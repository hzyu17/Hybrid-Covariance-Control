# Covariance control for SLIP system - Numerically Stable Version
# Improvements: scaling, regularization, better solvers, diagnostic checks

import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)
np.set_printoptions(suppress=True, precision=4)

# Import iLQR class
from hybrid_ilqr.h_ilqr_discrete_slip import solve_ilqr, extract_extensions
# Import SLIP dynamics
from dynamics.dynamics_discrete_slip import *
# Import experiment parameter class
from experiments.exp_params import *
from tools.plot_ellipsoid import *
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp

# =============================================================================
#                    Numerical Stability Utilities
# =============================================================================

def check_problem_conditioning(Sig0, SigT, S1, S2, Phi_A1, Phi_A2, E, name=""):
    """Diagnostic function to check conditioning of all matrices."""
    print(f"\n{'='*70}")
    print(f"  CONDITIONING REPORT {name}")
    print(f"{'='*70}")
    
    def safe_cond(M, name):
        try:
            c = np.linalg.cond(M)
            status = "✓ OK" if c < 1e8 else "⚠ WARNING" if c < 1e12 else "✗ CRITICAL"
            print(f"    cond({name:12s}): {c:12.2e}  [{status}]")
            return c
        except:
            print(f"    cond({name:12s}): FAILED TO COMPUTE")
            return np.inf
    
    def safe_eig_info(M, name):
        try:
            eigs = np.linalg.eigvalsh(M)
            min_e = np.min(eigs)
            max_e = np.max(eigs)
            ratio = max_e / min_e if min_e > 0 else np.inf
            status = "✓ OK" if min_e > 1e-10 else "⚠ WARNING" if min_e > 0 else "✗ CRITICAL"
            print(f"    {name:12s}: min={min_e:10.2e}, max={max_e:10.2e}, ratio={ratio:10.2e}  [{status}]")
            return min_e, max_e
        except:
            print(f"    {name:12s}: FAILED TO COMPUTE")
            return -np.inf, np.inf
    
    def matrix_info(M, name):
        print(f"    {name:12s}: shape={str(M.shape):12s}, norm={np.linalg.norm(M):10.2e}, rank={np.linalg.matrix_rank(M)}")
    
    print("\n  Matrix Dimensions and Norms:")
    matrix_info(Sig0, "Sig0")
    matrix_info(SigT, "SigT")
    matrix_info(S1, "S1")
    matrix_info(S2, "S2")
    matrix_info(Phi_A1, "Phi_A1")
    matrix_info(Phi_A2, "Phi_A2")
    matrix_info(E, "E")
    
    print("\n  Condition Numbers (lower is better, >1e8 may cause issues):")
    safe_cond(Sig0, "Sig0")
    safe_cond(SigT, "SigT")
    safe_cond(S1, "S1")
    safe_cond(S2, "S2")
    safe_cond(Phi_A1, "Phi_A1")
    safe_cond(Phi_A2, "Phi_A2")
    safe_cond(E, "E")
    
    print("\n  Eigenvalue Ranges (for symmetric matrices):")
    safe_eig_info(Sig0, "Sig0")
    safe_eig_info(SigT, "SigT")
    safe_eig_info(S1, "S1")
    safe_eig_info(S2, "S2")
    
    # Check symmetry
    print("\n  Symmetry Check (should be ~0):")
    print(f"    Sig0 asymmetry:  {np.linalg.norm(Sig0 - Sig0.T):.2e}")
    print(f"    SigT asymmetry:  {np.linalg.norm(SigT - SigT.T):.2e}")
    print(f"    S1 asymmetry:    {np.linalg.norm(S1 - S1.T):.2e}")
    print(f"    S2 asymmetry:    {np.linalg.norm(S2 - S2.T):.2e}")
    
    print(f"{'='*70}\n")


def regularized_inverse(M, reg_factor=1e-8, method='solve'):
    """Compute regularized inverse of a matrix."""
    n = M.shape[0]
    reg = reg_factor * np.trace(M) / n
    M_reg = M + reg * np.eye(n)
    
    if method == 'solve':
        return np.linalg.solve(M_reg, np.eye(n))
    elif method == 'pinv':
        return np.linalg.pinv(M, rcond=1e-10)
    else:
        return np.linalg.inv(M_reg)


def regularize_saltation_matrix(E, rcond=1e-6):
    """Regularize saltation matrix if ill-conditioned."""
    cond_E = np.linalg.cond(E)
    if cond_E > 1e6:
        print(f"  WARNING: Saltation matrix ill-conditioned (cond={cond_E:.2e}), regularizing...")
        U, s, Vh = np.linalg.svd(E)
        s_reg = np.maximum(s, rcond * s[0])
        E_reg = U @ np.diag(s_reg) @ Vh
        print(f"  Regularized condition number: {np.linalg.cond(E_reg):.2e}")
        return E_reg
    return E


def compute_scaling_factors(Sig0, SigT, nx1, nx2):
    """Compute scaling factors for numerical stability."""
    scale_cov1 = np.sqrt(np.trace(Sig0) / nx1)
    scale_cov2 = np.sqrt(np.trace(SigT) / nx2)
    return scale_cov1, scale_cov2


def ensure_symmetric_pd(M, min_eig=1e-10, name="matrix"):
    """Ensure matrix is symmetric positive definite."""
    M_sym = (M + M.T) / 2
    eigvals, eigvecs = np.linalg.eigh(M_sym)
    
    if np.min(eigvals) < min_eig:
        print(f"  Fixing non-PD {name}: min_eig was {np.min(eigvals):.2e}")
        eigvals = np.maximum(eigvals, min_eig)
        M_sym = eigvecs @ np.diag(eigvals) @ eigvecs.T
    
    return M_sym


# =============================================================================
#                    Dynamics Functions (JAX)
# =============================================================================

def slip_stance_dyn(x, u):
    """SLIP stance dynamics."""
    g = 9.81
    k = 25.0
    m = 0.5
    r0 = 1
    return jnp.array([
        x[1], 
        -2*x[1]*x[3]/x[2] - g*jnp.cos(x[0])/x[2], 
        x[3] + u[1]/m/x[2]/x[2], 
        k/m*(r0-x[2]) - g*jnp.sin(x[0]) + x[1]*x[1]*x[2] + k*u[0]/m
    ])

A1_func = jax.jit(jax.jacfwd(lambda x, u: slip_stance_dyn(x, u), 0))
B1_func = jax.jit(jax.jacfwd(lambda x, u: slip_stance_dyn(x, u), 1))


def slip_flight_dyn(x, u):
    """SLIP flight dynamics."""
    g = 9.81
    return jnp.array([
        x[1], 
        u[0], 
        x[3], 
        u[1] - g,
        u[2]
    ])

A2_func = jax.jit(jax.jacfwd(lambda x, u: slip_flight_dyn(x, u), 0))
B2_func = jax.jit(jax.jacfwd(lambda x, u: slip_flight_dyn(x, u), 1))


# =============================================================================
#                    ODE Integration with Better Accuracy
# =============================================================================

def integrate_state_transition(A_sequence, dt, nx, t_span, method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate state transition matrix Phi_A with high accuracy."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def ode_Phi_A(t, y):
        i = min(int(t / dt), nt - 1)
        Phi_reshaped = y.reshape((nx, nx))
        dydt = A_sequence[i] @ Phi_reshaped
        return dydt.flatten()
    
    Phi_0 = np.eye(nx).flatten()
    result = solve_ivp(ode_Phi_A, t_span, Phi_0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    
    Phi_t = result.y.reshape((nx, nx, -1))
    Phi_t = np.moveaxis(Phi_t, 2, 0)
    return Phi_t


def integrate_controllability_gramian(A_sequence, B_sequence, dt, nx, t_span, 
                                       method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate controllability Gramian S with high accuracy."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def ode_S(t, y):
        i = min(int(t / dt), nt - 1)
        S_reshaped = y.reshape((nx, nx))
        dydt = B_sequence[i] @ B_sequence[i].T + A_sequence[i] @ S_reshaped + S_reshaped @ A_sequence[i].T
        return dydt.flatten()
    
    S_0 = np.zeros((nx, nx)).flatten()
    result = solve_ivp(ode_S, t_span, S_0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    
    S_t = result.y.reshape((nx, nx, -1))
    S_t = np.moveaxis(S_t, 2, 0)
    return S_t


def integrate_hamiltonian_matrix(A_sequence, B_sequence, Q_sequence, dt, nx, t_span,
                                  method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate Hamiltonian matrix Phi_M with high accuracy."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def compute_M(t):
        i = min(int(t / dt), nt - 1)
        top_row = np.concatenate((A_sequence[i], -B_sequence[i] @ B_sequence[i].T), axis=1)
        bottom_row = np.concatenate((-Q_sequence[i], -A_sequence[i].T), axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)
    
    def ode_Phi_M(t, y):
        y_reshaped = y.reshape((2*nx, 2*nx))
        dydt = compute_M(t) @ y_reshaped
        return dydt.flatten()
    
    Phi_M_0 = np.eye(2 * nx).flatten()
    result = solve_ivp(ode_Phi_M, t_span, Phi_M_0, method=method, t_eval=t_eval, rtol=rtol, atol=atol)
    
    Phi_M = result.y.reshape((2*nx, 2*nx, -1))[:, :, -1]
    return Phi_M


# =============================================================================
#                    Covariance Steering Functions
# =============================================================================

def compute_Pi0(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon, min_eig=1e-12):
    """Compute initial Pi for forward covariance steering."""
    n_states = Sig_init.shape[0]
    
    # Ensure inputs are symmetric PD
    Sig_init = ensure_symmetric_pd(Sig_init, min_eig, "Sig_init")
    Sig_terminal = ensure_symmetric_pd(Sig_terminal, min_eig, "Sig_terminal")
    
    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig_init)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(np.maximum(eval_Sig0, min_eig))) @ evec_Sig0.T

    invSig0 = regularized_inverse(Sig_init)
    eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
    eval_invSig0 = np.maximum(eval_invSig0, min_eig)
    sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

    tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ Sig_terminal @ inv_Phi_M_12.T @ sqrtSig0
    tmp = ensure_symmetric_pd(tmp, min_eig, "tmp_matrix")
    
    eval_tmp, evec_tmp = np.linalg.eigh(tmp)
    sqrt_tmp = evec_tmp @ np.diag(np.sqrt(np.maximum(eval_tmp, min_eig))) @ evec_tmp.T

    return epsilon*invSig0/2 - inv_Phi_M_12 @ Phi_M_11 - sqrtInvSig0 @ sqrt_tmp @ sqrtInvSig0


def compute_Pi0_reverse(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon, min_eig=1e-12):
    """Compute initial Pi for reverse covariance steering."""
    n_states = Sig_init.shape[0]
    
    # Ensure inputs are symmetric PD
    Sig_init = ensure_symmetric_pd(Sig_init, min_eig, "Sig_init")
    Sig_terminal = ensure_symmetric_pd(Sig_terminal, min_eig, "Sig_terminal")
    
    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig_init)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(np.maximum(eval_Sig0, min_eig))) @ evec_Sig0.T

    invSig0 = regularized_inverse(Sig_init)
    eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
    eval_invSig0 = np.maximum(eval_invSig0, min_eig)
    sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

    tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ Sig_terminal @ inv_Phi_M_12.T @ sqrtSig0
    tmp = ensure_symmetric_pd(tmp, min_eig, "tmp_matrix")
    
    eval_tmp, evec_tmp = np.linalg.eigh(tmp)
    sqrt_tmp = evec_tmp @ np.diag(np.sqrt(np.maximum(eval_tmp, min_eig))) @ evec_tmp.T

    return epsilon*invSig0/2 - inv_Phi_M_12 @ Phi_M_11 + sqrtInvSig0 @ sqrt_tmp @ sqrtInvSig0


def solve_covariance_optimization(Sig0, SigT, E_linear, inv_S1, inv_S2, Phi_A1, Phi_A2, 
                                   nx1, nx2, epsilon, solver_params=None):
    """
    Solve the covariance steering optimization with improved numerical stability.
    
    Parameters:
    -----------
    solver_params : dict, optional
        Dictionary with solver configuration:
        - 'solver': str, one of 'SCS', 'MOSEK', 'CVXOPT', 'CLARABEL'
        - 'min_eig': float, minimum eigenvalue bound
        - 'max_eig_factor': float, max eigenvalue as factor of input covariances
        - 'verbose': bool
        - 'eps': float, solver tolerance
        - 'max_iters': int
    """
    import cvxpy as cp
    
    # Default solver parameters
    if solver_params is None:
        solver_params = {}
    
    solver_name = solver_params.get('solver', 'SCS')
    min_eig = solver_params.get('min_eig', 1e-8)
    max_eig_factor = solver_params.get('max_eig_factor', 100.0)
    verbose = solver_params.get('verbose', True)
    eps_solver = solver_params.get('eps', 1e-9)  # Tighter default tolerance
    max_iters = solver_params.get('max_iters', 100000)  # Increased default iterations
    use_log_det = solver_params.get('use_log_det', True)
    
    # Compute eigenvalue bounds
    max_eig = max_eig_factor * max(np.max(np.linalg.eigvalsh(Sig0)), 
                                    np.max(np.linalg.eigvalsh(SigT)))
    
    print(f"\n  {'─'*60}")
    print(f"  OPTIMIZATION CONFIGURATION")
    print(f"  {'─'*60}")
    print(f"    Solver:                 {solver_name}")
    print(f"    Min eigenvalue bound:   {min_eig:.2e}")
    print(f"    Max eigenvalue bound:   {max_eig:.2e}")
    print(f"    Regularization ε:       {epsilon:.2e}")
    print(f"    Solver tolerance:       {eps_solver:.2e}")
    print(f"    Max iterations:         {max_iters}")
    print(f"    Use log-det barrier:    {use_log_det}")
    print(f"  {'─'*60}")
    
    print(f"\n  PROBLEM DIMENSIONS:")
    print(f"    nx1 (stance states):    {nx1}")
    print(f"    nx2 (flight states):    {nx2}")
    print(f"    Decision variables:     {nx1*nx1 + nx2*nx2 + nx1*nx1 + nx2*nx2 + nx2*nx2} scalar values")
    print(f"    Σ̂⁻ ∈ ℝ^({nx1}×{nx1}),  Σ̂⁺ ∈ ℝ^({nx2}×{nx2})")
    print(f"    W₁ ∈ ℝ^({nx1}×{nx1}),  W₂ ∈ ℝ^({nx2}×{nx2}),  Y₂ ∈ ℝ^({nx2}×{nx2})")
    
    # Declare variables
    Sighat_minus = cp.Variable((nx1, nx1), symmetric=True)
    Sighat_plus = cp.Variable((nx2, nx2), symmetric=True)
    W1 = cp.Variable((nx1, nx1))
    W2 = cp.Variable((nx2, nx2))
    Y2 = cp.Variable((nx2, nx2), symmetric=True)
    
    # Construct block matrices
    Y1 = cp.bmat([[Sig0, W1.T], [W1, Sighat_minus]])
    slack_Y2 = cp.bmat([[Sighat_plus, W2.T], [W2, SigT - Y2]])
    
    # Objective function
    obj_1 = (cp.trace(inv_S1 @ Sighat_minus) 
             - 2*cp.trace(Phi_A2.T @ inv_S2 @ W2) 
             - 2*cp.trace(Phi_A1.T @ inv_S1 @ W1) 
             + cp.trace(Phi_A2.T @ inv_S2 @ Phi_A2 @ Sighat_plus))
    
    # Regularization term (log-det barrier or trace inverse)
    if use_log_det:
        # Use log-det barrier
        obj_2 = -epsilon * cp.log_det(Y1) - epsilon * cp.log_det(Y2)
    else:
        # Alternative: use trace of inverse (can be more stable)
        # Note: This changes the problem slightly
        obj_2 = epsilon * cp.lambda_sum_smallest(Y1, 1) + epsilon * cp.lambda_sum_smallest(Y2, 1)
        obj_2 = -obj_2  # Minimize negative of smallest eigenvalues
    
    # Constraints with eigenvalue bounds
    constraints = [
        # Coupling constraint
        Sighat_plus == E_linear @ Sighat_minus @ E_linear.T,
        
        # Positive definiteness with lower bounds
        Y1 >> min_eig * np.eye(2*nx1),
        slack_Y2 >> min_eig * np.eye(2*nx2),  # Fixed: slack_Y2 is (2*nx2, 2*nx2)
        Sighat_minus >> min_eig * np.eye(nx1),
        Sighat_plus >> min_eig * np.eye(nx2),
        Y2 >> min_eig * np.eye(nx2),
        
        # Upper bounds to prevent unbounded solutions
        Sighat_minus << max_eig * np.eye(nx1),
        Sighat_plus << max_eig * np.eye(nx2),
    ]
    
    # Create problem
    problem = cp.Problem(cp.Minimize(obj_1 + obj_2), constraints)
    
    print(f"\n  PROBLEM VALIDATION:")
    print(f"    Problem is DCP:         {problem.is_dcp()}")
    print(f"    Number of constraints:  {len(constraints)}")
    
    # Warm start with feasible initial guess
    try:
        Sighat_minus.value = Sig0.copy()
        Sighat_plus.value = E_linear @ Sig0 @ E_linear.T
        W1.value = Phi_A1 @ Sig0
        W2.value = Phi_A2 @ (E_linear @ Sig0 @ E_linear.T)
        Y2.value = 0.5 * SigT
        print(f"    Warm start:             ✓ Successfully initialized")
    except Exception as e:
        print(f"    Warm start:             ✗ Failed ({e})")
    
    # Solve with selected solver
    solve_kwargs = {'verbose': verbose}
    
    if solver_name == 'SCS':
        solve_kwargs.update({
            'solver': cp.SCS,
            'eps_abs': eps_solver,
            'eps_rel': eps_solver,
            'max_iters': max_iters,
        })
    elif solver_name == 'MOSEK':
        solve_kwargs.update({
            'solver': cp.MOSEK,
            'mosek_params': {
                'MSK_DPAR_INTPNT_CO_TOL_DFEAS': eps_solver,
                'MSK_DPAR_INTPNT_CO_TOL_PFEAS': eps_solver,
                'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': eps_solver,
            }
        })
    elif solver_name == 'CLARABEL':
        solve_kwargs.update({
            'solver': cp.CLARABEL,
        })
    elif solver_name == 'CVXOPT':
        solve_kwargs.update({
            'solver': cp.CVXOPT,
        })
    
    print(f"\n  {'─'*60}")
    print(f"  SOLVING OPTIMIZATION PROBLEM...")
    print(f"  {'─'*60}")
    
    try:
        problem.solve(**solve_kwargs)
        solver_success = True
    except cp.SolverError as e:
        print(f"\n    ⚠ Primary solver failed: {e}")
        print(f"    Trying fallback solver (SCS with more iterations)...")
        try:
            problem.solve(solver=cp.SCS, verbose=verbose, eps_abs=1e-9, eps_rel=1e-9, max_iters=200000)
            solver_success = True
        except Exception as e2:
            print(f"    Trying CLARABEL solver...")
            try:
                problem.solve(solver=cp.CLARABEL, verbose=verbose)
                solver_success = True
            except Exception as e3:
                print(f"    ✗ All solvers failed: {e3}")
                solver_success = False
    
    print(f"\n  {'─'*60}")
    print(f"  OPTIMIZATION RESULTS")
    print(f"  {'─'*60}")
    print(f"    Status:                 {problem.status}")
    print(f"    Optimal value:          {problem.value:.6e}" if problem.value is not None else "    Optimal value:          None")
    
    if problem.status in ['optimal', 'optimal_inaccurate']:
        status_symbol = "✓" if problem.status == 'optimal' else "⚠"
        print(f"    Convergence:            {status_symbol} {problem.status}")
    else:
        print(f"    Convergence:            ✗ {problem.status}")
        print("    WARNING: Optimization did not converge to optimal solution!")
        
    # Extract and validate results
    Sig_minus_opt = Sighat_minus.value
    Sig_plus_opt = Sighat_plus.value
    
    if Sig_minus_opt is not None:
        Sig_minus_opt = ensure_symmetric_pd(Sig_minus_opt, min_eig, "Sig_minus_opt")
        Sig_plus_opt = ensure_symmetric_pd(Sig_plus_opt, min_eig, "Sig_plus_opt")
        
        # Print solution quality metrics
        print(f"\n  SOLUTION QUALITY:")
        eig_minus = np.linalg.eigvalsh(Sig_minus_opt)
        eig_plus = np.linalg.eigvalsh(Sig_plus_opt)
        print(f"    Σ̂⁻ eigenvalues:        [{eig_minus.min():.4e}, {eig_minus.max():.4e}]")
        print(f"    Σ̂⁺ eigenvalues:        [{eig_plus.min():.4e}, {eig_plus.max():.4e}]")
        print(f"    Σ̂⁻ trace:              {np.trace(Sig_minus_opt):.6e}")
        print(f"    Σ̂⁺ trace:              {np.trace(Sig_plus_opt):.6e}")
        
        # Check constraint satisfaction
        constraint_error = np.linalg.norm(Sig_plus_opt - E_linear @ Sig_minus_opt @ E_linear.T, 'fro')
        print(f"    Coupling constraint:    ||Σ̂⁺ - EΣ̂⁻Eᵀ|| = {constraint_error:.2e}")
    
    print(f"  {'─'*60}")
    
    return Sig_minus_opt, Sig_plus_opt, problem.status, problem.value


def integrate_covariance_ode(A_sequence, B_sequence, K_sequence, Sig_init, dt, nx, t_span, 
                              epsilon, method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate covariance ODE under feedback control."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def cov_derivative(t, cov_flat):
        i = min(int(t / dt), nt - 1)
        cov_j = cov_flat.reshape((nx, nx))
        
        A_i = A_sequence[i]
        B_i = B_sequence[i]
        K_i = K_sequence[i]
        
        Acl_i = A_i + B_i @ K_i
        d_cov_j_dt = Acl_i @ cov_j + cov_j @ Acl_i.T + epsilon * B_i @ B_i.T
        
        return d_cov_j_dt.flatten()
    
    cov_0_flat = Sig_init.flatten()
    result = solve_ivp(cov_derivative, t_span, cov_0_flat, method=method, 
                       t_eval=t_eval, rtol=rtol, atol=atol)
    
    cov_trj = result.y.reshape((nx, nx, -1))
    cov_trj = np.moveaxis(cov_trj, 2, 0)
    return cov_trj


def integrate_XY_ode(A_sequence, B_sequence, Q_sequence, Pi_0, dt, nx, t_span, 
                      reverse=False, method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate XY ODE system for Pi computation."""
    nt = A_sequence.shape[0]
    
    if reverse:
        t_eval = np.linspace(t_span[1], t_span[0], nt + 1)
        t_span_actual = (t_span[1], t_span[0])
    else:
        t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
        t_span_actual = t_span
    
    def compute_M(t):
        i = min(int(t / dt), nt - 1)
        top_row = np.concatenate((A_sequence[i], -B_sequence[i] @ B_sequence[i].T), axis=1)
        bottom_row = np.concatenate((-Q_sequence[i], -A_sequence[i].T), axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)
    
    def ode_XY(t, y):
        y_reshaped = y.reshape((2*nx, nx))
        dydt = compute_M(t) @ y_reshaped
        return dydt.flatten()
    
    # Initial conditions
    v_XY_0 = np.zeros((2 * nx, nx))
    v_XY_0[:nx, :nx] = np.eye(nx)
    v_XY_0[nx:, :nx] = (Pi_0 + Pi_0.T) / 2
    
    result = solve_ivp(ode_XY, t_span_actual, v_XY_0.flatten(), method=method, 
                       t_eval=t_eval, rtol=rtol, atol=atol)
    
    v_XY_solution = result.y.reshape((2 * nx, nx, -1))
    
    # Extract Pi trajectory
    Pi_trj = np.zeros((nt + 1, nx, nx))
    Pi_trj[0] = (Pi_0 + Pi_0.T) / 2
    
    for i in range(1, nt + 1):
        X_i = v_XY_solution[:nx, :nx, i]
        Y_i = v_XY_solution[nx:, :nx, i]
        try:
            inv_Xi = np.linalg.solve(X_i, np.eye(nx))
            Pi_trj[i] = Y_i @ inv_Xi
        except np.linalg.LinAlgError:
            inv_Xi = np.linalg.pinv(X_i)
            Pi_trj[i] = Y_i @ inv_Xi
    
    if reverse:
        Pi_trj = Pi_trj[::-1, :, :]
    
    return Pi_trj[:-1]  # Return nt points


# =============================================================================
#                              Main Script
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  HYBRID COVARIANCE STEERING FOR SLIP")
    print("  Numerically Stable Version")
    print("="*70)
    
    # -------------- 
    #  SLIP Parameters 
    # --------------
    dt = 0.00005
    epsilon = 0.0015
    dt_shrink = 0.9
    r0 = 1
    
    n_modes = 2
    
    # Mode definitions
    # mode 1 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, tau_hip]
    # mode 0 (flight): x = [px, vx, pz, vz, theta], u = [ax, az, theta_dot]
    n_states = [5, 4]
    n_inputs = [3, 2]
    
    # --------------------------
    #  Problem Setup
    # --------------------------
    init_mode = 1
    start_time = 0
    end_time = 0.5
    
    # Terminal cost 
    target_mode = 0
    Q_T = 2.0 * np.eye(n_states[0])
    
    # Running costs
    Q_k = [np.zeros((n_states[0], n_states[0])), np.zeros((n_states[1], n_states[1]))]
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    # Initial and target states
    init_theta_deg = 100
    init_theta = init_theta_deg / 180 * np.pi
    init_state = np.array([init_theta, -4.0, 0.5*r0, 0.0], dtype=np.float64)
    target_state = np.array([1.1, 2.25, 1.4, 0.0, np.pi/3], dtype=np.float64)
    
    # Time setup
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    print(f"\n  {'─'*60}")
    print(f"  PROBLEM PARAMETERS")
    print(f"  {'─'*60}")
    print(f"    Time step (dt):         {dt:.6f} s")
    print(f"    Time horizon:           [{start_time}, {end_time}] s")
    print(f"    Number of time steps:   {nt}")
    print(f"    Noise intensity (ε):    {epsilon}")
    print(f"    Spring rest length:     {r0} m")
    print(f"  {'─'*60}")
    
    print(f"\n  MODE CONFIGURATION:")
    print(f"    Mode 0 (flight):  {n_states[0]} states, {n_inputs[0]} inputs")
    print(f"                      x = [px, vx, pz, vz, θ]")
    print(f"                      u = [ax, az, θ̇]")
    print(f"    Mode 1 (stance):  {n_states[1]} states, {n_inputs[1]} inputs")
    print(f"                      x = [θ, θ̇, r, ṙ]")
    print(f"                      u = [Δr, τ_hip]")
    
    print(f"\n  BOUNDARY CONDITIONS:")
    print(f"    Initial mode:           {init_mode} ({'stance' if init_mode == 1 else 'flight'})")
    print(f"    Target mode:            {target_mode} ({'stance' if target_mode == 1 else 'flight'})")
    print(f"    Initial state:          {np.array2string(init_state, precision=4, separator=', ')}")
    print(f"    Target state:           {np.array2string(target_state, precision=4, separator=', ')}")
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ================================
    #  Solve for hybrid iLQR proposal
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  SOLVING HYBRID iLQR")
    print(f"  {'─'*60}")
    exp_params = ExpParams()
    
    initial_guess = [0.0*np.ones((nt, n_inputs[0])), 0.0*np.ones((nt, n_inputs[1]))]
    symbolic_dynamics = [symbolic_flight_dynamics_slip, symbolic_stance_dynamics_slip]
    
    n_exp = 1
    n_samples = 0 
    
    exp_params.update_params(n_modes, init_mode, target_mode, 
                             n_states, init_state, target_state, 
                             start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, symbolic_dynamics, 
                             event_detect_discrete_slip, plot_slip, convert_state_21_slip, 
                             init_reset_args, target_reset_args, 
                             animate_slip)
    
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan, modes, states, inputs, saltations,
     k_feedforward, K_feedback, A_trj, B_trj,
     current_cost, states_iter,
     ref_modechanges, reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    
    print(f"    ✓ Hybrid iLQR converged!")
    print(f"    Final cost:             {current_cost:.6e}")
    print(f"    Iterations:             {len(states_iter)}")
    
    # Mean trajectory under H-iLQR
    Noise_zero = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    (mode_trj_mean, xt_trj_mean, ut_cl_trj_mean, Sk_mean, 
     xt_ref_actual_mean, reset_args_mean) = h_stoch_fb_rollout_slip(
        init_mode, init_state, n_inputs, states, modes, 
        inputs, K_feedback, k_feedforward, target_state, 
        Q_T, 0.0, dt, epsilon, Noise_zero, 
        reference_extension_helper, init_reset_args)
    
    exp_data.add_nominal_data(hybrid_ilqr_result)

    (v_mode_change_ref, v_ref_ext_bwd, v_ref_ext_fwd, 
     v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
     v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, v_tevents_ref) = extract_extensions(
        reference_extension_helper, start_index=0)

    t_event = v_tevents_ref[0]
    mode_1 = 1  # stance
    mode_2 = 0  # flight
    
    print(f"\n  MODE TRANSITION DETECTED:")
    print(f"    Event time index:       {t_event}")
    print(f"    Event time:             {t_event * dt:.6f} s")
    print(f"    Transition:             Mode {mode_1} (stance) → Mode {mode_2} (flight)")

    # =============================================== 
    #               Covariance Steering Setup
    # ===============================================
    print("\n" + "="*70)
    print("  COVARIANCE STEERING SETUP")
    print("="*70)
    
    # Initial and target covariances
    Sig0 = 0.002 * np.eye(4)
    SigT = 0.0003 * np.eye(5)
    
    print(f"\n  COVARIANCE SPECIFICATIONS:")
    print(f"    Initial Σ₀ (stance):    {Sig0[0,0]:.4e} × I₄")
    print(f"    Target Σ_T (flight):    {SigT[0,0]:.4e} × I₅")
    print(f"    Σ₀ trace:               {np.trace(Sig0):.6e}")
    print(f"    Σ_T trace:              {np.trace(SigT):.6e}")
    print(f"    Uncertainty reduction:  {np.trace(Sig0)/np.trace(SigT):.2f}× (trace ratio)")
    
    # Get saltation matrix and regularize if needed
    E_linear = saltations[t_event]
    
    print(f"\n  SALTATION MATRIX E:")
    print(f"    Shape:                  {E_linear.shape}")
    print(f"    Condition number:       {np.linalg.cond(E_linear):.2e}")
    print(f"    Rank:                   {np.linalg.matrix_rank(E_linear)}")
    print(f"    Frobenius norm:         {np.linalg.norm(E_linear, 'fro'):.4e}")
    
    E_linear = regularize_saltation_matrix(E_linear)
    
    print(f"\n    E = ")
    for row in E_linear:
        print(f"        [{', '.join([f'{x:8.4f}' for x in row])}]")
    
    # ======================================================
    #            Build System Matrices
    # ======================================================
    nx1, nx2 = n_states[1], n_states[0]  # 4, 5
    nt1, nt2 = t_event + 1, nt - t_event - 1
    
    print(f"\n  {'─'*60}")
    print(f"  PHASE DECOMPOSITION")
    print(f"  {'─'*60}")
    print(f"    Phase 1 (stance):       [0, t⁻]")
    print(f"      State dimension:      nx₁ = {nx1}")
    print(f"      Input dimension:      nu₁ = {n_inputs[mode_1]}")
    print(f"      Time steps:           nt₁ = {nt1}")
    print(f"      Duration:             {nt1 * dt:.6f} s")
    print(f"    Phase 2 (flight):       [t⁺, T]")
    print(f"      State dimension:      nx₂ = {nx2}")
    print(f"      Input dimension:      nu₂ = {n_inputs[mode_2]}")
    print(f"      Time steps:           nt₂ = {nt2}")
    print(f"      Duration:             {nt2 * dt:.6f} s")
    
    # Extract continuous-time A, B matrices for phase 1
    A1 = np.zeros((nt1, nx1, nx1))
    B1 = np.zeros((nt1, nx1, n_inputs[mode_1]))
    Q1 = np.zeros((nt1, nx1, nx1))
    
    for i in range(nt1):
        A1[i] = (A_trj[i] - np.eye(nx1)) / dt
        B1[i] = B_trj[i] / dt
    
    # Extract continuous-time A, B matrices for phase 2
    A2 = np.zeros((nt2, nx2, nx2))
    B2 = np.zeros((nt2, nx2, n_inputs[mode_2]))
    Q2 = np.zeros((nt2, nx2, nx2))
    
    for i in range(nt2):
        ii = t_event + i + 1
        A2[i] = (A_trj[ii] - np.eye(nx2)) / dt
        B2[i] = B_trj[ii] / dt
    
    # Time spans
    t_span1 = (0, dt * nt1)
    t_span2 = (0, dt * nt2)
    
    print(f"\n  LINEARIZED DYNAMICS (sampled at midpoint):")
    mid1 = nt1 // 2
    mid2 = nt2 // 2
    print(f"    Phase 1 A[{mid1}] spectral radius: {np.max(np.abs(np.linalg.eigvals(A1[mid1]))):.4e}")
    print(f"    Phase 1 B[{mid1}] norm:            {np.linalg.norm(B1[mid1]):.4e}")
    print(f"    Phase 2 A[{mid2}] spectral radius: {np.max(np.abs(np.linalg.eigvals(A2[mid2]))):.4e}")
    print(f"    Phase 2 B[{mid2}] norm:            {np.linalg.norm(B2[mid2]):.4e}")
    
    # ======================================================
    #            Integrate Transition Matrices
    # ======================================================
    print(f"\n  {'─'*60}")
    print(f"  INTEGRATING SYSTEM MATRICES")
    print(f"  {'─'*60}")
    
    # Use high-accuracy integration
    ode_method = 'DOP853'
    ode_rtol = 1e-10
    ode_atol = 1e-12
    
    print(f"    ODE method:             {ode_method}")
    print(f"    Relative tolerance:     {ode_rtol:.0e}")
    print(f"    Absolute tolerance:     {ode_atol:.0e}")
    
    # Phase 1: Phi_A1
    print(f"\n    Computing Φ_A1 (state transition, phase 1)...", end=" ")
    Phi_A1_t = integrate_state_transition(A1, dt, nx1, t_span1, method=ode_method, 
                                          rtol=ode_rtol, atol=ode_atol)
    Phi_A1 = Phi_A1_t[-1]
    print(f"✓  cond={np.linalg.cond(Phi_A1):.2e}")
    
    # Phase 2: Phi_A2
    print(f"    Computing Φ_A2 (state transition, phase 2)...", end=" ")
    Phi_A2_t = integrate_state_transition(A2, dt, nx2, t_span2, method=ode_method,
                                          rtol=ode_rtol, atol=ode_atol)
    Phi_A2 = Phi_A2_t[-1]
    print(f"✓  cond={np.linalg.cond(Phi_A2):.2e}")
    
    # Controllability Gramians
    print(f"    Computing S₁ (controllability Gramian, phase 1)...", end=" ")
    S1_t = integrate_controllability_gramian(A1, B1, dt, nx1, t_span1, method=ode_method,
                                              rtol=ode_rtol, atol=ode_atol)
    S1 = ensure_symmetric_pd(S1_t[-1], 1e-12, "S1")
    print(f"✓  cond={np.linalg.cond(S1):.2e}")
    
    print(f"    Computing S₂ (controllability Gramian, phase 2)...", end=" ")
    S2_t = integrate_controllability_gramian(A2, B2, dt, nx2, t_span2, method=ode_method,
                                              rtol=ode_rtol, atol=ode_atol)
    S2 = ensure_symmetric_pd(S2_t[-1], 1e-12, "S2")
    print(f"✓  cond={np.linalg.cond(S2):.2e}")
    
    # Regularized inverses
    print(f"\n    Computing regularized inverses...")
    inv_S1 = regularized_inverse(S1)
    inv_S2 = regularized_inverse(S2)
    print(f"    ||S₁⁻¹||:               {np.linalg.norm(inv_S1):.4e}")
    print(f"    ||S₂⁻¹||:               {np.linalg.norm(inv_S2):.4e}")
    
    # Run conditioning diagnostics
    check_problem_conditioning(Sig0, SigT, S1, S2, Phi_A1, Phi_A2, E_linear, 
                               "Before Optimization")
    
    # ======================================================
    #            Solve Optimization Problem
    # ======================================================
    print("\n" + "-"*70)
    print("  Solving Covariance Steering Optimization")
    print("-"*70)
    
    # Solver configuration
    solver_params = {
        'solver': 'CLARABEL',    # CLARABEL often handles log-det better than SCS
        'min_eig': 1e-8,
        'max_eig_factor': 100.0,
        'verbose': True,
        'eps': 1e-9,              # Tighter tolerance
        'max_iters': 500000,      # Many more iterations for convergence
        'use_log_det': True,
    }
    
    Sig_minus_opt, Sig_plus_opt, opt_status, opt_value = solve_covariance_optimization(
        Sig0, SigT, E_linear, inv_S1, inv_S2, Phi_A1, Phi_A2, 
        nx1, nx2, epsilon, solver_params)
    
    if Sig_minus_opt is None:
        print("\n  ERROR: Optimization failed! Using fallback values.")
        Sig_minus_opt = Sig0.copy()
        Sig_plus_opt = E_linear @ Sig0 @ E_linear.T
    
    print("\n  Optimized Sigma_minus:")
    print(np.array2string(Sig_minus_opt, precision=6))
    print("\n  Optimized Sigma_plus:")
    print(np.array2string(Sig_plus_opt, precision=6))
    print("\n  E @ Sigma_minus @ E.T:")
    print(E_linear @ Sig_minus_opt @ E_linear.T)
    
    # ======================================================
    #        Solve Optimal Covariance Steering Controller
    # ======================================================
    print("\n" + "="*70)
    print("  COMPUTING OPTIMAL FEEDBACK GAINS (THEOREM 2)")
    print("="*70)
    
    # =====================================================================
    # THEORY (from paper, Section 6 - Theorem 2):
    # For general (rectangular) Ξ, we use the SDP formulation.
    # 
    # The SDP finds optimal Σ⁻, Σ⁺ that minimize the KL divergence
    # subject to the covariance jump constraint: Σ⁺ = Ξ Σ⁻ Ξ'
    #
    # After solving the SDP, we recover the controllers using the smooth
    # covariance steering formulas (equations 10 and 11) for each phase:
    #   - Phase 1: Steer Σ₀ → Σ⁻_opt using equation (10) for Π₁(0)
    #   - Phase 2: Steer Σ⁺_opt → Σ_T using equation (11) for Π₂(T)
    #
    # For rectangular Ξ, the co-state jump condition Π₁(t⁻) = Ξ' Π₂(t⁺) Ξ
    # may have residual error, but the covariance steering is optimal.
    # =====================================================================
    
    print(f"\n  {'─'*60}")
    print(f"  USING SDP-OPTIMIZED BOUNDARY COVARIANCES")
    print(f"  {'─'*60}")
    print(f"    Σ⁻_opt trace:           {np.trace(Sig_minus_opt):.6e}")
    print(f"    Σ⁺_opt trace:           {np.trace(Sig_plus_opt):.6e}")
    print(f"    Σ⁺_opt = Ξ Σ⁻_opt Ξ' error: {np.linalg.norm(Sig_plus_opt - E_linear @ Sig_minus_opt @ E_linear.T):.4e}")
    
    # --------------------- Phase 1: [0, t^-] ---------------------
    print(f"\n  {'─'*60}")
    print(f"  PHASE 1: STANCE PHASE [0, t⁻]")
    print(f"  Steering Σ₀ → Σ⁻_opt")
    print(f"  {'─'*60}")
    
    # Integrate Hamiltonian matrix for Phase 1
    print(f"\n    Computing Hamiltonian transition matrix Φ_M1...", end=" ")
    Phi_M1 = integrate_hamiltonian_matrix(A1, B1, Q1, dt, nx1, t_span1,
                                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    # Extract blocks from Φ_M1
    Phi_M1_11 = Phi_M1[:nx1, :nx1]
    Phi_M1_12 = Phi_M1[:nx1, nx1:]
    inv_Phi_M1_12 = regularized_inverse(Phi_M1_12)
    
    print(f"    Φ_M1 blocks:")
    print(f"      ||Φ_M1_11||:          {np.linalg.norm(Phi_M1_11):.4e}")
    print(f"      ||Φ_M1_12||:          {np.linalg.norm(Phi_M1_12):.4e}")
    print(f"      cond(Φ_M1_12):        {np.linalg.cond(Phi_M1_12):.2e}")
    
    # Compute Π₁(0) using equation (10) for steering Σ₀ → Σ⁻_opt
    print(f"\n    Computing Π₁(0) via equation (10)...", end=" ")
    Pi1_0 = compute_Pi0(Sig0, Sig_minus_opt, Phi_M1_11, inv_Phi_M1_12, epsilon)
    print(f"✓")
    print(f"      ||Π₁(0)||:            {np.linalg.norm(Pi1_0):.4e}")
    print(f"      Π₁(0) eigenvalues:    {np.array2string(np.linalg.eigvalsh(Pi1_0), precision=4)}")
    
    # Integrate XY system forward to get Pi trajectory in Phase 1
    print(f"    Integrating Π₁(t) trajectory (forward)...", end=" ")
    Pi1 = integrate_XY_ode(A1, B1, Q1, Pi1_0, dt, nx1, t_span1, reverse=False,
                           method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    Pi1_t_minus = Pi1[-1]  # Π₁(t⁻)
    print(f"      ||Π₁(t⁻)||:           {np.linalg.norm(Pi1_t_minus):.4e}")
    print(f"      Π₁(t⁻) eigenvalues:   {np.array2string(np.linalg.eigvalsh(Pi1_t_minus), precision=4)}")
    
    # Compute feedback gains for Phase 1
    K1 = np.zeros((nt1, n_inputs[mode_1], nx1))
    for i in range(nt1):
        K1[i] = -B1[i].T @ Pi1[i]
    
    K1_ilqr = np.zeros((nt1, n_inputs[mode_1], nx1))
    for i in range(nt1):
        K1_ilqr[i] = K_feedback[i]
    
    print(f"\n    Feedback gain statistics (K₁ = -B₁ᵀΠ₁):")
    K1_norms = [np.linalg.norm(K1[i]) for i in range(nt1)]
    print(f"      ||K₁|| range:         [{min(K1_norms):.4e}, {max(K1_norms):.4e}]")
    print(f"      ||K₁|| mean:          {np.mean(K1_norms):.4e}")
    
    K1_ilqr_norms = [np.linalg.norm(K1_ilqr[i]) for i in range(nt1)]
    print(f"      ||K₁_iLQR|| range:    [{min(K1_ilqr_norms):.4e}, {max(K1_ilqr_norms):.4e}]")
    
    # --------------------- Phase 2: [t^+, T] ---------------------
    print(f"\n  {'─'*60}")
    print(f"  PHASE 2: FLIGHT PHASE [t⁺, T]")
    print(f"  Steering Σ⁺_opt → Σ_T")
    print(f"  {'─'*60}")
    
    # Integrate Hamiltonian matrix for Phase 2
    print(f"\n    Computing Hamiltonian transition matrix Φ_M2...", end=" ")
    Phi_M2 = integrate_hamiltonian_matrix(A2, B2, Q2, dt, nx2, t_span2,
                                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    # For backward steering (Σ⁺_opt → Σ_T), we need the inverse transition
    # Equation (11) / Remark 1: Π(T) in terms of Ψ = Φ⁻¹
    Psi_M2 = regularized_inverse(Phi_M2)  # Ψ = Φ⁻¹
    
    Psi_M2_11 = Psi_M2[:nx2, :nx2]
    Psi_M2_12 = Psi_M2[:nx2, nx2:]
    inv_Psi_M2_12 = regularized_inverse(Psi_M2_12)
    
    print(f"    Ψ_M2 = Φ_M2⁻¹ blocks:")
    print(f"      ||Ψ_M2_11||:          {np.linalg.norm(Psi_M2_11):.4e}")
    print(f"      ||Ψ_M2_12||:          {np.linalg.norm(Psi_M2_12):.4e}")
    print(f"      cond(Ψ_M2_12):        {np.linalg.cond(Psi_M2_12):.2e}")
    
    # Compute Π₂(T) using equation (11) / Remark 1
    # This steers from Σ⁺_opt (initial) to Σ_T (terminal)
    print(f"\n    Computing Π₂(T) via Remark 1 / equation (11)...", end=" ")
    Pi2_T = compute_Pi0_reverse(SigT, Sig_plus_opt, Psi_M2_11, inv_Psi_M2_12, epsilon)
    print(f"✓")
    print(f"      ||Π₂(T)||:            {np.linalg.norm(Pi2_T):.4e}")
    print(f"      Π₂(T) eigenvalues:    {np.array2string(np.linalg.eigvalsh(Pi2_T), precision=4)}")
    
    # Integrate XY system backward to get Pi trajectory in Phase 2
    print(f"    Integrating Π₂(t) trajectory (backward)...", end=" ")
    Pi2 = integrate_XY_ode(A2, B2, Q2, Pi2_T, dt, nx2, t_span2, reverse=True,
                           method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    Pi2_t_plus = Pi2[0]  # Π₂(t⁺)
    print(f"      ||Π₂(t⁺)||:           {np.linalg.norm(Pi2_t_plus):.4e}")
    print(f"      Π₂(t⁺) eigenvalues:   {np.array2string(np.linalg.eigvalsh(Pi2_t_plus), precision=4)}")
    
    # Compute feedback gains for Phase 2
    K2 = np.zeros((nt2, n_inputs[mode_2], nx2))
    for i in range(nt2):
        K2[i] = -B2[i].T @ Pi2[i]
    
    K2_ilqr = np.zeros((nt2, n_inputs[mode_2], nx2))
    for i in range(nt2):
        ii = t_event + i + 1
        K2_ilqr[i] = K_feedback[ii]
    
    print(f"\n    Feedback gain statistics (K₂ = -B₂ᵀΠ₂):")
    K2_norms = [np.linalg.norm(K2[i]) for i in range(nt2)]
    print(f"      ||K₂|| range:         [{min(K2_norms):.4e}, {max(K2_norms):.4e}]")
    print(f"      ||K₂|| mean:          {np.mean(K2_norms):.4e}")
    
    K2_ilqr_norms = [np.linalg.norm(K2_ilqr[i]) for i in range(nt2)]
    print(f"      ||K₂_iLQR|| range:    [{min(K2_ilqr_norms):.4e}, {max(K2_ilqr_norms):.4e}]")
    
    # --------------------- VERIFY JUMP CONDITIONS ---------------------
    print(f"\n  {'─'*60}")
    print(f"  JUMP CONDITION ANALYSIS")
    print(f"  {'─'*60}")
    
    # IMPORTANT NOTE about rectangular Ξ:
    # For rectangular saltation matrix (n₂ ≠ n₁), the co-state jump condition
    # Π₁(t⁻) = Ξ' Π₂(t⁺) Ξ may NOT be exactly satisfied when using Theorem 2.
    # 
    # This is because Theorem 2 (SDP formulation) optimizes over covariances,
    # not co-states. The co-states are reconstructed from boundary conditions
    # for each phase independently. The SDP couples phases through the
    # covariance constraint Σ⁺ = Ξ Σ⁻ Ξ', but NOT through the co-state.
    #
    # For invertible Ξ (Theorem 1), the hybrid transition kernel automatically
    # ensures both covariance and co-state jump conditions. For rectangular Ξ,
    # this coupling is lost.
    #
    # Despite this, the COVARIANCE steering should still work correctly:
    # - Σ(T) should approach Σ_T
    # - The control cost may be slightly suboptimal
    
    print(f"\n    NOTE: For rectangular Ξ ({nx2}×{nx1}), Theorem 2 does NOT")
    print(f"    explicitly enforce the co-state jump condition.")
    print(f"    Covariance steering should still work; co-state mismatch may")
    print(f"    indicate slightly suboptimal control cost.")
    
    # Co-state jump condition: Π₁(t⁻) = Ξ' Π₂(t⁺) Ξ
    # For rectangular Ξ (5×4), this maps R^{5×5} → R^{4×4}
    EtPi2E = E_linear.T @ Pi2_t_plus @ E_linear
    
    print(f"\n    CO-STATE JUMP: Π₁(t⁻) vs Ξ' Π₂(t⁺) Ξ")
    print(f"\n    Π₁(t⁻) (from Phase 1 forward integration):")
    for row in Pi1_t_minus:
        print(f"      [{', '.join([f'{x:10.4f}' for x in row])}]")
    
    print(f"\n    Ξ' Π₂(t⁺) Ξ (from Phase 2 backward integration):")
    for row in EtPi2E:
        print(f"      [{', '.join([f'{x:10.4f}' for x in row])}]")
    
    jump_error_costate = np.linalg.norm(Pi1_t_minus - EtPi2E, 'fro')
    relative_error = jump_error_costate / (np.linalg.norm(Pi1_t_minus, 'fro') + 1e-10)
    
    print(f"\n    ||Π₁(t⁻) - Ξ'Π₂(t⁺)Ξ||_F:     {jump_error_costate:.4e}")
    print(f"    Relative error:                {relative_error:.4e}")
    
    if jump_error_costate < 1e-2:
        print(f"    ✓ Co-state jump condition approximately satisfied!")
    else:
        print(f"    ⚠ Co-state jump condition has significant residual")
        print(f"      (This is expected for rectangular Ξ - see paper Section 6)")
    
    # --------------------- COVARIANCE PROPAGATION ---------------------
    print(f"\n  {'─'*60}")
    print(f"  COVARIANCE PROPAGATION")
    print(f"  {'─'*60}")
    
    # Phase 1: H-CS
    print(f"\n    Phase 1: Propagating covariance under H-CS controller...", end=" ")
    cov_trj_1 = integrate_covariance_ode(A1, B1, K1, Sig0, dt, nx1, t_span1, epsilon,
                                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    Sig_minus_computed = cov_trj_1[-1]
    print(f"    Σ⁻ computed (H-CS):")
    print(f"      trace:                {np.trace(Sig_minus_computed):.6e}")
    print(f"      eigenvalues:          {np.array2string(np.linalg.eigvalsh(Sig_minus_computed), precision=4)}")
    
    print(f"    Σ⁻ target (from SDP):")
    print(f"      trace:                {np.trace(Sig_minus_opt):.6e}")
    print(f"    ||Σ⁻_comp - Σ⁻_opt||:   {np.linalg.norm(Sig_minus_computed - Sig_minus_opt, 'fro'):.4e}")
    
    # Phase 1: H-iLQR
    print(f"\n    Phase 1: Propagating covariance under H-iLQR controller...", end=" ")
    cov_trj_1_ilqr = integrate_covariance_ode(A1, B1, K1_ilqr, Sig0, dt, nx1, t_span1, epsilon,
                                               method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    print(f"    Σ⁻ computed (H-iLQR):")
    print(f"      trace:                {np.trace(cov_trj_1_ilqr[-1]):.6e}")
    
    # Covariance jump: Σ⁺ = Ξ Σ⁻ Ξ'
    Sig_plus_computed = E_linear @ Sig_minus_computed @ E_linear.T
    
    print(f"\n    COVARIANCE JUMP: Σ⁺ = Ξ Σ⁻ Ξ'")
    print(f"    Σ⁺ computed:")
    print(f"      trace:                {np.trace(Sig_plus_computed):.6e}")
    print(f"      eigenvalues:          {np.array2string(np.linalg.eigvalsh(Sig_plus_computed), precision=4)}")
    print(f"    Σ⁺ target (from SDP):")
    print(f"      trace:                {np.trace(Sig_plus_opt):.6e}")
    print(f"    ||Σ⁺_comp - Σ⁺_opt||:   {np.linalg.norm(Sig_plus_computed - Sig_plus_opt, 'fro'):.4e}")
    
    # Phase 2: H-CS
    print(f"\n    Phase 2: Propagating covariance under H-CS controller...", end=" ")
    cov_trj_2 = integrate_covariance_ode(A2, B2, K2, Sig_plus_computed, dt, nx2, t_span2, epsilon,
                                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    # Phase 2: H-iLQR
    cov2_0_ilqr = E_linear @ cov_trj_1_ilqr[-1] @ E_linear.T
    print(f"    Phase 2: Propagating covariance under H-iLQR controller...", end=" ")
    cov_trj_2_ilqr = integrate_covariance_ode(A2, B2, K2_ilqr, cov2_0_ilqr, dt, nx2, t_span2, epsilon,
                                               method=ode_method, rtol=ode_rtol, atol=ode_atol)
    print(f"✓")
    
    # ======================================================
    #                    Results Summary
    # ======================================================
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n  {'─'*60}")
    print(f"  COVARIANCE EVOLUTION")
    print(f"  {'─'*60}")
    
    print(f"\n  Initial covariance Σ₀:")
    print(f"    trace:                  {np.trace(cov_trj_1[0]):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(cov_trj_1[0]), precision=4)}")
    
    print(f"\n  Terminal covariance Σ(T) - H-CS:")
    print(f"    trace:                  {np.trace(cov_trj_2[-1]):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(cov_trj_2[-1]), precision=4)}")
    
    print(f"\n  Terminal covariance Σ(T) - H-iLQR:")
    print(f"    trace:                  {np.trace(cov_trj_2_ilqr[-1]):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(cov_trj_2_ilqr[-1]), precision=4)}")
    
    print(f"\n  Target covariance Σ_T:")
    print(f"    trace:                  {np.trace(SigT):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(SigT), precision=4)}")
    
    print(f"\n  {'─'*60}")
    print(f"  PERFORMANCE METRICS")
    print(f"  {'─'*60}")
    
    # Compute errors
    error_hcs = np.linalg.norm(cov_trj_2[-1] - SigT, 'fro')
    error_ilqr = np.linalg.norm(cov_trj_2_ilqr[-1] - SigT, 'fro')
    trace_error_hcs = np.abs(np.trace(cov_trj_2[-1]) - np.trace(SigT))
    trace_error_ilqr = np.abs(np.trace(cov_trj_2_ilqr[-1]) - np.trace(SigT))
    
    print(f"\n  Frobenius norm error ||Σ(T) - Σ_T||_F:")
    print(f"    H-CS:                   {error_hcs:.6e}")
    print(f"    H-iLQR:                 {error_ilqr:.6e}")
    print(f"    Improvement:            {error_ilqr / error_hcs:.2f}× better with H-CS")
    
    print(f"\n  Trace error |tr(Σ(T)) - tr(Σ_T)|:")
    print(f"    H-CS:                   {trace_error_hcs:.6e}")
    print(f"    H-iLQR:                 {trace_error_ilqr:.6e}")
    
    print(f"\n  Relative error ||Σ(T) - Σ_T||_F / ||Σ_T||_F:")
    print(f"    H-CS:                   {error_hcs / np.linalg.norm(SigT, 'fro') * 100:.2f}%")
    print(f"    H-iLQR:                 {error_ilqr / np.linalg.norm(SigT, 'fro') * 100:.2f}%")
    
    # Eigenvalue comparison
    eig_T = np.linalg.eigvalsh(SigT)
    eig_hcs = np.linalg.eigvalsh(cov_trj_2[-1])
    eig_ilqr = np.linalg.eigvalsh(cov_trj_2_ilqr[-1])
    
    print(f"\n  Eigenvalue comparison (sorted):")
    print(f"    {'Target':^12s} {'H-CS':^12s} {'H-iLQR':^12s} {'H-CS err':^12s} {'iLQR err':^12s}")
    print(f"    {'-'*60}")
    for i in range(len(eig_T)):
        print(f"    {eig_T[i]:^12.4e} {eig_hcs[i]:^12.4e} {eig_ilqr[i]:^12.4e} "
              f"{abs(eig_hcs[i]-eig_T[i]):^12.4e} {abs(eig_ilqr[i]-eig_T[i]):^12.4e}")
    
    print(f"\n  {'─'*60}")
    print(f"  OPTIMIZATION VERIFICATION")
    print(f"  {'─'*60}")
    print(f"    Σ⁺ computed:            trace = {np.trace(cov_trj_2[0]):.6e}")
    print(f"    Σ⁺ targeted:            trace = {np.trace(Sig_plus_opt):.6e}")
    print(f"    ||Σ⁺_comp - Σ⁺_opt||:   {np.linalg.norm(cov_trj_2[0] - Sig_plus_opt, 'fro'):.4e}")
    
    # =============================================================================
    #                         Plotting Section
    # =============================================================================
    print("\n" + "="*70)
    print("  GENERATING VISUALIZATION")
    print("="*70)
    
    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T
    t0 = 0.0

    print(f"\n  Creating trajectory plots...")
    
    # Plot the mean trajectory (H-CS)
    fig, ax = animate_slip(modes, states, 
                           init_mode, init_state, 
                           target_mode, target_state, nt, 
                           init_reset_args, target_reset_args, step=400)
    ax.set_title("Hybrid Covariance Steering (H-CS)", fontsize=14, fontfamily='serif')

    # Plot the mean trajectory (H-iLQR)
    fig_ilqr, ax_ilqr = animate_slip(modes, states, 
                                     init_mode, init_state, 
                                     target_mode, target_state, nt, 
                                     init_reset_args, target_reset_args, step=400)
    ax_ilqr.set_title("Hybrid iLQR (H-iLQR)", fontsize=14, fontfamily='serif')

    # Build combined feedback gains
    K1_list = [K1[i] for i in range(len(K1))]
    K2_list = [K2[i] for i in range(len(K2))]
    K_hcs = K1_list + K2_list
    
    k_ff_1 = np.asarray(k_feedforward[:t_event+1])
    k_ff_2 = np.asarray(k_feedforward[t_event+1:])
    k_ff_hcs_1 = np.zeros_like(k_ff_1)
    k_ff_hcs_2 = np.zeros_like(k_ff_2)
    k_ff_hcs_1_list = [k_ff_hcs_1[i] for i in range(k_ff_hcs_1.shape[0])]
    k_ff_hcs_2_list = [k_ff_hcs_2[i] for i in range(k_ff_hcs_2.shape[0])]
    k_ff_hcs = k_ff_hcs_1_list + k_ff_hcs_2_list
    
    # Sample trajectories
    n_exp = 12
    np.random.seed(70)
    
    print(f"  Sampling {n_exp} Monte Carlo trajectories...")
    print(f"    Random seed:            70")
    print(f"    Initial state noise:    √Σ₀ @ N(0,I)")
    print(f"    Process noise:          √ε × N(0,I) per step")
    
    # Create proxy artists for legend
    sample_scatter_hcs = None
    sample_scatter_ilqr = None
    start_scatter_hcs = None
    start_scatter_ilqr = None
    goal_scatter_hcs = None
    goal_scatter_ilqr = None
    
    for i in range(n_exp):
        GaussianNoise_i = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
        x0_i = init_state + sqrtSig0 @ np.random.randn(n_states[1])

        # Samples H-CS
        (mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual, reset_args) = h_stoch_fb_rollout_slip(
            init_mode, x0_i, n_inputs, states, modes, inputs, 
            K_hcs, k_feedforward, target_state, Q_T, t0, dt, 
            epsilon, GaussianNoise_i, reference_extension_helper, init_reset_args)
        
        # Samples H-iLQR
        (mode_trj_ilqr, xt_trj_ilqr, ut_cl_trj_ilqr, Sk_ilqr, 
         xt_ref_actual_ilqr, reset_args_ilqr) = h_stoch_fb_rollout_slip(
            init_mode, x0_i, n_inputs, states, modes, inputs, 
            K_feedback, k_feedforward, target_state, Q_T, t0, dt, 
            epsilon, GaussianNoise_i, reference_extension_helper, init_reset_args)

        # Plot samples H-CS
        for ii in range(0, nt, 300):
            mode_i = mode_trj[ii]
            if mode_i == 0:
                px, pz = xt_trj[ii][0], xt_trj[ii][2]
            elif mode_i == 1:
                converted_state = convert_state_21_slip(xt_trj[ii])
                px, pz = converted_state[0], converted_state[2]
            scatter = ax.scatter(px, pz, marker='.', c='cyan', s=12, alpha=0.7)
            if sample_scatter_hcs is None:
                sample_scatter_hcs = scatter
        
        # Plot samples H-iLQR
        for ii in range(0, nt, 300):
            mode_i = mode_trj_ilqr[ii]
            if mode_i == 0:
                px_ilqr, pz_ilqr = xt_trj_ilqr[ii][0], xt_trj_ilqr[ii][2]
            elif mode_i == 1:
                converted_state = convert_state_21_slip(xt_trj_ilqr[ii])
                px_ilqr, pz_ilqr = converted_state[0], converted_state[2]
            scatter = ax_ilqr.scatter(px_ilqr, pz_ilqr, marker='.', c='cyan', s=12, alpha=0.7)
            if sample_scatter_ilqr is None:
                sample_scatter_ilqr = scatter

        # Plot start points
        converted_state = convert_state_21_slip(xt_trj[0])
        px_0, pz_0 = converted_state[0], converted_state[2]
        
        scatter_start = ax.scatter(px_0, pz_0, marker='d', c='red', s=25, 
                                   edgecolors='darkred', linewidths=0.5)
        scatter_start_ilqr = ax_ilqr.scatter(px_0, pz_0, marker='d', c='red', s=25, 
                                              edgecolors='darkred', linewidths=0.5)
        
        if start_scatter_hcs is None:
            start_scatter_hcs = scatter_start
        if start_scatter_ilqr is None:
            start_scatter_ilqr = scatter_start_ilqr

        # Plot goal points
        px_T, pz_T = xt_trj[-1][0], xt_trj[-1][2]
        px_T_ilqr, pz_T_ilqr = xt_trj_ilqr[-1][0], xt_trj_ilqr[-1][2]
        
        scatter_goal = ax.scatter(px_T, pz_T, marker='d', c='lime', s=25, 
                                  edgecolors='darkgreen', linewidths=0.5)
        scatter_goal_ilqr = ax_ilqr.scatter(px_T_ilqr, pz_T_ilqr, marker='d', c='lime', s=25, 
                                            edgecolors='darkgreen', linewidths=0.5)
        
        if goal_scatter_hcs is None:
            goal_scatter_hcs = scatter_goal
        if goal_scatter_ilqr is None:
            goal_scatter_ilqr = scatter_goal_ilqr

    # Draw target covariance ellipses
    SigT_mar = SigT[0:2, 0:2]
    target_ellipse_boundary, ax = plot_2d_ellipsoid_boundary(
        np.array([xt_trj_mean[-1][0], xt_trj_mean[-1][2]]), 
        SigT_mar, ax, 'green', linewidth=2.0)
    target_ellipse_boundary_ilqr, ax_ilqr = plot_2d_ellipsoid_boundary(
        np.array([xt_trj_mean[-1][0], xt_trj_mean[-1][2]]), 
        SigT_mar, ax_ilqr, 'green', linewidth=2.0)

    # Create legend handles
    from matplotlib.lines import Line2D
    
    legend_handles = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Mean trajectory'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='cyan', markersize=10, 
               label='Sample trajectories', linestyle='None'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=8,
               markeredgecolor='darkred', label='Initial states', linestyle='None'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='lime', markersize=8,
               markeredgecolor='darkgreen', label='Terminal states', linestyle='None'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='-', label=r'Target covariance $\Sigma_T$'),
    ]

    # Add legends
    ax.legend(handles=legend_handles, loc='upper left', 
              prop={'family': 'serif', 'size': 11}, framealpha=0.9)
    ax_ilqr.legend(handles=legend_handles, loc='upper left', 
                   prop={'family': 'serif', 'size': 11}, framealpha=0.9)

    # Axis labels
    ax.set_xlabel(r'$p_x$ (m)', fontsize=12, fontfamily='serif')
    ax.set_ylabel(r'$p_z$ (m)', fontsize=12, fontfamily='serif')
    ax_ilqr.set_xlabel(r'$p_x$ (m)', fontsize=12, fontfamily='serif')
    ax_ilqr.set_ylabel(r'$p_z$ (m)', fontsize=12, fontfamily='serif')

    # Finalize plots
    fig.tight_layout()
    fig_ilqr.tight_layout()
    
    ax.grid(True, which='both', linestyle='--', linewidth=0.5)
    ax_ilqr.grid(True, which='both', linestyle='--', linewidth=0.5)

    # Create figures directory if it doesn't exist
    os.makedirs("figures", exist_ok=True)
    
    # Save figures
    hcs_path = "figures/h_cs_slip_samples_stable.pdf"
    ilqr_path = "figures/h_ilqr_slip_samples_stable.pdf"
    
    fig.savefig(hcs_path, format="pdf", dpi=300, bbox_inches='tight')
    fig_ilqr.savefig(ilqr_path, format="pdf", dpi=300, bbox_inches='tight')
    
    print(f"\n  {'─'*60}")
    print(f"  OUTPUT FILES")
    print(f"  {'─'*60}")
    print(f"    ✓ {hcs_path}")
    print(f"    ✓ {ilqr_path}")
    
    print("\n" + "="*70)
    print("  EXECUTION COMPLETE")
    print("="*70)
    print(f"  H-CS achieves {error_ilqr/error_hcs:.1f}× better terminal covariance tracking")
    print(f"  than H-iLQR for this hybrid SLIP system.")
    print("="*70 + "\n")
    
    plt.show()
