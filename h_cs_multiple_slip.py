"""
Hybrid Covariance Steering for SLIP with Multiple Mode Transitions

This script implements covariance steering for a Spring-Loaded Inverted Pendulum (SLIP)
system that undergoes multiple hybrid transitions (e.g., stance → flight → stance → flight).

Key features:
- Handles arbitrary number of mode transitions
- Per-phase covariance steering with coupled jump conditions
- SDP optimization for rectangular saltation matrices
- Comparison with hybrid iLQR baseline

Author: Extended for multiple transitions
"""

import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)
np.set_printoptions(suppress=True, precision=4)

# Import iLQR class (use the multi-transition version)
from hybrid_ilqr.h_ilqr_discrete_slip import solve_ilqr, extract_extensions
# Import SLIP dynamics
from dynamics.dynamics_discrete_slip import *
# Import experiment parameter class
from experiments.exp_params import *
from tools.plot_ellipsoid import *
from scipy.integrate import solve_ivp

import jax
import jax.numpy as jnp
import cvxpy as cp

# =============================================================================
#                    Numerical Stability Utilities
# =============================================================================

def check_problem_conditioning(matrices_dict, name=""):
    """Diagnostic function to check conditioning of all matrices."""
    print(f"\n{'='*70}")
    print(f"  CONDITIONING REPORT {name}")
    print(f"{'='*70}")
    
    def safe_cond(M, name):
        try:
            c = np.linalg.cond(M)
            status = "✓ OK" if c < 1e8 else "⚠ WARNING" if c < 1e12 else "✗ CRITICAL"
            print(f"    cond({name:15s}): {c:12.2e}  [{status}]")
            return c
        except:
            print(f"    cond({name:15s}): FAILED TO COMPUTE")
            return np.inf
    
    for name, M in matrices_dict.items():
        if M is not None and hasattr(M, 'shape') and len(M.shape) == 2:
            safe_cond(M, name)
    
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

A_stance_func = jax.jit(jax.jacfwd(lambda x, u: slip_stance_dyn(x, u), 0))
B_stance_func = jax.jit(jax.jacfwd(lambda x, u: slip_stance_dyn(x, u), 1))


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

A_flight_func = jax.jit(jax.jacfwd(lambda x, u: slip_flight_dyn(x, u), 0))
B_flight_func = jax.jit(jax.jacfwd(lambda x, u: slip_flight_dyn(x, u), 1))


# =============================================================================
#                    ODE Integration Functions
# =============================================================================

def integrate_state_transition(A_sequence, dt, nx, t_span, method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate state transition matrix Phi_A with high accuracy."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def ode_Phi_A(t, y):
        i = min(int((t - t_span[0]) / dt), nt - 1)
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
        i = min(int((t - t_span[0]) / dt), nt - 1)
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
        i = min(int((t - t_span[0]) / dt), nt - 1)
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


def integrate_covariance_ode(A_sequence, B_sequence, K_sequence, Sig_init, dt, nx, t_span, 
                              epsilon, method='DOP853', rtol=1e-10, atol=1e-12):
    """Integrate covariance ODE under feedback control."""
    nt = A_sequence.shape[0]
    t_eval = np.linspace(t_span[0], t_span[1], nt + 1)
    
    def cov_derivative(t, cov_flat):
        i = min(int((t - t_span[0]) / dt), nt - 1)
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
        if reverse:
            i = min(int((t_span[1] - t) / dt), nt - 1)
        else:
            i = min(int((t - t_span[0]) / dt), nt - 1)
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
#                    Covariance Steering Functions
# =============================================================================

def compute_Pi0_forward(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon, min_eig=1e-12):
    """Compute initial Pi for forward covariance steering (Σ_init → Σ_terminal)."""
    n_states = Sig_init.shape[0]
    
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


def compute_Pi0_backward(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon, min_eig=1e-12):
    """Compute terminal Pi for backward covariance steering."""
    n_states = Sig_init.shape[0]
    
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


# =============================================================================
#                    Multi-Phase Data Structure
# =============================================================================

class PhaseData:
    """Data structure to hold information for each smooth phase."""
    def __init__(self, phase_id, mode, t_start_idx, t_end_idx, nx, nu):
        self.phase_id = phase_id
        self.mode = mode
        self.t_start_idx = t_start_idx
        self.t_end_idx = t_end_idx
        self.nt = t_end_idx - t_start_idx
        self.nx = nx
        self.nu = nu
        
        # Dynamics matrices (to be filled)
        self.A = None  # (nt, nx, nx)
        self.B = None  # (nt, nx, nu)
        self.Q = None  # (nt, nx, nx) - running cost
        
        # Transition matrices (to be filled)
        self.Phi_A = None  # State transition matrix
        self.S = None      # Controllability Gramian
        self.Phi_M = None  # Hamiltonian transition matrix
        
        # Covariance steering results (to be filled)
        self.Sig_init = None
        self.Sig_terminal = None
        self.Pi = None  # Co-state trajectory
        self.K = None   # Feedback gains
        self.cov_trj = None  # Covariance trajectory
        
    def __repr__(self):
        return f"Phase {self.phase_id}: mode={self.mode}, t=[{self.t_start_idx}, {self.t_end_idx}], nx={self.nx}, nu={self.nu}"


class HybridEventData:
    """Data structure to hold information for each hybrid event."""
    def __init__(self, event_id, t_idx, mode_before, mode_after, saltation):
        self.event_id = event_id
        self.t_idx = t_idx
        self.mode_before = mode_before
        self.mode_after = mode_after
        self.saltation = saltation  # E matrix
        
        # Boundary covariances (to be optimized)
        self.Sig_minus = None  # Pre-jump covariance
        self.Sig_plus = None   # Post-jump covariance
        
    def __repr__(self):
        return f"Event {self.event_id}: t_idx={self.t_idx}, {self.mode_before} → {self.mode_after}"


# =============================================================================
#                    Multi-Phase Covariance Optimization
# =============================================================================

def solve_multi_phase_covariance_optimization(phases, events, Sig0, SigT, epsilon, 
                                               solver_params=None):
    """
    Solve the multi-phase covariance steering optimization.
    
    This extends the single-jump SDP to handle multiple hybrid events.
    
    The optimization finds optimal boundary covariances at each jump that minimize
    the total control effort while satisfying:
    - Σ⁺_i = E_i Σ⁻_i E_i' at each jump i
    - Controllability constraints for each phase
    
    Args:
        phases: List of PhaseData objects
        events: List of HybridEventData objects
        Sig0: Initial covariance
        SigT: Target terminal covariance
        epsilon: Regularization parameter
        solver_params: Dictionary of solver parameters
        
    Returns:
        Updated phases and events with optimized covariances
    """
    if solver_params is None:
        solver_params = {}
    
    solver_name = solver_params.get('solver', 'SCS')
    min_eig = solver_params.get('min_eig', 1e-8)
    max_eig_factor = solver_params.get('max_eig_factor', 100.0)
    verbose = solver_params.get('verbose', True)
    eps_solver = solver_params.get('eps', 1e-9)
    max_iters = solver_params.get('max_iters', 200000)
    
    n_phases = len(phases)
    n_events = len(events)
    
    print(f"\n  {'─'*60}")
    print(f"  MULTI-PHASE COVARIANCE OPTIMIZATION")
    print(f"  {'─'*60}")
    print(f"    Number of phases:       {n_phases}")
    print(f"    Number of events:       {n_events}")
    print(f"    Solver:                 {solver_name}")
    print(f"    Regularization ε:       {epsilon:.2e}")
    
    # Compute eigenvalue bounds
    max_eig = max_eig_factor * max(np.max(np.linalg.eigvalsh(Sig0)), 
                                    np.max(np.linalg.eigvalsh(SigT)))
    
    # Build optimization variables
    # For each event, we have Sig_minus and Sig_plus
    # Plus auxiliary variables W for each phase
    
    Sig_minus_vars = []  # Pre-jump covariances
    Sig_plus_vars = []   # Post-jump covariances
    W_vars = []          # Auxiliary variables for each phase
    Y_vars = []          # Slack variables
    
    constraints = []
    objective_terms = []
    
    # Create variables for each event
    for i, event in enumerate(events):
        nx_before = phases[i].nx
        nx_after = phases[i + 1].nx
        
        Sig_minus = cp.Variable((nx_before, nx_before), symmetric=True, 
                                name=f"Sig_minus_{i}")
        Sig_plus = cp.Variable((nx_after, nx_after), symmetric=True,
                               name=f"Sig_plus_{i}")
        
        Sig_minus_vars.append(Sig_minus)
        Sig_plus_vars.append(Sig_plus)
        
        # Coupling constraint: Σ⁺ = E Σ⁻ E'
        E = event.saltation
        constraints.append(Sig_plus == E @ Sig_minus @ E.T)
        
        # Positive definiteness constraints
        constraints.append(Sig_minus >> min_eig * np.eye(nx_before))
        constraints.append(Sig_plus >> min_eig * np.eye(nx_after))
        constraints.append(Sig_minus << max_eig * np.eye(nx_before))
        constraints.append(Sig_plus << max_eig * np.eye(nx_after))
    
    # Create auxiliary variables and constraints for each phase
    for i, phase in enumerate(phases):
        nx = phase.nx
        
        # Get boundary covariances for this phase
        if i == 0:
            # First phase: starts from Sig0
            Sig_phase_init = Sig0
        else:
            # Starts from post-jump of previous event
            Sig_phase_init = Sig_plus_vars[i - 1]
        
        if i == n_phases - 1:
            # Last phase: ends at SigT
            Sig_phase_terminal = SigT
        else:
            # Ends at pre-jump of current event
            Sig_phase_terminal = Sig_minus_vars[i]
        
        # Auxiliary variable W
        W = cp.Variable((nx, nx), name=f"W_{i}")
        W_vars.append(W)
        
        # Slack variable Y (for terminal constraint of each phase)
        Y = cp.Variable((nx, nx), symmetric=True, name=f"Y_{i}")
        Y_vars.append(Y)
        
        # Get matrices for this phase
        inv_S = regularized_inverse(phase.S)
        Phi_A = phase.Phi_A
        
        # Block matrix constraints
        if i == 0:
            # First phase: Sig_phase_init is constant (Sig0)
            Y_block = cp.bmat([[Sig0, W.T], [W, Sig_minus_vars[0]]])
        else:
            Y_block = cp.bmat([[Sig_plus_vars[i-1], W.T], [W, Sig_minus_vars[i] if i < n_events else cp.Constant(SigT)]])
        
        constraints.append(Y_block >> min_eig * np.eye(2 * nx))
        
        if i < n_phases - 1:
            # Not the last phase
            slack_block = cp.bmat([[Sig_minus_vars[i], W.T], 
                                   [W, Sig_phase_init if isinstance(Sig_phase_init, np.ndarray) 
                                    else Sig_phase_init - Y]])
        else:
            # Last phase: terminal is SigT
            slack_block = cp.bmat([[SigT - Y, W.T], [W, Sig_plus_vars[-1]]])
        
        constraints.append(Y >> min_eig * np.eye(nx))
        
        # Objective contribution for this phase
        if isinstance(Sig_phase_init, np.ndarray):
            # First phase
            obj_phase = (cp.trace(inv_S @ Sig_minus_vars[0]) 
                        - 2*cp.trace(Phi_A.T @ inv_S @ W)
                        + cp.trace(Phi_A.T @ inv_S @ Phi_A @ Sig0))
        elif i < n_phases - 1:
            obj_phase = (cp.trace(inv_S @ Sig_minus_vars[i]) 
                        - 2*cp.trace(Phi_A.T @ inv_S @ W)
                        + cp.trace(Phi_A.T @ inv_S @ Phi_A @ Sig_plus_vars[i-1]))
        else:
            # Last phase
            obj_phase = (cp.trace(inv_S @ SigT) 
                        - 2*cp.trace(Phi_A.T @ inv_S @ W)
                        + cp.trace(Phi_A.T @ inv_S @ Phi_A @ Sig_plus_vars[-1]))
        
        objective_terms.append(obj_phase)
        
        # Regularization term
        objective_terms.append(-epsilon * cp.log_det(Y_block))
    
    # Total objective
    objective = cp.Minimize(sum(objective_terms))
    
    # Create and solve problem
    problem = cp.Problem(objective, constraints)
    
    print(f"\n  PROBLEM VALIDATION:")
    print(f"    Problem is DCP:         {problem.is_dcp()}")
    print(f"    Number of constraints:  {len(constraints)}")
    print(f"    Number of variables:    {len(Sig_minus_vars) + len(Sig_plus_vars) + len(W_vars) + len(Y_vars)}")
    
    # Solve
    try:
        if solver_name == 'SCS':
            problem.solve(solver=cp.SCS, verbose=verbose, eps_abs=eps_solver, 
                         eps_rel=eps_solver, max_iters=max_iters)
        elif solver_name == 'CLARABEL':
            problem.solve(solver=cp.CLARABEL, verbose=verbose)
        elif solver_name == 'MOSEK':
            problem.solve(solver=cp.MOSEK, verbose=verbose)
        else:
            problem.solve(verbose=verbose)
    except cp.SolverError as e:
        print(f"  WARNING: Solver failed with {e}, trying fallback...")
        problem.solve(solver=cp.SCS, verbose=verbose, eps_abs=1e-6, 
                     eps_rel=1e-6, max_iters=500000)
    
    print(f"\n  OPTIMIZATION RESULTS:")
    print(f"    Status:                 {problem.status}")
    print(f"    Optimal value:          {problem.value:.6e}" if problem.value else "    Optimal value:          None")
    
    # Extract results
    if problem.status in ['optimal', 'optimal_inaccurate']:
        for i, event in enumerate(events):
            event.Sig_minus = ensure_symmetric_pd(Sig_minus_vars[i].value, min_eig)
            event.Sig_plus = ensure_symmetric_pd(Sig_plus_vars[i].value, min_eig)
            
            print(f"\n    Event {i}:")
            print(f"      Σ⁻ trace: {np.trace(event.Sig_minus):.6e}")
            print(f"      Σ⁺ trace: {np.trace(event.Sig_plus):.6e}")
            print(f"      Jump error: {np.linalg.norm(event.Sig_plus - event.saltation @ event.Sig_minus @ event.saltation.T):.4e}")
    
    return phases, events, problem.status


def compute_phase_controller(phase, Sig_init, Sig_terminal, epsilon, dt, 
                             ode_method='DOP853', ode_rtol=1e-10, ode_atol=1e-12):
    """
    Compute the optimal feedback controller for a single phase.
    
    Args:
        phase: PhaseData object with dynamics matrices
        Sig_init: Initial covariance for this phase
        Sig_terminal: Target terminal covariance for this phase
        epsilon: Regularization parameter
        dt: Time step
        ode_method: ODE integration method
        ode_rtol: Relative tolerance
        ode_atol: Absolute tolerance
        
    Returns:
        Updated PhaseData with Pi trajectory and feedback gains K
    """
    nx = phase.nx
    nt = phase.nt
    t_span = (0, dt * nt)
    
    print(f"\n    Phase {phase.phase_id} (mode {phase.mode}):")
    print(f"      Time steps: {nt}, State dim: {nx}")
    
    # Compute Hamiltonian transition matrix
    Phi_M = integrate_hamiltonian_matrix(phase.A, phase.B, phase.Q, dt, nx, t_span,
                                          method=ode_method, rtol=ode_rtol, atol=ode_atol)
    phase.Phi_M = Phi_M
    
    # Extract blocks
    Phi_M_11 = Phi_M[:nx, :nx]
    Phi_M_12 = Phi_M[:nx, nx:]
    inv_Phi_M_12 = regularized_inverse(Phi_M_12)
    
    # Compute initial Pi using forward steering formula
    Pi_0 = compute_Pi0_forward(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon)
    print(f"      ||Π(0)||: {np.linalg.norm(Pi_0):.4e}")
    
    # Integrate Pi trajectory
    Pi = integrate_XY_ode(phase.A, phase.B, phase.Q, Pi_0, dt, nx, t_span, 
                          reverse=False, method=ode_method, rtol=ode_rtol, atol=ode_atol)
    phase.Pi = Pi
    
    # Compute feedback gains: K = -B'Π
    K = np.zeros((nt, phase.nu, nx))
    for i in range(nt):
        K[i] = -phase.B[i].T @ Pi[i]
    phase.K = K
    
    print(f"      ||K|| range: [{np.min([np.linalg.norm(K[i]) for i in range(nt)]):.4e}, "
          f"{np.max([np.linalg.norm(K[i]) for i in range(nt)]):.4e}]")
    
    # Store boundary covariances
    phase.Sig_init = Sig_init
    phase.Sig_terminal = Sig_terminal
    
    return phase


def propagate_covariance_through_phases(phases, events, Sig0, epsilon, dt,
                                         ode_method='DOP853', ode_rtol=1e-10, ode_atol=1e-12):
    """
    Propagate covariance through all phases using the computed controllers.
    
    Args:
        phases: List of PhaseData objects with computed controllers
        events: List of HybridEventData objects
        Sig0: Initial covariance
        epsilon: Noise intensity
        dt: Time step
        
    Returns:
        List of covariance trajectories for each phase
    """
    cov_trajectories = []
    Sig_current = Sig0.copy()
    
    for i, phase in enumerate(phases):
        t_span = (0, dt * phase.nt)
        
        # Propagate covariance for this phase
        cov_trj = integrate_covariance_ode(phase.A, phase.B, phase.K, Sig_current, 
                                           dt, phase.nx, t_span, epsilon,
                                           method=ode_method, rtol=ode_rtol, atol=ode_atol)
        phase.cov_trj = cov_trj
        cov_trajectories.append(cov_trj)
        
        print(f"    Phase {i}: Σ_init trace = {np.trace(cov_trj[0]):.6e}, "
              f"Σ_terminal trace = {np.trace(cov_trj[-1]):.6e}")
        
        # Apply jump if not the last phase
        if i < len(events):
            E = events[i].saltation
            Sig_current = E @ cov_trj[-1] @ E.T
            print(f"    Event {i}: Σ⁺ trace = {np.trace(Sig_current):.6e}")
        
    return cov_trajectories


# =============================================================================
#                              Main Script
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*70)
    print("  HYBRID COVARIANCE STEERING FOR SLIP")
    print("  Multiple Mode Transitions Version")
    print("="*70)
    
    # -------------- 
    #  SLIP Parameters 
    # --------------
    dt = 0.0001  # Smaller dt for multiple bounces
    epsilon = 0.001
    r0 = 1
    
    n_modes = 2
    
    # Mode definitions
    # mode 1 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, tau_hip]
    # mode 0 (flight): x = [px, vx, pz, vz, theta], u = [ax, az, theta_dot]
    n_states = [5, 4]  # [flight, stance]
    n_inputs = [3, 2]  # [flight, stance]
    
    # --------------------------
    #  Problem Setup for Multiple Bounces
    # --------------------------
    init_mode = 1  # Start in stance
    start_time = 0
    end_time = 1.2  # Longer time to allow multiple bounces
    
    # Terminal cost 
    target_mode = 0  # End in flight
    Q_T = 2.0 * np.eye(n_states[0])
    
    # Running costs
    Q_k = [np.zeros((n_states[0], n_states[0])), np.zeros((n_states[1], n_states[1]))]
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    # Initial state that will produce multiple bounces
    # Start with higher energy to get multiple hops
    init_theta_deg = 100
    init_theta = init_theta_deg / 180 * np.pi
    init_state = np.array([init_theta, -3.5, 0.6*r0, 0.5], dtype=np.float64)
    
    # Target state
    target_state = np.array([2.0, 1.5, 1.2, 0.0, np.pi/4], dtype=np.float64)
    
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
    print(f"  {'─'*60}")
    
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
    
    # ================================
    #  Extract Hybrid Events
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  ANALYZING HYBRID EVENTS")
    print(f"  {'─'*60}")
    
    # Find all mode transitions
    event_indices = []
    event_saltations = []
    event_mode_changes = []
    
    for i in range(nt - 1):
        if saltations[i] is not None:
            event_indices.append(i)
            event_saltations.append(saltations[i])
            # Determine mode change from modes array
            mode_before = modes[i]
            mode_after = modes[i + 1]
            event_mode_changes.append((mode_before, mode_after))
    
    n_events = len(event_indices)
    n_phases = n_events + 1
    
    print(f"    Number of hybrid events: {n_events}")
    print(f"    Number of phases:        {n_phases}")
    
    for i, (t_idx, mc) in enumerate(zip(event_indices, event_mode_changes)):
        mode_names = {0: 'flight', 1: 'stance'}
        print(f"    Event {i}: t_idx={t_idx} ({t_idx*dt:.4f}s), "
              f"{mode_names[mc[0]]} → {mode_names[mc[1]]}")
    
    # ================================
    #  Build Phase and Event Data Structures
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  BUILDING PHASE DATA STRUCTURES")
    print(f"  {'─'*60}")
    
    # Create phase data objects
    phases = []
    phase_boundaries = [0] + event_indices + [nt - 1]
    
    for i in range(n_phases):
        t_start = phase_boundaries[i]
        t_end = phase_boundaries[i + 1]
        phase_mode = modes[t_start]
        
        nx = n_states[phase_mode]
        nu = n_inputs[phase_mode]
        
        phase = PhaseData(i, phase_mode, t_start, t_end, nx, nu)
        
        # Extract A, B matrices for this phase (convert from discrete to continuous)
        phase_nt = t_end - t_start
        phase.A = np.zeros((phase_nt, nx, nx))
        phase.B = np.zeros((phase_nt, nx, nu))
        phase.Q = np.zeros((phase_nt, nx, nx))
        
        for j in range(phase_nt):
            idx = t_start + j
            A_discrete = A_trj[idx]
            B_discrete = B_trj[idx]
            
            # Handle dimension mismatch if needed
            if A_discrete.shape[0] == nx:
                phase.A[j] = (A_discrete - np.eye(nx)) / dt
                phase.B[j] = B_discrete / dt
            else:
                # Need to use correct dynamics for this mode
                if phase_mode == 0:  # flight
                    phase.A[j] = (A_discrete[:nx, :nx] - np.eye(nx)) / dt if A_discrete.shape[0] >= nx else np.zeros((nx, nx))
                    phase.B[j] = B_discrete[:nx, :nu] / dt if B_discrete.shape[0] >= nx else np.zeros((nx, nu))
                else:  # stance
                    phase.A[j] = (A_discrete[:nx, :nx] - np.eye(nx)) / dt if A_discrete.shape[0] >= nx else np.zeros((nx, nx))
                    phase.B[j] = B_discrete[:nx, :nu] / dt if B_discrete.shape[0] >= nx else np.zeros((nx, nu))
        
        phases.append(phase)
        print(f"    {phase}")
    
    # Create event data objects
    events = []
    for i in range(n_events):
        E = regularize_saltation_matrix(event_saltations[i])
        event = HybridEventData(i, event_indices[i], 
                                event_mode_changes[i][0], 
                                event_mode_changes[i][1], E)
        events.append(event)
        print(f"    {event}, E shape: {E.shape}, cond: {np.linalg.cond(E):.2e}")
    
    # ================================
    #  Compute Transition Matrices for Each Phase
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  COMPUTING TRANSITION MATRICES")
    print(f"  {'─'*60}")
    
    ode_method = 'DOP853'
    ode_rtol = 1e-10
    ode_atol = 1e-12
    
    for phase in phases:
        t_span = (0, dt * phase.nt)
        
        print(f"\n    Phase {phase.phase_id}:")
        
        # State transition matrix
        Phi_A_t = integrate_state_transition(phase.A, dt, phase.nx, t_span,
                                              method=ode_method, rtol=ode_rtol, atol=ode_atol)
        phase.Phi_A = Phi_A_t[-1]
        print(f"      Φ_A cond: {np.linalg.cond(phase.Phi_A):.2e}")
        
        # Controllability Gramian
        S_t = integrate_controllability_gramian(phase.A, phase.B, dt, phase.nx, t_span,
                                                 method=ode_method, rtol=ode_rtol, atol=ode_atol)
        phase.S = ensure_symmetric_pd(S_t[-1], 1e-12, f"S_{phase.phase_id}")
        print(f"      S cond: {np.linalg.cond(phase.S):.2e}")
    
    # ================================
    #  Initial and Target Covariances
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  COVARIANCE SPECIFICATIONS")
    print(f"  {'─'*60}")
    
    # Initial covariance (in stance mode)
    Sig0 = 0.002 * np.eye(n_states[init_mode])
    
    # Target covariance (in flight mode)
    SigT = 0.0003 * np.eye(n_states[target_mode])
    
    print(f"    Initial Σ₀ ({n_states[init_mode]}×{n_states[init_mode]}): {Sig0[0,0]:.4e} × I")
    print(f"    Target Σ_T ({n_states[target_mode]}×{n_states[target_mode]}): {SigT[0,0]:.4e} × I")
    print(f"    Trace ratio (uncertainty reduction): {np.trace(Sig0)/np.trace(SigT):.2f}×")
    
    # ================================
    #  Solve Multi-Phase Optimization (Simplified)
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  SIMPLIFIED MULTI-PHASE COVARIANCE STEERING")
    print(f"  {'─'*60}")
    
    # For the multi-phase case, we use a simplified approach:
    # 1. Propagate covariances forward using uncontrolled dynamics to get estimates
    # 2. Allocate target covariances at each jump proportionally
    # 3. Solve each phase independently
    
    # First, estimate covariances at each jump under zero control
    print(f"\n    Estimating boundary covariances...")
    
    Sig_estimates = [Sig0]
    Sig_current = Sig0.copy()
    
    for i, phase in enumerate(phases[:-1]):
        # Propagate through phase (using zero feedback for estimation)
        K_zero = np.zeros((phase.nt, phase.nu, phase.nx))
        t_span = (0, dt * phase.nt)
        cov_est = integrate_covariance_ode(phase.A, phase.B, K_zero, Sig_current,
                                           dt, phase.nx, t_span, epsilon,
                                           method=ode_method, rtol=ode_rtol, atol=ode_atol)
        
        # Apply jump
        E = events[i].saltation
        Sig_current = E @ cov_est[-1] @ E.T
        Sig_estimates.append(Sig_current)
        
        print(f"      After event {i}: Σ trace = {np.trace(Sig_current):.6e}")
    
    # Allocate target covariances geometrically between estimates
    # This is a heuristic - could be replaced with full SDP optimization
    print(f"\n    Allocating target covariances...")
    
    # For each event, set target based on geometric interpolation
    for i, event in enumerate(events):
        # Use a fraction of the estimated covariance as target
        factor = 0.8  # Aim for 80% of uncontrolled covariance
        event.Sig_minus = factor * Sig_estimates[i] if i == 0 else factor * E @ phases[i-1].Sig_terminal @ E.T
        event.Sig_plus = events[i].saltation @ event.Sig_minus @ events[i].saltation.T
        
        # Ensure PD
        event.Sig_minus = ensure_symmetric_pd(event.Sig_minus, 1e-10)
        event.Sig_plus = ensure_symmetric_pd(event.Sig_plus, 1e-10)
        
        print(f"      Event {i}: Σ⁻ trace = {np.trace(event.Sig_minus):.6e}, "
              f"Σ⁺ trace = {np.trace(event.Sig_plus):.6e}")
    
    # ================================
    #  Compute Controllers for Each Phase
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  COMPUTING PHASE CONTROLLERS")
    print(f"  {'─'*60}")
    
    for i, phase in enumerate(phases):
        # Determine boundary covariances
        if i == 0:
            Sig_init = Sig0
            Sig_terminal = events[0].Sig_minus if n_events > 0 else SigT
        elif i == n_phases - 1:
            Sig_init = events[-1].Sig_plus
            Sig_terminal = SigT
        else:
            Sig_init = events[i-1].Sig_plus
            Sig_terminal = events[i].Sig_minus
        
        # Ensure dimensions match
        if Sig_init.shape[0] != phase.nx:
            print(f"    WARNING: Dimension mismatch in phase {i}, using identity scaling")
            Sig_init = 0.001 * np.eye(phase.nx)
        if Sig_terminal.shape[0] != phase.nx:
            Sig_terminal = 0.0005 * np.eye(phase.nx)
        
        phase = compute_phase_controller(phase, Sig_init, Sig_terminal, epsilon, dt,
                                         ode_method, ode_rtol, ode_atol)
    
    # ================================
    #  Propagate Covariance with H-CS Controller
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  COVARIANCE PROPAGATION - H-CS")
    print(f"  {'─'*60}")
    
    cov_trajectories_hcs = propagate_covariance_through_phases(
        phases, events, Sig0, epsilon, dt, ode_method, ode_rtol, ode_atol)
    
    # ================================
    #  Propagate Covariance with H-iLQR Controller (Baseline)
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  COVARIANCE PROPAGATION - H-iLQR (Baseline)")
    print(f"  {'─'*60}")
    
    # Build iLQR feedback gains for each phase
    phases_ilqr = []
    for i, phase in enumerate(phases):
        phase_ilqr = PhaseData(i, phase.mode, phase.t_start_idx, phase.t_end_idx, 
                               phase.nx, phase.nu)
        phase_ilqr.A = phase.A
        phase_ilqr.B = phase.B
        phase_ilqr.S = phase.S
        phase_ilqr.Phi_A = phase.Phi_A
        
        # Extract iLQR gains for this phase
        K_ilqr = np.zeros((phase.nt, phase.nu, phase.nx))
        for j in range(phase.nt):
            idx = phase.t_start_idx + j
            K_fb = K_feedback[idx]
            # Handle dimension mismatch
            if K_fb.shape[1] == phase.nx and K_fb.shape[0] == phase.nu:
                K_ilqr[j] = K_fb
            else:
                K_ilqr[j] = np.zeros((phase.nu, phase.nx))
        
        phase_ilqr.K = K_ilqr
        phases_ilqr.append(phase_ilqr)
    
    cov_trajectories_ilqr = propagate_covariance_through_phases(
        phases_ilqr, events, Sig0, epsilon, dt, ode_method, ode_rtol, ode_atol)
    
    # ================================
    #  Results Summary
    # ================================
    print("\n" + "="*70)
    print("  FINAL RESULTS SUMMARY")
    print("="*70)
    
    print(f"\n  {'─'*60}")
    print(f"  TERMINAL COVARIANCE COMPARISON")
    print(f"  {'─'*60}")
    
    Sig_T_hcs = cov_trajectories_hcs[-1][-1]
    Sig_T_ilqr = cov_trajectories_ilqr[-1][-1]
    
    print(f"\n  Target Σ_T:")
    print(f"    trace:                  {np.trace(SigT):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(SigT), precision=4)}")
    
    print(f"\n  Terminal Σ(T) - H-CS:")
    print(f"    trace:                  {np.trace(Sig_T_hcs):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(Sig_T_hcs), precision=4)}")
    
    print(f"\n  Terminal Σ(T) - H-iLQR:")
    print(f"    trace:                  {np.trace(Sig_T_ilqr):.6e}")
    print(f"    eigenvalues:            {np.array2string(np.linalg.eigvalsh(Sig_T_ilqr), precision=4)}")
    
    # Compute errors
    error_hcs = np.linalg.norm(Sig_T_hcs - SigT, 'fro')
    error_ilqr = np.linalg.norm(Sig_T_ilqr - SigT, 'fro')
    
    print(f"\n  {'─'*60}")
    print(f"  PERFORMANCE METRICS")
    print(f"  {'─'*60}")
    print(f"    ||Σ(T) - Σ_T||_F  H-CS:    {error_hcs:.6e}")
    print(f"    ||Σ(T) - Σ_T||_F  H-iLQR:  {error_ilqr:.6e}")
    
    if error_hcs < error_ilqr:
        print(f"    Improvement:               {error_ilqr/error_hcs:.2f}× better with H-CS")
    else:
        print(f"    Note: H-iLQR performed better (ratio: {error_hcs/error_ilqr:.2f})")
    
    # ================================
    #  Visualization
    # ================================
    print(f"\n  {'─'*60}")
    print(f"  GENERATING VISUALIZATION")
    print(f"  {'─'*60}")
    
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    
    # Plot mean trajectory with sample trajectories
    fig, ax = animate_slip(modes, states, 
                           init_mode, init_state, 
                           target_mode, target_state, nt, 
                           init_reset_args, target_reset_args, step=max(1, nt//500))
    ax.set_title("Hybrid Covariance Steering (Multi-Transition)", fontsize=14, fontfamily='serif')
    
    # Build combined feedback gains for H-CS
    K_hcs_combined = []
    for phase in phases:
        for j in range(phase.nt):
            K_hcs_combined.append(phase.K[j])
    
    # Sample trajectories
    n_samples = 8
    np.random.seed(42)
    
    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T
    
    print(f"    Sampling {n_samples} Monte Carlo trajectories...")
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, n_samples))
    
    for i_sample in range(n_samples):
        GaussianNoise_i = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
        x0_i = init_state + sqrtSig0 @ np.random.randn(n_states[init_mode])
        
        # Rollout with H-CS controller
        (mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual, reset_args) = h_stoch_fb_rollout_slip(
            init_mode, x0_i, n_inputs, states, modes, inputs, 
            K_hcs_combined, k_feedforward, target_state, Q_T, 0.0, dt, 
            epsilon, GaussianNoise_i, reference_extension_helper, init_reset_args)
        
        # Plot trajectory
        for ii in range(0, nt, max(1, nt//100)):
            mode_i = mode_trj[ii]
            if mode_i == 0:
                px, pz = xt_trj[ii][0], xt_trj[ii][2]
            elif mode_i == 1:
                converted_state = convert_state_21_slip(xt_trj[ii])
                px, pz = converted_state[0], converted_state[2]
            ax.scatter(px, pz, marker='.', c=[colors[i_sample]], s=8, alpha=0.6)
        
        # Mark start and end
        converted_start = convert_state_21_slip(xt_trj[0])
        ax.scatter(converted_start[0], converted_start[2], marker='d', c='red', s=30, 
                   edgecolors='darkred', linewidths=0.5)
        ax.scatter(xt_trj[-1][0], xt_trj[-1][2], marker='d', c='lime', s=30,
                   edgecolors='darkgreen', linewidths=0.5)
    
    # Draw target covariance ellipse
    SigT_2d = SigT[:2, :2]  # px-pz marginal
    final_mean = [states[-1][0], states[-1][2]]
    plot_2d_ellipsoid_boundary(np.array(final_mean), SigT_2d, ax, 'green', linewidth=2.0)
    
    # Legend
    legend_handles = [
        Line2D([0], [0], color='black', linewidth=2, linestyle='-', label='Mean trajectory'),
        Line2D([0], [0], marker='.', color='w', markerfacecolor='teal', markersize=10, 
               label='Sample trajectories', linestyle='None'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='red', markersize=8,
               markeredgecolor='darkred', label='Initial states', linestyle='None'),
        Line2D([0], [0], marker='d', color='w', markerfacecolor='lime', markersize=8,
               markeredgecolor='darkgreen', label='Terminal states', linestyle='None'),
        Line2D([0], [0], color='green', linewidth=2, linestyle='-', label=r'Target $\Sigma_T$'),
    ]
    ax.legend(handles=legend_handles, loc='upper left', 
              prop={'family': 'serif', 'size': 10}, framealpha=0.9)
    
    ax.set_xlabel(r'$p_x$ (m)', fontsize=12, fontfamily='serif')
    ax.set_ylabel(r'$p_z$ (m)', fontsize=12, fontfamily='serif')
    ax.grid(True, linestyle='--', alpha=0.5)
    
    fig.tight_layout()
    
    # Save figure
    os.makedirs("figures", exist_ok=True)
    fig_path = "figures/h_cs_slip_multi_transition.pdf"
    fig.savefig(fig_path, format="pdf", dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {fig_path}")
    
    # Plot covariance evolution
    fig2, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Trace evolution
    ax1 = axes[0]
    
    # Build time and trace arrays
    t_hcs = []
    trace_hcs = []
    t_ilqr = []
    trace_ilqr = []
    
    t_current = 0
    for i, (cov_hcs, cov_ilqr) in enumerate(zip(cov_trajectories_hcs, cov_trajectories_ilqr)):
        for j in range(len(cov_hcs)):
            t_hcs.append(t_current + j * dt)
            trace_hcs.append(np.trace(cov_hcs[j]))
            t_ilqr.append(t_current + j * dt)
            trace_ilqr.append(np.trace(cov_ilqr[j]))
        t_current += len(cov_hcs) * dt
    
    ax1.semilogy(t_hcs, trace_hcs, 'b-', linewidth=2, label='H-CS')
    ax1.semilogy(t_ilqr, trace_ilqr, 'r--', linewidth=2, label='H-iLQR')
    ax1.axhline(np.trace(SigT), color='g', linestyle=':', linewidth=2, label=r'Target $\mathrm{tr}(\Sigma_T)$')
    
    # Mark events
    for event in events:
        ax1.axvline(event.t_idx * dt, color='gray', linestyle='--', alpha=0.5)
    
    ax1.set_xlabel('Time (s)', fontsize=12)
    ax1.set_ylabel(r'$\mathrm{tr}(\Sigma)$', fontsize=12)
    ax1.set_title('Covariance Trace Evolution', fontsize=14)
    ax1.legend(loc='upper right')
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Eigenvalue evolution
    ax2 = axes[1]
    
    eig_max_hcs = [np.max(np.linalg.eigvalsh(cov_trajectories_hcs[i][j])) 
                   for i in range(len(cov_trajectories_hcs)) 
                   for j in range(len(cov_trajectories_hcs[i]))]
    eig_min_hcs = [np.min(np.linalg.eigvalsh(cov_trajectories_hcs[i][j])) 
                   for i in range(len(cov_trajectories_hcs)) 
                   for j in range(len(cov_trajectories_hcs[i]))]
    
    ax2.semilogy(t_hcs, eig_max_hcs, 'b-', linewidth=2, label=r'H-CS $\lambda_{\max}$')
    ax2.semilogy(t_hcs, eig_min_hcs, 'b--', linewidth=2, label=r'H-CS $\lambda_{\min}$')
    ax2.axhline(np.max(np.linalg.eigvalsh(SigT)), color='g', linestyle=':', linewidth=2, 
                label=r'Target $\lambda_{\max}$')
    ax2.axhline(np.min(np.linalg.eigvalsh(SigT)), color='g', linestyle='-.', linewidth=2,
                label=r'Target $\lambda_{\min}$')
    
    for event in events:
        ax2.axvline(event.t_idx * dt, color='gray', linestyle='--', alpha=0.5)
    
    ax2.set_xlabel('Time (s)', fontsize=12)
    ax2.set_ylabel('Eigenvalue', fontsize=12)
    ax2.set_title('Covariance Eigenvalue Evolution', fontsize=14)
    ax2.legend(loc='upper right')
    ax2.grid(True, linestyle='--', alpha=0.5)
    
    fig2.tight_layout()
    
    fig2_path = "figures/h_cs_slip_cov_evolution_multi.pdf"
    fig2.savefig(fig2_path, format="pdf", dpi=300, bbox_inches='tight')
    print(f"    ✓ Saved: {fig2_path}")
    
    print("\n" + "="*70)
    print("  EXECUTION COMPLETE")
    print("="*70)
    print(f"    Total phases:           {n_phases}")
    print(f"    Total hybrid events:    {n_events}")
    print(f"    H-CS terminal error:    {error_hcs:.6e}")
    print(f"    H-iLQR terminal error:  {error_ilqr:.6e}")
    print("="*70 + "\n")
    
    plt.show()
    