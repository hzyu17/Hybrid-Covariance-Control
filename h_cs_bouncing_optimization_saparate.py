import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
hcs_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(hcs_dir, '..'))
sys.path.append(root_dir)

import jax.numpy as jnp

# Import iLQR class and reference extension handler
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr, extract_extensions
from experiments.exp_params import *
from dynamics.ode_solver.dynamics_bouncing import *
from dynamics.dynamics_discrete_bouncing import *
from tools.plot_ellipsoid import *

from matplotlib.font_manager import FontProperties
font_props = FontProperties(family='serif', size=20, weight='normal')
from scipy.integrate import solve_ivp


if __name__=='__main__':
    # ======================== H-iLQR ===========================
    n_exp = 1
        
    # Set desired state
    n_modes = 2

    # the state and control dimensions, mode-dependent
    n_states = [2, 2]
    n_inputs = [1, 1]

    # ------------------ one bounce --------------------
    dt = 0.001
    epsilon = 0.5

    dt_shrink = 0.9
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    Q_T = 25*np.eye(n_states[0])
    init_state = np.array([5, 1.5], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    target_state = np.array([2.5, 0], dtype=np.float64)  # Swing pendulum upright

    init_mode = 0
    target_mode = 0
    initial_guess = [0.5*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.5*np.ones((np.shape(time_span)[0],n_inputs[1]))]

    # ---------------- / bouncing example -----------------

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.zeros((n_states[0],n_states[0]), dtype=np.float64), np.zeros((n_states[1],n_states[1]), dtype=np.float64)] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0], dtype=np.float64), np.eye(n_inputs[1], dtype=np.float64)]
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ============================================================================================================
    #                                       Solve for hybrid ilqg proposal
    # ============================================================================================================
    exp_params = ExpParams()
    
    flow_dynamics = [symbolic_dynamics_bouncing, symbolic_dynamics_bouncing]
    
    n_samples = 0
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, dt_shrink, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, flow_dynamics, 
                             event_detect_bouncing_discrete, 
                             plot_bouncingball, 
                             convert_state_21_bouncing, 
                             init_reset_args, target_reset_args)
    exp_data = ExpData(exp_params)
    
    print("===================== Solving for h-iLQG proposal controller =====================")
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan,modes,states,inputs, saltations,
     k_feedforward,K_feedback,A_trj,B_trj,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result

    show_hilqr_results = True
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, trj_labels='iLQG-stochastic')
        plt.show()

    (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref, 
    v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref, 
    v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref) = extract_extensions(reference_extension_helper)
    
    print("v_tevents_ref: ", v_tevents_ref)

    # The event time at which the system enters mode 2 from mode 1.
    t_event = v_tevents_ref[0]
    mode_1 = 0
    mode_2 = 1
    
    # ==================== covariance steering ====================

    Sig0 = 0.2*np.eye(n_states[0])
    SigT = 0.05*np.eye(n_states[0])

    E_linear = np.array([[1.0, 0.0], [0.0, -0.6]], dtype=np.float64)

    A0 = np.array([[0, 1], [0, 0]], dtype=np.float64)
    B0 = np.array([[0],[1.0]], dtype=np.float64)
    Q0 = np.zeros((2, 2))

    A = np.tile(A0, (nt, 1, 1))
    B = np.tile(B0, (nt, 1, 1))
    Q = np.tile(Q0, (nt, 1, 1))


    # ========================= controlled covariances i-LQG =========================
    K_ilQG = np.asarray(K_feedback)
    cov_trj_lqg = np.zeros((nt, n_states[0], n_states[0]))
    cov_trj_lqg[0] = Sig0

    for i in range(0, t_event):
        Acl_i = A[i] + B[i]@K_ilQG[i]
        cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + epsilon*B[i]@B[i].T) * dt
    
    # hybrid time
    cov_trj_lqg[t_event+1] = E_linear@cov_trj_lqg[t_event]@E_linear.T

    for i in range(t_event+1, nt-1):
        Acl_i = A[i] + B[i]@K_ilQG[i]
        cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + epsilon*B[i]@B[i].T) * dt

    
    # ======================================================
    #           Convex Optimization Formulation
    # ======================================================
    import cvxpy as cp

    nx1, nx2 = n_states[0], n_states[1]
    nt1, nt2 = t_event+1, nt-t_event-1

    A1 = np.zeros((nt1, nx1, nx1))
    B1 = np.zeros((nt1, nx1, n_inputs[mode_1]))
    Q1 = np.zeros((nt1, nx1, nx1))

    for i in range(nt1):

        A1_i = A_trj[i]
        B1_i = B_trj[i]
        
        A1[i] = (A1_i-np.eye(nx1,nx1))/dt
        B1[i] = B1_i/dt

    A2 = np.zeros((nt2, nx2, nx2))
    B2 = np.zeros((nt2, nx2, n_inputs[mode_2]))
    Q2 = np.zeros((nt2, nx2, nx2))

    for i in range(nt2):
        ii = t_event+i+1

        A2_i = A_trj[ii]
        B2_i = B_trj[ii]

        A2[i] = (A2_i - np.eye(nx2,nx2)) / dt
        B2[i] = B2_i / dt

    # A1, A2 = A[0:t_event+1], A[t_event+1:]
    # B1, B2 = B[0:t_event+1], B[t_event+1:]

    Q1 = np.zeros((nt1, nx1, nx1))
    Q2 = np.zeros((nt2, nx2, nx2))

    # --------------------- compute Phi^{A_j} --------------------

    mode_1 = 0
    mode_2 = 1

    t_span1 = (0, dt * nt1)
    t_span2 = (0, dt * nt2)

    t_eval1 = np.linspace(0, dt*nt1, nt1+1)
    t_eval2 = np.linspace(0, dt*nt2, nt2+1)

    t_span1_reverse = (dt * nt1, 0)
    t_span2_reverse = (dt * nt2, 0)

    t_eval1_reverse = np.linspace(dt*nt1, 0, nt1+1)
    t_eval2_reverse = np.linspace(dt*nt2, 0, nt2+1)

    # --------------------- compute Phi^{A_j} --------------------
    Phi_A1 = np.zeros((nt1+1, nx1, nx1))
    Phi_A1_0 = np.eye(nx1).flatten()

    def ode_Phi_A1(t, y):
        i = min(int(t / dt), nt1-1)  
        A1_i = A1[i]
        Phi_reshaped = y.reshape((nx1, nx1))
        dydt = A1_i @ Phi_reshaped
        return dydt.flatten()  
    
    # Solve ODE
    result_PhiA1 = solve_ivp(ode_Phi_A1, t_span1, Phi_A1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    Phi_A1 = result_PhiA1.y.reshape((nx1, nx1, nt1+1))
    Phi_A1 = np.moveaxis(Phi_A1, 2, 0)
    Phi1 = Phi_A1[-1]

    def ode_Phi_A2(t, y):
        i = min(int(t / dt), nt2-1)  
        A2_i = A2[i]
        Phi_reshaped = y.reshape((nx2, nx2))
        dydt = A2_i @ Phi_reshaped
        return dydt.flatten()  
    
    Phi_A2 = np.zeros((nt2+1, nx2, nx2))
    Phi_A2_0 = np.eye(nx2).flatten()
    result_PhiA2 = solve_ivp(ode_Phi_A2, t_span2, Phi_A2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    Phi_A2 = result_PhiA2.y.reshape((nx2, nx2, nt2+1))
    Phi_A2 = np.moveaxis(Phi_A2, 2, 0)
    Phi2 = Phi_A2[-1]

    # --------------------- compute S_1, S_2 --------------------
    def ode_Phi_S1(t, y):
        i = min(int(t / dt), nt1-1)  
        PhiA1_i = Phi_A1[i]
        B1_i = B1[i]
        dydt = epsilon*PhiA1_i @ B1_i @ B1_i.T @ PhiA1_i.T
        return dydt.flatten()  
    
    S1_0 = np.zeros((nx1, nx1)).flatten()

    # Solve ODE
    result_S1 = solve_ivp(ode_Phi_S1, t_span1, S1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    S1_t = result_S1.y.reshape((nx1, nx1, nt1+1))
    S1_t = np.moveaxis(S1_t, 2, 0)
    S1 = S1_t[-1]

    inv_S1 = np.linalg.inv(S1)

    def ode_Phi_S2(t, y):
        i = min(int(t / dt), nt2-1)  
        PhiA2_i = Phi_A2[i]
        B2_i = B2[i]
        dydt = epsilon*PhiA2_i @ B2_i @ B2_i.T @ PhiA2_i.T
        return dydt.flatten()  
    
    S2_0 = np.zeros((nx2, nx2)).flatten()

    # Solve ODE
    result_S2 = solve_ivp(ode_Phi_S2, t_span2, S2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    S2_t = result_S2.y.reshape((nx2, nx2, nt2+1))
    S2_t = np.moveaxis(S2_t, 2, 0)
    S2 = S2_t[-1]
    inv_S2 = np.linalg.inv(S2)

    # --------------------- optimization formulation ---------------------
    # ---------- Declare variables ---------- 
    Sighat_minus, Sighat_plus = cp.Variable((nx1,nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)
    W1, W2  = cp.Variable((nx1,nx1)), cp.Variable((nx2,nx2))
    Y1, Y2 = cp.Variable((2*nx1,2*nx1)), cp.Variable((nx2,nx2))

    E = E_linear
    I = np.eye(nx2)
    
    Y1 = cp.bmat([[Sig0, W1.T], [W1, Sighat_minus]])
    slack_Y2 = cp.bmat([[Sighat_plus, W2.T], [W2, SigT-Y2]])
    
    obj_1 = cp.trace(inv_S1@Sighat_minus)/epsilon - 2*cp.trace(Phi2.T@inv_S2@W2)/epsilon - 2*cp.trace(Phi1.T@inv_S1@W1)/epsilon + cp.trace(Phi2.T@inv_S2@Phi2@Sighat_plus)/epsilon
    obj_2 = - cp.log_det(Y1) - cp.log_det(Y2)

    constraints = [Sighat_plus==E@Sighat_minus@E.T,
                    Y1>>0,
                    slack_Y2>>0,
                    Sighat_minus>>0,
                    Sighat_plus>>0
                    ]
    
    problem = cp.Problem(cp.Minimize(obj_1+obj_2), constraints)
    print("problem is DCP:", problem.is_dcp())

    problem.solve()
    Sig_minus_opt = Sighat_minus.value
    Sig_plus_opt = Sighat_plus.value

    print("Sig_minus_opt: ")
    print(Sig_minus_opt)
    print("Sig_plus_opt: ")
    print(Sig_plus_opt)
    print("E@Sigma_minus@E': ")
    print(E_linear@Sig_minus_opt@E_linear.T)


    # ==============================================================
    #         Solve the optimal covariance steering controller 
    # ==============================================================
    def compute_Pi0(Sig0, SigT, Phi_M_11, inv_Phi_M_12, epsilon):
        n_states = Sig0.shape[0]
        eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
        sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

        invSig0 = np.linalg.inv(Sig0)
        eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
        sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

        tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ SigT @ inv_Phi_M_12.T @ sqrtSig0
        tmp = (tmp + tmp.T) / 2
        
        eval_tmp, evec_tmp = np.linalg.eigh(tmp)
        sqrt_tmp = evec_tmp @ np.diag(np.sqrt(eval_tmp)) @ evec_tmp.T

        # ==================== Solve for Pi(t) ====================
        return epsilon*invSig0/2 - inv_Phi_M_12@Phi_M_11 - sqrtInvSig0@sqrt_tmp@sqrtInvSig0

    def compute_Pi0_reverse(Sig0, SigT, Phi_M_11, inv_Phi_M_12, epsilon):
        n_states = Sig0.shape[0]
        eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
        sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

        invSig0 = np.linalg.inv(Sig0)
        eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
        sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

        tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ SigT @ inv_Phi_M_12.T @ sqrtSig0
        tmp = (tmp + tmp.T) / 2
        
        eval_tmp, evec_tmp = np.linalg.eigh(tmp)
        sqrt_tmp = evec_tmp @ np.diag(np.sqrt(eval_tmp)) @ evec_tmp.T

        # ==================== Solve for Pi(t) ====================
        return epsilon*invSig0/2 - inv_Phi_M_12@Phi_M_11 + sqrtInvSig0@sqrt_tmp@sqrtInvSig0


    # ----------------------- [0, t^-] -----------------------
    
    # =================
    #  Solve \Phi ODE
    # =================
    
    def compute_M1(t_idx):
        i = min(int(t_idx / dt), nt1 - 1)  # Convert continuous time to nearest time index, cap at nt-1
        top_row = np.concatenate((A1[i], -B1[i] @ B1[i].T), axis=1)
        bottom_row = np.concatenate((-Q1[i], -A1[i].T), axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)
    
    def ode_system_Phi_M1(t, y):
        y_reshaped = y.reshape((2*nx1, 2*nx1))
        dydt = compute_M1(t) @ y_reshaped
        return dydt.flatten()  # Flatten because solve_ivp expects a flat array

    # Initial condition
    PhiM1_0 = np.eye(2 * nx1).flatten()  # Flatten the initial condition into a 1D array

    # Call the solver
    result = solve_ivp(ode_system_Phi_M1, t_span1, PhiM1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    Phi_M1 = result.y.reshape((2*nx1, 2*nx1, nt1+1))[:,:,-1]

    # Reversed transition kernel
    # Phi_M1 = np.linalg.inv(Phi_M1)
    
    Phi_M1_12 = Phi_M1[:nx1, nx1:]
    Phi_M1_11 = Phi_M1[:nx1, :nx1]

    # ====================
    #  End Solve \Phi ODE
    # ====================
    inv_Phi_M1_12 = np.linalg.inv(Phi_M1_12)
    Sig1_0 = Sig0
    Sig1_T = Sig_minus_opt

    Pi1_0 = compute_Pi0(Sig1_0, Sig1_T, Phi_M1_11, inv_Phi_M1_12, epsilon)

    # ==============
    #  Solve for XY
    # ==============
    def ode_system_M1_XY(t, y):
        y_reshaped = y.reshape((2*nx1, nx1))
        dydt = compute_M1(t) @ y_reshaped
        return dydt.flatten()  # Flatten because solve_ivp expects a flat array

    # Solve the ODE
    v_XY_M1_0 = np.zeros((2 * nx1, nx1))
    v_XY_M1_0[:nx1, :nx1] = np.eye(nx1)
    v_XY_M1_0[nx1:, :nx1] = (Pi1_0 + Pi1_0.T) / 2

    # Flatten initial conditions for use with solve_ivp
    v_XY_M1_0_flat = v_XY_M1_0.flatten()
    # result_M1_XY = solve_ivp(ode_system_M1_XY, t_span1_reverse, v_XY_M1_0_flat, method='RK23', t_eval=t_eval1_reverse)
    result_M1_XY = solve_ivp(ode_system_M1_XY, t_span1, v_XY_M1_0_flat, method='RK23', t_eval=t_eval1)


    # Reshape the result to retrieve v_XY at each time step
    v_XY_M1_solution = result_M1_XY.y.reshape((2 * nx1, nx1, -1))

    Pi1 = np.zeros((nt1, nx1, nx1), dtype=np.float64)
    Pi1[0] = (Pi1_0 + Pi1_0.T) / 2

    for i in range(nt1):
        X_i = v_XY_M1_solution[:nx1,:nx1,i]
        Y_i = v_XY_M1_solution[nx1:,:nx1,i]
        inv_Xi = np.linalg.inv(X_i)
        Pi1[i] = Y_i @ inv_Xi
        
    # Pi1 = Pi1[::-1,:,:]
    
    K1 = np.zeros((nt1, n_inputs[mode_1], nx1), dtype=np.float64)
    for i in range(nt1):
        B1_i = B1[i]
        K1[i] = -B1_i.T @ Pi1[i]

    # ===================
    #  End Solve for Pi
    # ===================

    # ------------- Covariance propagation ODE [0,t^-] -------------
    def cov1_derivative(t, cov_flat):
        i = min(int(t / dt), nt1 - 1)

        # Reshape the flattened covariance back to a matrix
        cov_j = cov_flat.reshape((nx1, nx1))
        
        # Compute A1_i and B1_i
        A1_i = A1[i]
        B1_i = B1[i]
        
        # Get K1 at time t
        K1_i = K1[i]
        
        # Compute Acl_i
        Acl_i = A1_i + B1_i @ K1_i
        
        d_cov_j_dt = Acl_i @ cov_j + cov_j @ Acl_i.T + epsilon*B1_i @ B1_i.T
        
        return d_cov_j_dt.flatten()

    cov1_0_flat = Sig0.flatten()
    # cov1_0_flat = Sig_minus_opt.flatten()
    solution_cov = solve_ivp(fun=cov1_derivative, t_span=t_span1, y0=cov1_0_flat, t_eval=t_eval1, method='RK23')
    
    cov_trj_1 = solution_cov.y.reshape((nx1, nx1, -1))
    cov_trj_1 = np.moveaxis(cov_trj_1, 2, 0)
    cov_trj_1 = cov_trj_1[:-1,:,:]

    # cov_trj_1 = cov_trj_1[::-1,:,:]
    print("------------------ Sigma_minus computed by solving covariance control in [0, t^-] ------------------")
    print(cov_trj_1[-1])
    print("Sigma plus integrated")
    print(E_linear@cov_trj_1[-1]@E_linear.T)

    # ======================================== [t^+, T] ========================================

    # =================
    #  Solve \Phi ODE
    # =================
    
    def compute_M2(t_idx):
        i = min(int(t_idx / dt), nt2 - 1)  # Convert continuous time to nearest time index, cap at nt-1
        top_row = np.concatenate((A2[i], -B2[i] @ B2[i].T), axis=1)
        bottom_row = np.concatenate((-Q2[i], -A2[i].T), axis=1)
        return np.concatenate((top_row, bottom_row), axis=0)
    
    def ode_system_PhiM2(t, y):
        y_reshaped = y.reshape((2*nx2, 2*nx2))
        dydt = compute_M2(t) @ y_reshaped
        return dydt.flatten()  

    # Initial condition
    Phi_M2_0 = np.eye(2 * nx2).flatten()  

    # Call the solver
    result_Phi_M2 = solve_ivp(ode_system_PhiM2, t_span2, Phi_M2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    Phi_M2 = result_Phi_M2.y.reshape((2*nx2, 2*nx2, nt2+1))[:,:,-1]
    
    Phi_M2 = np.linalg.inv(Phi_M2)

    Phi_M2_12 = Phi_M2[:nx2, nx2:]
    Phi_M2_11 = Phi_M2[:nx2, :nx2]
    inv_Phi_M2_12 = np.linalg.inv(Phi_M2_12)

    # ====================
    #  End Solve \Phi ODE
    # ====================
    Sig_minus = cov_trj_1[-1]
    # ======================================== Jumping time t^- to t^+ ========================================
    Sigma_plus = E_linear@Sig_minus@E_linear.T
    
    print("Sig_minus optimized: ")
    print(np.array2string(Sig_minus_opt, precision=4))
    
    Sig2_0 = SigT
    Sig2_T = Sig_plus_opt

    Pi2_0 = compute_Pi0_reverse(Sig2_0, Sig2_T, Phi_M2_11, inv_Phi_M2_12, epsilon)

    # ==============
    #  Solve for XY
    # ==============
    def ode_system_XY_M2(t, y):
        y_reshaped = y.reshape((2*nx2, nx2))
        dydt = compute_M2(t) @ y_reshaped
        return dydt.flatten()  
    
    # ------------------- Solve the ODE for Phi_M ------------------- 
    v_XY_M2_0 = np.zeros((2 * nx2, nx2))
    v_XY_M2_0[:nx2, :nx2] = np.eye(nx2)
    v_XY_M2_0[nx2:, :nx2] = (Pi2_0 + Pi2_0.T) / 2

    # Flatten initial conditions for use with solve_ivp
    v_XY_M2_0_flat = v_XY_M2_0.flatten()
    result = solve_ivp(ode_system_XY_M2, t_span2_reverse, v_XY_M2_0_flat, method='RK23', t_eval=t_eval2_reverse)

    # Reshape the result to retrieve v_XY at each time step
    v_XY_M2_t = result.y.reshape((2 * nx2, nx2, -1))

    Pi2 = np.zeros((nt2, nx2, nx2), dtype=np.float64)
    Pi2[0] = (Pi2_0 + Pi2_0.T) / 2
    for i in range(1, nt2):
        X2_i = v_XY_M2_t[:nx2, :nx2, i]
        Y2_i = v_XY_M2_t[nx2:, :nx2, i]
        inv_X2_i = np.linalg.solve(X2_i, np.eye(nx2))
        Pi2[i] = Y2_i @ inv_X2_i
        
    Pi2 = Pi2[::-1,:,:]
    # ===================
    #  End Solve for Pi2
    # ===================

    K2 = np.zeros((nt2, n_inputs[mode_2], nx2), dtype=np.float64)
    for i in range(nt2):
        B2_i = B2[i]
        K2[i] = -B2_i.T @ Pi2[i]

    print("============== Pi results ==============")
    print("Pi(t-): ")
    print(Pi1[-1])
    print("Pi(t+): ")
    print(Pi2[0])
    print("E'Pi(t+)E: ")
    print(E_linear.T@Pi2[0]@E_linear)

    # ========================= compute the controlled covariances =========================

    # ------------- Covariance propagation ODE in [t^+, T] -------------
    def cov_derivative_2(t, cov_flat):
        i = min(int(t / dt), nt2 - 1)  # Convert continuous time to nearest time index, cap at nt-1

        # Reshape the flattened covariance back to a matrix
        cov_j = cov_flat.reshape((nx2, nx2))
        
        # Compute A1_i and B1_i
        A2_i = A2[i]
        B2_i = B2[i]
        
        # Get K1 at time t
        K2_i = K2[i]
        
        # Compute Acl_i
        A2cl_i = A2_i + B2_i @ K2_i
        
        # Compute the derivative of the covariance matrix
        d_cov_j_dt = A2cl_i @ cov_j + cov_j @ A2cl_i.T + epsilon*B2_i @ B2_i.T
        
        # Flatten the derivative matrix back to a vector
        return d_cov_j_dt.flatten()

    # hybrid time
    # cov2_0 = E_linear@cov_trj_1[-1]@E_linear.T
    cov2_0 = Sig_plus_opt

    cov2_0_flat = cov2_0.flatten()
    solution_cov2 = solve_ivp(
        fun=cov_derivative_2,
        t_span=t_span2,
        y0=cov2_0_flat,
        t_eval=t_eval2,
        method='RK23' 
    )
    cov_trj_2 = solution_cov2.y.reshape((nx2, nx2, -1))
    cov_trj_2 = np.moveaxis(cov_trj_2, 2, 0)
    cov_trj_2 = cov_trj_2[:-1,:,:]
    
    print("==================== Terminal covariance computed ==================")
    print(cov_trj_2[-1])

    # =======================================
    #                Plotting
    # =======================================
    _, _, fig1, ax1 = plot_bouncingball(time_span, modes, 
                                        states, inputs, 
                                        init_state, target_state, 
                                        nt, ref_reset_args,
                                        trj_labels='LQG Mean',
                                        plot_start_goal=False)
    
    _, _, fig2, ax2 = plot_bouncingball(time_span, modes, 
                                        states, inputs, 
                                        init_state, target_state, 
                                        nt, ref_reset_args,
                                        trj_labels='LQG Mean',
                                        plot_start_goal=False)

    cov_trj = np.concatenate((cov_trj_1, cov_trj_2), axis=0)

    # plot covariance trajecotry
    for i in range(0, nt, 40):
        ellipse_boundary_lqg, ax1 = plot_2d_ellipsoid_boundary(states[i], cov_trj_lqg[i], ax1, 'k')

    for i in range(0, nt, 40):
        ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(states[i], cov_trj[i], ax2, 'b')

    # ---------------------- Plot the start and goal states ----------------------
    scatter_init_ax1 = ax1.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6)
    scatter_target_ax1 = ax1.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6)

    scatter_init_ax2 = ax2.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6)
    scatter_target_ax2 = ax2.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6)

    init_ellipse_boundary, ax1 = plot_2d_ellipsoid_boundary(init_state, Sig0, ax1, 'r', linewidth=5.0)
    target_ellipse_boundary, ax1 = plot_2d_ellipsoid_boundary(target_state, SigT, ax1, 'g', linewidth=5.0)

    init_ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(init_state, Sig0, ax2, 'r', linewidth=5.0)
    target_ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(target_state, SigT, ax2, 'g', linewidth=5.0)

    Sigma_minus_ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(states[t_event], Sig_minus_opt, ax2, 'k', linewidth=5.0)
    Sigma_plus_ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(states[t_event+1], Sig_plus_opt, ax2, 'c', linewidth=5.0)

    ax1.set_xlim(-5, 10)
    ax1.set_ylim(-15, 10)
    ax2.set_xlim(-5, 10)
    ax2.set_ylim(-15, 10)

    ax1.set_xlabel(r'$z$', fontproperties=font_props)
    ax1.set_ylabel(r'$\dot z$', fontproperties=font_props)
    ax1.legend([ellipse_boundary_lqg, scatter_init_ax1, scatter_target_ax1,
                init_ellipse_boundary, target_ellipse_boundary], [r'$3-\sigma$ H-iLQR',
                                                                  r'Initial Mean',
                                                                  r'Target Mean',
                                                                    r'Initial covariance $\Sigma_0$', 
                                                                    r'Target covariance $\Sigma_T$'], prop={'family': 'serif', 'size': 15})

    ax2.set_xlabel(r'$z$', fontproperties=font_props)
    ax2.set_ylabel(r'$\dot z$', fontproperties=font_props)
    ax2.legend([ellipse_boundary, scatter_init_ax2, scatter_target_ax2,
                init_ellipse_boundary, target_ellipse_boundary,
                Sigma_minus_ellipse_boundary, Sigma_plus_ellipse_boundary
                ], [r'$3-\sigma$ H-Covariance Steering', 
                    r'Initial Mean',
                    r'Target Mean',
                    r'Initial covariance $\Sigma_0$', 
                    r'Target covariance $\Sigma_T$',
                    r'$\Sigma^{-}$',
                    r'$\Sigma^{+}$'], 
                    prop={'family': 'serif', 'size': 15})
    
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig2.savefig(hcs_dir+'/covariance_steering_bouncing_hcs.pdf', dpi=2000)
    fig1.savefig(hcs_dir+'/covariance_steering_bouncing_hilqr.pdf', dpi=2000)
    plt.show()

    save_cov_trj = True
    K_hcs = np.concatenate((K1,K2),axis=0)
    if save_cov_trj:

        # Sample trajectories
        np.random.seed(60)
        eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
        sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T
        dt_shrinkrate = 0.9
        n_samples = 10
        xt_trj_samples = np.zeros((n_samples, nt, nx1))
        for i_s in range(n_samples):
            GaussianNoise_i = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
            x0_i = init_state + sqrtSig0@np.random.randn(n_states[1])
            (mode_trj, 
            xt_trj, 
            ut_cl_trj, 
            Sk, 
            xt_ref_actual, 
            reset_args) = hybrid_stochastic_feedback_rollout_discrete_bouncing(init_mode, x0_i, n_inputs, states, modes, 
                                                                                inputs, K_hcs, k_feedforward, target_state, 
                                                                                Q_T, 0.0, dt, 
                                                                                epsilon, GaussianNoise_i, dt_shrinkrate, 
                                                                                reference_extension_helper, init_reset_args)
            sample_i = np.asarray(xt_trj)
            xt_trj_samples[i_s] = sample_i
        cov_trj_swapped = np.transpose(cov_trj, (1, 2, 0))
        # np.savetxt('cov_trj_bouncing.csv', cov_trj_flatten, delimiter=',')
        scipy.io.savemat(hcs_dir+'/cov_trj_bouncing.mat', {'matrix': cov_trj_swapped})
        scipy.io.savemat(hcs_dir+'/sample_trj_bouncing.mat', {'matrix': xt_trj_samples})
        np.savetxt('mean_trj_bouncing.csv', states, delimiter=',')