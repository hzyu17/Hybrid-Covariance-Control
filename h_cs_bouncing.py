import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
hilqr_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(hilqr_dir, '..'))
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
    dt = 0.0015
    epsilon = 1.0

    dt_shrink = 0.9
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    Q_T = 60*np.eye(n_states[0])
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
    
    flow_dynamics = [sym_dyn_bouncing, sym_dyn_bouncing]
    
    n_samples = 0
    exp_params.update_params(n_modes, init_mode, target_mode, n_states, init_state, target_state, 
                             start_time, end_time, dt, initial_guess, 
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
     k_feedforward,K_feedback,
     A_trj,B_trj,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result

    show_hilqr_results = True
    if show_hilqr_results:
        time_span = np.arange(start_time, end_time, dt).flatten()
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, trj_labels='iLQG-stochastic')
        plt.show()
    
    # ==================== covariance steering ====================

    (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref, 
             v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref, 
             v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref) = extract_extensions(reference_extension_helper)
    
    print("v_tevents_ref: ", v_tevents_ref)

    # The event time at which the system enters mode 2 from mode 1.
    t_event = v_tevents_ref[0]

    Sig0 = 0.5*np.eye(n_states[0])
    SigT = 0.1*np.eye(n_states[0])

    E_linear = np.array([[1.0, 0.0], [0.0, -0.6]], dtype=np.float64)

    A0 = np.array([[0, 1], [0, 0]], dtype=np.float64)
    B0 = np.array([[0],[1.0]], dtype=np.float64)
    Q0 = np.zeros((2, 2))

    A = np.tile(A0, (nt, 1, 1))
    B = np.tile(B0, (nt, 1, 1))
    Q = np.tile(Q0, (nt, 1, 1))
    
    # ======================================================
    #           Convex Optimization Formulation
    # ======================================================
    import cvxpy as cp

    nx1, nx2 = n_states[0], n_states[1]
    nt1, nt2 = t_event, nt-t_event-2
    A1, A2 = A[0:t_event], A[t_event+1:]
    B1, B2 = B[0:t_event], B[t_event+1:]
    

    # --------------------- compute Phi^{A_j} --------------------

    t_span1 = (0, dt * nt1)
    t_span2 = (0, dt * nt2)

    t_eval1 = np.linspace(0, dt*nt1, nt1+1)
    t_eval2 = np.linspace(0, dt*nt2, nt2+1)

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
        dydt = PhiA1_i @ B1_i @ B1_i.T @ PhiA1_i.T
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
        dydt = PhiA2_i @ B2_i @ B2_i.T @ PhiA2_i.T
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
    
    obj_1 = cp.trace(inv_S1@Sighat_minus) - 2*cp.trace(Phi2.T@inv_S2@W2) - 2*cp.trace(Phi1.T@inv_S1@W1) + cp.trace(Phi2.T@inv_S2@Phi2@Sighat_plus)
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

    # ======================================================
    #         Closed-form solution for invertible E
    # ======================================================
    M = np.zeros((nt, 2*n_states[0], 2*n_states[0]), dtype=np.float64)

    for i in range(nt):
        top_row = np.concatenate((A[i], -B[i] @ B[i].T), axis=1)
        bottom_row = np.concatenate((-Q[i], -A[i].T), axis=1)
        M[i] = np.concatenate((top_row, bottom_row), axis=0)

    # the Phi at hybrid event
    Phi_miuns = np.zeros((2*n_states[0], 2*n_states[0]), dtype=np.float64)
    Phi_miuns_top_row = np.concatenate((E_linear, np.zeros((n_states[0], n_states[0]))), axis=1)
    Phi_miuns_bottom_row = np.concatenate((np.zeros((n_states[0], n_states[0])), np.linalg.inv(E_linear).T), axis=1)
    Phi_miuns = np.concatenate((Phi_miuns_top_row, Phi_miuns_bottom_row), axis=0)
    
    # integrate Phi, corresponding to M
    Phi = np.eye(2*n_states[0])
    for i in range(0, t_event):
        Phi = Phi + dt * (M[i] @ Phi)

    Phi = Phi_miuns@Phi

    for i in range(t_event+1, nt-1):
        Phi = Phi + dt * (M[i] @ Phi)

    Phi_11 = Phi[0:n_states[0], 0:n_states[0]]
    Phi_12 = Phi[0:n_states[0], n_states[0]:]
    Phi_21 = Phi[n_states[0]:, 0:n_states[0]]
    Phi_22 = Phi[n_states[0]:, n_states[0]:]

    I = np.eye(n_states[0])

    inv_Phi12 = np.linalg.solve(Phi_12, np.eye(n_states[0]))
    invSig0 = np.linalg.solve(Sig0, np.eye(n_states[0]))

    eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
    sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

    tmp = epsilon**2 * I/4 + sqrtSig0 @ inv_Phi12 @ SigT @ inv_Phi12.T @ sqrtSig0
    tmp = (tmp + tmp.T) / 2
        
    eval_tmp, evec_tmp = np.linalg.eigh(tmp)
    sqrt_tmp = evec_tmp @ np.diag(np.sqrt(eval_tmp)) @ evec_tmp.T

    # ==================== Solve for Pi(t) ====================
    Pi0 = epsilon*invSig0/2 - inv_Phi12@Phi_11 - sqrtInvSig0@sqrt_tmp@sqrtInvSig0

    Pi = np.zeros((nt, n_states[0], n_states[0]), dtype=np.float64)
    Pi[0] = (Pi0 + Pi0.T) / 2

    v_XY = np.zeros((nt, 2*n_states[0], n_states[0]), dtype=np.float64)
    v_XY[0, :n_states[0], :n_states[0]] = np.eye(n_states[0])
    v_XY[0, n_states[0]:, :n_states[0]] = Pi[0]

    for i in range(t_event):
        dXY = M[i]@v_XY[i]
        next_XY = v_XY[i] + dXY*dt
        dXY_next = M[i+1]@next_XY
        
        v_XY[i+1] = v_XY[i] + (dXY + dXY_next)*(dt/2.0) 
        X_next = v_XY[i+1,:n_states[0],:n_states[0]]
        Y_next = v_XY[i+1,n_states[0]:,:n_states[0]]
        inv_X_next = np.linalg.solve(X_next, np.eye(n_states[0]))
        Pi[i+1] = Y_next@inv_X_next

    Pi[t_event+1] = np.linalg.inv(E_linear).T@Pi[t_event]@np.linalg.inv(E_linear)
    v_XY[t_event+1] = Phi_miuns@v_XY[t_event]

    for i in range(t_event+1, nt-1):
        dXY = M[i]@v_XY[i]
        next_XY = v_XY[i] + dXY*dt
        dXY_next = M[i+1]@next_XY
        
        v_XY[i+1] = v_XY[i] + (dXY + dXY_next)*(dt/2.0) 
        X_next = v_XY[i+1,:n_states[0],:n_states[0]]
        Y_next = v_XY[i+1,n_states[0]:,:n_states[0]]
        inv_X_next = np.linalg.solve(X_next, np.eye(n_states[0]))
        Pi[i+1] = Y_next@inv_X_next

    K = np.zeros((nt, n_inputs[0], n_states[0]), dtype=np.float64)

    print("============== Pi results ==============")
    print("Pi(t-): ")
    print(Pi[t_event])
    print("Pi(t+): ")
    print(Pi[t_event+1])
    print("E'@Pi(t+)@E")
    print(E_linear.T@Pi[t_event+1]@E_linear)
    
    for i in range(nt):
        K[i] = -B[i].T @ Pi[i]

    # ========================= compute the controlled covariances =========================
    cov_trj = np.zeros((nt, n_states[0], n_states[0]))
    cov_trj[0] = Sig0

    for i in range(0, t_event):
        Acl_i = A[i] + B[i]@K[i]
        cov_trj[i+1] = cov_trj[i] + (Acl_i@cov_trj[i] + cov_trj[i]@Acl_i.T + epsilon*B[i]@B[i].T) * dt
    
    print("----------------- Sigma_minus computed -----------------")
    print(cov_trj[t_event])

    # hybrid time
    cov_trj[t_event+1] = E_linear@cov_trj[t_event]@E_linear.T

    print("----------------- Sigma_plus computed -----------------")
    print(cov_trj[t_event+1])

    for i in range(t_event+1, nt-1):
        Acl_i = A[i] + B[i]@K[i]
        cov_trj[i+1] = cov_trj[i] + (Acl_i@cov_trj[i] + cov_trj[i]@Acl_i.T + epsilon*B[i]@B[i].T) * dt

    # ========================= controlled covariances i-LQG =========================
    K_ilQG = np.asarray(K_feedback)
    cov_trj_lqg = np.zeros((nt, n_states[0], n_states[0]))
    cov_trj_lqg[0] = Sig0

    for i in range(0, t_event):
        Acl_i = A[i] + B[i]@K_ilQG[i]
        cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + B[i]@B[i].T) * dt
    
    # hybrid time
    cov_trj_lqg[t_event+1] = E_linear@cov_trj_lqg[t_event]@E_linear.T

    for i in range(t_event+1, nt-1):
        Acl_i = A[i] + B[i]@K_ilQG[i]
        cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + B[i]@B[i].T) * dt


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

    # plot covariance trajecotry
    for i in range(0, nt, 10):
        ellipse_boundary_lqg, ax1 = plot_2d_ellipsoid_boundary(states[i], cov_trj_lqg[i], ax1, 'k')

    for i in range(0, nt, 10):
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
                init_ellipse_boundary, target_ellipse_boundary], [r'$3-\sigma$ H-Covariance Steering', 
                                                                  r'Initial Mean',
                                                                  r'Target Mean',
                                                                  r'Initial covariance $\Sigma_0$', 
                                                                  r'Target covariance $\Sigma_T$'], prop={'family': 'serif', 'size': 15})
    
    fig1.tight_layout()
    fig2.tight_layout()
    fig2.savefig(hilqr_dir+'/covariance_steering_bouncing_hcovariancesteering.pdf', dpi=2000)
    fig1.savefig(hilqr_dir+'/covariance_steering_bouncing_hilqr.pdf', dpi=2000)
    plt.show()

    # ---------------------- Plot the controlled trajectories ----------------------
    eig_0 = np.sqrt(0.5)
    eig_T = np.sqrt(0.1)

    fig3, ax3 = plt.subplots()
    fig4, ax4 = plt.subplots()

    for i_sample in range(100):
        x0_i = init_state + sqrtSig0@np.random.randn(nx1)

        GaussianNoises = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
        mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual = stochastic_feedback_rollout_bouncing(init_mode, x0_i, n_inputs, states, ref_modechanges, 
                                                                                                inputs, K_ilQG, k_feedforward, target_state, Q_T, 0.0, end_time, 
                                                                                                epsilon, GaussianNoises, dt_shrink, 
                                                                                                reference_extension_helper, init_reset_args)

        xt_trj_arr = np.asarray(xt_trj)
        ax3.plot(time_span, xt_trj_arr[:, 0], 'b', alpha=0.5, linewidth=0.5)
        ax4.plot(time_span, xt_trj_arr[:, 1], 'b', alpha=0.5, linewidth=0.5)

    ax3.set_xlabel(r'Time $t$', fontproperties=font_props)
    ax3.set_ylabel(r'Position $z(t)$', fontproperties=font_props)

    ax4.set_xlabel(r'Time $t$', fontproperties=font_props)
    ax4.set_ylabel(r'Velocity $\dot z(t)$', fontproperties=font_props)

    ax3.grid(True)
    ax4.grid(True)

    ax3.plot([0.0, 0.0], [init_state[0] - eig_0, init_state[0] + eig_0], color='red', linewidth=6)
    ax3.plot([time_span[-1], time_span[-1]], [target_state[0] - eig_T, target_state[0] + eig_T], color='green', linewidth=6)

    ax4.plot([0.0, 0.0], [init_state[1] - eig_0, init_state[1] + eig_0], color='red', linewidth=6)
    ax4.plot([time_span[-1], time_span[-1]], [target_state[1] - eig_T, target_state[1] + eig_T], color='green', linewidth=6)

    plt.show()
