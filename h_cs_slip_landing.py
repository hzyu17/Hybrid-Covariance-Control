# Covariance control for SLIP system, landing from flight to stance

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
from hybrid_ilqr.h_ilqr_discrete import solve_ilqr, extract_extensions
# Import SLIP dynamics
from dynamics.dynamics_discrete_slip import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import experiment parameter class
from experiments.exp_params import *
from tools.plot_ellipsoid import *
from scipy.integrate import solve_ivp


import jax
import jax.numpy as jnp

def slip_stance_dyn(x, u):
    # inputs = Matrix([u1, u2])
    # states = Matrix([theta, theta_dot, r, r_dot])   
    g = 9.81
    k = 25.0
    m = 0.5
    r0 = 1
    return jnp.array([x[1], 
                      -2*x[1]*x[3]/x[2]-g*jnp.cos(x[0])/x[2], 
                      x[3] + u[1]/m/x[2]/x[2], 
                      k/m*(r0-x[2]) - g*jnp.sin(x[0]) + x[1]*x[1]*x[2] + k*u[0]/m
                      ])

A1_func = jax.jit(jacfwd(lambda x, u: slip_stance_dyn(x, u), 0))
B1_func = jax.jit(jacfwd(lambda x, u: slip_stance_dyn(x, u), 1))


def slip_flight_dyn(x, u):
    # inputs = Matrix([u1,u2,u3])
    # states = Matrix([x, x_dot, z, z_dot, theta])
    g = 9.81

    return jnp.array([x[1], 
                      u[0], 
                      x[3], 
                      u[1]-g,
                      u[2]
                      ])

A2_func = jax.jit(jacfwd(lambda x, u: slip_flight_dyn(x, u), 0))
B2_func = jax.jit(jacfwd(lambda x, u: slip_flight_dyn(x, u), 1))


def linear_stochastic_integration_euler_SLIP(x0, A, B, u, dt, eps, dW):   
    xt_next = x0 + (A@x0 + B@u)*dt + np.sqrt(eps) * B@dW
    return xt_next

def linear_hybrid_stochastic_integration_euler(xt, At, Bt, 
                                               current_mode, ut, 
                                                randN, eps, 
                                                dt, dt_shrink, t0, reset_arg):
    dW = np.sqrt(dt) * randN
    
    # Linear
    xt_next = linear_stochastic_integration_euler_SLIP(xt, At, Bt, ut, dt, eps, dW)

    args_guard = (xt, current_mode, ut, t0, xt_next, dt, dt_shrink, randN, eps, reset_arg)
    
    if (current_mode==0):
        guard_hit = guard_cond_slip_12(xt, xt_next, current_mode)
    
        if guard_hit:
            xt_next, next_mode, dW, new_reset_arg = guard_true_func_slip_12(args_guard)
        else:
            xt_next, next_mode, dW, new_reset_arg = guard_false_func_slip_12(args_guard)
            
    elif (current_mode==1):
        guard_hit = guard_cond_slip_21(xt, xt_next, current_mode)
    
        if guard_hit:
            xt_next, next_mode, dW, new_reset_arg = guard_true_func_slip_21(args_guard)
        else:
            xt_next, next_mode, dW, new_reset_arg = guard_false_func_slip_21(args_guard)

    return xt_next, next_mode, dW, new_reset_arg


# Feedback rollout on linearized systems
def linearized_hybrid_stochastic_feedback_rollout_discrete(init_mode, x0, n_inputs, xt_ref, ref_modes, 
                                                            ut, Kt, kt, At, Bt, t0, dt, 
                                                            epsilon, GaussianNoise, dt_shrinkrate, 
                                                            reference_extension_helper, init_reset_args,
                                                            cond_mode_mismatch_func=cond_mode_mismatch_slip,
                                                            reaction_mode_mismatch_func=reaction_mode_mismatch_slip):

    (v_event_modechange, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, _) = extract_extensions(reference_extension_helper, start_index = 0)

    # ----------------------- Construct the linearized dynamics around the extended states ----------------------- 
    A1_fwd = np.zeros((nt, nx1, nx1))
    B1_fwd = np.zeros((nt, nx1, n_inputs[mode_1]))
    for i in range(nt):
        A1_fwd[i] = A1_func(v_ref_ext_fwd[0][i], inputs[mode_1][0])
        B1_fwd[i] = B1_func(v_ref_ext_fwd[0][i], inputs[mode_1][0])
    
    A2_bwd = np.zeros((nt, nx2, nx2))
    B2_bwd = np.zeros((nt, nx2, n_inputs[mode_2]))
    for i in range(nt):
        A2_bwd[i] = A2_func(v_ref_ext_bwd[0][i], inputs[mode_2][0])
        B2_bwd[i] = B2_func(v_ref_ext_bwd[0][i], inputs[mode_2][0])
    
    # ----------------------- // Construct the linearized dynamics around the extended states ----------------------- 

    n_timestamps = len(xt_ref)
    
    # Returning trajectory    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0  
    
    mode_trj = np.zeros((n_timestamps), dtype=np.int64) 
    mode_trj[0] = init_mode
    
    # Closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    cnt_mismatch = 0
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    
    # Path cost
    Sk = 0
    
    # Hybrid event related 
    cnt_event = 0
    reset_args = [np.array([0.0]) for _ in range(n_timestamps)]
    event_args = [init_reset_args[0]]
    
    # -------------- roullout function --------------
    for ii_t in range(n_timestamps-1):   
        
        xt = xt_trj[ii_t]
        current_mode = mode_trj[ii_t]
        ref_current_mode = ref_modes[ii_t]
        
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        xref_i = xt_ref[ii_t] 
        
        reset_args[ii_t] = event_args[cnt_event]

        A_i = At[ii_t]
        B_i = Bt[ii_t]
        
        # ======== Handle mode mismatch ========
        if cond_mode_mismatch_func(current_mode, ref_current_mode):
            print("mode mismatch at time: ", ii_t)
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch_func(ii_t, current_mode, ref_current_mode, 
                                                                                v_ref_ext_fwd[0], v_ref_ext_bwd[0], 
                                                                                v_event_modechange[0],
                                                                                v_Kfb_ref_ext_fwd[0], v_kff_ref_ext_fwd[0],
                                                                                v_Kfb_ref_ext_bwd[0], v_kff_ref_ext_bwd[0],
                                                                                cnt_mismatch)
            
            if current_mode == v_event_modechange[0][1]:
                A_i = A2_bwd[ii_t]
                B_i = B2_bwd[ii_t]
            elif current_mode == v_event_modechange[0][0]:
                A_i = A1_fwd[ii_t]
                B_i = B1_fwd[ii_t]

        xt_ref_actual[ii_t] = xref_i
        delta_xt_i = xt - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = current_u
        
        noise_i = GaussianNoise[current_mode][ii_t]
        dW_i = np.sqrt(dt)*noise_i
        
        # ============================== One step integration ==============================        
        xt_next, next_mode, _, new_reset_arg = linear_hybrid_stochastic_integration_euler(xt, A_i, B_i, current_mode,
                                                                                           current_u, 
                                                                                            noise_i, epsilon, 
                                                                                            dt, dt_shrinkrate, 
                                                                                            t0, reset_args[ii_t])
        
        reset_args[ii_t+1] = new_reset_arg
        
        # ============================== // One step integration // ==============================     
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual, reset_args


if __name__ == '__main__':
    # -------------- 
    #  SLIP example 
    # --------------
    dt = 0.0005
    epsilon = 0.0015
    dt_shrink = 0.9
    r0 = 1
    
    n_modes = 2
    
    # --------------
    # SLIP Dynamics
    # --------------
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    # For the slip dynamics, mode 1 has 1 input, and mode 2 has 2 inputs. 
    n_states = [5, 4]
    n_inputs = [3, 2]
    
    # ------------------------------
    #   Case 3: Landing from flight
    # ------------------------------
    init_mode = 0
    
    # Time definitions
    start_time = 0
    end_time = 0.5
    
    # Terminal cost 
    target_mode = 1
    Q_T = 20.0*np.eye(n_states[0])
    
    # Running costs
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    

    init_state = np.array([0.0, 2.25, 1.4, 0.0, 2*np.pi/3], dtype=np.float64) 
    target_state = np.array([np.pi/3, -4.0, 0.5*r0, 0.0], dtype=np.float64) # Swing pendulum upright
    
    # ---------------- / slip example -----------------
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    print("nt: ", nt)
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ================================
    #  Solve for hybrid ilqr proposal
    # ================================
    exp_params = ExpParams()
    
    initial_guess = [0.0*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.0*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    symbolic_dynamics = [symbolic_flight_dynamics_slip, symbolic_stance_dynamics_slip]
    
    # place holders
    n_exp = 1
    n_samples = 0 
    
    exp_params.update_params(n_modes, init_mode, target_mode, 
                             n_states, init_state, target_state, 
                             start_time, end_time, dt, dt_shrink, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, symbolic_dynamics, 
                             event_detect_discrete_slip, plot_slip, convert_state_21_slip, 
                             init_reset_args, target_reset_args, 
                             animate_slip)
    
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True, verbose=False)
    
    (timespan,modes,states,inputs,saltations,
     k_feedforward,K_feedback,A_trj,B_trj,
     current_cost,states_iter,
     ref_modechanges,reference_extension_helper, ref_reset_args) = hybrid_ilqr_result
    

    # # ---------------------- mean trajectory under H-iLQR ----------------------------
    # Noise_zero = [np.zeros((nt, n_inputs[0])), np.zeros((nt, n_inputs[1]))]
    # (mode_trj_mean, 
    # xt_trj_mean, 
    # ut_cl_trj_mean, 
    # Sk_mean, 
    # xt_ref_actual_mean, 
    # reset_args_mean) = hybrid_stochastic_feedback_rollout_discrete_slip(init_mode, init_state, n_inputs, states, modes, 
    #                                                                     inputs, K_feedback, k_feedforward, target_state, 
    #                                                                     Q_T, 0.0, dt, 
    #                                                                     epsilon, Noise_zero, 0.9, 
    #                                                                     reference_extension_helper, init_reset_args)
    
    exp_data.add_nominal_data(hybrid_ilqr_result)

    (v_mode_change_ref, v_ref_ext_bwd, v_ref_ext_fwd, 
    v_Kfb_ref_ext_bwd, v_Kfb_ref_ext_fwd, 
    v_kff_ref_ext_bwd, v_kff_ref_ext_fwd, v_tevents_ref) = extract_extensions(reference_extension_helper, start_index = 0)

    print("v_tevents_ref: ", v_tevents_ref)
    t_event = v_tevents_ref[0]
    mode_1 = 1
    mode_2 = 0

    # =============================================== 
    #               Covariance Steering 
    # ===============================================

    Sig0 = 0.002*np.eye(5)
    SigT = 0.0003*np.eye(4)
    # SigT[2,2] = 0.0001

    E_linear = saltations[t_event]

    print("======== E_linear ========")
    print(E_linear)

    # ======================================================
    #             Convex Optimization Formulation
    # ======================================================
    import cvxpy as cp

    nx1, nx2 = n_states[1], n_states[0]
    nt1, nt2 = t_event+1, nt-t_event-1

    A1 = np.zeros((nt1, nx1, nx1))
    B1 = np.zeros((nt1, nx1, n_inputs[mode_1]))
    Q1 = np.zeros((nt1, nx1, nx1))

    for i in range(nt1):
        # A1_i = A1_func(states[i], inputs[mode_1][i])
        # B1_i = B1_func(states[i], inputs[mode_1][i])
        # A1_i = A1_func(xt_trj_mean[i], ut_cl_trj_mean[mode_1][i])
        # B1_i = B1_func(xt_trj_mean[i], ut_cl_trj_mean[mode_1][i])

        # A1[i] = A1_i
        # B1[i] = B1_i

        A1_i = A_trj[i]
        B1_i = B_trj[i]
        
        A1[i] = (A1_i-np.eye(nx1,nx1))/dt
        B1[i] = B1_i/dt

    A2 = np.zeros((nt2, nx2, nx2))
    B2 = np.zeros((nt2, nx2, n_inputs[mode_2]))
    Q2 = np.zeros((nt2, nx2, nx2))

    for i in range(nt2):
        ii = t_event+i+1
        # A2_i = A2_func(states[ii], inputs[mode_2][ii])
        # B2_i = B2_func(states[ii], inputs[mode_2][ii])
        # A2_i = A2_func(xt_trj_mean[ii], ut_cl_trj_mean[mode_2][ii])
        # B2_i = B2_func(xt_trj_mean[ii], ut_cl_trj_mean[mode_2][ii])

        # A2[i] = A2_i
        # B2[i] = B2_i

        A2_i = A_trj[ii]
        B2_i = B_trj[ii]

        A2[i] = (A2_i - np.eye(nx2,nx2)) / dt
        B2[i] = B2_i / dt

    t_span1 = (0, dt * nt1)
    t_span2 = (0, dt * nt2)

    t_eval1 = np.linspace(0, dt*nt1, nt1+1)
    t_eval2 = np.linspace(0, dt*nt2, nt2+1)

    t_span1_reverse = (dt * nt1, 0)
    t_span2_reverse = (dt * nt2, 0)

    t_eval1_reverse = np.linspace(dt*nt1, 0, nt1+1)
    t_eval2_reverse = np.linspace(dt*nt2, 0, nt2+1)

    # --------------------- compute Phi^{A_j} --------------------
    Phi_A1_t = np.zeros((nt1+1, nx1, nx1))
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
    Phi_A1_t = result_PhiA1.y.reshape((nx1, nx1, -1))
    Phi_A1_t = np.moveaxis(Phi_A1_t, 2, 0)
    Phi_A1 = Phi_A1_t[-1]

    # --------------------- Integrate Phi_A2 ---------------------
    def ode_Phi_A2(t, y):
        i = min(int(t / dt), nt2-1)  
        A2_i = A2[i]
        Phi_reshaped = y.reshape((nx2, nx2))
        dydt = A2_i @ Phi_reshaped
        return dydt.flatten()  
    
    Phi_A2_0 = np.eye(nx2).flatten()
    result_PhiA2 = solve_ivp(ode_Phi_A2, t_span2, Phi_A2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    Phi_A2_t = result_PhiA2.y.reshape((nx2, nx2, -1))
    Phi_A2_t = np.moveaxis(Phi_A2_t, 2, 0)
    Phi_A2 = Phi_A2_t[-1]

    # --------------------- compute S_1, S_2 --------------------
    def ode_Phi_S1(t, y):
        i = min(int(t / dt), nt1-1)  
        A1_i = A1[i]
        B1_i = B1[i]
        S1_reshaped = y.reshape((nx1, nx1))
        dydt = B1_i@B1_i.T + A1_i @ S1_reshaped + S1_reshaped@A1_i.T

        return dydt.flatten()  
    
    S1_0 = np.zeros((nx1, nx1)).flatten()
    # Solve ODE
    result_S1 = solve_ivp(ode_Phi_S1, t_span1, S1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    S1_t = result_S1.y.reshape((nx1, nx1, -1))
    S1_t = np.moveaxis(S1_t, 2, 0)
    S1 = S1_t[-1]

    inv_S1 = np.linalg.inv(S1)

    def ode_Phi_S2(t, y):
        i = min(int(t / dt), nt2-1)  
        A2_i = A2[i]
        B2_i = B2[i]
        S2_reshaped = y.reshape((nx2, nx2))
        dydt = B2_i@B2_i.T + A2_i @ S2_reshaped  + S2_reshaped@A2_i.T 
        return dydt.flatten()  
    
    S2_0 = np.zeros((nx2, nx2)).flatten()
    # Solve ODE
    result_S2 = solve_ivp(ode_Phi_S2, t_span2, S2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    S2_t = result_S2.y.reshape((nx2, nx2, -1))
    S2_t = np.moveaxis(S2_t, 2, 0)
    S2 = S2_t[-1]
    inv_S2 = np.linalg.inv(S2)

    # --------------------- optimization formulation ---------------------
    # ---------- Declare variables ---------- 
    Sighat_minus, Sighat_plus = cp.Variable((nx1,nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)
    W1, W2  = cp.Variable((nx1,nx1)), cp.Variable((nx2,nx2))
    Y1, Y2 = cp.Variable((2*nx1,2*nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)

    E = E_linear
    
    Y1 = cp.bmat([[Sig0, W1.T], [W1, Sighat_minus]])
    slack_Y2 = cp.bmat([[Sighat_plus, W2.T], [W2, SigT-Y2]])
    
    obj_1 = cp.trace(inv_S1@Sighat_minus) - 2*cp.trace(Phi_A2.T@inv_S2@W2) - 2*cp.trace(Phi_A1.T@inv_S1@W1) + cp.trace(Phi_A2.T@inv_S2@Phi_A2@Sighat_plus)
    obj_2 = - epsilon*cp.log_det(Y1) - epsilon*cp.log_det(Y2)

    constraints = [Sighat_plus==E@Sighat_minus@E.T,
                    Y1>>0,
                    slack_Y2>>0,
                    Sighat_minus>>0,
                    Sighat_plus>>0
                    ]
    
    problem = cp.Problem(cp.Minimize(obj_1+obj_2), constraints)
    print(" -------------- Problem is DCP -------------- :", problem.is_dcp())

    problem.solve()
    Sig_minus_opt = Sighat_minus.value
    Sig_plus_opt = Sighat_plus.value

    # ==============================================================
    #        Solve the optimal covariance steering controller 
    # ==============================================================
    def compute_Pi0(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon):
        n_states = Sig_init.shape[0]
        eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig_init)
        sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

        invSig0 = np.linalg.inv(Sig_init)
        eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
        sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

        tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ Sig_terminal @ inv_Phi_M_12.T @ sqrtSig0
        tmp = (tmp + tmp.T) / 2
        
        eval_tmp, evec_tmp = np.linalg.eigh(tmp)
        sqrt_tmp = evec_tmp @ np.diag(np.sqrt(eval_tmp)) @ evec_tmp.T

        # ==================== Solve for Pi(t) ====================
        return epsilon*invSig0/2 - inv_Phi_M_12@Phi_M_11 - sqrtInvSig0@sqrt_tmp@sqrtInvSig0

    def compute_Pi0_reverse(Sig_init, Sig_terminal, Phi_M_11, inv_Phi_M_12, epsilon):
        n_states = Sig_init.shape[0]
        eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig_init)
        sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T

        invSig0 = np.linalg.inv(Sig_init)
        eval_invSig0, evec_invSig0 = np.linalg.eigh(invSig0)
        sqrtInvSig0 = evec_invSig0 @ np.diag(np.sqrt(eval_invSig0)) @ evec_invSig0.T

        tmp = epsilon**2 * np.eye(n_states)/4 + sqrtSig0 @ inv_Phi_M_12 @ Sig_terminal @ inv_Phi_M_12.T @ sqrtSig0
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
        return dydt.flatten()  

    # Initial condition
    PhiM1_0 = np.eye(2 * nx1).flatten() 

    # Call the solver
    result = solve_ivp(ode_system_Phi_M1, t_span1, PhiM1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    Phi_M1 = result.y.reshape((2*nx1, 2*nx1, nt1+1))[:,:,-1]

    # Phi_M1 = np.linalg.inv(Phi_M1)
    
    Phi_M1_12 = Phi_M1[:nx1, nx1:]
    Phi_M1_11 = Phi_M1[:nx1, :nx1]
    inv_Phi_M1_12 = np.linalg.inv(Phi_M1_12)

    # ====================
    #  End Solve \Phi ODE
    # ====================

    # Sig1_0 = Sig_minus_opt  
    # Sig1_T = Sig0
    # Pi1_0 = compute_Pi0_reverse(Sig1_0, Sig1_T, Phi_M1_11, inv_Phi_M1_12, epsilon)

    Sig1_0 = Sig0   
    Sig1_T = Sig_minus_opt
    Pi1_0 = compute_Pi0(Sig1_0, Sig1_T, Phi_M1_11, inv_Phi_M1_12, epsilon)

    # ==============
    #  Solve for XY
    # ==============
    def ode_system_M1_XY(t, y):
        y_reshaped = y.reshape((2*nx1, nx1))
        dydt = compute_M1(t) @ y_reshaped
        return dydt.flatten()  

    # Solve the ODE
    v_XY_M1_0 = np.zeros((2 * nx1, nx1))
    v_XY_M1_0[:nx1, :nx1] = np.eye(nx1)
    v_XY_M1_0[nx1:, :nx1] = (Pi1_0 + Pi1_0.T) / 2

    # Flatten initial conditions for use with solve_ivp
    v_XY_M1_0_flat = v_XY_M1_0.flatten()
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
 
    K1_ilqr = np.zeros((nt1, n_inputs[mode_1], nx1), dtype=np.float64)
    for i in range(nt1):
        K1_ilqr[i] = K_feedback[i]

    # ===================
    #  End Solve for Pi
    # ===================

    # ------------- Covariance propagation ODE [0,t^-] -------------
    def cov_derivative(t, cov_flat):
        i = min(int(t / dt), nt1 - 1)  # Convert continuous time to nearest time index, cap at nt-1

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
    solution_cov = solve_ivp(
        fun=cov_derivative,
        t_span=t_span1,
        y0=cov1_0_flat,
        t_eval=t_eval1,
        method='RK23' 
    )
    cov_trj_1 = solution_cov.y.reshape((nx1, nx1, -1))
    cov_trj_1 = np.moveaxis(cov_trj_1, 2, 0)

    # cov_trj_1 = cov_trj_1[::-1,:,:]
    print("------------------ Sigma_minus computed by solving covariance control in [0, t^-] ------------------")
    print(cov_trj_1[-1])

    # ------------- Covariance propagation ODE [0,t^-], H-iLQR -------------
    def cov_derivative_ilqr(t, cov_flat):
        i = min(int(t / dt), nt1 - 1)  # Convert continuous time to nearest time index, cap at nt-1

        # Reshape the flattened covariance back to a matrix
        cov_j = cov_flat.reshape((nx1, nx1))
        
        # Compute A1_i and B1_i
        A1_i = A1[i]
        B1_i = B1[i]
        
        # Get K1 at time t
        K1_i = K1_ilqr[i]
        
        # Compute Acl_i
        Acl_i = A1_i + B1_i @ K1_i
        
        d_cov_j_dt = Acl_i @ cov_j + cov_j @ Acl_i.T + epsilon*B1_i @ B1_i.T
        
        return d_cov_j_dt.flatten()

    cov1_0_flat_ilqr = Sig0.flatten()
    # cov1_0_flat = Sig_minus_opt.flatten()
    solution_cov_ilqr = solve_ivp(
        fun=cov_derivative_ilqr,
        t_span=t_span1,
        y0=cov1_0_flat_ilqr,
        t_eval=t_eval1,
        method='RK23' 
    )
    cov_trj_1_ilqr = solution_cov_ilqr.y.reshape((nx1, nx1, -1))
    cov_trj_1_ilqr = np.moveaxis(cov_trj_1_ilqr, 2, 0)

    # cov_trj_1 = cov_trj_1[::-1,:,:]

    print("------------------ Sigma_minus computed by solving covariance control in [0, t^-], H-iLQR ------------------")
    print(cov_trj_1_ilqr[-1])

    # ======================================== [t^+, T] ========================================

    # =================
    #  Solve \Phi ODE
    # =================
    from scipy.integrate import solve_ivp
    
    # reversed M matrix
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

    Sig_plus = E_linear@Sig_minus@E_linear.T
    
    # ======================================== Jumping time t^- to t^+ ========================================
    # Sigma_plus = E_linear@Sig_minus@E_linear.T
    
    print("Sig_minus optimized: ")
    print(np.array2string(Sig_minus_opt, precision=4))
    print("Sig_plus optimized: ")
    print(np.array2string(Sig_plus_opt, precision=4))
    print("E@Sig_minus_opt@E'")
    print(E_linear@Sig_minus_opt@E_linear.T)

    eval_Sig_plus_opt, evec_Sig_plus_opt = np.linalg.eigh(Sig_plus_opt)
    print("Eigen values of the Sig_plus_opt")
    print(eval_Sig_plus_opt)

    Sig2_0 = SigT
    Sig2_T = Sig_plus

    # ==================== Solve for Pi(t) ====================
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
    
    K2_ilqr = np.zeros((nt2, n_inputs[mode_2], nx2), dtype=np.float64)
    for i in range(nt2):
        ii = t_event+i+1
        K2_ilqr[i] = K_feedback[ii]

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
        # --------------------- Reverse the dynamics ---------------------
        A2cl_i = A2_i + B2_i @ K2_i
        
        # Compute the derivative of the covariance matrix
        d_cov_j_dt = A2cl_i @ cov_j + cov_j @ A2cl_i.T + epsilon*B2_i @ B2_i.T
        
        # Flatten the derivative matrix back to a vector
        return d_cov_j_dt.flatten()

    # hybrid time
    # cov2_0 = E_linear@cov_trj_1[-1]@E_linear.T
    # cov2_0 = SigT
    cov2_0 = Sig_plus
    # cov2_0 = E_linear@cov_trj_1[-1]@E_linear.T

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

    ## ====================== Terminal covariance H-iLQR ======================

    # ------------- Covariance propagation ODE in [t^+, T] -------------
    def cov_derivative_2_ilqr(t, cov_flat):
        i = min(int(t / dt), nt2 - 1)  # Convert continuous time to nearest time index, cap at nt-1

        # Reshape the flattened covariance back to a matrix
        cov_j = cov_flat.reshape((nx2, nx2))
        
        # Compute A1_i and B1_i
        A2_i = A2[i]
        B2_i = B2[i]
        
        # Get K1 at time t
        K2_i = K2_ilqr[i]
        
        # Compute Acl_i
        # --------------------- Reverse the dynamics ---------------------
        A2cl_i = A2_i + B2_i @ K2_i
        
        # Compute the derivative of the covariance matrix
        d_cov_j_dt = A2cl_i @ cov_j + cov_j @ A2cl_i.T + epsilon*B2_i @ B2_i.T
        
        # Flatten the derivative matrix back to a vector
        return d_cov_j_dt.flatten()

    # hybrid time
    cov2_0_ilqr = E_linear@cov_trj_1_ilqr[-1]@E_linear.T

    cov2_0_ilqr_flat = cov2_0_ilqr.flatten()
    solution_cov2_ilqr = solve_ivp(
        fun=cov_derivative_2_ilqr,
        t_span=t_span2,
        y0=cov2_0_ilqr_flat,
        t_eval=t_eval2,
        method='RK23' 
    )
    cov_trj_2_ilqr = solution_cov2_ilqr.y.reshape((nx2, nx2, -1))
    cov_trj_2_ilqr = np.moveaxis(cov_trj_2_ilqr, 2, 0)

    
    # cov_trj_2 = cov_trj_2[::-1,:,:]
    print("==================== Initial covariance computed ==================")
    print(cov_trj_1[0])

    print("==================== Terminal covariance computed ==================")
    print(cov_trj_2[-1])

    print("==================== H-iLQR Terminal covariance computed ==================")
    print(cov_trj_2_ilqr[-1])

    print("==================== Sigma_plus computed ==================")
    print(cov_trj_2[0])

    print("==================== Sigma_plus targeted ==================")
    print(Sig_plus_opt)

    eval_Sig0, evec_Sig0 = np.linalg.eigh(Sig0)
    sqrtSig0 = evec_Sig0 @ np.diag(np.sqrt(eval_Sig0)) @ evec_Sig0.T
    t0 = 0.0
    dt_shrinkrate = 0.9

    # Plot the mean trajectory
    fig, ax = animate_slip(modes, states, 
                           init_mode, init_state, 
                           target_mode, target_state, nt, 
                           init_reset_args, target_reset_args, step=400)
    

    # Plot the mean trajectory
    fig_ilqr, ax_ilqr = animate_slip(modes, states, 
                           init_mode, init_state, 
                           target_mode, target_state, nt, 
                           init_reset_args, target_reset_args, step=400)

    # Plot the samples
    K1_list = [K1[i] for i in range(len(K1))]
    K2_list = [K2[i] for i in range(len(K2))]
    K_hcs = K1_list + K2_list
    
    k_ff_1 = np.asarray(k_feedforward[:t_event+1])
    k_ff_2 = np.asarray(k_feedforward[t_event+1:])
    k_ff_hcs_1 = np.zeros_like(k_ff_1)
    k_ff_hcs_2 = np.zeros_like(k_ff_2)
    k_ff_hcs_1_list = [k_ff_hcs_1[i] for i in range(k_ff_hcs_1.shape[0])]
    k_ff_hcs_2_list = [k_ff_hcs_2[i] for i in range(k_ff_hcs_2.shape[0])]
    k_ff_hcs = k_ff_hcs_1_list+k_ff_hcs_2_list
    
    # At_1 = [A1[i] for i in range(A1.shape[0])]
    # At_2 = [A2[i] for i in range(A2.shape[0])]
    # Bt_1 = [B1[i] for i in range(B1.shape[0])]
    # Bt_2 = [B2[i] for i in range(B2.shape[0])]

    # At = At_1 + At_2
    # Bt = Bt_1 + Bt_2

    # np.random.seed(70)
    for i in range(12):
        GaussianNoise_i = [np.random.randn(nt, n_inputs[0]), np.random.randn(nt, n_inputs[1])]
        x0_i = init_state + sqrtSig0@np.random.randn(n_states[1])

        # ---------------------- Samples H-CS ----------------------
        (mode_trj, 
        xt_trj, 
        ut_cl_trj, 
        Sk, 
        xt_ref_actual, 
        reset_args) = hybrid_stochastic_feedback_rollout_discrete_slip(init_mode, x0_i, n_inputs, 
                                                                       states, modes, 
                                                                        inputs, 
                                                                        K_hcs, k_feedforward, target_state, 
                                                                        Q_T, t0, dt, 
                                                                        epsilon, GaussianNoise_i, dt_shrinkrate, 
                                                                        reference_extension_helper, init_reset_args)
        
        # ---------------------- Samples H-iLQR ----------------------
        (mode_trj_ilqr, 
        xt_trj_ilqr, 
        ut_cl_trj_ilqr, 
        Sk_ilqr, 
        xt_ref_actual_ilqr, 
        reset_args_ilqr) = hybrid_stochastic_feedback_rollout_discrete_slip(init_mode, x0_i, n_inputs,
                                                                            states, modes, 
                                                                            inputs, 
                                                                            K_feedback, k_feedforward, target_state, 
                                                                            Q_T, t0, dt, 
                                                                            epsilon, GaussianNoise_i, dt_shrinkrate, 
                                                                            reference_extension_helper, init_reset_args)

        # (mode_trj, 
        # xt_trj, 
        # ut_cl_trj, 
        # Sk, 
        # xt_ref_actual, 
        # reset_args) = linearized_hybrid_stochastic_feedback_rollout_discrete(init_mode, x0_i, n_inputs, states, modes, 
        #                                                                     inputs, K_hcs, k_feedforward, At, Bt, t0, dt, 
        #                                                                     epsilon, GaussianNoise_i, dt_shrinkrate, 
        #                                                                     reference_extension_helper, init_reset_args)

        # Plot samples, H-CS
        for ii in range(0,nt,300):
            mode_i = mode_trj[ii]

            if mode_i == 0:
                px, pz = xt_trj[ii][0], xt_trj[ii][2]
            elif mode_i == 1:
                converted_state = convert_state_21_slip(xt_trj[ii])
                px, pz = converted_state[0], converted_state[2]

            ax.scatter(px, pz, marker='.', c='c', s=12)

        # Plot samples, H-iLQR
        for ii in range(0,nt,300):
            mode_i = mode_trj_ilqr[ii]

            if mode_i == 0:
                px_ilqr, pz_ilqr = xt_trj_ilqr[ii][0], xt_trj_ilqr[ii][2]
            elif mode_i == 1:
                converted_state = convert_state_21_slip(xt_trj_ilqr[ii])
                px_ilqr, pz_ilqr = converted_state[0], converted_state[2]

            ax_ilqr.scatter(px_ilqr, pz_ilqr, marker='.', c='c', s=12)

        # plot start and goals
        converted_state = convert_state_21_slip(xt_trj[0])
        px_0, pz_0 = converted_state[0], converted_state[2]
        ax.scatter(px_0, pz_0, marker='d', c='r', s=12)
        ax_ilqr.scatter(px_0, pz_0, marker='d', c='r', s=12)

        px_T, pz_T = xt_trj[-1][0], xt_trj[-1][2]
        ax.scatter(px_T, pz_T, marker='d', c='g', s=12)
        px_T_ilqr, pz_T_ilqr = xt_trj_ilqr[-1][0], xt_trj_ilqr[-1][2]
        ax_ilqr.scatter(px_T_ilqr, pz_T_ilqr, marker='d', c='g', s=12)

        # Draw covariances
        SigT_mar = SigT[0:2, 0:2]
        target_ellipse_boundary, ax = plot_2d_ellipsoid_boundary(np.array([xt_trj[-1][0], xt_trj[-1][2]]), SigT_mar, ax, 'g', linewidth=5.0)
        
    
    # ax.legend()
    ax.set_title("H-CS")
    ax_ilqr.set_title("H-iLQR")
    fig.savefig("h_cs_slip_samples.pdf", format="pdf", dpi=2000)
    plt.show()

    # for i in range(0, nt, 10):
    #     ellipse_boundary, ax2 = plot_2d_ellipsoid_boundary(states[i], cov_trj[i,0:2,0:2], ax2, 'b')

    # # ========================= controlled covariances i-LQG =========================
    # K_ilQG = np.asarray(K_feedback)
    # cov_trj_lqg = np.zeros((nt, n_states[0], n_states[0]))
    # cov_trj_lqg[0] = Sig0

    # for i in range(0, t_event):
    #     Acl_i = A[i] + B[i]@K_ilQG[i]
    #     cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + B[i]@B[i].T) * dt
    
    # # hybrid time
    # cov_trj_lqg[t_event+1] = E_linear@cov_trj_lqg[t_event]@E_linear.T

    # for i in range(t_event+1, nt-1):
    #     Acl_i = A[i] + B[i]@K_ilQG[i]
    #     cov_trj_lqg[i+1] = cov_trj_lqg[i] + (Acl_i@cov_trj_lqg[i] + cov_trj_lqg[i]@Acl_i.T + B[i]@B[i].T) * dt

