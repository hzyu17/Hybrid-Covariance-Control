## 2-dimensional SLIP dynamics
# mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
# mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
# reset maps: identity

import matplotlib.gridspec as gridspec
from dynamics.ode_solver.dynamics import *
from dynamics.guard_reset_slip import *

import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties

from matplotlib.lines import Line2D
from matplotlib.patches import Patch

font_props = FontProperties(family='serif', size=16, weight='normal')

import sympy as sp

# =================================
#   Flight dynamics Definitions
# =================================
# def symbolic_flight_dynamics_slip_continuoustime():
#     g = 9.81
#     x,x_dot,z,z_dot,theta,u = sp.symbols('x x_dot z z_dot theta u')

#     # Define the states and inputs
#     inputs = Matrix([u])
#     states = Matrix([x, x_dot, z, z_dot, theta])
    
#     # Defining the dynamics of the system
#     f_cont = Matrix([x_dot, 
#                     0,
#                     z_dot,
#                     -g,
#                     u])
    
#     # Take the jacobian with respect to states and inputs
#     A_disc = f_cont.jacobian(states)
#     B_disc = f_cont.jacobian(inputs)

#     f_cont_func = sp.lambdify((states,inputs),f_cont)
#     A_cont_func = sp.lambdify((states,inputs),A_disc)
#     B_cont_func = sp.lambdify((states,inputs),B_disc)
#     return (f_cont_func,A_cont_func,B_cont_func)


# Debug: flight dynamics, two inputs
def symbolic_flight_dynamics_slip_continuoustime():
    g = 9.81
    # x,x_dot,z,z_dot,theta,u1 = sp.symbols('x x_dot z z_dot theta u1')

    x,x_dot,z,z_dot,theta,u1,u2,u3 = sp.symbols('x x_dot z z_dot theta u1 u2 u3')

    # Define the states and inputs
    # inputs = Matrix([u1])
    inputs = sp.Matrix([u1,u2,u3])
    states = sp.Matrix([x, x_dot, z, z_dot, theta])
    
    # Defining the dynamics of the system
    # f_cont = Matrix([x_dot, 
    #                 u1,
    #                 z_dot,
    #                 -g,
    #                 0.0])
    
    f_cont = sp.Matrix([x_dot, 
                        u1,
                        z_dot,
                        u2-g,
                        u3])
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_cont.jacobian(states)
    B_disc = f_cont.jacobian(inputs)

    f_cont_func = sp.lambdify((states,inputs),f_cont)
    A_cont_func = sp.lambdify((states,inputs),A_disc)
    B_cont_func = sp.lambdify((states,inputs),B_disc)
    return (f_cont_func,A_cont_func,B_cont_func)

flight_dynamics_slip_return = symbolic_flight_dynamics_slip_continuoustime()
f_flight_cont_func = flight_dynamics_slip_return[0]

# ------------------------------------------------------------
# function definition for numerical integration in scipy
# ------------------------------------------------------------
def dyn_flight_slip(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state
        args[0]: control input
    """
   
    if len(args) == 0:
        # debug: two inputs
        u = np.array([0.0, 0.0, 0.0])
        # u = np.array([0.0])
    else:
        u = args[0]
        
    return f_flight_cont_func(x, u).flatten()

def gdWt_flight_slip(x0, dWt, eps):
    B = np.array([[0, 0, 0],[1, 0, 0],[0, 0, 0],[0, 1, 0],[0, 0, 1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt

# def gdWt_flight_slip(x0, dWt, eps):
#     B = np.array([[0],[0],[0],[0],[1.0]], dtype=np.float64)
#     return np.sqrt(eps) * B@dWt

# ---------------------------
#  Discrete-time definition
# ---------------------------

def symbolic_flight_dynamics_slip():
    g = 9.81
    x,x_dot,z,z_dot,theta,u1,u2,u3,dt = sp.symbols('x x_dot z z_dot theta u1 u2 u3 dt')

    # x,x_dot,z,z_dot,theta,u1,dt = sp.symbols('x x_dot z z_dot theta u1 dt')

    # Define the states and inputs
    inputs = sp.Matrix([u1,u2,u3])

    # inputs = Matrix([u1])
    states = sp.Matrix([x, x_dot, z, z_dot, theta])
    
    # Defining the dynamics of the system
    # f = Matrix([x_dot, 
    #             u1,
    #             z_dot,
    #             -g,
    #             0.0])
    
    f = sp.Matrix([x_dot, 
                    u1,
                    z_dot,
                    u2-g,
                    u3])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    
    return (f_disc_func,A_disc_func,B_disc_func)

# =================================
# Stance dynamics Definitions
# =================================

# ---------------------------
# Continuous-time definition
# ---------------------------
def symbolic_stance_dynamics_slip_continuoustimes():
    g = 9.81
    k = 25.0
    m = 0.5
    r0 = 1
    theta,theta_dot,r,r_dot,u1,u2 = sp.symbols('theta theta_dot r r_dot u1 u2')

    # Define the states and inputs
    inputs = sp.Matrix([u1, u2])
    states = sp.Matrix([theta, theta_dot, r, r_dot])
    
    # Defining the stance dynamics of the system
        
    # # debug: linear system
    # f_stance_cont = Matrix([theta_dot, 
    #                         u1,
    #                         r_dot,
    #                         k/m*(r0-r)+u2
    #                         ])
    
    f_stance_cont = sp.Matrix([theta_dot, 
                                -2*theta_dot*r_dot/r-g*sp.cos(theta)/r,
                                r_dot + u2/m/r/r,
                                k/m*(r0-r) - g*sp.sin(theta) + theta_dot*theta_dot*r + k*u1/m
                                ])
    
    # Take the jacobian with respect to states and inputs
    A_cont = f_stance_cont.jacobian(states)
    B_cont = f_stance_cont.jacobian(inputs)

    f_cont_func = sp.lambdify((states,inputs),f_stance_cont)
    A_cont_func = sp.lambdify((states,inputs),A_cont)
    B_cont_func = sp.lambdify((states,inputs),B_cont)
    return (f_cont_func, A_cont_func, B_cont_func)

f_stance_cont_func, _, _ = symbolic_stance_dynamics_slip_continuoustimes()

# ------------------------------------------------------------
# function definition for numerical integration in scipy
# ------------------------------------------------------------
def dyn_stance_slip(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state.
        args[0]: control input
    """
        
    if len(args) == 0:
        u = np.array([0.0, 0.0])
    else:
        u = args[0]
    
    return f_stance_cont_func(x, u).flatten()


def gdWt_stance_slip(xt, dWt, eps):
    k = 25.0
    m = 0.5
    
    r = xt[2]
    B = np.array([[0.0, 0.0],[0.0, 0.0],[0.0, 1/m/r/r],[k/m, 0.0]], dtype=np.float64)
    
    return np.sqrt(eps) * B@dWt


# ---------------------------
#  Discrete-time definition
# ---------------------------
def symbolic_stance_dynamics_slip():
    g = 9.81
    k = 25.0
    m = 0.5
    r0 = 1
    theta,theta_dot,r,r_dot,u1,u2,dt = sp.symbols('theta theta_dot r r_dot u1 u2 dt')

    # Define the states and inputs
    inputs = sp.Matrix([u1, u2])
    states = sp.Matrix([theta, theta_dot, r, r_dot])
    
    # # debug: linear system
    # f = Matrix([theta_dot, 
    #             u1,
    #             r_dot,
    #             k/m*(r0-r)+u2
    #             ])
    
    # Defining the stance dynamics of the system
    f = sp.Matrix([theta_dot, 
                    -2*theta_dot*r_dot/r-g*sp.cos(theta)/r,
                    r_dot + u2/m/r/r,
                    k/m*(r0-r) - g*sp.sin(theta) + m*r*r+k/m*u1
                    ])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)

def stochastic_integration_slip(mode, x0, u, t_span, epsilon, dW):
    if (mode==0):
        return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_flight_slip, gdWt_flight_slip)
    elif (mode==1):
        return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_stance_slip, gdWt_stance_slip)


# ----------------------------- 
# Define condition functions
# -----------------------------
# --------------------------------- Condition: mode mismatch --------------------------------- 
def cond_mode_mismatch_slip(current_mode, ref_current_mode): 
    return (current_mode != ref_current_mode)

# --------------------------------- Condition: early arrival ---------------------------------
def cond_early_arrival_slip(current_mode, ref_current_mode, event_modechange): 
    return (current_mode==event_modechange[1]) and (ref_current_mode==event_modechange[0]) 

# Condition: guard function hit
def cond_guard_function_hit_slip(xt, xt_next, guard_func): 
    return (guard_func(0.0, xt)>0) and (guard_func(0.0, xt_next)<=0)


guards_slip = {0:guard_slip_12, 1: guard_slip_21}
reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}

def hybrid_stochastic_feedback_rollout_slip(init_mode, x0, n_inputs, xt_ref, ref_modes, 
                                            ut, Kt, kt, target_state, Q_T, t0, tf, 
                                            epsilon, GaussianNoise, 
                                            ref_ext_helper, init_reset_args):

    (v_event_modechange, v_ext_bwd, v_ext_fwd, 
    v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
    v_kff_ext_bwd, v_kff_ext_fwd, _) = extract_extensions(ref_ext_helper, start_index = 0)
    
    n_timestamps = len(xt_ref)
    
    dt = (tf - t0) / n_timestamps
    dt_int = dt
    
    # returning trajectory    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0  
    
    mode_trj = np.zeros((n_timestamps), dtype=np.int64) 
    mode_trj[0] = init_mode
    
    # closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    # only consider the 1->2 reset for now 
    current_guard = guards_slip[init_mode]
    
    cnt_mismatch = 0
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    
    # path cost
    Sk = 0
    
    # hybrid event related 
    cnt_event = 0
    reset_args = init_reset_args
    event_args = [init_reset_args[0]]
    
    # -------------- roullout function --------------
    for ii_t in range(n_timestamps-1):   

        t0_i = t0 + ii_t*dt   
        
        current_mode = mode_trj[ii_t]
        xt = xt_trj[ii_t]
        
        ref_current_mode = ref_modes[ii_t]
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        xref_i = xt_ref[ii_t] 
        
        if cond_mode_mismatch_slip(current_mode, ref_current_mode):
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch(ii_t, current_mode, ref_current_mode, 
                                                                            v_ext_fwd[0], v_ext_bwd[0], 
                                                                            v_event_modechange[0],
                                                                            v_Kfb_ext_fwd[0], v_kff_ext_fwd[0],
                                                                            v_Kfb_ext_bwd[0], v_kff_ext_bwd[0],
                                                                            cnt_mismatch, cond_early_arrival_slip)
        
        xt_ref_actual[ii_t] = xref_i
        
        delta_xt_i = xt_trj[ii_t] - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = current_u
        
        noise_i = GaussianNoise[current_mode][ii_t]
        dW_i = np.sqrt(dt_int)*noise_i
        
        # ============================== One step integration ==============================        
        # ---- solver for the deterministic part
        t_span = (t0_i, t0_i + dt_int)
        
        xt_next = stochastic_integration_slip(current_mode, xt, current_u, t_span, epsilon, dW_i).flatten()
        next_mode = current_mode
        
        # Condition: Hit the guard function.  
        current_guard = guards_slip[current_mode]
        if cond_guard_function_hit_slip(xt, xt_next, current_guard): 
            
            args = (xt, current_mode, current_u, t0_i, t0_i+dt_int, xt_next, 
                    dt_int, GaussianNoise[current_mode][ii_t], epsilon, 
                    stochastic_integration_slip, guards_slip, reset_maps_slip, reset_args[ii_t])
            
            xt_next, next_mode, dW_i, new_reset_args = stoch_event_reactive_fun(args)
            dt_int = dt
            
            event_args.append(new_reset_args)
            cnt_event += 1
        
        # ============================== // One step integration // ==============================     
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += current_u.T@current_u/2.0 * dt + np.sqrt(epsilon) * np.dot(current_u.T, dW_i)
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    fig, ax = plt.subplots()
    time_span = np.arange(0, n_timestamps)
    plot_slip(time_span, mode_trj, xt_trj, ut_cl_trj, 
              x0, target_state, n_timestamps, reset_args, 
              figs=None, axes=None, color='k', alpha=1.0, step=2)
    
    ax.legend(loc='best', prop={'family': 'serif', 'size': 16})
    plt.show()
    
    # Terminal cost
    Sk += (xt-target_state)@Q_T@(xt-target_state) / 2.0
    
    show_mismatch = False
    if show_mismatch:
        # ======== Show mode mismatch ======== 
        fig2, axes = plt.subplots(1,2, figsize=(9, 6))
        ax5, ax6 = axes.flatten()
        ax5.grid(True)
        ax6.grid(True)
        
        ax5.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax5.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
        ax5.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--', label='Modified Reference')
        
        ax5.set_xlabel(r"z", fontproperties=font_props)
        ax5.set_ylabel(r"$\dot z$", fontproperties=font_props)
        ax5.legend(loc='best', prop={'family': 'serif', 'size': 16})
        plt.tight_layout()
        
        ax6.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax6.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
        ax6.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--',label='Modified Reference')
        ax6.set_xlabel(r"z", fontproperties=font_props)
        ax6.set_ylabel(r"$\dot z$", fontproperties=font_props)
        ax6.legend(loc='best', prop={'family': 'serif', 'size': 16})
        plt.tight_layout()
        
        plt.show()
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual


def event_detect_slip(x0, u, t0, tf, current_mode, reset_args, detection=True, backwards=False):
    # guard_slip_12.terminal=True
    # guard_slip_12.direction=1
    
    # guard_slip_21.terminal=True
    # guard_slip_21.direction=1
    
    # guards_slip = {0:guard_slip_12, 1: guard_slip_21}
    # reset_maps_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}
    
    Rxs_slip = {0:Rx_slip_12, 1:Rx_slip_21}
    Rts_slip = {0:Rt_slip_12, 1:Rt_slip_21}
    
    gxs_slip = {0:gx_slip_12, 1:gx_slip_21}
    gts_slip = {0:gt_slip_12, 1:gt_slip_21}
    
    smooth_dynamics_slip = {0:dyn_flight_slip, 1:dyn_stance_slip}
    
    return event_detect_onestep(x0, 
                                u, 
                                t0, 
                                tf, 
                                current_mode, 
                                smooth_dynamics_slip, 
                                guards_slip,
                                gxs_slip,
                                gts_slip,
                                reset_maps_slip,
                                Rxs_slip,
                                Rts_slip,
                                reset_args, detection, backwards)
    

def plot_slip_nexp(n_exp_indexs, exp_data, time_span, init_state, target_state, args=None):
    # fig11, axes11 = plt.subplots(1, 3, figsize=(15, 5))
    # fig12, axes12 = plt.subplots(1, 2, figsize=(10, 5))
    
    # ax11, ax12, ax13 = axes11.flatten()[0], axes11.flatten()[1], axes11.flatten()[2]
    # ax14, ax15 = axes12.flatten()[0], axes12.flatten()[1]
    
    
    fig11, ax11 = plt.subplots(1, 1, figsize=(4, 5))
    fig12, ax12 = plt.subplots(1, 1, figsize=(4, 5))
    fig13, ax13 = plt.subplots(1, 1, figsize=(4, 5))
    fig14, ax14 = plt.subplots(1, 1, figsize=(4, 5))
    fig15, ax15 = plt.subplots(1, 1, figsize=(4, 5))
    
    
    fig2, axes2 = plt.subplots(2, 2, figsize=(15, 5))
    fig3, ax3 = plt.subplots(1, 1, figsize=(4, 5))
    fig4, ax4 = plt.subplots(1, 1, figsize=(4, 5))
    
    
    (ax21, ax22, ax23, ax24) = axes2.flatten()[0], axes2.flatten()[1], axes2.flatten()[2], axes2.flatten()[3]
    
    # figs = np.array([fig11, fig12, fig2, fig3, fig4], dtype=object)
    figs = np.array([fig11, fig12, fig13, fig14, fig15, fig2, fig3, fig4], dtype=object)
    axes = np.array([ax11, ax12, ax13, ax14, ax15, ax3, ax21, ax22, ax23, ax24, ax4], dtype=object)  
    
    for i, i_exp in enumerate(n_exp_indexs):
        modes_pi = exp_data.get_data(i_exp).mode_trj_pi()
        states_pi = exp_data.get_data(i_exp).x_trj_pi()
        inputs_pi = exp_data.get_data(i_exp).u_trj_pi()
        # reset_args_pi = exp_data.get_data(i).reset_args()
        states_pi = exp_data.get_data(i_exp).x_trj_pi()
        states_pi = np.array(states_pi)
        
        modes_ilqg = exp_data.get_data(i_exp).mode_trj_ilqr()
        states_ilqg = exp_data.get_data(i_exp).x_trj_ilqr()
        inputs_ilqg = exp_data.get_data(i_exp).u_trj_ilqr()
        # reset_args_ilqg = exp_data.get_data(i).reset_args()
                
        nt = len(time_span)
        reset_args = [np.array([0.0]) for _ in range(nt)]
        states_pi = unpad_state_slip(modes_pi, states_pi)
        
        # h-pi
        time_span_discrete = np.arange(0, nt)
        figs, axes = plot_slip(time_span, modes_pi, states_pi, 
                                inputs_pi, init_state, target_state, 
                                nt, reset_args, figs=figs, axes=axes, 
                                color='r', alpha=0.9, step=1, trj_label='H-PI')
    
        # hilqr 
        figs, axes = plot_slip(time_span, modes_ilqg, states_ilqg, 
                                inputs_ilqg, init_state, target_state, 
                                nt, reset_args, figs=figs, axes=axes, 
                                color='b', step=1, trj_label='H-iLQR')
        
        
        (ax11, ax12, ax13, ax14, ax15, ax3, ax21, ax22, ax23, ax24, ax4) = axes.flatten()
        
        if (i==0):
            
            init_state_label = convert_state_21_slip(init_state, np.array([0.0]))
            
            ax11.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax11.scatter(time_span[0], init_state_label[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            
            ax12.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax12.scatter(time_span[0], init_state_label[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

            ax13.scatter(time_span[-1], target_state[2], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax13.scatter(time_span[0], init_state_label[2], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            
            ax14.scatter(time_span[-1], target_state[3], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax14.scatter(time_span[0], init_state_label[3], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            
            ax15.scatter(time_span[-1], target_state[4], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax15.scatter(time_span[0], init_state_label[4], color='r', marker='x', s=50.0, linewidths=6, label='Start')
            
            if ax11.get_legend() is None:
                ax11.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax12.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax13.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax14.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax15.legend(loc='best', prop={'family': 'serif', 'size': 16})
                
                ax21.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax22.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax23.legend(loc='best', prop={'family': 'serif', 'size': 16})
                ax24.legend(loc='best', prop={'family': 'serif', 'size': 16})
            
    return figs, axes


def plot_sample_trajectory_slip(nsamples_indexes, nt_i, time_span,
                                Kmodes_jax_i, Ksamples_jax_i, Ksamples_ut, 
                                Ksamples_reset_args, 
                                mode_trj_ilqr, xt_trj_ilqr, 
                                ut_trj_ilqr, reset_args_ilqr,
                                modes, states, inputs,ref_reset_args,
                                init_state, target_state):
    
    fig, axes = plt.subplots(2, 6, figsize=(20, 10))
    
    # hilqr 
    fig, axes = plot_slip(time_span, mode_trj_ilqr, xt_trj_ilqr, 
                            ut_trj_ilqr, init_state, target_state, 
                            nt_i, reset_args_ilqr, fig, axes, 'r', step=1)
    
    # reference
    fig, axes = plot_slip(time_span, modes, states, 
                            inputs, init_state, target_state, 
                            nt_i, ref_reset_args, fig, axes, 'k', step=1)
        
    for i_s in nsamples_indexes:
        sample_modes_i = Kmodes_jax_i[i_s]
        sample_states_i = Ksamples_jax_i[i_s]
        sample_inputs_i = Ksamples_ut[i_s]
        sample_states_i = unpad_state_slip(sample_modes_i, sample_states_i)
        sample_inputs_i = unpad_control_slip(sample_modes_i, sample_inputs_i, n_inputs=[3,2])
        
        sample_reset_args_i = Ksamples_reset_args[i_s]
        
        # # --------------------- Debug: plot the control gains --------------------- 
        # fig3, axes = plt.subplots(2, 1)
        # (ax6, ax7) = axes.flatten()
        
        # ax6.grid(True)
        # ax7.grid(True)
                        
        # ax6.plot(np.arange(nt), K_feedback_0_i[:, 0, 4], label='ref mode0 Kfb[4]')
        # ax6.plot(np.arange(nt), K_feedback_1_i[:, 0, 0], label='ref mode1 Kfb[0]')
        # ax6.plot(np.arange(nt), Ksamples_Kfb_mode_i[:, 0, 0], label='Ksamples Kfb[0]')
        # ax6.plot(np.arange(nt), Ksamples_Kfb_mode_i[:, 0, 4], label='Ksamples Kfb[4]')
        
        # ax7.plot(np.arange(nt), k_feedforward_0_i[:, 0], label='ref mode0 kff[0]')
        # ax7.plot(np.arange(nt), k_feedforward_1_i[:, 0], label='ref mode1 kff[0]')
        # ax7.plot(np.arange(nt), Ksamples_kff_mode_i[:, 0], label='Ksamples kff[0]')
        
        # ax6.legend()
        # ax7.legend()
        
        # fig3.tight_layout()
        # plt.show()
        fig, axes = plot_slip(time_span, sample_modes_i, sample_states_i, 
                             sample_inputs_i, init_state, target_state, 
                             nt_i, sample_reset_args_i, fig, axes, 'b', alpha=0.1, step=1)
    
    return fig, axes

spring_coils = 15
spring_amplitude = 0.01
ball_radius = 0.012

def plot_slip(time_span, modes, 
              states, inputs, 
              init_state, target_state, 
              nt, reset_args, 
              figs=None, axes=None, 
              color='k', alpha=1.0, 
              step=2, trj_label=None,
              show_mode_indicator=True,
              show_start_goal=True,
              legend_loc='best',
              convert_state_func=None):
    """
    Plot SLIP trajectory data with comprehensive legends.
    
    Args:
        time_span: Array of time values
        modes: Array of mode indices (0=flight, 1=stance)
        states: List of state vectors at each time step
        inputs: List of input arrays per mode
        init_state: Initial state vector
        target_state: Target state vector
        nt: Number of time steps
        reset_args: Reset arguments for state conversion
        figs: Optional existing figure handles
        axes: Optional existing axes handles
        color: Line color for trajectory
        alpha: Line transparency
        step: Plotting step size (for downsampling)
        trj_label: Label for trajectory in legend
        show_mode_indicator: Whether to show mode as background shading
        show_start_goal: Whether to mark start and goal states
        legend_loc: Legend location ('best', 'upper right', etc.)
        convert_state_func: Function to convert stance states to flight states
                           (defaults to convert_state_21_slip if None)
    
    Returns:
        figs: Array of figure handles
        axes: Array of axes handles
    """
    
    # Import conversion function if not provided
    if convert_state_func is None:
        from dynamics.dynamics_discrete_slip import convert_state_21_slip
        convert_state_func = convert_state_21_slip
    
    font_props = FontProperties(family='serif', size=18, weight='normal')
    legend_font_props = {'family': 'serif', 'size': 12}
    
    # =============== Create figures if not provided ===============
    if (figs is None) and (axes is None):
        fig11, ax11 = plt.subplots(1, 1, figsize=(5, 4))
        fig12, ax12 = plt.subplots(1, 1, figsize=(5, 4))
        fig13, ax13 = plt.subplots(1, 1, figsize=(5, 4))
        fig14, ax14 = plt.subplots(1, 1, figsize=(5, 4))
        fig15, ax15 = plt.subplots(1, 1, figsize=(5, 4))
        
        fig2, axes2 = plt.subplots(2, 2, figsize=(12, 8))
        fig3, ax3 = plt.subplots(1, 1, figsize=(5, 4))
        fig4, ax4 = plt.subplots(1, 1, figsize=(5, 4))
                
        (ax21, ax22, ax23, ax24) = axes2.flatten()
        
        # Initialize figures as new
        is_new_figure = True
    else:
        (fig11, fig12, fig13, fig14, fig15, fig2, fig3, fig4) = figs.flatten()
        (ax11, ax12, ax13, ax14, ax15, ax3, ax21, ax22, ax23, ax24, ax4) = axes.flatten()
        is_new_figure = False
    
    # =============== Convert stance states to flight coordinates ===============
    flight_mode_states = np.zeros((nt, 5))
    for i in range(nt):
        if modes[i] == 0:
            flight_mode_states[i] = states[i].flatten()
        elif modes[i] == 1:
            reset_arg = reset_args[i][0] if isinstance(reset_args[i], (list, np.ndarray)) else reset_args[i]
            flight_mode_states[i] = convert_state_func(states[i], reset_arg).flatten()
    
    # =============== Add mode shading (background) ===============
    if show_mode_indicator and is_new_figure:
        # Find mode transition points
        mode_changes = [0]
        for i in range(1, nt):
            if modes[i] != modes[i-1]:
                mode_changes.append(i)
        mode_changes.append(nt-1)
        
        # Shade regions by mode
        for ax in [ax11, ax12, ax13, ax14, ax15]:
            ylim = ax.get_ylim() if ax.get_ylim() != (0.0, 1.0) else None
            for j in range(len(mode_changes)-1):
                start_idx = mode_changes[j]
                end_idx = mode_changes[j+1]
                mode_val = modes[start_idx]
                
                if mode_val == 0:  # Flight
                    ax.axvspan(time_span[start_idx], time_span[end_idx], 
                              alpha=0.1, color='skyblue', zorder=0)
                else:  # Stance
                    ax.axvspan(time_span[start_idx], time_span[end_idx], 
                              alpha=0.1, color='salmon', zorder=0)
            if ylim:
                ax.set_ylim(ylim)
    
    # =============== Plot flight mode states ===============
    # Use a default label if none provided
    plot_label = trj_label if trj_label else 'Trajectory'
    
    line11, = ax11.plot(time_span[::step], flight_mode_states[::step, 0], 
                        linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    line12, = ax12.plot(time_span[::step], flight_mode_states[::step, 1], 
                        linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    line13, = ax13.plot(time_span[::step], flight_mode_states[::step, 2], 
                        linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    line14, = ax14.plot(time_span[::step], flight_mode_states[::step, 3], 
                        linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    line15, = ax15.plot(time_span[::step], flight_mode_states[::step, 4], 
                        linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    
    # =============== Collect mode-specific data ===============
    mode1_timestamps, mode1_states, mode1_inputs = [], [], []
    mode0_timestamps, mode0_states, mode0_inputs = [], [], []
    
    for i in range(0, nt-1, step):
        if modes[i] == 0:
            mode0_timestamps.append(time_span[i])
            mode0_states.append(states[i])
            mode0_inputs.append(inputs[modes[i]][i])
        elif modes[i] == 1:
            mode1_timestamps.append(time_span[i])
            mode1_states.append(states[i])
            mode1_inputs.append(inputs[modes[i]][i])
    
    mode0_timestamps = np.array(mode0_timestamps) if mode0_timestamps else np.array([])
    mode0_states = np.array(mode0_states) if mode0_states else np.array([]).reshape(0, 5)
    mode0_inputs = np.array(mode0_inputs) if mode0_inputs else np.array([]).reshape(0, 3)
    
    mode1_timestamps = np.array(mode1_timestamps) if mode1_timestamps else np.array([])
    mode1_states = np.array(mode1_states) if mode1_states else np.array([]).reshape(0, 4)
    mode1_inputs = np.array(mode1_inputs) if mode1_inputs else np.array([]).reshape(0, 2)
    
    # =============== Plot mode 0 (flight) control inputs ===============
    if len(mode0_inputs) > 0:
        ax3.plot(mode0_timestamps, mode0_inputs[:, 0], 
                linewidth=1.5, color='blue', alpha=alpha, label=r'$u_x$ (horiz. accel)')
        ax3.plot(mode0_timestamps, mode0_inputs[:, 1], 
                linewidth=1.5, color='red', alpha=alpha, label=r'$u_z$ (vert. accel)')
        ax3.plot(mode0_timestamps, mode0_inputs[:, 2], 
                linewidth=1.5, color='green', alpha=alpha, label=r'$\dot{\theta}$ (leg rate)')
    
    # =============== Plot mode 1 (stance) states ===============
    if len(mode1_states) > 0:
        ax21.plot(mode1_timestamps, mode1_states[:, 0], 
                 linewidth=1.5, color=color, alpha=alpha, label=plot_label)
        ax22.plot(mode1_timestamps, mode1_states[:, 1], 
                 linewidth=1.5, color=color, alpha=alpha, label=plot_label)
        ax23.plot(mode1_timestamps, mode1_states[:, 2], 
                 linewidth=1.5, color=color, alpha=alpha, label=plot_label)
        ax24.plot(mode1_timestamps, mode1_states[:, 3], 
                 linewidth=1.5, color=color, alpha=alpha, label=plot_label)
    
    # =============== Plot mode 1 (stance) control inputs ===============
    if len(mode1_inputs) > 0:
        ax4.plot(mode1_timestamps, mode1_inputs[:, 0], 
                linewidth=1.5, color='blue', alpha=alpha, label=r'$\Delta r$ (spring)')
        ax4.plot(mode1_timestamps, mode1_inputs[:, 1], 
                linewidth=1.5, color='red', alpha=alpha, label=r'$\tau_{hip}$ (torque)')
    
    # =============== Plot start and goal markers ===============
    if show_start_goal:
        # Convert init state if in stance mode
        if modes[0] == 1:
            init_state_flight = convert_state_func(init_state, np.array([0.0]))
        else:
            init_state_flight = init_state
        
        marker_size = 80
        
        # Start markers
        ax11.scatter(time_span[0], init_state_flight[0], color='red', marker='o', 
                    s=marker_size, zorder=5, edgecolors='darkred', linewidths=1.5, label='Start')
        ax12.scatter(time_span[0], init_state_flight[1], color='red', marker='o', 
                    s=marker_size, zorder=5, edgecolors='darkred', linewidths=1.5)
        ax13.scatter(time_span[0], init_state_flight[2], color='red', marker='o', 
                    s=marker_size, zorder=5, edgecolors='darkred', linewidths=1.5)
        ax14.scatter(time_span[0], init_state_flight[3], color='red', marker='o', 
                    s=marker_size, zorder=5, edgecolors='darkred', linewidths=1.5)
        ax15.scatter(time_span[0], init_state_flight[4], color='red', marker='o', 
                    s=marker_size, zorder=5, edgecolors='darkred', linewidths=1.5)
        
        # Goal markers
        ax11.scatter(time_span[-1], target_state[0], color='lime', marker='*', 
                    s=marker_size*1.5, zorder=5, edgecolors='darkgreen', linewidths=1.5, label='Target')
        ax12.scatter(time_span[-1], target_state[1], color='lime', marker='*', 
                    s=marker_size*1.5, zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax13.scatter(time_span[-1], target_state[2], color='lime', marker='*', 
                    s=marker_size*1.5, zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax14.scatter(time_span[-1], target_state[3], color='lime', marker='*', 
                    s=marker_size*1.5, zorder=5, edgecolors='darkgreen', linewidths=1.5)
        ax15.scatter(time_span[-1], target_state[4], color='lime', marker='*', 
                    s=marker_size*1.5, zorder=5, edgecolors='darkgreen', linewidths=1.5)
    
    # =============== Set axis labels ===============
    ax11.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax11.set_ylabel(r"$p_x$ (m)", fontproperties=font_props)
    ax11.set_title(r"Horizontal Position", fontproperties=font_props)

    ax12.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax12.set_ylabel(r"$v_x$ (m/s)", fontproperties=font_props)
    ax12.set_title(r"Horizontal Velocity", fontproperties=font_props)
    
    ax13.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax13.set_ylabel(r"$p_z$ (m)", fontproperties=font_props)
    ax13.set_title(r"Vertical Position", fontproperties=font_props)
    
    ax14.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax14.set_ylabel(r"$v_z$ (m/s)", fontproperties=font_props)
    ax14.set_title(r"Vertical Velocity", fontproperties=font_props)
    
    ax15.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax15.set_ylabel(r"$\theta$ (rad)", fontproperties=font_props)
    ax15.set_title(r"Leg Angle", fontproperties=font_props)
    
    ax3.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax3.set_ylabel(r"Input", fontproperties=font_props)
    ax3.set_title(r"Flight Phase Inputs", fontproperties=font_props)
    
    ax21.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax21.set_ylabel(r"$\theta$ (rad)", fontproperties=font_props)
    ax21.set_title(r"Stance: Leg Angle", fontproperties=font_props)
    
    ax22.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax22.set_ylabel(r"$\dot{\theta}$ (rad/s)", fontproperties=font_props)
    ax22.set_title(r"Stance: Angular Velocity", fontproperties=font_props)
    
    ax23.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax23.set_ylabel(r"$r$ (m)", fontproperties=font_props)
    ax23.set_title(r"Stance: Leg Length", fontproperties=font_props)

    ax24.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax24.set_ylabel(r"$\dot{r}$ (m/s)", fontproperties=font_props)
    ax24.set_title(r"Stance: Leg Velocity", fontproperties=font_props)
    
    ax4.set_xlabel(r"Time (s)", fontproperties=font_props)
    ax4.set_ylabel(r"Input", fontproperties=font_props)
    ax4.set_title(r"Stance Phase Inputs", fontproperties=font_props)
    
    # =============== Add legends ===============
    # Custom legend for flight state plots (includes mode shading explanation)
    if is_new_figure:
        # Create custom legend handles
        legend_handles_state = []
        
        # Trajectory line
        legend_handles_state.append(Line2D([0], [0], color=color, linewidth=1.5, 
                                           label=plot_label if plot_label else 'Trajectory'))
        
        # Mode shading indicators
        if show_mode_indicator:
            legend_handles_state.append(Patch(facecolor='skyblue', alpha=0.3, 
                                              edgecolor='none', label='Flight phase'))
            legend_handles_state.append(Patch(facecolor='salmon', alpha=0.3, 
                                              edgecolor='none', label='Stance phase'))
        
        # Start/goal markers
        if show_start_goal:
            legend_handles_state.append(Line2D([0], [0], marker='o', color='w', 
                                               markerfacecolor='red', markersize=10,
                                               markeredgecolor='darkred', markeredgewidth=1.5,
                                               label='Start', linestyle='None'))
            legend_handles_state.append(Line2D([0], [0], marker='*', color='w', 
                                               markerfacecolor='lime', markersize=12,
                                               markeredgecolor='darkgreen', markeredgewidth=1.5,
                                               label='Target', linestyle='None'))
        
        # Add legends to flight state plots
        for ax in [ax11, ax12, ax13, ax14, ax15]:
            ax.legend(handles=legend_handles_state, loc=legend_loc, prop=legend_font_props,
                     framealpha=0.9, fancybox=True)
            ax.grid(True, linestyle='--', alpha=0.5)
        
        # Add legends to stance state plots
        legend_handles_stance = [
            Line2D([0], [0], color=color, linewidth=1.5, label=plot_label if plot_label else 'Trajectory')
        ]
        for ax in [ax21, ax22, ax23, ax24]:
            ax.legend(handles=legend_handles_stance, loc=legend_loc, prop=legend_font_props,
                     framealpha=0.9, fancybox=True)
            ax.grid(True, linestyle='--', alpha=0.5)
        
        # Input plot legends
        ax3.legend(loc=legend_loc, prop=legend_font_props, framealpha=0.9, fancybox=True)
        ax3.grid(True, linestyle='--', alpha=0.5)
        
        ax4.legend(loc=legend_loc, prop=legend_font_props, framealpha=0.9, fancybox=True)
        ax4.grid(True, linestyle='--', alpha=0.5)
    
    # Tight layout for all figures
    for fig in [fig11, fig12, fig13, fig14, fig15, fig2, fig3, fig4]:
        fig.tight_layout()
    
    return (np.array([fig11, fig12, fig13, fig14, fig15, fig2, fig3, fig4], dtype=object), 
            np.array([ax11, ax12, ax13, ax14, ax15, ax3, ax21, ax22, ax23, ax24, ax4], dtype=object))


def plot_slip_comparison(time_span, modes_list, states_list, inputs_list,
                         init_state, target_state, nt, reset_args,
                         labels=None, colors=None, 
                         title_prefix="SLIP Trajectory Comparison",
                         convert_state_func=None):
    """
    Plot multiple SLIP trajectories for comparison with proper legends.
    
    Args:
        time_span: Array of time values
        modes_list: List of mode arrays for each trajectory
        states_list: List of state arrays for each trajectory
        inputs_list: List of input arrays for each trajectory
        init_state: Initial state vector
        target_state: Target state vector
        nt: Number of time steps
        reset_args: Reset arguments for state conversion
        labels: List of labels for each trajectory
        colors: List of colors for each trajectory
        title_prefix: Prefix for figure titles
        convert_state_func: Function to convert stance states to flight states
    
    Returns:
        figs: Array of figure handles
        axes: Array of axes handles
    """
    
    n_trajectories = len(states_list)
    
    # Default labels and colors
    if labels is None:
        labels = [f'Trajectory {i+1}' for i in range(n_trajectories)]
    if colors is None:
        cmap = plt.cm.tab10
        colors = [cmap(i) for i in range(n_trajectories)]
    
    # Plot first trajectory to create figures
    figs, axes = plot_slip(time_span, modes_list[0], states_list[0], inputs_list[0],
                           init_state, target_state, nt, reset_args,
                           color=colors[0], trj_label=labels[0],
                           show_mode_indicator=True, show_start_goal=True,
                           convert_state_func=convert_state_func)
    
    # Overlay remaining trajectories
    for i in range(1, n_trajectories):
        plot_slip(time_span, modes_list[i], states_list[i], inputs_list[i],
                 init_state, target_state, nt, reset_args,
                 figs=figs, axes=axes,
                 color=colors[i], trj_label=labels[i],
                 show_mode_indicator=False, show_start_goal=False,
                 convert_state_func=convert_state_func)
    
    # Update legends with all trajectories
    font_props = {'family': 'serif', 'size': 12}
    
    # Create comprehensive legend handles
    legend_handles = []
    for i, (label, color) in enumerate(zip(labels, colors)):
        legend_handles.append(Line2D([0], [0], color=color, linewidth=2, label=label))
    
    # Add mode indicators
    legend_handles.append(Patch(facecolor='skyblue', alpha=0.3, edgecolor='none', label='Flight phase'))
    legend_handles.append(Patch(facecolor='salmon', alpha=0.3, edgecolor='none', label='Stance phase'))
    
    # Add start/goal
    legend_handles.append(Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                 markersize=10, markeredgecolor='darkred', markeredgewidth=1.5,
                                 label='Start', linestyle='None'))
    legend_handles.append(Line2D([0], [0], marker='*', color='w', markerfacecolor='lime', 
                                 markersize=12, markeredgecolor='darkgreen', markeredgewidth=1.5,
                                 label='Target', linestyle='None'))
    
    # Update legends on all state plots
    (ax11, ax12, ax13, ax14, ax15, ax3, ax21, ax22, ax23, ax24, ax4) = axes.flatten()
    for ax in [ax11, ax12, ax13, ax14, ax15]:
        ax.legend(handles=legend_handles, loc='best', prop=font_props, 
                 framealpha=0.9, fancybox=True, ncol=1)
    
    # Set titles
    (fig11, fig12, fig13, fig14, fig15, fig2, fig3, fig4) = figs.flatten()
    fig11.suptitle(f"{title_prefix}: $p_x$", fontsize=14, fontfamily='serif')
    fig12.suptitle(f"{title_prefix}: $v_x$", fontsize=14, fontfamily='serif')
    fig13.suptitle(f"{title_prefix}: $p_z$", fontsize=14, fontfamily='serif')
    fig14.suptitle(f"{title_prefix}: $v_z$", fontsize=14, fontfamily='serif')
    fig15.suptitle(f"{title_prefix}: $\\theta$", fontsize=14, fontfamily='serif')
    
    return figs, axes


def animate_slip_with_legend(modes, states, 
                              init_mode, init_state, 
                              target_mode, target_state, 
                              nt, reset_args, target_reset_args,
                              step=100, figsize=(8, 6),
                              title="SLIP Animation",
                              convert_state_func=None):
    """
    Create an animated-style plot of SLIP trajectory with comprehensive legend.
    
    Args:
        modes: Array of mode indices
        states: List of state vectors
        init_mode: Initial mode
        init_state: Initial state vector
        target_mode: Target mode
        target_state: Target state vector
        nt: Number of time steps
        reset_args: Reset arguments
        target_reset_args: Target reset arguments
        step: Plotting step size
        figsize: Figure size tuple
        title: Plot title
        convert_state_func: State conversion function
    
    Returns:
        fig, ax: Figure and axes handles
    """
    
    if convert_state_func is None:
        from dynamics.dynamics_discrete_slip import convert_state_21_slip
        convert_state_func = convert_state_21_slip
    
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    
    r0 = 1.0  # Rest length of spring
    
    # Plot trajectory colored by mode
    flight_segments_x, flight_segments_z = [], []
    stance_segments_x, stance_segments_z = [], []
    
    for ii in range(0, nt, step):
        mode_i = modes[ii]
        reset_arg = reset_args[ii][0] if isinstance(reset_args[ii], (list, np.ndarray)) else reset_args[ii]
        
        if mode_i == 0:  # Flight
            px, pz = states[ii][0], states[ii][2]
            flight_segments_x.append(px)
            flight_segments_z.append(pz)
        elif mode_i == 1:  # Stance
            converted = convert_state_func(states[ii], reset_arg)
            px, pz = converted[0], converted[2]
            stance_segments_x.append(px)
            stance_segments_z.append(pz)
    
    # Plot trajectories
    if flight_segments_x:
        ax.scatter(flight_segments_x, flight_segments_z, c='blue', s=15, alpha=0.7, 
                   marker='o', label='Flight trajectory')
    if stance_segments_x:
        ax.scatter(stance_segments_x, stance_segments_z, c='orange', s=15, alpha=0.7,
                   marker='s', label='Stance trajectory')
    
    # Plot start state
    if init_mode == 1:
        init_converted = convert_state_func(init_state, np.array([0.0]))
        init_px, init_pz = init_converted[0], init_converted[2]
    else:
        init_px, init_pz = init_state[0], init_state[2]
    
    ax.scatter(init_px, init_pz, c='red', s=150, marker='o', 
               edgecolors='darkred', linewidths=2, zorder=10, label='Start')
    
    # Plot target state
    target_px, target_pz = target_state[0], target_state[2]
    ax.scatter(target_px, target_pz, c='lime', s=200, marker='*',
               edgecolors='darkgreen', linewidths=2, zorder=10, label='Target')
    
    # Draw ground
    x_min = min(min(flight_segments_x + stance_segments_x) - 0.5, init_px - 0.5)
    x_max = max(max(flight_segments_x + stance_segments_x) + 0.5, target_px + 0.5)
    ax.axhline(y=0, color='brown', linestyle='-', linewidth=3, label='Ground')
    ax.fill_between([x_min, x_max], [-0.1, -0.1], [0, 0], color='brown', alpha=0.3)
    
    # Labels and formatting
    ax.set_xlabel(r'$p_x$ (m)', fontsize=14, fontfamily='serif')
    ax.set_ylabel(r'$p_z$ (m)', fontsize=14, fontfamily='serif')
    ax.set_title(title, fontsize=16, fontfamily='serif')
    
    # Legend
    ax.legend(loc='upper left', prop={'family': 'serif', 'size': 11}, 
             framealpha=0.9, fancybox=True)
    
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.set_aspect('equal', adjustable='box')
    
    fig.tight_layout()
    
    return fig, ax


def plot_slip_flight_animate(state_flight, r0, ax=None, spring_color='k-'):
    if ax is None:
        fig, ax = plt.subplots()
        ax.grid(True)

    # Parameters
    x, _, z, _, theta = state_flight[0], state_flight[1], state_flight[2], state_flight[3], state_flight[4]
    
    theta_spring = np.pi / 2 - theta
    
    # Generate spring data
    t = np.linspace(0, 2 * np.pi * spring_coils, 1000)
    x_spring = spring_amplitude * np.sin(t)
    z_spring = np.linspace(0, 0.4*r0, 1000)
    
    x_spring_rot = x_spring * np.cos(theta_spring) + z_spring * np.sin(theta_spring)
    z_spring_rot = -x_spring * np.sin(theta_spring) + z_spring * np.cos(theta_spring)

    x_spring_rot = x_spring_rot + x - 0.7*r0*np.cos(theta)
    z_spring_rot = z_spring_rot + z - 0.7*r0*np.sin(theta)
    
    # Plot the spring
    # ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2.0)
    ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2.0, alpha=0.2)
    
    # Plot the line connecting the ends of the spring
    t1_x = np.linspace(x_spring_rot[-1], x, 1000)
    t1_z = np.linspace(z_spring_rot[-1], z, 1000)
    
    x_tail = x - r0*np.cos(theta)
    z_tail = z - r0*np.sin(theta)
    t2_x = np.linspace(x_tail, x_spring_rot[0], 1000)
    t2_z = np.linspace(z_tail, z_spring_rot[0], 1000)
    
    ax.plot(t2_x, t2_z, spring_color, lw=2.0, alpha=0.2)
    ax.plot(t1_x, t1_z, spring_color, lw=2.0, alpha=0.2)

    # Calculate spring end position    
    ball_x = x 
    ball_y = z 
    
    ball = plt.Circle((ball_x, ball_y), ball_radius, color='k',alpha=0.2)
    ax.add_patch(ball)
    
    # colors = ['r', 'g', 'k', 'b']
    # labels = ['Start', 'Goal', 'Flight', 'Stance']
    colors = ['r', 'k', 'b']
    labels = ['Start', 'Flight', 'Stance']
    proxy_artists = [plt.Line2D([0], [0], color=color, lw=2.0) for color in colors]
    # proxy_artists = [proxy_artists, 
    #                  plt.Line2D([0], [0], marker='D', color='r', markerfacecolor='green', markersize=10)
    #                 #  plt.Line2D([0], [0], marker='D', color='g', markerfacecolor='red', markersize=10)
    #                  ]
    
    ax.legend(proxy_artists, labels, loc='best', prop={'family': 'serif', 'size': 12})
    
    # ax.set_xlim(-0.5, 1.5)
    # ax.set_ylim(-0.5, 2.0)
    
    plt.tight_layout()
    

def plot_slip_stance_animate(state_stance, xp, ax=None, spring_color='b-'):
    if ax is None:
        fig, ax = plt.subplots()
        ax.grid(True)

    # Parameters
    theta, r = state_stance[0], state_stance[2]

    theta_spring = np.pi / 2 - theta
    
    # Generate spring data
    t = np.linspace(0, 2 * np.pi * spring_coils, 1000)
    x_spring = spring_amplitude * np.sin(t)
    y_spring = np.linspace(0, 0.5*r, 1000)
    
    x_spring_rot = x_spring * np.cos(theta_spring) + y_spring * np.sin(theta_spring)
    z_spring_rot = -x_spring * np.sin(theta_spring) + y_spring * np.cos(theta_spring)

    x_spring_rot = x_spring_rot + xp
    
    # Plot the spring
    # ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2.0)
    ax.plot(x_spring_rot, z_spring_rot, spring_color, lw=2.0, alpha=0.2)

    # Calculate spring end position
    spring_end_x = xp + r * np.cos(theta)
    spring_end_z = r * np.sin(theta)
    
    # Plot the line connecting the ends of the spring
    t1_x = np.linspace(x_spring_rot[-1], spring_end_x, 1000)
    t1_z = np.linspace(z_spring_rot[-1], spring_end_z, 1000)
    
    x_tail = xp
    z_tail = 0.0
    t2_x = np.linspace(x_tail, x_spring_rot[0], 1000)
    t2_z = np.linspace(z_tail, z_spring_rot[0], 1000)
    
    ax.plot(t2_x, t2_z, spring_color, lw=2.0, alpha=0.2)
    ax.plot(t1_x, t1_z, spring_color, lw=2.0, alpha=0.2)
    
    ball_x = spring_end_x
    ball_y = spring_end_z
    
    ball = plt.Circle((ball_x, ball_y), ball_radius, color='k', alpha=0.2)
    ax.add_patch(ball)
    
    plt.tight_layout()
    

def unpad_control_slip(modes, inputs_padded, n_inputs):
    nt = modes.shape[0]
    inputs = [np.zeros((nt, 3)), np.zeros((nt, 2))]
    
    for i in range(nt):
        
        if modes[i] == 0:
            input_i = inputs_padded[i, :n_inputs[0]]
        elif modes[i] == 1:
            input_i = inputs_padded[i, :n_inputs[1]]
            
        inputs[modes[i]][i] = input_i
    
    return inputs

def unpad_state_slip(modes, states_padded):
    
    nt = modes.shape[0]
    states = [np.array([0.0]) for _ in range(nt)]
    
    for i in range(nt):
        if modes[i] == 0:
            state_i = states_padded[i]
        elif modes[i] == 1:
            state_i = states_padded[i, 0:4]
        states[i] = state_i
    
    return states

    
if __name__ == '__main__':
    r0 = 1.0
    xp = 1.0
    stance_state = np.array([np.pi/4, 0.0, 0.9*r0, 0.0], dtype=np.float64)
    plot_slip_stance_animate(stance_state, xp)
    
    flight_state = np.array([3.0, 0.0, 3.0, 0.0, np.pi/3], dtype=np.float64)
    plot_slip_flight_animate(flight_state, r0)
    
    plt.show()
    
    
def animate_slip(modes, states, 
                 init_mode, init_state, 
                 target_mode, target_state, nt, 
                 reset_args, target_reset_args,step=1):
    r0 = 1
    fig, ax = plt.subplots(figsize=(8,9))
    # Define the desired font properties
    font = FontProperties()
    font.set_family('serif')     # Choose font family (e.g., 'sans-serif', 'serif')
    font.set_size(18)             # Set font size

    # Apply font properties to x and y tick labels
    for tick in ax.get_xticklabels():
        tick.set_fontproperties(font)
    for tick in ax.get_yticklabels():
        tick.set_fontproperties(font)
        
    # ax.tick_params(axis='both', which='major', labelsize=16)
    # ax.grid(True)
    for ii in range(0,nt,step):
        if modes[ii] == 0:
            plot_slip_flight_animate(states[ii].flatten(), r0, ax)
        elif modes[ii] == 1:
            plot_slip_stance_animate(states[ii].flatten(), reset_args[ii][0], ax)
    
    ii_end = nt-1
    if modes[ii_end] == 0:
        plot_slip_flight_animate(states[ii_end].flatten(), r0, ax)
    elif modes[ii_end] == 1:
        plot_slip_stance_animate(states[ii_end].flatten(), reset_args[ii_end][0], ax)

    # Plot start and goal 

    if init_mode == 0:
        plot_slip_flight_animate(init_state, r0, ax, 'r-')
    elif init_mode == 1:
        plot_slip_stance_animate(init_state, reset_args[0], ax, 'r-')
    
        
    if target_mode == 0:
        plot_slip_flight_animate(target_state, r0, ax, 'g-')
    elif target_mode == 1:
        plot_slip_stance_animate(target_state, target_reset_args, ax, 'g-')
        
    # Draw the ground
    ax.hlines(y=0, xmin=-0.5, xmax=1.5, color='k', linestyle='-', linewidth=3)
    
    return fig, ax