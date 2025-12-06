from dynamics.ode_solver.dynamics import *
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
font_props = FontProperties(family='serif', size=16, weight='normal')
import sympy as sp


def dyn_bouncing(t, x, *args):
    """
    Args:
        t (_type_): time variable
        x (_type_): state
        args[0]: control input
    """
   
    if len(args) == 0:
        u = np.array([0.0], dtype=np.float64)
    else:
        u = args[0]
    return np.array([x[1], u[0]-9.81], dtype=np.float64)


def dyn_bouncing_jax(t, x, *args):
      
    if len(args) == 0:
        u = jnp.array([0.0], dtype=jnp.float64)
    else:
        u = args[0]
    return jnp.array([x[1], u[0]-9.81], dtype=jnp.float64)

def f_euler_bouncing(x,u,dt):
    return x + dyn_bouncing_jax(0.0, x, u)*dt
    

def gdWt_bouncing(x0, dWt, eps):
    B = np.array([[0],[1.0]], dtype=np.float64)
    return np.sqrt(eps) * B@dWt
    
    
def symbolic_dynamics_bouncing():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = sp.Matrix([u])
    states = sp.Matrix([z, z_dot])
    # Defining the dynamics of the system
    f_contin = sp.Matrix([z_dot, u-g])
    
    A_contin = f_contin.jacobian(states)
    B_contin = f_contin.jacobian(inputs)

    A_contin_func = sp.lambdify((states,inputs),A_contin)
    B_contin_func = sp.lambdify((states,inputs),B_contin)
    
    return (f_contin,A_contin_func,B_contin_func)


def sym_dyn_bouncing():
    g = 9.81
    z,z_dot,u,dt = sp.symbols('z z_dot u dt')

    # Define the states and inputs
    inputs = sp.Matrix([u])
    states = sp.Matrix([z, z_dot])
    # Defining the dynamics of the system
    f = sp.Matrix([z_dot, u-g])

    # Discretize the dynamics usp.sing euler integration
    f_disc = states+f*dt
    
    # Take the jacobian with respect to states and inputs
    A_disc = f_disc.jacobian(states)
    B_disc = f_disc.jacobian(inputs)

    f_disc_func = sp.lambdify((states,inputs,dt),f_disc)
    A_disc_func = sp.lambdify((states,inputs,dt),A_disc)
    B_disc_func = sp.lambdify((states,inputs,dt),B_disc)
    return (f_disc_func,A_disc_func,B_disc_func)
    

def hybrid_stochastic_integration(x0, u, current_mode, t_span, epsilon, RandN, dt, dt_shrinkingrate):
    dW = np.sqrt(dt)*RandN
    xt_next = stochastic_integration(x0, u, t_span, epsilon, dW)
    t_next = t_span[1]
    
    # Sandwich rule for contact detection
    # dt_shrinkingrate = 0.7
    t = t_span[0]
    
    next_mode = current_mode
    # Guard condition: direction is -1     
    if event_condition(x0, xt_next, guard_bouncing_12): # Hit the guard function.
        args = (x0, current_mode, u, t, t_next, xt_next, dt, dt_shrinkingrate, RandN, epsilon, stochastic_integration, guard_bouncing_12, reset_map_bouncing_12)
        xt_next, next_mode, _ = stoch_event_reactive_fun(args)
        
    return xt_next, next_mode


def stochastic_integration_bouncing(x0, u, t_span, epsilon, dW):
    return stochastic_integration(x0, u, t_span, epsilon, dW, dyn_bouncing, gdWt_bouncing)


guards_bouncing = {0:guard_bouncing_12, 1: guard_bouncing_21}
reset_maps_bouncing = {0:reset_map_bouncing_12, 1:reset_map_bouncing_21}

# ----------------------------- 
# Define condition functions
# -----------------------------
# --------------------------------- Condition: mode mismatch --------------------------------- 
def cond_mode_mismatch_bouncing(current_mode, ref_current_mode): 
    return (current_mode != ref_current_mode)

# --------------------------------- Condition: early arrival ---------------------------------
def cond_early_arrival_bouncing(current_mode, ref_current_mode, event_modechange): 
    return (current_mode==event_modechange[1]) and (ref_current_mode==event_modechange[0]) 

# Condition: guard function hit
def cond_guard_function_hit_bouncing(xt, xt_next, guard_func): 
    return ((guard_func(0.0, xt)>0) and (guard_func(0.0, xt_next)<=0))


def stochastic_feedback_rollout_bouncing(init_mode, x0, n_inputs, xt_ref, ref_modechanges, 
                                        ut, Kt, kt, target_state, Q_T, t0, tf, 
                                        epsilon, GaussianNoise, dt_shrinkingrate, 
                                        ref_ext_helper, init_reset_args):

    (v_event_modechange, v_ext_bwd, v_ext_fwd, 
    v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
    v_kff_ext_bwd, v_kff_ext_fwd, _) = extract_extensions(ref_ext_helper, start_index = 0)
    
    n_timestamps = len(xt_ref)
    
    dt = (tf - t0) / n_timestamps
    dt_int = dt
    
    # returning trajectory
    mode_trj = np.zeros(n_timestamps, dtype=np.int64)
    mode_trj[0] = init_mode
    
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0   
    
    # closed-loop controls 
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[0])), np.zeros((n_timestamps, n_inputs[1]))]
    
    # only consider the 1->2 reset for now (bouncing)
    current_guard = guards_bouncing[init_mode]
    
    cnt_mismatch = 0
    xt_ref_actual = np.zeros_like(xt_ref)
    
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
        
        ref_current_mode = ref_modechanges[ii_t][0]
        reset_args[ii_t] = event_args[cnt_event]
        
        # ======== Handle mode mismatch ========
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t]
        
        xref_i = xt_ref[ii_t] 
        if cond_mode_mismatch_bouncing(current_mode, ref_current_mode):
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mode_mismatch(ii_t, 
                                                                          current_mode, ref_current_mode, 
                                                                            v_ext_fwd[0], v_ext_bwd[0], 
                                                                            v_event_modechange[0],
                                                                            v_Kfb_ext_fwd[0], v_kff_ext_fwd[0],
                                                                            v_Kfb_ext_bwd[0], v_kff_ext_bwd[0],
                                                                            cnt_mismatch, cond_early_arrival_bouncing)
            
        xt_ref_actual[ii_t] = xref_i
        
        delta_xt_i = xt_trj[ii_t] - xref_i
        u = ut[current_mode][ii_t] + K_fb_i@delta_xt_i + k_ff_i
        ut_cl_trj[current_mode][ii_t] = u
        
        dW_i = np.sqrt(dt_int)*GaussianNoise[current_mode][ii_t]
        
        # ============================== One step integration ==============================        
        # ---- solver for the deterministic part
        t_span = (t0_i, t0_i + dt_int)
        
        xt_next = stochastic_integration_bouncing(xt, u, t_span, epsilon, dW_i).flatten()
        
        current_guard = guards_bouncing[current_mode]
        next_mode = current_mode
        # Condition: Hit the guard function.  
        if cond_guard_function_hit_bouncing(xt, xt_next, current_guard): 
            
            args = (xt, current_mode, u, t0_i, t0_i+dt_int, xt_next, 
                    dt_int, GaussianNoise[current_mode][ii_t], epsilon, 
                    stochastic_integration_bouncing, guards_bouncing, reset_maps_bouncing, reset_args[ii_t])
            
            xt_next, next_mode, dW_i, new_reset_args = stoch_event_reactive_fun(args)
            dt_int = dt
            
            event_args.append(new_reset_args)
            cnt_event += 1
        
        # Collect cost: consider only the terminal state cost for now.
        Sk += u.T@u/2.0 * dt + np.sqrt(epsilon) * np.dot(u.T, dW_i)
        
        # Update trajectories
        xt_trj[ii_t+1] = xt_next
        mode_trj[ii_t+1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
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
        ax5.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
        plt.tight_layout()
        
        ax6.plot(xt_trj[:,0], xt_trj[:,1],color='b',linewidth=1.5,label='Rollout')
        ax6.plot(xt_ref[:,0], xt_ref[:,1],color='k',linewidth=2.5,label='Reference')
        ax6.plot(xt_ref_actual[:,0], xt_ref_actual[:,1],color='r',linewidth=1.5,linestyle='--',label='Modified Reference')
        ax6.set_xlabel(r"z", fontproperties=font_props)
        ax6.set_ylabel(r"$\dot z$", fontproperties=font_props)
        ax6.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
        plt.tight_layout()
        
        plt.show()
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual


def event_detect_bouncing(x0, u, t0, tf, current_mode, reset_args, detection=True, backwards=False):
    guard_bouncing_12.terminal=True
    guard_bouncing_12.direction=-1
    
    guard_bouncing_21.terminal=True
    guard_bouncing_21.direction=-1
    
    smooth_dynamics_bouncing = {0:dyn_bouncing, 1:dyn_bouncing}
    
    Rxs_bouncing = {0:Rx_bouncing_12, 1:Rx_bouncing_21}
    Rts_bouncing = {0:Rt_bouncing_12, 1:Rt_bouncing_21}
    
    gxs_bouncing = {0:gx_bouncing_12, 1:gx_bouncing_21}
    gts_bouncing = {0:gt_bouncing_12, 1:gt_bouncing_21}
    
    guards_bouncing_bouncing = {0:guard_bouncing_12, 1: guard_bouncing_21}
    reset_maps_bouncing_bouncing = {0:reset_map_bouncing_12, 1:reset_map_bouncing_21}
        
    return event_detect_onestep(x0, u, 
                                t0, tf, 
                                current_mode, 
                                smooth_dynamics_bouncing, 
                                guards_bouncing_bouncing,
                                gxs_bouncing, gts_bouncing,
                                reset_maps_bouncing_bouncing,
                                Rxs_bouncing, Rts_bouncing,
                                reset_args, detection, backwards)
            

def convert_state_21_bouncing(state_2):
    return state_2


def plot_bouncingball_nexp(exp_indexes, exp_data, time_span, init_state, 
                            target_state, args=None):
    print("----------------- Plotting bouncing ball results -----------------")
    # =============== plotting ===============
    if args is not None:
        fig1, axes_12, fig2, ax3 = args
    else:
        fig1, axes_12 = plt.subplots(1, 2)
        fig2, ax3 = plt.subplots()
        
    (ax1, ax2) = axes_12.flatten()
    
    (modes,states_ref,inputs, 
    k_feedforward, K_feedback, current_cost, 
    states_iter, ref_modechanges,
    ref_ext_helper, ref_reset_args) = exp_data.get_nominal_data()
    
    states_ref = np.array(states_ref)
    
    for i_exp in exp_indexes:
        states_pi = exp_data.get_data(i_exp).x_trj_pi()
        states_pi = np.array(states_pi)
        
        states_ilqg = exp_data.get_data(i_exp).x_trj_ilqr()
        states_ilqg = np.array(states_ilqg)
        
        # ----------- Plot the last iteration of iLQR controller ----------
        if i_exp == exp_indexes[-1]:
            
            ax1.plot(time_span[:], states_ilqg[:,0], color='b', alpha=0.8, label='H-iLQR', linewidth=0.7)
            ax1.plot(time_span[:], states_pi[:,0], color='r', alpha=0.8, label='H-PI', linewidth=0.7)
            ax1.plot(time_span[:], states_ref[:,0], color='k', alpha=1.0, label='Nominal', linewidth=0.7)
            
            ax2.plot(time_span[:], states_ilqg[:,1], color='b', alpha=0.8, label='H-iLQR', linewidth=0.7)
            ax2.plot(time_span[:], states_pi[:,1], color='r', alpha=0.8, label='H-PI', linewidth=0.7)
            ax2.plot(time_span[:], states_ref[:,1], color='k', alpha=1.0, label='Nominal', linewidth=0.7)
            
            # ------------- Plot the z-\dot_z figure -------------
            ax3.plot(states_ilqg[:,0], states_ilqg[:,1],color='b', alpha=0.8, label='H-iLQR', linewidth=0.7)
            ax3.plot(states_pi[:,0], states_pi[:,1],color='r', alpha=0.8, label='H-PI', linewidth=0.7)
            ax3.plot(states_ref[:,0], states_ref[:,1],color='k', alpha=1.0, label='Nominal', linewidth=0.7)
            
        else:
            
            ax1.plot(time_span[:], states_ilqg[:,0], color='b', alpha=0.8, linewidth=0.7)
            ax1.plot(time_span[:], states_pi[:,0], color='r', alpha=0.8, linewidth=0.7)
            ax1.plot(time_span[:], states_ref[:,0], color='k', alpha=1.0, linewidth=0.7)
            
            ax2.plot(time_span[:], states_ilqg[:,1], color='b', alpha=0.8, linewidth=0.7)
            ax2.plot(time_span[:], states_pi[:,1], color='r', alpha=0.8, linewidth=0.7)
            ax2.plot(time_span[:], states_ref[:,1], color='k', alpha=1.0, linewidth=0.7)
            
            ax3.plot(states_ilqg[:,0], states_ilqg[:,1],color='b', alpha=0.8, linewidth=0.7)
            ax3.plot(states_pi[:,0], states_pi[:,1],color='r', alpha=0.8, linewidth=0.7)
            ax3.plot(states_ref[:,0], states_ref[:,1],color='k', alpha=1.0, linewidth=0.7)
    
    # ----------- Plot the start and goal states -----------
    ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

    ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    # ----------- Plot the start and goal states -----------
    ax3.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
    ax3.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
    ax1.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
    ax2.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
    ax3.legend(loc='upper right', prop={'family': 'serif', 'size': 15})

    ax1.set_xlabel(r"Time", fontproperties=font_props)
    ax1.set_ylabel(r"$z$", fontproperties=font_props)
    ax1.set_title(r"Bouncing Ball Vertical Position", fontproperties=font_props)

    ax2.set_xlabel(r"Time", fontproperties=font_props)
    ax2.set_ylabel(r"$\dot z$", fontproperties=font_props)
    ax2.set_title(r"Bouncing Ball Vertical Velocity", fontproperties=font_props)
    
    font = FontProperties()
    font.set_family('serif')     # Choose font family (e.g., 'sans-serif', 'serif')
    font.set_size(18)             # Set font size

    # Apply font properties to x and y tick labels
    for tick in ax1.get_xticklabels():
        tick.set_fontproperties(font)
    for tick in ax1.get_yticklabels():
        tick.set_fontproperties(font)
    
    for tick in ax2.get_xticklabels():
        tick.set_fontproperties(font)
    for tick in ax2.get_yticklabels():
        tick.set_fontproperties(font)
        
    for tick in ax3.get_xticklabels():
        tick.set_fontproperties(font)
    for tick in ax3.get_yticklabels():
        tick.set_fontproperties(font)

    return fig1, axes_12, fig2, ax3

def plot_bouncingball(time_span, modes, states, inputs, init_state, 
                      target_state, nt, 
                      reset_args=None,
                      color='k', 
                      args=None, 
                      plot_start_goal=True,
                      trj_labels='iLQG-reference',
                      step=1):
    print("----------------- Plotting bouncing ball results -----------------")
    # =============== plotting ===============
    if args is not None:
        fig1, axes_12, fig2, ax3 = args
    else:
        fig1, axes_12 = plt.subplots(1, 2)
        fig2, ax3 = plt.subplots(1, 1, figsize=(10,10))
        
    (ax1, ax2) = axes_12.flatten()
    ax1.grid(True)
    ax2.grid(True)

    # ----------- Plot the reference -----------
    states = np.array(states)
    ax1.plot(time_span[:], states[:,0], color=color)
    ax2.plot(time_span[:], states[:,1], color=color)

    # =========== Plot the z-\dot_z figure ===========
    
    ax3.grid(True)

    # ----------- Plot the last iteration of iLQR controller ----------
    ax3.plot(states[:,0], states[:,1],color=color,label=trj_labels)

    
    if plot_start_goal:
        # ----------- Plot the start and goal states -----------
        ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
        
        # ----------- Plot the start and goal states -----------
        ax3.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax3.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
        
        ax1.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
        ax2.legend(loc='upper right', prop={'family': 'serif', 'size': 15})
        ax3.legend(loc='upper right', prop={'family': 'serif', 'size': 15})

    ax1.set_xlabel(r"Time", fontproperties=font_props)
    ax1.set_ylabel(r"$z$", fontproperties=font_props)
    ax1.set_title(r"Bouncing Ball Vertical Position", fontproperties=font_props)

    ax2.set_xlabel(r"Time", fontproperties=font_props)
    ax2.set_ylabel(r"$\dot z$", fontproperties=font_props)
    ax2.set_title(r"Bouncing Ball Vertical Velocity", fontproperties=font_props)
    
    ax3.set_xlabel(r"$z$", fontproperties=font_props)
    ax3.set_ylabel(r"$\dot z$", fontproperties=font_props)
    ax3.set_title(r"Bouncing Ball State Plot", fontproperties=font_props)

    return fig1, axes_12, fig2, ax3

