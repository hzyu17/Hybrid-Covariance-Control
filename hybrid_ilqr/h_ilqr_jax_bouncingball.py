import numpy as np
import os
import sys

file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr.h_ilqr_jax import *
from dynamics.trajectory_extension import *
# Import bouncing ball dynamics
from dynamics.dynamics_discrete_bouncing import *
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    
    # Set desired state
    n_modes = 2
    
    # the state and control dimensions, mode-dependent
    n_states = [2, 2]
    n_inputs = [1, 1]
    
    # ---------------- multiple bouncing example -----------------
    dt = 0.001
    epsilon = 2.0
    dt_shrink = 0.95
    
    start_time = 0
    end_time = 3.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([1.2, 0.0])  # Swing pendulum upright
     
    init_mode = 0
    target_mode = 0
    Q_T = 60*np.eye(n_states[0])

    # ---------------- / multiple bouncing example -----------------
    
    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # ---------------------------- Set the terminal cost ----------------------------

    n_exp = 1
    n_samples = 10
    
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ====================================
    #    Solve for hybrid ilqr proposal
    # ====================================
    exp_params = ExpParams()
    
    initial_guess = [1.0*np.ones((np.shape(time_span)[0],n_inputs[0])), 1.0*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    
    flow_dynamics = [sym_dyn_bouncing, sym_dyn_bouncing]
    niters = 10
    
    hilqr_obj = hybrid_ilqr_jax(n_states, n_inputs, 
                                init_mode, init_state, target_state, 
                                initial_guess, 
                                time_span, 
                                niters, 
                                is_detect=True, 
                                detect_func=event_detect_bouncing_discrete, 
                                smooth_dynamics=f_euler_bouncing, 
                                running_cost=bouncingball_cost, 
                                cost_args=0.0,
                                terminal_cost=deltx_norm_cost, 
                                terminal_cost_args=target_state)
    
    hybrid_ilqr_result = hilqr_obj.solve()
    
    
    (timespan,modes,states,inputs,
    k_feedforward,K_feedback,
    current_cost,states_iter,
    modechanges,ref_ext_helper) = hybrid_ilqr_result

    show_results = True
    if show_results:
        plot_bouncingball(time_span, modes, states, inputs, init_state, target_state, nt, color='k')
        plt.show()