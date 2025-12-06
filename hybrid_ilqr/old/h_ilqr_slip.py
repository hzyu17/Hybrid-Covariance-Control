import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr.h_ilqr import solve_ilqr
from dynamics.trajectory_extension import extract_extensions
# Import SLIP dynamics
from dynamics.dynamics_discrete_slip import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    # ------------- 
    # SLIP example 
    # -------------
    dt = 0.002
    epsilon = 2.0
    dt_shrink = 0.9
    r0 = 1
    
    n_modes = 2
    
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    # --------------
    # SLIP Dynamics
    # --------------
    # mode 1 (flight): x = [px, vx, pz, vz, theta], u = [theta_dot]
    # mode 2 (stance): x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
    
    # For the slip dynamics, mode 1 has 1 input, and mode 2 has 2 inputs. 
    n_states = [5, 4]
    n_inputs = [1, 2]
    
    # ----------------------------
    # Case 1: vertical bouncing
    # ----------------------------
    # start_time = 0
    # end_time = 1.5
    # init_mode = 0
    # init_state = np.array([0.0, 0.0, 2.0, 0.0, np.pi/2], dtype=np.float64)    # Define the initial state to be the origin with no velocity
    # target_state = np.array([0.0, 0.0, 2.3, 0.0, np.pi/2], dtype=np.float64)  # Swing pendulum upright

    # # Time definitions
    # start_time = 0
    # end_time = 1.5
    # time_span = np.arange(start_time, end_time, dt).flatten()
    # nt = len(time_span)
    
    # # Terminal cost 
    # target_mode = 0
    # Q_T = 0.01*np.eye(n_states[0])
    
    # # Running costs
    # Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    
    # --------------------------
    # Case 2: Running one step
    # --------------------------
    init_mode = 1
    
    # Time definitions
    start_time = 0
    end_time = 0.5
    
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    # Terminal cost 
    target_mode = 0
    Q_T = 80.0*np.eye(n_states[0])
    
    # Running costs
    Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]
    init_theta_deg = 100
    init_theta = init_theta_deg / 180 * np.pi
    init_state = np.array([init_theta, -4.0, 0.5*r0, 0.0], dtype=np.float64)
    target_state = np.array([1.1, 2.5, 1.5, 0.0, np.pi/3], dtype=np.float64)  # Swing pendulum upright
    init_reset_args = [np.array([0.0]) for _ in range(nt)]
    target_reset_args = [np.array([0.0]) for _ in range(nt)]
    
    # ---------------- / slip example -----------------
    
    # ================================
    # solve for hybrid ilqr proposal
    # ================================
    exp_params = ExpParams()
    
    initial_guess = [0.0*np.ones((np.shape(time_span)[0],n_inputs[0])), 0.0*np.ones((np.shape(time_span)[0],n_inputs[1]))]
    symbolic_dynamics = [symbolic_flight_dynamics_slip, symbolic_stance_dynamics_slip]
    
    # place holders
    n_exp = 1
    n_samples = 0 
    
    exp_params.update_params(n_modes, init_mode, target_mode, 
                             n_states, init_state, target_state, 
                             start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, 
                             Q_k, R_k, Q_T, symbolic_dynamics, 
                             event_detect_slip, plot_slip, convert_state_21_slip, 
                             init_reset_args, target_reset_args, 
                             animate_slip)
    
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True)
    
    (modes,states,inputs,
     k_feedforward,K_feedback,
     current_cost,states_iter,
     ref_modechanges,ref_ext_helper, ref_reset_args) = hybrid_ilqr_result
    
    exp_data.add_nominal_data(hybrid_ilqr_result)
    
    (v_mode_change_ref, v_ext_bwd, v_ext_fwd, 
    v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
    v_kff_ext_bwd, v_kff_ext_fwd, _) = extract_extensions(ref_ext_helper, start_index = 0)


    show_results = True
    if show_results:
        plot_slip(time_span, modes, states, inputs, init_state, target_state, nt, ref_reset_args)
        animate_slip(modes, states, init_mode, init_state, target_mode, target_state, nt, ref_reset_args, target_reset_args,step=5)
    