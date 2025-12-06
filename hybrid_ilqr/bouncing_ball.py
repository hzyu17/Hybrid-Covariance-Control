import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
sys.path.append(root_dir)


# Import iLQR class
from hybrid_ilqr import solve_ilqr
# Import Riccati class
from hybrid_riccati import *
# Importing path integral control
from hybrid_pathintegral.hybrid_pathintegral import *
# Import plotting
import matplotlib.pyplot as plt
# Import experiment parameter class
from experiments.exp_params import *


if __name__ == '__main__':
    # ---------------- bouncing example -----------------
    dt = 0.01
    epsilon = 2.0
    dt_shrink = 0.7
    
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([3.5, 0])  # Swing pendulum upright

    # ---------------- / bouncing example -----------------

    # ===== OR =====
    # dt = 5e-5
    # # ------------- verification with no contact ------------- 
    # start_time = 0
    # end_time = 1.0
    # time_span = np.arange(start_time, end_time, dt).flatten()
    # nt = len(time_span)

    # init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    # target_state = np.array([1.0, 0.0])

    # # ------------- /verification with no contact ------------- 

    # Set desired state
    n_states = 2
    n_inputs = 1

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = np.eye(n_inputs)

    # ---------------------------- Set the terminal cost ----------------------------
    Q_T = 200*np.eye(n_states)
    Q_T[0,0] = 2000.0

    n_exp = 1
    n_samples = 10

    # ====================================
    # solve for hybrid ilqr proposal
    # ====================================
    exp_params = ExpParams()
    initial_guess = 0.5*np.ones((np.shape(time_span)[0],n_inputs))
    exp_params.update_params(init_state, target_state, start_time, end_time, dt, initial_guess, 
                             epsilon, n_exp, n_samples, Q_k, R_k, Q_T, sym_dyn_bouncing,detect_bouncing)
    exp_data = ExpData(exp_params)
    hybrid_ilqr_result = solve_ilqr(exp_params, detect=True)
    
    (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps) = hybrid_ilqr_result
    
    exp_data.add_nominal_data((states,inputs,k_feedforward,K_feedback,current_cost,states_iter))


    show_results = True
    if show_results:
        print("plotting results")
        # =============== plotting ===============
        fig1, axes = plt.subplots(1, 2)
        (ax1, ax2) = axes.flatten()
        ax1.grid(True)
        ax2.grid(True)

        # ----------- Plot the start and goal states -----------
        ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        # ----------- Plot the reference -----------
        ax1.plot(time_span, states[:,0],'k',label='iLQR-deterministic')
        ax2.plot(time_span, states[:,1],'k',label='iLQR-deterministic')

        ax1.set_xlabel(r"Time")
        ax1.set_ylabel(r"$z$")
        ax1.set_title("Bouncing Ball Vertical Position")

        ax2.set_xlabel(r"Time")
        ax2.set_ylabel(r"$\dot z$")
        ax2.set_title("Bouncing Ball Vertical Velocity")

        ax1.legend()
        ax2.legend()

        # =========== Plot the z-\dot_z figure ===========
        fig2, ax5 = plt.subplots()
        ax5.grid(True)

        # ----------- Plot the last iteration of iLQR controller ----------
        ax5.plot(states[:,0], states[:,1],'k',label='iLQR-reference')

        # ----------- Plot the start and goal states -----------
        ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

        ax5.legend()
        
        plt.show()