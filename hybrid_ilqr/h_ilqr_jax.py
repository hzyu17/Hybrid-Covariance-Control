# Allows for general nonlinear cost functions and 
# jacobian computations using jax automatic differentiations 
# Only considering 1 same smooth flow (stance mode dynamics) in all the modes.

# For walking robot, we assume 2 modes: divided by the swing foot height and velocity sign.

import jax
from jax import grad, jacfwd, hessian
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from functools import partial

from dynamics.dynamics_discrete_bouncing import *

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.trajectory_extension import *
# from walking_3link import *

# from dynamics.saltation_matrix import compute_saltation

class hybrid_ilqr_jax:
    def __init__(self, 
                 nstates, ninputs,
                 init_mode, init_state, target_state,
                 initial_guess,
                 timespan, n_iterations, 
                 is_detect, detect_func,
                 smooth_dynamics,
                 running_cost, cost_args,
                 terminal_cost, terminal_cost_args):
        
        self._init_mode = init_mode

        self._nx = nstates
        self._nu = ninputs

        self._init_state = init_state
        self._target_state = target_state
        self._inputs = initial_guess
        self._initial_guess = initial_guess
        self._verbose = is_detect
        
        # time definitions
        self._timespan = timespan
        self._dt = timespan[1:] - timespan[:-1]
        self._starttime = timespan[0]
        self._endtime = timespan[-1]
        
        # self._timespan = np.arange(start_time, end_time, dt).flatten()
        self._nt = np.shape(self._timespan)[0]
        
        self._states = [np.zeros(self._nx[0]) for _ in range(self._nt)]
        
        # feedback and feedforward
        self._k_ff = [np.zeros((self._nu[0])) for _ in range(self._nt)]
        self._K_fb = [np.zeros((self._nu[0], self._nx[0])) for _ in range(self._nt)]
        
        # hybrid events and mode changes
        self._saltations = [np.array([None]) for i in range(self._nt)]
        self._modechanges = [np.array([0, 0]) for _ in range(self._nt)]
        self._modes = [0 for _ in range(self._nt)]
        
        # feedback and feedforward for the trajectory extensions 
        # shapes: feedback gains: [[K_fwd_extension, K_bwd_extension]_jump1, [K_fwd_extension, K_bwd_extension]_jump2, ...]
        # shapes: feedforwad gains: [[k_fwd_extension, k_bwd_extension]_jump1, [k_fwd_extension, k_bwd_extension]_jump2, ...]
        self._k_ff_ext = [np.zeros((self._nu[0])), np.zeros((self._nu[1]))]
        self._K_fb_ext = [np.zeros((self._nu[0], self._nx[0])), np.zeros((self._nu[1], self._nx[1]))]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        #   Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self._event_info = {}
        self._ref_ext_helper = []
        
        # Dynamics
        self._smooth_dyn = smooth_dynamics
        self._detect_func = partial(detect_func, detect=is_detect)

        # Jacobians of the smooth dynamics
        self._A = jax.jit(jacfwd(lambda x, u, dt: smooth_dynamics(x, u, dt), argnums=0))
        self._B = jax.jit(jacfwd(lambda x, u, dt: smooth_dynamics(x, u, dt), argnums=1))

        # Running costs
        self._running_cost = running_cost
        self._cost_args = cost_args
        
        # Gradients of the cost functions
        self._cost_x = jax.jit(grad(lambda x, u: running_cost(x, u, cost_args), 0))
        self._cost_u = jax.jit(grad(lambda x, u: running_cost(x, u, cost_args), 1))
        self._cost_xx = jax.jit(hessian(lambda x, u: running_cost(x, u, cost_args), 0))
        self._cost_uu = jax.jit(hessian(lambda x, u: running_cost(x, u, cost_args), 1))

        # Terminal costs
        self._terminal_cost = terminal_cost
        self._terminal_cost_args = terminal_cost_args

        # Gradients of the terminal costs
        self._terminal_cost_x = jax.jit(grad(lambda x: terminal_cost(x, terminal_cost_args), 0))
        # self._terminal_cost_xx = jacfwd(lambda x: self._terminal_cost_x(x), 0)
        self._terminal_cost_xx = jax.jit(jax.hessian(lambda x: self._terminal_cost(x, terminal_cost_args), 0))
        
        # Max iterations
        self._niters = n_iterations
        
    def desamble_control(self, modes, mode_inputs):
        nt = len(modes)
        inputs = np.zeros((nt, self._nu[0]))
        for i in range(nt):
            mode = modes[i]
            inputs[i,:] = mode_inputs[mode][i,:].flatten()

        return inputs
    
    def rollout(self, timespan, modes, states, inputs):        
        """Forward pass without feedback.

        Returns:
            Rollout trajectory and related hybrid information.
        """
        # temporary variables in the current forward pass
        dts = timespan[1:] - timespan[:-1]
        nt = len(timespan)
        
        saltations = [None for i in range(nt)]
        mode_changes = np.tile(np.array([0, 0]), (nt, 1))

        # Set the first state to be the initial
        current_state = self._init_state
        current_mode = self._init_mode
        
        modes[0] = current_mode
        states[0] = current_state
        mode_changes[0] = np.array([current_mode, current_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        event_info = {} # The dictionary that stores all the information of the jump dynamics and states.
        cnt_event = 0
        
        ii = 0
        while (ii < nt-1):
            
            x_i = states[ii]
            mode_i = modes[ii]
            
            # ------------------- 
            #  Open-loop control 
            # ------------------- 
            u_i = inputs[mode_i][ii]
            
            # ====================================
            #  Simulate forward using the control
            # ====================================
            t_ii = timespan[ii]
            dt_ii = dts[ii]
            
            (next_state, saltation, 
             mode_change, t_event, x_event, 
             x_reset, _) = self._detect_func(mode_i, x_i, u_i, t_ii, dt_ii, reset_args=None)

            # Hybrid event happened
            if mode_change[0] != mode_change[1]:     
                print("---- hybrid event happened at ----", t_event)            
                states[ii+1] = x_reset

                hybrid_index.add(ii)                
                saltations[ii] = saltation
                event_info[ii] = (t_event, x_event, x_reset, mode_change, self._K_fb_ext, self._k_ff_ext) # Add the default extension gains
                cnt_event += 1
                
            # No hybrid event
            else:                                 
                states[ii+1] = next_state
            
            # ---------------------
            #  Move forward in time
            # ---------------------            
            mode_changes[ii] = mode_change
            modes[ii+1] = mode_change[1]
            
            ii += 1
              
        return timespan, modes,states,inputs,saltations,mode_changes,event_info
    
    
    # def forward_pass(self, timespan, modes, states, inputs,
    #                  ref_ext_helper, learning_rate=1):
    def forward_pass(self, learning_rate=1):
        # temporary variables in the current forward pass
        timespan = self._timespan
        
        dts = timespan[1:] - timespan[:-1]
        nt = len(timespan)
        
        modes = [0 for _ in range(self._nt)]
        states = [np.array([0.0]) for _ in range(self._nt)]
        inputs = [np.zeros((self._nt, self._nu[0])), np.zeros((self._nt, self._nu[1]))]
        saltations = [None for _ in range(nt)]
        mode_changes = np.tile(np.array([0, 0]), (nt, 1))

        # Set the first states to be the initial     
        modes[0] = self._init_mode
        states[0] = self._init_state
        mode_changes[0] = np.array([self._init_mode, self._init_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        event_info = {} # The dictionary that stores all the information of the jump dynamics and states.
        cnt_event = 0
        hybrid_index_ref = 0
        
        # -------------------------------
        #  Current rollout hybrid events 
        # -------------------------------
        K_fb_ext = []
        k_ff_ext = []
        
        # Reference hybrid events and extensions from the last iteration
        (v_modechg_ref, v_ext_bwd, v_ext_fwd, 
        v_Kfb_ext_bwd, v_Kfb_ext_fwd, 
        v_kff_ext_bwd, v_kff_ext_fwd, v_tevents_ref) = extract_extensions(self._ref_ext_helper)
        
        # # ------------------------ Plot reference extensions ------------------------ 
        # states_arr = np.asarray(self._states)
        # fig, ax = plt.subplots()
        # ax.grid(True)
        # ax.plot(states_arr[:,0], states_arr[:,1], 'k')
        # for i_ext_trj_bwd in v_ext_bwd[1:2]:
        #     ax.plot(i_ext_trj_bwd[:,0], i_ext_trj_bwd[:,1], 'r', label="backward extension")
        # for i_ext_trj_fwd in v_ext_fwd[1:2]: 
        #     ax.plot(i_ext_trj_fwd[:,0], i_ext_trj_fwd[:,1], 'b', label="forward extension")
            
        # ax.legend()
        # ax.set_xlabel(r"$z$")
        # ax.set_ylabel(r"$\dot z$")
        # ax.set_title(r"Bouncing Ball State Plot")

        # plt.show()

        # Construct local extension feedback and feedforward gains
        for i_ext in range(len(v_modechg_ref)):
            K_fb_ext.append((v_Kfb_ext_fwd[i_ext], v_Kfb_ext_bwd[i_ext]))
            k_ff_ext.append((v_kff_ext_fwd[i_ext], v_kff_ext_bwd[i_ext]))
            
        
        # The main loop forward in time
        ii = 0
        while (ii < nt-1):
            
            x_i = states[ii]
            mode_i = modes[ii]
            
            # ------------------- 
            # Get the references 
            # ------------------- 
            u_i = self._inputs[mode_i][ii]
            
            # ===========================================
            #  Choose the feedback and feedforward gains
            # ===========================================
               
            ref_state = self._states[ii]
            mode_i_ref = self._modes[ii]
            
            # ----------------------------------------------- 
            #  Get the current (feedback, feedforward) gains 
            # ----------------------------------------------- 
            current_feedforward = learning_rate * self._k_ff[ii]
            current_feedback = self._K_fb[ii]
            
            # ---------------
            #  Mode Mismatch
            # --------------- 
            if mode_i != mode_i_ref:
                # print("mode mismatch at: ", ii)
                
                trj_extension = []
                fb_ext_trj = []
                ff_ext_trj = []
                
                hybrid_index_ref = np.argmin(abs(np.array(v_tevents_ref)-ii)) # find the nearest hybrid event in the reference
                ref_modechange_hybrid = v_modechg_ref[hybrid_index_ref]
                
                # ----------------------------------- Early Arrival ----------------------------------- 
                if ((mode_i == ref_modechange_hybrid[1]) and (mode_i_ref==ref_modechange_hybrid[0])):
                    # print(f"early arrival, Time: {ii}. Current mode: {mode_i}, Reference mode: {mode_i_ref}")
                    # print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        
                    trj_extension = v_ext_bwd[hybrid_index_ref]
                    
                    ff_ext_trj = v_kff_ext_bwd[hybrid_index_ref]
                    fb_ext_trj = v_Kfb_ext_bwd[hybrid_index_ref]
                    
                # ----------------------------------- Late Arrival ----------------------------------- 
                elif ((mode_i == ref_modechange_hybrid[0]) and (mode_i_ref==ref_modechange_hybrid[1])):
                    # print(f"late arrival, Time: {ii}. Current mode: {mode_i}, Reference mode: {mode_i_ref}")
                    # print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                    trj_extension = v_ext_fwd[hybrid_index_ref]
                    
                    ff_ext_trj = v_kff_ext_fwd[hybrid_index_ref]
                    fb_ext_trj = v_Kfb_ext_fwd[hybrid_index_ref]
                
                # Modify the reference to the extension
                ref_state = trj_extension[ii]
                current_feedback = fb_ext_trj[ii]
                current_feedforward = learning_rate * ff_ext_trj[ii]
                
            current_feedback_input = current_feedback@(x_i-ref_state)
    
            # Update the nominal control
            u_i = u_i + current_feedback_input + current_feedforward
            
            # ====================================
            #  Simulate forward using the control
            # ====================================
            t_ii = timespan[ii]
            dt_ii = dts[ii]
            
            (next_state, saltation, 
             mode_change, t_event, x_event, 
             x_reset, _) = self._detect_func(mode_i, x_i, u_i, t_ii, dt_ii, reset_args=None)

            # --------------------------------
            #       Hybrid event happened
            # --------------------------------
            if mode_change[0] != mode_change[1]:      
                print("---- hybrid event happened at ----", t_event)           
                states[ii+1] = x_reset

                hybrid_index.add(ii)                

                saltations[ii] = saltation
                event_info[ii] = (t_event, x_event, x_reset, mode_change, K_fb_ext[hybrid_index_ref], k_ff_ext[hybrid_index_ref])
                # hybrid_index_ref += 1
                cnt_event += 1
            else:
                states[ii+1] = next_state
                        
            # ---------------------
            #  Move forward in time
            # ---------------------            
            states[ii+1] = next_state
            inputs[mode_i][ii] = u_i.flatten()
            mode_changes[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
         
            ii += 1
        
        # ----- For five link system -----
        show_forwardpass = False
        # if show_forwardpass:
        #     from five_link.fivelink_simulation import animate_trj
        #     animate_trj(np.array(states))
        
        show_fwdpass = False
        if show_fwdpass:
            
            plot_bouncingball(self._timespan, modes, states, inputs, 
                                self._init_state, self._target_state, 
                                self._nt, reset_args=None, step=200)
        
        return (timespan, modes,states,inputs,saltations,mode_changes,event_info)
    

    def compute_cost(self,modes,states,inputs,timespan):
        # Initialize cost
        total_cost = 0.0
        dt = timespan[1:] - timespan[:-1]
        nt = len(timespan)
        
        for ii in range(0,nt-1):
            mode_i = modes[ii]
            x_i = states[ii] # Not being used currently
            u_i = inputs[mode_i][ii].flatten()

            total_cost = total_cost+self._running_cost(x_i, u_i, self._cost_args)*dt[ii]
            
            if np.isnan(total_cost):
                print("NaN found, stopping loop")
                print("ii: ", ii)
                print("x: ", x_i)
                print("u: ", u_i)
                break
        print("Running cost: ", total_cost)
        
        # Compute terminal cost
        print("Terminal cost: ", self._terminal_cost(states[-1], self._terminal_cost_args))
        total_cost = total_cost + self._terminal_cost(states[-1], self._terminal_cost_args)

        return total_cost

    def backward_pass(self, timespan, modes, states, inputs, saltations, event_info):
        
        nt = timespan.shape[0]
        dt = timespan[1:] - timespan[:-1]
        
        V_xx = self._terminal_cost_xx(self._states[-1])
        V_x = self._terminal_cost_x(self._states[-1])

        k_trj = [np.zeros((self._nu[0])) for _ in range(nt)]
        K_trj = [np.zeros((self._nu[0], self._nx[0])) for _ in range(nt)]
        
        k_ff_trj_ext = []
        K_fb_trj_ext = []
        
        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        
        # for loop backwards in time
        for idx in reversed(range(0, nt-1)):
            # Grab the current variables in the trajectory
            mode_i = modes[idx]
            x_i = states[idx]
            u_i = inputs[mode_i][idx]
            saltation_i = saltations[idx]
            dt_i = dt[idx]
            
            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self._cost_xx(x_i, u_i) # For now zeros, can add in a target to track later on
            l_uu = self._cost_uu(x_i, u_i)

            l_x = self._cost_x(x_i, u_i) # For now zeros, can add in a target to track later on
            l_u = self._cost_u(x_i, u_i)

            # Get the jacobian of the discretized dynamics
            A_k = self._A(x_i, u_i, dt_i)
            B_k = self._B(x_i, u_i, dt_i)
            
            if saltation_i is None:
                Q_x = l_x*dt_i + A_k.T@V_x
                Q_u = l_u*dt_i + B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*dt_i + B_k.T@V_xx@B_k
                Q_xx = l_xx*dt_i + A_k.T@V_xx@A_k
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._nu[0], self._nx[0]))
                
            else:
                # print("saltation_i: ", saltation_i)
                # print("Found contact dynamics! Computing the gains with saltation matrix.")
                Q_x = l_x*dt_i + A_k.T @ saltation_i.T @ V_x
                Q_u = l_u*dt_i + B_k.T @ saltation_i.T @ V_x
                Q_ux = B_k.T @ saltation_i.T @ V_xx @ saltation_i @ A_k
                Q_uu = l_uu*dt_i + B_k.T @ saltation_i.T @ V_xx @ saltation_i @ B_k
                Q_xx = l_xx*dt_i + A_k.T @ saltation_i.T @ V_xx @ saltation_i @ A_k    
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._nu[mode_i], self._nx[mode_i]))

                # Compute the (gains for the forward extension, gains for the backward extension): use the gain at the immediate next state (reseted)
                k_ff_trj_ext.append((k, k_trj[idx+1]))
                K_fb_trj_ext.append((K, K_trj[idx+1]))
                
                # update the hybrid dynamics information
                previous_event_info = list(event_info[idx])
                previous_event_info[4] = [K, K_trj[idx+1]]
                previous_event_info[5] = [k, k_trj[idx+1]]
                event_info[idx] = tuple(previous_event_info)

            # Store gains
            k_trj[idx] = k
            K_trj[idx] = K

            # Update the expected reduction
            current_cost_reduction_grad = -Q_u.T@k
            current_cost_reduction_hess = 0.5 * k.T @ (Q_uu) @ (k)
            current_cost_reduction = current_cost_reduction_grad + current_cost_reduction_hess

            expected_cost_reduction_grad +=  current_cost_reduction_grad
            expected_cost_reduction_hess +=  current_cost_reduction_hess
            expected_cost_reduction += + current_cost_reduction

            # Update hessian and gradient for value function (If we arent using regularization we can simplify this computation)
            # V_x = Q_x +K.T@Q_uu@k + K.T@Q_u + Q_ux.T@k
            # V_xx = (Q_xx+Q_ux.T@K+K.T@Q_ux+K.T@Q_uu@K)
            V_x = Q_x - K.T@Q_uu@k
            V_xx = Q_xx - K.T@Q_uu@K

        # Store expected cost reductions
        self._expt_cost_redugrad_ = expected_cost_reduction_grad
        self._expt_cost_reduhess_ = expected_cost_reduction_hess
        
        
        # Store the gain for the backward extensions
        K_fb_trj_ext.reverse()
        k_ff_trj_ext.reverse()
        
        return (k_trj,K_trj,k_ff_trj_ext,K_fb_trj_ext,expected_cost_reduction,event_info)    
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        
        # ------------------------------------
        #  First rollout using initial guess
        # ------------------------------------
        print("--------------------- Starting initial rollout ---------------------")
        (timespan, 
         modes, states, inputs, 
         saltations, mode_changes, event_info) = self.rollout(self._timespan, self._modes, self._states, self._inputs)
        
        
        # ----- For five link system -----
        show_forwardpass = True
        if show_forwardpass:
            from five_link.fivelink_simulation import FiveLinkSimulator, animate_trj, draw_5link
            u_trj_arr = self.desamble_control(modes, inputs)
            dts = timespan[1:] - timespan[:-1]
            fivelink_simulator = FiveLinkSimulator(self._init_state, u_trj_arr, dts)
    
            fivelink_simulator.simulate()
            fivelink_simulator.plot_results()
            
            plt.show()
            
            # animate_trj(np.array(states))
            
            # ========================================
            #       Save the animated trajectory 
            # ========================================
            # Assuming x_trj is defined and draw_5link is a function that draws on the given axes
            fig, ax = plt.subplots()

            states_arr = np.array(states)
            n_time_steps = states_arr.shape[0]
            step = 5
            frames = range(0, n_time_steps, step)

            def update(frame):
                ax.clear()  # clear previous drawings
                q_i = states_arr[frame, :7]
                draw_5link(q_i, ax, legend=True)
                return ax,

            # Create the animation
            anim = animation.FuncAnimation(fig, update, frames=frames, interval=5, blit=False)

            # Save the animation to a file (e.g., as an MP4)
            # Note: You need ffmpeg installed to save as mp4; alternatively, you can use writer='imagemagick' for a GIF.
            animation_filename = 'rollout_animation.mp4'
            anim.save(animation_filename, writer='ffmpeg', fps=30)

            # Optionally display the animation window after saving
            plt.show()
        
        
        # compute reference extensions
        print("====== Computing the reference trajectory extensions ======")
        self._ref_ext_helper = compute_trejactory_extension(event_info, 
                                                            timespan,
                                                            states,
                                                            self._nx, self._nu,
                                                            self._init_state, 
                                                            self._target_state, 
                                                            self._detect_func)
        
        # Store the rollout as default values
        self._timespan = timespan
        self._dt = timespan[1:] - timespan[:-1]
        self._states = states
        self._inputs = inputs
        self._saltations = saltations
        self._modes = modes
        self._modechanges = mode_changes
        self._event_info = event_info    

        # ------------ Plot first rollout ------------ 
        show_rollout = False
        if show_rollout:
            plt.figure(figsize=(6, 6))
            plt.subplot(1, 1, 1)
            ut = []
            for ii in range(len(timespan)):
                ut.append(inputs[modes[ii]][:, 0])
            ut_arr = np.array(ut)
            plt.plot(timespan, ut_arr[:, 0], label=r'$u_1$')
            plt.plot(timespan, ut_arr[:, 1], label=r'$u_2$')
            plt.legend(loc="best", fontsize=10)
            plt.title('Control Input Torque')
            plt.xlabel('Time (sec)')
            plt.grid()
            plt.show()

        print("===================== Finished initial rollout =====================")
        
        # ----------------------------------------------------
        #               Compute the initial cost 
        # ----------------------------------------------------
        current_cost = self.compute_cost(modes,states,inputs,timespan)
        
        # =============
        #   Main Loop
        # =============
        
        learning_speed = 0.9 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.001 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-4 # Determines optimality
        armijo_threshold = 0.05 # Determines if current line search solve is good (this is typically labeled as "c")
        
        for i_iter in range(0,self._niters):
            
            current_cost = self.compute_cost(self._modes,self._states,self._inputs,self._timespan)
            
            print('========== Starting Iteration: ',i_iter,', Current cost: ',current_cost, ' ==========')
            
            print("-------- Backward Pass --------")
                
            # --------------------------------------------------------
            # Compute the backwards pass and update the control gains
            # --------------------------------------------------------
            (k_feedforward,K_feedback,
             k_ff_trj_ext,K_fb_trj_ext,
             expected_reduction,
             updated_event_info) = self.backward_pass(self._timespan, 
                                                      self._modes, 
                                                      self._states, 
                                                      self._inputs, 
                                                      self._saltations, 
                                                      self._event_info)    
            
            # Store updated variables
            self._k_ff = k_feedforward
            self._K_fb = K_feedback
            self._expt_cost_redu = expected_reduction
            self._K_fb_ext = K_fb_trj_ext
            self._k_ff_ext = k_ff_trj_ext
            self._event_info = updated_event_info
            
            self._reference_extension_helper = compute_trejactory_extension(self._event_info, 
                                                                            self._timespan, 
                                                                            self._states,
                                                                            self._nx, self._nu,
                                                                            self._init_state, 
                                                                            self._target_state, 
                                                                            self._detect_func)
            
            # compute_trejactory_extension(self._timespan, self._states, self._event_info)
            
            print('-------- Expected cost reduction: ',expected_reduction, ' --------')
            
            if(abs(expected_reduction)<low_expected_reduction):
                print(" -------- Stopping optimization, Optimal trajectory found --------")
                break
            learning_rate = 1
            armijo_flag = 0
            
            while (learning_rate>low_learning_rate and armijo_flag == 0):
                # ---------------------------------------------
                # Forward pass under the updated control gains
                # ---------------------------------------------
                (new_timespan,new_modes,new_states,new_inputs,
                new_saltations,new_mode_changes,new_event_info)=self.forward_pass(learning_rate=learning_rate)
                
                # ---------------------------------------------------------
                #   Compute new costs and check the optimality conditions
                # ---------------------------------------------------------
                new_cost = self.compute_cost(new_modes,new_states, new_inputs, new_timespan)
                
                print("***** new_cost: ", new_cost, " ***** ")
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self._expt_cost_redugrad_ + learning_rate*learning_rate*self._expt_cost_reduhess_

                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    print(" -------- Next iteration, armijo condition is met --------")
                    
                    show_forwardpass = True
                    if show_forwardpass:
                        
                        from five_link.fivelink_simulation import FiveLinkSimulator, animate_trj, draw_5link
                        u_trj_arr = self.desamble_control(new_modes, new_inputs)
                        dts = new_timespan[1:] - new_timespan[:-1]
                        fivelink_simulator = FiveLinkSimulator(self._init_state, u_trj_arr, dts)
                
                        fivelink_simulator.simulate()
                        fivelink_simulator.plot_results()
                        
                        plt.show()
                        
                        # animate_trj(np.array(new_states))
                        
                        # ========================================
                        #       Save the animated trajectory 
                        # ========================================
                        # Assuming x_trj is defined and draw_5link is a function that draws on the given axes
                        fig, ax = plt.subplots()

                        new_states_arr = np.array(new_states)
                        n_time_steps = new_states_arr.shape[0]
                        step = 5
                        frames = range(0, n_time_steps, step)

                        def update(frame):
                            ax.clear()  # clear previous drawings
                            q_i = new_states_arr[frame, :7]
                            draw_5link(q_i, ax, legend=True)
                            return ax,

                        # Create the animation
                        anim = animation.FuncAnimation(fig, update, frames=frames, interval=5, blit=False)

                        # Save the animation to a file (e.g., as an MP4)
                        # Note: You need ffmpeg installed to save as mp4; alternatively, you can use writer='imagemagick' for a GIF.
                        animation_filename = 'iteration_'+str(i_iter)+'.mp4'
                        anim.save(animation_filename, writer='ffmpeg', fps=30)

                        # Optionally display the animation window after saving
                        # plt.show()
                        
                    # ------------------------------------------------------
                    # Accept the new trajectory if armijo condition is met
                    # ------------------------------------------------------
                    current_cost = new_cost
                    self._timespan = new_timespan
                    self._states = new_states
                    self._inputs = new_inputs
                    self._saltations = new_saltations
                    self._modechanges = new_mode_changes
                    self._modes = new_modes
                    self._event_info = new_event_info
                    
                    self._reference_extension_helper = compute_trejactory_extension(new_event_info, 
                                                                                    new_timespan,
                                                                                    new_states, 
                                                                                    self._nx, self._nu,
                                                                                    self._init_state, 
                                                                                    self._target_state, 
                                                                                    self._detect_func)
                    
                    states_iter.append(new_states)
                else:
                    # Decrease learning rate and continue line search
                    learning_rate = learning_speed*learning_rate
                    
            if(learning_rate<low_learning_rate):
                print(" -------- Stopping optimization, low learning rate --------")
                break
        
            if (i_iter == self._niters-1):
                print(" -------- Stopping optimization, reached max iteration --------")
            
                
                # --------------------------------------------------------------
                #  Compute the new trajectory extensions and the gains for them
                # --------------------------------------------------------------
                print("------ Computing the new trajectory extensions ------")
                self._ref_ext_helper = compute_trejactory_extension(new_event_info, 
                                                                    new_timespan, 
                                                                    new_states,
                                                                    self._nx, self._nu,
                                                                    self._init_state, 
                                                                    self._target_state, 
                                                                    self._detect_func)
            
        # Return the current trajectory
        timespan = self._timespan
        modes = self._modes
        states = self._states
        inputs = self._inputs
        modechanges = self._modechanges
        event_info = self._event_info
        show_results = False
        if show_results:
            pass

        ref_ext_helper = compute_trejactory_extension(event_info,
                                                        timespan, 
                                                        states,
                                                        self._nx, self._nu,
                                                        self._init_state, 
                                                        self._target_state, 
                                                        self._detect_func)

        return (timespan,modes,states,inputs,
                k_feedforward,K_feedback,
                current_cost,states_iter,
                modechanges,ref_ext_helper)


if __name__ == '__main__':
    # ---------------- 3link walking example -----------------
    dt = 0.005
    epsilon = 2.0
    dt_shrink = 0.95
    
    start_time = 0
    end_time = 2.0
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)

    # generate initial state
    omega_1 = 1.55
    init_state = sigma_three_link(omega_1, a)
    init_state, _, _ = resetmap_3link_12(start_time, init_state)

    # Target is to go back to the initial state
    target_state = init_state  
    
    init_mode = 0

    # Set desired state
    n_modes = 1
    
    # the state and control dimensions, mode-dependent
    n_states = [6, 6]
    n_inputs = [2, 2]

    # ---------------------------- 
    #   Define weighting matrices
    # ----------------------------

    # Q_k = np.zeros((n_states[0],n_states[0]))
    # Q_k = [np.zeros((n_states[0],n_states[0])), np.zeros((n_states[1],n_states[1]))] # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # # ---------------------------- Set the terminal cost ----------------------------
    # target_mode = 0
    # Q_T = 200*np.eye(n_states[0])
    # Q_T[0,0] = 2000.0

    n_exp = 1
    n_samples = 10
    n_iters = 20
    is_detect = True

    # ====================================
    #    Solve for hybrid ilqr proposal
    # ====================================

    tout, xout, uout, t_events, x_events, saltations = solve_limcycle_3link()

    initial_guess = [uout, uout] 

    target_hip_velocity = 2.0
    h_ilqr_solver = hybrid_ilqr_jax(n_states, n_inputs,
                                    init_state, target_state, 
                                    initial_guess, dt, 
                                    start_time, end_time, 
                                    n_iters, is_detect, 
                                    onestep_detect_3link, dyn_control_3link_discrete_jax, 
                                    hip_moving_cost, target_hip_velocity,
                                    statedeviation_norm_cost, target_state)
    
    h_ilqr_results = h_ilqr_solver.solve()
