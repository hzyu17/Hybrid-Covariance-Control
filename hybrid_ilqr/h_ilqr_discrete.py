import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties
from dynamics.dynamics_discrete_slip import *


class hybrid_ilqr:
    def __init__(self, nmodes,init_mode,target_mode,nstates,
                 init_state,target_state,initial_guess,
                 dt,start_time,end_time,
                 contact_detect,smooth_dynamics,
                 Q_k,R_k,Q_T,parameters,n_iterations,
                 detect,plot_func,state_convert_func, 
                 init_reset_args, target_reset_args, 
                 animate_func=None, verbose=True):
        
        self.nmodes_ = nmodes
        self._n_states = nstates
        self._init_mode = init_mode
        self._target_mode = target_mode
        self._init_state = init_state
        self._target_state = target_state
        self._inputs = initial_guess
        self._n_inputs = [np.shape(initial_guess[i_mode])[1] for i_mode in range(nmodes)]
        self._plot_states_func = plot_func
        self._state_convert_func = state_convert_func
        self._animate_func = animate_func
        self._verbose = verbose
        
        # time definitions
        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self._timespan = np.arange(start_time, end_time, dt).flatten()
        self._n_timesteps = np.shape(self._timespan)[0]
        
        # feedback and feedforward
        self.k_feedforward_ = [np.zeros((self._n_inputs[0])) for _ in range(self._n_timesteps)]
        self.K_feedback_ = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._n_timesteps)]
        
        # hybrid events and mode changes
        self._saltations = [np.array([None]) for i in range(self._n_timesteps)]
        self._modechanges = [np.array([0, 0]) for _ in range(self._n_timesteps)]
        self._modes = [0 for _ in range(self._n_timesteps)]
        self._modes[0] = init_mode
        self._reset_args = init_reset_args
        
        self._target_reset_args = target_reset_args
        self._init_reset_args = init_reset_args
        
        # feedback and feedforward for the trajectory extensions 
        # shapes: feedback gains: [[K_fwd_extension, K_bwd_extension]_jump1, [K_fwd_extension, K_bwd_extension]_jump2, ...]
        # shapes: feedforwad gains: [[k_fwd_extension, k_bwd_extension]_jump1, [k_fwd_extension, k_bwd_extension]_jump2, ...]
        self.K_feedforward_extensions_ = [[None], [None]]
        self.K_feedback_extensions_ = [[None], [None]]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        # Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self._hybrid_event_info = {}
        self._reference_extension_helper = []
        
        # Dynamics
        self.smooth_dyn_ = smooth_dynamics
        self.detection_func_ = contact_detect
        
        self.f_, self.A_func_, self.B_func_ = [None for _ in range(self.nmodes_)],[None for _ in range(self.nmodes_)],[None for _ in range(self.nmodes_)]
        self.A_, self.B_ = [None for _ in range(self._n_timesteps)], [None for _ in range(self._n_timesteps)]
        for ii in range(self.nmodes_):
            self.f_[ii], self.A_func_[ii], self.B_func_[ii] = smooth_dynamics[ii]()
        
        # Weighting
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters

        # Max iterations
        self.n_iterations_ = n_iterations
        
        # Flag detection
        self.detect_ = detect

    def rollout(self):        

        (timespan,modes,states,inputs,
         saltations,mode_changes,
         hybrid_event_info,reset_args) = self.forward_pass(use_feedback=False,
                                                           learning_rate=1,
                                                           check_modemismatch=False)

        # Store the trajectory(states, inputs)
        self._timespan = timespan
        self._states = states
        self._inputs = inputs
        self._saltations = saltations
        self._modechanges = mode_changes
        self._modes = modes
        self._hybrid_event_info = hybrid_event_info
        self._reset_args = reset_args
        
        return timespan, modes, states, inputs, saltations, mode_changes

    def compute_cost(self,timespan,modes,states,inputs):
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self._n_timesteps-1):
            dt = timespan[ii+1] - timespan[ii]
            current_mode = modes[ii]
            current_u = inputs[current_mode][ii].flatten()
            current_x = states[ii].flatten()

            current_cost_u = 0.5*current_u.T@self.R_k_[current_mode]@current_u # Right now only considering cost in input
            
            # state_diff = current_x - self._target_state
            # current_cost_x = 0.5*state_diff.T@self.Q_k_[current_mode]@state_diff
            # current_cost = current_cost_u + current_cost_x
            
            current_cost = current_cost_u
            
            total_cost = total_cost+current_cost*dt
            
        # Compute terminal cost
        terminal_state = states[-1]
        if modes[-1] != self._target_mode:
            terminal_state = self._state_convert_func(states[-1]).flatten()
        terminal_difference = (self._target_state - terminal_state).flatten()
        terminal_cost = 0.5*terminal_difference.T@self.Q_T_@terminal_difference
        total_cost = total_cost+terminal_cost
        return total_cost
    
    
    def backward_pass(self):
        V_xx = self.Q_T_
        
        end_difference = (self._states[-1] - self._target_state).flatten()
        V_x = self.Q_T_@end_difference
        
        k_trj = [np.zeros((self._n_inputs[0])) for _ in range(self._n_timesteps)]
        K_trj = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._n_timesteps)]

        A_trj = [None for _ in range(self._n_timesteps)]
        B_trj = [None for _ in range(self._n_timesteps)]
        
        k_feedforward_trj_extension = []
        K_feedback_trj_extension = []
        
        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        current_mode = self._modes[-1]
        current_x = self._states[-1]
        current_u = self._inputs[current_mode][-1]
        # Get the jacobian of the discretized dynamics
        A_k = self.A_func_[current_mode](current_x, current_u, self.dt_)
        B_k = self.B_func_[current_mode](current_x, current_u, self.dt_)

        A_trj[-1] = A_k
        B_trj[-1] = B_k
    

        # for loop backwards in time
        for idx in reversed(range(0, self._n_timesteps-1)):
            # Grab the current variables in the trajectory
            current_mode = self._modes[idx]
            current_x = self._states[idx]
            current_u = self._inputs[current_mode][idx]
            saltation = self._saltations[idx]

            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self.Q_k_[current_mode] 
            l_uu = self.R_k_[current_mode]

            l_x = self.Q_k_[current_mode]@(current_x).flatten() 
            l_u = self.R_k_[current_mode]@(current_u).flatten()

            # Get the jacobian of the discretized dynamics
            A_k = self.A_func_[current_mode](current_x, current_u, self.dt_)
            B_k = self.B_func_[current_mode](current_x, current_u, self.dt_)

            A_trj[idx] = A_k
            B_trj[idx] = B_k
            
            if saltation is None:
            
                Q_x = l_x*self.dt_ + A_k.T@V_x
                Q_u = l_u*self.dt_+ B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*self.dt_ + B_k.T@V_xx@B_k
                Q_xx = l_xx*self.dt_ + A_k.T@V_xx@A_k
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._n_inputs[current_mode], self._n_states[current_mode]))
                
            else:
                Q_x = l_x*self.dt_ + A_k.T @ saltation.T @ V_x
                Q_u = l_u*self.dt_ + B_k.T @ saltation.T @ V_x
                Q_ux = B_k.T @ saltation.T @ V_xx @ saltation @ A_k
                Q_uu = l_uu*self.dt_ + B_k.T @ saltation.T @ V_xx @ saltation @ B_k
                Q_xx = l_xx*self.dt_ + A_k.T @ saltation.T @ V_xx @ saltation @ A_k    
                
                # Compute gains           
                k = -np.linalg.solve(Q_uu, Q_u)
                K = -np.linalg.solve(Q_uu, Q_ux).reshape((self._n_inputs[current_mode], self._n_states[current_mode]))

                # Compute the (gains for the forward extension, gains for the backward extension): use the gain at the immediate next state (reseted)
                k_feedforward_trj_extension.append((k, k_trj[idx+1]))
                K_feedback_trj_extension.append((K, K_trj[idx+1]))
                
                # update the hybrid dynamics information
                previous_hybrid_event_info = list(self._hybrid_event_info[idx])
                previous_hybrid_event_info[4] = [K, K_trj[idx+1]]
                previous_hybrid_event_info[5] = [k, k_trj[idx+1]]
                self._hybrid_event_info[idx] = tuple(previous_hybrid_event_info)

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

            # Update hessian and gradient for value function
            V_x = Q_x - K.T@Q_uu@k
            V_xx = Q_xx - K.T@Q_uu@K

        # Store expected cost reductions
        self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
        self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
        self.expected_cost_reduction_ = expected_cost_reduction

        # Store gain schedule
        self.k_feedforward_ = k_trj
        self.K_feedback_ = K_trj
        
        # Store the gain for the backward extensions
        K_feedback_trj_extension.reverse()
        k_feedforward_trj_extension.reverse()
        self.K_feedback_extensions_ = K_feedback_trj_extension
        self.K_feedforward_extensions_ = k_feedforward_trj_extension
        
        return (k_trj,K_trj,expected_cost_reduction, A_trj, B_trj)

    def forward_pass(self, use_feedback=True, 
                     learning_rate=1,
                     check_modemismatch=True):
        
        if self._verbose:
            if (not use_feedback):
                print("---------- Initial rollout ----------")
            else:
                print(f"---------- Forward pass. Learning rate: {learning_rate} ----------")
        
        # Lists to collect the current forward pass trajectories (Dimensions might vary so use list)
        timespan = self._timespan
        modes = [0 for _ in range(self._n_timesteps)]
        states = [np.array([0.0]) for _ in range(self._n_timesteps)]
        inputs = [np.zeros((self._n_timesteps, self._n_inputs[0])), np.zeros((self._n_timesteps, self._n_inputs[1]))]
        saltations = [None for i in range(self._n_timesteps)]
        mode_changess = np.tile(np.array([0, 0]), (self._n_timesteps, 1))
        reset_args = self._init_reset_args
        
        # Set the first state to be the initial       
        modes[0] = self._init_mode
        states[0] = self._init_state
        mode_changess[0] = np.array([self._init_mode, self._init_mode])
        
        # Extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        hybrid_event_info = {} # The dictionary that stores all the information of the jump dynamics and states.
        
        # ===================================================================
        #    Reference hybrid events and extensions from the last iteration
        # ===================================================================
        if use_feedback:
            
            (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref, 
             v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref, 
             v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref) = extract_extensions(self._reference_extension_helper)
            
            if self._verbose:
                print(f"Reference trajectory bouncing event numbers: {len(v_ext_trj_bwd_ref)}")
                for i_bounce in range(len(v_ext_trj_bwd_ref)):
                    print(f"bounce {i_bounce}: From mode {v_mode_change_ref[i_bounce][0]} to mode {v_mode_change_ref[i_bounce][1]} at time {v_tevents_ref[i_bounce]}")
                print("------------------------------------------------")
        
        # ---------------------------------
        #   Current rollout hybrid events 
        # ---------------------------------
        cnt_event = 0
        hybrid_index_ref = 0
        event_args = self._init_reset_args[0]
        
        # ===============================
        #         Main iterations
        # ===============================
        for ii in range(self._n_timesteps-1):
            
            # ---------------------- 
            #   Get the references 
            # ---------------------- 
            current_mode = modes[ii]
            current_state = states[ii]
            current_input = self._inputs[current_mode][ii]
            reset_args[ii] = event_args
            
            # =======================================
            #   If it is not the first time rollout
            # =======================================
            if use_feedback:
                
                ref_state = self._states[ii]
                current_mode_ref = self._modes[ii]
                
                # -----------------------------------------------
                #  Get the current (feedback, feedforward) gains 
                # -----------------------------------------------
                current_feedforward = learning_rate * self.k_feedforward_[ii]
                current_feedback = self.K_feedback_[ii]
                
                # ---------------
                #  Mode Mismatch
                # --------------- 
                if (current_mode != current_mode_ref) and (check_modemismatch):
                    
                    trj_extension = []
                    fb_ext_trj = []
                    ff_ext_trj = []
                    
                    hybrid_index_ref = np.argmin(abs(np.array(v_tevents_ref)-ii))
                    ref_modechange_hybrid = v_mode_change_ref[hybrid_index_ref]
                    
                    # ----------------------------------- Early Arrival ----------------------------------- 
                    if ((current_mode == ref_modechange_hybrid[1]) and (current_mode_ref==ref_modechange_hybrid[0])):
                        if self._verbose:
                            print(f"early arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                            print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_trj_bwd_ref[hybrid_index_ref]
                        ff_ext_trj = v_kff_ext_trj_bwd_ref[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_trj_bwd_ref[hybrid_index_ref]
                        
                    # ----------------------------------- Late Arrival ----------------------------------- 
                    else:
                        if self._verbose:
                            print(f"late arrival, Time: {ii}. Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                            print(f"Reference mode change from mode {ref_modechange_hybrid[0]} to mode {ref_modechange_hybrid[1]} at time {v_tevents_ref[hybrid_index_ref]}")
                        
                        trj_extension = v_ext_trj_fwd_ref[hybrid_index_ref]
                        ff_ext_trj = v_kff_ext_trj_fwd_ref[hybrid_index_ref]
                        fb_ext_trj = v_Kfb_ext_trj_fwd_ref[hybrid_index_ref]
                    
                    # Modify the reference to the extension
                    ref_state = trj_extension[ii]
                    current_feedback = fb_ext_trj[ii]
                    current_feedforward = learning_rate * ff_ext_trj[ii]
                    
                    if self._verbose:
                        print("current_nominal_input: ", current_input)
                
                # ---------------------------
                #   // End Mode Mismatch //
                # --------------------------- 
                current_input = current_input + current_feedback@(current_state-ref_state) + current_feedforward
            
            # ====================
            #   Simulate forward
            # ====================
            t_ii = self._timespan[ii]
            dt = self._timespan[ii+1] - t_ii
            
            (next_state, saltation, mode_change, 
             t_event, x_event, x_reset, reset_byproduct) = self.detection_func_(current_mode, 
                                                                                current_state, 
                                                                                current_input, 
                                                                                t_ii, dt, 
                                                                                reset_args[ii], self.detect_)

            # -------------------------------
            #  Update the hybrid information
            # -------------------------------
            if saltation is not None:
                hybrid_index.add(ii)
                saltations[ii] = saltation
                hybrid_event_info[ii] = (t_event, x_event, x_reset, 
                                          mode_change, 
                                          self.K_feedback_extensions_[hybrid_index_ref], 
                                          self.K_feedforward_extensions_[hybrid_index_ref])
                timespan[ii+1] = t_event
            
            # Only consider the transition from mode 0 to mode 1 for now
            if (mode_change[0]!=mode_change[1]):
                if self._verbose:
                    print(f" ===================== At Time {ii}, the system has a mode change from mode {mode_change[0]} to mode {mode_change[1]} =====================")
                    print("-------- reset_byproduct --------")
                    print(reset_byproduct)
                event_args = reset_byproduct
                cnt_event += 1
            
            # ---------------------
            # Move forward in time
            # ---------------------
            states[ii+1] = next_state
            inputs[current_mode][ii] = current_input.flatten()
            mode_changess[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
        
        if self._verbose:
            print(f"--------------------- Total number of contacts: {cnt_event} ---------------------" )
        
        return (timespan,modes,states,inputs,saltations,mode_changess,hybrid_event_info,reset_args)
        
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        
        # ------------------------------------
        #  First rollout using initial guess
        # ------------------------------------
        [timespan,modes,states,inputs,saltations,modechanges] = self.rollout()
        
        print("===================== Finished initial rollout =====================")
        
        show_rollout = True
        r0 = 1
        if show_rollout:
            self._plot_states_func(self._timespan, modes, states, inputs, 
                                    self._init_state, self._target_state, 
                                    self._n_timesteps, reset_args=self._reset_args, step=50)
            
            if self._animate_func:
                fig, ax = self._animate_func(self._modes, self._states, self._init_mode, 
                                            self._init_state, self._target_mode, self._target_state, 
                                            self._n_timesteps, self._reset_args, self._target_reset_args, step=200)
                
                # Add grid
                ax.grid(True, linestyle='--', alpha=0.7)
                
                # Create legend handles
                legend_handles = [
                    Line2D([0], [0], color='black', linewidth=2, label='Flight trajectory'),
                    Line2D([0], [0], color='blue', linewidth=2, label='Stance trajectory'),
                    Line2D([0], [0], color='red', linewidth=2, label='Initial state'),
                    Line2D([0], [0], color='green', linewidth=2, label='Target state'),
                ]
                ax.legend(handles=legend_handles, loc='upper left', 
                         prop={'family': 'serif', 'size': 11}, framealpha=0.9)
                
                ax.set_xlabel(r'$p_x$ (m)', fontsize=12, fontfamily='serif')
                ax.set_ylabel(r'$p_z$ (m)', fontsize=12, fontfamily='serif')
                ax.set_title('Initial Rollout', fontsize=14, fontfamily='serif')
            
                fig.tight_layout()
                ax.set_xlim(-0.4, 1.25)
                ax.set_ylim(-0.2, 1.65)
                fig.savefig(root_dir+'/figures/slip/slip_jump_setting.pdf', dpi=2000)
                plt.show()
            
        # ----------------------------------------------------
        #  Compute the current cost of the initial trajectory
        # ----------------------------------------------------
        current_cost = self.compute_cost(timespan,modes,states,inputs)
        learning_speed = 0.95
        low_learning_rate = 0.001
        low_expected_reduction = 1e-4
        armijo_threshold = 0.05
        
        # =================================================================
        #                             Main Loop
        # =================================================================
        for ii in range(0,self.n_iterations_):   
            
            current_cost = self.compute_cost(self._timespan, self._modes, self._states, self._inputs)
            
            print('========== Starting Iteration: ',ii,', Current cost: ',current_cost, ' ==========')            
            print("-------- Backward Pass --------")
            
            # --------------------------------------------------------
            # Compute the backwards pass and update the control gains
            # --------------------------------------------------------
            (k_feedforward, K_feedback, expected_reduction, A_trj, B_trj) = self.backward_pass()    
            
            # --------------------------------------------------------------
            #  Compute the new trajectory extensions and the gains for them
            # --------------------------------------------------------------
            self._reference_extension_helper = self.compute_trejactory_extension(self._timespan, self._states, self._hybrid_event_info)
            
            print('-------- Expected cost reduction: ',expected_reduction, ' --------')
            
            if(abs(expected_reduction)<low_expected_reduction):
                print(" -------- Stopping optimization, Optimal trajectory found --------")
                break
            
            learning_rate = 1
            armijo_flag = 0
            
            # ----------------------------------------------
            #  Forward pass under the updated control gains
            # ----------------------------------------------
            while (learning_rate>low_learning_rate and armijo_flag == 0):
                
                # Forward pass: line search 
                (new_timespan, new_modes,new_states,new_inputs,
                 new_saltations,new_mode_changes,
                 new_hybrid_event_info,new_reset_args)=self.forward_pass(learning_rate)
                
                # --------------------------------- Plot forward pass ---------------------------------
                show_fwdpass = False
                if show_fwdpass:
                    self._plot_states_func(self._timespan, modes, states, inputs, 
                                            self._init_state, self._target_state, 
                                            self._n_timesteps, reset_args=self._reset_args, step=200)
                    
                    if self._animate_func:
                        fig, ax = self._animate_func(self._modes, self._states, self._init_mode, 
                                                    self._init_state, self._target_mode, self._target_state, 
                                                    self._n_timesteps, self._reset_args, self._target_reset_args, step=200)
                        
                        # Add grid
                        ax.grid(True, linestyle='--', alpha=0.7)
                        
                        # Create legend handles
                        legend_handles = [
                            Line2D([0], [0], color='black', linewidth=2, label='Flight trajectory'),
                            Line2D([0], [0], color='blue', linewidth=2, label='Stance trajectory'),
                            Line2D([0], [0], color='red', linewidth=2, label='Initial state'),
                            Line2D([0], [0], color='green', linewidth=2, label='Target state'),
                        ]
                        ax.legend(handles=legend_handles, loc='upper left', 
                                 prop={'family': 'serif', 'size': 11}, framealpha=0.9)
                        
                        ax.set_xlabel(r'$p_x$ (m)', fontsize=12, fontfamily='serif')
                        ax.set_ylabel(r'$p_z$ (m)', fontsize=12, fontfamily='serif')
                        ax.set_title(f'Forward Pass (lr={learning_rate:.3f})', fontsize=14, fontfamily='serif')
                    
                        fig.tight_layout()
                        ax.set_xlim(-0.4, 1.25)
                        ax.set_ylim(-0.2, 1.65)
                        plt.show()
                # ------------------------------- // Plot forward pass // -------------------------------

                new_cost = self.compute_cost(new_timespan, new_modes, new_states, new_inputs)

                print("new_cost: ", new_cost)
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self.expected_cost_reduction_grad_ + learning_rate*learning_rate*self.expected_cost_reduction_hess_

                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    print(" -------- Next iteration, armijo condition is met --------")
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
                    self._hybrid_event_info = new_hybrid_event_info
                    self._reference_extension_helper = self.compute_trejactory_extension(new_timespan, new_states, new_hybrid_event_info)
                    self._reset_args = new_reset_args
                    states_iter.append(new_states)
                else:
                    # Decrease learning rate and continue line search
                    learning_rate = learning_speed*learning_rate
                    
            if(learning_rate<low_learning_rate):
                print(" -------- Stopping optimization, low learning rate --------")
                break
          
            if (ii == self.n_iterations_-1):
                print(" -------- Stopping optimization, reached max iteration --------")
        
        timespan = self._timespan
        modes = self._modes
        states = self._states
        inputs = self._inputs
        saltations = self._saltations
        modechanges = self._modechanges
        hybrid_event_info = self._hybrid_event_info
        reset_args = self._reset_args
        
        ref_ext_helper = self.compute_trejactory_extension(timespan, states, hybrid_event_info)

        return (timespan,modes,states,inputs,saltations,
                k_feedforward,K_feedback, A_trj,B_trj,
                current_cost,states_iter,
                modechanges,ref_ext_helper,reset_args)
        
    
    def compute_trejactory_extension(self, timespan, states, hybrid_event_info):
        # NO hybrid events
        if len(hybrid_event_info.keys()) == 0:
            return []
        
        sorted_hybrid_index = sorted(hybrid_event_info.keys())
        ref_ext_helper = []
        
        i_events = []
        t_events = []
        x_events = []
        x_resets = []
        K_feedback_fwd_extensions = []
        k_feedforward_fwd_extensions = []
        K_feedback_bwd_extensions = []
        k_feedforward_bwd_extensions = []

        mode_changes = []
        
        i_events.append(0)
        t_events.append(self.start_time_)
        x_events.append(self._init_state)
        x_resets.append(self._init_state)
        K_feedback_fwd_extensions.append(np.zeros(1))
        K_feedback_bwd_extensions.append(np.zeros(1))
        k_feedforward_fwd_extensions.append(np.zeros(1))
        k_feedforward_bwd_extensions.append(np.zeros(1))
        mode_changes.append(np.array([0, 0]))
        
        for i_key in sorted_hybrid_index:
            i_events.append(i_key)
            t_events.append(hybrid_event_info[i_key][0])
            x_events.append(hybrid_event_info[i_key][1])
            x_resets.append(hybrid_event_info[i_key][2])
            mode_changes.append(hybrid_event_info[i_key][3])
            
            K_feedback_fwd_extensions.append(hybrid_event_info[i_key][4][0])
            K_feedback_bwd_extensions.append(hybrid_event_info[i_key][4][1])
            
            k_feedforward_fwd_extensions.append(hybrid_event_info[i_key][5][0])
            k_feedforward_bwd_extensions.append(hybrid_event_info[i_key][5][1])
        
        i_events.append(self._n_timesteps)
        t_events.append(self.end_time_)
        x_events.append(self._target_state)
        x_resets.append(self._target_state)
        K_feedback_fwd_extensions.append(np.zeros(1))
        K_feedback_bwd_extensions.append(np.zeros(1))
        k_feedforward_fwd_extensions.append(np.zeros(1))
        k_feedforward_bwd_extensions.append(np.zeros(1))
        
        if hybrid_event_info.keys():
            mode_changes.append(hybrid_event_info[sorted_hybrid_index[-1]][3])
        
        # Forward and backward trajectory extensions
        for ii, _ in enumerate(t_events[1:-1], start=1):
            if ii < len(t_events[1:-1]):
                i_event_next = i_events[ii+1]
            else:
                i_event_next = len(timespan)
            if ii > 1:
                i_event_prev = i_events[ii-1]
            else:
                i_event_prev = 0
                
            i_event = i_events[ii]
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            current_mode_i = mode_changes[ii][0]
            next_mode_i = mode_changes[ii][1]
            
            K_feedback_fwd_extension_i = K_feedback_fwd_extensions[ii]
            k_feedforward_fwd_extension_i = k_feedforward_fwd_extensions[ii]
            
            K_feedback_bwd_extension_i = K_feedback_bwd_extensions[ii]
            k_feedforward_bwd_extension_i = k_feedforward_bwd_extensions[ii]
            
            t_ext_fwd_i = self.end_time_
            
            # ============================== forward trajectory extension ==============================
            timespan_ext_fwd = timespan[i_event+1:]
            nt_ext_fwd = len(timespan_ext_fwd)
            xtrj_ext_fwd_i = np.zeros((nt_ext_fwd, self._n_states[current_mode_i]))
            xtrj_ext_fwd_i[0] = np.asarray(x_event_i)
            
            K_feedback_ext_fwd_i = np.tile(K_feedback_fwd_extension_i, (self._n_timesteps, 1, 1))
            k_feedforward_ext_fwd_i = np.tile(k_feedforward_fwd_extension_i, (self._n_timesteps, 1))
            
            for jj in range(nt_ext_fwd-1):
                current_state = xtrj_ext_fwd_i[jj]
                
                t_jj = timespan_ext_fwd[jj]
                dt = timespan_ext_fwd[jj+1] - t_jj
                
                current_input = np.zeros(self._n_inputs[current_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(current_mode_i, current_state, 
                                                                    current_input, t_jj, dt, 
                                                                    self._reset_args[jj], detection=False)
             
                xtrj_ext_fwd_i[jj+1] = next_state
            
            time_span_ext_fwd_padding = timespan[:i_event+1]
            nt_ext_padding_fwd = len(time_span_ext_fwd_padding)
            xtrj_ext_padding_fwd_i = np.zeros((nt_ext_padding_fwd, self._n_states[current_mode_i]))
            xtrj_ext_padding_fwd_i[i_event_prev+1:i_event+1] = np.asarray(states[i_event_prev+1:i_event+1])
            xtrj_ext_fwd_i = np.vstack((xtrj_ext_padding_fwd_i, xtrj_ext_fwd_i))
            
            # ============================== backward trajectory extension ==============================
            timespan_ext_bwd = timespan[:i_event+1][::-1]
            
            nt_ext_bwd = len(timespan_ext_bwd)
            xtrj_ext_bwd_i = np.zeros((nt_ext_bwd, self._n_states[next_mode_i]))
            xtrj_ext_bwd_i[0] = x_reset_i
            
            K_feedback_ext_bwd_i = np.tile(K_feedback_bwd_extension_i, (self._n_timesteps, 1, 1))
            k_feedforward_ext_bwd_i = np.tile(k_feedforward_bwd_extension_i, (self._n_timesteps, 1))
            
            for jj in range(nt_ext_bwd-1):
                current_state = xtrj_ext_bwd_i[jj]
                t_jj = timespan_ext_bwd[jj]
                dt = t_jj - timespan_ext_bwd[jj+1]
                
                current_input = np.zeros(self._n_inputs[next_mode_i])
                
                next_state, _, _, _, _, _, _ = self.detection_func_(next_mode_i,current_state, 
                                                                    current_input, t_jj, dt, 
                                                                    self._reset_args[jj], 
                                                                    detect=False, 
                                                                    backwards=True)
                
                xtrj_ext_bwd_i[jj+1] = next_state
            
            xtrj_ext_bwd_i = xtrj_ext_bwd_i[::-1]
            
            time_span_ext_bwd_padding = timespan[i_event+1:]
            nt_ext_padding_bwd = len(time_span_ext_bwd_padding)
            xtrj_ext_padding_bwd_i = np.zeros((nt_ext_padding_bwd, self._n_states[next_mode_i]))   
            xtrj_ext_padding_bwd_i[:i_event_next-i_event] = np.asarray(states[i_event+1:i_event_next+1])
            xtrj_ext_bwd_i = np.vstack((xtrj_ext_bwd_i, xtrj_ext_padding_bwd_i))
            
            ref_ext_helper.append({"Mode Change": np.array([current_mode_i, next_mode_i]), 
                                    "Trajectory Extensions": {current_mode_i:xtrj_ext_fwd_i, next_mode_i:xtrj_ext_bwd_i}, 
                                    "Feedback gains": {current_mode_i:K_feedback_ext_fwd_i, next_mode_i:K_feedback_ext_bwd_i}, 
                                    "Feedforward gains": {current_mode_i:k_feedforward_ext_fwd_i, next_mode_i:k_feedforward_ext_bwd_i}, 
                                    "event index":  i_event
                                    })
        
        return ref_ext_helper


def solve_ilqr(params, detect=True, verbose=True):
    # Dynamics functions
    smooth_dynamis = params.symbolic_dynamics()
    detect_integration = params.detection_func()

    # Tool functions
    plotting_function = params.plotting_function()    
    state_convert_function = params.state_convert_function()
    animate_function = params.animate_function()

    # Initialize timings
    dt = params._dt
    
    start_time = params._start_time
    end_time = params._end_time

    # Set desired state
    init_state = params._init_state
    target_state = params._target_state

    # Initial guess
    initial_guess = params._initial_guess
    
    # Reset arguments
    init_reset_args = params._init_reset_args
    target_reset_args = params._target_reset_args

    # Define weighting matrices
    Q_k = params._Q_k
    R_k = params._R_k

    # Set the terminal cost
    Q_T = params._Q_T

    # Set the physical parameters
    mass = 1
    gravity = 9.8
    parameters = np.array([mass,gravity])

    # Specify max number of iterations
    n_iterations = 50
    
    init_mode = params.current_mode()
    target_mode = params._target_mode
    
    nmodes = params.nmodes()
    nstates = params._nstates

    ilqr_ = hybrid_ilqr(nmodes, init_mode, target_mode, 
                        nstates, init_state, target_state, initial_guess,
                        dt, start_time, end_time,
                        detect_integration, smooth_dynamis,
                        Q_k, R_k, Q_T, parameters, n_iterations,
                        detect, plotting_function, state_convert_function, 
                        init_reset_args, target_reset_args, 
                        animate_function, verbose)
    
    return ilqr_.solve()
