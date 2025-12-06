import numpy as np
import matplotlib.pyplot as plt

class hybrid_ilqr:
    def __init__(self,init_mode,init_state,target_state,initial_guess,
                 dt,start_time,end_time,contact_detect,smooth_dynamics,Q_k,R_k,Q_T,parameters,n_iterations,detect):
        self.current_mode = init_mode
        self.init_state_ = init_state
        self.target_state_ = target_state
        self.inputs_ = initial_guess
        self.n_states_ = np.shape(init_state)[0]
        self.n_inputs_ = np.shape(initial_guess)[1]

        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self.time_span_ = np.arange(start_time, end_time, dt).flatten()
        self.n_timesteps_ = np.shape(self.time_span_)[0]
        self.saltations_ = [None for i in range(self.n_timesteps_)]
        self.modechanges_ = [np.array([0, 0]) for _ in range(self.n_timesteps_)]
        
        # ------------------------------------------------------------------------------------------------------------------ 
        # Map that maps the index at hybrid event to the event informations (t_event, x_event, x_reset, mode_change). 
        # ------------------------------------------------------------------------------------------------------------------ 
        self.txmode_map_ = {}
        
        # Dynamics
        self.smooth_dyn_ = smooth_dynamics
        self.detection_func_ = contact_detect
        
        self.f_, self.A_, self.B_ = smooth_dynamics[init_mode]()
        
        # Weighting
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters

        # Max iterations
        self.n_iterations_ = n_iterations
        
        # flag detection
        self.detect_ = detect

    def rollout(self):
        # states = np.zeros((self.n_timesteps_, self.n_states_))
        # inputs = np.zeros((self.n_timesteps_, self.n_inputs_))
        states = [0.0 for _ in range(self.n_timesteps_)]
        inputs = [0.0 for _ in range(self.n_timesteps_)]
        modes = np.zeros(self.n_timesteps_, dtype=np.int64)
        
        mode_changes = [np.array([0, 0]) for _ in range(self.n_timesteps_)]
        
        saltations = [None for i in range(self.n_timesteps_)]
        
        current_state = self.init_state_
        current_mode = 0
        states[0] = current_state
        mode_changes[0] = np.array([current_mode, current_mode])
        modes[0] = current_mode
        
        for ii in range(self.n_timesteps_-1):
            current_input = self.inputs_[ii,:]
            t_ii = self.time_span_[ii]
            
            next_state, saltation, mode_change, _, _, _ = self.detection_func_(current_state, current_input, t_ii, t_ii+self.dt_, current_mode, self.detect_)
            saltations[ii] = saltation
            next_state = next_state.flatten()
            
            # Store states and inputs
            states[ii+1] = next_state
            inputs[ii] = current_input # in case we have a control law, we store the input used
            mode_changes[ii+1] = mode_change
            modes[ii+1] = mode_change[1]
            
            current_state = next_state
            current_mode = mode_change[1]

        # Store the trajectory(states, inputs)
        self.states_ = states
        self.inputs_ = inputs
        self.saltations_ = saltations
        self.modechages_ = mode_changes
        self.modes_ = modes
        return states, inputs, saltations, mode_changes

    def compute_cost(self,states,inputs,dt):
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self.n_timesteps_-1):
            current_x = states[ii] # Not being used currently
            current_u = inputs[ii].flatten()

            current_cost = 0.5*current_u.T@self.R_k_@current_u # Right now only considering cost in input
            total_cost = total_cost+current_cost*dt
        
        # Compute terminal cost
        terminal_difference = (self.target_state_ - states[-1]).flatten()
        terminal_cost = 0.5*terminal_difference.T@self.Q_T_@terminal_difference
        total_cost = total_cost+terminal_cost
        return total_cost

    def backwards_pass(self):
        # First compute initial conditions (end boundary condition)
        # Value function hessian and gradient
        V_xx = self.Q_T_
        
        end_difference = (self.states_[-1] - self.target_state_).flatten()
        V_x = self.Q_T_@end_difference

        # Initialize storage variables
        k_trj = np.zeros((self.n_timesteps_,self.n_inputs_))
        K_trj = np.zeros((self.n_timesteps_,self.n_inputs_,self.n_states_))

        # Initialize cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0

        # for loop backwards in time
        for idx in reversed(range(0, self.n_timesteps_-1)):
            # Grab the current variables in the trajectory
            current_x = self.states_[idx]
            current_u = self.inputs_[idx]
            saltation = self.saltations_[idx]
            mode = self.modes_[idx]

            # R_k_updated
            # Define the expansion coefficients and the loss gradients
            l_xx = self.Q_k_ # For now zeros, can add in a target to track later on
            l_uu = self.R_k_

            l_x = self.Q_k_@np.zeros(self.n_states_).flatten() # For now zeros, can add in a target to track later on
            l_u = self.R_k_@(current_u).flatten()

            # Get the jacobian of the discretized dynamics
            self.f_, self.A_, self.B_ = self.smooth_dyn_[mode]()
            
            A_k = self.A_(current_x, current_u, self.dt_)
            B_k = self.B_(current_x, current_u, self.dt_)
            
            if saltation is None:
                Q_x = l_x*self.dt_ + A_k.T@V_x
                Q_u = l_u*self.dt_+ B_k.T@V_x
                Q_ux = B_k.T@V_xx@A_k
                Q_uu = l_uu*self.dt_ + B_k.T@V_xx@B_k
                Q_xx = l_xx*self.dt_ + A_k.T@V_xx@A_k

            else:
                print("Found contact dynamics!")
                Q_x = l_x*self.dt_ + A_k.T @ saltation.T @ V_x
                Q_u = l_u*self.dt_ + B_k.T @ saltation.T @ V_x
                Q_ux = B_k.T @ saltation.T @ V_xx @ saltation @ A_k
                Q_uu = l_uu*self.dt_ + B_k.T @ saltation.T @ V_xx @ saltation @ B_k
                Q_xx = l_xx*self.dt_ + A_k.T @ saltation.T @ V_xx @ saltation @ A_k                
            
            # Compute gains           
            k = -np.linalg.solve(Q_uu, Q_u)
            K = -np.linalg.solve(Q_uu, Q_ux)

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
        self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
        self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
        self.expected_cost_reduction_ = expected_cost_reduction

        # Store gain schedule
        self.k_feedforward_ = k_trj
        self.K_feedback_ = K_trj
        return (k_trj,K_trj,expected_cost_reduction)

    def forward_pass(self, learning_rate):
        states = np.zeros((self.n_timesteps_, self.n_states_))
        inputs = np.zeros((self.n_timesteps_, self.n_inputs_))
        mode_changess = np.tile(np.array([0, 0]), (self.n_timesteps_, 1))
        
        current_state = self.init_state_
        current_mode = 1

        # set the first state to be the initial
        states[0] = current_state
        mode_changess[0] = np.array([current_mode, current_mode])
        saltations = [None for i in range(self.n_timesteps_)]
        
        # extend reference trj, if a hybrid event is hit.
        hybrid_index = set()
        txmode_map = {}
        
        for ii in range(self.n_timesteps_-1):
            # Get the current gains and compute the feedforward and feedback terms
            current_feedforward = learning_rate * self.k_feedforward_[ii]
            current_feedback = self.K_feedback_[ii]@(current_state-self.states_[ii])
            current_input = self.inputs_[ii] + current_feedback + current_feedforward
            
            # simulate forward
            t_ii = self.time_span_[ii]
            
            next_state, saltation, mode_chage, t_event, x_event, x_reset = self.detection_func_(current_state, current_input, t_ii, t_ii+self.dt_, current_mode, self.detect_)
            saltations[ii] = saltation
            
            if saltation is not None:
                # Collect the tevent and xevent for the hybrid events
                hybrid_index.add(ii)
                txmode_map[ii] = (t_event, x_event, x_reset, mode_chage)
            
            # Store states and inputs
            states[ii+1] = next_state
            inputs[ii] = current_input.flatten()
            mode_changess[ii+1] = mode_chage

            # Update the current state
            current_state = next_state
            current_mode = mode_chage[1]
            
        return (states,inputs,current_feedback,current_feedforward,saltations,mode_changess,txmode_map)
    
    def compute_trejactory_extension(self, txmode_map):
        # txmode_map:
        sorted_hybrid_index = sorted(txmode_map.keys())
        
        i_events = []
        t_events = []
        x_events = []
        x_resets = []
        mode_changes = []
        
        mode_exttrjs_maps = []
        
        i_events.append(0)
        t_events.append(self.start_time_)
        x_events.append(self.init_state_)
        x_resets.append(self.init_state_)
        mode_changes.append(np.array([1, 1]))
        
        for i_key in sorted_hybrid_index:
            i_events.append(i_key)
            t_events.append(txmode_map[i_key][0])
            x_events.append(txmode_map[i_key][1])
            x_resets.append(txmode_map[i_key][2])
            mode_changes.append(txmode_map[i_key][3])
        
        i_events.append(self.n_timesteps_)
        t_events.append(self.end_time_)
        x_events.append(self.target_state_)
        x_resets.append(self.target_state_)
        if txmode_map.keys():
            mode_changes.append(txmode_map[sorted_hybrid_index[-1]][3])
        
        # forward and backward trajectory extensions
        for ii, tevent_i in enumerate(t_events[1:-1], start=1):
            i_event_i = i_events[ii]
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            current_mode_i = mode_changes[ii][0]
            next_mode_i = mode_changes[ii][1]
            
            # --------------------------------------
            # Choose a time span for the extensions
            # --------------------------------------
            t_ext_fwd_i = self.end_time_
            # t_ext_fwd_i = t_events[ii+1]
            # t_ext_bwd_i = tevent_i - (tevent_i-t_events[ii-1])
            
            # [0, t_event] for padding
            time_span_ext_fwd_padding = np.arange(0, tevent_i, self.dt_)
            # [t_event, t_trj_ext_fwd]
            timespan_ext_fwd = np.arange(tevent_i, t_ext_fwd_i, self.dt_)
            # [t_trj_ext_bwd: t_event]
            timespan_ext_bwd = np.arange(0, tevent_i, self.dt_)[::-1]
            
            # time span lengths
            nt_ext_fwd = len(timespan_ext_fwd)
            nt_ext_bwd = len(timespan_ext_bwd)
            nt_ext_padding_fwd = len(time_span_ext_fwd_padding)
            nt_ext_padding_bwd = nt_ext_fwd
            
            xtrj_ext_padding_fwd_i = np.zeros((nt_ext_padding_fwd, self.n_states_))
            xtrj_ext_fwd_i = np.zeros((nt_ext_fwd, self.n_states_))
            xtrj_ext_bwd_i = np.zeros((nt_ext_bwd, self.n_states_))
            # xtrj_ext_padding_bwd_i = np.zeros((nt_ext_padding_bwd, self.n_states_))
            xtrj_ext_padding_bwd_i = self.states_[i_event_i:]
            xtrj_ext_fwd_i[0] = x_event_i
            
            # ----------------------------------
            # simulate the forward extension
            # ----------------------------------
            current_state = x_event_i
            for jj in range(nt_ext_fwd-1):
                t_jj = timespan_ext_fwd[jj]
                
                # Using zero control, modify if needed.
                current_input = np.zeros(self.n_inputs_)
                
                next_state, _, _, _, _, _ = self.detection_func_(current_state, current_input, t_jj, t_jj+self.dt_, current_mode=current_mode_i, detection=False)
                
                # Store states and inputs
                xtrj_ext_fwd_i[jj+1] = next_state

                # Update the current state
                current_state = next_state
            
            xtrj_ext_fwd_i = np.vstack((xtrj_ext_padding_fwd_i, xtrj_ext_fwd_i))
            
            # ----------------------------------
            # simulate the backward extension
            # ----------------------------------
            xtrj_ext_bwd_i[0] = x_reset_i
            current_state = x_reset_i
            for jj in range(nt_ext_bwd-1):
                t_jj = timespan_ext_bwd[jj]
                
                # modify if needed
                current_input = np.zeros(self.n_inputs_)
                
                next_state, _, _, _, _, _ = self.detection_func_(current_state, current_input, t_jj, t_jj-self.dt_, current_mode=next_mode_i, detection=False)
                
                # Store states and inputs
                xtrj_ext_bwd_i[jj+1] = next_state

                # Update the current state
                current_state = next_state            
            
            # -------------------------------
            # reverse the backward extension 
            # -------------------------------   
            xtrj_ext_bwd_i = xtrj_ext_bwd_i[::-1]
            
            # --------------------- padding ---------------------
            xtrj_ext_bwd_i = np.vstack((xtrj_ext_bwd_i, xtrj_ext_padding_bwd_i))
            
            # ------------------------ collect the trajectory extensions ------------------------
            mode_exttrjs_maps.append((np.array([current_mode_i, next_mode_i]), {current_mode_i:xtrj_ext_fwd_i, next_mode_i:xtrj_ext_bwd_i}))
            
        return mode_exttrjs_maps
    
    
    def solve(self):
        # ------ collect the iteration data ------
        states_iter = []
        # Compute the rollout to get the initial trajectory with the
        # initial guess
        [states,inputs,saltations,modechanges] = self.rollout()
        
        # Compute the current cost of the initial trajectory
        current_cost = self.compute_cost(states,inputs,self.dt_)
        
        learning_speed = 0.9 # This can be modified, 0.95 is very slow
        low_learning_rate = 0.01 # if learning rate drops to this value stop the optimization
        low_expected_reduction = 1e-4 # Determines optimality
        armijo_threshold = 0.1 # Determines if current line search solve is good (this is typically labeled as "c")
        for ii in range(0,self.n_iterations_):
            print('Starting iteration: ',ii,', Current cost: ',current_cost)
            # Compute the backwards pass
            (k_feedforward,K_feedback,expected_reduction) = self.backwards_pass()    
            
            print('Expected cost reduction: ',expected_reduction)
            
            if(abs(expected_reduction)<low_expected_reduction):
                # If the expected reduction is low, then end the
                # optimization
                print("Stopping optimization, optimal trajectory")
                break
            learning_rate = 1
            armijo_flag = 0
            
            (new_states,new_inputs,new_feedback,new_feedforward,new_saltations,mode_changes,new_txmode_map)=self.forward_pass(learning_rate)
            new_cost = self.compute_cost(new_states, new_inputs, self.dt_)
            
            
            show_results = False
            if show_results:
                print("plotting results")
                # =============== plotting ===============
                fig1, axes = plt.subplots(1, 2)
                (ax1, ax2) = axes.flatten()
                ax1.grid(True)
                ax2.grid(True)

                start_time = 0
                end_time = 2.0
                time_span = np.arange(start_time, end_time, self.dt_).flatten()
                
                init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
                target_state = np.array([3.5, 0])  # Swing pendulum upright
        
                # ----------- Plot the start and goal states -----------
                ax1.scatter(time_span[-1], target_state[0], color='g', marker='x', s=50.0, linewidths=6, label='Target')
                ax1.scatter(time_span[0], init_state[0], color='r', marker='x', s=50.0, linewidths=6, label='Start')

                ax2.scatter(time_span[-1], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
                ax2.scatter(time_span[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

                # ----------- Plot the reference -----------
                for i in range(self.n_timesteps_):
                    ax1.scatter(time_span[i], self.states_[i][0], color='k')
                    ax2.scatter(time_span[i], self.states_[i][1], color='k')

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
                for i in range(self.n_timesteps_):
                    ax5.scatter(self.states_[i][0], self.states_[i][1], color='k')

                # ----------- Plot the start and goal states -----------
                ax5.scatter(target_state[0], target_state[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
                ax5.scatter(init_state[0], init_state[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

                ax5.legend()
                
                plt.show()

            # print("new_cost: ", new_cost) 
            
            # Execute linesearch until the armijo condition is met (for
            # now just check if the cost decreased) TODO add real
            # armijo condition
            while(learning_rate > 0.05 and armijo_flag == 0):
                # Compute forward pass
                (new_states,new_inputs,new_feedback,new_feedforward,new_saltations,mode_changes,new_txmode_map)=self.forward_pass(learning_rate)
                new_cost = self.compute_cost(new_states, new_inputs, self.dt_)

                print("new_cost: ", new_cost)
                
                # Calculate armijo condition
                cost_difference = (current_cost - new_cost)
                
                expected_cost_redu = learning_rate*self.expected_cost_reduction_grad_ + learning_rate*learning_rate*self.expected_cost_reduction_hess_
                armijo_flag = cost_difference/expected_cost_redu > armijo_threshold
                
                if(armijo_flag == 1):
                    # Accept the new trajectory if armijo condition is
                    # met
                    current_cost = new_cost
                    self.states_ = new_states
                    self.inputs_ = new_inputs
                    self.saltations_ = new_saltations
                    self.modechanges_ = mode_changes
                    self.txmode_map_ = new_txmode_map
                    
                    states_iter.append(new_states)
                    
                else:
                    # If no improvement, decrease the learning rate
                    learning_rate = learning_speed*learning_rate
                    
            if(learning_rate<low_learning_rate):
                # If learning rate is low, then stop optimization
                print("Stopping optimization, low learning rate")
                current_cost = new_cost
                self.states_ = new_states
                self.inputs_ = new_inputs
                self.saltations_ = new_saltations
                self.modechanges_ = mode_changes
                self.txmode_map_ = new_txmode_map
                
                states_iter.append(new_states)
                    
                break
            
        # Return the current trajectory
        states = self.states_
        inputs = self.inputs_
        modechanges = self.modechanges_
        txmode_map = self.txmode_map_
        
        show_results = False
        if show_results:
            fig1, axes = plt.subplots(1, 2)
            (ax1, ax2) = axes.flatten()
            ax1.grid(True)
            ax2.grid(True)
            
            ax1.plot(self.states_[:,0], self.states_[:,1],'k',label='iLQR-deterministic')
            ax1.scatter(self.target_state_[0], self.target_state_[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
            ax1.scatter(self.init_state_[0], self.init_state_[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')

            plt.show()
            
        mode_exttrjs_maps = self.compute_trejactory_extension(txmode_map)

        # extended_trjs = (ext_trjs_fwd, ext_trjs_bwd)
        
        return states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps


def solve_ilqr(params, detect=True):
    # Import dynamics
    smooth_dynamis = params.symbolic_dynamics()
    
    detect_integration = params.detection_func()

    # (f,A,B) = dynamis()
    
    # Initialize timings
    dt = params._dt
    
    start_time = params._start_time
    end_time = params._end_time
    time_span = np.arange(start_time, end_time, dt).flatten()

    # Set desired state
    n_states = 2
    n_inputs = 1
    init_state = params._init_state  # Define the initial state to be the origin with no velocity
    target_state = params._target_state  # Swing pendulum upright

    # Initial guess of zeros, but you can change it to any guess
    initial_guess = params._initial_guess
    
    # initial_guess = np.zeros((np.shape(time_span)[0],n_inputs))

    # Define weighting matrices
    Q_k = params._Q_k # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = params._R_k

    # Set the terminal cost
    Q_T = params._Q_T

    # Set the physical parameters of the system
    mass = 1
    gravity = 9.8
    parameters = np.array([mass,gravity])

    # Specify max number of iterations
    n_iterations = 10
    
    init_mode = params.current_mode()

    # ilqr_ = hybrid_ilqr(init_state,target_state,initial_guess,dt,start_time,end_time,detect_integration,f,A,B,Q_k,R_k,Q_T,parameters,n_iterations,detect)
    ilqr_ = hybrid_ilqr(init_mode,init_state,target_state,initial_guess,
                        dt,start_time,end_time,detect_integration,smooth_dynamis,
                        Q_k,R_k,Q_T,parameters,n_iterations,detect)
    
    (states,inputs,k_feedforward,K_feedback,
     current_cost,states_iter,
     modechanges,mode_exttrjs_maps) = ilqr_.solve()
        
    return (states,inputs,k_feedforward,K_feedback,current_cost,states_iter,modechanges,mode_exttrjs_maps)
        
