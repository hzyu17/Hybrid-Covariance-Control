"""
Hybrid iLQR for Multiple Mode Transitions

This module implements an iterative Linear Quadratic Regulator (iLQR) for hybrid 
dynamical systems that can handle multiple mode transitions (e.g., mode 0 -> 1 -> 0 -> 1).

Key features:
- Proper tracking of multiple hybrid events
- Mode mismatch detection that identifies which event caused the mismatch
- Per-event gain storage for trajectory extensions
- Trajectory extensions computed for each hybrid event

Author: Extended from original single-transition implementation
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.font_manager import FontProperties


# =============================================================================
# Helper Functions
# =============================================================================

def extract_extensions(ref_ext_helper):
    """
    Extract trajectory extensions from the reference extension helper.
    
    This function unpacks the extension helper data structure into separate
    lists for easier access during forward pass computations.
    
    Args:
        ref_ext_helper: List of dictionaries containing extension information
                       for each hybrid event.
    
    Returns:
        v_mode_change_ref: List of mode changes [mode_before, mode_after] for each event
        v_ext_trj_bwd_ref: List of backward extension trajectories (in post-jump mode)
        v_ext_trj_fwd_ref: List of forward extension trajectories (in pre-jump mode)
        v_Kfb_ext_trj_bwd_ref: Feedback gains for backward extensions
        v_Kfb_ext_trj_fwd_ref: Feedback gains for forward extensions
        v_kff_ext_trj_bwd_ref: Feedforward gains for backward extensions
        v_kff_ext_trj_fwd_ref: Feedforward gains for forward extensions
        v_tevents_ref: Event time indices
    """
    v_mode_change_ref = []
    v_ext_trj_bwd_ref = []
    v_ext_trj_fwd_ref = []
    v_Kfb_ext_trj_bwd_ref = []
    v_Kfb_ext_trj_fwd_ref = []
    v_kff_ext_trj_bwd_ref = []
    v_kff_ext_trj_fwd_ref = []
    v_tevents_ref = []
    
    for ext_info in ref_ext_helper:
        mode_change = ext_info["Mode Change"]
        pre_mode = mode_change[0]
        post_mode = mode_change[1]
        
        v_mode_change_ref.append(mode_change)
        v_ext_trj_fwd_ref.append(ext_info["Trajectory Extensions"][pre_mode])
        v_ext_trj_bwd_ref.append(ext_info["Trajectory Extensions"][post_mode])
        v_Kfb_ext_trj_fwd_ref.append(ext_info["Feedback gains"][pre_mode])
        v_Kfb_ext_trj_bwd_ref.append(ext_info["Feedback gains"][post_mode])
        v_kff_ext_trj_fwd_ref.append(ext_info["Feedforward gains"][pre_mode])
        v_kff_ext_trj_bwd_ref.append(ext_info["Feedforward gains"][post_mode])
        v_tevents_ref.append(ext_info["event index"])
    
    return (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref,
            v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref,
            v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref, v_tevents_ref)


# =============================================================================
# Main Hybrid iLQR Class
# =============================================================================

class hybrid_ilqr:
    """
    Hybrid iLQR supporting multiple mode transitions.
    
    This class implements an iterative Linear Quadratic Regulator for hybrid
    dynamical systems. It handles systems with discrete mode transitions
    (e.g., contact/flight phases in legged robots) and computes optimal
    control inputs that account for the hybrid nature of the dynamics.
    
    Key improvements over single-transition version:
    - Properly tracks multiple hybrid events during forward/backward passes
    - Correct mode mismatch detection for arbitrary transition sequences
    - Per-event gain storage for trajectory extensions
    - Robust handling of early/late arrivals at mode boundaries
    
    Attributes:
        nmodes_: Number of discrete modes in the hybrid system
        _n_states: List of state dimensions for each mode
        _n_inputs: List of input dimensions for each mode
        _states: Current trajectory states
        _inputs: Current trajectory inputs (per mode)
        _modes: Mode sequence along trajectory
        _saltations: Saltation matrices at hybrid events
        _hybrid_event_info: Dictionary storing hybrid event information
        gains_at_events_: Per-event feedback/feedforward gains
    """
    
    def __init__(self, nmodes, init_mode, target_mode, nstates,
                 init_state, target_state, initial_guess,
                 dt, start_time, end_time,
                 contact_detect, smooth_dynamics,
                 Q_k, R_k, Q_T, parameters, n_iterations,
                 detect, plot_func, state_convert_func,
                 init_reset_args, target_reset_args,
                 animate_func=None, verbose=True):
        """
        Initialize the hybrid iLQR solver.
        
        Args:
            nmodes: Number of discrete modes
            init_mode: Initial mode index
            target_mode: Target mode index
            nstates: List of state dimensions per mode
            init_state: Initial state vector
            target_state: Target state vector
            initial_guess: Initial control input guess (list per mode)
            dt: Time step
            start_time: Start time of trajectory
            end_time: End time of trajectory
            contact_detect: Function for detecting mode transitions
            smooth_dynamics: List of dynamics functions per mode
            Q_k: Running state cost matrices (list per mode)
            R_k: Running input cost matrices (list per mode)
            Q_T: Terminal state cost matrix
            parameters: Physical parameters (e.g., mass, gravity)
            n_iterations: Maximum number of iLQR iterations
            detect: Boolean flag for event detection
            plot_func: Function for plotting states
            state_convert_func: Function for converting states between modes
            init_reset_args: Initial reset arguments for mode transitions
            target_reset_args: Target reset arguments
            animate_func: Optional animation function
            verbose: Print debug information
        """
        # System dimensions
        self.nmodes_ = nmodes
        self._n_states = nstates
        self._n_inputs = [np.shape(initial_guess[i_mode])[1] for i_mode in range(nmodes)]
        
        # Boundary conditions
        self._init_mode = init_mode
        self._target_mode = target_mode
        self._init_state = init_state
        self._target_state = target_state
        
        # Initial guess
        self._inputs = initial_guess
        
        # Visualization functions
        self._plot_states_func = plot_func
        self._state_convert_func = state_convert_func
        self._animate_func = animate_func
        self._verbose = verbose
        
        # Time definitions
        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self._timespan = np.arange(start_time, end_time, dt).flatten()
        self._n_timesteps = np.shape(self._timespan)[0]
        
        # Initialize feedback and feedforward gains
        self.k_feedforward_ = [np.zeros((self._n_inputs[0])) for _ in range(self._n_timesteps)]
        self.K_feedback_ = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._n_timesteps)]
        
        # Hybrid events and mode changes
        self._saltations = [None for _ in range(self._n_timesteps)]
        self._modechanges = [np.array([0, 0]) for _ in range(self._n_timesteps)]
        self._modes = [0 for _ in range(self._n_timesteps)]
        self._modes[0] = init_mode
        
        # Reset arguments
        self._reset_args = init_reset_args
        self._target_reset_args = target_reset_args
        self._init_reset_args = init_reset_args
        
        # =====================================================================
        # KEY DATA STRUCTURE: Store feedback/feedforward gains PER hybrid event
        # Structure: {event_index: {'fwd': (K, k), 'bwd': (K, k)}}
        # 'fwd' = gains for pre-jump mode (forward extension)
        # 'bwd' = gains for post-jump mode (backward extension)
        # =====================================================================
        self.gains_at_events_ = {}
        
        # Hybrid event information map
        # Maps event index -> (t_event, x_event, x_reset, mode_change, K_ext, k_ext)
        self._hybrid_event_info = {}
        self._reference_extension_helper = []
        
        # Dynamics functions
        self.smooth_dyn_ = smooth_dynamics
        self.detection_func_ = contact_detect
        
        # Initialize dynamics Jacobian functions
        self.f_ = [None for _ in range(self.nmodes_)]
        self.A_func_ = [None for _ in range(self.nmodes_)]
        self.B_func_ = [None for _ in range(self.nmodes_)]
        self.A_ = [None for _ in range(self._n_timesteps)]
        self.B_ = [None for _ in range(self._n_timesteps)]
        
        for ii in range(self.nmodes_):
            self.f_[ii], self.A_func_[ii], self.B_func_[ii] = smooth_dynamics[ii]()
        
        # Cost weighting matrices
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters
        
        # Optimization settings
        self.n_iterations_ = n_iterations
        self.detect_ = detect
        
        # Expected cost reduction (computed in backward pass)
        self.expected_cost_reduction_ = 0
        self.expected_cost_reduction_grad_ = 0
        self.expected_cost_reduction_hess_ = 0

    def rollout(self):
        """
        Perform initial rollout without feedback control.
        
        This method simulates the system forward in time using only the
        initial control guess, without applying any feedback corrections.
        
        Returns:
            timespan: Time vector
            modes: Mode sequence
            states: State trajectory
            inputs: Input trajectory
            saltations: Saltation matrices at events
            mode_changes: Mode change information
        """
        (timespan, modes, states, inputs,
         saltations, mode_changes,
         hybrid_event_info, reset_args) = self.forward_pass(
             use_feedback=False,
             learning_rate=1,
             check_modemismatch=False
         )
        
        # Store trajectory
        self._timespan = timespan
        self._states = states
        self._inputs = inputs
        self._saltations = saltations
        self._modechanges = mode_changes
        self._modes = modes
        self._hybrid_event_info = hybrid_event_info
        self._reset_args = reset_args
        
        return timespan, modes, states, inputs, saltations, mode_changes

    def compute_cost(self, timespan, modes, states, inputs):
        """
        Compute the total cost of a trajectory.
        
        The cost consists of:
        - Running cost on inputs (quadratic)
        - Terminal cost on state error (quadratic)
        
        Args:
            timespan: Time vector
            modes: Mode sequence
            states: State trajectory
            inputs: Input trajectory (list per mode)
            
        Returns:
            total_cost: Scalar total cost value
        """
        total_cost = 0.0
        
        # Running cost
        for ii in range(0, self._n_timesteps - 1):
            dt = timespan[ii + 1] - timespan[ii]
            current_mode = modes[ii]
            current_u = inputs[current_mode][ii].flatten()
            
            # Input cost
            current_cost_u = 0.5 * current_u.T @ self.R_k_[current_mode] @ current_u
            total_cost += current_cost_u * dt
        
        # Terminal cost
        terminal_state = states[-1]
        if modes[-1] != self._target_mode:
            terminal_state = self._state_convert_func(states[-1]).flatten()
        terminal_difference = (self._target_state - terminal_state).flatten()
        terminal_cost = 0.5 * terminal_difference.T @ self.Q_T_ @ terminal_difference
        total_cost += terminal_cost
        
        return total_cost

    def backward_pass(self):
        """
        Perform backward pass to compute optimal gains.
        
        This method implements the backward Riccati recursion to compute
        feedback (K) and feedforward (k) gains. At hybrid events, the
        saltation matrix is used to propagate the value function across
        the mode transition.
        
        KEY CHANGE for multiple transitions:
        - Gains are stored per hybrid event in self.gains_at_events_
        - This allows proper matching of gains to events in forward pass
        
        Returns:
            k_trj: Feedforward gain trajectory
            K_trj: Feedback gain trajectory
            expected_reduction: Expected cost reduction
            A_trj: State Jacobian trajectory
            B_trj: Input Jacobian trajectory
        """
        # Initialize value function at terminal time
        V_xx = self.Q_T_
        end_difference = (self._states[-1] - self._target_state).flatten()
        V_x = self.Q_T_ @ end_difference
        
        # Initialize gain trajectories
        k_trj = [np.zeros((self._n_inputs[0])) for _ in range(self._n_timesteps)]
        K_trj = [np.zeros((self._n_inputs[0], self._n_states[0])) for _ in range(self._n_timesteps)]
        A_trj = [None for _ in range(self._n_timesteps)]
        B_trj = [None for _ in range(self._n_timesteps)]
        
        # Reset per-event gains storage
        self.gains_at_events_ = {}
        
        # Initialize expected cost reduction
        expected_cost_reduction = 0
        expected_cost_reduction_grad = 0
        expected_cost_reduction_hess = 0
        
        # Compute Jacobians at final time
        current_mode = self._modes[-1]
        current_x = self._states[-1]
        current_u = self._inputs[current_mode][-1]
        A_k = self.A_func_[current_mode](current_x, current_u, self.dt_)
        B_k = self.B_func_[current_mode](current_x, current_u, self.dt_)
        A_trj[-1] = A_k
        B_trj[-1] = B_k
        
        # Backward iteration through time
        for idx in reversed(range(0, self._n_timesteps - 1)):
            current_mode = self._modes[idx]
            current_x = self._states[idx]
            current_u = self._inputs[current_mode][idx]
            saltation = self._saltations[idx]
            
            # Cost expansion coefficients
            l_xx = self.Q_k_[current_mode]
            l_uu = self.R_k_[current_mode]
            l_x = self.Q_k_[current_mode] @ current_x.flatten()
            l_u = self.R_k_[current_mode] @ current_u.flatten()
            
            # Dynamics Jacobians
            A_k = self.A_func_[current_mode](current_x, current_u, self.dt_)
            B_k = self.B_func_[current_mode](current_x, current_u, self.dt_)
            A_trj[idx] = A_k
            B_trj[idx] = B_k
            
            # Q-function expansion
            if saltation is None:
                # Standard LQR update (no mode transition)
                Q_x = l_x * self.dt_ + A_k.T @ V_x
                Q_u = l_u * self.dt_ + B_k.T @ V_x
                Q_ux = B_k.T @ V_xx @ A_k
                Q_uu = l_uu * self.dt_ + B_k.T @ V_xx @ B_k
                Q_xx = l_xx * self.dt_ + A_k.T @ V_xx @ A_k
            else:
                # Hybrid update with saltation matrix
                # The saltation matrix captures the sensitivity of the
                # post-reset state to the pre-reset state
                Q_x = l_x * self.dt_ + A_k.T @ saltation.T @ V_x
                Q_u = l_u * self.dt_ + B_k.T @ saltation.T @ V_x
                Q_ux = B_k.T @ saltation.T @ V_xx @ saltation @ A_k
                Q_uu = l_uu * self.dt_ + B_k.T @ saltation.T @ V_xx @ saltation @ B_k
                Q_xx = l_xx * self.dt_ + A_k.T @ saltation.T @ V_xx @ saltation @ A_k
            
            # Compute optimal gains
            k = -np.linalg.solve(Q_uu, Q_u)
            K = -np.linalg.solve(Q_uu, Q_ux).reshape(
                (self._n_inputs[current_mode], self._n_states[current_mode])
            )
            
            # =====================================================================
            # KEY CHANGE: Store gains at hybrid events for extension computation
            # This enables proper gain matching in forward pass with multiple events
            # =====================================================================
            if saltation is not None:
                # Store gains for this event
                # 'fwd' = pre-jump gains (for forward extension in pre-jump mode)
                # 'bwd' = post-jump gains (for backward extension in post-jump mode)
                self.gains_at_events_[idx] = {
                    'fwd': (K, k),
                    'bwd': (K_trj[idx + 1], k_trj[idx + 1])
                }
                
                # Update hybrid event info with gains
                if idx in self._hybrid_event_info:
                    prev_info = list(self._hybrid_event_info[idx])
                    prev_info[4] = [K, K_trj[idx + 1]]  # [K_fwd, K_bwd]
                    prev_info[5] = [k, k_trj[idx + 1]]  # [k_fwd, k_bwd]
                    self._hybrid_event_info[idx] = tuple(prev_info)
            
            # Store gains
            k_trj[idx] = k
            K_trj[idx] = K
            
            # Update expected cost reduction
            current_cost_reduction_grad = -Q_u.T @ k
            current_cost_reduction_hess = 0.5 * k.T @ Q_uu @ k
            expected_cost_reduction_grad += current_cost_reduction_grad
            expected_cost_reduction_hess += current_cost_reduction_hess
            expected_cost_reduction += current_cost_reduction_grad + current_cost_reduction_hess
            
            # Update value function for next iteration
            V_x = Q_x - K.T @ Q_uu @ k
            V_xx = Q_xx - K.T @ Q_uu @ K
        
        # Store expected cost reductions
        self.expected_cost_reduction_grad_ = expected_cost_reduction_grad
        self.expected_cost_reduction_hess_ = expected_cost_reduction_hess
        self.expected_cost_reduction_ = expected_cost_reduction
        
        # Store gain schedules
        self.k_feedforward_ = k_trj
        self.K_feedback_ = K_trj
        
        return k_trj, K_trj, expected_cost_reduction, A_trj, B_trj

    def _find_matching_event(self, current_time_idx, current_mode, ref_modes,
                             v_tevents_ref, v_mode_change_ref):
        """
        Find which reference hybrid event corresponds to a mode mismatch.
        
        When the current mode differs from the reference mode, this method
        determines which hybrid event in the reference trajectory caused
        the mismatch. This is essential for multiple transitions where
        there could be several candidate events.
        
        The method considers both:
        1. Time proximity to candidate events
        2. Mode transition direction (determines early vs late arrival)
        
        Args:
            current_time_idx: Current time index in trajectory
            current_mode: Current mode of the system
            ref_modes: Reference mode sequence
            v_tevents_ref: Reference event time indices
            v_mode_change_ref: Reference mode changes
            
        Returns:
            event_idx: Index into the reference event lists
            is_early: True if arrived early at this event, False if late
        """
        ref_mode = ref_modes[current_time_idx]
        
        # Find candidate events that could explain the mode mismatch
        candidates = []
        for i, t_event in enumerate(v_tevents_ref):
            mode_before = v_mode_change_ref[i][0]
            mode_after = v_mode_change_ref[i][1]
            
            # Early arrival: we're already in post-jump mode but ref is still in pre-jump
            if current_mode == mode_after and ref_mode == mode_before:
                candidates.append((i, True, abs(current_time_idx - t_event)))
            
            # Late arrival: we're still in pre-jump mode but ref has moved to post-jump
            elif current_mode == mode_before and ref_mode == mode_after:
                candidates.append((i, False, abs(current_time_idx - t_event)))
        
        if not candidates:
            # Fallback: find closest event by time
            closest_idx = np.argmin(np.abs(np.array(v_tevents_ref) - current_time_idx))
            mode_before = v_mode_change_ref[closest_idx][0]
            mode_after = v_mode_change_ref[closest_idx][1]
            is_early = (current_mode == mode_after)
            return closest_idx, is_early
        
        # Return the closest matching candidate
        candidates.sort(key=lambda x: x[2])  # Sort by time distance
        return candidates[0][0], candidates[0][1]

    def forward_pass(self, use_feedback=True, learning_rate=1, check_modemismatch=True):
        """
        Perform forward pass with feedback control.
        
        This method simulates the system forward in time, applying the
        computed feedback and feedforward gains. When mode mismatches
        occur (current mode differs from reference), it uses trajectory
        extensions to compute appropriate feedback.
        
        KEY CHANGES for multiple transitions:
        1. Track which segment between events we're in
        2. Properly match mode mismatches to correct reference events
        3. Use per-event extension trajectories and gains
        
        Args:
            use_feedback: Apply feedback control (False for initial rollout)
            learning_rate: Step size for feedforward gains
            check_modemismatch: Check and handle mode mismatches
            
        Returns:
            timespan: Updated time vector
            modes: Mode sequence
            states: State trajectory
            inputs: Input trajectory
            saltations: Saltation matrices
            mode_changess: Mode change information
            hybrid_event_info: Hybrid event details
            reset_args: Reset arguments for mode transitions
        """
        if self._verbose:
            if not use_feedback:
                print("---------- Initial rollout ----------")
            else:
                print(f"---------- Forward pass. Learning rate: {learning_rate} ----------")
        
        # Initialize trajectory storage
        timespan = self._timespan.copy()
        modes = [0 for _ in range(self._n_timesteps)]
        states = [np.array([0.0]) for _ in range(self._n_timesteps)]
        inputs = [np.zeros((self._n_timesteps, self._n_inputs[i])) for i in range(self.nmodes_)]
        saltations = [None for _ in range(self._n_timesteps)]
        mode_changess = np.tile(np.array([0, 0]), (self._n_timesteps, 1))
        
        # Handle reset_args initialization
        if isinstance(self._init_reset_args, list):
            reset_args = self._init_reset_args.copy()
        else:
            reset_args = [self._init_reset_args[i] if i < len(self._init_reset_args) 
                         else self._init_reset_args[0] for i in range(self._n_timesteps)]
        
        # Set initial conditions
        modes[0] = self._init_mode
        states[0] = self._init_state
        mode_changess[0] = np.array([self._init_mode, self._init_mode])
        
        # Hybrid event tracking
        hybrid_event_info = {}
        cnt_event = 0
        
        # Initialize event arguments
        if isinstance(self._init_reset_args, list) and len(self._init_reset_args) > 0:
            event_args = self._init_reset_args[0]
        else:
            event_args = self._init_reset_args
        
        # =====================================================================
        # Extract ALL reference extensions for multiple events
        # =====================================================================
        v_mode_change_ref = []
        v_tevents_ref = []
        v_ext_trj_bwd_ref = []
        v_ext_trj_fwd_ref = []
        v_Kfb_ext_trj_bwd_ref = []
        v_Kfb_ext_trj_fwd_ref = []
        v_kff_ext_trj_bwd_ref = []
        v_kff_ext_trj_fwd_ref = []
        
        if use_feedback and self._reference_extension_helper:
            (v_mode_change_ref, v_ext_trj_bwd_ref, v_ext_trj_fwd_ref,
             v_Kfb_ext_trj_bwd_ref, v_Kfb_ext_trj_fwd_ref,
             v_kff_ext_trj_bwd_ref, v_kff_ext_trj_fwd_ref,
             v_tevents_ref) = extract_extensions(self._reference_extension_helper)
            
            if self._verbose:
                print(f"Reference trajectory has {len(v_ext_trj_bwd_ref)} hybrid events:")
                for i_bounce in range(len(v_ext_trj_bwd_ref)):
                    print(f"  Event {i_bounce}: mode {v_mode_change_ref[i_bounce][0]} -> "
                          f"{v_mode_change_ref[i_bounce][1]} at t_idx={v_tevents_ref[i_bounce]}")
                print("------------------------------------------------")
        
        # Track current event index for extension gains
        current_event_idx = 0
        
        # =====================================================================
        # Main forward integration loop
        # =====================================================================
        for ii in range(self._n_timesteps - 1):
            current_mode = modes[ii]
            current_state = states[ii]
            current_input = self._inputs[current_mode][ii]
            
            # Update reset arguments
            if isinstance(reset_args, list) and ii < len(reset_args):
                reset_args[ii] = event_args
            
            # Apply feedback control
            if use_feedback:
                ref_state = self._states[ii]
                current_mode_ref = self._modes[ii]
                
                # Get nominal gains
                current_feedforward = learning_rate * self.k_feedforward_[ii]
                current_feedback = self.K_feedback_[ii]
                
                # =====================================================================
                # MODE MISMATCH HANDLING for multiple transitions
                # =====================================================================
                if current_mode != current_mode_ref and check_modemismatch and v_tevents_ref:
                    # Find which event caused this mismatch
                    matched_event_idx, is_early = self._find_matching_event(
                        ii, current_mode, self._modes,
                        v_tevents_ref, v_mode_change_ref
                    )
                    
                    if self._verbose:
                        arrival_type = "EARLY" if is_early else "LATE"
                        print(f"Mode mismatch at t_idx={ii}: {arrival_type} arrival")
                        print(f"  Current mode: {current_mode}, Reference mode: {current_mode_ref}")
                        print(f"  Matched to event {matched_event_idx}: "
                              f"{v_mode_change_ref[matched_event_idx][0]} -> "
                              f"{v_mode_change_ref[matched_event_idx][1]}")
                    
                    # Get correct extension based on arrival type
                    if is_early:
                        # Early arrival: use backward extension (in post-jump mode)
                        # We arrived at the next mode before the reference did
                        trj_extension = v_ext_trj_bwd_ref[matched_event_idx]
                        fb_ext_trj = v_Kfb_ext_trj_bwd_ref[matched_event_idx]
                        ff_ext_trj = v_kff_ext_trj_bwd_ref[matched_event_idx]
                    else:
                        # Late arrival: use forward extension (in pre-jump mode)
                        # We're still in the previous mode while reference moved on
                        trj_extension = v_ext_trj_fwd_ref[matched_event_idx]
                        fb_ext_trj = v_Kfb_ext_trj_fwd_ref[matched_event_idx]
                        ff_ext_trj = v_kff_ext_trj_fwd_ref[matched_event_idx]
                    
                    # Apply extension reference and gains
                    if ii < len(trj_extension):
                        ref_state = trj_extension[ii]
                        current_feedback = fb_ext_trj[ii]
                        current_feedforward = learning_rate * ff_ext_trj[ii]
                    
                    current_event_idx = matched_event_idx
                
                # Compute control input with feedback
                current_input = (current_input +
                               current_feedback @ (current_state - ref_state) +
                               current_feedforward)
            
            # =====================================================================
            # Integrate dynamics with event detection
            # =====================================================================
            t_ii = self._timespan[ii]
            dt = self._timespan[ii + 1] - t_ii
            
            # Get current reset arguments
            current_reset_args = reset_args[ii] if isinstance(reset_args, list) else event_args
            
            (next_state, saltation, mode_change,
             t_event, x_event, x_reset, reset_byproduct) = self.detection_func_(
                current_mode, current_state, current_input,
                t_ii, dt, current_reset_args, self.detect_
            )
            
            # Record hybrid event
            if saltation is not None:
                saltations[ii] = saltation
                
                # Get gains for this event
                if current_event_idx < len(self._reference_extension_helper):
                    K_ext = self._reference_extension_helper[current_event_idx].get("Feedback gains", {})
                    k_ext = self._reference_extension_helper[current_event_idx].get("Feedforward gains", {})
                else:
                    # Fallback: use current gains
                    next_idx = min(ii + 1, self._n_timesteps - 1)
                    K_ext = [self.K_feedback_[ii], self.K_feedback_[next_idx]]
                    k_ext = [self.k_feedforward_[ii], self.k_feedforward_[next_idx]]
                
                hybrid_event_info[ii] = (t_event, x_event, x_reset, mode_change, K_ext, k_ext)
                timespan[ii + 1] = t_event
                cnt_event += 1
            
            # Handle mode transition
            if mode_change[0] != mode_change[1]:
                if self._verbose:
                    print(f"===== Mode change at t_idx={ii}: {mode_change[0]} -> {mode_change[1]} =====")
                event_args = reset_byproduct
            
            # Store results
            states[ii + 1] = next_state
            inputs[current_mode][ii] = current_input.flatten()
            mode_changess[ii + 1] = mode_change
            modes[ii + 1] = mode_change[1]
        
        if self._verbose:
            print(f"--------------------- Total hybrid events: {cnt_event} ---------------------")
        
        return (timespan, modes, states, inputs, saltations,
                mode_changess, hybrid_event_info, reset_args)

    def compute_trajectory_extension(self, timespan, states, hybrid_event_info):
        """
        Compute trajectory extensions for all hybrid events.
        
        For each hybrid event, this method computes:
        1. Forward extension: Continuing in pre-jump mode beyond the jump
        2. Backward extension: Propagating backward from reset state in post-jump mode
        
        These extensions are used to provide reference trajectories when
        mode mismatches occur (early or late arrivals at mode boundaries).
        
        KEY CHANGE for multiple transitions:
        - Properly handles multiple events by computing extensions that
          span between consecutive events, not just to boundaries
        - Each event gets its own extension trajectories and gains
        
        Args:
            timespan: Current time vector
            states: Current state trajectory
            hybrid_event_info: Dictionary of hybrid event information
            
        Returns:
            ref_ext_helper: List of extension dictionaries for each event
        """
        if not hybrid_event_info:
            return []
        
        sorted_hybrid_indices = sorted(hybrid_event_info.keys())
        ref_ext_helper = []
        
        # Build event lists with boundary conditions
        i_events = [0]
        t_events = [self.start_time_]
        x_events = [self._init_state]
        x_resets = [self._init_state]
        mode_changes = [np.array([self._init_mode, self._init_mode])]
        K_fb_fwd = [np.zeros((self._n_inputs[0], self._n_states[0]))]
        K_fb_bwd = [np.zeros((self._n_inputs[0], self._n_states[0]))]
        k_ff_fwd = [np.zeros(self._n_inputs[0])]
        k_ff_bwd = [np.zeros(self._n_inputs[0])]
        
        # Add all hybrid events
        for i_key in sorted_hybrid_indices:
            info = hybrid_event_info[i_key]
            i_events.append(i_key)
            t_events.append(info[0])
            x_events.append(info[1])
            x_resets.append(info[2])
            mode_changes.append(info[3])
            
            # Get gains from backward pass storage
            if i_key in self.gains_at_events_:
                K_fb_fwd.append(self.gains_at_events_[i_key]['fwd'][0])
                K_fb_bwd.append(self.gains_at_events_[i_key]['bwd'][0])
                k_ff_fwd.append(self.gains_at_events_[i_key]['fwd'][1])
                k_ff_bwd.append(self.gains_at_events_[i_key]['bwd'][1])
            else:
                # Fallback: use nominal gains
                K_fb_fwd.append(self.K_feedback_[i_key])
                K_fb_bwd.append(self.K_feedback_[min(i_key + 1, self._n_timesteps - 1)])
                k_ff_fwd.append(self.k_feedforward_[i_key])
                k_ff_bwd.append(self.k_feedforward_[min(i_key + 1, self._n_timesteps - 1)])
        
        # Add terminal event
        i_events.append(self._n_timesteps)
        t_events.append(self.end_time_)
        x_events.append(self._target_state)
        x_resets.append(self._target_state)
        mode_changes.append(mode_changes[-1] if mode_changes else np.array([0, 0]))
        K_fb_fwd.append(np.zeros((self._n_inputs[0], self._n_states[0])))
        K_fb_bwd.append(np.zeros((self._n_inputs[0], self._n_states[0])))
        k_ff_fwd.append(np.zeros(self._n_inputs[0]))
        k_ff_bwd.append(np.zeros(self._n_inputs[0]))
        
        # =====================================================================
        # Compute extensions for each hybrid event
        # =====================================================================
        for ii in range(1, len(t_events) - 1):
            i_event = i_events[ii]
            i_event_prev = i_events[ii - 1]
            i_event_next = i_events[ii + 1]
            
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            pre_mode = mode_changes[ii][0]
            post_mode = mode_changes[ii][1]
            
            # =================================================================
            # Forward extension: Continue in pre-jump mode beyond the jump
            # =================================================================
            timespan_ext_fwd = timespan[i_event + 1:]
            nt_ext_fwd = len(timespan_ext_fwd)
            xtrj_ext_fwd = np.zeros((nt_ext_fwd, self._n_states[pre_mode]))
            
            if nt_ext_fwd > 0:
                xtrj_ext_fwd[0] = np.asarray(x_event_i)
                
                for jj in range(nt_ext_fwd - 1):
                    current_state = xtrj_ext_fwd[jj]
                    t_jj = timespan_ext_fwd[jj]
                    dt = timespan_ext_fwd[jj + 1] - t_jj
                    current_input = np.zeros(self._n_inputs[pre_mode])
                    
                    # Get reset args
                    reset_idx = min(jj, len(self._reset_args) - 1) if isinstance(self._reset_args, list) else 0
                    current_reset_args = self._reset_args[reset_idx] if isinstance(self._reset_args, list) else self._reset_args
                    
                    next_state, _, _, _, _, _, _ = self.detection_func_(
                        pre_mode, current_state, current_input,
                        t_jj, dt, current_reset_args, detection=False
                    )
                    xtrj_ext_fwd[jj + 1] = next_state
            
            # Pad with actual trajectory before the event
            xtrj_ext_fwd_padded = np.zeros((self._n_timesteps, self._n_states[pre_mode]))
            
            # Fill in trajectory before event
            for jj in range(i_event_prev, i_event + 1):
                if jj < len(states):
                    s = states[jj]
                    if hasattr(s, '__len__') and len(s) == self._n_states[pre_mode]:
                        xtrj_ext_fwd_padded[jj] = s
                    else:
                        xtrj_ext_fwd_padded[jj] = np.zeros(self._n_states[pre_mode])
            
            # Fill in forward extension
            if nt_ext_fwd > 0:
                end_idx = min(i_event + 1 + nt_ext_fwd, self._n_timesteps)
                xtrj_ext_fwd_padded[i_event + 1:end_idx] = xtrj_ext_fwd[:end_idx - i_event - 1]
            
            # =================================================================
            # Backward extension: Propagate backward from reset in post-jump mode
            # =================================================================
            timespan_ext_bwd = timespan[:i_event + 1][::-1]
            nt_ext_bwd = len(timespan_ext_bwd)
            xtrj_ext_bwd = np.zeros((nt_ext_bwd, self._n_states[post_mode]))
            
            if nt_ext_bwd > 0:
                xtrj_ext_bwd[0] = x_reset_i
                
                for jj in range(nt_ext_bwd - 1):
                    current_state = xtrj_ext_bwd[jj]
                    t_jj = timespan_ext_bwd[jj]
                    dt = t_jj - timespan_ext_bwd[jj + 1]
                    current_input = np.zeros(self._n_inputs[post_mode])
                    
                    # Get reset args
                    reset_idx = min(jj, len(self._reset_args) - 1) if isinstance(self._reset_args, list) else 0
                    current_reset_args = self._reset_args[reset_idx] if isinstance(self._reset_args, list) else self._reset_args
                    
                    next_state, _, _, _, _, _, _ = self.detection_func_(
                        post_mode, current_state, current_input,
                        t_jj, dt, current_reset_args,
                        detect=False, backwards=True
                    )
                    xtrj_ext_bwd[jj + 1] = next_state
                
                # Reverse to get chronological order
                xtrj_ext_bwd = xtrj_ext_bwd[::-1]
            
            # Pad with actual trajectory after the event
            xtrj_ext_bwd_padded = np.zeros((self._n_timesteps, self._n_states[post_mode]))
            
            # Fill in backward extension
            if nt_ext_bwd > 0:
                xtrj_ext_bwd_padded[:i_event + 1] = xtrj_ext_bwd
            
            # Fill in trajectory after event
            for jj in range(i_event + 1, min(i_event_next, self._n_timesteps)):
                if jj < len(states):
                    s = states[jj]
                    if hasattr(s, '__len__') and len(s) == self._n_states[post_mode]:
                        xtrj_ext_bwd_padded[jj] = s
                    else:
                        xtrj_ext_bwd_padded[jj] = np.zeros(self._n_states[post_mode])
            
            # Create gains arrays (tiled for each timestep)
            K_fb_ext_fwd = np.tile(K_fb_fwd[ii], (self._n_timesteps, 1, 1))
            k_ff_ext_fwd = np.tile(k_ff_fwd[ii], (self._n_timesteps, 1))
            K_fb_ext_bwd = np.tile(K_fb_bwd[ii], (self._n_timesteps, 1, 1))
            k_ff_ext_bwd = np.tile(k_ff_bwd[ii], (self._n_timesteps, 1))
            
            # Store extension information
            ref_ext_helper.append({
                "Mode Change": np.array([pre_mode, post_mode]),
                "Trajectory Extensions": {
                    pre_mode: xtrj_ext_fwd_padded,
                    post_mode: xtrj_ext_bwd_padded
                },
                "Feedback gains": {
                    pre_mode: K_fb_ext_fwd,
                    post_mode: K_fb_ext_bwd
                },
                "Feedforward gains": {
                    pre_mode: k_ff_ext_fwd,
                    post_mode: k_ff_ext_bwd
                },
                "event index": i_event
            })
        
        return ref_ext_helper

    def solve(self):
        """
        Main optimization loop.
        
        This method runs the iLQR algorithm:
        1. Initial rollout with control guess
        2. Backward pass to compute gains
        3. Forward pass with line search
        4. Repeat until convergence
        
        Returns:
            timespan: Optimized time vector
            modes: Optimized mode sequence
            states: Optimized state trajectory
            inputs: Optimized input trajectory
            saltations: Saltation matrices
            k_feedforward: Feedforward gains
            K_feedback: Feedback gains
            A_trj: State Jacobians
            B_trj: Input Jacobians
            current_cost: Final cost
            states_iter: State trajectories from each iteration
            modechanges: Mode change information
            ref_ext_helper: Reference extension helper
            reset_args: Reset arguments
        """
        states_iter = []
        
        # =================================================================
        # Initial rollout
        # =================================================================
        timespan, modes, states, inputs, saltations, modechanges = self.rollout()
        print("===================== Finished initial rollout =====================")
        
        # Optional: Plot initial rollout
        show_rollout = True
        if show_rollout and self._plot_states_func:
            self._plot_states_func(self._timespan, modes, states, inputs,
                                  self._init_state, self._target_state,
                                  self._n_timesteps, reset_args=self._reset_args, step=50)
            
            if self._animate_func:
                fig, ax = self._animate_func(self._modes, self._states, self._init_mode,
                                            self._init_state, self._target_mode, self._target_state,
                                            self._n_timesteps, self._reset_args, self._target_reset_args, step=200)
                
                ax.grid(True, linestyle='--', alpha=0.7)
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
                plt.show()
        
        # =================================================================
        # Compute initial cost
        # =================================================================
        current_cost = self.compute_cost(timespan, modes, states, inputs)
        
        # Optimization parameters
        learning_speed = 0.98
        low_learning_rate = 0.001
        low_expected_reduction = 1e-4
        armijo_threshold = 0.05
        
        # Initialize gains for return
        k_feedforward = self.k_feedforward_
        K_feedback = self.K_feedback_
        A_trj = [None for _ in range(self._n_timesteps)]
        B_trj = [None for _ in range(self._n_timesteps)]
        
        # =================================================================
        # Main optimization loop
        # =================================================================
        for ii in range(self.n_iterations_):
            current_cost = self.compute_cost(
                self._timespan, self._modes, self._states, self._inputs
            )
            print(f'========== Iteration {ii}, Cost: {current_cost:.6f} ==========')
            
            # -------------------------------------------------------------
            # Backward pass
            # -------------------------------------------------------------
            print("-------- Backward Pass --------")
            k_feedforward, K_feedback, expected_reduction, A_trj, B_trj = self.backward_pass()
            
            # Compute trajectory extensions for all hybrid events
            self._reference_extension_helper = self.compute_trajectory_extension(
                self._timespan, self._states, self._hybrid_event_info
            )
            
            print(f'Expected cost reduction: {expected_reduction:.6f}')
            
            # Check convergence
            if abs(expected_reduction) < low_expected_reduction:
                print("-------- Converged: small expected reduction --------")
                break
            
            # -------------------------------------------------------------
            # Line search
            # -------------------------------------------------------------
            learning_rate = 1.0
            armijo_flag = False
            
            while learning_rate > low_learning_rate and not armijo_flag:
                # Forward pass with current learning rate
                (new_timespan, new_modes, new_states, new_inputs,
                 new_saltations, new_mode_changes,
                 new_hybrid_event_info, new_reset_args) = self.forward_pass(learning_rate)
                
                new_cost = self.compute_cost(new_timespan, new_modes, new_states, new_inputs)
                print(f"  Learning rate: {learning_rate:.4f}, New cost: {new_cost:.6f}")
                
                # Check Armijo condition
                cost_diff = current_cost - new_cost
                expected_redu = (learning_rate * self.expected_cost_reduction_grad_ +
                               learning_rate**2 * self.expected_cost_reduction_hess_)
                
                if expected_redu > 0:
                    armijo_flag = cost_diff / expected_redu > armijo_threshold
                
                if armijo_flag:
                    print("-------- Armijo condition satisfied --------")
                    # Accept new trajectory
                    current_cost = new_cost
                    self._timespan = new_timespan
                    self._states = new_states
                    self._inputs = new_inputs
                    self._saltations = new_saltations
                    self._modechanges = new_mode_changes
                    self._modes = new_modes
                    self._hybrid_event_info = new_hybrid_event_info
                    self._reset_args = new_reset_args
                    self._reference_extension_helper = self.compute_trajectory_extension(
                        new_timespan, new_states, new_hybrid_event_info
                    )
                    states_iter.append(new_states)
                else:
                    # Reduce learning rate
                    learning_rate *= learning_speed
            
            # Check if learning rate became too small
            if learning_rate < low_learning_rate:
                print("-------- Stopping: learning rate too small --------")
                break
            
            if ii == self.n_iterations_ - 1:
                print("-------- Stopping: reached max iterations --------")
        
        # =================================================================
        # Prepare return values
        # =================================================================
        timespan = self._timespan
        modes = self._modes
        states = self._states
        inputs = self._inputs
        saltations = self._saltations
        modechanges = self._modechanges
        hybrid_event_info = self._hybrid_event_info
        reset_args = self._reset_args
        
        ref_ext_helper = self.compute_trajectory_extension(timespan, states, hybrid_event_info)
        
        # Show final result
        if show_rollout and self._plot_states_func:
            self._plot_states_func(self._timespan, modes, states, inputs,
                                  self._init_state, self._target_state,
                                  self._n_timesteps, reset_args=self._reset_args, step=50)
            
            if self._animate_func:
                fig, ax = self._animate_func(self._modes, self._states, self._init_mode,
                                            self._init_state, self._target_mode, self._target_state,
                                            self._n_timesteps, self._reset_args, self._target_reset_args, step=200)
                
                ax.grid(True, linestyle='--', alpha=0.7)
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
                ax.set_title('Optimized Rollout', fontsize=14, fontfamily='serif')
                fig.tight_layout()
                plt.show()
        
        return (timespan, modes, states, inputs, saltations,
                k_feedforward, K_feedback, A_trj, B_trj,
                current_cost, states_iter,
                modechanges, ref_ext_helper, reset_args)


# =============================================================================
# Solver Wrapper Function
# =============================================================================

def solve_ilqr(params, detect=True, verbose=True):
    """
    Wrapper function to create and solve hybrid iLQR with multiple transitions.
    
    This function provides a convenient interface to the hybrid_ilqr class,
    extracting all necessary parameters from a params object.
    
    Args:
        params: Parameter object containing:
            - symbolic_dynamics(): Returns dynamics functions per mode
            - detection_func(): Returns event detection function
            - plotting_function(): Returns visualization function
            - state_convert_function(): Returns state conversion function
            - animate_function(): Returns animation function
            - _dt: Time step
            - _start_time, _end_time: Time bounds
            - _init_state, _target_state: Boundary states
            - _initial_guess: Initial control guess
            - _init_reset_args, _target_reset_args: Reset arguments
            - _Q_k, _R_k, _Q_T: Cost matrices
            - current_mode(): Returns initial mode
            - _target_mode: Target mode
            - nmodes(): Returns number of modes
            - _nstates: State dimensions per mode
        detect: Enable event detection
        verbose: Print debug information
        
    Returns:
        Same as hybrid_ilqr.solve()
    """
    # Get dynamics functions
    smooth_dynamics = params.symbolic_dynamics()
    detect_integration = params.detection_func()
    
    # Get visualization functions
    plotting_function = params.plotting_function()
    state_convert_function = params.state_convert_function()
    animate_function = params.animate_function()
    
    # Time parameters
    dt = params._dt
    start_time = params._start_time
    end_time = params._end_time
    
    # Boundary conditions
    init_state = params._init_state
    target_state = params._target_state
    initial_guess = params._initial_guess
    
    # Reset arguments
    init_reset_args = params._init_reset_args
    target_reset_args = params._target_reset_args
    
    # Cost matrices
    Q_k = params._Q_k
    R_k = params._R_k
    Q_T = params._Q_T
    
    # Physical parameters
    mass = 1
    gravity = 9.8
    parameters = np.array([mass, gravity])
    
    # Optimization settings
    n_iterations = 50
    
    # Mode information
    init_mode = params.current_mode()
    target_mode = params._target_mode
    nmodes = params.nmodes()
    nstates = params._nstates
    
    # Create and run solver
    ilqr = hybrid_ilqr(
        nmodes, init_mode, target_mode, nstates,
        init_state, target_state, initial_guess,
        dt, start_time, end_time,
        detect_integration, smooth_dynamics,
        Q_k, R_k, Q_T, parameters, n_iterations,
        detect, plotting_function, state_convert_function,
        init_reset_args, target_reset_args,
        animate_function, verbose
    )
    
    return ilqr.solve()


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    print("Hybrid iLQR for Multiple Mode Transitions")
    print("=========================================")
    print("This module provides the hybrid_ilqr class and solve_ilqr function.")
    print("Import and use with your dynamics module:")
    print("")
    print("  from hybrid_ilqr_multi_transition_complete import solve_ilqr")
    print("  from your_dynamics import YourParams")
    print("")
    print("  params = YourParams()")
    print("  results = solve_ilqr(params, detect=True, verbose=True)")