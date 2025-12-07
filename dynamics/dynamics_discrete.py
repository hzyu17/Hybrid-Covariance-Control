"""
Discrete dynamics utilities for hybrid systems.

This module provides functions for:
- Stochastic integration with guard detection
- Event detection and saltation matrix computation
- Stochastic feedback rollout for hybrid systems
"""

import numpy as np
from dynamics.trajectory_extension import extract_extensions
from dynamics.saltation_matrix import compute_saltation


# =============================================================================
#                           Stochastic Integration
# =============================================================================

def h_stoch_integr(xt, current_mode, ut, rand_noise, eps, dt, t0, reset_arg,
                   stoch_integr_func=None,
                   guards=None,
                   guard_true_funcs=None,
                   guard_false_funcs=None):
    """
    Perform one step of stochastic integration with hybrid guard detection.
    
    Args:
        xt: Current state
        current_mode: Current discrete mode (0 or 1)
        ut: Control input
        rand_noise: Gaussian noise sample
        eps: Noise scaling parameter
        dt: Time step
        t0: Current time
        reset_arg: Arguments for reset map
        stoch_integr_func: Stochastic integration function
        guards: Dict/list of guard functions for each mode
        guard_true_funcs: Dict/list of functions when guard is triggered
        guard_false_funcs: Dict/list of functions when guard is not triggered
    
    Returns:
        xt_next: Next state
        next_mode: Next discrete mode
        dW: Wiener process increment
        new_reset_arg: Updated reset arguments
    """
    dW = np.sqrt(dt) * rand_noise
    xt_next = stoch_integr_func(current_mode, xt, ut, dt, eps, dW)
    
    args_guard = (xt, current_mode, ut, t0, xt_next, dt, rand_noise, eps, reset_arg)
    
    guard_hit = guards[current_mode](xt, xt_next, current_mode)
    
    if guard_hit:
        xt_next, next_mode, dW, new_reset_arg = guard_true_funcs[current_mode](args_guard)
    else:
        xt_next, next_mode, dW, new_reset_arg = guard_false_funcs[current_mode](args_guard)
    
    return xt_next, next_mode, dW, new_reset_arg


# =============================================================================
#                              Guard Detection
# =============================================================================

def bisection_event_detection(xt_current, current_mode, u_current, t, xt_next, 
                               dt_int, smooth_dyn, guard, resetmap, reset_arg,
                               tol=1e-8):
    """
    Use bisection method to find exact event time when guard is triggered.
    
    Args:
        xt_current: State at start of interval
        current_mode: Current discrete mode
        u_current: Control input
        t: Start time
        xt_next: State at end of interval (after guard crossing)
        dt_int: Integration time step
        smooth_dyn: Smooth dynamics function
        guard: Guard function
        resetmap: Reset map function
        reset_arg: Reset map arguments
        tol: Bisection tolerance
    
    Returns:
        t_event: Time of event
        x_event: State at event
        x_reset: State after reset
        next_mode: Mode after reset
        new_reset_arg: Updated reset arguments
    """
    t_left, x_left = t, xt_current
    t_right = t + dt_int
    
    while (t_right - t_left) / 2.0 > tol:
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_right - t_left
        x_mid = x_left + smooth_dyn(t_left, x_left, u_current) * dt_new
        
        if guard(t_mid, x_mid) == 0:
            t_left, x_left = t_mid, x_mid
            break
        elif guard(t_left, x_left) * guard(t_mid, x_mid) < 0:
            t_right = t_mid
        else:
            t_left, x_left = t_mid, x_mid
    
    t_event, x_event = t_left, x_left
    x_reset, next_mode, new_reset_arg = resetmap(x_event, current_mode, reset_arg)
    
    return t_event, x_event, x_reset, next_mode, new_reset_arg


# =============================================================================
#                         Event Detection (One Step)
# =============================================================================

def event_detect_onestep_discrete(x0, u, t0, dt, current_mode,
                                   smooth_dynamics, guards, gxs, gts,
                                   reset_maps, Rxs, Rts, reset_args,
                                   guard_cond_funcs, 
                                   detection=None, detect=None, backwards=False):
    """
    Detect events and compute saltation matrices for one discrete time step.
    
    Args:
        x0: Initial state
        u: Control input
        t0: Initial time
        dt: Time step
        current_mode: Current discrete mode
        smooth_dynamics: List of smooth dynamics functions per mode
        guards: List of guard functions per mode
        gxs: List of guard state Jacobians per mode
        gts: List of guard time derivatives per mode
        reset_maps: List of reset map functions per mode
        Rxs: List of reset map state Jacobians per mode
        Rts: List of reset map time derivatives per mode
        reset_args: Arguments for reset maps
        guard_cond_funcs: List of guard condition functions per mode
        detection: Whether to perform event detection
        detect: Whether to perform event detection (alias for detection)
        backwards: Whether to integrate backwards in time
    
    Returns:
        xt_next: Next state
        saltation: Saltation matrix (None if no event)
        mode_mapping: Array [current_mode, next_mode]
        t_event: Event time (None if no event)
        x_event: State at event (None if no event)
        x_reset: State after reset (None if no event)
        reset_byproduct: Additional reset outputs
    """
    # Handle both parameter names for backwards compatibility
    if detection is not None:
        do_detection = detection
    elif detect is not None:
        do_detection = detect
    else:
        do_detection = True  # Default
    
    current_dyn = smooth_dynamics[current_mode]
    
    # Initialize outputs
    x_event, t_event, x_reset, saltation = None, None, None, None
    next_mode = current_mode
    reset_byproduct = (None,)
    
    # Smooth dynamics integration
    direction = -1 if backwards else 1
    xt_next = x0 + direction * current_dyn(t0, x0, u) * dt
    
    # Event detection
    if do_detection:
        guard_hit = guard_cond_funcs[current_mode](x0, xt_next, current_mode)
        
        if guard_hit:
            # Find exact event location via bisection
            t_event, x_event, x_reset, next_mode, reset_byproduct = bisection_event_detection(
                x0, current_mode, u, t0, xt_next, dt,
                current_dyn, guards[current_mode], reset_maps[current_mode], reset_args
            )
            
            # Compute saltation matrix
            saltation = _compute_saltation_at_event(
                t_event, x_event, x_reset, current_mode, int(next_mode),
                current_dyn, smooth_dynamics[int(next_mode)],
                Rxs[current_mode], Rts[current_mode],
                gxs[current_mode], gts[current_mode],
                reset_args
            )
            xt_next = x_reset
            next_mode = int(next_mode)
    
    mode_mapping = np.array([current_mode, next_mode])
    return xt_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct


def _compute_saltation_at_event(t_event, x_event, x_reset, current_mode, next_mode,
                                 current_dyn, next_dyn, Rx_func, Rt_func,
                                 gx_func, gt_func, reset_args):
    """
    Compute saltation matrix at an event.
    
    The saltation matrix captures the sensitivity of the trajectory
    to perturbations across a discrete transition.
    """
    # Jacobians of reset map (Rx_func returns jacobian directly, not tuple)
    R_x = Rx_func(x_event, current_mode, reset_args)
    R_t = Rt_func(x_event, current_mode, reset_args)
    
    # Derivatives of guard function
    g_x = gx_func(t_event, x_event)
    g_t = gt_func(t_event, x_event)
    
    # Vector fields before and after event
    F_1 = current_dyn(t_event, x_event)
    F_2 = next_dyn(t_event, x_reset)  # F2 evaluated at reset state
    
    return compute_saltation(F_1, F_2, R_t, R_x, g_t, g_x)


# Alias for backwards compatibility
event_detect_onestep_discrete_resetargs = event_detect_onestep_discrete


# =============================================================================
#                         Stochastic Feedback Rollout
# =============================================================================

def h_stoch_fb_rollout(init_mode, x0, n_inputs, xt_ref, ref_modes,
                        ut, Kt, kt, target_state, Q_T, t0, dt,
                        epsilon, GaussianNoise, ref_ext_helper, init_reset_args,
                        cond_mismatch_func=None,
                        reaction_mismatch_func=None,
                        h_stoch_integr_func=None):
    """
    Perform stochastic feedback rollout for hybrid systems.
    
    Args:
        init_mode: Initial discrete mode
        x0: Initial state
        n_inputs: List of input dimensions per mode
        xt_ref: Reference state trajectory
        ref_modes: Reference mode trajectory
        ut: Feedforward control inputs per mode
        Kt: Feedback gain matrices
        kt: Feedforward corrections (optional)
        target_state: Target state for terminal cost
        Q_T: Terminal cost weight matrix
        t0: Initial time
        dt: Time step
        epsilon: Noise scaling parameter
        GaussianNoise: Noise samples per mode
        ref_ext_helper: Reference extension helper data
        init_reset_args: Initial reset arguments
        cond_mismatch_func: Function to check mode mismatch
        reaction_mismatch_func: Function to handle mode mismatch
        h_stoch_integr_func: Stochastic integration function
    
    Returns:
        mode_trj: Mode trajectory
        xt_trj: State trajectory
        ut_cl_trj: Closed-loop control trajectory
        Sk: Path cost
        xt_ref_actual: Actual reference trajectory used
        reset_args: Reset arguments trajectory
    """
    # Extract reference extensions
    (v_event_modechange, v_ext_bwd, v_ext_fwd,
     v_Kfb_ext_bwd, v_Kfb_ext_fwd,
     v_kff_ext_bwd, v_kff_ext_fwd, _) = extract_extensions(ref_ext_helper, start_index=0)
    
    n_timestamps = len(xt_ref)
    
    # Initialize trajectories
    xt_trj = [np.array([0.0]) for _ in range(n_timestamps)]
    xt_trj[0] = x0
    
    mode_trj = np.zeros(n_timestamps, dtype=np.int64)
    mode_trj[0] = init_mode
    
    ut_cl_trj = [np.zeros((n_timestamps, n_inputs[mode])) for mode in range(len(n_inputs))]
    xt_ref_actual = [np.array([0.0]) for _ in range(n_timestamps)]
    reset_args = [np.array([0.0]) for _ in range(n_timestamps)]
    
    # Tracking variables
    cnt_mismatch = 0
    cnt_event = 0
    event_args = [init_reset_args[0]]
    Sk = 0
    
    # Rollout loop
    for ii_t in range(n_timestamps - 1):
        xt = xt_trj[ii_t]
        current_mode = mode_trj[ii_t]
        ref_current_mode = ref_modes[ii_t]
        
        # Get feedback gains and reference
        K_fb_i = Kt[ii_t]
        k_ff_i = kt[ii_t] if kt is not None else None
        xref_i = xt_ref[ii_t]
        reset_args[ii_t] = event_args[cnt_event]
        
        # Handle mode mismatch
        if cond_mismatch_func(current_mode, ref_current_mode):
            print(f"Mode mismatch at time: {ii_t}")
            xref_i, K_fb_i, k_ff_i, cnt_mismatch = reaction_mismatch_func(
                ii_t, current_mode, ref_current_mode,
                v_ext_fwd[0], v_ext_bwd[0], v_event_modechange[0],
                v_Kfb_ext_fwd[0], v_kff_ext_fwd[0],
                v_Kfb_ext_bwd[0], v_kff_ext_bwd[0],
                cnt_mismatch
            )
        
        xt_ref_actual[ii_t] = xref_i
        
        # Compute control input
        delta_xt = xt - xref_i
        current_u = ut[current_mode][ii_t] + K_fb_i @ delta_xt
        if k_ff_i is not None:
            current_u += k_ff_i
        
        ut_cl_trj[current_mode][ii_t] = current_u
        
        # Stochastic integration step
        noise_i = GaussianNoise[current_mode][ii_t]
        xt_next, next_mode, _, new_reset_arg = h_stoch_integr_func(
            xt, current_mode, current_u, noise_i, epsilon, dt, t0, reset_args[ii_t]
        )
        
        # Update trajectories
        reset_args[ii_t + 1] = new_reset_arg
        xt_trj[ii_t + 1] = xt_next
        mode_trj[ii_t + 1] = next_mode
    
    xt_ref_actual[-1] = xt_ref[-1]
    
    return mode_trj, xt_trj, ut_cl_trj, Sk, xt_ref_actual, reset_args