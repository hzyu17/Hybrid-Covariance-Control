"""
SLIP (Spring-Loaded Inverted Pendulum) discrete dynamics.

This module implements the hybrid dynamics for a SLIP model with:
- Mode 0: Flight phase (5 states: x, x_dot, z, z_dot, theta)
- Mode 1: Stance phase (4 states: theta, theta_dot, r, r_dot)
"""

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import numpy as np
from functools import partial
from dynamics.ode_solver.dynamics_slip import *
from dynamics.dynamics_discrete import (
    h_stoch_integr, 
    h_stoch_fb_rollout, 
    event_detect_onestep_discrete
)

# =============================================================================
#                              SLIP Parameters
# =============================================================================

SLIP_PARAMS = {
    'g': 9.81,      # Gravity
    'k': 25.0,      # Spring constant
    'm': 0.5,       # Mass
    'r0': 1.0       # Rest length of spring
}

g, k, m, r0 = SLIP_PARAMS['g'], SLIP_PARAMS['k'], SLIP_PARAMS['m'], SLIP_PARAMS['r0']


# =============================================================================
#                         Stochastic Integration
# =============================================================================

def stoch_integr_slip(mode, x0, u, dt, eps, dW):
    """
    Stochastic integration for SLIP dynamics.
    
    Args:
        mode: Current mode (0=flight, 1=stance)
        x0: Current state
        u: Control input
        dt: Time step
        eps: Noise scaling
        dW: Wiener process increment
    
    Returns:
        Next state after integration
    """
    if mode == 0:
        return _flight_dynamics(x0, u, dt, eps, dW)
    else:
        return _stance_dynamics(x0, u, dt, eps, dW)


def _flight_dynamics(x0, u, dt, eps, dW):
    """Flight mode dynamics: states = [x, x_dot, z, z_dot, theta]"""
    B = np.array([
        [0.0, 0.0, 0.0],
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float64)
    
    f = np.array([x0[1], u[0], x0[3], u[1] - g, u[2]], dtype=np.float64)
    return x0 + f * dt + np.sqrt(eps) * B @ dW


def _stance_dynamics(x0, u, dt, eps, dW):
    """Stance mode dynamics: states = [theta, theta_dot, r, r_dot]"""
    theta, theta_dot, r, r_dot = x0
    
    B = np.array([
        [0.0, 0.0],
        [0.0, 0.0],
        [0.0, 1/(m*r*r)],
        [k/m, 0.0]
    ], dtype=np.float64)
    
    f = np.array([
        theta_dot,
        -2*theta_dot*r_dot/r - g*np.cos(theta)/r,
        r_dot + u[1]/(m*r*r),
        k/m*(r0-r) - g*np.sin(theta) + theta_dot**2 * r + k*u[0]/m
    ], dtype=np.float64)
    
    return x0 + f * dt + np.sqrt(eps) * B @ dW


# =============================================================================
#                         Guard Conditions
# =============================================================================

def guard_cond_slip_12(xt, xt_next, current_mode):
    """Guard for flight -> stance transition."""
    return (current_mode == 0 and 
            guard_slip_12(0.0, xt) < 0 and 
            guard_slip_12(0.0, xt_next) > 0)


def guard_cond_slip_21(xt, xt_next, current_mode):
    """Guard for stance -> flight transition."""
    return (current_mode == 1 and 
            guard_slip_21(0.0, xt) <= 0 and 
            guard_slip_21(0.0, xt_next) > 0)


# =============================================================================
#                         Guard Event Handlers
# =============================================================================

def _bisection_find_event(xt_current, current_mode, u, t, dt_int, eps, rand_noise,
                          guard_func, reset_func, reset_arg, tol=1e-10):
    """
    Find exact event time using bisection method.
    
    Returns:
        xt_next: State after reset
        next_mode: Mode after transition
        dW_new: Adjusted Wiener increment
        new_reset_arg: Updated reset arguments
    """
    t_left, x_left = t, xt_current
    t_right = t + dt_int
    
    while (t_right - t_left) / 2.0 >= tol:
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_mid - t_left
        dW_new = np.sqrt(dt_new) * rand_noise
        
        x_mid = stoch_integr_slip(current_mode, x_left, u, dt_new, eps, dW_new)
        
        guard_left = guard_func(t_left, x_left)
        guard_mid = guard_func(t_mid, x_mid)
        
        if guard_mid == 0:
            break
        elif guard_left * guard_mid < 0:
            t_right = t_mid
        else:
            t_left, x_left = t_mid, x_mid
    
    # Reset function takes (x_event, current_mode, args) - no time argument
    xt_next, next_mode, new_reset_arg = reset_func(x_left, current_mode, reset_arg)
    dt_final = t_left - t
    dW_new = np.sqrt(dt_final) * rand_noise
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_true_slip_12(args):
    """Handle flight -> stance transition."""
    print("slip_cond_12: True")
    xt_current, current_mode, u, t, _, dt_int, rand_noise, eps, reset_arg = args
    return _bisection_find_event(
        xt_current, current_mode, u, t, dt_int, eps, rand_noise,
        guard_slip_12, reset_map_slip_12, reset_arg
    )


def guard_true_slip_21(args):
    """Handle stance -> flight transition."""
    xt_current, current_mode, u, t, _, dt_int, rand_noise, eps, reset_arg = args
    return _bisection_find_event(
        xt_current, current_mode, u, t, dt_int, eps, rand_noise,
        guard_slip_21, reset_map_slip_21, reset_arg
    )


def guard_false_slip(args):
    """No guard triggered - continue with current state."""
    _, current_mode, _, _, xt_next, dt_int, rand_noise, _, reset_arg = args
    dW = np.sqrt(dt_int) * rand_noise
    return xt_next, current_mode, dW, reset_arg


# =============================================================================
#                         Partial Function Definitions
# =============================================================================

# Guard functions organized by mode
SLIP_GUARDS = {0: guard_cond_slip_12, 1: guard_cond_slip_21}
SLIP_GUARD_TRUE_FUNCS = {0: guard_true_slip_12, 1: guard_true_slip_21}
SLIP_GUARD_FALSE_FUNCS = {0: guard_false_slip, 1: guard_false_slip}

# Stochastic integration with SLIP-specific guards
h_stoch_integr_slip = partial(
    h_stoch_integr,
    stoch_integr_func=stoch_integr_slip,
    guards=SLIP_GUARDS,
    guard_true_funcs=SLIP_GUARD_TRUE_FUNCS,
    guard_false_funcs=SLIP_GUARD_FALSE_FUNCS
)

# Mode mismatch handling
reaction_mode_mismatch_slip = partial(
    reaction_mode_mismatch,
    cond_early_arrival=cond_early_arrival_slip
)

# Stochastic feedback rollout for SLIP
h_stoch_fb_rollout_slip = partial(
    h_stoch_fb_rollout,
    cond_mismatch_func=cond_mode_mismatch_slip,
    reaction_mismatch_func=reaction_mode_mismatch_slip,
    h_stoch_integr_func=h_stoch_integr_slip
)


# =============================================================================
#                         Event Detection
# =============================================================================

# SLIP-specific dynamics and maps organized by mode
SLIP_SMOOTH_DYNAMICS = {0: dyn_flight_slip, 1: dyn_stance_slip}
SLIP_GUARDS_MAP = {0: guard_slip_12, 1: guard_slip_21}
SLIP_RESET_MAPS = {0: reset_map_slip_12, 1: reset_map_slip_21}
SLIP_RXS = {0: Rx_slip_12, 1: Rx_slip_21}
SLIP_RTS = {0: Rt_slip_12, 1: Rt_slip_21}
SLIP_GXS = {0: gx_slip_12, 1: gx_slip_21}
SLIP_GTS = {0: gt_slip_12, 1: gt_slip_21}
SLIP_GUARD_CONDS = {0: guard_cond_slip_12, 1: guard_cond_slip_21}


def event_detect_discrete_slip(current_mode, x0, u, t0, dt, reset_args,
                                detect=None, detection=None, backwards=False):
    """
    Detect events for SLIP dynamics.
    
    Args:
        current_mode: Current discrete mode (0=flight, 1=stance)
        x0: Current state
        u: Control input
        t0: Current time
        dt: Time step
        reset_args: Reset map arguments
        detect: Whether to perform event detection (alias for detection)
        detection: Whether to perform event detection
        backwards: Whether to integrate backwards
    
    Returns:
        Tuple of (xt_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct)
    """
    return event_detect_onestep_discrete(
        x0, u, t0, dt, current_mode,
        SLIP_SMOOTH_DYNAMICS,
        SLIP_GUARDS_MAP,
        SLIP_GXS,
        SLIP_GTS,
        SLIP_RESET_MAPS,
        SLIP_RXS,
        SLIP_RTS,
        reset_args,
        SLIP_GUARD_CONDS,
        detection=detection,
        detect=detect,
        backwards=backwards
    )