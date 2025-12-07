"""
SLIP (Spring-Loaded Inverted Pendulum) guard and reset map definitions.

Mode 0 (flight): x = [px, vx, pz, vz, theta]
Mode 1 (stance): x = [theta, theta_dot, r, r_dot]

Transitions:
- 12: Flight -> Stance (touchdown)
- 21: Stance -> Flight (liftoff)
"""

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import numpy as np
import jax
import jax.numpy as jnp
from jax import grad

from dynamics.saltation_matrix import *

# =============================================================================
#                              Constants
# =============================================================================

R0 = 1.0  # Rest length of spring


# =============================================================================
#                         Mode Change Helper
# =============================================================================

def mode_change_maps(current_mode):
    """Toggle between modes 0 and 1."""
    return 1 - current_mode


# =============================================================================
#                    Flight -> Stance (12) Transition
# =============================================================================

@jax.jit
def guard_slip_12(t, x):
    """
    Guard function for flight -> stance transition.
    Triggers when foot touches ground: r0*sin(theta) - z = 0
    
    Args:
        t: Time (unused, for interface compatibility)
        x: Flight state [px, vx, pz, vz, theta]
    
    Returns:
        Guard value (negative in flight, positive when ground contact)
    """
    return R0 * jnp.sin(x[4]) - x[2]


@jax.jit
def reset_map_slip_12(x_event, current_mode, args_reset):
    """
    Reset map for flight -> stance transition.
    
    Args:
        x_event: Flight state at event [px, vx, pz, vz, theta]
        current_mode: Current mode (should be 0)
        args_reset: Reset arguments (unused for this transition)
    
    Returns:
        x_reset: Stance state [theta, theta_dot, r, r_dot]
        new_mode: 1 (stance)
        new_args: Tuple containing foot contact x-position
    """
    px, vx, pz, vz, theta = x_event
    
    r = R0
    r_dot = -vx * jnp.cos(theta) + vz * jnp.sin(theta)
    theta_dot = (px * vz - pz * vx) / (r * r)
    
    x_reset = jnp.array([theta, theta_dot, r, r_dot])
    foot_x = px - jnp.cos(theta)  # Foot contact position
    
    return x_reset, 1, (foot_x,)


@jax.jit
def reset_map_slip_12_padding(t, x_event, current_mode, args_reset):
    """
    Reset map with zero-padded output for uniform state dimension.
    
    Returns 5D state with last element zeroed for compatibility.
    """
    px, vx, pz, vz, theta = x_event
    
    r = R0
    r_dot = -vx * jnp.cos(theta) + vz * jnp.sin(theta)
    theta_dot = (px * vz - pz * vx) / (r * r)
    
    x_reset = jnp.array([theta, theta_dot, r, r_dot, 0.0])
    
    return x_reset, 1, x_event[0]


# =============================================================================
#                    Stance -> Flight (21) Transition
# =============================================================================

@jax.jit
def guard_slip_21(t, x):
    """
    Guard function for stance -> flight transition.
    Triggers when spring reaches rest length: r - r0 = 0
    
    Args:
        t: Time (unused, for interface compatibility)
        x: Stance state [theta, theta_dot, r, r_dot]
    
    Returns:
        Guard value (negative when compressed, positive at liftoff)
    """
    return x[2] - R0


@jax.jit
def reset_map_slip_21(x_event, current_mode, args_reset):
    """
    Reset map for stance -> flight transition.
    
    Args:
        x_event: Stance state at event [theta, theta_dot, r, r_dot]
        current_mode: Current mode (should be 1)
        args_reset: Reset arguments containing (foot_x_position,)
    
    Returns:
        x_reset: Flight state [px, vx, pz, vz, theta]
        new_mode: 0 (flight)
        new_args: Pass through args_reset
    """
    xp = args_reset[0]  # Foot contact position
    theta, theta_dot, r, r_dot = x_event
    
    px = xp + R0 * jnp.cos(theta)
    vx = r_dot * jnp.cos(theta) - r * theta_dot * jnp.sin(theta)
    pz = R0 * jnp.sin(theta)
    vz = R0 * theta_dot * jnp.cos(theta) + r_dot * jnp.sin(theta)
    
    x_reset = jnp.array([px, vx, pz, vz, theta])
    
    return x_reset, 0, args_reset


# =============================================================================
#                    State Conversion Utilities
# =============================================================================

def convert_state_21_slip(state_2, foot_contact_pos=0.0):
    """
    Convert stance state (polar) to flight state (Cartesian).
    
    Args:
        state_2: Stance state [theta, theta_dot, r, r_dot]
        foot_contact_pos: X-coordinate of foot contact point
    
    Returns:
        Cartesian state [px, vx, pz, vz, theta]
    """
    polar_traj = np.array(state_2)
    assert polar_traj.shape[0] == 4, f'Expected (4,) polar state, got {polar_traj.shape}'
    
    if polar_traj.ndim == 1:
        polar_traj = np.expand_dims(polar_traj, axis=1)

    theta, theta_dot, r, r_dot = polar_traj[0], polar_traj[1], polar_traj[2], polar_traj[3]

    x = np.cos(theta) * r + foot_contact_pos
    x_dot = -r * theta_dot * np.sin(theta) + np.cos(theta) * r_dot
    z = np.sin(theta) * r
    z_dot = np.cos(theta) * theta_dot * r + np.sin(theta) * r_dot

    return np.array([x, x_dot, z, z_dot, theta])


# =============================================================================
#                    Jacobians and Derivatives
# =============================================================================

# Time derivatives of reset maps (both are time-independent)
def Rt_slip_12(x, current_mode, args):
    """Time derivative of reset map 12 (zero - time independent)."""
    return 0.0


def Rt_slip_21(x, current_mode, args):
    """Time derivative of reset map 21 (zero - time independent)."""
    return 0.0


# State Jacobians of reset maps
# NOTE: We differentiate only the state output [0], not the mode or args
Rx_slip_12 = jax.jacrev(
    lambda x, current_mode, args: reset_map_slip_12(x, current_mode, args)[0], 
    argnums=0
)

Rx_slip_21 = jax.jacrev(
    lambda x, current_mode, args: reset_map_slip_21(x, current_mode, args)[0], 
    argnums=0
)


# Guard derivatives
gt_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), argnums=0))
gx_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), argnums=1))

gt_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), argnums=0))
gx_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), argnums=1))


# =============================================================================
#                    Guard Attributes (for ODE solvers)
# =============================================================================

guard_slip_12.terminal = True
guard_slip_12.direction = -1

guard_slip_21.terminal = True
guard_slip_21.direction = 1