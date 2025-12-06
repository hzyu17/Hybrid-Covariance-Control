## 2-dimensional SLIP dynamics
# mode 1: x = [px, vx, pz, vz, theta], u = [theta_dot]
# mode 2: x = [theta, theta_dot, r, r_dot], u = [r_delta, \tau_hip]
# reset maps: identity

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))

sys.path.append(root_dir)

from dynamics.saltation_matrix import *
import numpy as np
import jax.numpy as jnp

import jax
from jax import grad 


def mode_change_maps(current_mode):
    new_mode = current_mode
    if (current_mode == 0):
        new_mode = 1
    elif (current_mode == 1):
        new_mode = 0
    return new_mode

# =========================================================
# The guard and reset map from flight mode to stance mode
# =========================================================

@jax.jit
def guard_slip_12(t, x):
    r0 = 1
    return  r0*jnp.sin(x[4]) - x[2]

# reset map from flight mode to stance mode
@jax.jit
def reset_map_slip_12(x_event, current_mode, args_reset):
    # x_event: [x, x_dot, z, z_dot, theta]
    
    r0 = 1
    x, xdot, z, zdot, theta = x_event
    
    x_reset = x_event
    args = (x_event, current_mode)
    
    stance_cond = jax.numpy.logical_and(z-r0*jnp.sin(theta) < 0, current_mode==0)
    
    def stance_true_fun(args):
        r = r0
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        # theta_dot = -(xdot * jnp.cos(theta) + zdot * jnp.sin(theta)) / r0
        theta_dot = (x*zdot - z*xdot) / r / r
        x_reset = jnp.array([theta, theta_dot, r, r_dot])
        return x_reset # We need to record the position x at the impact time
    
    def stance_false_fun(args):
        r = r0
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        theta_dot = (x*zdot - z*xdot) / r / r
        # theta_dot = -(xdot * jnp.cos(theta) + zdot * jnp.sin(theta)) / r0
        x_reset = jnp.array([theta, theta_dot, r, r_dot])
        return x_reset
    
    args = (x_event, current_mode)
    x_reset = jax.lax.cond(stance_cond, stance_true_fun, stance_false_fun, args)
    
    return x_reset, 1, (x_event[0]-jnp.cos(x_event[4]), )


# reset map from flight mode to stance mode
@jax.jit
def reset_map_slip_12_padding(t, x_event, current_mode, args_reset):
    # x_event: [x, x_dot, z, z_dot, theta]
    
    r0 = 1
    new_mode = current_mode
    x, xdot, z, zdot, theta = x_event
    
    x_reset = x_event
    args = (x_event, current_mode)
    
    stance_cond = jax.numpy.logical_and(z-r0*jnp.sin(theta) < 0, current_mode==0)
    
    def stance_true_fun(args):
        r = r0
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        # theta_dot = -(xdot * jnp.cos(theta) + zdot * jnp.sin(theta)) / r0
        theta_dot = (x*zdot - z*xdot) / r / r
        x_reset = jnp.array([theta, theta_dot, r, r_dot, 0.0])
        return x_reset # We need to record the position x at the impact time
    
    def stance_false_fun(args):
        r = r0
        r_dot = -xdot*jnp.cos(theta) + zdot*jnp.sin(theta)
        theta_dot = (x*zdot - z*xdot) / r / r
        # theta_dot = -(xdot * jnp.cos(theta) + zdot * jnp.sin(theta)) / r0
        x_reset = jnp.array([theta, theta_dot, r, r_dot, 0.0])
        return x_reset
    
    args = (x_event, current_mode)
    x_reset = jax.lax.cond(stance_cond, stance_true_fun, stance_false_fun, args)
    
    return x_reset, 1, x_event[0]

# =========================================================
# The guard and reset map from stance mode to flight mode
# =========================================================
# xs: [theta, theta_dot, r, r_dot]
# guard_21 = r - r0
def guard_slip_21(t, x):
    r0 = 1
    return x[2] - r0


# reset map from stance mode to flight mode
def reset_map_slip_21(x_event, current_mode, args_reset):
    xp = args_reset[0]
    # xp = 0.0
    r0 = 1
    # x_reset = x_event
    theta, theta_dot, r, r_dot = x_event
    
    px_reset = xp + r0*jnp.cos(theta)
    vx_reset = r_dot*jnp.cos(theta) - r*theta_dot*jnp.sin(theta)
    pz_reset = r0*jnp.sin(theta)
    vz_reset = r0*theta_dot*jnp.cos(theta) + r_dot*jnp.sin(theta)
    theta_reset = theta

    x_reset = jnp.array([px_reset, vx_reset, pz_reset, vz_reset, theta_reset])
    
    return x_reset, 0, args_reset


# --------------------------------------------------------------
# Conversion between the two state space (from stance to flight)
# --------------------------------------------------------------
def convert_state_21_slip(state_2, foot_contact_pos=0.0):
    """
    Utility function to convert an actuated SLIP CoM trajectory in polar coordinates to cartesian.
    This function assumes the extended SLIP model presented in "Optimal Control of a Differentially Flat
    Two-Dimensional Spring-Loaded Inverted Pendulum Model".
    :param trajectory: (4, k) Polar trajectory of the SLIP model during stance phase
    :param control_signal: (2, k) Optional leg length displacement and hip torque of exerted at every timestep during the
    stance phase.
    :param foot_contact_pos: Cartesian x coordinate of the foot contact point during the stance phase of the input
    trajectory.
    :return:
    """
    polar_traj = np.array(state_2)
    assert polar_traj.shape[0] == 4, 'Provide a valid (4, k) polar trajectory: %s' % polar_traj.shape
    if polar_traj.ndim == 1:
        polar_traj = np.expand_dims(polar_traj, axis=1)

    theta, theta_dot, r, r_dot = polar_traj[0], polar_traj[1], polar_traj[2], polar_traj[3]

    x = np.cos(theta) * r
    x_dot = -r * theta_dot * np.sin(theta) + np.cos(theta) * r_dot
    z = np.sin(theta) * r
    z_dot = np.cos(theta) * theta_dot * r + np.sin(theta) * r_dot

    # Center x dimension
    # print("foot_contact_pos: ", foot_contact_pos)
    x += foot_contact_pos

    cartesian_state = np.array([x, x_dot, z, z_dot, theta])
    return cartesian_state


# Define derivatives
def Rt_slip_12(x, current_mode, args):
    return 0.0

Rx_slip_12 = jax.jacrev(lambda x, current_mode, args: reset_map_slip_12(x, current_mode, args), 0)

def Rt_slip_21(x, current_mode, args):
    return 0.0

Rx_slip_21 = jax.jacrev(lambda x, current_mode, args: reset_map_slip_21(x, current_mode, args), 0)

gt_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), 0))
    
gx_slip_12 = jax.jit(grad(lambda t, x: guard_slip_12(t, x), 1))

gx_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), 1))
    
gt_slip_21 = jax.jit(grad(lambda t, x: guard_slip_21(t, x), 0))


guard_slip_12.terminal=True
guard_slip_12.direction=-1

guard_slip_21.terminal=True
guard_slip_21.direction=1
  