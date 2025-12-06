## 1 dimensional bouncing ball dynamics
# x = [z, \dot z]
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
from jax import jacfwd 
jax.config.update('jax_enable_x64', True)

m = 1

def linearize_bouncing(x):
    """
    Linearization.
    Args:
        x: state

    Returns:
        A, B: linear system matrices.
    """
    
    return np.array([[0, 1.0], [0.0, 0.0]], dtype=np.float64), np.array([0, 1.0/m])

@jax.jit
def guard_bouncing_12(t, x):
    return x[0]

@jax.jit
def guard_bouncing_21(t, x):
    return x[1]

def reset_map_bouncing_21(x_minus, current_mode, args_reset):
    x_plus = x_minus
    new_mode = 0
    return x_plus, new_mode, args_reset

@jax.jit
def idendity_map_bouncing(x_event):
    return x_event

@jax.jit
def reset_map_bouncing_21_jax(x_minus, current_mode, args_reset):
    x_plus = x_minus
    new_mode = 0
    return x_plus, new_mode, args_reset

@jax.jit
def impact_map_bouncing(x_event):
    return jnp.array([x_event[0], -0.6*x_event[1]], dtype=jnp.float64)

@jax.jit
def reset_map_bouncing_12_jax(x_minus, current_mode, args_reset):
    # coeff = jnp.array([[1.0, 0.0], [0.0, -0.6]], dtype=jnp.float64)
    x_plus = impact_map_bouncing(x_minus)
    new_mode = 1
    return x_plus, new_mode, args_reset

@jax.jit
def Rt_bouncing_12(x):
    return 0.0
Rx_bouncing_12 = jax.jacrev(impact_map_bouncing)

@jax.jit
def reset_map_bouncing_12(x_minus, current_mode, args_reset):
    x_plus = jnp.array([x_minus[0], -0.6*x_minus[1]], dtype=jnp.float64)
    new_mode = 1

    return x_plus, new_mode, args_reset


def Rt_bouncing_21(x):
    return 0.0
Rx_bouncing_21 = jax.jacrev(idendity_map_bouncing)

def gt_bouncing_12(t, x):
    return 0.0
gx_bouncing_12 = jax.jacrev(guard_bouncing_12, argnums=1)

def gt_bouncing_21(t, x):
    return 0.0
gx_bouncing_21 = jax.jacrev(guard_bouncing_21, argnums=1)

guard_bouncing_12.terminal=True
guard_bouncing_12.direction=-1

guard_bouncing_21.terminal=True
guard_bouncing_21.direction=1