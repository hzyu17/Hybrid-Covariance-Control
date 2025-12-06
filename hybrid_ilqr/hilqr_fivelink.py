# Allows for general nonlinear cost functions and 
# jacobian computations using jax automatic differentiations 
# Only considering 1 same smooth flow (stance mode dynamics) in all the modes.

# For walking robot, we assume 2 modes: divided by the swing foot height and velocity sign.

import jax
import numpy as np
import matplotlib.pyplot as plt

import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.trajectory_extension import *
from five_link.fivelink_simulation import *
from hybrid_ilqr.h_ilqr_jax import *
from experiments.exp_params import *
from five_link.cost_functions import *


if __name__ == '__main__':
    # Both modes share the same dynamics
    n_states = [14, 14]
    n_inputs = [4, 4]

    # ---------------------------- 
    # Define weighting matrices
    # ----------------------------
    Q_k = [np.eye(n_states[0]), np.eye(n_states[1])] 
    R_k = [np.eye(n_inputs[0]), np.eye(n_inputs[1])]

    # ---------------------------- Set the terminal cost ----------------------------
    Q_T = 200*np.eye(n_states[0])
    # Q_T[0,0] = 2000.0

    n_exp = 1
    n_samples = 10
    
    nt = 300
    
    time_span = np.linspace(0.0, 0.24, 240)
    
    # dt_trj = np.ones(nt)*0.001
    
    # nu = 4
    u_trj = np.zeros((nt, n_inputs[0]))
    u_trj[:30, 0] = 11 # u_1R
    u_trj[:30, 1] = -2 # u_2R
    u_trj[:30, 2] = -11 # u_1L
    u_trj[:30, 3] = -2.5 # u_2L

    u_trj[30:100, 0] = -8 # u_1R
    u_trj[30:100, 1] = 1 # u_2R
    u_trj[30:100, 2] = 8 # u_1L
    u_trj[30:100, 3] = 1.5 # u_2L
    
    u_trj[100:135, 0] = 13.5 # u_1R
    u_trj[100:135, 1] = -2 # u_2R
    u_trj[100:135, 2] = -11 # u_1L
    u_trj[100:135, 3] = -3.5 # u_2L
    
    u_trj[135:, 0] = -8 # u_1R
    u_trj[135:, 1] = 1 # u_2R
    u_trj[135:, 2] = 8 # u_1L
    u_trj[135:, 3] = 2.5 # u_2L
    
    initial_guess = [u_trj, u_trj]
    
    niters = 10
    target_com_vel = 2.0
    
    q_init = jnp.array([0, 0.658, 0, -0.6828+jnp.pi, 1.20, -0.6489+jnp.pi, 1.281])    
    qdot_init = jnp.zeros(7)
    x_init = jnp.concatenate([q_init, qdot_init])
    
    from five_link.fivelink_simulation import draw_5link
    fig, ax = plt.subplots()
    
    draw_5link(q_init, ax, legend=True)
    
    plt.show()
    
    target_state = x_init
    init_mode = 1
    
    hilqr_obj = hybrid_ilqr_jax(n_states, n_inputs, 
                                init_mode, x_init, target_state, 
                                initial_guess, 
                                time_span, 
                                niters, 
                                is_detect=True, 
                                detect_func=detect_fivelink, 
                                smooth_dynamics=f_euler_fivelink, 
                                running_cost=com_moving_cost, 
                                cost_args=target_com_vel,
                                terminal_cost=deltx_norm_cost_fivelink, 
                                terminal_cost_args=target_state)
    
    hybrid_ilqr_result = hilqr_obj.solve()