import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import numpy as np
from dynamics.ode_solver.dynamics_bouncing import *
from dynamics.dynamics_discrete import *


def stoch_integr_bouncing(mode, x0, u, dt, eps, dW):
    B = np.array([[0],[1.0]], dtype=np.float64)
    xt_next = x0 + np.array([x0[1], u[0]-9.81], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
    return xt_next


# ===============================================================================================================
#                                        Bouncing Guard condition handling 
# ===============================================================================================================
# -------------------------------- 
#      From mode 1 to mode 2 
# --------------------------------
def guard_cond_bouncing_12(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==0) and ((guard_bouncing_12(0.0,xt)>0) and (guard_bouncing_12(0.0,xt_next)<=0))

def guard_true_func_bouncing_12(args):
    (xt_current, current_mode, u_current, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_loop_body(xt, u, dt, dt_shrinkrate, RandN, eps, cnt_shrink):
        # Too far from the guard, shrink the step size
        dt_shrinked = dt * dt_shrinkrate
        dW_shrink = np.sqrt(dt_shrinked) * RandN
        
        xt_shrinked = stoch_integr_bouncing(current_mode, xt, u, dt_shrinked, eps, dW_shrink)
        
        new_condition = ((guard_cond_bouncing_12(xt, xt_shrinked, current_mode)) and (cnt_shrink < 20))
        cnt_shrink += 1
        
        return xt_shrinked, dt_shrinked, cnt_shrink, new_condition
    
    cnt_shrink = 0
    can_continue = True
    xt_shrinked = xt_next
    
    # Implementing the loop with Python's while
    while can_continue:
        xt_shrinked, dt_int, cnt_shrink, can_continue = while_loop_body(
            xt_current, u_current, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink
        )
    
    # Execute the reset map after the loop
    xt_next, next_mode, new_reset_arg = reset_map_bouncing_12(t, xt_shrinked, current_mode, reset_arg)
    dW_new = np.sqrt(dt_int) * RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_func_bouncing_12(args):
    (_, current_mode, _, _, xt_next, dt_int, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg

# --------------------------------
#       From mode 2 to mode 1 
# --------------------------------
def guard_cond_bouncing_21(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==1) and ((guard_bouncing_21(0.0,xt)>0) and (guard_bouncing_21(0.0,xt_next)<=0))

def guard_true_func_bouncing_21(args):
    (xt_current, current_mode, u_current, t, xt_next, dt_int, dt_shrinkrate, RandN, eps, reset_arg) = args
    
    def while_loop_body(xt, u, dt, dt_shrinkrate, RandN, eps, cnt_shrink):
        # Too far from the guard, shrink the step size
        dt_shrinked = dt * dt_shrinkrate
        dW_shrink = np.sqrt(dt_shrinked) * RandN
        
        xt_shrinked = stoch_integr_bouncing(current_mode, xt, u, dt_shrinked, eps, dW_shrink)
        
        new_condition = ((guard_cond_bouncing_21(xt, xt_shrinked, current_mode)) and (cnt_shrink < 20))
        cnt_shrink += 1
        
        return xt_shrinked, dt_shrinked, cnt_shrink, new_condition
    
    cnt_shrink = 0
    can_continue = True
    xt_shrinked = xt_next
    
    # Implementing the loop with Python's while
    while can_continue:
        xt_shrinked, dt_int, cnt_shrink, can_continue = while_loop_body(
            xt_current, u_current, dt_int, dt_shrinkrate, RandN, eps, cnt_shrink
        )
    
    # Execute the reset map after the loop
    xt_next, next_mode, new_reset_arg = reset_map_bouncing_21(t, xt_shrinked, current_mode, reset_arg)
    dW_new = np.sqrt(dt_int) * RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_func_bouncing_21(args):
    (_, current_mode, _, _, xt_next, dt_int, _, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg


# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

from functools import partial

reaction_mode_mismatch_bouncing = partial(reaction_mode_mismatch, cond_early_arrival=cond_early_arrival_bouncing)
h_stoch_integr_bouncing = partial(h_stoch_integr, 
                                    stoch_integr_func = stoch_integr_bouncing, 
                                    guard_0=guard_cond_bouncing_12,
                                    guard_true_func_0=guard_true_func_bouncing_12,
                                    guard_false_func_0=guard_false_func_bouncing_12,
                                    guard_1=guard_cond_bouncing_21,
                                    guard_true_func_1=guard_true_func_bouncing_21,
                                    guard_false_func_1=guard_false_func_bouncing_21)


h_stoch_fb_rollout_bouncing = partial(h_stoch_fb_rollout, 
                                        cond_mismatch_func=cond_mode_mismatch_bouncing,
                                        reaction_mismatch_func=reaction_mode_mismatch_bouncing,
                                        h_stoch_integr_func=h_stoch_integr_bouncing)    


def event_detect_bouncing_discrete(current_mode, 
                                   x0, u, 
                                   t0, dt, 
                                   reset_args, 
                                   detect=True, 
                                   backwards=False):
    
    smooth_dynamics_bouncing = {0:dyn_bouncing, 1:dyn_bouncing}
    
    Rxs_bouncing = {0:Rx_bouncing_12, 1:Rx_bouncing_21}
    Rts_bouncing = {0:Rt_bouncing_12, 1:Rt_bouncing_21}
    
    gxs_bouncing = {0:gx_bouncing_12, 1:gx_bouncing_21}
    gts_bouncing = {0:gt_bouncing_12, 1:gt_bouncing_21}
    
    guards_bouncing_bouncing = {0:guard_bouncing_12, 1: guard_bouncing_21}
    reset_maps_bouncing_bouncing = {0:reset_map_bouncing_12, 1:reset_map_bouncing_21}
    
    return event_detect_onestep_discrete(x0, u, t0, 
                                        dt, 
                                        current_mode, 
                                        smooth_dynamics_bouncing, 
                                        guards_bouncing_bouncing,
                                        gxs_bouncing,
                                        gts_bouncing,
                                        reset_maps_bouncing_bouncing,
                                        Rxs_bouncing, Rts_bouncing,
                                        reset_args, 
                                        guard_cond_bouncing_12,
                                        guard_cond_bouncing_21,
                                        detect, backwards)
    
def bouncingball_cost(x, u, args= 0.0):
    return u.T@u/2

@jax.jit
def deltx_norm_cost(x, x_tar):
    nx = x.shape[0]
    Q_T = 60.0*np.eye(nx)
    return (x-x_tar).T@Q_T@(x-x_tar)/2.0