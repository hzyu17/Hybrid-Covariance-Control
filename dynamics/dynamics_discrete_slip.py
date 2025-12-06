import os
import sys
file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

import numpy as np
from dynamics.ode_solver.dynamics_slip import *
from dynamics.dynamics_discrete import *

g = 9.81
k = 25.0
m = 0.5
r0 = 1

# g = 9.81
# # k = 25.0
# k = 5.0
# m = 2.5
# r0 = 1

# @jax.jit
def stoch_integr_slip(mode, x0, u, dt, eps, dW):   
    def mode0_dynamics_true_func_slip(args):
        (x0, u, dW, eps) = args
        # flight mode
        # [x, x_dot, z, z_dot, theta] = x0
        
        # Controlled: 3 inputs
        B = np.array([[0.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    [0.0, 0.0, 1.0]], dtype=np.float64)
        
        return x0 + np.array([x0[1], u[0], x0[3], u[1]-9.81, u[2]], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
    
        # B = np.array([[0.0],
        #             [0.0],
        #             [0.0],
        #             [0.0],
        #             [1.0]], dtype=np.float64)
        
        # return x0 + np.array([x0[1], 0, x0[3], -9.81, u[0]], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
    
    def mode0_dynamics_false_func_slip(args):
        (x0, u, dW, eps) = args
        # stance mode
        # [theta, theta_dot, r, r_dot] = x0
        
        theta, theta_dot, r, r_dot = x0[0], x0[1], x0[2], x0[3]
        
        # Defining the stance dynamics of the system
        B = np.array([[0.0, 0.0], 
                    [0.0, 0.0], 
                    [0.0, 1/m/r/r], 
                    [k/m, 0.0]], dtype=np.float64)
        
        # xt_next = x0 + np.array([theta_dot, 
        #                         u[0], 
        #                         r_dot, 
        #                         k/m*(r0-r)+u[1]], dtype=np.float64) * dt + np.sqrt(eps) * B@dW 
    
        xt_next = x0 + np.array([theta_dot, 
                                -2*theta_dot*r_dot/r-g*np.cos(theta)/r, 
                                r_dot + u[1]/m/r/r, 
                                k/m*(r0-r) - g*np.sin(theta) + theta_dot*theta_dot*r + k*u[0]/m], dtype=np.float64) * dt + np.sqrt(eps) * B@dW
        
        return xt_next
    
    xt_next = None
    args_choose_dynamics = (x0, u, dW, eps)
    if (mode == 0):
        xt_next = mode0_dynamics_true_func_slip(args_choose_dynamics)
    elif(mode == 1):
        xt_next = mode0_dynamics_false_func_slip(args_choose_dynamics)

    return xt_next


# ===============================================================================================================
#                                        SLIP Guard condition handling 
# ===============================================================================================================

# -------------------------------- From mode 1 (flight) to mode 2 (stance) --------------------------------
def guard_cond_slip_12(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==0) and (guard_slip_12(0.0,xt)<0) and (guard_slip_12(0.0,xt_next)>0)

def guard_true_slip_12(args):
    print("slip_cond_12: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, RandN, eps, reset_arg) = args
    
    # -----------------
    #    Bi-section 
    # -----------------    
    def bisection_while_body(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
        
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_mid - t_left
        dW_new = np.sqrt(dt_new)*randN
        
        x_mid = stoch_integr_slip(current_mode, x_left, u_current, dt_new, eps, dW_new)

        # Define conditions
        guard_left = guard_slip_12(t_left, x_left)
        guard_mid = guard_slip_12(t_mid, x_mid)
        
        # Check if guard condition at mid is zero
        continue_cond = (guard_mid != 0) and ((t_right-t_left)/2.0 >= tol)
        
        if (guard_left * guard_mid < 0):
            t_right = t_mid
            x_right = x_mid
        else:
            t_left = t_mid
            x_left = x_mid
            
        return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond

    tol = 1e-10
    t_left = t
    t_right = t+dt_int
    t_mid = t
    x_left = xt_current
    x_mid = xt_next
    x_right = xt_next
    continue_bisection = True
    
    while (continue_bisection):
        args = (t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection)
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = bisection_while_body(args)
    
    xt_next, next_mode, new_reset_arg = reset_map_slip_12(t_left, x_left, current_mode, reset_arg)
    dt_final = t_left - t
    dW_new = np.sqrt(dt_final) * RandN
        
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_slip_12(args):
    (_, current_mode, _, _, xt_next, dt_int, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg


# --------------------------------------------------  
#       From mode 2 (stance) to mode 1 (flight)
# --------------------------------------------------
def guard_cond_slip_21(xt, xt_next, current_mode):
    # assume time invariant guard for now
    return (current_mode==1) and (guard_slip_21(0.0,xt)<=0) and (guard_slip_21(0.0,xt_next)>0)

def guard_true_slip_21(args):
    print("slip guard condition 21: True")
    (xt_current, current_mode, u, t, xt_next, dt_int, RandN, eps, reset_arg) = args
    
    # -----------------
    #    Bi-section 
    # -----------------    
    def bisection_while_body(carry):
        t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond = carry
        
        t_mid = (t_left + t_right) / 2.0
        dt_new = t_mid - t_left
        dW_new = np.sqrt(dt_new)*randN
        
        x_mid = stoch_integr_slip(current_mode, x_left, u_current, dt_new, eps, dW_new)

        # Define conditions
        guard_left = guard_slip_21(t_left, x_left)
        guard_mid = guard_slip_21(t_mid, x_mid)
        
        # Check if guard condition at mid is zero
        continue_cond = (guard_mid != 0) and ((t_right-t_left)/2.0 >= tol)
        
        if (guard_left * guard_mid < 0):
            t_right = t_mid
            x_right = x_mid
        else:
            t_left = t_mid
            x_left = x_mid
            
        return t_left, t_mid, t_right, x_left, x_mid, x_right, u_current, tol, randN, continue_cond

    tol = 1e-10
    t_left = t
    t_right = t+dt_int
    t_mid = t
    x_left = xt_current
    x_mid = xt_next
    x_right = xt_next
    continue_bisection = True
    
    while (continue_bisection):
        args = (t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection)
        t_left, t_mid, t_right, x_left, x_mid, x_right, u, tol, RandN, continue_bisection = bisection_while_body(args)
    
    # Debug statement
    # print(f"guard left: {guard_slip_21(t_left, x_left)}, guard right: {guard_slip_21(t_right, x_right)}, guard mid: {guard_slip_21(t_mid, x_mid)}")  
    
    xt_next, next_mode, new_reset_arg = reset_map_slip_21(t_left, x_left, current_mode, reset_arg)
    dt_final = t_left - t
    dW_new = np.sqrt(dt_final) * RandN
    
    # # --------------------
    # #   Direct reset map 
    # # --------------------
    # xt_next, next_mode, new_reset_arg = reset_map_slip_21(t, xt_next, current_mode, reset_arg)
    # dW_new = np.sqrt(dt_int) * RandN
    
    return xt_next, next_mode, dW_new, new_reset_arg


def guard_false_slip_21(args):
    (_, current_mode, _, _, xt_next, dt_int, RandN, _, reset_arg) = args
    dW = np.sqrt(dt_int)*RandN
    return xt_next, current_mode, dW, reset_arg


# ===============================================================================================================
#                                       // End of SLIP guard condition handling //
# ===============================================================================================================

from functools import partial

reaction_mode_mismatch_slip = partial(reaction_mode_mismatch, 
                                      cond_early_arrival=cond_early_arrival_slip)

h_stoch_integr_slip = partial(h_stoch_integr, 
                                stoch_integr_func = stoch_integr_slip, 
                                guard_0=guard_cond_slip_12,
                                guard_true_func_0=guard_true_slip_12,
                                guard_false_func_0=guard_false_slip_12,
                                guard_1 = guard_cond_slip_21, 
                                guard_true_func_1 = guard_true_slip_21, 
                                guard_false_func_1 = guard_false_slip_21)


h_stoch_fb_rollout_slip = partial(h_stoch_fb_rollout, 
                                cond_mismatch_func=cond_mode_mismatch_slip,
                                reaction_mismatch_func=reaction_mode_mismatch_slip,
                                h_stoch_integr_func=h_stoch_integr_slip)    
    
    
def event_detect_discrete_slip(current_mode, 
                               x0, u, t0, dt, reset_args, 
                               detection=True, backwards=False):

    smooth_dynamics_slip = {0:dyn_flight_slip, 1:dyn_stance_slip}

    Rxs_slip = {0:Rx_slip_12, 1:Rx_slip_21}
    Rts_slip = {0:Rt_slip_12, 1:Rt_slip_21}

    gxs_slip = {0:gx_slip_12, 1:gx_slip_21}
    gts_slip = {0:gt_slip_12, 1:gt_slip_21}

    guards_slip_slip = {0:guard_slip_12, 1: guard_slip_21}
    reset_maps_slip_slip = {0:reset_map_slip_12, 1:reset_map_slip_21}
                                        
    return event_detect_onestep_discrete(x0, u, t0, 
                                        dt, 
                                        current_mode, 
                                        smooth_dynamics_slip, 
                                        guards_slip_slip,
                                        gxs_slip,
                                        gts_slip,
                                        reset_maps_slip_slip,
                                        Rxs_slip, Rts_slip,
                                        reset_args, 
                                        guard_cond_func_0=guard_cond_slip_12,
                                        guard_cond_func_1=guard_cond_slip_21,
                                        detection=detection, backwards=backwards)

