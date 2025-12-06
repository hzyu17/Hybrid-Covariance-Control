import os
import sys

file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

from dynamics.saltation_matrix import compute_saltation
from dynamics.trajectory_extension import *
from dynamics.guard_reset_bouncing import *

import scipy
import numpy as np

# -------------------------------------  
#  Function that handles mode mismatch   
# -------------------------------------  
def reaction_mode_mismatch(current_index, 
                            current_mode, ref_current_mode, 
                            ext_trj_fwd, ext_trj_bwd, 
                            ref_ext_modechange,
                            Kfb_ref_ext_fwd, kff_ref_ext_fwd,
                            Kfb_ref_ext_bwd, kff_ref_ext_bwd,
                            cnt_mismatch, cond_early_arrival=None):
    # Take the first hybrid event for now. Needs to find the correct corresponding one among all hybrid events.
    if (cond_early_arrival(current_mode, ref_current_mode, ref_ext_modechange)):
        extended_trj = ext_trj_bwd
        K_fb = Kfb_ref_ext_bwd
        k_ff = kff_ref_ext_bwd
    else:
        extended_trj = ext_trj_fwd
        K_fb = Kfb_ref_ext_fwd
        k_ff = kff_ref_ext_fwd
    
    xref_i = extended_trj[current_index]
    K_fb_i = K_fb[current_index]
    k_ff_i = k_ff[current_index]
    
    cnt_mismatch += 1
    
    return xref_i, K_fb_i, k_ff_i, cnt_mismatch


def stochastic_integration(x0, u, t_span, epsilon, dW, dyn_f, dyn_gdWt):
    """ Rollout function assuming constant control input during the time span.
    Returns:
        array: stochastic integrated state at tf.
    """
    args = (u, )
    # ============= ode solver =============
    
    # solution = scipy.integrate.solve_ivp(fun=lambda t, y: dyn_bouncing(t, y, *args), 
    #                                     t_span=t_span, y0=x0, method='RK45', 
    #                                     t_eval=t_eval, dense_output=True)
    
    # t0, tf = t_span[0], t_span[-1]
    
    # # Solve for the continuous trajectory before the contact 
    # t_sol = np.linspace(t0, tf, nt).flatten()
    
    # # The solved trajecoty, in shape (nx+nx*nx, nt)
    # f_disc = solution.sol(t_sol) 
    # x_next_det = f_disc[:, -1]
    
    # xt_next = x_next_det + np.sqrt(epsilon)*dW
    
    # ============= method 2: forward Euler =============
    t0, tf = t_span[0], t_span[-1]
    dt = tf - t0
    xt_next = x0 + dyn_f(t0, x0, *args)*dt + dyn_gdWt(x0, dW, epsilon)
    
    return xt_next

def event_condition(xt, xt_next, guard):
    # assume time invariant guard for now
    return (guard(0.0,xt)>0) and (guard(0.0,xt_next)<=0) 

def event_reactive_fun(args):
                        
    (xt_current, current_mode, u, 
     t, t_next, xt_next, dt_int, 
     RandN, epsilon, 
     smooth_integration_fun, guard_fun, reset_map_fun, reset_args) = args
    
    current_guard = guard_fun[current_mode]
    current_resetmap = reset_map_fun[current_mode]
    
    # -------------------
    #     Bi-section 
    # -------------------
    tol = 1e-8
    t_left = t
    x_left = xt_current
    t_right = t+dt_int
    
    while (t_right - t_left) / 2.0 > tol:
        t_mid = (t_left + t_right) / 2.0
        t_span = (t_left, t_right)
        dt_new = t_mid - t_left
        dW_new = np.sqrt(dt_new)*RandN
        
        x_mid = smooth_integration_fun(x_left, u, t_span, epsilon, dW_new)
        
        if current_guard(t_mid, x_mid) == 0:
            return (t_mid, x_mid)  # We've found the exact root
        
        elif current_guard(t_left, x_left) * current_guard(t_mid, x_mid) < 0:
            t_right = t_mid  # The root is in the left half
            x_right = x_mid
        else:
            t_left = t_mid  # The root is in the right half
            x_left = x_mid
            
        # print(f"t_left: {t_left}, t_right: {t_right}, midpoint: {t_mid}, x_mid: {x_mid}")  # Debug statement
    
    # t_event = (t_left + t_right) / 2.0
    # x_event = (x_left + x_right) / 2.0
    
    t_event = t_left
    dt_final = t_event - t_left
    dW = np.sqrt(dt_final)*RandN
    
    xt_next, next_mode, reset_byproduct = current_resetmap(t_left, x_left, current_mode, reset_args)
    
    # # -------------------
    # #   Direct reset map 
    # # -------------------
    # dW = np.sqrt(dt_int)*RandN
    # xt_next, next_mode, reset_byproduct = current_resetmap(t, xt_next, current_mode, reset_args)
        
    return xt_next, next_mode, dW, reset_byproduct


def event_detect_onestep(x0, u, t0, tf, current_mode, 
                         smooth_dynamics, 
                         guards, gxs, gts,
                         reset_maps, Rxs, Rts,
                         reset_args, detection=True, backwards=False):
    """Integrate controlled dynamics in a short period of time with hybrid event detection.

    Args:
        x0 (array): starting state
        u (array): control input
        t0 (scalar): start time
        tf (scalar): end time
        current_mode (int): the current mode
        smooth_dynamics(map): mode-dependent smooth dynamics 
        guards: mode-dependent guard functions
        gxs: mode-dependent \par g(t,x) / \par x
        gts: mode-dependent \par g(t,x) / \par t
        reset_maps: mode-dependent reset_map functions R(t,x)
        reset_controls: mode-dependent reset_control_functions
        Rxs: mode-dependent \par R(t,x) / \par x
        Rts: mode-dependent \par R(t,x) / \par t
        detection (bool, optional): With detection flag. Defaults to True.
        backwards (bool, optional): Integrate backwards flag. Defaults to False.

    Returns:
        tuple: Containing the next state and contact information if a hybrid event happens.
    """
    # Define the dynamics using the integration
    nt = 10000
    
    current_dyn = smooth_dynamics[current_mode]
        
    current_Rx = Rxs[current_mode]
    current_Rt = Rts[current_mode]
    
    current_gx = gxs[current_mode]
    current_gt = gts[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    args = (u, )
    if backwards:
        # integrate backwards
        t_span = (t0, tf)
    else:
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, nt)
        dyn_fun=lambda t, y: current_dyn(t, y, *args)
    
    x_next = None
    x_event = None
    t_event = None
    x_reset = None
    saltation = None
    next_mode = current_mode
    reset_byproduct = (None, )
    
    if detection:
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, 
                                            # method='RK23', 
                                            t_eval=t_eval, dense_output=True, 
                                            events=current_guard, vectorized=False)
        
        # Hit guard
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            x_reset, next_mode, reset_byproduct = current_resetmap(t_event, x_event, current_mode, reset_args)
            x0 = x_reset
            
            # ---------- Compute saltation matrix ---------- 
            R_x = current_Rx(t_event, x_event, current_mode, reset_args)[0]
            R_t = current_Rt(t_event, x_event, current_mode, reset_args)[0]
            
            g_x = current_gx(t_event, x_event)
            g_t = current_gt(t_event, x_event)
            
            next_dyn = smooth_dynamics[next_mode]
            
            F_1 = current_dyn(t_event, x_event, u)
            F_2 = next_dyn(t_event, x_reset, u) # Important, the F2 is evaluated at the reseted state!
            saltation = compute_saltation(F_1, F_2, R_t, R_x, g_t, g_x)
            
            t0 = t_event
            
            x_next = x_reset.flatten()
        
        # Had no contact
        else:
            x0 = None
            
            t = np.linspace(t0, tf, nt).flatten()
            
            # The solved trajecoty, in shape (nx+nx*nx, nt)
            f_disc = solution.sol(t) 
            
            x_next = f_disc[:, -1]
    
    else: # Do not detect contact 
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True)
        
        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, tf, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        f_disc = solution.sol(t) 
        
        x_next = f_disc[:, -1]
    
    mode_mapping = np.array([current_mode, next_mode])
    
    return x_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct

def event_detect_onestep_discrete(x0, u, t0, tf, current_mode, 
                                    smooth_dynamics, 
                                    guards, gxs, gts,
                                    reset_maps, Rxs, Rts,
                                    reset_args, detection=True, backwards=False):
    """Integrate controlled dynamics in a short period of time with hybrid event detection, discrete version.

    Returns:
        tuple: Containing the next state and contact information if a hybrid event happens.
    """
    # Define the dynamics using the integration
    nt = 1000
    
    current_dyn = smooth_dynamics[current_mode]
        
    current_Rx = Rxs[current_mode]
    current_Rt = Rts[current_mode]
    
    current_gx = gxs[current_mode]
    current_gt = gts[current_mode]
    
    current_guard = guards[current_mode]
    current_resetmap = reset_maps[current_mode]
    
    args = (u, )
    if backwards:
        # integrate backwards
        t_span = (t0, tf)
    else:
        t_span = (t0, tf)
        t_eval = np.linspace(t0, tf, nt)
        dyn_fun=lambda t, y: current_dyn(t, y, *args)
    
    x_next = None
    x_event = None
    t_event = None
    x_reset = None
    saltation = None
    next_mode = current_mode
    reset_byproduct = (None, )
    
    if detection:
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, 
                                            t_eval=t_eval, dense_output=True, 
                                            events=current_guard, vectorized=False)
        
        # Hit guard
        if len(solution.t_events[0]) > 0:
            t_event = solution.t_events[0][0]
            x_event = solution.y_events[0][0]
            x_reset, next_mode, reset_byproduct = current_resetmap(x_event, current_mode, reset_args)
            x0 = x_reset
            
            # ---------- Compute saltation matrix ---------- 
            R_x = current_Rx(t_event, x_event)
            R_t = current_Rt(t_event, x_event)
            
            g_x = current_gx(t_event, x_event)
            g_t = current_gt(t_event, x_event)
            
            next_dyn = smooth_dynamics[next_mode]
            
            F_1 = current_dyn(t_event, x_event, u)
            F_2 = next_dyn(t_event, x_reset, u) # Important, the F2 is evaluated at the reseted state!
            saltation = compute_saltation(F_1, F_2, R_t, R_x, g_t, g_x)
            
            t0 = t_event
            
            x_next = x_reset.flatten()
        
        # Had no contact
        else:
            x0 = None
            
            t = np.linspace(t0, tf, nt).flatten()
            
            # The solved trajecoty, in shape (nx+nx*nx, nt)
            f_disc = solution.sol(t) 
            
            x_next = f_disc[:, -1]
    
    else: # Do not detect contact 
        solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
                                            t_span=t_span, y0=x0, method='RK45', 
                                            t_eval=t_eval, dense_output=True)
        
        # Solve for the continuous trajectory before the contact 
        t = np.linspace(t0, tf, nt).flatten()
        
        # The solved trajecoty, in shape (nx+nx*nx, nt)
        f_disc = solution.sol(t) 
        
        x_next = f_disc[:, -1]
    
    mode_mapping = np.array([current_mode, next_mode])
    
    return x_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct


# def event_detect_onestep_discrete(x0, u, t0, tf, current_mode, 
#                                     smooth_dynamics, 
#                                     guards, gxs, gts,
#                                     reset_maps, Rxs, Rts,
#                                     reset_args, detection=True, backwards=False):
#     """Integrate controlled dynamics in a short period of time with hybrid event detection, discrete version.

#     Returns:
#         tuple: Containing the next state and contact information if a hybrid event happens.
#     """
#     # Define the dynamics using the integration
#     nt = 1000
    
#     current_dyn = smooth_dynamics[current_mode]
        
#     current_Rx = Rxs[current_mode]
#     current_Rt = Rts[current_mode]
    
#     current_gx = gxs[current_mode]
#     current_gt = gts[current_mode]
    
#     current_guard = guards[current_mode]
#     current_resetmap = reset_maps[current_mode]
    
#     args = (u, )
#     if backwards:
#         # integrate backwards
#         t_span = (t0, tf)
#     else:
#         t_span = (t0, tf)
#         t_eval = np.linspace(t0, tf, nt)
#         dyn_fun=lambda t, y: current_dyn(t, y, *args)
    
#     x_next = None
#     x_event = None
#     t_event = None
#     x_reset = None
#     saltation = None
#     next_mode = current_mode
#     reset_byproduct = (None, )
    
#     if detection:
#         solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
#                                             t_span=t_span, y0=x0, 
#                                             t_eval=t_eval, dense_output=True, 
#                                             events=current_guard, vectorized=False)
        
#         # Hit guard
#         if len(solution.t_events[0]) > 0:
#             t_event = solution.t_events[0][0]
#             x_event = solution.y_events[0][0]
#             x_reset, next_mode, reset_byproduct = current_resetmap(t_event, x_event, current_mode, reset_args)
#             x0 = x_reset
            
#             # ---------- Compute saltation matrix ---------- 
#             R_x = current_Rx(t_event, x_event, current_mode, reset_args)[0]
#             R_t = current_Rt(t_event, x_event, current_mode, reset_args)[0]
            
#             g_x = current_gx(t_event, x_event)
#             g_t = current_gt(t_event, x_event)
            
#             next_dyn = smooth_dynamics[next_mode]
            
#             F_1 = current_dyn(t_event, x_event, u)
#             F_2 = next_dyn(t_event, x_reset, u) # Important, the F2 is evaluated at the reseted state!
#             saltation = compute_saltation(F_1, F_2, R_t, R_x, g_t, g_x)
            
#             t0 = t_event
            
#             x_next = x_reset.flatten()
        
#         # Had no contact
#         else:
#             x0 = None
            
#             t = np.linspace(t0, tf, nt).flatten()
            
#             # The solved trajecoty, in shape (nx+nx*nx, nt)
#             f_disc = solution.sol(t) 
            
#             x_next = f_disc[:, -1]
    
#     else: # Do not detect contact 
#         solution = scipy.integrate.solve_ivp(fun=dyn_fun, 
#                                             t_span=t_span, y0=x0, method='RK45', 
#                                             t_eval=t_eval, dense_output=True)
        
#         # Solve for the continuous trajectory before the contact 
#         t = np.linspace(t0, tf, nt).flatten()
        
#         # The solved trajecoty, in shape (nx+nx*nx, nt)
#         f_disc = solution.sol(t) 
        
#         x_next = f_disc[:, -1]
    
#     mode_mapping = np.array([current_mode, next_mode])
    
#     return x_next, saltation, mode_mapping, t_event, x_event, x_reset, reset_byproduct