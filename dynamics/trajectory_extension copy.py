# Helper functions to handle reference trajectory extensions
import numpy as np

def compute_trejactory_extension(event_info, start_time, end_time, nt, dt,
                                  nx, nu, init_state, target_state, detection_func):
        
        # NO hybrid events
        if len(event_info.keys()) == 0:
            return []
        
        # event_info:
        sorted_hybrid_index = sorted(event_info.keys())
        ref_ext_helper = []
        
        i_events = []
        t_events = []
        x_events = []
        x_resets = []
        K_fb_fwd_extensions = []
        k_ff_fwd_extensions = []
        K_fb_bwd_extensions = []
        k_ff_bwd_extensions = []
        
        mode_changes = []
        
        i_events.append(0)
        t_events.append(start_time)
        x_events.append(init_state)
        x_resets.append(init_state)
        K_fb_fwd_extensions.append(np.zeros(1))
        K_fb_bwd_extensions.append(np.zeros(1))
        k_ff_fwd_extensions.append(np.zeros(1))
        k_ff_bwd_extensions.append(np.zeros(1))
        mode_changes.append(np.array([0, 0]))
        
        # event_info[i_key] = (t_event, x_event, x_reset, mode_change, K_feedback_extensions, K_feedforward_extensions)
        for i_key in sorted_hybrid_index:
            i_events.append(i_key)
            t_events.append(event_info[i_key][0])
            x_events.append(event_info[i_key][1])
            x_resets.append(event_info[i_key][2])
            mode_changes.append(event_info[i_key][3])
            
            K_fb_fwd_extensions.append(event_info[i_key][4][0])
            K_fb_bwd_extensions.append(event_info[i_key][4][1])
            
            k_ff_fwd_extensions.append(event_info[i_key][5][0])
            k_ff_bwd_extensions.append(event_info[i_key][5][1])
        
        i_events.append(nt)
        t_events.append(end_time)
        x_events.append(target_state)
        x_resets.append(target_state)
        K_fb_fwd_extensions.append(np.zeros(1))
        K_fb_bwd_extensions.append(np.zeros(1))
        k_ff_fwd_extensions.append(np.zeros(1))
        k_ff_bwd_extensions.append(np.zeros(1))
        
        if event_info.keys():
            mode_changes.append(event_info[sorted_hybrid_index[-1]][3])
        
        # Forward and backward trajectory extensions and (feedback, feedforward) gains for the two extensions
        for ii, tevent_i in enumerate(t_events[1:-1], start=1):
            i_event_i = i_events[ii]
            x_event_i = x_events[ii]
            x_reset_i = x_resets[ii]
            mode_i = mode_changes[ii][0]
            next_mode_i = mode_changes[ii][1]
            
            K_feedback_fwd_extension_i = K_fb_fwd_extensions[ii]
            k_feedforward_fwd_extension_i = k_ff_fwd_extensions[ii]
            
            K_feedback_bwd_extension_i = K_fb_bwd_extensions[ii]
            k_feedforward_bwd_extension_i = k_ff_bwd_extensions[ii]
            
            # ---------------------------------------
            #  Choose a time span for the extensions
            # ---------------------------------------
            t_ext_fwd_i = end_time
            if len(dt) > 1:
                dt_i = dt[i_event_i]
                
                # [0, t_event] for padding
                time_span_ext_fwd_padding = np.arange(0, tevent_i, dt_i)
                # [t_event, t_trj_ext_fwd]
                timespan_ext_fwd = np.arange(tevent_i, t_ext_fwd_i, dt_i)
                # [t_trj_ext_bwd: t_event]
                timespan_ext_bwd = np.arange(0, tevent_i, dt_i)[::-1]
                
                
            else:
                dt_i = dt
            
                # [0, t_event] for padding
                time_span_ext_fwd_padding = np.arange(0, tevent_i, dt_i)
                # [t_event, t_trj_ext_fwd]
                timespan_ext_fwd = np.arange(tevent_i, t_ext_fwd_i, dt_i)
                # [t_trj_ext_bwd: t_event]
                timespan_ext_bwd = np.arange(0, tevent_i, dt_i)[::-1]
            
            # time span lengths
            nt_ext_fwd = len(timespan_ext_fwd)
            nt_ext_bwd = len(timespan_ext_bwd)
            nt_ext_padding_fwd = len(time_span_ext_fwd_padding)
            nt_ext_padding_bwd = nt_ext_fwd
            
            xtrj_ext_padding_fwd_i = np.zeros((nt_ext_padding_fwd, nx[mode_i]))
            xtrj_ext_fwd_i = np.zeros((nt_ext_fwd, nx[mode_i]))
            xtrj_ext_fwd_i[0] = x_event_i
            
            xtrj_ext_bwd_i = np.zeros((nt_ext_bwd, nx[next_mode_i]))
            xtrj_ext_padding_bwd_i = np.zeros((nt_ext_padding_bwd, nx[next_mode_i]))    
            
            # Use the same gain for the whole extensions, and padding it to the whole time span
            K_feedback_ext_fwd_i = np.tile(K_feedback_fwd_extension_i, (nt, 1, 1))
            k_feedforward_ext_fwd_i = np.tile(k_feedforward_fwd_extension_i, (nt, 1))
            
            K_feedback_ext_bwd_i = np.tile(K_feedback_bwd_extension_i, (nt, 1, 1))
            k_feedforward_ext_bwd_i = np.tile(k_feedforward_bwd_extension_i, (nt, 1))
            
            # ----------------------------------
            # simulate the forward extension
            # ----------------------------------
            x_i = x_event_i
            for jj in range(nt_ext_fwd-1):
                t_jj = timespan_ext_fwd[jj]
                
                if len(dt) > 1:
                    if jj == nt_ext_fwd-2:
                        dt_jj = dt[-1]
                    else:
                        dt_jj = dt[jj]
                else:
                    dt_jj = dt
                
                # Using zero control, modify if needed.
                current_input = np.zeros(nu[mode_i])
                
                next_state, _, _, _, _, _, _ = detection_func(x_i, current_input, t_jj, t_jj+dt_jj, mode_i, detect=False, reset_args=None)
                
                # Store states and inputs
                xtrj_ext_fwd_i[jj+1] = next_state

                # Update the current state
                x_i = next_state
            
            xtrj_ext_fwd_i = np.vstack((xtrj_ext_padding_fwd_i, xtrj_ext_fwd_i))
            
            # ----------------------------------
            #   Simulate the backward extension
            # ----------------------------------
            xtrj_ext_bwd_i[0] = x_reset_i
            x_i = x_reset_i
            for jj in range(nt_ext_bwd-1):
                t_jj = timespan_ext_bwd[jj]
                if len(dt) > 1:
                    dt_jj = dt[jj]
                else:
                    dt_jj = dt
                
                # modify if needed
                current_input = np.zeros(nu[next_mode_i])
                
                next_state, _, _, _, _, _, _ = detection_func(x_i, current_input, t_jj, t_jj-dt_jj, next_mode_i, detect=False, reset_args=None)
                
                # Store states and inputs
                xtrj_ext_bwd_i[jj+1] = next_state

                # Update the current state
                x_i = next_state
            
            # -------------------------------
            # reverse the backward extension 
            # -------------------------------   
            xtrj_ext_bwd_i = xtrj_ext_bwd_i[::-1]
            
            # --------------------- padding ---------------------
            xtrj_ext_bwd_i = np.vstack((xtrj_ext_bwd_i, xtrj_ext_padding_bwd_i))
            
            # ------------------------ collect the trajectory extensions ------------------------
            ref_ext_helper.append({"Mode Change": np.array([mode_i, next_mode_i]), 
                                    "Trajectory Extensions": {mode_i:xtrj_ext_fwd_i, next_mode_i:xtrj_ext_bwd_i}, 
                                    "Feedback gains": {mode_i:K_feedback_ext_fwd_i, next_mode_i:K_feedback_ext_bwd_i}, 
                                    "Feedforward gains": {mode_i:k_feedforward_ext_fwd_i, next_mode_i:k_feedforward_ext_bwd_i}, 
                                    "event index": i_event_i})
            
        return ref_ext_helper


def extract_extensions(ref_ext_helper, start_index=0, padding=False):
    # ---------------------------------------------------
    #           Extract the extended references 
    # ---------------------------------------------------
    num_events = len(ref_ext_helper)
    
    if num_events == 0: # reference has no hybrid events
        return None, None, None, None, None, None, None, None
    
    v_mode_change = []
    v_ext_trj_fwd = []
    v_ext_trj_bwd = []
    v_Kfb_ext_trj_fwd = []
    v_kff_ext_trj_fwd = []
    v_Kfb_ext_trj_bwd = []
    v_kff_ext_trj_bwd = []
    v_tevents = []
    
    for i_event in range(num_events):
        # find out the mode changes
        MC_i = ref_ext_helper[i_event]["Mode Change"]
        Ext_Trjs_i = ref_ext_helper[i_event]["Trajectory Extensions"]
        Ext_Kfb_i = ref_ext_helper[i_event]["Feedback gains"]
        Ext_kff_i = ref_ext_helper[i_event]["Feedforward gains"]
        tevent_i = ref_ext_helper[i_event]["event index"]
        
        cur_mode_i = MC_i[0]
        next_mode_i = MC_i[1]
        
        v_mode_change.append((cur_mode_i, next_mode_i))
        v_tevents.append(tevent_i)
        
        if padding: # padding to the larger dimension of the two modes.
            n_states = np.array([Ext_Trjs_i[cur_mode_i].shape[1], Ext_Trjs_i[next_mode_i].shape[1]])
            n_inputs = np.array([Ext_Kfb_i[cur_mode_i].shape[1], Ext_Kfb_i[next_mode_i].shape[1]])
            
            max_nstate = np.max(n_states)
            max_ninput = np.max(n_inputs)
            
            nt_length = Ext_Trjs_i[cur_mode_i].shape[0] - start_index 
            
            ext_trj_fwd = np.zeros((nt_length, max_nstate))
            ext_trj_bwd = np.zeros((nt_length, max_nstate))
            ext_trj_fwd[:, :n_states[0]] = Ext_Trjs_i[cur_mode_i][start_index:]
            ext_trj_bwd[:, :n_states[1]] = Ext_Trjs_i[next_mode_i][start_index:]
            
            v_ext_trj_fwd.append(ext_trj_fwd)
            v_ext_trj_bwd.append(ext_trj_bwd)
            
            Kfb_ext_trj_fwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_bwd = np.zeros((nt_length, max_ninput, max_nstate))
            Kfb_ext_trj_fwd[:, :n_inputs[0], :n_states[0]] = Ext_Kfb_i[cur_mode_i][start_index:]
            Kfb_ext_trj_bwd[:, :n_inputs[1], :n_states[1]] = Ext_Kfb_i[next_mode_i][start_index:]
            
            v_Kfb_ext_trj_fwd.append(Kfb_ext_trj_fwd)
            v_Kfb_ext_trj_bwd.append(Kfb_ext_trj_bwd)
            
            kff_ext_trj_fwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_bwd = np.zeros((nt_length, max_ninput))
            kff_ext_trj_fwd[:, :n_inputs[0]] = Ext_kff_i[cur_mode_i][start_index:]
            kff_ext_trj_bwd[:, :n_inputs[1]] = Ext_kff_i[next_mode_i][start_index:]
            
            v_kff_ext_trj_fwd.append(kff_ext_trj_fwd)
            v_kff_ext_trj_bwd.append(kff_ext_trj_bwd)
            
        else:
            # Add the forward and backward extensions to the collection        
            v_ext_trj_fwd.append(Ext_Trjs_i[cur_mode_i][start_index:])
            v_ext_trj_bwd.append(Ext_Trjs_i[next_mode_i][start_index:])
            
            # Add the feedback gain for forward and backward extensions to the collection
            v_Kfb_ext_trj_fwd.append(Ext_Kfb_i[cur_mode_i][start_index:])
            v_Kfb_ext_trj_bwd.append(Ext_Kfb_i[next_mode_i][start_index:])
            
            # Add the feedforward gain for backward and backward extensions to the collection
            v_kff_ext_trj_fwd.append(Ext_kff_i[cur_mode_i][start_index:])
            v_kff_ext_trj_bwd.append(Ext_kff_i[next_mode_i][start_index:])
        
        
    return (v_mode_change, v_ext_trj_bwd, v_ext_trj_fwd, 
            v_Kfb_ext_trj_bwd, v_Kfb_ext_trj_fwd, v_kff_ext_trj_bwd, v_kff_ext_trj_fwd, v_tevents)