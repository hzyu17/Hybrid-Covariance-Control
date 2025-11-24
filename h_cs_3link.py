import numpy as np
import os
import sys
file_path = os.path.abspath(__file__)
exp_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(exp_dir, '..'))
dyn_dir = os.path.abspath(os.path.join(root_dir, 'dynamics'))
sys.path.append(root_dir)
sys.path.append(dyn_dir)

from dynamics.walking_3link import *

def h_cs_3link():
    global t_2, torque, y, force
    tout, xout, t_events, x_events, saltations = solve_limitcycles(n_steps=2)

    # ------ check interpolation ------
    show_states = False
    if show_states:
        fig, axes = plt.subplots(2, 3)
        axes = axes.flatten()

        axes[0].scatter(tout, xout[:, 0], color='k')
        axes[0].set_title(r'Interpolation $q_1$')

        axes[1].scatter(tout, xout[:, 1], s=1, color='k')
        axes[1].set_title(r'Interpolation $q_2$')

        axes[2].scatter(tout, xout[:, 2], s=1 , color='k')
        axes[2].set_title(r'Interpolation $q_3$')

        axes[3].scatter(tout, xout[:, 3], s=1 , color='k')
        axes[3].set_title(r'Interpolation $q_4$')

        axes[4].scatter(tout, xout[:, 4], s=1 , color='k')
        axes[4].set_title(r'Interpolation $q_5$')

        axes[5].scatter(tout, xout[:, 5], s=1 , color='k')
        axes[5].set_title(r'Interpolation $q_6$')

        plt.show()
    
    # -------- Linearization -------- 
    nt = tout.shape[0]
    ui = np.zeros((2))
    
    # add the initial time and the end time
    t_events.insert(0, 0.0)

    # find foot-touching event index
    te_indx = []
    for te_i in t_events:
        if len(np.where(tout==te_i)[0]) > 0:
            te_indx.append(np.where(tout==te_i)[0][0])

    ftouch_event = te_indx[1]

    # find the hip apex-to-apex trajectory
    hip_trj = r * np.cos(xout[:, 0])
    
    hip_apex_index = []
    hip_apex = []
    for ii in range(len(te_indx)-1):
        te_i = te_indx[ii]
        te_next = te_indx[ii+1]
        hip_apex_index.append(te_i + np.argmax(hip_trj[te_i:te_next]))
        hip_apex.append(np.max(hip_trj[te_i:te_next]))
    
    fig, ax = plt.subplots()
    ax.plot(range(len(hip_trj)), hip_trj, color='k')
    ax.scatter(hip_apex_index, hip_apex, c='green')
    ax.set_title("Hip Height")
    plt.show()
    
    # Linearization
    nx = 6
    nu = 2
    A_trj = np.zeros((nt, nx, nx))
    B_trj = np.zeros((nt, nx, nu))
    dt_trj = np.zeros(nt)
    for ii in range(nt):
        if ii < nt-1:
            dt_trj[ii] = tout[ii+1] - tout[ii]
        A_trj[ii] = jac_fxgu_x(tout[ii], xout[ii], ui, a)
        B_trj[ii] = jac_fxgu_u(tout[ii], xout[ii], ui, a)
    dt_trj[-1] = dt_trj[-2]

    nx1, nx2 = 6, 6
    nt1, nt2 = te_indx[1] - te_indx[0], te_indx[2] - te_indx[1]

    A1 = np.zeros((nt1, nx1, nx1))
    B1 = np.zeros((nt1, nx1, nu))
    Q1 = np.zeros((nt1, nx1, nx1))

    for i in range(nt1):

        dt = dt_trj[i]

        A1_i = A_trj[i]
        B1_i = B_trj[i]
        
        A1[i] = (A1_i-np.eye(nx1,nx1))/dt
        B1[i] = B1_i/dt

    A2 = np.zeros((nt2, nx2, nx2))
    B2 = np.zeros((nt2, nx2, nu))
    Q2 = np.zeros((nt2, nx2, nx2))

    for i in range(nt2):
        dt = dt_trj[i]

        ii = ftouch_event+i+1
        
        A2_i = A_trj[ii]
        B2_i = B_trj[ii]

        A2[i] = (A2_i - np.eye(nx2,nx2)) / dt
        B2[i] = B2_i / dt

    t_span1 = (0, dt * nt1)
    t_span2 = (0, dt * nt2)

    t_eval1 = np.linspace(0, dt*nt1, nt1+1)
    t_eval2 = np.linspace(0, dt*nt2, nt2+1)

    t_span1_reverse = (dt * nt1, 0)
    t_span2_reverse = (dt * nt2, 0)

    t_eval1_reverse = np.linspace(dt*nt1, 0, nt1+1)
    t_eval2_reverse = np.linspace(dt*nt2, 0, nt2+1)

    # --------------------- compute Phi^{A_j} --------------------
    Phi_A1_t = np.zeros((nt1+1, nx1, nx1))
    Phi_A1_0 = np.eye(nx1).flatten()

    def ode_Phi_A1(t, y, tout):
        # i = min(int(t / dt), nt1-1)  
        i = np.argmin(np.abs(tout-t))
        A1_i = A1[i]
        Phi_reshaped = y.reshape((nx1, nx1))
        dydt = A1_i @ Phi_reshaped
        return dydt.flatten()  
    
    # Solve ODE
    result_PhiA1 = solve_ivp(ode_Phi_A1, t_span1, Phi_A1_0, method='RK23', t_eval=t_eval1, args=(tout,))

    # Reshape the result to get the solution matrices at each time step
    Phi_A1_t = result_PhiA1.y.reshape((nx1, nx1, -1))
    Phi_A1_t = np.moveaxis(Phi_A1_t, 2, 0)
    Phi_A1 = Phi_A1_t[-1]

    # --------------------- Integrate Phi_A2 ---------------------
    def ode_Phi_A2(t, y):
        # i = min(int(t / dt), nt2-1)  
        i = np.argmin(np.abs(tout-t))
        A2_i = A2[i]
        Phi_reshaped = y.reshape((nx2, nx2))
        dydt = A2_i @ Phi_reshaped
        return dydt.flatten()  
    
    Phi_A2_0 = np.eye(nx2).flatten()
    result_PhiA2 = solve_ivp(ode_Phi_A2, t_span2, Phi_A2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    Phi_A2_t = result_PhiA2.y.reshape((nx2, nx2, -1))
    Phi_A2_t = np.moveaxis(Phi_A2_t, 2, 0)
    Phi_A2 = Phi_A2_t[-1]

    # --------------------- compute S_1, S_2 --------------------
    def ode_Phi_S1(t, y):
        # i = min(int(t / dt), nt1-1)  
        i = np.argmin(np.abs(tout-t))
        A1_i = A1[i]
        B1_i = B1[i]
        S1_reshaped = y.reshape((nx1, nx1))
        dydt = B1_i@B1_i.T + A1_i @ S1_reshaped + S1_reshaped@A1_i.T

        return dydt.flatten()  
    
    S1_0 = np.zeros((nx1, nx1)).flatten()
    
    # Solve ODE
    result_S1 = solve_ivp(ode_Phi_S1, t_span1, S1_0, method='RK23', t_eval=t_eval1)

    # Reshape the result to get the solution matrices at each time step
    S1_t = result_S1.y.reshape((nx1, nx1, -1))
    S1_t = np.moveaxis(S1_t, 2, 0)
    S1 = S1_t[-1]

    inv_S1 = np.linalg.inv(S1)

    def ode_Phi_S2(t, y):
        # i = min(int(t / dt), nt2-1)  
        i = np.argmin(np.abs(tout-t))
        A2_i = A2[i]
        B2_i = B2[i]
        S2_reshaped = y.reshape((nx2, nx2))
        dydt = B2_i@B2_i.T + A2_i @ S2_reshaped  + S2_reshaped@A2_i.T 
        return dydt.flatten()  
    
    S2_0 = np.zeros((nx2, nx2)).flatten()

    # Solve ODE
    result_S2 = solve_ivp(ode_Phi_S2, t_span2, S2_0, method='RK23', t_eval=t_eval2)

    # Reshape the result to get the solution matrices at each time step
    S2_t = result_S2.y.reshape((nx2, nx2, -1))
    S2_t = np.moveaxis(S2_t, 2, 0)
    S2 = S2_t[-1]
    inv_S2 = np.linalg.inv(S2)


    # --------------------- optimization formulation ---------------------
    import cvxpy as cp
    Sig0 = 0.01 * np.ones((nx1, nx1))
    SigT = 0.01 * np.ones((nx1, nx1))
    epsilon = 0.01

    Saltation = np.array(saltations[1])

    # ---------- Declare variables ---------- 
    Sighat_minus, Sighat_plus = cp.Variable((nx1,nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)
    W1, W2  = cp.Variable((nx1,nx1)), cp.Variable((nx2,nx2))
    Y1, Y2 = cp.Variable((2*nx1,2*nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)

    E = Saltation
    
    Y1 = cp.bmat([[Sig0, W1.T], [W1, Sighat_minus]])
    slack_Y2 = cp.bmat([[Sighat_plus, W2.T], [W2, SigT-Y2]])
    
    obj_1 = cp.trace(inv_S1@Sighat_minus) - 2*cp.trace(Phi_A2.T@inv_S2@W2) - 2*cp.trace(Phi_A1.T@inv_S1@W1) + cp.trace(Phi_A2.T@inv_S2@Phi_A2@Sighat_plus)
    obj_2 = - epsilon*cp.log_det(Y1) - epsilon*cp.log_det(Y2)

    constraints = [Sighat_plus==E@Sighat_minus@E.T,
                    Y1>>0,
                    slack_Y2>>0,
                    Sighat_minus>>0,
                    Sighat_plus>>0
                    ]
    
    problem = cp.Problem(cp.Minimize(obj_1+obj_2), constraints)
    print(" -------------- Problem is DCP -------------- :", problem.is_dcp())

    problem.solve()
    Sig_minus_opt = Sighat_minus.value
    Sig_plus_opt = Sighat_plus.value

    return tout, xout, t_events, x_events, saltations




if __name__ == '__main__':
    tout, xout, t_events, x_events, saltations = h_cs_3link()