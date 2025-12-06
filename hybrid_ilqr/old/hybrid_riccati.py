# hybrid Riccati equation
import numpy as np

import os
import sys
file_path = os.path.abspath(__file__)
script_filename = os.path.splitext(os.path.basename(file_path))[0]
current_dir = os.path.dirname(file_path)
root_dir = os.path.abspath(os.path.join(current_dir, '..'))
sys.path.append(root_dir)

# Import pendulum dynamics
from dynamics.dynamics_bouncing import *
# Import experiment parameter class
from experiments.exp_params import *

class hybrid_riccati():
    def __init__(self,init_state,target_state,init_K,dt,start_time,end_time,contact_detect_func,f_contin,A,B,Q_k,R_k,Q_T,parameters) -> None:
        self.init_state_ = init_state
        self.target_state_ = target_state
        self.n_inputs_, self.n_states_ = np.shape(init_K)
        
        self.dt_ = dt
        self.start_time_ = start_time
        self.end_time_ = end_time
        self.time_span_ = np.arange(start_time, end_time, dt).flatten()
        
        self.n_timesteps_ = np.shape(self.time_span_)[0]
        self.saltations_ = [None for i in range(self.n_timesteps_)]
        
        self.states_ = np.zeros((self.n_timesteps_, self.n_states_))
        self.inputs_ = np.zeros((self.n_timesteps_, self.n_inputs_))
        
        # Dynamics
        self.f_ = f_contin
        self._detection = contact_detect_func
        self.A_ = A
        self.B_ = B
        
        # Weighting
        self.Q_k_ = Q_k
        self.R_k_ = R_k
        self.Q_T_ = Q_T
        self.parameters_ = parameters
        
        # Hamiltonian system matrix
        self.M_ = np.zeros((2*self.n_states_, 2*self.n_states_))
        
        # value function
        self.PI_ = np.zeros((self.n_timesteps_, self.n_states_, self.n_states_))
        self.q_ = np.zeros((self.n_timesteps_, self.n_states_))
        
        # state feedback control
        self.K_ = np.zeros((self.n_timesteps_, self.n_inputs_, self.n_states_))
        self.k_ = np.zeros((self.n_timesteps_, self.n_inputs_))
        
        # first rollout
        self.rollout()
        
    def compute_cost(self,states,inputs,dt):
        print("self.R_k_", self.R_k_)
        print("self.Q_T_", self.Q_T_)
        print("self.n_timesteps_", self.n_timesteps_)
        print("self.target_state_", self.target_state_)
        print("dt", dt)
        # Initialize cost
        total_cost = 0.0
        for ii in range(0,self.n_timesteps_-1):
            current_x = states[ii,:] # Not being used currently
            current_u = inputs[ii,:].flatten()

            current_cost = 0.5*current_u.T@self.R_k_@current_u # Right now only considering cost in input
            total_cost = total_cost+current_cost*dt
        # Compute terminal cost
        terminal_difference = (self.target_state_-states[-1]).flatten()
        terminal_cost = 0.5*terminal_difference.T@self.Q_T_@terminal_difference
        total_cost = total_cost+terminal_cost
        return total_cost
    
    def rollout(self):
        states = np.zeros((self.n_timesteps_, self.n_states_))
        inputs = np.zeros((self.n_timesteps_, self.n_inputs_))
        saltations = [None for i in range(self.n_timesteps_)]
        current_state = self.init_state_
        states[0] = current_state

        for ii in range(self.n_timesteps_-1):
            current_input = self.inputs_[ii,:]
            t_ii = self.time_span_[ii]
            
            next_state, saltation = self._detection(current_state, current_input, t_ii, t_ii+self.dt_)
            saltations[ii] = saltation
            next_state = next_state.flatten()
            
            # next_state = self.f_(current_state, current_input, self.dt_).flatten()
            # Store states and inputs
            states[ii + 1,:] = next_state
            inputs[ii,:] = current_input # in case we have a control law, we store the input used
            # Update the current state
            current_state = next_state
        
        # Store the trajectory(states, inputs)
        self.states_ = states
        self.inputs_ = inputs
        self.saltations_ = saltations
        return states, inputs, saltations
            
    def solve(self):
        # for loop backwards in time
        inv_R_k = np.linalg.inv(self.R_k_)  
        self.PI_[self.n_timesteps_-1] = self.Q_T_
        
        X_Y = np.zeros((2*self.n_states_, self.n_states_))
        X_Y[0:self.n_states_,:] = np.eye(self.n_states_)
        X_Y[self.n_states_:2*self.n_states_,:] = self.Q_T_
        
        for idx in reversed(range(0, self.n_timesteps_-1)):
            # Grab the current variables in the trajectory
            current_x = self.states_[idx,:]
            current_u = self.inputs_[idx,:]
            saltation = self.saltations_[idx]

            # Get the M matrix components
            A_k = self.A_(current_x, current_u)
            B_k = self.B_(current_x, current_u)
            
            self.M_[0:self.n_states_, 0:self.n_states_] = A_k
            self.M_[0:self.n_states_, self.n_states_:2*self.n_states_] = -B_k@inv_R_k@B_k.T
            self.M_[self.n_states_:2*self.n_states_, 0:self.n_states_] = -self.Q_k_
            self.M_[self.n_states_:2*self.n_states_, self.n_states_:2*self.n_states_] = -A_k.T
            
            X_Y = X_Y - (self.M_@X_Y)*self.dt_
            # X_Y = scipy.linalg.expm(-self.M_*self.dt_)@X_Y
            X = X_Y[0:self.n_states_]
            Y = X_Y[self.n_states_:2*self.n_states_]
            self.PI_[idx] = Y@np.linalg.inv(X)
            
        self.q_[self.n_timesteps_-1] = -self.Q_T_@self.target_state_
        for idx in reversed(range(1, self.n_timesteps_)):
            current_x = self.states_[idx,:]
            current_u = self.inputs_[idx,:]
            
            # Get the linearization
            A_k = self.A_(current_x, current_u)
            B_k = self.B_(current_x, current_u)
            
            # self.q_[idx-1] = scipy.linalg.expm((A_k.T - self.PI_[idx]@B_k@inv_R_k@B_k.T)*self.dt_)@self.q_[idx]
            self.q_[idx-1] = self.q_[idx] + (A_k.T - self.PI_[idx]@B_k@inv_R_k@B_k.T)@self.q_[idx]*self.dt_
            
        # solve for the optimal states and controls
        self.states_ = np.zeros((self.n_timesteps_, self.n_states_))
        self.inputs_ = np.zeros((self.n_timesteps_, self.n_inputs_))
        self.states_[0] = self.init_state_
        
        for idx in range(self.n_timesteps_-1):
            # Get the M matrix components
            A_k = self.A_(self.states_[idx], self.inputs_[idx])
            B_k = self.B_(self.states_[idx], self.inputs_[idx])
            
            self.K_[idx] = -inv_R_k@B_k.T@self.PI_[idx]
            self.k_[idx] = -inv_R_k@B_k.T@self.q_[idx]
            
            self.inputs_[idx] = self.K_[idx]@self.states_[idx] + self.k_[idx]
            g = 9.81
            A_cl = A_k + B_k@self.K_[idx]
            a_cl = B_k@self.k_[idx] + np.array([0, -g])
            # self.states_[idx+1] = scipy.linalg.expm(A_cl*self.dt_)@self.states_[idx] + a_cl*self.dt_
            self.states_[idx+1] = self.states_[idx] + (A_k@self.states_[idx] + B_k@self.inputs_[idx])*self.dt_
            
        print("self.states_[-1]", self.states_[-1,:])
        cost = self.compute_cost(self.states_, self.inputs_, self.dt_)
        print("total cost: ", cost)
        
        fig1, axes = plt.subplots(1, 2)
        (ax1, ax2) = axes.flatten()
        ax1.grid(True)
        ax2.grid(True)
        
        ax1.plot(self.states_[:,0], self.states_[:,1],'k',label='iLQR-deterministic')
        ax1.scatter(self.target_state_[0], self.target_state_[1], color='g', marker='x', s=50.0, linewidths=6, label='Target')
        ax1.scatter(self.init_state_[0], self.init_state_[1], color='r', marker='x', s=50.0, linewidths=6, label='Start')
    
        plt.show()
        
        return self.states_, self.inputs_, self.K_, self.k_, self.PI_, self.q_
    

def solve_riccati(params):
    # Import dynamics
    dynamis = params.symbolic_dynamics()
    detect_integration = params.detection_func()
    (f,A,B) = dynamis()
    
    # Initialize timings
    dt = params._dt
    
    start_time = params._start_time
    end_time = params._end_time
    time_span = np.arange(start_time, end_time, dt).flatten()

    # Set desired state
    n_states = len(params._init_state)
    n_inputs = params._R_k.shape[0]
    
    init_state = params._init_state  # Define the initial state to be the origin with no velocity
    target_state = params._target_state  # Swing pendulum upright

    # Initial guess of zeros, but you can change it to any guess
    init_K = np.zeros((n_inputs, n_states))

    # Define weighting matrices
    Q_k = params._Q_k # zero weight to penalties along a strajectory since we are finding a trajectory
    R_k = params._R_k

    # Set the terminal cost
    Q_T = params._Q_T

    # Set the physical parameters of the system
    mass = 1
    gravity = 9.8
    parameters = np.array([mass,gravity])

    # # Specify max number of iterations
    # n_iterations = 1000

    hybrid_riccati_ = hybrid_riccati(init_state,target_state,init_K,dt,start_time,end_time,detect_integration,f,A,B,Q_k,R_k,Q_T,parameters)
    (states, inputs, K, k, PI, q) = hybrid_riccati_.solve()

    return (states, inputs, K, k, PI, q)

    
if __name__ == '__main__':
    # === ilqr parameters ===
    # Initialize timings
    dt = 0.01
    # dt_pathintegral = dt / 50.0
    dt_pathintegral = dt

    # Set desired state
    n_states = 2
    n_inputs = 1
    
    # Define weighting matrices
    Q_k = np.zeros((n_states,n_states)) # zero weight to penalties along a strajectory since we are finding a trajectory
    # R_k = 0.01*np.eye(n_inputs)
    R_k = np.eye(n_inputs)

    # Set the terminal cost
    Q_T = 20*np.eye(n_states)
    Q_T[0,0] = 200.0
    
    # === path integral parameters ===
    epsilon = 1
    n_samples = 10
    n_exp = 5
    
    # ------------- verification with no contact ------------- 
    start_time = 0
    end_time = 0.75
    time_span = np.arange(start_time, end_time, dt).flatten()
    nt = len(time_span)
    
    init_state = np.array([5, 1.5])    # Define the initial state to be the origin with no velocity
    target_state = np.array([4.0, -5.0])  # Swing pendulum upright
    
    # ------------- /verification with no contact ------------- 
    
    # === Do N experiments and compare the expected costs ===
    cost_pi_exp = np.zeros(n_exp)
    cost_ilqr_exp = np.zeros(n_exp)

    # Horizon
    nt_ode_solve = 1000 # number of points used to solve the ode
    
    exp_params = ExpParams()
    exp_params.update_params(init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_exp, n_samples, Q_k, R_k, Q_T)
    
    exp_data = ExpData(exp_params)
    
    # === solve for ilqr ===
    (states, inputs, K, k, PI) = solve_riccati(exp_params)
    
    