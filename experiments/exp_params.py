import pickle as pkl

class ExpParams():
    def __init__(self):
        self._current_mode = None
        self._nstates = []
        self._init_state = []
        self._target_state = []
        self._start_time = []
        self._end_time = []
        self._dt = []
        self._initial_guess = []
        self._epsilon = []
        self._n_exp = []
        self._n_samples = []
        self._Q_k = []
        self._R_k = []
        self._Q_T = []
        self._symbolic_dyn = None
        self._detection_func = None
        self._nmodes = None
                                 
    def update_params(self, nmodes, init_mode, target_mode, n_states, init_state, target_state, 
                      start_time, end_time, dt, initial_guess, epsilon, 
                      n_exp, n_samples, Q_k, R_k, Q_T, 
                      symbolic_dyn, detection_func, plotting_func, state_convert_func, 
                      init_reset_args, target_reset_args, animate_function=None):
        self._nmodes = nmodes
        self._nstates = n_states
        self._current_mode = init_mode
        self._target_mode = target_mode
        self._init_state = init_state
        self._target_state = target_state
        self._start_time = start_time
        self._end_time = end_time
        self._dt = dt
        self._initial_guess = initial_guess
        self._epsilon = epsilon
        self._n_exp = n_exp
        self._n_samples = n_samples
        self._Q_k = Q_k
        self._R_k = R_k
        self._Q_T = Q_T
        self._symbolic_dyn = symbolic_dyn
        self._detection_func = detection_func
        self._plotting_func = plotting_func
        self.state_converters_ = state_convert_func
        self._init_reset_args = init_reset_args
        self._target_reset_args = target_reset_args
        self._animate_function = animate_function
        
    def nmodes(self):
        return self._nmodes
    
    def current_mode(self):
        return self._current_mode
        
    def symbolic_dynamics(self):
        return self._symbolic_dyn
    
    def detection_func(self):
        return self._detection_func
    
    def plotting_function(self):
        return self._plotting_func
    
    def state_convert_function(self):
        return self.state_converters_
    
    def animate_function(self):
        return self._animate_function

class ExpData():
    def __init__(self, params):
        self._data = {}
        self._data['params'] = params
    
    def add_data(self, iter, iter_data):
        self._data[str(iter)] = iter_data

    def add_nominal_data(self, ilqr_solution):
        self._data['nominal'] = ilqr_solution
        
    def add_plotting_function(self, plotting_func):
        self._data['plotting_func'] = plotting_func
        
    def dump(self, file_name):
        with open(file_name, 'wb') as f:
            pkl.dump(self._data, f, pkl.HIGHEST_PROTOCOL)
            
    def load(self, file_name):
        with open(file_name, 'rb') as f:
            self._data = pkl.load(f)
            
    def get_data(self, iter):
        return self._data[str(iter)]
    
    def get_params(self):
        return self._data['params']
    
    def get_nominal_data(self):
        if 'nominal' in self._data.keys():
            return self._data['nominal']
        else:
            return []
    
    def get_plotting_function(self):
        if 'plotting_func' in self._data.keys():
            return self._data['plotting_func']
        else:
            return []


class DataOneSample():
    def __init__(self, mode_trj_pi, x_trj_pi, u_trj_pi, 
                 mode_trj_ilqr, x_trj_ilqr, u_trj_ilqr, 
                 allPathCosts, cost_pi, cost_ilqr, all_samples=None, allPathCosts_uncoupled=None):
        """ The data to save for a path integral control in [0, T].
        Args:
            mode_trj_pi (_type_): mode of the controlled path integral trajectory
            x_trj_pi (_type_): controlled state trajectory
            u_trj_pi (_type_): path integral controller
            mode_trj_ilqr (_type_): mode of the controlled i-lqr trajectory
            x_trj_ilqr (_type_): ilqr controlled state trajectory
            u_trj_ilqr (_type_): ilqr controller
            allPathCosts (_type_): All the PathCosts used. shape: [nt, n_samples]
        """
        self._mode_trj_pi = mode_trj_pi
        self._x_trj_pi = x_trj_pi
        self._u_trj_pi = u_trj_pi
        self._mode_trj_ilqr = mode_trj_ilqr
        self._x_trj_ilqr = x_trj_ilqr
        self._u_trj_ilqr = u_trj_ilqr
        self._allPathCosts = allPathCosts
        self._cost_pi = cost_pi
        self._cost_ilqr = cost_ilqr
        self._all_samples = all_samples
        self._allPathCosts_uncoupled = allPathCosts_uncoupled
    
    def mode_trj_pi(self):
        return self._mode_trj_pi
    
    def mode_trj_ilqr(self):
        return self._mode_trj_ilqr
    
    def x_trj_pi(self):
        return self._x_trj_pi
    
    def u_trj_pi(self):
        return self._u_trj_pi
    
    def x_trj_ilqr(self):
        return self._x_trj_ilqr
    
    def u_trj_ilqr(self):
        return self._u_trj_ilqr
    
    def allPathCosts(self):
        return self._allPathCosts
    
    def allPathCosts_uncoupled(self):
        return self._allPathCosts_uncoupled
    
    def cost_pi(self):
        return self._cost_pi
    
    def cost_ilqr(self):
        return self._cost_ilqr
    
    def all_samples(self):
        return self._all_samples
    
    