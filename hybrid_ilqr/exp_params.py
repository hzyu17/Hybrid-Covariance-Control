import pickle as pkl

class ExpParams():
    def __init__(self):
        self._init_state = []
        self._target_state = []
        self._start_time = []
        self._end_time = []
        self._dt = []
        self._dt_pathintegral = []
        self._epsilon = []
        self._n_exp = []
        self._n_samples = []
        self._Q_k = []
        self._R_k = []
        self._Q_T = []
        self._symbolic_dyn = None
        self._detection_func = None
        
    def update_params(self, init_state, target_state, start_time, end_time, dt, dt_pathintegral, epsilon, n_exp, n_samples, Q_k, R_k, Q_T, symbolic_dyn, detection_func):
        self._init_state = init_state
        self._target_state = target_state
        self._start_time = start_time
        self._end_time = end_time
        self._dt = dt
        self._dt_pathintegral = dt_pathintegral
        self._epsilon = epsilon
        self._n_exp = n_exp
        self._n_samples = n_samples
        self._Q_k = Q_k
        self._R_k = R_k
        self._Q_T = Q_T
        self._symbolic_dyn = symbolic_dyn
        self._detection_func = detection_func
        
    def symbolic_dynamics(self):
        return self._symbolic_dyn
    
    def detection_func(self):
        return self._detection_func

class ExpData():
    def __init__(self, params):
        self._data = {}
        self._data['params'] = params
    
    def add_data(self, iter, iter_data):
        self._data[str(iter)] = iter_data

    def add_nominal_data(self, ilqr_solution):
        self._data['nominal'] = ilqr_solution
        
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


class DataOneSample():
    def __init__(self, x_trj_pi, u_trj_pi, x_trj_ilqr, u_trj_ilqr, allPathCosts, cost_pi, cost_ilqr, allPathCosts_uncoupled=None):
        """ The data to save for a path integral control in [0, T].
        Args:
            x_trj_pi (_type_): controlled state trajectory
            u_trj_pi (_type_): path integral controller
            x_trj_ilqr (_type_): ilqr controlled state trajectory
            u_trj_ilqr (_type_): ilqr controller
            allPathCosts (_type_): All the PathCosts used. shape: [nt, n_samples]
        """
        self._x_trj_pi = x_trj_pi
        self._u_trj_pi = u_trj_pi
        self._x_trj_ilqr = x_trj_ilqr
        self._u_trj_ilqr = u_trj_ilqr
        self._allPathCosts = allPathCosts
        self._cost_pi = cost_pi
        self._cost_ilqr = cost_ilqr
        self._allPathCosts_uncoupled = allPathCosts_uncoupled
        
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