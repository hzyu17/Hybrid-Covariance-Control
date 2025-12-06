# Definition of a saltation matrix.
# 02/06/2024

import numpy as np

def compute_saltation(F1, F2, Rt, Rx, gt, gx):
    """
    Definition of a saltation matrix, given the jacobians it needs to compute.

    Parameters:
    - F1 (vector(n1)): The velocity field of the first mode.
    - F2 (vector(n2)): The velocity field of the second mode.
    - Rt (vector(n2)): The time derivative of the Reset Map.
    - Rx (matrix(n2, n1)): The partial derivative of Reset Map wrt the state.
    - gt (scalar): The time derivative of the guard.
    - gx (vector(n1)): The partial derivative of the guard wrt the state.
    
    Returns:
    SaltM (matrix(n2, n1)): Saltation Matrix.
    """

    return Rx + np.outer((F2 - Rx@F1 - Rt), gx) / (gt + np.dot(gx, F1))