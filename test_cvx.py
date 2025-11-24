import cvxpy as cp
import numpy as np

if __name__ == '__main__':

    nx1 = 4
    nx2 = 4
    Sighat_minus, Sighat_plus = cp.Variable((nx1,nx1), symmetric=True), cp.Variable((nx2,nx2), symmetric=True)
    var_t, T, Z, U = cp.Variable((1,1)), cp.Variable((nx2, nx2)), cp.Variable((nx2, nx2)), cp.Variable((nx2, nx2))
    
    W1, W2  = cp.Variable((nx1,nx1)), cp.Variable((nx2,nx2))
    
    E = cp.Parameter((nx2, nx1))
    I = cp.Parameter((nx2, nx2))
    Sig0, SigT = cp.Parameter((nx1, nx1), PSD=True), cp.Parameter((nx2, nx2), PSD=True)
    S1, S2 = cp.Parameter((nx1, nx1), PSD=True), cp.Parameter((nx2, nx2), PSD=True)
    Phi1, Phi2 = cp.Parameter((nx1, nx1)), cp.Parameter((nx1, nx1))

    controlled_Sig_0_T = cp.bmat([[Sighat_plus, W2.T], [W2, SigT]])
    prior_Sig_0_T = cp.bmat([[Sig0, W1.T], [W1, Sighat_minus]])

    slack_matrix_1 = cp.bmat([[U, W2], [W2.T, Sighat_plus]])

    # obj_1 = cp.trace(S1@Sighat_minus) - 2*cp.trace(Phi2.T@S2@W2) - 2*cp.trace(Phi1.T@S1@W1) + cp.trace(Phi2.T@S2@Phi2@Sighat_plus)
    obj_2 = -cp.log_det(controlled_Sig_0_T) - cp.log_det(prior_Sig_0_T)
    obj_3 = cp.log_det(I-U)

    constraints = [Sighat_plus==E@Sighat_minus@E.T,
                    controlled_Sig_0_T>>0,
                    prior_Sig_0_T>>0,
                    slack_matrix_1>>0,
                    U>>0]
    
    # prob = cp.Problem(cp.Minimize(obj_2), constraints)
    prob = cp.Problem(cp.Maximize(obj_3 - obj_2) , constraints)

    print("prob is DCP:", prob.is_dcp())
    # prob.solve()
