# LIPM Dynamics for Hybrid Covariance Steering
# Linear Inverted Pendulum Model for simplified walking

import numpy as np
import jax
import jax.numpy as jnp
from jax import jacfwd
from functools import partial

# ============================================
#           LIPM Parameters
# ============================================
class LIPMParams:
    def __init__(self, z0=1.0, g=9.81, step_length=0.3):
        self.z0 = z0              # CoM height (constant)
        self.g = g                # Gravity
        self.omega = np.sqrt(g / z0)  # Natural frequency
        self.step_length = step_length  # Nominal step length
        
# Global default parameters
DEFAULT_PARAMS = LIPMParams()

# ============================================
#           Continuous Dynamics
# ============================================
# State: x = [p_com, v_com]  (position and velocity of CoM)
# Control: u = [u_push]  (small ankle push, optional)
# Foot position p_foot is a parameter (fixed during stance)

def lipm_continuous_dynamics(x, u, p_foot, params=DEFAULT_PARAMS):
    """
    LIPM continuous dynamics during single support
    ẍ = ω²(x - p_foot) + u/m
    
    State: x = [position, velocity]
    Control: u = [push_force] (normalized by mass)
    """
    omega = params.omega
    
    p_com = x[0]
    v_com = x[1]
    
    # Dynamics: ẍ = ω²(x - p_foot) + u
    dp_com = v_com
    dv_com = omega**2 * (p_com - p_foot) + u[0]
    
    return jnp.array([dp_com, dv_com])


def lipm_dynamics_mode1(x, u, params=DEFAULT_PARAMS):
    """Mode 1: Left foot stance at origin (p_foot = 0)"""
    return lipm_continuous_dynamics(x, u, p_foot=0.0, params=params)


def lipm_dynamics_mode2(x, u, p_foot_right, params=DEFAULT_PARAMS):
    """Mode 2: Right foot stance at p_foot_right"""
    return lipm_continuous_dynamics(x, u, p_foot=p_foot_right, params=params)


# JAX-compiled Jacobians for linearization
@partial(jax.jit, static_argnums=(2,))
def get_A_matrix(x, u, p_foot, params=DEFAULT_PARAMS):
    """Get state Jacobian A = df/dx"""
    def dynamics(x):
        return lipm_continuous_dynamics(x, u, p_foot, params)
    return jacfwd(dynamics)(x)


@partial(jax.jit, static_argnums=(2,))
def get_B_matrix(x, u, p_foot, params=DEFAULT_PARAMS):
    """Get control Jacobian B = df/du"""
    def dynamics(u):
        return lipm_continuous_dynamics(x, u, p_foot, params)
    return jacfwd(dynamics)(u)


# Analytical A and B for LIPM (linear system!)
def get_A_analytical(params=DEFAULT_PARAMS):
    """LIPM A matrix (constant for linear system)"""
    omega = params.omega
    return np.array([[0, 1],
                     [omega**2, 0]])


def get_B_analytical(params=DEFAULT_PARAMS):
    """LIPM B matrix (constant for linear system)"""
    return np.array([[0],
                     [1]])


# ============================================
#           Guard Functions
# ============================================

def guard_foot_switch(x, p_foot, switch_offset, params=DEFAULT_PARAMS):
    """
    Guard function for foot switch
    Triggers when CoM passes the stance foot by switch_offset
    g(x) = x[0] - p_foot - switch_offset = 0
    """
    return x[0] - p_foot - switch_offset


def guard_mode1_to_mode2(x, params=DEFAULT_PARAMS):
    """Guard for switching from left stance to right stance"""
    # Switch when CoM is at step_length/2 past left foot (at origin)
    switch_offset = params.step_length / 2
    return guard_foot_switch(x, p_foot=0.0, switch_offset=switch_offset, params=params)


def guard_mode2_to_mode1(x, p_foot_right, params=DEFAULT_PARAMS):
    """Guard for switching from right stance to left stance"""
    switch_offset = params.step_length / 2
    return guard_foot_switch(x, p_foot=p_foot_right, switch_offset=switch_offset, params=params)


# ============================================
#           Reset Maps
# ============================================

def reset_foot_switch(x_minus, p_foot_new):
    """
    Reset map at foot switch
    For LIPM, the CoM state is continuous (no jump)
    x⁺ = x⁻
    
    But we track the new foot position separately
    """
    return x_minus.copy()


def compute_new_foot_position(x_minus, step_length):
    """
    Compute new foot position based on step length
    Place new foot at step_length ahead of CoM
    """
    return x_minus[0] + step_length / 2


# ============================================
#           Saltation Matrix
# ============================================

def compute_saltation_matrix(x_minus, p_foot_minus, p_foot_plus, params=DEFAULT_PARAMS):
    """
    Compute saltation matrix for LIPM foot switch
    
    E = ∂Δ/∂x⁻ + (f⁺ - ∂Δ/∂x⁻ · f⁻) · (∂g/∂x)ᵀ / (∂g/∂x · f⁻)
    
    For LIPM:
    - Reset map Δ(x) = x (identity), so ∂Δ/∂x = I
    - Guard g(x) = x[0] - p_foot - offset, so ∂g/∂x = [1, 0]
    
    The saltation accounts for the timing uncertainty at the guard surface.
    """
    omega = params.omega
    nx = 2
    
    # Jacobian of reset map (identity for continuous state)
    dDelta_dx = np.eye(nx)
    
    # Guard gradient
    dg_dx = np.array([1.0, 0.0])
    
    # Vector field before transition (at x_minus with p_foot_minus)
    f_minus = np.array([x_minus[1], 
                        omega**2 * (x_minus[0] - p_foot_minus)])
    
    # Vector field after transition (at x_plus = x_minus, with p_foot_plus)
    f_plus = np.array([x_minus[1], 
                       omega**2 * (x_minus[0] - p_foot_plus)])
    
    # Lie derivative of guard along f_minus
    Lg = np.dot(dg_dx, f_minus)
    
    if np.abs(Lg) < 1e-10:
        print("Warning: Guard is tangent to flow, saltation may be singular")
        return dDelta_dx
    
    # Saltation matrix
    # E = dDelta_dx + (f_plus - dDelta_dx @ f_minus) @ dg_dx.T / Lg
    jump_term = (f_plus - dDelta_dx @ f_minus).reshape(-1, 1) @ dg_dx.reshape(1, -1) / Lg
    E = dDelta_dx + jump_term
    
    return E


# ============================================
#           Simulation Utilities
# ============================================

def simulate_lipm_step(x0, u_trajectory, dt, p_foot, params=DEFAULT_PARAMS):
    """
    Simulate LIPM for one stance phase
    Uses Euler integration (could upgrade to RK4)
    """
    nt = len(u_trajectory)
    x_traj = np.zeros((nt + 1, 2))
    x_traj[0] = x0
    
    for i in range(nt):
        u_i = u_trajectory[i]
        dx = lipm_continuous_dynamics(jnp.array(x_traj[i]), jnp.array(u_i), p_foot, params)
        x_traj[i + 1] = x_traj[i] + dt * np.array(dx)
    
    return x_traj


def simulate_lipm_hybrid(x0, u_traj_mode1, u_traj_mode2, dt, params=DEFAULT_PARAMS):
    """
    Simulate full hybrid LIPM walking step
    Mode 1: Left stance -> Mode 2: Right stance
    """
    # Mode 1: Left foot at origin
    p_foot_left = 0.0
    x_traj_1 = simulate_lipm_step(x0, u_traj_mode1, dt, p_foot_left, params)
    
    x_minus = x_traj_1[-1]
    
    # Compute new foot position
    p_foot_right = compute_new_foot_position(x_minus, params.step_length)
    
    # Reset (continuous state for LIPM)
    x_plus = reset_foot_switch(x_minus, p_foot_right)
    
    # Mode 2: Right foot stance
    x_traj_2 = simulate_lipm_step(x_plus, u_traj_mode2, dt, p_foot_right, params)
    
    return x_traj_1, x_traj_2, p_foot_right


def detect_guard_crossing(x_traj, p_foot, params=DEFAULT_PARAMS):
    """
    Detect when guard is crossed in trajectory
    Returns index of crossing or -1 if no crossing
    """
    switch_offset = params.step_length / 2
    
    for i in range(len(x_traj) - 1):
        g_i = guard_foot_switch(x_traj[i], p_foot, switch_offset, params)
        g_ip1 = guard_foot_switch(x_traj[i + 1], p_foot, switch_offset, params)
        
        if g_i <= 0 and g_ip1 > 0:
            return i + 1
    
    return -1


# ============================================
#           Analytical Solutions
# ============================================

def lipm_analytical_solution(x0, t, p_foot, params=DEFAULT_PARAMS):
    """
    Analytical solution for uncontrolled LIPM
    x(t) = (x0 - p) cosh(ωt) + (v0/ω) sinh(ωt) + p
    v(t) = (x0 - p) ω sinh(ωt) + v0 cosh(ωt)
    """
    omega = params.omega
    p = p_foot
    
    x0_rel = x0[0] - p
    v0 = x0[1]
    
    x_t = x0_rel * np.cosh(omega * t) + (v0 / omega) * np.sinh(omega * t) + p
    v_t = x0_rel * omega * np.sinh(omega * t) + v0 * np.cosh(omega * t)
    
    return np.array([x_t, v_t])


def lipm_state_transition_matrix(t, params=DEFAULT_PARAMS):
    """
    State transition matrix for LIPM (homogeneous part)
    Φ(t) = [[cosh(ωt), sinh(ωt)/ω],
            [ω·sinh(ωt), cosh(ωt)]]
    """
    omega = params.omega
    
    c = np.cosh(omega * t)
    s = np.sinh(omega * t)
    
    return np.array([[c, s / omega],
                     [omega * s, c]])


def lipm_controllability_gramian(t, params=DEFAULT_PARAMS):
    """
    Controllability Gramian for LIPM over time horizon [0, t]
    W(t) = ∫₀ᵗ Φ(τ) B Bᵀ Φ(τ)ᵀ dτ
    
    For LIPM this has closed-form solution
    """
    omega = params.omega
    B = get_B_analytical(params)
    
    # Numerical integration for now (could derive analytical)
    n_steps = 100
    tau = np.linspace(0, t, n_steps)
    dt = t / n_steps
    
    W = np.zeros((2, 2))
    for i in range(n_steps):
        Phi_tau = lipm_state_transition_matrix(tau[i], params)
        W += Phi_tau @ B @ B.T @ Phi_tau.T * dt
    
    return W


# ============================================
#           Visualization
# ============================================

def plot_lipm_trajectory(x_traj, p_feet, ax=None, color='blue', label=None):
    """Plot LIPM trajectory with foot positions"""
    import matplotlib.pyplot as plt
    
    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    
    # Plot CoM trajectory
    ax.plot(x_traj[:, 0], np.zeros_like(x_traj[:, 0]) + 1.0, 
            color=color, linewidth=2, label=label)
    
    # Plot foot positions
    for p_foot in p_feet:
        ax.plot([p_foot, p_foot], [0, 0.1], 'k-', linewidth=3)
        ax.scatter(p_foot, 0, marker='s', s=100, c='black')
    
    ax.set_xlabel('x (m)')
    ax.set_ylabel('z (m)')
    ax.set_ylim(-0.1, 1.5)
    ax.grid(True)
    
    return ax


if __name__ == "__main__":
    # Test LIPM dynamics
    params = LIPMParams(z0=1.0, step_length=0.4)
    
    print("LIPM Parameters:")
    print(f"  z0 = {params.z0} m")
    print(f"  omega = {params.omega:.3f} rad/s")
    print(f"  step_length = {params.step_length} m")
    
    # Test analytical matrices
    A = get_A_analytical(params)
    B = get_B_analytical(params)
    print("\nA matrix:")
    print(A)
    print("\nB matrix:")
    print(B)
    
    # Test simulation
    x0 = np.array([-0.1, 0.5])  # Start behind foot, moving forward
    dt = 0.001
    nt = 200
    u_traj = np.zeros((nt, 1))
    
    x_traj = simulate_lipm_step(x0, u_traj, dt, p_foot=0.0, params=params)
    
    print(f"\nSimulated {nt} steps:")
    print(f"  Initial state: {x0}")
    print(f"  Final state: {x_traj[-1]}")
    
    # Compare with analytical
    t_final = nt * dt
    x_analytical = lipm_analytical_solution(x0, t_final, p_foot=0.0, params=params)
    print(f"  Analytical final: {x_analytical}")
    
    # Test saltation matrix
    x_minus = np.array([0.2, 0.8])
    E = compute_saltation_matrix(x_minus, p_foot_minus=0.0, 
                                  p_foot_plus=0.4, params=params)
    print("\nSaltation matrix at foot switch:")
    print(E)
