# Multi-Step Hybrid Covariance Steering for LIPM Walking
# Simulates realistic walking with multiple foot switches

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
from scipy.linalg import solve_continuous_are, expm

np.set_printoptions(suppress=True, precision=4)


class LIPMWalker:
    """
    Linear Inverted Pendulum Model for walking with hybrid covariance steering.
    
    State: x = [position, velocity] of CoM
    Control: u = ankle torque (normalized)
    
    Dynamics: ẍ = ω²(x - p_foot) + u
    where ω = sqrt(g/z0)
    """
    
    def __init__(self, z0=1.0, g=9.81, step_length=0.25):
        # Physical parameters
        self.z0 = z0
        self.g = g
        self.omega = np.sqrt(g / z0)
        self.step_length = step_length
        
        # State-space matrices (constant for LIPM)
        self.A = np.array([[0, 1],
                          [self.omega**2, 0]])
        self.B = np.array([[0], [1]])
        self.nx = 2
        self.nu = 1
        
    def dynamics(self, x, u, p_foot):
        """Continuous LIPM dynamics"""
        x_rel = x[0] - p_foot
        dx = np.array([x[1], self.omega**2 * x_rel + u])
        return dx
    
    def state_transition(self, T):
        """Analytical state transition matrix Φ(T) = exp(AT)"""
        omega = self.omega
        c, s = np.cosh(omega * T), np.sinh(omega * T)
        return np.array([[c, s/omega], [omega*s, c]])
    
    def controllability_gramian(self, T, n_steps=100):
        """Compute controllability gramian W = ∫₀ᵀ Φ(τ)BBᵀΦ(τ)ᵀ dτ"""
        dt = T / n_steps
        W = np.zeros((2, 2))
        for i in range(n_steps):
            t = i * dt
            Phi_t = self.state_transition(t)
            W += Phi_t @ self.B @ self.B.T @ Phi_t.T * dt
        return W
    
    def compute_saltation_matrix(self, x_trans, p_foot_old, p_foot_new):
        """
        Compute saltation matrix at foot switch.
        
        For LIPM:
        - State is continuous (Δ(x) = x)
        - Guard: g(x) = x - p_foot - d = 0 (CoM passes threshold)
        - Dynamics change due to foot position change
        
        E = I + (f⁺ - f⁻) ⊗ ∇g / (∇g · f⁻)
        """
        omega = self.omega
        
        # Gradients
        dg_dx = np.array([1.0, 0.0])  # Guard gradient
        
        # Vector fields (no control at transition)
        f_minus = np.array([x_trans[1], omega**2 * (x_trans[0] - p_foot_old)])
        f_plus = np.array([x_trans[1], omega**2 * (x_trans[0] - p_foot_new)])
        
        # Lie derivative
        Lg = dg_dx @ f_minus
        
        if np.abs(Lg) < 1e-10:
            return np.eye(2)
        
        # Saltation matrix
        E = np.eye(2) + np.outer(f_plus - f_minus, dg_dx) / Lg
        
        return E


class MultiStepCovarianceSteering:
    """
    Hybrid covariance steering controller for multi-step LIPM walking.
    """
    
    def __init__(self, walker, n_steps=4, T_step=0.4, dt=0.002, epsilon=0.001):
        self.walker = walker
        self.n_steps = n_steps
        self.T_step = T_step
        self.dt = dt
        self.epsilon = epsilon
        
        self.nt_per_step = int(T_step / dt)
        
        # Foot positions
        self.foot_positions = [i * walker.step_length for i in range(n_steps + 1)]
        
        # Storage
        self.saltation_matrices = []
        self.x_ref_steps = []
        self.K_steps = []
        self.Sig_minus = []
        self.Sig_plus = []
        
    def design_walking_gait(self):
        """
        Design a stable periodic walking reference trajectory.
        Uses orbital energy concept for LIPM stability.
        
        Key insight: For stable LIPM walking, the CoM must have sufficient
        velocity when passing over the stance foot to reach the next step.
        """
        omega = self.walker.omega
        L = self.walker.step_length
        
        # For periodic walking, use orbital energy
        # E = 0.5*v² - 0.5*ω²*(x-p)²
        # At step boundaries (x = p ± L/2), we need v such that walker reaches next step
        
        # Orbital energy for symmetric gait
        # Start at x = p - L/2 with v = v_start
        # End at x = p + L/2 with v = v_end
        # For periodic: v_start = v_end (magnitude)
        
        # Minimum velocity to complete step (from energy conservation)
        v_min = omega * L / 2 * np.sqrt(2)  # sqrt(2) factor for margin
        v_nominal = v_min * 1.1  # 10% margin
        
        print("\n" + "=" * 60)
        print("Designing Walking Gait")
        print("=" * 60)
        print(f"  Step length: {L} m")
        print(f"  Minimum velocity: {v_min:.3f} m/s")
        print(f"  Nominal velocity: {v_nominal:.3f} m/s")
        
        # Clear previous data
        self.saltation_matrices = []
        self.x_ref_steps = []
        
        for step in range(self.n_steps):
            p_foot = self.foot_positions[step]
            p_foot_next = self.foot_positions[step + 1]
            
            # Start behind foot, end ahead of foot
            x_start = p_foot - L/2 * 0.8
            x_end_target = p_foot + L/2 * 0.9
            
            # Initial velocity (maintain energy)
            v_start = v_nominal
            
            # Target final velocity (for next step)
            v_end_target = v_nominal
            
            x_init = np.array([x_start, v_start])
            x_target = np.array([x_end_target, v_end_target])
            
            # Simulate with feedback control to track the target
            x_ref = np.zeros((self.nt_per_step + 1, 2))
            x_ref[0] = x_init
            
            # Compute stabilizing LQR gain
            Q_lqr = np.diag([5.0, 1.0])
            R_lqr = np.array([[0.5]])
            P_lqr = solve_continuous_are(self.walker.A, self.walker.B, Q_lqr, R_lqr)
            K_lqr = -np.linalg.inv(R_lqr) @ self.walker.B.T @ P_lqr
            
            for i in range(self.nt_per_step):
                x_i = x_ref[i]
                t_frac = i / self.nt_per_step
                
                # Interpolated target
                x_des = (1 - t_frac) * x_init + t_frac * x_target
                
                # Feedback control
                u = (K_lqr @ (x_i - x_des)).item()
                u = np.clip(u, -5, 5)
                
                # Integrate
                dx = self.walker.dynamics(x_i, u, p_foot)
                x_ref[i + 1] = x_i + self.dt * dx
            
            self.x_ref_steps.append(x_ref)
            
            # Compute saltation at transition
            x_trans = x_ref[-1]
            E = self.walker.compute_saltation_matrix(x_trans, p_foot, p_foot_next)
            self.saltation_matrices.append(E)
            
            print(f"\n  Step {step + 1}:")
            print(f"    Start: pos={x_ref[0, 0]:.3f}, vel={x_ref[0, 1]:.3f}")
            print(f"    End:   pos={x_ref[-1, 0]:.3f}, vel={x_ref[-1, 1]:.3f}")
            print(f"    Saltation E (off-diag): {E[1, 0]:.3f}")
            print(f"    det(E) = {np.linalg.det(E):.4f}")
    
    def compute_covariance_targets(self, Sig0, SigT):
        """
        Compute optimal intermediate covariances at each transition.
        
        Strategy: Distribute covariance reduction across steps while
        respecting the saltation constraints Σ⁺ = E Σ⁻ Eᵀ
        """
        print("\n" + "=" * 60)
        print("Computing Optimal Intermediate Covariances")
        print("=" * 60)
        
        n = self.n_steps
        
        # Compute total "covariance budget"
        trace_0 = np.trace(Sig0)
        trace_T = np.trace(SigT)
        
        print(f"  Initial trace: {trace_0:.6f}")
        print(f"  Target trace: {trace_T:.6f}")
        
        # Simple strategy: geometric interpolation of trace
        traces = np.geomspace(trace_0, trace_T, n + 1)
        
        self.Sig_minus = []
        self.Sig_plus = []
        
        Sig_current = Sig0.copy()
        
        for k in range(n):
            E_k = self.saltation_matrices[k]
            
            # Target trace at this transition
            target_trace = traces[k + 1]
            
            # Scale current covariance to match target trace
            current_trace = np.trace(Sig_current)
            if current_trace > 1e-10:
                scale = target_trace / current_trace
            else:
                scale = 1.0
            
            Sig_minus_k = Sig_current * np.sqrt(scale)
            Sig_minus_k = (Sig_minus_k + Sig_minus_k.T) / 2
            
            # Ensure positive definiteness
            evals, evecs = np.linalg.eigh(Sig_minus_k)
            evals = np.maximum(evals, 1e-8)
            Sig_minus_k = evecs @ np.diag(evals) @ evecs.T
            
            # Apply saltation
            Sig_plus_k = E_k @ Sig_minus_k @ E_k.T
            Sig_plus_k = (Sig_plus_k + Sig_plus_k.T) / 2
            
            # Ensure positive definiteness after saltation
            evals, evecs = np.linalg.eigh(Sig_plus_k)
            evals = np.maximum(evals, 1e-8)
            Sig_plus_k = evecs @ np.diag(evals) @ evecs.T
            
            self.Sig_minus.append(Sig_minus_k)
            self.Sig_plus.append(Sig_plus_k)
            
            print(f"\n  Transition {k + 1}:")
            print(f"    Σ⁻ trace = {np.trace(Sig_minus_k):.6f}")
            print(f"    Σ⁺ trace = {np.trace(Sig_plus_k):.6f}")
            
            # Update for next step
            Sig_current = Sig_plus_k
        
        return self.Sig_minus, self.Sig_plus
    
    def compute_feedback_gains(self, Sig0, SigT):
        """
        Compute feedback gains for each stance phase.
        Uses LQR with terminal covariance weighting.
        """
        print("\n" + "=" * 60)
        print("Computing Feedback Controllers")
        print("=" * 60)
        
        A, B = self.walker.A, self.walker.B
        
        # Covariance targets for each segment
        Sig_inits = [Sig0] + self.Sig_plus
        Sig_targets = self.Sig_minus + [SigT]
        
        self.K_steps = []
        
        for k in range(self.n_steps):
            Sig_init_k = Sig_inits[k]
            Sig_target_k = Sig_targets[k]
            
            # Use LQR with weights based on covariance targets
            # Higher Q when we need to reduce covariance
            trace_ratio = np.trace(Sig_init_k) / max(np.trace(Sig_target_k), 1e-8)
            q_weight = min(max(trace_ratio, 1.0), 20.0)
            
            Q = q_weight * np.eye(2)
            R = np.array([[1.0]])
            
            # Solve ARE
            try:
                P = solve_continuous_are(A, B, Q, R)
                K_lqr = -np.linalg.inv(R) @ B.T @ P
            except:
                # Fallback to simple stabilizing gain
                K_lqr = np.array([[-10.0, -5.0]])
            
            # Use constant gain for this step (simpler, more stable)
            K_k = np.tile(K_lqr, (self.nt_per_step + 1, 1, 1))
            
            self.K_steps.append(K_k)
            
            print(f"  Step {k + 1}: Q weight = {q_weight:.2f}, K = {K_lqr.flatten()}")
        
        return self.K_steps
    
    def _solve_cov_steering_gains(self, Sig0, SigT):
        """
        Solve for covariance steering feedback gains using the 
        Chen-Georgiou-Pavon formulation.
        """
        A = self.walker.A
        B = self.walker.B
        T = self.T_step
        nt = self.nt_per_step
        dt = self.dt
        nx, nu = self.walker.nx, self.walker.nu
        epsilon = self.epsilon
        
        # Regularization for numerical stability
        Q = 0.01 * np.eye(nx)
        R = np.eye(nu)
        
        # Hamiltonian matrix
        M = np.block([
            [A, -B @ np.linalg.inv(R) @ B.T],
            [-Q, -A.T]
        ])
        
        # Compute Φ_M(T)
        Phi_M = expm(M * T)
        Phi_11 = Phi_M[:nx, :nx]
        Phi_12 = Phi_M[:nx, nx:]
        
        # Regularize Phi_12 inversion
        U, s, Vt = np.linalg.svd(Phi_12)
        s_inv = np.where(s > 1e-8, 1/s, 0)
        inv_Phi_12 = Vt.T @ np.diag(s_inv) @ U.T
        
        # Covariance square roots
        Sig0_reg = Sig0 + 1e-8 * np.eye(nx)
        SigT_reg = SigT + 1e-8 * np.eye(nx)
        
        evals0, evecs0 = np.linalg.eigh(Sig0_reg)
        sqrt_Sig0 = evecs0 @ np.diag(np.sqrt(np.maximum(evals0, 1e-10))) @ evecs0.T
        inv_Sig0 = evecs0 @ np.diag(1.0 / np.maximum(evals0, 1e-10)) @ evecs0.T
        sqrt_inv_Sig0 = evecs0 @ np.diag(1.0 / np.sqrt(np.maximum(evals0, 1e-10))) @ evecs0.T
        
        # Compute initial Pi
        tmp = epsilon**2 * np.eye(nx) / 4 + sqrt_Sig0 @ inv_Phi_12 @ SigT_reg @ inv_Phi_12.T @ sqrt_Sig0
        tmp = (tmp + tmp.T) / 2
        
        evals_tmp, evecs_tmp = np.linalg.eigh(tmp)
        sqrt_tmp = evecs_tmp @ np.diag(np.sqrt(np.maximum(evals_tmp, 1e-10))) @ evecs_tmp.T
        
        Pi0 = epsilon * inv_Sig0 / 2 - inv_Phi_12 @ Phi_11 - sqrt_inv_Sig0 @ sqrt_tmp @ sqrt_inv_Sig0
        Pi0 = (Pi0 + Pi0.T) / 2
        Pi0 = np.clip(Pi0, -100, 100)
        
        # Integrate Pi(t) using X-Y formulation
        def ode_XY(t, y):
            XY = y.reshape((2 * nx, nx))
            return (M @ XY).flatten()
        
        XY0 = np.zeros((2 * nx, nx))
        XY0[:nx, :] = np.eye(nx)
        XY0[nx:, :] = Pi0
        
        t_eval = np.linspace(0, T, nt + 1)
        result = solve_ivp(ode_XY, (0, T), XY0.flatten(), 
                          t_eval=t_eval, method='RK45', max_step=dt)
        
        XY_traj = result.y.reshape((2 * nx, nx, -1))
        
        # Extract feedback gains
        K_traj = np.zeros((nt + 1, nu, nx))
        
        for i in range(nt + 1):
            X_i = XY_traj[:nx, :, i]
            Y_i = XY_traj[nx:, :, i]
            
            # Regularized inversion
            U, s, Vt = np.linalg.svd(X_i)
            s_inv = np.where(s > 1e-8, 1/s, 0)
            inv_X_i = Vt.T @ np.diag(s_inv) @ U.T
            
            Pi_i = Y_i @ inv_X_i
            Pi_i = (Pi_i + Pi_i.T) / 2
            
            K_i = -np.linalg.inv(R) @ B.T @ Pi_i
            K_traj[i] = np.clip(K_i, -50, 50)
        
        return K_traj
    
    def propagate_covariance(self, Sig0):
        """Propagate covariance through all walking steps."""
        print("\n" + "=" * 60)
        print("Propagating Covariance")
        print("=" * 60)
        
        self.Sig_traj_steps = []
        Sig_current = Sig0.copy()
        
        for k in range(self.n_steps):
            K_k = self.K_steps[k]
            
            # Propagate through stance phase
            Sig_traj_k = self._propagate_step(Sig_current, K_k)
            self.Sig_traj_steps.append(Sig_traj_k)
            
            print(f"  Step {k + 1}: trace {np.trace(Sig_current):.6f} -> {np.trace(Sig_traj_k[-1]):.6f}")
            
            # Apply saltation at transition
            E_k = self.saltation_matrices[k]
            Sig_current = E_k @ Sig_traj_k[-1] @ E_k.T
            Sig_current = (Sig_current + Sig_current.T) / 2
            
            # Ensure positive definite
            evals, evecs = np.linalg.eigh(Sig_current)
            Sig_current = evecs @ np.diag(np.maximum(evals, 1e-10)) @ evecs.T
            
            print(f"    After saltation: trace = {np.trace(Sig_current):.6f}")
        
        return self.Sig_traj_steps
    
    def _propagate_step(self, Sig0, K_traj):
        """Propagate covariance for one stance phase."""
        A, B = self.walker.A, self.walker.B
        nt = self.nt_per_step
        dt = self.dt
        epsilon = self.epsilon
        
        Sig_traj = np.zeros((nt + 1, 2, 2))
        Sig_traj[0] = Sig0
        
        for i in range(nt):
            K_i = K_traj[i]
            Acl = A + B @ K_i
            Sig_i = Sig_traj[i]
            
            # Covariance ODE: dΣ/dt = Acl·Σ + Σ·Aclᵀ + ε·B·Bᵀ
            dSig = Acl @ Sig_i + Sig_i @ Acl.T + epsilon * B @ B.T
            Sig_new = Sig_i + dt * dSig
            Sig_new = (Sig_new + Sig_new.T) / 2
            
            # Ensure positive definiteness
            evals, evecs = np.linalg.eigh(Sig_new)
            Sig_traj[i + 1] = evecs @ np.diag(np.maximum(evals, 1e-10)) @ evecs.T
        
        return Sig_traj
    
    def monte_carlo(self, x0_mean, Sig0, n_samples=200):
        """Run Monte Carlo validation."""
        print("\n" + "=" * 60)
        print(f"Monte Carlo Simulation ({n_samples} samples)")
        print("=" * 60)
        
        np.random.seed(42)
        
        # Sample initial states
        sqrt_Sig0 = np.linalg.cholesky(Sig0 + 1e-10 * np.eye(2))
        x0_samples = x0_mean + (sqrt_Sig0 @ np.random.randn(2, n_samples)).T
        
        trajectories = []
        xT_samples = np.zeros((n_samples, 2))
        
        for i in range(n_samples):
            x = x0_samples[i].copy()
            traj = [x.copy()]
            
            for k in range(self.n_steps):
                p_foot = self.foot_positions[k]
                K_k = self.K_steps[k]
                x_ref_k = self.x_ref_steps[k]
                
                for j in range(self.nt_per_step):
                    # Control with noise
                    x_err = x - x_ref_k[j]
                    u = (K_k[j] @ x_err).item()
                    u += np.sqrt(self.epsilon) * np.random.randn()
                    
                    # Dynamics
                    dx = self.walker.dynamics(x, u, p_foot)
                    x = x + self.dt * dx
                    traj.append(x.copy())
            
            trajectories.append(np.array(traj))
            xT_samples[i] = x
        
        # Statistics
        xT_mean = np.mean(xT_samples, axis=0)
        SigT_mc = np.cov(xT_samples.T)
        
        print(f"  Terminal mean: {xT_mean}")
        print(f"  Terminal covariance:\n{SigT_mc}")
        print(f"  Terminal trace: {np.trace(SigT_mc):.6f}")
        
        self.trajectories = trajectories
        self.xT_samples = xT_samples
        self.SigT_mc = SigT_mc
        
        return trajectories, xT_samples
    
    def plot_results(self, x0_mean, Sig0, SigT, save_path=None):
        """Comprehensive visualization."""
        fig = plt.figure(figsize=(16, 12))
        
        # ===== Plot 1: Phase space with covariance ellipses =====
        ax1 = fig.add_subplot(2, 2, 1)
        
        # Sample trajectories
        for traj in self.trajectories[:50]:
            ax1.plot(traj[:, 0], traj[:, 1], 'c-', alpha=0.15, linewidth=0.5)
        
        # Reference trajectory
        x_ref_full = np.vstack(self.x_ref_steps)
        ax1.plot(x_ref_full[:, 0], x_ref_full[:, 1], 'k-', linewidth=2, label='Reference')
        
        # Covariance ellipses
        self._plot_ellipse(x0_mean, Sig0, ax1, 'blue', label=r'$\Sigma_0$')
        
        for k in range(self.n_steps):
            x_trans = self.x_ref_steps[k][-1]
            alpha = 0.3 + 0.5 * k / self.n_steps
            self._plot_ellipse(x_trans, self.Sig_minus[k], ax1, 'orange', alpha=alpha)
            self._plot_ellipse(x_trans, self.Sig_plus[k], ax1, 'red', alpha=alpha)
        
        self._plot_ellipse(x_ref_full[-1], SigT, ax1, 'green', label=r'$\Sigma_T$ (target)')
        self._plot_ellipse(np.mean(self.xT_samples, axis=0), self.SigT_mc, ax1, 
                          'purple', alpha=0.3, label=r'$\Sigma_T$ (MC)')
        
        # Terminal samples
        ax1.scatter(self.xT_samples[:, 0], self.xT_samples[:, 1],
                   c='red', s=10, alpha=0.3)
        
        ax1.set_xlabel('Position (m)', fontsize=12)
        ax1.set_ylabel('Velocity (m/s)', fontsize=12)
        ax1.set_title(f'Multi-Step LIPM Walking ({self.n_steps} steps)', fontsize=14)
        ax1.legend(loc='upper left', fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # ===== Plot 2: Covariance trace evolution =====
        ax2 = fig.add_subplot(2, 2, 2)
        
        t_all = []
        trace_all = []
        
        for k in range(self.n_steps):
            t_step = np.linspace(k * self.T_step, (k + 1) * self.T_step, self.nt_per_step + 1)
            trace_step = [np.trace(self.Sig_traj_steps[k][i]) for i in range(self.nt_per_step + 1)]
            
            color = plt.cm.viridis(k / self.n_steps)
            ax2.plot(t_step, trace_step, color=color, linewidth=2, label=f'Step {k+1}')
            
            # Mark transitions
            ax2.axvline(x=(k + 1) * self.T_step, color='gray', linestyle='--', alpha=0.5)
            
            t_all.extend(t_step.tolist())
            trace_all.extend(trace_step)
        
        ax2.axhline(y=np.trace(SigT), color='green', linestyle=':', linewidth=2, label='Target')
        ax2.axhline(y=np.trace(self.SigT_mc), color='purple', linestyle=':', linewidth=2, label='MC result')
        
        ax2.set_xlabel('Time (s)', fontsize=12)
        ax2.set_ylabel(r'Trace($\Sigma$)', fontsize=12)
        ax2.set_title('Covariance Trace Evolution', fontsize=14)
        ax2.legend(loc='upper right', fontsize=9)
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # ===== Plot 3: Feedback gains =====
        ax3 = fig.add_subplot(2, 2, 3)
        
        for k in range(self.n_steps):
            t_step = np.linspace(k * self.T_step, (k + 1) * self.T_step, self.nt_per_step + 1)
            K_k = self.K_steps[k]
            
            color = plt.cm.viridis(k / self.n_steps)
            ax3.plot(t_step, K_k[:, 0, 0], color=color, linestyle='-', linewidth=1.5)
            ax3.plot(t_step, K_k[:, 0, 1], color=color, linestyle='--', linewidth=1.5)
        
        # Legend entries
        ax3.plot([], [], 'k-', label=r'$K^{pos}$')
        ax3.plot([], [], 'k--', label=r'$K^{vel}$')
        
        ax3.set_xlabel('Time (s)', fontsize=12)
        ax3.set_ylabel('Feedback Gain', fontsize=12)
        ax3.set_title('Time-Varying Feedback Gains', fontsize=14)
        ax3.legend(loc='best', fontsize=10)
        ax3.grid(True, alpha=0.3)
        
        # ===== Plot 4: Walking animation (side view) =====
        ax4 = fig.add_subplot(2, 2, 4)
        
        # Ground
        ax4.axhline(y=0, color='saddlebrown', linewidth=4)
        ax4.fill_between([-0.2, self.foot_positions[-1] + 0.3], -0.1, 0, 
                        color='saddlebrown', alpha=0.3)
        
        # Feet
        for i, p in enumerate(self.foot_positions[:-1]):
            color = 'blue' if i % 2 == 0 else 'red'
            ax4.plot([p - 0.03, p + 0.03], [0, 0], color=color, linewidth=8, solid_capstyle='round')
            ax4.text(p, -0.08, f'{i+1}', ha='center', fontsize=10)
        
        # CoM trajectories
        z0 = self.walker.z0
        for traj in self.trajectories[:30]:
            ax4.plot(traj[:, 0], np.ones(len(traj)) * z0, 'c-', alpha=0.15, linewidth=0.5)
        
        ax4.plot(x_ref_full[:, 0], np.ones(len(x_ref_full)) * z0, 'k-', linewidth=2)
        
        # Stick figures at key times
        n_figures = self.n_steps * 2
        total_points = len(x_ref_full)
        
        for idx in np.linspace(0, total_points - 1, n_figures).astype(int):
            x_com = x_ref_full[idx, 0]
            step_idx = min(int(idx / self.nt_per_step), self.n_steps - 1)
            p_foot = self.foot_positions[step_idx]
            
            # Leg
            ax4.plot([p_foot, x_com], [0, z0], 'k-', linewidth=1.5, alpha=0.4)
            # CoM
            ax4.scatter([x_com], [z0], s=40, c='black', zorder=5, alpha=0.5)
        
        # Start and end markers
        ax4.scatter([x0_mean[0]], [z0], s=150, c='blue', marker='o', zorder=10, label='Start')
        ax4.scatter([x_ref_full[-1, 0]], [z0], s=150, c='green', marker='*', zorder=10, label='Goal')
        
        ax4.set_xlabel('Position (m)', fontsize=12)
        ax4.set_ylabel('Height (m)', fontsize=12)
        ax4.set_title('Walking Visualization', fontsize=14)
        ax4.set_ylim(-0.15, 1.3)
        ax4.set_xlim(-0.2, self.foot_positions[-1] + 0.3)
        ax4.set_aspect('equal')
        ax4.legend(loc='upper left', fontsize=10)
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path + '.pdf', dpi=300, bbox_inches='tight')
            plt.savefig(save_path + '.png', dpi=150, bbox_inches='tight')
            print(f"\nFigures saved to {save_path}.pdf/png")
        
        plt.show()
        
        return fig
    
    def _plot_ellipse(self, mean, cov, ax, color, alpha=0.3, label=None):
        """Plot 2σ covariance ellipse."""
        from matplotlib.patches import Ellipse
        
        evals, evecs = np.linalg.eigh(cov)
        angle = np.degrees(np.arctan2(evecs[1, 0], evecs[0, 0]))
        width, height = 4 * np.sqrt(np.maximum(evals, 1e-10))  # 2σ
        
        ellipse = Ellipse(mean, width, height, angle=angle,
                         facecolor=color, alpha=alpha, 
                         edgecolor=color, linewidth=2)
        ax.add_patch(ellipse)
        
        if label:
            ax.plot([], [], color=color, linewidth=2, label=label)


def main():
    print("=" * 70)
    print("  HYBRID COVARIANCE STEERING FOR MULTI-STEP LIPM WALKING")
    print("=" * 70)
    
    # Create LIPM walker with shorter steps for stability
    walker = LIPMWalker(z0=1.0, g=9.81, step_length=0.20)
    
    print(f"\nLIPM Parameters:")
    print(f"  CoM height: {walker.z0} m")
    print(f"  Natural frequency: {walker.omega:.3f} rad/s")
    print(f"  Step length: {walker.step_length} m")
    
    # Walking parameters
    n_steps = 4          # Number of walking steps
    T_step = 0.30        # Shorter stance for stability
    dt = 0.002           # Time step
    epsilon = 0.0005     # Lower noise intensity
    
    print(f"\nWalking Parameters:")
    print(f"  Number of steps: {n_steps}")
    print(f"  Stance duration: {T_step} s")
    print(f"  Total time: {n_steps * T_step:.2f} s")
    print(f"  Noise intensity: {epsilon}")
    
    # Create controller
    controller = MultiStepCovarianceSteering(
        walker, n_steps=n_steps, T_step=T_step, dt=dt, epsilon=epsilon
    )
    
    # Boundary conditions
    x0_mean = np.array([-0.05, 0.45])    # Start behind first foot
    Sig0 = 0.001 * np.eye(2)              # Initial covariance
    SigT = 0.0005 * np.eye(2)             # Target covariance
    
    print(f"\nBoundary Conditions:")
    print(f"  Initial state: {x0_mean}")
    print(f"  Initial cov trace: {np.trace(Sig0):.6f}")
    print(f"  Target cov trace: {np.trace(SigT):.6f}")
    print(f"  Covariance reduction: {np.trace(Sig0)/np.trace(SigT):.1f}x")
    
    # Design gait
    controller.design_walking_gait()
    
    # Compute optimal intermediate covariances
    controller.compute_covariance_targets(Sig0, SigT)
    
    # Compute feedback gains
    controller.compute_feedback_gains(Sig0, SigT)
    
    # Propagate covariance
    controller.propagate_covariance(Sig0)
    
    # Monte Carlo validation
    controller.monte_carlo(x0_mean, Sig0, n_samples=200)
    
    # Results summary
    print("\n" + "=" * 60)
    print("RESULTS SUMMARY")
    print("=" * 60)
    
    Sig_final = controller.Sig_traj_steps[-1][-1]
    E_final = controller.saltation_matrices[-1]
    Sig_final_post = E_final @ Sig_final @ E_final.T
    
    print(f"\nFinal propagated covariance trace: {np.trace(Sig_final_post):.6f}")
    print(f"Target covariance trace: {np.trace(SigT):.6f}")
    print(f"Monte Carlo covariance trace: {np.trace(controller.SigT_mc):.6f}")
    
    # Plot
    controller.plot_results(x0_mean, Sig0, SigT, save_path='lipm_multistep_walking')
    
    return controller


if __name__ == "__main__":
    controller = main()
