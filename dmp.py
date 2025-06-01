import numpy as np

class CanonicalSystem:
    """
    Skeleton of the discrete canonical dynamical system.
    """
    def __init__(self, dt: float, ax: float = 1.0):
        """
        Args:
            dt (float): Timestep duration.
            ax (float): Gain on the canonical decay.
        """
        # Initialize time parameters
        self.dt: float = dt
        self.ax: float = ax
        self.run_time: float = 1.0  # TODO: set total runtime
        self.timesteps: int = self.run_time/dt  # TODO: compute from run_time and dt
        self.x: float = 1.0  # phase variable

    def reset(self) -> None:
        """
        Reset the phase variable to its initial value.
        """
        self.x: float = 1.0

    def step(self, tau: float = 1.0, error_coupling: float = 1.0) -> float:
        """
        Advance the phase by one timestep.

        Returns:
            float: Updated phase value.
        """
        self.x: float = self.x + (-self.ax * self.x * error_coupling) * tau * self.dt
        return self.x

    def rollout(self, tau: float = 1.0, ec: float = 1.0) -> np.ndarray:
        """
        Generate the entire phase sequence.

        Returns:
            np.ndarray: Array of phase values over time.
        """
        xs = np.zeros(int(self.timesteps))
        self.reset()
        for i in range(int(self.timesteps)) :
            xs[i] = self.step(tau = tau, error_coupling = ec)
        return xs

class DMP:
    """
    Skeleton of the discrete Dynamic Motor Primitive.
    """
    def __init__(
        self,
        n_dmps: int,
        n_bfs: int,
        dt: float = 0.01,
        y0: float = 0.0,
        goal: float = 1.0,
        ay: float = 25.0,
        by: float = None
    ):
        """
        Args:
            n_dmps (int): Number of dimensions.
            n_bfs (int): Number of basis functions per dimension.
            dt (float): Timestep duration.
            y0 (float|array): Initial state.
            goal (float|array): Goal state.
            ay (float|array): Attractor gain.
            by (float|array): Damping gain.
        """
        # TODO: initialize parameters
        self.n_dmps: int = n_dmps
        self.n_bfs: int = n_bfs
        self.dt: float = dt
        self.y0: np.ndarray = y0
        self.goal: np.ndarray = goal
        self.ay: np.ndarray = np.ones(self.n_dmps) * ay
        self.by: np.ndarray = np.ones(self.n_dmps) * (self.ay / 4.0) if by is None else np.ones(self.n_dmps) * by
        self.w: np.ndarray = np.zeros((self.n_dmps, self.n_bfs))
        self.cs: CanonicalSystem = CanonicalSystem(dt = self.dt)
        self.reset_state()

    def reset_state(self) -> None:
        """
        Reset trajectories and canonical system state.
        """
        # TODO: reset y, dy, ddy and call self.cs.reset()
        self.y = np.zeros((self.n_dmps, 1))
        self.dy = np.zeros((self.n_dmps, 1))
        self.ddy = np.zeros((self.n_dmps, 1))
        self.cs = CanonicalSystem(dt = self.dt)
        self.cs.reset()

    def imitate(self, y_des: np.ndarray) -> np.ndarray:
        """
        Learn DMP weights from a demonstration.

        Args:
            y_des (np.ndarray): Desired trajectory, shape (T, D).

        Returns:
            np.ndarray: Interpolated demonstration (T', D).
        """
        # TODO: interpolate, compute forcing term, solve for w
        if y_des.ndim == 1:
            y_des = y_des[:, None]
        T_demo, D = y_des.shape

        t_demo = np.linspace(0, (T_demo - 1) * self.dt, T_demo)
        T = int(np.round(t_demo[-1] / self.dt)) + 1
        t_interp = np.linspace(0, t_demo[-1], T)
        y_interp = np.zeros((T, D))
        for d in range(D):
            y_interp[:, d] = np.interp(t_interp, t_demo, y_des[:, d])

        self.cs = CanonicalSystem(dt = self.dt)
        self.cs.run_time = t_demo[-1]
        self.cs.timesteps = T
        x_track = self.cs.rollout()

        dy = np.gradient(y_interp, axis = 0) / self.dt
        ddy = np.gradient(dy, axis = 0) / self.dt

        self.y0 = y_interp[0]
        self.goal = y_interp[-1]
        self.ay = np.ones(self.n_dmps) * 25.0 if self.ay is None else self.ay
        self.by = np.ones(self.n_dmps) * (self.ay / 4.0) if self.by is None else self.by

        centers = np.exp(-self.cs.ax * np.linspace(0, self.cs.run_time, self.n_bfs))
        widths = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / centers / self.cs.ax

        f_target = np.zeros_like(y_interp)
        for d in range(D):
            f_target[:, d] = ddy[:, d] - self.ay[d] * (self.by[d] * (self.goal[d] - y_interp[:, d]) - dy[:, d])

        basis = np.exp(-widths[None, :] * (x_track[:, None] - centers[None, :]) ** 2)

        self.w = np.zeros((D, self.n_bfs))
        for d in range(D):
            self.w[d, :] = np.sum(basis * f_target[:, d][:, None], axis = 0) / np.sum(basis, axis = 0)

        if y_interp.shape[1] == 1:
            return y_interp.T
        return y_interp

    def rollout(
        self,
        tau: float = 1.0,
        error: float = 0.0,
        new_goal: np.ndarray = None
    ) -> np.ndarray:
        """
        Generate a new trajectory from the DMP.

        Args:
            tau (float): Temporal scaling.
            error (float): Feedback coupling.
            new_goal (np.ndarray, optional): Override goal.

        Returns:
            np.ndarray: Generated trajectory (T x D).
        """
        # TODO: implement dynamical update loop
        goal = self.goal if new_goal is None else new_goal
        self.reset_state()
        self.y[:, 0] = self.y0
        self.dy[:, 0] = 0.0
        self.ddy[:, 0] = 0.0

        x_track = self.cs.rollout(tau = tau)
        T = len(x_track)
        y_track = np.zeros((T, self.n_dmps))

        for t in range(T):
            x = x_track[t]
            centers = np.exp(-self.cs.ax * np.linspace(0, 1, self.n_bfs))
            widths = np.ones(self.n_bfs) * self.n_bfs ** 1.5 / centers / self.cs.ax
            basis = np.exp(-widths * (x - centers) ** 2)
            for d in range(self.n_dmps):
                f = np.dot(basis, self.w[d]) / np.sum(basis)
                self.ddy[d, 0] = float(self.ay[d]) * (float(self.by[d]) * (float(goal[d]) - self.y[d, 0]) - self.dy[d, 0]) + f
                self.dy[d, 0] += self.ddy[d, 0] * self.dt * tau
                self.y[d, 0] += self.dy[d, 0] * self.dt * tau
                y_track[t, d] = self.y[d, 0]
        return y_track

# ==============================
# DMP Unit test
# ==============================
if __name__ == '__main__':

    import matplotlib.pyplot as plt

    # Test canonical system
    cs = CanonicalSystem(dt=0.05)
    x_track = cs.rollout()
    plt.figure()
    plt.plot(x_track, label='Canonical x')
    plt.title('Canonical System Rollout')
    plt.xlabel('Timestep')
    plt.ylabel('x')
    plt.legend()

    # Test DMP behavior with a sine-wave trajectory
    dt = 0.01
    T = 1.0
    t = np.arange(0, T, dt)
    y_des = np.sin(2 * np.pi * 2 * t)

    dmp = DMP(n_dmps=1, n_bfs=50, dt=dt)
    y_interp = dmp.imitate(y_des)
    y_run = dmp.rollout()

    plt.figure()
    plt.plot(t, y_des, 'k--', label='Original')
    plt.plot(np.linspace(0, T, y_interp.shape[1]), y_interp.flatten(), 'b-.', label='Interpolated')
    plt.plot(np.linspace(0, T, y_run.shape[0]), y_run.flatten(), 'r-', label='DMP Rollout')
    plt.title('DMP Imitation and Rollout')
    plt.xlabel('Time (s)')
    plt.ylabel('y')
    plt.legend()
    plt.tight_layout()
    plt.show()
