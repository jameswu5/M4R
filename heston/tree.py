import numpy as np
import matplotlib.pyplot as plt


class HestonTree:
    def __init__(self, n, mz, mv, K, T, r, kappa, theta, sigma, rho):
        self.n = n
        self.mz = mz
        self.mv = mv
        self.K = K
        self.T = T
        self.r = r
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.rho = rho

        self.dt = T / n

        self.tree_built = False
        self.VZ_grid = None
        self.price_grid = None
        self.V0_min = None
        self.V0_max = None
        self.S0_min = None
        self.S0_max = None

    def snap(self, points, value):
        """
        Find the biggest point in points that is <= value and return the index.
        points: a 1D array of monotonically increasing points
        value: the value to snap
        """
        min_val = points[0]
        max_val = points[-1]

        # Base cases
        if value < min_val:
            return 0
        if value >= max_val:
            return len(points) - 2

        return int((value - min_val) / (max_val - min_val) * (len(points) - 1))

    def build_grid(self, V_min, V_max, mv, Z_min, Z_max, mz):
        """
        Builds a grid of shape (n, mx, my, 2)
        V_min, V_max, Z_min, Z_max have shape (n,)
        """
        n = V_min.shape[0]

        u_v = np.linspace(0.0, 1.0, mv)
        u_z = np.linspace(0.0, 1.0, mz)

        v_grid = V_min[:, None] + (V_max - V_min)[:, None] * u_v[None, :]
        z_grid = Z_min[:, None] + (Z_max - Z_min)[:, None] * u_z[None, :]

        V = v_grid[:, :, None]        # (n, mv, 1)
        Z = z_grid[:, None, :]        # (n, 1, mz)

        grid = np.empty((n, mv, mz, 2))
        grid[..., 0] = V
        grid[..., 1] = Z

        return grid

    def v_next(self, y, v, z):
        """
        Evolves v according to the Euler scheme

        y: +1 or -1
        v: current variance
        z: current log-price (not used here but kept for symmetry)
        """
        return v + self.kappa * (self.theta - np.maximum(v, 0)) * self.dt + y * self.sigma * np.sqrt(np.maximum(v, 0) * self.dt)

    def z_next(self, y, v, z):
        """
        Evolves z according to the Euler scheme

        y: +1 or -1
        v: current variance
        z: current log-price (not used here but kept for symmetry)
        """
        return z + (self.r - 0.5 * v) * self.dt + y * np.sqrt(np.maximum(v, 0) * self.dt)

    def interpolate_price(self, v, z, k, VZ_grid, price_grid):
        """
        Interpolates the price at time step k-1 by computing weighted sum of
        four closest points of space at time step k for variance v and log-price z
        No discounting done here
        """
        v_points = VZ_grid[k, :, 0, 0]
        z_points = VZ_grid[k, 0, :, 1]
        expected_value = 0.0

        # Encode y1 and y2 as 2-bit binary number
        for state1 in range(4):
            y1 = 1 if (state1 & 0b01) else -1
            y2 = 1 if (state1 & 0b10) else -1

            v_ = self.v_next(y1, v, z)
            z_ = self.z_next(y2, v, z)

            v_idx = self.snap(v_points, v_)
            z_idx = self.snap(z_points, z_)

            v_tilde = (v_ - v_points[v_idx]) / (v_points[v_idx + 1] - v_points[v_idx])
            z_tilde = (z_ - z_points[z_idx]) / (z_points[z_idx + 1] - z_points[z_idx])

            # Sum over possible y3 and y4 in {0, 1}
            expected_value += 0.25 * (1 + y1 * y2 * self.rho) * \
                ((v_tilde * z_tilde) * price_grid[k, v_idx + 1, z_idx + 1] +
                 ((1 - v_tilde) * z_tilde) * price_grid[k, v_idx, z_idx + 1] +
                 (v_tilde * (1 - z_tilde)) * price_grid[k, v_idx + 1, z_idx] +
                 ((1 - v_tilde) * (1 - z_tilde)) * price_grid[k, v_idx, z_idx])

        return expected_value

    def build_tree(self, V0_min, V0_max, S0_min, S0_max, verbose=False):
        """
        Builds the Heston tree within the specified bounds for initial variance and stock price.
        Constructs the price tree using backward induction.

        Doesn't return anything but saves the grids internally.
        """
        # Save the bounds
        self.V0_min = V0_min
        self.V0_max = V0_max
        self.S0_min = S0_min
        self.S0_max = S0_max

        Z0_min = np.log(S0_min)
        Z0_max = np.log(S0_max)

        # Instead of generating every node, I will do proxy of just keeping track of up-most and down-most paths and readjusting
        V_max = np.zeros(self.n)
        V_min = np.zeros(self.n)
        Z_max = np.zeros(self.n)
        Z_min = np.zeros(self.n)

        V_up = V0_max
        V_down = V0_min
        Z_up = Z0_max
        Z_down = Z0_min

        for k in range(self.n):
            Z_up = Z_up + (self.r - 0.5 * V_down) * self.dt + np.sqrt(np.maximum(V_up, 0) * self.dt)
            Z_down = Z_down + (self.r - 0.5 * V_up) * self.dt - np.sqrt(np.maximum(V_down, 0) * self.dt)

            V_up = V_up + self.kappa * (self.theta - np.maximum(V_up, 0)) * self.dt + self.sigma * np.sqrt(np.maximum(V_up, 0) * self.dt)
            V_down = V_down + self.kappa * (self.theta - np.maximum(V_down, 0)) * self.dt - self.sigma * np.sqrt(np.maximum(V_down, 0) * self.dt)

            V_max[k] = V_up
            V_min[k] = V_down
            Z_max[k] = Z_up
            Z_min[k] = Z_down

        VZ_grid = self.build_grid(V_min, V_max, self.mv, Z_min, Z_max, self.mz)

        price_grid = np.zeros((self.n, self.mv, self.mz))

        # Fill in option values at maturity
        price_grid[-1, :, :] = np.maximum(np.exp(VZ_grid[-1, :, :, 1]) - self.K, 0)

        # Backward induction
        for k in range(self.n-2, -1, -1):
            if verbose:
                print(f"Building tree at step {k}")
            for i in range(self.mv):
                for j in range(self.mz):
                    v, z = VZ_grid[k, i, j, :]
                    expected_value = self.interpolate_price(v, z, k+1, VZ_grid, price_grid)
                    price_grid[k, i, j] = np.exp(-self.r * self.dt) * expected_value

        self.tree_built = True
        self.VZ_grid = VZ_grid
        self.price_grid = price_grid

    def price(self, V0, S0, k=0):
        """
        Prices the option given initial variance V0 and stock price S0 with horizon T.
        Requires that build_tree() has been called first.
        Also requires V0 and S0 to be within the specified bounds when build_tree() was called.
        """
        if not self.tree_built:
            raise ValueError("Tree not built yet. Call build_tree() first.")

        if V0 < self.V0_min or V0 > self.V0_max:
            raise ValueError(f"V0={V0} out of bounds ({self.V0_min}, {self.V0_max})")
        if S0 < self.S0_min or S0 > self.S0_max:
            raise ValueError(f"S0={S0} out of bounds ({self.S0_min}, {self.S0_max})")

        return self.interpolate_price(V0, np.log(S0), k, self.VZ_grid, self.price_grid)
