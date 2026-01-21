import numpy as np
import matplotlib.pyplot as plt


# Naive first implementation of Heston tree
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

        dv = v_points[1] - v_points[0]
        dz = z_points[1] - z_points[0]

        # Encode y1 and y2 as 2-bit binary number
        for state1 in range(4):
            y1 = 1 if (state1 & 0b01) else -1
            y2 = 1 if (state1 & 0b10) else -1

            v_ = self.v_next(y1, v, z)
            z_ = self.z_next(y2, v, z)

            # Find lower left corner for interpolation
            v_idx = int((v_ - v_points[0]) / dv)
            z_idx = int((z_ - z_points[0]) / dz)

            # Clamp indices to valid range
            v_idx = np.clip(v_idx, 0, self.mv - 2)
            z_idx = np.clip(z_idx, 0, self.mz - 2)

            v_tilde = (v_ - v_points[v_idx]) / dv
            z_tilde = (z_ - z_points[z_idx]) / dz

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


# Vectorised and optimized implementation of Heston tree
class HestonTreeFast:
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

    def build_grid(self, V_min, V_max, Z_min, Z_max):
        u_v = np.linspace(0.0, 1.0, self.mv)
        u_z = np.linspace(0.0, 1.0, self.mz)

        v_grid = V_min[:, None] + (V_max - V_min)[:, None] * u_v[None, :]
        z_grid = Z_min[:, None] + (Z_max - Z_min)[:, None] * u_z[None, :]

        V = v_grid[:, :, None]
        Z = z_grid[:, None, :]

        grid = np.empty((self.n, self.mv, self.mz, 2))
        grid[..., 0] = V
        grid[..., 1] = Z
        return grid

    def build_tree(self, V0_min, V0_max, S0_min, S0_max):
        self.V0_min = V0_min
        self.V0_max = V0_max
        self.S0_min = S0_min
        self.S0_max = S0_max

        Z0_min = np.log(S0_min)
        Z0_max = np.log(S0_max)

        V_min = np.empty(self.n)
        V_max = np.empty(self.n)
        Z_min = np.empty(self.n)
        Z_max = np.empty(self.n)

        V_up, V_dn = V0_max, V0_min
        Z_up, Z_dn = Z0_max, Z0_min

        for k in range(self.n):
            Z_up = Z_up + (self.r - 0.5 * V_dn) * self.dt + np.sqrt(max(V_up, 0) * self.dt)
            Z_dn = Z_dn + (self.r - 0.5 * V_up) * self.dt - np.sqrt(max(V_dn, 0) * self.dt)

            V_up = V_up + self.kappa * (self.theta - max(V_up, 0)) * self.dt \
                + self.sigma * np.sqrt(max(V_up, 0) * self.dt)
            V_dn = V_dn + self.kappa * (self.theta - max(V_dn, 0)) * self.dt \
                - self.sigma * np.sqrt(max(V_dn, 0) * self.dt)

            V_max[k] = V_up
            V_min[k] = V_dn
            Z_max[k] = Z_up
            Z_min[k] = Z_dn

        VZ_grid = self.build_grid(V_min, V_max, Z_min, Z_max)

        price = np.zeros((self.n, self.mv, self.mz))

        # terminal payoff
        price[-1] = np.maximum(np.exp(VZ_grid[-1, :, :, 1]) - self.K, 0.0)

        # Predefine stochastic branches
        y1 = np.array([-1, -1, +1, +1])[:, None, None]
        y2 = np.array([-1, +1, -1, +1])[:, None, None]
        prob = 0.25 * (1 + y1 * y2 * self.rho)

        discount = np.exp(-self.r * self.dt)

        # Backward induction
        for k in range(self.n - 2, -1, -1):
            v = VZ_grid[k, :, :, 0]
            z = VZ_grid[k, :, :, 1]

            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos * self.dt)

            v_drift = self.kappa * (self.theta - v_pos) * self.dt
            z_drift = (self.r - 0.5 * v) * self.dt

            # next variance and log-price for 4 branches vectorised
            v_next = v + v_drift + y1 * self.sigma * sqrt_v
            z_next = z + z_drift + y2 * sqrt_v

            # grid spacing
            Vmin = VZ_grid[k+1, 0, 0, 0]
            Zmin = VZ_grid[k+1, 0, 0, 1]
            dv = (VZ_grid[k+1, -1, 0, 0] - Vmin) / (self.mv - 1)
            dz = (VZ_grid[k+1, 0, -1, 1] - Zmin) / (self.mz - 1)

            # lower-left corner indices
            i0 = ((v_next - Vmin) / dv).astype(np.int32)
            j0 = ((z_next - Zmin) / dz).astype(np.int32)
            i0 = np.clip(i0, 0, self.mv - 2)
            j0 = np.clip(j0, 0, self.mz - 2)

            v0 = Vmin + i0 * dv
            z0 = Zmin + j0 * dz

            v_tilde = (v_next - v0) / dv
            z_tilde = (z_next - z0) / dz

            p = price[k+1]

            p00 = p[i0, j0]
            p01 = p[i0, j0 + 1]
            p10 = p[i0 + 1, j0]
            p11 = p[i0 + 1, j0 + 1]

            interp = (
                (1 - v_tilde)*(1 - z_tilde)*p00 +
                (1 - v_tilde)*z_tilde*p01 +
                v_tilde*(1 - z_tilde)*p10 +
                v_tilde*z_tilde*p11
            )

            price[k] = discount * np.sum(prob * interp, axis=0)

        self.VZ_grid = VZ_grid
        self.price_grid = price
        self.tree_built = True

    def price(self, V0, S0, k=0):
        if not self.tree_built:
            raise RuntimeError("Tree not built")
        if not (self.V0_min <= V0 <= self.V0_max):
            raise ValueError("V0 out of bounds")
        if not (self.S0_min <= S0 <= self.S0_max):
            raise ValueError("S0 out of bounds")

        z0 = np.log(S0)

        # interpolate at time step k using bilinear interpolation
        Vmin = self.VZ_grid[k, 0, 0, 0]
        Zmin = self.VZ_grid[k, 0, 0, 1]
        dv = (self.VZ_grid[k, -1, 0, 0] - Vmin) / (self.mv - 1)
        dz = (self.VZ_grid[k, 0, -1, 1] - Zmin) / (self.mz - 1)

        i0 = int((V0 - Vmin) / dv)
        j0 = int((z0 - Zmin) / dz)
        i0 = np.clip(i0, 0, self.mv - 2)
        j0 = np.clip(j0, 0, self.mz - 2)

        v_tilde = (V0 - (Vmin + i0*dv)) / dv
        z_tilde = (z0 - (Zmin + j0*dz)) / dz

        p = self.price_grid[k]

        result = (
            (1 - v_tilde) * (1 - z_tilde) * p[i0, j0] +
            (1 - v_tilde) * z_tilde * p[i0, j0 + 1] +
            v_tilde * (1 - z_tilde) * p[i0 + 1, j0] +
            v_tilde * z_tilde * p[i0 + 1, j0 + 1]
        )

        return result
