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

        self.information = {
            "tree_built": False,
            "exercise_type": None,
            "payoff_type": None,
        }

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

    def interpolate_price(self, v, z, k):
        Vmin = self.VZ_grid[k, 0, 0, 0]
        Zmin = self.VZ_grid[k, 0, 0, 1]

        dv = (self.VZ_grid[k, -1, 0, 0] - Vmin) / (self.mv - 1)
        dz = (self.VZ_grid[k, 0, -1, 1] - Zmin) / (self.mz - 1)

        # lower-left indices
        i0 = ((v - Vmin) / dv).astype(np.int32)
        j0 = ((z - Zmin) / dz).astype(np.int32)

        i0 = np.clip(i0, 0, self.mv - 2)
        j0 = np.clip(j0, 0, self.mz - 2)

        v0 = Vmin + i0 * dv
        z0 = Zmin + j0 * dz

        v_tilde = (v - v0) / dv
        z_tilde = (z - z0) / dz

        p = self.price_grid[k]

        return (
            (1 - v_tilde) * (1 - z_tilde) * p[i0, j0] +
            (1 - v_tilde) * z_tilde * p[i0, j0 + 1] +
            v_tilde * (1 - z_tilde) * p[i0 + 1, j0] +
            v_tilde * z_tilde * p[i0 + 1, j0 + 1]
        )

    def build_tree(self, V0_min, V0_max, S0_min, S0_max, option_type="call", exercise_type="european"):
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

        if option_type == "call":
            self.payoff = lambda x: np.maximum(x - self.K, 0.0)
        elif option_type == "put":
            self.payoff = lambda x: np.maximum(self.K - x, 0.0)
        else:
            raise ValueError(f"Payoff type ({option_type}) is not valid")

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

        self.VZ_grid = self.build_grid(V_min, V_max, Z_min, Z_max)
        self.price_grid = np.zeros((self.n, self.mv, self.mz))

        # terminal payoff
        self.price_grid[-1] = self.payoff(np.exp(self.VZ_grid[-1, :, :, 1]))

        # Predefine stochastic branches
        y1 = np.array([-1, -1, +1, +1])[:, None, None]
        y2 = np.array([-1, +1, -1, +1])[:, None, None]
        prob = 0.25 * (1 + y1 * y2 * self.rho)

        discount = np.exp(-self.r * self.dt)

        # Backward induction
        for k in range(self.n - 2, -1, -1):
            v = self.VZ_grid[k, :, :, 0]
            z = self.VZ_grid[k, :, :, 1]

            v_pos = np.maximum(v, 0.0)
            sqrt_v = np.sqrt(v_pos * self.dt)

            v_drift = self.kappa * (self.theta - v_pos) * self.dt
            z_drift = (self.r - 0.5 * v) * self.dt

            # next variance and log-price for 4 branches vectorised
            v_next = v + v_drift + y1 * self.sigma * sqrt_v
            z_next = z + z_drift + y2 * sqrt_v

            interp = self.interpolate_price(v_next, z_next, k+1)
            continuation = discount * np.sum(prob * interp, axis=0)

            if exercise_type == "american":
                exercise = self.payoff(np.exp(self.VZ_grid[k, :, :, 1]))
                self.price_grid[k] = np.maximum(continuation, exercise)
            elif exercise_type == "european":
                self.price_grid[k] = continuation
            else:
                raise ValueError(f"Exercise type ({exercise_type}) is not valid.")

        self.information['tree_built'] = True
        self.information['exercise_type'] = exercise_type
        self.information['payoff_type'] = option_type

    def price(self, V0, S0, k=0, continuation_value=False):
        """
        Prices the option given initial variance V0 and stock price S0 at time step k.
        Requires that build_tree() has been called first.

        Allows one of V0 and S0 to be a vector, enabling batch pricing. The other must be a scalar.
        """

        if not self.information['tree_built']:
            raise RuntimeError("Tree not built")

        V0 = np.asarray(V0, dtype=float)
        S0 = np.asarray(S0, dtype=float)

        if np.any(V0 < self.V0_min) or np.any(V0 > self.V0_max):
            raise ValueError("V0 out of bounds")
        if np.any(S0 < self.S0_min) or np.any(S0 > self.S0_max):
            raise ValueError("S0 out of bounds")

        Z0 = np.log(S0)

        # reshape to at least 2D to make batch-safe
        V0 = np.atleast_2d(V0)
        Z0 = np.atleast_2d(Z0)

        continuation = self.interpolate_price(V0, Z0, k)
        if self.information['exercise_type'] == "european" or continuation_value:
            price = continuation
        else:
            exercise = self.payoff(np.exp(Z0))
            price = np.maximum(continuation, exercise)

        return price.item() if price.size == 1 else price.flatten()
