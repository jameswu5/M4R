import numpy as np
import matplotlib.pyplot as plt


class HestonTreeFast:
    """Vectorised Heston tree pricer using fully-batched backward induction over four stochastic branches."""

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
        """Build a uniformly-spaced (V, Z) grid for each time step."""
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
        """Bilinearly interpolate the price grid at (v, z) and time step k; out-of-bounds values are clamped."""
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
        """Build the Heston price tree using vectorised backward induction over four stochastic branches."""
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
            Z_dn = Z_dn + (self.r - 0.5 * V_up) * self.dt - np.sqrt(max(V_up, 0) * self.dt)

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
        Price the option at initial variance V0 and asset price S0.

        One of V0 or S0 may be a vector for batch pricing; the other must be scalar.

        Parameters
        ----------
        V0 : float or array-like
            Initial variance; must lie within the bounds passed to build_tree.
        S0 : float or array-like
            Initial asset price; must lie within the bounds passed to build_tree.
        k : int, optional
            Time-step index at which to evaluate (default 0, i.e. the root).
        continuation_value : bool, optional
            If True, return the continuation value without early exercise (default False).

        Returns
        -------
        price : float or ndarray
            Option price(s); scalar when both inputs are scalar, otherwise a flattened array.
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
