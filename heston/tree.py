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
        Build a uniformly-spaced ``(V, Z)`` grid for each time step.

        Parameters
        ----------
        V_min : ndarray, shape (n,)
            Minimum variance at each time step.
        V_max : ndarray, shape (n,)
            Maximum variance at each time step.
        mv : int
            Number of variance grid points per time step.
        Z_min : ndarray, shape (n,)
            Minimum log-price at each time step.
        Z_max : ndarray, shape (n,)
            Maximum log-price at each time step.
        mz : int
            Number of log-price grid points per time step.

        Returns
        -------
        grid : ndarray, shape (n, mv, mz, 2)
            ``grid[k, i, j, 0]`` is the variance and ``grid[k, i, j, 1]``
            is the log-price at time step ``k``, variance index ``i``, and
            log-price index ``j``.
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
        Advance the variance by one Euler step.

        Parameters
        ----------
        y : {+1, -1}
            Brownian increment direction for the variance.
        v : float or ndarray
            Current variance.
        z : float or ndarray
            Current log-price (unused; kept for interface symmetry with
            ``z_next``).

        Returns
        -------
        v_new : float or ndarray
            Variance at the next time step.
        """
        return v + self.kappa * (self.theta - np.maximum(v, 0)) * self.dt + y * self.sigma * np.sqrt(np.maximum(v, 0) * self.dt)

    def z_next(self, y, v, z):
        """
        Advance the log-price by one Euler step.

        Parameters
        ----------
        y : {+1, -1}
            Brownian increment direction for the log-price.
        v : float or ndarray
            Current variance.
        z : float or ndarray
            Current log-price (unused in the update formula; kept for
            interface symmetry with ``v_next``).

        Returns
        -------
        z_new : float or ndarray
            Log-price at the next time step.
        """
        return z + (self.r - 0.5 * v) * self.dt + y * np.sqrt(np.maximum(v, 0) * self.dt)

    def interpolate_price(self, v, z, k, VZ_grid, price_grid):
        """
        Compute the expected discounted option value by interpolation.

        Evaluates the four stochastic branches ``(y1, y2) in {±1}^2``, advances
        ``(v, z)`` by one Euler step along each branch, and returns the
        probability-weighted sum of bilinearly-interpolated prices from the
        grid at step ``k``.  Discounting is not applied here.

        Parameters
        ----------
        v : float
            Current variance.
        z : float
            Current log-price.
        k : int
            Target time-step index in ``VZ_grid`` and ``price_grid`` (i.e.
            the step *after* the current one).
        VZ_grid : ndarray, shape (n, mv, mz, 2)
            Space grid produced by ``build_grid``.
        price_grid : ndarray, shape (n, mv, mz)
            Option values on the grid.

        Returns
        -------
        expected_value : float
            Probability-weighted interpolated option value (undiscounted).
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
        Build the Heston price tree and store the grids internally.

        Traces the extreme up/down paths to determine the bounding box of
        ``(V, Z)`` at each time step, builds a uniform space grid, then
        fills option values by backward induction using bilinear interpolation.
        Results are stored in ``self.VZ_grid`` and ``self.price_grid``.

        Parameters
        ----------
        V0_min : float
            Minimum initial variance for the pricing domain.
        V0_max : float
            Maximum initial variance for the pricing domain.
        S0_min : float
            Minimum initial asset price for the pricing domain.
        S0_max : float
            Maximum initial asset price for the pricing domain.
        verbose : bool, optional
            If True, print progress at each backward-induction step
            (default False).
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
        Price the option at initial variance ``V0`` and asset price ``S0``.

        Parameters
        ----------
        V0 : float
            Initial variance.  Must lie within the bounds passed to
            ``build_tree``.
        S0 : float
            Initial asset price.  Must lie within the bounds passed to
            ``build_tree``.
        k : int, optional
            Time-step index at which to evaluate the price (default 0,
            i.e. the root of the tree).

        Returns
        -------
        price : float
            Option price at time step ``k``.

        Raises
        ------
        ValueError
            If ``build_tree`` has not been called, or if ``V0`` or ``S0``
            are outside the bounds specified when building the tree.
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
        """
        Build a uniformly-spaced ``(V, Z)`` grid for each time step.

        Parameters
        ----------
        V_min : ndarray, shape (n,)
            Minimum variance at each time step.
        V_max : ndarray, shape (n,)
            Maximum variance at each time step.
        Z_min : ndarray, shape (n,)
            Minimum log-price at each time step.
        Z_max : ndarray, shape (n,)
            Maximum log-price at each time step.

        Returns
        -------
        grid : ndarray, shape (n, mv, mz, 2)
            ``grid[k, i, j, 0]`` is the variance and ``grid[k, i, j, 1]``
            is the log-price at time step ``k``, variance index ``i``, and
            log-price index ``j``.
        """
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
        """
        Bilinearly interpolate the option price grid at ``(v, z)`` and time step ``k``.

        Parameters
        ----------
        v : float or ndarray
            Variance value(s) at which to interpolate.
        z : float or ndarray
            Log-price value(s) at which to interpolate.
        k : int
            Time-step index into ``self.VZ_grid`` and ``self.price_grid``.

        Returns
        -------
        price : float or ndarray
            Bilinearly interpolated option price(s).  Out-of-bounds values
            are clamped to the nearest grid boundary.
        """
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
        """
        Build the Heston price tree using vectorised backward induction.

        Traces extreme paths to bound ``(V, Z)`` at each step, builds a
        uniform space grid, then fills option values backwards using fully
        vectorised bilinear interpolation over the four stochastic branches.
        Results are stored internally in ``self.VZ_grid``, ``self.price_grid``,
        and ``self.information``.

        Parameters
        ----------
        V0_min : float
            Minimum initial variance for the pricing domain.
        V0_max : float
            Maximum initial variance for the pricing domain.
        S0_min : float
            Minimum initial asset price for the pricing domain.
        S0_max : float
            Maximum initial asset price for the pricing domain.
        option_type : {'call', 'put'}, optional
            Type of the option (default ``'call'``).
        exercise_type : {'european', 'american'}, optional
            Exercise style; American options apply early-exercise at each
            node (default ``'european'``).

        Raises
        ------
        ValueError
            If ``option_type`` or ``exercise_type`` is not recognised.
        """
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
        Price the option at initial variance ``V0`` and asset price ``S0``.

        One of ``V0`` or ``S0`` may be a vector to enable batch pricing; the
        other must be a scalar.

        Parameters
        ----------
        V0 : float or array-like
            Initial variance.  Must lie within the bounds passed to
            ``build_tree``.
        S0 : float or array-like
            Initial asset price.  Must lie within the bounds passed to
            ``build_tree``.
        k : int, optional
            Time-step index at which to evaluate (default 0, i.e. the root).
        continuation_value : bool, optional
            If True, return the continuation value without applying early
            exercise, regardless of ``exercise_type`` (default False).

        Returns
        -------
        price : float or ndarray
            Option price(s).  Returns a scalar when both inputs are scalar,
            otherwise a flattened array.

        Raises
        ------
        RuntimeError
            If ``build_tree`` has not been called.
        ValueError
            If ``V0`` or ``S0`` are outside the bounds specified when building
            the tree.
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
