import numpy as np
import matplotlib.pyplot as plt


def snap(points, value):
    """
    Find the biggest point in points that is <= value by binary search and return the index.
    Note by construction, value is always within the range of points
    """
    # Enforce bounds (disabled for performance)
    # if value < points[0]:
    #     raise ValueError(f"Value {value:.5f} is out of bounds ({points[0]:.5f} to {points[-1]:.5f}) (too small).")
    # if value > points[-1]:
    #     raise ValueError(f"Value {value:.5f} is out of bounds ({points[0]:.5f} to {points[-1]:.5f}) (too large).")

    low = 0
    high = len(points) - 1
    while low < high:
        mid = (low + high + 1) // 2
        if points[mid] <= value:
            low = mid
        else:
            high = mid - 1

    # If it is the last point, move one back
    if low == len(points) - 1:
        low -= 1

    return low


def interpolate(S_x, S_y, x, y, f):
    """
    We don't need this function, but I implemented it to check how well it works.
    Bilinear interpolation of f(x, y) on grid defined by S_x and S_y.

    S_x: x_coordinates of grid
    S_y: y_coordinates of grid
    x: x coordinate of point to interpolate
    y: y coordinate of point to interpolate
    f: function f(x, y) to interpolate
    """

    x_idx, y_idx = snap(S_x, x), snap(S_y, y)
    x0, x1 = S_x[x_idx], S_x[x_idx + 1]
    y0, y1 = S_y[y_idx], S_y[y_idx + 1]

    x_tilde = (x - x0) / (x1 - x0)
    y_tilde = (y - y0) / (y1 - y0)

    c00 = (1 - x_tilde) * (1 - y_tilde)
    c10 = x_tilde * (1 - y_tilde)
    c01 = (1 - x_tilde) * y_tilde
    c11 = x_tilde * y_tilde

    return (c00 * f(x0, y0) +
            c10 * f(x1, y0) +
            c01 * f(x0, y1) +
            c11 * f(x1, y1))


def build_grid(V_min, V_max, mv, Z_min, Z_max, mz):
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


def construct_tree(V0, S0, K, n, mz, mv, T, r, kappa, theta, sigma, rho):
    Z0 = np.log(S0)
    dt = T / n

    # Instead of generating every node, I will do proxy of just keeping track of up-most and down-most paths and readjusting
    V_max = np.zeros(n)
    V_min = np.zeros(n)
    Z_max = np.zeros(n)
    Z_min = np.zeros(n)

    V_up = V_down = V0
    Z_up = Z_down = Z0

    for i in range(n):
        Z_up = Z_up + (r - 0.5 * V_down) * dt + np.sqrt(np.maximum(V_up, 0) * dt)
        Z_down = Z_down + (r - 0.5 * V_up) * dt - np.sqrt(np.maximum(V_down, 0) * dt)

        V_up = V_up + kappa * (theta - np.maximum(V_up, 0)) * dt + sigma * np.sqrt(np.maximum(V_up, 0) * dt)
        V_down = V_down + kappa * (theta - np.maximum(V_down, 0)) * dt - sigma * np.sqrt(np.maximum(V_down, 0) * dt)

        V_max[i] = V_up
        V_min[i] = V_down
        Z_max[i] = Z_up
        Z_min[i] = Z_down

    VZ_grid = build_grid(V_min, V_max, mv, Z_min, Z_max, mz)

    price_grid = np.zeros((n, mv, mz))

    # Fill in option values at maturity
    price_grid[-1, :, :] = np.maximum(np.exp(VZ_grid[-1, :, :, 1]) - K, 0)

    def v_next(y, v, z):
        return v + kappa * (theta - np.maximum(v, 0)) * dt + y * sigma * np.sqrt(np.maximum(v, 0) * dt)

    def z_next(y, v, z):
        return z + (r - 0.5 * v) * dt + y * np.sqrt(np.maximum(v, 0) * dt)

    def interpolate_price(v, z, k):
        """
        Interpolates the price at time step k-1 by projecting onto time step k
        for variance v and log-price z
        """
        v_points = VZ_grid[k, :, 0, 0]
        z_points = VZ_grid[k, 0, :, 1]
        expected_value = 0.0

        # Store 16 possible states as a 4-bit number 0b[Y4][Y3][Y2][Y1]
        for state in range(16):
            y1 = 1 if (state & 0b0001) else -1
            y2 = 1 if (state & 0b0010) else -1
            y3 = 1 if (state & 0b0100) else 0
            y4 = 1 if (state & 0b1000) else 0

            v_ = v_next(y1, v, z)
            z_ = z_next(y2, v, z)

            v_idx = snap(v_points, v_)
            z_idx = snap(z_points, z_)

            v_tilde = (v_ - v_points[v_idx]) / (v_points[v_idx + 1] - v_points[v_idx])
            z_tilde = (z_ - z_points[z_idx]) / (z_points[z_idx + 1] - z_points[z_idx])

            # Joint probability of y1, y2, y3, y4
            c = 1
            c *= v_tilde if y3 == 1 else (1 - v_tilde)
            c *= z_tilde if y4 == 1 else (1 - z_tilde)
            q = 0.25 * (1 + y1 * y2 * rho) * c

            expected_value += q * price_grid[k, v_idx + y3, z_idx + y4]

        return expected_value

    # Backward induction
    for k in range(n-2, -1, -1):
        for i in range(mv):
            for j in range(mz):
                v, z = VZ_grid[k, i, j, :]
                expected_value = interpolate_price(v, z, k+1)
                price_grid[k, i, j] = np.exp(-r * dt) * expected_value

    # Interpolate to get initial price
    price = interpolate_price(V0, Z0, 0)
    return price


# ---Unit tests---

def test_interpolate():
    S_x = np.array([1.0, 2.0, 3.0])
    S_y = np.array([1.0, 2.0, 3.0])

    def f(x, y):
        return x*2 + y**3

    test_points = [
        (1.5, 1.5),
        (2.5, 2.5),
        (1.0, 2.0),
        (2.0, 1.0),
        (2.0, 3.0),
        (3.0, 2.0),
    ]

    for x, y in test_points:
        interp_value = interpolate(S_x, S_y, x, y, f)
        true_value = f(x, y)
        print(f"Interpolated: {interp_value}, True: {true_value}, Difference: {abs(interp_value - true_value)}")


def test_build_grid():
    x_min = np.array([0.0, 0.5, 1.0])
    x_max = np.array([1.0, 1.5, 2.0])

    y_min = np.array([3.0, 2.5, 2.0])
    y_max = np.array([4.0, 4.5, 5.0])

    mx = 5
    my = 10
    grid = build_grid(x_min, x_max, mx, y_min, y_max, my)
    print("Grid shape:", grid.shape)
    print(grid)


def test_construct_tree():
    V0 = 0.04
    S0 = 100
    K = 100
    n = 100
    mz = 50
    mv = 50
    T = 1.0
    r = 0
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    price = construct_tree(V0, S0, K, n, mz, mv, T, r, kappa, theta, sigma, rho)

    print(f"Option Price: {price:.5f}")


if __name__ == "__main__":
    # test_interpolate()
    test_construct_tree()
    # test_build_grid()
