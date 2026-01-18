import numpy as np


def sample_binary(length, rho):
    """
    Sample Y_1, Y_2 in {-1, 1} as in the paper.

    Returns 2 nparrays of shape (length,) with entries in {-1, 1} and correlation rho.
    """

    uniform1 = np.random.rand(length)  # Determines Y1 = Y2
    uniform2 = np.random.rand(length)  # Determines Y1 = 1 or -1

    y_equal = uniform1 < (1 + rho) / 2

    y1 = np.where(uniform2 < 0.5, 1, -1)
    y2 = np.where(y_equal, y1, -y1)

    return y1, y2


def interpolate(S_x, S_y, x, y, f):
    """
    S_x: x_coordinates of grid
    S_y: y_coordinates of grid
    x: x coordinate of point to interpolate
    y: y coordinate of point to interpolate
    f: function f(x, y) to interpolate
    """

    def snap(points, value):
        """
        Find the biggest point in points that is <= value by binary search and return the index.
        Note by construction, value is always within the range of points
        """
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


def construct_tree(V0, S0, n, mz, mv, T, kappa, theta, sigma, rho):
    # Build tree for V
    V_layer = [V0]

    dt = T / n

    for i in range(n):
        next_V_layer = np.array([])
        for v in V_layer:
            up_v = v + kappa * (theta - max(v, 0)) * dt + sigma * np.sqrt(max(v, 0) * dt)
            down_v = v + kappa * (theta - max(v, 0)) * dt - sigma * np.sqrt(max(v, 0) * dt)
            next_V_layer = np.append(next_V_layer, [up_v, down_v])
        V_layer = next_V_layer
        print(V_layer)

        print(np.max(V_layer) == V_layer[0])
        print(np.min(V_layer) == V_layer[-1])


# ---Unit tests---

def test_sample_binary():
    length = 1000000
    rhos = [-0.9, -0.5, 0.0, 0.5, 0.9]

    for rho in rhos:
        y1, y2 = sample_binary(length, rho)
        empirical_rho = np.corrcoef(y1, y2)[0, 1]
        print(f"Empirical correlation: {empirical_rho:.4f}\t Target correlation: {rho}\tDifference: {abs(empirical_rho - rho):.4f}")


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


def test_construct_tree():
    V0 = 0.04
    S0 = 100
    n = 5
    mz = 100
    mv = 100
    T = 1.0
    kappa = 2.0
    theta = 0.04
    sigma = 0.3
    rho = -0.7
    construct_tree(V0, S0, n, mz, mv, T, kappa, theta, sigma, rho)


if __name__ == "__main__":
    test_interpolate()
