
import numpy as np
from numba import njit


@njit
def L96_2t_x_dot_y_dot_solver(t, xy_vars, K, J, F, h, b, c):
    """
    Compute the time derivatives for the two-time-scale Lorenz '96 model.

    Equations:
        d/dt X[k] = -X[k-1] (X[k-2] - X[k+1]) - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] (Y[j+2] - Y[j-1]) - c Y[j] + h.c/b X[k]

    Args:
        t (float): Current time (not used in the computation but required for   
            solvers).
        xy_vars (np.ndarray): Concatenated array of X and Y variables. Shape: 
            (K + K*J,).
        K (int): Number of slow variables (X).
        J (int): Number of fast variables (Y) per slow variable.
        F (float): Forcing term.
        h (float): Coupling coefficient.
        b (float): Ratio of amplitudes.
        c (float): Time-scale ratio.

    Returns:
        np.ndarray: Concatenated array of time derivatives for X and Y variables. Shape: (K + K*J,).
    """
    x, y = xy_vars[:K], xy_vars[K:]

    # Helper variables
    hcb = (h * c) / b
    y_summed = y.reshape((K, J)).sum(axis=-1)

    # L96 slow equation
    x_dot = -np.roll(x, 1) * (np.roll(x, 2) - np.roll(x, -1)) - \
        x + F - hcb * y_summed

    # L96 fast equation
    y_dot = (
        -c * b * np.roll(y, -1) * (np.roll(y, -2) - np.roll(y, 1))
        - c * y + hcb * np.repeat(x, J)
    )

    return np.concatenate((x_dot, y_dot))


@njit
def L96_2t_x_dot_y_dot_manual(x, y, F, h, b, c):
    """
    Compute the time derivatives for the two-time-scale Lorenz '96 model manually.

    Equations:
        d/dt X[k] = -X[k-1] (X[k-2] - X[k+1]) - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] (Y[j+2] - Y[j-1]) - c Y[j] + h.c/b X[k]

    Args:
        x (np.ndarray): Slow variables (X). Shape: (K,).
        y (np.ndarray): Fast variables (Y). Shape: (K*J,).
        F (float): Forcing term.
        h (float): Coupling coefficient.
        b (float): Ratio of amplitudes.
        c (float): Time-scale ratio.

    Returns:
        tuple: A tuple containing:
            - x_dot (np.ndarray): Time derivatives of X variables. Shape: (K,).
            - y_dot (np.ndarray): Time derivatives of Y variables. Shape: (K*J,).
            - coupling (np.ndarray): Coupling term. Shape: (K,).
    """

    # Compute K and J from the data
    JK, K = len(y), len(x)
    J = JK // K
    assert JK == J * K, "X and Y have incompatible shapes"

    # Helper variables
    hcb = (h * c) / b
    y_summed = y.reshape((K, J)).sum(axis=-1)

    # L96 slow equation
    x_dot = -np.roll(x, 1) * (np.roll(x, 2) - np.roll(x, -1)) - \
        x + F - hcb * y_summed

    # L96 fast equation
    y_dot = (
        -c * b * np.roll(y, -1) * (np.roll(y, -2) - np.roll(y, 1))
        - c * y + hcb * np.repeat(x, J)
    )

    return x_dot, y_dot, -hcb * y_summed


def calculate_xy_tendencies(y_hist, h, c, b, k, j, nt):
    """
    Compute the coupling tendencies for the Lorenz '96 model from small-scale Y variables.

    Args:
        y_hist (np.ndarray): History of the fast variables (Y) over time. 
            Shape: (nt+1, K*J).
        h (float): Coupling coefficient.
        c (float): Time-scale ratio.
        b (float): Ratio of amplitudes.
        k (int): Number of slow variables (X).
        j (int): Number of fast variables (Y) per slow variable.
        nt (int): Number of time steps.

    Returns:
        np.ndarray: Coupling tendencies for the slow variables (X) over time. 
            Shape: (nt+1, K).
    """
    xy_tend_hist = np.zeros((nt + 1, k))

    hcb = (h * c) / b
    y_summed = y_hist.reshape(
        (y_hist.shape[:-1] + (k, j))).sum(axis=-1)
    xy_tend_hist[1:] = -hcb * y_summed[:-1]

    return xy_tend_hist


@njit
def L96_eq1_x_dot(x: np.ndarray, f: float, advect: bool = True) -> np.ndarray:
    """
    Calculate the time rate of change for the X variables in the
    single time scale Lorenz '96 model.

    Equation:
        d/dt X[k] = -X[k-2] X[k-1] + X[k-1] X[k+1] - X[k] + F

    Args:
        x (np.ndarray): Values of X variables at the current time step. Shape: 
            (K,).
        f (float): Forcing term F.
        advect (bool): Whether to include the advection term in the 
            computation. Defaults to True.

    Returns:
        np.ndarray: Array of tendencies for the X variables. Shape: (K,).
    """
    k = len(x)
    x_dot = np.zeros(k)

    if advect:
        # Compute the advection term and add forcing
        x_dot = np.roll(x, 1) * (np.roll(x, -1) - np.roll(x, 2)) - x + f
    else:
        # Only include the linear damping and forcing terms
        x_dot = -x + f

    return x_dot
