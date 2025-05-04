"""
Provides implementations of the Lorenz '96 model, including
a two-time-scale version with slow (X) and fast (Y) variables.

Functions:
- L96_2t_x_dot_y_dot_solver: Computes the time derivatives for the two-time-scale Lorenz '96 model using a solver-friendly format.
- L96_2t_x_dot_y_dot_manual: Computes the time derivatives for the two-time-scale Lorenz '96 model manually.
- calculate_xy_tendencies: Computes the coupling tendencies from fast Y variables.

Classes:
- L96: A class for simulating the two-time-scale Lorenz '96 model.
"""

import logging
import numpy as np
from numba import njit
from scipy.integrate import solve_ivp, odeint


@njit
def L96_2t_x_dot_y_dot_solver(t, xy_vars, K, J, F, h, b, c):
    """
    Compute the time derivatives for the two-time-scale Lorenz '96 model.

    Equations:
        d/dt X[k] = -X[k-1] (X[k-2] - X[k+1]) - X[k] + F - h.c/b sum_j Y[j,k]
        d/dt Y[j] = -b c Y[j+1] (Y[j+2] - Y[j-1]) - c Y[j] + h.c/b X[k]

    Args:
        t (float): Current time (not used in the computation but required for solvers).
        xy_vars (np.ndarray): Concatenated array of X and Y variables. Shape: (K + K*J,).
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
        y_hist (np.ndarray): History of the fast variables (Y) over time. Shape: (nt+1, K*J).
        h (float): Coupling coefficient.
        c (float): Time-scale ratio.
        b (float): Ratio of amplitudes.
        k (int): Number of slow variables (X).
        j (int): Number of fast variables (Y) per slow variable.
        nt (int): Number of time steps.

    Returns:
        np.ndarray: Coupling tendencies for the slow variables (X) over time. Shape: (nt+1, K).
    """
    xy_tend_hist = np.zeros((nt + 1, k))

    hcb = (h * c) / b
    y_summed = y_hist.reshape(
        (y_hist.shape[:-1] + (k, j))).sum(axis=-1)
    xy_tend_hist[1:] = -hcb * y_summed[:-1]

    return xy_tend_hist


class L96:
    """
    Class for the two time-scale Lorenz '96 model.
    This model simulates a system with slow (X) and fast (Y) variables, 
    coupled through a set of differential equations.

    Attributes:
        x (numpy.ndarray): Current state or initial conditions for the slow variables (X).
        y (numpy.ndarray): Current state or initial conditions for the fast variables (Y).
        f (float): Forcing term for the slow variables.
        h (float): Coupling coefficient between the slow and fast variables.
        b (float): Ratio of amplitudes between the fast and slow variables.
        c (float): Time-scale ratio between the fast and slow variables.
        t (float): Current time or initial time.
        k (int): Number of slow variables (X).
        j (int): Number of fast variables (Y) per slow variable.
    """

    def __init__(self, K, J, F=18, h=1, b=10, c=10, t=0, seed=17):
        """
        Initialize the Lorenz '96 model with the given parameters.

        Args:
            K (int): Number of slow variables (X).
            J (int): Number of fast variables (Y) per slow variable.
            F (float): Forcing term for the slow variables. Default is 18.
            h (float): Coupling coefficient. Default is 1.
            b (float): Ratio of amplitudes. Default is 10.
            c (float): Time-scale ratio. Default is 10.
            t (float): Initial time. Default is 0.
            seed (int): Random seed for reproducibility. Default is 17.
        """
        np.random.seed(seed)

        self.logger = logging.getLogger(__name__)

        self.f, self.h, self.b, self.c = F, h, b, c
        # Initialize system state
        self.x, self.y, self.t = b * \
            np.random.randn(K), np.random.randn(J * K), t
        self.k, self.j = K, J

        self.jk = J * K    # Total number of fast variables
        self.range_k, self.range_jk = np.arange(
            self.k), np.arange(self.jk)  # For plotting

    def set_param(self, F=None, h=None, b=None, c=None, t=None):
        """
        Set model parameters.

        Args:
            dt (float, optional): Time step for numerical integration.
            F (float, optional): Forcing term for the slow variables.
            h (float, optional): Coupling coefficient.
            b (float, optional): Ratio of amplitudes.
            c (float, optional): Time-scale ratio.
            t (float, optional): Current time or initial time.
        """
        if F is not None:
            self.f = F
        if h is not None:
            self.h = h
        if b is not None:
            self.b = b
        if c is not None:
            self.c = c
        if t is not None:
            self.t = t
        return self

    def set_state(self, X, Y, t=None):
        """
        Set the state of the system (initial conditions or current state).

        Args:
            X (numpy.ndarray): State of the slow variables.
            Y (numpy.ndarray): State of the fast variables.
            t (float, optional): Current time or initial time.
        """
        # Set state
        self.x, self.y = X, Y
        if t is not None:
            self.t = t

        # Compute K and J from state
        self.k, self.jk = self.x.size, self.y.size
        self.j = self.jk // self.k

        # For plotting
        self.range_k, self.range_jk = np.arange(self.k), np.arange(self.jk)

        return self

    def randomize_IC(self):
        """
        Randomize the initial conditions (or current state) for the system.

        Returns:
            L96: The updated model instance with randomized initial conditions.
        """
        X, Y = self.b * \
            np.random.rand(self.x.size), np.random.rand(self.y.size)
        return self.set_state(X, Y)

    def run(self, si, t_total, store=False, return_coupling=False, dt=0.001, solver='manual', solver_method='RK45'):
        """
        Run the Lorenz '96 model for a total time `t_total`, sampling at intervals of `si`.

        Args:
            si (float): Sampling interval (time increment for each step).
            t_total (float): Total simulation time.
            store (bool, optional): If True, stores the final state as the initial conditions for the next run.
            return_coupling (bool, optional): If True, returns the coupling term in addition to X, Y, and time.
            dt (float, optional): Time step for numerical integration. Default is 0.001. Only used with manual integration.
            solver (str, optional): Integration method. Options are 'solve_ivp', 'ode_int', or 'manual'.
                                    Default is 'manual'.
            solver_method (str, optional): Method to use with the 'solve_ivp' solver. Default is 'RK45'.

        Returns:
            tuple:
                - x_hist (numpy.ndarray): History of the slow variables (X) over time.
                - y_hist (numpy.ndarray): History of the fast variables (Y) over time.
                - time (numpy.ndarray): Array of time points corresponding to the simulation.
                - xy_tend_hist (numpy.ndarray, optional): Coupling term history (if `return_coupling=True`).
        """

        implemented_methods = ['solve_ivp', 'odeint', 'manual']
        if solver not in implemented_methods:
            raise ValueError(
                f"Unknown method: {solver}. Method must be one of {implemented_methods}")
        if solver == 'manual' or solver == 'odeint':
            self.logger.info(
                "`solver_method=%s` is only used with `solver='solve_ivp'`.", solver_method)
        if solver == 'solve_ivp' or solver == 'odeint':
            self.logger.info(
                "`dt=%f` is only used with `solver='manual'`.", dt)

        # Number of time steps
        nt = int(t_total / si)

        # todo: remove copy() this after debugging
        x_0 = self.x.copy()
        y_0 = self.y.copy()
        t_0 = self.t

        # Result arrays
        x_hist, y_hist = (
            np.zeros((nt + 1, len(x_0))),
            np.zeros((nt + 1, len(y_0))),
        )

        x_hist[0, :] = x_0.copy()
        y_hist[0, :] = y_0.copy()

        # Evaluation interval with si size steps
        t_eval = np.arange(si, (si * nt) + si, si) + t_0

        if solver == 'solve_ivp':
            # Using solve_ivp
            ode_result = solve_ivp(L96_2t_x_dot_y_dot_solver, t_span=(0, nt),
                                   y0=np.concatenate((x_0, y_0)), method=solver_method,
                                   t_eval=t_eval, args=(self.k, self.j, self.f, self.h, self.b, self.c))

            # Collect results
            x_hist[1:] = np.swapaxes(ode_result.y[:self.k], 0, 1)
            y_hist[1:] = np.swapaxes(ode_result.y[self.k:], 0, 1)

            xy_tend_hist = calculate_xy_tendencies(
                y_hist, self.h, self.c, self.b, self.k, self.j, nt)

            time = t_0 + np.zeros((nt + 1))
            time[1:] = ode_result.t

        elif solver == 'odeint':
            # Using odeint
            ode_result = odeint(L96_2t_x_dot_y_dot_solver,  y0=np.concatenate((x_0, y_0)), t=t_eval,
                                args=(self.k, self.j, self.f, self.h, self.b, self.c), tfirst=True)
            # Collect results
            x_hist[1:] = ode_result[:, :self.k]
            y_hist[1:] = ode_result[:, self.k:]

            xy_tend_hist = calculate_xy_tendencies(
                y_hist, self.h, self.c, self.b, self.k, self.j, nt)

            time = np.arange(t_0, t_total+t_0+si, step=si)

        else:  # manual
            if si < dt:
                dt, ns = si, 1
            else:
                ns = int(si / dt + 0.5)
                assert (
                    abs(ns * dt - si) < 1e-14
                ), f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"

            xy_tend_hist = np.zeros((nt + 1, len(x_0)))
            
            x = x_0.copy()
            y = y_0.copy()
            time = t_0 + np.zeros((nt + 1))

            for n in range(nt):
                for s in range(ns):
                    # RK4 update of X,Y
                    x_dot1, y_dot1, xy_tend = L96_2t_x_dot_y_dot_manual(
                        x, y, self.f, self.h, self.b, self.c)
                    x_dot2, y_dot2, _ = L96_2t_x_dot_y_dot_manual(
                        x + 0.5 * dt * x_dot1, y + 0.5 * dt * y_dot1, self.f, self.h, self.b, self.c)
                    x_dot3, y_dot3, _ = L96_2t_x_dot_y_dot_manual(
                        x + 0.5 * dt * x_dot2, y + 0.5 * dt * y_dot2, self.f, self.h, self.b, self.c)
                    x_dot4, y_dot4, _ = L96_2t_x_dot_y_dot_manual(
                        x + dt * x_dot3, y + dt * y_dot3,   self.f, self.h, self.b, self.c)

                    x = x + (dt / 6.0) * ((x_dot1 + x_dot4) +
                                          2.0 * (x_dot2 + x_dot3))
                    y = y + (dt / 6.0) * ((y_dot1 + y_dot4) +
                                          2.0 * (y_dot2 + y_dot3))

                x_hist[n + 1] = x
                y_hist[n + 1] = y
                time[n + 1] = t_0 + si * (n + 1)
                xy_tend_hist[n + 1] = xy_tend

        if store:
            self.x, self.y, self.t = x_hist[-1], y_hist[-1], time[-1]
        if return_coupling:
            return x_hist, y_hist, time, xy_tend_hist

        return x_hist, y_hist, time

    def __repr__(self):
        return f"L96: K={self.k} J={self.j} F={self.f} h={self.h} b={self.b} c={self.c}"

    def __str__(self):
        return (self.__repr__() + f"\n X={self.x} \nY={self.y} \nt={self.t}")

    def copy(self):
        copy = L96(self.k, self.j, F=self.f, h=self.h, b=self.b, c=self.c)
        copy.set_state(self.x, self.y, t=self.t)
        return copy

    def print(self):
        print(self)
