"""
Provides implementations of General Circulation Models (GCMs) based on the Lorenz '96 model.
"""

import logging
from typing import Callable, Tuple

import numpy as np
from scipy.integrate import odeint, solve_ivp

from models.helpers import L96_eq1_x_dot
from utils.time_stepping import euler_forward


class GCM:
    """
    General Circulation Model (GCM) of the Lorenz '96 model using numerical solvers.
    """

    def __init__(self, f: float,
                 parameterization: Callable[[np.ndarray], np.ndarray] = lambda x: 0):
        """
        Initialize GCM instance.

        Args:
            f (float): Forcing term for the Lorenz '96 model.
            parameterization (Callable): Parameterization function to modify 
                the RHS of the tendency equation. Defaults to a zero function.
        """
        self.f = f
        self.parameterization = parameterization
        self.logger = logging.getLogger(__name__)

    def rhs(self, t: float, x: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side (RHS) of the tendency equation with parameterization.

        Args:
            t (float): Current time (not used in the computation but required 
                for ODE solvers).
            x (np.ndarray): Current state of the large-scale variables.

        Returns:
            np.ndarray: The computed RHS of the parameterized tendency 
                equation.
        """
        return L96_eq1_x_dot(x, self.f) + self.parameterization(x)

    def __call__(self, x_init: np.ndarray, si: float, t_total: float,
                 solver: str = 'solve_ivp', solver_method: str = 'RK45') -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrate the system forward in time.

        Integrate the system forward in time.

        Args:
            x_init (np.ndarray): Initial conditions for the large-scale state 
                variables. Shape: (K,).
            si (float): Sampling interval (time increment for each step).
            t_total (float): Total simulation time.
            method (str): Numerical solver to use ('solve_ivp' or 'odeint'). 
                Defaults to 'solve_ivp'.
            solver_method (str, optional): Method to use with the 'solve_ivp' 
                solver. Default is 'RK45'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - hist (np.ndarray): History of the large-scale state over 
                    time. Shape: (nt + 1, K).
                - time (np.ndarray): Array of time points corresponding to the 
                    simulation. Shape: (nt + 1,).
        """
        implemented_methods = ['solve_ivp', 'odeint']
        if solver not in implemented_methods:
            raise ValueError(
                f"Unknown method: {solver}. Method must be one of {implemented_methods}")
        if solver == 'odeint':
            self.logger.info(
                "`solver_method=%s` is only used with `solver=solve_ivp`.", solver_method)

        # Number of time steps
        nt = int(t_total / si)

        # Initialize history array for storing the state variables
        hist = np.zeros((nt + 1, len(x_init))) * np.nan
        t_eval = np.arange(0, t_total, si)

        hist[0] = x_init

        # Solve the ODE using the specified RHS
        if solver == 'odeint':
            ode_result = odeint(self.rhs, y0=x_init, t=t_eval, tfirst=True)

            hist[1:] = ode_result
            # Simulation time
            time = np.arange(0, t_total + si, step=si)

        elif solver == 'solve_ivp':
            ode_result = solve_ivp(
                self.rhs, (0, nt), y0=x_init, method=solver_method, t_eval=t_eval)

            hist[1:] = np.swapaxes(ode_result.y, 0, 1)
            # Simulation time
            time = si * np.arange(nt + 1)
            time[1:] = ode_result.t

        return hist, time


class GCMManual:
    """
    General Circulation Model (GCM) of the Lorenz '96 model using manual time-stepping methods.
    """

    def __init__(self, f: float,
                 parameterization: Callable[[np.ndarray], np.ndarray] = lambda x: 0):
        """
        Initialize GCM instance.

        Args:
            f (float): Forcing term for the Lorenz '96 model.
            parameterization (Callable): Parameterization function to modify 
                the RHS of the tendency equation. Defaults to a zero function.
        """
        self.f = f
        self.parameterization = parameterization

    def rhs(self, x: np.ndarray) -> np.ndarray:
        """
        Compute the right-hand side (RHS) of the tendency equation with parameterization.

        Args:
            x (np.ndarray): Current state of the large-scale variables.

        Returns:
            np.ndarray: The computed RHS of the parameterized tendency 
                equation.
        """
        return L96_eq1_x_dot(x, self.f) + self.parameterization(x)

    def __call__(self, x_init: np.ndarray, dt: float, si: float, t_total:
                 float, time_stepping_func: Callable = euler_forward) -> Tuple[np.ndarray, np.ndarray]:
        """
        Integrates the system forward in time.

        Args:
            x_init (np.ndarray): Initial conditions for the large-scale state 
                variables. Shape: (K,).
            dt (float): Time step for numerical integration.
            si (float): Sampling interval (time increment for each step).
            t_total (float): Total simulation time.
            time_stepping_func (Callable): Time-stepping function (e.g., RK4). 
                Defaults to 'euler_forward'.

        Returns:
            Tuple[np.ndarray, np.ndarray]: A tuple containing:
                - hist (np.ndarray): History of the large-scale state over 
                    time. Shape: (nt + 1, K).
                - time (np.ndarray): Array of time points corresponding to the 
                    simulation. Shape: (nt + 1,).
        """
        # Number of time steps
        nt = int(t_total / si)

        # Compute number of integration steps
        if si < dt:
            dt, ns = si, 1
        else:
            ns = int(si / dt + 0.5)
            assert (
                abs(ns * dt - si) < 1e-14
            ), f"si is not an integer multiple of dt: si={si} dt={dt} ns={ns}"

        # Initialize history array for storing the state variables
        hist = np.zeros((nt + 1, len(x_init))) * np.nan
        time = np.zeros((nt + 1))

        hist[0] = x_init

        x = x_init

        for n in range(nt):
            for _ in range(ns):
                x = time_stepping_func(self.rhs, dt, x)

            hist[n + 1], time[n + 1] = x, si * (n + 1)

        return hist, time
