"""Module providing utility functions for notebooks."""

import sys

sys.path.insert(1, '../src')

import numpy as np
from lorenz96_model_tutorial import L96_eq1_xdot
from scipy.integrate import solve_ivp
import flags

FLAGS = flags.Flags({
    'K': 8,  # Number of global-scale variables X
    'J': 32,  # Number of local-scale Y variables per single global-scale X variable
    'h': 1.0,  # Coupling coefficient
    'F': 20.0,  # Forcing
    'b': 10.0,  # spatial-scale ratio
    # time-scale ratio, 4 (small time-scale separation, hard) or 10 (large time-scale separation, easy)
    'c': 10.0,


    'dt': 0.01,  # Time step
    'si': 0.1,  # Sampling time interval (50 atmospheric days)
    'int_scheme': 'RK4',  # Integration scheme
    # Truth:
    # dt: 0.001
    # si: 10
    # int: RK4

    # Forecast:
    # dt: 0.005
    # si: 10
    # int: RK2

    'n_ics': 300,  # Number of initial conditions
    'n_ens': 40,  # Number of ensemble members
    't_total': 300,  # Total time for the model run
    'time_steps': 20000
})


def add_source_to_path():
    """Add the source directory to sys.path."""
    sys.path.insert(1, '../src')


def run_member(args):
    """Run a single ensemble member."""
    i_init, i_member, init_state_indices, x_true, det_param_p3_poly, phi, sigma_e = args

    gcm_model = GCM_stochastic(
        FLAGS['F'], det_param_p3_poly, noise_params=[phi, sigma_e])

    init_state = x_true[init_state_indices[i_init]]
    x_pred, t = gcm_model(init_state, si=FLAGS['si'], nt=int(
        FLAGS['t_total'] / FLAGS['si']))

    return i_init, i_member, x_pred, t


class GCM_stochastic:
    """GCM with stochastic parameterization in rhs of equation for tendency"""

    def __init__(self, f, param_poly, noise_params):
        self.f = f
        self.param_poly = param_poly

        self.phi = noise_params[0]
        self.sigma_e = noise_params[1]

        self.noise_t_minus_1 = np.zeros_like(self.phi)

    def rhs(self, t, x):
        """
        Compute the right-hand side (RHS) of the tendency equation.

        Args:
            x (numpy.ndarray): Current large-scale state of the system.
            param (list or numpy.ndarray): Parameters of the parameterization function.

        Returns:
            numpy.ndarray: The computed RHS of the tendency equation.
        """
        noise_t = self.phi * self.noise_t_minus_1 + \
            self.sigma_e * np.sqrt(1-self.phi**2) * \
            np.random.normal(0, 1, size=self.phi.shape)

        self.noise_t_minus_1 = noise_t

        return L96_eq1_xdot(x, self.f) + (self.param_poly(x) + noise_t)

    def __call__(self, x_init, si, nt):
        """
        Integrate the system forward in time.

        Args:
            x_init (numpy.ndarray): Initial conditions for the large-scale state.
            si (float): Time increment for each step.
            nt (int): Number of forward steps to take.
            param (list or numpy.ndarray, optional): Parameters of the parameterization function. Defaults to [0].

        Returns:
            tuple: A tuple containing:
                - hist (numpy.ndarray): History of the large-scale state over time.
                - time (numpy.ndarray): Array of time points corresponding to the simulation.
        """
        time = si * np.arange(nt + 1)  # simulation time
        # empty history for large-scale state
        hist = np.zeros((nt + 1, len(x_init))) * np.nan
        x = x_init.copy()  # initial large-scale values
        t_eval = np.arange(0, FLAGS['t_total'], si)

        hist[0] = x

        ode_result = solve_ivp(self.rhs, (0, nt), y0=x,
                               method='RK45', t_eval=t_eval)
        hist[1:] = np.swapaxes(ode_result.y, 0, 1)
        time[1:] = ode_result.t

        return hist, time
