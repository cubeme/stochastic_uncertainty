"""
Provides functionality for running ensemble forecasts
using stochastic polynomial parameterizations in parallel with multiprocessing.

Functions:
- run_stochastic_member: Runs a single ensemble member using a GCM with stochastic parameterization.
- run_stochastic_member_manual: Runs a single ensemble member using a manually stepped GCM with stochastic parameterization.
- run_stochastic_ensemble_parallel_multiprocessing: Runs an ensemble forecast in parallel using multiprocessing.
"""

from models.gcm_models import GCM, GCMManual
from parameterization.stochastic_poly_parameterization import PolynomialAR1Parameterization
import multiprocessing as mp
import numpy as np


def run_stochastic_member_solver(args):
    """
    Run a single ensemble member using a GCM with stochastic parameterization.

    Args:
        args (tuple): A tuple containing:
            - i_init (int): Index of the initial state.
            - i_member (int): Index of the ensemble member.
            - init_state (np.ndarray): Initial conditions for the state variables.
            - f (float): Forcing term for the GCM.
            - si (float): Sampling interval.
            - t_total (float): Total simulation time.
            - coefs (np.ndarray): Polynomial coefficients for the parameterization.
            - phi (np.ndarray): AR(1) coefficients for the noise.
            - sigma_e (np.ndarray): Standard deviations of the noise.
            - seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - i_init (int): Index of the initial state.
            - i_member (int): Index of the ensemble member.
            - x_pred (np.ndarray): Predicted state variables over time.
            - t (np.ndarray): Corresponding time points.
    """
    i_init, i_member, init_state, f, si, t_total, coefs, phi, sigma_e, solver, solver_method, seed = args

    stoch_param = PolynomialAR1Parameterization(coefs, phi, sigma_e, seed=seed)
    gcm_model = GCM(f, stoch_param)

    x_pred, t = gcm_model(init_state, si=si, t_total=t_total,
                          solver=solver, solver_method=solver_method)

    return i_init, i_member, x_pred, t


def run_stochastic_member_manual(args):
    """
    Run a single ensemble member using a manually stepped GCM with stochastic parameterization.

    Args:
        args (tuple): A tuple containing:
            - i_init (int): Index of the initial state.
            - i_member (int): Index of the ensemble member.
            - init_state (np.ndarray): Initial conditions for the state variables.
            - f (float): Forcing term for the GCM.
            - si (float): Sampling interval.
            - t_total (float): Total simulation time.
            - coefs (np.ndarray): Polynomial coefficients for the parameterization.
            - phi (np.ndarray): AR(1) coefficients for the noise.
            - sigma_e (np.ndarray): Standard deviations of the noise.
            - dt (float): Time step for numerical integration.
            - seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - i_init (int): Index of the initial state.
            - i_member (int): Index of the ensemble member.
            - x_pred (np.ndarray): Predicted state variables over time.
            - t (np.ndarray): Corresponding time points.
    """
    i_init, i_member, init_state, f, si, t_total, coefs, phi, sigma_e, dt, seed = args

    stoch_param = PolynomialAR1Parameterization(coefs, phi, sigma_e, seed=seed)
    gcm_model = GCMManual(f, stoch_param)

    x_pred, t = gcm_model(init_state, si=si, t_total=t_total, dt=dt)

    return i_init, i_member, x_pred, t


def run_stochastic_ensemble_parallel_multiprocessing(n_init_states, n_ens, init_states, t_total,
                                                     f, si, k, member_func,  coefs, phi, sigma_e,
                                                     seeds, perturb=False, *args):
    """
    Run an ensemble forecast using the given GCM model in parallel with multiprocessing.

    Args:
        n_init_states (int): Number of initial states.
        n_ens (int): Number of ensemble members per initial state.
        step_size_init_states (int): Step size for selecting initial states.
        x_true (np.ndarray): Array of true state variables from which initial states are selected.
        t_total (float): Total simulation time.
        f (float): Forcing term for the GCM.
        si (float): Sampling interval.
        k (int): Number of state variables.
        member_func (callable): Function to run a single ensemble member. Options: 
            - `run_stochastic_member_solver`: Runs a single ensemble member using a GCM using scipy solver time stepping.
            - `run_stochastic_member_manual`: Runs a single ensemble member using a manually stepped GCM.
        seeds (list): List of random seeds for each simulation.
        perturb (bool): Whether to perturb the initial states.
        *args: Additional arguments required by the `member_func`. 
            Options when using scipy solver (`member_func=run_stochastic_member_solver`)
            - solver (str): Integration method for the GCM. Options are 'solve_ivp' or 'odeint'.
            - solver_method (str): Method to use with the 'solve_ivp' solver. Default is 'RK45'.
            Options when using manual solver (`member_func=run_stochastic_member_manual`)
            - dt (float): Time step for numerical integration (used with manual integration).

    Returns:
        tuple: A tuple containing:
            - x_ens_forecast (np.ndarray): Ensemble forecast of the state variables.
              Shape: [n_init_states, n_ens, time_steps, K].
            - t_ens_forecast (np.ndarray): Corresponding time steps for the forecast.
              Shape: [n_init_states, n_ens, time_steps].
            - init_state_indices (np.ndarray): Indices of the initial states used in the forecast.
    """

    # Initialize arrays to store ensemble forecasts for state variables and time
    nt = int(t_total / si)
    x_ens_forecast = np.zeros([n_init_states, n_ens, nt + 1, k])
    t_ens_forecast = np.zeros([n_init_states, n_ens, nt + 1])
    
    if perturb:
        rg = np.random.default_rng(seed=seeds[-1]+1)
        init_states += rg.normal(0, 1, size=init_states.shape)

    # Prepare arguments for each process
    tasks = [(i_init, i_member, init_states[i_init], f, si, t_total, coefs, phi, sigma_e, *args)
             for i_init in range(n_init_states) for i_member in range(n_ens)]
    tasks = [(*task_tuple, s) for task_tuple, s in zip(tasks, seeds)]

    # Use multiprocessing Pool for parallel execution
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(member_func, tasks)

    # Collect results
    for i_init, i_member, x_pred, t in results:
        x_ens_forecast[i_init, i_member] = x_pred
        t_ens_forecast[i_init, i_member] = t

    return x_ens_forecast, t_ens_forecast
