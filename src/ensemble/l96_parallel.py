"""
Provides functionality for running multiple Lorenz '96 (L96) simulations
in parallel using multiprocessing.
"""


import multiprocessing as mp
from multiprocessing import Pool

import numpy as np

from models.lorenz96 import L96


def run_single_l96_solver(args):
    """
    Run a single L96 simulation for a given initial state.

    Args:
        args (tuple): A tuple containing:
            - x_init (np.ndarray): Initial conditions for the slow variables (X).
            - y_init (np.ndarray): Initial conditions for the fast variables (Y).
            - t_init (float): Initial time.
            - k (int): Number of slow variables (X).
            - j (int): Number of fast variables (Y) per slow variable.
            - f (float): Forcing term.
            - h (float): Coupling coefficient.
            - b (float): Spatial-scale ratio.
            - c (float): Time-scale ratio.
            - si (float): Sampling interval.
            - t_total (float): Total simulation time.
            - solver (str): Scipy solver used for integration. 
            - solver_method (str): Solver method used for integration (e.g., 'RK45').

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): Array of slow variables (X) over time.
            - y (np.ndarray): Array of fast variables (Y) over time.
            - t (np.ndarray): Array of time points.
    """
    # init_state, config = args
    # x_init, y_init, t_init = init_state
    x_init, y_init, t_init, k, j, f, h, b, c, si, t_total, solver, solver_method = args

    # Initialize the L96 model
    # Seed is irrelevant, as initial state gets set
    model = L96(k, j, f, h, b, c, seed=0)
    model.set_state(x_init, y_init, t_init)

    # Run the simulation
    x, y, t, u = model.run(
        si=si,
        t_total=t_total,
        solver=solver,
        solver_method=solver_method,
        return_coupling=True
    )

    return x, y, t, u


def run_single_l96_manual(args):
    """
    Run a single L96 simulation for a given initial state.

    Args:
        args (tuple): A tuple containing:
            - x_init (np.ndarray): Initial conditions for the slow variables (X).
            - y_init (np.ndarray): Initial conditions for the fast variables (Y).
            - t_init (float): Initial time.
            - k (int): Number of slow variables (X).
            - j (int): Number of fast variables (Y) per slow variable.
            - f (float): Forcing term.
            - h (float): Coupling coefficient.
            - b (float): Spatial-scale ratio.
            - c (float): Time-scale ratio.
            - si (float): Sampling interval.
            - t_total (float): Total simulation time.
            - dt (float): Time step for numerical integration.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): Array of slow variables (X) over time.
            - y (np.ndarray): Array of fast variables (Y) over time.
            - t (np.ndarray): Array of time points.
    """
    # init_state, config = args
    # x_init, y_init, t_init = init_state
    x_init, y_init, t_init, k, j, f, h, b, c, si, t_total, dt = args

    # Initialize the L96 model
    # Seed is irrelevant, as initial state gets set
    model = L96(k, j, f, h, b, c, seed=0)
    model.set_state(x_init, y_init, t_init)

    # Run the simulation
    x, y, t, u = model.run(
        si=si,
        t_total=t_total,
        dt=dt,
        solver='manual',
        return_coupling=True
    )

    return x, y, t, u


def run_l96_parallel(x_init_states, y_init_states, t_init_states, config,
                     perturb=False, seed=None, num_processes=mp.cpu_count()):
    """
    Run multiple L96 simulations in parallel using multiprocessing.

    Args:
        init_states (list): List of initial states, where each state is a tuple (x_init, y_init, t_init).
        config (dict): Configuration dictionary with simulation parameters. Must include:
            - 'K' (int): Number of slow variables (X).
            - 'J' (int): Number of fast variables (Y) per slow variable.
            - 'F' (float): Forcing term.
            - 'h' (float): Coupling coefficient.
            - 'b' (float): Spatial-scale ratio.
            - 'c' (float): Time-scale ratio.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'solver' (str): Integration scheme ('manual', 'solve_ivp' or 'odeint').
            - 'solver_method' (str): Integration method when using scipy solver 'solve_ivp'. 
            - 'dt' (float): Time step for numerical integration when using manual integration. 
        perturb (bool, optional): Whether to perturb the initial states. Defaults to False.
        seed (int, optional): Random seed used for perturbation of initial states. 
            Must be set when `perturb=True`. Defaults to None. 
        num_processes (int, optional): Number of processes to use. Defaults to the number of CPU cores.

    Returns:
        tuple: A tuple containing:
            - x_per_state (np.ndarray): Array of slow variables (X) for all states.
            - y_per_state (np.ndarray): Array of fast variables (Y) for all states.
            - t_per_state (np.ndarray): Array of time points for all states.
            - u_per_state (np.ndarray): Coupling term history for all states.
    """
    if perturb and seed is None:
        raise ValueError(
            "Random seed is required to perturb initial states.")

    if perturb:
        rg = np.random.default_rng(seed=seed)
        x_init_states += rg.normal(0, 1, size=x_init_states.shape)

    if config['solver'] == 'manual':

        tasks = [(x_init_states[i], y_init_states[i], t_init_states[i],
                  config['K'], config['J'], config['F'], config['h'], config['b'],
                  config['c'], config['si'], config['t_total'], config['dt'])
                 for i in range(t_init_states.shape[0])]

        # Use multiprocessing to run simulations in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_single_l96_manual, tasks)

    else:
        tasks = [(x_init_states[i], y_init_states[i], t_init_states[i],
                  config['K'], config['J'], config['F'], config['h'], config['b'],
                  config['c'], config['si'], config['t_total'],
                  config['solver'], config['solver_method']) for i in range(t_init_states.shape[0])]

        # Use multiprocessing to run simulations in parallel
        with Pool(processes=num_processes) as pool:
            results = pool.map(run_single_l96_solver, tasks)

    # Combine results
    x_per_state, y_per_state, t_per_state, u_per_state = zip(*results)

    return (
        np.array(x_per_state),
        np.array(y_per_state),
        np.array(t_per_state),
        np.array(u_per_state)
    )
