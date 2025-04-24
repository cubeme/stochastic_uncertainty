"""
Provides functionality for running multiple Lorenz '96 (L96) simulations
in parallel using multiprocessing.

Functions:
- run_single_l96_simulation: Runs a single L96 simulation for a given initial state and configuration.
- run_l96_parallel: Runs multiple L96 simulations in parallel using multiprocessing.
"""


from multiprocessing import Pool
import multiprocessing as mp
import numpy as np
from models.lorenz96_model import L96


def run_single_l96_simulation(args):
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
            - method (str): Integration method (e.g., 'RK4').
            - seed (int): Random seed for reproducibility.

    Returns:
        tuple: A tuple containing:
            - x (np.ndarray): Array of slow variables (X) over time.
            - y (np.ndarray): Array of fast variables (Y) over time.
            - t (np.ndarray): Array of time points.
    """
    # init_state, config = args
    # x_init, y_init, t_init = init_state
    x_init, y_init, t_init, k, j, f, h, b, c, si, t_total, dt, method, seed = args

    # Initialize the L96 model
    model = L96(k, j, f, h, b, c, seed=seed)
    model.set_state(x_init, y_init, t_init)

    # Run the simulation
    x, y, t, u = model.run(
        si=si,
        t_total=t_total,
        dt=dt,
        method=method,
        return_coupling=True
    )

    return x, y, t, u


def run_l96_parallel(init_states, config, num_processes=mp.cpu_count(), seeds=None):
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
            - 'dt' (float): Time step for numerical integration.
            - 'method' (str): Integration method (e.g., 'RK4').
            - 'seed' (int): Random seed for reproducibility.
        num_processes (int, optional): Number of processes to use. Defaults to the number of CPU cores.
        seeds (list, optional): List of random seeds for each simulation. If provided, each simulation gets a unique seed.

    Returns:
        tuple: A tuple containing:
            - x_per_state (np.ndarray): Array of slow variables (X) for all states.
            - y_per_state (np.ndarray): Array of fast variables (Y) for all states.
            - t_per_state (np.ndarray): Array of time points for all states.
            - u_per_state (np.ndarray): Coupling term history for all states.
    """
    # todo: which seed version should be use? 
    if seeds is None:
        # Every L96 model gets the same seed
        tasks = [(x_init, y_init, t_init, config['K'], config['J'], config['F'],
                  config['h'], config['b'], config['c'], config['si'], config['t_total'],
                  config['dt'], config['method'], config['seed']) for (x_init, y_init, t_init) in init_states]
    else:
        # Every L96 model gets a different seed
        # Should be just one init state in this case
        tasks = [(x_init, y_init, t_init, config['K'], config['J'], config['F'],
                  config['h'], config['b'], config['c'], config['si'], config['t_total'],
                  config['dt'], config['method'], seed) for (x_init, y_init, t_init) in init_states for seed in seeds]

    # Use multiprocessing to run simulations in parallel
    with Pool(processes=num_processes) as pool:
        results = pool.map(run_single_l96_simulation, tasks)

    # Combine results
    x_per_state, y_per_state, t_per_state, u_per_state = zip(*results)

    return (
        np.array(x_per_state),
        np.array(y_per_state),
        np.array(t_per_state),
        np.array(u_per_state)
    )
