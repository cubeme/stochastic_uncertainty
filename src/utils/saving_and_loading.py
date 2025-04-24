"""
Provides utility functions for saving and loading simulation results.

Functions:
- save_output_l96: Save Lorenz '96 simulation results to the specified output folder.
- load_output_l96: Load Lorenz '96 simulation results from the specified output folder.
- save_output_gcm: Save General Circulation Model (GCM) simulation results.
- load_output_gcm: Load General Circulation Model (GCM) simulation results.
- save_output_ensemble: Save ensemble simulation results.
- load_output_ensemble: Load ensemble simulation results.
- save_output_l96_ensemble_simulation: Save Lorenz '96 ensemble simulation results.
- load_output_l96_ensemble_simulation: Load Lorenz '96 ensemble simulation results.
"""

import os
from pathlib import Path
import numpy as np


def save_output_l96(output_folder, config, x, y, t, u, extra_ident=""):
    """
    Save Lorenz '96 simulation results to the specified output folder.

    The results are saved in .npy format the following directory structure:
    `output_folder`/
        L96/
            c{c}_dt{dt}_si{si}_time{t_total}/
                x.npy
                y.npy
                t.npy
                u.npy

    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
        x (numpy.ndarray): Array of slow variables (X).
        y (numpy.ndarray): Array of fast variables (Y).
        t (numpy.ndarray): Array of time points.
        u (numpy.ndarray): Coupling term history.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}"
    )

    output_path = Path(output_folder, 'L96', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "y.npy"), y)
    np.save(os.path.join(output_path, "t.npy"), t)
    np.save(os.path.join(output_path, "u.npy"), u)


def load_output_l96(output_folder, config, extra_ident=""):
    """
    Load Lorenz '96 simulation results from the specified output folder.

    The results are loaded from the following directory structure:
        `output_folder`/
            L96/
                c{c}_dt{dt}_si{si}_time{t_total}/
                    x.npy
                    y.npy
                    t.npy
                    u.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of slow variables (X).
            - y (numpy.ndarray): Array of fast variables (Y).
            - t (numpy.ndarray): Array of time points.
            - u (numpy.ndarray): Coupling term history.
    """
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}"
    )

    save_folder = Path(output_folder, 'L96', extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    y = np.load(os.path.join(save_folder, "y.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))
    u = np.load(os.path.join(save_folder, "u.npy"))
    return x, y, t, u


def save_output_gcm(output_folder, config, parameterization, x, t, extra_ident=""):
    """
    Save GCM simulation results to the specified output folder.

    The results are saved in .npy format the following directory structure:
        `output_folder`/
            gcm/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_rs{seed}/
                        x.npy
                        t.npy

    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        x (numpy.ndarray): Array of state variables.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"rs{config['seed']}"
    )

    output_path = Path(output_folder, 'gcm',
                       parameterization, extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "t.npy"), t)


def load_output_gcm(output_folder, config, parameterization, extra_ident=""):
    """
    Load GCM simulation results from the specified output folder.

    The results are loaded from the following directory structure:
        `output_folder`/
            gcm/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_rs{seed}/
                        x.npy
                        t.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of state variables.
            - t (numpy.ndarray): Array of time points.
    """
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"rs{config['seed']}"
    )

    save_folder = Path(output_folder, 'gcm',
                       parameterization, extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))

    return x, t


def save_output_ensemble(output_folder, config, parameterization, x, t, init_idx, seeds, extra_ident=""):
    """
    Save ensemble simulation results to the specified output folder.

    The results are saved in .npy format in the following directory structure:
        `output_folder`/
            ensemble/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_init{init_states}_ens{n_ens}_rs{seed}/
                        x.npy
                        t.npy
                        init_indices.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'init_states' (int): Number of initial states.
            - 'n_ens' (int): Number of ensemble members.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        x (numpy.ndarray): Array of ensemble state variables.
        t (numpy.ndarray): Array of time points.
        init_idx (numpy.ndarray): Array of initial state indices.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    seed_str = f"{seeds[0]}-{seeds[-1]}"
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"init{config['init_states']}_"
        f"ens{config['n_ens']}_"
        f"rs{seed_str}"
    )

    output_path = Path(output_folder, 'ensemble',
                       parameterization, extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "t.npy"), t)
    np.save(os.path.join(output_path, "init_indices.npy"), init_idx)


def load_output_ensemble(output_folder, config, parameterization, seeds, extra_ident=""):
    """
    Load ensemble simulation results from the specified output folder.


    The results are loaded from the following directory structure:
        `output_folder`/
            ensemble/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_init{init_states}_ens{n_ens}_rs{seed}/
                        x.npy
                        t.npy
                        init_indices.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'init_states' (int): Number of initial states.
            - 'n_ens' (int): Number of ensemble members.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of ensemble state variables.
            - t (numpy.ndarray): Array of time points.
            - init_idx (numpy.ndarray): Array of initial state indices.
    """
    seed_str = f"{seeds[0]}-{seeds[-1]}"
    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"init{config['init_states']}_"
        f"ens{config['n_ens']}_"
        f"rs{seed_str}"
    )
    save_folder = Path(output_folder, 'ensemble',
                       parameterization, extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))
    init_indices = np.load(os.path.join(save_folder, "init_indices.npy"))

    return x, t, init_indices


def save_output_l96_ensemble_simulation(output_folder, config, x, y, t, u, seeds=None, extra_ident=""):
    """
    Save ensemble simulation results to the specified output folder.

    The results are saved in .npy format in the following directory structure:
        `output_folder`/
            ensemble/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_init{init_states}_ens{n_ens}_rs{seed}/
                        x.npy
                        t.npy
                        init_indices.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'init_states' (int): Number of initial states.
            - 'n_ens' (int): Number of ensemble members.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        x (numpy.ndarray): Array of ensemble state variables.
        t (numpy.ndarray): Array of time points.
        init_idx (numpy.ndarray): Array of initial state indices.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    if seeds is not None:
        seed_str = f"{seeds[0]}-{seeds[-1]}"
    else:
        seed_str = config['seed']

    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"init{config['init_states']}_"
        f"ens{config['n_ens']}_"
        f"rs{seed_str}"
    )

    output_path = Path(output_folder, 'ensemble', 'l96', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "y.npy"), y)
    np.save(os.path.join(output_path, "t.npy"), t)
    np.save(os.path.join(output_path, "u.npy"), u)


def load_output_l96_ensemble_simulation(output_folder, config, seeds=None, extra_ident=""):
    """
    Save ensemble simulation results to the specified output folder.

    The results are saved in .npy format in the following directory structure:
        `output_folder`/
            ensemble/
                `parameterization`/
                    c{c}_dt{dt}_si{si}_time{t_total}_init{init_states}_ens{n_ens}_rs{seed}/
                        x.npy
                        t.npy
                        init_indices.npy
    Args:
        output_folder (str): Path to the output folder.
        config (dict): Configuration dictionary containing simulation parameters. Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'init_states' (int): Number of initial states.
            - 'n_ens' (int): Number of ensemble members.
            - 'seed' (int): Random seed for reproducibility.
        parameterization (str): Name of the parameterization used in the simulation.
        x (numpy.ndarray): Array of ensemble state variables.
        t (numpy.ndarray): Array of time points.
        init_idx (numpy.ndarray): Array of initial state indices.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of slow variables (X).
            - y (numpy.ndarray): Array of fast variables (Y).
            - t (numpy.ndarray): Array of time points.
            - u (numpy.ndarray): Coupling term history.

    """
    if seeds is not None:
        seed_str = f"{seeds[0]}-{seeds[-1]}"
    else:
        seed_str = config['seed']

    f_ident = (
        f"c{config['c']}_"
        f"dt{config['dt']}_"
        f"si{config['si']}_"
        f"time{config['t_total']}_"
        f"init{config['init_states']}_"
        f"ens{config['n_ens']}_"
        f"rs{seed_str}"
    )

    output_path = Path(output_folder, 'ensemble', 'l96', extra_ident, f_ident)

    x = np.load(os.path.join(output_path, "x.npy"))
    y = np.load(os.path.join(output_path, "y.npy"))
    t = np.load(os.path.join(output_path, "t.npy"))
    u = np.load(os.path.join(output_path, "u.npy"))

    return x, y, t, u
