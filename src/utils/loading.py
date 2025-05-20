"""
Provides utility functions for loading simulation results.
"""

import os
from pathlib import Path

import numpy as np

from utils.identifiers import get_ident_str


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
    f_ident = get_ident_str('l96', config)

    save_folder = Path(output_folder, 'L96', extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    y = np.load(os.path.join(save_folder, "y.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))
    u = np.load(os.path.join(save_folder, "u.npy"))
    return x, y, t, u


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
    f_ident = get_ident_str('gcm', config)

    save_folder = Path(output_folder, 'gcm',
                       parameterization, extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))

    return x, t


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
        parameterization (str): Name of the parameterization used in the simulation.
        seeds (list): List of random seeds used for reproducibility.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of ensemble state variables.
            - t (numpy.ndarray): Array of time points.
            - init_idx (numpy.ndarray): Array of initial state indices.
    """
    f_ident = get_ident_str('ensemble', config, seeds=seeds)

    save_folder = Path(output_folder, 'ensemble',
                       parameterization, extra_ident, f_ident)

    x = np.load(os.path.join(save_folder, "x.npy"))
    t = np.load(os.path.join(save_folder, "t.npy"))

    return x, t


def load_output_l96_ensemble_simulation(output_folder, config, seeds, extra_ident=""):
    """
    Load ensemble simulation results from the specified output folder.

    The results are loaded from the following directory structure:
        `output_folder`/
            ensemble/
                l96/
                    c{c}_dt{dt}_si{si}_time{t_total}_init{init_states}_ens{n_ens}_rs{seed}/
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
            - 'init_states' (int): Number of initial states.
            - 'n_ens' (int): Number of ensemble members.
        seeds (list): List of random seeds used for reproducibility.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.

    Returns:
        tuple: A tuple containing:
            - x (numpy.ndarray): Array of slow variables (X).
            - y (numpy.ndarray): Array of fast variables (Y).
            - t (numpy.ndarray): Array of time points.
            - u (numpy.ndarray): Coupling term history.
    """
    f_ident = get_ident_str('ensemble', config, seeds=seeds)

    output_path = Path(output_folder, 'ensemble', 'l96', extra_ident, f_ident)

    x = np.load(os.path.join(output_path, "x.npy"))
    y = np.load(os.path.join(output_path, "y.npy"))
    t = np.load(os.path.join(output_path, "t.npy"))
    u = np.load(os.path.join(output_path, "u.npy"))

    return x, y, t, u
