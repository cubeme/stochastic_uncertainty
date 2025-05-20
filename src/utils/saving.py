"""
Provides utility functions for saving simulation results.
"""

import os
from pathlib import Path

import numpy as np

from utils.identifiers import get_ident_str


def save_output_l96(output_folder, config, x, y, t, u, extra_ident=""):
    """
    Save Lorenz '96 simulation results to the specified output folder.

    The results are saved in .npy format in the following directory structure:
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
    f_ident = get_ident_str('l96', config)

    output_path = Path(output_folder, 'L96', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "y.npy"), y)
    np.save(os.path.join(output_path, "t.npy"), t)
    np.save(os.path.join(output_path, "u.npy"), u)


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
        t (numpy.ndarray): Array of time points.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    f_ident = get_ident_str('gcm', config)

    output_path = Path(output_folder, 'gcm',
                       parameterization, extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "t.npy"), t)


def save_output_ensemble(output_folder, config, parameterization, x, t, seeds, extra_ident=""):
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
        parameterization (str): Name of the parameterization used in the simulation.
        x (numpy.ndarray): Array of ensemble state variables.
        t (numpy.ndarray): Array of time points.
        seeds (list): List of random seeds used for reproducibility.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    f_ident = get_ident_str('ensemble', config, seeds=seeds)

    output_path = Path(output_folder, 'ensemble',
                       parameterization, extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "t.npy"), t)


def save_output_l96_ensemble_simulation(output_folder, config, x, y, t, u, seeds, extra_ident=""):
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
        x (numpy.ndarray): Array of slow variables (X).
        y (numpy.ndarray): Array of fast variables (Y).
        t (numpy.ndarray): Array of time points.
        u (numpy.ndarray): Coupling term history.
        seeds (list): List of random seeds used for reproducibility.
        extra_ident (str, optional): Additional identifier for the folder. Default is an empty string.
    """
    f_ident = get_ident_str('ensemble', config, seeds=seeds)

    output_path = Path(output_folder, 'ensemble', 'l96', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "x.npy"), x)
    np.save(os.path.join(output_path, "y.npy"), y)
    np.save(os.path.join(output_path, "t.npy"), t)
    np.save(os.path.join(output_path, "u.npy"), u)
