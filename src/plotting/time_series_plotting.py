
import os
import logging
import math

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_l96_x_with_random_y(x, y, t, time_end, time_start=0, dpi=150,
                             output_folder="", config=None):
    """
    Plots time series for Lorenz '96 slow variables (X) and randomly selected fast variables (Y).

    This function generates a time series plot for each slow variable (X) and a randomly selected
    corresponding fast variable (Y) from the Lorenz '96 model.

    Args:
        x_true (numpy.ndarray): Array of slow variables (X) with shape (time, k).
        y_true (numpy.ndarray): Array of fast variables (Y) with shape (time, jk).
        t (numpy.ndarray): Array of time points with shape (time,).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF. 
            If this is an empty string, the plot will not be saved. Default is an empty string.        
        config (dict, optional): Configuration dictionary containing simulation parameters. 
            Used for naming the output file if 'output_folder' is given. Defaults to None. 
            Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """

    logger = logging.getLogger(__name__)

    # Validate that config is provided if output_folder is specified
    if output_folder != "" and config is None:
        raise ValueError("'config' is None but 'output_folder' is set. \
            Config argument is required for saving the plot.")

    k = x.shape[-1]
    j = y.shape[-1]/k

    sns.set_theme()

    fig = plt.figure(figsize=(10, k * 2), dpi=dpi)

    for i in range(k):
        # Calculate subplot index for the current variable
        subplot_index = int(f"{math.ceil(k/2)}2{i+1}")
        plt.subplot(subplot_index)

        # Select a random index for the fast variable (Y)
        random_y_index = np.random.randint(0, j)
        y_index = f"Y_{random_y_index, i}"

        # Plot the slow variable (X)
        plt.plot(t[time_start:time_end],
                 x[time_start:time_end, i], label=f"$X_{i}(t)$")
        # Plot randomly selected fast variable (Y)
        plt.plot(t[time_start:time_end], y[time_start:time_end, i],
                 label=f"${y_index}(t)$")

        plt.xlabel("Time $t$")
        plt.title(f"Time series for $X_{i}$, ${y_index}(t)$")
        plt.legend()

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"l96_training_time_series_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info("X/Y time series plot saved successfully to %s", save_path)

    return fig


def plot_residuals_with_noise(residuals, ar1_noise, t, time_end, time_start=0, dpi=150,
                              output_folder="", config=None):
    """
    Plots residuals and AR(1) noise time series for Lorenz '96 model.

    This function generates a time series plot for each residual and its corresponding AR(1) noise.

    Args:
        residuals (numpy.ndarray): Array of residuals with shape (time, k).
        ar1_noise (numpy.ndarray): Array of AR(1) noise with shape (time, k).
        t (numpy.ndarray): Array of time points with shape (time,).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.
        config (dict, optional): Configuration dictionary containing simulation parameters.
            Used for naming the output file if 'output_folder' is given. Defaults to None.
            Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'seed' (int): Random seed for reproducibility.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    logger = logging.getLogger(__name__)

    # Validate that config is provided if output_folder is specified
    if output_folder != "" and config is None:
        raise ValueError("'config' is None but 'output_folder' is set. \
            Config argument is required for saving the plot.")

    k = residuals.shape[-1]

    sns.set_theme()

    # Create a figure with subplots for each residual
    fig = plt.figure(figsize=(10, k), dpi=dpi)

    for i in range(k):
        # Calculate subplot index for the current variable
        subplot_index = int(f"{math.ceil(k/2)}2{i+1}")
        plt.subplot(subplot_index)

        # Plot the residuals
        plt.plot(t[time_start:time_end],
                 residuals[time_start:time_end, i], label=f"$Res_{i}(t)$")
        # Plot the AR(1) noise
        plt.plot(t[time_start:time_end],
                 ar1_noise[time_start:time_end, i], label=f"$AR(1)_{i}(t)$")

        plt.xlabel("Time $t$")
        plt.title(f"Residuals for $X_{i}$ with AR(1) noise")
        plt.legend(fontsize=7, loc='upper right')

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"residuals_ar1_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}_rs{config['seed']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "Residuals with AR(1) noise time series plot saved successfully to %s", save_path)

    return fig


def plot_gcm_time_series_comparison(t, x_full, x_no_param, x_det_param, x_stoch_param, time_end,
                                    time_start=0, dpi=150, output_folder="", config=None):
    """
    Plots time series comparison for GCM simulations with different parameterizations.

    This function generates a time series plot for each variable in the GCM simulation, comparing:
    - Full Lorenz '96 model (x_full)
    - No parameterization (x_no_param)
    - Deterministic parameterization (x_det_param)
    - Stochastic parameterization (x_stoch_param)

    Args:
        t (numpy.ndarray): Array of time points with shape (time,).
        x_full (numpy.ndarray): Array of variables from the full Lorenz '96 model 
            with shape (time, k).
        x_no_param (numpy.ndarray): Array of variables from GCM with no 
            parameterization with shape (time, k).
        x_det_param (numpy.ndarray): Array of variables from GCM deterministic 
            parameterization with shape (time, k).
        x_stoch_param (numpy.ndarray): Array of variables from GCM stochastic 
            parameterization with shape (time, k).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        output_folder (str, optional): Path to the folder where the plot will be saved as PDF.
            If this is an empty string, the plot will not be saved. Default is an empty string.
        config (dict, optional): Configuration dictionary containing simulation parameters.
            Used for naming the output file if 'output_folder' is given. Defaults to None.
            Must include:
            - 'c' (float): Time-scale ratio.
            - 'dt' (float): Time step for numerical integration.
            - 'si' (float): Sampling interval.
            - 't_total' (float): Total simulation time.
            - 'seed' (int): Random seed for reproducibility.

    Returns:
        matplotlib.figure.Figure: The generated figure object.
    """
    logger = logging.getLogger(__name__)
    
    # Validate that config is provided if output_folder is specified
    if output_folder != "" and config is None:
        raise ValueError("'config' is None but 'output_folder' is set. \
            Config argument is required for saving the plot.")

    sns.set_theme()

    k = x_full.shape[-1]

    # Create a figure with subplots for each variable
    fig = plt.figure(figsize=(12, k * 2), dpi=dpi)

    for i in range(k):
        # Calculate subplot index for the current variable
        subplot_index = int(f"{math.ceil(k/2)}2{i+1}")
        plt.subplot(subplot_index)

        # Plot the full Lorenz '96 model
        plt.plot(t[time_start:time_end], x_full[time_start:time_end, i],
                 label="Full L96")
        # Plot the no parameterization case
        plt.plot(t[time_start:time_end], x_no_param[time_start:time_end, i],
                 '--', label="No parameterization")
        # Plot the deterministic parameterization case
        plt.plot(t[time_start:time_end], x_det_param[time_start:time_end, i],
                 label="Deterministic parameterization")
        # Plot the stochastic parameterization case
        plt.plot(t[time_start:time_end], x_stoch_param[time_start:time_end, i],
                 label="Stochastic parameterization")

        plt.xlabel("Time $t$")
        plt.title(f"Time series for $X_{i}$")

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"gcm_time_series_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}_rs{config['seed']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "GCM time series comparison plot saved successfully to %s", save_path)

    return fig
