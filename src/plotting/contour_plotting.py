import os
import logging

import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


def plot_l96_x_and_y(x, y, t, time_end, time_start=0, dpi=150,
                     cmap='viridis', output_folder="", config=None):
    """
    Plots a contour plot for Lorenz '96 slow variables (X) and fast variables (Y).

    This function generates two contour plots in one figure:
    - The first plot shows the evolution of the slow variables X over time.
    - The second plot shows the evolution of the corresponding fast variables Y over time.

    Args:
        x_true (numpy.ndarray): Array of slow variables (X) with shape (time, k).
        y_true (numpy.ndarray): Array of fast variables (Y) with shape (time, jk).
        t (numpy.ndarray): Array of time points.
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        cmap (str, optional): Colormap to use for the contour plots. Default is 'viridis'.
        output_folder (str, optional): Path to the folder where the plot will be saved as a PDF. 
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
    jk = y.shape[-1]
    j = jk/k

    sns.set_theme()
    fig = plt.figure(figsize=(10, 6), dpi=dpi)

    # Plot the slow variables (X)
    plt.subplot(121)  # nrows, ncols, index
    plt.contourf(np.arange(k), t[time_start:time_end],
                 x[time_start:time_end, :], cmap=cmap)
    plt.colorbar()
    plt.xlabel("k")
    plt.ylabel("t")
    plt.title("$X_k(t)$")

    # Plot the fast variables (Y)
    plt.subplot(122)
    plt.contourf(np.arange(jk) / j, t[time_start:time_end],
                 y[time_start:time_end, :], levels=np.linspace(-1, 1, 10),  cmap=cmap)
    plt.xlabel("k+j/J")
    plt.ylabel("t")
    plt.title("$Y_{j,k}(t)$")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"l96_training2d_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info("L96 contour plot saved successfully to %s", save_path)

    return fig


def plot_gcm_contour_comparison(t, x_full, x_no_param, x_det_param, x_stoch_param, time_end,
                                time_start=0, dpi=150, cmap='viridis', output_folder="", config=None):
    """
    Plots contour comparison for GCM simulations with different parameterizations.

    This function generates a contour plot for each variable in the GCM simulation, comparing:
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
        x_det_param (numpy.ndarray): Array of variables from GCM with deterministic 
            parameterization with shape (time, k).
        x_stoch_param (numpy.ndarray): Array of variables from GCM with stochastic 
            parameterization with shape (time, k).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        cmap (str, optional): Colormap to use for the contour plots. Default is 'viridis'.
        output_folder (str, optional): Path to the folder where the plot will be saved as a PDF.
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

    k = x_full.shape[-1]
    range_k = np.arange(k)

    sns.set_theme()

    fig = plt.figure(dpi=dpi, figsize=(17, 8))

    # Define contour plot limits and levels
    limits = {"vmin": -12, "vmax": 12,
              "levels": np.linspace(-12, 12, 12), "extend": "both"}

    # Plot the full Lorenz '96 model
    plt.subplot(141)
    plt.contourf(range_k, t[time_start:time_end],
                 x_full[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("Full L96")

    # Plot the no parameterization case
    plt.subplot(142)
    plt.contourf(range_k, t[time_start:time_end],
                 x_no_param[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("GCM without parameterization")

    # Plot the deterministic parameterization case
    plt.subplot(143)
    plt.contourf(range_k, t[time_start:time_end],
                 x_det_param[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("GCM with deterministic parameterization")

    # Plot the stochastic parameterization case
    plt.subplot(144)
    plt.contourf(range_k, t[time_start:time_end],
                 x_stoch_param[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("GCM with stochastic parameterization")

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"gcm_2d_comparison_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}_rs{config['seed']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "GCM contour plot comparison saved successfully to %s", save_path)

    return fig


def plot_gcm_contour_difference(t, x_full, x_det_param, x_stoch_param, time_end, time_start=0,
                                dpi=150, cmap='viridis', output_folder="", config=None):
    """
    Plots contour differences for GCM simulations with deterministic and 
    stochastic parameterizations.

    This function generates three contour plots:
    - The first plot shows the full Lorenz '96 model (x_full).
    - The second plot shows the difference between the full L96 model and the GCM 
      with deterministic parameterization.
    - The third plot shows the difference between the full L96 model and the GCM 
      with stochastic parameterization.

    Args:
        t (numpy.ndarray): Array of time points with shape (time,).
        x_full (numpy.ndarray): Array of variables from the full Lorenz '96 model 
            with shape (time, k).
        x_det_param (numpy.ndarray): Array of variables from GCM with deterministic 
            parameterization with shape (time, k).
        x_stoch_param (numpy.ndarray): Array of variables from GCM with stochastic 
            parameterization with shape (time, k).
        time_end (int): Index of the last time point to include in the plot.
        time_start (int, optional): Index of the first time point to include in the plot. 
            Default is 0.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        cmap (str, optional): Colormap to use for the contour plots. Default is 'viridis'.
        output_folder (str, optional): Path to the folder where the plot will be saved as a PDF.
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

    k = x_full.shape[-1]
    range_k = np.arange(k)

    sns.set_theme(style='ticks')

    fig = plt.figure(dpi=dpi, figsize=(17, 8))

    # Define contour plot limits and levels
    limits = {"vmin": -12, "vmax": 12,
              "levels": np.linspace(-12, 12, 12), "extend": "both"}

    # Plot the full Lorenz '96 model
    plt.subplot(131)
    plt.contourf(range_k, t[time_start:time_end],
                 x_full[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("Full L96")

    # Plot the difference between the full model and GCM with deterministic parameterization
    plt.subplot(132)
    plt.contourf(range_k, t[time_start:time_end], x_full[time_start:time_end,
                 :] - x_det_param[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("L96 - Deterministic parameterization")
    plt.colorbar()

    # Plot the difference between the full model and GCM with stochastic parameterization
    plt.subplot(133)
    plt.contourf(range_k, t[time_start:time_end], x_full[time_start:time_end,
                 :] - x_stoch_param[time_start:time_end, :], cmap=cmap, **limits)
    plt.xlabel("$k$")
    plt.ylabel("$t$")
    plt.title("L96 - Stochastic parameterization")
    plt.colorbar()

    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"gcm_2d_difference_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}_rs{config['seed']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "GCM contour plot difference saved successfully to %s", save_path)

    return fig
