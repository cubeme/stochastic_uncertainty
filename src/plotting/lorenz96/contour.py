import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

