
import logging
import math
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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
        y_index = f"Y_{{{random_y_index, i}}}"

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

