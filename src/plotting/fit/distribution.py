
import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import numpy.polynomial.polynomial as poly
import seaborn as sns

from parameterization.ar1_process import compute_ar1_noise


def plot_x_u_distribution_with_polynomial_fit(x, u, coefs, dpi=150, poly_label="$poly(X_k)$",
                                              output_folder="", config=None):
    """
    Plot the joint distribution of X and U with a polynomial fit.

    This function generates a 2D histogram of the slow variables (X) and the coupling term (U),
    and overlays a polynomial fit on the histogram.

    Args:
        x (numpy.ndarray): Array of slow variables (X) with shape (n_samples, k).
        u (numpy.ndarray): Array of coupling terms (U) with shape (n_samples, k).
        coefs (numpy.ndarray): Coefficients of the polynomial fit.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        poly_label (str, optional): Label for the polynomial fit in the legend. 
            Default is "$poly(X_k)$".
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

    sns_style = 'rocket'
    sns.set_theme(style='ticks', palette=sns_style, color_codes=True)

    fig = plt.figure(dpi=dpi)

    # 2D histogram of X vs U
    plt.hist2d(x.flatten(), u.flatten(), bins=40, density=True, cmap=sns_style)
    plt.xlabel("$X_k$")
    plt.ylabel(r"$U_k = \frac{hc}{b}\sum_j Y_{j,k}$")
    plt.colorbar(label="PDF")

    # Fits from polynomials
    plot_x = np.linspace(x.flatten().min(), x.flatten().max(), 100)
    plt.plot(plot_x, poly.polyval(plot_x, coefs),
             color='xkcd:sky blue', label=poly_label)

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"det_param_fit_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "X/U density with polynomial fit plot saved successfully to %s", save_path)

    return fig


def plot_x_u_distribution_with_stochastic_polynomial_fit(x, u, coefs, phi, sigma_e, dpi=150,
                                                         poly_label="$poly(X_k) + AR(1)$",
                                                         output_folder="", config=None):
    """
    Plot the joint distribution of X and U with a stochastic polynomial fit.

    This function generates a 2D histogram of the slow variables (X) and the coupling term (U),
    and overlays a stochastic polynomial fit on the histogram. The stochastic fit includes
    a polynomial component and an AR(1) noise component.

    Args:
        x (numpy.ndarray): Array of slow variables (X) with shape (n_samples, k).
        u (numpy.ndarray): Array of coupling terms (U) with shape (n_samples, k).
        coefs (numpy.ndarray): Coefficients of the polynomial fit.
        phi (float): AR(1) coefficient for the noise model.
        sigma_e (float): Standard deviation of the noise in the AR(1) model.
        dpi (int, optional): Resolution of the plot in dots per inch. Default is 150.
        poly_label (str, optional): Label for the polynomial fit in the legend. 
            Default is "$poly(X_k) + AR(1)$".
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

    sns_style = 'rocket'
    sns.set_theme(style='ticks', palette=sns_style, color_codes=True)

    fig = plt.figure(dpi=dpi)

    # 2D histogram of X vs U
    plt.hist2d(x.flatten(), u.flatten(), bins=40, density=True, cmap=sns_style)
    plt.xlabel("$X_k$")
    plt.ylabel(r"$U_k = \frac{hc}{b}\sum_j Y_{j,k}$")
    plt.colorbar(label="PDF")

    # Compute AR1 noise
    ar1_noise = compute_ar1_noise(
        phi, sigma_e, poly.polyval(x, coefs).shape[0], seed=config['seed'])

    # Plot fits from polynomials + AR1 noise
    plot_x = np.linspace(x.flatten().min(), x.flatten().max(), 100)
    plt.plot(plot_x, poly.polyval(plot_x, coefs) + ar1_noise.flatten()
            [:poly.polyval(plot_x, coefs).shape[0]], color='xkcd:sky blue', label=poly_label)

    plt.legend()
    plt.tight_layout()

    # Save the plot if an output folder is specified
    if output_folder != "":
        f_name = f"stoch_param_fit_c{config['c']}_dt{config['dt']}" + \
            f"_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "X/U density with stochastic polynomial fit plot saved successfully to %s", save_path)

    return fig
