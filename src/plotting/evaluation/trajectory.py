import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_trajectory(x_ens, x_true, t, plot_step=1, dpi=150, output_folder="", config=None):
    logger = logging.getLogger(__name__)

    sns.set_theme()

    fig = plt.figure(figsize=(30, 6), dpi=dpi)

    time = t[0]

    mean_ens = np.mean(x_ens, axis=0)
    std_ens = np.std(x_ens, axis=0)

    if len(x_true.shape) > 1:
        # Ensemble with multiple initial states
        mean_truth = np.mean(x_true, axis=0)
        std_truth = np.std(x_true, axis=0)
    else:
        # Ensemble with single initial state
        mean_truth = x_true
        std_truth = np.zeros_like(x_true)

    line_ens, = plt.plot(time[::plot_step], mean_ens[::plot_step], 'b-')
    fill_ens = plt.fill_between(time[::plot_step], (mean_ens - std_ens)[::plot_step],
                                (mean_ens + std_ens)[::plot_step], color='b', alpha=0.3)

    line_truth, = plt.plot(time[::plot_step], mean_truth[::plot_step], 'r--')
    fill_truth = plt.fill_between(time[::plot_step], (mean_truth - std_truth)[::plot_step],
                                  (mean_truth + std_truth)[::plot_step], color='r', alpha=0.3)

    plt.margins(x=0.01)
    plt.xlabel('Simulation Time')
    plt.ylabel('Mean $X_k$')

    plt.legend([(line_ens, fill_ens), (line_truth, fill_truth)],
               ['Stochastic ensemble', 'Truth'])

    if output_folder != "":
        f_name = f"trajectory_ens_is{config['n_init_states']}_mem{config['n_ens']}_" + \
            f"c{config['c']}_dt{config['dt']}_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "Ensemble trajectory plot successfully saved to %s", save_path)
    return fig
