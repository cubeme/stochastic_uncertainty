import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_stacked_rank_histogram(filled_bins, stack_steps, title="", dpi=150, output_folder="", config=None):
    logger = logging.getLogger(__name__)

    sns.set_theme()

    fig = plt.figure(figsize=(4, 4), dpi=dpi)

    plt.bar(np.arange(
        1, filled_bins.shape[0]+1), filled_bins, width=1.0, color='w', edgecolor='k')
    plt.title(title)

    plt.xticks([])
    plt.yticks([])

    plt.tight_layout()

    if output_folder != "":
        f_name = f"rank_histogram_ens_is{config['n_init_states']}_mem{config['n_ens']}_" + \
            f"stack{stack_steps}_c{config['c']}_dt{config['dt']}_si{config['si']}_" + \
            f"time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "Ensemble rank histogram successfully saved to %s", save_path)
    return fig


def compute_stacked_rank_histogram(x_ens, x_true, stack_steps):
    filled_bins = np.zeros((x_ens.shape[0],), dtype=int)

    for i in range(0, x_ens.shape[1], stack_steps):

        data_i_sorted = np.sort(x_ens[:, i])

        bin_edges_i = np.zeros((data_i_sorted.shape[0]+1,))
        bin_edges_i[:-1] = data_i_sorted
        # Add additional bin edge to cover all values
        bin_edges_i[-1] = np.inf

        if len(x_true.shape) > 1:
            hist_i, _ = np.histogram(x_true[:, i], bin_edges_i)
        else:
            hist_i, _ = np.histogram(x_true[i], bin_edges_i)

        filled_bins += hist_i

    return filled_bins
