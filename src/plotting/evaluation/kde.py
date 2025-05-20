import logging
import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def plot_ensemble_kde(x_ens, x_true, dpi=150, output_folder="", config=None):
    logger = logging.getLogger(__name__)

    sns.set_theme()

    fig = plt.figure(figsize=(5, 4), dpi=dpi)

    mean_ens = np.mean(x_ens, axis=0)

    if len(x_true.shape) > 1:
        # Ensemble with multiple initial states
        mean_truth = np.mean(x_true, axis=0)
    else:
        # Ensemble with single initial state
        mean_truth = x_true

    ax = sns.kdeplot(mean_ens, fill=True, color='b', label='Ensemble')
    ax = sns.kdeplot(mean_truth, fill=True, color='r', label='Truth')

    plt.legend()
    sns.move_legend(ax, "upper right")

    plt.xlabel('$X$')
    plt.ylabel('Density $P(X)$')

    plt.tight_layout()

    if output_folder != "":
        f_name = f"trajectory_ens_is{config['n_init_states']}_mem{config['n_ens']}_" + \
            f"c{config['c']}_dt{config['dt']}_si{config['si']}_time{config['t_total']}.pdf"
        save_path = os.path.join(output_folder, f_name)

        plt.savefig(save_path, format="pdf", bbox_inches="tight")
        logger.info(
            "Ensemble KDE plot successfully saved to %s", save_path)
    return fig
