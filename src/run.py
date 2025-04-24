from typing import Dict, Text, Union

import json
import os

from absl import app
from absl import flags
from absl import logging

from datetime import datetime

import jax.numpy as jnp
import numpy as np

from tqdm import tqdm

# some more imports

# If you are using the plotting_file_skeleton.py here
# https://gist.github.com/nikikilbertus/c593cab6e15cbe4096b5b2951b522b1a
# then also import that module, e.g.,
# import plotting

Results = Dict[Text, Union[float, np.ndarray]]


# Don't be afraid of *many* command line arguments.
# Everything should be a flag!
# ------------------------------- DUMMY PARAMS --------------------------------
flags.DEFINE_enum("enum", "choice1",
                  ("choice1", "choice2", "choice3"),
                  "An enum parameter.")
flags.DEFINE_string("string", "test", "String parameter.")
flags.DEFINE_bool("bool", False, "Boolean parameter.")
flags.DEFINE_float("float", 0.0, "Float parameter.")
flags.DEFINE_integer("integer", 0, "Integer parameter.")
# ---------------------------- INPUT/OUTPUT -----------------------------------
# I always keep those the same!
flags.DEFINE_string("data_dir", "../data/",
                    "Directory of the input data.")
flags.DEFINE_string("output_dir", "../results/",
                    "Path to the output directory (for results).")
flags.DEFINE_string("output_name", "",
                    "Name for result folder. Use timestamp if empty.")
flags.DEFINE_bool("plot", True,
                  "Whether to store plots while running.")
# ------------------------------ MISC -----------------------------------------
# I always keep this one!
flags.DEFINE_integer("seed", 0, "The random seed.")
FLAGS = flags.FLAGS

# local functions

# =============================================================================
# MAIN
# =============================================================================


def main(_):
    # ---------------------------------------------------------------------------
    # Directory setup, save flags, set random seed
    # ---------------------------------------------------------------------------
    FLAGS. alsologtostderr = True

    if FLAGS.output_name == "":
        dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    else:
        dir_name = FLAGS.output_name
    out_dir = os.path.join(os.path.abspath(FLAGS.output_dir), dir_name)
    logging.info(f"Save all output to {out_dir}...")
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    FLAGS.log_dir = out_dir
    logging.get_absl_handler().use_absl_log_file(program_name="run")

    logging.info("Save FLAGS (arguments)...")
    with open(os.path.join(out_dir, 'flags.json'), 'w') as fp:
        json.dump(FLAGS.flag_values_dict(), fp, sort_keys=True, indent=2)

    logging.info(f"Set random seed {FLAGS.seed}...")
    # Depending on whether we are using JAX, just numpy, or something else
    # key = random.PRNGKey(FLAGS.seed)
    # np.random.seed(FLAGS.seed)

    # ---------------------------------------------------------------------------
    # Load data
    # ---------------------------------------------------------------------------

    # FILL THIS IN

    # ---------------------------------------------------------------------------
    # Allocate results dictionary
    # ---------------------------------------------------------------------------
    # I always try to keep my results in the format
    # Results = Dict[Text, Union[float, np.ndarray]
    # This makes it super easy to store it to an npz file (see below).
    results = {}

    # ---------------------------------------------------------------------------
    # Do work and fill in results dictionary
    # ---------------------------------------------------------------------------

    # FILL THIS IN

    # ---------------------------------------------------------------------------
    # Store aggregate results
    # ---------------------------------------------------------------------------
    logging.info(f"Store results...")
    result_path = os.path.join(out_dir, "results.npz")
    np.savez(result_path, **results)

    # ---------------------------------------------------------------------------
    # Plot aggregate results
    # ---------------------------------------------------------------------------
    if FLAGS.plot:
        logging.info(f"Plot final aggregate results...")
        # If you're using the plotting template (see beginning) then run
        plotting.plot_all(results, os.path.join(out_dir, 'figures'))

    logging.info(f"DONE")


if __name__ == "__main__":
    app.run(main)
