
import logging

import numpy as np

from ensemble.gcm_polynomial_ar1 import (
    run_stochastic_ensemble_parallel_multiprocessing,
    run_stochastic_member_manual, run_stochastic_member_solver)
from models.execute import initialize_stoch_gcm, run_gcm
from parameterization.helpers import (load_ar1_parameters,
                                      load_poly_coefficients)
from utils.saving import save_output_ensemble, save_output_gcm

logger = logging.getLogger(__name__)


def run_stoch_gcm(poly_order, init_conditions, config, output_folder):

    coefs = load_poly_coefficients(poly_order, output_folder, config)
    phi, sigma_e = load_ar1_parameters(output_folder, config)

    gcm = initialize_stoch_gcm(config, coefs, phi, sigma_e)

    x, t = run_gcm(gcm, init_conditions, config)

    save_output_gcm(output_folder, config, "stoch_param", x, t)
    logging.info("Successfully saved stochastic GCM output to %s",
                 output_folder)


def run_stoch_gcm_ensemble(init_states, poly_order, perturb, config, output_folder):

    coefs = load_poly_coefficients(poly_order, output_folder, config)
    phi, sigma_e = load_ar1_parameters(output_folder, config)

    seeds = np.arange(config['seed'], config['seed'] +
                      config['init_states'] + 1, dtype=int)

    logging.info("Running %s stochastic GCM ensemble with %i initial states "
                 "and %i members for %f time steps", config['solver'],
                 config['n_init_states'], config['n_ens'], config['t_total'])

    if config['solver'] == 'manual':
        x, t = run_stochastic_ensemble_parallel_multiprocessing(init_states, config,
                                                                run_stochastic_member_manual,
                                                                coefs, phi, sigma_e, seeds,
                                                                perturb, config['dt'])
    else:
        x, t = run_stochastic_ensemble_parallel_multiprocessing(init_states, config,
                                                                run_stochastic_member_solver,
                                                                coefs, phi, sigma_e, seeds, perturb,
                                                                config['solver'],
                                                                config['solver_method'])

    save_output_ensemble(output_folder, config, "stoch_param", x, t, seeds)
    logging.info(
        "Successfully saved stochastic GCM ensemble output to %s",  output_folder)
