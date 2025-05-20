import logging

import numpy.polynomial.polynomial as poly

from models.gcm import GCM, GCMManual
from models.lorenz96 import L96
from parameterization.helpers import (compute_poly_ar1_parameters,
                                      fit_deterministic_poly_parameterization)
from parameterization.polynomial_ar1_param import PolynomialAR1Parameterization

logger = logging.getLogger(__name__)


def initialize_det_gcm(config, coefs):
    def det_param(x):
        return poly.polyval(x, coefs)

    if config['solver'] == 'manual':
        return GCMManual(config['F'], det_param)

    return GCM(config['F'], det_param)


def initialize_stoch_gcm(config, coefs, phi, sigma_e):
    stoch_param = PolynomialAR1Parameterization(coefs, phi, sigma_e, config['seed'])

    if config['solver'] == 'manual':
        return GCMManual(config['F'], stoch_param)

    return GCM(config['F'], stoch_param)


def run_gcm(gcm, init_conditions, config):
    if config['solver'] == 'manual':
        logging.info("Running manual GCM for %f time steps",
                     config['t_total'])
        return gcm(init_conditions, dt=config['dt'],
                   si=config['si'], t_total=config['t_total'])

    logging.info("Running solver GCM for %f time steps", config['t_total'])
    return gcm(init_conditions, si=config['si'],
               t_total=config['t_total'], solver=config['solver'],
               solver_method=config['solver_method'])


def initialize_l96(config):
    m = L96(config['K'], config['J'], config['F'], config['h'],
            config['b'], config['c'], seed=config['seed'])

    if config['t_spin_up'] > 0:
        # Spin up model and save final state as new initial state
        _ = m.run(si=config['si'], t_total=config['t_spin_up'], store=True,
                  solver=config['solver'], solver_method=config['solver_method'])

    return m


def run_l96(m, config, store=True, return_coupling=True):
    logging.info("Running L96 for %f time steps", config['t_total'])

    x, y, t, u = m.run(config['si'], config['t_total'], dt=config['dt'],
                       solver=config['solver'], solver_method=config['solver_method'],
                       store=store, return_coupling=return_coupling)
    return x, y, t, u
