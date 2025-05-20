import os
from pathlib import Path

import numpy as np
import numpy.polynomial.polynomial as poly

from parameterization.ar1_process import fit_ar1_noise_parameters
from parameterization.polynomial_ar1_param import PolynomialAR1Parameterization
from utils.identifiers import get_ident_str


def fit_deterministic_poly_parameterization(x, u, poly_order):
    return poly.polyfit(x.flatten(), u.flatten(), poly_order)


def compute_poly_ar1_parameters(x, u, coefs, config):
    residuals = u - poly.polyval(x, coefs)
    phi, sigma_e, phi_cov, sigma_e_cov = fit_ar1_noise_parameters(
        residuals, seed=config['seed'])

    return phi, sigma_e


def save_poly_coefficients(output_folder, config, coefs, extra_ident=""):
    poly_order = len(coefs) - 1

    f_ident = get_ident_str('coefs', config, poly_order=poly_order)

    output_path = Path(output_folder, 'coefs', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "coefs.npy"), coefs)


def load_poly_coefficients(poly_order, output_folder, config, extra_ident=""):

    f_ident = get_ident_str('coefs', config, poly_order=poly_order)

    save_folder = Path(output_folder, 'coefs', extra_ident, f_ident)

    return np.load(os.path.join(save_folder, "coefs.npy"))


def save_ar1_parameters(output_folder, config, phi, sigma_e, extra_ident=""):
    f_ident = get_ident_str('ar1', config)

    output_path = Path(output_folder, 'ar1', extra_ident, f_ident)
    output_path.mkdir(parents=True, exist_ok=True)

    np.save(os.path.join(output_path, "phi.npy"), phi)
    np.save(os.path.join(output_path, "sigma_e.npy"), sigma_e)


def load_ar1_parameters(output_folder, config, extra_ident=""):
    f_ident = get_ident_str('ar1', config)

    save_folder = Path(output_folder, 'ar1', extra_ident, f_ident)

    phi = np.load(os.path.join(save_folder, "phi.npy"))
    sigma_e = np.load(os.path.join(save_folder, "sigma_e.npy"))

    return phi, sigma_e
