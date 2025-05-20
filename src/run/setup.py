from models.execute import initialize_l96, run_l96
from parameterization.helpers import (compute_poly_ar1_parameters,
                                      fit_deterministic_poly_parameterization,
                                      save_ar1_parameters,
                                      save_poly_coefficients)
from utils.loading import load_output_l96
from utils.saving import save_output_l96


def run_lorenz96(config, output_folder):
    m = initialize_l96(config)

    x, y, t, u = run_l96(m, config)

    save_output_l96(output_folder, config, x, y, t, u)


def fit_poly_coefficients(poly_order, config, output_folder):
    x_true, y_true, t_true, u_true = load_output_l96(output_folder, config)

    coefs = fit_deterministic_poly_parameterization(x_true, u_true, poly_order)

    save_poly_coefficients(output_folder, config, coefs)


def fit_ar1_parameters(coefs, config, output_folder):
    x_true, y_true, t_true, u_true = load_output_l96(output_folder, config)
    phi, sigma_e = compute_poly_ar1_parameters(x_true, u_true, coefs, config)

    save_ar1_parameters(output_folder, config, phi, sigma_e)
