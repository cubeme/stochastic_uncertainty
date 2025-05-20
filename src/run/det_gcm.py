import logging

from models.execute import initialize_det_gcm, run_gcm
from parameterization.helpers import load_poly_coefficients
from utils.saving import save_output_gcm

logger = logging.getLogger(__name__)


def run_det_gcm(poly_order, init_conditions, config, output_folder):

    coefs = load_poly_coefficients(poly_order, output_folder, config)

    gcm = initialize_det_gcm(config, coefs)

    x, t = run_gcm(gcm, init_conditions, config)

    save_output_gcm(output_folder, config, "det_param", x, t)
    logging.info("Successfully saved stochastic GCM output to %s",
                 output_folder)
