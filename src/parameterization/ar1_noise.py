"""
Provides functions to fit and generate first-order autoregressive (AR(1)) noise.

Functions:
- fit_ar1_noise_parameters: Fits AR(1) parameters (phi and sigma_e) from residuals.
- compute_ar1_noise: Generates AR(1) noise for a specified number of time steps.
"""

from typing import Tuple
import numpy as np
from scipy.optimize import curve_fit


def fit_ar1_noise_parameters(residuals: np.ndarray, seed: int = 17) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Fit AR(1) noise parameters (phi and sigma_e) from the residuals.

    Args:
        residuals (numpy.ndarray): Residuals array of shape (time_steps, k), where `k` is the number of X variables.
        seed (int, optional): Random seed for reproducibility. Default is 17.

    Returns:
        tuple: A tuple containing:
            - phi (numpy.ndarray): AR(1) coefficient for each X variable (shape: (k,)).
            - sigma_e (numpy.ndarray): Noise standard deviation for each X variable (shape: (k,)).
            - phi_cov (numpy.ndarray): Covariance of phi for each X variable (shape: (k,)).
            - sigma_e_cov (numpy.ndarray): Covariance of sigma_e for each X variable (shape: (k,)).
    """

    k = residuals.shape[1]  # Number of X variables (k)

    rg = np.random.default_rng(seed=seed)
    # Time indices for residuals
    time_idx = np.arange(0, residuals.shape[0], dtype=int)

    # Backward-shifted time indices (cyclic shift)
    bwd_shifted_time_idx = np.roll(time_idx, 1)

    def ar_1_process_per_k(shifted_time, phi, sigma_e):
        """AR(1) process model for a single X variable."""

        # Ensure time indices are integers
        shifted_time = shifted_time.astype(int)

        # Draw random noise from normal Gaussian
        z_t = rg.normal(0, 1)

        return phi * residuals[shifted_time, k_opt] + sigma_e * np.sqrt(1-phi**2) * z_t

    # Optimized parameters (phi, sigma_e) for each X variable
    p_opt = np.zeros((2, k))
    # Covariance matrices for each X variable
    p_cov = np.zeros((2, 2, k))

    # Fit AR(1) parameters for each X variable
    for k_opt in range(k):
        p_opt_k, p_cov_k = curve_fit(
            ar_1_process_per_k, bwd_shifted_time_idx, residuals[:, k_opt],
            bounds=([-1, -np.inf], [1, np.inf]))

        p_opt[:, k_opt] = p_opt_k
        p_cov[:, :, k_opt] = p_cov_k

    # Extract phi and sigma_e along with their covariances
    phi = p_opt[0]
    sigma_e = p_opt[1]
    phi_cov = p_cov[0]
    sigma_e_cov = p_cov[1]

    return phi, sigma_e, phi_cov, sigma_e_cov


def compute_ar1_noise(phi: np.ndarray, sigma_e: np.ndarray, steps: int, seed: int = 17) -> np.ndarray:
    """
    Generate AR(1) noise specified number of time steps.

    Args:
        phi (numpy.ndarray): AR(1) coefficients for each X variable (shape: (k,)).
        sigma_e (numpy.ndarray): Noise standard deviations for each X variable (shape: (k,)).
        steps (int): Number of time steps to generate noise for.
        seed (int, optional): Random seed for reproducibility. Default is 17.

    Returns:
        numpy.ndarray: Generated AR(1) noise of shape (steps, k), where `k` is the number of X variables.
    """
    k = phi.shape[0]  # Number of X variables (k)

    rg = np.random.default_rng(seed=seed)
    ar1_noise = np.zeros((steps, k))

    for i in range(1, steps):
        ar1_noise[i, :] = phi * ar1_noise[i - 1] + \
            sigma_e * np.sqrt(1-phi**2) * rg.normal(0, 1, size=k)

    return ar1_noise
