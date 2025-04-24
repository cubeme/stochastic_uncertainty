"""
This module provides a polynomial parameterization with an autoregressive AR(1) noise process.

Classes:
- PolynomialAR1Parameterization: Combines a polynomial function and AR(1) noise to model stochastic processes.
"""

from typing import Tuple
import numpy as np
from numba import njit


class PolynomialAR1Parameterization:
    """
    Polynomial parameterization with an autoregressive AR(1) noise model.

    The parameterization is defined as:
        U_p = P(X) + e(t),
    where P(X) is a polynomial function of the state variables X, and e(t) is an AR(1) noise process:
        e(t) = phi * e(t-1) + sigma_e * sqrt(1 - phi^2) * z(t),
    where z(t) is Gaussian white noise.
    """

    def __init__(self, coefs: np.ndarray, phi: np.ndarray, sigma_e: np.ndarray, seed: int = 17):
        """
        Initialize the parameterization.

        Args:
            coefs (np.ndarray): Polynomial coefficients.
            phi (np.ndarray): Autoregressive coefficients for the AR(1) noise process. Shape: (K,).
            sigma_e (np.ndarray): Standard deviations of the noise term in the AR(1) process. Shape: (K,).
            seed (int, optional): Random seed for reproducibility. Default is 17.
        """
        self.coefs = coefs
        self.phi = np.asarray(phi)
        self.sigma_e = np.asarray(sigma_e)

        # Validate phi values
        if np.any((self.phi < -1) | (self.phi > 1)):
            raise ValueError("phi must satisfy -1 <= phi <= 1.")

        # Initialize the previous noise value to zero
        self.noise_t_minus_1 = np.zeros_like(self.phi)

        self.rg = np.random.default_rng(seed=seed)

    @staticmethod
    @njit
    def eval_poly(x: np.ndarray, coefs: np.ndarray) -> np.ndarray:
        """Evaluate the polynomial for the given state variables."""
        return np.polynomial.polynomial.polyval(x, coefs)

    @staticmethod
    @njit
    def ar1_noise(phi: np.ndarray, sigma_e: np.ndarray, noise_t_minus_1: np.ndarray, rg: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate AR(1) noise for the current time step.

        Args: 
            phi (np.ndarray): Autoregressive coefficients for the AR(1) noise process. Shape: (K,).
            sigma_e (np.ndarray): Standard deviations of the noise term in the AR(1) process. Shape: (K,).
            noise_t_minus_1 (np.ndarray): Noise values for the previous time step. 
                                          Shape: (K,), where K is the number of X variables.
            rg (np.random.default_rng): Random generator instance. 
        Returns:
            np.ndarray: Noise values for the current time step.
                        Shape: (K,), where K is the number of X variables.
        """
        noise_t = phi * noise_t_minus_1 + sigma_e * \
            np.sqrt(1 - phi**2) * rg.normal(0, 1, size=phi.shape)

        # Update the previous noise value
        noise_t_minus_1 = noise_t

        return noise_t, noise_t_minus_1

    def __call__(self, x):
        """
        Compute the parameterization value for the given state variables.

        Args:
            x (numpy.ndarray): State variables for which to compute the parameterization.
                               Shape: (K,), where K is the number of variables.

        Returns:
            numpy.ndarray: Parameterization values for the given state variables.
                           Shape: (K,).
        """
        ar1_noise, self.noise_t_minus_1 = self.ar1_noise(
            self.phi, self.sigma_e, self.noise_t_minus_1, self.rg)
        # Compute the deterministic polynomial parameterization and add AR(1) noise
        return self.eval_poly(x, self.coefs) + ar1_noise
