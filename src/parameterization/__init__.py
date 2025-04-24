"""
The `parameterization` module provides tools for implementing and managing parameterizations in simulations.

Submodules:
- stochastic_poly_parameterization: Implements polynomial parameterizations with stochastic AR(1) noise.
- ar1_noise: Functions for fitting and generating AR(1) noise.

Exports:
- PolynomialAR1Parameterization: Combines a polynomial function and AR(1) noise to model stochastic processes.
- fit_ar1_noise_parameters: Fit AR(1) noise parameters from residuals.
- compute_ar1_noise: Generate AR(1) noise for a specified number of time steps.
"""

# Import key classes and functions from submodules for easier access
from .stochastic_poly_parameterization import PolynomialAR1Parameterization

from .ar1_noise import (
    fit_ar1_noise_parameters,
    compute_ar1_noise
)

# Define the public API of the module
__all__ = [
    "PolynomialAR1Parameterization",
    "fit_ar1_noise_parameters",
    "compute_ar1_noise"
]
