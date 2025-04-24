"""
The `models` module provides implementations of various models used in simulations.

Submodules:
- lorenz96_model: Implements the Lorenz '96 model.
- gcm_models: Implements General Circulation Models (GCMs) based on the Lorenz '96 model.

Exports:
- L96: The Lorenz '96 model class.
- L96_2t_x_dot_y_dot_solver: Time stepping for the two-time-scale Lorenz '96 model using numerical solvers.
- L96_2t_x_dot_y_dot_manual: Manual time stepping for the two-time-scale Lorenz '96 model.
- GCM: General Circulation Model using numerical solvers.
- GCMManual: General Circulation Model using manual time-stepping methods.
"""

# Import key classes and functions from submodules for easier access
from .lorenz96_model import (
    L96,
    L96_2t_x_dot_y_dot_solver,
    L96_2t_x_dot_y_dot_manual
)

from .gcm_models import (
    GCM,
    GCMManual
)

# Define the public API of the module
__all__ = [
    "L96",
    "L96_2t_x_dot_y_dot_solver",
    "L96_2t_x_dot_y_dot_manual",
    "GCM",
    "GCMManual"
]
