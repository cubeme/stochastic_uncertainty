"""
The `utils` module provides utility functions for saving, loading, and managing simulation data,
as well as time-stepping methods for numerical integration.

Submodules:
- saving_and_loading: Functions for saving and loading simulation results.
- time_stepping: Functions for numerical integration using various time-stepping methods.

Exports:
- save_output_l96: Save Lorenz '96 simulation results.
- save_output_gcm: Save General Circulation Model (GCM) simulation results.
- save_output_ensemble: Save ensemble simulation results.
- load_output_l96: Load Lorenz '96 simulation results.
- load_output_gcm: Load General Circulation Model (GCM) simulation results.
- load_output_ensemble: Load ensemble simulation results.
- euler_forward: Perform a single Euler forward time-stepping step.
- RK2: Perform a single second-order Runge-Kutta (RK2) time-stepping step.
- RK4: Perform a single fourth-order Runge-Kutta (RK4) time-stepping step.
- RK4_two_variables: Perform a single RK4 time-stepping step for two coupled variables.
"""

# Import key functions from submodules for easier access
from .saving_and_loading import (
    save_output_l96,
    save_output_gcm,
    save_output_ensemble,
    load_output_l96,
    load_output_gcm,
    load_output_ensemble
)

from .time_stepping import (
    RK2,
    RK4,
    RK4_two_variables,
    euler_forward
)

# Define the public API of the module
__all__ = [
    "save_output_l96",
    "save_output_gcm",
    "save_output_ensemble",
    "load_output_l96",
    "load_output_gcm",
    "load_output_ensemble",
    "RK2",
    "RK4",
    "RK4_two_variables",
    "euler_forward"
]
