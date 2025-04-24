"""
The `ensemble` module provides tools for running ensemble simulations.

Submodules:
- stochastic_poly_ensemble: Functions for running stochastic polynomial ensemble simulations.
- l96_parallel_ensemble: Functions for parallelizing Lorenz '96 simulations.

Exports:
- run_stochastic_ensemble_parallel_multiprocessing: Run stochastic ensemble simulations in parallel using multiprocessing.
- run_stochastic_member: Run a single stochastic ensemble member.
- run_stochastic_member_manual: Run a single stochastic ensemble member with manual time-stepping.
- run_l96_parallel: Run Lorenz '96 ensemble simulations in parallel.
- run_single_l96_simulation: Run a single Lorenz '96 simulation. 
"""

# Import key functions and classes from submodules for easier access
from .stochastic_poly_ensemble import (
    run_stochastic_ensemble_parallel_multiprocessing,
    run_stochastic_member,
    run_stochastic_member_manual
)

from .l96_parallel_ensemble import (
    run_l96_parallel, 
    run_single_l96_simulation
)

# Define the public API of the module
__all__ = [
    "run_stochastic_ensemble_parallel_multiprocessing",
    "run_stochastic_member",
    "run_stochastic_member_manual",
    "run_l96_parallel", 
    "run_single_l96_simulation"
]