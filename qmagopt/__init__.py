"""Q-MagOpt: constraint-aware quantum optimization for magnetic navigation.

A compact, self-contained research demonstrator connecting magnetic-anomaly navigation
with constrained QAOA.  It is a benchmark and an architecture study, not a claim of
quantum advantage: every instance here is solved exactly, and far faster, by the
classical dynamic program in :mod:`qmagopt.classical`.
"""

from __future__ import annotations

from .classical import (
    solve_brute_force,
    solve_dynamic_programming,
    solve_greedy,
    solve_simulated_annealing,
)
from .maps import (
    MagneticMap,
    SensorModel,
    d_optimality,
    e_optimality,
    make_synthetic_magnetic_map,
    position_crlb_m,
    route_fisher,
)
from .metrics import information_summary, route_metrics, surrogate_fidelity
from .problem import NavigationProblem, NavigationWeights, build_demo_problem, with_information_model
from .qaoa import QaoaConfig, qaoa_depth_scan, solve_penalty_qaoa, solve_xy_qaoa
from .results import SolverResult

__version__ = "0.2.0"

__all__ = [
    "MagneticMap",
    "NavigationProblem",
    "NavigationWeights",
    "QaoaConfig",
    "SensorModel",
    "SolverResult",
    "build_demo_problem",
    "d_optimality",
    "e_optimality",
    "information_summary",
    "make_synthetic_magnetic_map",
    "position_crlb_m",
    "qaoa_depth_scan",
    "route_fisher",
    "route_metrics",
    "solve_brute_force",
    "solve_dynamic_programming",
    "solve_greedy",
    "solve_penalty_qaoa",
    "solve_simulated_annealing",
    "solve_xy_qaoa",
    "surrogate_fidelity",
    "with_information_model",
]
