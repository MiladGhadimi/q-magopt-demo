"""Per-route metrics, including the estimation-theoretic ones the objective proxies."""

from __future__ import annotations

import math

import numpy as np

from .maps import d_optimality, e_optimality, position_crlb_m
from .problem import NavigationProblem
from .results import SolverResult


def route_metrics(
    problem: NavigationProblem,
    result: SolverResult,
    *,
    optimum: float | None = None,
) -> dict:
    """Summarize one solver result.

    ``approximation_ratio`` compares against the exact optimum when one is supplied; see
    :func:`approximation_ratio` for its definition.
    """
    route = result.route
    distance = 0.0
    motion_energy = 0.0
    information = 0.0
    risk = 0.0

    for t, r in enumerate(route):
        information += float(problem.magnetic_map.information[r, t])
        risk += float(problem.magnetic_map.risk[r, t])
        if t:
            dy = abs(route[t] - route[t - 1])
            distance += math.sqrt(1.0 + dy * dy)
            motion_energy += 1.0 + 0.9 * dy

    fisher = problem.route_fisher(route)

    row = {
        "method": result.method,
        "route": "-".join(map(str, route)),
        "objective": result.objective,
        "distance": distance,
        "motion_energy": motion_energy,
        "magnetic_information": information,
        "risk": risk,
        "position_crlb_m": position_crlb_m(fisher),
        "log_det_fisher": d_optimality(fisher),
        "min_eigen_fisher": e_optimality(fisher),
        "hard_feasible": result.hard_feasible,
        "success_probability": result.success_probability,
        "one_hot_probability": result.one_hot_probability,
        "feasible_probability": result.feasible_probability,
        "conditional_feasible_probability": result.conditional_feasible_probability,
        "seconds": result.seconds,
    }

    if optimum is not None:
        row["approximation_ratio"] = approximation_ratio(problem, result.objective, optimum)
    return row


def approximation_ratio(problem: NavigationProblem, objective: float, optimum: float) -> float:
    """Normalized score against the exact optimum, scaled by the feasible cost range.

    Defined as ``(C_worst - C) / (C_worst - C_opt)`` over the *hard-feasible* routes, so
    1.0 is optimal and 0.0 is the worst route a legal plan could take.  The usual
    ``C_opt / C`` ratio is unusable here because the objective mixes rewards and costs
    and can cross zero; normalizing by the worst *feasible* route rather than the worst
    bitstring also keeps the scale informative instead of squashing every method against
    the enormous penalty-laden states.  A value below 0 means the returned route is
    worse than any feasible one, which happens only when a solver returns an infeasible
    route -- worth seeing rather than hiding.
    """
    feasible = problem.feasible_routes()
    if not feasible:
        return float("nan")
    worst = max(problem.route_cost(r) for r in feasible)
    denominator = worst - optimum
    if denominator <= 1e-12:
        return 1.0 if abs(objective - optimum) < 1e-9 else float("nan")
    return float((worst - objective) / denominator)


def information_summary(problem: NavigationProblem) -> dict:
    """Instance-level context: how much position information the corridor can offer at all."""
    feasible = problem.feasible_routes()
    if not feasible:
        raise RuntimeError("Instance has no hard-feasible route")

    crlbs = np.array([problem.route_crlb_m(r) for r in feasible])
    costs = np.array([problem.route_cost(r) for r in feasible])
    best_cost_route = feasible[int(np.argmin(costs))]

    return {
        "feasible_routes": len(feasible),
        "prior_position_sigma_m": problem.magnetic_map.prior_position_sigma_m,
        "sensor_sigma_nt": problem.magnetic_map.sensor.sigma_nt,
        "best_possible_crlb_m": float(np.min(crlbs)),
        "worst_feasible_crlb_m": float(np.max(crlbs)),
        "crlb_of_cost_optimal_route_m": problem.route_crlb_m(best_cost_route),
    }


def surrogate_fidelity(problem: NavigationProblem) -> dict:
    """How faithfully the separable QUBO information term tracks exact D-optimality.

    The optimizers can only see a sum of per-cell scores, because that is what a
    quadratic encoding admits.  The quantity a navigator actually cares about is
    ``log det J`` of the accumulated route information, which is not separable.  This
    reports the rank correlation between the two over all feasible routes -- the
    quantitative version of the caveat that the QUBO objective is a surrogate.
    """
    feasible = problem.feasible_routes()
    if len(feasible) < 3:
        return {"feasible_routes": len(feasible), "spearman_rho": float("nan")}

    separable = np.array(
        [
            sum(float(problem.magnetic_map.information[r, t]) for t, r in enumerate(route))
            for route in feasible
        ]
    )
    exact = np.array([problem.route_log_det_information(route) for route in feasible])
    return {
        "feasible_routes": len(feasible),
        "spearman_rho": _spearman(separable, exact),
        "pearson_r": _pearson(separable, exact),
    }


def _rank(a: np.ndarray) -> np.ndarray:
    """Average ranks, so ties do not bias the correlation."""
    order = np.argsort(a, kind="mergesort")
    ranks = np.empty(len(a), dtype=float)
    ranks[order] = np.arange(len(a), dtype=float)
    _, inverse, counts = np.unique(a, return_inverse=True, return_counts=True)
    sums = np.zeros(len(counts))
    np.add.at(sums, inverse, ranks)
    return (sums / counts)[inverse]


def _pearson(a: np.ndarray, b: np.ndarray) -> float:
    if np.std(a) < 1e-12 or np.std(b) < 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    return _pearson(_rank(a), _rank(b))
