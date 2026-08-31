"""Tests for the reporting metrics."""

from __future__ import annotations

import itertools
import math
import statistics

import pytest

from qmagopt import build_demo_problem, solve_dynamic_programming, solve_greedy
from qmagopt.metrics import (
    approximation_ratio,
    information_summary,
    route_metrics,
    surrogate_fidelity,
)


def test_approximation_ratio_anchors_at_one_and_zero():
    problem = build_demo_problem()
    exact = solve_dynamic_programming(problem)
    feasible = problem.feasible_routes()
    worst = max(problem.route_cost(r) for r in feasible)

    assert approximation_ratio(problem, exact.objective, exact.objective) == pytest.approx(1.0)
    assert approximation_ratio(problem, worst, exact.objective) == pytest.approx(0.0)


def test_approximation_ratio_is_monotone():
    problem = build_demo_problem()
    exact = solve_dynamic_programming(problem)
    costs = sorted(problem.route_cost(r) for r in problem.feasible_routes())
    ratios = [approximation_ratio(problem, c, exact.objective) for c in costs]
    assert all(a >= b - 1e-12 for a, b in itertools.pairwise(ratios))


def test_route_metrics_reports_the_expected_fields():
    problem = build_demo_problem()
    exact = solve_dynamic_programming(problem)
    row = route_metrics(problem, exact, optimum=exact.objective)

    expected = {
        "method",
        "route",
        "objective",
        "distance",
        "motion_energy",
        "magnetic_information",
        "risk",
        "position_crlb_m",
        "log_det_fisher",
        "min_eigen_fisher",
        "hard_feasible",
        "success_probability",
        "one_hot_probability",
        "feasible_probability",
        "conditional_feasible_probability",
        "seconds",
        "approximation_ratio",
    }
    assert set(row) == expected
    assert row["approximation_ratio"] == pytest.approx(1.0)
    assert row["route"] == "-".join(map(str, exact.route))
    assert math.isfinite(row["position_crlb_m"])


def test_route_metrics_distance_matches_the_objective_terms():
    problem = build_demo_problem()
    greedy = solve_greedy(problem)
    row = route_metrics(problem, greedy)
    manual = sum(
        math.sqrt(1.0 + (greedy.route[t] - greedy.route[t - 1]) ** 2) for t in range(1, problem.steps)
    )
    assert row["distance"] == pytest.approx(manual)
    assert "approximation_ratio" not in row


def test_information_summary_brackets_the_achievable_accuracy():
    problem = build_demo_problem()
    summary = information_summary(problem)
    assert summary["feasible_routes"] == len(problem.feasible_routes())
    assert summary["best_possible_crlb_m"] <= summary["worst_feasible_crlb_m"]
    # Any route beats dead reckoning alone.
    assert summary["worst_feasible_crlb_m"] < summary["prior_position_sigma_m"]


def test_surrogate_fidelity_is_a_valid_correlation():
    problem = build_demo_problem()
    report = surrogate_fidelity(problem)
    assert report["feasible_routes"] > 0
    assert -1.0 - 1e-9 <= report["spearman_rho"] <= 1.0 + 1e-9
    assert -1.0 - 1e-9 <= report["pearson_r"] <= 1.0 + 1e-9


def test_surrogate_usually_but_not_always_tracks_exact_information():
    """The separable term is a decent surrogate on average and a poor one sometimes.

    This is a documented finding rather than a tolerance to be widened: a sum of
    per-cell scores cannot see gradient-direction diversity, which is exactly what
    ``log det J`` rewards.  The median must stay high; individual instances are allowed
    to fall apart, and that is the motivation for the ``d_optimal`` objective, which
    encodes the exact criterion instead of approximating it.
    """
    values = [surrogate_fidelity(build_demo_problem(seed=s))["spearman_rho"] for s in range(100, 130)]
    values = [v for v in values if not math.isnan(v)]
    assert len(values) >= 25
    assert statistics.median(values) > 0.7, values
    assert min(values) < 0.5, "expected at least one instance where the surrogate misleads"
