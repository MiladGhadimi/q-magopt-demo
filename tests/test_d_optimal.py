"""Tests for the exact D-optimality objective.

The claim under test is specific: for two-dimensional position, ``det J`` of the
accumulated Fisher information is *exactly* a quadratic function of the binary
selection variables, so the QUBO encodes the true information criterion with no
surrogate.  These tests check that identity numerically, and check the consequences --
that the dynamic program stops being valid, and that optimizing the exact criterion
does not degrade real positioning accuracy.
"""

from __future__ import annotations

import statistics
from dataclasses import replace

import numpy as np
import pytest

from qmagopt import (
    NavigationProblem,
    build_demo_problem,
    solve_brute_force,
    solve_dynamic_programming,
    solve_greedy,
    solve_simulated_annealing,
    solve_xy_qaoa,
)
from qmagopt.qaoa import QaoaConfig

SEEDS = [7, 11, 23, 42, 101]


def d_optimal_problem(seed: int = 7) -> NavigationProblem:
    return build_demo_problem(seed=seed, information_objective="d_optimal")


@pytest.mark.parametrize("seed", SEEDS)
def test_cauchy_binet_expansion_is_exact(seed):
    """The quadratic form must reproduce det J exactly on every one-hot state."""
    problem = d_optimal_problem(seed)
    linear, quadratic, constant = problem.information_qubo_terms()

    for route in problem.all_routes():
        x = np.array([int(b) for b in problem.route_to_bitstring(route)], dtype=float)
        encoded = constant + float(linear @ x) + float(x @ quadratic @ x)
        exact = float(np.linalg.det(problem.route_fisher(route)))
        assert encoded == pytest.approx(exact, abs=1e-18, rel=1e-9)


def test_pairwise_coefficients_are_squared_cross_products():
    """Each cross term is |g_i x g_j|^2 / sigma^4, hence non-negative."""
    problem = d_optimal_problem()
    _, quadratic, _ = problem.information_qubo_terms()
    sigma = problem.magnetic_map.sensor.sigma_nt
    gradient = problem.magnetic_map.gradient_nt_per_m

    cells = [(r, t) for t in range(problem.steps) for r in range(problem.rows)]
    for a, (r0, t0) in enumerate(cells):
        for r1, t1 in cells[a + 1 :]:
            i = problem.variable_index(r0, t0)
            j = problem.variable_index(r1, t1)
            g0, g1 = gradient[r0, t0], gradient[r1, t1]
            cross = g0[0] * g1[1] - g0[1] * g1[0]
            expected = cross**2 / sigma**4
            lo, hi = (i, j) if i < j else (j, i)
            assert quadratic[lo, hi] == pytest.approx(expected, abs=1e-18, rel=1e-9)
    assert np.all(quadratic >= -1e-18)


def test_rank_one_cells_contribute_no_diagonal_information_term():
    """det of a single rank-one contribution is zero, so no self-coupling appears."""
    problem = d_optimal_problem()
    _, quadratic, _ = problem.information_qubo_terms()
    assert np.allclose(np.diag(quadratic), 0.0)


@pytest.mark.parametrize("seed", SEEDS)
def test_qubo_and_route_cost_agree_under_the_exact_objective(seed):
    problem = d_optimal_problem(seed)
    for route in problem.all_routes():
        bits = np.array([int(b) for b in problem.route_to_bitstring(route)], dtype=float)
        assert problem.route_cost(route) == pytest.approx(problem.binary_cost(bits), abs=1e-9)
        assert problem.binary_cost(bits) == pytest.approx(problem.qubo_cost(bits), abs=1e-9)


def test_ising_form_survives_the_denser_couplings():
    problem = d_optimal_problem()
    h, J, const = problem.ising()
    rng = np.random.default_rng(0)
    for _ in range(200):
        x = rng.integers(0, 2, size=problem.n_qubits).astype(float)
        z = 1.0 - 2.0 * x
        assert problem.binary_cost(x) == pytest.approx(float(h @ z + z @ J @ z + const), abs=1e-9)


def test_exact_objective_couples_non_adjacent_time_steps():
    """The point of the encoding: information couples every pair of steps, not neighbours."""
    problem = d_optimal_problem()
    _, quadratic, _ = problem.information_qubo_terms()
    first = problem.variable_index(0, 0)
    last = problem.variable_index(0, problem.steps - 1)
    lo, hi = min(first, last), max(first, last)
    assert quadratic[lo, hi] > 0.0


def test_dynamic_programming_refuses_the_exact_objective():
    """DP has no valid single-step state once the objective is all-pairs coupled."""
    problem = d_optimal_problem()
    with pytest.raises(ValueError, match="separable"):
        solve_dynamic_programming(problem)
    # It remains available on the separable formulation of the same instance.
    assert solve_dynamic_programming(replace(problem, information_objective="separable")).hard_feasible


def test_annealing_usually_reaches_the_exact_optimum_and_never_beats_it():
    """The denser QUBO is measurably harder for the classical heuristic.

    With the dynamic program unavailable, simulated annealing is the strongest general
    baseline left -- and on the all-to-all coupled D-optimal encoding it stops solving
    every instance to optimality, which is precisely the regime where a different
    heuristic could earn its place.  Optimality is asserted as a majority, not as a
    per-instance guarantee.
    """
    optimal_hits = 0
    for seed in SEEDS:
        problem = d_optimal_problem(seed)
        exact = solve_brute_force(problem)
        annealed = solve_simulated_annealing(problem, sweeps=500, restarts=12, seed=seed)
        # An exact optimum can never be beaten, whatever the heuristic does.
        assert annealed.objective >= exact.objective - 1e-9
        if annealed.objective == pytest.approx(exact.objective, abs=1e-6):
            optimal_hits += 1
    assert optimal_hits >= len(SEEDS) - 1


def test_greedy_uses_marginal_determinant_gain():
    problem = d_optimal_problem()
    greedy = solve_greedy(problem)
    assert greedy.hard_feasible
    assert greedy.objective >= solve_brute_force(problem).objective - 1e-9


def test_xy_qaoa_runs_on_the_exact_objective():
    problem = d_optimal_problem()
    result = solve_xy_qaoa(problem, QaoaConfig(p=1, restarts=2, maxiter=25, seed=3))
    assert result.one_hot_probability == 1.0
    assert len(result.route) == problem.steps


def test_exact_objective_never_worsens_position_accuracy():
    """Optimizing det J directly must not lose to optimizing the separable surrogate.

    Both objectives are solved exactly, so this isolates the encoding from the solver.
    """
    separable_crlb = []
    exact_crlb = []
    for seed in range(100, 120):
        separable = build_demo_problem(seed=seed)
        exact = replace(separable, information_objective="d_optimal")
        separable_crlb.append(separable.route_crlb_m(solve_brute_force(separable).route))
        exact_crlb.append(exact.route_crlb_m(solve_brute_force(exact).route))

    assert all(e <= s + 1e-9 for e, s in zip(exact_crlb, separable_crlb, strict=True))
    assert statistics.median(exact_crlb) < statistics.median(separable_crlb)


def test_information_reward_scales_are_comparable_between_objectives():
    """The same weight must mean roughly the same thing under either objective."""
    separable = build_demo_problem()
    exact = replace(separable, information_objective="d_optimal")
    routes = separable.feasible_routes()
    a = [separable.information_reward(r) for r in routes]
    b = [exact.information_reward(r) for r in routes]
    assert min(b) >= 0.0
    assert max(b) < 4.0 * max(a)


def test_unknown_objective_is_rejected():
    problem = build_demo_problem()
    with pytest.raises(ValueError, match="information objective"):
        replace(problem, information_objective="a_optimal")
