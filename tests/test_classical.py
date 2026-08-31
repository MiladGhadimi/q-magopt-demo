"""Tests for the classical baselines, including exactness of the dynamic program."""

from __future__ import annotations

import pytest

from qmagopt import (
    NavigationProblem,
    build_demo_problem,
    solve_brute_force,
    solve_dynamic_programming,
    solve_greedy,
    solve_simulated_annealing,
)
from qmagopt.maps import make_synthetic_magnetic_map

SEEDS = [7, 11, 23, 42, 101]


@pytest.mark.parametrize("seed", SEEDS)
def test_dynamic_programming_matches_brute_force(seed):
    """The DP recursion must reproduce the enumerated optimum exactly, not approximately."""
    problem = build_demo_problem(seed=seed)
    exact = solve_brute_force(problem)
    dp = solve_dynamic_programming(problem)
    assert dp.objective == pytest.approx(exact.objective, abs=1e-9)
    assert dp.hard_feasible


@pytest.mark.parametrize("seed", SEEDS)
def test_dynamic_programming_is_no_worse_than_greedy(seed):
    problem = build_demo_problem(seed=seed)
    exact = solve_dynamic_programming(problem)
    greedy = solve_greedy(problem)
    assert greedy.hard_feasible
    assert exact.objective <= greedy.objective + 1e-10


@pytest.mark.parametrize("seed", SEEDS)
def test_dynamic_programming_beats_or_matches_annealing(seed):
    """Simulated annealing works on the same QUBO, so it can tie but never win."""
    problem = build_demo_problem(seed=seed)
    exact = solve_dynamic_programming(problem)
    annealed = solve_simulated_annealing(problem, sweeps=250, restarts=6, seed=seed)
    assert exact.objective <= annealed.objective + 1e-9


def test_annealing_reaches_the_optimum_on_the_reference_instance():
    problem = build_demo_problem()
    exact = solve_dynamic_programming(problem)
    annealed = solve_simulated_annealing(problem, sweeps=500, restarts=12, seed=0)
    assert annealed.hard_feasible
    assert annealed.objective == pytest.approx(exact.objective, abs=1e-9)
    assert annealed.extras["decoded_one_hot"] == 1.0


def test_annealing_is_deterministic_for_a_fixed_seed():
    problem = build_demo_problem()
    a = solve_simulated_annealing(problem, sweeps=120, restarts=3, seed=5)
    b = solve_simulated_annealing(problem, sweeps=120, restarts=3, seed=5)
    assert a.route == b.route
    assert a.objective == pytest.approx(b.objective)


def test_annealing_energy_matches_the_qubo():
    problem = build_demo_problem()
    result = solve_simulated_annealing(problem, sweeps=200, restarts=4, seed=3)
    bits = problem.route_to_bitstring(result.route)
    x = [int(b) for b in bits]
    assert result.extras["qubo_energy"] == pytest.approx(problem.qubo_cost(x), abs=1e-8)


def test_all_solvers_return_the_declared_number_of_steps():
    problem = build_demo_problem()
    for solver in (solve_brute_force, solve_dynamic_programming, solve_greedy):
        assert len(solver(problem).route) == problem.steps


def test_blocked_corridor_reports_no_route():
    """A wall across the corridor must raise, not silently return a broken plan."""
    magnetic_map = make_synthetic_magnetic_map(3, 5, seed=7)
    walled = NavigationProblem(
        magnetic_map=magnetic_map,
        start_row=1,
        goal_row=1,
        max_row_jump=1,
        blocked_cells=frozenset({(0, 2), (1, 2), (2, 2)}),
    )
    with pytest.raises(RuntimeError):
        solve_dynamic_programming(walled)
    with pytest.raises(RuntimeError):
        solve_brute_force(walled)


def test_timings_are_recorded():
    problem = build_demo_problem()
    assert solve_dynamic_programming(problem).seconds is not None
    assert solve_greedy(problem).seconds >= 0.0
