"""Tests for the routing problem, its constraints and its QUBO / Ising encodings."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

from qmagopt import NavigationProblem, NavigationWeights, build_demo_problem
from qmagopt.maps import make_synthetic_magnetic_map


@pytest.fixture
def problem() -> NavigationProblem:
    return build_demo_problem()


def test_reference_instance_shape(problem):
    assert problem.magnetic_map.shape == (3, 5)
    assert problem.rows == 3
    assert problem.steps == 5
    assert problem.n_qubits == 15


def test_variable_indexing_is_a_bijection(problem):
    seen = {problem.variable_index(r, t) for t in range(problem.steps) for r in range(problem.rows)}
    assert seen == set(range(problem.n_qubits))


def test_bitstring_round_trip_for_every_route(problem):
    for route in problem.all_routes():
        assert problem.decode_bitstring(problem.route_to_bitstring(route)) == route


def test_non_one_hot_strings_do_not_decode(problem):
    assert problem.decode_bitstring("0" * problem.n_qubits) is None
    assert problem.decode_bitstring("1" * problem.n_qubits) is None
    with pytest.raises(ValueError):
        problem.decode_bitstring("101")


def test_blocked_cell_route_is_rejected(problem):
    """A correctly sized route through the blocked cell must fail on the obstacle itself."""
    blocked_route = (1, 1, 1, 1, 1)
    assert (1, 3) in problem.blocked_cells
    assert len(blocked_route) == problem.steps
    assert blocked_route[0] == problem.start_row and blocked_route[-1] == problem.goal_row
    assert all(abs(blocked_route[t + 1] - blocked_route[t]) <= problem.max_row_jump for t in range(4))
    # Every other constraint is satisfied, so only the obstacle can be the cause.
    assert not problem.is_hard_feasible(blocked_route)


def test_wrong_length_route_is_rejected_but_not_confused_with_infeasibility(problem):
    assert not problem.is_hard_feasible((1, 1, 1, 1, 1, 1))
    with pytest.raises(ValueError):
        problem.route_cost((1, 1, 1))


@pytest.mark.parametrize(
    "route,reason",
    [
        ((0, 0, 0, 0, 1), "wrong start"),
        ((1, 0, 0, 0, 0), "wrong goal"),
        ((1, 2, 0, 2, 1), "jump larger than max_row_jump"),
        ((1, 1, 1, 1, 1), "passes through the blocked cell"),
    ],
)
def test_specific_constraint_violations(problem, route, reason):
    assert not problem.is_hard_feasible(route), reason


def test_feasible_routes_agree_with_enumeration(problem):
    expected = [r for r in itertools.product(range(3), repeat=5) if problem.is_hard_feasible(r)]
    assert problem.feasible_routes() == expected
    assert len(expected) > 0


def test_qubo_reproduces_binary_cost_on_random_states(problem):
    rng = np.random.default_rng(0)
    Q, offset = problem.qubo()
    for _ in range(300):
        x = rng.integers(0, 2, size=problem.n_qubits).astype(float)
        assert problem.binary_cost(x) == pytest.approx(float(x @ Q @ x + offset), abs=1e-9)


def test_qubo_reproduces_binary_cost_on_every_valid_route(problem):
    for route in problem.all_routes():
        bits = np.array([int(b) for b in problem.route_to_bitstring(route)], dtype=float)
        assert problem.binary_cost(bits) == pytest.approx(problem.qubo_cost(bits), abs=1e-9)


def test_route_cost_matches_qubo_on_one_hot_states(problem):
    """On the feasible sector the one-hot penalty vanishes, so both costs coincide."""
    for route in problem.all_routes():
        bits = np.array([int(b) for b in problem.route_to_bitstring(route)], dtype=float)
        assert problem.route_cost(route) == pytest.approx(problem.binary_cost(bits), abs=1e-9)


def test_ising_energy_matches_binary_cost(problem):
    rng = np.random.default_rng(1)
    h, J, const = problem.ising()
    for _ in range(300):
        x = rng.integers(0, 2, size=problem.n_qubits).astype(float)
        z = 1.0 - 2.0 * x
        energy = float(h @ z + z @ J @ z + const)
        assert problem.binary_cost(x) == pytest.approx(energy, abs=1e-9)


def test_ising_coupling_matrix_is_strictly_upper_triangular(problem):
    _, J, _ = problem.ising()
    assert np.allclose(np.tril(J), 0.0)


def test_penalties_are_strong_enough_to_bind(problem):
    report = problem.penalty_report()
    assert report["penalties_are_binding"] is True
    assert report["relaxed_optimum"] == pytest.approx(report["feasible_optimum"])


def test_weak_penalties_are_detected():
    """The report must fail loudly when a violation would pay for itself."""
    weak = build_demo_problem(weights=NavigationWeights(blocked=0.05, endpoint=0.05, invalid_transition=0.05))
    assert weak.penalty_report()["penalties_are_binding"] is False


def test_transition_cost_is_symmetric_and_penalizes_long_jumps(problem):
    assert problem.transition_cost(0, 2) == pytest.approx(problem.transition_cost(2, 0))
    assert (
        problem.transition_cost(0, 2) > problem.transition_cost(0, 1) + problem.weights.invalid_transition / 2
    )


def test_invalid_instances_are_rejected():
    magnetic_map = make_synthetic_magnetic_map(3, 5, seed=7)
    with pytest.raises(ValueError):
        NavigationProblem(magnetic_map, start_row=9)
    with pytest.raises(ValueError):
        NavigationProblem(magnetic_map, goal_row=-1)
    with pytest.raises(ValueError):
        NavigationProblem(magnetic_map, max_row_jump=0)
    with pytest.raises(ValueError):
        NavigationProblem(magnetic_map, blocked_cells=frozenset({(0, 99)}))
    with pytest.raises(ValueError):
        NavigationProblem(magnetic_map, start_row=1, blocked_cells=frozenset({(1, 0)}))


def test_route_analysis_helpers_are_consistent(problem):
    route = (1, 2, 2, 2, 1)
    assert problem.route_crlb_m(route) > 0.0
    assert np.isfinite(problem.route_log_det_information(route))
    assert problem.route_fisher(route).shape == (2, 2)
