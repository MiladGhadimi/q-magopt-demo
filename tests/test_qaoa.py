"""Tests for the statevector QAOA simulators.

These check the simulation itself -- mixer correctness, subspace preservation, the
cost tables -- rather than the quality of the variational answer, which is a research
result and belongs in the benchmark rather than in an assertion.
"""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import expm

from qmagopt import QaoaConfig, build_demo_problem, solve_penalty_qaoa, solve_xy_qaoa
from qmagopt.qaoa import (
    MAX_PENALTY_QUBITS,
    _apply_x_mixer,
    _bit_matrix,
    _interp_parameters,
    _mixer_adjacency,
    _objective_value,
    _PenaltyTables,
    _qaoa_state_x,
    _qaoa_state_xy,
    _xy_layer_matrix,
    qaoa_depth_scan,
)

PAULI_X = np.array([[0.0, 1.0], [1.0, 0.0]])


def _dense_x_mixer(beta: float, n: int) -> np.ndarray:
    """Reference ``exp(-i beta sum_j X_j)`` built explicitly, for small n."""
    total = np.zeros((1 << n, 1 << n), dtype=complex)
    for q in range(n):
        operator = np.array([[1.0]])
        # Qubit q has place value 2^q, so it is the *last* factor in the Kronecker order.
        for k in range(n - 1, -1, -1):
            operator = np.kron(operator, PAULI_X if k == q else np.eye(2))
        total += operator
    return expm(-1j * beta * total)


@pytest.mark.parametrize("n", [1, 2, 3, 4])
@pytest.mark.parametrize("beta", [0.0, 0.37, 1.9])
def test_x_mixer_matches_dense_matrix_exponential(n, beta):
    rng = np.random.default_rng(n)
    state = rng.normal(size=1 << n) + 1j * rng.normal(size=1 << n)
    state /= np.linalg.norm(state)
    assert np.allclose(_apply_x_mixer(state, beta, n), _dense_x_mixer(beta, n) @ state)


def test_x_mixer_preserves_norm_and_leaves_input_untouched():
    n = 6
    rng = np.random.default_rng(0)
    state = rng.normal(size=1 << n).astype(complex)
    state /= np.linalg.norm(state)
    original = state.copy()
    out = _apply_x_mixer(state, 0.8, n)
    assert np.isclose(np.linalg.norm(out), 1.0)
    assert np.allclose(state, original)


def test_bit_matrix_orders_bits_least_significant_first():
    bits = _bit_matrix(3)
    assert bits.shape == (8, 3)
    assert np.array_equal(bits[5], [1, 0, 1])  # 5 = 0b101


@pytest.mark.parametrize("graph", ["complete", "ring"])
@pytest.mark.parametrize("rows", [2, 3, 5])
def test_xy_layer_matrix_is_unitary(rows, graph):
    mat = _xy_layer_matrix(rows, 0.63, graph)
    assert np.allclose(mat @ mat.conj().T, np.eye(rows), atol=1e-12)


def test_mixer_adjacency_shapes():
    assert np.array_equal(_mixer_adjacency(3, "complete"), np.ones((3, 3)) - np.eye(3))
    ring = _mixer_adjacency(5, "ring")
    assert np.array_equal(ring, ring.T)
    assert ring.sum() == 10  # each of five nodes has two neighbours
    with pytest.raises(ValueError):
        _mixer_adjacency(3, "star")


def test_xy_layer_matrix_at_zero_is_the_identity():
    assert np.allclose(_xy_layer_matrix(4, 0.0), np.eye(4))


def test_penalty_cost_table_matches_binary_cost():
    problem = build_demo_problem()
    tables = _PenaltyTables.build(problem)
    rng = np.random.default_rng(2)
    for _ in range(200):
        index = int(rng.integers(0, 1 << problem.n_qubits))
        bits = [(index >> q) & 1 for q in range(problem.n_qubits)]
        assert tables.costs[index] == pytest.approx(problem.binary_cost(bits), abs=1e-9)


def test_penalty_tables_identify_the_right_sectors():
    problem = build_demo_problem()
    tables = _PenaltyTables.build(problem)
    assert int(tables.one_hot.sum()) == problem.rows**problem.steps
    assert int(tables.feasible.sum()) == len(problem.feasible_routes())
    assert np.all(tables.feasible <= tables.one_hot)

    for index in np.flatnonzero(tables.one_hot)[:50]:
        bits = "".join("1" if (int(index) >> q) & 1 else "0" for q in range(problem.n_qubits))
        assert problem.decode_bitstring(bits) == tuple(int(r) for r in tables.route_of_state[index])


def test_penalty_simulation_is_capped():
    big = build_demo_problem(rows=4, cols=6)  # 24 qubits
    assert big.n_qubits > MAX_PENALTY_QUBITS
    with pytest.raises(ValueError, match="capped"):
        _PenaltyTables.build(big)


def test_zero_angles_leave_the_uniform_superposition():
    problem = build_demo_problem()
    tables = _PenaltyTables.build(problem)
    zeros = np.zeros(2)

    state = _qaoa_state_x(tables.costs, zeros, 1, problem.n_qubits)
    probs = np.abs(state) ** 2
    assert np.allclose(probs, 1.0 / len(probs))

    _, route_costs = problem.route_costs()
    xy_state = _qaoa_state_xy(route_costs, zeros, 1, problem.rows, problem.steps, "complete")
    xy_probs = np.abs(xy_state) ** 2
    assert np.allclose(xy_probs, 1.0 / len(xy_probs))
    assert float(np.dot(xy_probs, route_costs)) == pytest.approx(float(np.mean(route_costs)))


def test_xy_ansatz_stays_normalized_inside_the_one_hot_subspace():
    problem = build_demo_problem()
    _, costs = problem.route_costs()
    params = np.array([0.7, 1.1, 0.4, 0.9])
    state = _qaoa_state_xy(costs, params, 2, problem.rows, problem.steps, "complete")
    assert np.isclose(float(np.sum(np.abs(state) ** 2)), 1.0)
    assert len(state) == problem.rows**problem.steps


def test_cvar_never_exceeds_the_mean():
    """CVaR averages the best tail, so it must be at most the full expectation."""
    rng = np.random.default_rng(4)
    costs = rng.normal(size=500)
    probs = rng.random(500)
    probs /= probs.sum()
    mean = _objective_value(costs, probs, QaoaConfig(objective="expectation"))
    for alpha in (0.05, 0.25, 0.75):
        cvar = _objective_value(costs, probs, QaoaConfig(objective="cvar", cvar_alpha=alpha))
        assert cvar <= mean + 1e-9
    full = _objective_value(costs, probs, QaoaConfig(objective="cvar", cvar_alpha=1.0))
    assert full == pytest.approx(mean, abs=1e-9)


def test_interp_grows_the_schedule_by_one_layer():
    params = np.array([0.5, 1.2])  # p = 1
    grown = _interp_parameters(params, 1)
    assert grown.shape == (4,)
    assert grown[0] == pytest.approx(0.5)  # first gamma is carried through
    assert grown[2] == pytest.approx(1.2)
    deeper = _interp_parameters(grown, 2)
    assert deeper.shape == (6,)


def test_xy_solver_never_leaves_the_one_hot_subspace():
    problem = build_demo_problem()
    result = solve_xy_qaoa(problem, QaoaConfig(p=1, restarts=1, maxiter=15, seed=3))
    assert result.one_hot_probability == 1.0
    assert len(result.route) == problem.steps
    assert all(0 <= r < problem.rows for r in result.route)
    assert 0.0 <= result.feasible_probability <= 1.0 + 1e-9
    assert result.conditional_feasible_probability == pytest.approx(result.feasible_probability)


def test_penalty_solver_reports_a_small_one_hot_sector():
    """Most of the penalty encoding's Hilbert space is not a route at all."""
    problem = build_demo_problem()
    result = solve_penalty_qaoa(problem, QaoaConfig(p=1, restarts=1, maxiter=15, seed=3))
    assert 0.0 < result.one_hot_probability < 1.0
    assert result.feasible_probability <= result.one_hot_probability + 1e-12
    assert result.extras["hilbert_dim"] == 2**problem.n_qubits


def test_solvers_are_deterministic_for_a_fixed_seed():
    problem = build_demo_problem()
    config = QaoaConfig(p=1, restarts=2, maxiter=20, seed=17)
    assert solve_xy_qaoa(problem, config).route == solve_xy_qaoa(problem, config).route
    assert solve_penalty_qaoa(problem, config).parameters == pytest.approx(
        solve_penalty_qaoa(problem, config).parameters
    )


def test_depth_scan_returns_one_result_per_depth():
    problem = build_demo_problem()
    scan = qaoa_depth_scan(problem, "xy", p_max=3, base_config=QaoaConfig(restarts=1, maxiter=15))
    assert scan.depths == [1, 2, 3]
    assert len(scan.results) == 3
    for p, result in zip(scan.depths, scan.results, strict=True):
        assert len(result.parameters) == 2 * p

    with pytest.raises(ValueError):
        qaoa_depth_scan(problem, "quantum-annealing", p_max=1)


def test_invalid_configurations_are_rejected():
    with pytest.raises(ValueError):
        QaoaConfig(p=0)
    with pytest.raises(ValueError):
        QaoaConfig(restarts=0)
    with pytest.raises(ValueError):
        QaoaConfig(cvar_alpha=0.0)
    with pytest.raises(ValueError):
        QaoaConfig(cvar_alpha=1.5)
