"""Statevector QAOA: a penalty encoding with the X mixer, and a one-hot-preserving XY mixer.

Both solvers are exact statevector simulations of small instances, so the reported
probabilities are the true output distributions of the ansatz rather than shot
estimates.  Nothing here is a claim of quantum advantage: the same instances are solved
exactly and far faster by dynamic programming in :mod:`qmagopt.classical`.

What the comparison is actually for
-----------------------------------
The penalty encoding lives in the full ``2^(rows*steps)`` Hilbert space, where the vast
majority of basis states are not valid routes at all.  The XY variant restricts the
dynamics to the one-hot subspace of dimension ``rows^steps``, which for the reference
instance is 243 states instead of 32768.  Simply reporting "more feasible probability"
would therefore be measuring the size of the search space, not the quality of the
ansatz.  Every result also carries the *conditional* feasibility inside the one-hot
sector, which is the honest comparison.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy.optimize import minimize

from .problem import NavigationProblem
from .results import SolverResult

MixerGraph = Literal["complete", "ring"]
Objective = Literal["expectation", "cvar"]

#: Simulating more than this many qubits as a dense statevector is not the point of a demo.
MAX_PENALTY_QUBITS = 20


@dataclass(frozen=True)
class QaoaConfig:
    """Settings shared by both variational solvers."""

    p: int = 1
    restarts: int = 4
    maxiter: int = 120
    seed: int = 21
    objective: Objective = "expectation"
    #: Tail fraction used when ``objective="cvar"``; small values chase good samples
    #: rather than a good mean, which is the standard fix for flat QAOA landscapes.
    cvar_alpha: float = 0.15
    mixer_graph: MixerGraph = "complete"
    #: Seed layer ``p`` from the optimized layer ``p-1`` via the INTERP heuristic.
    interp: bool = True
    optimizer: str = "COBYLA"

    def __post_init__(self) -> None:
        if self.p < 1:
            raise ValueError("p must be at least 1")
        if self.restarts < 1:
            raise ValueError("restarts must be at least 1")
        if not 0.0 < self.cvar_alpha <= 1.0:
            raise ValueError("cvar_alpha must lie in (0, 1]")


# --------------------------------------------------------------------- utilities


def _bit_matrix(n: int) -> np.ndarray:
    """All ``2^n`` binary vectors as rows, with column ``q`` holding bit ``q``."""
    idx = np.arange(1 << n, dtype=np.uint32)
    return ((idx[:, None] >> np.arange(n, dtype=np.uint32)[None, :]) & 1).astype(np.float64)


def _objective_value(costs: np.ndarray, probs: np.ndarray, config: QaoaConfig) -> float:
    """Expectation value, or CVaR over the best ``cvar_alpha`` fraction of the distribution."""
    if config.objective == "expectation":
        return float(np.dot(probs, costs))

    order = np.argsort(costs)
    sorted_costs = costs[order]
    sorted_probs = probs[order]
    cumulative = np.cumsum(sorted_probs)
    alpha = config.cvar_alpha
    cutoff = int(np.searchsorted(cumulative, alpha))
    cutoff = min(cutoff, len(sorted_costs) - 1)
    weights = sorted_probs[: cutoff + 1].copy()
    # Trim the last bucket so the weights sum to exactly alpha.
    excess = float(cumulative[cutoff] - alpha)
    if excess > 0.0:
        weights[cutoff] = max(weights[cutoff] - excess, 0.0)
    total = float(weights.sum())
    if total <= 0.0:
        return float(sorted_costs[0])
    return float(np.dot(weights, sorted_costs[: cutoff + 1]) / total)


def _interp_parameters(params: np.ndarray, p: int) -> np.ndarray:
    """INTERP heuristic of Zhou et al.: linearly interpolate depth-``p`` angles to ``p+1``.

    Optimized QAOA angle schedules vary smoothly with the layer index, so interpolating
    a converged schedule gives a far better starting point than a random one and keeps
    deeper circuits from stalling in poor local minima.
    """
    gammas, betas = params[:p], params[p:]
    out = []
    for block in (gammas, betas):
        new = np.zeros(p + 1)
        for i in range(1, p + 2):
            lower = block[i - 2] if 2 <= i <= p + 1 else 0.0
            upper = block[i - 1] if 1 <= i <= p else 0.0
            new[i - 1] = (i - 1) / p * lower + (p - i + 1) / p * upper
        out.append(new)
    return np.concatenate(out)


def _optimize(objective, config: QaoaConfig, p: int, rng: np.random.Generator, warm_start: np.ndarray | None):
    """Multi-restart local optimization of the variational angles."""
    best = None
    starts = []
    if warm_start is not None:
        starts.append(np.asarray(warm_start, dtype=float))
    while len(starts) < config.restarts:
        starts.append(np.concatenate([rng.uniform(0.0, 2 * np.pi, size=p), rng.uniform(0.0, np.pi, size=p)]))
    for x0 in starts:
        res = minimize(
            objective,
            x0,
            method=config.optimizer,
            options={"maxiter": config.maxiter, "rhobeg": 0.6},
        )
        if best is None or res.fun < best.fun:
            best = res
    assert best is not None
    return best


# ------------------------------------------------------------ penalty / X mixer


@dataclass
class _PenaltyTables:
    """Precomputed, fully vectorized lookup tables over the ``2^n`` computational basis."""

    costs: np.ndarray
    one_hot: np.ndarray
    feasible: np.ndarray
    route_of_state: np.ndarray  # (2^n, steps), meaningful only where one_hot is True

    @classmethod
    def build(cls, problem: NavigationProblem) -> _PenaltyTables:
        n = problem.n_qubits
        if n > MAX_PENALTY_QUBITS:
            raise ValueError(
                f"Penalty-QAOA simulation is capped at {MAX_PENALTY_QUBITS} qubits; "
                f"this instance needs {n}. Reduce rows or steps."
            )
        bits = _bit_matrix(n)
        Q, offset = problem.qubo()
        costs = np.einsum("ki,ki->k", bits @ Q, bits) + offset

        layered = bits.reshape(-1, problem.steps, problem.rows)
        one_hot = np.all(layered.sum(axis=2) == 1, axis=1)
        routes = np.argmax(layered, axis=2).astype(np.int16)

        feasible = one_hot.copy()
        if problem.steps >= 1:
            feasible &= routes[:, 0] == problem.start_row
            feasible &= routes[:, -1] == problem.goal_row
        for t in range(problem.steps - 1):
            feasible &= np.abs(routes[:, t + 1] - routes[:, t]) <= problem.max_row_jump
        for r, t in problem.blocked_cells:
            feasible &= routes[:, t] != r

        return cls(costs=costs, one_hot=one_hot, feasible=feasible, route_of_state=routes)


def _apply_x_mixer(state: np.ndarray, beta: float, n: int) -> np.ndarray:
    """Apply ``exp(-i beta sum_j X_j)`` in place of a dense matrix product.

    Each qubit is handled by viewing the statevector as ``(high, 2, low)`` where the
    middle axis is that qubit's bit, so the cost is ``O(n 2^n)`` rather than ``O(4^n)``.
    """
    out = np.array(state, dtype=complex, copy=True)
    c = math.cos(beta)
    s = -1j * math.sin(beta)
    for q in range(n):
        stride = 1 << q
        view = out.reshape(-1, 2, stride)
        a = view[:, 0, :].copy()
        b = view[:, 1, :].copy()
        view[:, 0, :] = c * a + s * b
        view[:, 1, :] = s * a + c * b
    return out


def _qaoa_state_x(costs: np.ndarray, params: np.ndarray, p: int, n: int) -> np.ndarray:
    """State after ``p`` layers of cost phase and transverse-field mixing, from ``|+>^n``."""
    state = np.full(len(costs), 1 / math.sqrt(len(costs)), dtype=complex)
    for gamma, beta in zip(params[:p], params[p:], strict=True):
        state = state * np.exp(-1j * gamma * costs)
        state = _apply_x_mixer(state, float(beta), n)
    return state


def solve_penalty_qaoa(
    problem: NavigationProblem,
    config: QaoaConfig | None = None,
    *,
    tables: _PenaltyTables | None = None,
    warm_start: np.ndarray | None = None,
) -> SolverResult:
    """Penalty-QUBO QAOA with the standard transverse-field mixer.

    The one-hot constraint is enforced only by the quadratic penalty, so the ansatz
    spends most of its amplitude on states that do not decode to a route at all.
    """
    config = config or QaoaConfig()
    start = time.perf_counter()
    n = problem.n_qubits
    tables = tables or _PenaltyTables.build(problem)
    costs = tables.costs
    rng = np.random.default_rng(config.seed)

    def objective(params: np.ndarray) -> float:
        state = _qaoa_state_x(costs, params, config.p, n)
        return _objective_value(costs, np.abs(state) ** 2, config)

    best = _optimize(objective, config, config.p, rng, warm_start)
    probs = np.abs(_qaoa_state_x(costs, best.x, config.p, n)) ** 2

    one_hot_prob = float(probs[tables.one_hot].sum())
    feasible_prob = float(probs[tables.feasible].sum())

    feasible_idx = np.flatnonzero(tables.feasible)
    if feasible_idx.size:
        pick = int(feasible_idx[int(np.argmax(probs[feasible_idx]))])
    else:
        one_hot_idx = np.flatnonzero(tables.one_hot)
        # Surface the failure rather than papering over it with a repaired route.
        pick = int(one_hot_idx[int(np.argmax(probs[one_hot_idx]))]) if one_hot_idx.size else -1

    if pick >= 0:
        route = tuple(int(r) for r in tables.route_of_state[pick])
        route_prob = float(probs[pick])
    else:
        route = tuple([problem.start_row] * problem.steps)
        route_prob = 0.0

    return SolverResult(
        method=f"Penalty QAOA, X mixer (p={config.p})",
        route=route,
        objective=problem.route_cost(route),
        hard_feasible=problem.is_hard_feasible(route),
        success_probability=route_prob,
        feasible_probability=feasible_prob,
        one_hot_probability=one_hot_prob,
        parameters=tuple(float(v) for v in best.x),
        seconds=time.perf_counter() - start,
        extras={"expected_qubo_energy": float(np.dot(probs, costs)), "hilbert_dim": float(1 << n)},
    )


# ------------------------------------------------------- constraint-preserving XY


def _mixer_adjacency(rows: int, graph: MixerGraph) -> np.ndarray:
    """Adjacency of the mixer graph on one layer's candidate rows.

    ``complete`` couples every pair of rows and mixes fastest; ``ring`` couples only
    neighbours and is the variant that stays shallow on hardware with limited
    connectivity.  Both preserve the one-excitation subspace.
    """
    if graph == "complete":
        return np.ones((rows, rows)) - np.eye(rows)
    if graph == "ring":
        A = np.zeros((rows, rows))
        for i in range(rows):
            A[i, (i + 1) % rows] = 1.0
            A[(i + 1) % rows, i] = 1.0
        return A
    raise ValueError(f"Unknown mixer graph: {graph!r}")


def _xy_layer_matrix(rows: int, beta: float, graph: MixerGraph = "complete") -> np.ndarray:
    """``exp(-i beta A)`` on one layer's one-excitation subspace.

    The XY mixer ``sum_(i,j) in E (X_i X_j + Y_i Y_j)`` restricted to states with exactly
    one excitation acts as a continuous-time quantum walk on the mixer graph, i.e. as
    twice its adjacency matrix.  The factor of two is absorbed into ``beta``.  Layers
    act on disjoint qubits, so the full mixer factorizes exactly into these blocks.
    """
    A = _mixer_adjacency(rows, graph)
    vals, vecs = np.linalg.eigh(A)
    return (vecs * np.exp(-1j * beta * vals)) @ vecs.conj().T


def _apply_layer_matrix(state: np.ndarray, mat: np.ndarray, axis: int, dims: tuple[int, ...]) -> np.ndarray:
    """Apply a single-layer operator to one tensor axis of the route-indexed state."""
    tensor = state.reshape(dims)
    moved = np.moveaxis(tensor, axis, 0)
    shape = moved.shape
    mixed = mat @ moved.reshape(dims[axis], -1)
    return np.moveaxis(mixed.reshape(shape), 0, axis).reshape(-1)


def _qaoa_state_xy(
    costs: np.ndarray, params: np.ndarray, p: int, rows: int, steps: int, graph: MixerGraph
) -> np.ndarray:
    """State in the one-hot subspace, starting from the uniform superposition of routes."""
    dims = (rows,) * steps
    state = np.full(len(costs), 1 / math.sqrt(len(costs)), dtype=complex)
    for gamma, beta in zip(params[:p], params[p:], strict=True):
        state = state * np.exp(-1j * gamma * costs)
        mat = _xy_layer_matrix(rows, float(beta), graph)
        for axis in range(steps):
            state = _apply_layer_matrix(state, mat, axis, dims)
    return state


def solve_xy_qaoa(
    problem: NavigationProblem,
    config: QaoaConfig | None = None,
    *,
    warm_start: np.ndarray | None = None,
) -> SolverResult:
    """QAOA confined to the one-hot subspace by an XY mixer.

    Exactly one row per time step is occupied by construction, so the one-hot penalty
    disappears from the objective entirely.  Start/goal, obstacle and jump constraints
    remain soft penalties, which is what makes this a partial rather than complete
    constraint-preserving encoding -- an honest intermediate point, and the natural
    place for a transition-preserving mixer in future work.
    """
    config = config or QaoaConfig()
    start = time.perf_counter()
    routes, costs = problem.route_costs()
    rng = np.random.default_rng(config.seed)

    def objective(params: np.ndarray) -> float:
        state = _qaoa_state_xy(costs, params, config.p, problem.rows, problem.steps, config.mixer_graph)
        return _objective_value(costs, np.abs(state) ** 2, config)

    best = _optimize(objective, config, config.p, rng, warm_start)
    probs = (
        np.abs(_qaoa_state_xy(costs, best.x, config.p, problem.rows, problem.steps, config.mixer_graph)) ** 2
    )

    feasible_mask = np.fromiter((problem.is_hard_feasible(r) for r in routes), dtype=bool, count=len(routes))
    feasible_prob = float(probs[feasible_mask].sum())
    feasible_idx = np.flatnonzero(feasible_mask)
    pick = (
        int(feasible_idx[int(np.argmax(probs[feasible_idx]))]) if feasible_idx.size else int(np.argmax(probs))
    )
    route = routes[pick]

    return SolverResult(
        method=f"XY-QAOA, one-hot preserving (p={config.p})",
        route=route,
        objective=problem.route_cost(route),
        hard_feasible=problem.is_hard_feasible(route),
        success_probability=float(probs[pick]),
        feasible_probability=feasible_prob,
        one_hot_probability=1.0,  # exact: the mixer never leaves the one-hot subspace
        parameters=tuple(float(v) for v in best.x),
        seconds=time.perf_counter() - start,
        extras={"expected_route_cost": float(np.dot(probs, costs)), "hilbert_dim": float(len(costs))},
    )


# ---------------------------------------------------------------- depth scaling


@dataclass
class DepthScan:
    """Results of running one ansatz at increasing depth with INTERP warm starts."""

    method: str
    depths: list[int] = field(default_factory=list)
    results: list[SolverResult] = field(default_factory=list)


def qaoa_depth_scan(
    problem: NavigationProblem,
    solver: str = "xy",
    *,
    p_max: int = 3,
    base_config: QaoaConfig | None = None,
) -> DepthScan:
    """Run depths ``1..p_max``, seeding each depth from the previous optimum.

    Without the INTERP warm start, deeper circuits routinely score *worse* than shallow
    ones because the optimizer stalls; with it, the depth trend is meaningful.
    """
    base = base_config or QaoaConfig()
    scan = DepthScan(method=solver)
    tables = _PenaltyTables.build(problem) if solver == "penalty" else None
    previous: np.ndarray | None = None

    for p in range(1, p_max + 1):
        config = QaoaConfig(
            p=p,
            restarts=base.restarts,
            maxiter=base.maxiter,
            seed=base.seed + p,
            objective=base.objective,
            cvar_alpha=base.cvar_alpha,
            mixer_graph=base.mixer_graph,
            interp=base.interp,
            optimizer=base.optimizer,
        )
        warm = previous if (base.interp and previous is not None) else None
        if solver == "penalty":
            result = solve_penalty_qaoa(problem, config, tables=tables, warm_start=warm)
        elif solver == "xy":
            result = solve_xy_qaoa(problem, config, warm_start=warm)
        else:
            raise ValueError(f"Unknown solver: {solver!r}")
        scan.depths.append(p)
        scan.results.append(result)
        if result.parameters is not None:
            previous = _interp_parameters(np.asarray(result.parameters), p)

    return scan
