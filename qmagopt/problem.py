"""Layered constrained routing problem and its QUBO / Ising encodings."""

from __future__ import annotations

import itertools
import math
from collections.abc import Iterable, Iterator
from dataclasses import dataclass, replace
from typing import Literal

import numpy as np

from .maps import (
    MagneticMap,
    d_optimality,
    make_synthetic_magnetic_map,
    position_crlb_m,
    route_fisher,
)

#: How the objective rewards magnetic observability.
#:
#: ``separable``
#:     Sum of per-cell normalized information scores.  Cheap, sparse, and the usual
#:     choice -- but it is only a surrogate for what navigation actually needs.
#: ``d_optimal``
#:     The exact D-optimality criterion ``det J`` of the accumulated route information.
#:     For two-dimensional position this is *exactly quadratic* in the selection
#:     variables, so it fits a QUBO with no approximation at all (see
#:     :meth:`NavigationProblem.information_qubo_terms`).
InformationObjective = Literal["separable", "d_optimal"]


@dataclass(frozen=True)
class NavigationWeights:
    """Objective weights.

    ``distance`` and ``energy`` scale dimensionless per-step quantities, ``risk`` and
    ``magnetic_information`` scale scores normalized to [0, 1], and the remaining
    weights are penalties.  Penalty magnitudes must exceed the largest achievable
    objective improvement from violating the corresponding constraint, otherwise the
    optimum of the relaxed problem is infeasible; :meth:`NavigationProblem.penalty_report`
    checks this explicitly.
    """

    distance: float = 1.0
    energy: float = 0.30
    risk: float = 1.8
    magnetic_information: float = 2.4
    invalid_transition: float = 9.0
    blocked: float = 12.0
    endpoint: float = 12.0
    one_hot: float = 11.0


@dataclass(frozen=True)
class NavigationProblem:
    """Route selection over a layered graph in a GNSS-denied corridor.

    At each navigation step (map column) ``t`` exactly one candidate row must be
    occupied.  The objective rewards magnetically informative cells while penalizing
    distance, motion energy, operational risk, blocked cells and non-local row changes.

    Binary variables are ordered ``index = t * rows + r``, so bits are grouped by time
    step.  Both the QUBO and the statevector simulators follow that convention.
    """

    magnetic_map: MagneticMap
    start_row: int = 1
    goal_row: int = 1
    max_row_jump: int = 1
    blocked_cells: frozenset[tuple[int, int]] = frozenset()
    weights: NavigationWeights = NavigationWeights()
    information_objective: InformationObjective = "separable"

    def __post_init__(self) -> None:
        rows, steps = self.magnetic_map.shape
        if self.information_objective not in ("separable", "d_optimal"):
            raise ValueError(f"Unknown information objective: {self.information_objective!r}")
        if not 0 <= self.start_row < rows:
            raise ValueError(f"start_row {self.start_row} outside 0..{rows - 1}")
        if not 0 <= self.goal_row < rows:
            raise ValueError(f"goal_row {self.goal_row} outside 0..{rows - 1}")
        if self.max_row_jump < 1:
            raise ValueError("max_row_jump must be at least 1")
        for r, t in self.blocked_cells:
            if not (0 <= r < rows and 0 <= t < steps):
                raise ValueError(f"Blocked cell {(r, t)} is outside the map")
        if (self.start_row, 0) in self.blocked_cells:
            raise ValueError("The start cell is blocked; the instance has no feasible route")
        if (self.goal_row, steps - 1) in self.blocked_cells:
            raise ValueError("The goal cell is blocked; the instance has no feasible route")

    # ------------------------------------------------------------------ geometry

    @property
    def rows(self) -> int:
        return self.magnetic_map.shape[0]

    @property
    def steps(self) -> int:
        return self.magnetic_map.shape[1]

    @property
    def n_qubits(self) -> int:
        return self.rows * self.steps

    def variable_index(self, row: int, t: int) -> int:
        """Binary-variable index for occupying ``row`` at step ``t``."""
        return t * self.rows + row

    # ------------------------------------------------------------------- costs

    def node_base_cost(self, row: int, t: int) -> float:
        """Per-cell cost excluding any magnetic-information reward.

        Risk plus the soft obstacle and endpoint penalties.  Kept separate because the
        ``d_optimal`` objective's information term is pairwise, not per-cell.
        """
        w = self.weights
        c = w.risk * float(self.magnetic_map.risk[row, t])
        if (row, t) in self.blocked_cells:
            c += w.blocked
        if t == 0 and row != self.start_row:
            c += w.endpoint
        if t == self.steps - 1 and row != self.goal_row:
            c += w.endpoint
        return c

    def node_cost(self, row: int, t: int) -> float:
        """Cost of occupying a single cell under the ``separable`` information objective.

        Only meaningful when ``information_objective == "separable"``; the ``d_optimal``
        objective has no per-cell information term to fold in, so this returns the base
        cost and the information reward is applied at route level instead.
        """
        c = self.node_base_cost(row, t)
        if self.information_objective == "separable":
            c -= self.weights.magnetic_information * float(self.magnetic_map.information[row, t])
        return c

    # ---------------------------------------------------- exact D-optimality terms

    def _fisher_scale(self) -> float:
        """Reference ``det J``, so the D-optimal reward is dimensionless and O(1).

        Computed greedily: walk the columns and take the cell giving the largest marginal
        determinant gain, ignoring routing constraints.  That is an *achievable* selection
        rather than a loose analytic bound like ``det <= (tr/2)^2``, so realistic routes
        score near 1 instead of near 0, and it costs ``O(steps * rows)`` rather than an
        enumeration -- it stays defined for instances far too large to enumerate.
        """
        accumulated = self.magnetic_map.prior_fisher.copy()
        for t in range(self.steps):
            gains = [
                float(np.linalg.det(accumulated + self.magnetic_map.fisher[r, t])) for r in range(self.rows)
            ]
            accumulated = accumulated + self.magnetic_map.fisher[int(np.argmax(gains)), t]
        return max(float(np.linalg.det(accumulated)), 1e-30)

    def information_qubo_terms(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Exact D-optimality as ``(linear, quadratic, constant)`` in the binary variables.

        For two-dimensional position the accumulated information is
        ``J(x) = J_0 + sum_i x_i F_i`` with each ``F_i = g_i g_i^T / sigma^2`` rank one,
        and the determinant of a 2x2 matrix is a quadratic form in its entries.  Writing
        it out with ``x_i^2 = x_i`` gives

            det J(x) = det J_0
                     + sum_i x_i [ J0_11 F_i,22 + J0_22 F_i,11 - 2 J0_12 F_i,12 ]
                     + sum_{i<j} x_i x_j |g_i x g_j|^2 / sigma^4,

        where the pairwise coefficient is the squared cross product of the two gradient
        vectors -- the Cauchy-Binet expansion.  The rank-one diagonal terms vanish
        identically because ``det F_i = 0``.

        Two things follow.  The exact information criterion needs **no surrogate**: it is
        already quadratic and drops straight into a QUBO.  And it is genuinely pairwise
        across *all* pairs of time steps, which is why it rewards visiting cells whose
        gradients point in *different* directions -- and why a dynamic program over a
        single-step state cannot solve it.  The price is an all-to-all coupled QUBO.
        """
        n = self.n_qubits
        linear = np.zeros(n)
        quadratic = np.zeros((n, n))
        J0 = self.magnetic_map.prior_fisher

        cells = [(r, t) for t in range(self.steps) for r in range(self.rows)]
        F = {(r, t): self.magnetic_map.fisher[r, t] for r, t in cells}

        for r, t in cells:
            i = self.variable_index(r, t)
            Fi = F[(r, t)]
            linear[i] = J0[0, 0] * Fi[1, 1] + J0[1, 1] * Fi[0, 0] - 2.0 * J0[0, 1] * Fi[0, 1]

        for a, (r0, t0) in enumerate(cells):
            for r1, t1 in cells[a + 1 :]:
                i = self.variable_index(r0, t0)
                j = self.variable_index(r1, t1)
                Fi, Fj = F[(r0, t0)], F[(r1, t1)]
                coefficient = Fi[0, 0] * Fj[1, 1] + Fj[0, 0] * Fi[1, 1] - 2.0 * Fi[0, 1] * Fj[0, 1]
                lo, hi = (i, j) if i < j else (j, i)
                quadratic[lo, hi] += coefficient

        return linear, quadratic, float(np.linalg.det(J0))

    def information_reward(self, route: Iterable[int]) -> float:
        """Observability reward for a decoded route, on a common scale across objectives.

        Both modes land on roughly ``[0, steps]`` so that ``weights.magnetic_information``
        keeps the same meaning whichever objective is selected.
        """
        route = self._as_route(route)
        if self.information_objective == "separable":
            return sum(float(self.magnetic_map.information[r, t]) for t, r in enumerate(route))
        determinant = float(np.linalg.det(self.route_fisher(route)))
        return self.steps * determinant / self._fisher_scale()

    def transition_cost(self, r0: int, r1: int) -> float:
        """Cost of moving from row ``r0`` to row ``r1`` over one column."""
        dy = abs(r1 - r0)
        distance = math.sqrt(1.0 + dy * dy)
        energy = 1.0 + 0.9 * dy
        cost = self.weights.distance * distance + self.weights.energy * energy
        if dy > self.max_row_jump:
            cost += self.weights.invalid_transition
        return cost

    def route_cost(self, route: Iterable[int]) -> float:
        """Objective value of a fully decoded route (one row per step)."""
        route = self._as_route(route)
        total = sum(self.node_base_cost(r, t) for t, r in enumerate(route))
        total += sum(self.transition_cost(route[t], route[t + 1]) for t in range(self.steps - 1))
        total -= self.weights.magnetic_information * self.information_reward(route)
        return float(total)

    def is_hard_feasible(self, route: Iterable[int]) -> bool:
        """True when the route satisfies every constraint exactly, not just softly."""
        route = tuple(int(r) for r in route)
        if len(route) != self.steps:
            return False
        if any(not 0 <= r < self.rows for r in route):
            return False
        if route[0] != self.start_row or route[-1] != self.goal_row:
            return False
        if any((r, t) in self.blocked_cells for t, r in enumerate(route)):
            return False
        return all(abs(route[t + 1] - route[t]) <= self.max_row_jump for t in range(self.steps - 1))

    def _as_route(self, route: Iterable[int]) -> tuple[int, ...]:
        route = tuple(int(r) for r in route)
        if len(route) != self.steps:
            raise ValueError(f"Expected {self.steps} route rows, got {len(route)}")
        if any(not 0 <= r < self.rows for r in route):
            raise ValueError("Route contains an out-of-range row")
        return route

    # ------------------------------------------------------------- enumeration

    def all_routes(self) -> Iterator[tuple[int, ...]]:
        """Every one-hot-decodable route; ``rows ** steps`` of them."""
        return itertools.product(range(self.rows), repeat=self.steps)

    def feasible_routes(self) -> list[tuple[int, ...]]:
        """Every route satisfying all hard constraints."""
        return [r for r in self.all_routes() if self.is_hard_feasible(r)]

    def route_costs(self) -> tuple[list[tuple[int, ...]], np.ndarray]:
        """All one-hot routes with their objective values, in a fixed order."""
        routes = list(self.all_routes())
        costs = np.fromiter((self.route_cost(r) for r in routes), dtype=float, count=len(routes))
        return routes, costs

    # -------------------------------------------------------------- encodings

    def route_to_bitstring(self, route: Iterable[int]) -> str:
        """Encode a route as a bitstring whose position ``t * rows + r`` is variable ``(t, r)``."""
        route = self._as_route(route)
        bits = ["0"] * self.n_qubits
        for t, r in enumerate(route):
            bits[self.variable_index(r, t)] = "1"
        return "".join(bits)

    def decode_bitstring(self, bits: str) -> tuple[int, ...] | None:
        """Decode a bitstring, returning ``None`` if any layer is not one-hot."""
        if len(bits) != self.n_qubits:
            raise ValueError(f"Expected a {self.n_qubits}-bit string, got {len(bits)}")
        route = []
        for t in range(self.steps):
            chunk = bits[t * self.rows : (t + 1) * self.rows]
            ones = [i for i, b in enumerate(chunk) if b == "1"]
            if len(ones) != 1:
                return None
            route.append(ones[0])
        return tuple(route)

    def binary_cost(self, bits: np.ndarray) -> float:
        """Penalty-QUBO objective on an arbitrary binary vector (one-hot not assumed)."""
        flat = np.asarray(bits, dtype=float).ravel()
        x = flat.reshape(self.steps, self.rows)
        w = self.weights
        total = 0.0
        for t in range(self.steps):
            total += w.one_hot * float((np.sum(x[t]) - 1) ** 2)
            for r in range(self.rows):
                total += float(x[t, r]) * self.node_base_cost(r, t)
        for t in range(self.steps - 1):
            for r0 in range(self.rows):
                for r1 in range(self.rows):
                    total += float(x[t, r0]) * float(x[t + 1, r1]) * self.transition_cost(r0, r1)

        if self.information_objective == "separable":
            for t in range(self.steps):
                for r in range(self.rows):
                    total -= (
                        w.magnetic_information * float(x[t, r]) * float(self.magnetic_map.information[r, t])
                    )
        else:
            linear, quadratic, constant = self.information_qubo_terms()
            determinant = constant + float(linear @ flat) + float(flat @ quadratic @ flat)
            total -= w.magnetic_information * self.steps * determinant / self._fisher_scale()
        return float(total)

    def qubo(self) -> tuple[np.ndarray, float]:
        """Upper-triangular QUBO matrix ``Q`` and offset with ``cost(x) = x^T Q x + offset``.

        Using ``x^2 = x`` for binary variables, each one-hot penalty
        ``lambda (sum_r x_tr - 1)^2`` contributes ``-lambda`` to every diagonal entry of
        its layer, ``2 lambda`` to every within-layer pair, and ``lambda`` to the offset.
        Verified against :meth:`binary_cost` in the test suite.
        """
        n = self.n_qubits
        Q = np.zeros((n, n), dtype=float)
        lam = self.weights.one_hot
        offset = lam * self.steps

        for t in range(self.steps):
            for r in range(self.rows):
                i = self.variable_index(r, t)
                Q[i, i] += self.node_base_cost(r, t) - lam
            for r0 in range(self.rows):
                for r1 in range(r0 + 1, self.rows):
                    i = self.variable_index(r0, t)
                    j = self.variable_index(r1, t)
                    Q[i, j] += 2.0 * lam

        for t in range(self.steps - 1):
            for r0 in range(self.rows):
                for r1 in range(self.rows):
                    i = self.variable_index(r0, t)
                    j = self.variable_index(r1, t + 1)
                    Q[i, j] += self.transition_cost(r0, r1)

        if self.information_objective == "separable":
            for t in range(self.steps):
                for r in range(self.rows):
                    i = self.variable_index(r, t)
                    Q[i, i] -= self.weights.magnetic_information * float(self.magnetic_map.information[r, t])
        else:
            # Exact D-optimality: linear terms on the diagonal, Cauchy-Binet cross terms
            # as all-to-all couplings.  No approximation is made here.
            linear, quadratic, constant = self.information_qubo_terms()
            gain = self.weights.magnetic_information * self.steps / self._fisher_scale()
            Q -= gain * quadratic
            Q[np.diag_indices(n)] -= gain * linear
            offset -= gain * constant

        return Q, float(offset)

    def ising(self) -> tuple[np.ndarray, np.ndarray, float]:
        """Ising form ``(h, J, offset)`` with ``E(z) = sum_i h_i z_i + sum_{i<j} J_ij z_i z_j + offset``.

        Obtained from the QUBO by ``x_i = (1 - z_i) / 2``, so ``z_i = +1`` means the cell
        is *not* occupied.  This is the form a gate-based or annealing backend consumes.
        """
        Q, offset = self.qubo()
        diag = np.diag(Q).copy()
        off = Q - np.diag(diag)  # strictly upper triangular

        J = off / 4.0
        h = -diag / 2.0 - (off.sum(axis=1) + off.sum(axis=0)) / 4.0
        const = offset + diag.sum() / 2.0 + off.sum() / 4.0
        return h, J, float(const)

    def qubo_cost(self, bits: np.ndarray) -> float:
        """Evaluate the QUBO form directly; equals :meth:`binary_cost`."""
        x = np.asarray(bits, dtype=float).ravel()
        Q, offset = self.qubo()
        return float(x @ Q @ x + offset)

    # --------------------------------------------------------------- analysis

    def route_fisher(self, route: Iterable[int], *, include_prior: bool = True) -> np.ndarray:
        """Accumulated position Fisher information along a route (1/m^2)."""
        return route_fisher(self.magnetic_map, self._as_route(route), include_prior=include_prior)

    def route_crlb_m(self, route: Iterable[int]) -> float:
        """Cramer-Rao lower bound on horizontal position error for a route, in metres."""
        return position_crlb_m(self.route_fisher(route))

    def route_log_det_information(self, route: Iterable[int]) -> float:
        """Exact D-optimality score ``log det J`` for a route."""
        return d_optimality(self.route_fisher(route))

    def penalty_report(self) -> dict[str, float | bool]:
        """Check that the penalty weights actually make the relaxed optimum feasible.

        A penalty formulation is only correct when no constraint violation can pay for
        itself.  This compares the best unconstrained-in-that-respect gain against each
        penalty, and confirms by enumeration that the global QUBO minimum is feasible.
        """
        w = self.weights
        info_span = float(np.ptp(self.magnetic_map.information))
        risk_span = float(np.ptp(self.magnetic_map.risk))
        best_node_gain = w.magnetic_information * info_span + w.risk * risk_span

        max_jump_saving = max(
            self.transition_cost(r0, r1) - w.invalid_transition
            for r0 in range(self.rows)
            for r1 in range(self.rows)
        ) - min(self.transition_cost(r0, r1) for r0 in range(self.rows) for r1 in range(self.rows))

        routes, costs = self.route_costs()
        feasible = np.fromiter((self.is_hard_feasible(r) for r in routes), dtype=bool, count=len(routes))
        best_overall = float(np.min(costs))
        best_feasible = float(np.min(costs[feasible])) if feasible.any() else float("inf")

        return {
            "blocked_penalty": w.blocked,
            "endpoint_penalty": w.endpoint,
            "invalid_transition_penalty": w.invalid_transition,
            "max_node_gain_from_violation": best_node_gain,
            "max_transition_gain_from_violation": float(max_jump_saving),
            "relaxed_optimum": best_overall,
            "feasible_optimum": best_feasible,
            "penalties_are_binding": bool(abs(best_overall - best_feasible) < 1e-9),
        }


def build_demo_problem(
    seed: int = 7,
    *,
    rows: int = 3,
    cols: int = 5,
    information_model: str = "fisher",
    information_objective: InformationObjective = "separable",
    blocked_cells: frozenset[tuple[int, int]] | None = None,
    weights: NavigationWeights | None = None,
) -> NavigationProblem:
    """The reference instance: a 3-row, 5-step corridor with one blocked cell."""
    magnetic_map = make_synthetic_magnetic_map(
        rows=rows,
        cols=cols,
        seed=seed,
        information_model=information_model,  # type: ignore[arg-type]
    )
    if blocked_cells is None:
        blocked_cells = frozenset({(1, 3)}) if rows >= 3 and cols >= 5 else frozenset()
    return NavigationProblem(
        magnetic_map=magnetic_map,
        start_row=rows // 2,
        goal_row=rows // 2,
        max_row_jump=1,
        blocked_cells=blocked_cells,
        weights=weights or NavigationWeights(),
        information_objective=information_objective,
    )


def with_information_model(problem: NavigationProblem, model: str) -> NavigationProblem:
    """Return the same instance scored by a different per-cell information model."""
    return replace(problem, magnetic_map=problem.magnetic_map.with_information_model(model))  # type: ignore[arg-type]
