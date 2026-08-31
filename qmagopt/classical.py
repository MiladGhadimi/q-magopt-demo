"""Classical baselines: exact enumeration, exact dynamic programming, greedy, annealing.

The point of this module is to keep the quantum results honest.  The layered objective
is separable, so dynamic programming solves it exactly in ``O(steps * rows^2)`` -- and
simulated annealing attacks the *same* QUBO the QAOA solvers receive.  Any claim about
the variational methods has to be read against these.
"""

from __future__ import annotations

import time

import numpy as np

from .problem import NavigationProblem
from .results import SolverResult

__all__ = [
    "SolverResult",
    "solve_brute_force",
    "solve_dynamic_programming",
    "solve_greedy",
    "solve_simulated_annealing",
]


def solve_brute_force(problem: NavigationProblem) -> SolverResult:
    """Exact optimum by enumerating every hard-feasible route.

    Exponential in the number of steps and only usable on demo-sized instances; it
    exists to certify the dynamic program rather than to compete with it.
    """
    start = time.perf_counter()
    best_route: tuple[int, ...] | None = None
    best_cost = float("inf")
    for route in problem.all_routes():
        if not problem.is_hard_feasible(route):
            continue
        cost = problem.route_cost(route)
        if cost < best_cost:
            best_route, best_cost = route, cost
    if best_route is None:
        raise RuntimeError("Instance has no hard-feasible route")
    return SolverResult(
        method="Brute force (exact)",
        route=best_route,
        objective=best_cost,
        hard_feasible=True,
        seconds=time.perf_counter() - start,
    )


def solve_dynamic_programming(problem: NavigationProblem) -> SolverResult:
    """Exact optimum of the layered route graph under hard constraints.

    Constraints are enforced structurally rather than by penalty: blocked cells and
    illegal jumps are simply never expanded, and the final layer is pinned to the goal.
    """
    if problem.information_objective != "separable":
        raise ValueError(
            "Dynamic programming is exact only for the separable objective. The d_optimal "
            "objective couples every pair of time steps through the Cauchy-Binet cross "
            "terms, so no single-step state summarizes the past; use solve_brute_force, "
            "simulated annealing or a QAOA solver instead."
        )
    start = time.perf_counter()
    R, T = problem.rows, problem.steps
    inf = float("inf")
    dp = np.full((T, R), inf)
    prev = np.full((T, R), -1, dtype=int)

    dp[0, problem.start_row] = problem.node_cost(problem.start_row, 0)

    for t in range(1, T):
        for r in range(R):
            if (r, t) in problem.blocked_cells:
                continue
            if t == T - 1 and r != problem.goal_row:
                continue
            best, best_prev = inf, -1
            for rp in range(R):
                if dp[t - 1, rp] == inf or abs(r - rp) > problem.max_row_jump:
                    continue
                c = dp[t - 1, rp] + problem.transition_cost(rp, r) + problem.node_cost(r, t)
                if c < best:
                    best, best_prev = c, rp
            dp[t, r] = best
            prev[t, r] = best_prev

    goal = problem.goal_row
    if not np.isfinite(dp[-1, goal]):
        raise RuntimeError("No feasible route: the corridor is blocked under the jump limit")

    route = [goal]
    r = goal
    for t in range(T - 1, 0, -1):
        r = int(prev[t, r])
        route.append(r)
    route.reverse()
    route_t = tuple(route)

    return SolverResult(
        method="Dynamic programming (exact)",
        route=route_t,
        objective=problem.route_cost(route_t),
        hard_feasible=True,
        seconds=time.perf_counter() - start,
    )


def solve_greedy(problem: NavigationProblem) -> SolverResult:
    """Myopic baseline: take the locally best legal cell that still reaches the goal.

    Under the separable objective "locally best" is the cheapest next cell.  Under the
    D-optimal objective it is the cell with the largest marginal determinant gain given
    the information already accumulated -- the standard greedy heuristic for a
    determinant-style sensor-placement objective, and the natural classical competitor
    once the problem stops being a shortest path.  The reachability guard keeps the
    route feasible either way, so the gap to the exact optimum measures myopia alone.
    """
    start = time.perf_counter()
    weights = problem.weights
    scale = problem._fisher_scale()
    route = [problem.start_row]
    accumulated = problem.magnetic_map.prior_fisher + problem.magnetic_map.fisher[problem.start_row, 0]

    for t in range(1, problem.steps - 1):
        rp = route[-1]
        remaining = problem.steps - 1 - t
        candidates = []
        for r in range(problem.rows):
            if abs(r - rp) > problem.max_row_jump or (r, t) in problem.blocked_cells:
                continue
            if abs(problem.goal_row - r) > remaining * problem.max_row_jump:
                continue
            score = problem.transition_cost(rp, r) + problem.node_base_cost(r, t)
            if problem.information_objective == "separable":
                score -= weights.magnetic_information * float(problem.magnetic_map.information[r, t])
            else:
                gain = float(np.linalg.det(accumulated + problem.magnetic_map.fisher[r, t])) - float(
                    np.linalg.det(accumulated)
                )
                score -= weights.magnetic_information * problem.steps * gain / scale
            candidates.append((score, r))
        if not candidates:
            raise RuntimeError("Greedy baseline reached a dead end")
        chosen = min(candidates)[1]
        route.append(chosen)
        accumulated = accumulated + problem.magnetic_map.fisher[chosen, t]

    if problem.steps > 1:
        route.append(problem.goal_row)
    route_t = tuple(route)

    return SolverResult(
        method="Greedy local",
        route=route_t,
        objective=problem.route_cost(route_t),
        hard_feasible=problem.is_hard_feasible(route_t),
        seconds=time.perf_counter() - start,
    )


def solve_simulated_annealing(
    problem: NavigationProblem,
    *,
    sweeps: int = 400,
    restarts: int = 8,
    seed: int = 0,
) -> SolverResult:
    """Single-spin-flip simulated annealing on the *same penalty QUBO* the QAOA sees.

    This is the baseline that matters for a fair reading of the variational results: it
    consumes the identical encoding, including the one-hot penalty, and it is the kind
    of method a quantum heuristic would actually have to beat.
    """
    start = time.perf_counter()
    Q, offset = problem.qubo()
    n = problem.n_qubits
    Q_sym = Q + Q.T - np.diag(np.diag(Q))
    diag = np.diag(Q).copy()
    rng = np.random.default_rng(seed)

    scale = float(np.max(np.abs(Q_sym))) or 1.0
    best_x: np.ndarray | None = None
    best_energy = float("inf")

    for _ in range(restarts):
        x = rng.integers(0, 2, size=n).astype(float)
        energy = float(x @ Q @ x + offset)
        for sweep in range(sweeps):
            temperature = scale * (1.0 - sweep / sweeps) + 1e-9
            for i in rng.permutation(n):
                # Flipping x_i changes the energy by (1 - 2 x_i) (Q_ii + sum_j!=i Q_sym_ij x_j).
                local = diag[i] + float(Q_sym[i] @ x) - Q_sym[i, i] * x[i]
                delta = (1.0 - 2.0 * x[i]) * local
                if delta <= 0.0 or rng.random() < np.exp(-delta / temperature):
                    x[i] = 1.0 - x[i]
                    energy += delta
        energy = float(x @ Q @ x + offset)
        if energy < best_energy:
            best_energy, best_x = energy, x.copy()

    assert best_x is not None
    bits = "".join("1" if v > 0.5 else "0" for v in best_x)
    route = problem.decode_bitstring(bits)
    decoded = route is not None
    if route is None:
        # Report the failure honestly instead of repairing it into a valid-looking route.
        route = tuple([problem.start_row] * problem.steps)

    return SolverResult(
        method="Simulated annealing (same QUBO)",
        route=route,
        objective=problem.route_cost(route),
        hard_feasible=decoded and problem.is_hard_feasible(route),
        seconds=time.perf_counter() - start,
        extras={"qubo_energy": best_energy, "decoded_one_hot": float(decoded)},
    )
