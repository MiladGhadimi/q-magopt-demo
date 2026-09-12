"""Self-contained reference implementation of the Q-MagOpt interface this study uses.

Why this exists
---------------
`integrated_benchmark.py` depends on the private `qmagopt` package, so the
experiment bundle could not be run or audited by anyone without that checkout.
This module reimplements exactly the four `NavigationProblem` members the study
calls -- `feasible_routes`, `route_fisher`, `information_qubo_terms` and
`route_to_bitstring` -- over the corridor lattice, so the benchmark runs
standalone.

Status of this reimplementation
-------------------------------
It is a reference, not the integration under test. It was validated against the
recorded outputs of the original `qmagopt` run (commit 13b6d30a, results in
`integrated_results_qmagopt_v1/trials.csv`): it reproduces all 577 feasible
candidates per map, all 90 selected routes, all route lengths, and all 90 route
determinants to within 1e-9 relative. `test_integration.py` re-checks that
agreement, and `integrated_benchmark.py --planner qmagopt` cross-asserts the two
planners on every map when `qmagopt` is importable.

The prior
---------
`route_fisher` returns PRIOR + sum of per-waypoint Fisher matrices, where the
prior is isotropic with standard deviation PRIOR_SIGMA_M. The value 200 m was
recovered from the original run's recorded determinants (identical to 14
significant figures across all 90 cases); it is a property of the Q-MagOpt
objective, not a choice made here. The D-optimal objective is therefore the
determinant of the POSTERIOR information matrix, det(J_prior + J_route), not
det(J_route). See PROTOCOL.md.
"""
from __future__ import annotations

import itertools
from typing import Iterator, Tuple

import numpy as np

ROWS = 3
COLUMNS = 9
START_ROW = 1
END_ROW = 1
MAX_ROW_JUMP = 1

PRIOR_SIGMA_M = 200.0
PRIOR_INFORMATION = np.eye(2) / PRIOR_SIGMA_M ** 2

Route = Tuple[int, ...]


class ReferenceProblem:
    """Route-selection problem over a ROWS x COLUMNS corridor lattice.

    Parameters
    ----------
    fisher : (ROWS, COLUMNS, 2, 2) array
        Per-waypoint Fisher information, already divided by the sensor variance.
    """

    def __init__(self, fisher: np.ndarray, prior: np.ndarray = PRIOR_INFORMATION):
        fisher = np.asarray(fisher, dtype=float)
        if fisher.shape != (ROWS, COLUMNS, 2, 2):
            raise ValueError(f"fisher must be {(ROWS, COLUMNS, 2, 2)}, got {fisher.shape}")
        self.fisher = fisher
        self.prior = np.asarray(prior, dtype=float)

    def feasible_routes(self) -> Iterator[Route]:
        """Every row assignment with fixed endpoints and at most one row of lateral
        movement per column step. Yields 577 routes for a 3x9 lattice."""
        interior = COLUMNS - 2
        for middle in itertools.product(range(ROWS), repeat=interior):
            route = (START_ROW,) + middle + (END_ROW,)
            if np.max(np.abs(np.diff(route))) <= MAX_ROW_JUMP:
                yield route

    def route_fisher(self, route: Route) -> np.ndarray:
        """Posterior information matrix for a route: prior + accumulated waypoint Fisher."""
        total = self.prior.copy()
        for column, row in enumerate(route):
            total = total + self.fisher[row, column]
        return total

    def route_to_bitstring(self, route: Route) -> str:
        """One-hot encoding over (row, column), row-major, matching `fisher.reshape(-1, 2, 2)`."""
        bits = np.zeros((ROWS, COLUMNS), dtype=int)
        for column, row in enumerate(route):
            bits[row, column] = 1
        return "".join(str(b) for b in bits.ravel())

    def information_qubo_terms(self) -> Tuple[np.ndarray, np.ndarray, float]:
        """Exact quadratic encoding of det(route_fisher) in the one-hot variables.

        With J = P + sum_i x_i F_i and a = F[0,0], b = F[1,1], c = F[0,1]:

            det J = (P00 + a.x)(P11 + b.x) - (P01 + c.x)^2
                  = const + lin.x + x.Q.x

        where const = det P, lin = P00*b + P11*a - 2*P01*c and
        Q = outer(a, b) - outer(c, c). The diagonal of Q is exact for binary x
        because x_i^2 = x_i.
        """
        flat = self.fisher.reshape(-1, 2, 2)
        a, b, c = flat[:, 0, 0], flat[:, 1, 1], flat[:, 0, 1]
        p = self.prior
        const = float(p[0, 0] * p[1, 1] - p[0, 1] ** 2)
        linear = p[0, 0] * b + p[1, 1] * a - 2.0 * p[0, 1] * c
        quadratic = np.outer(a, b) - np.outer(c, c)
        return linear, quadratic, const
