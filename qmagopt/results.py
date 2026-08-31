"""Common result container shared by classical and variational solvers."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SolverResult:
    """Outcome of one solver run on one instance.

    The probability fields are only meaningful for the statevector solvers and are
    ``None`` for deterministic classical methods.  They are deliberately kept distinct:

    ``one_hot_probability``
        Total probability on states that decode to *some* route, i.e. the mass that
        survives the one-hot constraint.  It is 1 by construction for the XY solver.
    ``feasible_probability``
        Total probability on states satisfying *every* hard constraint (one-hot,
        start/goal, obstacles and jump limits).
    ``success_probability``
        Probability of the single returned route.
    """

    method: str
    route: tuple[int, ...]
    objective: float
    hard_feasible: bool
    success_probability: float | None = None
    feasible_probability: float | None = None
    one_hot_probability: float | None = None
    parameters: tuple[float, ...] | None = None
    seconds: float | None = None
    extras: dict[str, float] = field(default_factory=dict)

    @property
    def conditional_feasible_probability(self) -> float | None:
        """Feasibility mass *within* the one-hot sector.

        This is the comparison that isolates the effect of the mixer from the effect of
        working in a smaller Hilbert space, and is the fairer of the two numbers when
        contrasting penalty-QAOA with XY-QAOA.
        """
        if self.feasible_probability is None or self.one_hot_probability is None:
            return None
        if self.one_hot_probability <= 0.0:
            return 0.0
        return float(self.feasible_probability / self.one_hot_probability)
