"""Synthetic magnetic anomaly map and a Fisher-information observability model.

The central quantity here is not a hand-tuned "interestingness" score but the Fisher
information a scalar total-field magnetometer carries about horizontal position.

Measurement model
-----------------
A total-field magnetometer flying over a mapped anomaly field ``B(p)`` returns

    z = B(p) + n,          n ~ N(0, sigma^2),

where ``sigma`` combines sensor noise and map representation error (in practice the
second term dominates, even for a quantum magnetometer).  The log-likelihood of a
position hypothesis ``p`` is ``-(z - B(p))^2 / (2 sigma^2)``, whose score is
``(z - B(p)) grad B(p) / sigma^2``.  The Fisher information matrix for position is
therefore the outer product

    J(p) = grad B(p) grad B(p)^T / sigma^2         [units: 1/m^2]

Two consequences drive the whole project:

1. ``J(p)`` is **rank one**.  A single scalar measurement constrains position only
   along the local field-gradient direction; the perpendicular direction is
   unobservable from that measurement alone.
2. Information is therefore a property of a *trajectory*, not of a cell.  Position
   becomes observable only when measurements are accumulated across cells whose
   gradient directions differ.  That non-separability is exactly what makes route
   selection a real optimization problem rather than a per-cell lookup.

The demonstrator exposes both views: the exact route-level criterion
(``route_fisher`` and its D-optimality / CRLB summaries) and a separable per-cell
surrogate (``MagneticMap.information``) that stays quadratic and is therefore
mappable to a QUBO and to quantum hardware.  ``scripts/benchmark.py`` measures how
well the surrogate tracks the exact criterion.

Caveat: the CRLB is a *local* bound.  It quantifies how sharply the likelihood peaks
around the true position, not whether a distant part of the map produces a similar
field value.  Global ambiguity (a genuine failure mode in magnetic navigation) is not
captured by any of the scores here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

InformationModel = Literal["fisher", "heuristic"]

#: Horizontal position states estimated from the map (east, north).
POSITION_DIM = 2


@dataclass(frozen=True)
class SensorModel:
    """Error budget of a scalar total-field magnetometer used for map matching.

    Attributes
    ----------
    noise_nt:
        RMS noise of the magnetometer over one measurement interval, in nT.  The
        default is representative of an optically pumped / quantum scalar sensor,
        which is close to negligible next to the map error.
    map_error_nt:
        RMS disagreement between the stored anomaly map and the true field, in nT.
        This is the dominant term in real magnetic-navigation error budgets.
    """

    noise_nt: float = 0.01
    map_error_nt: float = 2.0

    def __post_init__(self) -> None:
        if self.noise_nt < 0.0 or self.map_error_nt < 0.0:
            raise ValueError("Sensor error terms must be non-negative")
        if self.sigma_nt <= 0.0:
            raise ValueError("Total measurement sigma must be positive")

    @property
    def sigma_nt(self) -> float:
        """Total 1-sigma measurement uncertainty in nT (noise and map error in quadrature)."""
        return float(np.hypot(self.noise_nt, self.map_error_nt))


@dataclass(frozen=True)
class MagneticMap:
    """A magnetic anomaly map with its derived position-observability structure.

    Arrays are indexed ``[row, col]``.  Rows increase northward, columns eastward,
    and both are spaced by ``cell_size_m``.
    """

    field_nt: np.ndarray
    risk: np.ndarray
    cell_size_m: float
    sensor: SensorModel
    prior_position_sigma_m: float
    information_model: InformationModel
    #: Field gradient in nT/m, shape ``(rows, cols, 2)`` ordered (d/d_east, d/d_north).
    gradient_nt_per_m: np.ndarray
    #: Per-measurement Fisher information, shape ``(rows, cols, 2, 2)``, units 1/m^2.
    fisher: np.ndarray
    #: Dimensionless per-cell observability score in [0, 1] used by the QUBO objective.
    information: np.ndarray
    _heuristic_information: np.ndarray = field(repr=False)

    @property
    def shape(self) -> tuple[int, int]:
        return self.field_nt.shape  # type: ignore[return-value]

    @property
    def fisher_trace(self) -> np.ndarray:
        """Trace of the per-cell Fisher matrix, ``|grad B|^2 / sigma^2``, in 1/m^2."""
        return np.trace(self.fisher, axis1=2, axis2=3)

    @property
    def prior_fisher(self) -> np.ndarray:
        """Information already held before the leg starts (INS / dead-reckoning prior)."""
        return np.eye(POSITION_DIM) / (self.prior_position_sigma_m**2)

    @classmethod
    def from_field(
        cls,
        field_nt: np.ndarray,
        risk: np.ndarray,
        *,
        cell_size_m: float = 250.0,
        sensor: SensorModel | None = None,
        prior_position_sigma_m: float = 200.0,
        information_model: InformationModel = "fisher",
    ) -> MagneticMap:
        """Build a map and derive its gradient, Fisher information and cell scores."""
        field_nt = np.asarray(field_nt, dtype=float)
        risk = np.asarray(risk, dtype=float)
        if field_nt.ndim != 2:
            raise ValueError("field_nt must be a 2-D array")
        if risk.shape != field_nt.shape:
            raise ValueError("risk must have the same shape as field_nt")
        if cell_size_m <= 0.0:
            raise ValueError("cell_size_m must be positive")
        if prior_position_sigma_m <= 0.0:
            raise ValueError("prior_position_sigma_m must be positive")
        if information_model not in ("fisher", "heuristic"):
            raise ValueError(f"Unknown information model: {information_model!r}")

        sensor = sensor or SensorModel()

        # np.gradient returns derivatives along each axis; axis 0 is north, axis 1 is east.
        d_north, d_east = np.gradient(field_nt, cell_size_m, cell_size_m)
        gradient = np.stack([d_east, d_north], axis=-1)

        # J = g g^T / sigma^2, evaluated per cell.
        fisher = np.einsum("rci,rcj->rcij", gradient, gradient) / sensor.sigma_nt**2

        fisher_trace = np.trace(fisher, axis1=2, axis2=3)
        fisher_score = _normalize(fisher_trace)
        heuristic_score = _heuristic_information(field_nt)

        information = fisher_score if information_model == "fisher" else heuristic_score

        return cls(
            field_nt=field_nt,
            risk=risk,
            cell_size_m=cell_size_m,
            sensor=sensor,
            prior_position_sigma_m=prior_position_sigma_m,
            information_model=information_model,
            gradient_nt_per_m=gradient,
            fisher=fisher,
            information=information,
            _heuristic_information=heuristic_score,
        )

    def with_information_model(self, model: InformationModel) -> MagneticMap:
        """Return the same map scored by a different per-cell information model."""
        if model == self.information_model:
            return self
        return MagneticMap.from_field(
            self.field_nt,
            self.risk,
            cell_size_m=self.cell_size_m,
            sensor=self.sensor,
            prior_position_sigma_m=self.prior_position_sigma_m,
            information_model=model,
        )


def _normalize(a: np.ndarray) -> np.ndarray:
    """Scale an array into [0, 1]; constant arrays map to zeros."""
    a = np.asarray(a, dtype=float)
    lo, hi = float(np.min(a)), float(np.max(a))
    if hi - lo < 1e-12:
        return np.zeros_like(a)
    return (a - lo) / (hi - lo)


def _heuristic_information(field_nt: np.ndarray) -> np.ndarray:
    """The original ad-hoc score, retained only as an ablation baseline.

    It blends normalized gradient magnitude with distance of the field value from the
    map median.  The gradient half turns out to be a crude, unit-free stand-in for the
    Fisher trace; the "rarity" half has no estimation-theoretic justification.
    """
    d_north, d_east = np.gradient(field_nt)
    gradient = np.hypot(d_east, d_north)
    rarity = np.abs(field_nt - np.median(field_nt))
    return 0.68 * _normalize(gradient) + 0.32 * _normalize(rarity)


def route_fisher(
    magnetic_map: MagneticMap,
    route: tuple[int, ...] | list[int],
    *,
    include_prior: bool = True,
) -> np.ndarray:
    """Accumulated position Fisher information along a route.

    ``route[t]`` is the row occupied at navigation step (column) ``t``.  Information
    from independent measurements adds, so the route matrix is the sum of the per-cell
    rank-one matrices, optionally seeded with the dead-reckoning prior.
    """
    rows, cols = magnetic_map.shape
    route = tuple(int(r) for r in route)
    if len(route) != cols:
        raise ValueError(f"Route must have {cols} entries, got {len(route)}")
    if any(not 0 <= r < rows for r in route):
        raise ValueError("Route contains an out-of-range row")

    total = magnetic_map.prior_fisher.copy() if include_prior else np.zeros((POSITION_DIM, POSITION_DIM))
    for t, r in enumerate(route):
        total = total + magnetic_map.fisher[r, t]
    return total


def position_crlb_m(fisher_matrix: np.ndarray) -> float:
    """Cramer-Rao lower bound on horizontal position RMS error, in metres.

    Returns ``sqrt(trace(J^-1))``.  A singular ``J`` means at least one direction is
    unobservable, which is reported as infinite error rather than a numerical artefact.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)
    eigenvalues = np.linalg.eigvalsh(fisher_matrix)
    if np.any(eigenvalues <= 1e-15 * max(1.0, float(np.max(eigenvalues)))):
        return float("inf")
    return float(np.sqrt(np.sum(1.0 / eigenvalues)))


def d_optimality(fisher_matrix: np.ndarray) -> float:
    """D-optimality criterion ``log det J``: the exact, non-separable information score.

    Maximizing it minimizes the volume of the position uncertainty ellipse.  It is not
    a sum of per-cell terms, so it cannot be written as a QUBO -- which is why the
    optimizers use the separable surrogate and this serves as the reference metric.
    """
    fisher_matrix = np.asarray(fisher_matrix, dtype=float)
    sign, logdet = np.linalg.slogdet(fisher_matrix)
    return float(logdet) if sign > 0 else float("-inf")


def e_optimality(fisher_matrix: np.ndarray) -> float:
    """E-optimality criterion: information along the *worst-observed* direction, 1/m^2."""
    return float(np.min(np.linalg.eigvalsh(np.asarray(fisher_matrix, dtype=float))))


def make_synthetic_magnetic_map(
    rows: int = 3,
    cols: int = 5,
    seed: int = 7,
    *,
    cell_size_m: float = 250.0,
    sensor: SensorModel | None = None,
    prior_position_sigma_m: float = 200.0,
    information_model: InformationModel = "fisher",
    anomaly_count: int = 3,
) -> MagneticMap:
    """Create a reproducible synthetic magnetic-anomaly map with plausible magnitudes.

    The field superposes smooth regional structure with a few localized dipole-like
    anomalies, at amplitudes (tens to ~150 nT over kilometre scales) typical of crustal
    anomaly maps used for magnetic navigation.  It is a stand-in for a real survey
    product, not a geophysical model.
    """
    if rows < 2 or cols < 2:
        raise ValueError("The map needs at least 2 rows and 2 columns")

    rng = np.random.default_rng(seed)
    north, east = np.mgrid[0:rows, 0:cols].astype(float)

    field = 35.0 * np.sin(0.9 * east) + 22.0 * np.cos(1.3 * north + 0.25 * east)

    for _ in range(anomaly_count):
        amplitude = rng.uniform(60.0, 150.0) * rng.choice([-1.0, 1.0])
        centre_east = rng.uniform(0.0, cols - 1)
        centre_north = rng.uniform(0.0, rows - 1)
        width = rng.uniform(0.7, 1.6)
        field = field + amplitude * np.exp(
            -((east - centre_east) ** 2 + (north - centre_north) ** 2) / (2.0 * width**2)
        )

    # Small-scale roughness: real anomaly maps are not band-limited to the survey grid.
    field = field + rng.normal(0.0, 3.0, size=(rows, cols))

    # An anomaly map is a residual after the core field is removed, so it is zero-mean by
    # construction.  A constant offset leaves the gradient -- and therefore every
    # information quantity here -- unchanged; this only fixes the physical interpretation.
    field = field - float(np.mean(field))

    # Operational risk (threat exposure, terrain, airspace), deliberately uncorrelated
    # with magnetic content so the objective has a genuine trade-off to resolve.
    risk = _normalize(rng.normal(0.0, 1.0, size=(rows, cols)))
    risk = 0.15 + 0.7 * risk

    return MagneticMap.from_field(
        field,
        risk,
        cell_size_m=cell_size_m,
        sensor=sensor,
        prior_position_sigma_m=prior_position_sigma_m,
        information_model=information_model,
    )
