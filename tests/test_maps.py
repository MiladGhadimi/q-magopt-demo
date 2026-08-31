"""Tests for the magnetic map and its Fisher-information model."""

from __future__ import annotations

import numpy as np
import pytest

from qmagopt.maps import (
    MagneticMap,
    SensorModel,
    d_optimality,
    e_optimality,
    make_synthetic_magnetic_map,
    position_crlb_m,
    route_fisher,
)


def test_map_is_reproducible_and_zero_mean():
    a = make_synthetic_magnetic_map(3, 5, seed=7)
    b = make_synthetic_magnetic_map(3, 5, seed=7)
    assert np.array_equal(a.field_nt, b.field_nt)
    # An anomaly map is a residual after the core field is removed.
    assert abs(float(np.mean(a.field_nt))) < 1e-9


def test_information_score_is_normalized():
    m = make_synthetic_magnetic_map(4, 6, seed=3)
    assert m.information.min() >= 0.0
    assert m.information.max() <= 1.0 + 1e-12
    assert np.isclose(m.information.max(), 1.0)


def test_per_cell_fisher_is_rank_one():
    """A single scalar measurement constrains position only along the gradient."""
    m = make_synthetic_magnetic_map(4, 6, seed=11)
    for r in range(m.shape[0]):
        for t in range(m.shape[1]):
            J = m.fisher[r, t]
            assert np.allclose(J, J.T)
            eigenvalues = np.linalg.eigvalsh(J)
            assert eigenvalues[0] > -1e-12  # positive semi-definite
            assert np.linalg.matrix_rank(J, tol=1e-12 * max(1.0, eigenvalues[-1])) <= 1


def test_fisher_matches_closed_form():
    """J must equal grad(B) grad(B)^T / sigma^2 for the stated gradient and sigma."""
    m = make_synthetic_magnetic_map(3, 5, seed=5)
    expected = np.einsum("rci,rcj->rcij", m.gradient_nt_per_m, m.gradient_nt_per_m) / m.sensor.sigma_nt**2
    assert np.allclose(m.fisher, expected)


def test_gradient_orientation_and_units():
    """A field varying only eastward yields an east-only gradient with the right slope."""
    rows, cols, cell = 3, 5, 100.0
    slope = 0.4  # nT per metre
    east = np.arange(cols)[None, :] * cell
    field = np.repeat(slope * east, rows, axis=0)
    m = MagneticMap.from_field(field, np.zeros((rows, cols)), cell_size_m=cell)
    assert np.allclose(m.gradient_nt_per_m[..., 0], slope)
    assert np.allclose(m.gradient_nt_per_m[..., 1], 0.0)


def test_constant_offset_does_not_change_information():
    """Only the gradient carries position information, so a DC shift is invisible."""
    base = make_synthetic_magnetic_map(3, 5, seed=9)
    shifted = MagneticMap.from_field(
        base.field_nt + 12345.0, base.risk, cell_size_m=base.cell_size_m, sensor=base.sensor
    )
    assert np.allclose(base.fisher, shifted.fisher)
    assert np.allclose(base.information, shifted.information)


def test_route_fisher_is_additive_over_measurements():
    m = make_synthetic_magnetic_map(3, 4, seed=2)
    route = (0, 1, 2, 1)
    manual = sum((m.fisher[r, t] for t, r in enumerate(route)), np.zeros((2, 2)))
    assert np.allclose(route_fisher(m, route, include_prior=False), manual)
    assert np.allclose(route_fisher(m, route), manual + m.prior_fisher)


def test_more_measurements_never_reduce_information():
    """Information is monotone: adding a measurement cannot raise the CRLB."""
    m = make_synthetic_magnetic_map(3, 5, seed=4)
    partial = m.prior_fisher.copy()
    previous = position_crlb_m(partial)
    for t, r in enumerate((1, 2, 2, 2, 1)):
        partial = partial + m.fisher[r, t]
        current = position_crlb_m(partial)
        assert current <= previous + 1e-9
        previous = current


def test_prior_only_crlb_matches_prior_sigma():
    m = make_synthetic_magnetic_map(3, 5, seed=1, prior_position_sigma_m=150.0)
    # sqrt(trace(J^-1)) over two independent axes each with sigma 150 m.
    assert position_crlb_m(m.prior_fisher) == pytest.approx(150.0 * np.sqrt(2.0))


def test_singular_information_reports_infinite_error():
    m = make_synthetic_magnetic_map(3, 5, seed=1)
    single = m.fisher[1, 1]  # rank one
    assert position_crlb_m(single) == float("inf")
    assert d_optimality(single) == float("-inf")
    assert e_optimality(single) == pytest.approx(0.0, abs=1e-12)


def test_heuristic_model_differs_from_fisher_model():
    fisher = make_synthetic_magnetic_map(3, 5, seed=7, information_model="fisher")
    heuristic = fisher.with_information_model("heuristic")
    assert heuristic.information_model == "heuristic"
    assert not np.allclose(fisher.information, heuristic.information)
    # The underlying physics is untouched; only the scoring changes.
    assert np.allclose(fisher.fisher, heuristic.fisher)
    assert fisher.with_information_model("fisher") is fisher


def test_sensor_sigma_combines_terms_in_quadrature():
    sensor = SensorModel(noise_nt=3.0, map_error_nt=4.0)
    assert sensor.sigma_nt == pytest.approx(5.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rows": 1, "cols": 5},
        {"rows": 3, "cols": 1},
    ],
)
def test_degenerate_map_sizes_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_synthetic_magnetic_map(**kwargs)


def test_invalid_construction_is_rejected():
    with pytest.raises(ValueError):
        SensorModel(noise_nt=0.0, map_error_nt=0.0)
    with pytest.raises(ValueError):
        MagneticMap.from_field(np.zeros((3, 4)), np.zeros((3, 4)), cell_size_m=0.0)
    with pytest.raises(ValueError):
        MagneticMap.from_field(np.zeros((3, 4)), np.zeros((2, 2)))
    with pytest.raises(ValueError):
        MagneticMap.from_field(np.zeros((3, 4)), np.zeros((3, 4)), information_model="magic")


def test_route_fisher_rejects_bad_routes():
    m = make_synthetic_magnetic_map(3, 5, seed=1)
    with pytest.raises(ValueError):
        route_fisher(m, (0, 1, 2))
    with pytest.raises(ValueError):
        route_fisher(m, (0, 1, 2, 1, 9))
