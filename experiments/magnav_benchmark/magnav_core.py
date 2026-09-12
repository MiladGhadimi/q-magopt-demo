"""Q-MagNav numerical core, MIT (c) 2026 Milad Ghadimi.
Extracted from app.py blob 5ab5d1b6008ccad8107ce5fc554219dd251654e3.
Source: https://github.com/MiladGhadimi/q-magnav-demo

Changes from the source blob, all of them structural:
  - UI removed.
  - Optional known GNSS cutoff added (`gnss_cutoff`).
  - Optional GNSS spoofing switch added (`gps_spoof_enabled`), replacing the
    previous practice of pushing `gps_spoof_start_pct` past the horizon.
  - The trajectory is now an explicit argument to `simulate_sensors`. The source
    blob called a `make_trajectory` helper that was NOT part of this extraction;
    calling it here would have raised NameError, and the previous version of this
    file worked only because the benchmark assigned `nav.make_trajectory` at
    runtime. Callers now pass the trajectory directly.
  - `simulate_sensors` and `run_particle_filter` accept an explicit
    `numpy.random.Generator`. Without one they fall back to the source blob's
    `cfg.seed + 100` / `cfg.seed + 500` derivation, which is preserved only for
    backwards compatibility; it collides across trials and should not be used
    for studies (see PROTOCOL.md, "Randomization").

The sensor, odometry, magnetic and filter equations are unchanged.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Optional, Tuple
import numpy as np

@dataclass
class DemoConfig:
    seed: int = 7
    n_steps: int = 220
    n_particles: int = 1500
    map_size_m: float = 10_000.0
    grid_n: int = 140

    gps_noise_m: float = 18.0
    gps_spoof_enabled: bool = True
    gps_spoof_start_pct: float = 0.48
    gps_spoof_growth_m_per_step: float = 11.0
    gps_spoof_side_offset_m: float = 350.0

    odo_noise_m: float = 5.0
    odo_bias_m_per_step: float = 0.9
    process_noise_m: float = 10.0

    mag_noise_nt: float = 2.0
    classical_mag_noise_nt: float = 8.0
    quantum_mag_noise_nt: float = 2.0
    magnetic_bias_drift_nt_per_step: float = 0.015

    gps_residual_threshold_m: float = 100.0
    spoof_confirm_steps: int = 3

def make_magnetic_map(cfg: DemoConfig) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create a synthetic magnetic anomaly map B(x,y) in nanoTesla.

    The map is a smooth background plus localized positive/negative anomalies.
    This represents a simplified 2D magnetic anomaly field.
    """
    rng = np.random.default_rng(cfg.seed)
    xs = np.linspace(0.0, cfg.map_size_m, cfg.grid_n)
    ys = np.linspace(0.0, cfg.map_size_m, cfg.grid_n)
    X, Y = np.meshgrid(xs, ys)

    # Base field around Earth magnetic-field magnitude, with a weak regional trend.
    B = 50_000.0 + 0.0015 * X - 0.0010 * Y

    # Fixed anomalies for reproducibility.
    anomalies = [
        (1800, 2500, +90, 900, 650),
        (3100, 7200, -75, 800, 950),
        (5400, 4200, +110, 700, 700),
        (7600, 6500, -95, 1000, 750),
        (8300, 2300, +65, 650, 850),
        (4700, 8200, +55, 950, 550),
        (6700, 3500, -45, 600, 600),
    ]

    for cx, cy, amp, sx, sy in anomalies:
        B += amp * np.exp(-(((X - cx) ** 2) / (2 * sx ** 2) + ((Y - cy) ** 2) / (2 * sy ** 2)))

    # Add a small smooth-ish texture.
    B += 7.0 * np.sin(X / 1100.0) * np.cos(Y / 1500.0)
    B += 3.0 * np.sin((X + Y) / 900.0)

    return xs, ys, B

def bilinear_interpolate(xs: np.ndarray, ys: np.ndarray, Z: np.ndarray, points: np.ndarray) -> np.ndarray:
    """
    Bilinear interpolation for points shaped (N,2), with columns x,y.
    Returns Z(x,y).
    """
    x = np.clip(points[:, 0], xs[0], xs[-1])
    y = np.clip(points[:, 1], ys[0], ys[-1])

    ix = np.searchsorted(xs, x) - 1
    iy = np.searchsorted(ys, y) - 1
    ix = np.clip(ix, 0, len(xs) - 2)
    iy = np.clip(iy, 0, len(ys) - 2)

    x0, x1 = xs[ix], xs[ix + 1]
    y0, y1 = ys[iy], ys[iy + 1]

    z00 = Z[iy, ix]
    z10 = Z[iy, ix + 1]
    z01 = Z[iy + 1, ix]
    z11 = Z[iy + 1, ix + 1]

    wx = (x - x0) / np.maximum(x1 - x0, 1e-9)
    wy = (y - y0) / np.maximum(y1 - y0, 1e-9)

    return (
        (1 - wx) * (1 - wy) * z00
        + wx * (1 - wy) * z10
        + (1 - wx) * wy * z01
        + wx * wy * z11
    )

def simulate_sensors(
    cfg: DemoConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    Bmap: np.ndarray,
    mag_noise_nt: float,
    trajectory: np.ndarray,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Simulate ground truth, GNSS, dead reckoning, odometry, and magnetic measurement.

    `trajectory` is the (n_steps, 2) ground-truth path, supplied by the caller.
    `rng` should be an independent generator per trial; omitting it reproduces
    the source blob's `cfg.seed + 100` derivation and is not collision-free.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed + 100)
    truth = np.asarray(trajectory, dtype=float)
    if truth.ndim != 2 or truth.shape[1] != 2:
        raise ValueError(f"trajectory must have shape (n, 2), got {truth.shape}")
    if len(truth) != cfg.n_steps:
        raise ValueError(f"cfg.n_steps={cfg.n_steps} does not match len(trajectory)={len(truth)}")
    n = cfg.n_steps

    true_B = bilinear_interpolate(xs, ys, Bmap, truth)

    drift = np.cumsum(rng.normal(0.0, cfg.magnetic_bias_drift_nt_per_step, size=n))
    mag_meas = true_B + drift + rng.normal(0.0, mag_noise_nt, size=n)

    # GPS / GNSS: reliable first, then spoofed with a growing false offset.
    gps = truth + rng.normal(0.0, cfg.gps_noise_m, size=(n, 2))
    spoof_start = int(cfg.gps_spoof_start_pct * n) if cfg.gps_spoof_enabled else n

    for k in range(spoof_start, n):
        g = k - spoof_start
        gps[k, 0] += cfg.gps_spoof_growth_m_per_step * g
        gps[k, 1] += cfg.gps_spoof_side_offset_m * np.sin(g / 19.0)

    # Odometry increments: approximate local motion with noise and small bias.
    true_delta = np.diff(truth, axis=0, prepend=truth[:1])
    heading_bias = np.deg2rad(4.0)
    R = np.array(
        [
            [np.cos(heading_bias), -np.sin(heading_bias)],
            [np.sin(heading_bias), np.cos(heading_bias)],
        ]
    )
    odo_delta = true_delta @ R.T
    odo_delta += rng.normal(0.0, cfg.odo_noise_m, size=(n, 2))
    odo_delta += np.array([cfg.odo_bias_m_per_step, -0.35 * cfg.odo_bias_m_per_step])

    # Dead reckoning starts at the first GPS point, then integrates odometry.
    dead = np.zeros_like(truth)
    dead[0] = gps[0]
    for k in range(1, n):
        dead[k] = dead[k - 1] + odo_delta[k]

    return {
        "truth": truth,
        "gps": gps,
        "dead": dead,
        "odo_delta": odo_delta,
        "true_B": true_B,
        "mag_meas": mag_meas,
        "spoof_start": np.array([spoof_start]),
    }

def systematic_resample(weights: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    n = len(weights)
    positions = (rng.random() + np.arange(n)) / n
    indexes = np.zeros(n, dtype=np.int64)

    cumulative_sum = np.cumsum(weights)
    i, j = 0, 0
    while i < n:
        if positions[i] < cumulative_sum[j]:
            indexes[i] = j
            i += 1
        else:
            j += 1
    return indexes

def run_particle_filter(
    cfg: DemoConfig,
    xs: np.ndarray,
    ys: np.ndarray,
    Bmap: np.ndarray,
    data: Dict[str, np.ndarray],
    mag_sigma: float,
    gnss_cutoff: int | None = None,
    rng: Optional[np.random.Generator] = None,
) -> Dict[str, np.ndarray]:
    """
    Particle-filter localization.

    State: 2D position.
    Prediction: odometry increments + process noise.
    Measurement: magnetic anomaly likelihood.
    GNSS is used only while it is consistent with the fused estimate.

    `rng` should be an independent generator per trial; omitting it reproduces
    the source blob's `cfg.seed + 500` derivation and is not collision-free.
    """
    if rng is None:
        rng = np.random.default_rng(cfg.seed + 500)

    n = cfg.n_steps
    N = cfg.n_particles
    map_size = cfg.map_size_m

    gps = data["gps"]
    odo = data["odo_delta"]
    mag = data["mag_meas"]

    particles = gps[0] + rng.normal(0.0, 70.0, size=(N, 2))
    weights = np.ones(N) / N

    estimates = np.zeros((n, 2))
    uncertainty = np.zeros(n)
    gps_residual = np.zeros(n)
    gps_trusted = np.ones(n, dtype=bool)
    ooda_state = np.empty(n, dtype=object)

    suspicious_counter = 0
    trust_gps = True

    for k in range(n):
        if k > 0:
            particles += odo[k] + rng.normal(0.0, cfg.process_noise_m, size=(N, 2))
            particles[:, 0] = np.clip(particles[:, 0], 0.0, map_size)
            particles[:, 1] = np.clip(particles[:, 1], 0.0, map_size)

        # Magnetic likelihood
        predicted_B = bilinear_interpolate(xs, ys, Bmap, particles)
        mag_err = predicted_B - mag[k]
        mag_likelihood = np.exp(-0.5 * (mag_err / max(mag_sigma, 1e-6)) ** 2)

        weights *= mag_likelihood + 1e-300
        weights_sum = np.sum(weights)
        if not np.isfinite(weights_sum) or weights_sum <= 0:
            weights = np.ones(N) / N
        else:
            weights /= weights_sum

        # Preliminary estimate before deciding GPS trust at this step.
        preliminary = np.average(particles, weights=weights, axis=0)
        gps_residual[k] = float(np.linalg.norm(gps[k] - preliminary))

        # Simple GNSS spoofing detector:
        # if GNSS disagrees strongly with the magnetic/odometry fusion for multiple steps,
        # stop using GNSS as a trusted measurement.
        if gps_residual[k] > cfg.gps_residual_threshold_m:
            suspicious_counter += 1
        else:
            suspicious_counter = max(0, suspicious_counter - 1)

        if suspicious_counter >= cfg.spoof_confirm_steps:
            trust_gps = False

        gps_trusted[k] = trust_gps

        if gnss_cutoff is not None and k >= gnss_cutoff:
            trust_gps = False
            gps_trusted[k] = False

        # If still trusted, apply GPS likelihood too.
        if trust_gps:
            gps_err = np.linalg.norm(particles - gps[k], axis=1)
            gps_likelihood = np.exp(-0.5 * (gps_err / max(cfg.gps_noise_m, 1e-6)) ** 2)
            weights *= gps_likelihood + 1e-300
            weights /= np.sum(weights)

        # Estimate and uncertainty.
        estimates[k] = np.average(particles, weights=weights, axis=0)
        diffs = particles - estimates[k]
        uncertainty[k] = float(np.sqrt(np.average(np.sum(diffs ** 2, axis=1), weights=weights)))

        # Resample when effective sample size is low.
        ess = 1.0 / np.sum(weights ** 2)
        if ess < 0.55 * N:
            idx = systematic_resample(weights, rng)
            particles = particles[idx]
            weights = np.ones(N) / N

        if trust_gps:
            ooda_state[k] = "GNSS trusted: fuse GPS + odometry + magnetic map"
        else:
            ooda_state[k] = "GNSS denied/spoofed: switch to magnetic-map-aided navigation"

    return {
        "estimate": estimates,
        "uncertainty": uncertainty,
        "gps_residual": gps_residual,
        "gps_trusted": gps_trusted,
        "ooda_state": ooda_state,
    }
