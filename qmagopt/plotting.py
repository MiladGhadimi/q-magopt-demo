"""Figures for the demonstrator.

Colour choices follow one rule set: a fixed categorical order for methods (never
cycled, always paired with a distinct marker and dash so identity never rests on
colour alone), a single-hue sequential ramp for magnitudes, and a two-hue diverging
ramp with a neutral grey midpoint for the signed anomaly field.  ``results/benchmark.csv``
is the table view for the low-contrast slots.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm

from .problem import NavigationProblem
from .results import SolverResult

# Fixed categorical order; a method keeps its colour wherever it appears.
SERIES_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e34948", "#4a3aa7"]
SERIES_MARKERS = ["o", "s", "^", "D", "P", "v"]
SERIES_DASHES = ["-", "--", "-.", ":", (0, (5, 1, 1, 1)), (0, (3, 1, 3, 1))]

TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"
TEXT_MUTED = "#8a8880"
SURFACE = "#fcfcfb"

#: Single hue, light to dark: magnitude only.
SEQUENTIAL = LinearSegmentedColormap.from_list("qmag_seq", ["#eaf1fb", "#2a78d6", "#123a6b"])
#: Two hues with a neutral grey midpoint: a signed anomaly around zero.
DIVERGING = LinearSegmentedColormap.from_list(
    "qmag_div", ["#123a6b", "#2a78d6", "#d9d8d2", "#eb6834", "#7a2f13"]
)


def _style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(TEXT_MUTED)
        ax.spines[side].set_linewidth(0.8)
    ax.tick_params(colors=TEXT_SECONDARY, labelsize=9, length=3)
    ax.xaxis.label.set_color(TEXT_SECONDARY)
    ax.yaxis.label.set_color(TEXT_SECONDARY)
    ax.title.set_color(TEXT_PRIMARY)


def _save(fig, out: str | Path) -> Path:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    return out


def plot_magnetic_field(problem: NavigationProblem, out: str | Path) -> Path:
    """The anomaly field with its gradient, beside the Fisher information it implies.

    The two panels make the central physical point visible: information concentrates
    where the field *changes*, not where it is large.
    """
    mmap = problem.magnetic_map
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), facecolor=SURFACE)

    # A diverging ramp is only honest when the data actually straddles zero; an anomaly
    # map that happens not to gets the sequential ramp instead.
    limit = float(np.max(np.abs(mmap.field_nt))) or 1.0
    straddles_zero = float(mmap.field_nt.min()) < 0.0 < float(mmap.field_nt.max())
    if straddles_zero:
        im0 = axes[0].imshow(
            mmap.field_nt,
            origin="lower",
            aspect="auto",
            cmap=DIVERGING,
            norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        )
    else:
        im0 = axes[0].imshow(mmap.field_nt, origin="lower", aspect="auto", cmap=SEQUENTIAL)
    fig.colorbar(im0, ax=axes[0], label="Anomaly (nT)")

    grad = mmap.gradient_nt_per_m
    ys, xs = np.mgrid[0 : mmap.shape[0], 0 : mmap.shape[1]]
    axes[0].quiver(xs, ys, grad[..., 0], grad[..., 1], color=TEXT_PRIMARY, alpha=0.65, scale=3.0, width=0.004)
    axes[0].set_title("Magnetic anomaly field and its gradient", fontsize=11)

    im1 = axes[1].imshow(mmap.fisher_trace, origin="lower", aspect="auto", cmap=SEQUENTIAL)
    fig.colorbar(im1, ax=axes[1], label=r"tr $J$  (m$^{-2}$)")
    axes[1].set_title(
        f"Position Fisher information  ($\\sigma$ = {mmap.sensor.sigma_nt:.2f} nT)", fontsize=11
    )

    for ax in axes:
        _style_axes(ax)
        ax.set_xlabel("Navigation step / map column")
        ax.set_ylabel("Candidate row")
        ax.set_xticks(range(mmap.shape[1]))
        ax.set_yticks(range(mmap.shape[0]))

    fig.tight_layout()
    return _save(fig, out)


def plot_routes(problem: NavigationProblem, results: list[SolverResult], out: str | Path) -> Path:
    """Routes drawn over the per-cell information ramp, labelled by position CRLB."""
    fig, ax = plt.subplots(figsize=(9.5, 5.0), facecolor=SURFACE)
    im = ax.imshow(problem.magnetic_map.information, origin="lower", aspect="auto", cmap=SEQUENTIAL)
    fig.colorbar(im, ax=ax, label="Per-cell information score (normalized tr $J$)")

    xs = np.arange(problem.steps)
    # Nudge overlapping routes apart so a shared segment stays readable.
    offsets = np.linspace(-0.09, 0.09, max(len(results), 1))

    for i, result in enumerate(results):
        crlb = problem.route_crlb_m(result.route)
        ax.plot(
            xs,
            np.asarray(result.route, dtype=float) + offsets[i],
            color=SERIES_COLORS[i % len(SERIES_COLORS)],
            marker=SERIES_MARKERS[i % len(SERIES_MARKERS)],
            linestyle=SERIES_DASHES[i % len(SERIES_DASHES)],
            linewidth=2.0,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=f"{result.method}  ·  cost {result.objective:.2f}  ·  CRLB {crlb:.1f} m",
        )

    for r, t in sorted(problem.blocked_cells):
        ax.scatter([t], [r], marker="X", s=190, color=TEXT_PRIMARY, zorder=5)
    if problem.blocked_cells:
        ax.scatter([], [], marker="X", s=120, color=TEXT_PRIMARY, label="Blocked cell")

    _style_axes(ax)
    ax.set_xlabel("Navigation step / map column")
    ax.set_ylabel("Candidate row")
    ax.set_xticks(range(problem.steps))
    ax.set_yticks(range(problem.rows))
    ax.set_title("Information-aware route selection in a GNSS-denied corridor", fontsize=12)
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=1,
        frameon=False,
        fontsize=9,
        labelcolor=TEXT_SECONDARY,
    )
    fig.tight_layout()
    return _save(fig, out)


def plot_depth_scan(
    scans: dict[str, list[tuple[int, float]]], out: str | Path, *, ylabel: str, title: str
) -> Path:
    """Trend of a scalar against QAOA depth, one line per ansatz."""
    fig, ax = plt.subplots(figsize=(7.2, 4.2), facecolor=SURFACE)

    for i, (label, series) in enumerate(scans.items()):
        depths = [d for d, _ in series]
        values = [v for _, v in series]
        color = SERIES_COLORS[i % len(SERIES_COLORS)]
        ax.plot(
            depths,
            values,
            color=color,
            marker=SERIES_MARKERS[i % len(SERIES_MARKERS)],
            linestyle=SERIES_DASHES[i % len(SERIES_DASHES)],
            linewidth=2.0,
            markersize=8,
            markeredgecolor=SURFACE,
            markeredgewidth=1.4,
            label=label,
        )
        # Direct-label the endpoint so identity survives without reading the legend.
        ax.annotate(
            f"{values[-1]:.3f}",
            xy=(depths[-1], values[-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=TEXT_SECONDARY,
            fontsize=9,
            va="center",
        )

    _style_axes(ax)
    ax.grid(axis="y", color=TEXT_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("QAOA depth $p$")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=12)
    ax.set_xticks(sorted({d for series in scans.values() for d, _ in series}))
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    return _save(fig, out)


def plot_surrogate_fidelity(problem: NavigationProblem, out: str | Path) -> Path:
    """Separable QUBO information term against the exact D-optimality of the same route.

    A tight monotone cloud means the quadratic surrogate the optimizers see is a fair
    stand-in for the criterion a navigator cares about; scatter is the cost of forcing
    a non-separable objective into a QUBO.
    """
    feasible = problem.feasible_routes()
    separable = np.array(
        [
            sum(float(problem.magnetic_map.information[r, t]) for t, r in enumerate(route))
            for route in feasible
        ]
    )
    exact = np.array([problem.route_log_det_information(route) for route in feasible])

    fig, ax = plt.subplots(figsize=(6.4, 4.6), facecolor=SURFACE)
    ax.scatter(
        separable,
        exact,
        s=70,
        color=SERIES_COLORS[0],
        edgecolor=SURFACE,
        linewidth=1.4,
        zorder=3,
        label="Hard-feasible route",
    )

    if len(feasible) >= 2 and np.std(separable) > 1e-12:
        slope, intercept = np.polyfit(separable, exact, 1)
        line_x = np.linspace(separable.min(), separable.max(), 2)
        ax.plot(
            line_x,
            slope * line_x + intercept,
            color=TEXT_MUTED,
            linewidth=1.5,
            linestyle="--",
            zorder=2,
            label="Least-squares trend",
        )

    _style_axes(ax)
    ax.grid(color=TEXT_MUTED, alpha=0.18, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.set_xlabel("Separable surrogate  $\\sum_t I(r_t, t)$   (what the QUBO sees)")
    ax.set_ylabel("Exact  $\\log\\det J$   (what navigation needs)")
    ax.set_title("Does the quadratic surrogate track real observability?", fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=TEXT_SECONDARY, loc="best")
    fig.tight_layout()
    return _save(fig, out)
