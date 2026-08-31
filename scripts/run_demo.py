"""Solve the reference instance with every method and write the figures and table."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from qmagopt import (  # noqa: E402
    QaoaConfig,
    build_demo_problem,
    information_summary,
    route_metrics,
    solve_brute_force,
    solve_dynamic_programming,
    solve_greedy,
    solve_penalty_qaoa,
    solve_simulated_annealing,
    solve_xy_qaoa,
    surrogate_fidelity,
)
from qmagopt.plotting import plot_magnetic_field, plot_routes, plot_surrogate_fidelity  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Q-MagOpt GNSS-denied navigation demo")
    parser.add_argument("--seed", type=int, default=7, help="Instance seed")
    parser.add_argument("--p", type=int, default=2, help="QAOA depth")
    parser.add_argument("--rows", type=int, default=3, help="Candidate rows per step")
    parser.add_argument("--cols", type=int, default=5, help="Navigation steps")
    parser.add_argument(
        "--objective",
        choices=["expectation", "cvar"],
        default="cvar",
        help="Variational objective; CVaR targets the good tail rather than the mean",
    )
    parser.add_argument(
        "--information-model",
        choices=["fisher", "heuristic"],
        default="fisher",
        help="Per-cell observability score: Fisher-derived, or the legacy ad-hoc blend",
    )
    parser.add_argument(
        "--information-objective",
        choices=["separable", "d_optimal"],
        default="separable",
        help=(
            "How observability enters the objective: a separable per-cell sum, or the "
            "exact det J criterion (quadratic via Cauchy-Binet; disables the DP baseline)"
        ),
    )
    parser.add_argument("--mixer", choices=["complete", "ring"], default="complete")
    parser.add_argument("--quick", action="store_true", help="Fewer optimizer iterations")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    problem = build_demo_problem(
        seed=args.seed,
        rows=args.rows,
        cols=args.cols,
        information_model=args.information_model,
        information_objective=args.information_objective,
    )

    config = QaoaConfig(
        p=args.p,
        restarts=2 if args.quick else 8,
        maxiter=40 if args.quick else 200,
        seed=21,
        objective=args.objective,
        mixer_graph=args.mixer,
    )

    # The dynamic program is exact only for the separable objective; the D-optimal
    # objective couples every pair of time steps, so enumeration takes over as the
    # exact reference.
    if problem.information_objective == "separable":
        exact = solve_dynamic_programming(problem)
    else:
        exact = solve_brute_force(problem)

    results = [
        exact,
        solve_greedy(problem),
        solve_simulated_annealing(
            problem, sweeps=150 if args.quick else 500, restarts=4 if args.quick else 12
        ),
        solve_penalty_qaoa(problem, config),
        solve_xy_qaoa(problem, config),
    ]

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = [route_metrics(problem, r, optimum=exact.objective) for r in results]
    csv_path = out_dir / "benchmark.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    context = information_summary(problem)
    fidelity = surrogate_fidelity(problem)

    print("\nQ-MagOpt demo")
    print("=" * 104)
    print(
        f"Instance: {problem.rows}x{problem.steps} corridor, seed {args.seed}, "
        f"{problem.n_qubits} qubits, {context['feasible_routes']} hard-feasible routes"
    )
    print(
        f"Information objective: {problem.information_objective}"
        + ("  (exact det J, all-pairs coupled)" if problem.information_objective == "d_optimal" else "")
    )
    print(
        f"Sensor sigma {context['sensor_sigma_nt']:.2f} nT · dead-reckoning prior "
        f"{context['prior_position_sigma_m']:.0f} m · best achievable CRLB "
        f"{context['best_possible_crlb_m']:.1f} m · worst {context['worst_feasible_crlb_m']:.1f} m"
    )
    print(
        f"Surrogate vs exact log-det information: Spearman rho = {fidelity['spearman_rho']:.3f} "
        f"over {fidelity['feasible_routes']} feasible routes"
    )
    print("-" * 104)
    header = (
        f"{'Method':<38}{'Route':<12}{'Cost':>8}{'Ratio':>8}{'CRLB m':>9}{'P(1-hot)':>10}{'P(feas|1h)':>12}"
    )
    print(header)
    print("-" * 104)
    for row in rows:
        ratio = row.get("approximation_ratio")
        one_hot = row["one_hot_probability"]
        conditional = row["conditional_feasible_probability"]
        print(
            f"{row['method']:<38}{row['route']:<12}{row['objective']:>8.3f}"
            f"{(f'{ratio:.3f}' if ratio is not None else '-'):>8}"
            f"{row['position_crlb_m']:>9.1f}"
            f"{(f'{one_hot:.4f}' if one_hot is not None else '-'):>10}"
            f"{(f'{conditional:.4f}' if conditional is not None else '-'):>12}"
        )
    print("-" * 104)

    figures = [
        plot_magnetic_field(problem, out_dir / "magnetic_map.png"),
        plot_routes(problem, results, out_dir / "routes.png"),
        plot_surrogate_fidelity(problem, out_dir / "surrogate_fidelity.png"),
    ]
    print(f"\nWrote {csv_path}")
    for figure in figures:
        print(f"Wrote {figure}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
