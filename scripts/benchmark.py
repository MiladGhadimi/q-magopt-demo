"""Statistical benchmark: many instances, several depths, both variational objectives.

A single instance proves nothing about an optimizer.  This script sweeps random
instances and QAOA depths, and reports distributions rather than anecdotes:

* approximation ratio against the exact dynamic-programming optimum,
* one-hot survival and *conditional* feasibility, which separates the mixer's effect
  from the effect of simply searching a smaller space,
* how faithfully the separable QUBO information term tracks exact D-optimality.
"""

from __future__ import annotations

import argparse
import csv
import statistics
import sys
from collections import defaultdict
from dataclasses import replace
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from qmagopt import (  # noqa: E402
    QaoaConfig,
    build_demo_problem,
    solve_brute_force,
    solve_dynamic_programming,
    solve_penalty_qaoa,
    solve_simulated_annealing,
    solve_xy_qaoa,
    surrogate_fidelity,
)
from qmagopt.metrics import approximation_ratio  # noqa: E402
from qmagopt.plotting import plot_depth_scan  # noqa: E402
from qmagopt.qaoa import _interp_parameters, _PenaltyTables  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the Q-MagOpt statistical benchmark")
    parser.add_argument("--instances", type=int, default=12, help="Number of random instances")
    parser.add_argument(
        "--encoding-instances",
        type=int,
        default=40,
        help="Instances for the separable-vs-D-optimal comparison (exact solves only, so cheap)",
    )
    parser.add_argument("--p-max", type=int, default=3, help="Highest QAOA depth")
    parser.add_argument("--rows", type=int, default=3)
    parser.add_argument("--cols", type=int, default=5)
    parser.add_argument("--restarts", type=int, default=4)
    parser.add_argument("--maxiter", type=int, default=140)
    parser.add_argument("--quick", action="store_true", help="Small sweep for CI and smoke tests")
    parser.add_argument("--out", type=Path, default=ROOT / "results")
    return parser


def _run_scan(problem, solver: str, objective: str, p_max: int, restarts: int, maxiter: int, seed: int):
    """Depths 1..p_max for one ansatz, each seeded from the previous optimum (INTERP)."""
    tables = _PenaltyTables.build(problem) if solver == "penalty" else None
    previous = None
    out = []
    for p in range(1, p_max + 1):
        config = QaoaConfig(
            p=p,
            restarts=restarts,
            maxiter=maxiter,
            seed=seed + p,
            objective=objective,  # type: ignore[arg-type]
        )
        if solver == "penalty":
            result = solve_penalty_qaoa(problem, config, tables=tables, warm_start=previous)
        else:
            result = solve_xy_qaoa(problem, config, warm_start=previous)
        out.append((p, result))
        if result.parameters is not None:
            previous = _interp_parameters(np.asarray(result.parameters), p)
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    instances = 3 if args.quick else args.instances
    p_max = 2 if args.quick else args.p_max
    restarts = 2 if args.quick else args.restarts
    maxiter = 40 if args.quick else args.maxiter
    encoding_instances = 6 if args.quick else args.encoding_instances

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    fidelities: list[float] = []

    # Headline encoding comparison, run over its own larger sample: it needs only two
    # exact solves per instance, so it is cheap, and a wider sample keeps the reported
    # medians from swinging with the handful of instances where the two objectives
    # happen to pick the same route.
    separable_crlb: list[float] = []
    exact_crlb: list[float] = []
    for index in range(encoding_instances):
        problem = build_demo_problem(seed=100 + index, rows=args.rows, cols=args.cols)
        if not problem.feasible_routes():
            continue
        d_optimal = replace(problem, information_objective="d_optimal")
        separable_crlb.append(problem.route_crlb_m(solve_brute_force(problem).route))
        exact_crlb.append(d_optimal.route_crlb_m(solve_brute_force(d_optimal).route))

    for index in range(instances):
        seed = 100 + index
        problem = build_demo_problem(seed=seed, rows=args.rows, cols=args.cols)
        if not problem.feasible_routes():
            continue

        exact = solve_dynamic_programming(problem)
        annealed = solve_simulated_annealing(problem, sweeps=400, restarts=8, seed=seed)
        fidelities.append(surrogate_fidelity(problem)["spearman_rho"])

        rows.append(
            {
                "instance": seed,
                "solver": "dynamic_programming",
                "objective": "-",
                "p": 0,
                "cost": exact.objective,
                "approximation_ratio": 1.0,
                "one_hot_probability": "",
                "feasible_probability": "",
                "conditional_feasible_probability": "",
                "crlb_m": problem.route_crlb_m(exact.route),
                "seconds": exact.seconds,
            }
        )
        rows.append(
            {
                "instance": seed,
                "solver": "simulated_annealing",
                "objective": "-",
                "p": 0,
                "cost": annealed.objective,
                "approximation_ratio": approximation_ratio(problem, annealed.objective, exact.objective),
                "one_hot_probability": "",
                "feasible_probability": "",
                "conditional_feasible_probability": "",
                "crlb_m": problem.route_crlb_m(annealed.route),
                "seconds": annealed.seconds,
            }
        )

        for solver in ("penalty", "xy"):
            for objective in ("expectation", "cvar"):
                for p, result in _run_scan(
                    problem, solver, objective, p_max, restarts, maxiter, seed=1000 + index
                ):
                    rows.append(
                        {
                            "instance": seed,
                            "solver": solver,
                            "objective": objective,
                            "p": p,
                            "cost": result.objective,
                            "approximation_ratio": approximation_ratio(
                                problem, result.objective, exact.objective
                            ),
                            "one_hot_probability": result.one_hot_probability,
                            "feasible_probability": result.feasible_probability,
                            "conditional_feasible_probability": result.conditional_feasible_probability,
                            "crlb_m": problem.route_crlb_m(result.route),
                            "seconds": result.seconds,
                        }
                    )

        print(f"instance {seed}: done ({index + 1}/{instances})")

    csv_path = out_dir / "benchmark_sweep.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # ---------------------------------------------------------------- aggregation
    grouped: dict[tuple[str, str, int], list[dict]] = defaultdict(list)
    for row in rows:
        grouped[(row["solver"], row["objective"], row["p"])].append(row)

    print("\nQ-MagOpt benchmark sweep")
    print("=" * 96)
    print(f"{instances} instances · {args.rows}x{args.cols} corridors · depths 1..{p_max}")
    print(
        f"Separable surrogate vs exact log-det information: median Spearman rho = "
        f"{statistics.median(fidelities):.3f} "
        f"(min {min(fidelities):.3f}, max {max(fidelities):.3f})"
    )
    pairs = list(zip(exact_crlb, separable_crlb, strict=True))
    never_worse = sum(e <= s + 1e-9 for e, s in pairs)
    strictly_better = sum(e < s - 1e-9 for e, s in pairs)
    print(
        f"Encoding comparison over {len(pairs)} instances, both objectives solved exactly -- position CRLB:"
    )
    print(
        f"  separable surrogate  median {statistics.median(separable_crlb):5.2f} m   "
        f"mean {statistics.fmean(separable_crlb):5.2f} m"
    )
    print(
        f"  exact D-optimal      median {statistics.median(exact_crlb):5.2f} m   "
        f"mean {statistics.fmean(exact_crlb):5.2f} m   "
        f"({1.0 - statistics.fmean(exact_crlb) / statistics.fmean(separable_crlb):+.1%} on the mean)"
    )
    print(
        f"  never worse on {never_worse}/{len(pairs)} instances; "
        f"strictly better on {strictly_better}/{len(pairs)}"
    )
    print("-" * 96)
    print(
        f"{'Solver':<22}{'Objective':<13}{'p':>3}{'ratio (median)':>16}"
        f"{'ratio (mean)':>14}{'P(feas|1-hot)':>16}{'s':>8}"
    )
    print("-" * 96)

    for key in sorted(grouped, key=lambda k: (k[0], k[1], k[2])):
        solver, objective, p = key
        bucket = grouped[key]
        ratios = [float(r["approximation_ratio"]) for r in bucket]
        conditional = [
            float(r["conditional_feasible_probability"])
            for r in bucket
            if r["conditional_feasible_probability"] not in ("", None)
        ]
        seconds = [float(r["seconds"]) for r in bucket if r["seconds"] is not None]
        print(
            f"{solver:<22}{objective:<13}{p:>3}"
            f"{statistics.median(ratios):>16.3f}"
            f"{statistics.fmean(ratios):>14.3f}"
            f"{(statistics.fmean(conditional) if conditional else float('nan')):>16.4f}"
            f"{(statistics.fmean(seconds) if seconds else float('nan')):>8.2f}"
        )
    print("-" * 96)

    # ------------------------------------------------------------------- figures
    def series(solver: str, objective: str, field: str) -> list[tuple[int, float]]:
        out = []
        for p in range(1, p_max + 1):
            bucket = grouped.get((solver, objective, p), [])
            values = [float(r[field]) for r in bucket if r[field] not in ("", None)]
            if values:
                out.append((p, statistics.fmean(values)))
        return out

    feasibility_fig = plot_depth_scan(
        {
            "XY mixer (expectation)": series("xy", "expectation", "conditional_feasible_probability"),
            "XY mixer (CVaR)": series("xy", "cvar", "conditional_feasible_probability"),
            "Penalty / X mixer (expectation)": series(
                "penalty", "expectation", "conditional_feasible_probability"
            ),
            "Penalty / X mixer (CVaR)": series("penalty", "cvar", "conditional_feasible_probability"),
        },
        out_dir / "feasibility_vs_depth.png",
        ylabel="Feasible probability within the one-hot sector",
        title=f"Constraint satisfaction vs circuit depth (mean of {instances} instances)",
    )

    quality_fig = plot_depth_scan(
        {
            "XY mixer (expectation)": series("xy", "expectation", "approximation_ratio"),
            "XY mixer (CVaR)": series("xy", "cvar", "approximation_ratio"),
            "Penalty / X mixer (expectation)": series("penalty", "expectation", "approximation_ratio"),
            "Penalty / X mixer (CVaR)": series("penalty", "cvar", "approximation_ratio"),
        },
        out_dir / "quality_vs_depth.png",
        ylabel="Approximation ratio vs exact optimum",
        title=f"Solution quality vs circuit depth (mean of {instances} instances)",
    )

    print(f"\nWrote {csv_path}")
    print(f"Wrote {feasibility_fig}")
    print(f"Wrote {quality_fig}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
