"""Run the frozen PROTOCOL.md study.

    python integrated_benchmark.py                     # standalone reference planner
    python integrated_benchmark.py --planner qmagopt   # the Q-MagOpt integration

With --planner qmagopt the reference planner is run alongside and the two are
asserted to agree on every map, so the standalone path stays honest.
"""
from pathlib import Path
from dataclasses import replace
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import argparse
import json
import time
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
from scipy import stats
import magnav_core as nav
import reference_planner as ref

OUT = Path(__file__).parent / 'integrated_results'
METHODS = ['shortest', 'separable', 'd_optimal']

# Randomization. Every trial draws from an independent SeedSequence keyed on
# (map, repeat, noise level). The key deliberately excludes `method`, so the
# three routing methods remain paired on identical sensor and filter noise.
# Including the noise level is what makes the 2 nT and 8 nT arms independent
# replications rather than one draw at two scalings. Sensor and filter streams
# are spawned children, so no two trials can share a stream.
ENTROPY = 20260911

# Planner sensor budget: 2 nT sensor RMS + 2 nT map error -> variance 8 nT^2.
PLANNER_SENSOR_VARIANCE_NT2 = 8.0
MAP_ERROR_NT = 2.0
DISTANCE_BUDGET_M = 8800.0


def trial_generators(map_seed, repeat, noise_nt):
    """Independent (sensor, filter) generators for one paired trial."""
    key = [ENTROPY, int(map_seed), int(repeat), int(round(float(noise_nt) * 1000))]
    sensor_seq, filter_seq = np.random.SeedSequence(key).spawn(2)
    return np.random.default_rng(sensor_seq), np.random.default_rng(filter_seq)


def field_map(seed):
    """Stored map B and measurement-generating map B+error. The error realization
    is fixed per map, which is why maps are the bootstrap cluster unit."""
    rng = np.random.default_rng(seed)
    axis = np.linspace(0, 10000, 140)
    X, Y = np.meshgrid(axis, axis)
    B = 50000 + .0015*X - .001*Y
    for _ in range(12):
        cx, cy = rng.uniform(500, 9500, 2)
        sx, sy = rng.uniform(450, 1300, 2)
        amp = rng.uniform(45, 130) * rng.choice([-1, 1])
        B += amp*np.exp(-.5*((X-cx)**2/sx**2+(Y-cy)**2/sy**2))
    B += 7*np.sin(X/1100+rng.uniform(0, 6))*np.cos(Y/1500+rng.uniform(0, 6))
    error = gaussian_filter(rng.normal(size=B.shape), 4)
    error -= error.mean()
    error *= MAP_ERROR_NT/np.sqrt(np.mean(error**2))
    return axis, B, B+error


def waypoint_fisher(axis, B):
    """Per-waypoint Fisher information on the corridor lattice, from the STORED map."""
    xs = np.linspace(1000, 9000, 9)
    ys = np.array([4750, 5000, 5250])
    X, Y = np.meshgrid(xs, ys)
    points = np.column_stack([X.ravel(), Y.ravel()])
    sample = lambda a: nav.bilinear_interpolate(axis, axis, a, points).reshape(3, 9)
    # B[i, j] is at y=axis[i], x=axis[j], so np.gradient returns d/dy first.
    gy, gx = np.gradient(B, axis, axis)
    grad = np.stack([sample(gx), sample(gy)], axis=-1)
    fisher = np.einsum('...i,...j->...ij', grad, grad)/PLANNER_SENSOR_VARIANCE_NT2
    return xs, ys, grad, fisher, np.trace(fisher, axis1=-2, axis2=-1)


def build_problem(grad, fisher, planner):
    """The reference problem, plus the Q-MagOpt problem when that path is selected."""
    reference = ref.ReferenceProblem(fisher)
    if planner == 'reference':
        return reference, None
    from qmagopt import MagneticMap, SensorModel, NavigationProblem
    base = MagneticMap.from_field(np.zeros((3, 9)), np.zeros((3, 9)), cell_size_m=1000,
                                  sensor=SensorModel(noise_nt=2, map_error_nt=MAP_ERROR_NT))
    trace = np.trace(fisher, axis1=-2, axis2=-1)
    # `information` is not read by the d_optimal objective; it is set only to keep
    # the MagneticMap internally consistent.
    base = replace(base, gradient_nt_per_m=grad, fisher=fisher,
                   information=(trace-trace.min())/(np.ptp(trace)+1e-30))
    return NavigationProblem(base, information_objective='d_optimal'), reference


def check_planners_agree(problem, reference):
    """Assert the reference reproduces Q-MagOpt, including the declared prior."""
    routes = list(problem.feasible_routes())
    assert sorted(routes) == sorted(reference.feasible_routes()), 'candidate sets differ'
    rng = np.random.default_rng(0)
    for i in rng.choice(len(routes), size=min(40, len(routes)), replace=False):
        route = routes[int(i)]
        np.testing.assert_allclose(problem.route_fisher(route),
                                   reference.route_fisher(route), rtol=1e-9, atol=0)


def plan(axis, B, planner='reference'):
    """Exact selection over the same finite routes and distance budget."""
    t0 = time.perf_counter()
    xs, ys, grad, fisher, trace = waypoint_fisher(axis, B)
    problem, reference = build_problem(grad, fisher, planner)
    if reference is not None:
        check_planners_agree(problem, reference)

    candidates = []
    for route in problem.feasible_routes():
        points = np.column_stack([xs, ys[list(route)]])
        length = np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
        if length <= DISTANCE_BUDGET_M:
            info = problem.route_fisher(route)
            # The objective is the POSTERIOR determinant: the prior is part of it.
            assert np.allclose(info - ref.PRIOR_INFORMATION,
                               sum(fisher[r, t] for t, r in enumerate(route)), rtol=1e-9), \
                'route_fisher is not prior + accumulated waypoint Fisher'
            candidates.append((route, length, np.linalg.det(info),
                               sum(trace[r, t] for t, r in enumerate(route))))

    selected = [min(candidates, key=lambda z: (z[1], z[0])),
                max(candidates, key=lambda z: (z[3], -z[1])),
                max(candidates, key=lambda z: (z[2], -z[1]))]

    # Validate the exact QUBO determinant encoding on a sample of ALL candidates,
    # not only the three selected ones.
    lin, quad, const = problem.information_qubo_terms()
    rng = np.random.default_rng(1)
    sample = rng.choice(len(candidates), size=min(50, len(candidates)), replace=False)
    for i in list(sample) + [0, len(candidates)-1]:
        route, _, det, _ = candidates[int(i)]
        x = np.array(list(problem.route_to_bitstring(route)), dtype=float)
        assert np.isclose(const+lin@x+x@quad@x, det, rtol=1e-8, atol=1e-18)

    paths = {}
    for method, (route, length, det, tr) in zip(METHODS, selected):
        dense_x = np.linspace(1000, 9000, 161)
        truth = np.column_stack([dense_x, np.interp(dense_x, xs, ys[list(route)])])
        assert np.allclose(truth[0], [1000, 5000])
        assert np.allclose(truth[-1], [9000, 5000])
        assert length <= DISTANCE_BUDGET_M
        paths[method] = (truth, length, route, det)
    return paths, time.perf_counter()-t0, len(candidates)


def evaluate(axis, B, true_B, truth, map_seed, repeat, noise, particles):
    """One mission: simulate along `truth`, filter against the stored map B."""
    sensor_rng, filter_rng = trial_generators(map_seed, repeat, noise)
    cfg = nav.DemoConfig(seed=0, n_steps=len(truth), n_particles=particles,
                         gps_spoof_enabled=False, gps_residual_threshold_m=float('inf'))
    # The trajectory is passed explicitly; no module-level state is mutated.
    data = nav.simulate_sensors(cfg, axis, axis, true_B, noise, truth, rng=sensor_rng)
    t0 = time.perf_counter()
    result = nav.run_particle_filter(cfg, axis, axis, B, data,
                                     mag_sigma=np.hypot(noise, MAP_ERROR_NT),
                                     gnss_cutoff=20, rng=filter_rng)
    elapsed = time.perf_counter()-t0
    assert np.all(np.isfinite(result['estimate']))
    assert not result['gps_trusted'][20:].any()
    assert data['spoof_start'][0] == len(truth)
    err = np.linalg.norm(result['estimate']-truth, axis=1)[20:]
    return dict(rmse_m=float(np.sqrt(np.mean(err**2))),
                final_error_m=float(err[-1]), p95_error_m=float(np.quantile(err, .95)),
                failure=int(np.any(err > 250)), runtime_s=elapsed)


def summarize(frame):
    """Primary: ratio of map-averaged means, paired map-cluster bootstrap.
    Reported alongside per-map win rates and a paired rank test, because the
    ratio-of-means estimator is dominated by the highest-RMSE maps."""
    summaries = []
    rng = np.random.default_rng(99181)
    for noise, group in frame.groupby('noise_nt'):
        means = group.groupby(['map_seed', 'method']).mean(numeric_only=True)
        maps = sorted(group.map_seed.unique())
        idx = rng.integers(0, len(maps), (5000, len(maps)))
        for method in METHODS:
            g = group[group.method == method]
            row = dict(noise_nt=float(noise), method=method,
                       mean_rmse_m=float(g.rmse_m.mean()), failure_rate=float(g.failure.mean()),
                       mean_p95_m=float(g.p95_error_m.mean()), mean_length_m=float(g.length_m.mean()),
                       mean_runtime_s=float(g.runtime_s.mean()))
            for baseline in ['shortest', 'separable']:
                a = means.xs(method, level='method').loc[maps].rmse_m.to_numpy()
                b = means.xs(baseline, level='method').loc[maps].rmse_m.to_numpy()
                gains = 100*(1-a[idx].mean(axis=1)/b[idx].mean(axis=1))
                row['reduction_vs_'+baseline+'_pct'] = float(100*(1-a.mean()/b.mean()))
                row['ci_vs_'+baseline] = np.quantile(gains, [.025, .975]).tolist()
                wins = int((b > a).sum())
                row['maps_better_than_'+baseline] = wins
                row['sign_test_p_vs_'+baseline] = (
                    float(stats.binomtest(wins, len(maps), 0.5).pvalue) if method != baseline else 1.0)
                row['wilcoxon_p_vs_'+baseline] = (
                    float(stats.wilcoxon(a, b).pvalue) if method != baseline else 1.0)
            summaries.append(row)
    failure_intervals = []
    rng_failure = np.random.default_rng(19)
    for noise, g in frame.groupby('noise_nt'):
        rates = g.groupby(['map_seed', 'method']).failure.mean().unstack()
        indices = rng_failure.integers(0, len(rates), (5000, len(rates)))
        for baseline in ['shortest', 'separable']:
            delta = (rates[baseline]-rates.d_optimal).to_numpy()*100
            failure_intervals.append(dict(noise_nt=float(noise), baseline=baseline,
                failure_reduction_percentage_points=float(delta.mean()),
                ci=np.quantile(delta[indices].mean(axis=1), [.025, .975]).tolist()))
    (OUT/'failure_intervals.json').write_text(json.dumps(failure_intervals, indent=2))
    (OUT/'summary.json').write_text(json.dumps(summaries, indent=2))
    return summaries


def plot(frame):
    """Paired slope plot. Bars from zero hid the per-map pairs, which are the
    part of this result a reader most needs to see."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, axs = plt.subplots(1, 2, figsize=(11, 4.8), layout='constrained')
    colors = ['#64748b', '#d97706', '#087f8c']
    for ax, (noise, g) in zip(axs, frame.groupby('noise_nt')):
        vals = g.groupby(['map_seed', 'method']).rmse_m.mean().unstack()[METHODS]
        for _, r in vals.iterrows():
            ax.plot(range(3), r, alpha=.35, color='#94a3b8', linewidth=.9,
                    marker='o', markersize=2.5, zorder=1)
        means = vals.mean()
        ax.plot(range(3), means, color='#0f172a', linewidth=2.2, zorder=3)
        ax.scatter(range(3), means, s=70, c=colors, zorder=4, edgecolor='#0f172a', linewidth=1.1)
        for i, v in enumerate(means):
            ax.annotate(f'{v:.1f}', (i, v), textcoords='offset points', xytext=(0, 12),
                        ha='center', fontsize=9, fontweight='bold')
        wins = int((vals.shortest > vals.d_optimal).sum())
        ax.set_xticks(range(3), ['Shortest', 'Separable', 'D-optimal'])
        ax.set_xlim(-.35, 2.35)
        ax.set_ylabel('Post-GNSS-loss RMSE (m)')
        ax.set_title(f'Sensor noise: {noise:g} nT   '
                     f'(D-optimal beats shortest on {wins}/{len(vals)} maps)', fontsize=10)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle('Route planning evaluated with the same particle filter\n'
                 'Thin lines: one synthetic map each. Heavy line: mean across maps.')
    fig.savefig(OUT/'comparison.png', dpi=180)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--maps', type=int, default=30)
    parser.add_argument('--repeats', type=int, default=10)
    parser.add_argument('--particles', type=int, default=800)
    parser.add_argument('--planner', choices=['reference', 'qmagopt'], default='reference',
                        help='reference: standalone; qmagopt: the integration, cross-checked')
    args = parser.parse_args()
    OUT.mkdir(exist_ok=True)
    rows = []
    for map_seed in range(2000, 2000+args.maps):
        axis, B, true_B = field_map(map_seed)
        paths, plan_s, candidates = plan(axis, B, args.planner)
        for repeat in range(args.repeats):
            for noise in [2., 8.]:
                for method, (truth, length, route, det) in paths.items():
                    metrics = evaluate(axis, B, true_B, truth, map_seed, repeat, noise, args.particles)
                    rows.append(dict(map_seed=map_seed, repeat=repeat, noise_nt=noise, method=method,
                                     length_m=length, route='-'.join(map(str, route)), det=det,
                                     planning_s=plan_s, candidates=candidates, **metrics))
        pd.DataFrame(rows).to_csv(OUT/'trials.csv', index=False)
        print(f'map {map_seed}: {len(rows)} runs complete', flush=True)
    frame = pd.DataFrame(rows)
    print(json.dumps(summarize(frame), indent=2))
    plot(frame)
    (OUT/'config.json').write_text(json.dumps(
        dict(vars(args), entropy=ENTROPY, prior_sigma_m=ref.PRIOR_SIGMA_M), indent=2))

if __name__ == '__main__': main()
