"""Focused numerical checks; python -m unittest test_integration -v."""
import unittest
import numpy as np
import pandas as pd
from pathlib import Path
import integrated_benchmark as bench
import magnav_core as nav
import reference_planner as ref

V1 = Path(__file__).parent / 'integrated_results_qmagopt_v1' / 'trials.csv'


class IntegrationTests(unittest.TestCase):
    def test_bilinear_affine_field(self):
        a = np.linspace(0, 10000, 30)
        X, Y = np.meshgrid(a, a)
        p = np.array([[1234., 4321.], [8900., 1500.]])
        np.testing.assert_allclose(nav.bilinear_interpolate(a, a, 2*X-3*Y+7, p), 2*p[:, 0]-3*p[:, 1]+7)

    def test_maps_are_independent_and_error_budget_is_separate(self):
        a, b, t = bench.field_map(2000)
        _, b2, _ = bench.field_map(2001)
        self.assertFalse(np.allclose(b, b2))
        self.assertAlmostEqual(float(np.sqrt(np.mean((t-b)**2))), 2., places=9)
        np.testing.assert_array_equal(b, bench.field_map(2000)[1])

    def test_flat_map_selects_shortest_for_all_objectives(self):
        a = np.linspace(0, 10000, 140)
        paths, _, _ = bench.plan(a, np.zeros((140, 140)))
        for method in bench.METHODS:
            np.testing.assert_array_equal(paths[method][0], paths['shortest'][0])
            self.assertEqual(paths[method][1], 8000.)

    def test_paired_runs_and_cutoff(self):
        a, b, t = bench.field_map(2000)
        paths, _, _ = bench.plan(a, b)
        truth = paths['shortest'][0]
        first = bench.evaluate(a, b, t, truth, 2000, 0, 2., 200)
        second = bench.evaluate(a, b, t, truth.copy(), 2000, 0, 2., 200)
        for k in ['rmse_m', 'final_error_m', 'p95_error_m', 'failure']:
            self.assertEqual(first[k], second[k])

    def test_determinant_does_not_guarantee_lower_rms_bound(self):
        # D-optimality and A-optimality can rank positive definite matrices differently.
        A = np.diag([1., 100.]); B = np.diag([9., 9.])
        self.assertGreater(np.linalg.det(A), np.linalg.det(B))
        self.assertGreater(np.trace(np.linalg.inv(A)), np.trace(np.linalg.inv(B)))

    # --- checks added after the v1 audit -------------------------------------

    def test_noise_levels_draw_independent_randomness(self):
        """v1 keyed seeds on (map, repeat) only, so the 8 nT magnetic noise was
        exactly 4x the 2 nT noise and every other stream was identical."""
        s2, f2 = bench.trial_generators(2000, 0, 2.)
        s8, f8 = bench.trial_generators(2000, 0, 8.)
        x2, x8 = s2.normal(size=64), s8.normal(size=64)
        self.assertFalse(np.allclose(x2, x8))
        self.assertFalse(np.allclose(4*x2, x8))
        self.assertFalse(np.allclose(f2.normal(size=64), f8.normal(size=64)))

    def test_no_stream_collides_across_trials(self):
        """v1 derived sensor=seed+100 and filter=seed+500 from a base spaced 100
        apart, so map m's sensor stream equalled map m-4's filter stream."""
        seen = {}
        for map_seed in range(2000, 2030):
            for repeat in range(10):
                for noise in (2., 8.):
                    for role, gen in zip(('sensor', 'filter'),
                                         bench.trial_generators(map_seed, repeat, noise)):
                        key = gen.normal(size=4).tobytes()
                        self.assertNotIn(key, seen,
                                         f'{(role, map_seed, repeat, noise)} collides with {seen.get(key)}')
                        seen[key] = (role, map_seed, repeat, noise)

    def test_methods_stay_paired_on_identical_noise(self):
        """Pairing is the point of the design: the key must not include method."""
        key = dict(map_seed=2000, repeat=3, noise_nt=8.)
        s1, f1 = bench.trial_generators(**key)
        s2, f2 = bench.trial_generators(**key)
        np.testing.assert_array_equal(s1.normal(size=32), s2.normal(size=32))
        np.testing.assert_array_equal(f1.normal(size=32), f2.normal(size=32))

    def test_trajectory_is_explicit_not_monkeypatched(self):
        """v1's magnav_core called an undefined make_trajectory and worked only
        because the benchmark assigned it at runtime."""
        self.assertFalse(hasattr(nav, 'make_trajectory'))
        cfg = nav.DemoConfig(seed=0, n_steps=5, n_particles=10, gps_spoof_enabled=False)
        a = np.linspace(0, 10000, 20)
        traj = np.column_stack([np.linspace(1000, 9000, 5), np.full(5, 5000.)])
        data = nav.simulate_sensors(cfg, a, a, np.zeros((20, 20)), 2., traj,
                                    rng=np.random.default_rng(0))
        np.testing.assert_array_equal(data['truth'], traj)
        with self.assertRaises(ValueError):
            nav.simulate_sensors(cfg, a, a, np.zeros((20, 20)), 2., traj[:3],
                                 rng=np.random.default_rng(0))

    def test_spoofing_is_switched_off_not_pushed_past_the_horizon(self):
        cfg = nav.DemoConfig(seed=0, n_steps=12, n_particles=10, gps_spoof_enabled=False)
        a = np.linspace(0, 10000, 20)
        traj = np.column_stack([np.linspace(1000, 9000, 12), np.full(12, 5000.)])
        d = nav.simulate_sensors(cfg, a, a, np.zeros((20, 20)), 2., traj,
                                 rng=np.random.default_rng(0))
        self.assertEqual(int(d['spoof_start'][0]), 12)

    def test_qubo_encoding_is_exact_on_every_feasible_route(self):
        """v1 asserted this on the 3 selected routes only."""
        a, b, _ = bench.field_map(2000)
        _, _, _, fisher, _ = bench.waypoint_fisher(a, b)
        problem = ref.ReferenceProblem(fisher)
        lin, quad, const = problem.information_qubo_terms()
        routes = list(problem.feasible_routes())
        self.assertEqual(len(routes), 577)
        for route in routes:
            x = np.array(list(problem.route_to_bitstring(route)), dtype=float)
            self.assertEqual(x.sum(), 9)
            np.testing.assert_allclose(const + lin@x + x@quad@x,
                                       np.linalg.det(problem.route_fisher(route)), rtol=1e-8)

    def test_planner_prior_is_the_declared_200_m(self):
        """The objective is det(J_prior + J_route). The prior was undeclared in v1;
        this pins it so a silent change fails loudly."""
        self.assertEqual(ref.PRIOR_SIGMA_M, 200.0)
        a, b, _ = bench.field_map(2000)
        _, _, _, fisher, _ = bench.waypoint_fisher(a, b)
        problem = ref.ReferenceProblem(fisher)
        route = (1,)*9
        np.testing.assert_allclose(
            problem.route_fisher(route) - sum(fisher[r, t] for t, r in enumerate(route)),
            np.eye(2)/200.0**2, rtol=1e-12)

    @unittest.skipUnless(V1.exists(), 'v1 qmagopt results not present')
    def test_reference_planner_reproduces_recorded_qmagopt_output(self):
        """The reference planner is only trustworthy because it matches the real
        library's recorded routes, lengths and determinants on all 30 maps."""
        v1 = pd.read_csv(V1).drop_duplicates(['map_seed', 'method']).set_index(['map_seed', 'method'])
        for map_seed in range(2000, 2030):
            axis, B, _ = bench.field_map(map_seed)
            paths, _, candidates = bench.plan(axis, B, 'reference')
            self.assertEqual(candidates, int(v1.loc[(map_seed, 'shortest')].candidates))
            for method, (_, length, route, det) in paths.items():
                row = v1.loc[(map_seed, method)]
                self.assertEqual('-'.join(map(str, route)), row.route)
                self.assertAlmostEqual(length, float(row.length_m), places=6)
                np.testing.assert_allclose(det, float(row.det), rtol=1e-9)


if __name__ == '__main__': unittest.main()
