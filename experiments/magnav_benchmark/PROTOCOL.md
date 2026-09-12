# Frozen experiment protocol

Written before benchmark outcomes, 2026-09-11. No result-driven seed or parameter selection.
Amended 2026-09-11 after an audit of the first run (v1); see "Amendments" at the end.
The amendments change randomization and disclosure only. The question, the methods,
the outcome definitions and the analysis plan are unchanged from the pre-registered version.

Question: does route-level D-optimal information improve measured particle-filter
localization over shortest and separable-information routes under GNSS loss?

Integration: Q-MagOpt's NavigationProblem, route_fisher and exact determinant
encoding select a 3-row, 9-column corridor route. Q-MagNav's particle filter,
bilinear interpolation and sensor simulator evaluate its continuous interpolation.
Exact classical enumeration isolates route objective from solver performance.
No QAOA performance or quantum advantage is tested.

30 independent synthetic Gaussian-anomaly maps (seeds 2000–2029), 10 noise
repeats each, sensor RMS 2 and 8 nT. A separate smooth 2 nT RMS error field is
added to the measurement-generating map, never exposed to route selection.
Routes share endpoints and 161 measurement epochs. Nine waypoints span 8 km,
row spacing 250 m, maximum one-row jump; length cannot exceed 8.8 km.
Each mission has equal duration/sample count; longer routes imply higher speed.
Heading bias is the source simulator's fixed 4 degrees; no heading state is estimated.
GNSS is removed at epoch 20. The magnetic consistency detector is disabled in
this experiment to isolate route choice; no spoof-detection claim is made.
800 particles; source process noise 10 m, odometry noise 5 m per epoch.

## Randomization

Each trial draws from an independent `numpy.random.SeedSequence` keyed on
(map seed, repeat, sensor noise level), with sensor and filter streams as spawned
children. The key deliberately excludes the routing method, so the three methods
are paired on identical sensor, odometry, initial GNSS and filter noise. Including
the noise level is what makes the 2 nT and 8 nT arms independent replications.
Spawned children guarantee no two trials share a stream.

## Planner objective

The planner scores a route by the determinant of the POSTERIOR information matrix,
det(J_prior + J_route), where J_route is the sum of per-waypoint Fisher matrices
computed on the stored map with variance 8 nT² (2 nT sensor + 2 nT map error), and
J_prior is isotropic with standard deviation 200 m. The prior is part of the Q-MagOpt
objective and materially affects the ranking; it is asserted in `plan()` and pinned by
`test_planner_prior_is_the_declared_200_m`. Routes are fixed at the nominal 2 nT
design; the 8 nT level tests robustness of those routes, not a replanned design.

Primary outcome: post-cutoff positional RMSE per mission, averaged first within
map then across maps. 95% intervals use paired map-cluster bootstrap, 5000 draws.
Because the ratio-of-means estimator is dominated by the highest-RMSE maps, the
per-map win rate, an exact sign test and a paired Wilcoxon signed-rank test are
reported beside it for every comparison. A claim rests on the bootstrap interval
and the rank evidence agreeing; where they disagree, both are stated.

Secondary: final error, post-cutoff 95th percentile error, any error >250 m
(a prespecified illustrative failure proxy, not a validated operational limit),
route length and planning/runtime. Report both information baselines, all seeds,
and both noise levels, including negative results. Aspirational target: >=20%
RMSE reduction vs shortest with <=10% additional distance. Do not tune to hit it.

Limitations: synthetic map; no flight data, realistic dynamics or heading/bias
estimation; open-loop route planning; local/static Fisher proxy ignores filter
dynamics and global ambiguity; same filter tuning for all methods; no guarantee
of optimality over arbitrary continuous paths. Map seeds are independent draws
from one generator, not independent geographic survey datasets. The two noise
arms share maps and routes, so their results remain correlated (measured r = 0.48
between paired mission RMSEs) even with independent noise draws.

## Amendments after the v1 audit

1. **Randomization rewritten.** v1 keyed seeds on (map, repeat) only, deriving
   sensor = base+100 and filter = base+500 from a base spaced 100 apart. Two
   defects followed: the 8 nT magnetic noise was exactly four times the 2 nT noise
   with every other stream identical (paired mission RMSEs correlated at r = 0.77),
   so the two arms were one draw at two scalings rather than two replications; and
   map m's sensor stream equalled map m−4's filter stream in 260 of 300 cells,
   coupling clusters the bootstrap treats as independent. Both are regression-tested.
2. **Rank evidence added to the analysis plan**, as described above.
3. **The 200 m planner prior is now declared**, having been undeclared in v1.
4. **Trajectory injection made explicit.** v1 assigned `nav.make_trajectory` at
   runtime to satisfy a call to a name that did not exist in the extracted module.
5. **Results regenerated.** All numbers in README.md come from the amended run.
   The v1 results are retained unchanged in `integrated_results_qmagopt_v1/`.
