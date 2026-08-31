# Research note: constrained quantum optimization in a magnetic-navigation decide loop

## Research question

When GNSS is unavailable, a vehicle localizes by matching a measured magnetic field
against a stored anomaly map. Map matching works well over some terrain and badly over
other terrain, and the planner gets to choose which terrain it flies over. So:

> Can a route planner trade operational cost against **map-matching observability**, and
> is that trade-off naturally expressible in the encodings quantum optimizers consume?

This note answers the second half precisely. The first half is a formulation and
benchmarking exercise, not a demonstration of quantum advantage.

---

## 1. Where the information comes from

A scalar total-field magnetometer flying over a mapped anomaly field `B(p)` returns

```
z = B(p) + n,        n ~ N(0, sigma^2)
```

where `sigma` combines sensor noise and map representation error. In a real error
budget the second term dominates: a quantum scalar magnetometer contributes on the order
of picotesla, while anomaly maps disagree with the true field at the nanotesla level.
The demonstrator's defaults reflect this (`noise_nt = 0.01`, `map_error_nt = 2.0`).

The log-likelihood of a position hypothesis is `-(z - B(p))^2 / (2 sigma^2)`, its score
is `(z - B(p)) grad B(p) / sigma^2`, and so the Fisher information matrix for horizontal
position is the outer product

```
J(p) = grad B(p) grad B(p)^T / sigma^2          [1/m^2]
```

Two facts follow, and they shape everything else.

**J is rank one.** One scalar reading pins position down only along the local gradient
direction; the perpendicular direction is completely unobservable from that measurement.
This is checked directly in `tests/test_maps.py::test_per_cell_fisher_is_rank_one`.

**Information is therefore a property of the trajectory, not of a cell.** Position
becomes observable only by accumulating measurements whose gradient directions *differ*.
Two beautiful, high-gradient cells whose gradients are parallel are worth far less
together than their individual scores suggest. This is precisely what a per-cell score
cannot see — and precisely what makes route selection a real optimization problem
rather than a lookup.

Accumulated information over a route is the sum of the per-step matrices seeded with the
dead-reckoning prior, `J_route = J_0 + sum_t J(p_t)`, which yields two reportable
quantities in physical units: the position CRLB `sqrt(tr(J_route^-1))` in metres, and the
D-optimality criterion `det J_route`.

---

## 2. The main result: exact D-optimality is already a QUBO

The standard move at this point is to invent a separable per-cell score — the original
version of this project used a normalized blend of gradient magnitude and "rarity" — and
accept it as a surrogate, because a QUBO only admits quadratic terms and `log det` is not
separable.

That concession is unnecessary. Write the accumulated information as
`J(x) = J_0 + sum_i x_i F_i` over binary selection variables `x_i` with each
`F_i = g_i g_i^T / sigma^2`. For **two-dimensional** position, `det` of a 2x2 matrix is a
quadratic form in its entries, and each entry is linear in `x`. Expanding with
`x_i^2 = x_i`:

```
det J(x) = det J_0
         + sum_i    x_i     [ J0_11 F_i,22 + J0_22 F_i,11 - 2 J0_12 F_i,12 ]
         + sum_{i<j} x_i x_j  |g_i x g_j|^2 / sigma^4
```

This is the Cauchy–Binet expansion. Three things are worth pointing out:

* The **pairwise coefficient is the squared cross product of the two gradients**. It is
  large exactly when the two cells' gradients are perpendicular and zero when they are
  parallel. The encoding rewards gradient-direction diversity by construction — the
  thing a separable score is blind to.
* The **diagonal terms vanish identically**, because each `F_i` is rank one and so
  `det F_i = 0`. A cell contributes nothing to the determinant on its own.
* The expression is **exact**. There is no surrogate, no linearization, no auxiliary
  variables, and no increase in qubit count.

It is implemented in `NavigationProblem.information_qubo_terms` and verified against the
true determinant over every route in `tests/test_d_optimal.py::test_cauchy_binet_expansion_is_exact`
(agreement to ~1e-18).

### What it costs

Information couples **every pair** of time steps, not just adjacent ones. That has two
consequences, one good for this project's thesis and one bad.

The bad one: the QUBO becomes all-to-all coupled, which on hardware means either full
connectivity or minor-embedding overhead. The separable formulation stays banded.

The good one: **the dynamic program stops being valid**. A shortest-path recursion needs
a single-step state that summarizes the past; under D-optimality the whole accumulated
matrix is the state. `solve_dynamic_programming` raises rather than silently returning a
wrong answer. Simulated annealing on the same QUBO also stops solving every instance to
optimality (`tests/test_d_optimal.py::test_annealing_usually_reaches_the_exact_optimum_and_never_beats_it`).
This is the regime where a different heuristic could plausibly earn its place — and it is
reached by taking the physics seriously, not by inflating the problem size.

### Does it matter numerically?

Solving both objectives exactly on the same 40 instances — so the comparison isolates the
encoding from the solver — the exact criterion is never worse (40/40), strictly better on
22/40, and improves the median position CRLB from 10.84 m to 8.78 m (mean 11.69 m to
9.62 m). The two objectives agree on the route in most instances, which is why the
aggregate gain depends on how many instances are sampled; the *never worse* property is
the robust statement.

The separable surrogate's rank correlation with exact `log det J` has a median around
0.89 but ranges down to near zero — and to negative values on wider samples: usually a
decent proxy, occasionally an actively misleading one.

---

## 3. Constraint handling

At each step `t` exactly one candidate row is occupied: `sum_r x[t,r] = 1`.

The penalty formulation enforces this with `lambda sum_t (sum_r x[t,r] - 1)^2`. The XY
formulation instead initializes each time layer in its one-excitation subspace and mixes
with an XY mixer, which restricted to that subspace is a continuous-time quantum walk on
the mixer graph. Layers act on disjoint qubits, so the full mixer factorizes exactly and
the one-hot constraint is preserved by construction. Start/goal, obstacles and jump
limits remain soft penalties — an honest intermediate point rather than a complete
constraint-preserving encoding, and the natural place for a transition-preserving mixer
in future work.

### The comparison has to be conditional

The penalty ansatz searches all `2^15 = 32768` basis states; the XY ansatz searches
`3^5 = 243`. Reporting that XY puts more probability on feasible states would mostly be
reporting that its search space is 135× smaller. Every result therefore carries the
**conditional** feasibility inside the one-hot sector, which is the number that isolates
the mixer's contribution.

Under that fair comparison the XY advantage is real but far smaller than the raw numbers
suggest — roughly 5×, not 200× — and it grows with circuit depth, while the penalty
formulation's does not reliably. Two supporting choices matter: a CVaR objective over the
best tail of the distribution rather than the plain expectation value, and INTERP warm
starting of the angle schedule from depth `p` to `p+1`. Without INTERP, deeper circuits
routinely score *worse* because the classical optimizer stalls.

---

## 4. Honest limitations

* **No quantum advantage, and none claimed.** Under the separable objective, dynamic
  programming solves these instances exactly in microseconds; simulated annealing on the
  same QUBO reaches the optimum on essentially every instance. The QAOA variants are
  worse than both on solution quality at the depths simulated.
* **The CRLB is a local bound.** It measures how sharply the likelihood peaks around the
  true position. It says nothing about *global* ambiguity — a distant part of the map
  producing a similar field value — which is a genuine failure mode of magnetic
  navigation and is not modelled here at all.
* **The map is synthetic.** Amplitudes and gradients are plausible for a crustal anomaly
  survey, but no real survey product or flight data is used.
* **Static, single-vehicle, discretized corridor.** No vehicle dynamics, no wind, no
  filter in the loop, no re-planning, and position is discretized to a small grid.
* **Instances are tiny.** 15 qubits, chosen so the full statevector and the exact optimum
  are both available. Nothing here probes the scaling regime that would matter.
* **The exact D-optimal encoding is 2-D specific.** The determinant of an `n x n` matrix
  is degree `n` in its entries, so the Cauchy–Binet trick gives a quadratic form only for
  two-dimensional position. Adding altitude or heading to the state would need auxiliary
  variables or a different criterion (E-optimality, or a linearized A-optimal surrogate).

---

## 5. Where this should go next

1. **Real data.** Swap the synthetic map for an open airborne magnetic-navigation dataset
   and re-derive the sensor error budget from the survey's own specification.
2. **Filter-in-the-loop information.** Replace the static CRLB with entropy or covariance
   reduction from an actual EKF/UKF or particle filter, which also captures the global
   ambiguity the CRLB misses.
3. **Transition-preserving mixers.** Move the jump-limit constraint out of the penalty
   and into the mixer, so the ansatz preserves *connected* routes rather than just
   one-hot layers.
4. **Stronger classical baselines.** CP-SAT or MILP on the exact D-optimal objective,
   which is where the classical difficulty actually starts.
5. **Hardware study.** The all-to-all coupling of the exact encoding against the banded
   separable one is a concrete, measurable embedding-overhead question: does the better
   objective survive the connectivity cost under noise?
6. **Survey planning.** Turn the problem around — plan where a UAV should fly to *build*
   or refresh an anomaly map under battery and time constraints, an orienteering problem
   whose objective is the same determinant.

---

## References in spirit

The formulation draws on standard results rather than novel theory: the Cramér–Rao bound
and Fisher information for parameter estimation; D- and E-optimal design criteria from
optimal experimental design; XY / one-hot-preserving mixers for constrained QAOA
(Hadfield et al.'s Quantum Alternating Operator Ansatz); the CVaR variational objective
(Barkoutsos et al.); and INTERP parameter transfer across circuit depth (Zhou et al.).
The magnetic-navigation framing follows the public literature on magnetic anomaly
navigation as a GNSS-denied positioning method.
