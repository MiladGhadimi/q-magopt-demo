# v1 results (superseded)

Output of the first run, using the real `qmagopt` package at commit 13b6d30a.
Retained unchanged for comparison and because `test_integration.py` validates the
standalone `reference_planner.py` against the routes and determinants recorded here.

These numbers are **not** the study's results. They were produced under the
randomization defects described in `../README.md` → Corrections: the 2 nT and 8 nT
arms shared one noise realization, and sensor/filter RNG streams collided across
map clusters. Current results are in `../integrated_results/`.
