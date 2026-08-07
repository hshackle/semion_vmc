# semion_vmc

Numerical studies of semions on a lattice.

Current contents: an exact-diagonalization reproduction of the semion
(abelian-anyon) results of Kirchner et al., *Numerical simulation of
non-abelian anyons*, arXiv:2206.14730 — see [NOTES.md](NOTES.md) for the
algorithm, conventions and validation, and `results/` for figures.

## Layout

- `semion_ed/core.py` — basis, hop amplitude rules, sparse Hamiltonian,
  translation operators (semions / general abelian anyons / hard-core bosons).
- `semion_ed/momentum.py` — translation orbits, momentum-sector bases,
  sector Hamiltonians.
- `tests/run_checks.py` — consistency suite (braiding, torus holonomies,
  symmetries, sector completeness). Run: `python -m tests.run_checks`.
- `scripts/run_lss.py` — level-spacing-statistics production run
  (semions 8×8 N=4; HCB 9×9 N=4).
- `scripts/run_quench.py` — 30×2 ladder quench (fermions / HCB / semions).
- `scripts/make_figures.py` — figures analogous to the paper's Figs. 6 and 8.
