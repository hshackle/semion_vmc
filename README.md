# semion_vmc

Numerical studies of semions on a lattice.

- **`ed/`** — exact-diagonalization reproduction of the semion
  (abelian-anyon) results of Kirchner et al., *Numerical simulation of
  non-abelian anyons*, arXiv:2206.14730. See [ed/NOTES.md](ed/NOTES.md)
  for the algorithm, conventions and validation, and `ed/results/` for
  the figures (level-spacing statistics and quench dynamics).
- **`vmc/`** — neural-quantum-state variational Monte Carlo with the
  anyonic translational symmetries built into the ansatz. Specification
  and exact validation targets: [vmc/SUMMARY.md](vmc/SUMMARY.md).

## ed/ layout

Run everything from inside `ed/`:

```
cd ed
python -m tests.run_checks        # consistency suite
python -m scripts.run_lss semion  # level statistics production runs
python -m scripts.run_quench      # ladder quench
python -m scripts.make_figures    # figures
```

- `semion_ed/core.py` — basis, hop amplitude rules, sparse Hamiltonian,
  translation operators (semions / general abelian anyons / hard-core bosons).
- `semion_ed/momentum.py` — translation orbits, momentum-sector bases,
  sector Hamiltonians.
- `tests/run_checks.py` — braiding, torus holonomies, symmetries,
  sector completeness.
- `scripts/` — production runs and figure generation.
