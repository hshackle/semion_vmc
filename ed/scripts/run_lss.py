"""Production run: momentum-sector spectra for level-spacing statistics.

Reproduces the data behind Fig. 6 of arXiv:2206.14730 for
  - semions on an 8x8 torus, N=4;
  - hard-core bosons on a 9x9 torus, N=4, column coupling r_col=0.5;
in the momentum sectors (0,0), (0, 2pi/L), (2pi/L, 4pi/L).

Usage: python -m scripts.run_lss semion|hcb
Saves eigenvalues to results/lss_<model>_k<jx><jy>.npy
"""

import os
import sys
import time

import numpy as np
import scipy.linalg as sla

sys.path.insert(0, ".")
from semion_ed.core import AnyonModel, hcb_model  # noqa: E402
from semion_ed.momentum import orbit_reps, sector_basis, sector_hamiltonian  # noqa: E402

SECTORS = [(0, 0), (0, 1), (1, 2)]   # in units of 2*pi/L


def main(which: str):
    os.makedirs("results", exist_ok=True)
    if which == "semion":
        model = AnyonModel(8, 8, 4, convention="paper")
    elif which == "hcb":
        model = hcb_model(9, 9, 4, r_col=0.5)
    else:
        raise SystemExit(f"unknown model {which}")
    t0 = time.time()
    print(f"[{which}] building H (dim {model.dim}) ...", flush=True)
    H = model.hamiltonian()
    print(f"[{which}] H built: nnz={H.nnz}  ({time.time()-t0:.0f}s)", flush=True)
    canon, reps = orbit_reps(model)
    print(f"[{which}] {len(reps)} translation orbits  ({time.time()-t0:.0f}s)",
          flush=True)
    for jx, jy in SECTORS:
        out = f"results/lss_{which}_k{jx}{jy}.npy"
        if os.path.exists(out):
            print(f"[{which}] {out} exists, skipping", flush=True)
            continue
        V = sector_basis(model, jx, jy, canon, reps)
        print(f"[{which}] sector ({jx},{jy}): dim {V.shape[1]} "
              f"({time.time()-t0:.0f}s)", flush=True)
        Hk = sector_hamiltonian(H, V)
        herm = np.abs(Hk - Hk.conj().T).max()
        print(f"[{which}] sector ({jx},{jy}): |Hk - Hk^dag|_max = {herm:.2e} "
              f"({time.time()-t0:.0f}s)", flush=True)
        Hk = 0.5 * (Hk + Hk.conj().T)
        del V
        ev = sla.eigvalsh(Hk, overwrite_a=True, check_finite=False)
        del Hk
        np.save(out, ev)
        print(f"[{which}] sector ({jx},{jy}): saved {len(ev)} eigenvalues "
              f"({time.time()-t0:.0f}s)", flush=True)
    print(f"[{which}] done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(sys.argv[1])
