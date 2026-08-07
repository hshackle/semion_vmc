"""Quench dynamics on a 30x2 ladder (Fig. 8 of arXiv:2206.14730).

N=4 particles start in a zigzag pattern in the middle of the ladder,
sites (13,0), (14,1), (15,0), (16,1); for semions the initial state is an
equal superposition of both sheets. Column densities <n_i(t)> are recorded
for t in [0, 50] (units of 1/t_hop).

Usage: python -m scripts.run_quench [fermion|hcb|semion ...]
"""

import os
import sys
import time

import numpy as np
import scipy.sparse.linalg as spla

sys.path.insert(0, ".")
from semion_ed.core import AnyonModel, hcb_model, rank_configs  # noqa: E402

LX, LY, N = 30, 2, 4
SITES0 = [13 * LY + 0, 14 * LY + 1, 15 * LY + 0, 16 * LY + 1]
TIMES = np.linspace(0.0, 50.0, 201)


def column_density_free_fermions():
    """Exact free-fermion evolution from the zigzag Slater determinant."""
    ns = LX * LY
    H = np.zeros((ns, ns))
    for x in range(LX):
        for y in range(LY):
            s = x * LY + y
            H[((x + 1) % LX) * LY + y, s] -= 1.0
            H[x * LY + (y + 1) % LY, s] -= 1.0
    H = H + H.T
    w, U = np.linalg.eigh(H)
    n = np.zeros((len(TIMES), LX))
    for it, t in enumerate(TIMES):
        prop = (U * np.exp(-1j * w * t)) @ U.conj().T   # e^{-iHt}
        amp = prop[:, SITES0]                            # (ns, 4)
        dens = (np.abs(amp) ** 2).sum(axis=1)            # site densities
        n[it] = dens.reshape(LX, LY).sum(axis=1)
    return n


def column_density_manybody(model):
    idx0 = int(rank_configs(np.sort(np.array(SITES0)), model.C))
    M = model.M
    psi0 = np.zeros(model.dim, dtype=complex)
    for f in range(M):
        psi0[idx0 * M + f] = 1.0 / np.sqrt(M)
    H = model.hamiltonian()
    print(f"  H: dim {model.dim}, nnz {H.nnz}", flush=True)
    xs, _ = model.xy(model.configs)
    counts = np.zeros((model.nconf, LX), dtype=np.float64)
    for i in range(LX):
        counts[:, i] = (xs == i).sum(axis=1)
    psi_t = spla.expm_multiply(-1j * H.tocsc(), psi0,
                               start=TIMES[0], stop=TIMES[-1],
                               num=len(TIMES), endpoint=True)
    n = np.zeros((len(TIMES), LX))
    for it in range(len(TIMES)):
        p = np.abs(psi_t[it]) ** 2
        p = p.reshape(model.nconf, M).sum(axis=1)
        n[it] = p @ counts
        norm = p.sum()
        if abs(norm - 1) > 1e-6:
            print(f"  warning: norm deviation {norm-1:.2e} at t={TIMES[it]:.2f}",
                  flush=True)
    return n


def main(which):
    os.makedirs("results", exist_ok=True)
    t0 = time.time()
    for w in which:
        out = f"results/quench_{w}.npz"
        if os.path.exists(out):
            print(f"{out} exists, skipping", flush=True)
            continue
        print(f"[{w}] running ...", flush=True)
        if w == "fermion":
            n = column_density_free_fermions()
        elif w == "hcb":
            n = column_density_manybody(hcb_model(LX, LY, N, r_col=0.5))
        elif w == "semion":
            n = column_density_manybody(AnyonModel(LX, LY, N, convention="paper"))
        else:
            raise SystemExit(f"unknown model {w}")
        np.savez(out, t=TIMES, n=n)
        print(f"[{w}] saved {out}  ({time.time()-t0:.0f}s)", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:] or ["fermion", "hcb", "semion"])
