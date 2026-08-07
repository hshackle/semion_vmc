"""Momentum-sector construction (App. E of arXiv:2206.14730).

Momentum states are built by projecting basis states |c, f> with the
translation operators:

    |c, k, A> ~ sum_{j=0}^{Lx-1} sum_{l=0}^{Ly-1} e^{i(j kx + l ky)}
                T_y^l T_x^j |c, f>

T_x and T_y act on configurations as shifts and on the sheet space as
monomial (generalized permutation) matrices; see AnyonModel.Tx_action /
Ty_action. Per translation orbit we project the M seed vectors |c_ref, f>
into the sector and orthonormalize with an SVD; columns with non-zero
singular value form the sector basis. Since [T_x, T_y] = 0, T_x^Lx =
T_y^Ly = 1 (verified in the tests), these states exactly block-diagonalize H.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from .core import AnyonModel, rank_configs


def compose(mon1, mon2):
    """Monomial composition: apply mon1 first, then mon2.

    mon = (dsheet, phases[M]): |f> -> phases[f] |(f+dsheet)%M>.
    """
    d1, p1 = mon1
    d2, p2 = mon2
    M = len(p1)
    f = np.arange(M)
    return ((d1 + d2) % M, p1 * p2[(f + d1) % M])


def orbit_reps(model: AnyonModel, chunk: int = 20000):
    """Canonical representative (minimal colex rank over translations) per config.

    Returns (canon, reps) where canon[i] is the canonical rank of config i and
    reps is the sorted array of unique canonical ranks.
    """
    Lx, Ly = model.Lx, model.Ly
    Nc = model.nconf
    canon = np.full(Nc, np.iinfo(np.int64).max, dtype=np.int64)
    cfg = model.configs
    for lo in range(0, Nc, chunk):
        hi = min(lo + chunk, Nc)
        block = cfg[lo:hi]
        x, y = model.xy(block)
        best = np.full(hi - lo, np.iinfo(np.int64).max, dtype=np.int64)
        for dx in range(Lx):
            for dy in range(Ly):
                sites = np.sort(((x + dx) % Lx) * Ly + (y + dy) % Ly, axis=1)
                r = rank_configs(sites, model.C)
                np.minimum(best, r, out=best)
        canon[lo:hi] = best
    return canon, np.unique(canon)


def orbit_group_action(model: AnyonModel, rep_sites: np.ndarray):
    """All T_y^l T_x^j applied to the orbit representative.

    Returns lists conf_rank[j][l] and monomial[j][l] such that
    T_y^l T_x^j |rep, f> = phases[f] |conf, (f+dsheet)%M>.
    """
    Lx, Ly, M = model.Lx, model.Ly, model.M
    ident = (0, np.ones(M, dtype=np.complex128))
    ranks = [[None] * Ly for _ in range(Lx)]
    mons = [[None] * Ly for _ in range(Lx)]
    sites_x, mon_x = rep_sites, ident
    for j in range(Lx):
        sites, mon = sites_x, mon_x
        for l in range(Ly):
            ranks[j][l] = int(rank_configs(sites, model.C))
            mons[j][l] = mon
            new_sites, step = model.Ty_action(sites)
            sites, mon = new_sites, compose(mon, step)
        new_sites, step = model.Tx_action(sites_x)
        sites_x, mon_x = new_sites, compose(mon_x, step)
    return ranks, mons


def sector_basis(model: AnyonModel, kx_idx: int, ky_idx: int,
                 canon=None, reps=None, tol: float = 1e-8) -> sp.csr_matrix:
    """Sparse isometry V (dim x n_sector) whose columns span the (kx, ky) sector.

    kx = 2*pi*kx_idx/Lx, ky = 2*pi*ky_idx/Ly; T_x V = e^{-i kx} V etc.
    """
    Lx, Ly, M = model.Lx, model.Ly, model.M
    if canon is None or reps is None:
        canon, reps = orbit_reps(model)
    kx = 2 * np.pi * kx_idx / Lx
    ky = 2 * np.pi * ky_idx / Ly
    phase_jl = np.exp(1j * (kx * np.arange(Lx)[:, None] +
                            ky * np.arange(Ly)[None, :]))
    col_idx, col_ptr, col_val = [], [0], []
    ncols = 0
    for rep in reps:
        rep_sites = model.configs[rep]
        ranks, mons = orbit_group_action(model, rep_sites)
        # accumulate projected seed vectors for f0 = 0..M-1
        acc = {}
        for j in range(Lx):
            for l in range(Ly):
                d, p = mons[j][l]
                w = phase_jl[j, l]
                base = ranks[j][l] * M
                for f0 in range(M):
                    key = base + (f0 + d) % M
                    if key not in acc:
                        acc[key] = np.zeros(M, dtype=np.complex128)
                    acc[key][f0] += w * p[f0]
        keys = np.fromiter(acc.keys(), dtype=np.int64, count=len(acc))
        mat = np.array([acc[k] for k in keys])          # (nkeys, M)
        # orthonormalize the M seed columns
        U, s, _ = np.linalg.svd(mat, full_matrices=False)
        for m in range(len(s)):
            if s[m] > tol * np.sqrt(Lx * Ly):
                order = np.argsort(keys)
                col_idx.append(keys[order])
                col_val.append(U[order, m])
                col_ptr.append(col_ptr[-1] + len(keys))
                ncols += 1
    if ncols == 0:
        return sp.csr_matrix((model.dim, 0), dtype=np.complex128)
    V = sp.csc_matrix((np.concatenate(col_val), np.concatenate(col_idx),
                       np.array(col_ptr)), shape=(model.dim, ncols))
    return V.tocsc()


def sector_hamiltonian(H: sp.csr_matrix, V: sp.csc_matrix,
                       block: int = 2048, dtype=np.complex128) -> np.ndarray:
    """Dense H_k = V^dagger H V computed in column blocks."""
    n = V.shape[1]
    Hk = np.empty((n, n), dtype=dtype)
    Vh = V.getH().tocsr()
    for lo in range(0, n, block):
        hi = min(lo + block, n)
        W = H @ V[:, lo:hi]
        Hk[:, lo:hi] = (Vh @ W).toarray()
    return Hk
