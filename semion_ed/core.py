"""Exact diagonalization of abelian anyons (semions) hopping on an Lx x Ly torus.

Implements the abelian specialization of the algorithm of Kirchner, Millar,
Ayeni, Smith, Slingerland & Pollmann, "Numerical simulation of non-abelian
anyons", arXiv:2206.14730 (see Sec. II.B, Sec. V and App. D of the paper).

Model
-----
Hard-core anyons with exchange phase theta = pi*p/M (semions: theta = pi/2,
M = 2) hop on a periodic square lattice with the tight-binding Hamiltonian

    H = -t sum_i ( T_{r_i, r_i+e_x} + T_{r_i, r_i+e_y} + h.c. ).

On the torus the Hilbert space carries an extra M-fold "sheet" index f
(the anyonic charge threading the torus); the total number of anyons must
satisfy e^{2 i theta N} = 1 (for semions: N even).

Amplitude rules (identical anyons of charge eta=1)
--------------------------------------------------
Sites are (x, y), x in 0..Lx-1, y in 0..Ly-1. Cut A sits between rows
Ly-1 and 0; cut B between columns Lx-1 and 0.

+x hop from (x,y) to (x+1,y):
    phase exp(i theta (n_iup - n_jdn)) where n_iup counts anyons in column x
    strictly above y and n_jdn counts anyons in column x+1 strictly below y.
    If x == Lx-1 (cut B): additionally sheet f -> (f-1) mod M and, in the
    paper's convention, an extra factor exp(-i pi N / M).

+y hop from (x,y) to (x,y+1):
    trivial in the bulk. If y == Ly-1 (cut A):
    paper convention (App. D / Fig. 12): phase exp(i theta (nL - nR)) where
    nL (nR) counts anyons in columns strictly left (right) of x — anyons in
    the same column do not braid — times exp(-i pi N / M) * exp(2 i theta f).
    ref convention (Ref. 43.10761 / Sec. II.B): phase
    exp(2 i theta nL) * exp(i theta n_samecol) * exp(2 i theta f).

Negative-direction hops are the hermitian conjugates.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from itertools import combinations

import numpy as np
import scipy.sparse as sp


# ----------------------------------------------------------------------------
# configuration enumeration / ranking (colex order of sorted site tuples)
# ----------------------------------------------------------------------------

def binom_table(nmax: int, kmax: int) -> np.ndarray:
    """C[n, k] for 0 <= n <= nmax, 0 <= k <= kmax (int64)."""
    C = np.zeros((nmax + 1, kmax + 1), dtype=np.int64)
    C[:, 0] = 1
    for n in range(1, nmax + 1):
        for k in range(1, kmax + 1):
            C[n, k] = C[n - 1, k - 1] + C[n - 1, k]
    return C


def enumerate_configs(nsites: int, N: int) -> np.ndarray:
    """All N-site configs as sorted site tuples, in colex order. Shape (Nc, N)."""
    configs = np.array(list(combinations(range(nsites), N)), dtype=np.int64)
    # colex order == sort by reversed tuple; combinations gives lex order.
    # Rank them and re-sort so that rank_configs(configs[i]) == i.
    C = binom_table(nsites, N)
    ranks = rank_configs(configs, C)
    order = np.argsort(ranks)
    out = configs[order]
    assert np.array_equal(rank_configs(out, C), np.arange(len(out)))
    return out


def rank_configs(sites_sorted: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Colex rank of sorted site tuples: sum_i C(s_i, i+1)."""
    N = sites_sorted.shape[-1]
    r = np.zeros(sites_sorted.shape[:-1], dtype=np.int64)
    for i in range(N):
        r += C[sites_sorted[..., i], i + 1]
    return r


# ----------------------------------------------------------------------------
# model
# ----------------------------------------------------------------------------

@dataclass
class AnyonModel:
    """Identical abelian anyons (charge eta=1) with statistics theta = pi*p/M.

    M is the number of sheets (wave-function components). Use theta=0, M=1
    for hard-core bosons. convention: 'paper' (arXiv:2206.14730, App. F
    string-net phases) or 'ref' (Wen-Dagotto style, Sec. II.B sketch).
    """
    Lx: int
    Ly: int
    N: int
    theta: float = math.pi / 2
    M: int = 2
    convention: str = "paper"
    t: float = 1.0
    r_col: float = 1.0  # multiplies y-hops (used for HCB ladder isotropy)

    configs: np.ndarray = field(init=False)
    C: np.ndarray = field(init=False)

    def __post_init__(self):
        assert self.convention in ("paper", "ref")
        ns = self.Lx * self.Ly
        self.C = binom_table(ns, self.N)
        self.configs = enumerate_configs(ns, self.N)
        # consistency of anyon model on the torus: e^{2 i theta N} = 1
        phase = 2 * self.theta * self.N / (2 * math.pi)
        if abs(phase - round(phase)) > 1e-12:
            raise ValueError("N anyons with statistics theta must satisfy "
                             "exp(2 i theta N) = 1 on the torus")

    # -- helpers ----------------------------------------------------------
    @property
    def nsites(self) -> int:
        return self.Lx * self.Ly

    @property
    def nconf(self) -> int:
        return len(self.configs)

    @property
    def dim(self) -> int:
        return self.nconf * self.M

    @property
    def cut_extra(self) -> complex:
        """Extra per-cut-crossing factor (paper convention).

        App. D: for theta = pi*p/M with p odd the anyon model is Z_M^(n+1/2),
        whose F-move string contributes exp(-i pi N / M) per crossing; for p
        even (Z_M^n, e.g. hard-core bosons at theta=0) the F-moves are trivial.
        """
        if self.convention != "paper":
            return 1.0 + 0.0j
        p = round(self.theta * self.M / math.pi)
        if p % 2 == 0:
            return 1.0 + 0.0j
        return np.exp(-1j * math.pi * self.N / self.M)

    def xy(self, sites: np.ndarray):
        return sites // self.Ly, sites % self.Ly

    # -- single-anyon hop terms (vectorized over all configs) --------------
    def _hops(self, direction: str):
        """Forward hop terms for all configs and all anyons.

        Yields tuples (src_conf, dst_conf, amp0, dsheet, sheet_phase_exp)
        as flat arrays, where the amplitude for sheet f is
        amp0 * exp(2i*theta*f)**sheet_phase_exp and the final sheet is
        (f + dsheet) mod M.
        """
        Lx, Ly, th = self.Lx, self.Ly, self.theta
        cfg = self.configs
        xs, ys = self.xy(cfg)                      # (Nc, N)
        Nc, N = cfg.shape
        src_all, dst_all, amp_all, dsh_all, spe_all = [], [], [], [], []
        for k in range(N):
            x, y = xs[:, k], ys[:, k]
            if direction == "x":
                nx2, ny2 = (x + 1) % Lx, y
            else:
                nx2, ny2 = x, (y + 1) % Ly
            tgt = nx2 * Ly + ny2
            blocked = (cfg == tgt[:, None]).any(axis=1)
            ok = ~blocked
            if not ok.any():
                continue
            xo, yo = xs[ok], ys[ok]                # (n, N)
            xk, yk = x[ok], y[ok]
            amp = np.ones(ok.sum(), dtype=np.complex128)
            dsheet = 0
            spe = np.zeros(ok.sum(), dtype=np.int64)   # exponent of exp(2i th f)
            if direction == "x":
                n_iup = ((xo == xk[:, None]) & (yo > yk[:, None])).sum(axis=1)
                n_jdn = ((xo == nx2[ok][:, None]) & (yo < yk[:, None])).sum(axis=1)
                amp *= np.exp(1j * th * (n_iup - n_jdn))
                crossB = xk == Lx - 1
                if crossB.any():
                    amp[crossB] *= self.cut_extra
                    # sheet shift applies only where crossB; handle by split
                    # (emit two batches: bulk and cross)
                    for cross_flag, dsh in ((~crossB, 0), (crossB, -1)):
                        if not cross_flag.any():
                            continue
                        newc = cfg[ok][cross_flag].copy()
                        newc[:, k] = tgt[ok][cross_flag]
                        newc.sort(axis=1)
                        src_all.append(np.flatnonzero(ok)[cross_flag])
                        dst_all.append(rank_configs(newc, self.C))
                        amp_all.append(amp[cross_flag])
                        dsh_all.append(np.full(cross_flag.sum(), dsh))
                        spe_all.append(spe[cross_flag])
                    continue
            else:
                crossA = yk == Ly - 1
                if crossA.any():
                    ca = crossA
                    if self.convention == "paper":
                        nL = (xo[ca] < xk[ca][:, None]).sum(axis=1)
                        nR = (xo[ca] > xk[ca][:, None]).sum(axis=1)
                        amp[ca] *= np.exp(1j * th * (nL - nR)) * self.cut_extra
                    else:
                        nL = (xo[ca] < xk[ca][:, None]).sum(axis=1)
                        nsame = ((xo[ca] == xk[ca][:, None]) &
                                 (yo[ca] != yk[ca][:, None])).sum(axis=1)
                        amp[ca] *= np.exp(2j * th * nL) * np.exp(1j * th * nsame)
                    spe[ca] = 1                      # factor exp(2 i theta f)
            newc = cfg[ok].copy()
            newc[:, k] = tgt[ok]
            newc.sort(axis=1)
            src_all.append(np.flatnonzero(ok))
            dst_all.append(rank_configs(newc, self.C))
            amp_all.append(amp)
            dsh_all.append(np.full(ok.sum(), dsheet))
            spe_all.append(spe)
        return (np.concatenate(src_all), np.concatenate(dst_all),
                np.concatenate(amp_all), np.concatenate(dsh_all),
                np.concatenate(spe_all))

    # -- Hamiltonian -------------------------------------------------------
    def hamiltonian(self) -> sp.csr_matrix:
        """Sparse H = -t sum (T_x + r_col*T_y + h.c.) on the full basis.

        Basis index = conf_index * M + f.
        """
        M = self.M
        rows, cols, vals = [], [], []
        for direction, weight in (("x", self.t), ("y", self.t * self.r_col)):
            src, dst, amp, dsh, spe = self._hops(direction)
            for f in range(M):
                fac = amp * np.exp(2j * self.theta * f) ** spe
                rows.append(dst * M + (f + dsh) % M)
                cols.append(src * M + f)
                vals.append(-weight * fac)
        rows = np.concatenate(rows)
        cols = np.concatenate(cols)
        vals = np.concatenate(vals)
        A = sp.coo_matrix((vals, (rows, cols)), shape=(self.dim, self.dim)).tocsr()
        return (A + A.getH()).tocsr()

    # -- translation operators --------------------------------------------
    def translate_config(self, sites: np.ndarray, dx: int, dy: int) -> np.ndarray:
        x, y = self.xy(sites)
        return np.sort(((x + dx) % self.Lx) * self.Ly + (y + dy) % self.Ly, axis=-1)

    def Tx_action(self, sites: np.ndarray):
        """T_x on a single config: returns (new_sites, monomial).

        monomial = (dsheet, phases[M]) meaning |f> -> phases[f] |(f+dsheet)%M>.
        Paper (App. E): FDAV factor is Utilde_x per cut-B crossing, no string
        phases: Utilde_x = cut_extra * (shift f -> f-1).
        """
        x, y = self.xy(sites)
        nB = int((x == self.Lx - 1).sum())
        phases = np.full(self.M, self.cut_extra ** nB, dtype=np.complex128)
        return self.translate_config(sites, 1, 0), (-nB % self.M, phases)

    def Ty_action(self, sites: np.ndarray):
        """T_y on a single config: returns (new_sites, monomial).

        Each anyon crossing cut A braids with all non-crossing anyons in other
        columns (+theta per anyon to its left, -theta per anyon to its right in
        the paper convention) and contributes cut_extra * exp(2 i theta f).
        """
        x, y = self.xy(sites)
        cross = y == self.Ly - 1
        nA = int(cross.sum())
        amp = 1.0 + 0.0j
        for xk in x[cross]:
            nL = int((x[~cross] < xk).sum())
            nR = int((x[~cross] > xk).sum())
            if self.convention == "paper":
                amp *= np.exp(1j * self.theta * (nL - nR)) * self.cut_extra
            else:
                nsame = int((x[~cross] == xk).sum())
                amp *= np.exp(2j * self.theta * nL) * np.exp(1j * self.theta * nsame)
        f = np.arange(self.M)
        phases = amp * np.exp(2j * self.theta * f) ** nA
        return self.translate_config(sites, 0, 1), (0, phases)

    def _T_matrix(self, action) -> sp.csr_matrix:
        M = self.M
        rows = np.empty(self.dim, dtype=np.int64)
        cols = np.empty(self.dim, dtype=np.int64)
        vals = np.empty(self.dim, dtype=np.complex128)
        for i, sites in enumerate(self.configs):
            new_sites, (dsh, phases) = action(sites)
            j = int(rank_configs(new_sites, self.C))
            for f in range(M):
                idx = i * M + f
                rows[idx] = j * M + (f + dsh) % M
                cols[idx] = idx
                vals[idx] = phases[f]
        return sp.coo_matrix((vals, (rows, cols)),
                             shape=(self.dim, self.dim)).tocsr()

    def Tx_matrix(self) -> sp.csr_matrix:
        return self._T_matrix(self.Tx_action)

    def Ty_matrix(self) -> sp.csr_matrix:
        return self._T_matrix(self.Ty_action)


def hcb_model(Lx: int, Ly: int, N: int, t: float = 1.0,
              r_col: float = 1.0) -> AnyonModel:
    """Hard-core bosons: theta = 0, single sheet."""
    return AnyonModel(Lx, Ly, N, theta=0.0, M=1, t=t, r_col=r_col)
