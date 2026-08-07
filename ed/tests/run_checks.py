"""Consistency checks for the semion ED implementation (small lattices).

Run:  python -m tests.run_checks
"""

import sys
import math

import numpy as np
import scipy.sparse as sp
import scipy.linalg as sla

sys.path.insert(0, ".")
from semion_ed.core import AnyonModel, hcb_model, rank_configs  # noqa: E402
from semion_ed.momentum import orbit_reps, sector_basis, sector_hamiltonian  # noqa: E402

FAIL = 0


def check(name, ok, detail=""):
    global FAIL
    print(f"[{'PASS' if ok else 'FAIL'}] {name} {detail}")
    if not ok:
        FAIL += 1


# ---------------------------------------------------------------------------
# scalar re-implementation of the hop rules (independent of the vectorized one)
# ---------------------------------------------------------------------------

def scalar_hop(model, sites, k, direction):
    """Forward hop (+x or +y) of anyon k. Returns (new_sites_sorted, amp_f, dsheet)
    where amp_f is an array over sheets f, or None if blocked."""
    Lx, Ly, th, M = model.Lx, model.Ly, model.theta, model.M
    x, y = sites // Ly, sites % Ly
    xk, yk = x[k], y[k]
    if direction == "x":
        tx, ty = (xk + 1) % Lx, yk
    else:
        tx, ty = xk, (yk + 1) % Ly
    tgt = tx * Ly + ty
    if tgt in sites:
        return None
    amp = np.ones(M, dtype=complex)
    dsheet = 0
    if direction == "x":
        n_iup = sum(1 for j in range(len(sites)) if x[j] == xk and y[j] > yk)
        n_jdn = sum(1 for j in range(len(sites)) if x[j] == tx and y[j] < yk)
        amp *= np.exp(1j * th * (n_iup - n_jdn))
        if xk == Lx - 1:
            amp *= model.cut_extra
            dsheet = -1
    else:
        if yk == Ly - 1:
            if model.convention == "paper":
                nL = sum(1 for j in range(len(sites)) if x[j] < xk)
                nR = sum(1 for j in range(len(sites)) if x[j] > xk)
                amp *= np.exp(1j * th * (nL - nR)) * model.cut_extra
            else:
                nL = sum(1 for j in range(len(sites)) if x[j] < xk)
                ns = sum(1 for j in range(len(sites))
                         if x[j] == xk and j != k)
                amp *= np.exp(2j * th * nL) * np.exp(1j * th * ns)
            amp *= np.exp(2j * th * np.arange(M))
    new = sites.copy()
    new[k] = tgt
    return np.sort(new), amp, dsheet


def scalar_hamiltonian(model):
    """Dense H built from the scalar rules."""
    M, dim = model.M, model.dim
    H = np.zeros((dim, dim), dtype=complex)
    for i, sites in enumerate(model.configs):
        for k in range(model.N):
            for direction, w in (("x", model.t), ("y", model.t * model.r_col)):
                out = scalar_hop(model, sites.copy(), k, direction)
                if out is None:
                    continue
                new, amp, dsh = out
                j = int(rank_configs(new, model.C))
                for f in range(M):
                    H[j * M + (f + dsh) % M, i * M + f] += -w * amp[f]
    return H + H.conj().T


def path_monomial(model, sites0, moves):
    """Apply a sequence of single-anyon moves; moves = list of (site, step)
    with step in {'+x','-x','+y','-y'}. Negative steps use the conjugate of the
    corresponding forward hop. Returns (final_sites, dsheet, amp_f)."""
    M = model.M
    sites = np.array(sorted(sites0), dtype=np.int64)
    dsheet, amp = 0, np.ones(M, dtype=complex)
    Ly = model.Ly
    for site, step in moves:
        k = int(np.where(sites == site)[0][0])
        if step in ("+x", "+y"):
            new, a, dsh = scalar_hop(model, sites.copy(), k, step[1])
            # amp for current sheet fifo: |f> has accumulated dsheet already;
            # a is indexed by the sheet the state is currently in.
            f = (np.arange(M) + dsheet) % M
            amp = amp * a[f]
            dsheet = (dsheet + dsh) % M
            sites = new
        else:
            # inverse of forward hop from the destination site
            x, y = site // Ly, site % Ly
            if step == "-x":
                src = ((x - 1) % model.Lx) * Ly + y
            else:
                src = x * Ly + (y - 1) % Ly
            tmp = sites.copy()
            tmp[k] = src
            tmp = np.sort(tmp)
            kk = int(np.where(tmp == src)[0][0])
            new, a, dsh = scalar_hop(model, tmp.copy(), kk, step[1])
            assert int(rank_configs(new, model.C)) == int(rank_configs(sites, model.C))
            # inverse monomial: |f> -> conj(a[(f-dsh)%M]) |(f-dsh)%M>
            f = (np.arange(M) + dsheet - dsh) % M
            amp = amp * np.conj(a[f])
            dsheet = (dsheet - dsh) % M
            sites = tmp
    return sites, dsheet, amp


# ---------------------------------------------------------------------------

def run_model_checks(model, label, full_spectrum_check=True):
    H = model.hamiltonian()
    check(f"{label}: H hermitian", abs(H - H.getH()).max() < 1e-12)
    Hs = scalar_hamiltonian(model)
    check(f"{label}: vectorized H == scalar H",
          np.abs(H.toarray() - Hs).max() < 1e-12)
    Tx, Ty = model.Tx_matrix(), model.Ty_matrix()
    I = sp.identity(model.dim, dtype=complex, format="csr")
    check(f"{label}: Tx unitary", abs(Tx @ Tx.getH() - I).max() < 1e-12)
    check(f"{label}: Ty unitary", abs(Ty @ Ty.getH() - I).max() < 1e-12)
    Px = I
    for _ in range(model.Lx):
        Px = Tx @ Px
    Py = I
    for _ in range(model.Ly):
        Py = Ty @ Py
    check(f"{label}: Tx^Lx == 1", abs(Px - I).max() < 1e-10)
    check(f"{label}: Ty^Ly == 1", abs(Py - I).max() < 1e-10)
    check(f"{label}: [Tx,Ty] == 0", abs(Tx @ Ty - Ty @ Tx).max() < 1e-10)
    check(f"{label}: [H,Tx] == 0", abs(H @ Tx - Tx @ H).max() < 1e-10)
    check(f"{label}: [H,Ty] == 0", abs(H @ Ty - Ty @ H).max() < 1e-10)

    # momentum sectors: dims add up, eigenvalues reproduce the full spectrum
    canon, reps = orbit_reps(model)
    evals = []
    dims = 0
    ok_inv = True
    ok_iso = True
    for jx in range(model.Lx):
        for jy in range(model.Ly):
            V = sector_basis(model, jx, jy, canon, reps)
            n = V.shape[1]
            dims += n
            if n == 0:
                continue
            ok_iso &= abs((V.getH() @ V) - sp.identity(n)).max() < 1e-8
            Hk = sector_hamiltonian(H, V)
            ok_inv &= (abs(H @ V - V @ sp.csr_matrix(Hk)).max() < 1e-8)
            evals.append(np.linalg.eigvalsh(Hk))
            # sector states are Tx/Ty eigenstates with the right eigenvalue
            lx = np.exp(-2j * np.pi * jx / model.Lx)
            ly = np.exp(-2j * np.pi * jy / model.Ly)
            ok_iso &= abs(Tx @ V - lx * V).max() < 1e-8
            ok_iso &= abs(Ty @ V - ly * V).max() < 1e-8
    check(f"{label}: sector dims sum to full dim", dims == model.dim,
          f"({dims} vs {model.dim})")
    check(f"{label}: sector bases are isometries/eigenstates", ok_iso)
    check(f"{label}: sectors invariant under H", ok_inv)
    if full_spectrum_check:
        ev_full = np.linalg.eigvalsh(H.toarray())
        ev_sec = np.sort(np.concatenate(evals))
        check(f"{label}: union of sector spectra == full spectrum",
              np.abs(ev_full - ev_sec).max() < 1e-8)
    return H


def main():
    # ----- semions, 4x4, N=2 -------------------------------------------------
    m = AnyonModel(4, 4, 2, convention="paper")
    run_model_checks(m, "semion 4x4 N=2 (paper)")

    # braid check: one semion around another, contractible CCW loop -> -1
    A, B = 1 * 4 + 1, 1 * 4 + 2      # (x=1,y=1), (x=1,y=2); site = x*Ly+y
    loop = [(1 * 4 + 1, "+x"), (2 * 4 + 1, "+y"), (2 * 4 + 2, "+y"),
            (2 * 4 + 3, "-x"), (1 * 4 + 3, "-x"), (0 * 4 + 3, "-y"),
            (0 * 4 + 2, "-y"), (0 * 4 + 1, "+x")]
    sites, dsh, amp = path_monomial(m, [A, B], loop)
    check("braid: CCW loop around one semion = e^{2 i theta} = -1",
          dsh == 0 and np.allclose(amp, -1), f"amp={amp}")

    # tau_1 / rho_1 for the ordered two-anyon configuration (0,0), (1,1)
    s0 = [0 * 4 + 0, 1 * 4 + 1]
    tau_moves = [((x % 4) * 4 + 0, "+x") for x in range(4)]
    sites, dsh, amp = path_monomial(m, s0, tau_moves)
    th, N, M = m.theta, m.N, m.M
    expect = np.exp(1j * th * (N - 1)) * np.exp(-1j * np.pi * N / M)
    check("tau_1 = e^{i th (N-1)} e^{-i pi N/M} X",
          dsh == 1 and np.allclose(amp, expect), f"amp={amp}, expect={expect}")
    rho_moves = [(0 * 4 + (y % 4), "+y") for y in range(4)]
    sites, dsh, amp = path_monomial(m, s0, rho_moves)
    expect = (np.exp(-1j * th * (N - 1)) * np.exp(-1j * np.pi * N / M)
              * np.exp(2j * th * np.arange(M)))
    check("rho_1 = e^{-i th (N-1)} e^{-i pi N/M} diag(1, e^{2 i th})",
          dsh == 0 and np.allclose(amp, expect), f"amp={amp}, expect={expect}")

    # ----- semions, 4x4, N=4 -------------------------------------------------
    m4 = AnyonModel(4, 4, 4, convention="paper")
    run_model_checks(m4, "semion 4x4 N=4 (paper)")

    # ----- semions, ladder 4x2, N=2 ------------------------------------------
    ml = AnyonModel(4, 2, 2, convention="paper")
    run_model_checks(ml, "semion 4x2 N=2 (paper)")

    # ----- ref convention: internal consistency + spectrum comparison --------
    mr = AnyonModel(4, 4, 2, convention="ref")
    Hr = mr.hamiltonian()
    check("semion 4x4 N=2 (ref): H hermitian", abs(Hr - Hr.getH()).max() < 1e-12)
    Hs = scalar_hamiltonian(mr)
    check("semion 4x4 N=2 (ref): vectorized == scalar",
          np.abs(Hr.toarray() - Hs).max() < 1e-12)
    ev_p = np.linalg.eigvalsh(m.hamiltonian().toarray())
    ev_r = np.linalg.eigvalsh(Hr.toarray())
    same = np.abs(ev_p - ev_r).max() < 1e-8
    print(f"[INFO] paper vs ref convention full spectra "
          f"{'agree' if same else 'differ'} (N=2, 4x4); max dev "
          f"{np.abs(ev_p - ev_r).max():.2e}")

    # ----- odd N rejected ----------------------------------------------------
    try:
        AnyonModel(4, 4, 3)
        check("odd semion number rejected", False)
    except ValueError:
        check("odd semion number rejected", True)

    # ----- HCB sanity --------------------------------------------------------
    hb = hcb_model(4, 3, 1)
    Hb = hb.hamiltonian().toarray()
    kx = 2 * np.pi * np.arange(4) / 4
    ky = 2 * np.pi * np.arange(3) / 3
    ev_exp = np.sort((-2 * np.cos(kx)[:, None] - 2 * np.cos(ky)[None, :]).ravel())
    check("HCB single particle == tight-binding dispersion",
          np.abs(np.linalg.eigvalsh(Hb) - ev_exp).max() < 1e-10)
    hb2 = hcb_model(4, 4, 3)
    run_model_checks(hb2, "HCB 4x4 N=3")

    print()
    if FAIL:
        print(f"{FAIL} CHECK(S) FAILED")
        sys.exit(1)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
