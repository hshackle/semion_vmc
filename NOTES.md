# Exact diagonalization of semions on a lattice

Reproduction of the abelian-anyon (semion) results of

> N. Kirchner, D. Millar, B. M. Ayeni, A. Smith, J. K. Slingerland, F. Pollmann,
> *Numerical simulation of non-abelian anyons*, arXiv:2206.14730
> (Phys. Rev. B 107, 195129 (2023)).

The paper develops a fusion-diagram algorithm for simulating tight-binding
Hamiltonians of arbitrary (abelian and non-abelian) anyons on a periodic
square lattice. Here we implement only the **abelian specialization**
(paper Sec. II.B and App. D), applied to **semions** (exchange phase
θ = π/2), and reproduce the two semion results of Sec. VI:

1. **Energy level-spacing statistics** (Fig. 6): N = 4 semions on an 8×8
   torus show Poisson statistics in the (0,0) momentum sector, GOE in
   (0, 2π/L), and GUE in (2π/L, 4π/L) — i.e. free semions behave like an
   *interacting, time-reversal-breaking* system, with an antiunitary
   (reflection × TR) symmetry surviving on the k_x = 0 axis.
2. **Quench dynamics** (Fig. 8): N = 4 particles released from a zigzag
   pattern in the middle of a 30×2 ladder. Free fermions show persistent
   interference fringes; hard-core bosons and semions thermalize (the
   column density becomes homogeneous at large times).

## Model

Hard-core semions hop on an L_x × L_y torus:

    H = −t Σ_i ( T_{r_i, r_i+e_x} + T_{r_i, r_i+e_y} + h.c. )

Semions have no Fock space (the total number N must be **even**:
e^{2iθN} = 1 must hold on the torus), so the Hamiltonian is defined
directly through translation operators. On the torus the Hilbert space is

    (hard-core configurations of N semions) ⊗ (sheet index f ∈ {0, 1}),

where the two "sheets" are the wave-function components labelled by the
anyonic charge (1 or s) threading the torus — the origin of the 2-fold
topological ground-state degeneracy of the semion sector.

## Amplitude rules (paper convention, App. D)

Sites (x, y), x ∈ 0..L_x−1, y ∈ 0..L_y−1. Cut A lies between rows
L_y−1 and 0; cut B between columns L_x−1 and 0. With θ = π/2, M = 2,
and the same-charge anyon count N:

| move | amplitude | sheet action |
|---|---|---|
| +x bulk | exp[iθ(n_i↑ − n_j↓)] | — |
| +x across cut B | exp[iθ(n_i↑ − n_j↓)] · e^{−iπN/M} | f → f−1 (mod 2) |
| +y bulk | 1 | — |
| +y across cut A | exp[iθ(n_L − n_R)] · e^{−iπN/M} | × e^{2iθf} = (−1)^f |

n_i↑ = anyons in the origin column strictly above the mover; n_j↓ = anyons
in the destination column strictly below it; n_L / n_R = anyons in columns
strictly left / right of the mover (same-column anyons do **not** braid
during a cut-A crossing). Negative moves are hermitian conjugates.
The factor e^{−iπN/M} is the F-move string of the ℤ₂^{(1/2)} (semion)
fusion category (App. D, Eq. τ₁/ρ₁); it is +1 for N ≡ 0 (mod 4).

These rules reproduce the paper's holonomies for the ordered reference
configuration (verified exactly in `tests/run_checks.py`):

    τ₁ = e^{iθ(N−1)} e^{−iπN/M} X          (x-loop of anyon 1)
    ρ₁ = e^{−iθ(N−1)} e^{−iπN/M} diag(1, e^{2iθ})   (y-loop)
    τ_{j+1} = e^{−2iθ} τ_j ,  ρ_{j+1} = e^{2iθ} ρ_j ,
    τ ρ τ⁻¹ ρ⁻¹ = e^{−2iθ(N−1)}   (consistent only for N even)

### Convention caveat

The simpler "flux-string" convention sketched in the paper's Sec. II.B
(following Refs. [43, 45] there) differs from the paper's own convention by
external fluxes through the torus handles (an extra factor
e^{−iθ(N−1)} e^{−iπN/M} per cut-A crossing and e^{−iπN/M} per cut-B
crossing). We implemented both (`convention="paper" | "ref"`); their **full
spectra genuinely differ** (max deviation ~0.94 t for 4 semions on 4×4) and
the momentum sectors are shuffled — so reproducing the paper's
sector-resolved Fig. 6 requires the paper convention, which is fixed
physically by requiring trivial Wilson-loop phases in the doubled-semion
string-net realization (paper App. F).

## Translation operators and momentum sectors

T_x (T_y) translate all anyons one site in x (y). On the sheet space
(paper App. E):

- T_x: multiply by (e^{−iπN/M})^{n_B} and shift f → f − n_B, where n_B is
  the number of anyons in the last column; no braid phases arise.
- T_y: each anyon crossing cut A contributes e^{iθ(n_L − n_R)} counting
  only **non-crossing** anyons (crossing anyons do not braid each other),
  times e^{−iπN/M} (−1)^f.

These satisfy T_x^{L_x} = T_y^{L_y} = 1, [T_x, T_y] = [H, T_x] = [H, T_y] = 0
(all verified numerically to 1e-10). Momentum states are built per
translation orbit by projecting the M sheet seeds with
Σ_{j,l} e^{i(jk_x + lk_y)} T_y^l T_x^j and orthonormalizing via SVD;
the sector bases are exact isometries and exactly block-diagonalize H
(verified: union of all sector spectra = full spectrum on 4×4 and 4×2).

## Numerical details

- 8×8, N = 4 semions: dim = 2·C(64,4) = 1,270,752; 64 momentum sectors of
  dimension ≈ 19,900. Each sector Hamiltonian is built as V†HV (sparse,
  blocked) and fully diagonalized densely (complex Hermitian, ~6.3 GB).
- HCB comparison (9×9, N = 4, column coupling r_col = 0.5 as in the paper's
  Fig. 6): same machinery with θ = 0, M = 1.
- Level statistics: consecutive-gap ratios r_n = min(δ_n, δ_{n−1})/max(...),
  histogrammed against the Poisson / GOE / GUE surmises quoted in the paper.
- Quench (30×2 ladder): initial state = zigzag sites (13,0),(14,1),(15,0),(16,1)
  in an equal superposition of both sheets; evolution by Krylov
  `expm_multiply`; fermions are evolved exactly as free particles.
  On the two-leg ladder the two y-bonds per rung add coherently; for HCBs
  this doubles the rung coupling, compensated by r_col = 0.5 (for semions
  the two terms differ — bulk vs cut A — so no compensation is applied),
  exactly as discussed in the paper.

## Results

### Level-spacing statistics (paper Fig. 6) — `results/fig_lss.png`

Mean consecutive-gap ratios r̄ (reference values: Poisson 0.3863,
GOE 0.5307, GUE 0.6027; two superposed GOE blocks ≈ 0.42):

| sector | semions 8×8 (dim) | HCB 9×9 (dim) | paper's assignment |
|---|---|---|---|
| (0, 0)         | 0.3906 (19 902) | 0.3966 (20 540) | Poisson / Poisson |
| (0, 2π/L)      | 0.5310 (19 840) | 0.4230 (20 540) | GOE / intermediate (2 blocks) |
| (2π/L, 4π/L)   | 0.6032 (19 840) | 0.5269 (20 540) | GUE / GOE |

The P(r) histograms trace the corresponding curves across the whole range,
reproducing the paper's Fig. 6 panels quantitatively. Physics: the free-semion
Hamiltonian breaks time reversal and all individual reflections (they reverse
braiding orientation), leaving GUE statistics in a generic momentum sector;
on the k_x = 0 axis the antiunitary combination (x-reflection × TR) survives,
giving GOE; at (0,0) four such combinations remain unresolved and the mixture
of symmetry blocks yields Poisson. HCBs are TR symmetric: GOE in the generic
sector, two superposed GOE blocks (r̄ ≈ 0.42) on the k_x = 0 axis, Poisson
at (0,0).

### Quench dynamics (paper Fig. 8) — `results/fig_quench.png`

Column densities ⟨n_i(t)⟩ on the 30×2 ladder from the zigzag initial state:

- **Fermions** (free): wave packets pass through each other; interference
  fringes and wrap-around recurrences (bright center revivals at t ≈ 16, 31,
  47) persist undamped — no thermalization.
- **HCBs** (r_col = 0.5): fronts cannot pass through; interference washes
  out and the density is nearly homogeneous for t ≳ 20.
- **Semions**: qualitatively like HCBs — ballistic fronts and first
  recurrences remain visible but decay steadily toward a homogeneous
  distribution, i.e. free semions relax like an interacting system.

Both observations match the paper's conclusion: tight-binding semions,
whose Hamiltonian contains nothing but statistics, exhibit level repulsion
and thermalizing quench dynamics — statistics alone acts as an interaction.

### Reproduction fidelity caveats

- The paper's histograms include Fibonacci and Ising anyons (non-abelian),
  which are outside the scope of this abelian implementation.
- Bin-level agreement with the paper's markers is excellent by eye; the
  underlying eigenvalue data of the paper (Zenodo record 6777951) is
  access-restricted, so the comparison is against the published figures.
