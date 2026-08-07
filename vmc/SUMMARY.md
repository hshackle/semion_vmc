# VMC spec: neural quantum states for semions on the torus

Goal: variational Monte Carlo with a neural quantum state (NQS) for the
tight-binding semion Hamiltonian implemented and validated in `../ed/`,
with the **anyonic translational symmetries built into the ansatz** so that
states carry sharp momentum quantum numbers (k_x, k_y). Ground truth for
every validation step comes from the exact diagonalization in `../ed/`
(Kirchner et al., arXiv:2206.14730, abelian specialization; see
`../ed/NOTES.md` for the algorithm and its validation).

## 1. System specification

- L_x × L_y square lattice, periodic in both directions. Site index
  `s = x*Ly + y`, x ∈ 0..L_x−1, y ∈ 0..L_y−1. Cut A lies between rows
  L_y−1 and 0; cut B between columns L_x−1 and 0.
- N identical hard-core semions (exchange phase θ = π/2). **N must be even**
  (torus consistency: e^{2iθN} = 1).
- Hilbert space: `(hard-core configurations c of N particles) ⊗ (sheet f ∈ {0,1})`.
  The sheet is the topological degree of freedom (anyonic charge threading
  the torus); dim = 2·C(L_x·L_y, N).
- Hamiltonian: H = −t Σ_i (T_{r_i,r_i+e_x} + T_{r_i,r_i+e_y} + h.c.), t = 1.
- **Convention: the "paper" convention only** (`convention="paper"` in the ED
  code). The alternative "ref" convention is a different physical flux
  choice with a different spectrum — do not mix them.

Amplitude rules (matrix elements of the forward hops; negative hops are
hermitian conjugates). With cut factor `cN = exp(−iπN/2)` (= +1 for N ≡ 0 mod 4):

| move of one particle | amplitude | sheet |
|---|---|---|
| (x,y) → (x+1,y), x < L_x−1 | exp[iθ(n_i↑ − n_j↓)] | f unchanged |
| (L_x−1,y) → (0,y) (cut B) | exp[iθ(n_i↑ − n_j↓)] · cN | f → f ⊕ 1 |
| (x,y) → (x,y+1), y < L_y−1 | 1 | f unchanged |
| (x,L_y−1) → (x,0) (cut A) | exp[iθ(n_L − n_R)] · cN · (−1)^f | f unchanged |

n_i↑ = particles in the origin column strictly above the mover; n_j↓ =
particles in the destination column strictly below it; n_L / n_R = particles
in columns strictly left / right of the mover (same-column particles do not
contribute at cut A). Hops onto occupied sites are forbidden (hard core).

**Do not re-derive these rules — import them.** `../ed/semion_ed/core.py`
provides `AnyonModel(Lx, Ly, N, convention="paper")` with vectorized hop
generation (`_hops`), and `../ed/tests/run_checks.py` contains an
independent scalar reference (`scalar_hop`). The local energy of a sample
(c, f) is a sum over ≤ 4N connected states (c′, f′):

    E_loc(c,f) = Σ H_{(c′,f′),(c,f)} · ψ(c′,f′) / ψ(c,f)

## 2. Anyonic translation symmetry

The operators T_x, T_y translate every particle one site in +x / +y. They
are **monomial** operators: each basis state maps to a single basis state
times a computable phase, with a possible sheet flip:

- T_x |c, f⟩ = (cN)^{n_B(c)} |c + e_x, f ⊕ (n_B(c) mod 2)⟩, where n_B(c) =
  number of particles in column L_x−1. No braid phases arise.
- T_y |c, f⟩ = φ_y(c) · [cN · (−1)^f]^{n_A(c)} |c + e_y, f⟩, where n_A(c) =
  number of particles in row L_y−1 and φ_y(c) = Π over crossing particles of
  exp[iθ(n_L^{nc} − n_R^{nc})], counting only **non-crossing** particles
  (crossing particles do not braid with each other).

These are implemented as `AnyonModel.Tx_action(sites)` / `Ty_action(sites)`,
returning `(new_sites, (dsheet, phases[2]))` meaning |f⟩ → phases[f] |f+dsheet⟩.
Exact operator identities (all verified in `../ed`):

    T_x^{L_x} = T_y^{L_y} = 1,   [T_x, T_y] = [H, T_x] = [H, T_y] = 0.

So the symmetry group is an honest abelian group Z_{L_x} × Z_{L_y} acting by
monomial matrices — a momentum-projected ansatz is exact, no projective
complications. Momenta are quantized as k_x = 2πj/L_x, k_y = 2πl/L_y.

### Symmetric ansatz

Let ψ_net(c, f) be any complex-valued network. Define the group element
g = (j, l) acting as U_g = T_y^l T_x^j, with U_g |c,f⟩ = ω_g(c,f) |c_g, f_g⟩
(compose `Tx_action`/`Ty_action`; ω is a phase, f_g = f ⊕ (sheet flips)).
The momentum-(k_x, k_y) state is the projection

    |ψ_k⟩ = Σ_{j=0}^{L_x−1} Σ_{l=0}^{L_y−1} e^{i(j k_x + l k_y)} U_g |ψ_net⟩,

identical in form to the ED momentum states (`../ed/semion_ed/momentum.py`).
In amplitude form:

    ψ_k(c, f) = Σ_g e^{i(j k_x + l k_y)} · ω_g(c₀, f₀) · ψ_net(c₀, f₀)
                where (c₀, f₀) = U_g^{-1}(c, f);

equivalently, enumerate forward images of each (c₀,f₀) — the group is
abelian and U_g^{-1} = U_{(L_x−j) mod L_x, (L_y−l) mod L_y}. This is the
standard GCNN/symmetrized-NQS construction, except the characters are
multiplied by the **exactly known anyonic phases ω_g and sheet flips** —
this is the whole point of the exercise. Cost: L_x·L_y network evaluations
per amplitude (64 for 8×8); amortize by batching all group images.

An equivalent covariance check (use it as a unit test): a correct ψ_k obeys

    ψ_k(T_x(c,f)) = e^{i k_x} · ω_x(c,f) · ψ_k(c,f),  and similarly for T_y,

i.e. amplitudes on a translation orbit are fixed by one representative up to
exactly computable phases.

## 3. Ansatz and sampling recommendations

- **Complex amplitudes are mandatory.** The generic momentum sector is GUE
  class (no antiunitary symmetry); there is no real/Marshall gauge. Use a
  holomorphic-complex network or separate log-modulus and phase heads.
- Treat the sheet as a 2-component output: network maps the occupation
  configuration c (e.g. an L_x×L_y binary image, periodic convolutions are
  natural) to two complex amplitudes (ψ(c,0), ψ(c,1)). Then either sample
  (c, f) jointly or sample c from Σ_f |ψ_k(c,f)|² and treat f exactly.
- Metropolis proposals: single-particle hops to empty neighbors (matches H
  connectivity); add occasional long-range particle moves for ergodicity.
  The sheet needs no proposal if f is summed exactly.
- Start from small systems where **exact summation over the full basis** is
  feasible (4×4 N=2: dim 240; 4×2 N=2: dim 56) to debug E_loc and the
  symmetrization with zero sampling noise.
- Watch for: the two sheets double every ED degeneracy pattern; at generic k
  the spectrum is complex (level repulsion GUE) and optimization can be
  stiff — SR / minSR with complex parameters recommended.
- The spectra are symmetric under E → −E (bipartite hopping), so an
  unconverged state stuck near E ≈ 0 is maximally wrong, not "halfway".

## 4. Validation targets (exact, from ../ed, t = 1, paper convention)

Global ground states (all sit in the (k_x,k_y) = (0,0) sector):

| system | dim | E₀ |
|---|---|---|
| 4×2, N=2 | 56 | −5.8760125453 |
| 4×4, N=2 | 240 | −7.0389129320 |
| 4×4, N=4 | 3 640 | −11.3772922185 |
| 6×6, N=4 | 117 810 | −13.9159290464 |
| 8×8, N=4 | 1 270 752 | −14.8308385845 |

Sector-resolved ground-state energies (momenta in units 2π/L; sector dims
in parentheses):

| system | (0,0) | (0,1) | (1,2) |
|---|---|---|---|
| 4×2, N=2 | −5.8760125453 (7) | −3.6955181300 (7) | −4.9506193271 (7) |
| 4×4, N=2 | −7.0389129320 (15) | −6.0152249412 (15) | −2.6131259298 (15) |
| 4×4, N=4 | −11.3772922185 (238) | −9.8507072764 (224) | −8.8389994376 (224) |
| 6×6, N=4 | −13.9159290464 (3298) | −13.1158865155 (3264) | −12.2783082430 (3264) |
| 8×8, N=4 | −14.8308385845 (19902) | −14.3644578911 (19840) | −13.8294561114 (19840) |

Full sector spectra for 8×8 N=4 are stored in
`../ed/results/lss_semion_k{00,01,12}.npy`. For any other small system,
generate exact numbers with `../ed/semion_ed/` (see `../ed/scripts/run_lss.py`
for the sector pipeline); exact sector bases come from
`semion_ed.momentum.sector_basis`, so overlaps ⟨ψ_NQS|ψ_exact⟩ can be
measured directly on ≤ 6×6 systems.

## 5. Suggested milestones

1. **E_loc correctness**: exact summation (no sampling) of ⟨H⟩ for a random
   unsymmetrized ψ on 4×4 N=2; compare against dense H from
   `AnyonModel.hamiltonian()`. Must agree to machine precision.
2. **Symmetrization correctness**: verify the covariance identity above and
   that exact summation of the projected ansatz reproduces sector-restricted
   expectation values; optimize on 4×4 N=2 per sector → match the sector E₀
   table to ~1e−6.
3. **Sampling**: repeat milestone 2 with Metropolis sampling.
4. **Scale to ED frontier**: 6×6 N=4 (overlap + energy per sector), then
   8×8 N=4 (energy vs stored spectra).
5. **Beyond ED**: 10×10, N=4/6 (dim ~10⁸–10¹⁰) — new territory; monitor
   variance ⟨H²⟩−⟨H⟩² → 0 as the eigenstate criterion.

## 6. Pitfalls checklist

- Wrong convention (`ref` vs `paper`) silently shifts sector labels and
  energies — always instantiate with `convention="paper"`.
- N odd is inconsistent (constructor raises).
- The cut factor cN = e^{−iπN/2} is −1 for N ≡ 2 (mod 4) — it matters for
  N=2 test systems even though it is +1 for N=4.
- Same-column particles do NOT braid at cut A; the T_y phase φ_y counts only
  non-crossing particles. Reuse `Ty_action` rather than re-implementing.
- Amplitude ratios in E_loc need the sheet flip at cut B: the connected
  state of (c, f) under a cut-B hop is (c′, f⊕1).
