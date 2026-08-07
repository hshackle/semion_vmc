# Neural Quantum State Setup with Anyonic Translation Symmetry

## 1. Goal

Construct a complex neural quantum state for lattice semions on a torus that transforms in a definite momentum sector under the **physical anyonic translation operators**, not merely under translation of the occupation array.

The recommended implementation is

\[
\boxed{
\text{2D translation-equivariant backbone}
\;+\;
\text{two-component complex output}
\;+\;
\text{exact anyonic momentum projection}.
}
\]

The two-component output represents the torus topological fiber. The anyonic translation matrices are supplied by the same translation oracle used to define the Hamiltonian.

---

## 2. Variational state

For a particle configuration \(C\), let the unprojected network output

\[
\boldsymbol\phi_\theta(C)
=
\begin{pmatrix}
\phi_{\theta,0}(C)\\
\phi_{\theta,1}(C)
\end{pmatrix}
\in\mathbb C^2.
\]

The network does not need to impose an exchange rule directly. Semionic statistics is already encoded in the Hamiltonian and in the configuration-dependent action of translations.

A useful base architecture is a periodic complex CNN:

```text
occupation array
    -> periodic convolution
    -> residual blocks
    -> periodic convolution
    -> complex two-component output
```

A graph network or lattice transformer can be substituted without changing the projection procedure.

At fixed particle number, the input is a binary occupation array

\[
n_C(x,y)\in\{0,1\},
\qquad
\sum_{x,y}n_C(x,y)=N.
\]

No autoregressive factorization is required. Fixed \(N\) can instead be enforced by the Monte Carlo proposal.

---

## 3. Anyonic translation action

Let \(g\) be a lattice translation. The translation oracle returns

\[
gC
\]

and a configuration-dependent matrix

\[
U_g(C)\in U(2).
\]

The physical action on a vector-valued wavefunction is

\[
[\mathcal R_g\boldsymbol\psi](C)
=
U_g(g^{-1}C)\boldsymbol\psi(g^{-1}C).
\]

The matrices must satisfy the cocycle relation

\[
U_{gh}(C)
=
U_g(hC)U_h(C).
\]

This identity is essential. It ensures that the configuration-dependent translation rule still defines a genuine representation of the lattice translation group on the full Hilbert space.

A momentum-\(\mathbf k\) state obeys

\[
\mathcal R_g\boldsymbol\psi_{\mathbf k}
=
e^{i\mathbf k\cdot g}\boldsymbol\psi_{\mathbf k}.
\]

Equivalently,

\[
\boldsymbol\psi_{\mathbf k}(gC)
=
e^{-i\mathbf k\cdot g}
U_g(C)\boldsymbol\psi_{\mathbf k}(C).
\]

The phase convention may be reversed if active and passive translations are defined differently. The implementation should choose one convention and test it explicitly.

---

## 4. Exact anyonic momentum projection

Given an unconstrained seed network \(\boldsymbol\phi_\theta(C)\), define

\[
\boxed{
\boldsymbol\psi_{\theta,\mathbf k}(C)
=
\frac{1}{|G|}
\sum_{g\in G}
e^{-i\mathbf k\cdot g}
U_g(g^{-1}C)
\boldsymbol\phi_\theta(g^{-1}C).
}
\]

This is the anyonic analogue of ordinary momentum projection.

The projection enforces the symmetry exactly:

\[
\mathcal R_h\boldsymbol\psi_{\theta,\mathbf k}
=
e^{i\mathbf k\cdot h}
\boldsymbol\psi_{\theta,\mathbf k}.
\]

It also handles configurations with nontrivial translation stabilizers automatically. If \(hC=C\), then the amplitude must satisfy

\[
U_h(C)\boldsymbol\psi(C)
=
e^{i\mathbf k\cdot h}\boldsymbol\psi(C).
\]

If the required eigenvalue is absent from \(U_h(C)\), the projected amplitude vanishes on that translation orbit.

### Reference implementation

```python
def projected_amplitude(config, momentum, model, translation_oracle):
    psi = zeros_complex((2,))

    for g in translation_group(Lx, Ly):
        preimage = translate_config(config, inverse(g))
        _, U = translation_oracle(preimage, g)

        phi = model(preimage)  # shape: (2,), complex
        phase = exp(-1j * dot(momentum, g))

        psi += phase * (U @ phi)

    return psi / (Lx * Ly)
```

This direct implementation requires \(L_xL_y\) network evaluations per configuration. It should be implemented first because it is transparent and easy to validate.

---

## 5. Efficient twisted Fourier pooling

The full projection can be reduced to one CNN evaluation if the backbone returns a site-resolved equivariant feature field.

Let

\[
\mathbf h_\theta(C,\mathbf r)\in\mathbb C^c
\]

be the output of a periodic translation-equivariant CNN. Apply a shared sitewise head

\[
\mathbf q_\theta(C,\mathbf r)
=
W\mathbf h_\theta(C,\mathbf r)
\in\mathbb C^2.
\]

Choose a reference site \(\mathbf r_0\). Translation equivariance gives

\[
\boldsymbol\phi_\theta(g^{-1}C)
=
\mathbf q_\theta(C,g\mathbf r_0)
\]

up to the convention used for the translation action.

The projected wavefunction becomes

\[
\boxed{
\boldsymbol\psi_{\theta,\mathbf k}(C)
=
\frac{1}{|G|}
\sum_{g\in G}
e^{-i\mathbf k\cdot g}
U_g(g^{-1}C)
\mathbf q_\theta(C,g\mathbf r_0).
}
\]

This is a **twisted Fourier pooling layer**. The computational steps are:

1. evaluate the periodic CNN once;
2. produce a two-component vector at each site;
3. multiply each site vector by the corresponding anyonic transport matrix;
4. multiply by the momentum character;
5. sum over all translations.

### Pseudocode

```python
def twisted_fourier_pool(config, momentum, backbone, head, oracle):
    features = backbone(config)   # [Lx, Ly, channels]
    q = head(features)            # [Lx, Ly, 2], complex

    psi = zeros_complex((2,))

    for dx in range(Lx):
        for dy in range(Ly):
            g = (dx, dy)
            preimage = translate_config(config, (-dx, -dy))
            _, U = oracle(preimage, g)

            phase = exp(-1j * (momentum[0] * dx + momentum[1] * dy))
            psi += phase * (U @ q[dx, dy])

    return psi / (Lx * Ly)
```

The mapping between `q[dx, dy]` and \(g\mathbf r_0\) depends on array-indexing and active/passive translation conventions. It should be fixed by comparing against the direct projector, not by inspection.

---

## 6. Backbone design

A practical starting architecture is:

```text
binary occupation map
    -> periodic 3x3 convolution
    -> several residual convolutional blocks
    -> site-resolved complex feature map
    -> shared complex linear head to dimension 2
    -> twisted Fourier pooling
```

Recommended choices:

- periodic padding;
- no ordinary global pooling before the anyonic projection;
- complex output represented either natively or by paired real channels;
- receptive field large enough to cover the torus;
- fixed-particle-number sampling rather than an autoregressive number mask.

The seed network need not itself satisfy the anyonic translation law. Exact projection imposes the symmetry on the final state.

Because the anyonic action is applied only at the output, ordinary nonlinearities inside the backbone are allowed. Restrictions such as norm-gated nonlinearities arise only if the hidden layers themselves are required to transform as topological doublets.

---

## 7. Sampling and VMC estimator

The wavefunction is a two-component vector

\[
\boldsymbol\psi_\theta(C)\in\mathbb C^2.
\]

It is convenient to sample only the occupation configuration and sum over the topological fiber exactly.

Define the configuration weight

\[
w_\theta(C)
=
\boldsymbol\psi_\theta(C)^\dagger
\boldsymbol\psi_\theta(C).
\]

Sample from

\[
p_\theta(C)
=
\frac{w_\theta(C)}
{\sum_{C'}w_\theta(C')}.
\]

For Hamiltonian blocks

\[
H_{C,C'}\in\mathbb C^{2\times2},
\]

the scalar local-energy estimator is

\[
\boxed{
E_{\mathrm{loc}}(C)
=
\frac{
\boldsymbol\psi_\theta(C)^\dagger
\displaystyle\sum_{C'}H_{C,C'}
\boldsymbol\psi_\theta(C')
}{
\boldsymbol\psi_\theta(C)^\dagger
\boldsymbol\psi_\theta(C)
}.
}
\]

Then

\[
E_\theta
=
\mathbb E_{C\sim p_\theta}
\left[
E_{\mathrm{loc}}(C)
\right].
\]

This is preferable to sampling a discrete topological-sector index because the fiber dimension is only two.

### Metropolis proposal

Use particle-conserving local moves:

1. choose an occupied site;
2. choose an empty nearest neighbor;
3. move the particle;
4. accept with

\[
A(C\to C')
=
\min\left[
1,
\frac{w_\theta(C')q(C'\to C)}
{w_\theta(C)q(C\to C')}
\right].
\]

For a symmetric proposal, the proposal-probability ratio is one.

At higher density, supplement local hops with nonlocal particle moves or cluster-like proposals to reduce autocorrelation.

---

## 8. Optimization

Standard complex VMC optimization applies. Suitable methods include:

- stochastic reconfiguration;
- natural gradient;
- MinSR;
- Adam for initial pretraining, followed by stochastic reconfiguration.

For a scalar wavefunction, the logarithmic derivative is

\[
O_\mu(C)
=
\partial_{\theta_\mu}\log\psi_\theta(C).
\]

For the two-component state sampled with weight

\[
w_\theta(C)
=
\boldsymbol\psi^\dagger\boldsymbol\psi,
\]

a convenient generalized derivative is

\[
O_\mu(C)
=
\frac{
\boldsymbol\psi_\theta(C)^\dagger
\partial_{\theta_\mu}\boldsymbol\psi_\theta(C)
}{
\boldsymbol\psi_\theta(C)^\dagger
\boldsymbol\psi_\theta(C)
}.
\]

Automatic differentiation can evaluate the energy gradient directly without manually deriving this expression.

The projected sum may have severe phase cancellation. During initial debugging, use complex double precision. If numerical underflow becomes important, stabilize the translated sum using a complex log-sum-exp procedure.

---

## 9. Minimal software interface

The NQS code only needs three model-specific interfaces.

### Anyonic translation

```python
translate_with_fiber(config, displacement)
    -> translated_config, U
```

where `U.shape == (2, 2)`.

### Hamiltonian connectivity

```python
connected_configurations(config)
    -> list[(config_prime, H_block)]
```

where each `H_block.shape == (2, 2)`.

### Neural amplitude

```python
wavefunction(params, config, momentum)
    -> complex vector of shape (2,)
```

The neural module should not directly implement braid phases. Those phases belong in the translation and Hamiltonian oracles.

---

## 10. Validation tests

The symmetry implementation should be validated before VMC optimization.

### 10.1 Translation cocycle

For random \(C,g,h\),

\[
U_{gh}(C)
\stackrel{?}{=}
U_g(hC)U_h(C).
\]

### 10.2 Commuting global translations

For primitive translations \(T_x,T_y\),

\[
U_x(T_yC)U_y(C)
\stackrel{?}{=}
U_y(T_xC)U_x(C).
\]

This is the correct test. The individual \(2\times2\) matrices \(U_x(C)\) and \(U_y(C)\) need not commute at the same base configuration.

### 10.3 Projected covariance

For random \(C,g\),

\[
\boldsymbol\psi_{\mathbf k}(gC)
\stackrel{?}{=}
e^{-i\mathbf k\cdot g}
U_g(C)\boldsymbol\psi_{\mathbf k}(C).
\]

### 10.4 Direct versus pooled projector

For random network parameters and configurations, compare:

\[
\boldsymbol\psi_{\mathrm{direct}}(C)
\]

against

\[
\boldsymbol\psi_{\mathrm{pooled}}(C).
\]

They should agree to floating-point precision.

### 10.5 Projector idempotence

\[
P_{\mathbf k}^2=P_{\mathbf k}.
\]

### 10.6 Hamiltonian covariance

\[
[\widehat T_g,H]=0.
\]

For small systems, construct the matrices explicitly and test the commutator norm.

### 10.7 Exact-diagonalization benchmark

For small lattices:

1. diagonalize the Hamiltonian in a fixed momentum sector;
2. optimize the projected NQS in the same sector;
3. compare the energy and wavefunction overlap.

The \(N=2\) problem is the most useful initial benchmark.

---

## 11. Recommended implementation order

1. Implement the direct projected network using \(L_xL_y\) independent forward passes.
2. Verify the anyonic covariance condition exactly.
3. Benchmark against exact diagonalization.
4. Implement the site-resolved CNN and twisted Fourier pooling.
5. Verify agreement with the direct projector.
6. Add fixed-\(N\) Metropolis sampling.
7. Add stochastic reconfiguration or MinSR.
8. Only afterward consider point-group symmetry or fiber-covariant hidden layers.

The translation oracle is the critical component. A standard CNN with an exact, correct projector is preferable to a more elaborate equivariant architecture built on an unverified anyonic translation action.

---

## 12. Compact architecture summary

\[
\boxed{
\begin{aligned}
C
&\xrightarrow{\text{periodic 2D CNN}}
\mathbf q_\theta(C,\mathbf r)\in\mathbb C^2,\\[2mm]
\boldsymbol\psi_{\theta,\mathbf k}(C)
&=
\frac{1}{L_xL_y}
\sum_{g}
e^{-i\mathbf k\cdot g}
U_g(g^{-1}C)
\mathbf q_\theta(C,g\mathbf r_0),\\[2mm]
p_\theta(C)
&\propto
\boldsymbol\psi_{\theta,\mathbf k}(C)^\dagger
\boldsymbol\psi_{\theta,\mathbf k}(C),\\[2mm]
E_{\mathrm{loc}}(C)
&=
\frac{
\boldsymbol\psi(C)^\dagger
\sum_{C'}H_{C,C'}\boldsymbol\psi(C')
}{
\boldsymbol\psi(C)^\dagger\boldsymbol\psi(C)
}.
\end{aligned}
}
\]

This gives a non-autoregressive, two-dimensional NQS with exact anyonic translation symmetry and exact summation over the torus topological fiber.

---

## References

- N. Kirchner, D. Millar, B. M. Ayeni, A. Smith, J. K. Slingerland, and F. Pollmann, “Numerical simulation of non-Abelian anyons,” *Phys. Rev. B* **107**, 195129 (2023), [arXiv:2206.14730](https://arxiv.org/abs/2206.14730).

- T. S. Cohen and M. Welling, “Group Equivariant Convolutional Networks,” [arXiv:1602.07576](https://arxiv.org/abs/1602.07576).

- T. S. Cohen, M. Weiler, B. Kicanaoglu, and M. Welling, “Gauge Equivariant Convolutional Networks and the Icosahedral CNN,” [arXiv:1902.04615](https://arxiv.org/abs/1902.04615).

- D. Luo, Z. Chen, K. Hu, Z. Zhao, V. M. Hur, and B. K. Clark, “Gauge-Invariant and Anyonic-Symmetric Autoregressive Neural Network for Quantum Lattice Models,” *Phys. Rev. Research* **5**, 013216 (2023), [arXiv:2101.07243](https://arxiv.org/abs/2101.07243).
