# Algorithm Spec: Lindblad -> Pauli-Lindblad Synthesis

Status: draft (2026-04-12) -- extracted from scout deep-read.
Pairs with: `design/lindblad_sim_skeleton.md` (uses this as the MagnusSynth kernel).

**Primary reference.** Malekakhlagh, Seif, Puzzuoli, Govia, van den Berg,
*Efficient Lindblad synthesis for noise model construction*, npj QI 2025,
arXiv:2502.03462v1. Equation numbers below from the v1 HTML.

**Secondary reference.** van den Berg, Minev, Kandala, Temme,
*Probabilistic error cancellation with sparse Pauli-Lindblad models*,
Nat. Phys. 2023, arXiv:2201.09866 (sparse PL generator + Pauli fidelity).

**Source status (2026-04-12).** LaTeX tarball extracted to
`/tmp/lindblad_tex/Main.tex` (1082 lines). All closed-form $\lambda_k$
expressions below are verbatim from the tex source; equation-label
references are authoritative.

---

## 1. Inputs & outputs

**Inputs.**
- Gate Hamiltonian $H_g \in \mathbb{C}^{d\times d}$, $d=2^n$, Hermitian,
  time-(quasi-)independent in the rotating frame (paper eq. 6-8).
- Collapse operators $\{L_j\}$ with rates $\beta_j \ge 0$, or GKS matrix
  $\beta_{jk}$; optional coherent shifts $\delta_j$.
- Gate duration $\tau_g$ and gate angle $\theta = \omega_g \tau_g$
  (with $\omega_g \in \{\omega_{cz}, \omega_{cx}\}$).
- Pauli basis $\mathcal{K} \subseteq \{I,X,Y,Z\}^{\otimes n}\setminus\{I^{\otimes n}\}$
  (typically weight-1 and weight-2 on device edges).
- Magnus truncation order $N \in \{1,2,3,4\}$.

**Output.** Rate vector $\{\lambda_k\}_{k\in\mathcal{K}}$ with
$\lambda_k \ge 0$ (non-negativity only guaranteed to the truncation order;
see open questions below).

**Assumptions.**
- Weak noise: $\beta\tau_g \ll 1$, equivalently $\beta/\omega_g \ll 1$ for
  two-qubit gates. Magnus convergence radius.
- Markovianity -- time-local Lindblad master equation.
- $H_g$ time-independent in a convenient frame. Time-dependent $H_g(t)$
  requires $U_g(t) = \mathcal{T}\exp(-i\int H_g(s)\,ds)$ via piecewise
  integration (not v1).

---

## 2. Algorithm pseudocode

```
INPUT  H_g, {L_j, beta_j}, tau_g, K, N
OUTPUT {lambda_k : k in K}

// Step 1 -- interaction-frame jump operators (paper eq. 6-8)
//   P_{jI}(t) = U_g(t)^dag L_j U_g(t)
//   L_I(t)(rho) = -i sum_j delta_j [P_{jI}(t), rho]
//               + sum_{jk} beta_{jk} ( P_{jI}(t) rho P_{kI}^dag(t)
//                                    - 1/2 { P_{kI}^dag(t) P_{jI}(t), rho } )
eigendecomp H_g = V D V^dag
U_g(t) = V * diag(exp(-i D t)) * V^dag        // pure-phase matrix elements
                                              // in H_g eigenbasis
PjI(t) = U_g(t)^dag * L_j * U_g(t)

// Step 2 -- Magnus terms (paper eq. 11, 12; App. C for higher orders)
Omega_1 = integrate( L_I(t1),                      t1 in [0,tau_g] )
Omega_2 = 0.5 * integrate_double(
            comm( L_I(t1), L_I(t2) ),              0 <= t2 <= t1 <= tau_g )
Omega_3 = (1/6) * integrate_triple(
            comm(L_I(t1), comm(L_I(t2), L_I(t3)))
          + comm(L_I(t3), comm(L_I(t2), L_I(t1))), 0 <= t3 <= t2 <= t1 <= tau_g )
// VERIFIED prefactor is 1/12 (paper eq. TDLindPT-G4 Sol), NOT 1/24 (BCOR textbook).
Omega_4 = (1/12) * integrate_quadruple(
            comm(L_I(t'), comm(L_I(t''), comm(L_I(t'''), L_I(t''''))))
          + comm(L_I(t'), comm([L_I(t''), L_I(t''')], L_I(t'''')))
          + comm([[L_I(t'), L_I(t'')], L_I(t''')], L_I(t''''))
          + comm(L_I(t''), comm(L_I(t'''), comm(L_I(t''''), L_I(t')))),
            0 <= t'''' <= t''' <= t'' <= t' <= tau_g )

// Step 3 -- effective generator (paper eq. 9-10)
L_eff = (1/tau_g) * sum_{n=1..N} Omega_n

// Step 4 -- Pauli twirl projection
//   Twirled generator is diagonal in Pauli basis.
//   Diagonal coeff: alpha_b = -(1/d) tr( P_b * L_eff(P_b) )
//   (alpha_b is a *rate* = 1/time; its integrated form is alpha_b * tau_g.)
//   Rates recovered via Walsh-Hadamard on {0,1}^{2n} (2201.09866 eq. (1)):
//     alpha_b = 2 sum_k lambda_k <b,k>_sp           // forward map
//     lambda_k = -(1/4^n) sum_b (-1)^{<b,k>_sp} alpha_b  // for k != I
//     lambda_I = 0   (by convention; not a physical rate)
//
// Derivation sketch: let W_{bk} = (-1)^{<b,k>_sp}, T = sum_k lambda_k.
// Then (W lambda)_b = T - alpha_b (since <b,k>_sp = (1 - W_{bk})/2). W is
// self-inverse up to a factor of 4^n, so applying W and using that
// sum_b (-1)^{<b,k>_sp} vanishes for k != I gives the formula above.
// For the 1-qubit case this collapses to the direct linear solve
//   lambda_X = (alpha_Y + alpha_Z - alpha_X) / 4  (etc. by symmetry).

// Step 5 -- Dyson cross-check (paper eq. 13)
//   T exp( int L_I ) = I + Omega_1 + Omega_2 + 1/2 Omega_1^2 + O(L_I^3)
//   Compare Magnus-truncated channel vs Dyson-truncated channel.
```

**Key simplification for constant $H_g$.** Matrix elements of $P_{jI}(t)$
in the $H_g$ eigenbasis are **pure phases $e^{i(E_a-E_b)t}$**. All Magnus
time integrals become sums of exponentials times polynomials in $t$ --
integrate **analytically**, no numerical quadrature. This is what makes
closed-form Appendix C results possible.

**Twirl representation.** $\mathcal{L}_{eff}$ is a $d^2\times d^2$ map
$M_d\to M_d$ in the Pauli transfer matrix (PTM) representation; the
diagonal entries are $-\alpha_b$. Off-diagonals measure residual coherence
and must be small under the weak-noise assumption -- assert
`||off-diagonal|| < tol` as a correctness check.

---

## 3. Closed-form fixtures (Appendix E / `App:WhyPauliLind`)

All expressions verbatim from `/tmp/lindblad_tex/Main.tex`. Index convention:
$P_b$ written as string label `ab` = $P_a \otimes P_b$ on (left, right) qubit;
$i\equiv I$. Rates: $\beta_{\downarrow j}$ amplitude damping on qubit $j$,
$\beta_{\phi j}$ pure dephasing on qubit $j$, for $j\in\{l, r\}$.

### Single-qubit identity + AD + PD (exact, non-perturbative)

Paper line 812, $\tau_g$-scale:
$$
\lambda_x = \lambda_y = \tfrac14\beta_{\downarrow}\tau_g,\quad
\lambda_z = \tfrac12\beta_\phi\tau_g.
$$
Not perturbative -- exact twirled result for identity.

### Single-qubit $X_\theta$ + AD + PD (paper eqs. 869-874)

$$
\lambda_x = \tfrac{\theta}{4}\tfrac{\beta_\downarrow}{\omega_x},
$$
$$
\lambda_y = \tfrac{2\theta+\sin 2\theta}{16}\tfrac{\beta_\downarrow}{\omega_x}
         + \tfrac{2\theta-\sin 2\theta}{8}\tfrac{\beta_\phi}{\omega_x},
$$
$$
\lambda_z = \tfrac{2\theta-\sin 2\theta}{16}\tfrac{\beta_\downarrow}{\omega_x}
         + \tfrac{2\theta+\sin 2\theta}{8}\tfrac{\beta_\phi}{\omega_x}.
$$

### Two-qubit $CZ_\theta$ + AD + PD (paper eqs. 896-906)

$\theta = \omega_{cz}\tau_g$. PD contributions separable from AD:
$$
\lambda_{iz} = \tfrac{\theta}{2}\tfrac{\beta_{\phi r}}{\omega_{cz}},\quad
\lambda_{zi} = \tfrac{\theta}{2}\tfrac{\beta_{\phi l}}{\omega_{cz}},
$$
$$
\lambda_{ix}=\lambda_{iy}=\tfrac{2\theta+\sin 2\theta}{16}\tfrac{\beta_{\downarrow r}}{\omega_{cz}},\quad
\lambda_{xi}=\lambda_{yi}=\tfrac{2\theta+\sin 2\theta}{16}\tfrac{\beta_{\downarrow l}}{\omega_{cz}},
$$
$$
\lambda_{zx}=\lambda_{zy}=\tfrac{2\theta-\sin 2\theta}{16}\tfrac{\beta_{\downarrow r}}{\omega_{cz}},\quad
\lambda_{xz}=\lambda_{yz}=\tfrac{2\theta-\sin 2\theta}{16}\tfrac{\beta_{\downarrow l}}{\omega_{cz}}.
$$
At Clifford angles $\theta = n\pi/2$ the degeneracy becomes 4-fold:
$\lambda_{ix}=\lambda_{iy}=\lambda_{zx}=\lambda_{zy}$ and
$\lambda_{xi}=\lambda_{yi}=\lambda_{xz}=\lambda_{yz}$.

### Two-qubit $CX_\theta$ + AD + PD (paper eqs. 929-956)

$\theta = \omega_{cx}\tau_g$. AD and PD **mix** in $\lambda_{iy}, \lambda_{iz},
\lambda_{zy}, \lambda_{zz}$:
$$
\lambda_{ix} = \tfrac{\theta}{4}\tfrac{\beta_{\downarrow r}}{\omega_{cx}},\quad
\lambda_{zi} = \tfrac{\theta}{2}\tfrac{\beta_{\phi l}}{\omega_{cx}},
$$
$$
\lambda_{iy} = \tfrac{12\theta + 8\sin 2\theta + \sin 4\theta}{128}\tfrac{\beta_{\downarrow r}}{\omega_{cx}}
            + \tfrac{4\theta - \sin 4\theta}{64}\tfrac{\beta_{\phi r}}{\omega_{cx}},
$$
$$
\lambda_{iz} = \tfrac{4\theta - \sin 4\theta}{128}\tfrac{\beta_{\downarrow r}}{\omega_{cx}}
            + \tfrac{12\theta + 8\sin 2\theta + \sin 4\theta}{64}\tfrac{\beta_{\phi r}}{\omega_{cx}},
$$
$$
\lambda_{zy} = \tfrac{12\theta - 8\sin 2\theta + \sin 4\theta}{128}\tfrac{\beta_{\downarrow r}}{\omega_{cx}}
            + \tfrac{4\theta - \sin 4\theta}{64}\tfrac{\beta_{\phi r}}{\omega_{cx}},
$$
$$
\lambda_{zz} = \tfrac{4\theta - \sin 4\theta}{128}\tfrac{\beta_{\downarrow r}}{\omega_{cx}}
            + \tfrac{12\theta - 8\sin 2\theta + \sin 4\theta}{64}\tfrac{\beta_{\phi r}}{\omega_{cx}},
$$
$$
\lambda_{xi} = \lambda_{yi} = \tfrac{2\theta + \sin 2\theta}{16}\tfrac{\beta_{\downarrow l}}{\omega_{cx}},\quad
\lambda_{xx} = \lambda_{yx} = \tfrac{2\theta - \sin 2\theta}{16}\tfrac{\beta_{\downarrow l}}{\omega_{cx}}.
$$

**Correction from earlier scout.** Initial scout transcribed
$\lambda_{iz}=\lambda_{zz}=\frac{4\theta-\sin 4\theta}{128}\frac{\beta_{\downarrow r}}{\omega_{cx}}$
and missed PD contributions entirely. Verbatim paper formulae above
supersede.

### Two-qubit phase noise (subsection SubApp:2QPhNoise, lines 962-1001)

Quadratic-in-$\delta$ dependence (coherent noise $H_\delta = (\delta/2)ZZ$).
Not transcribed here; see paper lines 962-1001 if PECOS needs coherent-
noise fixtures before v1 ships.

### Three-qubit ZZ crosstalk (paper eqs. 1009-1011)

**Only non-trivial case**: $CX_\theta \otimes I$ with $IZZ$ crosstalk between
target and spectator. $H_g = (\omega_{cz}/2)(IXI-ZXI)$,
$H_\delta = (\delta_{izz}/2)IZZ$. Produces weight-2 **and weight-3** PL terms:
$$
\lambda_{iyz} = \lambda_{zyz} = \tfrac{\sin^4\theta}{16}\tfrac{\delta_{izz}^2}{\omega_{cx}^2},
$$
$$
\lambda_{izz} = \tfrac{[2\theta + \sin 2\theta]^2}{64}\tfrac{\delta_{izz}^2}{\omega_{cx}^2},\quad
\lambda_{zzz} = \tfrac{[2\theta - \sin 2\theta]^2}{64}\tfrac{\delta_{izz}^2}{\omega_{cx}^2}.
$$
**Important for PECOS:** weight-3 terms break the standard weight-2-only
sparse-PL sparsity assumption -- `PauliLindbladModel` must allow
user-specified basis $\mathcal{K}$ with weight > 2.

### Four-qubit ZZ crosstalk (paper eqs. 1044-1062)

Only case (iv) -- $CX_\theta \otimes X_\theta C$ with $IZZI$ crosstalk on
middle two qubits -- is non-trivial (case (iii) reduces to 3Q). Yields
weight-3 and weight-4 PL terms.

$H_g = (\omega_{cx}/2)[(IXII-ZXII) + (IIIX-IIZX)]$,
$H_\delta = (\delta_{izzi}/2)IZZI$:
$$
\lambda_{iyyi} = \lambda_{iyyz} = \lambda_{izzz} = \lambda_{zyyi} = \lambda_{zyyz}
= \tfrac{[4\theta - \sin 4\theta]^2}{4096}\tfrac{\delta_{izzi}^2}{\omega_{cx}^2},
$$
$$
\lambda_{iyzi} = \lambda_{izyi} = \lambda_{iyzz} = \lambda_{zzyi}
= \tfrac{\sin^4\theta [3 + \cos 2\theta]^2}{256}\tfrac{\delta_{izzi}^2}{\omega_{cx}^2},
$$
$$
\lambda_{iyzz} = \lambda_{zyzz} = \lambda_{zzyi} = \lambda_{zzyz}
= \tfrac{\sin^8\theta}{64}\tfrac{\delta_{izzi}^2}{\omega_{cx}^2},
$$
$$
\lambda_{izzi} = \tfrac{[12\theta + 8\sin 2\theta + \sin 4\theta]^2}{4096}\tfrac{\delta_{izzi}^2}{\omega_{cx}^2},
$$
$$
\lambda_{zzzz} = \tfrac{[12\theta - 8\sin 2\theta + \sin 4\theta]^2}{4096}\tfrac{\delta_{izzi}^2}{\omega_{cx}^2}.
$$

Note: paper appears to have duplicate labels in the first group
($\lambda_{iyyz}$ appears twice) -- possible typo; verify against any
erratum before Rust transcription.

### Leading-order precision (paper App:LindPertPrecision)

For $CX_{\pi/4}$ at $\beta_\downarrow/\omega_{cx} \approx 10^{-2}$: deviation
$\sim O(10^{-5})$. At $10^{-1}$: deviation $\sim O(10^{-4})$. Use as
guidance for convergence-regime defaults in `MagnusSynth`.

**Test fixture usage.** Feed $H_{CR}$ or $H_{CZ}$ and $L\in\{\sigma^-, Z\}$
into the algorithm; compare against closed forms to `< 1e-10`. For 3Q/4Q
crosstalk, feed $H_\delta = (\delta/2)P$ (coherent, not incoherent) and
verify quadratic scaling in $\delta$.

---

## 4. Effort revision (post-latex-extract)

**Scout initial estimate:** 200-300 formulae. **Actual (from verbatim
tex extract):** ~25-30 distinct $\lambda_k$ expressions across the
whole appendix. Most cases collapse: paper notes "the only non-trivial
case is $CX_\theta \otimes I$ with $IZZ$" etc. Much less transcription
work than scout estimated.

Breakdown of distinct formulae:
- 1Q identity (AD+PD): 3 entries (non-perturbative).
- 1Q $X_\theta$: 3 entries with AD+PD mixing.
- 2Q $CZ_\theta$: 8 entries (mostly 2-fold/4-fold degenerate).
- 2Q $CX_\theta$: 9 entries with AD+PD mixing on 4 of them.
- 2Q phase noise: untranscribed; coherent ZZ, quadratic in $\delta$
  (lines 962-1001).
- 3Q ZZ crosstalk: 3 entries ($CX \otimes I$ only).
- 4Q ZZ crosstalk: 5 groups (many-fold degenerate) for
  $CX_\theta \otimes X_\theta C$ only.

Rust lookup form: `(gate_type, pauli_label) -> fn(theta, beta_ad_l,
beta_ad_r, beta_pd_l, beta_pd_r, omega) -> f64`. One afternoon. Test each
against a numerical Magnus order-2 integration on the same inputs.

**Ambiguity flag.** Paper's 4Q section has an apparent label typo
(`lambda_iyyz` listed twice in one group). Manual review + possible
erratum check required.

---

## 5. Sparse Pauli-Lindblad generator (arXiv:2201.09866)

**Generator.**
$$
\mathcal{L}(\rho) = \sum_{k\in\mathcal{K}}\lambda_k\bigl(P_k\rho P_k^\dagger - \rho\bigr),
\quad \lambda_k \ge 0.
$$

**Pauli fidelity (2201.09866 eq. 1, 2311.15408 eq. 1).**
$$
f_b = \tfrac{1}{2^n}\operatorname{tr}\bigl(P_b\,\Lambda(P_b)\bigr)
    = \exp\!\Bigl(-2\sum_{k\in\mathcal{K}}\lambda_k\,\langle b,k\rangle_{sp}\Bigr).
$$

**Symplectic inner product.** Write $P = i^{x\cdot z}X^x Z^z$,
$(x,z)\in\mathbb{F}_2^{2n}$. Then
$$
\langle b,k\rangle_{sp} = x_b\cdot z_k + z_b\cdot x_k \pmod 2 \in \{0,1\},
$$
i.e. `0` if $P_b, P_k$ commute, `1` if they anticommute. Implementation:
bitwise XOR + popcount + `& 1`. `O(n/64)` per pair.

**Forward sampling over duration $t$.** Each $k$ acts as an independent
single-Pauli channel $(1-p_k)\mathbb{1} + p_k\,P_k\cdot P_k$ with
$$
p_k = \tfrac12\bigl(1 - e^{-2\lambda_k t}\bigr).
$$
Per shot: for each $k$, draw Bernoulli($p_k$); if `1`, apply $P_k$.
All $|\mathcal{K}|$ draws independent, $O(|\mathcal{K}|)$ per shot.

For PEC the signed form $\gamma_k = \text{sign}(\lambda_k)$ would be
tracked; forward QEC simulation uses $\lambda_k \ge 0$ only.

---

## 6. Complexity & data structures

- **Magnus order $N$.** Each $\Omega_n$: $n$-fold nested commutator of
  $d\times d$ matrices + analytic time integral. Dominant matmul cost
  $O(N\cdot d^3)$; commutator sum $O(N\cdot M^n)$ with $M$ = number of
  jump operators. For $n=2$, $d=4$, $d^3=64$ -- trivial.
- **Pauli basis $\mathcal{K}$.** 1-local = $3n$; 2-local on device edges
  $=9|E|$. On 100-qubit heavy-hex ($|E|\approx 140$): ~1560 terms.
  $O(n^2)$ worst case.
- **Dense path memory.** State matrix $d\times d$ complex
  ($16\,d^2$ bytes). PTM $d^2\times d^2$ ($16\,d^4$ bytes). For 4-qubit
  ($d=16$): 0.5 MB -- fine.
- **Sparse path (> 6 qubits).** Represent $\mathcal{L}$ as a list of
  `(P_j, P_k, beta_jk)` triples and form $\Omega_n$ symbolically in
  Pauli basis via the Pauli-group multiplication table; never
  materialize the PTM.
- **Rust types.** `faer::Mat<Complex64>` for small dense path ($d\le 16$).
  `SparsePauliOp` (`Vec<(PauliLabel, Complex64)>`) for sparse path.
  Commutators via Pauli-group table -- this is where grug gets the
  80/20 win.

---

## 7. Open questions / risks

- **Positivity of $\lambda_k$.** Magnus-truncated generator is **not
  guaranteed** GKS-positive at finite order. Paper dodges via weak-noise
  assumption. PECOS policy decision: clip to $\max(0, \lambda_k)$
  (lossy), warn/error on negative, or bump order (expensive). Start with
  "warn + clip" and log the truncation residual.

- **Omega_3 / Omega_4 prefactor verification.** Resolved (2026-04-12):
  $\Omega_3$ has prefactor $1/6$ and $\Omega_4$ has prefactor **$1/12$**
  (paper eq. TDLindPT-G4 Sol, line 688 of Main.tex), with 4 specific
  nested-commutator terms explicitly listed. Note: textbook BCOR uses
  $1/24$ with a different term decomposition -- the paper's form is
  equivalent by commutator identities but the prefactor is $1/12$ as
  written. Use paper's form verbatim.

- **Time-dependent $H_g(t)$.** Paper assumes quasi-time-independent.
  Real pulse shapes (Gaussian, DRAG) break this. Dyson path handles
  numerically (time-ordered product); Magnus path needs piecewise
  integration. Out of scope v1.

- **Catastrophic cancellation near $\theta=0$.** Formulae like
  $(2\theta-\sin 2\theta)/16$ lose precision at small $\theta$
  ($\approx \theta^3/6$). Rust impl **must** use a Taylor branch for
  $|\theta|<\epsilon$. Standard `sinmx` trick; test with
  $\theta=10^{-10}$.

- **PTM off-diagonal residuals.** Weak-noise -> near-diagonal in Pauli
  basis. Assert `||off-diagonal|| < tol`; do not silently discard.

- **Pauli basis $\mathcal{K}$ completeness.** If physical noise generates
  a $\lambda_k$ outside $\mathcal{K}$ (e.g. amplified weight-3 ZZ), it is
  silently dropped on projection. Log the norm of the projected-away part;
  error if above a user-settable threshold.

---

## 8. Implementation phases

**Phase 1 -- numerical (gold standard).**
Eigendecompose $H_g$, integrate $\Omega_1$ and $\Omega_2$ analytically in
the eigenbasis, assemble $\mathcal{L}_{eff}$ as a PTM, diagonal-read the
$\alpha_b$, Walsh-Hadamard to $\lambda_k$. Test against Table 1 (CX,
amplitude damping, right qubit only). This is the order-2 MagnusSynth
default.

**Phase 2 -- sparse Pauli-Lindblad sampler + DemStabSim glue.**
Implement `PauliLindbladModel::sample(t)` via independent Bernoullis
(Section 5). Implement `DemStabNoiseModel` for `PauliLindbladModel`
(skeleton Section "Glue into DemStabSim"). Rep-code memory experiment
parity test.

### Shipped work (2026-04-13)

Implemented in `exp/pecos-lindblad/` Phases 1-5:

| # | Scope | Phase |
|---|---|---|
| 1 | 1Q identity (exact) | [shipped] |
| 2 | 1Q X_theta (leading-order) | [shipped] |
| 3 | 2Q CZ_theta + n-qubit Walsh-Hadamard | [shipped] |
| 4 | 2Q CX_theta + block-diagonal exp | [shipped] |
| 5 | PL summary helpers + DemStabSim scalar-collapse scaffold | [shipped] |

28 tests verify all four paper closed-form fixtures (1Q ident, X_theta,
CZ_theta, CX_theta) to tol 1e-8 (1e-12 for the exact identity result).

### Order-1 scope limit

The `synthesize_numerical` entry point implements **Omega_1 only**. This is
correct and tight for incoherent noise (amplitude damping, pure
dephasing) because the rates enter linearly in `beta`. Coherent noise
cases (2Q phase noise, 3Q ZZ crosstalk, 4Q ZZ crosstalk) have rates
**quadratic in delta** and require Omega_2 + Pauli-twirl:

- For purely coherent `L(rho) = -i[H_delta, rho]`:
  `Tr(P_b * L(P_b))/d = 0` (first-order diagonal element vanishes by
  cyclicity of trace for Hermitian H_delta).
- Thus the Omega_1-diagonal shortcut gives `alpha_b = 0`, and the
  extracted `lambda_k = 0`, which is wrong.
- The correct second-order result comes from twirling the full channel
  `exp(Omega_1 + Omega_2 + ...)`, where quadratic cross-terms in the
  expansion produce non-vanishing Pauli-diagonal contributions.

### Open for future phases

- Phase 6: coherent-noise path. Either (a) implement Omega_2 + twirl, or
  (b) add general `d x d` Hermitian matrix exponentiation and compute the
  exact channel `U_err rho U_err^dag`. Target: 3Q IZZ crosstalk
  (paper eqs. 1009-1011, weight-3 rates).
- Phase 7: proper `pecos-qec::NoiseConfig` generalization and
  per-gate-type Pauli-Lindblad input to `DemStabSim` (see
  `design/lindblad_sim_skeleton.md` "Glue into DemStabSim" section).
- Phase 8: 4-qubit ZZ crosstalk with weight-4 rates (paper eqs.
  1044-1062). Blocked on Phase 6.

**Phase 3 -- closed-form Appendix C lookup.**
Transcribe Tables 1-2 into a Rust `const` table keyed by
`(GateType, PauliLabel)`. Property-test each entry against Phase 1
numerical path. Add Taylor branches for small $\theta$.

**Phase 4 -- higher-order Magnus.**
Implement $\Omega_3, \Omega_4$ with Blanes-Casas-Oteo-Ros; verify against
Phase 3 + Phase 1 random Lindbladians. Out-of-regime detection.

**Phase 5 -- Appendix D multi-qubit ZZ crosstalk.**
Re-scrape paper for D.7/D.8 formulae. Transcribe. Property-test. 200-300
entries.

Phases 1-2 are the MVP that unblocks DemStabSim-with-Lindblad-noise.
Phases 3-5 are refinements.

---

## 9. References

Paper:
- arXiv:2502.03462 -- Malekakhlagh et al., *Efficient Lindblad synthesis*
- arXiv:2201.09866 -- van den Berg et al., *Sparse Pauli-Lindblad PEC*
- arXiv:2311.15408 -- Chen et al., *Learning sparse PL*

Cross-check:
- arXiv:2407.03576 -- 4th-order commutator-free Magnus in Liouville space
- Blanes, Casas, Oteo, Ros, *The Magnus expansion and some of its
  applications* (Phys. Rep. 2009) -- textbook for $\Omega_n$ formulae

Next scout TODO:
- 2Q phase noise (paper subsection SubApp:2QPhNoise, lines 962-1001):
  coherent-noise test fixtures. Low priority; only needed if coherent
  $ZZ$ test path is in v1.
- Verify paper's apparent 4Q label typo ($\lambda_{iyyz}$ listed twice).

Source checked out at `/tmp/lindblad_tex/Main.tex` (ephemeral). For a
permanent copy, pull from `arxiv.org/e-print/2502.03462`.
