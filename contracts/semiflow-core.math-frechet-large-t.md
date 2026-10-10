# semiflow-core — Math Specification §63 (Fréchet gradient at large `λ_max·t`, ADR-0203)

NORMATIVE. This file **is** §63 of the math contract `contracts/semiflow-core.math.md`,
kept in its own file for size (the pre-commit git-guard caps a single file at 1024 KB;
precedent: §62 / ADR-0202). Section numbering, sub-section anchors (§63.1 …) and
cross-references to other §§ refer to `contracts/semiflow-core.math.md`.

## §63 — Two-sided graded Duhamel quadrature for the graph / symmetric-operator Fréchet gradient (ADR-0203, NORMATIVE library; CITATION mathematics)

> **Scope.** Replaces the *quadrature rule* used by `graph_expmv_frechet` (§54.5 A2,
> and through it `symmetric_op_expmv_frechet`, §55.5). The quantity computed, the
> public signature, the channel layout and the ascending-channel accumulation
> (ADR-0184 D4) are unchanged. Symmetric positive-semidefinite generators only
> (§54.6 boundary, unchanged). No new numerical method class: composite
> Gauss–Legendre on a geometric mesh is textbook (Davis–Rabinowitz 1984 §2.7;
> Schwab 1998 §3.3 for geometric meshes against boundary layers). What this section
> makes NORMATIVE is the mesh, the evaluation order that keeps memory at
> `O(nnz + 12 vectors)`, the bilinear contraction contract, and the a-priori error
> bound the gates use.

### §63.1 — Setting and the quantity

`L` symmetric PSD, `σ(L) ⊂ [0, λ_max]`, `ρ̄ ≥ λ_max` the cached Gershgorin bound
(`Laplacian::spectral_radius_bound`, `SymmetricOperator` analogue). Generator
`A = −L`. Parameters `θ_k` with symmetric directions `M_k := ∂L/∂θ_k`
(`∂A/∂θ_k = −M_k`, the `GeneratorSensitivity` convention). For one channel,
`J = ⟨d, e^{−tL} v⟩` (`d = dj_c`, `v = u0_c`). Duhamel (exact, any `[L, M_k]`):

```text
g_k := ∂J/∂θ_k = ∫₀ᵗ ⟨ a(σ), (∂A/∂θ_k) b(σ) ⟩ dσ,
      a(σ) = e^{−(t−σ)L} d,   b(σ) = e^{−σL} v.                               [§63.1.a]
```

Eigen form (Daleckii–Krein), `L = Σ λ_i φ_i φ_iᵀ`, `α_i = ⟨φ_i,d⟩`, `β_j = ⟨φ_j,v⟩`:

```text
g_k = Σ_{i,j} α_i β_j (φ_iᵀ M_k φ_j) · f¹(λ_i, λ_j),
f¹(x, y) = −∫₀ᵗ e^{−(t−σ)x − σy} dσ
         = −t·e^{−t·min(x,y)}·φ₁(t|x−y|),  φ₁(z) = (1 − e^{−z})/z = −expm1(−z)/z. [§63.1.b]
```

The second line is the cancellation-free form used by every oracle in this section.
Define the **absolute magnitude** and the **norm scale** of parameter `k`:

```text
G_k := Σ_{i,j} |α_i| |β_j| |φ_iᵀ M_k φ_j| |f¹(λ_i, λ_j)|,
N_k := t · ‖M_k‖₂ · ‖d‖₂ · ‖v‖₂.                                               [§63.1.c]
```

`‖M_k‖₂ = 2` for an edge weight (`(e_i−e_j)(e_i−e_j)ᵀ`), `1` for a symmetric entry
pair (`e_ie_jᵀ + e_je_iᵀ`) and for a diagonal entry (`e_ie_iᵀ`).

**§63.1.d Precondition: `L` is PSD (NORMATIVE, Amendment 2).** The Chebyshev
mapping (§54.3, `B = (2/ρ̄)L − I` on `[0, ρ̄]`), the contraction bound
`‖e^{−τL}x‖ ≤ ‖x‖` behind the decay skip (§63.5.b), and Proposition 63.4 all
assume `σ(L) ⊂ [0, ρ̄]`. `Laplacian` satisfies this by construction.
`SymmetricOperator::from_csr` checks symmetry, finiteness and `diag ≥ 0` but does
**not** check positive semidefiniteness: a symmetric matrix with non-negative
diagonal and large off-diagonal entries can be indefinite. For such input the
results of `graph_expmv`, `graph_expmv_frechet` and `symmetric_op_expmv_frechet` are
unspecified and no error is raised. Sufficient, `O(nnz)`-checkable condition: weak
diagonal dominance `L_ii ≥ Σ_{j≠i} |L_ij|` for every row (Gershgorin), which every
conductance network with non-negative leaks satisfies. Callers with other operators
must establish PSD themselves. (Adding a diagonal-dominance check, or an `Unsupported`
error for operators that fail it, is a follow-up API decision, not part of ADR-0203.)

### §63.2 — Root cause of the pre-ADR-0203 error (NORMATIVE diagnosis)

The pre-ADR-0203 rule was ONE 8-point Gauss–Legendre panel on `σ ∈ [0, t]`. Each
mode pair contributes `c_ij·e^{−λ_i t}·e^{μσ}`, `μ = λ_i − λ_j ∈ [−λ_max, λ_max]`.
The Gauss–Legendre remainder on a panel of length `h` (Abramowitz–Stegun 25.4.30) is

```text
E = C_8 · h^{17} · f^{(16)}(ξ),   C_8 = (8!)⁴ / (17 · (16!)³) = 1.697e-23.      [§63.2.a]
```

With `h = t`, `|f^{(16)}| ≤ |μ|^{16}·max|f|`, the relative error of a pair is
`≈ C_8·(|μ|t)^{17}·e^{−(·)}`: below `u` only for `|μ| t ≲ 3`, `≈ 1e−9` at
`|μ|t ≈ 13`, `O(1)` beyond `|μ|t ≈ 30` until the boundary-layer share of the
integral shrinks like `1/(λt)`. The propagators were never the problem:
a numpy replica of the 8-node rule with EXACT eigen propagators reproduces the
semiflow Python errors to all printed digits (ADR-0203 §Evidence). The integrand
has boundary layers of width `1/λ_max` at both ends of `[0, t]` that a fixed rule
cannot resolve.

### §63.3 — The mesh (NORMATIVE)

Split `[0, t]` at `H = t/2` (exact in binary). Each half is graded towards its outer
endpoint. Distances are measured from that endpoint (`r = σ` on the left half,
`r = t − σ` on the right half). Constants: `q = 3/2`, `δ = 2`.

```text
d_0 = min(H, δ/ρ̄)                        (ρ̄ = 0 ⇒ d_0 = H)
d_k = min(H, q·d_{k−1}),  k = 1, 2, …,  stop at the first d_K = H.
panels of one half:  P_0 = [0, d_0],  P_k = [d_{k−1}, d_k]  (k = 1..K).      [§63.3.a]
```

`K = ⌈ log_q(H/d_0) ⌉₊` and every graded panel satisfies `len(P_k) ≤ (q−1)·d_{k−1}`.
The mesh is built by the loop above (multiplications and one `min`), never through
`ln`, so it is bit-identical across `libm` / platform builds (ADR-0200).
Each panel carries the 8 Gauss–Legendre nodes `r = lo + h·x_q`, weights `h·w_q`
(the existing `GL8` table). Node count: `n_nodes = 16·(K + 1)`.

**Distances, never positions (NORMATIVE).** Nodes are carried as their distance `r`
to the outer endpoint of their half. `σ = t − r` is never formed and differenced:
for `λ_max t > 1/u` that subtraction loses all digits of `r` (observed `1.5e−10`
relative error at `|μ|t = 1e7` in the probe of ADR-0203, which disappears when
distances are carried).

### §63.4 — A-priori quadrature bound (NORMATIVE)

**Proposition 63.4.** For every pair term of §63.1.b and every `t > 0`, the
§63.3 rule has relative error, measured against `|f¹(λ_i, λ_j)|`, at most

```text
ε_Q = 1.1e−14.                                                                  [§63.4.a]
```

Consequently, for every parameter `k`: `|g_k^{rule} − g_k| ≤ ε_Q · G_k`
(exact propagators and exact arithmetic assumed; §63.5 adds both).

*Proof sketch.* Take `μ < 0` (the term is largest at `σ = 0`; `μ > 0` is the
mirror image on the other half). Let `S = ∫₀ᵗ e^{−|μ|σ}dσ ≥ (1 − e^{−1})/|μ|` when
`|μ|t ≥ 1`.
(i) Innermost panel `[0, d_0]`: `|μ|·d_0 ≤ ρ̄·δ/ρ̄ = δ`, so by §63.2.a the error
over `S·|μ|` is `≤ C_8·δ^{17} = 2.2e−18`.
(ii) Graded panel `P_k` on the near half: with `x_k = |μ|·d_{k−1}`,
`h ≤ (q−1)·d_{k−1}`, and `max_{P_k} e^{−|μ|σ} = e^{−x_k}`, the panel error over
`S·|μ|` is `≤ C_8·((q−1)x_k)^{17}·e^{−x_k}`. The `x_k` form a geometric sequence of
ratio `q`; `sup_{x_0} Σ_k ((x_k/2)^{17} e^{−x_k}) = 3.95e8` (`q = 3/2`), giving
`6.7e−15`.
(iii) Far half: every panel has `σ ≥ t/2` and length `≤ t/6`, so its error is
`≤ C_8·(x/3)^{17}·e^{−x}`, `x = |μ|t/2`, summed over a geometric sequence of
lengths: `≤ 4.5e−18`.
Sum `(i)+(ii)+(iii)`, times `1/(1 − e^{−1}) = 1.582` for the lower bound on `S`:
`1.06e−14 ≤ ε_Q`. For `|μ|t < 1` the remainder is `≤ C_8·(|μ|h)^{16}·h/t·S·|μ|`,
far smaller. `μ = 0` is integrated exactly. Using `ρ̄ ≥ λ_max` instead of
`λ_max` only shrinks `d_0`. ∎

`ε_Q` is independent of `λ_max·t`: this is the property the old rule lacked. The
probe of ADR-0203 measures `sup = 4.9e−16` over `|μ|t ∈ [1e−3, 1e8]` for single
exponentials, consistent with (not fitted to) the bound.

### §63.5 — Evaluation order and memory (NORMATIVE)

Per half, the vector that is *near* its source is computed directly, the vector that
is *far* from its source is advanced incrementally. On the left half (`r = σ`)
the near vector is `b(σ) = e^{−rL}v` and the far vector is `a(σ) = e^{−(t−r)L}d`.
On the right half (`r = t−σ`) the near vector is `a = e^{−rL}d` and the far vector
is `b = e^{−(t−r)L}v`. One routine serves both halves with the roles swapped; the
contraction always receives `(a, b)` in that order, so no symmetry of `∂A/∂θ_k`
is assumed.

```text
half_sweep(near_src, far_src, roles):
    far ← e^{−H L} far_src;  r_far ← H                     # far vector at r = H
    for panel P_k, k = K, K−1, …, 0:                       # outer → inner
        nodes r_1 < … < r_8 of P_k, weights w_1 … w_8
        near_1 ← e^{−r_1 L} near_src
        near_q ← e^{−(r_q − r_{q−1}) L} near_{q−1},  q = 2..8      # 8 buffers
        for q = 8, 7, …, 1:                                         # r descending
            far ← e^{−(r_far − r_q) L} far;  r_far ← r_q            # distance grows
            accumulate_bilinear(w_q, a, b, grad)   # (a,b) per roles
        if ‖far‖₂ ≤ ε_skip·‖far_src‖₂: return                       # §63.5.b
```

Step lengths are differences of distances (`r_far − r_q`, `r_q − r_{q−1}`),
never of positions. Both halves run per channel, left half first, channels in
ascending order (ADR-0184 D4).

**§63.5.a Memory.** 8 near buffers + 1 far buffer + 1 contraction temporary +
the propagator's own scratch (5 vectors for Chebyshev): `≤ 15·N` floats plus the
operator. Nothing scales with `λ_max·t`, the number of nodes, or `n_params`.

**§63.5.b Decay skip (exact accounting).** Along a chain, `‖e^{−τL}x‖₂ ≤ ‖x‖₂`
(`L` PSD). Once `‖far‖₂ ≤ ε_skip‖far_src‖₂`, every remaining node of the half has
`|⟨a, (∂A/∂θ_k) b⟩| ≤ ‖M_k‖₂·‖far‖₂·‖near_src‖₂`, and the remaining weights sum to
at most `H`. The dropped part is therefore `≤ ε_skip·N_k/2` per half. `ε_skip`
is the propagator tolerance `tol` of the `GraphKrylovChernoff`. When `λ_min·t ≫ 1`
(SPD generator with a leak) the skip removes almost all of the near-vector work.

### §63.6 — Cost (NORMATIVE statement, ADVISORY gate)

Let `C(τ)` be the SpMV count of one action `e^{−τL}x` on the chosen path
(`graph_expmv_matvec_count`). Per channel:

```text
far chains:   2·C(H) + 2·C(H)              (midpoint + sweep, both halves)
near vectors: 2 · Σ_{k=0..K} C(d_k)        (first node direct, then 7 steps)
contractions: n_nodes = 16(K+1) calls of accumulate_bilinear.                  [§63.6.a]
```

**§63.6.c Bound (Amendment 1, replaces the "≈ 5·C(t)" estimate).** Propagated
time per channel: far chains `2·(H + H) = 2t`; near vectors, per half,
`Σ_{k=0..K} d_k ≤ d_K + d_{K−1}·q/(q−1) ≤ 4H` (the last panel is clipped, so
`d_{K−1}` can be close to `H`), i.e. `≤ 4t` for both halves. Total `≤ 6t`, so
`Σ z_i ≤ 3·ρ̄t` over all calls (`z = ρ̄τ/2`).

*Amendment 4 (ADR-0205).* A Chebyshev call is ONE expansion of degree `m(z)`, the
smallest `m` with `2Σ_{k>m} e^{−z}I_k(z) ≤ tol/4`. That tail is `P(|X| > m)` for
the Skellam law `X = P₁ − P₂`, `P_{1,2} ~ Poisson(z/2)` (variance `z`, unit
jumps), so Bennett's inequality gives
`P(|X| ≥ x) ≤ 2·exp(−x²/(2(z + x/3)))` and, with `L = ln(8/tol)`,

```text
m(z) ≤ ⌈L/3 + √(L²/9 + 2zL)⌉ − 1 ≤ 2L/3 + √(2Lz).                            [§63.6.d]
```

(gate `G_CHEB_SQRT_COST`; measured `m/bound ≤ 0.955` for `z ∈ [1e−3, 1e7]`,
`tol ∈ [1e−14, 1e−6]`). Summing over the `N = 2 + 32(K+1)` calls with
`Σ√zᵢ ≤ √(N·Σzᵢ)`:

```text
spmv_upper ≤ B(ρ̄t) := √(6L·N·ρ̄t) + N·2L/3.                                   [§63.6.c]
```

At `tol = 1e−12`: `B = 1.9e4 / 4.8e4 / 1.4e5 / 4.5e5` at
`ρ̄t = 1e3 / 1e4 / 1e5 / 1e6`, against `5.0e4 / 8.3e4 / 2.4e5 / 1.6e6` for the
Amendment-1 bound `m_Z·(3ρ̄t/Z_SAFE + 2 + 32(K+1))` of the substep kernel
(`Z_SAFE = 200`, `m_Z = 101`), which the gate also asserts is never exceeded by
the new `B`. The cost per channel grows like `√(ρ̄t·log ρ̄t)` rather than `ρ̄t`.
For `ρ̄t ≲ 10` the new rule makes 34 calls instead of 16.
The node count is `O(log(ρ̄t))`: `ρ̄t ∈ {1, 10, 1e2, 1e4, 1e6} → K ∈ {0, 3, 8, 20, 31}`,
`n_nodes ∈ {16, 64, 144, 336, 512}`.

**Independence of `n_params` (NORMATIVE).** The number of actions above does not
depend on `n_params`. Per node the contraction is ONE call
`accumulate_bilinear(w, a, b, grad)` that adds `w·⟨a, (∂A/∂θ_k) b⟩` to every
`grad[k]`. Its cost is `O(nnz)` for the three in-tree sensitivities:

```text
EdgeWeightSensitivity:     grad[k] −= w·(a_i − a_j)·(b_i − b_j)
EntrySensitivity (i≠j):    grad[k] −= w·(a_i b_j + a_j b_i);   (i=j): −= w·a_i b_i
NodeTimescaleSensitivity:  p = L(D b), s = L(D a)  (2 SpMV per node), then
                           grad[k] −= w·½a_k^{−½}·(a_k p_k + b_k s_k)            [§63.6.b]
```

The trait default (any external implementor) loops `apply_param_deriv` and is
`O(n·n_params)` per node: unchanged behaviour, documented as the slow path.
The pre-ADR-0203 code used that slow path for ALL sensitivities, so the
"all-edges gradient for the price of a few actions" claim of §54.5 did not hold
before this section (`O(n·|E|)` per node).

### §63.7 — Total error bound used by the gates (NORMATIVE)

Write `u = 2^{−53}`, `r` = max stored entries in a row of `L`, `n` = dimension,
`tol` the propagator tolerance, `N_chain` the largest number of propagator
*evaluations* (Chebyshev expansions, Lanczos outer steps) along any single chain of
§63.5, and `W` the largest rounding weight `Σ wᵢ` along any single chain, with
`wᵢ = ⌈zᵢ/2⌉ + mᵢ` for a Chebyshev evaluation of `zᵢ = ρ̄τᵢ/2` and degree `mᵢ`,
`wᵢ = mᵢ²` for a Lanczos outer step of dimension `mᵢ`. Then

```text
|g_k − g_k^ref| ≤ τ_k := (ε_Q + 2n²u)·G_k + η·N_k,
η = 2·(N_chain·tol + (r+3)·W·u)  +  ε_skip  +  (r + n)·u·ρ̄t.               [§63.7.a]
```

Terms, each derived, none fitted:

1. `ε_Q·G_k`: §63.4. `2n²u·G_k`: rounding of the oracle's double sum §63.1.b.
2. `2·N_chain·tol·N_k`: each evaluation adds `≤ tol·‖input‖₂` (Bessel-tail bound of
   `chebyshev_degree`, §54.3), norms are non-increasing along a chain, and
   `|⟨δa, M b⟩| + |⟨a, M δb⟩| ≤ ‖M‖(‖δa‖‖b‖ + ‖a‖‖δb‖)` integrated over weights
   summing to `t`.
3. `2·(r+3)·W·u·N_k`: forward rounding of the Chebyshev three-term recurrence
   with `‖B‖₂ ≤ 1`. A local error `eⱼ` (`‖eⱼ‖ ≤ (r+3)u‖v‖`, the `r`-term SpMV plus
   the update) injected at step `j` reaches `t_k = T_k(B)v` through `U_{k−j}(B)`,
   and `max_{[−1,1]} |U_i| = i + 1` (Clenshaw 1955; Higham 2002 Ch. 3), so
   `‖δt_k‖ ≤ (r+3)u‖v‖·k(k+1)/2`. The series weights `t_k` by `2c_k`,
   `c_k = e^{−z}I_k(z)`, hence the output error is at most
   `(r+3)u‖v‖·Σ_{k≥1} c_k k(k+1) = (r+3)u‖v‖·(z/2 + Σ c_k k)`: `Σ_{k≥1} c_k k² = z/2`
   is the variance identity of the Skellam distribution `p_k = e^{−z}I_{|k|}(z)`,
   and `Σ c_k k ≤ √(z/2)·√(1/2) ≤ m` (Cauchy–Schwarz). The `+m` also covers the
   coefficient and accumulation rounding (`O(m u)`). Lanczos keeps the `m²`
   per outer step of the original analysis.

   *Amendment 4 (ADR-0205).* The original term was `2·N_chain·(r+3)·m_max²·u`,
   with `m_max ≤ 101` from substeps of `z ≤ 200`. One Chebyshev expansion of
   degree `≈ √(2z ln(1/tol))` replaces those substeps; `m_max²` would then grow
   like `z` per evaluation instead of per chain. `W` sums the actual weights
   along the chain. The new `η` is below the old one at every point of the grid
   (`r ∈ {3, 5, 12}`, `n = 12`, `tol = 1e−12`; ratio new/old at
   `λ_max t = 1, 10, 1e2, 1e4, 1e6`: `0.92–0.97, 0.74–0.88, 0.30–0.52, 0.06–0.13,
   0.04–0.06`), so the gate tightens.
4. `ε_skip·N_k`: §63.5.b, both halves.
5. `(r + n)·u·ρ̄t·N_k`: the *inherent* conditioning. Perturbing `L` by `ΔL` changes
   `g_k` by at most `t·‖ΔL‖₂·N_k` (Duhamel twice; `‖e^{−τL'} − e^{−τL}‖ ≤ τ‖ΔL‖`).
   Any SpMV-based method has backward error `‖ΔL‖ ≤ r·u·‖|L|‖₂ ≤ r·u·ρ̄`; a dense
   symmetric eigensolver (the oracle) has `‖ΔL‖ ≤ n·u·‖L‖₂`. No f64 algorithm
   that reads `L` can beat this term; it is why `λ_max·t = 1e6` cannot deliver
   `1e−15` relative accuracy for slow modes in any library.

`N_chain`, `W` (`chain_weight`) and `n_nodes` are returned by the pure predictor
`graph_expmv_frechet_plan` (§63.8); the gate does not measure them.

**§63.7.b Sharp algorithmic check with exact propagators (NORMATIVE, Amendment 2).**
Term 5 of §63.7.a, the inherent `(r+n)·u·ρ̄t` floor, dominates `τ_k` at large
`ρ̄t`. Combined with `N_k ≫ G_k` for most parameters, this lets the end-to-end
bound accept algorithmic regressions (mesh, weights, sweep step lengths, role
swap, skip) of several orders of magnitude at `λ_max t = 1e6`. The floor is genuine
for any method that applies `L` in floating point (the SpMV `(Lx)_i = L_ii x_i − Σ…`
cancels on slow vectors with error `u·L_ii`), so it cannot be removed from the
end-to-end gate. It can be removed from a second check that runs the library's own
sweep with propagators that apply exact eigenvalues.

Let the sweep of §63.5 be generic over a crate-private propagator
`x ↦ P(τ)x`. Production uses `GraphKrylovChernoff`. The check uses
`P_E(τ)x = Ṽ(e^{−τλ̃} ⊙ (Ṽᵀx))`, where `(λ̃, Ṽ)` comes from the same Jacobi
decomposition as the oracle. Oracle and propagator then describe the same matrix
`Ṽ diag(λ̃) Ṽᵀ`, so the eigensolver's backward error cancels: there is no `ρ̄t`
term. Per call, `‖P_E(τ)x − Ṽ e^{−τΛ̃} Ṽᵀ x‖₂ ≤ κ_E·u·‖x‖₂` with
`κ_E = 2n^{3/2} + n + 2`. The terms are: two dense products, each `≤ γ_n‖|Ṽ|‖₂ ≤ n·√n·u`;
loss of orthogonality of `Ṽ`, `≤ n·u`; and `exp` of an argument carried with relative
error `u`, `≤ (1 + max_z z e^{−z})u ≤ 2u`. Chains restart at every panel (near) or
run once (far), so the longest chain has `L_E = 1 + 8(K+1)` calls. Contraction
rounding is at most `(n_nodes + 4)·u` relative to `Σ w·|bilinear| ≤ N_k`. Hence

```text
|g_k^{sweep,P_E} − g_k^ref| ≤ τ_k^E := (ε_Q + 2n²u)·G_k
                                     + ( 2·L_E·κ_E + n_nodes + 4 )·u·N_k,
evaluated with the decay skip OFF (predicate never fires).                         [§63.7.b]
```

At `n = 12` (`κ_E = 97.1`), `ρ̄t = 1e6` (`K = 31`): `(2·257·97.1 + 516)·u ≈ 5.6e−12`. This is
independent of `ρ̄t` up to `log(ρ̄t)`, and about 600× tighter than the floor term
`(r+n)·u·ρ̄t ≈ 3.5e−9` alone. The gate requires the legacy one-panel rule, driven by the
same `P_E`, to violate `τ_k^E` at every `λ_max t ≥ 10` on every carrier. Under
§63.7.a the legacy rule exceeds the full `τ_k` at `1e6` by only about 4× (reviewer
measurement), so this second check is what catches mesh, weight, step-length and
role-swap regressions that §63.7.a cannot see. It does not test the Chebyshev
propagator; §63.7.a and the existing `G_GRAPH_EXPMV_*` gates do.

*Correction (Amendment 3).* Amendment 2 also claimed this check covers the decay
skip, via a second run with `ε_skip = tol`. That claim was false. On F1 and F3
(Laplacians) the zero mode keeps `‖far‖` above any `ε_skip`, and on F2
`λ_min·t ≲ 2`. Propagator call counts were identical with the skip on and off at
every point, so the skip path never ran. §63.7.c replaces it.

**§63.7.c Decay-skip check (NORMATIVE, Amendment 3).**

*Dropped part.* §63.5 tests the skip once per panel and returns on the first
success, so it fires at most once per half and per channel. At the firing check,
the computed far vector satisfies `‖far‖ ≤ ε_skip‖far_src‖`. With propagator
error, the true far vector satisfies `‖far‖ ≤ (ε_skip + L_E κ_E u)‖far_src‖`.
Norms do not increase along the chain (`L` PSD), every near vector satisfies
`‖near‖ ≤ ‖near_src‖`, and the dropped weights sum to at most `H`. So the dropped
part of `g_k` is at most `(ε_skip + L_E κ_E u)·N_k/2` per half. The same bound
holds for an implementation that keeps sweeping and only skips contractions,
because the dropped weight still totals `≤ H`. Over both halves:

```text
|g_k^{sweep,P_E,skip} − g_k^ref| ≤ τ_k^E + ε_skip·N_k                            [§63.7.c]
```

(`L_E κ_E u·N_k` is already inside `τ_k^E`.) No new constant is needed when the
skip fires: by construction it can fire only once per half. `N_k` is summed over
channels when `n_cols > 1`.

*Why a uniform leak cannot test it.* For `L' = L + cI`, `e^{−τL'} = e^{−cτ}e^{−τL}`.
The Duhamel integrand then carries the constant factor
`e^{−c(t−σ)}·e^{−cσ} = e^{−ct}`, so `g' = e^{−ct}g`. The skip fires only when
`e^{−ct} ≲ ε_skip`, and then the whole gradient is already `≲ ε_skip·N_k`.
Accuracy under a uniform leak (the SOFC ambient leak) is covered by the bound,
but no fixture of that kind can discriminate a broken skip. A discriminating
fixture needs the decay to be non-uniform between the two sources.

*Onset.* Let `dj` lie in the span of eigenvectors with `λ ≥ λ_f`, with residual
slow content `ρ_res·‖dj‖` (projection rounding, `ρ_res ≤ 2n^{3/2}u`). On the left
half the far vector is `a = e^{−DL}dj` at distance `D ∈ [H, t − r_min]`, so
`‖a‖ ≤ (e^{−λ_f D} + ρ_res)‖dj‖`. Consequences:
- The skip is guaranteed to fire by the last check of the left half if
  `λ_f·(t − δ/ρ̄) ≥ ln(2/ε_skip)` and `ρ_res ≤ ε_skip/2`.
- It fires at the first check if `λ_f·t/2 ≥ ln(2/ε_skip)`.
- On the right half, the far vector `b = e^{−DL}u0` keeps its slow content, so the
  right half does not skip.

The right half carries the whole gradient (the boundary layer at `σ → t`), so a
skip that wrongly fires there loses `|g_k|`. That is the discrimination.


### §63.8 — Public surface (NORMATIVE)

```text
graph_expmv_frechet(gk, u0_cols, dj_cols, n_cols, t, sens, grad, scratch)   UNCHANGED
GeneratorSensitivity::accumulate_bilinear(&self, t, w, a, b, grad, scratch)
        -> Result<(), SemiflowError>                    NEW, provided default
graph_expmv_frechet_plan(rho_bar, t, tol, &path) -> FrechetPlan              NEW, pure
FrechetPlan { panels_per_half: u32, n_nodes: u32, propagator_calls: u32,
              n_chain: u64, m_max: u32, chain_weight: u64, spmv_upper: u64 }
GraphKrylovChernoff::lambda_max_bound(&self) -> F,  ::tol(&self) -> F,
                     ::path(&self) -> KrylovPath                              NEW accessors
Python: symmetric_op_expmv_frechet(..., tol: float = 1e-12)                   NEW kwarg
```

`accumulate_bilinear` contract: for every `k`, `grad[k] += w·⟨a, (∂A/∂θ_k) b⟩`,
`a.len() == b.len() == n`, `grad.len() == n_params()`; it must not allocate per `k`
(scratch only). The `KrylovPath::ImplicitEuler` path is accepted but outside the
§63.7 bound: its action carries an `O(Δt)` bias (§59), so the gradient is that of
the backward-Euler propagator to `O(Δt)`. Documented, not gated.

### §63.9 — Acceptance gates (NORMATIVE)

| Gate | Definition | Threshold | Oracle |
|------|-----------|-----------|--------|
| `G_FRECHET_QUAD_CONSTANT` | §63.3 rule on `e^{−|μ|σ}`, `e^{−|μ|(t−σ)}`, `|μ|t ∈ logspace(−3, 8, 600)`, `ρ̄ ∈ {|μ|, 2|μ|}` | rel-err `≤ ε_Q = 1.1e−14` | closed form `−expm1(−|μ|t)/|μ|` |
| `G_FRECHET_LARGE_T_ORACLE` | `graph_expmv_frechet` (Edge, Entry incl. diagonal) on three `n=12` fixtures (F3 clustered edge carrier, Amendment 1), `λ_max t ∈ {1, 10, 1e2, 1e4, 1e6}` | `|g_k − g_k^ref| ≤ τ_k` (§63.7.a) for every `k`; legacy-rule replica (exact propagators) must VIOLATE `τ_k^Q = (ε_Q+2n²u)·G_k + (r+n)·u·ρ̄t·N_k` for some `k` at every `λ_max t ≥ 10`, per carrier; informative `k` per point PER CARRIER (F3-edge ≥ 2/3 bridges, F2-entry ≥ 1), never pooled | Jacobi eigen + §63.1.b; scipy-free |
| `G_FRECHET_BILINEAR_ONE_PASS` | counting wrapper: `accumulate_bilinear` called exactly `n_nodes·n_cols` times, `apply_param_deriv` zero times for overriding impls; override vs default path | `|Δg_k| ≤ 8u·N_k` | default trait path |
| `G_FRECHET_LARGE_T_NOSTD_DIGEST` | new `semiflow-nostd-check` scenario `graph_frechet_large_t` | digest identical in every configuration; FD sanity `≤ 1e−6` | committed digest (ADR-0200) |
| `G_PY_FRECHET_LARGE_T` | `symmetric_op_expmv_frechet` on F2 and F3 (bridge pairs ≥ 2/3 informative), same `λ_max t` grid, `tol=1e−12` | `τ_k` of §63.7.a; scipy `expm_frechet` agrees with the eigen oracle to `1e−12·G_k` for `λ_max t ≤ 1e2` | numpy `eigh` + §63.1.b |
| `G_FRECHET_SWEEP_EXACT_PROP` (Amendments 2, 3) | library sweep (crate-private, generic over propagator and skip predicate) driven by the exact-eigenvalue propagator `P_E`; part A: F1/F2/F3 × `λ_max t` grid, skip off; part B: skip fixture S1 (F3 with `dj` projected onto the fast eigenspace), `ε_skip = 1e−12` | A: `≤ τ_k^E` (§63.7.b), legacy rule violates `τ_k^E` at every `λ_max t ≥ 10` per carrier; B: `≤ τ_k^E + ε_skip N_k` (§63.7.c), fewer calls with skip on at `λ_max t ≥ 1e2`, broken skips (always-fire; absolute norm on signals ×1e−16) violate it | same Jacobi decomposition as `P_E` |
| `G_FRECHET_COST_LOG_NODES` (ADVISORY) | `n_nodes(ρ̄t)` formula; `n_nodes(1e6)/n_nodes(1e2) ≤ 4`; `spmv_upper ≤ B(ρ̄t)` (§63.6.c); all-edges vs one-edge wall time on `N=2000` | ratio `≤ 2` | instrumentation |

Unchanged and still RELEASE_BLOCKING: `G_GRAPH_FRECHET_FD` (`≤ 1e−7`),
`tests/graph_frechet_fd.rs::g_graph_frechet_fd_triangle` (`≤ 1e−7`),
`G_SYMOP_ENTRY_FRECHET` (`≤ 1e−7` per entry), `T_ADJOINT_STATE_SENSITIVITY`
(§43, untouched code path but the trait gains a default method).

### §63.10 — References

- M. Abramowitz, I. A. Stegun (1964), *Handbook of Mathematical Functions*, 25.4.29–30.
- P. J. Davis, P. Rabinowitz (1984), *Methods of Numerical Integration*, 2nd ed., §2.7.
- C. Schwab (1998), *p- and hp-Finite Element Methods*, §3.3 (geometric meshes).
- Ju. L. Daleckii, S. G. Krein (1965), *Integration and differentiation of functions
  of Hermitian operators*, AMS Transl. 47:1–30.
- A. H. Al-Mohy, N. J. Higham (2009), *Computing the Fréchet derivative of the matrix
  exponential*, SIAM J. Matrix Anal. Appl. 30(4):1639–1657 (scipy `expm_frechet`).
- C. W. Clenshaw (1955), *A note on the summation of Chebyshev series*, MTAC 9:118–120.
- N. J. Higham (2002), *Accuracy and Stability of Numerical Algorithms*, 2nd ed., SIAM.
- ADR-0203 (authority), ADR-0185 (§54), ADR-0186 (§55), ADR-0200 (digests), ADR-0184 (D4).
