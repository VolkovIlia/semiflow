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

For the Chebyshev path with `ρ̄t ≫ 2·Z_SAFE` (substepped, `C(τ) ≈ c·ρ̄τ`), the
geometric sum gives `Σ_k C(d_k) ≤ q/(q−1)·C(H) = 3·C(H)`, so the total is
`≈ 5·C(t)` SpMVs, against `≈ 8·C(t)` for the old 16-action rule (which was also
wrong). For `ρ̄t ≲ 10` the per-call minimum degree dominates and the new rule costs
`~2×` the old one in calls (`n_nodes = 16` or `32` instead of 8 GL nodes × 2).
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
*evaluations* (Chebyshev substeps, Lanczos outer steps) along any single chain of
§63.5, `m_max` the largest polynomial degree among them. Then

```text
|g_k − g_k^ref| ≤ τ_k := (ε_Q + 2n²u)·G_k + η·N_k,
η = 2·N_chain·(tol + (r+3)·m_max²·u)  +  ε_skip  +  (r + n)·u·ρ̄t.          [§63.7.a]
```

Terms, each derived, none fitted:

1. `ε_Q·G_k`: §63.4. `2n²u·G_k`: rounding of the oracle's double sum §63.1.b.
2. `2·N_chain·tol·N_k`: each evaluation adds `≤ tol·‖input‖₂` (Bessel-tail bound of
   `chebyshev_degree`, §54.3), norms are non-increasing along a chain, and
   `|⟨δa, M b⟩| + |⟨a, M δb⟩| ≤ ‖M‖(‖δa‖‖b‖ + ‖a‖‖δb‖)` integrated over weights
   summing to `t`.
3. `2·N_chain·(r+3)m_max²u·N_k`: forward rounding of an `m`-term Chebyshev three-term
   recurrence with `‖B‖₂ ≤ 1` is `O(m²u)` per evaluation: a local error `e` injected
   at step `j` propagates like a Chebyshev polynomial of degree `≤ m − j`, and
   `max_{[−1,1]} |U_k| = k + 1` (Clenshaw 1955; Higham 2002 Ch. 3 for the
   running-error framework), times the `r`-term SpMV inner products. Conservative.
4. `ε_skip·N_k`: §63.5.b, both halves.
5. `(r + n)·u·ρ̄t·N_k`: the *inherent* conditioning. Perturbing `L` by `ΔL` changes
   `g_k` by at most `t·‖ΔL‖₂·N_k` (Duhamel twice; `‖e^{−τL'} − e^{−τL}‖ ≤ τ‖ΔL‖`).
   Any SpMV-based method has backward error `‖ΔL‖ ≤ r·u·‖|L|‖₂ ≤ r·u·ρ̄`; a dense
   symmetric eigensolver (the oracle) has `‖ΔL‖ ≤ n·u·‖L‖₂`. No f64 algorithm
   that reads `L` can beat this term; it is why `λ_max·t = 1e6` cannot deliver
   `1e−15` relative accuracy for slow modes in any library.

`N_chain`, `m_max` and `n_nodes` are returned by the pure predictor
`graph_expmv_frechet_plan` (§63.8); the gate does not measure them.

### §63.8 — Public surface (NORMATIVE)

```text
graph_expmv_frechet(gk, u0_cols, dj_cols, n_cols, t, sens, grad, scratch)   UNCHANGED
GeneratorSensitivity::accumulate_bilinear(&self, t, w, a, b, grad, scratch)
        -> Result<(), SemiflowError>                    NEW, provided default
graph_expmv_frechet_plan(rho_bar, t, tol, &path) -> FrechetPlan              NEW, pure
FrechetPlan { panels_per_half: u32, n_nodes: u32, propagator_calls: u32,
              n_chain: u64, m_max: u32, spmv_upper: u64 }
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
| `G_FRECHET_LARGE_T_ORACLE` | `graph_expmv_frechet` (Edge, Entry incl. diagonal) on two `n=12` fixtures, `λ_max t ∈ {1, 10, 1e2, 1e4, 1e6}` | `|g_k − g_k^ref| ≤ τ_k` (§63.7.a) for every `k`; legacy-rule replica (exact propagators) must VIOLATE `τ_k^Q = (ε_Q+2n²u)·G_k + (r+n)·u·ρ̄t·N_k` for some `k` at every `λ_max t ≥ 10`; `≥ 1` informative `k` per point | Jacobi eigen + §63.1.b; scipy-free |
| `G_FRECHET_BILINEAR_ONE_PASS` | counting wrapper: `accumulate_bilinear` called exactly `n_nodes·n_cols` times, `apply_param_deriv` zero times for overriding impls; override vs default path | `|Δg_k| ≤ 8u·N_k` | default trait path |
| `G_FRECHET_LARGE_T_NOSTD_DIGEST` | new `semiflow-nostd-check` scenario `graph_frechet_large_t` | digest identical in every configuration; FD sanity `≤ 1e−6` | committed digest (ADR-0200) |
| `G_PY_FRECHET_LARGE_T` | `symmetric_op_expmv_frechet` on the stiff fixture, same `λ_max t` grid, `tol=1e−12` | `τ_k` of §63.7.a; scipy `expm_frechet` agrees with the eigen oracle to `1e−12·G_k` for `λ_max t ≤ 1e2` | numpy `eigh` + §63.1.b |
| `G_FRECHET_COST_LOG_NODES` (ADVISORY) | `n_nodes(ρ̄t)` formula; `n_nodes(1e6)/n_nodes(1e2) ≤ 4`; all-edges vs one-edge wall time on `N=2000` | ratio `≤ 2` | instrumentation |

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
