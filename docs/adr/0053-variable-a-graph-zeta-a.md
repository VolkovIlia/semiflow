# ADR-0053 — Variable-coefficient graph diffusion `VarCoefGraphHeatChernoff`

- **Status**: ACCEPTED (v2.2 Wave A — implementation shipped; G13 slope gate pending HW run)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave A (graph time-dependence)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0047, ADR-0048, ADR-0049 (math.md §12), ADR-0003
  (ζ-A τ²-correction on 1D), ADR-0046 (precision-policy bands).
- **Supersedes / amends**: nothing; answers v2.0 "Out of scope" item
  "Variable-a graph ζ-A" (ROADMAP.md line ~440).
- **Mathematical foundation**: math.md §14.2 (CITATION: Belkin-Niyogi 2008
  "Towards a theoretical foundation for Laplacian-based manifold methods"
  for the symmetric conjugation `L̃ = D^{−½/a} L D^{−½·a}`; NORMATIVE:
  ζ-A correction expansion through τ²).

## Context

v2.1 ships graph heat semigroups for the *combinatorial* Laplacian
`L_G[i,j] = −w(i,j) + δ_{ij}·deg_w(i)` — equivalent to assuming uniform
conductivity `a(·) ≡ 1`. Real applications (heat conduction in
heterogeneous power grids, drug diffusion in tissue networks, opinion
dynamics on social graphs with node-specific stubbornness) require a
NODE-DEPENDENT conductivity coefficient `a: V → (0, ∞)`.

The natural generalisation is the *weighted* Laplacian
`L_a := A^{1/2} L_G A^{1/2}` where `A = diag(a(0), …, a(N−1))`. The
order-1 Chernoff `S(τ) = I − τ · L_a` then corresponds to the discrete
analogue of `∂_t u = ∂·(a·∂u)` in the continuum, but suffers the same
order-collapse to global order-1 that v0.3.x's `DiffusionChernoff`
required ζ-A correction to fix.

## Decision

Introduce `VarCoefGraphHeatChernoff<F: SemiflowFloat = f64>` —
order-2 Chernoff with τ²-correction matching the BCH expansion of
`exp(−τ · L_a)`:

```rust
//! crates/semiflow-core/src/graph_var_coef.rs (NEW FILE, ~360 LoC)

pub struct VarCoefGraphHeatChernoff<F: SemiflowFloat = f64> {
    /// Topology + base Laplacian (combinatorial, `a ≡ 1`).
    graph: Arc<Graph<F>>,
    laplacian: Arc<Laplacian<F>>,
    /// Per-node conductivity `a: V → (0, ∞)`. Length N.
    a: Vec<F>,
    /// Cached `√a`. Used in the conjugation and in the τ²-correction.
    sqrt_a: Vec<F>,
    /// Spectral-radius bound for safety (caller-supplied).
    rho_bar: F,
}

impl<F: SemiflowFloat> VarCoefGraphHeatChernoff<F> {
    /// Construct from topology + conductivity.
    /// `a` MUST have `a.len() == graph.n_nodes()`; each `a[i] > 0` and finite.
    pub fn new(
        graph: Arc<Graph<F>>,
        a: Vec<F>,
        rho_bar: F,
    ) -> Result<Self, SemiflowError>;
}

impl<F: SemiflowFloat> ChernoffFunction<F> for VarCoefGraphHeatChernoff<F> {
    type S = GraphSignal<F>;
    fn apply_into(/* ... */) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 } // order-2 on smooth a; falls to 1 on rough a
    fn growth(&self) -> (f64, f64) { (1.0, self.rho_bar.to_f64().unwrap() * a_sup_2) }
}
```

**The ζ-A correction kernel.** Following ADR-0003 (1D ζ-A), the
τ²-correction term for graph operator `L_a = A^{1/2} L_G A^{1/2}` is

```text
S(τ) f := f − τ · L_a · f + (τ² / 2) · L_a² · f
        − (τ² / 12) · D_a^{(2)} · f
```

where `D_a^{(2)}` is the discrete graph analogue of the *Stratonovich*
correction `(1/2) ∂_x (a · ∂_x ((d/dx) ln(a) · ∂_x ·))` from 1D. On a
graph, this collapses to (math.md §14.2):

```text
[D_a^{(2)} f]_i := Σ_{j ∈ N(i)}   w(i, j) · (a(j) − a(i)) ·
                                   (a(j)·f(j) − a(i)·f(i))
                                   / (a(i) · a(j))
```

(symmetric — `(a_j − a_i)·(a_jf_j − a_if_i)` is symmetric in `i↔j` under
the weight-symmetric edge sum). This adds one extra sparse mat-vec
beyond `L_a² · f`, giving total cost 4 sparse mat-vecs per
`apply_into` (vs Wave 2.1A's 1 mat-vec).

The implementation reuses the v0.3.x ζ-A code skeleton (per-node
`a_at_node` cache + 1 sparse mat-vec for `L_a`, 1 for `L_a²`, 1 for
`D_a^{(2)}`). No new dependency.

## Rationale

- **Order-2 on smooth `a`**. Verified by self-convergence to slope ≤ −1.95.
  On rough `a` (e.g., `a(i) ∈ {0.1, 10}` randomly per-node) the τ²-correction
  doesn't help, and the order drops to 1 (matches 1D ζ-A regime envelope
  per MEMORY "Validated regime v0.3.1" — σ < 0.50 ∧ σ·T·β < 0.40 there;
  here, the analogous regime is `a_sup / a_inf < 5` for graph diameter ≲ 50).
- **Reuses sparse mat-vec infrastructure**. `Laplacian::apply_into_slice`
  is the hot path; ζ-A is layered above it without dense-matrix
  intermediates.
- **No new traits**. `VarCoefGraphHeatChernoff` is a plain
  `ChernoffFunction<F, S = GraphSignal<F>>` implementation — drops into
  `ChernoffSemigroup::evolve` and `StrangSplitGraph` unchanged.
- **CFL constraint**: `τ · ρ̄ · max(a) < 0.5` to keep τ²-correction
  numerically stable. Documented in constructor rustdoc.

## Consequences

- New module `src/graph_var_coef.rs` (~360 LoC); under file cap.
- Public surface +1 type. Additive minor.
- `lib.rs` re-export adds `VarCoefGraphHeatChernoff`.
- Cross-talk with `MagnusGraphHeatChernoff`: ADR-0053 is for *time-independent*
  variable conductivity. The product
  `VarCoefMagnus<a, w(t)>` (variable-a AND variable edge-weight) is OUT
  of v2.2 scope — see Risks §R3.

## Acceptance gates

- **G13 slope gate** (NORMATIVE). Path graph `P_n` with `n ∈ {32, 64, 128,
  256}`, `a(i) = 1 + 0.5·cos(2π · i/n)`, `t_final = 0.05`. Self-convergence
  at 2× refinement in `n_steps`. OLS slope on `(log n_steps, log err_sup)`
  over `n_steps ∈ {25, 50, 100, 200, 400}`. Threshold: slope ≤ −1.95 (f64,
  smooth `a`), slope ≤ −1.50 (f32, ADR-0046 precision-band).
- **T12N_var_a_graph sympy gate** (NORMATIVE) — derive the τ²-correction
  symbolically on a 4-node path with symbolic `a = [a_0, a_1, a_2, a_3]`,
  verify the closed-form matches the formula in math.md §14.2 within
  `simplify`-normal form. Pure symbolic; no library runtime.

## Out of scope (v2.2)

- **Combined variable-a + time-varying** (`VarCoefMagnusGraphHeatChernoff`).
  Cross-product of ADR-0051 + ADR-0053. Deferred to v2.3+ once both
  individually stabilise.
- **Higher-order ζ-A (τ⁴, τ⁶) on graphs**. Order-4 ζ-A would require
  nested commutators `[L_a, [L_a, D_a^{(2)}]]` — high implementation
  complexity for marginal use. Deferred.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `a(i)` near zero causes division blow-up in `D_a^{(2)}` | Constructor rejects `a[i] < 1e-12 · max(a)`. |
| R2 | f32 underflow at `a_sup/a_inf > 100` | f32 slope threshold relaxed to −1.50 per ADR-0046; documented in G13. |
| R3 | Combining ADR-0053 + ADR-0051 (var-a AND var-weight) tempting but unsupported | rustdoc explicitly forbids; ADR-0053 §"Out of scope" §R3 cross-references this risk. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/graph_var_coef.rs` | ~360 |
| `tests/g13_var_a_graph_slope.rs` | ~140 |
| `.dev-docs/verification/scripts/verify_v2_2_variable_a_graph_residual.py` | ~150 |
| math.md §14.2 | ~120 |
| ADR-0053 (this) | ~200 |
| **Total** | **~970** |

## References

- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) — Theorem 6.
- M. Belkin, P. Niyogi, *Found. Comp. Math.* **8** (2008) 7–37 —
  symmetric Laplacian conjugation `L_a = A^{1/2} L_G A^{1/2}`.
- F. R. K. Chung, *Spectral Graph Theory* (1997), §1.2 — weighted Laplacian.
- ADR-0003 (ζ-A τ²-correction on 1D) — direct precedent for the discrete
  formula.
