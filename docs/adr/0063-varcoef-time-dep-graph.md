# ADR-0063 — Variable-coefficient time-dependent graph Magnus `VarCoefMagnusGraphHeatChernoff`

- **Status**: ACCEPTED (v2.4 Wave A)
- **Date**: 2026-05-22
- **Wave**: v2.4 Wave A (Graph Completeness)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0051 (Magnus K=4 graph), ADR-0053 (Variable-a
  graph ζ-A), ADR-0054 (Time-discontinuous graph Laplacian),
  ADR-0026 (Generic-over-Float), ADR-0046 (precision-policy bands).
- **Mathematical foundation**: math.md §20 (CITATION: Iserles+ 2000
  §6, Blanes+ 2009 §3, Horn-Johnson 2013 §7.5; NORMATIVE: the closure-driven
  GL₂ sampling of both `a(t)` AND `L_G(t)`).

## Context

`VarCoefGraphHeatChernoff` (ADR-0053) handles variable node coefficient
`a(x)` via the symmetric scaling `L_a = A^{1/2} L_G A^{1/2}` — but only
for **fixed-time** problems. Separately, `MagnusGraphHeatChernoff`
(ADR-0051) handles **time-dependent** Laplacian `L_G(t)` via order-4
Magnus with GL₂ quadrature — but constant-coefficient only.

There is no composition. Users with both `a(t)` and `L_G(t)` (network
diffusion where edge weights AND node mobility change with time —
e.g., temperature-dependent diffusivity in a network of varying
connectivity) must manually compose ζ-A scaling per step inside their
Magnus driver — error-prone and bypasses convergence-radius checks.

The fix: ship `VarCoefMagnusGraphHeatChernoff<F>` that samples **both**
`a(t)` AND `L_G(t)` at the GL₂ abscissae `c₁ = (3 − √3)/6`, `c₂ = (3 +
√3)/6`, assembles `L_a(c_i τ) = sqrt(a(c_i τ)) ⊙ L_G(c_i τ) ⊙ sqrt(a(c_i
τ))` at each, and feeds the operator-form into the existing K=4 Magnus
machinery.

## Decision

Ship `VarCoefMagnusGraphHeatChernoff<F: SemiflowFloat = f64>` in a NEW
module `crates/semiflow-core/src/varcoef_magnus_graph.rs`. Implements
`ChernoffFunction<F, S = GraphSignal<F>>` with `order() == 4`.

```rust
pub type WeightAtTime<F> = Arc<dyn Fn(F) -> Vec<F> + Send + Sync>;

pub struct VarCoefMagnusGraphHeatChernoff<F: SemiflowFloat = f64> {
    n_nodes: usize,
    lap_at_t: LaplacianAtTime<F>,    // alias for SegmentWeightFn<F>
    a_at_t: WeightAtTime<F>,          // length-N vector at time t
    rho_bar_max: F,                   // max_t rho_bar(L_G(t))
    a_sup_max: F,                     // max_t sqrt(max_i a_i(t))
    convergence_radius_check: bool,
}

impl<F: SemiflowFloat> VarCoefMagnusGraphHeatChernoff<F> {
    pub fn new(
        n_nodes: usize,
        lap_at_t: LaplacianAtTime<F>,
        a_at_t: WeightAtTime<F>,
        rho_bar_max: F,
        a_sup_max: F,
    ) -> Result<Self, SemiflowError>;

    pub fn with_radius_check(self, enabled: bool) -> Self;
}

/// Helper: estimate (rho_bar_max, a_sup_max) over [t0, t1] via n_samples points.
pub fn compute_rho_bar<F: SemiflowFloat>(
    lap_at_t: &LaplacianAtTime<F>,
    a_at_t: &WeightAtTime<F>,
    interval: (F, F),
    n_samples: usize,
) -> (F, F);
```

The `apply_into` kernel:

1. **Sample** at each GL₂ abscissa `c_i · τ`:
   - `lap_at_t(c_i · τ)` → `Arc<Laplacian<F>>`
   - `a_at_t(c_i · τ)` → `Vec<F>` (length `N`, all entries finite > 0;
     DomainViolation if not)
2. **Compute** `sqrt_a_i[k] = a_i[k].sqrt()` into a pre-allocated buffer.
3. **Build** the operator closure `apply_A_i(v, out) = A_i · v = −L_a(c_i·τ) · v`
   via `apply_la_on_slice(L_G(c_i·τ), sqrt_a_i, v, out, tmp1, tmp2)`.
4. **Assemble** `Ω_4(τ) = (τ/2)(A_1 + A_2) + (√3 τ²/12) [A_2, A_1]`.
5. **Apply** `exp(Ω_4) · src` via degree-4 Taylor truncation (factored
   `apply_exp_omega4_kernel` from `magnus_graph.rs:649`, refactored
   per the "operator-form" change below).
6. **Check** `ρ̄_max · a_sup_max² · τ < π/2`, else
   `SemiflowError::OutOfMagnusRadius { tau, rho_estimate }`.

### Operator-form: code duplication (NORMATIVE)

The original plan considered refactoring `apply_omega4`
(`magnus_graph.rs:727`) to accept a `&dyn Fn(&[F], &mut [F])` operator
closure so `VarCoefMagnusGraphHeatChernoff` could reuse the same
inner kernel. In practice the operator-closure approach requires
interior mutability (or `&mut` callback) on the scratch buffers used
by `apply_la_on_slice`, which made the borrow-checker contortions
worse than the duplicated math. **The decision is to duplicate the
~90-LoC `apply_omega4` + `apply_exp_omega4_kernel` pair into
`varcoef_magnus_graph.rs` as `apply_omega4_la` +
`apply_exp_omega4_la_kernel`** (the duplication path documented as
the safe fallback in this ADR's R1).

`apply_la_on_slice` (`graph_var_coef.rs:208`) is promoted to
`pub(crate)` and shared. `magnus_graph.rs` is **NOT** modified, so
the G11 byte-equality gate (`tests/g11_magnus_graph_slope.rs`) passes
unchanged — verified post-implementation.

### Precision policy (NORMATIVE)

Generic `<F: SemiflowFloat>`. The new operations are sqrt + element-wise
multiplication — both stable at f32. Per ADR-0046 the f32 slope band
for K=4 graph kernels is ≤ −3.50; same threshold applies here. f64
band: ≤ −3.85.

## Rationale

- **Closes the v2.3 gap.** v2.3 Phase 5 shipped `VarCoefGraphHeat`
  Python binding (constant-time) and `MagnusGraphHeat` Python binding
  (constant-coefficient time-dependent), but no composition. v2.4
  unifies both via this type.
- **Re-uses existing infrastructure.** `apply_la_on_slice`
  (`graph_var_coef.rs:208`), `LaplacianAtTime<F>`
  (`magnus_graph.rs:121`), `validate_magnus_radius`
  (`magnus_graph.rs:462`) — all reused unchanged.
- **Caller-side radius estimation, with optional helper.**
  `rho_bar_max` and `a_sup_max` are caller-supplied for predictability
  (Magnus convergence radius is provably tight only with worst-case
  bounds), but `compute_rho_bar` is exposed for users who do not have
  closed-form bounds.
- **Cross-binding parity from day 1.** Python + FFI + WASM exposed in
  same milestone (P3/P4 of v2.4).

## Consequences

- `src/varcoef_magnus_graph.rs` projected ~380 LoC (under 500-LoC cap).
- `src/magnus_graph.rs` operator-form refactor ~+50 LoC (still under
  the per-file cap from ADR-0051; existing G11 slope gate proves
  byte-equality post-refactor).
- Public surface +1 type + 1 helper function + 1 type alias
  (`WeightAtTime<F>`). Additive minor bump.
- `lib.rs` re-exports `VarCoefMagnusGraphHeatChernoff`,
  `WeightAtTime`, `compute_rho_bar`.

## Acceptance gates

- **G22 slope gate** (NORMATIVE). Time-dependent `P_64`,
  `w(t) = 1 + 0.3·sin(πt)`, `a_i(t) = 1 + 0.5·cos(πt) · i / N`,
  `t_final = 0.5`, `n_steps ∈ {5, 10, 20, 40, 80}`. Slope ≤ −3.85
  (f64) / ≤ −3.50 (f32).
- **T17N sympy gate** (NORMATIVE). On a symbolic 4×4 path Laplacian
  with degree-1 polynomial `w(t)` and `a(t)`, verify
  `Ω_4_library(τ) − Ω_true(τ) = O(τ⁵)` via sympy series expansion
  through `τ⁴`. Pure symbolic; no library runtime.
- **G11 byte-equality regression** (NORMATIVE). Existing
  `tests/g11_magnus_graph_slope.rs` for `MagnusGraphHeatChernoff` MUST
  produce byte-identical output post-refactor to closure-form
  `apply_omega4`. If not, refactor is reverted.
- **P3 OutOfMagnusRadius proptest** (NORMATIVE). Existing
  `graph_proptest_magnus.rs` extended: random `(L_G(t), a(t))` with
  `ρ̄ · a_sup² · τ ≥ π/2` MUST return `OutOfMagnusRadius`.

## Out of scope (v2.4)

- **Variable-coefficient K=6.** Composing this with Magnus K=6
  (ADR-0056) would need 3 sample points × (lap + a) × 4 commutators —
  ~24 SpMV per step. Deferred to v2.5+ if customer demand emerges.
- **Time-discontinuous a(t).** Currently `a_at_t` is presumed
  continuous. ADR-0054 covers time-discontinuous `L_G(t)`; the same
  treatment for `a(t)` (jump detection + restart) is deferred to v2.5+.
- **Non-symmetric L_a.** Would lose self-adjointness; out of scope
  indefinitely (whole graph stack assumes symmetric PSD).

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `apply_omega4` operator-form refactor breaks bit-equality of G11 | RESOLVED — chose duplication path from day 1. G11 unaffected because `magnus_graph.rs` is not modified. |
| R2 | Caller-supplied `rho_bar_max` / `a_sup_max` underestimate true bound → Magnus diverges silently | `convergence_radius_check: true` by default; runtime check at every step. `compute_rho_bar` helper exposed. |
| R3 | `a(t)` returns Vec each call → allocator pressure | `WeightAtTime` is `Fn(F) → Vec<F>`; library caches the vec inside `apply_into` via a reusable scratch buffer (one allocation amortised across τ-step loop). |
| R4 | f32 `sqrt(a)` round-off near `a → 0` | DomainViolation if `a_i ≤ 0`; rustdoc warns to use `a_i ≥ epsilon` in practice. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/varcoef_magnus_graph.rs` | ~380 |
| `src/magnus_graph.rs` refactor | ~+50 |
| `tests/varcoef_magnus_graph_zero_alloc.rs` | ~70 |
| `tests/varcoef_magnus_proptest.rs` | ~130 |
| `tests/varcoef_magnus_slope.rs` (G22) | ~140 |
| `scripts/verify_varcoef_magnus_graph_sympy.py` | ~160 |
| math.md §20 | ~100 |
| ADR-0063 (this) | ~210 |
| **Total** | **~1240** |

## References

- A. Iserles, H. Z. Munthe-Kaas, S. P. Nørsett, A. Zanna, *Acta
  Numerica* **9** (2000) §6 — Magnus expansion.
- S. Blanes, F. Casas, J. A. Oteo, J. Ros, *Phys. Rep.* **470** (2009)
  §3 + Tables 5–6 — Magnus expansion convergence + weights.
- R. A. Horn, C. R. Johnson, *Matrix Analysis* (2nd ed., Cambridge
  2013), §7.5 — Schur product preserves PSD.
- ADR-0051 (Magnus K=4 graph) — base Magnus machinery.
- ADR-0053 (Variable-a graph ζ-A) — `L_a` operator-form precedent.
- ADR-0054 (Time-discontinuous `L_G(t)`) — companion for discontinuous
  weight handling (out of scope for `a(t)` in v2.4).
