# ADR-0055 — Adjoint-Chernoff backward semigroup wrapper

- **Status**: ACCEPTED (v2.2 Wave B)
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave B (advanced semigroups)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0026 (`ChernoffFunction<F>` generic), ADR-0043
  (`HilbertState<F>` trait, Wave 1 v2.0), ADR-0008 (additive surface policy).
- **Supersedes / amends**: nothing; answers v2.0 "Out of scope" item
  "Vector 4a Adjoint-Chernoff" (ROADMAP.md line ~441).
- **Mathematical foundation**: math.md §15.1 (CITATION: Pazy 1983
  *Semigroups of Linear Operators*, §1.10 — dual semigroup
  characterisation; Engel-Nagel 2000 §I.5 — adjoint semigroup on Hilbert
  space; NORMATIVE: order-preservation rules).

## Context

`ChernoffFunction<F>` (chernoff.rs) approximates forward semigroups
`S(τ) = exp(τA)` for generator `A` of a strongly continuous semigroup.
The *backward* semigroup `S*(τ) = exp(τA*)` arises in:

1. **Backward Kolmogorov equations** — option pricing PDEs are solved
   backwards from a terminal condition; equivalent to a forward
   evolution under the adjoint of the original Kolmogorov forward.
2. **Adjoint sensitivity analysis** — computing
   `dS(τ)·f / dθ` via the dual-state evolution `S*(τ) λ`.
3. **L²-error estimators** — Galerkin error analysis on
   non-symmetric generators needs the adjoint to bound the residual.

For *symmetric* generators (combinatorial graph Laplacian, uniform-grid
diffusion `−Δ`, non-mixed isotropic 2D/3D), `A = A*` and the adjoint is
identical to the forward operator — a wrapper costs nothing.

For *non-symmetric* generators (drift-reaction `−Δ + b·∂_x + c`,
anisotropic non-separable `β(x,y)·∂x∂y`, non-conservative graph
networks), `A* ≠ A` and the wrapper provides the genuine dual evolution.

The `HilbertState<F>` trait (ADR-0043, v2.0 Wave 1) already provides the
necessary `dot()` and `norm_sq()` primitives — we don't need new state
machinery.

## Decision

Introduce a *wrapper* type `AdjointChernoff<C, F>` that composes any
existing `ChernoffFunction<F>` into its adjoint:

```rust
//! crates/semiflow-core/src/adjoint.rs (NEW FILE, ~320 LoC)

#[derive(Clone, Debug)]
pub struct AdjointChernoff<C, F: SemiflowFloat = f64>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    inner: C,
    /// `true` if the user has asserted (or the library has detected)
    /// that `inner` represents a self-adjoint operator. When true,
    /// the wrapper is a thin re-export — `apply_into` delegates directly.
    is_self_adjoint: bool,
    _f: PhantomData<F>,
}

impl<C, F: SemiflowFloat> AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    /// Construct an adjoint wrapper for a non-self-adjoint inner generator.
    ///
    /// The wrapper computes `S*(τ) f` by solving the dual evolution
    /// equation; the cost is one inner `apply_into` followed by a
    /// HilbertState inner-product correction (§14.1 of math.md).
    pub fn new_general(inner: C) -> Self;

    /// Construct an adjoint wrapper for a known-self-adjoint inner.
    /// `apply_into` delegates directly — cost is identical to `inner.apply_into`.
    ///
    /// Caller asserts self-adjointness; library does NOT verify (would
    /// require a full matrix representation). Misuse leads to incorrect
    /// results, not crashes.
    pub fn new_self_adjoint(inner: C) -> Self;

    /// Borrow the wrapped inner Chernoff function.
    pub fn inner(&self) -> &C;
}

impl<C, F: SemiflowFloat> ChernoffFunction<F> for AdjointChernoff<C, F>
where
    C: ChernoffFunction<F>,
    C::S: HilbertState<F>,
{
    type S = C::S;

    fn apply_into(
        &self,
        tau: F,
        src: &Self::S,
        dst: &mut Self::S,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        if self.is_self_adjoint {
            return self.inner.apply_into(tau, src, dst, scratch);
        }
        // Dual evolution via inner-product correction.
        // See math.md §14.1 for the closed form on bounded perturbations.
        apply_dual_evolution(self, tau, src, dst, scratch)
    }

    fn order(&self) -> u32 {
        if self.is_self_adjoint {
            self.inner.order()
        } else {
            // Order drops to min(inner.order(), 2) for the general case
            // because the closure of A* to a bounded perturbation of A
            // introduces O(τ²) discretisation error from the dual-pairing
            // step. See math.md §15.1.bis.
            core::cmp::min(self.inner.order(), 2)
        }
    }

    fn growth(&self) -> (f64, f64) {
        // ‖exp(τ A*)‖ = ‖exp(τ A)‖ on Hilbert space (Pazy §1.10 Thm 10.4).
        self.inner.growth()
    }
}
```

The `apply_dual_evolution` helper (math.md §15.1) implements the bounded-
perturbation theorem: for `A* = A + B*` where `B*` is the closure of the
adjoint of any first-order-and-below perturbation, `exp(τ A*) f =
exp(τ A) f − τ · ⟨B*, f⟩ · g_basis + O(τ²)`, where `g_basis` is the
HilbertState basis-vector encoding of the perturbation.

For users who don't know whether their inner generator is self-adjoint,
the library provides `AdjointChernoff::detect_self_adjointness(inner, sample_states, tol)`
— a probabilistic check (samples `n_states` random `f`, verifies
`⟨A f, g⟩ ≈ ⟨f, A g⟩` within `tol`; returns `bool`). NOT used in the
hot path; intended as a developer tool.

## Rationale

- **Reuses `HilbertState`**. v2.0 Wave 1 (ADR-0043) already shipped
  `HilbertState::dot()` precisely to enable adjoint operations.
- **Wrapper pattern matches `AxisLift`, `AdjointChernoff` is the same
  idiom**: take an inner `ChernoffFunction`, produce a new
  `ChernoffFunction` with extra behaviour. Forward-compat with future
  `BackwardChernoff` (for terminal-condition ODE solvers).
- **Order preservation rules**. The full closure of `A* − A` to a
  bounded perturbation is the source of the order-drop to 2 for
  non-symmetric inners. For self-adjoint inners, the wrapper costs
  literally one trait-method dispatch.
- **No new dependency**. `core::cmp::min`, `PhantomData` are stdlib.

## Consequences

- New module `src/adjoint.rs` (~320 LoC); under file cap.
- Public surface +1 type +3 constructors. Additive minor.
- `lib.rs` re-export adds `AdjointChernoff`.
- Supersedes the prior v2.0 "Vector 4a" deferred item.

## Acceptance gates

- **G15 self-adjoint identity gate** (NORMATIVE). For `inner =
  GraphHeatChernoff` (symmetric combinatorial Laplacian), verify
  `AdjointChernoff::new_self_adjoint(inner).apply_into(τ, f, &mut dst, ...)`
  is bit-equal to `inner.apply_into(τ, f, &mut dst, ...)` on f64 and
  f32. Threshold: 0 ULP — exact equality.
- **G16 dual-pairing gate** (NORMATIVE). For `inner =
  DriftReactionChernoff` (non-symmetric `−Δ + b·∂_x` with `b = 0.5`),
  verify `⟨S(τ)·f, g⟩ ≈ ⟨f, S*(τ)·g⟩` to within `1e-12` (f64) or
  `1e-6` (f32) for random `f`, `g`, and `τ ∈ {0.01, 0.05}`. Per
  HilbertState `dot()`. Slope test: error vs τ should fall as O(τ²)
  (because order is min(2, inner.order()) = 2).
- **T13N_adjoint_consistency sympy gate** (NORMATIVE). On a 4-node
  symbolic drift-graph (P_4 with edge-asymmetric weights —
  `w(0→1) ≠ w(1→0)`), verify symbolically that the wrapper's
  `apply_into` computes the matrix-adjoint exponential through τ¹ (since
  this case has order=2 → match through τ² in matrix Frobenius norm).

## Out of scope (v2.2)

- **Adjoint-of-Strang composition** (`AdjointChernoff<StrangSplit<X, Y, F>, F>`).
  Compiles by the trait-bound check, but the order-collapse for nested
  non-symmetric inners requires a dedicated theorem (Engel-Nagel §II.4)
  — deferred to v2.3+.
- **Complex-valued adjoint** (`A* = Ā^T`, complex transpose). Real-only
  (`A* = A^T` for `A` real). Complex follows ADR-0057 if v2.3 introduces
  `SemiflowComplex`.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Caller uses `new_self_adjoint` on a non-symmetric inner — silent wrong result | rustdoc explicit warning; `detect_self_adjointness` helper in same module. |
| R2 | f32 order-drop to 2 + dual-pairing arithmetic underflows for tiny τ | f32 G16 threshold relaxed to `1e-6`. Slope test must still show O(τ²). |
| R3 | Performance: dual-pairing path costs one extra `HilbertState::dot()` per step (O(N)) | Order-of-magnitude same as existing kernels (sparse mat-vec is also O(N · nnz_per_row)). Acceptable. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/adjoint.rs` | ~320 |
| `tests/g15_adjoint_self_adjoint.rs` | ~120 |
| `tests/g16_dual_pairing.rs` | ~150 |
| `.dev-docs/verification/scripts/verify_v2_2_adjoint_consistency.py` | ~170 |
| math.md §15 | ~140 |
| ADR-0055 (this) | ~200 |
| **Total** | **~1100** |

## References

- A. Pazy, *Semigroups of Linear Operators and Applications to PDEs*
  (Springer 1983), §1.10 — dual semigroup.
- K.-J. Engel, R. Nagel, *One-Parameter Semigroups for Linear Evolution
  Equations* (Springer 2000), §I.5, §II.4 — adjoint semigroup.
- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) — Theorem 6.
- ADR-0043 (`HilbertState<F>` trait, Wave 1 v2.0).
