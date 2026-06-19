# ADR-0058 — `NonSeparableAniso<S: Discrete<F>>` unification

- **Status**: ACCEPTED (v2.2 Wave C) — **SUPERSEDES ADR-0033**
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave C (refactor + bindings)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0023 (anisotropic non-separable 2D), ADR-0026 (Generic-over-Float),
  ADR-0033 (NonSeparable2D deprecation policy — **SUPERSEDED by this ADR**),
  ADR-0043 (`State<F>` 3-layer split, `Discrete<F>`).
- **Mathematical foundation**: math.md §18 (NORMATIVE; CITATION: math.md
  §10.7-ter unchanged — generic refactor is API-only).

## Context

v0.7.0 shipped `NonSeparable2DChernoff<X, Y, F>` for the
constant-coupling case `L = Lx + Ly + c·∂x∂y` (math.md §10.7;
ADR-0016).

v0.9.0 shipped `NonSeparable2DAnisotropicChernoff<X, Y, F>` for the
position-dependent case `L = Lx + Ly + β(x,y)·∂x∂y` (math.md §10.7-ter;
ADR-0023).

ADR-0033 (v0.9.0) opted to "keep both" types alive — the constant-c
case is faster (avoids per-node `β(x,y)` evaluation), and v0.9.0 wasn't
ready for a type-system unification. Two years of usage have shown:

1. The constructors `NonSeparable2DChernoff::new(x, y, c, c_bound, grid)`
   and `NonSeparable2DAnisotropicChernoff::new(x, y, β, β_bound, grid)`
   share 95% structural equality.
2. The two `apply_into` implementations differ only in the mixed-Taylor
   coupling step — `c · ∂x∂y · f` vs `β(x,y) · ∂x∂y · f` (and the
   norm-bound book-keeping).
3. The future direction for graph-mixed operators (e.g.,
   `NonSeparableMixed<GraphHeat, GraphHeat, F, GraphSignal<F>>` for
   cross-domain coupled graph PDEs) needs a generic abstraction.

## Decision

Introduce a new generic type `NonSeparableMixedChernoff<X, Y, F, S>`:

```rust
//! crates/semiflow-core/src/nonseparable_mixed.rs (NEW FILE, ~550 LoC)
//! Replaces (via type aliases) `src/nonseparable2d.rs` (514 LoC) and
//! `src/nonseparable2d_aniso.rs` (481 LoC).

pub trait MixedDerivOperator<F: SemiflowFloat, S: State<F>>: Send + Sync {
    /// Sup-norm bound on the coupling coefficient.
    fn norm_bound(&self) -> F;
    /// Apply `∂x∂y · f` weighted by the coupling, in-place: `dst ← coupling · ∂x∂y · src`.
    fn apply_mixed_into(&self, src: &S, dst: &mut S, grid: &MixedGrid<F, S>);
    /// `true` iff the coupling is identically zero (fast-path).
    fn is_zero(&self) -> bool;
}

#[derive(Clone, Debug)]
pub struct NonSeparableMixedChernoff<X, Y, F: SemiflowFloat = f64, S = GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    pub x: AxisLift<X, F>,
    pub y: AxisLift<Y, F>,
    coupling: Box<dyn MixedDerivOperator<F, S>>,
    pub grid: MixedGrid<F, S>,
}

// v2.2: collapses to GridFn2D<F> only (graph case is v2.3+).
impl<X, Y, F: SemiflowFloat> NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone,
{
    /// Scalar coupling constructor — mirrors v0.7.0 `NonSeparable2DChernoff::new`.
    pub fn with_scalar_c(
        x_inner: X,
        y_inner: Y,
        c: fn(F, F) -> F,
        c_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError>;

    /// Position-dependent coupling constructor — mirrors v0.9.0 anisotropic.
    pub fn with_beta(
        x_inner: X,
        y_inner: Y,
        beta: fn(F, F) -> F,
        beta_norm_bound: f64,
        grid: Grid2D<F>,
    ) -> Result<Self, SemiflowError>;
}

// Backward-compat type aliases — zero-source-change migration.
pub type NonSeparable2DChernoff<X, Y, F = f64> = NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;
pub type NonSeparable2DAnisotropicChernoff<X, Y, F = f64> = NonSeparableMixedChernoff<X, Y, F, GridFn2D<F>>;
```

The existing constructors `NonSeparable2DChernoff::new(x, y, c, c_bound, grid)`
and `NonSeparable2DAnisotropicChernoff::new(x, y, β, β_bound, grid)` are
RE-EXPORTED FROM THE GENERIC IMPL (alias-bound) — callers don't change
a single line.

```rust
// v2.1 user code (unchanged):
let op = NonSeparable2DChernoff::new(x_inner, y_inner, |x, y| 0.3, 0.3, grid)?;
// works in v2.2; calls `NonSeparableMixedChernoff::with_scalar_c` internally.

let op_aniso = NonSeparable2DAnisotropicChernoff::new(x_inner, y_inner, |x, y| 0.3 * x, 1.5, grid)?;
// works in v2.2; calls `NonSeparableMixedChernoff::with_beta` internally.
```

Internal `apply_into` dispatches on `coupling.is_zero()` for the
fast-path (collapses to `Strang2D`), then on `coupling.apply_mixed_into(...)`
for the slow-path. Two concrete `MixedDerivOperator` impls live in
`nonseparable_mixed.rs`:

- `ScalarCoupling<F>` — `c: F` constant.
- `BetaCoupling<F>` — `beta: fn(F, F) -> F` + per-node cache.

## Rationale

- **Single source of truth.** The Strang-style palindromic 5-leg
  composition (math.md §10.7-ter Theorem 7-bis) is now expressed once,
  not twice. Future bug fixes apply uniformly.
- **Type-alias preserves backward-compat.** No source migration burden
  on downstream users.
- **Sets stage for graph-mixed operators**. v2.3+ can add
  `type NonSeparableMixedGraphChernoff<X, Y, F = f64> =
   NonSeparableMixedChernoff<X, Y, F, GraphSignal<F>>;`
  with `GraphSignal<F>` already `impl Discrete<F>` (v2.0 Wave 1).
- **Supersedes ADR-0033 decisively.** ADR-0033 punted; now we resolve.

## Consequences

- New file `src/nonseparable_mixed.rs` (~550 LoC; Override #1 carve-out
  EXPANSION — math co-location with §10.7-ter; new module replaces
  514 + 481 = 995 total LoC across two files).
- Existing `src/nonseparable2d.rs` and `src/nonseparable2d_aniso.rs`
  become 30-LoC thin re-export shims (or are removed entirely in favour
  of `pub use crate::nonseparable_mixed::*` in `lib.rs`).
- **Constitution Override #1 check**: total LoC under Override #1 file-list
  may go DOWN (995 LoC → 550 LoC) or up by 1 file (new file in carve-out
  list, two old files removed). Net Override #1 capacity does not change.

## Acceptance gates

- **G_NS2D and G_NS2D_aniso slope gates** (existing v0.7.0 + v0.9.0).
  Re-pass byte-identical via the type-aliased call paths. No new
  slope gate added in v2.2; the refactor MUST NOT change numerics.
- **G20 alias-identity gate** (NORMATIVE — new). Verify that
  `<NonSeparable2DChernoff as ChernoffFunction>::apply_into(τ, f)`
  byte-equals `<NonSeparableMixedChernoff::with_scalar_c(...) as ChernoffFunction>::apply_into(τ, f)`
  for arbitrary `f` and `τ ∈ {0.001, 0.01, 0.1}`. Threshold: 0 ULP.
- **No new sympy gate.** Mathematical content is unchanged (math.md §10.7-ter
  is referenced from §17, not rewritten).

## Out of scope (v2.2)

- **Graph extensions** (`NonSeparableMixedGraphChernoff` for cross-domain
  graph PDEs). v2.3+.
- **3D non-separable** (`NonSeparable3DMixedChernoff`). v2.3+.
- **Removal of the old type aliases.** They stay as aliases (zero
  maintenance cost). Removal would be MAJOR.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Trait `MixedDerivOperator<F, S>` may not capture future graph-mixed semantics cleanly | The trait is private to `nonseparable_mixed.rs` for v2.2 (only two impls); v2.3+ can promote to pub trait once usage stabilises. |
| R2 | Type alias dispatch confuses cargo doc / IDE | Verified via cargo doc CI gate; rustdoc renders both alias names. |
| R3 | The `apply_into` fast-path for `c == 0` was inline in v0.7.0; via trait dispatch it costs one v-table hop | v-table hop on `Box<dyn>` is ~1-2 ns; negligible vs sparse-mat-vec or apply step (~µs). |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/nonseparable_mixed.rs` | ~550 (Override #1 carve-out) |
| Remove (-) `src/nonseparable2d.rs` (514) | -514 |
| Remove (-) `src/nonseparable2d_aniso.rs` (481) | -481 |
| `lib.rs` re-export shims | +30 |
| `tests/g20_alias_identity.rs` | ~120 |
| math.md §18 (refactor pointer) | ~50 |
| ADR-0058 (this) | ~210 |
| **Net change** | **−435 LoC overall** |

## References

- ADR-0023 (anisotropic non-separable 2D).
- ADR-0033 (NonSeparable2D deprecation policy — SUPERSEDED here).
- ADR-0026 (Generic-over-Float; trait `MixedDerivOperator` follows the
  same pattern as `MagnusGraphHeat6thChernoff`'s coupling).
- math.md §10.7-ter (Theorem 7-bis; unchanged).
