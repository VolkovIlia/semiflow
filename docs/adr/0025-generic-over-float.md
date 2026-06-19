# ADR-0025 — v0.9.0 Generic-over-Float refactor (lifts ADR-0004)

**Status**: Proposed
**Date**: 2026-05-07
**Authors**: ai-solutions-architect
**Lifts**: ADR-0004 (v0.1.0 f64 monomorphic decision)
**Cross-refs**: ROADMAP.md v0.9.0 third bullet ("Generic-over-Float — type-parameterise
core types over f32/f64 …"), ADR-0011 (`SemiflowError`), ADR-0012 (`Strang2D<X, Y>`),
ADR-0023 (`NonSeparable2DAnisotropicChernoff`), ADR-0024 (`Strang3D<X, Y, Z>`)

## Context

ADR-0004 (v0.1.0) deliberately fixed all core types to `f64`, citing three costs of
premature generalisation (trait-bound noise, forced `num-traits` dependency, rustdoc
legibility) and stated: *"The clean refactor target is v0.5 when the adaptive
controller (`remizov-adaptive`) lands with a real motivation for non-f64 scalars
(interval arithmetic for guaranteed bounds)."*

v0.5.0 shipped `Strang2D` and the tensor catalogue without lifting ADR-0004. v0.6.0
delivered `AdaptivePI` — still on `f64`. The ADR-0004 trigger (interval arithmetic)
never fired because the adaptive controller's practical need was step-size control, not
scalar-type flexibility. The `num-traits` crate is now an **explicit** `[dependencies]`
entry in `semiflow-core/Cargo.toml` (v0.2+), so the "forced transitive dep" cost from
v0.1.0 is already sunk.

The v0.9.0 trigger is twofold. First, `f32` SIMD throughput: AVX2 processes 8
single-precision lanes vs 4 double-precision (2× bandwidth); on AVX-512, the ratio is
4×. Downstream callers running large 3D grids at single-precision tolerance (e.g.
real-time simulation, game physics) have no opt-in path today. Second, downstream
library consumers have requested `num_complex::Complex<f64>` coefficients for
spectral methods; the monomorphic `f64` surface blocks this entirely. An `Interval<F>`
type for guaranteed bounds — the original ADR-0004 motivator — becomes feasible once
the generic foundation lands (separate crate, separate ADR; this ADR is its prerequisite).

## Decision

Adopt a project-local sealed trait `SemiflowFloat: num_traits::Float + AddAssign +
MulAssign + SubAssign + DivAssign + Send + Sync + Copy + Debug + Display + PartialOrd
+ 'static` (exact bound list refined by the wave 2 engineer to satisfy all current
method signatures; `RemAssign` added if any implementation requires it). Provide blanket
implementations for `f32` and `f64` only; the trait is sealed so downstream crates
cannot implement it for unknown types. Type-parameterise all data-bearing core types
and all `ChernoffFunction` implementations over `<F: SemiflowFloat = f64>` — the `= f64`
default ensures every existing callsite compiles unchanged via type inference (zero
breaking change). Coefficient literals (e.g. `0.5`, `2.0`, ζ-correction coefficients)
are rewritten as `F::from(literal).unwrap()` for exact fractions; any non-exact
conversion is documented with a comment. Coefficient function pointers of the form
`fn(f64, f64) -> f64` in `NonSeparable2DChernoff` and `NonSeparable2DAnisotropicChernoff`
become `fn(F, F) -> F`. SIMD intrinsics (AVX2/NEON, `#[cfg(feature = "simd")]`) remain
`f64`-specialised — they require concrete float types; the generic path dispatches
through the scalar fallback when `F ≠ f64`. Concrete `f32x8`/`f32x4` SIMD support is
deferred to a future ADR. The `num-traits` direct `[dependencies]` entry is already
present (v0.2+); no new crate is added.

## Considered alternatives

- **(a) Partial generic — data types only, `ChernoffFunction` stays `f64`**: rejected.
  A `Grid1D<f32>` paired with an `f64`-only operator forces explicit scalar conversion
  at every apply call, which defeats the f32 SIMD bandwidth gain and leaves a
  confusing half-generic surface. Full generic is cleaner and the complexity cost is
  the same once the trait alias is defined.
- **(b) Const-generic precision `<const PREC: usize>`**: rejected. Float types are
  nominal in Rust's type system; `<const PREC: usize = 64>` has no standard meaning
  and no `f32`/`f64` unification path. Not a Rust idiom.
- **(c) Trait alias instead of a supertrait `SemiflowFloat`**: rejected. Rust stable
  does not have trait aliases. The idiomatic workaround is `pub trait SemiflowFloat:
  Float + … {}` plus a blanket `impl<T: Float + …> SemiflowFloat for T {}`. This is
  the approach adopted; the sealed-trait variant narrows the blanket to only `f32`
  and `f64` to prevent misuse.
- **(d) Skip f32, generalise only for `Interval`/complex**: rejected. The f32 SIMD
  bandwidth doubling is a concrete, measurable gain for engineering callers on large
  3D grids (Block D scope). Deferring f32 to a later ADR after building the generic
  skeleton would require a second parameter-sweeping refactor pass for no architectural
  benefit.
- **(e) Monomorphisation explosion (compile time and binary size)**: documented risk,
  mitigated by `default = f64`. The common instantiation is one concrete type; `f32`
  instantiation is opt-in. Estimated workspace build overhead: +5–15% (single extra
  instantiation per type when f32 is used).

## Consequences

- **+1 trait** (`SemiflowFloat`) **+2 blanket impls** (`f32`, `f64`). Zero new
  crate dependencies; `num-traits = "0.2"` is already in `[dependencies]`.
- **All affected public types** (see ROADMAP.md type list: `Grid1D`, `GridFn1D`,
  `BoundaryPolicy`, `ShiftChernoff1D`, `DiffusionChernoff`, `Diffusion4thChernoff`,
  `Diffusion6thChernoff`, `DriftReactionChernoff`, `LiouvilleChernoff`,
  `TruncatedExpDiffusionChernoff`, `TruncatedExpDiffusion4thChernoff`, `AdaptivePI`,
  `AdvectionDiffusionChernoff`, `Grid2D`, `GridFn2D`, `Axis`, `AxisLift<C>`,
  `Strang2D<X, Y>`, `NonSeparable2DChernoff<X, Y>`, `NonSeparable2DAnisotropicChernoff<X, Y>`,
  `Grid3D`, `GridFn3D`, `Strang3D<X, Y, Z>`) gain an `<F = f64>` parameter.
  Existing callsites compile unchanged via the default.
- **API break risk**: type inference may fail at callsites with ambiguous float
  literals (e.g. `Grid1D::new(0.0, 1.0, 100)`). Wave 2 engineer must run the full
  test suite and fix any callsite that does not infer `F = f64` automatically.
  Affected callsites will require an explicit turbofish or a `let` annotation —
  a one-time migration, not a semantic change.
- **SIMD path** (`simd` feature): `f64` intrinsics are unchanged bit-for-bit.
  `f32` users fall through to the scalar path; no regression for current benchmarks.
  `heat_2d` 4.38× speedup (v0.8.1 baseline) is unaffected.
- **Compile time**: minor increase; bounded by the `f64` monomorphisation remaining
  the only common instantiation in practice.
- **Binary size**: unchanged for `f64`-only binaries; approximately 2× the
  type-bearing code if a caller explicitly instantiates both `f32` and `f64` paths
  (the caller's explicit choice).
- **`Send + Sync` bound on `SemiflowFloat`**: excludes `Rc<…>`-based pseudo-scalars.
  Acceptable; `parallel` feature already requires `Send` through `std::thread::scope`.
- **`BoundaryPolicy` and `InterpKind`**: these are pure-enum geometry adapters
  independent of scalar type; they remain `F`-agnostic and require no changes.

## Forward compatibility

- **v1.0+** MAY add `f32` SIMD intrinsics (`f32x8` on AVX2, `f32x4` on NEON) once
  this refactor lands — separate ADR; SIMD code is already feature-gated and isolated.
- **v1.0+** MAY relax `Send + Sync` from `SemiflowFloat` for single-threaded users —
  separate ADR; requires auditing the parallel feature interaction.
- **v1.0+** MAY add first-class `Complex<f64>` support for spectral methods by
  implementing `SemiflowFloat` for `num_complex::Complex<f64>` — separate ADR;
  requires re-evaluating the `PartialOrd` and `Display` bounds (complex numbers
  have no natural total order).
- **`Interval<F>` type** for guaranteed bounds (the original ADR-0004 motivator)
  becomes feasible once this refactor lands; would live in a separate crate
  (`remizov-interval`) satisfying the `SemiflowFloat` bounds — separate ADR.

## Verification

The Block D wave 2 engineer deliverable produces a passing run of:

```
cargo build --workspace                                                         # f64 default — clean
cargo build --workspace --features simd,parallel,slow-tests                    # f64 + SIMD + parallel
cargo test --workspace                                                          # all tests green, f64 default
cargo run -p xtask -- test-fast                                                 # 169+ tests pass (Block C baseline preserved)
cargo clippy --workspace --all-targets --features slow-tests -- -D warnings    # 0 warnings
```

A new integration test (e.g. `tests/generic_float_smoke.rs`) instantiates
`Grid1D<f32>`, `GridFn1D<f32>`, `DiffusionChernoff<f32>`, `Strang2D<f32>`, and
`Strang3D<f32>` — verifying both that the generic path **compiles** and that it
**produces correct numerical results** within f32 tolerance (e.g. 1D heat-equation
oracle at sup-error ≤ 1 × 10⁻⁵ relative to the f64 reference solution), guarding
against silent precision-only regressions in the scalar fallback path.
