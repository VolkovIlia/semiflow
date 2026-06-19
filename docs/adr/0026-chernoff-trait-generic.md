# ADR-0026 — v0.9.0 ChernoffFunction trait generalised over SemiflowFloat

**Status**: Proposed
**Date**: 2026-05-08
**Authors**: ai-solutions-architect
**Lifts**: ADR-0025 deferral ("generic ChernoffFunction is future scope")
**Cross-refs**: ADR-0025 (Block D pilot — data-bearing types generic over F),
ADR-0018 (parallel SIMD bit-equality contract — parallel impls stay f64),
ADR-0011 (`SemiflowError` reused, no new variants), ADR-0012 (`Strang2D<X, Y>`),
ADR-0024 (`Strang3D<X, Y, Z>`)

## Context

ADR-0025 (Block D pilot, commit 166b7d8) lifted the ADR-0004 f64 monomorphism
for every data-bearing type (`Grid1D<F>`, `GridFn1D<F>`, `DiffusionChernoff<F>`,
`State<F>`, and the full 2D/3D grid catalogue) but kept the `ChernoffFunction`
trait itself concrete. The ADR-0025 decision text stated explicitly: *"ChernoffFunction
trait implementation ONLY for concrete `<f64>` (generic ChernoffFunction is future
scope)."* Block D Waves 1–3 propagated `F` through all leaf 1D Chernoff type
fields and through `Grid2D/3D<F>` and `GridFn2D/3D<F>` — but composition types
(`AxisLift`, `Strang2D`, `Strang3D`, `NonSeparable2D*`, `AdaptivePI`,
`StrangSplit`) remained trait-bound to `f64` because their inner types' `ChernoffFunction`
impls were still `f64`-concrete. The consequence: an `f32`-instantiated `Grid2D<f32>`
could be constructed but no generic operator could act on it through the `Strang2D`
or `AxisLift` path. This ADR closes that deferral.

## Decision

Parameterise the `ChernoffFunction` trait over `F: SemiflowFloat = f64`, replacing
every `f64`-concrete method signature with the corresponding `F`-generic form:
`fn apply(&self, tau: F, f: &Self::S) -> Result<Self::S, SemiflowError>` and
`fn order(&self) -> usize` (unchanged; order is a structural property, not a
scalar). The `= f64` default ensures every existing callsite compiles unchanged
via Rust type inference — zero breaking change. Each of the 7 leaf 1D Chernoff
types (`DiffusionChernoff`, `Diffusion4thChernoff`, `Diffusion6thChernoff`,
`DriftReactionChernoff`, `LiouvilleChernoff`, `TruncatedExpDiffusionChernoff`,
`TruncatedExpDiffusion4thChernoff`, and `AdvectionDiffusionChernoff`) gains
`impl<F: SemiflowFloat> ChernoffFunction<F> for Type<F>`, replacing the prior
concrete `impl ChernoffFunction for Type<f64>`. Composition types — `AxisLift<C, F>`,
`Strang2D<X, Y, F>`, `Strang3D<X, Y, Z, F>`, `StrangSplit<D, R, F>`,
`NonSeparable2DChernoff<X, Y, F>`, `NonSeparable2DAnisotropicChernoff<X, Y, F>`,
`AdaptivePI<C, F>` — gain an explicit `<F>` parameter on their `ChernoffFunction`
impl and propagate it through inner-type bounds (e.g. `where C: ChernoffFunction<F>`);
the `= f64` default on each struct's own parameter ensures `let s = Strang2D::new(x, y, grid)?`
continues to infer `F = f64` without a turbofish. The parallel-feature impls of
`Strang2D`, `NonSeparable2DChernoff`, and `NonSeparable2DAnisotropicChernoff` remain
`f64`-only — concretely `#[cfg(feature = "parallel")] impl ChernoffFunction<f64>
for Type<f64>` — per ADR-0018's parallel-SIMD bit-equality contract: the
`STRANG2D_PARALLEL_BIT_EQUAL` gate requires f64 SIMD intrinsics and is
release-blocking; generalising to `F ≠ f64` would either drop SIMD (regressing
v0.8.1's heat_2d 4.38× speedup) or require f32 SIMD intrinsics (separate ADR
scope). The generic serial path is the `F ≠ f64` fallback and is reached whenever
the `parallel` feature is off or the caller instantiates `f32`. SIMD intrinsics on
leaf 1D types (`#[cfg(feature = "simd")]`) stay on their concrete `f64` impls — no
`TypeId` branching, no transmute. Numeric literal conversion in generic `apply`
bodies uses `two()`, `half()`, and `from_f64()` helpers from `float.rs`; non-exact
conversions carry a comment.

## Considered alternatives

- **(a) Keep `ChernoffFunction` f64-monomorphic** (the deferred state from
  ADR-0025): rejected. Composition types remain f64-bound, blocking any f32
  path through `Strang2D`, `NonSeparable2D*`, or `AdaptivePI`. The Generic-over-Float
  effort is structurally incomplete without the composition layer; the whole value
  proposition of ADR-0025 — f32 SIMD bandwidth for large 3D grids — cannot be
  realised unless the operator can act generically on the generic grid.
- **(b) Generic trait plus generic parallel impl**: rejected. ADR-0018's
  `STRANG2D_PARALLEL_BIT_EQUAL` release-blocking contract requires concrete f64 SIMD
  intrinsics. Generalising the parallel path to arbitrary `F` either drops the
  SIMD kernel (regresses v0.8.x perf baseline) or requires a separate f32-SIMD ADR;
  either path is out of this block's scope.
- **(c) Two traits — `ChernoffFunctionF64` (concrete, SIMD) + `ChernoffFunction<F>`
  (generic, scalar)**: rejected. Doubles the public surface area; downstream implementors
  must choose which trait to impl; blanket delegation between the two traits leaks
  combinatorial bound complexity. Single-trait with a concrete `impl ChernoffFunction<f64>
  for Type<f64>` parallel carve-out is uniform without duplicating the trait itself.
- **(d) Trait alias `pub trait ChernoffFunctionF<F> = ChernoffFunction<F = F>`**:
  rejected; Rust trait aliases are unstable on MSRV 1.78. The `= f64` default
  on the trait parameter is the stable equivalent.

## Consequences

- All 14 `ChernoffFunction`-implementing types are now generic over `F` (7 leaf
  1D + 7 composition); existing `f64` callsites compile unchanged via the default.
- 6 new f32 composition-type smokes added to `tests/generic_float_smoke.rs` (26
  total, +7 over Wave 3 baseline of 191 workspace tests → 198/0/1 fast-tests).
- `apply` method signature changes `tau: f64 → tau: F`; callers passing f64
  literals continue to work via type inference with no turbofish required.
- Parallel impls (`Strang2D` parallel, `NonSeparable2DChernoff` parallel,
  `NonSeparable2DAnisotropicChernoff` parallel) stay f64-only via the
  `#[cfg(feature = "parallel")] impl ChernoffFunction<f64>` carve-out.
- v0.8.x SIMD bit-equality regressions preserved (`diffusion4_unit` 9/9,
  `simd_bit_equal` 2/2); heat_2d 4.38× parallel speedup unaffected.
- `BoundaryPolicy` and `InterpKind` are pure-enum geometry adapters independent
  of scalar type; no changes required.

## Forward compatibility

- v1.0+ MAY add f32 SIMD intrinsics (`f32x8` on AVX2, `f32x4` on NEON) for the
  `Strang2D` parallel path once a dedicated f32-SIMD ADR lands; the generic serial
  fallback already validates the f32 code path end-to-end.
- v1.0+ MAY add first-class `Complex<f64>` spectral methods by implementing
  `SemiflowFloat` for `num_complex::Complex<f64>` — requires re-evaluating the
  `PartialOrd` and `Display` bounds (separate ADR).
- An `Interval<F>` type for guaranteed-bounds Chernoff (the original ADR-0004
  motivator) becomes feasible once a `SemiflowFloat for Interval<f64>` impl lands
  in a separate `remizov-interval` crate — separate ADR; this ADR is its prerequisite.

## Verification

```
cargo build --workspace                                                          # f64 default, clean
cargo build --workspace --features simd,parallel                                # f64 + SIMD + parallel path
cargo clippy --workspace --all-targets --features slow-tests -- -D warnings    # 0 warnings
cargo run -p xtask -- test-fast                                                 # 198 passed, 0 failed, 1 ignored
cargo test -p semiflow-core --test generic_float_smoke                           # 26/26 (incl. 6 new composition f32 smokes)
cargo test -p semiflow-core --test diffusion4_unit                               # 9/9 SIMD bit-equality preserved
cargo test -p semiflow-core --test simd_bit_equal                                # 2/2 SIMD regression
python3 .dev-docs/verification/scripts/verify_v0_7_0_nonseparable.py           # T7N_* 6/6 PASS
python3 .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py     # T9N_* 6/6 PASS
python3 .dev-docs/verification/scripts/verify_v0_9_0_3d_tensor.py              # T10N_* 6/6 PASS
```
