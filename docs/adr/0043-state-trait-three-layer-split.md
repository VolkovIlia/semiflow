# ADR-0043 — `State<F>` Trait 3-Layer Split (Wave 3 of v2.0)

**Status**: ACCEPTED
**Date**: 2026-05-20
**Wave**: 3 of 5 (v2.0 MAJOR)
**Supersedes (partially)**: ADR-0038 (`State` trait experimental — Wave 3 graduates `State` from experimental to STABLE in v2.0)
**Depends on**: ADR-0041 (Wave 1 scratch arena), ADR-0042 (Wave 2 in-place Strang)
**Foreshadows**: ADR-0044 (Wave 4 generic `AdaptivePI<C, F>`)
**Approved by**: ai-solutions-architect (Wave 3 design pass)

## Decision

Split the existing `State<F>` trait (4 methods, `Clone` supertrait,
`crates/semiflow-core/src/state.rs`) into a **3-layer hierarchy**:

1. **`State<F>`** — zero-allocation Banach-vector primitives sufficient for forward Chernoff
   iteration: `len`, `axpy_into`, `copy_from`, `zero_into`, `norm_sup`, `scale_into`
   (default `unimplemented!`, must override). `Clone` supertrait **removed**.

2. **`HilbertState<F>: State<F>`** — inner-product extension: `dot`, `norm_sq` (default),
   `norm_l2` (default). For L²-flavoured controllers and adjoint-Chernoff (Wave 4+).

3. **`Discrete<F>: State<F>`** — graph/manifold/lattice extension: GAT `Neighbours<'a>`,
   `get`, `set`, `indices`, `neighbours`. Eliminates `Box<dyn Iterator>` per v0.14.0 spike.
   **Tensor types do NOT implement `Discrete<F>`** (no canonical neighbour set).

The `ChernoffFunction::apply_into` default bridge changes from `*dst = self.apply(tau,
src)?` to `dst.copy_from(&tmp)`. The `apply` and default-bridge `apply_into` carry
`where Self::S: Clone` bounds; override impls do not need `Clone`.

`ChernoffSemigroup::evolve` keeps `S: State<f64> + Clone` (needs `f.clone()` for `buf_a`
initialisation). `buf_b` initialisation changes from `f.zeroed_like()` to
`f.clone(); buf_b.zero_into()` — explicit and auditable.

## Rationale

1. Wave 1 plumbing exposed missing zero-alloc primitives: `copy_from` and `zero_into`
   close the gap that previously forced callers through `zeroed_like()` + Clone-based
   assignment.
2. `Clone` supertrait made allocation invisible; removing it forces call-sites to be
   explicit (scratch-arena or `+ Clone` bound). Concrete `GridFnXD<F>` keep `Clone`
   via `#[derive(Clone)]`; no user-facing change for the 99% case.
3. `HilbertState<F>` separates dot-product cost so graph-PDE implementors (who may
   lack a canonical inner product) don't have to provide `dot`.
4. `Discrete<F>` enables graph PDEs without polluting tensor types. GATs (stable
   Rust 1.65, MSRV is 1.78) eliminate the `Box<dyn Iterator>` allocation flagged
   by the v0.14.0 spike (`crates/remizov-graph-spike/src/lib.rs:24-26`).
5. GAT `type Neighbours<'a>` removes per-call heap allocation in hot graph loops.
6. `apply_into(&mut S, …)` with the new primitives is the only API-shape that
   supports Wave 5 fused-loop optimisations without another MAJOR break.

## Math fidelity guarantee (NORMATIVE)

Wave 3 changes are **plumbing only**. Per-node arithmetic in every math-bearing
kernel is **byte-identical** to v0.13.0 on the f64 path. Verification:

- All 18 NORMATIVE sympy gates (T9N_*, T10N_*, T11N_*) re-pass.
- All 6 slope gates (G3, G3⁴, G3⁶, G3⁶-2D, G4_NS2D_aniso, G5_3D) re-pass on prod-HW.
- SIMD bit-equality regressions re-pass: `diffusion6_simd`, `chernoff1d_parallel_bit_equal`,
  `nonseparable2d_parallel_bit_equal`, `strang_inplace_byte_equal`, `apply_into_byte_equal`.
- New `state_trait_contract.rs` proptest verifies algebraic laws (10 invariants).

## Breaking change scope

**v2.0 MAJOR BREAKING** — user confirmed direct cut.

### What breaks
- Custom `impl State<F> for MyType`: must implement new method set (`len`, `axpy_into`,
  `copy_from`, `zero_into`, `norm_sup`; override `scale_into`).
- Generic bounds `T: State<F>` calling `t.clone()`: add explicit `+ Clone` bound.
- Generic bounds calling `s.axpy(...)` / `s.scale(...)` / `s.zeroed_like()`: change to
  `s.axpy_into(...)` / `s.scale_into(...)` / manual clone+zero pattern.

### What does NOT break
- Concrete `GridFnXD<F>` users: inherent shim methods `axpy`, `scale`, `zeroed_like`
  are retained on concrete types — no source change needed.
- `ChernoffSemigroup` / `AdaptivePI` / all `DiffusionChernoff` etc. users: public API
  unchanged.

## Acceptance criteria (release-blocking)

- AC-1: 18 NORMATIVE sympy gates green.
- AC-2: 6 slope gates re-pass on prod-HW.
- AC-3: SIMD bit-equality regressions re-pass (AVX2 + NEON).
- AC-4: Wave 1 carry-forward: `apply_into_byte_equal` 6/6, `zero_alloc_steady` 2/2,
  `parallel_scratch_drain` 7/7.
- AC-5: Wave 2 carry-forward: `strang_inplace_byte_equal` 7/7, `strang_inplace_alloc_count` 2/2.
- AC-6: `state_trait_contract.rs` proptest passes (10 invariants).
- AC-7: `graph_heat_oracle.rs` passes for `PathGraphFn` against eigenvalue oracle.
- AC-8: `cargo semver-checks` reports expected delta only (whitelist `.cargo/semver-checks-allowlist.toml`).
- AC-9: Suckless: functions ≤50 lines, files ≤500 LoC, dep cap ≤3.
- AC-10: `cargo check -p semiflow-core --no-default-features` green.
- AC-11: Default-bridge compat: downstream impl without `apply_into` override compiles+runs.
- AC-12: `docs/migration/v1-to-v2.md` exists with all 4 method removals/renames.

## Alternatives rejected

1. **Flat `State<F>` + all new methods**: forces graph implementors to implement `dot`;
   forces tensor implementors to implement `neighbours`. Violates minimal surface principle.
2. **Two-layer (merge `HilbertState` + `Discrete`)**: `HilbertState` and `Discrete` are
   orthogonal; some types need one but not the other.
3. **Four-layer (split `Discrete` further)**: YAGNI; spike validated 3-layer shape.
4. **Promote spike whole-hog into core**: adds `std::collections::BTreeMap` dep; spike
   recommends `VecState` as primary type — design takes >1 wave.
5. **`&mut dyn State<F>` everywhere**: defeats inlining; SIMD regressions would break.

## Out of scope (deferred)

- Wave 4: `AdaptivePI<C, F>` generic-over-F + `HilbertState`-powered Richardson.
- v2.x: `WeightedGraphFn<F>` / `VecState<F>` concrete graph types.
- Explicit boundary-condition enum on `Discrete<F>`.
- `unsafe`-based SIMD on `State::axpy_into`.
- Version bump to `2.0.0-rc.1` (deferred to Wave 5 release prep).
