# ADR-0042 — In-place pencil ping-pong for Strang2D / Strang3D (Wave 2)

**Status**: PROPOSED
**Date**: 2026-05-20
**Supersedes**: none
**Cross-refs**: ADR-0022 Amendment 1 (Strang3D serial in-place — extended here to Strang2D + parallel paths), ADR-0018 (parallel scratch / SIMD bit-equality — preserved), ADR-0035 (v1.0.0 API stability — Wave 2 is ADDITIVE), ADR-0036 (1D parallel bit-equal contract), ADR-0039 (small-N fused-axis ABORTED — palindromic structure preserved), ADR-0041 (Wave 1 scratch arena & `apply_into` — Wave 2 consumes it).

## Decision

Wire the Wave 1 `ChernoffFunction::apply_into` + `ScratchPool<F>` machinery through the full Strang composition layer. Four additive constructs:

1. **`Strang2D::apply_into`** override — in-place 3-leg palindromic ping-pong `X(τ/2) → Y(τ) → X(τ/2)` operating on a single working state borrowed from `ScratchPool<F>`. Zero intermediate `GridFn2D<F>` allocations per step.
2. **`Strang3D::apply_into`** override — in-place 5-leg palindromic ping-pong `X(τ/2) → Y(τ/2) → Z(τ) → Y(τ/2) → X(τ/2)`. Extends ADR-0022 Amendment 1's serial in-place path to the trait surface; eliminates the `f.clone()` at the entry of `apply_strang3d_full`.
3. **`AxisLift::apply_into` / `AxisLift3D::apply_into`** overrides — per-pencil ping-pong that swaps two slice-views over `GridFnXD::values` instead of allocating `out = zeroed_like()`. Per-pencil 1D `Vec<F>` scratch is borrowed once from `ScratchPool<F>` at the start of the pass and reused across all pencils in the pass.
4. **Per-thread `thread_local!` `ScratchPool<f64>`** in `strang2d_parallel.rs` and `strang3d_parallel.rs` (gated on `feature="parallel"`). Replaces the per-thread `vec![0.0; nx]` / `vec![0.0; ny]` / `vec![0.0; nz]` row/col/pencil buffers currently allocated inside each `*_chunk` helper. Capacity is grow-only with a high-water mark reset only on explicit drain (see Acceptance criterion 7).

Strang2D's `f1/f2/f3` allocations (currently 3 × O(N²) `Vec<F>` per step) and Strang3D's `f.clone()` entry allocation (1 × O(N³) `Vec<F>` per step) collapse to a single 2-buffer ping-pong (2 × O(N²) or O(N³) buffers per `ScratchPool`, allocated once across all time-steps). Result: **0 heap allocations per step in steady state** for `ChernoffSemigroup<Strang2D>::evolve` and `ChernoffSemigroup<Strang3D>::evolve`.

## Rationale

ADR-0041 (Wave 1) eliminated leaf-kernel allocations for `Diffusion*Chernoff`, `Magnus*Chernoff`, `TruncatedExp*Chernoff`, and `NonSeparable2DChernoff`. But every `ChernoffSemigroup<Strang2D>::evolve` step still pays 3 × O(N²) allocs (Strang2D serial) or 1 × O(N²) + 3 × O(N)·O(N²) (parallel) per call, because Strang composition itself was untouched. The v2.0 memory KPI (−60…−80 % across F2/F4/F5/F7/F8) cannot be reached without W2 — Strang dominates the residual.

The in-place ping-pong pattern is forced by two constraints simultaneously: (a) ADR-0035 v1.0.0 API freeze precludes mutating the `apply` signature, so `apply_into` is the only legal channel; (b) ADR-0018's SIMD bit-equality release-blocker forbids any change to the per-node arithmetic, which the slice-level ping-pong respects by construction (only output storage moves; the inner `apply_into` calls remain bit-identical to ADR-0041's leaf overrides).

The palindromic 3-leg (2D) / 5-leg (3D) structure CANNOT be flattened — ADR-0039 already aborted the fused-axis variant because `‖fused − palindromic‖` measured O(τ¹) not O(τ²) due to FP noise structure. Wave 2 PRESERVES palindromic execution unchanged. Only the *storage strategy between legs* changes (2-buffer ping-pong instead of fresh-clone-per-leg).

Alternatives rejected:
- **Refactor `apply_into` to take `&mut Self::S` directly without intermediate ping-pong** (impossible — palindromic structure requires reading the previous state while writing the next).
- **Single in-place buffer with no ping-pong** (impossible — per-pencil `apply_into` of `DiffusionChernoff` reads the source slice while writing the destination; aliasing would violate `&` / `&mut` exclusivity).
- **Move `Strang*::apply` body unchanged, add `apply_into` that wraps `apply` and `mem::swap` the result** (preserves correctness but does NOT eliminate allocations — `apply` still calls `f.zeroed_like()` internally).
- **Rayon / `crossbeam_channel` for the parallel path** (rejected — adds deps, violates `≤3 direct` budget, redundant with existing `std::thread::scope`).

## Math fidelity guarantee

For every separable / non-separable composition `S ∈ {Strang2D, Strang3D, AxisLift, AxisLift3D}` and every valid `(tau, src)` accepted by `S::apply`, the f64 value at every index of `dst.values` after `S::apply_into(tau, &src, &mut dst, &mut scratch)` is bit-identical to `S::apply(tau, &src)?.values[i]`.

Holds by construction:
- Palindromic leg order (X(τ/2), Y(τ), X(τ/2) in 2D; X(τ/2), Y(τ/2), Z(τ), Y(τ/2), X(τ/2) in 3D) is preserved verbatim.
- Each per-leg `AxisLift::apply_into` calls the same leaf `inner.apply_into` (ADR-0041) with identical `tau` and identical `src` slice contents.
- Buffer-parity rule (§2/§3 of contract) ensures legs see exactly the same input bits they would have seen under the allocate-every-leg path.
- The parallel-path per-thread pool replaces `vec![0.0; n]` (which constructs zero-initialised storage that is overwritten by `copy_from_slice`) with `pool.borrow_vec(n)` (which `resize`s an existing buffer to `n` and zero-fills then is also overwritten by `copy_from_slice`). Final contents are identical.
- ADR-0018 SIMD bit-equality (`STRANG2D_PARALLEL_BIT_EQUAL`, `strang3d_parallel_bit_equal`) regressions are re-run as a release gate.
- New proptest `tests/strang_inplace_byte_equal.rs` enforces this property over random `(τ ∈ (0, τ_max), grids, seeds)`.

## Acceptance criteria (release-blocking)

1. All 18 NORMATIVE sympy gates PASS.
2. All 6 slope gates (G3, G3⁴, G3⁶, G3⁶-2D, G4_NS2D_aniso, G5_3D) PASS on prod-HW.
3. ADR-0018 SIMD bit-equality regressions PASS (`STRANG2D_PARALLEL_BIT_EQUAL`, `strang3d_parallel_bit_equal`, `simd_bit_equal`).
4. ADR-0041 `tests/apply_into_byte_equal.rs` (Wave 1) PASS unchanged.
5. New `tests/strang_inplace_byte_equal.rs` PASS (proptest, ≥256 cases per scenario, 2D + 3D + AxisLift + AxisLift3D).
6. New `tests/strang_inplace_alloc_count.rs` PASS — **0 allocs/step after 3 warmup steps** for `ChernoffSemigroup<Strang2D>::evolve` and `ChernoffSemigroup<Strang3D>::evolve` (allocation counter from Wave 1; `dhat` fallback).
7. New `tests/parallel_scratch_drain.rs` PASS — thread-local pool high-water mark settles within 4 calls; explicit `drain_thread_local_pools()` test hook clears all pools and pool re-fills on next call.
8. Miri PASS on `tests/strang_inplace_byte_equal.rs` (catches use-after-borrow on `ScratchPool`).
9. `cargo xtask test-fast`, `test-full`, `test-flagship` all green.
10. Direct-dep budget unchanged at ≤3 (`num-traits + libm`; `num-complex` reserved).
11. File-cap respected per Wave 2 constitution check: `strang3d.rs` ≤ 700 (Override #1) achieved by splitting `AxisLift3D` to `strang3d_axislift.rs`; `strang2d.rs` ≤ 500 achieved by moving fused-order tests to integration suite.
12. v1.0.0 API surface intact: `cargo semver-checks` reports only ADDITIVE changes (one new `apply_into` override per Strang type; no `pub` signature mutation).
13. Existing `Strang2D::apply` and `Strang3D::apply` call sites (15+ tests + 2 benches verified, see contract §10) compile and pass unchanged.

## Out of scope (deferred)

- `State<F>` trait split (W3): generic `State<F>` with `swap`/`as_mut_slice` methods on `GridFn{1,2,3}D`. W2 uses ad-hoc helpers on each concrete type.
- Generic `f32` on parallel paths: parallel-Strang remains f64-only (ADR-0018 carve-out). W2 thread-local pool is `ScratchPool<f64>`.
- AdaptivePI scratch integration (W4): AdaptivePI's `u_full`, `u_half_a`, `u_half` triple-allocation in `adaptive.rs:293-295` remains untouched in W2.
- FFI/PyO3/WASM zero-copy state buffers (W5): `cdylib` boundary still copies `GridFn*D::values` across the FFI seam.
- Y-axis pencil SIMD packing (gather-into-contiguous-then-SIMD): the strided gather currently uses scalar loads. SIMD-on-strided-loads deferred to a hypothetical W6.

**Designed-By**: ai-solutions-architect
**Reviewed-By**: pending
