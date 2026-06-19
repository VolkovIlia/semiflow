# ADR-0041 — Scratch arena & `apply_into` additive trait method (Wave 1)

**Status**: PROPOSED
**Date**: 2026-05-20
**Supersedes**: none
**Cross-refs**: ADR-0004 (monomorphism; superseded by ADR-0025/0026 — F threads through unchanged), ADR-0018 (parallel scratch / SIMD bit-equality), ADR-0022 Amendment 1 (Strang3D scratch-pool serial), ADR-0025/0026 (SemiflowFloat + ChernoffFunction<F>), ADR-0035 (v1.0.0 API stability — Wave 1 is ADDITIVE), ADR-0036 (1D parallel bit-equal contract).

## Decision

Introduce two ADDITIVE constructs that eliminate per-step heap allocations in the Chernoff hot loop without touching kernel math:

1. **`ScratchPool<F: SemiflowFloat>`** — a single-threaded, owned, grow-only bump-allocator of `Vec<F>` buffers, kept in `crates/semiflow-core/src/scratch.rs`. `no_std + alloc`, zero new dependencies, ≤200 LoC. RAII handle `ScratchVec<'a, F>` returns the buffer to the pool on Drop. Single-borrow exclusivity invariant is statically enforced via `&mut ScratchPool<F>` on `borrow_vec` (no `RefCell`, no `unsafe`).
2. **`ChernoffFunction<F>::apply_into(&self, tau: F, src: &Self::S, dst: &mut Self::S, scratch: &mut ScratchPool<F>) -> Result<(), SemiflowError>`** — additive trait method with a default impl that bridges to existing `apply`. Math-bearing kernels override with allocation-free implementations. Default-impl bridge guarantees backward compatibility.

`ChernoffSemigroup::evolve` is rewritten to ping-pong between two pre-allocated `[S; 2]` buffers using `apply_into`. Result: **0 heap allocations per step in steady state** for f64 math-bearing kernels.

## Rationale

The v2.0 release KPIs (memory −60…−80 % on F2/F4/F5/F7/F8, 0 per-step allocations) cannot be met by W2–W5 alone — even with in-place Strang composition (W2) and zero-copy bindings (W5), the inner `ChernoffSemigroup::evolve` still allocates one `Vec<F>` per leaf kernel per step. W1 fixes leaf + loop and is a prerequisite for W2.

The additive trait method (rather than redefining `apply`) is forced by ADR-0035 v1.0.0 API freeze. Keeping `apply` intact + adding `apply_into` with default impl ships under MINOR.

Alternatives rejected: global thread-local arena (no_std + hidden globals), `bumpalo` dep (adds dep, std-only), `SmallVec` (capacities too large), caller-passed `&mut Vec<F>` per kernel (combinatorial trait signature explosion).

## Math fidelity guarantee

For every math-bearing kernel K ∈ {DiffusionChernoff, Diffusion4thChernoff, Diffusion6thChernoff, Magnus4thDiffusionChernoff, MagnusKChernoff, TruncatedExpDiffusionChernoff, TruncatedExp4thDiffusionChernoff, NonSeparable2DChernoff} and every valid `(tau: f64, src)` accepted by `apply`, the f64 value at every index of `dst.values` after `apply_into` is bit-identical to `apply(tau, src)?.values[i]`.

Holds by construction: only output-Vec storage moves; per-node ζ-A / γ-A / Magnus / TruncatedExp arithmetic, `parallel_eval` chunk order, SIMD `catmull_rom` / `quintic_hermite` sample path, Fourier weights (7/12, 3/16, 1/48), τ²-correction (1, 1/2, 1/4), K=4 Magnus expansion, palindromic Strang order, PI gains — all untouched.

Gates: 18 sympy NORMATIVE + 6 slope + SIMD bit-equality + new `apply_into_byte_equal` proptest.

## Acceptance criteria (release-blocking)

1. All 18 NORMATIVE sympy gates PASS.
2. All 6 slope gates (G3, G3⁴, G3⁶, G3⁶-2D, G4_NS2D_aniso, G5_3D) PASS on prod-HW.
3. SIMD bit-equality regressions PASS.
4. New `tests/apply_into_byte_equal.rs` proptest PASSES.
5. New `tests/zero_alloc_steady.rs` PASSES — 0 allocs/step after 3 warmup steps for every kernel.
6. Miri PASS on `tests/zero_alloc_steady.rs`.
7. `cargo xtask test-fast`, `test-full`, `test-flagship` all green.
8. Direct-dep budget unchanged at ≤3 (only dev-dep grows: +allocation-counter or +dhat fallback).
9. File-cap respected.
10. v1.0.0 API surface intact: `cargo semver-checks` reports only ADDITIVE changes.

## Out of scope (deferred)
- Per-thread parallel pools → W2
- Pencil ping-pong in Strang2D/3D → W2
- State<F> trait split → W3
- AdaptivePI scratch integration → W4
- FFI/PyO3/WASM zero-copy → W5

**Designed-By**: ai-solutions-architect
**Reviewed-By**: pending
