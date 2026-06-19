# ADR-0036 — Transparent parallel `apply` for 1D Chernoff types

**Status**: Accepted
**Date**: 2026-05-13
**Authors**: docs-writer (ai-solutions-architect — originating design)
**Cross-refs**: ADR-0018 (parallel `Strang2D::apply`), ADR-0019 (SIMD
intrinsics), ADR-0025 (generic-over-Float), ADR-0026 (`ChernoffFunction`
trait generic), `crates/semiflow-core/src/parallel1d.rs`,
`crates/semiflow-core/tests/chernoff1d_parallel_bit_equal.rs`,
`contracts/semiflow-core.properties.yaml` gate
`CHERNOFF1D_PARALLEL_BIT_EQUAL` (release-blocking).

## Context

Seven stand-alone 1D Chernoff types — `ShiftChernoff1D`,
`DiffusionChernoff`, `Diffusion4thChernoff`, `Diffusion6thChernoff`,
`TruncatedExpDiffusionChernoff`, `TruncatedExp4thDiffusionChernoff`,
`DriftReactionChernoff` — were structurally single-threaded prior to
v0.12.0. ADR-0018 (v0.8.0) added a parallel path to `Strang2D::apply` and
`Strang3D::apply` that fans out across rows and pencils; however, each
per-row 1D `apply` call remained serial. Standalone 1D users — flagship
gates F1–F4, F9, F10; FFI / PyO3 / WASM bindings; the `heat_1d` criterion
bench — could not benefit from multi-threaded speedup regardless of grid
size. For a 1D grid of N = 4096 nodes on a 12-core host the serial path
leaves roughly an order of magnitude of compute throughput unused.

## Decision

All seven f64 stand-alone 1D Chernoff types route their `apply` method
through `crate::parallel1d::parallel_eval` under `#[cfg(feature =
"parallel")]`. The dispatcher partitions the output index range
`0..n` across `std::thread::scope` threads using deterministic
ceiling-division (`n.div_ceil(n_threads)`) with disjoint per-thread
writes, mirroring the ADR-0018 bit-equality contract: byte-identical f64
output across thread counts {1, 2, 4, 8}. The cutoff
`MIN_POINTS_PER_THREAD = 1024` ensures that per-row calls issued from
`Strang2D::apply_parallel` and `Strang3D::apply_parallel` (typical row
N ≤ 512) fall back to serial automatically, preventing nested-thread
oversubscription. Generic-over-Float paths (F = f32) remain serial:
ADR-0019 SIMD auto-vectorisation reorders f32 operations, which would
break bit-equality under thread rescheduling.

## Bit-equality test gate

File: `tests/chernoff1d_parallel_bit_equal.rs` (262 LoC).
Feature gate: `--features parallel,slow-tests --release`.
Matrix: 7 types × N ∈ {1024, 2048, 4096} × n_steps ∈ {1, 4} ×
n_threads ∈ {1, 2, 4, 8} — 168 assertions per type, 1176 total.
The gate is release-blocking (`CHERNOFF1D_PARALLEL_BIT_EQUAL` in
`contracts/semiflow-core.properties.yaml`).

## Consequences

**Positive**:
- Transparent speedup for standalone 1D users compiled with
  `--features parallel`; no API change, source-compatible with all callers.
- ADR-0018 bit-equality invariants extended to the 1D layer.
- No new runtime dependencies (stdlib only: `std::thread::scope`,
  `std::sync::Mutex`, `std::cell::Cell`). Dep count remains 2.

**Negative / deferred**:
- Threshold N < 1024 standalone benchmarks remain serial. A future
  `pub fn set_parallel_threshold_1d(usize)` can lower the floor if
  benchmarks justify it.
- WASM (`wasm32-unknown-unknown` without the `threads` feature) is
  mutually exclusive with `--features parallel`; no change for those users.

## Related

ADR-0018 (parallel `Strang2D`), ADR-0019 (SIMD), ADR-0025
(generic-over-Float), ADR-0026 (`ChernoffFunction` trait generic).
