# ADR-0066 — tracking-alloc Cargo feature

- **Status**: ACCEPTED (v2.5.1)
- **Date**: 2026-05-26
- **Wave**: v2.5.1 (cache-residency + HFT benchmarks)
- **Authors**: claude (docs-writer)
- **Depends on**: ADR-0028 (FFI/PyO3/WASM v0.10 — no_std constraints),
  ADR-0025 (Generic-over-Float).
- **Mathematical foundation**: none — implementation/tooling only.

## Context

The cache-residency experiment (Phase 1) required measuring hot-loop peak
allocation accurately at runtime, without RSS inflation from shared libs.
Standard OS-level RSS (`/proc/PID/status` VmRSS) includes code segments, libc,
and other read-only mappings — at N=512 the program-data component is only
~152 KB of the ~2 MB RSS. A Cargo feature that installs a `#[global_allocator]`
wrapper with `AtomicUsize` counters was the lowest-overhead approach: it adds
zero code when the feature is off, and the two atomic reads (peak / delta)
outside the hot loop add no measurable latency.

## Decision

Add a Cargo feature `tracking-alloc` to `crates/semiflow-core` (off by default).
When enabled, a `struct TrackingAlloc(System)` implements `GlobalAlloc` by
delegating every call to `System` and updating two `AtomicUsize` globals
(`PEAK_BYTES`, `CURRENT_BYTES`). Exposed helpers: `reset_tracking()`,
`peak_bytes() -> usize`, `current_bytes() -> usize`.

The `cev_european_call.rs` example uses the feature for the `--measure` CLI
flag (hot-loop working-set reporting) and the `--measure-only` flag (skip GBM
file rewrite, read-only). The `cache_phase4` criterion bench uses it to assert
zero net allocation in the hot loop (AC-5 invariant).

## Consequences

- Production users with `tracking-alloc` disabled pay zero code-size cost and
  zero runtime cost; the feature is purely opt-in.
- Examples and benches asserting zero-allocation in the hot loop can enable the
  feature via `--features tracking-alloc` without touching production code.
- The feature is additive to the public API surface (no SemVer break, PATCH bump).
- Two `AtomicUsize` globals are introduced under `#[cfg(feature = "tracking-alloc")]`
  in `crates/semiflow-core/src/alloc_tracker.rs`; they are `#[doc(hidden)]` and
  not part of the stability guarantee.
