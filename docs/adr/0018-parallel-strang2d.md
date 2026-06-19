# ADR-0018 — v0.8.0 Block B production parallel Strang2D::apply (opt-in)

**Status**: Accepted
**Date**: 2026-05-05
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0012 (tensor-product 2D), ADR-0013 (4th-order spatial),
ADR-0017 (v0.8.0 perf baseline + lint), ROADMAP.md v0.8.0 PERFORMANCE THEME
item 1, `crates/semiflow-core/tests/heat_2d_oracle_4th.rs` (reference parallel
kernel), `contracts/semiflow-core.tensor.yaml` schema 0.7.1 (Strang2D
`feature_flags` / `thread_safety` / `parallel_correctness`),
`contracts/semiflow-core.properties.yaml` schema 0.7.1 gates
`STRANG2D_PARALLEL_BIT_EQUAL` (release-blocking) and `STRANG2D_PARALLEL_SPEEDUP`
(informational), `docs/perf-baseline-v0_7_0.md`.

`Strang2D::apply` ships an internal multi-threaded path behind a new opt-in
`parallel` Cargo feature: when the feature is enabled, the per-row X-pass and
the per-column Y-pass fan out via `std::thread::scope` (stable since the crate
MSRV 1.78); when it is disabled (the default), the v0.7.0 serial code path is
byte-identical and `--no-default-features` continues to compile no_std + alloc.
The kernel is the production-grade port of the test-only 8-thread idiom proven
in `tests/heat_2d_oracle_4th.rs::parallel_strang2d_step` — `Vec::chunks_mut` for
the X-pass, two-phase gather/scatter through a column-major temp for the Y-pass,
zero `unsafe`, zero atomics, zero reductions, zero new dependencies (rayon and
crossbeam are explicitly rejected to keep the dep count at 2 — `num-traits`,
`libm` — per the project suckless invariant). Determinism is preserved by
construction: every f64 written to the output buffer is the result of a single
1D `apply` on a single row or column with no thread-count-dependent FP
rearrangement, so the implementation owes a hard `STRANG2D_PARALLEL_BIT_EQUAL`
contract — bit-identical output across thread counts {1, 2, 4, 8} — which is
the SOLE acceptance criterion for landing the kernel; the ROADMAP "≥ 6× at
N=1600²" speedup target is enforced as the informational `STRANG2D_PARALLEL_SPEEDUP`
warn-gate, not release-blocking, because wallclock varies by host CPU.
**Alternatives considered**: rayon (rejected: would push the dep count from
2 to ≥4, violates the project suckless dep-count invariant); default-on
(rejected: ROADMAP v0.8.0 item 1 explicitly says "behind a `parallel` feature
flag", and default-on would force every `--no-default-features` no_std consumer
to inherit `std::thread`); manual `unsafe` parallel column writes (rejected:
violates `#![forbid(unsafe_code)]` at the crate root, and `chunks_mut` on a
column-major temp gives the borrow checker enough to prove disjoint writes
without it). **Consequences**: `cargo build --no-default-features` still compiles
no_std + alloc clean; `cargo build --features parallel` implies `std` and
adds a `Send + Sync` bound to `X` and `Y` ONLY at the parallel `apply` call site
behind `#[cfg(feature = "parallel")]` — every v0.7.0 inner type
(`DiffusionChernoff`, `Diffusion4thChernoff`, `Diffusion6thChernoff`,
`TruncatedExpDiffusionChernoff`, `TruncatedExp4thDiffusionChernoff`,
`DriftReactionChernoff`) auto-implements those marker traits today, so the
public API is source-compatible; v0.5.0 regression bit-equality (G3-2D, G3⁴-2D)
holds because the bit-equal contract makes the parallel build a strict
substitute for the serial build at the byte level.
