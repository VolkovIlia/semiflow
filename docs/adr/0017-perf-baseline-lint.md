# ADR-0017 — v0.8.0 performance baseline + lint hygiene

**Status**: Accepted
**Date**: 2026-05-03
**Authors**: agentic-engineer
**Cross-refs**: ADR-0012 (tensor-product 2D), ADR-0015 (6th-order spatial),
ROADMAP.md v0.8.0 PERFORMANCE THEME items 1–6, `docs/perf-baseline-v0_7_0.md`,
`crates/semiflow-core/benches/heat_2d.rs`, `crates/semiflow-core/benches/advdiff_2d.rs`.

v0.8.0 opens the **PERFORMANCE THEME** milestone: the mathematical kernel is frozen at v0.7.0; all v0.8.x work targets throughput (parallel Strang2D, SIMD intrinsics, G3⁶-2D flagship gate). Block A is a no-math no-API-change preparatory block: the 66 pre-existing Clippy warnings catalogued in the v0.7.0 CHANGELOG ("Pre-existing debt") are addressed mechanically — doc-backtick wrapping (21 sites), similar-name disambiguations (17 sites), lossless cast rewrites or bounded-domain `#[allow]` with rationale comment (19 sites), float-equality replacements in test asserts (3 sites), module-name-repetition `#[allow]` for stable public types (2 sites), two function splits to satisfy the 50-line suckless limit, and one redundant-borrow fix — with zero behavior change and full v0.5.0 bit-equal regression. Two new bench files (`benches/heat_2d.rs`, `benches/advdiff_2d.rs`) using `std::time::Instant` (no criterion dep) establish the N=400²/800²/1600² wall-clock baseline recorded in `docs/perf-baseline-v0_7_0.md`; these numbers are the falsification surface for the ≥5× speedup gate targeted by Blocks B+C.
