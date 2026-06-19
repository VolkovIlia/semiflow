# ADR-0021 — v0.8.0 release

**Status**: Accepted
**Date**: 2026-05-06
**Authors**: agentic-engineer
**Cross-refs**: ROADMAP.md v0.8.0; ADR-0017 (perf baseline + lint);
ADR-0018 (parallel Strang2D); ADR-0019 (SIMD intrinsics, Amendment 2026-05-06);
ADR-0020 (G3⁶-2D flagship, Amendment 2026-05-06 #2);
`docs/audit-findings-v0_8_0.md`.

v0.8.0 ships Blocks A (lint hygiene + 2D bench scaffolding, commits 1399e7f +
a1d8c78), B (production parallel `Strang2D::apply` with `STRANG2D_PARALLEL_BIT_EQUAL`
3/3 pass, commits 300054f + 3a33e98), and C (SIMD intrinsics AVX2/NEON with
`SIMD_BIT_EQUAL` 2/2 pass + `SIMD_BIT_EQUAL_PARALLEL` 3/3 pass, commits a927aac +
7d39938 + 3091c12) as independently green performance additions; Block D ships the
G3⁶-2D FLAGSHIP test infrastructure (`tests/convergence_rate_6th_2d.rs`, `#[ignore]`)
with gate calibration deferred to v0.8.1 per ADR-0020 Amendment 2026-05-06 #2 after
two consecutive runtime-budget failures (v0.8.1 plan: N up to 2048, raised budget —
calibration adjustment, not architectural change). Math is frozen since v0.7.0:
the researcher-agent fidelity audit (`docs/audit-findings-v0_8_0.md`, 2026-05-06)
confirms zero algorithmic change, all numerical constants preserved verbatim, and the
v0.5.0 bit-equal regression vector reproduces byte-for-byte both on the plain scalar
build and with `parallel,simd` enabled. Suckless invariants hold: runtime deps
unchanged at 2 (`num-traits`, `libm`); largest src file 460 LoC (≤500); all
functions ≤50 lines; `cargo clippy --all-targets --features parallel,simd,slow-tests
-- -D warnings` exits 0. Workspace lint policy moved `unsafe_code = "forbid"` →
`"deny"` (required for scoped `#![allow(unsafe_code)]` inside `src/simd/x86_64.rs`
and `src/simd/aarch64.rs`; enforcement substitute is `xtask check-unsafe-scope`
which fails the build if any `unsafe` or `#[allow(unsafe_code)]` appears outside
`src/simd/`, per ADR-0019 Amendment 2026-05-06); the bit-equal contracts of
Blocks B and C compose correctly (`SIMD_BIT_EQUAL_PARALLEL` confirms the combined
path byte-matches the v0.7.0 scalar/serial reference end-to-end.
