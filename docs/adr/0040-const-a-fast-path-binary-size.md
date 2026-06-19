# ADR-0040 — Const-a fast path + WASM binary-size profile (v0.13.0 Wave D)

**Status**: Accepted
**Date**: 2026-05-20
**Context**: v0.13.0 Wave D (D1 + D2 + D3). Architectural performance improvements (memory-Pareto reinforcement, Vector 2 / H-MEM brand).

## D1 — Const-a fast path (`DiffusionChernoff::new_const_a`)

When `a(x) ≡ const`, `a'(x) ≡ 0` and `a''(x) ≡ 0`. The ζ-A τ²-correction evaluates to exactly zero, and the inner Strang shift `x_pre = x + τ/2·a'(x)` reduces to `x_pre = x`. A new `Storage::ConstA` variant + `DiffusionChernoff::new_const_a(a_value, a_norm_bound, grid)` constructor activates these reductions automatically at call-time by branching on the storage variant. Gate `CONST_A_BIT_EQUAL` verifies bit-identical output vs the `with_closure` path with zero derivative closures.

## D2 — WASM bundle size profile (`[profile.release-wasm]`)

A new workspace profile `[profile.release-wasm]` (opt-level=z, lto=true, codegen-units=1, panic=abort, strip=true) targets <500 KB stripped WASM bundle. `cargo xtask wasm-build --size` selects this profile. Default `wasm-build` continues to use `[profile.release]` for reproducible CI.

## D3 — Binary-size CI gate (`binary-size-check`)

A new `cargo xtask binary-size-check` command audits built artefacts vs budgets. A `binary-size-check` CI job (depends on wasm-build, ffi-build, py-build) fails PRs that produce over-budget artefacts.
