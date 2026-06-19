# ADR-0139 — OctonicHermite + ChebyshevSpectralWithBC generic samplers (v8.1.0, A-2)

**Status:** ACCEPTED · **Date:** 2026-06-08 · **Branch:** `feat/v8.1.0-debt-closure`
**Theme:** v8.1.0 debt closure — closes honest-defer from ADR-0133 Amendment 1 · **Parent:** ADR-0133 · **Gates:** G1 (f64 byte-identity, 0 ULP, RELEASE_BLOCKING), G2 (Dual-AD gradient ≤1e-9, slow-tests)

ADR-0133 Amendment 1 genericised `SepticHermite` over `F: SemiflowFloat` and honestly deferred `OctonicHermite` and `ChebyshevSpectralWithBC` as f64-only; this ADR closes that defer. Both samplers are extracted into child modules (`grid_chebyshev_octonic/octonic_generic.rs`, `grid_chebyshev/chebyshev_generic.rs`) following the identical pattern, keeping both parent files within the 500-line budget. The f64 path is untouched and byte-identical (0 ULP, G1). No SIMD in the generic path (§46.5 carve-out). `Grid1D::interp_generic` now dispatches all five `InterpKind` variants for any `F: SemiflowFloat`.
