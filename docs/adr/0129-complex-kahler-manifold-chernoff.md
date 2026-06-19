# ADR-0129 — Complex Kähler-manifold Chernoff (scalar CP¹ Fubini-Study backend)

**Status:** PROPOSED (PRE-FLIGHT GREEN) · **Date:** 2026-06-06 · **Branch:** `feat/v7.0.0-debt-closure`
**Item:** v7.0.0 backlog #15 (§24.7 deferred) · **Gate:** `G_KAHLER_CURV`
**Numbering:** backlog assigned 0124 (consumed by matrix-2d3d-strang); reallocated to **0129**.

## Context

§24.7 ("Real-valued only") deferred complex Kähler structures + holomorphic line bundles "to v4.0 B6 SemiflowComplex". Shipped `ManifoldChernoff` (ADR-0072) provides real Torus/Sphere2/Hyperbolic2 with the MMRS-2023 `[1+(τ/12)R]` scalar-curvature correction (§24.2).

## Decision

Ship a **scalar** Kähler backend: complex projective line CP¹ with the Fubini-Study metric, as an additive `BoundedGeometryManifold`-style impl over `SemiflowComplex`. Key fact (PRE-FLIGHT-confirmed): CP¹ with Fubini-Study is **isometric to the round S²** — the conformal factor `4/(1+|z|²)²` is identical to S² stereographic projection, so the scalar Laplace-Beltrami operator (hence heat semigroup) is the SAME real elliptic operator the shipped Sphere2 backend already evolves. Therefore MMRS-2023 R/12 convergence theory applies VERBATIM (no new theorem); the NEW content is only the complex affine chart (`z ∈ ℂ`, complex-modulus exp_map/curvature over `SemiflowComplex`). PRE-FLIGHT `scripts/verify_complex_kahler.py` 3/3 PASS: FS scalar curvature **R=2** constant (S²-isometric); FS metric == S² stereographic metric; `[1+(τ/12)R]` tangent to `exp(τR/12)` to O(τ²). Gate `G_KAHLER_CURV`: curvature-corrected self-convergence slope ≤ −1.95 (mirror G26 sphere). **Holomorphic line-bundle sections** (genuinely complex-valued state — the harder object) remain a SEPARATE future deferral; this item ships the scalar Kähler heat backend only.

## Consequences

Additive; reuses the shipped `ManifoldChernoff` R/12 machinery + `manifold_curvature_kit.py`. No new deps. The "Kähler" label is honest at the scalar-heat level (complex chart, FS metric) but NOT yet a line-bundle/holomorphic-section engine — rustdoc must state this scope boundary to avoid over-claiming.
