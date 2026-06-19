# ADR-0127 — Complex-λ Laplace-Chernoff Resolvent over `SemiflowComplex`

- **Status**: Accepted (PRE-FLIGHT GO)
- **Date**: 2026-06-06
- **Wave**: v7.0.0 Phase 4 (TIER-2 SemiflowComplex cluster) — additive, NON-BREAKING.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0069 (Laplace-Chernoff resolvent, real-λ surface this extends), ADR-0079 (`SemiflowComplex` trait), ADR-0025/0026 (Generic-over-Float / `ChernoffFunction` generic), ADR-0041 (`apply_into` + `ScratchPool`).
- **Supersedes / amends**: lifts the §22.7 limitation #1 ("Real λ only … deferred to v4.0 B6 SemiflowComplex"); strictly additive on the public surface (NEW method, no signature change to `eval`).
- **Mathematical foundation**: math.md §22.9 (NEW — complex-λ extension; CITATION Pazy 1983 §1.5 Hille-Yosida, Engel-Nagel 2000 §II.1 resolvent on the right half-plane).
- **Acceptance gates added**: `G_CPLX_RES` (RELEASE_BLOCKING — complex-λ resolvent-identity residual ≤ 1e-3), `T_CPLX_RES` (NORMATIVE sympy/scipy — 4 sub-checks, `scripts/verify_complex_resolvent.py`).

## Decision

The shipped real-λ resolvent integral `(λI−A)⁻¹g = ∫₀^∞ e^{−λt}S(t)g dt` extends to **complex λ with `Re λ > ω`** unchanged: the imaginary part of λ contributes only a unit-modulus oscillation `e^{−i(Im λ)t}` that does not affect absolute convergence (governed solely by `e^{−(Re λ−ω)t}`). The same Gauss-Laguerre 32-pt quadrature `R̃(λ)g = (1/λ)Σ_k w_k (C(s_k/(λn)))^n g` applies with a **complex** change of variable `s = λt` (Cauchy contour rotation, valid for `|arg λ| < π/2`). Ship one additive method `LaplaceChernoffResolvent::eval_complex<Cx: SemiflowComplex>(self, lambda: Cx, g) -> Result<…>` on inner Chernoff functions whose state carries `SemiflowComplex` values; the real `eval` is untouched. The validation guard MUST check `Re λ > ω` (NOT `|λ| > ω`): PRE-FLIGHT sub-check 3 shows a large-modulus left-half-plane λ (`|λ|=5.02, Re λ=−0.5`) makes the integral DIVERGE (residual ~1e+92) — this is the §22.7-analogue of the Killing idempotent-indicator trap. PRE-FLIGHT (`T_CPLX_RES PASS`) confirms: real-axis reduction 4.2e-6, complex-λ worst residual 6.3e-4 ≤ 1e-3, canonical datum λ=1+1i → 3.3e-5 (~30× margin), Cauchy-Riemann residual 1.9e-12 (holomorphic).

## Consequences

Additive only — no 4th dependency (`num-complex` is already 3/3-budget; matrix ops reuse the existing scalar machinery via the inner Chernoff state). Enables complex spectral analysis (resonances, Heston/SABR characteristic-function closure at complex frequencies). Margin is `Re λ`-dominated; the engineer gate pins `λ=1+1i` to stay ~30× under budget. Trade-off: `Re λ → ω⁺` with large `Im λ` approaches the 1e-3 budget — documented, gate datum avoids that corner.
