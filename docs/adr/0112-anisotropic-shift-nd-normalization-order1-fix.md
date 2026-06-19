# ADR-0112 — AnisotropicShiftChernoffND: F(0)=I normalization fix, honest order-1, N(D) ladder, ζ⁶/ζ⁸ order() semantics

- **Status**: Accepted
- **Date**: 2026-05-30
- **Authors**: ai-solutions-architect
- **Supersedes / amends**: amends ADR-0081 (corrects the order claim, the discrete formula §32.4 eq 32.3, the §32.5 N=128 spec, and removes the fabricated "§D=5 fallback"); amends the order() contract semantics jointly with ADR-0110.
- **Mathematical foundation**: math.md §32.4 (re-derived eq 32.3) + §32.5 (corrected gate). Triggered by the read-only audit findings A1/P1 (CRITICAL), A2/B1/A3 (D≥4 resolution + fabricated citation), A4 (false-pass), and A1-HONEST-DEFER (ζ⁶/ζ⁸).

## Context

The read-only QA/code audit (`.dev-docs/reports/{qa-math-honesty-review,code-review,audit-backlog}-today.md`) found `AnisotropicShiftChernoffND::apply_into` (`shift_nd.rs:383-395`) mathematically broken. It computed `acc = Σ_q w_q·f(y_q)` then `dst = e^{τc}·acc` with NO normalization. The 5-point physicist Gauss-Hermite weights sum to √π per axis, so for `f≡1, c=0` the operator returns `π^(D/2) ≠ 1`: the Chernoff precondition `F(0)=I` is violated and `(F(τ/n))^n` diverges as `π^(nD/2)→∞`. The `G_DDIM` RELEASE_BLOCKING gate did not catch it because the divergence produces `inf−inf=NaN` and the sup-norm reduction `f64::max(0.0, NaN)=0.0` masked the error as `err=0.0`, yielding a `NaN` slope and a panic that CI never ran (tests are `#[cfg(feature="slow-tests")]`). The normative contract math.md §32.4 eq (32.3) enshrined the same un-normalized formula and additionally used node scale `√(2τ)` where the kernel requires `2√τ`. Compounding: `order()` returned 2 claiming a "ζ²-d-D correction baked into the baseline" that does not exist in the code; the D=5 gate was relaxed `-1.95→-1.7` citing a non-existent "ADR-0081 §D=5 fallback"; and §32.5 hardcoded `N=128` for all D (274 GB at D=5).

## Decision

**1. Normalization + node scale (the CRITICAL fix).** Re-derive eq (32.3) by whitening the §32.2 Gaussian integral: substitute `y = x_k + 2√τ·σ·s` (σ = Cholesky factor of A(x_k)). The exponent collapses to `−sᵀs`, the Jacobian `(2√τ)^D·√detA` cancels the `(4πτ)^{D/2}·√detA` prefactor to exactly `π^{−D/2}`. The corrected NORMATIVE formula is
`(F_A(τ)f)_k = e^{τc(x_k)}·π^{−D/2}·Σ_q w_q·f(x_k + τ·b(x_k) + 2√τ·σ·η_q)`.
Both corrections are independently required: `π^{−D/2}` restores `F(0)=I` exactly (`π^{−D/2}·Σw_q = 1`); the `2√τ` scale gives the correct heat variance `2τ·A` (the `√(2τ)` scale was off by a factor 2). Verified by sympy/numpy (ADR-0112 appendix). There is NO residual `1/√detA`/`(4πτ)^{D/2}` factor — both are absorbed by the whitening.

**2. Order is 1, not 2 — set `order()=1`, gate `-0.95`.** The frozen-coefficient kernel reproduces the exact heat semigroup to all orders for CONSTANT A (the Gaussian average IS the exact solution), but for variable A the leading per-step mismatch is `(τ²/2)(a·a''·f'' + 2a·a'·f''')` (sympy-proven; empirical 1D self-convergence slope = −1.02). This is order-1 (global O(1/n)) — exactly what Theorem 32.1 already states. A genuine order-2 lift needs the d-D analog of the 1D ζ-A correction (ADR-0008): explicit `∂_m A(x)` gradient/Hessian closures plus a τ² correction polynomial, with the missing-odd-derivative obstruction of §9.2.2 reappearing in d-D. That is a major additive API + derivation and is **DEFERRED** (math §32.6). v6.0.0 ships honest order-1. The "ζ²-d-D correction baked into the baseline" claim (ADR-0081 §Rationale, math §32.2/§32.5, `shift_nd.rs:400`) was false and is retracted.

**3. N(D) ladder (replaces N=128).** Publish the normative table the tests use: N_AXIS(D)={128,32,8,6}, n_ref(D)={2048,2048,512,512} for D={2,3,4,5}. N(D) is the largest per-axis count keeping N^D·5^D within CI budget; the self-convergence gate measures temporal order (spatial floor cancels common-mode), so coarse N(D) does not bias slope. *(This claim is empirically WRONG for D=2,3 — see AMENDMENT 1 below.)*

**4. Remove fabricated citation; restore D=5 to -0.95.** "ADR-0081 §D=5 fallback" does not exist. The D=5 gate uses the SAME -0.95 as D≤4 (no curse-of-dimensionality relaxation is needed for a temporal-order self-convergence measurement). The fabricated reference is removed from the D=4 and D=5 tests, CHANGELOG, and ROADMAP.

**5. Self-masking gate fix.** The slope sup-norm reduction MUST propagate NaN/non-finite, and the assertion MUST be `assert!(slope.is_finite() && slope <= -0.95)`. A divergent run must NEVER read as `err=0.0`. Add a REQUIRED `F(0)=I` smoke sub-test per D (`‖F(τ)·1 − 1‖_∞ < 1e-12` at τ∈{0, T/16, T/128}) as the direct normalization guard.

**6. ζ⁶/ζ⁸ order() semantics (A1-HONEST-DEFER) — QUALIFY, do not change return values.** `Diffusion6thZeta6Chernoff::order()=6` and `Diffusion8thZeta8Chernoff::order()=8` are correct as **LOCAL** Taylor-tangency orders (sympy-proven T23N). They are NOT empirical global-order guarantees — the global gates `G_zeta6/8_TRUTHFUL_ORDER` are deferred to v7.0+ (ADR-0110 AMENDMENT 1). The contract (traits.yaml `order` semantics) is qualified: `order()` is the LOCAL consistency exponent m (single-step O(τ^{m+1})); consumers (AdaptivePI/ClassicalPI gain scheduling) MAY use it for PI tuning (gains are tuning constants, stable under the weaker reading) but MUST NOT assume a measured global slope of −m at feasible step sizes. Return values stay 6/8 (renaming/zeroing would break the load-bearing AdaptivePI consumer and lose the truthful local-order claim).

## Rationale

Correctness over expedience: this is a math library. Adding a half-derived ζ²-d-D correction to "save" the -1.95 gate would be the dishonest move; the audit explicitly praised the ζ⁶/ζ⁸ honest-defer posture and condemned the silent order-2 claim. Order-1 + a real F(0)=I guard + a finite-slope assertion is the suckless, honest fix. The N(D) ladder documents reality (the tests already use it); the uniform N=128 was never runnable. Qualifying (not mutating) ζ⁶/ζ⁸ `order()` preserves the truthful local-order claim while protecting consumers from over-reading it.

## Consequences

- `shift_nd.rs::apply_into` divides by `π^(D/2)` and uses node scale `2√τ`; `order()` returns 1; rustdoc order/ζ²-correction claims corrected. Engineer task (separate, bug-fixer): see implementation spec.
- The 4 slope tests: add `slope.is_finite()` assertion + NaN-propagating reduction + F(0)=I smoke; gate -0.95; remove fabricated citation; encode N_AXIS/N_REF ladder (D=5 gate -1.95→-0.95).
- math.md §32.2/§32.4/§32.5/§32.6 corrected (this ADR). properties.yaml G_DDIM gate/threshold/ladder corrected. traits.yaml `order` semantics qualified. No public type/signature change (additive prose + one return-value change 2→1).
- Schema: properties.yaml PATCH (gate semantics); traits.yaml PATCH (order semantics prose). math.md append/correct.
- CHANGELOG/ROADMAP: remove fabricated "ADR-0081 §D=5 fallback"; record order 2→1 honest correction (BREAKING for any caller that read `order()==2` — but no such caller exists; AnisotropicShiftChernoffND is not yet wired into AdaptivePI).

## Appendix — derivation evidence (re-runnable)

- Normalization/scale: `1/√π·Σ w_q f(x+2√τ·η_q)` matches the exact N(x0,2τa) heat average to 1e-10; the pre-v6 `√(2τ)`/no-norm form gave 2.397 vs exact 1.404. F(0)=I exact.
- Order: sympy `exact−disc` τ² coefficient = `(a/2)(a''·f'' + 2a'·f''')` ≠ 0 ⇒ order-1; empirical 1D variable-a self-convergence slope = −1.02.

## AMENDMENT 1 (2026-05-30) — N(D) ladder corrected to a UNIFORM COARSE grid; the "common-mode cancellation" reasoning was wrong for D≥2

- **Status**: Accepted
- **Author**: ai-solutions-architect
- **Trigger**: adversarial QA pass (probe `tests/adversarial_qa_probe.rs`, 5 experiments, slow-tests gated, run 2026-05-30) empirically FALSIFIED §Decision 3.

**What was wrong.** §Decision 3 published the ladder `N_AXIS(D)={128,32,8,6}`, `n_ref(D)={2048,2048,512,512}` and justified the *fine* D=2/D=3 grids with the claim "the self-convergence gate measures temporal order; the spatial floor cancels common-mode, so coarse N(D) does not bias slope." **That reasoning is false for this kernel.** The adversarial QA pass measured:

| D | N_AXIS | measured slope | verdict |
|---|--------|----------------|---------|
| 2 | 128 | ≈ −0.05 (non-monotone) | floor-dominated — FAILS the −0.95 gate |
| 2 | 8   | ≈ −1.03 | clean order-1 — PASSES |
| 3 | 32  | ≈ −0.67 | floor-polluted — FAILS the −0.95 gate |
| 3 | 8   | ≈ −1.20 | clean order-1 — PASSES |
| 4 | 8   | ≈ −1.08 | clean order-1 — PASSES |
| 5 | 6   | ≈ −0.9614 | order-1, THIN margin vs the −0.95 gate |

**Root cause (the corrected reasoning).** The "spatial error cancels common-mode" premise holds only for a discretisation error that is *identical* between the sweep iterate `u_n` and the reference `u_ref`. It is NOT. The dominant spatial error here is the per-step **multilinear interpolation** error of `GridFnND::sample` at the off-grid Gauss-Hermite shifted points `x_k + 2√τ·σ·η_q`. This interpolation error accumulates as **O(n·dx²)** along each trajectory and `u_n` (n steps) and `u_ref` (n_ref ≫ n steps) take a *different number of steps* — so the accumulated interpolation floors do NOT cancel in `‖u_n − u_ref‖_∞`. On a *fine* grid (small dx) the interpolation floor is small in absolute terms but, critically, it sits at a *temporal-step-count* scaling that flattens the measured temporal slope toward 0 once it dominates the genuine O(1/n) truncation signal. A **coarse** grid (large dx) inflates dx² so the *temporal-truncation* signal dominates the interpolation floor across the sweep, recovering the true order-1 slope. The "exact for constant A" property (Appendix) is a property of the CONTINUOUS frozen-coefficient formula; the discretised kernel always carries an O(dx²) interpolation error regardless of A, so it cannot rescue a fine-grid measurement.

**Corrected normative ladder (matches the test files verbatim, verified 2026-05-30):**

| D | N_AXIS | N_REF | N_SWEEP |
|---|--------|-------|---------|
| 2 | **8** | 512 | {32,64,128,256} |
| 3 | **8** | 512 | {32,64,128,256} |
| 4 | **8** | 512 | {16,32,64,128} |
| 5 | **6** | 512 | {16,32,64,128} |

`N_AXIS(D) = {8, 8, 8, 6}`; `N_REF(D) = 512` uniformly for ALL D (NOT the {2048,2048,512,512} originally claimed — the actual tests use 512 across the board). The grid is now deliberately COARSE so the temporal-truncation signal dominates the interpolation floor.

**D=5 honesty note.** At D=5 the curse of dimensionality forces N_AXIS=6 (6⁵=7776 nodes; N_AXIS=8 would be 8⁵=32768 × 5⁵=3125 quad/pt, beyond CI budget). The measured slope ≈ −0.9614 clears the −0.95 gate by a **thin margin** (~1.5%). This is honestly the curse-of-dimensionality boundary: at N=6 the coarse-grid temporal signal only just dominates the interpolation floor. Order-1 at D=5 is demonstrated, but with a thin margin — it is NOT a comfortable pass. Any future tightening of the gate or change to the D=5 quadrature must re-measure this margin.

**Why an amendment, not a silent rewrite.** §Decision 3 and §Rationale are preserved verbatim above (with an inline pointer) so the audit trail records WHY the original "common-mode cancellation" reasoning was adopted and WHY it was empirically overturned. The original reasoning was a plausible-but-untested analogy to the v2.2 `G_NS2D_aniso` self-convergence gate; that gate uses a finite-difference Laplacian whose spatial error genuinely is step-count-independent, whereas this kernel's interpolation error is not. The analogy failed because the error mechanisms differ.

**adversarial_qa_probe.rs decision — KEEP as a permanent documented diagnostic (option a).** The file is a 483-line, slow-tests-gated, no-slope-assert investigation harness with clear module rustdoc (5 experiments: D=2/D=3 N_AXIS reconciliation, finite+monotone err-ladder genuineness, range-robust anti-cherry-pick slope, constant-A exact sanity, D=5 full sweep). It is the **empirical witness** for this AMENDMENT and the load-bearing evidence for the N(D) calibration. Keeping it permanently:
1. documents the floor-vs-temporal-signal tradeoff as runnable code (not just prose);
2. protects against the most likely future regression — someone "optimising" N upward to a fine grid and silently reintroducing the floor-domination bug this amendment fixes;
3. costs nothing in CI (slow-tests gated, no asserts).
The engineer/QA should: (i) ensure the module rustdoc explicitly states it documents the N(D) calibration and the interpolation-floor-vs-temporal-signal tradeoff (cite this ADR AMENDMENT 1), and (ii) keep it slow-tests gated with NO release-gating asserts (it is a diagnostic, not a gate — `G_DDIM` remains the release gate). Do NOT delete it as scratch. *(I do not edit the .rs — this is the architect's recorded decision for the engineer/QA to act on.)*

**Consequences of AMENDMENT 1.**
- math.md §32.5 N(D) table + "Why the gate is N-robust" prose corrected to the coarse uniform ladder + interpolation-floor reasoning + D=5 thin-margin note (this amendment).
- properties.yaml G_DDIM N_AXIS/n_ref corrected to {8,8,8,6}/512 (this amendment).
- No code change beyond what §Decision already mandated; the tests already use the corrected ladder (the original §Decision 3 ladder was never the ladder the tests ran).
- No new BREAKING surface: `order()` stays 1; the gate threshold stays −0.95; only the documented N(D)/n_ref values change to match reality.

---

## AMENDMENT 2 (2026-06-05) — order-2 ζ² correction as an ADDITIVE constructor (lifts the §32.5 Note / §32.6 deferred order-2)

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — additive `with_zeta2_correction` constructor; `order()` of the existing type UNCHANGED).
- **Backlog item**: v7.0.0 freeze item #23 (Phase 5). The freeze maps this to "AMENDMENT 1 to ADR-0112"; since this file already carries an AMENDMENT 1 (the N(D)/coarse-grid calibration above), the ζ² lift lands as **AMENDMENT 2**.
- **Mathematical foundation**: PRE-FLIGHT `scripts/verify_anisotropic_shift_zeta2_ddim.py` (executed 2026-06-05).
- **Acceptance gate added**: `G_AS_ZETA2_DDIM` (RELEASE_BLOCKING) — order-2 self-convergence slope ≤ −1.95 on the COARSE-grid datum (the AMENDMENT 1 protocol).

**Context.** The v6.0.0 baseline kernel (eq 32.3) is honest order-1 for variable A; §32.5 Note / §32.6 deferred the order-2 ζ² lift, citing the d-D analog of the 1-D ζ-A correction (math.md §9.2.3.B, ADR-0008) — explicit ∂A closures plus a τ²-correction polynomial. This amendment authorises that lift as an ADDITIVE constructor, keeping the order-1 type's `order()` untouched (the order-2 lift is a NEW constructor `AnisotropicShiftChernoffND::with_zeta2_correction(... grad_a_ij ...)`, not a mutation).

**PRE-FLIGHT (sympy, executed 2026-06-05).** Reduced d-D to representative D=1 (regression vs known §9.2.3.B) and D=2 (genuine off-diagonal coupling a₁₂≠0). Via exact Gaussian-moment (Isserlis/Wick) expansion of the frozen-coefficient average vs `e^{τL}`:
- τ⁰, τ¹ deficit = 0 at both D (frozen kernel is already order-1, confirming Theorem 32.1).
- **All-constant (A and b constant) τ²-deficit = 0** at both D → confirms §32.2: the frozen kernel is EXACT for constant coefficients (sanity check validating the probe).
- **A-gradient sources a non-trivial τ²-deficit**: D=1 → −1249/9800, D=2 → −8129/29400 (variable A, constant b). This is the ζ² correction TARGET; it vanishes when ∂A→0, so an explicit C₂ built from ∂A closures and f-derivatives KILLS it exactly (the d-D analog of the §9.2.3.B bracket).

**Load-bearing engineer caveat (surfaced by PRE-FLIGHT).** The §32 frozen kernel freezes A AND b AND c. The PRE-FLIGHT shows **variable b ALSO sources a τ²-deficit** (constant-A variable-b: D=1 → −42121/180075, D=2 → −191783/360150). So a FULL order-2 lift for the general operator needs BOTH ∂A and ∂b correction terms. The ζ² correction as scoped (the diffusion ∂A piece) is sufficient for the gate ONLY because the §32.5 gate datum uses b≡0 (zero drift). The engineer MUST either (a) restrict `G_AS_ZETA2_DDIM` to the b≡0 datum (matching §32.5), or (b) additionally include the ∂b drift-gradient correction. **Recommendation: scope v7.0.0 to (a)** — the ∂A ζ² lift on the b≡0 gate datum — and document the ∂b drift-gradient term as a follow-on (§32.6 future-extension note), since the §32.5 gate is already b≡0.

**Honest order caveat (mirror of §9.2.3.B "Order claim revised").** Like the 1-D ζ-A, the GLOBAL empirical rate for variable A is capped near O(τ¹) by interpolation/FD noise on f-derivatives at off-grid GH nodes. The ζ² correction kills the τ²-deficit at the ANALYTIC (local-tangency) level — which is the mathematically-true order-2 claim and what `G_AS_ZETA2_DDIM` measures on a COARSE grid (temporal signal above the interpolation floor, per AMENDMENT 1). If the COARSE-grid slope does not reach −1.95 the engineer falls back to documenting the analytic τ²-kill as the order-2 evidence with an honest implementation-ceiling note (precedent: §9.2.3.B Amendment 2).

**Consequences.** New additive constructor + ∂A closure inputs; `order()` of the base type unchanged; math.md §32.6 ζ² bullet promoted from "deferred" to "shipped (b≡0 datum)"; `G_AS_ZETA2_DDIM` added to properties.yaml; no BREAKING surface; 3/3 dep cap untouched (sympy-derived C₂ is pure in-tree arithmetic).
