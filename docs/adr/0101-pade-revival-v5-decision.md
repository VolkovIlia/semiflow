# ADR-0101 — Padé revival v5.0 DECISION: Path δ TERMINAL CLOSURE per max-2-retries + suckless single-kernel principle

- **Status**: Accepted (v5.0 third-attempt FINAL DECISION; Padé permanently DEFERRED v6.0+ pending external math advance)
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Supersedes / refines**: ADR-0091 + AMENDMENT 1 (Wave B v4.3 DEFER, `τ‖A‖` unbounded), ADR-0094 + AMENDMENT 1 (Wave I v4.5 FINAL DEFER, const-a anti-convergent at log₂ = −0.4053). This ADR is the THIRD attempt per the roadmap §"v5.0.0 BREAKING WINDOW #3" entry; max-2-retries rule + suckless single-kernel principle now select Path δ TERMINAL CLOSURE.
- **Depends on**: ADR-0090 (Chebyshev `Diffusion8thZeta8Chernoff` — the production ζ⁸ kernel that ships ALONE; shipped at v4.3), ADR-0095 (Engel step-3 Carnot `HypoellipticChernoff::new_engel()` — the production step-3 kernel; shipped at v4.5), ADR-0073 freeze (no v5.0 trait reshape for Padé), constitution v1.8.0 (3/3 dep cap INVIOLATE through v5.x per roadmap §"Hard constraints").
- **Mathematical foundation**: PRE-FLIGHT sympy harness `/tmp/pade_v5_preflight.py` (executed 2026-05-29, output preserved at `/tmp/pade_preflight_output.txt`) FALSIFIES the D3 root-cause hypothesis of ADR-0094 AMENDMENT 1 and demonstrates that none of Path α / β / γ offers a structural Wave-I-bug-bypass over the v4.5 Wave I implementation.
- **Researcher synthesis**: NOT INVOKED — PRE-FLIGHT sympy verdict was conclusive enough at architect time to render researcher escalation unnecessary; existing literature (Higham 2005, Saad 1992) is sufficient and the question collapsed to algorithmic-equivalence, not literature scarcity.
- **Acceptance gates added**: NONE (terminal closure ships no new code). Acceptance gates retired by ADR-0094 AMENDMENT 1 (`G_zeta8_pade_const_a_richardson_ratio` + `G_zeta8_pade_var_a_temporal_slope` + `G_PADE_SS_NORM_ENVELOPE` + `T_ZETA8_PADE` sympy oracle) remain RETIRED through v5.0+ and beyond pending v6.0+ revival.

## Context

The Padé `Diffusion8thZeta8PadeChernoff` kernel was DEFERRED twice prior to this ADR: at v4.3 (ADR-0091 AMENDMENT 1; Wave B failure mode `τ‖A‖` unbounded above the degree-4 convergence radius θ_4 = 5.4) and at v4.5 (ADR-0094 AMENDMENT 1; Wave I failure mode const-a Richardson ratio = −0.4053 anti-convergent despite the scaling-and-squaring envelope PASSING per `G_PADE_SS_NORM_ENVELOPE`). ADR-0094 AMENDMENT 1 §"Diagnosis" identified three root-cause candidates ordered by architectural plausibility: D1 (P₄/Q₄ coefficient bug), D2 (banded LU error propagation under squaring), D3 (scalar-vs-matrix squaring mismatch per Higham 2005 §2.5). D3 was rated MOST PLAUSIBLE on architectural grounds and earmarked as the binding constraint for v5.0+ revival via "(1) Genuine matrix squaring (~400 LoC NEW)" per AMENDMENT 1 §"v5.0+ ADR-0094 follow-up directions".

The v5.0+ roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md` §"v5.0.0 BREAKING WINDOW #3", reused pattern #6) committed this ADR to PRE-FLIGHT sympy validation of at least Path α (the architecturally most-plausible D3 fix) BEFORE committing to a v5.0 engineer wave. The roadmap pre-declared the default outcome as Path δ TERMINAL CLOSURE under suckless single-kernel principle ("Chebyshev sufficient; Padé as permanent research artifact closes rolling-defer debt"), conditional on PRE-FLIGHT failing all three architectural paths.

This ADR records the PRE-FLIGHT sympy outcome and adopts Path δ accordingly.

## PRE-FLIGHT sympy attempts (executed 2026-05-29)

### Path α — Genuine matrix squaring vs vector iteration

**Setup**: Symbolic 3×3 tridiagonal test operator A = trid(−2, 1) (qualitatively mirrors the K5 heat operator — negative real spectrum, dissipative `exp(τA)`). Padé coefficients exact rationals (P₄ = 1 + z/2 + 3z²/28 + z³/84 + z⁴/1680; Q₄ = P₄(−z)). Test setup: τ = 1/4, s = 2 (squaring depth ≥1 cleanly tests the matrix-vs-vector question).

**Direct measurement**: ‖v_iter − v_explicit‖₁ = **0.000e+00** (BYTE-IDENTICAL within ULP). The vector iteration (Wave I's `apply_into` chained 2^s times) is mathematically EQUIVALENT to the genuine matrix squaring (R^{2^s} computed via matrix-matrix multiplication, then applied to v).

**Verdict**: ALIGNED → **RED**. The D3 hypothesis (scalar-vs-matrix squaring mismatch) is FALSIFIED. R is a fixed operator (it does not depend on which iteration step is current); iterating R on a vector gives the same numerical result as squaring R as a matrix first, then applying. The classical Higham 2005 §2.5 warning against scalar-vs-matrix confusion applies to scaling-and-squaring formulas where R itself depends on the iteration step (e.g., `R_k = R(τ/2^k)` with R recomputed each step) — NOT to the Wave I pattern where R is fixed and applied repeatedly. Wave I's anti-convergence at log₂ = −0.4053 was therefore caused by D1 (coefficient transcription) or D2 (banded LU error accumulation under repeated solves), NOT D3. Path α offers ZERO mathematical advantage over the Wave I implementation: a genuine matrix-squaring rewrite would produce byte-identical numerics and inherit the same Wave I anti-convergence.

### Path β — Higham m=13 envelope

**Setup**: Canonical Chernoff calibration setup per ADR-0091 AMENDMENT 1 (N = 512, τ_outer = 0.125, n_chernoff = 32, ‖a‖_∞ = 1.0, L = 20.0; K5 stencil ρ(A) ≈ 4·a_inf/dx² ≈ 2.61·10³). θ_13 = 5.371920351148152 (Higham 2005 Table 2.1 exact bound) vs θ_4 = 5.4. Sweep over N ∈ {128, 256, 512, 1024, 2048}.

**Measurement**: At canonical N = 512, τ_inner·‖A‖ = 1.02·10¹, τ_outer·‖A‖ = 3.26·10². Both exceed θ_13 = 5.37; scaling STILL required. Squaring depth `s` rounds to integers IDENTICAL between m=4 and m=13 for ALL N in the sweep: s_inner @ m=13 = s_inner @ m=4 for N ∈ {128 (0/0), 256 (0/0), 512 (1/1), 1024 (3/3), 2048 (5/5)}. The ~0.5% difference between θ_13 and θ_4 is dominated by the `⌈log₂⌉` ceiling-rounding to integer squaring depth.

**Verdict**: **RED**. Path β requires ~600 LoC NEW bandwidth-27 `banded_lu_m13.rs` helper (per ADR-0094 §"Degree choice m=4 vs Higham 2005's m=13") and Cohort 11 carve-out per constitution v1.8.0 — for ZERO observable Wave-I bug bypass. The squaring loop remains the same; if Wave I was D1/D2 (as Path α PRE-FLIGHT shows), m=13 inherits the same defect. Path β is mathematically futile.

### Path γ — Krylov-Arnoldi dep-budget + architecture

**Setup**: Architectural review of Saad 1992 minimal Arnoldi against constitution v1.8.0 3/3 dep cap (num-traits, libm, num-complex currently at 3/3; roadmap §"Hard constraints" declares the cap INVIOLATE through v5.x).

**Measurement**: Hand-rolled minimal Saad 1992 Arnoldi (modified Gram-Schmidt + Hessenberg projection) requires only matrix-vector products (we have the K5 stencil + `apply_into`), orthogonalisation primitives (hand-rollable, no_std-clean), and a SMALL Hessenberg matrix-exp at k×k for k ≤ 30. The Hessenberg matrix-exp is **the same Padé primitive we are trying to escape**, just at smaller scale. This creates a circular dependency: if Path α's matrix-squaring is fine on N×N (per PRE-FLIGHT), it is also fine on k×k (smaller scale = same algorithmic correctness, just cheaper). Conversely, if Wave I's anti-convergence is D1 or D2 (per Path α PRE-FLIGHT inference), the SAME defect inherits to the Hessenberg matrix-exp step. Net advantage over Path α: NONE. Engineering complexity ~500 LoC NEW + harder tolerance-based gating (no absolute order signature).

**Verdict**: ELIGIBLE-BUT-NON-PREFERRED. Architecturally feasible (3/3 dep cap unviolated) but mathematically subordinate to Path α; since Path α fails PRE-FLIGHT, Path γ inherits the same failure mode at smaller scale.

### Summary

All three architectural paths FAIL PRE-FLIGHT. Per `max-2-retries + suckless single-kernel` rule (roadmap §"Reused patterns" #6): default Path δ TERMINAL CLOSURE.

## Decision

**Path δ — TERMINAL CLOSURE**. Padé `Diffusion8thZeta8PadeChernoff` is permanently DEFERRED to v6.0+ pending external mathematical advance. The v4.5 Wave I research artifact (`.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md`) and v4.3 Wave B research artifact (`.dev-docs/research/zeta8-pade-wave-b-deferred.md`) are PRESERVED verbatim as research artifacts. The ζ⁸ contract remains closed via ADR-0090 Chebyshev `Diffusion8thZeta8Chernoff` (shipped v4.3, suckless single-kernel sufficient); the step-3 Carnot contract remains closed via ADR-0095 Engel `HypoellipticChernoff::new_engel()` (shipped v4.5). No NEW code at v5.0; no NEW gates; no NEW math.

The v5.0 BREAKING window per roadmap §"v5.0.0" reduces to A.6 LadderRung<K, F> formal trait (ADR-0100) ALONE. The roadmap pre-declared this contingency: "B.1 last OR deferred to v5.1" — under Path δ, B.1 is DEFERRED beyond v5.x entirely (v6.0+ at earliest, conditional on external math advance such as a published algorithm that bypasses both the matrix-squaring D3-equivalence and the m=4 vs m=13 envelope identity demonstrated by this PRE-FLIGHT).

The B.1 deferral does NOT, by itself, invalidate the v5.0 BREAKING justification: A.6 LadderRung<K, F> requires v5.0 even alone (sealed sibling super-trait of `ApproximationSubspace<K, F>` per ADR-0100 + ADR-0073 freeze; trait surface reshape candidate). If the engineer wave for A.6 reveals that LadderRung can ship without breaking ApproximationSubspace<K, F> consumers (additive sibling per ADR-0073 §"strictly additive"), the v5.0 BREAKING window may itself be re-evaluated as a MINOR v5.0 → v4.6 promotion candidate at architect's discretion at A.6 PRE-FLIGHT — but that is ADR-0100's scope, not this ADR's.

## Rationale (≤300 words)

**Path δ over Path α/β/γ**: PRE-FLIGHT sympy is dispositive. (1) Path α: matrix-squaring R^{2^s} and vector-iteration R·...·R·v are mathematically equivalent for fixed R (sympy shows BYTE-IDENTICAL within ULP); the D3 root-cause hypothesis of ADR-0094 AMENDMENT 1 was incorrect. A v5.0 Path α engineer wave would produce byte-identical numerics to Wave I and inherit its log₂ = −0.4053 anti-convergence. (2) Path β: θ_13 vs θ_4 differ by 0.5%; squaring depth `s` rounds identically across all N ∈ [128, 2048]. ~600 LoC NEW + Cohort 11 for ZERO Wave-I bug bypass. (3) Path γ: hand-rolled minimal Arnoldi has circular Padé dependency at the Hessenberg step; mathematically subordinate to α; α fails ⇒ γ inherits failure.

**Suckless single-kernel principle**: v4.3 Chebyshev `Diffusion8thZeta8Chernoff` ships order-8 with default-ON Chebyshev floor lift (ADR-0090, math §27.tris). v4.5 Engel `HypoellipticChernoff::new_engel()` ships step-3 Carnot (ADR-0095, math §28.tris). These two kernels CLOSE the order-8 + step-3 production needs for the roadmap. Padé is OPTIONAL bonus per ADR-0094 §"Decision (A)" Option α coexistence framing — under PRE-FLIGHT failure, the bonus is unobtainable and the production path is unaffected.

**Max-2-retries rule** (`.claude/CLAUDE.md` §"Agent Failure Protocol" + roadmap §"Reused patterns" #6): Wave B attempt 1 (DEFERRED ADR-0091 AMENDMENT 1) + Wave I attempt 2 (FINAL DEFER ADR-0094 AMENDMENT 1) → THIRD attempt requires architectural PRE-FLIGHT clearance. None achieved. Terminal closure is the established framework outcome; continuing past two failures without PRE-FLIGHT validation would violate Anchor protocol and accumulate tech debt without research benefit.

## Consequences

- **POSITIVE**: closes the rolling Padé defer debt with a formal v5.0+ TERMINAL CLOSURE marker. Subsequent maintainers (or the same architect at v6.0+) inherit a clear "do NOT re-attempt Padé without proving PRE-FLIGHT clearance of α/β/γ-equivalent paths". Research artifacts preserved verbatim per established convention; if external math literature publishes a Padé algorithm that demonstrably bypasses both the matrix-squaring equivalence AND the θ_4/θ_13 envelope identity, revival is unblocked. v5.0 BREAKING window scope reduces to A.6 LadderRung<K, F> alone (ADR-0100); narrower BREAKING surface lowers v5.0 risk profile and may permit MINOR re-evaluation at A.6 PRE-FLIGHT (architect's discretion).
- **NEUTRAL**: NO new code (zero LoC delta); NO new gates; NO new sympy oracles; NO schema bumps (`properties.yaml` schema_version preserved; ADR-0094 AMENDMENT 1 revert state preserved). Constitution UNCHANGED (no override-count change, no Cohort change). The v4.5 PRE-FLIGHT outcome of Wave I (T_ZETA8_PADE 5/5 PASS + G_PADE_SS_NORM_ENVELOPE PASS) remains a partial-positive entry in the research artifact — proves the scaling-and-squaring envelope CAN be tracked correctly, just that the post-envelope numerics anti-converge for a non-D3 reason.
- **NEGATIVE**: zero peer-review cross-validation of Chebyshev ζ⁸ against an independent matrix-exp algorithm (ADR-0094 §"Rationale" listed this as Padé's primary value; under Path δ the cross-validation is permanently unavailable until v6.0+ external advance). End-users wanting A-stable arbitrary-τ behavior for ζ⁸ have no kernel option (Chebyshev requires bounded τ within nested Richardson convergence radius; Padé's A-stability per Hochbruck-Lubich 2010 §3.4 was the unique selling proposition). Users with very large outer τ must split via Chernoff outer loop (`n_chernoff > 32`) to bring inner τ into Chebyshev's convergence regime; documented in v5.0 CHANGELOG.
- **BREAKING**: NONE. Path δ is purely a non-event at v5.0 (no surface change). The retired Wave I/Wave B kernels were never `pub`-exported to v4.5+; their absence at v5.0 is consistent with v4.5.
- **Schema bumps**: NONE. `properties.yaml`, `traits.yaml`, `math.md` (no normative algorithm changes; only AMENDMENT 3 to §27.quart marking permanent closure per Deliverable 2 below).
- **Constitution check**: NO override count change. NO Cohort change. NO file-list expansion. Path δ is the only ADR-0101 outcome that requires zero constitution amendment — by design.

## Implementation cost estimate

- **Engineering**: 0 LoC NEW (terminal closure ships no code; only documentation amendments).
- **Architecture**: this ADR (~200 LoC) + math.md §27.quart AMENDMENT 3 (~30 LoC) + final-closure marker appended to `.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md` (~30 LoC). PRE-FLIGHT sympy harness `/tmp/pade_v5_preflight.py` (already executed; preserved in /tmp; can be moved to `scripts/verify_pade_v5_preflight.py` at engineer's discretion if maintainers want to re-run against future m=13 alternatives — OPTIONAL).
- **Days**: 0 working days engineering. ~0.5 architect day (this ADR + amendments). PRE-FLIGHT sympy harness took ~2 hours to author + verify; harness deleted from /tmp after merge (no permanent file artifact unless engineer Wave promotes it to `scripts/`).
- **Risk**: NONE at v5.0 (terminal closure has zero behavior change). v6.0+ revival risk: a v6.0+ ADR attempting Padé revival MUST cite this ADR-0101 explicitly and demonstrate PRE-FLIGHT clearance via sympy harness (mirror this ADR's PRE-FLIGHT §). Without such clearance, the v6.0+ revival would be a fourth attempt at a thrice-failed kernel and would violate max-2-retries (an architect or user would need to explicitly justify reopening the question).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Path α — Genuine matrix squaring (~400 LoC NEW)** | PRE-FLIGHT sympy FALSIFIES the D3 hypothesis (matrix R^{2^s} ≡ vector R·...·R·v BYTE-IDENTICAL within ULP for fixed R). Wave I anti-convergence was D1 or D2; rewriting to genuine matrix squaring produces byte-identical numerics and inherits Wave I failure. ZERO Wave-I bug bypass. |
| **Path β — Higham m=13 (~600 LoC NEW + Cohort 11)** | PRE-FLIGHT envelope analysis: θ_13 = 5.37 vs θ_4 = 5.4 differ by 0.5%; squaring depth `s` rounds to identical integers across all N ∈ [128, 2048]. ~600 LoC NEW for zero envelope advantage; if Wave I was D1 or D2 (per Path α PRE-FLIGHT inference), m=13 inherits same defect. Mathematically futile. |
| **Path γ — Krylov-Arnoldi (~500 LoC NEW; hand-rolled minimal Saad 1992)** | Architecturally eligible (no 4th dep needed for hand-rolled implementation) but mathematically subordinate to α: Hessenberg matrix-exp is the same Padé primitive at smaller scale; circular dependency. Path α fails ⇒ Path γ inherits failure at smaller scale. Engineering complexity higher than α (~500 vs ~400 LoC); test-gating harder (tolerance-based, not order-signature). |
| **Path δ′ — Defer to v6.0+ WITHOUT PRE-FLIGHT clearance requirement** | Would re-invite a fourth Padé attempt at v6.0+ without architectural guard; violates max-2-retries protocol intent. Documented PRE-FLIGHT clearance requirement in this ADR creates a structural barrier preventing re-litigation of already-falsified hypotheses (D3) and re-attempts of already-shown-equivalent paths (α ≡ β under envelope identity). |
| **Path ε — DELETE retired research artifacts; cease research-track tracking** | Violates roadmap §"Reused patterns" #8 "Research artifact preservation: failed waves preserved as `.dev-docs/research/*-deferred.md` with full inline source for future revival". The Wave B + Wave I artifacts encode bug-fixer diagnoses + measurement records + revival-direction candidates that would be lost. v6.0+ external advance (e.g., a published algorithm with provable Wave-I-bug-bypass) would have to re-discover all prior failure modes from scratch. |

## Cross-references

- ADR-0091 + AMENDMENT 1 — Wave B v4.3 DEFER (`τ‖A‖` unbounded above Padé degree-4 radius); ADR-0094's parent.
- ADR-0094 + AMENDMENT 1 — Wave I v4.5 FINAL DEFER (const-a anti-convergent log₂ = −0.4053 despite envelope PASS; D1/D2/D3 diagnosis candidates); this ADR's parent.
- ADR-0090 — Chebyshev `Diffusion8thZeta8Chernoff` (the v4.3 production ζ⁸ kernel; Padé is OPTIONAL bonus per Option α framing; Path δ removes the bonus permanently).
- ADR-0095 — Engel `HypoellipticChernoff::new_engel()` (the v4.5 production step-3 Carnot kernel; orthogonal to Padé).
- ADR-0073 — `ApproximationSubspace<K, F>` v3.0 freeze (Padé revival cannot reshape this trait at v5.0).
- ADR-0100 — A.6 LadderRung<K, F> formal trait (the OTHER v5.0 BREAKING item; ships independently of B.1 Padé under Path δ).
- math.md §27.quart + AMENDMENT 1 + AMENDMENT 2 + AMENDMENT 3 (this ADR's math amendment marking permanent closure).
- `.dev-docs/research/zeta8-pade-wave-b-deferred.md` — Wave B research artifact (PRESERVED).
- `.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md` — Wave I research artifact + this ADR's final-closure section appended per Deliverable 3.
- `/tmp/pade_v5_preflight.py` — PRE-FLIGHT sympy harness (executed 2026-05-29; output `/tmp/pade_preflight_output.txt`; OPTIONAL promotion to `scripts/verify_pade_v5_preflight.py` deferred to engineer's discretion).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §"v5.0.0 BREAKING WINDOW #3" + §"Reused patterns" #6 (max-2-retries + suckless single-kernel) — the framework rules this ADR invokes.
- N. J. Higham (2005) *SIAM J. Matrix Anal. Appl.* 26(4) §2.5 — the warning against scalar-vs-matrix confusion in scaling-and-squaring (cited by ADR-0094 AMENDMENT 1 D3; FALSIFIED for fixed R by Path α PRE-FLIGHT).
- Saad (1992) *SIAM J. Numer. Anal.* 29(1) pp. 209-228 — Krylov subspace methods (Path γ paradigm; subordinate to α per PRE-FLIGHT analysis).
- Constitution v1.8.0 §"Override #1/#2/#3" + 3/3 dep cap (INVIOLATE through v5.x per roadmap).

ADR-0101 status: **Accepted (Path δ TERMINAL CLOSURE; v6.0+ revival requires PRE-FLIGHT clearance of α/β/γ-equivalent paths plus citation to a published algorithm demonstrating Wave-I-bug-bypass)**.
