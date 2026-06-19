# ADR-0131 — Order-4 ζ⁴ drift-reaction sibling via Path β (resolves §27.7 OPEN)

**Status:** PROPOSED (PRE-FLIGHT GREEN — MATH RESOLUTION) · **Date:** 2026-06-06 · **Branch:** `feat/v7.0.0-debt-closure`
**Item:** v7.0.0 backlog #18 (§27.7 OPEN math problem) · **Gate:** `G_DR_ZETA4_TRUTHFUL_ORDER`
**Numbering:** backlog assigned 0127 (consumed by complex-lambda-resolvent); reallocated to **0131**.

## Context

§27.7 (v3.0 framing) called the ζ⁴ τ²-correction WITH drift an OPEN problem: "the τ²-coefficient with drift introduces additional monomials beyond the diffusion case (Galkin-Remizov §3.3)". That obstruction belongs to the **RETIRED** 6-monomial `P_2[A]` correction-operator framework (citation-error, retired at v4.1 per §27 AMENDMENT).

## Decision

The §27.7 OPEN is **RESOLVED, not deferred** — the obstruction was a stale-premise artifact. The NORMATIVE ζ⁴ algorithm pivoted to **Path β = Richardson-over-symmetric-K5** (§27 AMENDMENT 1/2): no correction monomials at all. Richardson is BLIND to which operator the symmetric order-2 base approximates — it needs only (i) order-2 consistency and (ii) time-symmetry (palindromic), so the global error is odd-in-τ and the leading τ³ term cancels. For the drift-reaction generator `L = a∂ₓₓ + b∂ₓ + c`, the palindromic Strang base `S(τ)=e^{τR/2}∘D(τ)∘e^{τR/2}` (R=b∂ₓ+c, D=a∂ₓₓ) IS time-symmetric and order-2, so `Fβ(τ)=(4/3)S(τ/2)²−(1/3)S(τ)` is order-4. PRE-FLIGHT `scripts/verify_drift_reaction_zeta4.py` 3/3 PASS (non-commuting 3×3 model, `[D,R]≠0`): symmetric Strang has τ²-error = 0, τ³ ≠ 0; Richardson gives τ³=τ⁴=0, τ⁵≠0 (order-4); the OLD "extra drift monomials" = exactly the `(1/2)[R,D]` BCH commutator that asymmetry leaves at τ² and Path β annihilates for free. Ship additive `DriftReactionZeta4Chernoff` = Richardson-over-(symmetric drift-reaction Strang base). Gate `G_DR_ZETA4_TRUTHFUL_ORDER`: pair-slope ≤ −3.5.

## Consequences

Additive (new constructor, no mutation of existing `order()` contracts per the v7.0 additive-order policy). Closes a long-standing OPEN math note by re-grounding it in the post-pivot framework. math.md §27.7 amended from "OPEN" → "RESOLVED via Path β". Caller regularity: `f ∈ D(L⁴)`, `a,b,c ∈ C⁴_b` (mirrors Path β diffusion). NORMATIVE math creation — engineer must implement against the symmetric-Strang base, NOT a 4-term Taylor (which overflows at finite dx per §27 AMENDMENT 2).
