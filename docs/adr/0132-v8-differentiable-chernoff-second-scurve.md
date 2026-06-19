# ADR-0132 — v8.0.0 umbrella: Differentiable Chernoff Semigroup (second S-curve)

**Status:** ACCEPTED · **Date:** 2026-06-06 · **Shipped:** 2026-06-08 · **Branch:** `feat/v8.0.0-planning`
**Theme:** v8.0.0 strategic umbrella · **Child ADRs:** 0133, 0134, 0135, 0136

## Context

The kernel-accuracy S-curve — ζ-ladder (ζ²→ζ⁸), manifolds, hypoelliptic, resolvent, complex, matrix, soft-killing — is saturated as of v7.0.0 (ADRs 0117–0131). Catastrophe-model signals confirm saturation: critical-slowdown was observed in ζ⁶/ζ⁸ pre-v7 (honest-defer → multi-amendment recovery), and operator-level Padé in the time domain hit a hard physical ceiling (‖A‖ ∝ 1/dx² → ∞) twice over ADRs 0094/0101/0125, each time requiring a coordinate-change escape. The next S-curve axis is unused **product structure**: the Chernoff product `(F(τ))ⁿ` is differentiated, jump-compressed, boundary-regularized, and recursively deepened in ADRs 0133–0136. This ADR names the axis, ratifies the four child proposals, and records two permanent anti-directions.

## Decision

Open v8.0.0 on the **differentiability and amortization** axis, treating the Chernoff product structure `(F(τ))ⁿ` as the primary unused valuable resource (ВПР). Four child ADRs (0133–0136) define the concrete directions: forward-mode dual-number AD (F1), resolvent time-jump amortization (F2), order-2 hard absorbing wall via resolvent-regularized projector (F3), and step-k Carnot closure via recursive palindrome-of-palindromes (F4). Two permanent non-goals are recorded: (X1) operator-level Padé in the time domain (twice-failed, physically infeasible); (X2) deeper ζ¹⁰/ζ¹² laddering (spatial floor already dominates; pure compromise); (X3) becoming a general-purpose PDE solver framework (dilutes the Chernoff moat).

## Consequences

v8.0.0 may or may not be a BREAKING window depending on whether F1 `Dual<F>: SemiflowFloat` integration requires trait-bound changes. Child ADRs carry individual gate names (`G_DUAL_AD_GRADIENT`, `G_DUAL_ZERO_ALLOC`, `G_RESOLVENT_JUMP_ORDER`, `G_HARD_WALL_ORDER2`, `G_CARNOT_STEP4`). All four directions are additive in the library surface; no v7.0 kernel semantics change. The anti-directions become explicit rejection criteria for future amendment proposals.
