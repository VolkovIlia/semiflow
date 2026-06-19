# ADR-0161 — (CONTINGENCY) Honest NO-GO for the coupled TT evolver, IF the rank bound fails in practice

**Status:** PROPOSED (CONTINGENCY — ACCEPTED only if `g_tt_coupled` genuinely fails after a real implementation attempt) · **Date:** 2026-06-10 · **Branch:** `feat/v9.1.0-genuine-scurve`
**Theme:** v9.1.0 — Shift C honest-boundary safety net (mirrors §50.7 / ADR-0158 negative-result precedent)
**Parent:** ADR-0159 (Amendment 1) · **Math:** §52.9 (refutation criterion)

## Context

The v9.1.0 design (ADR-0159 Amendment 1, design spec §3) commits to `CoupledTtChernoff` as **outcome (i): genuinely achievable**, on the strength of feasibility probes showing the coupled evolver grows rank from a rank-1 IC but stays BOUNDED (O(1) local, ⌊d/2⌋ dense) for the correlated-Gaussian class. The probes are dense-tensor / analytic-precision evidence, NOT the shipped no_std evolver with deterministic Jacobi-SVD rounding. There is a residual implementation risk: the pair-bond contraction could inflate rank faster than the in-tree TT-rounding removes it (e.g. accumulated SVD truncation error, conditioning of the Jacobi SVD on the coupled cores, or the explicit `(I+τL)ⁿ` step accumulating more rank than the exact semigroup at the chosen ε), making practical cost super-polynomial in d. That is an **outcome (iii)** signal for the *implemented method* (distinct from the proven-low-rank *analytic* result).

## Decision (contingent)

This ADR is the pre-committed honest record IF, after a genuine implementation attempt (design spec Phases 4–6), the `g_tt_coupled` gate FAILS its rank-polynomiality or accuracy condition and the failure is shown to be intrinsic (not a tunable bug) — i.e. the coupled evolver cannot hold sub-polynomial rank in practice for the correlated-Gaussian class.

**Mandated response on a genuine failure:**
1. **Do NOT weaken `g_tt_coupled` to pass.** The gate is the arbiter; the probe is only evidence.
2. Record the obstruction with the SAME rigor as §50.7: the measured rank trajectory vs d, the ε-vs-rank trade-off, the d at which it breaks, and the mathematical reason (analogous to the §50.7 `O(m^d)` spatial-merge argument).
3. **Downgrade the §52 claim honestly:** retain `CoupledTtChernoff` for the d where it provably works (e.g. local coupling only, or d ≤ d_max), label the high-d general-coupling claim as research-track, and update §52 STATUS to the achieved boundary — NOT to a second overclaim.
4. Redirect Shift C effort to the residual research-track direction (ADR-0158 path-space RQMC for the regime where TT rank is uncapped).

## Consequences

An honest "this part is achievable only up to boundary X, here is the real limit" is a SUCCESS in this project's negative-result culture (precedent: §50.7 INTRINSIC LIMIT, ADR-0158, the G_zeta4 escalation). A second overclaim is the only failure mode. If the gate PASSES (the probe-predicted outcome (i)), this ADR is WITHDRAWN at v9.1.0 tag and the §52.9 genuine-escape claim stands as validated. This ADR exists so the engineer has a pre-authorized honest exit and never feels pressure to game the gate.
