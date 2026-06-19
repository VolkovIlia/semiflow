# ADR-0160 — Integrity-gate hardening: RELEASE_BLOCKING gates must ASSERT (no print-only always-green)

**Status:** ACCEPTED (v9.1.0 PLAN) · **Date:** 2026-06-10 · **Branch:** `feat/v9.1.0-genuine-scurve`
**Theme:** v9.1.0 — integrity fixes corroborated by two adversarial audits (independent of Shift B / Shift C outcomes)
**Gates:** hardens `g_gridless` / `g_gridless_dim_scaling`; adds `G_TT_STRANG_IDENTITY`; corrects `T_GRIDLESS` claim
**Math:** §50.6 (T_GRIDLESS wording), §52.3 (bit-identity), §52.5 (Regime-L hard assert)
**Parent:** ADR-0154 · **Source:** `.dev-docs/reports/v9-third-scurve-audit-reviewer.md` (findings #4, #6), `v9-math-fidelity-audit.md` (D5)

## Context

Both adversarial audits flagged three integrity defects in the v9.0.0 §50/§52 gate layer, all transparency/fidelity (no fraud, no fabricated numbers), all cheap to fix:

1. **Print-only RELEASE_BLOCKING gates (audit finding #4, LOW-MEDIUM).** `g_gridless` and `g_gridless_dim_scaling` are spec'd RELEASE_BLOCKING but implemented as `"documentation gate, always passes"` / `"printed honestly, not asserted"`. A reader scanning CI sees all-green even on a refuted hypothesis (the §50 variance NO-GO at 1.417× and the d≥4 accuracy collapse are invisible at the CI level — only the printed body reveals them).
2. **§52.3 NORMATIVE bit-identity has no test (audit finding #6, LOW).** §52.3 requires `TtChernoff::new` (rank-1) to be bit-identical to `Strang2D`/`Strang3D`; grep confirms no such test exists. A NORMATIVE clause is unproven.
3. **False "reuses T_ADJOINT_FP_TIGHTNESS machinery" claim (audit D5, LOW).** `scripts/gridless_kit.py` re-derives a Taylor stencil inline and imports nothing; the "reuses … machinery" wording is a code-level overstatement. The oracle proves generator consistency to O(τ) + a coefficient-sum check, NOT a non-trivial adjoint-identity proof.

## Decision

1. **A RELEASE_BLOCKING gate MUST `assert!`/`panic!` on its invariant (NORMATIVE §50.6).** A "documentation-only always-passes RELEASE_BLOCKING gate" is FORBIDDEN. Where a refuted hypothesis is *intentionally* recorded green (e.g. the §50 INTRINSIC LIMIT), it MUST be a SEPARATE clearly-named `#[ignore]` documentation record, and the RELEASE_BLOCKING gate of the same name MUST hard-assert the actual invariant that still holds (e.g. d=2 accuracy < 5e-3, `Var_det > 0`), so a regression in the validated envelope FAILS the test process. The anti-gaming asserts already present (e.g. `Var_det > 0`) are kept; the *ratio/slope* metrics that are honestly NO-GO are moved to the documentation record, never spun as a green RELEASE_BLOCKING pass.
2. **Add `G_TT_STRANG_IDENTITY`** (RELEASE_BLOCKING, `tests/g_tt_strang_identity.rs`): rank-1 `TtState` / no-coupling ⇒ 0-ULP identical to `Strang2D`/`Strang3D` (ADR-0018 bit-equality culture). Enforces the §52.3 NORMATIVE clause.
3. **Correct the `T_GRIDLESS` claim** in §50.6 and the `gridless_kit.py` docstring to state exactly what the oracle proves (generator consistency to O(τ) + coefficient-sum sanity; inline stencil re-derivation, NO code reuse of `T_ADJOINT_FP_TIGHTNESS`).

## Consequences

Test-only + docstring/contract-wording changes; no production kernel semantics change. CI now surfaces NO-GO as a failing process, not a green checkmark with a buried print. The §50 negative-result *science* is UNCHANGED and remains exemplary — only its CI surfacing is corrected (the refuted ratio becomes a documented `#[ignore]` record; the validated-envelope invariant becomes a hard assert). Independent of Shift B / Shift C implementation outcomes — Phase 0 of the v9.1.0 plan, lowest risk, done first. Verification of the hardened gates: temporary fault-injection must make them FAIL, then revert.
