# ADR-0126 — Order-2 Killing via SOFT killing-RATE symmetric Strang `e^{−τκ/2}·C·e^{−τκ/2}` (`Killing2ndChernoff`); the hard absorbing-wall §21 remains irreducibly order-1

- **Status**: Accepted (PRE-FLIGHT GO with scope refinement; engineer wave authorised — additive `Killing2ndChernoff<C, K, F>` driven by a killing-RATE field, NOT a region indicator)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Backlog item**: v7.0.0 freeze item #26 (predicted ADR-0133; actual sequential allocation = **0126**).
- **Depends on**: ADR-0068 (`KillingChernoff<C, R>` v2.6 — the order-1 hard-indicator wrapper this ADR does NOT modify), constitution v2.0.0 (3/3 dep cap INVIOLATE — adds ZERO deps).
- **Mathematical foundation**: `scripts/verify_killing_order2_preflight.py` (executed 2026-06-05). Both checks PASS — see §PRE-FLIGHT.
- **Acceptance gates added**: `G_KILLING_ORDER2` (slow-tests; self-convergence `slope ≤ −1.95` on the soft-killed semigroup `e^{t(L−κ)}`).

## Context

Butko 2018 §5 conjectures order-2 killing via the Strang-style symmetrization `𝟙_R^{1/2}·C(τ)·𝟙_R^{1/2}` (math §21.6). v2.6 ships only the order-1 hard-indicator `KillingChernoff`. Before authorising an order-2 wave, the architect had to settle whether the conjecture, taken literally over the **hard indicator** `𝟙_R`, can actually beat order-1.

## Decision

**Scope refinement (load-bearing, honest).** Over the *hard* absorbing indicator the conjecture is VACUOUS: `𝟙_R` is idempotent (`0^½=0`, `1^½=1`), so `𝟙_R^{1/2}=𝟙_R` and the "symmetric split" degenerates to `𝟙_R·C·𝟙_R` — which includes the **pre-multiply over-killing form that math §21.2 explicitly REJECTS**. The boundary jump makes `[L,𝟙_R]` an irreducible O(τ) term; the hard wall is order-1, period.

The conjecture has teeth only for **soft killing** — a smooth bounded killing **rate** `κ(x) ≥ 0` (a reaction term `−κu`; Feynman-Kac with continuous weight `e^{−∫κ}`). There `R^{1/2}(τ) := e^{−τκ/2}` is a genuine non-idempotent factor, `[L,κ]` is bounded (no boundary jump), and the palindrome `e^{−τκ/2}·C(τ)·e^{−τκ/2}` reaches order-2. Ship this as a **new** additive `Killing2ndChernoff<C, K, F>` parameterised by a killing-rate field `κ(x)`, NOT by a `KillingRegion`. The v2.6 hard-indicator `KillingChernoff` is UNCHANGED (stays order-1, correct for absorbing walls). ZERO new deps.

## PRE-FLIGHT sympy (executed 2026-06-05; `scripts/verify_killing_order2_preflight.py`)

- **CHECK A (negative, expected)**: `𝟙_R^{1/2} = 𝟙_R` symbolically (3-node `diag(1,1,0)`); the symmetric split is the rejected pre+post over-killing form ⟹ hard wall stays order-1.
- **CHECK B (positive, GO)**: `L = ½·discrete ∂_xx` (3×3), `κ = diag(0.3,0.7,0.2)`, **`[L,κ] ≠ 0`** (faithful non-commuting model). `S(τ) = e^{−τκ/2}·(e^{τL}+O(τ³))·e^{−τκ/2}` minus `e^{τ(L−κ)}` has τ¹ and τ² coefficient matrices **exactly zero**, τ³ nonzero ⟹ **order-2**.

VERDICT: **GO** for soft killing-rate order-2; hard absorbing-wall order-1 is mathematically irreducible (scoped honestly).

## Consequences

- **Engineer wave** builds `Killing2ndChernoff<C, K, F>`: a killing-RATE trait `KillingRate<F>` supplying `κ(x) ≥ 0` per node; `apply_into` = half-step `e^{−τκ/2}` (pointwise scalar exp) → inner `C.apply_into(τ)` → half-step `e^{−τκ/2}`. Gate `G_KILLING_ORDER2` self-convergence `slope ≤ −1.95` vs `e^{t(L−κ)}`.
- **Honest scope note (math §21 amendment)**: order-2 is for the *rate* formulation; absorbing Dirichlet walls (`u|_{∂R}=0`) remain the order-1 §21 `KillingChernoff`. The two are DIFFERENT operators (soft continuous killing vs hard absorbing boundary), not faster/slower versions of one — the ADR + math note MUST prevent users mistaking `Killing2ndChernoff` for a higher-order absorbing-wall solver.
- **`κ ≥ 0` guard**: constructor validates non-negativity (a negative rate is mass creation, not killing) — fail-closed `DomainViolation`.
