# ADR-0149 — F3 KilledDirichlet variable-coefficient sharp order: GO (order-2 is a THEOREM for variable a(x), not an empirical cap)

**Status:** ACCEPTED (PRE-FLIGHT / proof obligation — RESEARCH ONLY, no Rust change) · **Date:** 2026-06-08 · **Branch:** `feat/v8.2.0-math`
**Theme:** v8.2.0 — Wave-2 B-6 math proof obligation
**Gate:** `T_KILLED_GEN_CN` (sympy; recommend ADD sub-check E) · `G_HARD_WALL_ORDER2` (recommend re-derive probe to generic jet)
**Parent:** ADR-0135 (Amendment 2, GO) · math.md §44.ter · honesty precedent ADR-0136 Amendment 1

## Context

ADR-0135 Am. 2 / §44.ter shipped `KilledDirichletChernoff<F>` (`killed_dirichlet.rs`): an order-2 hard absorbing wall via the Crank–Nicolson Cayley map `(I−τ/2 L^R)⁻¹(I+τ/2 L^R)` of the killed Dirichlet generator `L^R = ∂_x(a(x)∂_x) + b(x)∂_x`. The shipped oracle `T_KILLED_GEN_CN` (CHECK A) verifies order-2 with a **CONSTANT** `a = 1/2`. B-6 asks whether the τ²-cancellation that yields order-2 SURVIVES VARIABLE `a(x)` (and drift `b(x)`) — **as a theorem**, closing the "variable-coefficient sharp order not proven — empirically gated only" gap that §44.7/§44.4 honestly flag for the obstacle evolver. Per the ADR-0136 Am. 1 honesty precedent, any verification MUST use a generic non-origin-symmetric high-degree jet (low-degree/origin-symmetric probes over-report order because `L^k` lowers polynomial degree).

## Decision

**RULING: GO — order-2 is a THEOREM for variable coefficients, not an empirical cap.** The order-2 of the Cayley / (1,1)-Padé map is the PURE MATRIX IDENTITY `(I−τ/2 G)⁻¹(I+τ/2 G) = e^{τG} + (1/12)τ³G³ + O(τ⁴)` (★), which holds for ANY fixed matrix `G`; the derivation never uses constant-coefficient structure. Variable `a(x)`/`b(x)` only change the NUMERIC entries of the tridiagonal `L^R` (off-diagonals → `a_{k±1/2}/dx²`, drift → `b_k/2dx`), not the algebraic form of (★). Hence `τ¹ = τ² = 0` with leading remainder `τ³` independently of coefficient variation. Literature concurs: CN for `−∇·(κ∇·)` with Dirichlet BC is unconditionally `O(h²+τ²)` for variable `κ` (Springer ACDM 2017; arXiv:2011.05178 Strang–CN equivalence) — the only known caveats are Rannacher-startup smoothing for NON-SMOOTH data and corner compatibility, both global-rate/regularity issues, NOT coefficient-variation order drops.

The sympy oracle `scripts/killed_dirichlet_varcoef_kit.py` (`PREFLIGHT-KILLED-VARCOEF`, exit 0) PROVES this on honest finite variable-coefficient matrix models, six sub-checks A–F.

**Recommendation (theorem upgrade):**
1. Upgrade §44.ter.2 from coefficient-agnostic-BY-ASSUMPTION (it says "bounded `a`" but only the oracle's constant `a` was exercised) to an EXPLICIT variable-coefficient NORMATIVE theorem, citing the symbolic CHECK C witness and the variable-`a` jet CHECK E.
2. ADD a variable-coefficient sub-check **(E)** to `T_KILLED_GEN_CN` (generic degree-≥6 non-origin-symmetric jet on a variable-`a` killed generator), per the ADR-0136 anti-over-report discipline.
3. Re-derive the `G_HARD_WALL_ORDER2` empirical self-convergence probe to use a generic non-origin-symmetric initial datum (not a low-degree/symmetric one) so the measured slope is honest.
4. **No Rust change.** `killed_dirichlet.rs` already takes `a`/`b` as closures and assembles the variable-coefficient stencil correctly (`assemble_cn_row` uses `(self.a)(x)`, `(self.b)(x)` per node); the kernel was correct — only the math/oracle claim was narrower than the implementation.

## Findings (sympy verdict, `scripts/killed_dirichlet_varcoef_kit.py`, exit 0)

- **CHECK A (variable-`a` matrix order):** strictly varying half-node `a ∈ {3/7, 5/4, 2/3, 9/5}` + varying drift `b` → `τ¹ = τ² = 0`, `τ³ ≠ 0`. Order-2 survives variable coefficients.
- **CHECK B (hard BC exact):** Dirichlet boundary rows = identity ∀τ under variable `a` — wall exact, width 0, coefficient-independent.
- **CHECK C (DECISIVE — symbolic identity):** fully symbolic distinct-entry tridiagonal `G` (`p1,p2,p3,l2,l3,u1,u2`) gives `τ² ≡ 0` and `τ³ residual ≡ G³/12` SYMBOLICALLY ⟹ order-2 is a MATRIX IDENTITY, not a constant-`a` accident. **This is the theorem.**
- **CHECK D (no blow-up):** refine variable-`a` grid (3→4 interior) → τ³ constant bounded (`≈1.64` → `≈4.81`, O(1), no `1/dx` power) — no boundary layer from variable coefficients.
- **CHECK E (honesty, ADR-0136):** generic degree-6 NON-origin-symmetric nodal jet → `τ¹ = τ² = 0`, genuinely nonzero τ³ residual ⟹ order EXACTLY 2 on real data, not over-reported by a degenerate probe.
- **CHECK F (non-self-adjoint):** drift-dominated NON-symmetric variable-`a` `L^R` still has `τ² = 0` — order needs no self-adjointness (only A-stability does).

**Literature:** Hochbruck–Lubich (Acta Numerica 19, 2010, §3.4: (1,1)-Padé A-stable order-2 unconditional); Springer ACDM (2017) CN for variable-coefficient diffusion unconditionally `O(h²+τ²)`; arXiv:2011.05178 (Strang–CN order-2 for variable-coefficient diffusion–reaction with Dirichlet BC); Strang (SINUM 5:3, 1968). Caveat (not a coefficient issue): non-smooth-data global rate needs Rannacher backward-Euler startup; corner compatibility for the full-rate constant.

## Consequences

Variable-coefficient hard-wall order-2 is now a PROVEN theorem (was implicitly assumed). The shipped kernel needs no change. Honest scope preserved: order is exactly 2 (genuine τ³ remainder, no overclaim); the free-boundary American/obstacle `O(√τ)` cap (§44.ter.5) is unchanged. The variable-coefficient sub-check (E) hardens `T_KILLED_GEN_CN` against the ADR-0136 over-report failure mode. Both this GO and the (counterfactual) NO-GO would have been honest outcomes; the symbolic identity (CHECK C) makes GO unambiguous. **Kernel ships either way.**
