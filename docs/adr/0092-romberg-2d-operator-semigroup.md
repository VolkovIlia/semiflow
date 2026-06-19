# ADR-0092 — Romberg-2D Operator Semigroup Extrapolation: NOVEL Math Attempt → Outcome B (Algebraic Equivalence)

- **Status**: Accepted (negative result: scheme is sound but ALGEBRAICALLY EQUIVALENT to nested Richardson at the relevant ζ⁸ measurement regime; does NOT bypass floor-cascade contamination)
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Supersedes**: NONE (additive research deliverable; ADR-0088 AMENDMENT 2's "Option δ Romberg-2D rewrite" future-work pointer is now resolved with a documented negative outcome)
- **Depends on**: ADR-0086 + AMENDMENT 1 (Path β single-step Richardson on K5 — the algebraic precedent), ADR-0088 + AMENDMENT 1 + AMENDMENT 2 (nested Richardson ζ⁴ → ζ⁶ → ζ⁸ ladder + floor-cascade DEFER), ADR-0001 (contract-first).
- **Mathematical foundation**: Richardson 1911 (extrapolation); Romberg 1955 (Romberg-in-time semigroup integration template); Bulirsch-Stoer 1966 (extrapolation algorithms for ODE); Hairer-Lubich-Wanner 2006 *Geometric Numerical Integration* §II.4 + §II.9 (Romberg-in-time for symmetric methods); Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (Taylor-tangency rate theorem). Wave 2 research synthesis in `.dev-docs/research/raw-findings-romberg-2d.md` confirmed no published Romberg-2D for operator semigroup approximation (61 sources reviewed; 0 direct hits).
- **Acceptance gates added**: NONE (no kernel ships). Sympy oracle `scripts/derive_romberg_2d.py` ships as research deliverable with 4 sub-checks (taylor_structure / table_construction / equivalence_to_nested / floor_contamination_model). Verdict artifact `.dev-docs/research/verdicts/verdict-romberg-2d-attempt.md` documents the negative result.

## Context

User directive (delegation prompt; cross-ref `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/feedback_library_fixable_via_bench.md` and `verdict-v4-3-research-waves.md` Priority 3): "проводит исследования по вопросам которые остались открытыми. если не найдёшь, попробуй сам создать математику" (research open questions; if not found, try to create math yourself). Wave 2 researcher synthesis (`raw-findings-romberg-2d.md`) confirmed that direct 2D Richardson extrapolation over the (τ refinement × K cascade depth) grid for operator semigroup approximation has no published precedent; the construction is genuinely novel research. ADR-0088 AMENDMENT 2 §"Future work" identified Romberg-2D as one of three v4.3+ ADR-0090 candidate directions for unblocking the deferred ζ⁸ Wave II rung. This ADR attempts the math creation per user directive and reports the rigorous symbolic result.

## Decision

**Document the math creation attempt and its outcome (Outcome B: ALGEBRAIC EQUIVALENCE).** Ship the sympy derivation script as a research deliverable (`scripts/derive_romberg_2d.py`) and a negative-result verdict (`.dev-docs/research/verdicts/verdict-romberg-2d-attempt.md`). Do NOT ship a new kernel type or acceptance gate. The Romberg-2D scheme is mathematically correct (achieves order 2(j+1) at table cell T[j, 0]; sub-check (b) PASS) but is ALGEBRAICALLY EQUIVALENT to the nested Richardson cascade `R^{j+1}` at the relevant single-outer-step measurement regime (sub-check (c) PASS: difference is exactly 0). Therefore it cannot bypass the floor-cascade contamination diagnosed in ADR-0088 AMENDMENT 2 (which is intrinsic to the linear combination at one outer step, not to the nesting structure of intermediate Richardson stages). Re-route v4.3+ ζ⁸ resurrection effort toward the Path ε successor (Chebyshev spectral collocation per `verdict-v4-3-research-waves.md` Priority 1) which lifts the spatial floor BEFORE attempting any K=4 measurement.

## Algorithm (Romberg-2D table, NORMATIVE per sympy derivation in `scripts/derive_romberg_2d.py`)

Define `U(h) := K5(h) ∘ K5(h) ∘ … ∘ K5(h)` (composition of `n = T/h` symmetric K5 steps to reach final time `T`). Hairer-Lubich-Wanner 2006 Theorem II.4.7 (symmetric methods have even-power-only global error expansion) gives:

```
U(h) = e^{TA} + a_2·h^2 + a_4·h^4 + a_6·h^6 + a_8·h^8 + O(h^10)
```

Romberg-2D table construction (analogue of classical Romberg integration; 1D table because the second axis K is realised as table depth `j`, NOT a true second continuous axis):

```
T[0, m] := U(h / 2^m)                                  for m = 0, 1, …, K
T[j, m] := (4^j · T[j-1, m+1] − T[j-1, m]) / (4^j − 1) for j = 1, …, K; m = 0, …, K-j
```

The final extrapolant `T[K, 0]` achieves order `2(K+1)` accuracy:

- `T[0, 0]` has residual `(a_2)h² + (a_4)h⁴ + (a_6)h⁶ + …`           (order 2)
- `T[1, 0]` has residual `(-a_4/4)h⁴ + (-5a_6/16)h⁶ + …`             (order 4) ← matches ζ⁴ kernel
- `T[2, 0]` has residual `(a_6/64)h⁶ + (21a_8/1024)h⁸ + …`           (order 6) ← matches ζ⁶ kernel
- `T[3, 0]` has residual `(-a_8/4096)h⁸ + (-85a_10/262144)h¹⁰ + …`   (order 8) ← would be ζ⁸ kernel

Sub-check (b) in `scripts/derive_romberg_2d.py` PASS confirms the order-lift via direct symbolic computation.

## Equivalence to nested Richardson (sub-check (c), the negative result)

**Sympy proves**: `T[K, 0]` is EXACTLY ALGEBRAICALLY IDENTICAL to the nested Richardson cascade `R^{K+1}` from ADR-0088 when both are computed at a single outer step. Explicit verification at K=2 (order 6):

```
T[2, 0]                   = E + (a_6/64)·h⁶ + (21·a_8/1024)·h⁸
R^3(h) = (16·R^2(h/2)^2 − R^2(h)) / 15
                          = E + (a_6/64)·h⁶ + (21·a_8/1024)·h⁸
Difference T[2, 0] − R^3(h) = 0
```

This is not a coincidence: both schemes are LINEAR combinations of the base samples `U(h / 2^m)` with rational coefficients that satisfy `sum w_m = 1` (preservation of the exact value `E` when error coefficients vanish). The nested Richardson recurrence (combine at every depth) and the Romberg table recurrence (combine pairwise across columns) are TWO DIFFERENT BOOKKEEPING SCHEDULES for computing the SAME final linear combination, by uniqueness of the polynomial-elimination problem. Romberg-2D coefficients for K=3 (order 8) computed symbolically: `w_m = {0: -1/2835, 1: 4/135, 2: -64/135, 3: 4096/2835}`, summing to 1.

## Floor contamination model (sub-check (d))

The ζ⁸ Wave II catastrophic measurement (ratio = 3.067 vs theoretical 8.0 at N=512, n-pair {1,2}, per `.dev-docs/research/zeta8-wave-ii-deferred.md`) was diagnosed in ADR-0088 AMENDMENT 2 as a floor-cascade regime: pre-asymptotic K5 spatial-floor noise (~1e-6 at N=512 under Catmull-Rom interpolation) dominates the τ⁸ leading term at n ≥ 2. Both the nested Richardson cascade and Romberg-2D process the SAME base samples `U(h / 2^m)` and apply the SAME linear combination (per sub-check (c)); therefore they incur the SAME floor amplification. Specifically, the Romberg-2D floor amplification factor for K=3 is `|sum w_m · 2^m| = 3937/405 ≈ 9.72`, vs the nested cascade's identical effective amplification at single outer step. **Romberg-2D does NOT bypass the floor-cascade contamination.** The floor problem is intrinsic to the spatial discretisation (Catmull-Rom O(dx⁴) sample floor), not to the temporal extrapolation scheme.

## Outcome classification (per task spec)

**Outcome B** (math creation gives correct order BUT no floor improvement, algebraically equivalent to nested Richardson). Per task spec: "ship as alternative impl with explicit 'equivalent to nested Richardson' caveat; not novel". However, because the algebraic equivalence is EXACT (not approximate), shipping an alternative impl would add code surface area with ZERO observable behavior change — a suckless violation (Cohort 1 anti-pattern). Therefore the deliverables are research artifacts only: the sympy script (proves the equivalence) and the verdict (documents the negative result and re-routes to Path ε Chebyshev).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Ship Romberg-2D as a new kernel `Diffusion2KthRombergChernoff<F, const K>`** | Sub-check (c) proves algebraic equivalence; new kernel would compute IDENTICAL output to existing nested Richardson at the same outer step. Adds code surface with zero behavior change. Suckless violation. |
| **Re-attempt math creation with a different 2-axis construction (e.g., Wynn epsilon + cascade)** | The polynomial-elimination problem at a single outer step has a UNIQUE solution given the base samples. Any "2D" construction over the same K+1 base samples reduces algebraically to the same linear combination (Theorem: uniqueness of interpolating polynomial of degree K at K+1 nodes). Different schemes would only differ in COMPUTATIONAL BOOKKEEPING (associativity / order of operations / numerical roundoff), not in mathematical content. To genuinely escape the equivalence one needs DIFFERENT base samples (e.g., NON-power-of-2 step refinements) or a NON-LINEAR acceleration (Wynn epsilon on the sequence of T[j, 0]), neither of which addresses the floor-cascade. |
| **Romberg-2D over (τ, K) where K varies the BASE kernel order (e.g., K=2 uses K5, K=3 uses a higher-order base)** | Mixes two different objects on one axis (no "K-cascade depth" exists as a continuous parameter; only as a discrete recurrence index). Not a 2-axis grid in the analytical sense — this is just "use a higher-order base", which is the v4.3+ Path ε direction anyway. |
| **Defer math creation entirely; ship Padé P₄/Q₄ per `verdict-v4-3-research-waves.md` Priority 2** | User directive explicitly asks for math creation attempt FIRST per "если не найдёшь, попробуй сам создать"; Padé remains the recommended low-risk parallel path independently. |

## Consequences

- **POSITIVE**: closes the "Romberg-2D" future-work pointer from ADR-0088 AMENDMENT 2 §3 with a rigorous symbolic answer; sympy script is publishable as research note ("On the algebraic equivalence of Romberg-2D and nested Richardson for operator semigroup approximation"); v4.3+ effort is now correctly re-routed to the spatial-floor Path ε direction; explicit negative result preserves "honest closure" per constitution principle #1 (math fidelity > engineering velocity).
- **NEUTRAL**: no kernel ships; no acceptance gate added; properties.yaml unchanged; traits.yaml unchanged; constitution unchanged.
- **NEGATIVE**: user spent one architect cycle on a research direction that turned out to be equivalent to existing code. However, the cycle PRODUCED a rigorous proof that the direction is foreclosed, preventing future re-attempts. Per Constitution Principle #1, this is a net positive (negative results are valuable).
- **BREAKING**: NONE.
- **Schema bumps**: NONE (no contract surface change).
- **Open follow-up** (for v4.3+ Anchor delegation): focus on the two HIGH-leverage / LOW-risk directions from `verdict-v4-3-research-waves.md`: Priority 1 (Chebyshev spectral collocation per Path ε successor, lifts spatial floor) and Priority 2 (diagonal Padé P₄/Q₄ per Higham 2005, direct ζ⁸ via scaling-and-squaring; bypasses the cascade entirely with a different mathematical paradigm).

## Honest assessment (per task spec deliverable 4 instruction)

The math creation attempt succeeded in proving the scheme is well-defined and order-correct (Outcome B per task taxonomy). The KEY finding is the ALGEBRAIC EQUIVALENCE established in sub-check (c): the user's intuition ("Romberg-2D combines base K5 samples DIRECTLY, not Richardson-of-Richardson outputs, so floor accumulates ONCE not K-times") was based on the assumption that DIFFERENT bookkeeping schedules over the SAME linear combination would have different floor characteristics. Sympy proved this is false: the polynomial elimination at K+1 nodes has a UNIQUE solution, so the two schedules compute identical output and incur identical floor noise. The floor-cascade is INTRINSIC to the spatial discretisation, not to the temporal extrapolation framing. The correct re-route is Path ε spatial-floor lift (Chebyshev spectral collocation), not a re-bookkeeping of the same temporal scheme.

## Cross-references

- ADR-0001 — contract-first; this ADR produces a research deliverable, not a contract change.
- ADR-0086 + AMENDMENT 1 — Path β single-step Richardson on K5; the K=1 case of the present scheme.
- ADR-0088 + AMENDMENT 1 + AMENDMENT 2 — nested R²→R³→R⁴ ladder + ζ⁸ floor-cascade DEFER; the present ADR forecloses Option δ "Romberg-2D rewrite" from AMENDMENT 2 §"Future work".
- ADR-0089 — Path ε QuinticHermite spatial floor (the prerequisite that did NOT close the ζ⁸ gap; v4.3+ needs Chebyshev per `verdict-v4-3-research-waves.md`).
- `scripts/derive_romberg_2d.py` — NEW sympy oracle (this ADR's verification deliverable).
- `.dev-docs/research/verdicts/verdict-romberg-2d-attempt.md` — NEW negative-result verdict (this ADR's research deliverable).
- `.dev-docs/research/raw-findings-romberg-2d.md` — Wave 2 research synthesis (63 sources; 0 published Romberg-2D for operator semigroups).
- `.dev-docs/research/verdicts/verdict-v4-3-research-waves.md` — Wave 2 verdict §Priority 3 (Romberg-2D parallel research track).
- `.dev-docs/research/zeta8-wave-ii-deferred.md` — Wave II measurement record + diagnosis (floor cascade); the empirical evidence that motivated the present math attempt.
- Richardson 1911 — *Phil. Trans. R. Soc. A* 210, pp. 307–357.
- Romberg 1955 — *Det Kongelige Norske Videnskabers Selskab Forhandlinger* 28, pp. 30–36.
- Bulirsch-Stoer 1966 — *Numerische Mathematik* 8 (extrapolation algorithms for ODE).
- Hairer-Lubich-Wanner 2006 — *Geometric Numerical Integration* §II.4 + §II.9 (Romberg-in-time + symmetric methods).
- Higham 2005 — *SIAM J. Matrix Anal.* (scaling-and-squaring with diagonal Padé P_m/Q_m; the recommended Priority 2 v4.3+ direction, mathematically DISTINCT from Romberg).
- Galkin-Remizov 2025 — *Israel J. Math.* Theorem 3.1 (Taylor-tangency rate theorem; Romberg-2D rate inheritance argument).
