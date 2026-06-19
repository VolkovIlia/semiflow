# ADR-0013 — v0.6.0 4th-order spatial finite-difference stencils

**Status**: Accepted
**Date**: 2026-05-01
**Authors**: ai-solutions-architect
**Supersedes**: none. Implements Phase 1 of the approved plan
`/home/volk/.claude/plans/composed-cooking-reef.md` (v0.6.0).
**Cross-refs**: ADR-0008 (ζ-A self-adjoint diffusion), ADR-0011 (TruncatedExp
integrator v0.4.0), ADR-0012 (tensor-product 2D), ADR-0014 (adaptive
PI controller — sibling v0.6.0 ADR), `contracts/semiflow-core.math.md`
§9.2.4 (ζ⁴ FD on f-derivatives) and §9.2.5 (TruncatedExp 4th-order stencil),
`.dev-docs/verification/scripts/verify_v0_6_0_zeta4.py` and
`verify_v0_6_0_magnus4.py` (both reproducible, exit 0).

Adopt **two new spatial-4th-order finite-difference paths** as additive
sibling types next to `DiffusionChernoff` (ζ-A) and
`TruncatedExpDiffusionChernoff`, plus the `Strang2D` global-order cap lift
`min(2)` → `min(4)`. (i) `Diffusion4thChernoff` — γ-A inner-Strang
baseline UNCHANGED (the 5-point K-kernel weights `(7/12, 3/16, 1/48)`
already match the constant-a Fourier symbol to order ξ⁴; math.md
§9.2.1) — but the τ²-correction polynomial `a·a'·f''' + ½·a·a''·f'' +
¼·a'·a''·f'` now computes `f', f'', f'''` via 7-point central FD
(Fornberg 1988) with truncation residues O(Δ⁶), O(Δ⁶), O(Δ⁴)
respectively (sympy gate Z⁴_spatial-order). Stencil step
`Δ = max(3·dx, τ^{3/4})`: the floor `3·dx` controls FP cancellation
(stencil span ≥ 6 grid cells), and the `τ^{3/4}`-scaling balances
truncation and round-off so that the τ²-correction's contribution to
local error is `τ²·O(Δ⁴) ≤ O(τ⁵)` — safely inside the local-O(τ⁵) ζ⁴
budget. The 5-point γ-A K-factor is bit-equal to v0.5.0 — no change.
(ii) `TruncatedExp4thDiffusionChernoff` — replaces the v0.4.0 3-point
divergence-form stencil `G` (`[1, -2, 1]·a/dx²` at constant a) with the
5-point divergence-form stencil
`G⁴ f|_i = [-a_{i+3/2}(f_{i+2}-f_{i+1})/12 + 5·a_{i+1/2}(f_{i+1}-f_i)/4
- 5·a_{i-1/2}(f_i-f_{i-1})/4 + a_{i-3/2}(f_{i-1}-f_{i-2})/12]/dx²`.
**Architect correction (sympy-derived)**: the original plan draft used
flux coefficients `(7/6, -7/6)` which would have collapsed to the
non-standard 5-point Laplacian `[-1/12, 5/4, -7/3, 5/4, -1/12]·a₀/dx²`;
the corrected coefficients `(5/4, -5/4)` enforce the standard 4th-order
central Laplacian `[-1/12, 4/3, -5/2, 4/3, -1/12]·a₀/dx²` collapse for
constant a (sympy gate M⁴_const-a-fast — verbatim agreement). For
variable a the residue is **honestly O(dx²)** with strictly tighter
constant than v0.5.0 (Mickens-improvement; sympy gate
M⁴_spatial-order shows leading variable-a residue
`-a'·f'''/12 - a''·f''/8 - a'''·f'/24` — non-zero at dx², zero at
constant-a). No 5-point divergence-form stencil sampling `a` only at
half-grid nodes can achieve variable-a spatial 4th order (Strikwerda
§6.4 + Mickens 1994 §3.2 impossibility); the v0.6.0 honest claim is
4th-order spatial **at constant a** (which is the regime of the
flagship 2D heat oracle eq. 10.7 — the very target of the v0.6.0 slope
gate G3⁴-2D ≤ -3.80) and tighter dx²-constant for variable a. K=4
truncated power series unchanged from v0.4.0; new operator-norm bound
`‖G⁴‖ ≤ 16·‖a‖_∞ / (3·dx²)` (Fourier symbol at kdx=π) yields a 25%
tighter CFL `τ < 3·dx² / (8·‖a‖_∞)` (constants `CFL_NUMER = 3`,
`CFL_DENOM = 8`); the v0.4.0 `SemiflowError::CflViolated`
variant is reused verbatim. (iii) `Strang2D::order()` cap lifts
`ox.min(oy).min(2)` → `ox.min(oy).min(4)` — Theorem 7 (math.md §10.3)
gives global Strang2D order = per-axis order on separable L; the
v0.5.0 cap-at-2 was conservative. Existing v0.5.0 unit test
`strang2d.rs:153-155` `assert_eq!(s.order(), 2)` is updated in Phase 4
(Engineer task). All v0.5.0 callers using `DiffusionChernoff` and
`TruncatedExpDiffusionChernoff` remain bit-equal — the new types are strict
additions, not replacements; sympy gates Z⁴_const-a and M⁴_const-a-fast
are the bit-equal regression guarantees. Rejected alternatives:
(a) 9-point central FD for f, f', f''' (would lift variable-a Δ-order
to O(Δ⁸) but doubles f.sample stencil span — boundary-layer issues
within `4·Δ` of edges, and the τ²·O(Δ⁸) budget yields no improvement
once the K-kernel's spatial 4th-order locks the floor); (b) genuine
4th-order Mickens-compact divergence-form stencil with `a` sampled at
both grid AND half-grid points (would lift variable-a spatial to O(dx⁴)
but requires 11 closure evaluations per flux — quadruples per-step cost
for marginal CEV benefit; deferred to v0.7+ pending production
mileage); (c) replacing v0.5.0 TruncatedExp in place (regression risk for
existing `truncated_exp_variable_zeta_liouville_oracle_slope` ≤ -1.95 gate —
sibling-type costs nothing). Consequences: exactly **+2 public types**
(`Diffusion4thChernoff`, `TruncatedExp4thDiffusionChernoff`); **+0
dependencies**; **+0 SemiflowError variants** (CflViolated reused);
existing CFL constants `TRUNC_CFL_NUMER=1, TRUNC_CFL_DENOM=2`
unchanged for v0.4.0 type, NEW constants `CFL_NUMER=3,
CFL_DENOM=8` for the v0.6.0 type only; observable headline:
`AxisLift<Diffusion4thChernoff>` and `AxisLift<TruncatedExp4thDiffusionChernoff>`
instantiate without `axis.rs` changes (generic bounds satisfied);
`Strang2D<Diffusion4thChernoff, Diffusion4thChernoff>` constant-a per
axis achieves global slope ≤ -3.80 (G3⁴-2D — the v0.6.0 release
gate). Folded ADR: the originally-planned ADR-0015 (TruncatedExp K=4 +
4th-order CFL bound) is **subsumed into this ADR §3** (CFL constants
above) — the new bound is short enough that a separate ADR would be
ceremonial overhead per the suckless-conventions guardrail.

**Mechanism (audited 2026-05-02)**: The 4th-order spatial accuracy
emerges jointly from (a) Fourier-symbol ξ⁴-matched K-kernel weights
(7/12, 3/16, 1/48) and (b) cubic-Hermite interpolation in
`f.sample(off-grid)` (ADR-0005, O(dx⁴)) called at shifted grid points.
Empirical 2D slope -3.99 ≤ -3.85 (G3⁴-2D, N ∈ {400, 800, 1600}, n=2000)
confirms the asymptotic dx⁴ floor is reached at N=1600 (ratio
{7.27, 34.73}, the 34.73 > 16 = 2⁴ being the asymptotic-regime
signature). See math.md §9.2.4 mechanism clarification.

**Amendment 1 (v0.6.1, 2026-05-02)**: "Fourth-order" in this ADR
refers EXCLUSIVELY to the dx-axis spatial accuracy `q = 4` (the
Fornberg 7-point central FD residue O(Δ⁶)/O(Δ⁶)/O(Δ⁴) for
`Diffusion4thChernoff`, the 5-point divergence-form Laplacian for
`TruncatedExp4thDiffusionChernoff` constant-`a`). The τ-axis Chernoff
consistency order `p` remains `2` for both types — unchanged from
v0.5.0 ζ-A and v0.4.0 Magnus, because the γ-A baseline / Strang
composition / K=4 truncated power series all saturate the τ-axis
order at 2 (sympy gates Z⁴_τ², M⁴_τ²). The two axes are orthogonal
and `ChernoffFunction::order()` advertises ONLY the τ-axis `p`
per math.md §11.1.bis (NORMATIVE).

**Defect cross-reference (D1)**: The v0.6.0 implementations of
`Diffusion4thChernoff::order()` (`diffusion4.rs:145-148`) and
`TruncatedExp4thDiffusionChernoff::order()` (`truncated_exp4.rs`) returned
`4`, conflating `q` with `p`. This violated this ADR's contractual
intent (the "4th-order" name describes the *spatial* lift, not the
Chernoff order) and propagated an incorrect Richardson divisor
`2^4 − 1 = 15` through `AdaptivePI` (correct: `2^2 − 1 = 3`).
v0.6.1 PATCH fixes the implementation to return `2`. See
`docs/audit-findings-v0_6_0.md` D1 and math.md §11.1.bis.

**Amendment 2 (v0.7.0, 2026-05-02, audit D2)**: `Magnus4thDiffusionChernoff` shipped in v0.6.0 implements a **truncated Taylor series** (K=4 power series of G⁴), NOT the genuine Magnus expansion. The type was renamed to `TruncatedExp4thDiffusionChernoff` in v0.7.0 (clean break). Constants `MAGNUS4_CFL_NUMER` and `MAGNUS4_CFL_DENOM` were renamed to `CFL_NUMER` and `CFL_DENOM`; `MAGNUS_TRUNC_ORDER` renamed to `TRUNC_ORDER`. No algorithm or CFL change. See `docs/audit-findings-v0_6_0.md` D2.

**Strang2D cap consequence**: The cap lift `min(2) → min(4)` (§ paragraph
above) was motivated by an incorrect reading of the inner `order()`
returns; with the corrected `order() = 2`, `Strang2D::order() =
min(2, 2, 4) = 2` — matching the canonical Strang global order
(Theorem 7, math.md §10) and the type-level invariant. The cap of
`4` in `Strang2D::order()` is RETAINED purely as a forward-compatibility
ceiling for hypothetical higher-τ-order inner functions (e.g., a
future Magnus K=6 path that genuinely lifts τ-axis order to `4`).

**Amendment 3 (v9.2.0, 2026-06-19, HW-floor recalibration of the G3⁴-2D
SPATIAL flagship gate)**: The flagship spatial gate
(`spatial_convergence_2d_4th`, math.md §9.2.1) is recalibrated and
HONESTLY re-scoped from a two-sided "asymptotic ratio→16×, slope→−4.0"
target to a **one-sided spatial-order floor**: `OLS slope ≤ −3.85` ⇔
`spatial order ≥ ~4`. The threshold `−3.85` is UNCHANGED — it always was
a one-sided lower bound, never a `≈ −4.0 ± ε` equality. The earlier prose
("ASYMPTOTIC", "Expected ≈ −4.0", "ratio 34.73 > 16 = asymptotic signature")
mislabeled the regime: 34.73 > 16 is *steeper* than the asymptotic limit,
i.e. itself super-convergent/floor-noise, not the asymptotic plateau. On
the current i7-12700K the original fine window `{400,800,1600}` at `n=2000`
is **floor-contaminated** (measured temporal floor ≈1.2e-9 at n=2000, ~5×
the docstring's optimistic 2.4e-10), collapsing N=800/1600 to a bogus
shallow slope −1.4472 (this arc produced two false-greens by trusting that
floor estimate). The recalibrated coarse window `N∈{200,300,400}` at
`n=4000` is floor-clean (floor ≈0.3e-9 < 3.5% of the smallest spatial
error 8.75e-9) and measures slope −5.8679 — **pre-asymptotic and
super-convergent**: ratios {10.81, 5.40} sit ABOVE the 4th-order prediction
{5.06, 3.16} and DECREASE toward it, the textbook signature of a genuinely
≥4th-order scheme with higher-order (dx⁶,dx⁸) terms still active. The
scheme's true asymptotic spatial order is and remains **exactly 4** (a
property of `src/`, unchanged); the window simply does not reach the dx⁴
plateau on this HW within a 30s budget. The gate still discriminates a
real order-regression: a 2nd-order degradation yields slope −2.0, which
FAILS −3.85 by a wide margin (verified). Rejected alternative (Option B):
the fine asymptotic window made literal by raising `n≈12000–16000` to push
the floor below the N=1600 spatial error (~3.4e-11) — ~1hr wallclock AND it
re-enters the near-f64-noise regime that already produced two false-greens;
it buys prose-literalism at the price of both cost and numerical safety, so
it is rejected. The fix is documentation-only: stop calling the window
"asymptotic", state the true order (4) and the expected pre-asymptotic
measurement (≥4, ≈−5.87 here) in their correct distinct roles. `src/`
untouched. See math.md §9.2.1 "v9.2.0 recalibration note".
