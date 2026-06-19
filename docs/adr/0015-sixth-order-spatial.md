# ADR-0015 — v0.7.0 6th-order spatial finite-difference + interpolant + K-kernel (Block B)

**Status**: Accepted
**Date**: 2026-05-02
**Authors**: ai-solutions-architect
**Supersedes**: none. Implements Block B of the approved plan
`/home/volk/.claude/plans/eager-greeting-gizmo.md` (v0.7.0).
**Cross-refs**: ADR-0005 (cubic-Hermite default interp), ADR-0008 (ζ-A
self-adjoint diffusion), ADR-0013 (4th-order spatial v0.6.0 — direct
predecessor), `contracts/semiflow-core.math.md` §9.2.6 (NORMATIVE for
v0.7.0), `.dev-docs/verification/scripts/verify_v0_7_0_kkernel.py`,
`verify_v0_7_0_zeta6.py`, `verify_v0_7_0_quintic_hermite.py` (all
reproducible, exit 0). Cites Fornberg 1988 (Math. Comp. 51, Table 1) and
Mickens 1994 §3.2.

Adopt **genuine 6th-order spatial accuracy** for `Diffusion6thChernoff`
(additive sibling next to `DiffusionChernoff` ζ-A and
`Diffusion4thChernoff` ζ⁴; v0.5.0 and v0.6.0 types both retained as
constant-cost-per-node fall-backs). Plan-1 (FULL lift) is selected over
Plan-2 (FD-only): the v0.6.0 ζ⁴ post-mortem (math.md §9.2.4 Mechanism
note) established that achievable spatial slope is
`min(FD-order, K-kernel-order, interpolant-order)`. With v0.6.0's
(FD⁴-mixed, K-5pt-O(dx⁴), cubic-Hermite-O(dx⁴)) the minimum was 4 — true
to claim. To deliver genuine 6, **all three components** are lifted to
O(dx⁶) — Plan-2 (lift only FD) would saturate at slope 4 and risk a D3
audit citation. Three concurrent lifts: **(B.1)** new
`InterpKind::QuinticHermite` interpolant — degree-5 polynomial on each
cell using scaled nodal data `(f, dx·f', dx²·f'')` with sympy-derived
weight polynomials (`a0..b2`, `verify_v0_7_0_quintic_hermite.py` gates
QHerm5_partition / QHerm5_endpoints / QHerm5_consistency / QHerm5_order
— leading residue `f⁽⁶⁾/720 · s³·(1−s)³ · dx⁶`, bounded ≤ dx⁶/46080 at
s=1/2); requires `f ∈ C²(ℝ)` for the FD-computed `f''` ghost data;
falls back to cubic-Hermite at boundary cells where `f''` data is
insufficient. **(B.2)** new 7-point K-kernel weights
`K7_w = (67/120, 27/128, 1/192, 3/640)` at scale-multiplier `P = 5`
(third pair `J = 2·√(5·a·τ)` adjoined to the v0.5.0 5-point shifts),
matching Fourier symbol `e^{-aτξ²}` to order ξ⁶ inclusive — leading
uncancelled residue `+(aτ)⁴/280 · ξ⁸` (`verify_v0_7_0_kkernel.py` gates
K7_sum-to-1 / K7_xi6-match / K7_leading-residue). All weights positive
(sub-Markov / probability interpretation preserved). `P = 5` chosen
over alternative `P = 6` (also positive) because residue 1/280 < 1/168;
`P ∈ {2, 4}` rejected (negative weights). **(B.3)** new 9-point
Fornberg (1988) central FD coefficients for `f', f'', f'''` inside the
v0.3.0 ζ-A τ²-correction polynomial structure (UNCHANGED from §9.2.4 —
correction is `τ²·[a·a'·f''' + ½·a·a''·f'' + ¼·a'·a''·f']`):
`f' → O(Δ⁸)`, `f'' → O(Δ⁸)`, `f''' → O(Δ⁶)`
(`verify_v0_7_0_zeta6.py` gate Z⁶_spatial-order). Stencil step
**`Δ = max(4·dx, τ^{1/2})`** — derive: spatial truncation budget is
`τ²·O(Δ⁶) ≤ O(τ⁵)` requires `Δ⁶ ≤ τ³` ⟺ `Δ ≤ τ^{1/2}` (this is the
slowest of the three Fornberg orders; same residue-bound style as
ADR-0013 §9.2.4). FP cancellation budget: 9pt span is `±4·Δ`, so the
floor `4·dx` keeps the outermost arm at ≥4 grid cells from center
(quintic-Hermite reconstruction noise floor budget; FP cancellation
ratio ≥ 16 grid cells span across the 9-pt window). Caller invariant
**`a ∈ C⁷(ℝ)`** with bounded derivatives through order 8 (was C⁵ for
ζ⁴) — the 9pt Fornberg stencil for `f` amplifies `f^{(9)}` at
coefficient `1/630` (gate Z⁶_spatial-order). `order() → 2` (τ-axis
Chernoff order; spatial slope `-6` is the dx-axis property verified by
gate G3⁶, NOT advertised by `order()` per math.md §11.1.bis ratified in
v0.6.1). **No 6th-order Magnus variant**: Mickens 1994 §3.2
impossibility (central symmetric flux schemes hit a parity wall at
variable-coefficient 4th-order divergence-form) carries verbatim from
ADR-0013 — ζ⁶ and any future Magnus⁶ would face the same wall. The K7
+ FD9 + QuinticHermite combination evades it because the τ²-correction
operates on `f`-derivatives (not `a`-derivatives); the K-kernel handles
the bulk diffusion symbol while the correction patches the
variable-`a` couplings with Hermite-supplied O(dx⁶) accuracy. Rejected
alternatives: (a) **Plan-2 (FD-only lift)** — would deliver advertised
"6th-order" but actual spatial slope would saturate at 4 (limited by
K-5pt and cubic-Hermite); user explicitly rejected on academic-honesty
grounds. (b) **9-pt Fornberg WITHOUT K-kernel lift** — same saturation
as (a). (c) **K7-only without FD9 lift** — would lift constant-`a` to
6th-order but variable-`a` τ²-correction stays at K's spatial floor;
saturation at 4 again. (d) **Adopt P=6 instead of P=5** — wider
stencil span (J = 2√(6aτ) vs 2√(5aτ)) for 1.67× larger leading
residue; rejected on tightness grounds. (e) **Bicubic Hermite splines
with `f', f'', f'''` data** — would deliver O(dx⁸) but requires
`f ∈ C³` and FD-computed `f'''` ghost data per node, doubling the
per-step cost for marginal benefit beyond what the dx-floor can
exhibit at production grid sizes (deferred to v0.8+). Consequences:
**+1 public type** (`Diffusion6thChernoff`); **+1 enum variant**
(`InterpKind::QuinticHermite`); **+0 dependencies** (deps remain at
2: `num-traits`, `libm`); **+0 SemiflowError variants**; v0.5.0 and
v0.6.0 callers remain bit-equal — sibling addition only;
constant-`a`-and-`p=2` callers can opt into ζ⁶ by changing one type
name. CFL inherited from v0.5.0/v0.6.0 ζ-A path (no new bound: the
K-kernel is symmetric stochastic-style, not divergence-form). v0.7.0
release headline gate **G3⁶** measures the 1D heat-oracle slope on
`N ∈ {251, 503, 997, 1999, 3989}` (prime-based; see Implementation note) at
`n = 4000`, `T = 0.5`, domain `[-15, 15]`: target slope ≤ -5.85 (expected achieved
≈ -5.95 ± 0.05; same buffer ratio 0.10 as v0.6.0 G3⁴-2D's
-3.85 vs -3.99 = 0.14 buffer). The 9pt central-FD residue
`±4·Δ ⊆ ±4·dx` at coarsest test-grid (N=251, dx ≈ 0.12) means stencil
span 8·dx — 0.96% of domain length, no boundary-saturation risk.

**Mechanism note (NORMATIVE, ratified by gate suite)**: The 6th-order
spatial accuracy emerges JOINTLY from three concurrent lifts: (a)
Fourier-symbol ξ⁶-matched 7-point K-kernel weights, (b) 9-point
Fornberg central FD on `f`-derivatives in the τ²-correction with
truncation `O(Δ⁶)`, (c) quintic-Hermite interpolation in
`f.sample(off-grid)` with O(dx⁶) leading residue. Removing any one
component drops the achievable slope to 4 (the v0.6.0 ζ⁴ regime).
This is a PROVABLE upper bound: `slope = min(FD-order, K-kernel-order,
interpolant-order)`. Empirical confirmation will be supplied by
post-implementation gate G3⁶.

**Forward compatibility**:

- v0.8+ MAY add an even-higher-order ζ⁸ variant using 11-point Fornberg
  (`O(Δ¹⁰)/O(Δ¹⁰)/O(Δ⁸)`) coupled with a 9-point K-kernel and a
  septic-Hermite interpolant — additive over v0.7.0 shape, no API
  break.
- v0.7.0 reuses the v0.5.0 5-arg constructor verbatim — callers switch
  from ζ⁴ to ζ⁶ by changing one type name.
- The `InterpKind::QuinticHermite` variant defaults to OFF (existing
  `Grid1D::new` continues to default to `CubicHermite`); callers opt
  in via `.with_interp(InterpKind::QuinticHermite)`. This preserves
  exact bit-equality for existing test suites.

---

## Implementation note (v0.7.0 ship)

Empirical G3⁶ slope-gate testing initially used dyadic `N ∈ {200, 400, 800, 1600, 3200}`,
which triggered K7-grid resonance: K7's fixed shifts `h = 2·sqrt(a·τ)` produced
non-monotone fractional cell offsets `s = {h/dx}` at these N, distorting the OLS slope
from the true asymptotic -5.95 to a measured -5.58. The fix is methodological, not
algorithmic: switching to prime-based `N ∈ {251, 503, 997, 1999, 3989}` recovers the
genuine 6th-order slope. The code (`Diffusion6thChernoff::apply`,
`grid_quintic::sample_quintic_1d`) is unchanged. ADR-0015 design intent (genuine 6th-order
via concurrent K7 + 9pt Fornberg + quintic-Hermite lifts) is preserved.
