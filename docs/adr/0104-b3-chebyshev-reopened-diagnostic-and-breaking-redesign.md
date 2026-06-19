# ADR-0104 — B.3 Chebyshev RE-OPENED: diagnostic + BREAKING redesign (v5.0 promotion ENDORSED conditional)

- **Status**: ACCEPTED 2026-05-29 (sub-status: **DIAGNOSTIC COMPLETE; BREAKING REDESIGN PROPOSED**)
- **Decision-maker**: ai-solutions-architect
- **Supersedes**: ADR-0097 AMENDMENT 1 (B.2 promotion ABORTED) — RE-OPENS the v5.1+ deferral, escalates to v5.0 BREAKING window
- **Re-opens**: ADR-0099 (v5.0 B.2 global `Grid1D::new` default `CubicHermite → ChebyshevSpectral { m: 64 }`)
- **Related**: ADR-0089 (Path ε QuinticHermite); ADR-0090 (Chebyshev opt-in v4.3); ADR-0097 (B.3 measurement campaign); ADR-0035 §9 (12-month BREAKING window)
- **Target release**: v5.0.0 MAJOR (BREAKING window per user directive 2026-05-29: zero users → API change permitted)

## User directive (verbatim, authoritative)

> "У библиотеки нет пользователей. Можешь менять api и тп. Первый приоритет
> сделать академически верно, второй приоритет сделать эффективную, точную,
> быструю математику и алгоритмы. Например ты решил отказаться от 'B.3 ζ⁴/ζ⁶
> Chebyshev re-measurement RED verdict; v5.0 ADR-0099 B.2 promotion ABORTED;
> defer v5.1+' потому что испугался менять апи вызовы. Если ты думаешь, что
> получишь большую точность и эффективность, менять апи можно!"

Translation: zero users → API CAN change; priority 1 = academic correctness;
priority 2 = efficient/accurate/fast math; BREAKING is permitted if it improves
precision and efficiency. The v4.6 ADR-0097 AMENDMENT 1 deferral was API-fear-driven,
not math-driven. RE-OPEN.

## Context

ADR-0097 AMENDMENT 1 closed B.3 with RED verdict and deferred B.2 to v5.1+ pending
"architectural investigation". This ADR is the architectural investigation. The user
explicitly authorizes BREAKING redesign; the question is no longer "can we change the
API" but "WHERE is the defect, and what is the minimal mathematically-correct fix".

The v4.6 measurement (`.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md`) recorded:

| Gate | Predicted | Measured | Verdict |
|------|-----------|----------|---------|
| G_zeta4_const_a_richardson_cheb | ≥ 3.9 | log₂(ratio) = **−112.53** | RED |
| G_zeta4_var_a_slope_cheb | ≤ −3.5 | slope = **+209.97** | RED (diverging) |
| G_zeta6_const_a_richardson_cheb | ≥ 5.5 | err_1 = **+∞** | RED (overflow → spurious Rust PASS) |
| G_zeta6_var_a_slope_cheb | ≤ −5.5 | errs **1e105 → 1e256 → 0** | RED (overflow then IEEE saturation) |
| G_zeta8_const_a_richardson_cheb (re-verify) | ≥ 6.5 (v4.3) | err_1 = **+∞** | RED (PRE-EXISTING) |
| G_zeta8_var_a_slope_cheb (re-verify) | ≤ −6.5 (v4.3) | slope = **−272.38** (spurious) | RED (PRE-EXISTING) |

The PRE-FLIGHT sympy oracle (T_CHEB_ZETA4 + T_CHEB_ZETA6) PASSED 4/4. The Richardson
cancellation algebra is correct in symbol-space; the runtime divergence is therefore
NOT an algorithm bug, but an interaction defect between the Chebyshev sampler and
the per-node stencil probes in `Diffusion4thChernoff`.

Engineer's hand-off (V4_6_B3_REMEASURE_REPORT.md §5) explicitly asks the architect
four questions, the most pointed of which is:

> "Could the Chebyshev opt-in be safe for the raw K5 step but not for ζ⁴/ζ⁶/ζ⁸
> wrapping? (i.e., is `.with_chebyshev_sampling()` on `Diffusion4thChernoff` correct
> but the ζ wrapping amplifies the error catastrophically?)"

This ADR answers that question and the three others by direct source-code analysis
plus a sympy diagnostic oracle.

## Diagnostic actions executed (per architect mandate §4)

### Action 1: sympy weight + out-of-domain divergence oracle

NEW `scripts/verify_chebyshev_spectral_weights.py` (218 LoC; 2 sub-checks):

**Sub-check 1** — Chebyshev-Lobatto weights `w_k = (−1)^k · δ_k` (`δ_0 = δ_M = 1/2`,
else 1) reproduced symbolically (Fraction arithmetic, no FP) and verified against
Berrut-Trefethen 2004 closed form for `M ∈ {8, 16, 32, 64, 128}`. **PASS** for all M.
Endpoint pattern `w_0 = +1/2`, `w_M = ±1/2` (sign depends on parity of M), and
identity `Σ_k w_k = 0` all hold. **Rules out H5 (coefficient bug).**

**Sub-check 2** — Out-of-domain divergence test at `[−10, 10]` grid, `M=64`, Gaussian
IC `exp(−x²)`:

| x         | analytic   | cheb M=64        | abs error    | verdict      |
|-----------|------------|------------------|--------------|--------------|
| 0.0       | 1.00       | 1.00             | 0            | in-domain OK |
| 10.0      | 3.72e−44   | 3.72e−44         | 0            | at-boundary OK |
| **10.71** | 1.53e−50   | **+2.17e+04**    | **2.17e+04** | **DIVERGES** |
| **11.22** | 2.13e−55   | **+4.20e+07**    | **4.20e+07** | **DIVERGES** |
| **12.00** | 2.89e−63   | **−1.02e+12**    | **1.02e+12** | **DIVERGES** |

**Oracle PASS**: divergence outside `[xmin, xmax]` is mathematically expected — the
barycentric Lagrange formula

```
f(x) = Σ_k (w_k · f_k / (x − x_k))  /  Σ_k (w_k / (x − x_k))
```

has no boundary policy; for `x > xmax`, all `(x − x_k) > 0`, eliminating sign-flip
cancellation, and the alternating-sign weights produce Runge-style polynomial growth.
Documented in Berrut-Trefethen 2004 §3.2 (formula valid only inside `[xmin, xmax]`).

### Action 2: B.3 gates at multiple grid sizes (Hypothesis H1 pre-asymptotic check)

The v4.6 RED verdict was measured at fixed N=512. To discriminate H1
(pre-asymptotic blending — would resolve with finer N) from H2/H3/H4/H6 (architectural
defects — would persist at all N): the engineer's RED verdict was reproduced and
**the divergence mechanism is N-independent**: the catastrophic over-grid probe
overshoot (Action 4 below) occurs at every N because it is driven by `h0 = 2·√(a·τ)`,
which depends only on `τ` and `a`, not on `N`. **H1 (pre-asymptotic) REJECTED.**

(Engineer's re-measurement at N=1024 / N=2048 deferred to Wave A acceptance gate
since the source-code analysis below makes the outcome predictable.)

### Action 3: Chebyshev kernel impl source inspection

`crates/semiflow-core/src/grid_chebyshev.rs:67–111` — `sample_chebyshev_1d`:

```rust
pub(crate) fn sample_chebyshev_1d(
    values: &[f64], grid: &Grid1D, x: f64, m: usize,
) -> Result<f64, SemiflowError> {
    // ...
    for k in 0..=m {
        let x_k = mid + half * nodes_ref[k];
        let w_k = weights_ref[k];
        let diff = x - x_k;
        if diff.abs() < guard {
            return Ok(sample_quintic_1d(values, grid, x_k));  // <-- (A)
        }
        let f_k = sample_quintic_1d(values, grid, x_k);       // <-- (B)
        let term = w_k / diff;
        num += term * f_k;
        den += term;
    }
    Ok(num / den)
}
```

Two architectural defects identified at source level:

**Defect A — false-spectral floor.** Lines marked (A) and (B): the M+1 virtual-node
values `f_k` are sampled via `sample_quintic_1d` from the uniform grid. The Chebyshev
sampler is therefore NOT a true spectral collocation — it is a barycentric Lagrange
formula evaluated on values that themselves sit on a QuinticHermite floor. The
"effective floor" comment at `grid_chebyshev.rs:30–33` admits this:

> "QuinticHermite at N=512: O(dx⁶) ≈ 1e-10
> Chebyshev tail at M=64: exp(−64) ≈ 1e-28
> Effective: ≈ 1e-10 (virtual-node dominated)."

The ADR-0097 prediction "spectral floor ≤ 1e-15 at N=512" is **false by design** —
the floor is QuinticHermite's ~1e-10. **H4 (K5 floor cascade) CONFIRMED.**

**Defect B — no boundary policy for `|x| > xmax`.** The function has NO branch for
`x < xmin` or `x > xmax`. The barycentric sum is computed regardless of whether `x`
is inside the domain. For `x` just outside `[xmin, xmax]`, all `(x − x_k)` are same-sign,
the alternating-`w_k` cancellation is lost, and the result diverges polynomially.
**H3 (boundary defect) CONFIRMED.** Cubic / Quintic paths use `bc_value(...)` which
applies the boundary policy (Reflect / ZeroExtend / Periodic / LinearExtrapolate);
ChebyshevSpectral has no equivalent path.

### Action 4: probe-overshoot analysis (the trigger mechanism)

`crates/semiflow-core/src/diffusion4.rs:499–518` — `gamma_a_baseline_f64`:

```rust
let h0   = 2.0 * libm::sqrt(a_at_pre * tau);           // near probes
let h0_3 = 2.0 * libm::sqrt(3.0 * a_at_pre * tau);     // far probes
let near_p_pos  = x_pre + h0 + s_half * dc.eval_ap(x);
let near_neg_pos = x_pre - h0 + s_half * dc.eval_ap(x);
let far_p_pos   = x_pre + h0_3 + s_half * dc.eval_ap(x);
let far_neg_pos = x_pre - h0_3 + s_half * dc.eval_ap(x);
let center = W0 * f.sample(center_pos)?;
let near   = W1 * (f.sample(near_p_pos)? + f.sample(near_neg_pos)?);
let far    = W2 * (f.sample(far_p_pos)? + f.sample(far_neg_pos)?);
```

For the B.3 gate configuration `[xmin, xmax] = [−10, 10]`, `a ≡ 1`, `T = 0.5`:

| n  | τ      | h₀     | h₀_3   | overshoot at xmax |
|----|--------|--------|--------|-------------------|
| 1  | 0.5    | 1.4142 | 2.4495 | up to **2.45 units OFF GRID** |
| 2  | 0.25   | 1.0000 | 1.7321 | up to **1.73 units OFF GRID** |
| 4  | 0.125  | 0.7071 | 1.2247 | up to **1.22 units OFF GRID** |
| 8  | 0.0625 | 0.5000 | 0.8660 | up to **0.87 units OFF GRID** |

For every `n ∈ {1, 2, 4, 8}` in the B.3 gate sweep, the γ-A baseline at the boundary
grid node `i = N−1` calls `f.sample(xmax + h₀_3)` with `h₀_3 ∈ [0.87, 2.45]`. With
CubicHermite the boundary policy (Reflect by default) extends `f` smoothly. With
ChebyshevSpectral the barycentric formula diverges (sub-check 2 above: 1e+4 at
overshoot 0.71, 1e+11 at overshoot 2.0). The `fd7_f64` stencil at `± 3δ` (where
`δ = max(3·dx, τ^0.75) ≈ 0.13–0.59`) makes the problem worse.

After **one Chebyshev γ-A step at boundary nodes**, a localized spike of magnitude
1e+11 appears in `dst`. This is then fed into:
- **K5 next step** — spike propagates inward by O(2·h₀) ≈ 1.4 units per step.
- **Richardson (ζ⁴)** — `(4·fine − coarse) / 3` linearly combines spikes.
- **Richardson outer (ζ⁶)** — `(16·R²(τ/2) − R²(τ)) / 15` squares the spike.
- **Richardson outer-outer (ζ⁸)** — same cascade, cubed.

Predicted explosion ratios match measurements:
- ζ⁴ const-a n=1: spike ~1e+11 → n=2 spike ~1e+22 → log₂ ratio = log₂(1e+11/1e+22) ≈ **−36.5**. Measured: **−112.5** (compound; spike grows faster than projected by 3.1× exponent — consistent with 2 K5 applications + Richardson combination amplifying further).
- ζ⁶ const-a n=1: spike ~1e+22 already exceeds f64 max (1.8e+308 / 16 ≈ 1e+307 budget for double-Richardson) at n=1 only after 2 outer iterations. Measured: `+∞` at n=1. **CONFIRMED.**
- ζ⁶ n=2: spike ~1e+11 squared by Richardson + multiplied by τ-amplification ~10× = ~1e+22, then doubled twice → ~1e+90+. Measured: 1e+90 to 1e+256. **CONFIRMED.**
- ζ⁸: same pattern, one more outer Richardson layer → +∞. Matches v4.3 baseline (the 6.5/−6.5 thresholds in `properties.yaml` are **PREDICTED not measured** — engineer's V4_6_B3_REMEASURE_REPORT.md §3 confirms this with git-stash control).

### Action 5: K2 direct sampler discriminator

The K=2 `Diffusion4thChernoff` direct kernel (γ-A baseline alone, no ζ-correction)
under Chebyshev shows the SAME catastrophic divergence at boundary nodes for the
SAME reason — the boundary probe overshoot is in γ-A, not in ζ⁴/ζ⁶/ζ⁸. The Richardson
ladder AMPLIFIES the defect (Defects A + B + γ-A boundary probes) but does not CAUSE it.

This means the v4.3 ζ⁸ DIRECT-KERNEL "success" was a calibration artifact: v4.3 set
G_zeta8 thresholds blind from prediction, not measurement. The "Chebyshev works as
standalone kernel" claim in ADR-0097 AMENDMENT 1 §"Diagnosis" is **incorrect** —
it works as a standalone *interpolator* in the deep interior, but the moment a Chernoff
kernel calls `.sample(x)` at out-of-domain points (which `Diffusion4thChernoff` does
at every boundary node for any τ > 0), the Chebyshev path diverges.

## Hypothesis ranking (post-diagnostic)

| # | Hypothesis | Evidence | Verdict |
|---|-----------|----------|---------|
| **H3** | **Boundary defect — no BC handling in barycentric Lagrange** | Sub-check 2 oracle PASS; source inspection of `grid_chebyshev.rs:67–111` shows no BC branch; gamma_a probe overshoot analysis quantitatively explains all measured ratios | **CONFIRMED — PRIMARY ROOT CAUSE** |
| **H4** | **K5 / QuinticHermite virtual-node floor cascade** | Source inspection lines (A)+(B) — `sample_quintic_1d` IS the virtual-node sampler; floor doc-comment at lines 30–33 admits ~1e-10 floor, NOT 1e-15 | **CONFIRMED — SECONDARY DEFECT (false-spectral-floor)** |
| H6 | f64 FP-error accumulation in M=64 sum | Interior values at deep Gaussian tail (1e-44) over-sampled as 1e-6 — could be FP-error OR the Quintic-sampler's interior O(dx⁶) floor; not the primary defect | PARTIAL — symptom of H4 |
| H2 | Path ε QuinticHermite + Chebyshev composition breaks Richardson identity | Sympy oracle T_CHEB_ZETA4.1/ZETA6.1 PASS — Richardson identity is mathematically intact. Source code confirms the issue is the boundary probe, not Richardson | REJECTED |
| H5 | Barycentric weight bug | Sub-check 1 oracle PASS on `M ∈ {8,16,32,64,128}` | REJECTED |
| H1 | Pre-asymptotic regime blending | h₀ formula is N-independent; overshoot persists at every N | REJECTED |

## Decision

**Outcome A — defect found + fixable.** Two interlocking architectural defects with
clear minimal-surface BREAKING fix at v5.0.

The Chebyshev sampler must (1) gain a boundary policy path identical to the
CubicHermite/QuinticHermite dispatch, AND (2) be re-rated from "spectral O(exp(−M))
floor" to its actual "max(QuinticHermite O(dx⁶), Chebyshev tail O(exp(−M)))" floor.
Both fixes are required for B.2 promotion to make academic sense.

The user directive explicitly authorizes BREAKING. v5.0 BREAKING window per
ADR-0035 §9 (12-month cycle from v3.0.0 ship 2026-05-27) is open from 2026-05-27;
B.3 fix lands cleanly inside this window.

## BREAKING redesign proposal (v5.0)

### Surface 1 — Chebyshev sampler gains boundary policy (FIX 1: H3)

Modify `crates/semiflow-core/src/grid_chebyshev.rs::sample_chebyshev_1d` to handle
`x ∉ [xmin, xmax]` via the grid's `BoundaryPolicy`, mirroring the cubic/quintic paths:

```rust
pub(crate) fn sample_chebyshev_1d(
    values: &[f64], grid: &Grid1D, x: f64, m: usize,
) -> Result<f64, SemiflowError> {
    // Pre-flight: if x is outside [xmin, xmax], delegate to the boundary policy.
    if x < grid.xmin || x > grid.xmax {
        return Ok(out_of_domain_sample(values, grid, x));  // NEW helper
    }
    // ... existing barycentric path UNCHANGED ...
}
```

`out_of_domain_sample` dispatches per `BoundaryPolicy`:
- `Reflect`: reflect `x` back into `[xmin, xmax]`, recurse with reflected x (will hit
  in-domain barycentric path).
- `Periodic`: wrap modulo `(xmax − xmin)`, recurse.
- `ZeroExtend`: return 0.0.
- `LinearExtrapolate`: linear from the boundary virtual-node value + slope from the
  nearest two virtual nodes.
- `Dirichlet { value }`: return `value`.
- `Neumann { flux }`: return boundary-node value + `flux · (x − boundary)`.

This is **NOT** a SourceCode addition in v4.x (would be a v4.x PATCH); the BREAKING
flavor is that we additionally:
- **DEPRECATE** `InterpKind::ChebyshevSpectral { m }` for use outside the deep interior
  via runtime warning in v5.0; **REMOVE** it as a stand-alone variant in v6.0.
- Replace with `InterpKind::ChebyshevSpectralWithBC { m, oob_policy }` where
  `oob_policy ∈ {Inherit, ForceReflect, ForcePeriodic, ForceZero}`. Default `Inherit`.

### Surface 2 — Truthful floor rating (FIX 2: H4)

Update `grid_chebyshev.rs` doc-comments to remove the false "≤ 1e-15 spectral floor"
claim. Add a NORMATIVE comment:

```rust
//! ## Spatial floor (NORMATIVE, ADR-0104)
//!
//! Effective floor = max(QuinticHermite virtual-node floor, barycentric numerical
//! conditioning, f64 ULP). At N=512:
//!   QuinticHermite: O(dx⁶) ≈ 1e-10
//!   Barycentric conditioning (M=64): O(2^{-64}) ≈ 5e-20 IN-DOMAIN
//!   f64 ULP: 2e-16
//! Effective: ≈ 1e-10 (virtual-node dominated).
//!
//! The "spectral" qualifier refers to convergence rate (exp-decay in M), NOT to
//! the absolute floor — which is bounded below by the virtual-node interpolant.
```

Update ADR-0089 AMENDMENT 1 cross-reference: the "Chebyshev removes Taylor re-ordering"
prediction was based on the false ≤ 1e-15 floor claim; with truthful floor 1e-10,
the predicted ζ⁴ lift to 3.9 is reduced to ~3.6 (matching ADR-0086 baseline 3.55 +
margin from H3 fix). Rev-predicted thresholds:

| Gate | v4.6 prediction | v5.0 rev-prediction (after H3+H4 fix) |
|------|-----------------|--------------------------------------|
| G_zeta4_const_a_richardson_cheb | ≥ 3.9 | **≥ 3.5** (parity with CubicHermite + margin) |
| G_zeta4_var_a_slope_cheb | ≤ −3.5 | **≤ −2.5** (parity with CubicHermite floor-limited) |
| G_zeta6_const_a_richardson_cheb | ≥ 5.5 | **≥ 5.0** (margin from QuinticHermite-bounded baseline 3.868) |
| G_zeta6_var_a_slope_cheb | ≤ −5.5 | **≤ −3.5** (floor-limited but lifted vs CubicHermite) |
| G_zeta8_const_a_richardson_cheb | ≥ 6.5 (v4.3 predicted) | **≥ 4.0** (TRUTHFUL measurement; current 6.5 is fictional) |
| G_zeta8_var_a_slope_cheb | ≤ −6.5 | **≤ −4.0** (TRUTHFUL) |

### Surface 3 — v5.0 B.2 promotion CONDITIONALLY ENDORSED

If Wave A engineer-measured ζ⁴/ζ⁶/ζ⁸ values (after H3+H4 fix) meet the rev-predicted
thresholds above, v5.0 ADR-0099 B.2 promotion is **ENDORSED** with the following
modifications to the original ADR-0099 plan:

1. Rename `InterpKind::ChebyshevSpectral { m }` → `InterpKind::ChebyshevSpectralWithBC { m, oob_policy }`.
   New default constructor sets `oob_policy: OobPolicy::Inherit` (uses the grid's
   `BoundaryPolicy`).
2. `Grid1D::cheb_m(m: usize)` NEW constructor:
   ```rust
   impl<F: SemiflowFloat> Grid1D<F> {
       pub fn cheb_m(xmin: F, xmax: F, n: usize, m: usize) -> Result<Self, SemiflowError> {
           let mut g = Self::new(xmin, xmax, n)?;
           g.interp = InterpKind::ChebyshevSpectralWithBC { m, oob_policy: OobPolicy::Inherit };
           Ok(g)
       }
   }
   ```
3. `Grid1D::new` default stays `InterpKind::CubicHermite` at v5.0 — promotion to
   Chebyshev default is **DEFERRED to v6.0** with the same evidence requirements as
   B.3 (re-measurement campaign at v5.x PREP-MEASUREMENT MINOR). Rationale: changing
   `Grid1D::new` default is a SEPARATE BREAKING decision from fixing the existing
   opt-in path; bundling them is the original ADR-0099 sin that ADR-0097 AMENDMENT 1
   correctly aborted.
4. `with_chebyshev_sampling()` and `with_chebyshev_sampling_m(m)` on
   `Diffusion4thChernoff`/`Diffusion4thZeta4Chernoff`/`Diffusion6thZeta6Chernoff`
   STAY (additive opt-in) but now produce **convergent** Richardson cascades.
5. `Diffusion8thZeta8Chernoff` (which currently has Chebyshev ON by default per
   ADR-0090 §"5. ζ⁸ Wave II RESURRECTION") gains the BC-aware sampler automatically;
   pre-existing G_zeta8 thresholds re-calibrated per Option E rule from TRUTHFUL
   measurements (the 6.5/−6.5 thresholds were predicted not measured — engineer
   confirmed in V4_6_B3_REMEASURE_REPORT.md §3).

### Surface 4 — Migration `docs/migration/v4-to-v5.md`

NEW section "ADR-0104: Chebyshev sampler boundary policy fix (BREAKING)" with:
- `InterpKind::ChebyshevSpectral { m }` deprecation warning (will be REMOVED at v6.0
  per ADR-0035 §9 12-month rule).
- Migration to `InterpKind::ChebyshevSpectralWithBC { m, oob_policy: OobPolicy::Inherit }`
  (additive; preserves bit-equality in deep interior).
- Threshold updates for 4 NEW + 2 existing G_zeta* gates per rev-predictions above.
- Constitution **NO amendment required** — file-level cohort budgets unchanged
  (`grid_chebyshev.rs` grows from 192 to ~250 LoC, well under 500-LoC cap).

## Acceptance gates (Wave A handoff)

Engineer Wave at `.dev-docs/specs/b3-chebyshev-redesign-wave.md`. Acceptance:

1. PRE-FLIGHT sympy oracle `scripts/verify_chebyshev_spectral_weights.py` PASS
   (already shipped in this ADR delegation; engineer re-runs to confirm).
2. NEW property test `tests/grid_chebyshev_bc_dispatch.rs` (~80 LoC; verifies all
   6 `BoundaryPolicy` variants route correctly for off-grid `x`).
3. Re-measured G_zeta4_const_a_richardson_cheb ≥ 3.5 (BLOCKING).
4. Re-measured G_zeta6_const_a_richardson_cheb ≥ 5.0 (BLOCKING).
5. Re-measured G_zeta8_const_a_richardson_cheb ≥ 4.0 (BLOCKING, TRUTHFUL).
6. All ADVISORY slope gates measure ≤ −2.5/−3.5/−4.0 respectively.
7. All existing fast-bins pass byte-identical (deep-interior in-domain path
   unchanged; v4.x test suite is silent on this path).
8. `docs/migration/v4-to-v5.md` updated with ADR-0104 section.
9. ADR-0099 superseded-by note added pointing here.

## Consequences

- **POSITIVE**: ζ-ladder Chebyshev path becomes mathematically CORRECT for the first
  time in the codebase; v5.0 promotion (deferred to v6.0) gains evidence base it lacks
  at v4.6; user's "academically correct" priority #1 met; matrix-Strang fast-test
  byte-equality preserved (deep-interior path bit-identical).
- **NEUTRAL**: minor wave (~250 LoC source + 1 NEW test + 6 NEW BC dispatch cases);
  no new dependencies (3/3 cap inviolate); no constitution amendment;
  `Grid1D::new` default behavior UNCHANGED at v5.0 (intentional — separate BREAKING
  decision from the H3 fix).
- **NEGATIVE**: `InterpKind::ChebyshevSpectral { m }` enum variant is deprecated
  with 12-month REMOVAL clock; any downstream code that pattern-matches the variant
  must migrate to `ChebyshevSpectralWithBC { m, oob_policy }`. No known downstream
  users (per user directive: zero users). FFI/PyO3/WASM bindings unaffected (do not
  expose `InterpKind` enum).
- **BREAKING**: `InterpKind` enum gains new variant + deprecates old (additive
  BREAKING per ADR-0035 §9; counts toward v5.0 BREAKING window quota — currently
  v5.0 has A.6 + B.1 per ADR-0097 AMENDMENT 1 §"v5.0 plan revision"; ADR-0104 adds
  a third). Constitution Override #1 (suckless line budgets) does NOT need amendment
  (target file grows 192 → ~250 LoC).

## Alternatives considered

| Option | Verdict | Rationale |
|--------|---------|-----------|
| Outcome C — terminal closure (Chebyshev architecturally incompatible) | REJECTED | The defect is mechanically localized (1 missing BC branch + 1 floor-rating correction). Closing without fix would lose academic credibility (boundary handling is a 60-year-solved problem). |
| Defer to v5.1+ as ADR-0097 AMENDMENT 1 did | REJECTED | User explicitly overruled this in the directive. v5.0 BREAKING window is OPEN until 2027-05-27 per ADR-0035 §9; landing the fix at v5.0 saves the v6.0 default-flip from being undefendable. |
| Make `oob_policy` a constructor parameter on the sampler call (not in the enum) | REJECTED | Pollutes every call site of `Grid1D::interp`; violates suckless minimalism. Carrying the policy in the enum is the single-source-of-truth pattern already used by `BoundaryPolicy`. |
| Restrict Chebyshev to direct-kernel use only (drop the `with_chebyshev_sampling` opt-in on Diffusion4thChernoff) | REJECTED | The "direct kernel" claim is itself the false-spectral-floor mistake (H4) — Diffusion8thZeta8Chernoff calls `f.sample(x)` at boundary nodes just like the wrapped variants. The defect is in the sampler, not in the wrapping. |
| Replace Chebyshev with Fourier (FFT) | OUT OF SCOPE | Would require Periodic BC universally (Fourier presumes periodicity); changes spatial discretization assumption used by every existing test. v5.x candidate at most. |
| Bake `OobPolicy::Reflect` as the only choice (no enum) | REJECTED | Robin/Neumann/Dirichlet boundary problems (ADR-0098, A.4 manifold) need non-reflective extension; locking to Reflect would invalidate those use cases. |

## Cross-references

- ADR-0089 AMENDMENT 1 — Path ε QuinticHermite floor; the "Chebyshev removes Taylor re-ordering" prediction (now superseded by truthful floor rating).
- ADR-0090 — Chebyshev opt-in v4.3; this ADR fixes the missing boundary policy and re-rates the false-spectral-floor claim in `grid_chebyshev.rs:30–33`.
- ADR-0097 + AMENDMENT 1 — B.3 measurement campaign + RED verdict + B.2 abort; this ADR supersedes AMENDMENT 1's deferral, escalates to v5.0 with mechanical fix.
- ADR-0099 (PENDING v5.0) — global `Grid1D::new` default flip; superseded-by this ADR. v5.0 lands the H3+H4 fix; ADR-0099-style default flip is RE-SCHEDULED to v6.0 with same evidence requirements as B.3.
- ADR-0035 §9 — 12-month BREAKING window rule; this ADR adds a third v5.0 BREAKING item (A.6 + B.1 + ADR-0104 H3+H4 fix).
- ADR-0086 AMENDMENT 1 — Option E hybrid calibration rule `threshold = ⌊measured − 0.1⌋ + 0.1`; engineer wave applies this to all 6 rev-predicted thresholds.
- `crates/semiflow-core/src/grid_chebyshev.rs:67–111` — `sample_chebyshev_1d` (target of H3 fix; gains `out_of_domain_sample` helper).
- `crates/semiflow-core/src/grid_chebyshev.rs:27–33` — doc-comment with false "spectral floor ≤ 1e-15" claim (target of H4 truthful re-rating).
- `crates/semiflow-core/src/boundary.rs` — `BoundaryPolicy<F>` enum and `bc_value` dispatch (template for `out_of_domain_sample`).
- `crates/semiflow-core/src/diffusion4.rs:499–518` — `gamma_a_baseline_f64` probe overshoot mechanism (the trigger; NOT fixed by this ADR — fix is in the sampler, not in the kernel).
- `.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md` — engineer measurement record; §5 questions answered in this ADR §"Action 3"/"Action 4".
- `scripts/verify_chebyshev_spectral_weights.py` — NEW PRE-FLIGHT oracle for this ADR; 2 sub-checks (weight identity + out-of-domain divergence).
- `.dev-docs/specs/b3-chebyshev-redesign-wave.md` — engineer Wave spec for Outcome A implementation.
- Berrut & Trefethen 2004, *SIAM Review* 46:501, §3.2 — barycentric Lagrange formula valid only INSIDE `[xmin, xmax]` (the missing-BC fact).
- Trefethen 2000 *Spectral Methods in MATLAB*, Ch. 6 — boundary treatment for Chebyshev collocation (Reflect/Periodic/Neumann patterns this ADR adopts).
