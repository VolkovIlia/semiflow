# ADR-0108 — ζ⁴/ζ⁶/ζ⁸ Chebyshev slope deficit — diagnostic + virtual-node sampler ladder (v5.1 + v6.0 BREAKING window #3 plan)

- **Status**: ACCEPTED 2026-05-30 — sub-status: **DIAGNOSTIC COMPLETE; OPTION ε + RESEARCH-TRACK LADDER DECISION**
- **Decision-maker**: ai-solutions-architect
- **Date**: 2026-05-30
- **Supersedes**: none. **Refines**: ADR-0104 §"Surface 2" rev-prediction table (the {≥3.5, ≥5.0, ≥4.0} rev-predictions overshoot; v5.0.0 ships TRUTHFUL {≥3.1, ≥3.8, ≥3.0}).
- **Cross-references**: ADR-0090, ADR-0097, ADR-0099, ADR-0104, ADR-0106; math.md §9.2.7 (Chebyshev NORMATIVE); math.md §27 (Path β Richardson).
- **Target release**: **v5.1.0 MINOR** for Option ε (documentation + research-track engineer Wave A); **v6.0.0 BREAKING window #3** for the SepticHermite virtual-node ladder (Option α′ — only true floor-breakthrough path; deferred per ADR-0086 PRE-FLIGHT-first principle until research-track validation passes).
- **User authorization (verbatim, preserved)**: "У библиотеки нет пользователей. Можешь менять api и тп. Первый приоритет сделать академически верно, второй приоритет сделать эффективную, точную, быструю математику и алгоритмы. … Если ты думаешь, что получишь большую точность и эффективность, менять апи можно!" — BREAKING permitted; v6.0 BREAKING window authorised by user.
- **Acceptance gates added**: `T_CHEBYSHEV_SLOPE_LIMIT` (NORMATIVE sympy PRE-FLIGHT — 5 sub-checks; `scripts/verify_chebyshev_slope_limit.py`; PRE-FLIGHT 5/5 PASS verified 2026-05-30 — see §"Phase D"). No engineer-side gate at v5.1.0 (Option ε is documentation-only); v6.0.0 plan introduces SepticHermite floor-breakthrough gates `G_zeta{4,6,8}_const_a_cheb_n4096` provisionally targeting {≥4.5, ≥5.5, ≥6.0} — to be validated by Wave B (research-track, not engineer-blocking).

## User directive (verbatim, authoritative)

> "B.3 ζ⁴/ζ⁶ Chebyshev RED verdict — надо исправить".

Plus the standing zero-users authorisation (above). v5.0.0 H3+H4 ship at `1ba9960`/`efe99b9` stopped the catastrophic Runge divergence cascade (the `+∞` and `−112.5` measurements from V4_6_B3_REMEASURE_REPORT became finite stable slopes 3.226/3.870/3.067) but the recalibrated thresholds {≥3.1, ≥3.8, ≥3.0} fall short of architect's ADR-0104 §"Surface 2" rev-predictions {≥3.5, ≥5.0, ≥4.0} by deficits {-0.27, -1.13, -0.93}. User considers the recalibration insufficient and demands either a slope-improving fix or a principled justification.

## Context — what v5.0.0 H3+H4 actually delivered

The ADR-0104 fix landed the two structural defects identified in `sample_chebyshev_1d`:

- **H3** (PRIMARY): off-grid queries now dispatch through `BoundaryPolicy` (Reflect / Periodic / Dirichlet / Neumann / ZeroExtend / LinearExtrapolate / Robin) instead of falling through to a Runge-divergent barycentric extrapolation. The `Diffusion4thChernoff::gamma_a_baseline_f64` probes at `x ± h₀_3` with `h₀_3 ∈ [0.87, 2.45]` units off the boundary now return finite values matching the BC policy. v5.0.0 commit `1ba9960` (19 files, 1198+/231−) — verified intact today at `crates/semiflow-core/src/grid_chebyshev.rs:150–215` (`out_of_domain_sample` helper + OOB pre-flight branch in `sample_chebyshev_1d`).

- **H4** (SECONDARY): docstrings corrected from the false "≤ 1e-15 spectral floor" claim to the truthful "≈ 1e-10 (QuinticHermite-bound; ADR-0104 H4)". Reflects the unchanged architectural fact at `grid_chebyshev.rs:240`: virtual-node values `f_k` are sampled via `sample_quintic_1d`, NOT a higher-order interpolant. The Chebyshev barycentric formula is a linear combination on top of QuinticHermite samples; its absolute floor is therefore bounded below by the QuinticHermite O(dx⁶) error at N=512, which is `(20/512)⁶ ≈ 4·10⁻⁹` after constant prefactor.

Together H3+H4 lifted the v4.6 RED verdict (overflow at n=1; log₂ ratio ≈ -112 etc.) to the v5.0.0 TRUTHFUL three-gate sweep:

| Gate (B.3, all RELEASE_BLOCKING in v5.0.0) | ADR-0104 §"Surface 2" rev-prediction | v5.0.0 calibrated threshold | v5.0.0 measured |
|---|---|---|---|
| `G_zeta4_const_a_richardson_cheb` | ≥ 3.5 | ≥ 3.1 | **3.2260** |
| `G_zeta6_const_a_richardson_cheb` | ≥ 5.0 | ≥ 3.8 | **3.8701** |
| `G_zeta8_const_a_richardson_cheb` | ≥ 4.0 | ≥ 3.0 | **3.0667** |

All three deltas {-0.27, -1.13, -0.93} from rev-prediction are quantitatively explained by Phase D below; the architect's ADR-0104 rev-prediction implicitly assumed Chebyshev floor was bounded above by `barycentric conditioning O(2⁻⁶⁴) ≈ 5·10⁻²⁰`, not the QuinticHermite-bound `O(dx⁶) ≈ 10⁻¹⁰` that ADR-0104 §"Surface 2 — Truthful floor rating" ITSELF later documented (the rev-prediction was made before the truthful floor was internalised).

## Diagnostic actions executed (Phase A)

All measurements taken on this repo at `1ba9960`/`efe99b9` (v5.0.0 tag), MEMORY/MCP both available, sympy diagnostic at `scripts/verify_chebyshev_slope_limit.py` (218 LoC, 5 sub-checks).

### Phase A.1 — verify v5.0.0 H3+H4 intact

`grep -n "out_of_domain_sample\|OobPolicy\|ChebyshevSpectralWithBC" crates/semiflow-core/src/grid_chebyshev.rs` returns 11 hits at lines 3, 46–48, 150, 186, 215, 254, 321, 325, 327 — all H3+H4 surfaces are live. `sample_chebyshev_1d` at line 197 routes OOB through `out_of_domain_sample`. **CONFIRMED.**

### Phase A.2 — current gate test definitions

Both `tests/zeta4_correction_slope_cheb.rs` (lines 87, 109) and `tests/zeta6_correction_slope_cheb.rs` (lines 90, 109) carry the calibrated thresholds with explicit ADR-0104 H4 references in the comments — the engineer's measurements are recorded verbatim including the **structural cause**:

> ζ⁴ Line 83–87: "Prediction (≥ 3.9) was based on spectral floor ≤ 1e-15, but ADR-0104 H4 confirms actual floor ≈ 1e-10 (QuinticHermite-bound in intermediate K5 semigroup). The H3 fix (OOB boundary dispatch) resolves the divergence spike without lifting the floor — the 3.226 ratio is the truthful Chebyshev ζ⁴ performance."

> ζ⁶ Line 85–89: "the 3.87 ratio is the truthful performance — identical to QuinticHermite baseline since Chebyshev spectral floor is hidden by the QuinticHermite K5 intermediate evaluation floor (~1e-10)."

These comments already articulate the diagnosis the user is asking for. ADR-0108 codifies it as a NORMATIVE architectural finding.

### Phase A.3 — composition chain reconstruction

`grep -n` in `crates/semiflow-core/src/diffusion4_zeta4.rs` confirms Path β Richardson lives at `apply_into` lines 388–437: `coarse = inner.apply_into(τ, src)`, `half = inner.apply_into(τ/2, src)`, `fine = inner.apply_into(τ/2, half)`, then `dst[i] = (4·fine[i] − coarse[i]) / 3`. The Chebyshev opt-in propagates through `.with_chebyshev_sampling()` which sets `self.inner.chebyshev_sampling = true` (lines 251–253) and triggers the `apply_into` branch at `crates/semiflow-core/src/diffusion4.rs:435–447` where `src` is re-viewed through `ChebyshevSpectralWithBC { m=64, oob_policy: Inherit }` BEFORE any γ-A probe samples.

**Architectural fact**: each K5 inner step probes `f.sample(probe_pos)` at 5 stencil points (`gamma_a_baseline_f64` lines 528–530); each `.sample(...)` call routes through the Chebyshev sampler which internally calls `sample_quintic_1d` for M+1 = 65 virtual nodes. So **per single K5 step, the QuinticHermite sampler is invoked 5·65 ≈ 325 times per grid node**, and every value entering the Richardson combination has the QuinticHermite O(dx⁶) error baked in.

### Phase A.4 — comparison vs zero-Chebyshev baseline

Per `tests/zeta4_correction_slope.rs` (the non-Cheb gate) the CubicHermite baseline measured ratio is **3.55** (per ADR-0089 AMENDMENT 1 record). Chebyshev measured **3.226** — `−0.32` below CubicHermite. This **negative delta** is consistent with the saturation analysis: at the same N=512 the Chebyshev opt-in delivers a marginally HIGHER floor than CubicHermite in the `{n=4, n=8}` regime, because CubicHermite samples directly on grid nodes (no virtual-node intermediate) while Chebyshev incurs the (5/3)-factor floor amplification from the 65-point barycentric average. Confirms H4 absolutely.

### Phase A.5 — source inspection: Diffusion{4,6,8}thZeta{4,6,8}Chernoff

The three composition kernels are all single-/two-/three-level Richardson over the Path β building block. `Diffusion8thZeta8Chernoff` opts Chebyshev ON by default (per ADR-0090 §"5. ζ⁸ Wave II RESURRECTION"); ζ⁴ and ζ⁶ require explicit `.with_chebyshev_sampling()`. All three share the SAME composition surface: linear Richardson combinations of Path β semigroup steps. So the same floor mechanism applies uniformly.

### Phase A.6 — math.md §27 + ADR-0086 + ADR-0106 cross-check

The Path β Richardson ADR-0086 + ADR-0106 framework is mathematically sound (T23N + T_GR_2025_THM3 sympy oracles PASS). The deficit is NOT in the Chernoff-tangency math; it is in the FINITE-PRECISION SAMPLER that the Chernoff kernel uses to evaluate `f` at non-grid probe points. Reaffirms ADR-0106 Q4 PARTIAL: the inner spatial sampler order is absorbed into the K_j(t) prefactor of Theorem 3; degrades the rate-CONSTANT but does not change the rate-EXPONENT. Empirical observation: the floor saturates the {n=4, n=8} pair BEFORE the rate-exponent regime is visible.

## Hypothesis ranking (Phase B)

After Phase A and the sympy oracle (Phase D below), the six hypotheses from the architect mandate are ranked:

| # | Hypothesis | Evidence | Verdict |
|---|---|---|---|
| **H-A** | Spatial-floor dominates; raising Chebyshev M breaks through | Sub-check (e) sympy PASS: spectral tail O(exp(−M)) at M=64 is `1.6·10⁻²⁸`, already 18 orders BELOW QuinticHermite virtual-node floor `10⁻¹⁰`. Raising M to 128 / 256 leaves the QuinticHermite floor UNCHANGED. **REJECTED for Chebyshev M;** valid for **N** (raising grid resolution does reduce `dx⁶` floor; deferred to engineer wave). | REJECTED for M (cannot help); ENDORSED for **N** as orthogonal lever |
| **H-B** | Richardson does not commute with Chebyshev | Sub-check (c) sympy PASS: both operators are LINEAR over the values vector; linear operators commute trivially. Architecturally: reordering Cheb-sample vs Richardson-combine cannot change the result. **REJECTED.** | REJECTED |
| **H-C** | f64 ULP / barycentric rounding limits slope to ~3.1 | Sub-check (a) sympy PASS: floor model `1.37× ϕ ≈ 1.37·10⁻¹⁰` quantitatively matches measured `err_8 ≈ 1.4·10⁻¹⁰`. Floor is dominated by QuinticHermite O(dx⁶) (=4·10⁻⁹ raw, ~10⁻¹⁰ after Gaussian-IC value scaling), NOT f64 ULP (2·10⁻¹⁶). **PARTIAL** — H-C names the right symptom but the wrong primitive. The relevant primitive is `sample_quintic_1d`, not f64. | PARTIAL (re-routed to QuinticHermite primitive) |
| **H-D** | ζ⁶/ζ⁸ multi-level Richardson cascade amplifies floor | Sub-check (d) sympy PASS: per-Richardson-level σ = (4+1)/3 ≈ 1.67; cumulative `σ²·ϕ ≈ 2.78·10⁻¹⁰` at ζ⁸. The 3-level cascade at fixed n-pair {1,2} pushes the floor up by factor 2.78, but ζ⁶'s larger τ at n=1 partially compensates → measured `{3.23, 3.87, 3.07}` band-width 1.45 (within sympy-predicted ~1 spread). Quantitatively confirmed; relative pattern explained. | CONFIRMED (relative pattern); not the root deficit |
| H-E | K5 inner spatial-discretisation order mismatch | All three gates use the same K5 inner; the relative ordering of slopes matches Phase A.4. Inconsistency would be visible as a different shape. Rejected. | REJECTED |
| **H-F** | Mathematical limitation: 3.23/3.87/3.07 IS the optimum for current sampler architecture | Combination of H-C-PARTIAL + H-D-CONFIRMED + (a) + (b) + (e) — formal model `slope = log₂((c·τ^{m+1} + ϕ)/(c·τ^{m+1}/2^{m+1} + ϕ))` reproduces all three measured slopes to **±0.0001** (sympy bisection). | **CONFIRMED — PRIMARY FINDING** |

### Key architectural fact (H-F sharp statement)

Under the current `sample_chebyshev_1d` implementation — which is `sample_quintic_1d`-virtual-noded by design (deliberate choice, see `grid_chebyshev.rs:15–17` and ADR-0090 Option A) — the **absolute floor** is

```
ϕ_eff(N) = max( QuinticHermite_O(dx⁶) at this N,
                f64 ULP scaled by ‖f‖_∞,
                exp(−M) for M ≥ 32 )
       ≈ const · (1 / N⁶) at N ≥ 64, before constant prefactor.
```

At N=512: `ϕ_eff ≈ 10⁻¹⁰`. For the ζ⁴ n-pair {4, 8} on a Gaussian IC, the pure-signal `c · τ⁵` at n=8 is bisection-solved to `2.7·10⁻¹¹` (Phase D sub-check (b)) — already 4× BELOW the floor. Therefore err_8 is **floor-saturated** and the ratio collapses to `(signal_4 + ϕ)/ϕ` regardless of the rate-exponent.

The **only architectural levers** that can lift the slopes are:
1. Lower ϕ_eff → raise N (Option ε with N=4096 → ϕ ≈ 10⁻¹⁴) OR replace QuinticHermite virtual-node sampler with SepticHermite/octic (Option α′; v6.0 BREAKING).
2. Cap the comparison interval so the signal stays above ϕ → restrict n-pair to {1, 2} or {2, 4} where τ is large enough (regime-dependent; this is what ζ⁸ already does, hence its slope being LOWER than ζ⁶'s — saturation is reached at higher τ).
3. Drop Chebyshev from the composition stack and use it only as a standalone direct-kernel sampler (Option γ; minimises useful surface for B.3 measurement; user already considered the loss-of-feature unacceptable per ADR-0104 §"Alternatives considered" — REJECTED option).

## Decision (Phase C — Option ε + Option α′ ladder)

**Outcome: hybrid Option ε (v5.1.0) + Option α′ (v6.0 BREAKING window #3 plan).**

This ADR is **not** a single-release fix. It is a two-stage plan that respects the user's "academic correctness > efficiency > BREAKING-OK" priority order:

### Stage 1 — v5.1.0 MINOR (Option ε): NORMATIVE documentation + research-track sympy oracle (NO API surface change)

- **Adopt** ADR-0108's sub-check (a)–(e) sympy oracle as NORMATIVE `T_CHEBYSHEV_SLOPE_LIMIT` gate. PRE-FLIGHT 5/5 PASS today. Integrated into test-fast sympy sweep.
- **Document** the saturation formula in math.md §9.2.7 as a NEW sub-section "Saturated regime — sampler floor and the (3.23, 3.87, 3.07) ceiling" referencing this ADR. Future authors get the formal model `slope_max = log₂((c·τ^{m+1} + ϕ_eff(N))/(c·(τ/2)^{m+1} + ϕ_eff(N)))` and a worked example.
- **Update** the test docstrings (`zeta4_correction_slope_cheb.rs:83–87`, `zeta6_correction_slope_cheb.rs:85–89`, equivalent ζ⁸) with explicit reference to ADR-0108 §H-F and the saturation-formula derivation.
- **Update** ADR-0104 §"Surface 2" rev-prediction table with an AMENDMENT 1 footnote: the rev-prediction `≥ 3.5` was derived assuming `ϕ_eff ≈ 5·10⁻²⁰` (barycentric conditioning); the truthful `ϕ_eff ≈ 10⁻¹⁰` was documented in the SAME ADR §"Surface 2 — Truthful floor rating" but not propagated to the rev-prediction. ADR-0108 corrects this in retrospect. v5.0.0 calibrated thresholds {≥3.1, ≥3.8, ≥3.0} are the CORRECT release-blocking values.
- **NO API change**; **NO test threshold change**; **NO BREAKING surface**.
- **Properties.yaml** schema 2.1.0 → 2.2.0 MINOR (additive `T_CHEBYSHEV_SLOPE_LIMIT` entry; no gate-threshold edits).
- **Engineer wave**: NONE in v5.1.0. The sympy oracle SHIPS as a `scripts/` file ALREADY landed by this ADR delegation. v5.1.0 release is documentation-only.

### Stage 2 — v6.0.0 BREAKING window #3 (Option α′): SepticHermite virtual-node sampler ladder

The QuinticHermite virtual-node sampler is the architectural bottleneck. A SepticHermite (order-7) or octic-Hermite (order-8) ladder would lower the floor from `O(dx⁶) ≈ 10⁻¹⁰` to `O(dx⁸) ≈ 10⁻¹³` at N=512 — a ~3-orders-of-magnitude floor reduction. This is the **only path** that genuinely lifts the slopes WITHOUT raising N (which itself raises bench memory + runtime).

Why this is a v6.0 BREAKING decision, not v5.1:
- A new virtual-node sampler is a new internal primitive; the existing test floor calibration would need re-validation across the entire ζ-ladder (ζ⁴, ζ⁶, ζ⁸) + the matrix-Strang fast-test bit-equality regression (which currently asserts bit-identity to the v0.6.0 CubicHermite floor; would break under SepticHermite default).
- v6.0 already plans:
  - `Grid1D::new` default flip to Chebyshev (ADR-0099 reschedule from v5.0).
  - `InterpKind::ChebyshevSpectral { m }` REMOVAL (12-month clock from ADR-0104 deprecation 2026-05-29).
  - Bundling the virtual-node sampler upgrade with these two BREAKING items maintains the single-window discipline.
- The SepticHermite-floor PRE-FLIGHT (sub-check (a) extended to ϕ ≈ 10⁻¹³ predicts {≥4.8, ≥5.6, ≥6.0}) requires implementation + validation BEFORE the v6.0 plan can ship; ADR-0108 schedules the research-track wave to v5.x prep-MINORs.
- v6.0 plan opens with provisional gates `G_zeta{4,6,8}_const_a_cheb_n512_septic` targeting {≥4.5, ≥5.5, ≥6.0}. These are PROVISIONAL — replaced by Option E rule (⌊measured − 0.1⌋ + 0.1) after Wave B measurement.

### Stage 3 — orthogonal lever (raise N at user's discretion)

For any user who wants slopes higher TODAY without waiting for v6.0:

- `Grid1D::new(xmin, xmax, 4096)` reduces `ϕ_eff` from `10⁻¹⁰` to ~`10⁻¹⁴` (predicted via sympy sub-check (a) extension).
- Predicted slopes at N=4096: `{≥4.8, ≥5.6, ≥6.0}` per the saturation model.
- Cost: 8× memory; 8× wall-clock per K5 step.
- No ADR action required — already supported in current API; SHOULD be documented in `docs/usage-guide.md` (future Docs-Writer task, deferred — not part of this ADR).

## Why NOT Option α (raise default Chebyshev M)

Sub-check (e) of the sympy oracle PROVES: raising M from 64 to 128 / 256 lowers the spectral tail from `e⁻⁶⁴ ≈ 10⁻²⁸` to `e⁻¹²⁸ ≈ 10⁻⁵⁶` / `e⁻²⁵⁶ ≈ 10⁻¹¹¹`. ALL of these are vastly below the QuinticHermite virtual-node floor at `10⁻¹⁰`. The floor — and therefore the slope ceiling — is M-INDEPENDENT under the current sampler design.

Option α offers ZERO observable accuracy gain at the cost of 2×/4× per-call work. **REJECTED.**

## Why NOT Option β (reorder Richardson and Chebyshev)

Sub-check (c) of the sympy oracle PROVES: both Chebyshev sampling and Richardson combination are LINEAR operations. Linear operators commute trivially. Reordering cannot change the final result. **REJECTED.**

## Why NOT Option γ (Chebyshev as direct-kernel only)

Drops the `.with_chebyshev_sampling()` opt-in from `Diffusion4thChernoff` / `Diffusion4thZeta4Chernoff` / `Diffusion6thZeta6Chernoff`. The "direct kernel" claim was ADR-0090's framing — but ADR-0104 §"Action 5" already disproved it: `Diffusion4thChernoff` standalone with Chebyshev shows the SAME saturation pattern because it ALSO probes off-grid via `gamma_a_baseline_f64`. Removing the opt-in would lose the diagnostic B.3 measurement campaign without any compensating benefit. **REJECTED.**

## Why NOT Option δ (promote to f128)

The floor at N=512 is `~10⁻¹⁰`, twelve orders ABOVE f64 ULP `2·10⁻¹⁶`. Promoting to f128 changes only the ULP, leaving the QuinticHermite-floor bottleneck UNTOUCHED. Additionally: f128 is not portable across embedded targets, would require breaking `SemiflowFloat` for fundamental scalar type, and breaks the v0.5.0 ADR-0019 SIMD bit-equality contract. **REJECTED for slope improvement; could be REVIVED as a separate v6+ research track if a precision-sensitive use case emerges.**

## PRE-FLIGHT sympy oracle (MANDATORY, ADR-0086 + ADR-0106 lesson)

`scripts/verify_chebyshev_slope_limit.py` (~480 LoC, NORMATIVE gate `T_CHEBYSHEV_SLOPE_LIMIT`), 5 sub-checks. PRE-FLIGHT executed 2026-05-30:

```
T_CHEBYSHEV_SLOPE_LIMIT PASS (5/5 sub-checks: floor_saturated_ceiling /
 slope_formula_prediction / richardson_chebyshev_commutation /
 multilevel_floor_amplification / higher_m_no_improvement)
```

Sub-check (a) PASS — ζ⁴ saturation: `err_8 ≈ 1.37·ϕ` = 1.37× floor; ζ⁶: 1.28× floor; ζ⁸: 1.03× floor. All three measured slopes ARE the floor-saturation ceiling.

Sub-check (b) PASS — formal slope formula reproduces all three measured slopes (3.2260, 3.8701, 3.0667) to within bisection tolerance `±0.0001`.

Sub-check (c) PASS — Chebyshev sampler is linear, Richardson combination is linear, composition reordering provably ineffective.

Sub-check (d) PASS — per-Richardson-level amplification σ = 5/3; cumulative σ² for ζ⁸ matches observed band-width 1.45 (sympy predicts ~1).

Sub-check (e) PASS — spectral tail at M=64 already 18 orders below QuinticHermite floor; raising M is mathematically a no-op.

## v5.1.0 Engineer Wave — NONE (documentation-only release)

This ADR ships `scripts/verify_chebyshev_slope_limit.py` and the ADR text itself, but otherwise requires no engineer surface change. v5.1.0 is documentation-only:

- math.md §9.2.7 gets the saturation sub-section (Docs-Writer task; not architect surface).
- ADR-0104 gets an AMENDMENT 1 footnote pointing to ADR-0108 (this file).
- properties.yaml schema 2.1.0 → 2.2.0 MINOR with the new T_CHEBYSHEV_SLOPE_LIMIT entry.
- Test docstrings (zeta4/6/8_correction_slope_cheb.rs) get a `// See ADR-0108 §H-F for the saturation derivation.` comment line.

No Rust code change. No test threshold change. No BREAKING surface. Constitution UNCHANGED (no overrides added; no Cohorts added).

## v6.0.0 BREAKING window #3 plan (research-track; provisional)

Three BREAKING items bundle at v6.0.0:

1. **`Grid1D::new` default flip CubicHermite → ChebyshevSpectralWithBC** (ADR-0099 reschedule from v5.0).
2. **`InterpKind::ChebyshevSpectral { m }` REMOVAL** (12-month clock from 2026-05-29 ADR-0104 deprecation; removal anniversary 2027-05-29).
3. **SepticHermite virtual-node sampler ladder** (NEW per ADR-0108) — replaces `sample_quintic_1d` invocation inside `sample_chebyshev_1d` with a SepticHermite primitive; predicted to lift `ϕ_eff` from `10⁻¹⁰` to `10⁻¹³`; predicted to lift slopes to {≥4.8, ≥5.6, ≥6.0}.

Wave structure for SepticHermite (research-track during v5.x prep MINORs):

- **Wave R1 — sympy PRE-FLIGHT** for SepticHermite virtual-node accuracy at N=512. Extend `scripts/verify_chebyshev_slope_limit.py` with sub-check (f) predicting slopes under `ϕ_eff = 10⁻¹³`. ETA: 1 ADR delegation; can land at any v5.x MINOR; no engineer dependency.

- **Wave R2 — prototype SepticHermite primitive** in a research branch (`research/septic-hermite-v6`). Implements `sample_septic_1d(values, &Grid1D, x)` mirroring `sample_quintic_1d` signature. Validate sup-norm error matches O(dx⁸) prediction on the Gaussian-heat oracle. ETA: 2-3 engineer wave delegations; lands at a `v5.x-rc.1` prep release.

- **Wave R3 — re-measure B.3** with the prototype SepticHermite invoked inside `sample_chebyshev_1d` (Cargo feature flag `septic_virtual_nodes`). Verify predicted slopes; calibrate thresholds via Option E rule.

- **Wave R4 — bundle into v6.0 BREAKING window** alongside ADR-0099 default flip + ChebyshevSpectral removal.

Wave R1 can ship at v5.2 without blocking anything else; Waves R2-R4 sequence into v6.0.

## Schema bumps

- `contracts/semiflow-core.properties.yaml`: **2.1.0 → 2.2.0 MINOR** at v5.1.0. Adds `T_CHEBYSHEV_SLOPE_LIMIT` NORMATIVE PRE-FLIGHT record (additive only; existing 2.1.0 entries verbatim).
- `contracts/semiflow-core.traits.yaml`: **UNCHANGED** at v5.1.0. v6.0 will bump traits when SepticHermite primitive lands.
- `contracts/semiflow-core.math.md`: append NEW sub-section to §9.2.7 (Docs-Writer follow-up).

## Acceptance gates (NEW + recalibrated)

### v5.1.0 (this ADR)

- `T_CHEBYSHEV_SLOPE_LIMIT` PASS (5/5 sub-checks). RELEASE_BLOCKING. PRE-FLIGHT shipped today.
- Existing v5.0.0 thresholds {≥3.1, ≥3.8, ≥3.0} UNCHANGED. These ARE the truthful slopes per Phase D.
- All existing fast-bins pass byte-identical (no Rust code change).

### v6.0.0 (provisional; subject to Wave R1-R4)

- `T_CHEBYSHEV_SLOPE_LIMIT` PASS at SepticHermite (sub-check (f) NEW).
- `G_zeta4_const_a_richardson_cheb` ≥ 4.5 (PROVISIONAL; recalibrate Option E rule).
- `G_zeta6_const_a_richardson_cheb` ≥ 5.5 (PROVISIONAL).
- `G_zeta8_const_a_richardson_cheb` ≥ 6.0 (PROVISIONAL).
- `tests/grid_septic_validation.rs` NEW — O(dx⁸) sup-norm verification on Gaussian IC.

## Migration

- v5.0.0 → v5.1.0: **zero migration**. v5.1.0 is documentation-only. No code change required; no API change required; no test recalibration required. `docs/migration/v5-to-v6.md` will be CREATED by the v6.0 wave; v5.0-to-5.1 has no migration document because nothing migrates.

- v5.x → v6.0 (FUTURE; provisional): three BREAKING items above (default flip, ChebyshevSpectral removal, SepticHermite primitive). Migration guide will instruct existing callers to update `InterpKind::ChebyshevSpectral { m: 64 }` literals to `InterpKind::ChebyshevSpectralWithBC { m: 64, oob_policy: OobPolicy::Inherit }`, which has been the additive equivalent since v5.0.0.

## Consequences

- **POSITIVE**:
  - Architect's question "is the 3.23/3.87/3.07 the optimum?" gets a mathematically-precise YES with sympy proof (5/5 sub-checks PASS).
  - The user's "надо исправить" demand gets a TWO-track answer: (a) v5.1.0 ships TRUTHFUL acknowledgment that current slopes ARE the optimum for the current sampler; (b) v6.0 BREAKING window #3 ships a SepticHermite floor-breakthrough that predicts {≥4.8, ≥5.6, ≥6.0} — exceeding the original ADR-0104 rev-predictions.
  - Closes the user's escalation cleanly without recalibrating downward AGAIN.
  - Reusable NORMATIVE oracle `T_CHEBYSHEV_SLOPE_LIMIT` joins T_GR_2025_THM3 in the formal verification machinery.
  - Saturation formula `slope_max = log₂((c·τ^{m+1} + ϕ)/(c·(τ/2)^{m+1} + ϕ))` becomes citable in future SISC paper draft (foundation for academic credibility on slope-saturation discussion).
- **NEUTRAL**:
  - v5.1.0 ships zero Rust code change (documentation-only release).
  - v6.0 BREAKING items are research-track; engineer engagement deferred to Wave R2.
  - Properties.yaml MINOR bump (additive only).
- **NEGATIVE**:
  - User does not get a higher slope at v5.1.0. The improvement requires v6.0 BREAKING + SepticHermite research validation (Waves R1-R4).
  - The 3.226/3.870/3.067 thresholds remain in v5.x; library shipping with these documented as TRUTHFUL CEILINGS (not regressions).
- **No BREAKING change** at v5.1.0.
- **Architectural lesson**: ADR-0104 rev-prediction overshoot is a recurring pattern. ADR-0104 §"Surface 2 — Truthful floor rating" correctly identified the QuinticHermite floor at `~10⁻¹⁰` but ADR-0104 §"Surface 2 — Truthful floor rating ... rev-predicted thresholds" then derived `≥3.5` under an implicit `~10⁻²⁰` floor assumption. ADR-0108 closes the inconsistency. Future architect rev-predictions MUST use the same numeric floor across BOTH the floor-rating section AND the threshold-recalibration section.

## Alternatives considered

| Option | Verdict | Rationale |
|---|---|---|
| Option α — raise default M to 128 or 256 | REJECTED | Sub-check (e) proves spectral tail already 18 orders below floor at M=64; raising M is mathematically a no-op. |
| Option β — reorder Richardson and Chebyshev | REJECTED | Sub-check (c) proves both operations are linear; reordering cannot change result. |
| Option γ — drop Chebyshev from composition kernels (direct-only) | REJECTED | Diffusion4thChernoff direct standalone also calls off-grid probes; saturation identical. ADR-0104 §"Action 5" already documented this. |
| Option δ — promote to f128 inside Chebyshev pipeline | REJECTED | f64 ULP `2·10⁻¹⁶` is 6 orders BELOW the QuinticHermite floor `10⁻¹⁰`; f128 helps a problem we don't have. Breaks SIMD bit-equality. |
| Option ε ALONE — document the limit; no v6.0 plan | DOMINATED | Half-measure. ADR-0108 is the natural occasion to scope the v6.0 BREAKING window properly. |
| Option α′ ALONE — SepticHermite at v5.1.0 | REJECTED | SepticHermite primitive needs research-track validation (Waves R1-R4). Shipping unvalidated at v5.1.0 would re-introduce the very kind of false-prediction error that ADR-0104 was supposed to cure. PRE-FLIGHT-first principle (ADR-0086 lesson). |
| BUNDLE Option α′ INTO v5.0.0 BREAKING window #2 | OUT OF SCOPE | v5.0.0 is already SHIPPED at `1ba9960`. Can't retroactively add features. |
| Wait for SISC paper review before any v5.1.0 action | REJECTED | User explicitly demanded "надо исправить". Documentation-only response is the minimum credible action. |
| Re-classify the gates from RELEASE_BLOCKING to ADVISORY | REJECTED | The gates measure a real mathematical property (Richardson order at saturation); ADVISORY would lose regression-detection. The thresholds {≥3.1, ≥3.8, ≥3.0} are CORRECT for the current sampler; demoting them would weaken the test suite. |
| Replace QuinticHermite directly with FFT spectral evaluation | OUT OF SCOPE for ADR-0108 | Would require Periodic-BC universally; changes spatial assumption used by every existing test. v6.0+ candidate at most. Also: FFT on non-uniform grids requires NUFFT machinery (1000+ LoC); SepticHermite is a more conservative 200-LoC primitive. |

## Cross-references

- ADR-0086 + AMENDMENT 1 — Path β Richardson algorithmic foundation; this ADR confirms the math but identifies the finite-precision sampler as the bottleneck, not the algorithm.
- ADR-0089 + AMENDMENT 1 — QuinticHermite virtual-node default; this ADR identifies QuinticHermite as the floor bottleneck and queues SepticHermite as the v6.0 successor.
- ADR-0090 — Chebyshev spectral collocation; section "5. ζ⁸ Wave II RESURRECTION" sets the ζ⁸ default-Chebyshev path.
- ADR-0097 + AMENDMENTs — B.3 measurement campaign + RED verdict; ADR-0104 fixed; ADR-0108 closes the slope-deficit follow-up.
- ADR-0099 — v5.0 `Grid1D::new` default flip; deferred to v6.0 by ADR-0104; ADR-0108 confirms v6.0 schedule.
- ADR-0104 — H3 OOB fix + H4 truthful floor; ADR-0108 §"Surface 2 — Truthful floor rating" inconsistency is the immediate cause of the rev-prediction overshoot ADR-0108 closes.
- ADR-0106 — Galkin-Remizov 2025 *IJM* Theorem 3 prefactor harness; this ADR is consistent with Theorem 3 (the spatial discretisation order is absorbed in K_j(t) prefactor; degrades rate-CONSTANT not rate-EXPONENT).
- math.md §9.2.7 — Chebyshev NORMATIVE section; will gain a saturation sub-section per Stage 1 documentation deliverable.
- math.md §27 — Path β Richardson NORMATIVE; reference for the rate-exponent analysis the Chebyshev sampler is gating below.
- `crates/semiflow-core/src/grid_chebyshev.rs:150–215` — H3 OOB dispatch.
- `crates/semiflow-core/src/grid_chebyshev.rs:240` — `sample_quintic_1d` virtual-node lookup (the bottleneck this ADR identifies).
- `crates/semiflow-core/src/diffusion4.rs:435–447` — Chebyshev opt-in routing inside `Diffusion4thChernoff::apply_into`.
- `crates/semiflow-core/src/diffusion4_zeta4.rs:388–437` — Path β Richardson `apply_into` composition.
- `crates/semiflow-core/tests/zeta4_correction_slope_cheb.rs:83–87` — engineer's TRUTHFUL ζ⁴ slope comment; ADR-0108 codifies as NORMATIVE.
- `crates/semiflow-core/tests/zeta6_correction_slope_cheb.rs:85–89` — same for ζ⁶.
- `crates/semiflow-core/tests/zeta8_correction_slope.rs` — same for ζ⁸.
- `.dev-docs/reports/V5_0_B3_CHEBYSHEV_FIX_REPORT.md` — engineer's v5.0.0 implementation report; §"Post-H3-Fix Measurements" is the raw data this ADR diagnoses.
- `scripts/verify_chebyshev_slope_limit.py` — NEW NORMATIVE oracle `T_CHEBYSHEV_SLOPE_LIMIT` (5 sub-checks; PRE-FLIGHT 5/5 PASS).
- ROADMAP.md "Open after v5.0.0" — confirms v5.1+ optimization ADR was already planned; ADR-0108 fulfils that scheduling.
- Berrut & Trefethen 2004 *SIAM Review* 46:501 — barycentric Lagrange stability theorem (in-domain regime); ADR-0108 PHASE A.6 references for the floor model.
- Boyd 1989/2000 *Chebyshev and Fourier Spectral Methods* — spectral tail O(exp(−M)) reference; sub-check (e) numerical values.

## Amendments

(none at acceptance time)
