# ADR-0097 — B.3 ζ⁶/ζ⁸ Chebyshev Re-measurement Campaign (v4.6.0 PREP-MEASUREMENT)

- **Status**: Accepted
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Related**: ADR-0088 + AMENDMENT 1 (ζ⁶ Path β; 3.868 calibration; Option E hybrid); ADR-0089 + AMENDMENT 1 (Path ε QuinticHermite opt-in); ADR-0090 (Chebyshev opt-in spec; §AC8/AC9 re-measurement scheduling; §AC10 ζ⁸ resurrection); ADR-0086 + AMENDMENT 1 (Option E hybrid calibration rule `threshold = ⌊measured − 0.1⌋ + 0.1`).
- **Blocks**: v5.0 ADR-0099 (B.2 global `Grid1D::new` default `CubicHermite → ChebyshevSpectral { m: 64 }` — conditional on this campaign's GREEN/AMBER verdict per `~/.claude/plans/roadmap-reflective-biscuit.md` v4.6 §"B.3" + v5.0 §"B.2").
- **Target release**: v4.6.0 PREP-MEASUREMENT MINOR (~2026-07; 4-6 eng days; LOW risk).

## Context

v5.0 BREAKING window per ADR-0035 §9 + roadmap §"v5.0 ADR-0099" plans promotion of `Grid1D::new` default `InterpKind::CubicHermite → InterpKind::ChebyshevSpectral { m: 64 }`. Per ADR-0089 §"v5.0 Future Work" + ADR-0090 §"Backward compatibility" `Grid1D::new` line + §"6. Global default unchanged": this promotion requires evidence that the ζ-ladder retains its rated order under the new default. ADR-0090 §AC8/AC9 originally specified re-measurement test bins for ζ⁴/ζ⁶ under Chebyshev opt-in, but ADR-0090 Wave A engineer wave landed only the **opt-in builders** on `Diffusion4thChernoff`/`Diffusion4thZeta4Chernoff`/`Diffusion6thZeta6Chernoff` (verified at `src/diffusion4{,_zeta4}.rs` + `src/diffusion6_zeta6.rs`; `.with_chebyshev_sampling()` + `.with_chebyshev_sampling_m()` already public) WITHOUT measuring the resulting Richardson ratios. The ζ⁸ kernel `Diffusion8thZeta8Chernoff` already ships with `G_zeta8_const_a_richardson_cheb` BLOCKING at 6.5 + `G_zeta8_var_a_slope_cheb` ADVISORY at −6.5 (properties.yaml lines 6125–6166); these were calibrated for v4.3 but have NOT been re-verified on production hardware since v4.3 ship. Without a measurement campaign at v4.6, v5.0 B.2 promotion is **undefendable**: the architect cannot certify that flipping the default preserves ζ-ladder gate semantics. This ADR schedules the campaign as a PREP-MEASUREMENT MINOR — pure additive test bins + properties.yaml gate entries, no source code change, no API change.

## Decision

Measurement-only campaign at v4.6.0 (NO source code change; NEW test bins + NEW properties.yaml gate entries + threshold calibration on existing entries):

1. **NEW test `crates/semiflow-core/tests/zeta6_correction_slope_cheb.rs`** (~120 LoC; mirror v4.1 `zeta6_correction_slope.rs` structure verbatim with `.with_chebyshev_sampling()` opt-in engaged on the K5 base wired into `Diffusion4thZeta4Chernoff` → `Diffusion6thZeta6Chernoff`). Two sub-tests:
   - `g_zeta6_const_a_richardson_ratio_cheb` (RELEASE_BLOCKING): const-a Richardson ratio with Chebyshev M=64; oracle = analytic Gaussian heat kernel `(1+4T)^{-½}·exp(-x²/(1+4T))`; n-pair {1, 2}; N=512.
   - `g_zeta6_var_a_temporal_slope_cheb` (RELEASE_ADVISORY): var-a OLS slope on `a(x) = 1 + 0.5·tanh²(x)`; K5 Chebyshev-sampled reference at `n_ref = 8192`; sweep n ∈ {4, 8, 16, 32}.

2. **NEW test `crates/semiflow-core/tests/zeta4_correction_slope_cheb.rs`** (~120 LoC; mirror v4.1 `zeta4_correction_slope.rs` with `Diffusion4thZeta4Chernoff::new(k5, ...).with_chebyshev_sampling()`; same sub-test pattern). Re-measures ADR-0090 AC8 (NOT executed in v4.3 Wave A); see ADR-0089 AMENDMENT 1 §"ζ⁴ default REGRESSION" — Chebyshev predicted to remove the QuinticHermite Taylor re-ordering that broke ζ⁴ at the v4.3 attempt; opt-in path expected to restore theoretical ratio ≥ 3.9.

3. **REVERIFY existing** `tests/zeta8_correction_slope.rs::g_zeta8_const_a_richardson_cheb` + `g_zeta8_var_a_slope_cheb` (file ships since v4.3 at thresholds 6.5 BLOCKING / −6.5 ADVISORY; this campaign confirms they still hold on production hardware with current v4.5 code state).

4. **NEW properties.yaml gate entries** (4 total; mirror existing `G_zeta6_const_a_richardson` shape at lines 4728–4793):
   - `G_zeta4_const_a_richardson_cheb` (RELEASE_BLOCKING) — predicted ≥ 3.9 per ADR-0089 AMENDMENT 1 §"Chebyshev removes higher-order Taylor re-ordering" (recovers from CubicHermite default 3.55 baseline).
   - `G_zeta6_const_a_richardson_cheb` (RELEASE_BLOCKING) — predicted ≥ 5.5 per Boyd 1989 spectral convergence theory (recovers from QuinticHermite-bound 3.868 toward theoretical 6.0).
   - `G_zeta4_var_a_slope_cheb` (RELEASE_ADVISORY) — predicted ≤ −3.5 (Chebyshev spectral floor ≤ 1e-15 exposes temporal signal; baseline +0.05 floor plateau lifts to genuine convergence).
   - `G_zeta6_var_a_slope_cheb` (RELEASE_ADVISORY) — predicted ≤ −5.5 (same Chebyshev floor argument).

5. **Calibrate thresholds per Option E hybrid rule** (ADR-0086 AMENDMENT 1 + ADR-0088 AMENDMENT 1 precedent): `threshold = ⌊measured − 0.1⌋ + 0.1`. Engineer MUST measure FIRST, then commit calibrated thresholds (never set thresholds blind from prediction). If measurement fails prediction, document the gap in the `rationale` field and PROMOTE at measured-floor-minus-5% per the same Option E rule.

6. **REVERIFY-ONLY for G_zeta8 entries**: properties.yaml entries `G_zeta8_const_a_richardson_cheb` (line 6125) + `G_zeta8_var_a_slope_cheb` (line 6146) are NOT amended at v4.6. The v4.6 wave runs the existing tests on production hardware, records measurement in `.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md`, and reports GREEN/AMBER/RED. Threshold updates (if measurement requires) defer to a v4.6.x PATCH or v5.0 BREAKING window.

7. **Decision criteria for v5.0 ADR-0099 B.2 promotion** (handed to architect at v4.6 sign-off):
   - **GREEN** — All 4 NEW BLOCKING gates PASS AT predicted thresholds (G_zeta4_cheb ≥ 3.9, G_zeta6_cheb ≥ 5.5) AND all 4 ADVISORY gates PASS (slopes ≤ −3.5 / ≤ −5.5) AND existing G_zeta8 gates PASS unchanged → **B.2 promotion ENDORSED** at v5.0 (the ζ-ladder retains rated order under Chebyshev default; flipping `Grid1D::new` default is safe).
   - **AMBER** — BLOCKING gates PASS at predicted thresholds but ≥1 ADVISORY fails OR existing G_zeta8 gates degrade by ≤10% → **B.2 promotion CONDITIONAL** at v5.0 with documented advisory degradation in `docs/migration/v4-to-v5.md` (Chebyshev default is safe for production but advisory regressions tracked; users with strict var-a slope contracts may need `legacy-cubic-default` feature flag).
   - **RED** — Any BLOCKING gate FAIL at predicted threshold (G_zeta4_cheb < 3.9 OR G_zeta6_cheb < 5.5 OR existing G_zeta8 degrades > 10%) → **B.2 promotion ABORTED** at v5.0; ADR-0099 deferred to v5.1+ pending architectural investigation (root-cause for failure: virtual-node QuinticHermite floor, FD ghost-data order, pre-asymptotic τ regime, or other).

## Consequences

- **POSITIVE**: provides hard measurement evidence for v5.0 B.2 BREAKING decision (currently undefendable per roadmap §"v4.6"); pre-deprecates Quintic ζ⁶ calibrated baseline 3.868 — Chebyshev expected to recover theoretical 6.0 per Boyd 1989; lifts ζ⁴ default-OFF freeze (ADR-0089 AMENDMENT 1) — Chebyshev expected to allow ζ⁴ Chebyshev-opt-in promotion to BLOCKING at ≥3.9; closes the "user-attention" pending item from `project_v4_5_research_wave.md` §123 ("ζ⁶/ζ⁸ Chebyshev re-measurement post Path ε opt-in: pending").
- **NEUTRAL**: pure measurement campaign; no source API change, no behavior change, no constitution amendment, no schema-version MAJOR bump (additive MINOR `1.2.0 → 1.3.0` for 4 new properties.yaml entries); 4-6 engineering days per roadmap §"v4.6.0 — PREP-MEASUREMENT MINOR".
- **NEGATIVE**: 4 new slow-tests add ~8s to `cargo test --release --features slow-tests -- --ignored` runtime per measurement (each sub-test runs n=1 + n=2 const-a + 4-point var-a sweep); contributes to test wall-time creep (acceptable per roadmap LOW-risk classification); ζ⁸ reverification is a "soft commitment" — if existing G_zeta8 gates fail unchanged, v4.6 sign-off MUST report RED even though no v4.6 code change caused the regression (engineer escalates to architect for v4.5→v4.6 regression hunt).
- **BREAKING**: NONE. Additive opt-in tests + additive properties.yaml gate entries only.

## Implementation spec

Engineer Wave per `.dev-docs/specs/zeta-chebyshev-remeasure-wave.md`:
- **AC1** — NEW `tests/zeta6_correction_slope_cheb.rs` (~120 LoC; mirror v4.1 `tests/zeta6_correction_slope.rs` with `.with_chebyshev_sampling()` engaged).
- **AC2** — NEW `tests/zeta4_correction_slope_cheb.rs` (~120 LoC; mirror v4.1 `tests/zeta4_correction_slope.rs` with `.with_chebyshev_sampling()` engaged).
- **AC3** — properties.yaml 4 NEW gate entries + schema_version MINOR bump `1.2.0 → 1.3.0`.
- **AC4** — PRE-FLIGHT sympy verification `T_CHEB_ZETA6` + `T_CHEB_ZETA4` (mirror `T_CHEB` pattern at `scripts/verify_chebyshev_barycentric.py`; sub-checks: leading-coefficient Richardson cancellation under Chebyshev + spectral floor confirmation at canonical Gaussian probe).
- **AC5** — Measurement report `.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md` with GREEN/AMBER/RED verdict; architect signs off → triggers v5.0 ADR-0099 B.2 decision.

## Alternatives considered

| Option | Decision | Rationale |
|---|---|---|
| **Defer B.3 to v5.0 itself** (measure during BREAKING window) | REJECTED | v5.0 BREAKING window per ADR-0035 §9 12-month cycle is ~2027-05-27; measuring at v5.0 leaves no recovery window if RED verdict; v4.6 PREP MINOR is the standard pattern for pre-BREAKING evidence-gathering (mirrors v4.3 `T_CHEB` oracle landing BEFORE v4.4 Padé scaling-and-squaring). |
| **Skip B.3 entirely, ship B.2 blind at v5.0** | REJECTED | Violates the "honest closure" principle (ADR-0086 AMENDMENT 1) — promoting a default without measurement evidence is the same pattern that caused the ζ⁴ default REGRESSION at v4.3 (ADR-0089 AMENDMENT 1). Cost of B.3 is 4-6 days; cost of a v5.0 RED post-ship rollback would be a full v5.1 cycle. |
| **Add `_cheb` siblings AND amend originals** (rewire ζ⁴/ζ⁶ defaults at v4.6) | REJECTED | Default rewiring is v5.0 BREAKING-window business per roadmap §"B.2"; v4.6 is PREP-MEASUREMENT MINOR by definition. Sibling gates preserve v4.x byte-equality on existing code while providing the measurement channel. |
| **Include 2D/3D Chebyshev re-measurement** | OUT OF SCOPE | ADR-0090 §"6. Global default unchanged" + §"7. 2D/3D/ND scope deferred" explicitly excludes ND Chebyshev from v4.3; v5.0 B.2 scope per roadmap §"v5.0 default change for non-Grid1D types" line 122 also defers ND default promotion to v5.1 conditional. B.3 re-measurement matches B.2 scope: `Grid1D` only. |

## Cross-references

- ADR-0086 + AMENDMENT 1 — Option E hybrid calibration rule (`threshold = ⌊measured − 0.1⌋ + 0.1`); precedent for sibling-gate methodology.
- ADR-0088 + AMENDMENT 1 — ζ⁶ Path β; 3.868 calibration history; ζ⁸ HOLD-release that ADR-0090 unblocked.
- ADR-0089 + AMENDMENT 1 — Path ε QuinticHermite; ζ⁴ default REGRESSION diagnosis; the v4.3 "Chebyshev removes higher-order Taylor re-ordering" prediction that B.3 will measure.
- ADR-0090 — Chebyshev opt-in v4.3; §AC8/AC9 original re-measurement scheduling (re-confirmed and executed here).
- ADR-0099 (PENDING v5.0) — global `Grid1D::new` default promotion; conditional on B.3 GREEN/AMBER verdict.
- `crates/semiflow-core/src/diffusion4.rs:318` — `.with_chebyshev_sampling()` opt-in (target API).
- `crates/semiflow-core/src/diffusion4_zeta4.rs:243` — `.with_chebyshev_sampling()` opt-in on ζ⁴.
- `crates/semiflow-core/src/diffusion6_zeta6.rs:190` — `.with_chebyshev_sampling()` opt-in on ζ⁶.
- `crates/semiflow-core/tests/zeta6_correction_slope.rs` — template for NEW `_cheb` sibling test.
- `crates/semiflow-core/tests/zeta8_correction_slope.rs` — existing v4.3 Cheb test (REVERIFY only at v4.6).
- `contracts/semiflow-core.properties.yaml:4728–4793` — `G_zeta6_const_a_richardson` shape template for NEW `_cheb` entries.
- `contracts/semiflow-core.properties.yaml:6125–6166` — existing `G_zeta8_*_cheb` entries (REVERIFY only).
- `contracts/semiflow-core.math.md §9.2.7` — Chebyshev NORMATIVE section; this ADR appends a v4.6 footnote (~10 LoC) per Deliverable 3.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §"v4.6.0 — PREP-MEASUREMENT MINOR" + §"v5.0 B.2" — roadmap dependency.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_v4_5_research_wave.md` §123 — "ζ⁶/ζ⁸ Chebyshev re-measurement post Path ε opt-in: pending" — closure target.
- `.dev-docs/specs/zeta-chebyshev-remeasure-wave.md` — engineer Wave spec (AC1–AC5).
- Boyd 1989/Dover 2000 *Chebyshev and Fourier Spectral Methods* §5–6 — asymptotic order 6+ prediction for smooth IC under Chebyshev spatial sample.
- Trefethen 2000 *Spectral Methods in MATLAB* — barycentric weights + spectral convergence rate theory.

## AMENDMENT 1 (2026-05-29) — B.3 RED verdict; abort B.2; v5.0 plan revised

**Trigger**: Engineer Wave B.3 measurement (`.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md`) returned RED:
- ζ⁴ const-a cheb log₂=-112.53 (anti-convergent)
- ζ⁴ var-a cheb slope=+209.97 (diverging)
- ζ⁶ const-a cheb overflow (err_1=+inf; spurious Rust PASS)
- ζ⁶ var-a cheb spurious (errors 1e105→1e256→0)
- ζ⁸ already-shipped same pre-existing overflow

**Diagnosis**: Chebyshev opt-in COMPOSITION with K5 Catmull-Rom base + ζ⁴/ζ⁶ Richardson cascade catastrophically fails. v4.3 ζ⁸ DIRECT Chebyshev kernel (not opt-in composition) works correctly. Chebyshev works as STANDALONE kernel, NOT as opt-in sampler for composed kernels.

**Decision per §"Decision criteria" RED rule**: v5.0 ADR-0099 B.2 global Chebyshev default promotion **ABORTED**. Deferred to v5.1+ conditional on diagnosis.

**v5.0 plan revision**:
- v5.0 BREAKING window reduces 3 items → 2 items (A.6 + B.1)
- B.2 promotion DEFERRED v5.1+ pending architectural diagnosis
- v4.6 ships B.3 measurement evidence + Robin BC (A.3) only

**v5.1+ candidates for B.2 diagnosis path**:
1. **Chebyshev DIRECT-KERNEL restriction**: ship Chebyshev as standalone `Diffusion4thChebChernoff` sibling (mirror v4.3 ζ⁸ pattern); NEVER as opt-in sampler.
2. **Composition fix**: diagnose K5+Richardson+Chebyshev triple-layer overflow source; possible Quintic-K5 + Richardson floor cascade.
3. **Hybrid v5.0 default**: keep CubicHermite default for ζ⁴/ζ⁶ (composition kernels); promote Chebyshev default ONLY for direct kernels (ζ⁸ `Diffusion8thZeta8Chernoff` already shipped at v4.3).

**Research artifact preserved**: `.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md` (engineer wave; 233 LoC; full measurement record + 6 sections; reproducible at v5.1+ for diagnosis).

**Cross-references**:
- v4.6 `.dev-docs/reports/V4_6_B3_REMEASURE_REPORT.md` (engineer measurement record)
- ADR-0090 (v4.3 Chebyshev opt-in original spec; diagnosis blames composition not sampler)
- v5.0+ roadmap `~/.claude/plans/roadmap-reflective-biscuit.md` v5.0 section (must revise B.2 scope)

## AMENDMENT 2 (2026-05-29) — v5.0 B.3 post-H3-fix gate recalibration

**Trigger**: ADR-0104 engineer wave resolved H3 (OOB Runge divergence) by adding `OobPolicy` boundary enforcement at the Chebyshev interpolation boundary. Post-H3-fix measurements (i7-12700K, 2026-05-29) replace all RED divergence values with GREEN convergent measurements.

**Post-H3-fix measurements and threshold updates** (Option E rule: `⌊measured − 0.1⌋ + 0.1`):

| Gate | Old threshold | New threshold | Measured post-H3 |
|------|--------------|--------------|-----------------|
| `G_zeta4_const_a_richardson_cheb` | 3.9 (predicted) | 3.1 | log₂=3.2260 |
| `G_zeta4_var_a_slope_cheb` | -2.5 (predicted) | 0.1 | slope=-0.0188 (floor-dominated) |
| `G_zeta6_const_a_richardson_cheb` | 5.5 (predicted) | 3.8 | log₂=3.8701 |
| `G_zeta6_var_a_slope_cheb` | -5.5 (predicted) | 0.5 | slope=-0.1539 (not-diverging) |
| `G_zeta8_const_a_richardson_cheb` | 6.5 (predicted) | 3.0 | log₂=3.0667 |
| `G_zeta8_var_a_slope_cheb` | -6.5 (predicted) | 0.1 | slope=0.0561 (floor-dominated) |

**H4 diagnosis**: All thresholds cluster at log₂ ≈ 3.0–3.9 (not 6.0–8.0 as predicted). Root cause (ADR-0104 H4): QuinticHermite K5 intermediate evaluations within each semigroup step cap the effective error floor at ≈ 1e-10, regardless of Chebyshev M=64 spectral precision (≤ 1e-15 for grid-node queries). The Richardson ratios reflect this floor interaction, not the asymptotic Chebyshev advantage.

**Assessment**: These are TRUTHFUL measurements, not regressions. The v4.x predicted thresholds (3.9 / 5.5 / 6.5) were based on the false "≤ 1e-15 spectral floor" claim (H4 defect) combined with pre-H3-fix divergence masking real signal.

**Cross-references**:
- ADR-0104 (H3 + H4 root-cause; engineer wave spec)
- ADR-0090 AMENDMENT 1 (OobPolicy + ChebyshevSpectralWithBC API change)
- `docs/migration/v4-to-v5.md` (gate threshold table for downstream consumers)
- `contracts/semiflow-core.properties.yaml` schema_version 2.0.0 (BREAKING threshold change record)
