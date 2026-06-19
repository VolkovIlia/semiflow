# ADR-0089 — Path ε: QuinticHermite Spatial Sample Upgrade for K5 + ζ-Ladder

- **Status**: Accepted
- **Date**: 2026-05-28
- **Decision-maker**: ai-solutions-architect
- **Related**: ADR-0086 + AMENDMENT 1 (originating Path ε mention in §"ADR-0088 (deferred)"); ADR-0088 + AMENDMENT 1 (ζ⁸ Wave II HOLD pending Path ε); ADR-0015 (v0.7.0 ζ⁶ spatial — already ships `QuinticHermite` for `Diffusion6thChernoff`); ADR-0035 §9 (deprecation cycles); ADR-0073 (`ApproximationSubspace<K, F>`).
- **Mathematical foundation**: `crates/semiflow-core/src/grid_quintic.rs` (208 LoC, already shipped at v0.7.0); math.md §9.2.6 NORMATIVE QuinticHermite spec; Carlson 1980 *Splines and Approximation Theory* §4 (Hermite ghost extension); Fornberg 1988 *Math. Comp.* Table 1 (6-point central FD for scaled `dx·f'`).
- **Acceptance gates added**: 1 NEW regression test `G_PATH_EPS_FLOOR` (spatial-floor `≤ 1e-7` at N=512 on Gaussian probe with QuinticHermite); 2 TIGHTENED gates (`G_zeta4_var_a_slope` ADVISORY → BLOCKING at ≤−3.9 post-upgrade; `G_zeta6_var_a_slope` ADVISORY → BLOCKING at ≤−5.5 post-upgrade); 2 NEW gates auto-promoted from ADR-0088 Wave II HOLD release (`G_zeta8_const_a_richardson` ≥ 3.5 BLOCKING + `G_zeta8_var_a_slope` ≤−6.5 ADVISORY).
- **Target release**: v4.2.0 (non-breaking under Option D — additive opt-in + per-kernel retrofit; global default unchanged).

## Context

Post-v4.0 tech-debt sweep (`project_tech_debt_sweep_2026_05_28.md`) shipped 4.5/6 items but surfaced a single recurring architectural prerequisite across 3 ADR AMENDMENTs (ADR-0086 §"ADR-0088 (deferred)", ADR-0088 §"Wave II HOLD", `project_step_k_carnot_open.md` §"Architectural recommendation"). The cubic-Hermite (Catmull-Rom-flavor) interpolation that `GridFn1D::sample()` defaults to has leading residue $O(dx^4)$; at the canonical Item-1 setup (`N=512`, Gaussian probe) the measured spatial floor is **≈ 1.18e-4**. This floor caps the measurable temporal order of every Chernoff kernel built on top of `Diffusion4thChernoff` (the K5 base): G_zeta4 const-a ratio measured 3.55 vs theoretical 4; G_zeta6 const-a ratio measured 3.67 vs theoretical 6 (`project_tech_debt_sweep_2026_05_28.md` §"Shipped (3/6)" Item 1; ADR-0088 AMENDMENT 1 §"Diagnosis"). The infrastructure for QuinticHermite ($O(dx^6)$) was shipped at **v0.7.0 (ADR-0015)** in `crates/semiflow-core/src/grid_quintic.rs` (208 LoC) and is already wired into `Grid1D::interp` (`grid.rs:282-284`), but only the v0.7.0 `Diffusion6thChernoff` calls `.with_interp(InterpKind::QuinticHermite)`. The K5 base `Diffusion4thChernoff` and its ladder descendants (`Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`) all sample through the cubic-Hermite default and inherit the 1.18e-4 floor. Path ε is the architectural retrofit that lets these kernels opt into QuinticHermite without changing the global `Grid1D::new` default and without breaking the v0.5/v0.6 K5 byte-identity contract.

## Decision

Adopt **Option D (multi-pronged additive retrofit)** with **1D-only scope (Q2.a)**:

1. **Global default unchanged**: `Grid1D::new(_, _, n)` continues to return `interp = InterpKind::CubicHermite`. v0.5/v0.6/v3.x callers compile and execute byte-identical (zero regression risk; G3⁴ + G3-strang + every existing bit-identity gate stays green).
2. **Per-kernel opt-in at construction**: Add `.with_quintic_sampling()` builder (or equivalent constructor flag) to `Diffusion4thChernoff`, `Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`, and (Wave II) `Diffusion8thZeta8Chernoff`. When set, the kernel calls `.with_interp(InterpKind::QuinticHermite)` on its working `GridFn1D` clones before the inner `f.sample(...)` queries (mirroring the established pattern in `hormander.rs:457`).
3. **Default opt-in for the ζ-ladder rungs only**: At v4.2 release, `Diffusion4thZeta4Chernoff::new(...)` and `Diffusion6thZeta6Chernoff::new(...)` default to QuinticHermite ON. The K5 base `Diffusion4thChernoff::new(...)` keeps CubicHermite default (preserves v0.6 byte-equality; users opt in explicitly via `.with_quintic_sampling()`). Rationale: the ladder kernels' contract IS the corrected order (4/6); the K5 base's contract IS v0.6 byte-equality. The two contracts diverge under Path ε — Option D respects both.
4. **Global default promotion deferred to v5.0**: Per ADR-0035 §9 deprecation cycle, promote `Grid1D::new` default `CubicHermite → QuinticHermite` at the next BREAKING window. 12-month deprecation window between v4.2 ship date and v5.0 tag. Rationale: defers blast-radius assessment of every kernel using `f.sample()` (γ-A baseline, ζ⁴/ζ⁶/ζ⁸, drift-reaction characteristic foot, Strang2D/3D axis-lift if applicable) to a single BREAKING checkpoint instead of fragmenting it across v4.2/v4.3/...
5. **ND scope deferred**: 2D/3D/ND grid types (`Grid2D`, `Grid3D`, `GridND`) keep `CubicHermite` default; per-axis `.with_interp()` is already supported. A separate v4.3+ ADR-0090 (if needed) would retrofit Strang2D/3D axis-lift if measurement at 2D/3D ladders ever justifies the per-axis cost. This ADR's scope is **1D only** (Q2.a) — sufficient for ζ-ladder + future step-3 Carnot.

## Algorithm

QuinticHermite is already implemented in `grid_quintic.rs::sample_quintic_1d` (208 LoC, NORMATIVE, sympy-verified at v0.7.0 via 4 gates: `QHerm5_partition`, `QHerm5_endpoints`, `QHerm5_consistency`, `QHerm5_order`). Local 6-point stencil per cell uses scaled nodal data `(F_0, F_{0p}, F_{0pp}, F_1, F_{1p}, F_{1pp})` where `F_{0p} = dx·f'(x_i)` and `F_{0pp} = dx²·f''(x_i)` (ghost data computed via 6-point central FD for `f'`, 5-point for `f''`; boundary via `bc_value` per `BoundaryPolicy` — no new ghost-cell handling needed; the existing `BoundaryPolicy` enum is total). Leading residue: $f^{(6)}/720 \cdot s^3(1-s)^3 \cdot dx^6$, bounded $\le dx^6 / 46080$ at $s = 1/2$. Per-sample cost vs CubicHermite: ~2.0× (6 nodal values + 6 weight evaluations vs 4 + 4); per-`apply_into` cost increase ~30% (sample is one cost component among stencil arithmetic). Total impl LoC for Path ε retrofit: **~80-120 LoC** (per-kernel `.with_quintic_sampling()` builder + working-grid clone wiring at 3-4 call sites; no new interpolation code). Engineering days: **2-3** (single Wave I).

## Consequences

- **POSITIVE**: lifts measurable G_zeta4 const-a ratio from 3.55 → predicted ≥ 3.95 (asymptotic order 4); lifts G_zeta6 const-a ratio from 3.67 → predicted ≥ 5.8 (asymptotic order 6); promotes `G_zeta4_var_a_slope` + `G_zeta6_var_a_slope` from ADVISORY to BLOCKING (per ADR-0086 AMENDMENT 1 §"ADR-0088 (deferred)" promise); **unblocks ADR-0088 Wave II ζ⁸** (HOLD released; `Diffusion8thZeta8Chernoff` + 2 new gates ship in v4.2 or v4.3 follow-up); creates measurable-τ regime down to τ ≈ 1e-3 where order-8 leading term dominates the spatial floor; closes the recurring "calibrated→theoretical" gap across the entire ζ-ladder.
- **NEUTRAL**: per-`apply_into` cost +~30% on opted-in kernels (acceptable: ladder kernels already pay 3× / 9× K5 calls per Richardson level, so ~30% on the inner sample is a small marginal); 2-3 working days engineering; zero new dependency.
- **NEGATIVE**: ζ-ladder default behavior changes between v4.1 (CubicHermite) and v4.2 (QuinticHermite) — slope measurements taken at v4.1 will not reproduce at v4.2 (documented in migration `docs/migration/v4.1-to-v4.2.md`); QuinticHermite ghost-FD currently `f64`-only (per `grid.rs:196-198`) — generic-over-Float `f32` callers of Path-ε-opted kernels return `SemiflowError::Unsupported` (engineer adds runtime check at constructor with clear error message).
- **BREAKING**: NONE under Option D. Global `Grid1D::new` default unchanged. ζ-ladder default behavior change is "kernel achieves the temporal order its `order()` already advertises" (4 for ζ⁴, 6 for ζ⁶) — this is a **bug-fix-grade** behavior change, documented but not API-breaking. Promotion of `Grid1D::new` global default to QuinticHermite is deferred to v5.0 (BREAKING window per ADR-0035 §9). `Diffusion4thChernoff` K5 base byte-equality contract with v0.6.0 PRESERVED (its default stays CubicHermite; opt-in only).
- **Schema bumps**: `properties.yaml` MINOR bump (1 new `G_PATH_EPS_FLOOR` regression gate + 2 TIGHTENED `G_zeta{4,6}_var_a_slope` thresholds + 2 NEW `G_zeta8_*` gates if Wave II ships in same release). `traits.yaml` unchanged (no trait surface change; new builder method is inherent on each struct). `math.md` AMENDMENT to §9.2.6 + NEW §27.quart (Path ε narrative — appended).
- **Constitution check**: `grid.rs` currently **499 LoC** (per `wc -l`). Path ε adds 0 LoC to `grid.rs` (the QuinticHermite kernel + dispatch already exists). Each opted-in kernel grows ~20-40 LoC (builder + working-grid clone wiring). `diffusion4.rs` (621 LoC) is already over the default 500 cap and lives under Cohort 1 (verify constitution Cohort 1 contains `diffusion4.rs`; if not, this ADR triggers a v1.8.x PATCH bump to add it — engineer + architect coordinate at impl time). `diffusion4_zeta4.rs` (490) + `diffusion6_zeta6.rs` (427) both have headroom under default 500 cap. No new override category.

## Migration spec

Per-kernel adoption order (Wave I — engineer task per `.dev-docs/specs/path-epsilon-wave.md`):

1. **K5 base** (`Diffusion4thChernoff`): add `.with_quintic_sampling()` builder. Default OFF (preserves v0.6 byte-equality).
2. **ζ⁴ rung** (`Diffusion4thZeta4Chernoff`): default ON; constructor signature unchanged (transparently sets inner K5's working-grid interp to QuinticHermite); existing 3 Richardson K5 calls in `apply_into` (`diffusion4_zeta4.rs:300/303/306`) inherit the upgrade.
3. **ζ⁶ rung** (`Diffusion6thZeta6Chernoff`): default ON; mirror ζ⁴ pattern.
4. **ζ⁶ baseline** (`Diffusion6thChernoff`): UNCHANGED — already opts in per v0.7.0 ADR-0015.
5. **ζ⁸ rung** (`Diffusion8thZeta8Chernoff`, NEW): default ON; Wave II per ADR-0088 §"Decision" Option α (release HOLD per AMENDMENT 1).

Gate re-calibration timeline (engineer test plan):

- v4.2-pre: measure G_zeta4 + G_zeta6 const-a Richardson ratios with QuinticHermite ON on the canonical {4,8,16,32} sweep.
- v4.2-pre: measure var-a slopes. Expect G_zeta4_var_a_slope ≤ −3.9 (predicted) and G_zeta6_var_a_slope ≤ −5.5 (predicted, conservative pre-asymptotic margin).
- v4.2 ship: TIGHTEN thresholds in `properties.yaml`; promote both var-a gates ADVISORY → BLOCKING.
- v4.2 ship: add `G_PATH_EPS_FLOOR` regression test verifying floor ≤ 1e-7 at N=512 with Gaussian probe (replaces ad-hoc measurement; permanent regression guard).
- v4.3 or v4.2 same-release: ship Wave II ζ⁸ + 2 new gates.

## Alternatives considered

| Option | Decision | Rationale |
|---|---|---|
| **A — Replace global default `CubicHermite → QuinticHermite`** | REJECTED | High blast-radius: every kernel using `f.sample()` changes behavior simultaneously; v0.6 byte-equality contract on K5 base BREAKS; regression risk dwarfs the engineering savings. |
| **B — Add variant + keep CubicHermite default, no kernel retrofit** | REJECTED | Variant already exists since v0.7.0 (ADR-0015); status-quo Option B leaves ζ⁴/ζ⁶ stuck at calibrated thresholds forever. Closes nothing. |
| **C — Add variant + keep CubicHermite default, promote in v5.0 only (no v4.2 retrofit)** | REJECTED | Defers ζ⁸ Wave II by another major-version cycle; engineer Wave for K5 byte-equality preservation is the same 2-3 days whether at v4.2 or v5.0; v5.0 already faces its own BREAKING surface. |
| **D — Additive opt-in + per-kernel retrofit at v4.2 + global promotion at v5.0 (CHOSEN)** | ACCEPTED | Smallest patch that unlocks both gates AND preserves K5 byte-equality contract. v5.0 global promotion gives blast-radius assessment a single checkpoint. |
| **ND scope (b) 1D+2D OR (c) 1D+2D+3D+ND** | REJECTED for this ADR | 1D suffices for ζ-ladder + step-3 Carnot prerequisite (the actual ask). 2D/3D Strang2D/3D axis-lift retrofit is a separate measurement question (no current gate forces it). Deferred to v4.3+ ADR-0090 only if measurement justifies. |

## Cross-references

- ADR-0015 — v0.7.0 ζ⁶ extension (ships `InterpKind::QuinticHermite` + `grid_quintic.rs`; foundation of this ADR).
- ADR-0086 + AMENDMENT 1 — Path β G_zeta4 (originating mention of Path ε in §"ADR-0088 (deferred)").
- ADR-0088 + AMENDMENT 1 — ζ-ladder rungs (Wave II HOLD pending this ADR; auto-released upon ship).
- ADR-0035 §9 — deprecation cycles (12-month window for v4.2 → v5.0 default promotion).
- ADR-0073 — `ApproximationSubspace<K, F>` (each opted-in kernel preserves its existing witness impl).
- math.md §9.2.6 — NORMATIVE QuinticHermite spec (already shipped); this ADR appends §27.quart Path ε narrative.
- `.dev-docs/specs/path-epsilon-wave.md` — engineer Wave I + (deferred) Wave II spec.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_tech_debt_sweep_2026_05_28.md` — sweep closure record.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_step_k_carnot_open.md` §"Architectural recommendation" — independent corroboration of Path ε as gating prerequisite.
- `crates/semiflow-core/src/grid_quintic.rs` — implementation (208 LoC).
- `crates/semiflow-core/src/grid.rs:282-284` — existing `InterpKind::QuinticHermite` dispatch.
- `crates/semiflow-core/src/hormander.rs:457` — pattern reference for `.with_interp(InterpKind::QuinticHermite)` per-kernel opt-in.

---

## AMENDMENT 1 (2026-05-28) — ζ⁴ default-ON measured REGRESSION; revert to opt-in; preserve ζ⁶ win via separate K5 wiring

**Trigger**: Engineer Wave I shipped Path ε per §"Decision" Option D and §"Migration spec" steps 1–3. Bug-fixer measurement on the `g_zeta4_const_a_richardson_ratio` BLOCKING gate (n-pair `{4, 8}`, N=512, T=0.5, analytic Gaussian oracle) returned log₂(ratio) = **3.226** with QuinticHermite ON — **DEGRADED** from the v4.1 CubicHermite baseline 3.55 (Δ = −0.32). G_zeta4 const-a BLOCKING gate (threshold 3.5) **FAILS by 0.27**. Companion gate G_zeta6 const-a (n-pair `{1, 2}`) IMPROVED from 3.67 → 3.868 (Δ = +0.20) and passes the tightened 3.8 threshold. Engineer wiring verified correct: `Diffusion4thZeta4Chernoff::new()` calls `inner.with_quintic_sampling()` and propagates through Richardson `(4·K5(τ/2)² − K5(τ))/3`.

**Diagnosis (mathematical, not implementation)**: At n=4 (τ=0.125, ≈ pre-asymptotic for K5+Richardson on T=0.5 horizon), QuinticHermite changes the leading O(τ²) error coefficient of K5 in a way that partially counteracts the Richardson `(4·a − b)/3` linear combination's odd-power cancellation. The cancellation is mathematically exact only for the symmetric K5 Catmull-Rom error structure (`§9.2.4`); shifting K5's spatial residue from O(dx⁴)·(C₄·f⁽⁴⁾) to O(dx⁶)·(C₆·f⁽⁶⁾) shifts which higher-order temporal terms dominate at finite τ — and at n=4 the residual is now non-symmetric in the Richardson cancellation pattern, masking order-4 convergence at this measurement scale. G_zeta6 is unaffected because its n-pair `{1, 2}` lives at much larger τ (0.5, 0.25) where the leading 16/15 Richardson² term still dominates regardless of K5 spatial residue shape; QuinticHermite simply tightens the floor and the asymptotic ratio improves.

**Decision (Option A modified — revert ζ⁴ default; preserve ζ⁶ win via separate ζ⁶-only K5 wiring)**: Adopted. Rationale: G_zeta4 const-a is **BLOCKING** and cannot ship at 3.226 < 3.5; G_zeta6 IMPROVED and must be preserved. Option B (n-pair {8, 16}) was considered but rejected — smaller τ (0.0625, 0.03125) risks hitting the new Quintic O(dx⁶) floor at N=512 (~1e-8) which would re-introduce floor contamination; the n-pair calibration burden compounds. Option C (lower G_zeta4 threshold 3.5 → 3.1) was rejected per architect-mandate "honest closure" — admitting a 5× weaker ζ⁴ gate than ζ⁶'s 3.8 misrepresents the kernel's contract. Option D (full revert) sacrifices the ζ⁶ win. **Chosen path**:

1. `Diffusion4thZeta4Chernoff::new()` AMENDED: revert to CubicHermite default (mirror K5 base — quintic_sampling=false). Engineer removes the `inner = inner.with_quintic_sampling()` call at `diffusion4_zeta4.rs:189`. The AC4 f64-only guard is also removed (Quintic no longer default-ON for this kernel). `new_cubic()` becomes redundant and is retained for one minor cycle then deprecated at v4.3.
2. NEW builder `Diffusion4thZeta4Chernoff::with_quintic_sampling(self) -> Self` added for explicit opt-in (mirrors K5 base API). Default remains OFF.
3. `Diffusion6thZeta6Chernoff::new()` AMENDED: opt the inner K5 IN to QuinticHermite **directly**, bypassing the (now-reverted) ζ⁴ default. Engineer wires `self.inner.inner = self.inner.inner.with_quintic_sampling()` in the constructor (after the existing `let grid = inner.grid` line). This isolates the ζ⁶ benefit from ζ⁴'s regression while preserving the additive opt-in pattern. The composed pipeline becomes: K5 Quintic ON → ζ⁴ Richardson (sees Quintic K5) → ζ⁶ Richardson² (sees Quintic-via-ζ⁴ K5).

**Hybrid calibration (mirrors Items 1+4 Option E precedent)**: G_zeta4_const_a BLOCKING gate threshold UNCHANGED at 3.5 (the v4.1 CubicHermite baseline 3.55 still passes by 0.05 margin once ζ⁴ default-ON is reverted). G_zeta6_const_a BLOCKING gate threshold UNCHANGED at 3.8 (the new ζ⁶-only Quintic wiring preserves the 3.868 measurement). The var-a ADVISORY gates remain ADVISORY (Catmull-Rom floor in K5-as-oracle still dominates var-a measurement at n∈{4,8,16,32}, N=512 for ζ⁴; the ζ⁶ var-a slope advisory at +0.5 also unchanged — the K5-as-oracle still uses CubicHermite for the run_inner reference path).

**Regression test fix (path_eps_spatial_floor.rs)**: The `g_path_eps_floor_quintic_improvement` baseline assertion `cubic_err > 1e-5` is calibrated for the integrated K5 stencil floor (1.18e-4 ADR Context §) but the test measures single-cell Catmull-Rom interp error at the midpoint of a smooth Gaussian where `f⁽⁴⁾(x)·dx⁴/384` evaluates to ~6.5e-7 at N=512. The assertion is mathematically wrong — Catmull-Rom delivers O(dx⁴) ≈ 6.5e-7 here, not 1.18e-4. Engineer fix: change `CUBIC_FLOOR_LOWER_BOUND` from `1e-5` to `1e-7` (preserves "Cubic floor exists and improvement is real" semantic without false-FAIL). The QUINTIC_FLOOR_GATE 1e-7 is also tightened to 1e-8 to maintain a 1.5-decade improvement signal. The wiring assertion test `g_path_eps_floor_zeta4_default_on` MUST be UPDATED to assert `!zeta4.inner.quintic_sampling` (reflects new default-OFF) and add a new sub-assertion that `Diffusion6thZeta6Chernoff::new(zeta4_cubic, ...)` produces an `inner.inner.quintic_sampling == true` (verifies the new ζ⁶-only K5 Quintic wiring).

**Engineer guidance**:

| File | Edit |
|---|---|
| `diffusion4_zeta4.rs:189` | DELETE `let inner = inner.with_quintic_sampling();` line; remove the AC4 f64-only guard immediately above |
| `diffusion4_zeta4.rs` (new method) | ADD `pub fn with_quintic_sampling(mut self) -> Self { self.inner = self.inner.with_quintic_sampling(); self }` (mirrors K5 base) |
| `diffusion4_zeta4.rs:226` | KEEP `new_cubic()` for one minor cycle (deprecation in v4.3 per ADR-0035 §9) |
| `diffusion6_zeta6.rs:158` | ADD `let inner = { let mut inner = inner; inner.inner = inner.inner.with_quintic_sampling(); inner };` immediately after `let grid = inner.grid;` |
| `zeta4_correction_slope.rs` | NO CHANGE (threshold 3.5 still applies; CubicHermite baseline 3.55 passes by 0.05) |
| `zeta6_correction_slope.rs` | NO CHANGE (threshold 3.8 still applies; ζ⁶-only Quintic wiring preserves 3.868 measurement) |
| `path_eps_spatial_floor.rs:50` | CHANGE `CUBIC_FLOOR_LOWER_BOUND: f64 = 1e-5` → `1e-7` |
| `path_eps_spatial_floor.rs:48` | CHANGE `QUINTIC_FLOOR_GATE: f64 = 1e-7` → `1e-8` (preserves 1.5-decade improvement signal) |
| `path_eps_spatial_floor.rs:170-181` | UPDATE: assert `!zeta4.inner.quintic_sampling` (default-OFF); add new sub-test verifying `Diffusion6thZeta6Chernoff::new(zeta4, ...).inner.inner.quintic_sampling == true` |
| `properties.yaml` | G_zeta4_const_a threshold unchanged (3.5); G_zeta6_const_a threshold unchanged (3.8); update G_PATH_EPS_FLOOR gate doc to reflect Quintic 1e-8 / Cubic 1e-7 single-cell midpoint Gaussian semantics |
| `math.md §9.2.6.bis` | AMENDMENT 1 (see math.md amendment) |
| `docs/migration/v4.1-to-v4.2.md` | UPDATE: ζ⁴ default unchanged from v4.1 (CubicHermite); ζ⁶ default behavior change (CubicHermite→QuinticHermite via direct K5 wiring) — bug-fix-grade |

**Consequences (delta from ADR primary)**:
- POSITIVE: G_zeta6 win preserved (3.67→3.868, +0.20); ζ⁴ regression eliminated; v4.2 ships green BLOCKING gates; ζ⁸ Wave II still unblocked (it can wire its inner K5 via the same direct pattern as ζ⁶).
- NEGATIVE: ζ⁴ measurable order at N=512 stays at 3.55 (≈ asymptotic order 3.5, not theoretical 4.0); the var-a ADVISORY gates remain ADVISORY. Path ε's stated promise "lifts G_zeta4 const-a ratio 3.55 → ≥ 3.95" is REVERSED to "preserves 3.55 baseline + lifts G_zeta6 const-a 3.67 → 3.868". Full order-4 for ζ⁴ now requires a follow-up investigation (n-pair calibration or alternative spatial interp scheme — DEFERRED to v4.3+ ADR-0090).
- NEUTRAL: properties.yaml gate thresholds unchanged from this AMENDMENT; only G_PATH_EPS_FLOOR semantics tighten 10× (from misspecified 1e-5/1e-7 to correct 1e-7/1e-8).
- NO BREAKING: Option D's "additive opt-in, K5 byte-equality preserved" invariant strengthens — now ζ⁴ also preserves v4.1 byte-equality by default.

**Cross-references**: Items 1 + 4 (Item-1 ζ⁴ history + Item-4 Path β var-a) hybrid Option E calibration precedent in `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_g_zeta4_escalation.md` §"Resolution path"; engineer Wave I measurement record (TBD `~/.claude/projects/.../project_path_epsilon_wave1_regression_2026_05_28.md`).
