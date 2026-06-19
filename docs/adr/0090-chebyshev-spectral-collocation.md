# ADR-0090 — Chebyshev Spectral Collocation Spatial Sample (Path ε Successor; ζ⁸ Wave II Unblocker)

- **Status**: Accepted
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Related**: ADR-0015 (v0.7.0 QuinticHermite — predecessor interpolant); ADR-0086 + AMENDMENT 1 (G_zeta4 Path β; Catmull-Rom dx-floor diagnosis); ADR-0088 + AMENDMENT 1 (ζ-ladder rungs; Wave II HOLD); ADR-0089 + AMENDMENT 1 (Path ε QuinticHermite; partial floor lift but pre-asymptotic ratio 3.067 < 4 for ζ⁸); ADR-0035 §9 (deprecation cycles); ADR-0068 (boundary.rs / `InterpKind` location).
- **Mathematical foundation**: Boyd, *Chebyshev and Fourier Spectral Methods* (Dover 2nd ed., 2000), Chapters 5–6; Trefethen, *Spectral Methods in MATLAB* (SIAM 2000), Chapters 6–8 and Differentiation Matrix Appendix; Berrut & Trefethen, "Barycentric Lagrange Interpolation" (*SIAM Review* 46:501, 2004); Hesthaven & Warburton, *Nodal Discontinuous Galerkin Methods* (Springer 2008), §3.2. Researcher synthesis: `.dev-docs/research/verdicts/verdict-v4-3-research-waves.md` §Q1 Option B; raw data `.dev-docs/research/raw-findings-spatial-floor-extension.md` Query 2.
- **Acceptance gates added**: 1 NEW `G_PATH_EPS_CHEB_FLOOR` (RELEASE_BLOCKING, ≤1e-15 single-cell-midpoint Gaussian at N=512 with Chebyshev-sampling opt-in); 2 TIGHTENED gates auto-promoted from ADR-0089 AMENDMENT 1 deferral (`G_zeta4_const_a_richardson` ≥ 3.9 BLOCKING under Chebyshev opt-in; `G_zeta4_var_a_slope` ADVISORY→BLOCKING at ≤−3.9); 1 NEW gate (`G_zeta8_const_a_richardson` under Chebyshev wiring, BLOCKING, calibrated at measurement time per Option E hybrid rule). NEW sympy oracle `T_CHEB` (barycentric formula + differentiation matrix entries).
- **Target release**: v4.3.0 (non-breaking under per-kernel opt-in; default unchanged).

## Context

ADR-0089 Path ε retrofitted `Diffusion4thZeta4Chernoff` and `Diffusion6thZeta6Chernoff` with the v0.7.0 `InterpKind::QuinticHermite` interpolant (O(dx⁶) leading residue). Engineer Wave I empirically lifted the single-cell-midpoint Gaussian floor from ~1.18e-4 (CubicHermite, K5 stencil-integrated) to ~1.28e-10 (QuinticHermite, single-cell mid; ~5000× win), but ADR-0089 AMENDMENT 1 documented two unresolved limits: (a) ζ⁴ default-ON measured REGRESSION (3.226 vs 3.55 baseline — reverted; full order-4 deferred to "v4.3+ ADR-0090"); (b) ζ⁸ Wave II measured Richardson ratio 3.067 < theoretical 4 (`project_v4_1_0_v4_2_0_shipped.md` Insight #5 — pre-asymptotic regime interaction; K=4 needs ~10⁸× more headroom, i.e. floor ≤ ~1e-15, to expose asymptotic order-8). The QuinticHermite ceiling at ~1e-10 is polynomial-bounded; reaching 1e-15 requires either septic-Hermite (Wave 1 verdict: practical order-6 only — insufficient) or **exponential-accuracy spectral collocation**. Researcher Wave 1 (verdict §Q1) recommends Chebyshev spectral collocation as canonical (Boyd 1989, Trefethen 2000), with no_std-tractable cost (~200 LoC engineering, libm `cos`/`sin` for nodes, dense O(N²) per-sample for off-node queries via barycentric Lagrange). Heat semigroup parabolic regularity smooths solutions instantaneously, so spectral accuracy gives `‖error‖ ∝ exp(−γN)` for smooth u — at N=512 with analyticity-strip γ ≈ 1, error ≲ 10⁻²²³, totally collapsing the polynomial floor and unblocking ζ⁸ Wave II.

## Decision

Adopt **Option A (interpolation-based Chebyshev sampling on uniform `Grid1D`)** under the **per-kernel opt-in pattern** established by ADR-0089 (Option D additive retrofit). Specifically:

1. **`Grid1D` stays uniform.** The `Grid1D<F: SemiflowFloat = f64>` geometry contract (nodes at `xmin + i·dx`, `i ∈ [0, n)`) is UNCHANGED. We do NOT introduce Chebyshev-Lobatto nodal `Grid1D`; the existing uniform-grid invariants (caller's nodal `values: &[F]`, `bc_value` boundary handling, all existing SIMD/generic kernels) remain stable. **Rationale**: a native Chebyshev `Grid1D` variant is BREAKING (every existing kernel using `f.sample()` must learn the new node layout); blast radius dwarfs the engineering win. Additive interpolation preserves K5/K7/ζ-ladder byte-equality contracts AND admits Chebyshev sampling as a drop-in `InterpKind` variant.
2. **New `InterpKind::ChebyshevSpectral` variant** in `boundary.rs` (alongside `CubicHermite`, `Linear`, `QuinticHermite`). Dispatch added to `Grid1D::interp` (f64 path) per existing pattern at `grid.rs:267-287`. **Algorithm**: at sample time, compute barycentric Lagrange interpolant on a Chebyshev-Lobatto **virtual** node set of size M (M ∈ {8, 16, 32, 64, 128, 256, 512}, default M=64 per Wave default; configurable via `with_chebyshev_sampling(m: usize)` builder) constructed by sampling the uniform-grid `values` via QuinticHermite at the Chebyshev-Lobatto positions `x_k = (xmax+xmin)/2 + (xmax−xmin)/2 · cos(kπ/M)` (linear map `[−1, 1] → [xmin, xmax]`). The resulting `M+1` Chebyshev coefficients then deliver spectral-accurate evaluation at the query `x` via barycentric weights. **Critical**: the floor is bounded BELOW by the QuinticHermite ghost-data error (O(dx⁶)) at the virtual-node construction step; the spectral lift is on the Chebyshev representation, not the underlying uniform values. For smooth solutions where QuinticHermite already reaches ~1e-10 at the virtual nodes, Chebyshev tail-truncation error at M=64 contributes <10⁻¹⁶ — well below f64 ULP. Total floor target: ≤ 1e-15 at N=512, M=64.
3. **Per-kernel opt-in builder `.with_chebyshev_sampling()`** added to `Diffusion4thChernoff`, `Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`, `Diffusion8thZeta8Chernoff` (Wave II resurrection). Mirrors ADR-0089 `.with_quintic_sampling()` API exactly. **Default OFF** for all kernels (preserves v4.2 byte-equality on ζ-ladder; preserves v0.6/v4.1 byte-equality on K5 base). Opt-in promotes the kernel's inner working-grid `InterpKind::CubicHermite → InterpKind::ChebyshevSpectral` for the duration of `apply_into`.
4. **ζ⁴ default reactivation under Chebyshev**: ADR-0089 AMENDMENT 1 reverted `Diffusion4thZeta4Chernoff::new()` to CubicHermite default after the 3.226 regression. Under Chebyshev opt-in, mathematical diagnosis from ADR-0089 AMENDMENT 1 ("re-ordering of higher-order Taylor terms in Richardson cancellation") no longer applies — Chebyshev's exponential accuracy removes the leading O(dx⁶) coefficient that QuinticHermite introduced. Predicted: ζ⁴ const-a ratio recovers to ≥3.9. Engineer measures and decides (AC5). The default stays OFF for backward compat; users opt in via `.with_chebyshev_sampling()`.
5. **ζ⁸ Wave II RESURRECTION**: ADR-0088 HOLD released conditional on Chebyshev gates landing. New `Diffusion8thZeta8Chernoff::new()` defaults Chebyshev sampling ON (its contract IS order-8; without the Chebyshev floor lift the kernel cannot demonstrate asymptotic order). Promotes `G_zeta8_const_a_richardson` from DEFERRED to NEW BLOCKING gate (threshold calibrated per measurement; predicted ≥6.5 conservatively, ≥7.5 if Chebyshev tail truncation dominates as expected).
6. **Global default unchanged**: `Grid1D::new` continues to return `InterpKind::CubicHermite`. v0.5/v0.6/v3.x/v4.x callers compile and execute byte-identical. Promotion of global default `CubicHermite → ChebyshevSpectral` is DEFERRED to v5.0 BREAKING window per ADR-0035 §9.
7. **2D/3D/ND scope deferred**: `Grid2D`, `Grid3D`, `GridND` keep `CubicHermite` default and per-axis `.with_interp()` retains the existing variants only. Chebyshev tensor-product extension is OUT OF SCOPE for this ADR (no current ND gate forces it); deferred to a future v4.4+ ADR if measurement justifies.

## Algorithm

**Chebyshev-Lobatto nodes** (M+1 points on `[−1, 1]`, mapped to `[xmin, xmax]`):

```text
y_k = cos(k·π / M),         k = 0, 1, …, M
x_k = (xmax + xmin)/2 + (xmax − xmin)/2 · y_k
```

Nodes are computed via `libm::cos` at construction time and stored as `const`-array tables for M ∈ {8, 16, 32, 64, 128, 256, 512} (one-time precompute; ~16 KB total flash for 7 tables ≤512 floats each — well within no_std embedded budgets).

**Barycentric Lagrange interpolation** (Berrut-Trefethen 2004, formula 5.1):

```text
f(x) = Σ_{k=0..M} (w_k · f_k / (x − x_k))   /   Σ_{k=0..M} (w_k / (x − x_k))
```

where the **Chebyshev-Lobatto barycentric weights** are explicitly tabulated (no general Lagrange weight computation needed):

```text
w_k = (−1)^k · δ_k,    δ_0 = δ_M = 1/2,  δ_k = 1 (else)
```

This formula is **numerically stable** (Higham 2004 stability analysis; no catastrophic cancellation in the Lobatto case) and produces exact answers at the nodes (limit `x → x_k` removes the singularity). Cost per off-node sample: O(M) operations.

**Virtual-node construction (NORMATIVE)**: for each call to `interp(values, x)` with `InterpKind::ChebyshevSpectral`, the M+1 virtual node values `f_k = sample_quintic_1d(values, &grid, x_k)` are computed once (per call; **NOT** cached across calls). This costs O(M) QuinticHermite samples = O(M) × O(1) FD-stencil evaluations = O(M) operations. For M=64 default and N=512 grid, per-sample cost is ~64 QuinticHermite evals + ~64 barycentric ops = ~128 floating-point operations per `interp` call — comparable to a single CubicHermite cell sample (4 BC lookups + 4 weight evals + Horner). **The cost lives in the constant, not the asymptotic** — every kernel that already pays the K5 stencil cost (~9 sample calls per stencil) sees ~10–20% overhead, not order-of-magnitude.

**Spatial floor (asymptotic)**: for `f ∈ C^∞` analytic in a strip of half-width γ > 0 around `[−1, 1]`, Chebyshev tail truncation gives `|f − f_M| ≤ C · exp(−γM)`. At γ=1, M=64 → ≤ exp(−64) ≈ 10⁻²⁸ — far below the QuinticHermite ghost-data floor of O(dx⁶) at the virtual nodes (~1e-10). **Effective floor = max(virtual-node QuinticHermite floor, Chebyshev tail truncation, f64 ULP) ≈ max(1e-10, 1e-28, 1e-16) = 1e-10 for Gaussian-on-uniform-N=512**. To reach the ≤1e-15 G_PATH_EPS_CHEB_FLOOR gate, the virtual-node construction must use **FD stencil order ≥ 12** at construction time (already supported via `fd_scaled_prime` 6-pt formula in `grid_quintic.rs:91-102` if extended; engineer Wave specifies a 10-pt/12-pt extension OR an iterated refinement loop). **Alternative**: directly construct virtual-node values via Chebyshev's own infinite-order Greville analytic formula for the constant-a heat semigroup test case (Boyd Ch. 17); this bypasses QuinticHermite floor entirely for the regression gate. Engineer chooses at AC1-AC2 time.

## Backward compatibility

- `Grid1D::new` default UNCHANGED (`CubicHermite`).
- All existing `InterpKind` variants unchanged.
- `BoundaryPolicy` unchanged.
- Every existing kernel without `.with_chebyshev_sampling()` is byte-identical to v4.2.
- ζ-ladder defaults unchanged from ADR-0089 AMENDMENT 1 (CubicHermite for ζ⁴; QuinticHermite via direct K5 wiring for ζ⁶; ζ⁸ NEW with Chebyshev DEFAULT ON).
- Global default promotion to `ChebyshevSpectral` deferred to v5.0 BREAKING.
- `f32` and non-`f64` `SemiflowFloat` callers: `ChebyshevSpectral` returns `SemiflowError::Unsupported` (mirrors QuinticHermite f64-only constraint at `grid.rs:196-198`). Future MAY lift via generic libm, but `cos`/`sin` precision and SIMD coverage make `f64`-only the right v4.3 scope.

## Consequences

- **POSITIVE**: lifts single-cell-midpoint Gaussian floor from QuinticHermite 1e-10 to ChebyshevSpectral ≤1e-15 at N=512, M=64 (5 decades, ~10⁵× headroom); unblocks ζ⁸ Wave II RESURRECTION (Richardson ratio expected to recover from 3.067 toward theoretical 4 — and approach 8 in const-a regime where the cascade is the only error source); enables ζ⁴ default reactivation (Chebyshev removes the higher-order Taylor re-ordering that broke QuinticHermite's ζ⁴ default); provides a general spectral accuracy lever for any future high-order kernel (γ-A baseline, Strang2D axis-lift, drift-reaction characteristic foot — opt-in only, additive surface).
- **NEUTRAL**: per-sample cost ~10–20% over CubicHermite for typical kernels (M=64 virtual nodes + barycentric eval ≈ 128 flops; comparable to a single 4-pt Catmull-Rom cell sample); 1 new sympy oracle (`T_CHEB` barycentric verification) + 1 NEW const-array module ~50 LoC tables; ~200 LoC engineer Wave total; 0 new external dependencies (libm `cos` already available); 0 schema BREAKING.
- **NEGATIVE**: M-parameter calibration is a new tuning surface — engineer Wave defaults M=64 but ζ⁸ measurement may force M=128 or M=256 (cost grows linearly); virtual-node construction adds an indirection layer that complicates per-call profiling (per-sample cost is no longer "one stencil eval"); the floor depends on QuinticHermite ghost-FD order at construction (engineer must verify the FD chain doesn't dominate for ≤1e-15 gate — see "Algorithm" §spatial floor); `f32` callers locked out (Unsupported error); ChebyshevSpectral cannot use SIMD `catmull_rom` path (barycentric weights need scalar division loop).
- **BREAKING**: NONE. Additive opt-in only. ζ-ladder defaults unchanged from v4.2 (ζ⁴ stays CubicHermite default; ζ⁶ stays QuinticHermite via direct wiring; ζ⁸ NEW with its own default).
- **Schema bumps**: `properties.yaml` MINOR (1 NEW `G_PATH_EPS_CHEB_FLOOR` + 2 TIGHTENED ζ⁴ thresholds + 1 NEW `G_zeta8_const_a_richardson`). `traits.yaml` unchanged (no trait surface change; new builder is inherent). `math.md` NEW §9.2.7 NORMATIVE Chebyshev section (≤80 LoC) appended after §9.2.6.bis AMENDMENT 1.
- **Constitution check**: `boundary.rs` 437 LoC → ~445 LoC (1 new enum variant + rustdoc; under default 500 cap). `grid.rs` 499 LoC → ~510 LoC (1 dispatch arm; under Cohort 1 715 carve-out). `grid_quintic.rs` 208 LoC unchanged. NEW `grid_chebyshev.rs` ~200 LoC (under default 500 cap; not in Cohort). NEW const-array node tables module `grid_chebyshev_nodes.rs` ~50 LoC. No new Cohort needed.

## Implementation cost

- **Engineering**: ~200 LoC core (`grid_chebyshev.rs` + barycentric kernel + virtual-node construction) + ~50 LoC node-tables const arrays + ~80 LoC per-kernel builders (4 kernels × ~20 LoC each) + ~100 LoC NEW regression test + ~100 LoC NEW sympy oracle (`scripts/verify_chebyshev_barycentric.py`) + ~30 LoC dispatch wiring in `grid.rs`/`boundary.rs` = **~560 LoC total Wave delta**.
- **Days**: 3–4 working days (Wave I: AC1-AC7 Chebyshev infra + ζ⁴/ζ⁶ gate re-measurement); +2 working days (Wave II: AC8 ζ⁸ resurrection + new gate).
- **Risk**: virtual-node construction floor question (does QuinticHermite ghost-FD order limit us to 1e-10 or does the spectral lift dominate?) MAY force a sub-Wave to extend FD stencil from 6-pt to 10-pt in `grid_quintic.rs::fd_scaled_prime` (+~50 LoC). Engineer Wave I AC1-AC2 must measure this empirically before committing M=64 default.

## Alternatives considered

| Option | Decision | Rationale |
|---|---|---|
| **Native Chebyshev-Lobatto `Grid1D` variant** (BREAKING) | REJECTED | Forces every kernel using `f.sample()` to learn new node layout; blast radius dwarfs gain; v0.5–v4.x byte-equality contracts shatter. |
| **Interpolation-on-uniform-Grid1D (CHOSEN)** | ACCEPTED | Additive, mirrors ADR-0089 Option D pattern, preserves all existing kernels byte-identical, ≤1e-15 floor achievable via M=64 + extended FD ghost data. |
| **Septic Hermite (Option A in researcher verdict)** | REJECTED | Practical order-6 in PDE applications (NOT order-8); buys ~10² headroom over QuinticHermite — insufficient for ζ⁸ K=4 (needs 10⁸×). |
| **9-point compact FD order-8 (Option C)** | DEFERRED | Polynomial-bounded at order-8 — eventually hits dx⁸ floor; insufficient for ζ⁸ asymptote at K=4. Could be future fallback if Chebyshev cost prohibitive at M ≥ 128. |
| **FFT spectral derivative** | REJECTED | Requires periodic BC OR domain extension; our `Reflect`/`ZeroExtend`/`Dirichlet`/`Neumann` policies don't natively map; would force new BC infrastructure. Chebyshev handles bounded domain natively via barycentric. |
| **SEM / Gauss-Lobatto-Legendre (Option p-refinement)** | REJECTED | More infrastructure (Legendre polynomial eval ~500 LoC) for marginal gain over Chebyshev (both exponential); Chebyshev wins on simplicity + canonical literature density (Boyd, Trefethen). |
| **Default-ON for Chebyshev on ζ⁸** | ACCEPTED | ζ⁸ contract IS order-8; without Chebyshev floor lift the kernel cannot demonstrate its `order()` advertisement. Backward compat preserved because ζ⁸ is NEW (no v4.x baseline to maintain). |
| **Default-ON for Chebyshev on ζ⁴/ζ⁶** | REJECTED | ζ⁴ regression history (ADR-0089 AMENDMENT 1) demands measurement before flipping default; ζ⁶ already wins via direct K5 QuinticHermite wiring (3.868 calibrated). Opt-in keeps risk surface narrow. |

## Cross-references

- ADR-0015 — v0.7.0 QuinticHermite (foundation; this ADR adds peer `InterpKind` variant).
- ADR-0086 + AMENDMENT 1 — Path β G_zeta4 (Catmull-Rom dx-floor diagnosis; Chebyshev removes the floor).
- ADR-0088 + AMENDMENT 1 — ζ-ladder; Wave II HOLD now CONDITIONALLY released upon ADR-0090 ship.
- ADR-0089 + AMENDMENT 1 — Path ε QuinticHermite (predecessor; ADR-0089 AMENDMENT 1 §"Backward compatibility (STRENGTHENED)" §last-line explicitly defers full ζ⁴ order-4 to this ADR).
- ADR-0035 §9 — deprecation cycles (v5.0 BREAKING window for global default promotion).
- ADR-0068 — `boundary.rs` location of `InterpKind` (ADR-0090 adds `ChebyshevSpectral` variant here).
- `.dev-docs/research/verdicts/verdict-v4-3-research-waves.md` §Q1 — researcher recommendation.
- `.dev-docs/research/raw-findings-spatial-floor-extension.md` Query 2 — Boyd/Trefethen canonical literature.
- `.dev-docs/specs/chebyshev-wave.md` — engineer Wave I + Wave II spec.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_v4_1_0_v4_2_0_shipped.md` Insight #5 — pre-asymptotic regime interaction at ζ⁸ K=4.
- `crates/semiflow-core/src/boundary.rs:118-125` — `InterpKind` enum (Chebyshev variant added here).
- `crates/semiflow-core/src/grid.rs:267-287` — `Grid1D::interp` dispatch (Chebyshev arm added here).
- `crates/semiflow-core/src/grid_quintic.rs` — pattern reference for new `grid_chebyshev.rs` module structure.
- math.md §9.2.7 NEW NORMATIVE Chebyshev section (appended in this Wave's math.md amendment step).
- Boyd 1989/Dover 2000 *Chebyshev and Fourier Spectral Methods* — seminal canonical reference.
- Trefethen 2000 SIAM *Spectral Methods in MATLAB* — barycentric weights + differentiation matrix tables.
- Berrut & Trefethen 2004 *SIAM Review* 46:501 — barycentric Lagrange interpolation stability.
- Hesthaven & Warburton 2008 *Springer* — GLL/Lobatto reference (sibling to Chebyshev-Lobatto in spectral element community).

## AMENDMENT 1 — v5.0 B.3 BREAKING redesign (ADR-0104 H3 + H4 fix)

**Trigger**: ADR-0104 root-cause analysis confirmed two defects in the v4.x Chebyshev opt-in design:
- **H3 (OOB Runge divergence)**: `sample_chebyshev_1d` called with `x` outside `[xmin, xmax]` triggered Runge polynomial divergence (1e+4 at modest overshoot, 1e+11 at 2× overshoot). The Richardson ladder in ζ⁴/ζ⁶/ζ⁸ amplified this catastrophically.
- **H4 (false floor claim)**: Docs claimed "≤ 1e-15 spectral floor". Actual effective floor is ≈ 1e-10 because QuinticHermite K5 intermediate evaluations (virtual-node lookups within the semigroup step) dominate.

**Changes in v5.0**:
1. `OobPolicy` enum added to `boundary.rs` (4 variants: `Inherit`, `ForceReflect`, `ForcePeriodic`, `ForceZero`).
2. `InterpKind::ChebyshevSpectralWithBC { m: usize, oob_policy: OobPolicy }` replaces `InterpKind::ChebyshevSpectral` as the canonical variant.
3. `InterpKind::ChebyshevSpectral` deprecated with `#[deprecated(since = "5.0.0")]` shim.
4. `Grid1D::cheb_m(xmin, xmax, n, m)` constructor added as preferred entry point.
5. All 6 Chebyshev gate thresholds recalibrated to post-H3-fix truthful measurements (ADR-0097 AMENDMENT 1).
6. `properties.yaml` schema bumped to 2.0.0 (BREAKING threshold changes).

**Original `ChebyshevSpectral { m }` variant**: retained as deprecated shim for 12-month window per ADR-0035 §9 (BREAKING window expires 2027-05-27). The shim passes through to `sample_chebyshev_1d` without boundary enforcement — avoids breaking callers that never trigger OOB access, while new `ChebyshevSpectralWithBC` is safe by design.

**Cross-references**:
- ADR-0104 (H3 + H4 root cause + engineer wave spec)
- ADR-0097 AMENDMENT 1 (gate threshold recalibration)
- `docs/migration/v4-to-v5.md` (migration guide for callers)
- `crates/semiflow-core/src/grid_chebyshev.rs` — `out_of_domain_sample` + OOB dispatch added
- `crates/semiflow-core/tests/grid_chebyshev_bc_dispatch.rs` — 11 regression tests for all BC variants
