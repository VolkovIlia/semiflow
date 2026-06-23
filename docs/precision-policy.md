# Precision Policy

semiflow's composition types (`Strang2D`, `Strang3D`, `AdaptivePI`, …) are
generic over `F: SemiflowFloat ∈ {f32, f64}`. The math is the same; the
floating-point guarantees differ.

## Quick reference

| Property | f64 (default) | f32 |
|---|---|---|
| Order of consistency | unchanged | unchanged |
| Strang slope-gate band | ≥ −1.95 | ≥ −1.80 |
| 4th-order slope-gate band | ≥ −3.85 | ≥ −3.50 |
| 6th-order slope-gate band | ≥ −5.80 | **disabled** (no asymptotic window) |
| Sympy oracle gates | NORMATIVE | VACUOUSLY SATISFIED |
| Parallel bit-equality | f64↔f64 byte-equal | f32↔f32 byte-equal (within precision class) |
| Recommended τ_min | 1e-4 | 5e-3 |

## When to pick f32

- Memory bandwidth is the bottleneck (large 2D/3D grids).
- The slope band (one mantissa-decade wider than f64) is acceptable for your
  use case.
- 6th-order spatial accuracy is **not** required (the asymptotic regime is
  unreachable at 23-bit mantissa).
- Mesh refinement or parameter-sweep studies where halving grid memory
  enables a finer resolution.

## When to pick f64 (default)

- Tight error budgets (slope ≥ −1.95 required).
- Sixth-order spatial schemes (`Diffusion6thChernoff`,
  `TruncatedExp6thChernoff`).
- Any sympy-verified property is load-bearing in your pipeline.
- Long-time integration where rounding accumulates.

## What is and is not gated

semiflow ships two classes of correctness gate:

1. **Sympy oracle gates** (`T9N_*`, `T10N_*`, `T11N_*`) probe the *mathematical*
   structure of the Chernoff formula using arbitrary-precision symbolic
   algebra. They run on f64 only and remain authoritative for the math
   fidelity contract. On f32 they are *vacuously satisfied* — the gate
   describes the formula, not the rounding model.
2. **Slope gates** (G3, G3⁴, G3⁶, G4_NS2D_aniso, G5_3D) probe *numerical*
   self-convergence. They run on **both** f64 and f32 at the bands tabled
   above. G3⁶ has no asymptotic window on f32 and is therefore disabled
   there.

## Cross-precision composition

Mixed-precision composition (f32 grid + f64 internal compute, or vice versa)
is **not supported in v2.0**. A `Strang2D<X, Y, f32>` instance produces f32
output; a `Strang2D<X, Y, f64>` instance produces f64 output. They cannot be
chained.

## Gate Calibration Record (post-v4.2.0)

### v7.0.0 — OctonicHermite keystone + ζ⁶/ζ⁸ TRUTHFUL_ORDER PASS (ADR-0117–0120; all f64)

`InterpKind::OctonicHermite` (degree-9, O(dx¹⁰)) ships as the high-precision option.
Virtual-node floor at N=512 ≈ 9.1·10⁻¹⁶ (1638× below SepticHermite, 1.1e8× below
the retired QuinticHermite). `QuinticHermite` and `legacy-quintic` REMOVED (ADR-0109
12-month deprecation clock closed). `G_zeta6_TRUTHFUL_ORDER` and
`G_zeta8_TRUTHFUL_ORDER` promoted from HONEST-DEFER to active RELEASE_BLOCKING gates
via finest-pair (8→16) strategy (ADR-0119 AMENDMENT 2, N=8192, L=±32, T=10).

| Gate | Threshold | Measured | Notes |
|------|-----------|----------|-------|
| `G_OCTONIC_HERMITE_FLOOR` | ≤ 5e-15 | 9.1e-16 | OctonicHermite virtual-node floor (ADR-0117) |
| `G_zeta6_TRUTHFUL_ORDER` | ≤ -5.95 | verified PASS | finest-pair (8→16) lower-bound (ADR-0119 AMENDMENT 2) |
| `G_zeta8_TRUTHFUL_ORDER` | ≤ -7.95 | verified PASS | same strategy |
| `g_zeta4_const_a_richardson_ratio` | ≥ 3.1 (was 3.5) | 3.226 | re-calibrated; 3.5 was QuinticHermite-era (ADR-0120) |

### v6.0.0 — SepticHermite floor breakthrough (ADR-0110; all f64)

Default `InterpKind` changed `QuinticHermite` → `SepticHermite` (7-pt degree-7 Hermite; ADR-0109). Spatial floor 1e-10 → 1.89e-12 (67× lower). `InterpKind::ChebyshevSpectral { m }` REMOVED (ADR-0104 12-month clock fulfilled). Two new RELEASE_BLOCKING gate classes ship. Note: `QuinticHermite` and the `legacy-quintic` feature were subsequently REMOVED at v7.0.0 (ADR-0109 12-month clock closed):

| Gate | Threshold | Measured | Notes |
|------|-----------|----------|-------|
| `G_SEPTIC_HERMITE_FLOOR` | ≤ 5e-12 | 1.89e-12 | SepticHermite virtual-node floor (ADR-0109) |
| `G_zeta4_TRUTHFUL_ORDER` | ≤ -3.5 | -3.6573 | TRUE math order-4 in pre-asymp regime (ADR-0110 AMENDMENT 1); margin 0.16 |

const-a Chebyshev gates (`G_zeta4/6/8_const_a_cheb`, measured 3.226 / 3.870 / 3.067) live in the pre-asymp-temporal-transition regime (math.md §40.5.bis) where §39.2 saturation formula does not apply; thresholds REVERT to v5.0.0 baselines {≥ 3.1, ≥ 3.8, ≥ 3.0} (ADR-0109 AMENDMENT 1 — NOT a downward recalibration).

ζ⁶/ζ⁸ `TRUTHFUL_ORDER` empirical gates HONEST-DEFERRED to v7.0+ (OCTONIC-Hermite + higher-order K5 stencil required simultaneously; root cause: off-by-one GLOBAL/LOCAL scope error + K5 3-point spatial stencil saturation at large T_FINAL, diagnosed via `T_ZETA_TRUTHFUL_ORDER_AMENDMENT1` 4/4 sympy PASS). ζ⁶/ζ⁸ academic honesty maintained via existing `G_zeta_K_const_a_richardson_cheb` gates + `T23N_zeta6` sympy oracle + Galkin-Remizov 2025 IJM Theorem 3.1 LOCAL tangency (PROVEN).

Sympy oracles added:

| Gate | Status | Release |
|------|--------|---------|
| `T_SEPTIC_HERMITE` | 6/6 NORMATIVE | v6.0.0 (ADR-0109) |
| `T_ZETA_TRUTHFUL_ORDER` | 6/6 NORMATIVE | v6.0.0 (ADR-0110; historical — original 6-sub-check oracle) |
| `T_ZETA_TRUTHFUL_ORDER_AMENDMENT1` | 4/4 NORMATIVE | v6.0.0 (ADR-0110 AMENDMENT 1; authoritative for engineer-wave calibration) |
| `T_ZETA_CONST_A` | 6/6 NORMATIVE | v6.0.0 (ADR-0109 §40.5.bis three-regime taxonomy) |

Schema bumps: `properties.yaml` 2.2.0 → 3.0.0 MAJOR; `traits.yaml` 2.3.0 → 3.0.0 MAJOR; constitution v1.9.1 → v2.0.0 MAJOR.

### v5.1.0 — Chebyshev saturation formula §39 NORMATIVE (ADR-0108)

Measured const-a Chebyshev slopes {3.226, 3.870, 3.067} mathematically CONFIRMED as the floor-saturated CEILING of the QuinticHermite-bound K5 sampler at N=512. NORMATIVE saturation formula (math.md §39.2):

```
slope_eff(N) = log₂((c·τ^{m+1} + φ) / (c·(τ/2)^{m+1} + φ))
```

where φ ≈ 1e-10 (QuinticHermite floor). Formula reproduces all three measurements to ±0.0001 — confirms measurements are floor-saturated optima, NOT regressions. v5.0.0 thresholds {≥ 3.1, ≥ 3.8, ≥ 3.0} CODIFIED as truthful saturation-bounded values; ADR-0104 rev-prediction inconsistency CLOSED.

| Gate | Status | Release |
|------|--------|---------|
| `T_CHEBYSHEV_SLOPE_LIMIT` | 5/5 NORMATIVE | v5.1.0 (ADR-0108; PRE-FLIGHT oracle for saturation formula) |

Schema: `properties.yaml` 2.1.0 → 2.2.0 MINOR (additive; no gate-threshold edits). ZERO Rust source change; ZERO API change.

### v5.0.0 — Chebyshev recalibration (ADR-0104 H3+H4 fix; all f64)

Chebyshev-path gate thresholds corrected from optimistic predictions to
truthful measured values. The effective floor of the K5 + QuinticHermite path
is ≈ 1e-10 (not 1e-15 as previously documented). Richardson log₂ values are
bounded by this floor.

| Gate ID | Threshold | Measured | Notes |
|---------|-----------|----------|-------|
| `G_zeta4_const_a_cheb` | ≥ 3.1 | 3.2260 | recalibrated TRUTHFUL (v4.3 prediction 6.5 retired) |
| `G_zeta4_var_a_slope_cheb` | ≤ 0.1 | −0.0188 | floor-dominated; not-diverging |
| `G_zeta6_const_a_cheb` | ≥ 3.8 | 3.8701 | |
| `G_zeta6_var_a_slope_cheb` | ≤ 0.5 | −0.1539 | floor-dominated |
| `G_zeta8_const_a_cheb` | ≥ 3.0 | 3.0667 | |
| `G_zeta8_var_a_slope_cheb` | ≤ 0.1 | 0.0561 | floor-dominated |

Calibration rule: Option E (ADR-0086 AMENDMENT 1) applied to Chebyshev path.

### v5.0.0 — Sympy oracles added (NORMATIVE, f64 only)

| Gate | Status | Release |
|------|--------|---------|
| `T_GR_2025_THM3` | 5/5 NORMATIVE | v5.0.0 (ADR-0106 — G_zeta4 escalation RESOLVED) |
| `T_ADJOINT_FP_TIGHTNESS` | 6/6 NORMATIVE | ADR-0107 (post-v5.0 math creation; engineer-wave G_ADJOINT_FP_TIGHTNESS_VAGUE deferred) |
| `T_CHEBYSHEV_WEIGHTS` | 2/2 NORMATIVE | v5.0.0 (ADR-0104) |

### v4.8.0 — SubordinatedChernoff gates (ADR-0103)

| Gate | Threshold | Status |
|------|-----------|--------|
| `G_SUBORD_ORDER1` | slope ≤ −0.95 for ≥ 2 of 3 backends | PASS (3/3) |
| `T_SUBORD` | 5/5 NORMATIVE | v4.8.0 |

### v4.7.0 — LadderRung trait oracle (ADR-0100)

| Gate | Status |
|------|--------|
| `T_LADDER_RUNG` | 4/4 NORMATIVE |

### v9.0.0 — New types inherit existing precision bands (ADR-0154/0155/0156/0159; all f64)

The three new kernel families added at v9.0.0 do not introduce new slope gates
or sympy oracles. Their precision characteristics are inherited from existing
types or are subject to honest scope restrictions.

**`ReverseChernoff<F>` / `CheckpointSchedule` (ADR-0156):**
The gradient computation uses the forward-mode `Dual<F>` pass (§51.4), which
produces 0-ULP results compared to a separate forward-mode reference — no
precision-band retuning required. The primal forward pass uses
`DiffusionChernoff<F>`, which is already covered by the Strang slope-gate band
(≥ −1.95 for f64, ≥ −1.80 for f32). **NARROW scope (§51.5):** constant-a
`DiffusionChernoff` only.

**`TtChernoff<F>` / `TtState<F>` (ADR-0159):**
Per-axis shift arithmetic is identical to the 1D `DiffusionChernoff` Chernoff
kernel (eq. 50.3). The storage bound `O(d·n·r²)` is polynomial in d **only for
the linear diagonal-A Gaussian class** where `r ≤ d/2` (Rohrbach et al. 2022).
For off-diagonal A, variable coefficients, or non-Gaussian IC the TT rank is
not algebraically capped and may blow up — this is a **research-track** regime;
no precision gate applies. No new slope gates at v9.0.0.

**`GridlessChernoff<F, D>` / `ParticleReduction` (ADR-0155):**
The 3-branch per-axis Chernoff step (eq. 50.3) is order-2 for diagonal A
(commuting-axes exact splitting). Particle-reduction quality (`WeightedVoronoi`)
is subject to a committed negative result for high-d: the d=2 validated scope
marker is normative. Off-diagonal A, variable coefficients, and d > ~10 are
**research-track** — rank and particle-count explosions are expected.
No new slope gates at v9.0.0.

### v9.1.0 — CoupledTtChernoff inherits TtChernoff precision (ADR-0162)

`CoupledTtChernoff<F>` adds a spectral (FFT-diagonal) coupling factor to the
tensor-train carrier. The spectral apply is machine-exact at 1.2e-15 vs dense `expm`
(f64; `G_TT_COUPLED_EXACT`). No new slope bands: the coupling factor is
exact-in-time (circulant commutativity), not order-2 approximate. The per-axis
Chernoff shift uses the v9.1.0 cubic-band-split (QTT-op-rank ≤ 3); shift
precision is ≤ O(dx⁴) local error. **NARROW scope**: constant-coefficient
correlated-Gaussian / adjacent-pairs only (see ADR-0162 §"Fail-loud construction").
No new slope gates at v9.1.0.

### v9.2.0 — S³ POC types: precision matches in-class guarantees (ADR-0169)

All six `s3-poc` evolvers operate via FFT, tridiagonal solves, and Strang splitting —
no new floating-point mechanisms. Their precision characteristics:

| Type | Precision characteristic |
|------|--------------------------|
| `S3DriftSpectralEvolver` | EXACT-in-time: complex Fourier symbol; imag residue ≤1e-12 returned by `evolve` (`g_s3_drift_spectral`) |
| `S3DenseCouplingEvolver` | EXACT-in-time for rank-1-dense coupling; same spectral path as `S3DriftSpectralEvolver` |
| `S3VarCoefEvolver` | ORDER-2 (slope ≤ −1.95, `g_s3_varcoef_spectral`); inherits Strang splitting band |
| `S3NonSepVarCoefEvolver` | ORDER-2 (slope ≤ −1.95, `g_s3_nonsep_varcoef`); inherits Strang splitting band |
| `S3BurgersColeHopf` | EXACT-in-time via Cole-Hopf transform; heat-equation spectral step machine-exact |
| `S3ReactionDiffusion<F>` | ORDER-2 (Strang-split); inherits Strang splitting band |

No new precision bands required (all within existing ≥ −1.95 order-2 policy for f64).
The `s3-poc` surface does not expose an f32 path in v9.2.0 — f32 precision for S³ types
is uncharacterised and deferred.

## References

- ADR-0046 "Precision-policy bands" — full derivation of the bands.
- ADR-0025, ADR-0026 — generic-over-`F` design.
- ADR-0086 AMENDMENT 1 — Option E calibration rule (Richardson ratio interpretation).
- ADR-0104 — Chebyshev H3 OOB fix + H4 floor correction; gate table authority.
- ADR-0106 — G_zeta4 Galkin-Remizov 2025 IJM Theorem 3 prefactor RESOLVED.
- ADR-0108 — ζ⁴/ζ⁶/ζ⁸ Chebyshev slope deficit diagnostic; §39 saturation formula NORMATIVE.
- ADR-0109 — SepticHermite floor breakthrough; v6.0.0 BREAKING Window #3.
- ADR-0110 + AMENDMENT 1 — `G_zeta_K_TRUTHFUL_ORDER` pre-asymptotic gate framework; ζ⁶/ζ⁸ HONEST DEFER.
- `contracts/v2/wave5-precision-policy.md` — machine-readable band tables.
