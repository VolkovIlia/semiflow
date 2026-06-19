# ADR-0046: Precision-Policy Bands (f32 vs f64 Composition Tolerances)

**Status**: ACCEPTED — Amendment 1 (2026-06-12, ACCEPTED) replaces the
unattainable f32 spatial-slope assertions for `gf1_2d_f32` / `gf2_3d_f32` with
honest **floor-band gates** (Option B) after the regime map proved f32 is
round-off-limited with no asymptotic band.
**Date**: 2026-05-20 (Amendment 1: 2026-06-12)
**Architect**: ai-solutions-architect
**Supersedes**: none
**Superseded by**: none
**Related**: ADR-0025 (Generic-over-Float v0.9.0), ADR-0026 (ChernoffFunction
generic), ADR-0045 (Zero-copy bindings + f32 lift), ADR-0163 (f64 G3⁶-2D floor
recalibration — the precedent Amendment 1 mirrors), ADR-0120 (f64 1D floor-free
basket precedent), ADR-0053 (G13 f32 order-2 coarse-basket precedent).

---

## Context

After Wave 5 lifts the f64 lock on `Strang2D`/`Strang3D` parallel composition
(ADR-0045 §5.3), callers can run composition on `F = f32`. The numerical
question this raises is **what convergence guarantee survives**.

The semiflow-core math fidelity contract today (post-v0.9.0, ADR-0025/ADR-0026)
makes two kinds of claim:

1. **Sympy oracle gates** (T9N_*, T10N_*, T11N_*): closed-form τ-Taylor
   expansions of the Chernoff kernel, derived in arbitrary precision in
   `.dev-docs/verification/scripts/`. These probe the *mathematical*
   structure of the formula — not its floating-point implementation.
2. **Slope gates** (G3, G3⁴, G3⁶, G3⁶-2D, G4_NS2D_aniso, G5_3D): numerical
   self-convergence on a grid sequence with τ ∈ {τ₀, τ₀/2, τ₀/4, ...}; the
   expected log-log slope is the *order* claimed by the kernel.

The IEEE-754 floor matters for slope gates but not for sympy gates:

- **f64**: 52-bit mantissa → ≈ 1e-16 relative epsilon. Slope gates run with
  τ_min ~ 1e-4 and reference grid N ~ 4096; the dominant error is truncation,
  rounding is invisible.
- **f32**: 23-bit mantissa → ≈ 1.2e-7 relative epsilon. At τ ~ 1e-4 the
  rounding floor swamps a τ² truncation term (1e-8 truncation vs 1e-7 rounding
  per step × 100 steps ≈ 1e-5 accumulated noise). A slope gate at "≥ −1.95" on
  f64 cannot mechanically run on f32 — the asymptote bends at the rounding
  floor.

The honest reading: f32 composition has the same **order of consistency** as
f64 (it's the same formula), but it cannot demonstrate that order beyond a
precision-dependent cutoff τ_floor. The slope tolerance band shifts to
accommodate the cutoff; the cutoff is not a bug.

---

## Decision

### 3.1 f64 path — unchanged

All gates retain v1.0.0 tolerances:

| Gate | Tolerance | Notes |
|---|---|---|
| T9N_* (sympy, var-a 1D) | EXACT match | arbitrary-precision oracle, no IEEE-754 sensitivity |
| T10N_* (sympy, 3D tensor) | EXACT match | as above |
| T11N_* (sympy, non-sep 2D) | EXACT match | as above |
| G3 (1D, order 2) | slope ≥ −1.95 | self-convergence, τ-sweep |
| G3⁴ (1D, order 4) | slope ≥ −3.85 | as above |
| G3⁶ (1D, order 6) | slope ≥ −5.80 | as above |
| G3⁶-2D (2D, order 2 in τ) | slope ≥ −1.95 | flagship, ADR-0020 |
| G4_NS2D_aniso (non-sep 2D) | slope ≥ −1.95 | ADR-0023 |
| G5_3D (3D, order 2) | slope ≥ −1.95 | ADR-0024 |

### 3.2 f32 path — relaxed bands; sympy gates VACUOUS

**Sympy gates**: NOT required on f32. Sympy uses `mpmath` arbitrary precision;
the gate is *mathematical*, not numerical. f32 rounding is a property of the
runtime, not of the closed-form expansion. Forcing sympy to "round to f32"
would invent a fake oracle (the rounding model in mpmath is not bit-for-bit
identical to the LLVM f32 ABI). VACUOUSLY SATISFIED — same status as MCP
under constitution Override #2.

**Slope gates on f32** — relaxed bands:

| Gate | f64 tolerance | f32 tolerance | Rationale |
|---|---|---|---|
| G3 (1D, order 2) | ≥ −1.95 | ≥ −1.80 | one f32 mantissa-decade rounding floor lifts τ_min sweep from 1e-4 to 5e-3, narrowing the asymptotic window |
| G3⁴ (1D, order 4) | ≥ −3.85 | ≥ −3.50 | as above; higher order is more sensitive to rounding floor |
| G3⁶ (1D, order 6) | ≥ −5.80 | **N/A — gate DISABLED on f32** | 6th-order has rounding floor at ~1e-3 in τ; no asymptotic window left at f32 mantissa width |
| G3⁶-2D (2D flagship) | ≥ −1.95 | **N/A — flagship is f64-only** | bit-equal proptest is f64; flagship tag remains f64 |
| G4_NS2D_aniso | ≥ −1.95 | ≥ −1.80 | same widening as G3 |
| G5_3D | ≥ −1.95 | ≥ −1.80 | same widening as G3; n_steps = 200 sufficient to clear rounding floor at τ_min = 5e-3 |
| Strang2D-3D parallel bit-equality (ADR-0018) | byte-equal | **DOES NOT APPLY** | bit-equal is f64 SIMD path; f32 path has its own (looser) byte-equal vs serial |

### 3.3 New f32 byte-equal carve-out

For the f32 path, the "bit-equal" gate from ADR-0018 is replaced by **byte-equal
of f32 serial vs parallel** (no cross-precision comparison). I.e.:
`strang2d_parallel<f32>::apply == strang2d_serial<f32>::apply` byte-for-byte
on a 5-grid × 3-thread matrix. This is the analog of the f64 ADR-0018 gate,
applied within the f32 precision class. `tests/strang2d_parallel_bit_equal_f32.rs`
(new) holds the regression.

### 3.4 Documentation: `docs/precision-policy.md`

A new rustdoc-facing document at `docs/precision-policy.md` summarises ADR-0046
for end users. Outline (≤ 1 screen):

```markdown
# Precision Policy

semiflow-core's composition (`Strang2D`, `Strang3D`, `AdaptivePI`, …) is
generic over `F: SemiflowFloat ∈ {f32, f64}`. The math is the same; the
floating-point guarantees differ.

| Property | f64 | f32 |
|---|---|---|
| Order of consistency | 2 (Strang) / 4 (4th-order) / 6 (6th-order) | unchanged |
| Slope-gate asymptotic | ≥ −1.95 (Strang) | ≥ −1.80 (Strang) |
| 6th-order slope gate | ≥ −5.80 | DISABLED (no asymptotic window) |
| Sympy oracle | NORMATIVE | VACUOUSLY SATISFIED (sympy is arb-prec) |
| Parallel bit-equality | f64↔f64 byte-equal | f32↔f32 byte-equal (no cross-precision) |
| Recommended τ_min | 1e-4 | 5e-3 |

Choose f32 when: memory bandwidth is the bottleneck, the slope band is
acceptable, and 6th-order accuracy is not needed.

Choose f64 (default) when: tight error budgets, 6th-order spatial schemes,
or any sympy-verified property is load-bearing.

See ADR-0046 for the full derivation of the bands and the rounding-floor
argument.
```

### 3.5 Test wiring

`tests/generic_float_strang.rs` (new) runs the slope sweep on Strang2D and
Strang3D for `F ∈ {f64, f32}`, asserting:

```rust
const F64_BAND: f64 = -1.95;
const F32_BAND: f64 = -1.80;
assert_slope_at_least::<f64>(strang2d_slope_sweep::<f64>(), F64_BAND);
assert_slope_at_least::<f64>(strang2d_slope_sweep::<f32>(), F32_BAND);
```

Existing sympy verification scripts in `.dev-docs/verification/scripts/`
remain f64-only and unchanged; the `generic_float_strang.rs` rustdoc cites
this ADR explicitly.

---

## Consequences

### Positive
1. **Honest math fidelity**: f32 gates exist and are gated in CI, but at a band
   that matches IEEE-754 reality. No fake oracle.
2. **f32 path is a first-class citizen** for memory-bound workloads (parameter
   sweeps, multi-resolution mesh refinement).
3. **f64 path is unchanged**: existing users see no behavioural shift, no new
   gates to interpret, no narrowed tolerances.

### Negative
1. **Two slope-gate tables** to maintain. Mitigation: `xtask` test runner
   parameterises a single `assert_slope_at_least` call with the band per
   precision.
2. **6th-order gate DISABLED on f32**: a user writing `Diffusion6thChernoff<f32>`
   will get the right *order* of consistency but cannot prove it
   asymptotically. Mitigation: rustdoc on `Diffusion6thChernoff` cites
   ADR-0046; the type definition is unchanged.

---

## Acceptance Criteria

1. `tests/generic_float_strang.rs` passes with f32 band −1.80 and f64 band
   −1.95.
2. `tests/strang2d_parallel_bit_equal_f32.rs` (new) passes — f32 serial =
   f32 parallel byte-for-byte.
3. `docs/precision-policy.md` exists and is linked from `crates/semiflow-core/src/lib.rs`
   crate-level rustdoc.
4. Existing 18 sympy gates and 6 f64 slope gates remain green.
5. ADR-0046 referenced from ADR-0045 §5.5 and from the
   `tests/generic_float_strang.rs` rustdoc.

---

## Alternatives Considered

- **A1: Run sympy gates "as if" rounded to f32**. Rejected: mpmath ↔ LLVM f32
  rounding is not bit-equal; would invent a fake oracle.
- **A2: Disable f32 entirely on composition**. Rejected: serial Strang2D/3D
  is already f32-capable (ADR-0025); leaving parallel f64-only creates a
  permanent asymmetry and blocks Wave 5's memory KPI.
- **A3: One unified tolerance band across both precisions**. Rejected:
  f32 cannot pass −1.95 with current τ_min; relaxing f64 to f32's band
  weakens existing guarantees. Per-precision bands are the only honest
  choice.
- **A4: Allow user to override the band via env var**. Rejected: gates are
  CI hygiene, not user configuration. Bands are project-defined.

---

## Out of Scope

- **f16 / bf16**: ADR-0025 sealed `SemiflowFloat` at `{f32, f64}`.
- **Mixed-precision composition** (f32 input, f64 internal): possible future
  ADR; not in Wave 5.

---

## References

- ADR-0018, ADR-0025, ADR-0026, ADR-0035, ADR-0042, ADR-0045.
- `docs/precision-policy.md` (new in Wave 5).
- `contracts/v2/wave5-precision-policy.md` (sister contract holding the
  band tables in machine-readable form).
- math.md §11.1.bis (Strang palindromic structure, precision-agnostic).
- IEEE-754-2019 §3.6 ("binary32" 23-bit mantissa, "binary64" 52-bit
  mantissa).

---

## Amendment 1 (2026-06-12) — f32 spatial-slope bands are floor-saturated; recalibrate the basket to floor-safe COARSE grids

### What broke and why it is NOT the method

The two **spatial self-convergence** f32 gates in
`crates/semiflow-core/tests/generic_float_strang.rs` —
`gf1_2d_f32_slope` (`Strang2D<f32>`) and `gf2_3d_f32_slope` (`Strang3D<f32>`) —
fit the OLS log-log slope on the basket `N ∈ {64, 128}` and gate at `−1.80`
(§3.2). They now report **−1.3459** (2D) and **−1.4473** (3D), OUTSIDE the band.

This is **pre-existing**: byte-identical `self_err` at baseline commit `5069d85`
(verified via `git worktree`); the recent tech-debt refactor was bit-identical
and did NOT cause it. The Strang composition is correct (the f64 twins
`gf1_2d_f64_slope` / `gf2_3d_f64_slope` PASS at `−1.95`).

Root cause: the **f32 round-off floor**, the exact analogue of the f64
SepticHermite interpolation floor that ADR-0163 fixed for the G3⁶-2D flagship.
f32 has machine-ε ≈ 1.2e-7; the `WrapDiff` Catmull-Rom path rounds intermediate
coefficients to ~7 digits per step, accumulating over `n_steps = 200` to a
`self_err` floor of ≈1e-5..4e-5 (§3 of the parent ADR predicted this). The data:

| Gate | N=32 self_err | N=64 self_err | N=128 self_err | slope{64,128} |
|------|---------------|---------------|----------------|---------------|
| gf1_2d_f32 | 4.0233e-5 | 3.0756e-5 | 1.2100e-5 | **−1.3459** |
| gf2_3d_f32 | 4.8995e-5 | 4.0472e-5 | 1.4842e-5 | **−1.4473** |

At N=32 `self_err` is ALREADY ≈4e-5 — the f32 floor DOMINATES the dx²
discretization error across the WHOLE {32,64,128} basket. The 32→64 segment
(4.0e-5→3.1e-5, ratio ~1.3) is almost flat = pure floor; the 64→128 segment
(3.1e-5→1.2e-5) is only partially de-floored. So the measured slope reflects
floor-noise decay, **not** the true order-2. The `−1.80` band in §3.2 assumed
{64,128} was floor-safe ("the floor is well below the spatial error for short
sweeps") — the data falsifies that assumption for this dense-grid heat problem.

This is the f32 sibling of the floor diagnosis already accepted twice in the
f64 line (ADR-0120 1D, ADR-0163 2D) and once in the f32 line (ADR-0053 G13,
which moved to a coarse `n_steps ∈ {1,2,3}` basket and an honest `−1.50` band).

### Diagnostic-first (identical-parameter regime map)

`crates/semiflow-core/tests/generic_float_strang_f32_regime.rs` (NEW, `#[ignore]`d,
non-asserting) sweeps a WIDER, mostly-COARSER range
`N ∈ {8, 12, 16, 24, 32, 48, 64}` at the **identical** gate parameters
(T=0.2, n_steps=200, a=0.1, [−5,5]ᵈ, `WrapDiff` Catmull-Rom). Because the
parameters match the gate exactly, any flattening is unambiguously the f32 floor,
never a code or temporal artifact (mirrors the ADR-0163 method note). It prints
per-N `self_err` plus consecutive segment slopes for both 2D and 3D. Run:

```text
cargo test -p semiflow-core --features parallel,simd,slow-tests --release \
  -- --ignored --nocapture f32_regime
```

> **STATUS: regime map RUN — COMPLETE (2026-06-12).** The diagnostic was run at
> the identical gate parameters; the measured maps are pasted in
> "Regime-map result" below. They show NO floor-safe asymptotic band, so
> **Option A is impossible and Option B is ADOPTED** (see "Decision" below). The
> gate constants in `generic_float_strang.rs` are now FINALIZED to the Option-B
> floor-band predicates.

### Regime-map result (RUN 2026-06-12, identical gate params: T=0.2, n_steps=200, a=0.1, [−5,5]ᵈ, WrapDiff Catmull-Rom)

```text
GF1_2D f32 (Strang2D<f32>):
  N=  8  dx=1.4286  self_err=6.2828e-3  seg=—
  N= 12  dx=0.9091  self_err=3.6038e-4  seg=-7.050
  N= 16  dx=0.6667  self_err=8.5467e-4  seg=+3.002   ← err INCREASED
  N= 24  dx=0.4348  self_err=2.5243e-4  seg=-3.008
  N= 32  dx=0.3226  self_err=4.0233e-5  seg=-6.384
  N= 48  dx=0.2128  self_err=4.4286e-5  seg=+0.237   ← err INCREASED
  N= 64  dx=0.1587  self_err=3.0756e-5  seg=-1.267

GF2_3D f32 (Strang3D<f32>):
  N=  8  5.6048e-3  seg=—
  N= 12  4.1926e-4  seg=-6.395
  N= 16  1.1167e-3  seg=+3.405   ← err INCREASED
  N= 24  3.6138e-4  seg=-2.782
  N= 32  4.8995e-5  seg=-6.946
  N= 48  6.7174e-5  seg=+0.778   ← err INCREASED
  N= 64  4.0472e-5  seg=-1.761
```

**Interpretation (decisive — no asymptotic band).** `self_err` is
**NON-MONOTONE** (it *increases* at N=16 and again at N=48 in both 2D and 3D) and
the consecutive segment slopes swing wildly between **+3.4 and −7.0**. This is
**f32 round-off NOISE, not a convergence law** — a real order-2 law would give
monotone decrease with segment slopes clustered near −2. There is **no
floor-safe band**: only N=8 (`self_err` ≈ 6e-3) is clearly above the
≈1e-5..1e-3 f32 floor, and N=8 (dx ≈ 1.43) is pre-asymptotic (order not yet
developed at the coarsest grid). With **at most ONE above-floor point** you
cannot fit a real order — every multi-point fit straddles the floor and measures
noise decay. **Option A (recalibrate to a floor-safe coarse basket) is therefore
impossible**: the predicted N∈{12,16,24} window the recommendation hoped for is
exactly where the non-monotone +3.0/+3.4 floor-noise excursion lives (N=12→16
err *rises*). f32 is genuinely round-off-limited for this dense 2D/3D Strang heat
problem and **cannot demonstrate order-2**.

### Candidate resolutions

**Option A (preferred IF a floor-safe coarse band exists).** Recalibrate the
gate basket to COARSE grids where dx² discretization error >> the ~1e-5 floor,
so f32 genuinely demonstrates ≈order-2, and set the window honestly from the
measured in-band segment slopes. This is the f32 analogue of ADR-0163/ADR-0120:
a correctness-preserving basket move, NOT a relaxation. The basket's FINEST
point must keep `self_err ≳ 100×` floor (≥ ~1e-3) AND avoid the very-coarsest
pre-asymptotic point (order not yet developed at the largest dx). Predicted
floor-safe band: the mid-coarse grids around `N ∈ {12, 16, 24}` (dx ≈ 0.91,
0.67, 0.43; dx² error ≈ 8e-4..2e-3, i.e. ~100×–200× the floor) — with N=8
likely pre-asymptotic and N≥32 already floor-contaminated. **Do NOT keep a
−1.80 slope on a floored basket; a slope gate is only honest on a floor-safe
band where the slope is REAL.**

**Option B (IF no floor-safe band exists — even the coarsest sane grid is
at/below the floor, or is pre-asymptotic).** This is an HONEST finding, not a
hack. Re-state the f32 gate to test what f32 CAN guarantee for this problem: a
**round-off-floor band gate** — assert that `self_err` stays BELOW a documented
f32 accuracy ceiling (≈5e-5, the observed floor + margin) AND is
monotone-non-increasing in N (error never blows up as the grid refines). This
catches REAL regressions (method broke → error explodes; or floor lifts →
ceiling tripped) without being a tautology, and explicitly documents that f32
is round-off-limited here so a −1.80 slope is physically unattainable. Slope is
DROPPED for these two gates (not lowered to −1.3 to sneak past — that would be
the hack the directive forbids).

### Decision — Option B ADOPTED (Option A impossible)

The regime map falsifies Option A: there is no coarse sub-basket with both
`self_err ≳ 100×` floor AND a developed order-2 (every multi-point window
straddles the non-monotone floor noise; the only clearly-above-floor point, N=8,
is pre-asymptotic). **Option B is adopted for BOTH `gf1_2d_f32` and
`gf2_3d_f32`.** The unattainable `−1.80` slope assertion is REMOVED (slope is
dropped entirely for these two gates — NOT lowered to −1.3 to sneak past, which
would be the hack the directive forbids) and replaced by a **f32 round-off
floor-band gate** that asserts the regression-sensitive invariants f32 CAN
guarantee. There is no dense-grid 1D f32 spatial analogue (the 1D f32 path is
gated by G11/G13 on floor-safe coarse baskets per §3.2 / ADR-0053), so scope is
exactly these two gates.

The gates were RENAMED `gf1_2d_f32_slope → gf1_2d_f32_floor_band` and
`gf2_3d_f32_slope → gf2_3d_f32_floor_band` (semantics changed, so the name
follows). The f64 gates `gf1_2d_f64_slope` / `gf2_3d_f64_slope` and
`SLOPE_GATE_F64 = −1.95` are UNCHANGED and still PASS. The now-dead
`SLOPE_GATE_F32` constant was removed.

#### Finalized Option-B predicates (exact, in `assert_f32_floor_band`)

Self-convergence on `N ∈ {32, 64, 128}` (coarse→fine), with the two band
constants:

```rust
const F32_CEILING: f64 = 5.0e-4;  // accuracy ceiling
const F32_FLOOR:   f64 = 1.0e-7;  // floor-of-floors (anti-tautology wall)
```

The gate asserts, for `errs` = `[self_err(32), self_err(64), self_err(128)]`:

1. **Band containment** — every probe inside `[F32_FLOOR, F32_CEILING]`:
   `assert!((F32_FLOOR..=F32_CEILING).contains(&e))` for each `e`.
   - Upper wall (`≤ 5e-4`) fails LOUD if the f32 method diverges or a kernel
     breaks (error blows up at any N).
   - Lower wall (`≥ 1e-7`) prevents a degenerate-zero tautology: a `self_err`
     of ~0 would mean self-convergence collapsed (broken probe wiring), not a
     real solve — so that is ALSO a failure, not a pass.
2. **Refinement-helps** — `assert!(finest < coarsest)`, i.e.
   `self_err(128) < self_err(32)`. Even though per-segment behaviour is
   round-off noise, the gross coarse→fine trend must still reduce error; a
   method that gets *worse* under refinement trips this.

**Ceiling/band justification from the data.** Observed in-basket `self_err`
spans `self_err(32)` ≈ 4.0e-5..4.9e-5 (worst, coarsest) down to `self_err(128)`
≈ 1.2e-5..1.5e-5 (finest); floor-noise excursions top out at ≈6e-3 only at the
pre-asymptotic N=8, which is OUTSIDE the gate basket. The ceiling `5.0e-4` sits
~10× above the worst benign in-basket value (`self_err(32)` ≈ 5e-5) and ~30×
above the finest — wide enough to never trip on benign f32 jitter, tight enough
to catch a gross (≳ 13×) uniform blow-up directly. The floor `1e-7` is ~120×
below the smallest observed in-basket error, so it only ever fires on a
genuinely degenerate (≈ zero) result.

**Regression sensitivity (verified by construction).** If the finest-grid
`self_err` were 10× larger (≈ 1.5e-4 vs the observed ≈ 1.5e-5), it would EXCEED
`F32_CEILING = 5e-4`? No — 1.5e-4 < 5e-4, so predicate (1) alone would still
pass at 10×. The bite comes from BOTH predicates together plus the headroom
choice: a 10× regression at the finest grid that does NOT also lift the coarse
grids would make `finest (1.5e-4) > coarsest (4e-5)` → predicate (2)
"refinement-helps" FAILS LOUD. A uniform ≳ 13× blow-up across all N trips the
ceiling directly (`> 5e-4`). A divergent method (error growing with N) trips
predicate (2) immediately. Thus the gate catches the three real failure modes —
divergence, finest-grid blow-up, and refinement-makes-it-worse — without
asserting an unattainable slope and without being vacuously true. (The directive's
"10× finest" scenario is caught via predicate (2); a stronger ceiling-only
catch would require lowering `F32_CEILING` toward ~1e-4, which the observed
in-basket coarse value 4e-5 makes risky for false positives — predicate (2)
provides the tighter, false-positive-free catch instead.)

### Honesty statement

This Amendment is **honesty-restoring**, NOT a relaxation. The latent bug was the
`−1.80` slope assertion itself: f32 round-off makes order-2 physically
unobservable for this dense 2D/3D Strang heat problem, so the old gate was
asserting something that could only pass by accident of where the floor noise
happened to land. Option B replaces it with a floor-band assertion that is true
of correct f32 behaviour and fails loud on a genuine regression. The `−1.95` f64
bands and all sympy oracle gates are untouched.

### Wiring (FINALIZED)

- `crates/semiflow-core/tests/generic_float_strang.rs`: the two f32 gates renamed
  to `gf1_2d_f32_floor_band` / `gf2_3d_f32_floor_band`, slope assert replaced by
  `assert_f32_floor_band(label, &errs)` against `F32_CEILING`/`F32_FLOOR`. Dead
  `SLOPE_GATE_F32` removed; `SLOPE_GATE_F64` and the f64 gates UNCHANGED. File =
  498 lines (≤ 500), all fns ≤ 50 lines, `check-lints` PASS, `clippy -D warnings`
  = 0 errors.
- `crates/semiflow-core/tests/generic_float_strang_f32_regime.rs`: the `#[ignore]`d
  regime-map diagnostic, KEPT in-tree (zero CI cost) as the reproducible evidence
  for this Amendment and a tool for future precision/interpolant changes that move
  the floor. Its module doc updated to record the no-floor-safe-band finding.
- `contracts/semiflow-core.properties.yaml`: these two f32 spatial-slope gates are
  NOT pinned by name in the properties contract (confirmed via grep — only the
  f64 G3/G5/G3⁶ slope gates and the graph G11/G13 f32 gates are pinned). No
  machine-readable pin to update; §3.2 of this ADR remains the source of truth.

### Updated f32 disposition table (supersedes the `gf*_f32` rows of §3.2)

| Gate | §3.2 band | Amendment 1 final disposition |
|---|---|---|
| `gf1_2d_f32_floor_band` (Strang2D) | ~~≥ −1.80 on {64,128}~~ | **floor-band gate** — `self_err ∈ [1e-7, 5e-4]` ∀N AND finest < coarsest (slope DROPPED — round-off-limited, no asymptotic band) |
| `gf2_3d_f32_floor_band` (Strang3D) | ~~≥ −1.80 on {64,128}~~ | same floor-band gate (identical resolution) |
| `gf1_2d_f64_slope` / `gf2_3d_f64_slope` | ≥ −1.95 | UNCHANGED (PASS) |
