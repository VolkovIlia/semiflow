# ADR-0006 — Strang operator splitting `L = A + B` for v0.2.0 (supersedes formal-adjoint design)

**Status**: Accepted
**Date**: 2026-04-29 (revised — same-day supersession)
**Authors**: ai-solutions-architect (Stage 0+1 v2)
**Resolves risk**: R4 (technical-constraints.md)
**Supersedes**: ADR-0006 v1 (formal-adjoint `S^*(τ/2) ∘ S(τ/2)`, drafted earlier the same day)

## Status note (supersession)

The earlier same-day draft of this ADR proposed formal-adjoint symmetrization
`S_strang(τ) = S^*(τ/2) ∘ S(τ/2)` with `(ShiftChernoff1D)^* = (a, −b, c)`.
Stage 4 QA empirically measured for the heat-kernel oracle (`a=0.5, b=0, c=0`):
G1 at `n=100` = **1.6 × 10⁻⁴** (gate `< 1.0 × 10⁻⁴` → FAIL),
G2 at `n=1000` = **2.0 × 10⁻⁵** (gate `< 1.0 × 10⁻⁶` → FAIL by 20×),
G3-strang slope = **−0.96** (gate `≤ −1.95` → FAIL — first-order behaviour).
Diagnosis: for `b ≡ 0`, `ShiftChernoff1D` is self-adjoint, so
`S^*(τ/2) ∘ S(τ/2) = S(τ/2)²`, equivalent to running the base function with
`2n` iterations at half step — only a 2× constant reduction (matches the
empirical 1.6e-4 ≈ 3.2e-4 / 2), no order lift. This ADR replaces that approach
with operator splitting `L = A + B`, per the user's no-relaxation mandate.

## Context

`L f = a(x) f''(x) + b(x) f'(x) + c(x) f(x)` decomposes naturally into
diffusion `A f = a(x) f''(x)` and drift+reaction `B f = b(x) f'(x) + c(x) f(x)`.
The Galkin–Remizov 2025 "Chernoff product" framework permits composing
order-`p` Chernoff approximants of `e^{τA}` and `e^{τB}` to approximate
`e^{τL}`; the canonical Strang composition is
`Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2)` (palindromic, second-order if `D` is order ≥ 2
and `R` is exact, or order 2 generically when both are order ≥ 2 and
time-symmetric — Hairer–Lubich–Wanner, *Geometric Numerical Integration*,
§III.5, Thm 4.1).

For the v0.1.0 heat-kernel test (`a=0.5, b=0, c=0`), `B = 0`, so the splitting
collapses to the diffusion step alone — making the oracle insensitive to any
clever B-step design and forcing all error to come from the A-step
approximation. The test oracle is therefore amended (§5 below) to
**advection-diffusion** with `(α, β) = (0.5, 0.5)`, which has the same
closed-form structure as the v0.1.0 heat kernel under a Galilean change of
variables and exercises both `A` and `B` non-trivially.

## Decision

Adopt **option (δ)**: a custom **second-order time-symmetric Chernoff for
the diffusion operator** combined with an **exact closed-form characteristic
solver for drift+reaction**, composed via the Strang sandwich.

### Rationale (option choice)

Numerical-Fourier analysis (§9 in `contracts/semiflow-core.math.md`) shows
that the natural alternative — option (β), Strang sandwich of formula-(6)
restricted to `(a, 0, 0)` (a *first-order* Chernoff) with exact `R` — yields
**global O(τ)**, *not* O(τ²): the per-step error symbol
`-(a²τ²/12) ξ⁴ + O(τ³)` is purely O(τ²) locally, so over `n = T/τ` steps the
global error is O(τ). Strang lift to order 2 requires the inner methods to
be **order ≥ 2 and time-symmetric** (or both flows exact). Hence the
diffusion step must itself be order 2.

### API surface

Three new types in the public interface; **`ShiftChernoff1D` is preserved
unchanged** for v0.1.0 backward compatibility (formula (6) for the full
operator `L`):

```rust
/// Chernoff function for A = a(x)·∂²_x.
///
/// 5-point time-symmetric formula matching `e^{τA}` to O(τ³) per step:
/// (D(τ)f)(x) = (7/12) f(x)
///             + (3/16) [f(x + 2√(a(x)τ)) + f(x − 2√(a(x)τ))]
///             + (1/48) [f(x + 2√(3 a(x)τ)) + f(x − 2√(3 a(x)τ))]
pub struct DiffusionChernoff { a: fn(f64) -> f64 }

/// Chernoff function for B = b(x)·∂_x + c(x).
///
/// EXACT closed-form characteristics (no time discretization):
/// (R(τ)f)(x) = exp(τ·c(x)) · f(x + τ·b(x))
pub struct DriftReactionChernoff { b: fn(f64) -> f64, c: fn(f64) -> f64,
                                    c_norm_bound: f64 }

/// Strang composition: Φ(τ) = D(τ/2) ∘ R(τ) ∘ D(τ/2).
pub struct StrangSplit<D: ChernoffFunction, R: ChernoffFunction> {
    pub diffusion: D,
    pub drift_reaction: R,
}
```

`ChernoffFunction` trait surface is unchanged: no new methods, no adjoint.
`StrangSplit::apply(τ, &f)` calls `diffusion.apply(τ/2, .)`,
`drift_reaction.apply(τ, .)`, `diffusion.apply(τ/2, .)` in sequence,
threading state via three single-buffer `axpy` operations on `GridFn1D`.

### Convergence proof outline (delegated to Stage 5 formal-verifier)

For constant `(a, b, c)` (test-oracle regime) with `[A, B] = 0`:
`e^{τA}·e^{τB} = e^{τ(A+B)} = e^{τL}` exactly. `R` is exact for constant
`b, c`. `D` matches `e^{τA}` to O(τ³) per step (5-point Fourier symbol
`(7/12) + (3/8) cos(2√(aτ)ξ) + (1/24) cos(2√(3aτ)ξ) = e^{-aτξ²} + O(τ³)`,
verified algebraically below). Local error of `D(τ/2)·R(τ)·D(τ/2)` is
therefore O(τ³); over `n = T/τ` steps, global error is O(τ²). For variable
coefficients (v0.3+) the Strang palindromic structure cancels the leading
`τ³` commutator term so global O(τ²) persists; that result is canonical
(Hairer–Lubich–Wanner §III.5.4) and not algebraically re-proved here. Stage 5
formal-verifier MUST verify the 3-equation linear system for `D`'s Fourier
symbol coefficients (`w_0 = 7/12, w_1 = 3/16, w_2 = 1/48`) — see
`contracts/semiflow-core.math.md` §9.2.

### Test oracle change (G1, G2, G3-strang)

The v0.1.0 oracle (`a=0.5, b=c=0`, heat kernel) is replaced for v0.2.0
acceptance by **advection-diffusion**:

```
∂_t u = (1/2) ∂²_x u + (1/2) ∂_x u,    u(0,x) = e^{-x²}
```

Closed form (Galilean substitution `v(t,y) = u(t, y − βt)` reduces to pure
heat with `α = 1/2`):

$$
u(t,x) = (1 + 2t)^{-1/2} \exp\!\Big(-\frac{(x + t/2)^2}{1 + 2t}\Big)
$$

At `t=1`: `u(1,x) = (3)^{-1/2}·exp(-(x+0.5)²/3)`. Mass centre translates from
`0` to `−0.5` — `B`-step is non-trivial and the slope test exercises both
operators. v0.1.0 heat-kernel oracle (`b=0`) is preserved as `G1-legacy /
G2-legacy` for regression.

## Consequences

**New tests**: G1, G2 (advection-diffusion), G3-strang (slope ≤ −1.95 over
`n ∈ {32, 64, 128, 256, 512, 1024}`), G4-strang (proptest with bound
`(1 + |c|·τ + 20·τ²)·‖f‖∞`). Acceptance-criteria.md amendment 2026-04-29 v2.

**New properties**: `strang_split_second_order_advdiff` (numerical),
`strang_palindrome_invariance` (`Φ(τ)f` ≡ reverse-order recompose with same
state up to interp noise), `diffusion_chernoff_taylor_order_2`,
`drift_reaction_exact_characteristic`. Replace previously-deleted
`strang_*` properties.

**Suckless budget impact** (each new file ≤ 250 lines, each fn ≤ 50 lines):
`crates/semiflow-core/src/diffusion.rs` ≤ 200, `src/drift_reaction.rs` ≤ 200,
`src/strang.rs` ≤ 150. No new direct dependencies. Internal `axpy`-based
threading reuses existing `GridFn1D` buffers; no extra allocations per step.

**Backward compatibility (v0.1.0 ↔ v0.2.0)**: `ShiftChernoff1D`, `Grid1D`,
`GridFn1D`, `ChernoffSemigroup`, `State`, `ChernoffFunction` all unchanged
(no methods added, no signatures altered). v0.1.0 callers compile unchanged
on v0.2.0. New types are additive.

## Verification handoff

**Stage 4 (Agentic QA)** — implement six tests in
`crates/semiflow-core/tests/strang_advdiff.rs`:

1. G1 numeric (n=100, advdiff oracle, sup-norm < 1e-4).
2. G2 numeric (n=1000, advdiff oracle, sup-norm < 1e-6).
3. G3-strang slope (n ∈ {32, 64, 128, 256, 512, 1024}, slope ≤ −1.95).
4. G4-strang proptest (10 000 cases, `(1 + |c|·τ + 20·τ²)·‖f‖∞`).
5. G1-legacy / G2-legacy regression with `ShiftChernoff1D` (unchanged).
6. Strang palindrome property: `StrangSplit<D, R>::apply(τ, f)` agrees with
   `StrangSplit<D, R>::apply(τ, .)` post-composed with itself at `τ/2`
   (consistency self-check).

**Stage 5 (formal-verifier)** — algebraic checks in
`.dev-docs/verification/strang-splitting-derivation.md`:

1. Verify `D`'s 3-equation linear system (Taylor coefficients): solve
   `w_0 + 2w_1 + 2w_2 = 1`, `4 w_1 + 12 w_2 = 1`, `16 w_1 + 144 w_2 = 6`,
   recover `(7/12, 3/16, 1/48)`. All weights positive (probability
   interpretation valid; spectral radius `cos²(z)·... ≤ 1`).
2. Verify `D`'s Fourier symbol matches `e^{-aτξ²}` to O(τ³).
3. Verify `R(τ)f(x) = e^{τc} f(x + τb)` is the exact propagator of
   `B = b∂_x + c` for constant coefficients (linear ODE along
   characteristics).
4. Verify the per-step Fourier-symbol product
   `cos²-weighted-D(τ/2) · e^{τ(c+iβξ)} · same-D(τ/2) = e^{τ(-aξ² + iβξ + c)} + O(τ³)`
   (commutativity for constants makes this an algebraic identity rather
   than a BCH expansion).
5. Verify quasi-contractivity: `‖D(τ)‖∞ ≤ 1` (positive weights summing to 1
   ⇒ positivity-preserving) and `‖R(τ)‖∞ ≤ e^{|c|τ}` ⇒
   `‖StrangSplit(τ)‖∞ ≤ 1·e^{|c|τ}·1 ≤ 1 + |c|τ + (|c|τ)²·e^{|c|τ}/2 ≤
   1 + |c|τ + 20τ²` for `|c| ≤ 5`, `τ ≤ 0.1` (G4-strang constant `C = 20`).

**Stage 3 (Engineer, follow-up)** — implement against contracts in this
ADR's API-surface block; no API invention.

---

## Amendment 3 (2026-04-29, same-day) — Production grid `N = 8000` (not `N = 1000`)

**Trigger**: Stage 7 QA measured v0.2.0 G2 = 1.385×10⁻⁵ (FAIL by 14×) and
G3-strang slope = +0.53 (FAIL — *positive*) on the post-sign-fix
`StrangSplit` implementation. Empirical error curve at `N = 1000` is
U-shaped (minimum at `n = 64`), the canonical signature of a spatial
discretization floor that the v0.1.0 heat-kernel baseline never exposed
(because `b ≡ 0` ⇒ zero advection ⇒ no rising spatial error to dominate).

**Diagnosis** (architect, this amendment):

1. **Algebra of the 5-point diffusion Chernoff** independently re-verified
   in `contracts/semiflow-core.math.md §9.2`. Weights `(7/12, 3/16, 1/48)`
   are correct. Leading symbol error is `+(aτ)³/30 · ξ⁶`, giving local
   `O(τ³)` and global `O(τ²)` per §9.4. **No code-side or formula-side
   bug.**

2. **Constant-coefficient regime clarification** added in §9.7. For the
   test oracle `[A, B] = 0`, so `Φ(τ) = D(τ/2)² · e^{τB}` and
   `e^{τL} = e^{τA} · e^{τB}`. The Strang test exercises `D`'s local
   `O(τ³)` directly, not the BCH-commutator cancellation (which is
   trivially zero here). The test still meaningfully covers production
   pieces (D-weights, state-threading, interpolation kernel).
   Variable-coefficient testing is deferred to v0.3+ (R6).

3. **Spatial-error budget** (§9.7 in math.md): empirical floor at
   `N = 1000` calibrates to `K_s ≈ 86.5 · Δx⁴` with mild `n^0.43` growth.
   At `N = 1000`: floor ≈ 1.4×10⁻⁵ (dominates G2 and the G3-strang slope
   tail). At `N = 8000`: floor ≈ 3.7×10⁻⁹ — well below the smallest
   measured time error in the slope window (8.9×10⁻⁹ at `n = 1024`).
   Predicted slope at `N = 8000` is **−1.99** (margin 0.04 below the
   `−1.95` gate). G1 prediction at `N = 8000`: 9.3×10⁻⁷ (107× under gate);
   G2 prediction: 9.5×10⁻⁹ (105× under gate).

**Resolution**: amend G1, G2, G3-strang grid resolution from `N = 1000`
to `N = 8000`. This is a **harness** parameter refinement, **not** a
gate relaxation:

- All gate ε's unchanged: `1×10⁻⁴`, `1×10⁻⁶`, slope `≤ −1.95`,
  proptest constant `20`.
- The original `N = 1000` was a copy-paste from the v0.1.0 heat-kernel
  configuration where spatial discretization happened to suffice (no
  advection, no order-2 time integration). The v0.2.0 advection-diffusion
  oracle + Strang-order-2 time integration mathematically demand a
  finer spatial grid for the same accuracy budget — this is the cost
  of the higher-order time discretization, not a relaxation of any
  acceptance threshold.

**Compute / memory cost**: `N = 8000` increases per-Strang-step grid
work by 8× over `N = 1000`. At `n = 1024 × N = 8000 × ~9 interps/step`
the slope-test point measures ~7.4×10⁷ cell-ops ≈ tens of milliseconds;
state-buffer memory ~192 KB. Within CI budgets.

**Files modified by this amendment** (architect Stage 0):

- `contracts/semiflow-core.math.md` — §6.2 (`N: 1000 → 8000`, grid-resolution
  note added), §9.2 (leading symbol error coefficient corrected from
  `−a³τ³/6` to `+(aτ)³/30`, full derivation expanded), §9.7 added
  (spatial-error budget + constant-coefficient clarification).
- `.dev-docs/product/acceptance-criteria.md` — G1, G2, G3-strang grid
  spec updated to `N = 8000`; amendment-3 entry added.
- `docs/adr/0006-strang-symmetrization.md` — this amendment.

**Files NOT modified** (per architect scope):

- `crates/semiflow-core/src/*.rs` — no source change required (algebra
  is correct).
- `crates/semiflow-core/tests/strang_advdiff.rs` — Stage 4 QA must update
  test `Grid1D::new(N=8000, ...)` calls in G1, G2, G3-strang harnesses
  (3 sites). G1-legacy / G2-legacy / G4 / G4-strang unchanged.

**Open issues / risks**:

- *Slope margin*: predicted `−1.99` leaves only 0.04 below the `−1.95`
  gate. If reality deviates by more than 0.04 (e.g., due to BC reflection
  artifacts at very fine grids, or accumulated round-off effects not
  captured in the empirical model), QA may need to raise to `N = 10000`
  in a v0.2.1 amendment. Predicted slope at `N = 10000` is `−2.00`
  (model margin 0.05).
- *G6 performance gate*: unchanged because G6 still measures the v0.1.0
  heat-kernel configuration (`n = 100, N = 1000`), not the v0.2.0
  advection-diffusion harness. A separate v0.2.0 performance gate (G6-strang)
  could be added if needed; deferred.

**Re-delegation**: Stage 4 QA re-executes after applying `N = 8000` in
the test harnesses. Expected: G1 ~9.3×10⁻⁷, G2 ~9.5×10⁻⁹, G3-strang
slope ~−1.99.

---

## Amendment 4 (2026-04-29, same-day) — Empirical floor recalibration: `N = 8000 → 100000`

**Trigger**: Stage 4 QA re-executed v0.2.0 G3-strang at `N = 8000`
(per Amendment-3) and measured slope `−1.07` (FAIL — gate `≤ −1.95`).
G1 = 2.69×10⁻⁷ and G2 = 1.09×10⁻⁷ both passed (G2 with only ~9×
margin under 1e-6, smaller than the 105× predicted in Amendment-3).
The G3-strang error table at `N = 8000`:

| n    | empirical err |
|------|---------------|
| 32   | 2.65×10⁻⁶     |
| 64   | 6.58×10⁻⁷     |
| 128  | 1.65×10⁻⁷     |
| 256  | 5.11×10⁻⁸     |
| 512  | 3.36×10⁻⁸     |
| 1024 | 1.12×10⁻⁷ (rebound — 3.3× above n=512) |

The slope on the first 4 points (`n ∈ {32, 64, 128, 256}`) is `−1.91`
— clean order-2 in the time-dominated regime. The implementation IS
achieving order-2 accuracy where the time error dominates. The
rebound between `n = 512` and `n = 1024` is the failure mode.

**Diagnosis** (architect, Path B → Path A):

1. **Not an implementation bug**. The 5-point diffusion algebra
   (weights `(7/12, 3/16, 1/48)`) was independently verified
   (math.md §9.2). The drift sign `f.sample(x + τ·b)` is correct
   per math.md §9.3 (verified earlier in this same-day cycle).

2. **Amendment-3's `Δx⁴` floor model was wrong**. Cross-validating
   the floor at `n = 1024` between the two anchor measurements
   (`N = 1000` from pre-Amendment-3 QA gives `1.38×10⁻⁵`; `N = 8000`
   from post-Amendment-3 QA gives `1.12×10⁻⁷`):

   - Floor ratio: `124`
   - Grid ratio: `8`
   - If `floor ∝ Δx^p`: `8^p = 124` ⇒ `p = log(124)/log(8) ≈ 2.32`
   - If `floor ∝ Δx⁴` (Amendment-3): `8^4 = 4096`, predicting floor
     ratio of 4096 — empirically off by factor 33×.

   Amendment-3 used the canonical 4th-order accuracy of cubic Hermite
   for the asymptotic Δx-scaling; in the regime tested, the empirical
   exponent is significantly smaller. This is the root cause of
   Amendment-3's slope-prediction error.

3. **The floor grows linearly in `n` at fixed `N`**. At `N = 8000`,
   the floor at `n = 512` is `3.4×10⁻⁸` and at `n = 1024` is
   `1.1×10⁻⁷` — ratio `≈ 3.3`, very close to `2 × 1.65` (i.e.,
   doubling `n` roughly doubles the floor). This linear `n`-growth
   is *per-step interpolation accumulation* in the palindromic
   Strang sandwich when the drift shift `τβ` is sub-grid. At
   `N = 8000` the drift-shift-to-grid-spacing ratio crosses 1 at
   `n* = T·β/Δx = 200`; for `n > 200`, each Strang step performs
   sub-grid drift resampling whose Catmull-Rom Hermite kernel error
   does not fully cancel between successive steps. The error
   accumulates linearly in `n`. The diffusion shifts
   `h = 2√(α·τ)` and `H = 2√(3α·τ)` remain super-grid throughout
   the slope window (`h/Δx ≥ 17.7` at `N = 8000, n = 1024`), so the
   diffusion half-steps are not the source of the floor.

4. **Refined empirical model**:

   ```
   err(n, N) ≈ √[(C_t / n²)² + (K · n · Δx^p)²]
   ```

   with `C_t ≈ 2.7×10⁻³`, `K ≈ 1.17×10⁻⁴`, `p ≈ 2.32`. This model
   fits all 6 points at `N = 8000` within ratio 0.59-1.02 (the
   `n = 512` point is conservatively over-predicted, biasing the
   model's predicted slope to be *less* negative than the actual
   empirical slope — i.e., real measurements should always be a
   little better than the model).

**Resolution**: `N : 8000 → 100000`. Predicted at `N = 100000`:

- floor at `n = 1024`: `3.2×10⁻¹⁰` (~8× below time error
  `2.6×10⁻⁹`)
- G3-strang slope: `−1.998` (margin `0.048` below `−1.95`)
- G1 (`n = 100`): `2.7×10⁻⁷` (gate `1×10⁻⁴`, margin 370×)
- G2 (`n = 1000`): `2.7×10⁻⁹` (gate `1×10⁻⁶`, margin 370×)

**Sensitivity to `p`** (confidence in the recommendation):

| `p`  | predicted slope at `N = 100000` |
|------|---------------------------------|
| 2.0  | `−1.989` (PASS, margin `0.039`) |
| 2.32 | `−1.998` (PASS, best fit)       |
| 4.0  | `−2.000` (PASS, theoretical)    |

`p ≥ 2.0` is the defensible regime (cubic Hermite is theoretically
≥ 2nd-order accurate at any sub-grid shift); `N = 100000` clears the
gate over the entire defensible range. Adversarial `p = 1.5` (which
would imply implausibly poor interp-kernel behavior) is the only case
that would require a still-finer grid; that case is dismissed as
unphysical.

**Why this is not a gate relaxation**: The acceptance gates `1×10⁻⁴`,
`1×10⁻⁶`, slope `≤ −1.95`, proptest constant `20` are **unchanged**.
Only the harness parameter `N` is updated, calibrated against
empirical data rather than estimated under the prior incorrect
`Δx⁴`-scaling assumption. The corrected `N = 100000` is the smallest
engineering-friendly round value that defends the slope gate under
all defensible `p`-sensitivities. The empirical model's `C_t` and `K`
are calibrated jointly across two anchors with different `N`, so
prediction at `N = 100000` is forward extrapolation of measured
physics rather than hopeful estimation.

**Compute / memory cost**: `N = 100000` is 12.5× over Amendment-3's
`N = 8000`. Per-Strang-step grid work scales linearly with `N`;
G3-strang slope sweep totals 32+64+128+256+512+1024 = 2016 Strang
steps, each performing ~9 sub-grid evaluations. Estimated wall-clock
on the CI baseline (release mode, single-thread, x86_64): ~50 s for
the full G1+G2+G3 sweep (vs. ~4 s at `N = 8000`). Memory: three state
buffers × `N = 100000` × 8 B ≈ 2.4 MB. Both within CI budgets — well
under any reasonable `cargo test --release` deadline.

The G6 performance gate remains unaffected because G6 benchmarks the
v0.1.0 heat-kernel configuration (`n = 100, N = 1000`), not the
v0.2.0 advection-diffusion harness.

**Files modified by this amendment** (architect Stage 0):

- `contracts/semiflow-core.math.md` — §6.2 (`N: 8000 → 100000`,
  grid-resolution note rewritten with Amendment-4 rationale), §9.7
  (model recalibrated; floor exponent `p` revised from `≈4` to
  `≈2.32`; calibration constants recomputed with two anchors;
  sensitivity table added).
- `.dev-docs/product/acceptance-criteria.md` — G1, G2, G3-strang grid
  spec updated to `N = 100000`; amendment-4 entry added.
- `docs/adr/0006-strang-symmetrization.md` — this amendment.

**Files NOT modified** (per architect scope):

- `crates/semiflow-core/src/*.rs` — no source change required (algebra
  is correct; sign is correct).
- `crates/semiflow-core/tests/strang_advdiff.rs` and
  `tests/convergence_rate_strang.rs` — Stage 4 QA must update
  `Grid1D::new(N=100000, ...)` calls in G1, G2, G3-strang harnesses
  (3-5 sites). G1-legacy / G2-legacy / G4 / G4-strang unchanged.

**Risks remaining**:

- *Pessimistic-`p` scenario*: if the asymptotic floor exponent is
  `p < 2.0` (which would require sub-canonical accuracy of the
  Catmull-Rom Hermite kernel for sub-grid drift shifts — physically
  implausible), `N = 100000` may not suffice. The fallback would be
  `N = 200000` (model predicts slope `−1.999` at `p = 2.0`, `−1.989`
  at `p = 1.5`). Cost would be ~100 s wall-clock, ~5 MB memory —
  still acceptable.
- *Test runtime*: 50 s for `cargo test --release` on the slope sweep
  is significant. If this proves intractable for CI (unlikely on a
  modern x86_64 baseline), an alternative is to split G3-strang into
  a release-only-nightly job (similar to G5). This is a v0.2.x
  contingency, not a v0.2.0 blocker.

**Re-delegation**: Stage 4 QA re-executes after applying `N = 100000`
in the test harnesses. Expected: G1 ~2.7×10⁻⁷, G2 ~2.7×10⁻⁹,
G3-strang slope ~−1.998 (likely more negative due to model's ~0.06
conservative bias observed at `N = 8000`).

---

## Amendment 6.1 (v0.2.3, REVISED) — variable-`a(x)` Chernoff-correct lift via single-pass symmetric mean (formula β)

**Status (2026-04-30, end-of-day)**: **SUPERSEDED by ADR-0008** (option γ,
v0.3.0 API break). The formula β below is **NOT shipped**; the project
skips the planned v0.2.3 micro-bump and goes directly to v0.3.0 with
the option-γ inner-Strang divergence-form Chernoff. **Amendment 6.1 is
retained in full** because:

1. The **symmetric-stencil impossibility theorem** (§6 of the
   `.dev-docs/verification/variable-diffusion-beta-derivation.md`
   reference document) is a mathematical result that remains canonical
   — it explains why ADR-0008's γ-A also achieves only local-O(τ²) /
   global-O(τ) for variable `a` (the symmetric K-factor inside γ-A's
   inner-Strang inherits the same obstruction).
2. The **Liouville-transform manufactured oracle** `a(x) = (1+γx)²`
   (§9 of the β-derivation document) is preserved verbatim as the QA
   gate for v0.3.0's `diffusion_chernoff_variable_gamma_liouville_oracle`
   (renamed from `*_order1_liouville_oracle`).
3. The **constant-`a` reduction lemma** (§4) applies verbatim to γ-A
   (and is sympy-proven for γ-A by the inner-Strang's `S(0) = id`
   reduction when `a' ≡ 0`).

**Original (now-superseded) status block**:
SUPERSEDES Amendment 6 (formula α below — REJECTED by Stage-5
formal verifier on 2026-04-30 for τ¹ generator mismatch generating the
wrong PDE; full report at
`.dev-docs/verification/variable-diffusion-midpoint-derivation.md`).

**Re-design rationale** (architect Stage-5 → Stage-1 round-trip): formula
α's per-leg midpoint `a`-evaluation produced asymmetric shifts
$\widetilde h^+ \ne \widetilde h^-$, biasing the symmetric Taylor pair
$f(x + \widetilde h^+) + f(x - \widetilde h^-)$ at order $\tau^1$ by a
parasitic $\tfrac{1}{2}\,a'(x)\,f'(x)$ term. The iterated approximant
converged to $e^{T\mathcal{M}}f$ with
$\mathcal{M} = a\,\partial^2 + \tfrac{1}{2} a'\,\partial$ — the wrong PDE.

**Chosen replacement — formula β (single-pass symmetric arithmetic mean)**:
each shift pair shares ONE lifted shift derived from
$\bar a_\bullet = (a(x + h_0/2) + a(x - h_0/2))/2$ for the near pair
(and similarly with $H_0$ for the far pair). Stencil symmetry
$h^+ = h^- = h$ is restored, so $D_\beta'(0)\,f = a(x)\,f''(x) = A f$
exactly (sympy-verified, see
`.dev-docs/verification/variable-diffusion-beta-derivation.md` §3.2).
The wrong-PDE bug is eliminated.

**Impossibility result**: formula β has a τ² structural deficit
$\frac{1}{4} a a'' f'' - a\,a'\,f'''$ that **cannot** be removed by any
symmetric stencil over $(\pm h, \pm H)$ shifts. The $a a' f'''$ term
contains an **odd derivative of $f$**, which is unreachable by any
symmetric pair $f(x \pm s) + f(x \mp s) = 2f + s^2 f'' + s^4/12 f^{(4)} + \cdots$
(only even derivatives appear). Six recipe variants tested
(α, β, β'-Picard, trap3, Simpson, central-only) all retain a non-zero
$a a' f'''$ coefficient. Conclusion: the v0.2.0 stencil topology
**cannot deliver** local $O(\tau^3)$ for variable `a`. Full proof:
`.dev-docs/verification/variable-diffusion-beta-derivation.md` §6.

**Revised order claim**:
| Regime | Local order | Global order | Generator |
|--------|-------------|--------------|-----------|
| Constant `a` | $O(\tau^3)$ exact | $O(\tau^2)$ via Strang | $A$ ✓ |
| Variable `a ∈ C^2$ | $O(\tau^2)$ | $O(\tau)$ | $A$ ✓ (correct, vs α's $\mathcal M$) |

The contract-claimed "global $O(\tau^2)$ for variable `a`" is **revised
down** to global $O(\tau)$. v0.2.3 v.s. v0.2.2: same convergence rate
but **correct generator** (v0.2.2 silently degraded for variable `a`;
α actively converged to wrong PDE; β converges to right PDE at
v0.2.2's rate). The order-2 lift for variable `a` is **deferred to
v0.5+** behind an explicit `a'(x)` API (option γ) or Magnus integrator
(option ε).

**Pointwise formula β (engineer copies into `apply_at_node`)**:

```
let x      = dc.grid.x_at(i);
let a0     = (dc.a)(x);                               // central-a base
validate_a_x(a0, x)?;
let h0     = 2 * sqrt(a0 * τ);                         // base scale (near)
let H0     = 2 * sqrt(3 * a0 * τ);                     // base scale (far)

// Single-pass symmetric arithmetic mean per leg-pair:
let a_pos_near = (dc.a)(x + h0/2);                    // 4 evaluations total
let a_neg_near = (dc.a)(x - h0/2);                    // (vs 8 in α)
let a_pos_far  = (dc.a)(x + H0/2);
let a_neg_far  = (dc.a)(x - H0/2);
let abar_near  = 0.5 * (a_pos_near + a_neg_near);
let abar_far   = 0.5 * (a_pos_far  + a_neg_far);
validate_a_x(abar_near, x)?;                           // averaged a inherits ellipticity
validate_a_x(abar_far,  x)?;

// Symmetric lifted shifts (h^+ = h^- = h, H^+ = H^- = H):
let h      = 2 * sqrt(abar_near * τ);
let H      = 2 * sqrt(3 * abar_far  * τ);

result[i] = (7/12)·f[i]
          + (3/16)·(f.sample(x + h) + f.sample(x - h))   // SYMMETRIC shifts
          + (1/48)·(f.sample(x + H) + f.sample(x - H));
```

**QA gate replacement (per Stage-5 verifier mandate)**: the property
`diffusion_chernoff_variable_order2` (Richardson self-consistency,
slope ≤ -1.95) is replaced by `diffusion_chernoff_variable_order1_liouville_oracle`
using the manufactured-solution profile $a(x) = (1 + \gamma x)^2$. This
profile admits the Liouville transform $y = \ln(1 + \gamma x)/\gamma$,
$v = a^{-1/4} u$, mapping $u_t = a u_{xx}$ to $v_t = v_{yy} - (\gamma^2/4) v$
— a **constant-potential** Schrödinger-like equation with closed-form
heat-kernel solution (sympy-verified, `verify_liouville_oracle.py`).
This provides the absolute closed-form oracle that the verifier
mandated as a replacement for self-consistency (which "tests order, not
correctness").

The Liouville-oracle gate asserts slope $\le -0.95$ (β's theoretical
global $O(\tau)$). A future v0.5+ implementation passing the same gate
with slope $\le -1.95$ would empirically demonstrate the order-2 lift,
auto-detected.

**Files modified for Amendment 6.1**:

- `contracts/semiflow-core.math.md` §9.2.2 — REWRITTEN (β formula,
  τ¹ Chernoff proof, τ² impossibility, Liouville oracle).
- `contracts/semiflow-core.traits.yaml` `DiffusionChernoff.apply.semantics`
  — REWRITTEN (β pseudocode, single-pass symmetric mean).
- `contracts/semiflow-core.properties.yaml` — REPLACE
  `diffusion_chernoff_variable_order2` with
  `diffusion_chernoff_variable_order1_liouville_oracle`. KEEP
  `diffusion_chernoff_constant_fast_path_exact` (β passes by §4 reduction).
- `.dev-docs/verification/variable-diffusion-beta-derivation.md` (NEW)
  — Stage-1 architect re-derivation with sympy proof of β.
- `.dev-docs/verification/scripts/verify_v0_2_3_beta.py`,
  `.../verify_v0_2_3_variants.py`, `.../verify_liouville_oracle.py`
  — sympy reproduction (build-time only).
- `docs/adr/0006-strang-symmetrization.md` — this Amendment 6.1.
- `contracts/semiflow-core.errors.yaml` — UNCHANGED (β has same error
  signature as α: 5 `validate_a_x` checks per node, all returning
  `DomainViolation` on failure).

**Performance**: 4 `a(·)` evaluations per node (vs. 8 in α, vs. 1 in
v0.2.2). Predicted overhead: ~3-5% per step vs. v0.2.2 — within ±5%
target. Half the cost of the rejected α formula.

**Public API**: unchanged. Same `DiffusionChernoff::new(a, a_norm_bound, grid)`
constructor. Engineer rewrites only `apply_at_node` body.

**Forward-compatibility for v0.3.0 (2D tensor product)**: tensor 2D
Strang inherits β transparently (each axis applies β along its own
direction). Global order: $O(\tau^2)$ for constant `a`, $O(\tau)$ for
variable `a` — same as 1D, same caveat.

**Why the order-1 claim is acceptable for v0.2.3 milestone**: the
production oracle (G1, G2, G3-strang, G4-strang) uses constant
`α = 0.5`, which the β formula handles **bit-equal** to v0.2.2 by §4
of the verification document. The variable-`a` lift to global
$O(\tau^2)$ was a *speculative* enhancement; with the Stage-5
impossibility result it's deferred. The architecturally honest position
is to ship a **correct-generator** scheme at v0.2.2's convergence rate
and revisit the order lift in v0.5+ with appropriate API change.

---

## Amendment 6 (v0.2.3, REJECTED 2026-04-30) — variable-`a(x)` order-2 lift via per-leg midpoint-`a` (formula α)

**Status (2026-04-30, end-of-day)**: **REJECTED by Stage-5 formal
verifier** (original disposition, retained verbatim) **AND superseded
by ADR-0008** (the v0.3.0 API break makes this entire amendment moot —
v0.3.0 ships γ-A inner-Strang divergence-form, not α and not β). **DO
NOT IMPLEMENT.** Retained for historical record and to document the
architectural lesson (per-leg midpoint asymmetry breaks Chernoff
consistency for variable coefficients — α was wrong, β was correct
but order-limited, γ-A is mathematically rigorous AND order-limited
by the same impossibility theorem).

**Trigger**: v0.2.0 GAP 1 (symmetric counterpart of Amendment 5).
v0.2.2 lifted `DriftReactionChernoff::apply` from local O(τ²) / global
O(τ) to local O(τ³) / global O(τ²) for variable `b ∈ C²`, `c ∈ C¹`
(midpoint + trapezoidal RK2). The `DiffusionChernoff::apply` 5-point
formula (§9.2) currently evaluates `a` only at the *central* node `x_i`
(see `crates/semiflow-core/src/diffusion.rs:162` — `let a_x = (dc.a)(x);`),
giving local O(τ²) / global O(τ) for variable `a(x)`. After Amendment 5
the drift-reaction half is order-2 in variable coefficients but the
diffusion half is not, so the Strang sandwich is bottlenecked at global
O(τ) for variable `a` even though variable `b, c` are now order-2.
v0.2.3 closes that gap symmetrically.

**Chosen formula — option (α), midpoint-evaluated `a` per shift segment**.
Three candidates were enumerated (`/home/volk/.claude/plans/wondrous-marinating-turing.md`,
user-approved):

- **(α)** Midpoint-evaluated `a`: each of the four shift legs evaluates
  `a` at the midpoint of *its own* segment. Local O(τ³) for `a ∈ C²`,
  bit-equal-modulo-IEEE for constant `a`, weights `(7/12, 3/16, 1/48)`
  unchanged.
- **(β)** Symmetric average `ā(x) = (a(x + h/2) + a(x − h/2))/2` via
  Picard iteration. Same order, but requires fixed-point iteration on
  the shift scale, breaking the constant-`a` bit-equality (Picard's
  initial guess vs. converged value differ by O(τ²) in IEEE arithmetic
  even for constants) and pushing `apply_at_node` over its 50-line cap.
  Rejected.
- **(γ)** Self-adjoint decomposition using `a'(x)`. Requires the public
  API to accept the derivative as a separate `fn` pointer, breaking the
  v0.2.0 single-`a`-pointer contract. Rejected per plan.

**Decision: (α)**. Algebraic justification:

1. **Structural mirror of Amendment 5**: midpoint quadrature for the
   characteristic length-scale `2·√(a·τ)` is the natural diffusion
   counterpart of midpoint quadrature for the drift foot-point
   `τ·b(x + (τ/2)·b(x))`. The Stage-5 verification toolchain re-uses the
   same Taylor-expansion-to-O(τ³) machinery (math.md §9.2 extension).
2. **Reduction-to-constant lemma**: when `a(x) ≡ a₀`, every midpoint
   evaluation returns `a₀`, so `h_pos_near = h_neg_near = 2·√(a₀·τ)`
   and `H_pos_far = H_neg_far = 2·√(3·a₀·τ)` — *bit-equal* to v0.2.2
   (the IEEE arithmetic path is structurally identical, no new ops).
   v0.2.0/v0.2.1/v0.2.2 acceptance gates G1, G2, G3-strang, G4-strang
   (all use constant `α = 0.5`) are regression-safe by construction.
3. **Weights `(7/12, 3/16, 1/48)` unchanged**: the lift adds *shift-scale*
   locality (one extra `a(·)` per shift leg) without disturbing the
   Fourier-symbol weight algebra. The 3-equation linear system in §9.2
   is still solved by the same triple. Quasi-contractivity preserved
   (positive weights summing to 1 ⇒ `‖D(τ)‖_∞ ≤ 1`).
4. **File / function budgets**: `apply_at_node` grows from ~25 to ~38
   lines (under the 50-line cap); `diffusion.rs` grows from 178 to ~198
   lines (under the 200-line cap from §142 above).

**Pointwise formula** (engineer copies into `apply_at_node` verbatim):

```
let x   = dc.grid.x_at(i);
let a0  = (dc.a)(x);                                  // central a (used for shift scales)
validate_a_x(a0, x)?;
let h   = 2·sqrt(a0·τ);                                // near-wing scale (set by central a)
let H   = 2·sqrt(3·a0·τ);                              // far-wing scale  (set by central a)

// Midpoint a-evaluations per shift leg (variable-a lift):
let a_pos_near = (dc.a)(x + h/2);                      validate_a_x(a_pos_near, x + h/2)?;
let a_neg_near = (dc.a)(x − h/2);                      validate_a_x(a_neg_near, x − h/2)?;
let a_pos_far  = (dc.a)(x + H/2);                      validate_a_x(a_pos_far,  x + H/2)?;
let a_neg_far  = (dc.a)(x − H/2);                      validate_a_x(a_neg_far,  x − H/2)?;

// Per-leg shift scales — each uses a evaluated at its own segment midpoint:
let h_pos_near = 2·sqrt(a_pos_near·τ);
let h_neg_near = 2·sqrt(a_neg_near·τ);
let H_pos_far  = 2·sqrt(3·a_pos_far·τ);
let H_neg_far  = 2·sqrt(3·a_neg_far·τ);

result[i] = (7/12)·f[i]
          + (3/16)·(f.sample(x + h_pos_near) + f.sample(x − h_neg_near))
          + (1/48)·(f.sample(x + H_pos_far)  + f.sample(x − H_neg_far));
```

**Order claim**: local O(τ³) for `a ∈ C²(ℝ)` (Taylor expansion in §9.2.2
of math.md). Global O(τ²) by Chernoff product theorem. Constant-`a`
reduces to v0.2.2 closed form bit-equal modulo IEEE rearrangement
(≤ 4 ULPs envelope; tolerance `1e-13` in property
`diffusion_chernoff_constant_fast_path_exact`).

**Performance target**: ±5% per-step latency vs v0.2.2. Cost analysis:
v0.2.2 makes 1 `a(·)` call + 4 `f.sample(·)` calls per node. v0.2.3
makes 5 `a(·)` calls (1 central + 4 midpoint) + 4 `f.sample(·)` calls
per node. Extra: 4 `a(·)` calls + 4 `validate_a_x` checks ≈ 8-12 ns
per node on x86_64 (vs. ~150 ns/node baseline dominated by
`f.sample`). Predicted overhead: ~5-8% per step. Within the ±5% target
when the user-supplied `a` closure is trivial; for complex `a` the
relative cost shrinks because `f.sample` (cubic Hermite) still
dominates. G6 budget (10 ms at `n=100, N=1000`) unaffected (G6
benchmarks v0.1.0 `ShiftChernoff1D` heat-kernel, not `DiffusionChernoff`).

**Public API unchanged**: same struct fields (`a, a_norm_bound, grid`),
same constructor `DiffusionChernoff::new(a: fn(f64) -> f64, a_norm_bound: f64, grid: Grid1D)`,
same `ChernoffFunction::apply / order / growth` signatures. Only the
internal `apply_at_node` body is rewritten. v0.2.0/v0.2.1/v0.2.2
callers compile unchanged on v0.2.3.

**Forward-compatibility for v0.3.0 (2D tensor product)**: the tensor
2D Strang `Φ²ᴰ = D_x(τ/2)·D_y(τ/2)·R_x(τ)·R_y(τ)·D_y(τ/2)·D_x(τ/2)`
inherits the variable-`a` lift transparently when `D_x` and `D_y` each
apply the v0.2.3 midpoint-`a` 5-point formula along their respective
axes. With variable `a(x, y)` the two axis-aligned diffusion steps
remain order-2 individually, and Strang preserves global O(τ²) by
the same palindromic-trapezoidal-symmetry argument as Amendment 5
§7. v0.2.3 is the prerequisite for the v0.3.0 2D extension on the
diffusion side, just as Amendment 5 was on the drift-reaction side.

**Files modified** (architect Stage 0):

- `contracts/semiflow-core.math.md` — §9.2 restructured into §9.2.1
  (constant-`a`, unchanged) and §9.2.2 (variable-`a` lift, NEW).
  Reduction lemma + Strang cross-reference added.
- `contracts/semiflow-core.traits.yaml` — `schema_version: 0.2.2 → 0.2.3`.
  `DiffusionChernoff` block: `Invariants` paragraph (I1 totality,
  I2 variable-coefficient order-2, I3 constant-coefficient
  regression-safety) added; `apply` `semantics` block rewritten with
  pointwise formula above; `order` semantics paragraph clarified
  for variable-`a`.
- `contracts/semiflow-core.properties.yaml` — `schema_version: 0.2.2 → 0.2.3`.
  Two new properties:
  `diffusion_chernoff_constant_fast_path_exact` (1000 cases,
  tolerance 1e-13 against v0.2.2 closed form) and
  `diffusion_chernoff_variable_order2` (200 cases, log-log slope ≤ -1.95
  against the smooth-bounded oracle `a(x) = a₀ + α₀·sech²(γx)` using a
  Richardson self-consistency reference at `n_ref = 8192`).
- `.dev-docs/verification/variable-diffusion-midpoint-derivation.md`
  (NEW) — symbolic Taylor expansion proving local O(τ³) and exact
  reduction at constant-`a`. Stage 5 formal-verifier handoff.
- `docs/adr/0006-strang-symmetrization.md` — this Amendment 6.

**Files NOT modified** (per architect scope):

- `contracts/semiflow-core.errors.yaml` — no new error variants. The
  midpoint-`a` lift adds 4 extra `a(·)` evaluations and 4 extra
  `validate_a_x` checks per node; the existing `DomainViolation`
  variant covers the `a(x_mid) <= 0` failure mode (no new variant
  needed). The emission_table entry for `DiffusionChernoff::apply`
  remains `[DomainViolation, Unsupported]`.
- `crates/semiflow-core/src/diffusion.rs` — Engineer (Stage 6)
  implements the rewritten `apply_at_node` per the formula above.
  All other functions (`apply`, `order`, `growth`, `validate_tau`,
  `validate_a_x`, constants `W0/W1/W2`) unchanged.
- `crates/semiflow-core/tests/strang_advdiff.rs` — G1, G2, G3-strang
  harnesses use constant `a = 0.5`, hence bit-equal to v0.2.2 by the
  reduction lemma. No test changes required for v0.2.3.
- `crates/semiflow-core/src/strang.rs`, `drift_reaction.rs`, `grid.rs`,
  `grid_fn.rs`, `state.rs`, `chernoff.rs`, `shift1d.rs`, `error.rs`,
  `lib.rs` — unchanged.

**Variable-`a` oracle (engineer-relevant)**: `a(x) := a₀ + α₀·sech²(γx)`
with `a₀ > 0`, `α₀ ≥ 0`, `γ > 0`. This profile is `C^∞`, bounded with
bounded derivatives of all orders, strictly elliptic (`a(x) ≥ a₀ > 0`),
and satisfies all hypotheses of Theorem 6 (Remizov 2025). The PDE
`u_t = a(x)·u_xx` does **not** admit an elementary closed-form solution
for generic Gaussian initial data, so the property
`diffusion_chernoff_variable_order2` uses a **Richardson-extrapolated
self-consistency reference**: run the v0.2.3 implementation itself at
`n_ref = 8192` and compare against runs at `n ∈ {32, 64, 128, 256,
512}`. The error metric `‖D(T/n)^n f − D(T/n_ref)^{n_ref} f‖_∞` is
asymptotic to the true error up to `(n_ref/n)^{-2}` correction, which
is `≤ 4·10⁻⁴` of the time error for the smallest `n` in the sweep.
Log-log slope of error vs. `τ` over the 5-point sweep is asserted
`≤ -1.95`.

**Why no closed-form oracle**: for variable `a(x)`, elementary closed
forms require the Liouville transform to land on constant-coefficient
heat in transformed variables — only `a(x) = α₀·exp(2γx)` admits this,
and even then the transform produces a *drift-diffusion* equation in
`y`, not pure heat (see math.md §9.2.2). v0.2.2 (drift-reaction)
worked around this by choosing `b(x) = -γx` linear-restoring drift,
which has elementary characteristics. There is no analogous variable-`a`
profile whose `u_xx`-PDE has elementary Gaussian closed-form. The
self-consistency reference is the standard variable-coefficient
empirical-order test in the numerical-PDE literature (LeVeque,
*Finite Difference Methods for ODEs and PDEs*, 2007, §9.6;
Hairer-Lubich-Wanner §III.5.7). It tests *exactly* what we want:
the order at which the implementation's error decays as `τ → 0`,
indifferent to the unknown true value.

---

## Amendment 5 (v0.2.2) — RK2 lift for `DriftReactionChernoff::apply`

**Trigger**: v0.2.0 GAP 2 (audit). The current `R(τ)f(x) = e^{τc(x)} f(x + τb(x))`
is *exact* for constant `(b, c)` but only first-order accurate for variable
coefficients (math.md §9.3, pre-v0.2.2). Amendment 4's "constant-coefficient
regime" caveat in §9.7 explicitly defers variable-`b` testing to v0.3+; v0.2.2
lifts that restriction by upgrading `R` to a Runge–Kutta-2 (midpoint +
trapezoidal) characteristic step:
$X(τ,x) := x + τ\,b\big(x + (τ/2)\,b(x)\big)$ and
$R(τ)f(x) := \exp\big((τ/2)(c(x) + c(X(τ,x)))\big) \cdot f(X(τ,x))$,
local $O(τ^3)$ / global $O(τ^2)$ for $b \in C^2$, $c \in C^1$. **Public API
unchanged**: same struct fields (`b, c, c_norm_bound, grid`), same constructor
`new(b, c, c_norm_bound, grid)`, same `ChernoffFunction::apply / order / growth`
signatures — only the internal `apply_at_node` is rewritten. The reduction-to-
constant lemma (math.md §9.3) proves the new formula collapses *bit-equal modulo
floating-point rearrangement* to the v0.2.1 closed form when $b, c$ are constant,
so v0.2.0/v0.2.1 acceptance gates G1, G2, G3-strang, G4-strang, G1/G2-legacy
are regression-safe by construction (zero perf regression target ±1% with
`c_norm_bound`-based growth bound retained verbatim). Strang composition
preserves global $O(τ^2)$ in the variable-coefficient regime via the
palindromic-trapezoidal symmetry of the new $R$ (Hairer–Lubich–Wanner §III.5,
Thm 4.1, canonical — no re-derivation required); v0.3.0 (2D tensor product)
inherits the upgrade transparently. New properties enforce both regimes:
`drift_reaction_constant_fast_path_exact` (1000 cases, tolerance $10^{-13}$)
and `drift_reaction_variable_order2` (200 cases, log-log slope $\le -1.95$
against the linear-restoring oracle $b(x) = -\gamma x$, $c \equiv -\kappa\gamma$
with closed-form $u(τ,x) = e^{-\kappa\gamma τ} \exp(-x^2 e^{-2γτ}/(2\sigma^2))$
for Gaussian initial data — the simplest variable-$b$ ODE with elementary
closed-form characteristic flow). **Supersedes**: Amendment 4's deferral of
variable-coefficient testing to v0.3+. Files modified: `contracts/semiflow-core.math.md`
§9.3 (rewrite), `contracts/semiflow-core.traits.yaml` `DriftReactionChernoff`
(invariants I1–I3 added), `contracts/semiflow-core.properties.yaml` (two new
properties), `.dev-docs/verification/variable-drift-rk2-derivation.md` (NEW
algebraic derivation). Files NOT modified: `contracts/semiflow-core.errors.yaml`
(no new error variants — RK2 is total over the same input domain as v0.2.1),
`crates/semiflow-core/src/drift_reaction.rs` (Engineer Stage 6 implements next).
