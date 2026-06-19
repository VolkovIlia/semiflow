# ADR-0024 — v0.9.0 3D tensor product (Grid3D, GridFn3D, AxisLift3D, Strang3D)

**Status**: Proposed
**Date**: 2026-05-07
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0012 (`Strang2D<X, Y>` — 2D ancestor; 3D inherits per-axis
splitting verbatim and degenerates to it when $L_z = 0$), ADR-0011
(`SemiflowError::CflViolated` reused, no new variants), Theorem 7
(`contracts/semiflow-core.math.md` §10.3 — 2D base case, generalised to N axes
in §10.8.2 as Theorem 7'), `verify_v0_9_0_3d_tensor.py` (Block C wave 2
engineer deliverable), `contracts/semiflow-core.math.md` §10.8 (NORMATIVE,
math companion). Cites HLW (2006) §III.5.2 for higher-dimensional Strang
splitting.

## Context

v0.5.0 (ADR-0012, math.md §10) shipped `Strang2D<X, Y>` for separable 2D
tensor-product operators $L = (L_x \otimes I) + (I \otimes L_y)$ with
per-axis Chernoff legs $A := L_x \otimes I$ and $B := I \otimes L_y$
satisfying $[A, B] = 0$ (axis-disjoint commutation). Theorem 7 lifts the
per-axis order to a global order-2 Strang composition without invoking
BCH commutator cancellation (Remark 10.1 — purely per-axis inheritance).
The math.md §10.7 deferral note and the §10.8 forward pointer at
line 3091-3093 explicitly listed "**3D and higher**: the same construction
extends inductively but is not part of v0.5.0 scope".

Three years and four releases later (v0.5–v0.8.1), every per-axis leg
the v0.5.0 catalogue ships (`DiffusionChernoff`, `TruncatedExpDiffusionChernoff`,
`Diffusion4thChernoff`, `AdvectionDiffusionChernoff`, `LiouvilleChernoff`)
has been validated for axis-lifted use; the 2D code path is hardened
(G3-strang-2D and G3⁶-2D gates green at flagship-N in v0.8.1). The
inductive extension to N=3 is the natural next architectural step,
unblocking 3D heat / advection-diffusion / option-pricing applications
without forcing artificial reduction-of-dimension via slicing.

## Decision

Adopt **`Grid3D`**, **`GridFn3D`**, **`AxisLift3D<C>`** (or alternatively
extend the existing `Axis` enum with a `Z` variant — Block C wave 2 chooses
between an explicit `AxisLift3D` type and an `Axis::Z` extension; the two
are observably identical at the math level by Lemma 10.1, and the choice
is purely an API-ergonomics call to be made when the implementation is
profiled), and **`Strang3D<X, Y, Z>`** as the v0.9.0 3D tensor-product
foundation for separable operators $L = (L_x \otimes I \otimes I) +
(I \otimes L_y \otimes I) + (I \otimes I \otimes L_z)$ on
$H = H_x \otimes H_y \otimes H_z$. Per-axis legs $A := L_x \otimes I \otimes I$,
$B := I \otimes L_y \otimes I$, $C := I \otimes I \otimes L_z$ act on
disjoint tensor factors, hence pairwise commute (**Lemma 10.1**, inductive
form, math.md §10.8.1) so $[A, B] = [A, C] = [B, C] = 0$ identically. The
canonical Strang3D composition is the **palindromic 5-leg** $S_{3D}(\tau) =
e^{\tau A/2}\,e^{\tau B/2}\,e^{\tau C}\,e^{\tau B/2}\,e^{\tau A/2}$ with X
outermost (minimises cache-miss cost on row-major x-fastest storage; bit-equal
under exact arithmetic to any other leg ordering since A, B, C pairwise
commute, but the type fixes the order for FP determinism). By the
**reduction theorem** (math.md §10.8.3), pairwise commutation collapses
$S_{3D}(\tau) = e^{\tau (A + B + C)} = e^{\tau L}$ exactly with **zero BCH
residue** — Theorem 7 generalises inductively to N axes (Theorem 7' in
§10.8.2), with $\mathrm{order}(\mathrm{Strang3D}) = \min(\mathrm{order}(L_x),
\mathrm{order}(L_y), \mathrm{order}(L_z))$ (limited by the weakest axis;
Remark 10.1 applies verbatim — no commutator bonus or penalty in 3D
separable). Memory layout is **row-major `Vec<f64>` of length $nx \cdot ny
\cdot nz$, x-fastest**, with $\mathrm{idx}(i,j,k) = k \cdot nx \cdot ny + j
\cdot nx + i$ (consistent extension of `Grid2D::idx(i,j) = j \cdot nx + i`,
see `crates/semiflow-core/src/grid2d.rs:7-8`); per-axis strides are
$\mathrm{stride}_x = 1$, $\mathrm{stride}_y = nx$, $\mathrm{stride}_z = nx
\cdot ny$, so each per-axis lift is a strided 1D walk reusing the
v0.5.0 `AxisLift` apply path. CFL aggregation is the strictest per-axis
bound: $\tau < \min(\tau_{\max}^x, \tau_{\max}^y, \tau_{\max}^z)$,
propagated as **`SemiflowError::CflViolated`** from the offending
per-axis lift (no new error variant). When $L_z \equiv 0$ ($C = 0$),
$S_{3D}$ collapses to the existing v0.5.0 `Strang2D` apply on $(X, Y)$
bit-equal under FP (gate T10N_zero-axis ratifies this).

## Considered alternatives

- **(a) Single 3D `ChernoffFunction` for non-separable mixed-derivative
  operators** (e.g. $L = \sum_{i \le j} \beta_{ij}(x) \partial_i \partial_j$):
  rejected, out of scope. Requires genuine 3D 27-point cross-stencils, full
  3D BCH residue analysis, and a separate ADR. Mirrors the v0.5→v0.7
  separable→non-separable progression in 2D; v0.9.0 ships the separable
  inductive base.
- **(b) Const-generic `Strang<N>` for arbitrary tensor rank**: rejected,
  suckless. An explicit `Strang3D` type parallels the existing `Strang2D`
  and avoids const-generic plumbing through the Chernoff trait; the
  generalisation to higher N is a separate architectural decision when
  4D becomes a use case (no current caller demand).
- **(c) Reuse `Strang2D` recursively as 2D-of-1D (treat Z as outer Strang
  over a "2D leg")**: rejected. Breaks the contract clarity of `Strang2D`
  (its leg parameters are 1D Chernoff types per spec, not composite
  operators); would require a `Compose<Strang2D, AxisLift1D>` adapter that
  is structurally a `Strang3D` with extra indirection.

## Consequences

- **+4 public types**: `Grid3D`, `GridFn3D`, `AxisLift3D` (or `Axis::Z`
  extension — Block C wave 2 chooses; the math-level contract is
  identical), `Strang3D<X, Y, Z>`. Additive — no v0.5–v0.8.x API break.
- **+0 dependencies, +0 `SemiflowError` variants**: `CflViolated` reused
  with the same semantic field overload (per-axis surfaced).
- **+0 changes to existing source files** beyond `Axis` if the `Axis::Z`
  variant is chosen (one variant addition, otherwise net-zero); a new
  module `grid3d.rs` (~150 LoC) plus `grid_fn3d.rs` (~200 LoC),
  `axis_lift3d.rs` (~120 LoC, or absorbed into existing `axis.rs` for
  the `Axis::Z` route), and `strang3d.rs` (~180 LoC) are added.
- **`contracts/semiflow-core.tensor.yaml`** total_additions schema field
  rises from 7 (v0.5.0) → 11 (v0.9.0) — Block C engineer amendment.
- **No new boundary policy work**: per-axis `BoundaryPolicy` is unchanged;
  Z axis reuses the v0.5.0 enum and `Reflect` / `ZeroExtend` /
  `LinearExtrapolate` rules verbatim.
- **No perf-baseline regression**: the v0.8.1 2D parallel tile-scratch
  path (heat_2d 4.38× speedup) is untouched. 3D performance gates are
  separate — out of v0.9.0 Block B math scope; Block C wave 2 may add a
  3D parallel tiling ADR.

## Forward compatibility

- **v1.0+** MAY add `NonSeparable3DChernoff` for 3D mixed-derivative
  operators (separate ADR; structurally a 3D analogue of v0.7.0
  `NonSeparable2DChernoff`).
- **v1.0+** MAY add `Strang4D` for $H_x \otimes H_y \otimes H_z \otimes H_w$
  separable operators (separate ADR; the inductive Lemma 10.1 already
  covers the math, so the ADR is purely API/perf scope).
- **4th-order spatial in 3D**: supported the moment per-axis
  `Diffusion4thChernoff` (v0.6.0, ADR-0014/0015) is plumbed through
  `AxisLift3D`. No new math.md section needed (per-axis inheritance
  applies); the Block C wave 2 implementation is the gate.
- **3D parallel tiling** (analogue of v0.8.1 Block A 2D tile-scratch):
  separate perf-track ADR; structurally a `(j, k)`-tile parallelisation
  of the X-leg, with `(i, k)`-tile for the Y-leg and `(i, j)`-tile for
  the Z-leg.

## Verification

The Block C wave 2 engineer deliverable produces a passing run of all four
acceptance commands:

1. `python3 .dev-docs/verification/scripts/verify_v0_9_0_3d_tensor.py`
   — exits 0 with all six **T10N_*** sympy gates passing
   (math.md §10.8.6).
2. Regression guard: every prior verifier (`verify_v0_5_0_tensor_2d.py`
   through `verify_v0_9_0_nonseparable_aniso.py`) continues to exit 0
   (no v0.5–v0.8.x semantics change).
3. `cargo run -p xtask -- test-fast` — green workspace.
4. `cargo build --workspace --release` — clean release build.

The empirical slope gate **G5_3D** (math.md §10.8.7) at threshold $\le
-1.95$ over $N \in \{16, 32, 64\}$ (per-axis grid) ratifies the
order-2 claim end-to-end on a constant-coefficient 3D heat oracle (closed-form
3D Gaussian product, eq. 10.8.7 — exact; no high-resolution N_ref needed,
strictly cheaper than the v0.9.0 Block A G4_NS2D_aniso 4-point basket).

## Amendment 2026-05-09 (v0.11.0 G5_3D recalibration — basket $\{16,32,64\}\to\{32,64,128,256\}$)

The first prod-HW run of G5_3D (Intel i7-12700K, AVX2, `RUSTFLAGS="-C
target-cpu=native"`, `--features parallel,simd,slow-tests --release`,
2026-05-09 — deferred from v0.9.0 per the Block C ship note in
`project_v0_9_0_block_c_shipped.md`) found $N = 16$ inside the
**pre-asymptotic regime**: at $N = 16$ the spacing $\mathrm{dx} = 10/15
\approx 0.667$ under-resolves the Gaussian initial datum
$u_0 = e^{-(x^2+y^2+z^2)}$ ($\sigma = 1/\sqrt{2} \approx 0.707$, half-width
spans only $\sim 2$ grid points), distorting the OLS regression (per-N
errors $\{2.37 \times 10^{-2},\, 1.51 \times 10^{-2},\, 4.50 \times 10^{-3}\}$,
ratios 1.57 / 3.36 — $\log_2$ progression $0.65 \to 1.75$ monotonically
approaching the order-2 asymptote but not yet at it; OLS slope $-1.1978$
fails the $\le -1.95$ gate by $-0.75$). The math is unchanged: §10.8.7
order-2 claim and Theorem 7' inductive tensor-collapse proof (§10.8.2)
are intact; the failure is **test-calibration**, not implementation
regression. **Decision** — recalibrate the spatial basket to $N \in
\{32, 64, 128, 256\}$, mirroring the v0.9.0 G4_NS2D_aniso 2D self-convergence
redesign (commit `0180292`, math.md §10.7-ter.7 — same "skip the coarsest
pre-asymptotic point" convention adopted there: drop $N = 32$ from OLS in
2D, drop $N = 16$ from OLS in 3D), keep the gate threshold at $\le -1.95$
unchanged (the basket move into the asymptotic regime is precisely where
the order-2 floor is achievable; predicted slope $\approx -1.95$ to
$-2.00$ per the empirical $\log_2$-ratio extrapolation), preserve the
constant-coefficient closed-form 3D Gaussian oracle (eq. 10.8.7-osc,
exact at every $N$). Memory: $256^3 \times 8\,\mathrm{B} \times 3$ working
buffers $\approx 1.6\,\mathrm{GB}$ peak — well under the 31 GB prod-HW
budget. Wallclock: dominated by the $N = 256$ leg ($N^3$ scaling vs $64^3$
gives $\sim 64\times$ cost amplification for the largest single point;
total revalidation budget $\approx 50$ minutes on i7-12700K with the
parallel-Strang2D path inside each Z-leg). The amendment is **additive
calibration** (constitution v1.1.0 principle "additive surface, never
subtractive"): the $N \in \{32, 64\}$ legs from the original basket remain;
$N \in \{128, 256\}$ are new asymptotic anchor points; only the
under-resolved $N = 16$ is dropped. Reproducibility command: `cargo run
-p xtask -- test-flagship -- --ignored g5_3d_slope`. See
`audit-findings-v0_11_0.md` (docs-writer scope) for the raw per-N
diagnostic data and the bug-fixer one-line edit at
`crates/semiflow-core/tests/strang_3d_slope.rs` (`N_SPATIAL` constant).
