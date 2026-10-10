# ADR-0204 — Build each step's interpolation data once: prepared sampling

- **Status**: Accepted
- **Date**: 2026-10-10
- **Supersedes / amends**: none. Complements ADR-0041 (scratch arena), ADR-0109
  (septic default), ADR-0117 (octonic), ADR-0104 (Chebyshev sampling).
- **Contract**: math §40 / §41.bis / §9.2.7 unchanged (same numbers to the bit);
  gate `G_PLAN_BIT_EQUAL` in `contracts/semiflow-core.properties.yaml`.

## Context

The grid engines evaluate `S(τ)f` node by node; each node takes 1–34 off-grid
samples `f(x + shift)`. `GridFn1D::sample` then rebuilds, for every sample, the
ghost-extended nodal data of the interpolant:

- septic Hermite (the default since ADR-0109): `f, dx·f′, dx²·f″, dx³·f‴` at both
  cell ends from 8-, 7- and 6-point FD stencils — 44 `bc_value` calls (each a
  boundary-policy dispatch, `exp` for Robin) per sample, ~440 per node per step
  for variable-`a` `DiffusionChernoff`;
- octonic Hermite: 5 data per end, 10-point stencils;
- Chebyshev spectral: all `M + 1` virtual nodes, each a septic sample — about
  29 000 `bc_value` calls per node per step at `M = 64`.

All of it depends only on the state being sampled, which is the same for every
sample of the step. The ζ-corrections of `Diffusion4th`/`6th` additionally sampled
their shared 7/9 stencil points three times (once per derivative) and called
`libm::pow` per node for step-constant values. Measured: `Strang2D` 400², 10 steps,
3.30 s (≈690 ns per 1-D node update); `Diffusion4th` + Chebyshev sampling, N = 256,
20 steps, 104 ms.

## Decision

1. **Prepared sampling.** `sample_table::PreparedGridFn` fills, once per step, a
   ghost-extended table over `[−24, n − 1 + 24]`: `[f, dx·f′, dx²·f″, dx³·f‴]` per
   node (septic, stride 4, so a cell's eight data are contiguous), the five octonic
   data, the plain values (Catmull–Rom), or the `M + 1` virtual-node values
   (Chebyshev). Every entry is produced by the SAME function the direct sampler
   calls (the FD stencils now take a value fetcher: `bc_value`, or a slice read
   where the stencil is interior — the same number), and a sample combines them in
   the same left-to-right order. Cells outside the margin fall back to the direct
   formula. The result is bit-identical by construction, which the gate checks.
2. **Engines** (`DiffusionChernoff`, `ShiftChernoff1D`, `DriftReactionChernoff`
   incl. its transpose, `Diffusion4th`/`6th` incl. octonic/Chebyshev re-views
   without cloning `src`, `DriftReactionZeta4`) take the table from the scratch
   pool in `apply_into`; their node kernels are generic over `Sample1D`
   (`GridFn1D` = direct, `PreparedGridFn` = table), so both paths share one
   formula. Compositions (`AxisLift`, `Strang2D/3D`, `StrangSplit`, the ζ-ladders)
   inherit the speedup through the inner `apply_into`.
3. **Shared ζ stencils.** The three Fornberg derivative sums reuse one set of
   samples; `Δ`, `Δ^k` are computed once per step. Same samples, same sums.

Rejected: *assembling `S(τ)` as a sparse matrix* (one row per node). It is not
faster than the table (≈110 non-zeros per row for septic variable-`a` versus ≈80
multiply-adds with the table), changes the floating-point order (would re-record
9 digests and break five plan-vs-direct 0-ULP gates), and its memory grows with
the shift range. *Caching per-sample weights across steps* (a τ-keyed plan) is
deferred: it needs a trait hook and gives a further ≈2–3×, but the table alone
already removes the dominant cost.

## Consequences

Measured on this container (best of 5; before → after, identical errors):
`ShiftChernoff1D` N = 1000, 100 steps 17.9 → 6.3 ms; `DiffusionChernoff` const-a
29.2 → 9.4 ms, closure a ≡ ½ 60.9 → 20.0 ms, variable a 73.2 → 27.9 ms;
`Diffusion6th` N = 1024, 50 steps 21.9 → 6.8 ms; `Diffusion4th` Chebyshev
sampling 104 → 2.7 ms (39×); `Strang2D` 400² 3.30 → 1.14 s; `Strang3D` 64³
8.2 → 3.8 s. Memory: one table of `4(n + 48)` doubles per step (per row in 2-D/3-D),
from the scratch pool — zero allocations in steady state. No API change.

## Honest limits

- f64 only: the f32 / `Dual` engines still use the generic direct samplers.
- `AdaptivePI` and other per-call users benefit (the table is per call), but
  coefficient closures, `sqrt` and the Hermite weight polynomials are still
  evaluated per node per step; a τ-keyed plan would remove those.
- The 2-D/3-D compositions still copy rows/columns and sweep Y/Z with stride.

## Gate

`G_PLAN_BIT_EQUAL` (`src/sample_table_tests.rs`): for every interpolant (septic,
octonic, cubic, linear, Chebyshev × three `OobPolicy`) × every `BoundaryPolicy` ×
`n ∈ {8, 37, 128}`, at > 100 000 probes reaching past the ghost margin, at every
node and every virtual node: `prepared.sample(x)` equals `Grid1D::interp` to the
bit (or both return the same error), on the lane path and under the force-scalar
hook. End-to-end witness: every `semiflow-nostd-check` digest of a grid engine is
unchanged.
