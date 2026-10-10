# ADR-0207 — Richardson extrapolation in the step count

- **Status**: Accepted
- **Date**: 2026-10-10
- **Supersedes / amends**: none (additive).
- **Contract**: math §64.2; gates `G_RICHARDSON_WEIGHTS`, `G_RICHARDSON_SHIFT1D`,
  `G_RICHARDSON_COST_GAIN` (`contracts/semiflow-core.properties.yaml`).

## Context

Every engine in the crate has a fixed consistency order `p` (1 for the shift
engines, 2 for the diffusion family and for Strang splitting, 4/6/8 for the ζ
ladders). Reaching `1e−6` with an order-1 engine takes `≈ 10⁴` steps. The global
error of a Chernoff product `(S(t/n))ⁿf` has an expansion
`c_p n^{−p} + c_{p+1} n^{−(p+1)} + …` for data in the domain of a sufficiently
high power of the generator (Galkin & Remizov 2025), so the leading terms can be
cancelled by combining runs with different `n` — for any engine, without touching
its kernel.

## Decision

1. `richardson_weights(p, L)`: weights `w_j` for step counts `(j+1)·n`, `j < L`,
   solving `Σ w_j = 1`, `Σ w_j (j+1)^{−q} = 0` for `q = p, …, p+L−2` (Gaussian
   elimination with partial pivoting, `L ≤ 6`). The harmonic sequence `n, 2n, 3n`
   keeps the cost at `L(L+1)/2` base runs (the Romberg sequence `n, 2n, 4n`
   costs `2^L − 1`).
2. `StepIntegrator` (leading exponent + `integrate_into(t0, t, n, …)`) with impls
   for `Evolver<C, F>` (any `ChernoffFunction`) and `ReactionDiffusion`;
   `extrapolate_into(integ, t0, t, n, L, src, dst, scratch)` and the convenience
   `Evolver::evolve_extrapolated_into(t, L, src, dst, scratch)`.
3. Tolerance-driven kernels (`order() == u32::MAX`) have no expansion and are
   rejected.

Rejected: geometric (Romberg) step sequence (cost); extrapolation inside the
step (would change every engine and its digests); adaptive level selection (the
error expansion is an assumption on the data, not something a cheap estimator
certifies).

## Consequences

`ShiftChernoff1D` (order 1) on the heat equation: observed order 1.01 / 2.03 /
3.06 with 1 / 2 / 3 levels; at an equal number of steps (96) three levels give
`7.2e−8` against `3.3e−4` (4600×). Reaction–diffusion: order 2 → 3.1 with two
levels. Purely additive API.

## Honest limits

- The combination amplifies rounding and spatial error by `Σ|w_j|` (9 for
  `p = 1`, `L = 3`; 28 for `L = 4`). At the spatial error floor extrapolation
  cannot help; the gates measure above it.
- Rough data (outside the domain of the needed generator powers) has no clean
  expansion; the gain is then smaller.
- Grid engines carry an interpolation error per step (`∝ n·h^q`); extrapolation
  cancels time error only.

## Gate

See the gate list above.
