# ADR-0208 — Semilinear reaction–diffusion systems on every engine

- **Status**: Accepted
- **Date**: 2026-10-10
- **Supersedes / amends**: none (additive). Python surface: ADR-0209.
- **Contract**: math §64.1; gates `G_SEMILIN_LINEAR_SYSTEM_EXACT`,
  `G_SEMILIN_SOURCE_MMS`, `G_SEMILIN_FISHER_WAVE`, `G_SEMILIN_NAGUMO_WAVE`,
  `G_SEMILIN_GRAY_SCOTT_2D`.

## Context

The crate evolved linear equations: diffusion, drift, reaction `c(x)u`, on
grids, graphs and manifolds. Nonlinear dynamics existed only as Allen–Cahn
through ETDRK4 on a CSR generator. The bulk of applied PDE work — pattern
formation (Gray–Scott, Brusselator), excitable media (FitzHugh–Nagumo),
population fronts (Fisher–KPP), phase separation (Allen–Cahn), source terms
`f(t, x)` — is semilinear: `∂ₜu_k = L_k u_k + f_k(t, x, u)` with a few
species. Every linear engine of the crate is already a good `e^{τL_k}`.

## Decision

1. `Kinetics<F>`: `K` species, rates evaluated for ALL nodes in one call on
   flat species-major buffers (`u[k·N + i]`), coordinates axis-major. Optional
   `exact_flow`. One batched call per Runge–Kutta stage is what makes a
   host-language (Python/NumPy) reaction cost `O(steps)` calls, not
   `O(steps·N)`.
2. Catalogue: `FisherKpp` (exact logistic flow), `AllenCahnReaction` (exact
   flow), `Nagumo`, `GrayScott`, `FitzHughNagumo`, `Brusselator`,
   `LinearReaction` (exact `e^{hC}`), `FnReaction` (any pointwise closure).
3. `NodalField<F>` (values + coordinates) for `GridFn1D/2D/3D` and
   `GraphSignal`; `Species<S>` (one field per species) implements `State`.
4. `ReactionDiffusion<F, S>`: one boxed engine per species (each its own
   diffusivity, boundary policy, interpolant, dimension composition), Strang
   splitting `R(τ/2)∘L(τ)∘R(τ/2)` with adjacent reaction half-steps merged,
   reaction by exact flow or classical RK4 with `r` substeps (stiff kinetics),
   time-dependent kinetics supported. Order `min(2, min_k order(C_k))`;
   `StepIntegrator`, so Richardson (ADR-0207) raises it. A non-finite reaction
   result is an error, not a silent `inf`.

Rejected: an implicit/IMEX reaction solver (needs Jacobians and a nonlinear
solve; RK4 substeps or an exact flow cover the catalogue; stiff kinetics remain a
limit); per-node callbacks for Python (`O(N·steps)` crossings); a const-generic
`K` (the Python surface needs a runtime `K`).

## Consequences

Measured: commuting linear system exact to `8.6e−16`; MMS order 2.05 (3.1 with
two Richardson levels); Fisher–KPP and Nagumo exact fronts order 2.00 / 2.06
(`sup` error `6.9e−7` / `3.2e−5` at 40 steps); Gray–Scott 2-D (periodic,
`Strang2D` engines) self-convergence 2.07, positivity kept. Additive API.

## Honest limits

- Strang splitting is order 2; stiff kinetics need substeps (explicit RK4
  stability `h·|∂f/∂u| ≲ 2.8`).
- Positivity is not guaranteed by the scheme (RK4 and septic interpolation can
  undershoot); the gate measures it on Gray–Scott only.
- Coupled diffusion (cross-diffusion, `L` mixing species) is out of scope; use
  `MatrixDiffusionChernoff` for linear coupled systems.

## Gate

See the gate list above.
