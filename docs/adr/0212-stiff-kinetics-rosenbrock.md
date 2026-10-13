# ADR-0212 — Stiff kinetics: adaptive Rosenbrock (ROS2) reaction step

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: amends ADR-0208 (the reaction half-step of
  `ReactionDiffusion`); additive (the number was reserved in ADR-0205).
- **Contract**: math §69 (`contracts/semiflow-core.math-stiff-kinetics.md`); gates
  `G_STIFF_KINETICS_LINEAR`, `G_STIFF_KINETICS_FD_JAC`, `G_STIFF_ROBERTSON`.

## Context

`ReactionDiffusion` (ADR-0208) integrates kinetics without an exact flow by classical
RK4 with a fixed number of substeps. Stiff chemistry — Robertson, combustion,
atmospheric mechanisms, fast–slow ecological models — has Jacobian eigenvalues
`|λ| ≫ 1/h`, so RK4 is stable only for `h·|λ| ≲ 2.8`: on Robertson's problem it
needs at least 40 000 substeps per half-step of 20 time units (9.4 s for 200 nodes)
and returns NaN below that. An L-stable implicit method is required.

## Decision

1. **ROS2** (Verwer, Spee, Blom & Hundsdorfer, SIAM J. Sci. Comput. 20 (1999)),
   `γ = 1 + 1/√2`, per node `W = I − γhJ` (`K×K`, `LU` with partial pivoting):
   `W k₁ = f(t, u)`, `W k₂ = f(t + h, u + hk₁) − 2k₁`, `u⁺ = u + h(3k₁ + k₂)/2`.
   Two stages, one `LU` per node and step, L-stable (`R(∞) = 0`), and of order 2 for
   ANY matrix `J` (a W-method): a forward-difference Jacobian, or the omitted
   `∂f/∂t` of a non-autonomous `f`, keeps order 2. Linear invariants with
   `eᵀJ = 0` (mass in Robertson) are conserved to rounding.
2. **Adaptive step, one size for the field.** Local error
   `u⁺ − (u + hk₁) = h(k₁ + k₂)/2` (the embedded first-order solution), mixed norm
   `max_{k,i} |e|/(atol + rtol·max(|u|, |u⁺|))`, accept at `≤ 1`, next step
   `h·clamp(0.9/√err, 0.2, 5)`; a singular `W` or a non-finite stage counts as a
   rejection (`×0.2`). Every stage is ONE batched kinetics call (vectorised Python
   callbacks stay `O(steps)` calls). The controller is pure arithmetic:
   reproducible bit for bit. The first try is the whole half-step.
3. **Jacobian.** `Kinetics` gains a provided `jacobian_batch(t, n, x, u, jac) →
   Option<Result<()>>` (layout `jac[(k·K + l)·N + i] = ∂f_k/∂u_l`; default `None`).
   The catalogue models without an exact flow (Nagumo, Gray–Scott,
   FitzHugh–Nagumo, Brusselator) implement it analytically; otherwise the solver
   forms forward differences, `δ = √ε·max(1, |u_l|)` (exactly representable), from
   `K` extra batched evaluations.
4. **API** (additive): `ReactionSolver { Rk4 (default), Rosenbrock(RosenbrockOptions)
   }`, `RosenbrockOptions { rtol = 1e−6, atol = 1e−9, max_steps = 100 000 }` (both
   `#[non_exhaustive]`), `ReactionDiffusion::with_reaction_solver`. An exact flow
   still takes precedence; RK4 stays the default, bit-identical. Stage buffers come
   from the scratch pool once per evolution. Python:
   `ReactionDiffusion*(…, reaction_solver="rk4" | "rosenbrock", rtol, atol,
   jacobian=None)`, `jacobian(t, x, u) -> (K, K, N)` for a callable reaction.
   `dense_exp` moves verbatim (bit-identical) with the new `LU` into `dense_la.rs`.

## Consequences

Robertson on 200 nodes (no diffusion), `T = 40`, Python, same session:

| integrator | wall clock | max relative error |
|---|---:|---:|
| RK4, 40 000 substeps (smallest stable count tried) | 9.4 s | 2.9e−12 |
| RK4, 20 000 substeps | — | NaN (unstable) |
| ROS2, `rtol = 1e−3`, analytic / FD Jacobian | 80 / 112 ms | 6.0e−4 |
| ROS2, `rtol = 1e−5` | 0.54 / 0.78 s | 6.2e−6 |
| ROS2, `rtol = 1e−7` | 2.6 / 3.7 s | 5.8e−8 |

## Honest limits

- Order 2 only: cost grows like `tol^{−1/2}`; at `rtol ≲ 1e−8` on Robertson the
  step cap (100 000 per half-step) is reached — use a looser `rtol` or raise
  `max_steps` (`ConvergenceFailed` reports the step size reached).
- One step size for the whole field: the stiffest node dictates the step.
- Positivity is not guaranteed (measured `min ≥ 0` on the gates).
- A diffusion step between half-steps is not part of the error control: the
  splitting error of ADR-0208 is unchanged.

## Gates

- `G_STIFF_KINETICS_LINEAR` (`tests/g_stiff_kinetics.rs`): `u′ = Cu`, eigenvalues
  `−10⁶`, `−1`, identity diffusion, `T = 1`. RK4 with one substep (`hλ = −10⁴`)
  blows up (error `> 10⁶`); single ROS2 steps converge with order 2 (slopes 1.87,
  1.93 over `n = 20, 40, 80`; window `[1.8, 2.2]`); adaptive ROS2 at
  `rtol ∈ {1e−4, 1e−6, 1e−8}`: error `≤ 10·rtol` (measured `0.92·rtol`).
- `G_STIFF_KINETICS_FD_JAC`: the catalogue Jacobians equal central differences of
  `eval_batch` to `≤ 1e−7` (measured `≤ 2.8e−10`); the forward-difference run agrees
  with the analytic one to `≤ 10·rtol` at `rtol = 1e−8`.
- `G_STIFF_ROBERTSON`: Robertson (`k₁ = 0.04, k₂ = 3·10⁷, k₃ = 10⁴`), `T = 40`,
  analytic and forward-difference Jacobians, against SciPy Radau (`rtol = 1e−13`):
  per-species relative error `≤ 10·rtol` for `rtol ∈ {1e−3, 1e−5, 1e−7}`,
  `|Σyₖ − 1| ≤ 1e−12`, `min yₖ ≥ −1e−12`.
