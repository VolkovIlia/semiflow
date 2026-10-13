# §69 — Stiff kinetics: adaptive ROS2 reaction step (ADR-0212, NORMATIVE library)

## §69.1 — Scheme

For the pointwise system `u′ = f(t, x, u)` (`K` species, `N` nodes, species-major
`u`) and a step `h`, with `γ = 1 + 1/√2` and per node `W_i = I − γhJ_i`
(`J_i = ∂f/∂u` at `(t, x_i, u_i)`):

`W k₁ = f(t, u)`,  `W k₂ = f(t + h, u + hk₁) − 2k₁`,  `u⁺ = u + h(3k₁ + k₂)/2`.

Stability function `R(z) = (1 + (1 − 2γ)z + (γ² − 2γ + ½)z²)/(1 − γz)²` with
`γ² − 2γ + ½ = 0`: `R(∞) = 0` (L-stable). Order 2 for every `J` (W-method).

## §69.2 — Step control (NORMATIVE)

`e = h(k₁ + k₂)/2`, `err = max_{k,i} |e_{k,i}|/(atol + rtol·max(|u_{k,i}|, |u⁺_{k,i}|))`
(a component with `e = 0` contributes 0, also at a zero scale; `∞` if `W` is singular
or a stage value is non-finite). Accept iff `err ≤ 1`; the next
step is `h·clamp(0.9/√err, 0.2, 5)` (`err = 0`: `×5`), a rejected step shrinks by
`max(0.2, 0.9/√err)`. The first try of a half-step `[t, t + H]` is `H`; the last
accepted step ends exactly at `t + H`. At most `max_steps` attempts, else
`ConvergenceFailed`. `J` and `f(t, u)` are evaluated once per accepted state; a
non-finite `f(t, u)` or `J` there is a `DomainViolation` (no retry: the `∞` rule
applies to stage values only), and so is a non-final step `h ≤ 16ε·elapsed` on the
half-step's local clock (no progress).

## §69.3 — Jacobian

`Kinetics::jacobian_batch` when provided, layout `jac[(k·K + l)·N + i]`; otherwise
`J_{kl,i} = (f_k(u + δ_i e_l) − f_k(u))/δ_i`, `δ_i = (u_{l,i} + √ε·max(1, |u_{l,i}|)) −
u_{l,i}`.

## §69.4 — Gates

`G_STIFF_KINETICS_LINEAR`, `G_STIFF_KINETICS_PER_NODE`, `G_STIFF_KINETICS_FD_JAC`,
`G_STIFF_ROBERTSON` (ADR-0212).
