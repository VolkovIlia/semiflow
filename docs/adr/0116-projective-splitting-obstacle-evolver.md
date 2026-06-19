# ADR-0116 — Projective-splitting obstacle evolver / variational inequalities (v6.3.0)

- Status: Accepted
- Date: 2026-06-03
- Designed-By: ai-solutions-architect
- Supersedes: none. Extends ADR-0115 (graph adjoint honesty boundary), ADR-0068 (KillingChernoff post-projection pattern).
- Authority for: math.md §44.

## Additive kernel family: `ObstacleChernoff<C, O, F>` (CONFIRMED)

A new post-projection wrapper family — `ObstacleChernoff<C, O, F>` backed by `Obstacle<F>` trait with `ConstantObstacle` and `ClosureObstacle` implementations — composes any library `ChernoffFunction` approximating `e^{ΔτL}` with the metric projection `Π_g(W) = max(W, g)` onto the closed convex cone `K = {V : V ≥ g}`, realizing the projective-splitting iterate `V^{n+1} = Π_g(S(Δτ)Vⁿ)` for obstacle problems / variational inequalities / optimal-stopping (`min(∂_τV − LV, V − g) = 0`). The design mirrors `KillingChernoff` (ADR-0068) as an additive zero-alloc post-step wrapper. `order()` returns `1` — declared honest: for constant-coefficient generators with a convex obstacle this is the proven global order (Jaillet–Lamberton–Lapeyre 1990; Leduc 2025 `O(1/n)` convex / `O(1/√n)` non-convex); for variable-coefficient `L` the sharp composite order is **not a theorem here** but is empirically gated (`G_OBSTACLE_SLOPE_SMOOTH` OLS slope −0.997 ≤ −0.95; `G_OBSTACLE_SLOPE_AMERICAN` slope −0.825 ≤ −0.45; math.md §44.4 honesty posture). The stationary membrane correctness gate `G_OBSTACLE_STATIONARY` passes (sup-error 5.55e-3 ≤ 2.5e-2, membrane analytic oracle, RELEASE_BLOCKING). The sympy PRE-FLIGHT oracle `T_OBSTACLE_PROJECTION` passes 6/6 (projection identity, idempotence, nonexpansiveness, active-set Jacobian, stationary membrane, complementarity). `growth()` returns the inner's homogeneous growth — exact for `g ≤ 0` (sub-Markov), otherwise carrying a documented additive `‖g⁺‖_∞` offset that the `Growth` struct cannot express; the operative stability certificate is `Π_g`-nonexpansiveness (Theorem 44.1.3, math.md §44.2), not the multiplicative bound. The `c(x) > 0 / ω > 0` non-contractive regime (where `Π_g`-nonexpansiveness fails) is **CONJECTURAL**, documented with reference to the Trotter projection counterexample (arXiv math/0109049), and not gated. Viscosity convergence under monotone + nonexpansive inner follows by Barles–Souganidis 1991 (Theorem 44.2); the m-accretivity / proximal-map characterization of `max(W, g)` follows Crandall–Liggett 1971 and Brezis–Pazy 1972 (Theorem 44.1); forward–backward splitting interpretation follows Lions–Mercier 1979.

## Active-set adjoint primitive (CONFIRMED; does NOT implement `AdjointApply<F>`)

A separate inherent primitive `ObstacleChernoff::apply_active_set_adjoint_into` realizes the linearized backward step `λ_next = S*(Δτ)[diag(𝟙[W_fwd > g]) · λ]` — the active-set diagonal mask (Theorem 44.3, Clarke/Bouligand Jacobian a.e.) applied first, followed by the inner adjoint — where `W_fwd = S(Δτ)Vⁿ` is the pre-projection forward state, frozen at the correct linearization point. This primitive is NOT exposed via `ChernoffFunction::apply_adjoint_into` and `ObstacleChernoff` MUST NOT implement `AdjointApply<F>`: the projection is non-differentiable at the kink, so claiming the genuine-adjoint supertrait would be dishonest (same honesty posture as ADR-0115 §42 for the Magnus graph adjoint). When the inner does not support `apply_adjoint_into`, the primitive returns `UnsupportedOperation`. The gate `T_OBSTACLE_ADJOINT` passes (active-set adjoint vs central-FD rel err 1.03e-12, O(ε²)); 6/6 proptest invariants pass (lower-bound, idempotence, nonexpansiveness, monotonicity, composite order-preservation, active-set consistency). Second-order greeks (Γ) at the free boundary are **OPEN** (math.md §44.5).

## Normative in-core / out-of-core boundary

IN-CORE: projective splitting `Π_g ∘ S(Δτ)` and the active-set adjoint primitive. OUT-OF-CORE (MUST NOT ship as engine paths): the penalty method `∂_τV = LV + ρ·max(g−V, 0)` and PSOR / implicit LCP — both may appear as reference oracles only. All financial applicability (price ↔ log-price maps, discounting, full pricers) belongs in a downstream finance project; the shipped obstacles are abstract math objects. The family is D=1 only (shared `GridFn1D` coordinate-access constraint; multi-asset obstacles deferred).

## Bench (new_kernels4 Group 5)

~705 Kelem/s/step.

## References (math.md §44.8)

Crandall–Liggett 1971; Brezis–Pazy 1972; Lions–Mercier 1979; Barles–Souganidis 1991; Jaillet–Lamberton–Lapeyre 1990; Leduc 2025 (doi:10.3390/math13020213); Howison–Reisinger–Witte 2013 (doi:10.1137/090776089); Peskir 2005; arXiv math/0109049 (Trotter counterexample); Chernoff 1968; Remizov 2025.
