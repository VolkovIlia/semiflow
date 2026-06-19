# ADR-0014 — v0.6.0 Adaptive PI step controller (`AdaptivePI<C>`)

**Status**: Accepted
**Date**: 2026-05-01
**Authors**: ai-solutions-architect
**Supersedes**: none. Implements Phase 1 of the approved plan
`/home/volk/.claude/plans/composed-cooking-reef.md` (v0.6.0).
**Cross-refs**: ADR-0011 (Magnus integrator), ADR-0013 (4th-order
spatial — sibling v0.6.0 ADR), `contracts/semiflow-core.math.md` §11
(adaptive PI controller, NORMATIVE), `contracts/semiflow-core.errors.yaml`
(new `AdaptiveStepRejected` variant), `contracts/semiflow-core.traits.yaml`
(`AdaptivePI<C>` and `AdaptiveOutcome<S>` struct entries),
references Söderlind 2002 (PI controllers for stiff ODEs) + Hairer–
Lubich–Wanner *Geometric Numerical Integration* §IV.2 (step-size
control) and §II.3 ("Lady Windermere's fan" — semigroup-splitting
local-truncation argument).

Adopt **`AdaptivePI<C>` as a generic semigroup-splitting integrator**
(NOT a `ChernoffFunction`). `AdaptivePI` wraps any `C: ChernoffFunction`
and performs adaptive sub-stepping over a target time `T`: each substep
applies `S(τ_i) ≈ e^{τ_i A}` on `[t_i, t_{i+1}]` and the substeps are
glued via the semigroup property `e^{(t_2-t_0)A} = e^{(t_2-t_1)A} ·
e^{(t_1-t_0)A}` of the *target* semigroup — the integrator is therefore
a **splitting scheme along the time axis**, with convergence governed
by per-step **local truncation** `O(τ^{p+1})` where `p = func.order()`
(NOT by Theorem 6's `O(1/n)` Chernoff-product bound). This is HLW §II.3
Lady-Windermere-fan accumulation: total error ≤ Σ local errors over
substeps × stability constant ≤ T · O(τ^p) under quasi-contractivity.
Step controller: half-step Richardson local-error estimator
`err = (u_half − u_full) / (2^p − 1)` (1/3 for p=2, 1/15 for p=4,
1/63 for p=6); PI gains `α = 0.7/p, β = 0.4/p` (Söderlind 2002 default
"PI.4.7" parameter set, standard for diffusion-dominated semigroups);
mixed-tolerance norm `tol = tol_abs + tol_rel · max(‖u‖_∞,
‖u_full‖_∞)`; controller update
`τ_new = τ · safety · (tol/err)^α · (err_prev/err)^β` clamped to
`[min_ratio·τ, max_ratio·τ]`. Defaults `safety=0.9, min_ratio=0.2,
max_ratio=5.0, tol_abs=1e-8, tol_rel=1e-6, max_substeps=100_000`. Per
substep: 3 `func.apply` calls (one full + two halves). **NEW error
variant** `SemiflowError::AdaptiveStepRejected { last_tau: f64, last_err:
f64, steps_attempted: usize }` — returned ONLY when
`steps_attempted >= max_substeps` (runaway protection, pathological
toleration mismatch); the PI controller's bounded multiplicative
update rule guarantees that within max_substeps the controller drives
err below tol or escalates to this hard error. **API**: `AdaptivePI<C>{
func, tol_abs, tol_rel, safety, alpha, beta, min_ratio, max_ratio,
max_substeps }` with method
`evolve_adaptive(t: f64, u0: &C::S) -> Result<AdaptiveOutcome<C::S>,
SemiflowError>` returning `AdaptiveOutcome { final_state: C::S,
steps_accepted: usize, steps_rejected: usize, last_tau: f64 }`.
**`AdaptivePI` does NOT implement `ChernoffFunction`** — there is no
fixed `n`, the substep count is a runtime function of (tol, the local
solution roughness, and `func.order()`). This decouples the
semigroup-splitting interpretation from the Chernoff-product machinery
and SHOULD be documented prominently in `src/adaptive.rs` rustdoc and
in the math contract §11 (NORMATIVE clarification — see plan §3 R3).
Generic over any `C: ChernoffFunction`: works on 1D heat, drift-
reaction, ζ-A, Magnus, ζ⁴, Magnus⁴, `Strang2D`, even nested
`StrangSplit<DiffusionChernoff, DriftReactionChernoff>` — no
specialisation per inner type. Rejected alternatives: (a) embedded
Runge–Kutta pair (would require a different generator-flow API and
duplicates step-control logic specific to ODE state vectors —
incompatible with `ChernoffFunction`'s semigroup signature); (b) fixed
local-truncation extrapolation (Richardson) without PI smoothing
(empirically ~30% more rejections at the same tol — Söderlind 2002 §3);
(c) implementing `AdaptivePI` as a `ChernoffFunction` (would require
fixing `n` at construction — defeating the purpose). Consequences:
**+1 public generic struct**, **+1 helper struct** (`AdaptiveOutcome`),
**+1 SemiflowError variant** (additive under `non_exhaustive`), **+0
dependencies**; G_PI flagship gate is `steps_adaptive(tol=1e-5) ≤ 0.7
× n_fixed_for_same_err` on stiff CEV (high-corner, where v0.5.0 fixed
step is wasteful) — failure does NOT block release (gate is observable,
not normative; Magnus⁴ + ζ⁴ deliver release-blocking value
independent of adaptive efficiency).

**Amendment 1 (v0.6.1, 2026-05-02)**: The Richardson local-error
estimator divisor `2^p − 1` and the PI gains `α = 0.7/p, β = 0.4/p`
in this ADR derive from the **τ-axis** Chernoff consistency order
`p = func.order()` — NOT from any inner type's dx-axis spatial
accuracy `q`. For all currently-shipped inner functions
(`ShiftChernoff1D` p=1; `DiffusionChernoff`, `Diffusion4thChernoff`,
`TruncatedExpDiffusionChernoff`, `TruncatedExp4thDiffusionChernoff`,
`StrangSplit<…>`, `Strang2D<…>` all p=2), the divisor is `2^2 − 1 = 3`
and the gains are `α = 0.35, β = 0.20`. The dx-axis spatial accuracy
`q ∈ {1, 2, 4}` is orthogonal and observable only via the
G3 / G3⁴ / G3-2D / G3⁴-2D convergence-slope gates — `q` MUST NOT be
substituted into the PI controller's exponents. See math.md §11.1.bis
(NORMATIVE clarification, v0.6.1) and `docs/audit-findings-v0_6_0.md`
D1 — the v0.6.0 incident in which `Diffusion4thChernoff::order()` and
`TruncatedExp4thDiffusionChernoff::order()` returned `4` (their dx-axis
spatial order) propagated an incorrect divisor `2^4 − 1 = 15` and
gains `α = 0.175, β = 0.10` through `AdaptivePI<C>`, degrading the
controller's local-error estimate by a factor of 5× and biasing
substep selection toward over-aggressive growth.
