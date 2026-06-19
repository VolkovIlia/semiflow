# ADR-0044 — `StepController<F>` Trait + H211b Advisory Filter (Wave 4 of v2.0)

**Status**: PROPOSED
**Date**: 2026-05-20
**Wave**: 4 of 5 (v2.0 MAJOR)
**Depends on**: ADR-0041 (Wave 1 scratch arena), ADR-0042 (Wave 2 in-place Strang),
ADR-0043 (Wave 3 `State<F>` 3-layer split)
**Foreshadowed-by**: ADR-0043 (Wave 3 explicitly cites Wave 4 use of `HilbertState::dot`
and `norm_sq`)
**Designed-by**: ai-solutions-architect (Wave 4 design pass)
**Math fidelity gate**: §11.1.bis NORMATIVE (`ClassicalPI` only). `H211bFilter` is
plumbing/advisory; documented at ADR scope, NOT promoted into math.md.

## Decision

Wave 4 lifts the f64 lock on `AdaptivePI<C>` and factors the PI step-size law into a
pluggable `StepController<F>` trait:

1. **`AdaptivePI<C, F: SemiflowFloat = f64, K: StepController<F> = ClassicalPI<F>>`** —
   the integrator becomes generic over the scalar `F` (via `C: ChernoffFunction<F>`,
   ADR-0025/0026) **and** over the step-size law `K`. The triple default
   `(F = f64, K = ClassicalPI<F>)` keeps all v1.x call-sites compiling unchanged.

2. **`StepController<F>` trait** — two methods (`propose_accept`, `propose_reject`)
   that return the next-τ multiplier given current/previous error norms, target
   tolerance, and the inner function's order `p`. State is carried inside the
   implementor (e.g. previous filtered error). The trait operates over the same
   `F: SemiflowFloat` already used by the inner Chernoff function — no new bounds.

3. **`ClassicalPI<F>` impl (NORMATIVE default)** — encodes the Söderlind 2002 "PI.4.7"
   law with gains `α = 0.7/p, β = 0.4/p`. This is the law currently inlined in
   `pi_step_factor` and `reject_step_factor` (`adaptive.rs:159–177`). Bit-identical
   to the v1.0.0 accepted-step trajectory on the F9 (CEV) oracle.

4. **`H211bFilter<F>` impl (advisory, opt-in only)** — Söderlind 2003 "Digital filters in
   adaptive time-stepping" H211b filter. **NOT promoted into math.md §11.bis**. Lives at
   the ADR level only. Targets F9 step-variance reduction. Opt-in via builder
   `AdaptivePI::new(func).with_controller(H211bFilter::default())`.

5. **Zero-allocation Richardson** — the half-step error path migrates off `Clone` (used
   today at `adaptive.rs:194` `u_half.clone()` and `adaptive.rs:316`
   `apply_full_and_half`) onto Wave 1 `ScratchPool<F>` for scratch states plus Wave 3
   `HilbertState::dot`/`norm_sq` for the L²-flavoured error norm. The sup-norm path
   stays available via `State::norm_sup`; the trait surface admits both norms behind a
   single `propose_*` API (controller decides which).

## Rationale

1. **Generic-over-F (Wave 4 of the multi-wave generic lift).** Wave 3 graduated `State<F>`
   and `HilbertState<F>` to STABLE. `AdaptivePI` was the last public type with an
   inherited f64 lock — Wave 4 removes it. f32 paths become possible (memory pressure on
   wasm/embedded), at the cost of relaxed tolerances driven by f32 ULP.

2. **F9 step-variance is the operational pain point.** The CEV oracle
   (`tests/cev_european_call.rs`, `tests/cev_european_call_sweep.rs`,
   `tests/cev_boundary_stress.rs`, `tests/cev_high_lam_oracle.rs`) accepts steps at
   widely varying τ when the Classical PI gains over-react to oscillatory error norms.
   Söderlind 2003 H211b applies a low-pass digital filter that smooths step-size
   trajectories with negligible loss of L² accuracy — measured 2–3× IQR(step) reduction
   on stiff CEV regimes in published benchmarks.

3. **Math-fidelity invariance.** §11.1.bis is NORMATIVE and stays untouched. The
   default controller is bit-identical to v1.0.0 behaviour — same gains, same safety,
   same clamping, same I-term seed. H211b is an opt-in plumbing layer documented in
   ADR-0044 + the contract; it is explicitly NOT a math.md amendment.

4. **Trait factoring is essentially free.** Monomorphisation eliminates dynamic dispatch
   at the `with_controller` boundary; binary size delta is bounded by the number of
   instantiated `(C, F, K)` triples (currently 1 in core, 2 with H211b opt-in).

5. **Composability with Wave 1/3.** Wave 1 `ScratchPool<F>` + Wave 3
   `HilbertState::dot`/`norm_sq` cleanly land the zero-alloc Richardson path. No new
   primitives are needed at the State layer.

## Math fidelity

- **§11.1.bis (NORMATIVE) unchanged.** `ClassicalPI<F>` reproduces the §11.1.bis gains
  `α = 0.7/p, β = 0.4/p`, the safety factor `0.9`, the clamp interval `[0.2 τ, 5.0 τ]`,
  and the I-term seed `err_prev = 1.0` on the first step.
- **H211b is ADVISORY plumbing.** Documented in this ADR + `contracts/v2/wave4-
  stepcontroller.md §4`. Not referenced by §11.bis. Not a NORMATIVE law.
- **Accepted-step trajectory invariance.** On the F9 oracle, the sequence of accepted
  `(τᵢ, t_curr_i)` pairs under `AdaptivePI::new(...)` (default controller) is required
  to be bit-identical to the v1.0.0 trace. A captured baseline trace in
  `tests/adaptive_classical_bit_equal.rs` enforces this.

## Acceptance criteria (v2.0 Wave 4 gate)

1. **All 18 NORMATIVE sympy gates re-pass.** No symbolic regression.
2. **All 6 NORMATIVE slope gates re-pass** (G1, G2, G3-1D/2D/⁴-2D/⁶-2D, G4_NS2D_aniso,
   G5_3D — verify exact list against `crates/semiflow-core/tests/`).
3. **SIMD bit-equality preserved.** AVX2/NEON paths untouched.
4. **F9 (CEV) oracle re-passes under `ClassicalPI` with bit-identical accepted-step
   trajectory.** Enforced by `tests/adaptive_classical_bit_equal.rs` proptest +
   baseline-trace fixture.
5. **F9 IQR(step) ≥ 2× reduction under `H211bFilter`** at `L²(state_final_h211b -
   reference) ≤ 1.05 × L²(state_final_classical - reference)`.
6. **Generic-over-F f32 smoke**: `tests/adaptive_generic_f32.rs` runs a 1D heat sweep
   with `AdaptivePI<DiffusionChernoff<f32>, f32>` and validates convergence at relaxed
   f32 tolerance (`tol_rel ≥ 1e-4`).
7. **Wave 1/2/3 regressions** — all current adaptive tests
   (`tests/adaptive_*.rs`) re-pass unchanged.
8. **Zero new direct deps** in semiflow-core (`Cargo.toml` direct-deps count stays at 2).
9. **No `unsafe_code`** introduced; `#![deny(unsafe_code)]` workspace lint unchanged.
10. **File-cap respected**: `adaptive.rs` ≤ 500 lines after expansion (carve-out 700
    available if monomorphisation pressures exceed; see LoC budget in contract §LoC).

## Out of scope (explicitly deferred)

- **Changing the default controller to H211b.** Default stays `ClassicalPI` per
  §11.1.bis. Opt-in only.
- **Mutating the §11.1.bis NORMATIVE gains** `α = 0.7/p, β = 0.4/p`. Touch math.md only
  via a future, properly-graduated ADR if ever needed.
- **PID controllers / DEADBEAT laws.** Not part of Wave 4 scope.
- **Per-component / vectorised tolerance.** Tolerance stays scalar (mixed abs/rel
  via `State::norm_sup` of `u_curr` and `u_full`).
- **Adaptive ORDER selection.** Wave 4 keeps `p = func.order()` static.
- **Promoting H211b into math.md.** Stays at ADR + contract scope only.
- **Removing the f64 short-cut for `growth()`.** `ChernoffFunction::growth() -> (f64,
  f64)` is unchanged per ADR-0025 §"Generics".

## Migration

- **f64 callers**: zero source change. `AdaptivePI::new(func)` returns
  `AdaptivePI<C, f64, ClassicalPI<f64>>` and behaves identically to v1.0.0.
- **f32 callers (NEW)**: `AdaptivePI::<_, f32>::new(func)` where
  `func: ChernoffFunction<f32>`.
- **H211b opt-in**: `AdaptivePI::new(func).with_controller(H211bFilter::default())`.
  Returns a typed `AdaptivePI<C, F, H211bFilter<F>>`.
- **Public field access** (`pi.alpha`, `pi.beta`) is preserved — they remain
  `f64` on `ClassicalPI<F>` for visual continuity with §11.1.bis, even when `F = f32`.
  The internal computation casts to `F` once per call. See contract §3 for the rule.

## Consequences

- **Positive**: F9 stiff regimes gain a low-pass step-size filter at zero math-fidelity
  cost. f32 path opens (wasm/embedded). Trait factoring opens future controller variants
  (digital filters, PID, predictive) without touching `adaptive.rs` core logic.
- **Negative**: `adaptive.rs` grows from 330 → ~480 lines (still ≤ 500 cap). Builder
  surface widens by one method. `AdaptivePI` becomes a triple-parameter generic — minor
  ergonomic cost mitigated by defaults.
- **Risk**: Bit-identity on the Classical path requires careful ordering of `safety *
  e * e_prev` (FP multiplication associates differently than `safety * e_prev * e`).
  Enforced by the bit-equal proptest.

## References

- Söderlind, G. (2002). "Automatic control and adaptive time-stepping." *Numer.
  Algorithms* **31**, 281–310. — Source of the PI.4.7 gains `α = 0.7/p, β = 0.4/p`.
- Söderlind, G. (2003). "Digital filters in adaptive time-stepping." *ACM TOMS* **29**,
  1–26. — Source of the H211b filter (`b = 1/4`, `c = 1/2`).
- HLW = Hairer, Lubich, Wanner, *Geometric Numerical Integration* (2nd ed.), §IV.2.
- `contracts/semiflow-core.math.md §11.1.bis` — NORMATIVE axis-distinction clarification.
- ADR-0041, ADR-0042, ADR-0043 — Wave 1/2/3 foundations.
