# ADR-0070 — Howland-Lifted Nonautonomous Chernoff (B2)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v2.7 (second math pillar; additive minor)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction` trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0051 (Magnus graph variable-time — graph-side analog of nonautonomous lift), ADR-0063 (var-coef time-dep graph), ADR-0068 (v2.6 infrastructure).
- **Supersedes / amends**: none — strictly additive on the public surface. The new super-trait `TimedChernoffFunction<F>` is *additive over* `ChernoffFunction<F>`; existing impls are unaffected.
- **Mathematical foundation**: math.md §23 (NORMATIVE library — `HowlandLift` semantics; CITATION Howland 1974 *Trans. AMS* 207 for the augmented-generator construction; Magnus 1954 / Casas-Iserles 2009 for higher-order time integration that v3.x will revisit).
- **Acceptance gates added**: G25 (RELEASE_BLOCKING slope on time-dependent heat), T20N (NORMATIVE sympy — augmented-generator commutator identity).

## Context

The continuous-space analog of `MagnusGraphHeatChernoff` (v2.1, ADR-0051) for **time-dependent operators** `L(s)` on a Hilbert space `X` is the **Howland lift**: augment `X` to `Y := L²([0,T], X)` and define the *time-extended* operator

```
L̂ := -∂_s + L(s)   on Y
```

Howland (1974, *Trans. AMS* 207) proved that `L̂` generates a C₀-semigroup `Û(t)` on `Y` and that the time-evolution `U(t₂, t₁) f₀` of the original Cauchy problem `∂_t u = L(t) u`, `u(t₁) = f₀` is recovered from `Û` via the **shift identity**:

```
[Û(τ) f̂](s)  =  U(s, s − τ) f̂(s − τ)         (Howland 1974 Theorem 1)
```

The Chernoff approximation of `Û` becomes

```
F̂(τ) f̂(s)  :=  F(τ, s − τ) f̂(s − τ)         (left-endpoint sampling)
```

where `F(τ, t)` is a *time-parameterised* Chernoff function for `L(t)` evaluated at time `t` (the left endpoint of each Chernoff step). The result is a generic, additive lift from any "time-aware" Chernoff function to the nonautonomous semigroup — the continuous-space counterpart of the Magnus-graph machinery from v2.1.

The current `ChernoffFunction<F>` trait has no notion of an "outer time coordinate" — `apply_into(τ, src, dst)` carries only the *step* `τ`, not the absolute time `t`. Existing impls are correct for autonomous operators (constant `L`); they cannot represent `L(s)` directly without smuggling state through interior mutability or per-construction binding.

v2.7 ships an **additive super-trait** `TimedChernoffFunction<F>: ChernoffFunction<F>` with the new method `apply_at(t, τ, src, dst, scratch)`, plus a wrapper `HowlandLift<C: TimedChernoffFunction<F>, F>` that consumes any timed Chernoff function and evolves on a discretized `L²([0,T], X) ≅ Vec<C::S>`.

This is **scoped** for v2.7: order-1 only (the shift in time gives order-1 unless higher-order Magnus is used on the augmented operator — see §"Limitations"). No per-time-sample spatial interpolation in v2.7 (uniform grid in `s`; defer adaptive-`s` to v3.0).

## Decision

Ship two additive public-surface items in v2.7:

- **`pub trait TimedChernoffFunction<F: SemiflowFloat = f64>: ChernoffFunction<F>`** — additive super-trait. Single new method:
  ```rust
  fn apply_at(
      &self,
      t: F,
      tau: F,
      src: &Self::S,
      dst: &mut Self::S,
      scratch: &mut ScratchPool<F>,
  ) -> Result<(), SemiflowError>;
  ```
  Semantics: zero-alloc apply at *absolute time* `t` with *step* `tau`. The implementor's frozen `L(t)` is sampled at `t`; the step `tau` is applied. The default impl is a **no-op bridge** to the supertrait's `apply_into(tau, src, dst, scratch)` — this allows autonomous impls (`ShiftChernoff1D`, `DiffusionChernoff`, `Strang2D<...>`, etc.) to declare `impl TimedChernoffFunction<F> for _ {}` with zero method body when their generator does not depend on `t`. Time-dependent impls (the user's `TimedDiffusionChernoff` adapter for `L(s) = a(s)·∂²`) override `apply_at` to consult `t`.

- **`pub struct HowlandLift<C: TimedChernoffFunction<F>, F: SemiflowFloat = f64>`** — wrapper. Constructor takes the inner `C`, the time horizon `T: F`, and the number of time samples `n_t: usize` (`n_t ≥ 2`). Implements `ChernoffFunction<F>` with `Self::S = HowlandState<C::S, F>` (new public state type — see traits.yaml). `apply_into(τ, src, dst, scratch)` per the §23 algorithm: for each time-sample index `i ∈ [0, n_t)`, dispatch
  ```
  if i == 0:  dst[0] := zero (boundary convention f̂(s) = 0 for s < 0)
  else:       inner.apply_at(t_i, Δs, &src[i-1], &mut dst[i], scratch)
  ```
  where `Δs = T / (n_t - 1)` and `t_i = i · Δs`. **Step `τ` MUST equal `Δs`** (the shift in `s` matches the integration step; rejected at runtime with `DomainViolation` otherwise — see math.md §23.4).
  `order()` returns `min(C::order(), 1)` (Howland shift is order-1 unless higher-order Magnus is layered on top). `growth()` returns `(M_c · exp(T · |ω_c|), 0)` where `(M_c, ω_c) = inner.growth()` (the time-shift is unitary on `L²([0,T])`; the inner growth integrates over the full horizon).

- **`pub struct HowlandState<S: State<F>, F: SemiflowFloat = f64>`** — new public state type backing the discretized `L²([0,T], X)`. Internally `Vec<S>` of length `n_t` (time samples). Implements `State<F>` via component-wise dispatch (`axpy` over all `n_t` samples; `norm_sup` is the max across all samples).

File layout: `crates/semiflow-core/src/howland.rs` (~400 LoC budget, default 500-LoC cap with 100-LoC headroom — no constitution carve-out needed). Module added to `traits.yaml` `modules:` list. Schema bumps included in the v2.7 batch (properties.yaml 0.8.0 → 0.9.0 per ADR-0069 — G25 + T20N gates added at the same bump; traits.yaml 0.6.0 → 0.7.0).

## Rationale

- **Why a super-trait, not a refactor of `ChernoffFunction`?** Refactoring `ChernoffFunction::apply_into` to take `t: F` would break every existing impl (ShiftChernoff1D, DiffusionChernoff, Strang2D, NS2D, etc. — ~14 types touched). It would also be wasteful: most impls represent autonomous operators and would ignore the new parameter. The additive super-trait pattern (a) preserves backward compatibility (every existing impl trivially satisfies `TimedChernoffFunction` via the default bridge), (b) lets time-dependent impls add `apply_at` opt-in, and (c) marks the v2.7 surface as candidate for **folding** into the v3.0 trait redesign once A1/B2/A4/B4 use cases have crystallised (see roadmap-reflective-biscuit.md §v3.0).
- **Why default-bridge `apply_at` to `apply_into`?** Existing autonomous impls (`L` does not depend on `t`) MUST implement `TimedChernoffFunction` for free — otherwise `HowlandLift<DiffusionChernoff, _>` cannot construct, which would force the user to write a trivial wrapper. The default `fn apply_at(_, t: F, tau: F, ...) { self.apply_into(tau, src, dst, scratch) }` ignores `t` and delegates. Time-dependent impls override.
- **Why `Vec<C::S>` for `HowlandState`, not contiguous flat `Vec<F>`?** The inner state `C::S` can be any `State<F>` impl: `GridFn1D`, `GridFn2D`, `GridFn3D`, `GraphSignal`. Forcing a flat layout would require a trait method `S::flatten_into(&mut [F])` and the inverse — premature abstraction. `Vec<S>` is the suckless choice: each time sample is a full inner state, addressed by index. Memory cost: `n_t` × `sizeof(S)` for the state vector. For the G25 test (`n_t ∈ {32, 64, 128, 256}`, `S = GridFn1D` at `N = 256`) the storage is ≤ 256 × 256 × 8 B = 512 KiB — well within budget.
- **Why uniform `Δs` grid (no adaptive `s`)?** v2.7 ships the order-1 version. Adaptive `s` (refining samples where `L(s)` varies fast) requires either (a) per-sample interpolation between time grid points or (b) a `StepController`-style adaptive control on the `s`-axis. Both are scope creep for v2.7. Defer to v3.x once the v2.7 use cases (Heston time-dependent vol) have informed the API.
- **Why `apply_at(t, τ, ...)` with `t` as the LEFT endpoint of the step?** Howland 1974 §2 derives the lifted-semigroup formula with the left-endpoint sampling convention: `F̂(τ) f̂(s) = F(τ, s − τ) f̂(s − τ)`. The right-endpoint convention `F(τ, s)` gives an equivalent O(τ) approximation but biases the shift formula. We adopt Howland's left-endpoint convention verbatim — math.md §23.2 derives the choice.
- **Why reject `τ ≠ Δs` at runtime?** The shift formula `[F̂(τ) f̂](s) = F(τ, s−τ) f̂(s−τ)` requires the time-shift to match the integration step exactly: if `τ ≠ Δs`, the sampling `f̂(s − τ)` falls between grid points and requires `s`-axis interpolation (out-of-scope for v2.7 — see preceding rationale point). Returning `DomainViolation` is the contract-honest path. The matched-step requirement is documented in math.md §23.4 and asserted in `HowlandLift::apply_into`.
- **Why `boundary convention f̂(s) = 0 for s < 0`?** This is the initial-condition convention: the time-extended state encodes the trajectory `u(s)` for `s ∈ [0, T]`, and `s < 0` is "before initial time" with no defined value. The shift `f̂(s − τ)` at `s = 0` would read `f̂(−τ)` which is undefined; we set it to zero. Equivalent to the absorbing-boundary convention in killing-functional terms.
- **Why order-1 globally?** The Howland shift `[Û(τ)f̂](s) = U(s, s−τ)f̂(s−τ)` is first-order accurate in `τ`: the Taylor expansion of `U(s, s−τ)` at `τ = 0` gives `I + τ L(s) + O(τ²)`, but the left-endpoint sampling of `L` (i.e., evaluating at `s − τ` rather than the midpoint `s − τ/2`) introduces an `O(τ²)` error term in the generator — capping the global rate at order 1. Higher-order Howland is *constructable* (Magnus on the augmented operator gives order-2 via midpoint sampling, order-4 via Casas-Iserles palindromic) but requires either (a) a midpoint `apply_at(t_mid, τ, ...)` variant or (b) the v3.0 `ApproxSubspace<4>` machinery from the roadmap. Ship order-1 now via the simple left-endpoint formula; ladder up in v3.x.
- **Why a Howland-specific G25 oracle (`a(s) = 1 + 0.5·s`)?** A linear-in-`s` diffusion coefficient admits a closed-form solution via Green's function on the time-integrated diffusivity `A(t) = ∫₀^t a(s) ds = t + 0.25·t²` (heat-kernel-with-time-dependent-conductivity, standard exercise). This gives an exact oracle for slope-testing the order-1 convergence without depending on a Chernoff sub-oracle — clean separation between the Howland-shift accuracy and the inner Chernoff accuracy.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Modify `ChernoffFunction::apply_into` to take `t: F` parameter | API break across 14 existing impls; most don't need `t`. Forces v3.0 trait redesign one release early — premature. |
| Drop the super-trait; require users to write per-call `LaplacianAtTime`-style closures | The Magnus-graph approach (ADR-0051) uses a closure `Box<dyn Fn(F) -> Arc<Laplacian<F>>>` because the graph's Laplacian is a *value*. Continuous-space `L(t)` is an *operator*; representing it as a closure of state functions is awkward and inflexible. A trait method is the natural design. |
| Higher-order Howland (Magnus-2 / Casas-Iserles-4) in v2.7 | Requires midpoint `apply_at(t_mid, τ, ...)` AND a higher-order integrator. Two-step PR. Defer to v3.x once the order-1 baseline is shipped and benchmarked. |
| Adaptive time-sample grid (non-uniform `Δs`) | Requires `s`-axis interpolation between grid points OR adaptive `StepController`-on-`s`. Premature for v2.7; the use cases (Heston time-dependent vol) tolerate uniform sampling. |
| Continuous-space Magnus directly (no Howland lift) | Requires evaluating `[L(s), L(s')]` commutators on a continuous spatial domain — needs `ApproxSubspace<k>` Chernoff that lives in v3.0. Howland sidesteps the commutator by lifting to L²([0,T], X) where the augmented `L̂ = -∂_s + L(s)` is again autonomous. |
| Make `HowlandState` private; expose only `Vec<C::S>` | The user constructs the initial state and reads the final state — both need a typed handle. `HowlandState` provides the `State<F>` impl needed by `ChernoffSemigroup::evolve` and a clean `from_initial_condition(f0: C::S, n_t)` constructor. |
| Default `apply_at` to *error* (not delegate to `apply_into`) | Forces every existing impl to either (a) explicitly implement `TimedChernoffFunction` with a boilerplate body or (b) opt out. The default-bridge pattern is the suckless choice: autonomous impls work for free. |
| Reject `τ ≠ Δs` at construction time instead of runtime | Construction takes `T` and `n_t`; `τ` is determined per-call by the outer `ChernoffSemigroup`. The check belongs at the per-call boundary (`apply_into`). |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface. The super-trait `TimedChernoffFunction<F>` is opt-in; no existing impl is touched.
- **New module `crates/semiflow-core/src/howland.rs`** (~400 LoC budget, default 500-LoC cap with 100-LoC headroom). No constitution amendment needed.
- **Existing autonomous impls** (`ShiftChernoff1D`, `DiffusionChernoff`, `DiffusionChernoff4`, `Strang2D`, `Strang3D`, etc.) gain a one-line `impl TimedChernoffFunction<F> for X<F> {}` (default-bridged). Engineer wave applies these mechanically. **No method body changes.**
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm`).
- **Schema bumps** (combined with ADR-0069): `properties.yaml` 0.8.0 → 0.9.0; `traits.yaml` 0.6.0 → 0.7.0. math.md is append-only (§23 NEW).
- **New gates**: G25 (RELEASE_BLOCKING — Howland slope ≤ −0.95 on time-dependent heat `a(s) = 1 + 0.5s`); T20N (NORMATIVE sympy — augmented-generator commutator identity, verifies left-endpoint shift convention).
- **No L-gate for Howland in v2.7.** The HFT use case (Heston time-dependent vol) shares L_HESTON_PTICK with the broader Heston example. The Howland lift adds enough per-tick complexity (n_t-sample shift) that a separate L_HOWLAND gate would be premature — defer to v2.8 if benchmark evidence warrants.
- **CITATIONs added to math.md §23**: Howland 1974 *Trans. AMS* 207 (augmented-generator lift); Magnus 1954 *Comm. Pure Appl. Math.* 7 (cited as the higher-order ladder target for v3.x); Casas-Iserles 2009 *J. Comput. Phys.* (palindromic Magnus, cited for the deferral rationale).

## Migration

None for end-users. The new super-trait is opt-in; existing code compiles and runs unchanged. Users who want nonautonomous evolution implement `TimedChernoffFunction` on their custom Chernoff impl (or use one of the engineer-wave-provided demos in `examples/`).

The G25 oracle (time-dependent heat with `a(s) = 1 + 0.5·s`) is a new test only; no migration burden.

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `TimedChernoffFunction<F>`, `HowlandLift<C, F>`, `HowlandState<S, F>`.
- ADR-0026 — `ChernoffFunction<F>` super-trait; `TimedChernoffFunction<F>` extends it additively.
- ADR-0041 — `apply_into` + `ScratchPool`; `apply_at` mirrors this contract signature.
- ADR-0051 — Magnus-graph variable-time; the graph-side analog (discrete time-dependent operator). Howland lifts continuous-space `L(s)` to `Y = L²([0,T], X)`; ADR-0051 lifts discrete-graph `L_G(s)` via Magnus-K=4 directly. Different math, same problem class.
- ADR-0063 — var-coef time-dep graph; further graph-side time-dependent infrastructure.
- ADR-0068 — v2.6 infrastructure baseline.
- ADR-0069 — Laplace-Chernoff resolvent (v2.7 companion ADR; shared release window).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v2.7 — release-level roadmap.
- math.md §23 (NEW v2.7) — Howland-lifted Chernoff normative spec.
- math.md §16, §20 — Magnus-K=4 and Magnus-K=6 (cited as the higher-order analog for v3.x ladder).

## Amendments

(none at acceptance time)
