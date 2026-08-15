# ADR-0193 — Adaptive stepping and coefficient schedules over variable-coefficient `Shift1D`

- **Status**: Proposed (Issues #22, #23; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: none — purely ADDITIVE. No existing signature,
  behaviour or gate changes; `evolve_with_time_schedule` is untouched.
- **Contract**: no new math. Both items are binding-layer entry points onto
  `ShiftChernoff1D` and `semiflow::AdaptivePI`, which already carry their own
  normative sections (math.md §1–§2 formula (6); ADR-0014/§11.1.bis for the PI
  controller). Recorded here because they widen the *public Python surface*.

## Context

Two adjacent gaps, both of the same shape: the core supports the case, the
binding does not expose it.

**#22.** `AdaptivePI`'s `kernel=` menu reaches only constant-coefficient
kernels. Its `"shift"` arm hard-codes `a = 0.5, b = 0, c = 0`
(`crates/semiflow-py/src/adaptive.rs:249`) and uses the fn-pointer constructor,
so a Black-Scholes-type generator `½σ²S²·u_SS + rS·u_S − r·u` had no adaptive
path at all — `n_steps` was hand-tuned per (grid, maturity, vol) triple, by
bisection. This bites hardest exactly where it is least affordable: the shift
kernel is order 1, so the accuracy/steps trade-off is steep.

**#23.** `Shift1D.evolve_with_time_schedule` takes a **scalar** schedule for `a`
only, with `b` and `c` fixed for the whole run. Time-dependent Feynman–Kac
killing `c(x, t)` — the core of optimal-execution policy evaluation, e.g.
`∂_τ u = ½σ²·u_pp − γν(t)(p − ην(t))·u` with killing linear in `p` and a
time-varying slope — was therefore inexpressible, as was any space-varying
coefficient inside a schedule (schedules and `with_arrays` were mutually
exclusive). The workaround was M frozen-coefficient macro-segments, each
constructing a fresh `Shift1D` and re-sampling every coefficient array, with the
state round-tripping through numpy between segments.

## Decision

**1. `AdaptivePI.with_arrays(...)` rather than a new `kernel=` string.** The
existing `AdaptiveVariant::Shift` arm already holds
`AdaptivePI<ShiftChernoff1D<f64>>`, and `ShiftChernoff1D::with_closure` produces
*the same type* as the hard-coded `::new` path — so this is a constructor, not
new machinery. No enum arm, no controller change, no error-path change. A
`kernel="shift_arrays"` string was rejected: the coefficient arrays have to
arrive somehow, and threading them through a constructor whose signature is
shaped for the constant-coefficient menu would make every other arm carry three
unused parameters.

**2. `evolve_with_coefficient_schedule` as a sibling, not a widening of
`evolve_with_time_schedule`.** The existing method's `a_schedule` is typed as a
float array and its `b`/`c` are scalars; accepting per-segment
scalar-or-array for all three cannot be layered onto that signature without
changing what existing calls mean. The new entry point takes three independent
per-segment sequences, each entry either a float or a length-`n` array.
`a_schedule` defines `n_segments` and the others must match it.

**3. The new path leaves the object consistent; the old one is left alone.**
`evolve_with_time_schedule` updates only `inner.current.values`, leaving
`inner.semigroup.func` on the construction-time coefficients — so a subsequent
`evolve()` silently reverts to the original `a`. That is surprising, but it is
existing behaviour and changing it would alter results for current callers, so
it is documented rather than fixed. The new method rebuilds the semigroup from
the final segment, and a test pins the difference: after a schedule ending at
`a = 2.0`, a follow-up `evolve` matches a fresh kernel at `a = 2.0` to 1e-12 and
demonstrably does *not* match one at the construction-time `a = 0.5`.

**4. The new path ping-pongs instead of cloning.** The scalar-`a` walk allocates
a fresh state clone on every step (`let mut next = state.clone()`); the new one
uses two buffers and `mem::swap`, matching every other evolve loop in the crate.

**5. Corrected an inaccurate documented error kind.** `AdaptivePI`'s rustdoc and
`.pyi` both advertised `kind='CflViolated'` when `max_substeps` is exceeded. The
core returns `AdaptiveStepRejected` from that path
(`crates/semiflow/src/adaptive.rs:298`), which `classify_core_error` maps to
`'ConvergenceFailed'`. `CflViolated` is emitted by `truncated_exp`,
`graph_var_coef` and `nonseparable_mixed`, never by `AdaptivePI`. A caller
following the documentation would have caught the wrong exception kind.

## Rationale

Both items are cases where the honest amount of new code is small and the
temptation is to build more than that. The adaptive item in particular looked
like it needed a new integrator variant until the existing enum turned out to
already hold the right type — the constraint was the *constructor*, not the
machinery. Reusing `closure_from_array` and the existing PI controller keeps the
new surface to two entry points with no new numerical claims to gate.

## Consequences

- `AdaptivePI` gains `with_arrays`; `Shift1D` gains
  `evolve_with_coefficient_schedule`. Both PyO3-only, inheriting the existing
  asymmetry (neither `AdaptivePI` nor the schedule methods exist in FFI or WASM).
- New module `crates/semiflow-py/src/shift1d_schedule_py.rs` — `shift1d_py.rs`
  was at 404/500 lines and the parsing logic does not fit.
- `.pyi` stubs updated, including the corrected error kind and a pointer from
  the old schedule method to the new one.

## Honest limits

- Schedules remain **piecewise constant in time**. Genuine joint `a(x, t)` is
  still out of scope — core coefficient closures are purely spatial, and this
  ADR does not change that. What is new is that each piece may now vary in
  space, and that `b` and `c` get pieces at all.
- `with_arrays` inherits the shift kernel's order 1. Adaptivity chooses the step
  size; it does not raise the order, and the PI controller's gains are scheduled
  from `order()` — so tightening `tol_*` buys accuracy at a first-order rate.
- Coefficient arrays are sampled once at construction and interpolated with
  Catmull-Rom, exactly as `Shift1D.with_arrays` does. A coefficient with a jump
  between nodes is smoothed by that interpolant; the conservative
  divergence-form path (§56) is the tool for genuine material interfaces.
- No new gate. Both items are surface, not mathematics: the numerical paths they
  reach are the ones `G_SHIFT1D_*` and the AdaptivePI gates already cover. The
  tests added are behavioural (coefficients are live, tolerance converges,
  schedule lengths validate, the object stays consistent), not accuracy gates,
  and they are deliberately labelled as such.
