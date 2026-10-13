# ADR-0215 — τ-keyed step plans for the grid engines

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: amends ADR-0204 (prepared sampling) — additive trait
  hook; every result bit-identical.
- **Contract**: math §71 (`contracts/semiflow-core.math-step-plans.md`); gates
  `G_STEP_PLAN_BIT_EQUAL`, `G_STEP_PLAN_USED`, `G_STEP_PLAN_ZERO_ALLOC`.
- **Amendment 1 (review)**: plans only for `≥ 2` uses, engine-state words, exact
  address bits, length checks, parallel fill, 32 MiB cap, error order (below).

## Context

After ADR-0204 one `DiffusionChernoff` step still recomputes, for every node and
every step, what depends only on `(τ, grid, coefficients)`: the coefficient calls
(constant `a`: 1; variable `a`: 7 `a′`, 2 `a`, 1 `a″` — each a Python callback in the
bindings), two or three square roots, and for each of the 5 (constant `a`) or 10
(γ-A baseline + ζ-A correction) samples the cell lookup (two divisions, a floor) and
the Hermite weight polynomials (septic ≈ 70 flops, octonic ≈ 100). Fixed-τ loops
apply the same engine with the same `τ` over and over: the `n` steps of `Evolver`
and `ChernoffSemigroup`, and — inside one Strang step — every row of an axis sweep
(`400` rows of a `400²` grid, `64²` pencils per axis of a `64³` grid).

## Decision

1. **Trait hook** (provided methods of `ChernoffFunction`, dyn-compatible,
   semver-additive):
   `plan_len(τ, proto) → Option<(data, work)>` (default `None`),
   `fill_plan(τ, proto, data)`, `apply_planned_into(τ, data, work, src, dst, scratch)`
   (default `apply_into`). The caller owns the buffers (scratch pool), so a planned
   steady state allocates nothing; the plan's work area replaces the per-step
   ghost-table buffer.
2. **Plan contents** (`DiffusionChernoff<f64>`, all three storages, and
   `ShiftChernoff1D<f64>`: `τ·c(x)` and its three samples per node): a header — tag,
   `τ`, state length, the engine's address (as a bit pattern, lossless on tagged
   pointers) and the sampling grid's bit pattern (bounds, `n`, boundary policy with
   payload, interpolant) — then the engine's own state (its grid, and the identity
   of its coefficients: fn-pointer or closure addresses, or the constant `a`; for
   `ShiftChernoff1D` also the public `a`, `b`, `c`), then per node the
   planned samples `[table offset | −1, x, weights]` (`plan_sample`: the cell, the
   ghost-table coverage and the interpolant's weights that
   `PreparedGridFn::sample` computes; Catmull–Rom stores `s`, linear and
   Chebyshev store only `x`) and, for variable `a`, the ζ scalars `2δ, δ², 2δ³,
   a·a′, a·a″/2, a′·a″/4` and `τ²`, each by the kernel's own expression. A planned
   step fills the ghost table into the work area and combines the stored weights
   with it in the kernel's order: **bit-identical by construction**. A plan
   presented with another `τ`, sampling grid, length, engine state, or a buffer of
   the wrong length, is ignored (the engine runs `apply_into`), so a plan can only
   make a step faster. Plans above `2²²` doubles (32 MiB) or whose size overflows
   are not built. Building validates `τ` and `a` like `apply_into` and probes the
   interpolant right after node 0's coefficient check, so it raises the error
   `apply_into` raises first (in node order; with threaded node loops both report
   whichever thread fails first). Under the `parallel` feature the per-node fill is
   threaded like `apply_into`.
3. **Consumers** build a plan only when it is used at least twice (a single
   application would pay the plan's cost for nothing). `Evolver::evolve_into`,
   `ChernoffSemigroup::evolve` and the Richardson levels (one plan for all `n` steps), `AxisLift`/`AxisLift3D` sweeps
   (one plan per sweep for all pencils), the threaded Strang 2-D/3-D sweeps (one
   shared read-only plan, a work area per thread; the unplanned path is unchanged),
   `ReactionDiffusion` (one plan per species engine per evolution) and the Python
   chunked 1-D evolution.
4. **Scratch pool: best fit.** `ScratchPool::take_vec` now hands out the SMALLEST
   free buffer that fits (was: the first). With first fit a large plan buffer could
   serve a small request, so the next large request allocated in every call; best
   fit keeps a fixed request sequence allocation-free after one warm-up. Buffer
   contents never depend on the choice (every taken buffer is zero-filled).

## Consequences

`perf_suite` (release; best of 5, 2-D/3-D best of 1), two sessions, each pairing a
run without and with the plans. The host is shared and noisy (rows the plans do
not touch moved by up to 2× between sessions), so ranges are reported:

| workload | session A before → after | session B before → after | speed-up |
|---|---:|---:|---:|
| heat1d `ShiftChernoff1D`, N = 1000, n = 100 | — | 11.0 → 3.5 ms | 3.1× |
| heat1d `DiffusionChernoff` const-a | 18.9 → 5.0 ms | 17.8 → 5.6 ms | 3.2–3.8× |
| heat1d fn-ptr `a ≡ ½` | 30.1 → 8.8 ms | 36.2 → 13.7 ms | 2.6–3.4× |
| heat1d variable-`a` closure | 40.1 → 8.9 ms | 52.0 → 9.5 ms | 4.5–5.5× |
| heat2d `Strang2D` 400², n = 10 | 1 799 → 596 ms | 1 937 → 910 ms | 2.1–3.0× |
| heat3d `Strang3D` 64³, n = 10 | 5 550 → 2 534 ms | 6 104 → 3 822 ms | 1.6–2.2× |

Deterministic: the coefficient functions (Python callbacks in the bindings) are
evaluated once per plan — once per `evolve`, per Strang sweep or per Python chunk —
instead of once per step and pencil.

## Honest limits

- Producers: `DiffusionChernoff<f64>` and `ShiftChernoff1D<f64>`. Other engines
  (`Diffusion4th/6th`, drift–reaction, `f32`/`Dual`) keep `plan_len = None`
  (unchanged behaviour); wrappers that do not forward the hook run unplanned.
- `ChernoffSemigroup::evolve` drops its plan with its local pool (≤ 3
  allocations per call, was ≤ 4); the plan is allocated (zero-filled) on every
  call, while `Evolver` keeps it in the caller's `ScratchPool`.
- Memory: 50 (constant `a`), 106 (variable `a`, septic) or 126 (octonic) doubles
  per node, against 4–5 for the ghost table it replaces; capped at 32 MiB.
- A plan belongs to the engine value it was filled from: a dropped engine's plan
  presented to a NEW engine at the same address with the same coefficients and
  grid is used — harmlessly, since it computes the same step.
- Short pencils (`64³`) are bound by the per-pencil ghost-table fill
  (`MARGIN = 24` cells each side), which the plan does not remove.
- The address check identifies the engine only while it is alive; a plan kept
  past its engine's lifetime and presented to a new engine at the same address
  is the caller's error (crate consumers build plans per call).

## Gates

- `G_STEP_PLAN_BIT_EQUAL` (`tests/g_step_plan_bit_equal.rs`,
  `tests/g_step_plan_consumers.rs`): planned vs forced
  unplanned (`NoPlan` wrapper), 0 ULP, three `DiffusionChernoff` storages and
  `ShiftChernoff1D` (drift and reaction) × five interpolants × eight
  boundary policies × `n ∈ {8, 37, 128}` × `τ ∈ {1e−4, 5}` through `Evolver` and
  `ChernoffSemigroup`; `Strang2D`/`Strang3D` serial (`24²`/`8³`, every policy) and
  threaded (`64²`/`32³`, every policy × septic, Catmull–Rom, linear); fallbacks for
  another `τ`, another engine of the same kind and each sampling-grid field; the
  error `apply_into` reports first (unsupported interpolant before an invalid `a`);
  Richardson levels (base `n ∈ {1, 2}`, levels 1–4) and `ReactionDiffusion` (with
  and without cross-diffusion, plain and extrapolated); with `parallel`, the
  threaded 1-D fill and apply (4 threads forced, `n = 8192`, incl. an error).
- `G_STEP_PLAN_USED` (`tests/g_step_plan_used.rs`): the planned branch runs — a
  perturbed stored weight changes the planned result (every producer), and the
  coefficient calls of `Evolver`, `ChernoffSemigroup`, `ReactionDiffusion` and the
  Strang sweeps (serial and threaded) do not grow with the step or pencil count.
  Crate unit tests show each identity word matters (a forged header gives other
  bits) and that a changed engine field or reassigned engine is detected.
- `G_STEP_PLAN_ZERO_ALLOC`: a planned `Evolver::evolve_into` allocates per call
  only its two ping-pong states (the same count for `n = 2` and `n = 50`, ≤ 2 with a
  warm pool); a planned `Strang2D` step allocates nothing in steady state;
  `ChernoffSemigroup::evolve` ≤ 3 allocations per call.
