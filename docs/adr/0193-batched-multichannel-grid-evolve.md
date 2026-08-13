# ADR-0193 — Batched multi-channel evolve for 1-D grid kernels

- **Status**: Proposed (Issue #19; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: none — purely ADDITIVE. Extends the ADR-0184 batched
  contract from graph kernels to the 1-D grid family; no existing signature,
  layout or gate changes.
- **Contract**: `contracts/semiflow-core.properties.yaml` gate
  `G_GRID1D_BATCH_ULP`. **No new math.md section** — batching is an identity
  transform on each channel, not a numerical method. That is the point of the
  0-ULP gate, and saying so here is deliberate: a `§N` section would imply a
  numerical claim that does not exist.

## Context

ADR-0184 gave the **graph** kernels `evolve_batched`; the **grid** 1-D family
(`ShiftChernoff1D`, `DiffusionChernoff`, `Diffusion4thChernoff`,
`DriftReactionChernoff`, …) still evolved one state per object. Three workloads
from the issue degrade to Python loops as a result: pricing a strike strip is
11–50 independent solves under the *same* generator with only `u0` differing;
bump Greeks are ±bump re-solves that could amortise the coefficient setup; and a
Fokker–Planck backtest evolves ~250 density anchors under one local-vol
operator, paying object construction and a GIL round-trip per anchor.

## Decision

**1. Generic over the kernel, not specialised to `ShiftChernoff1D`.**
`evolve_batched_1d<C, F>` is bounded by
`C: ChernoffFunction<F, S = GridFn1D<F>>`, which covers the whole 1-D family in
one function. Specialising would have meant one copy per kernel type for no gain
— the batching logic touches nothing kernel-specific.

**2. `[C, N]` channel-major in Rust, `[N, C]` in Python.** Identical to
ADR-0184 D1, and for the same reason: contiguity per channel is what lets
`chunks_mut(n)` hand disjoint output slices to workers with no synchronisation.
The transpose is dissolved into the GIL-boundary copy that a numpy round-trip
requires anyway, reusing `gather_nc_to_cn` / `scatter_cn_to_nc` rather than
copying them.

**3. Channel-parallel and node-parallel are mutually exclusive, not nested.**
`parallel1d::parallel_eval_into` already threads over *nodes* inside
`apply_into`. Naive channel-parallelism would give `C × T` threads. The dispatch
rule is a pure function of `(C, N)`:

```
channel-parallel  iff  n_cols >= 2  AND  n < 2 * min_points_per_thread()
```

`resolve_threads_1d` returns 1 below that same threshold, so the two regimes are
*exactly complementary*: no `(C, N)` engages both, and none is left with neither
axis available. Large `N` is already saturated node-wise; small `N` is precisely
where node parallelism declines and channels are the only axis left — and it is
the regime the motivating workloads live in (`n = 513…801`).

**4. Belt-and-braces: workers pin their own node parallelism to 1.**
`FORCE_THREADS_1D` is a thread-local, so a spawned worker gets a fresh slot and
`pin_single_thread_1d()` cannot disturb the caller — including a test harness
that set it for its own thread. This makes oversubscription structurally
impossible even if the threshold in rule 3 is later retuned, rather than relying
on the two rules staying in sync.

**5. 0-ULP, asserted on bit patterns.** Every channel runs the identical
`apply_into` sequence at the identical `tau`, and there is no cross-channel
reduction, so bit-equality with `C` sequential solves is achievable by
construction (ADR-0184 D5). `G_GRID1D_BATCH_ULP` compares `f64::to_bits()`, not
a tolerance, across `n ∈ {64, 512, 4096, 8192}` — straddling the dispatch
boundary in both directions — × `n_cols ∈ {1, 2, 3, 5, 8}` × `n_steps ∈ {1, 7}`,
in both the serial and parallel builds.

A companion test perturbs one channel's input and asserts that channel changed
and no other did. Bit-equality against a sequential reference would *not* catch
shared-buffer cross-talk if both paths were wrong in the same way; this does.

## Rationale

The whole item is a transposition of an existing, gated design onto a sibling
state type — so the interesting decisions are the two the graph path did not
have to make: what to do about the pre-existing node-level parallelism, and
whether to specialise per kernel. Making the two parallel regimes complementary
by *construction* rather than by tuning is what keeps this from becoming a
performance-tuning surface with its own failure modes.

## Consequences

- New `crates/semiflow/src/grid_batched.rs`; `pub mod` appended to an existing
  line in `lib.rs`, which is at the 500-line cap.
- `parallel1d` gains `pin_single_thread_1d()` (`pub(crate)`).
- PyO3: `Shift1D.evolve_batched(t, u0_nc, n_steps=100)`, functional (does not
  mutate the object). PyO3-only, inheriting the ADR-0184 asymmetry — no batched
  surface exists in FFI or WASM for the graph path either.

## Honest limits

- **Batching is throughput, not accuracy.** Results are bit-identical to `C`
  sequential solves, so no per-channel error estimate improves and no
  convergence property changes. The gate exists to keep that true.
- **No speedup is claimed here.** Channel-parallelism helps only below the node
  threshold; above it the serial path runs and node parallelism does the work.
  Neither regime is measured by this ADR, and the README's standing position on
  wallclock applies unchanged.
- **The `[2048, 4096)` band is served by neither axis** — node parallelism needs
  `n ≥ 2 × 2048` to engage and channel parallelism defers to it from `n ≥ 4096`.
  That dead zone is inherited from the ADR-0036 threshold, not introduced here,
  but it is real and worth naming.
- **f32 batching is not exposed at the Python boundary**, mirroring
  `reject_f32_for_batched` on the graph classes. The core function is generic
  over `F`; only the binding is f64.
- **Coefficients are shared across channels by construction.** This batches over
  initial conditions, not over generators. A strike strip and bump-Greeks in `u0`
  fit; bumping a *coefficient* (vega, rho) still needs one object per bump.
