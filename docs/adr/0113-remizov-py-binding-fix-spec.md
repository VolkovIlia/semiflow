# ADR-0113 — `semiflow-py` four-gap binding fix: Magnus-K4 weight parity, reverse Schrödinger, time-schedule coefficients, graph-rho naming

**Status**: Accepted (binding-side contract corrections; no core change)
**Date**: 2026-05-31
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0028 (PyO3 f64-only sibling boundary), ADR-0031 (three-phase
GIL-release), ADR-0034 (Python-callable coefficient closures DEFEAT `py.detach`),
ADR-0051 (Magnus K=4 graph kernel), ADR-0056 (Magnus K=6), ADR-0057/ADR-0061
(Schrödinger palindromic Strang + Cayley kinetic sign), ADR-0059 (graph callback
GIL R2), ADR-0111 (full-parity binding contract — §1 dispatch, §2 numpy I/O,
§4 GIL policy + pre-sampled-array mandate). Constitution v2.0.0 Override #1
(function cap ≤50 ENFORCED), Override #6 (MCP WAIVED — binding has no runtime).

## Context

Python testing of `semiflow-py` surfaced four confirmed binding-side gaps plus one
naming inconsistency. Two required CORE VERIFICATION (do the kernels actually
support the proposed fix?). Findings:

- **Gap #1 (MagnusGraphHeat K=4)** — binding constructor is typed `new(graph: &GraphPath, lap_at_t, rho_bar)`.
  `GraphPath` exposes only `new(n_nodes)` with unit edge weights, so the
  *headline* time-varying-weight feature is reachable only through the
  `lap_at_t` callback (which already accepts `Graph`/`Laplacian` via
  `extract_laplacian_arc`), never through the topology carrier. **CORE VERIFIED**:
  `MagnusGraphHeatChernoff::new(graph: Arc<Graph<F>>, lap_at_t: LaplacianAtTime<F>, rho_bar_max, conv_check)`
  (`crates/semiflow-core/src/magnus_graph.rs:225`) already takes a
  `t → Arc<Laplacian<F>>` callback whose returned Laplacians may carry **varying
  edge weights** (only `row_ptr`/`col_idx` topology must be fixed; debug_assert
  enforces). The K=4 core is feature-complete for time-varying weights — identical
  to the K=6 core. The compute helper `compute_magnus_graph` already builds a
  correct `LaplacianAtTime` and runs a per-step `apply_into_at` loop. **The defect
  is purely the constructor's narrow `&GraphPath` parameter type + positional
  `rho_bar`.** Fix is to mirror `MagnusGraphHeat6::new` (`magnus6.rs:101`).

- **Gap #3 (Schrödinger reverse evolution)** — `validate_evolve_params`
  (`schrodinger.rs:466`) rejects `t < 0`. Schrödinger is unitary/time-reversible.
  **CORE VERIFIED + NUMERICAL ROUND-TRIP RUN**: the kernel `apply_strang_step`
  (`schrodinger.rs:334`) is `S(τ) = V(τ/2)·K(τ)·V(τ/2)`. Both factors are analytic
  in τ with **no sign branch**: the V-rotation angle `α = V(x)·τ/2` flips sign
  (giving the exact inverse rotation), and the Crank-Nicolson Cayley map
  `A = (τ/2)·L` is unitary for any real A regardless of sign (`A²` is sign-invariant;
  the linear `Am`/`Ar` terms carry the sign), so `K(−τ) = K(τ)⁻¹`. Hence
  `S(−τ) = S(τ)⁻¹` exactly and norm is preserved. A round-trip harness (n=128,
  T=1.0, 200 steps each way, harmonic V, complex packet with momentum, feeding
  negative τ directly through `apply_into`) measured **‖ψ_back − ψ₀‖₂ = 1.19e-13**
  and norm drift **3.67e-14** — machine precision. The kernel is correct for τ<0.
  **SUBTLETY**: relaxing `validate_evolve_params` alone is INSUFFICIENT. The compute
  path calls `ChernoffSemigroup::evolve(t, …)` (`chernoff.rs:395`) which has its own
  `t < 0` guard (and `evolve_into`/`validate_t` at lines 299/429 likewise). The
  binding must therefore bypass the semigroup helper for negative t and run a manual
  `apply_into` loop with `τ = t/n_steps < 0` (exactly the loop the round-trip
  harness validated). No core change.

- **Gap #4 (time-varying-coefficient 1-D kernels)** — `Shift1D`/`DriftReaction1D`
  freeze coefficients at construction. `ShiftChernoff1D::with_closure`
  (`shift1d.rs:128`) stores `Arc<dyn Fn(f64)->f64 + Send + Sync>` **spatial** closures
  `x → f64`; there is **no time argument** and `apply_into` receives only `τ`.
  **CORE FINDING**: the core has NO per-step time-rebinding mechanism for these
  kernels, and adding a Python `β(t)` callback in the hot loop is forbidden — it
  re-acquires the GIL and defeats `py.detach` (ADR-0034/ADR-0111 §4). However,
  constructing a new `ShiftChernoff1D` is cheap (stores 3 Arcs + a `Grid1D` copy;
  no sampling at construction — `eval_a/b/c` are lazy). Therefore a **binding-side
  piecewise-constant-in-time segment loop** over existing constructors works and is
  GIL-safe: a pre-sampled time schedule (coefficient values on a time grid, passed
  as numpy arrays) is indexed by Rust inside `py.detach`, building one
  `with_closure` instance per segment with a captured constant `β_k`. **No core
  change required.** This mirrors the segment-walking pattern already proven in
  `MagnusGraphHeatChernoff::evolve_with_traj` (`magnus_graph.rs:296`).

- **Gap #5 (graph-rho naming)** — `MagnusGraphHeat` (K=4) takes `rho_bar` positional;
  `MagnusGraphHeat6` / `VarCoefMagnusGraph` take `rho_bar_max` keyword-only;
  `VarCoefGraphHeat` takes `rho_bar` positional. The `rho_bar` vs `rho_bar_max`
  distinction is **semantically real** — static kernels bound `ρ̄(L_G)` once
  (`rho_bar`); time-varying Magnus kernels bound the **peak over time**
  `max_t ρ̄(L_G(t))` (`rho_bar_max`).

## Decision

**D1 — Gap #1: rewrite `MagnusGraphHeat` (K=4) constructor to mirror `MagnusGraphHeat6`.**
Adopt the K=6 signature shape exactly:
```
#[pyo3(signature = (graph=None, laplacian=None, *, lap_at_t, rho_bar_max, convergence_check=true))]
fn new(graph: Option<&PyGraph>, laplacian: Option<&PyLaplacian>,
       lap_at_t: Py<PyAny>, rho_bar_max: f64, convergence_check: bool) -> PyResult<Self>
```
Resolve topology via the shared `resolve_graph(graph, laplacian)` helper (reuse the
one in `magnus6.rs`, or lift it to `graph_py.rs` as `pub(crate)` and import in both —
preferred, removes duplication). The K=4 `MagnusGraphHeat` no longer accepts a bare
`GraphPath` typed parameter; **`GraphPath` back-compat is preserved at the value
level** because `PyGraph` and the `lap_at_t` callback's `extract_laplacian_arc`
already accept `GraphPath`. Plumb `convergence_check` into
`MagnusGraphHeatChernoff::new(…, convergence_check)` (currently hard-coded `true` in
`compute_magnus_graph`). This is a **BREAKING** Python-signature change
(`rho_bar` → `rho_bar_max`, positional → keyword-only, `graph` first positional now
`Option`); justified because the prior surface made the headline feature inert and
v6.x is pre-1.0 for `semiflow-py` per ADR-0028.

**D2 — Gap #3: native reverse Schrödinger via relaxed validation + manual signed-τ loop.**
Relax `validate_evolve_params` to accept negative finite `t` (keep the
`!t.is_finite()` and `n_steps == 0` rejections; drop only `t < 0`). Because
`ChernoffSemigroup::evolve` re-imposes `t ≥ 0`, change `compute_schrodinger` to run
a **manual ping-pong `apply_into` loop** with `τ = t / n_steps` (which is negative
when `t < 0`) instead of `ChernoffSemigroup::evolve`. The kernel handles signed τ
correctly (verified, residual 1.19e-13). Decision: **relax in place — do NOT add a
separate `evolve_reverse`** (a sign on `t` is the natural, discoverable API; a second
method would duplicate the loop and the docstring). Update the `evolve` docstring:
`t` may be negative for backward (time-reversed) unitary evolution; norm is preserved
to machine precision. This is **NORMATIVE** new semantics → recorded here.

**D3 — Gap #4: binding-side pre-sampled time-schedule, piecewise-constant segments. No core change.**
Add a `with_time_schedule` constructor (or a `evolve_with_time_schedule` segmented
helper) to `Shift1D` and `DriftReaction1D` that accepts the spatial coefficient(s)
PLUS a 1-D `numpy.ndarray[float64]` time schedule and a matching time-grid (or an
implicit uniform `[0, t_final]` grid of length `n_segments+1`). The Rust evolve loop,
inside `py.detach`, walks segments: for segment `k` it builds
`ShiftChernoff1D::with_closure(move |_| beta_k, b_fn_k, c_fn_k, norm_k, grid)` and
runs `n_steps_per_segment` `apply_into` steps with `τ = (t_{k+1} − t_k)/n_steps_per_segment`,
ping-ponging the state. No Python callback enters the loop (GIL stays released);
the schedule is a plain `Vec<f64>` captured by value. Default interpolation:
**piecewise-constant** (segment value held). Optional linear interpolation between
schedule nodes MAY be added later but is OUT OF SCOPE here (keep minimal). Spatial
coefficients (`b(x)`, `c(x)`, and the non-scheduled coefficients) remain pre-sampled
arrays exactly as today. This is **NORMATIVE** new API → recorded here.
Scope note: a genuine `a(x,t)` (jointly space-time-varying, smooth) would require a
core API extension (`ShiftChernoff1D` closures take only `x`); that is explicitly
DEFERRED — the piecewise-constant-in-time schedule covers the stated β(t) use case
without touching the core.

**D4 — Gap #5: canonical graph-rho naming policy.**
- ALL graph-kernel rho parameters are **keyword-only** (`*,` in the pyo3 signature).
  This fixes the two positional offenders: `MagnusGraphHeat` (K=4, fixed by D1) and
  `VarCoefGraphHeat` (`graph_extra.rs:328` — change to keyword-only `rho_bar`).
- Name by semantics, NOT uniformly:
  - **`rho_bar`** for STATIC-bound kernels (time-independent generator):
    `GraphHeat`, `GraphHeat4th`, `GraphHeat6`, `VarCoefGraphHeat`.
  - **`rho_bar_max`** for TIME-VARYING Magnus kernels (peak over time):
    `MagnusGraphHeat` (K=4, via D1), `MagnusGraphHeat6`, `VarCoefMagnusGraph`.
  Document the distinction in each class docstring. The `.pyi` stub and any tests
  MUST follow the keyword-only signatures and the per-kernel name.

## Consequences

- **No `semiflow-core` change** for any of the four gaps. All fixes are in
  `crates/semiflow-py/src/{graph_py.rs, schrodinger.rs, shift1d_py.rs, drift_reaction_py.rs, graph_extra.rs}`
  plus the `.pyi` stub. The `compute_magnus_graph` helper gains a `convergence_check`
  parameter (D1); `compute_schrodinger` switches from `ChernoffSemigroup::evolve` to a
  manual signed-τ loop (D2).
- **BREAKING Python signatures** in this wave: `MagnusGraphHeat.__init__`
  (`rho_bar` → `rho_bar_max`, keyword-only); `VarCoefGraphHeat.__init__`
  (`rho_bar` positional → keyword-only). Acceptable pre-1.0 (ADR-0028).
- **New NORMATIVE semantics**: negative-`t` Schrödinger evolve (D2); time-schedule
  coefficient API on `Shift1D`/`DriftReaction1D` (D3); graph-rho naming policy (D4).
- GIL invariant (ADR-0111 §4) preserved everywhere: D2's manual loop and D3's segment
  loop both run inside `py.detach` with only `Send+Sync` captures; add/keep the
  `send_assertions.rs` lines.
- Function-cap (≤50 lines, Override #1 ENFORCED): the D3 segment loop must be a
  private `compute_*_time_schedule` helper extracted from the pymethod body
  (mirror `compute_magnus_graph` / `build_shift_scalar`).
- Reverse-evolution accuracy is the kernel's order-2 Strang accuracy run backward;
  exact reversibility holds at the algebraic level (round-trip residual 1.19e-13 is
  numerical-arithmetic floor, NOT discretization error, because the same step
  sequence is inverted).
