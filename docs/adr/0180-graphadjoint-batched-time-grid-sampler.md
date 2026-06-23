# ADR-0180 — GraphAdjoint batched time-grid Laplacian sampler

**Status:** Proposed (design-only; no implementation, no version assigned)
**Date:** 2026-06-23
**Authors:** ai-solutions-architect
**Branch:** `graphadjoint-batched-sampler` (worktree `sf-ga-sampler`)
**Cross-refs:** ADR-0179 (batched setup-time sampler; this ADR **extends** its
named wall "time-dependent `LaplacianAtTime` is architecturally out of scope"),
ADR-0028 (FFI/PyO3/WASM split — canonical C ABI, opaque handle, `SemiflowStatus`,
`catch_panic!`, `[profile.release-ffi]` `panic=unwind`), ADR-0171 (S³ carrier
conventions — CSR ragged flattening, fail-loud `OutOfDomain` walls, null-safe
idempotent free), ADR-0178 (`tt_varcoef_ffi.rs` — variable coefficients ALREADY
cross as pre-sampled CSR arrays), ADR-0115 (GraphAdjoint / state-adjoint),
ADR-0031 (PyO3 GIL release via `py.detach`), ADR-0051 (Magnus K=4 GL₄ design).

## Context

`GraphAdjoint` (`crates/semiflow-py/src/graph_adjoint.rs`,
`crates/semiflow-core/src/magnus_graph_adjoint.rs`) drives a backward costate
sweep over a **time-dependent** Laplacian `L_G(t)` supplied by a constructor
callback `lap_at_t: t → Graph|Laplacian`. The callback is invoked *inside* the
integrator hot loop, **per sub-step**.

ADR-0179 §"Fail-loud deferral wall" classified this callback as
*architecturally* out of scope: "this callback is invoked repeatedly per time
step and returns a whole vector, so sample-once-at-construction cannot serve it."
That conclusion is correct **only if you treat the sample times as unknown**.
They are not.

### The resource ADR-0179 missed

For a costate sweep of `t_horizon` over `n_steps` steps of size `τ = t_horizon /
n_steps`, the **set of sample times is fully determined at construction** by
`(t_horizon, n_steps)` plus the fixed GL₄ abscissae `c₁=(3−√3)/6`,
`c₂=(3+√3)/6`. The integrator samples `L_G` only at

```
T = { jτ + c₁τ , jτ + c₂τ  :  j = 0 … n_steps−1 }     (|T| = 2·n_steps)
```

(verified in `magnus_graph_adjoint.rs:169-172` and `:375-378`:
`lap1 = lap_at_t(t_start + c1·τ)`, `lap2 = lap_at_t(t_start + c2·τ)`; the sweep
sets `t_start = (n_steps−1−k)·τ`). For a **fixed graph topology** only the edge
weights (`Laplacian::vals`, the CSR `row_ptr`/`col_idx` invariant) change with t.

## Decision

Ship a **pre-sampled time-grid array path**: the host computes the Laplacian
weight sequence at the known GL₄ sample times **in its own language, once, at
construction**, and passes it down as a flat array `vals_seq` (+ the shared CSR
pattern once). Evolve replays the pre-sampled per-sub-step operators. **No live
host callback inside the integrator; no per-step C-ABI crossing; no GIL
re-acquisition; sample-once safety (ADR-0179) restored for the time-dependent
case.** This is **chosen variant (a)** — pure array path, NO callback ABI at all.
Variant (b), a batched callback invoked once at construction, is rejected for
phase-1 (see §"Variant (b) rejection"). The existing closure constructor
(`MagnusGraphHeatChernoff::new` + PyO3 `lap_at_t`) is **kept working, additively**.

### TRIZ resolution (АП → ТП → ФП → ИКР → решение)

- **НЭ (АП):** evolving a time-varying `L_G(t)` requires evaluating the host's
  `lap_at_t` at many t, but a live per-step host crossing across the C-ABI is
  unsafe (unwind across FFI = UB) and expensive (~200× + GIL-defeat, ADR-0179/0034).
- **ТП:** instrument = the host weight-supplier. ТП-1: supplier is *live during
  evolve* → weights track t (useful) but reintroduces per-step crossing (harm).
  ТП-2: supplier is *absent during evolve* (sampled once) → no crossing (no harm)
  but seemingly cannot track t (loses the useful function). Chosen half: ТП-2
  (keep sample-once safety), pushed to the limit.
- **ОЗ / ОВ / ВПР:** zone = the FFI/GIL boundary; **operative time = construction,
  BEFORE the `py.detach`/evolve window**; resource already in the system = the
  **time grid `T` is computable from `(t_horizon, n_steps, c₁, c₂)` at
  construction** + the topology is fixed (only `vals` vary).
- **ФП:** the weight supply must be *present* (to define `L_G(t)` at every needed
  t) **and** *absent* (no live crossing during evolve). **Resolution in time:**
  present at construction, absent during evolve — sample the entire weight
  sequence at the known grid `T` once, store it, then evolve replays it.
- **ИКР:** the integrator *itself*, using the grid it already knows, obtains every
  `L_G(t)` it needs with **zero** extra machinery during evolve — the host simply
  hands over a pre-computed array; the per-step callback disappears.
- **Решение:** pre-sampled `vals_seq[2·n_steps × nnz]` + shared CSR pattern;
  `GraphAdjoint::from_presampled(...)`. The contradiction is **resolved, not
  compromised**: time-varying weights AND sample-once safety hold simultaneously.
  (Not a golden mean — there is no live callback left to trade against.)

### Honest scope-narrowing found in the source (CRITICAL)

The grid is **`2·n_steps`**, NOT `n_steps`. Magnus K=4 evaluates `L_G` at **two
GL₄ sub-step abscissae** `t_start + c₁τ` and `t_start + c₂τ` per step
(`magnus_graph.rs:10`, `magnus_graph_adjoint.rs:169-172`), never at the step
boundary. The pre-sampled sequence MUST therefore be **GL₄-aware**: it stores one
weight vector per `(step, abscissa)` pair, length `2·n_steps` in step order
`[(j=0,c₁),(j=0,c₂),(j=1,c₁),…]`. A naïve "one Laplacian per step grid point"
array would be silently wrong at O(τ²) (the commutator term `[A₂,A₁]` mixes the
two abscissae). This is why the API takes `n_steps` and reproduces the abscissa
schedule internally rather than letting the host guess. For the **VarCoef**
kernel the host must additionally pre-sample `a(t)` on the same `2·n_steps` grid.

### Core surface (additive; does NOT touch the closure constructor)

A new input type carries the pre-sampled sequence + shared pattern:

```rust
/// Pre-sampled, fixed-topology, GL4-aware Laplacian weight sequence.
/// `vals_seq.len() == 2 * n_steps * nnz`, laid out per (step, abscissa) in
/// schedule order [(0,c1),(0,c2),(1,c1),(1,c2),…]; each nnz-block reuses the
/// shared CSR pattern `row_ptr`/`col_idx`.
pub struct PreSampledLaplacianSeq<F> {
    row_ptr: Vec<usize>,   // len n_nodes+1  (shared)
    col_idx: Vec<u32>,     // len nnz        (shared)
    vals_seq: Vec<F>,      // len 2*n_steps*nnz
    n_steps: usize,
    kind: LaplacianKind,   // Combinatorial | SymNormalized
}
```

Plus a constructor `MagnusGraphHeatChernoff::from_presampled(graph,
PreSampledLaplacianSeq, rho_bar, conv_check)` and a VarCoef sibling
`VarCoefMagnusGraphHeatChernoff::from_presampled(…, a_seq)`. Evolve walks
`vals_seq` block-by-block; each block is reconstituted into a `Laplacian` via a
**new core helper `Laplacian::from_csr_parts(n_nodes, row_ptr, col_idx, vals,
kind)`** (no public CSR ctor exists today — `assemble_combinatorial` is the only
path; this helper recomputes the Gershgorin `spectral_radius_bound` from rows,
preserving the cached-bound invariant). Index for adjoint step k uses the SAME
`t_s = (n_steps−1−k)·τ` schedule the closure path uses, so the replayed abscissa
order is byte-identical.

### Exact C-ABI surface (canonical; mirror PyO3/WASM after)

```c
/* Construct a pre-sampled Magnus graph adjoint. NO callback. */
SemiflowStatus smf_graph_adjoint_new_presampled(
    const SmfGraph*   topo,        /* fixed topology (validates pattern)   */
    const uintptr_t*  row_ptr,     /* len n_nodes+1  (shared CSR pattern)  */
    uintptr_t         row_ptr_len,
    const uint32_t*   col_idx,     /* len nnz                              */
    uintptr_t         nnz,
    const double*     vals_seq,    /* len 2*n_steps*nnz, schedule order    */
    uintptr_t         n_steps,
    double            t_horizon,
    double            rho_bar_max,
    int32_t           convergence_check,
    int32_t           kind,        /* 0=combinatorial, 1=normalized        */
    SmfGraphAdjoint** out);        /* opaque; free w/ smf_graph_adjoint_free */

/* Backward costate sweep λ_n → λ_0. n_steps MUST equal construction n_steps. */
SemiflowStatus smf_graph_adjoint_evolve_state_adjoint(
    const SmfGraphAdjoint* h,
    const double* lambda_n, uintptr_t lambda_len,  /* len n_nodes          */
    uintptr_t     n_steps,
    double*       out,      uintptr_t out_len);     /* len n_nodes          */

uintptr_t     smf_graph_adjoint_n_nodes(const SmfGraphAdjoint* h);
void          smf_graph_adjoint_free(SmfGraphAdjoint* h);   /* null-safe   */
```

Reuse `SemiflowStatus`, `catch_panic!`, opaque-handle idiom, null-check-before-
`catch_panic!`. **No new error variants** (ADR-0171): pattern/length mismatches
→ `OutOfDomain` (3); buffer too small → `GridMismatch` (1); null → `NullPtr` (5);
Magnus radius → existing `OutOfMagnusRadius`/`ConvergenceFailed`. `vals_seq` is
**copied** into the handle at construction; caller may free it immediately
(ownership not shared, strictly safer than the live-callback `user_data`
lifetime). A VarCoef variant `smf_graph_adjoint_new_presampled_varcoef(…,
const double* a_seq /* len 2*n_steps*n_nodes */, double a_sup_max)` mirrors this.

### PyO3 + WASM trampolines — the GIL story

The Python UX **keeps a callable** but it is invoked **sample-once at
construction under the GIL**, never per-step:

1. **Under the GIL (construction):** the binding computes the grid `T` from
   `(t_horizon, n_steps, c₁, c₂)`, calls the Python `lap_at_t` once per grid
   point (or, preferred, once with a vectorized `lap_at_ts(ts: np.ndarray)`
   batched callable — collapses `2·n_steps` GIL crossings to 1), extracts each
   `Laplacian`'s `vals` (validating the CSR pattern matches the topology, see
   wall), and builds the owned `PreSampledLaplacianSeq`. For VarCoef it also
   samples `a(t)` on the same grid. This replaces the current per-step
   `Python::attach` inside `py.detach` (`graph_adjoint.rs:206-209`).
2. **`py.detach` (evolve):** pure-Rust replay over `vals_seq`; **GIL fully
   released, ADR-0031 preserved byte-for-byte**, no `Python::attach` in the loop.

A new keyword constructor `GraphAdjoint(..., presample=True)` (or a classmethod
`GraphAdjoint.from_presampled(...)`) selects this path; default stays the
existing closure path (additive). The WASM trampoline
(`semiflow-wasm`) exposes `GraphAdjoint.fromPresampled(topo, rowPtr, colIdx,
valsSeq, nSteps, tHorizon, …)` taking `Float64Array`/`Uint32Array`; single
JS↔WASM crossing at construction; `Result<_, JsValue>`, `panic=abort` profile
(ADR-0028 Amendment 1). WASM is single-threaded so no `Send+Sync` split — the
host callable is never touched after construction (the structural ADR-0179 win).

## Fail-loud walls (honest INTRINSIC_LIMIT)

Construction-time typed rejection, never a silent fallback:

1. **Topology/sparsity changes with t** → `OutOfDomain`. The shared
   `row_ptr`/`col_idx` is supplied ONCE; if any host-sampled Laplacian had a
   different pattern, the host either cannot flatten it into `vals_seq`
   (different `nnz`) or the PyO3/WASM trampoline detects `row_ptr`/`col_idx`
   mismatch against the topology while sampling and raises before `py.detach`.
   Phase-1 supports **fixed topology, time-varying weights** only.
2. **Nonlinear / state-dependent Laplacian** (a function of the costate, not of t
   alone) → out of scope, undocumentable as a pre-sampled grid; the array path
   structurally cannot represent it (there is no state to sample against at
   construction). Named wall, no attempt.
3. **`n_steps` mismatch** between construction and evolve → `OutOfDomain`
   (the replay schedule is fixed at construction; a different `n_steps` would
   index a grid that was never sampled).
4. **`vals_seq.len() != 2·n_steps·nnz`** → `OutOfDomain` (GL₄-aware length check;
   guards the "one-per-step instead of one-per-abscissa" mistake).

## Parity gate

**`G_GRAPH_ADJOINT_SAMPLED_PARITY`** (RELEASE_BLOCKING, `slow-tests`,
`crates/semiflow-core/tests/`): on a known time-varying-weight **fixed-topology**
example (e.g. path-8, `w_ij(t) = 1 + 0.5·sin(t)` on each edge, `t_horizon=0.5`,
`n_steps=64`, both Magnus and VarCoef kernels), the **pre-sampled-array path** and
the **existing closure path** (`MagnusGraphHeatChernoff::new` + per-step
`lap_at_t` closure) MUST produce the same costate `λ_0`:

```rust
// closure path
let lam0_closure = closure_kernel.evolve_state_adjoint_into(τ, n_steps, &λn, …)?;
// pre-sampled path: vals_seq built by sampling the SAME closure on the GL4 grid
let lam0_sampled = presampled_kernel.evolve_state_adjoint_into(τ, n_steps, &λn, …)?;
assert_eq!(lam0_sampled.values(), lam0_closure.values()); // 0 ULP — same float ops
```

**Assertion: bit-exact (0 ULP).** Both paths feed identical `Laplacian` values in
identical abscissa order into identical `apply_omega4` kernels, so the float op
sequence is unchanged — `assert_eq!` on the raw `&[f64]`, not an ε-tolerance.
(`<1e-12` ε-form is the fallback ONLY if a reduction-order difference is ever
introduced; phase-1 introduces none, so 0 ULP is required.) A cross-binding
sub-check (core ↔ FFI ↔ PyO3 ↔ WASM, mirroring ADR-0179 `G_CALLBACK_SAMPLED_PARITY`)
is deferred to the binding-parity harness; the in-core 0-ULP gate is the
release blocker.

## Consequence

Closes the last PyO3-only deferral in the bindings: C and WASM hosts can now drive
a time-dependent `GraphAdjoint` via the pure array path, and PyO3 gains a fully
GIL-released time-dependent evolve. Strictly safer than the live `smf_mghc_new`
callback (`ghc_mghc.rs:233`, which assembles a Laplacian inside the integrator
loop and `panic!`s across the boundary on failure): no foreign code runs during
evolve, no unwind risk, no `user_data` lifetime obligation past construction.
The closure path remains for hosts that genuinely need on-demand sampling at
integrator-chosen t (and accept its hazards). **Phase-1 is deliberately narrow —
fixed topology, weights as a function of t alone — and the wall is documented, not
overclaimed.**
