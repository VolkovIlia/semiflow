# ADR-0179 — Callback ABI for host-language closures across the C ABI

**Status:** Proposed (design-only; no implementation, no version assigned)
**Date:** 2026-06-23
**Authors:** ai-solutions-architect
**Branch:** `callback-abi-design` (worktree `sf-callback-abi`)
**Cross-refs:** ADR-0028 (FFI/PyO3/WASM split — C ABI is canonical, opaque-handle +
`SemiflowStatus` + `catch_panic!` + `[profile.release-ffi]` `panic=unwind` baseline),
ADR-0171 (S³ carrier-handle conventions — opaque handle, CSR ragged flattening,
fail-loud walls pre-checked as `OutOfDomain`), ADR-0034 (core `with_closure` /
per-node callback signature `double(*)(double, void*)`, 200× slowdown + GIL-defeat
documented, `with_a_array` / `with_a_cfunction` escape hatches reserved), ADR-0031
(PyO3 GIL release via `py.detach`), ADR-0026 (`ChernoffFunction` generic; `Send+Sync`),
ADR-0018 (parallel `Strang2D` requires `D: Send+Sync`), ADR-0178
(`tt_varcoef_ffi.rs` — variable coefficients ALREADY cross as pre-sampled CSR arrays).
**Feasibility note:** `docs/adr/0179-feasibility-note.md`.

## Context — what the ABI must serve, and what already exists

The core takes Rust closures / `fn`-pointers at these surfaces:

| Call site (`crates/semiflow-core/src/`) | Signature | Represents |
|---|---|---|
| `diffusion.rs` `DiffusionChernoff::new` | `a, a_prime, a_double_prime: fn(F)->F` | scalar `a(x)` 1D diffusion coef + derivatives |
| `diffusion.rs` `DiffusionChernoff::{with_closure, with_closure_local}` (ADR-0034) | `Fn(F)->F + Send+Sync` ×3 | owned `a(x)` variant (+ WASM thread-local alias) |
| `diffusion4.rs`, `diffusion6.rs`, `truncated_exp.rs`, `truncated_exp4.rs` `with_closure` | same `Fn(F)->F` triplet | 4th/6th-order `a(x)` siblings |
| `drift_reaction.rs` `with_closure` | `b(x)`, `c(x)` (`Storage2`) | advection / reaction coefs |
| `shift1d.rs` `ShiftChernoff1D::with_closure` | `a(x)`, `b(x)`, `c(x)` (`Storage3`) | full 1D drift-diffusion-reaction |
| `schrodinger.rs` `SchrodingerChernoff::new` | `impl Fn(F)->F` for `V(x)` | **pre-sampled at construction** into `v_at_node: Vec<F>` — the sample-once precedent |
| `varcoef_magnus_graph.rs` `VarCoefMagnusGraph::new` | `Box<dyn Fn(F)->Vec<F> + Send+Sync>` (`WeightAtTime`/`LaplacianAtTime`) | **time-dependent** weights / Laplacian, invoked per step |
| `grid_fn{,2d,3d}.rs`, `grid_nd.rs` `from_fn` | `Fn(F)->F` … `Fn(&[F;D])->F` | initial-condition / generic D-dim sampling |
| `adjoint_fp.rs` `MeasureState::pair` | `G: Fn(&[F;D])->F` | custom functional ⟨μ, f⟩ over a measure-state |

**But the variable-coefficient APIs already cross the ABI without closures** by
taking *pre-sampled arrays*: `graph_var_coef.rs` `VarCoefGraphHeatChernoff::new`
takes `a: Vec<F>`; `tt_varcoef_ffi.rs` (`smf_varcoef_tt_evolver_new`, ADR-0178)
crosses per-axis `aⱼ/bⱼ/vⱼ` as CSR-flattened `*const f64`; `coeff2d.rs`
(`closure_2d_from_array`) already converts a NumPy array into a pure-Rust
bilinear-interpolant closure *before* `py.detach`. The closure is therefore
**not fundamentally needed to vary a coefficient** — it is needed only when the
host wants the library to *sample the coefficient itself* at integrator-chosen
points (irregular nodes, on-demand functionals).

## Decision

Ship a **single, batched, sampling callback** for setup-time coefficient
acquisition. Do NOT ship a per-node callback in the integrator hot loop.

### Callback type (the ONE blessed signature)

```c
/* Batched scalar sampler: fill out[i] = f(xs[i]) for all i in one host call.
 * Return SmfStatus::Ok (0) on success; any non-zero = host signalled failure
 * (e.g. a Python exception was raised / a JS throw was caught in the trampoline).
 * The trampoline NEVER lets a host exception unwind across this boundary. */
typedef int (*smf_batch_scalar_fn)(
    const double* xs,   /* in : sample points, length n            */
    double*       out,  /* out: caller-Rust-owned buffer, length n */
    size_t        n,
    void*         user_data);

/* D-dim variant for GridFnND / AdjointFokkerPlanck::pair functionals.
 * xs is row-major [n * d]; point i = xs[i*d .. i*d + d]. */
typedef int (*smf_batch_vec_fn)(
    const double* xs,   /* in : n points × d coords, row-major */
    size_t        d,
    double*       out,  /* out: length n */
    size_t        n,
    void*         user_data);
```

A per-node single-point `double(*)(double, void*)` (the ADR-0034 sketch) is
**explicitly rejected** as a shipped surface — see feasibility note §"Per-node
verdict". The batched signature collapses N FFI/GIL crossings into one.

### Phase-1 extern "C" surface (scalar `a(x)` for 1D diffusion only)

```c
SmfStatus smf_diffusion1d_new_sampled(
    double xmin, double xmax, size_t n,
    smf_batch_scalar_fn a,          /* required: samples a(x) at grid + δ-stencil pts */
    smf_batch_scalar_fn a_prime,    /* optional: NULL ⇒ central-difference of a       */
    smf_batch_scalar_fn a_dd,       /* optional: NULL ⇒ central-difference of a'       */
    void*  user_data,
    double a_norm_bound,            /* 0.0 ⇒ auto from max|a| (cf. coeff2d::magnitude_max) */
    SmfDiffusion1dHandle** out_ev); /* opaque handle, freed by smf_diffusion1d_free */
```

The binding samples each callback **once each, at construction**, into owned
`Vec<f64>`s, then builds the core evolver via `with_closure` over those owned
arrays (a `move`-captured interpolant closure, exactly the `coeff2d.rs`
pattern). The callbacks are **never stored** and **never invoked after the
constructor returns**. Opaque handle, `SemiflowStatus` return, null-check
before `catch_panic!`, CSR/fail-loud conventions per ADR-0028 / ADR-0171.

### Unwind / panic safety (mechanism + justification)

The host callback runs foreign code that may raise. **Contract: a host
exception MUST NOT unwind across the FFI boundary** (Python exception or JS
throw propagating through Rust frames = UB).

- **C/FFI caller:** the C callback returns an `int` status; a non-zero return
  is the failure channel. The C author is contractually forbidden from
  `longjmp`/C++-throwing out of the callback. The Rust trampoline checks the
  returned status and converts non-zero → `SemiflowStatus::OutOfDomain` (or a
  new `CallbackFailed` variant if finer signalling is wanted — deferred, reuse
  `OutOfDomain` for phase-1 per ADR-0171 "no new variants").
- **PyO3 trampoline:** the Python callable is invoked under
  `Python::with_gil(|py| obj.call1(...))`; a raised Python exception surfaces as
  `PyErr` (a `Result`, *not* an unwind) — the trampoline returns non-zero, the
  constructor returns the `PyErr` to Python. No Rust panic is generated.
- **WASM trampoline:** `js_sys::Function::call1` returns `Result<JsValue,
  JsValue>`; a JS throw is captured as the `Err` arm — converted to non-zero.
- **Defense in depth:** the whole constructor body is still inside the existing
  `catch_panic!` (`crates/semiflow-ffi/src/panic.rs`, requires
  `[profile.release-ffi]` `panic=unwind`). If a trampoline bug nonetheless
  panics in Rust, `catch_unwind` returns `SemiflowStatus::Panic`; the
  half-built handle is dropped before return (constructor owns it until the
  out-param is written), so **no leak** — improving on ADR-0034's per-node
  "state is leaked" hazard, because sampling-at-construction means there is no
  in-flight integrator state to leak.

### Reentrancy / lifetime / ownership / threading

- **Lifetime of `user_data`:** caller must keep it valid only for the *duration
  of the constructor call*. Because callbacks are sample-once-then-discarded,
  `user_data` need NOT outlive the handle (strictly weaker, and safer, than
  ADR-0034's "valid until `smf_state_free`"). Documented in the header.
- **Ownership:** the handle owns only the sampled `Vec<f64>`s; it owns nothing
  foreign. Free is null-safe and idempotent (ADR-0171).
- **Threading:** sampling happens on the calling thread, serially, *before* any
  parallel `Strang2D` (ADR-0018) work begins. The interpolant closure built
  from the owned arrays is `Send+Sync` (it captures only `Arc<Vec<f64>>` +
  scalars — `coeff2d.rs` shows this is already `Send+Sync`). **The host
  callable itself is never touched by worker threads**, so the WASM
  non-`Send+Sync` `js_sys::Function` problem (ADR-0034 §WASM) evaporates — there
  is no need for a separate `with_closure_local`; the single sampled path serves
  all three hosts. This is the structural win of sampling over per-node calls.

### PyO3 trampoline — the GIL story

The conflict ADR-0034 flagged (per-node Python callback inside the
`py.detach` window forces GIL re-acquisition, defeating ADR-0031) **does not
arise here**: sampling runs in the **pre-flight phase under the GIL**, before
the `py.detach` compute window opens — identical to `coeff2d.rs`. Sequence:
(1) under GIL: call the Python callable once with a NumPy array of sample
points, `extract::<Vec<f64>>()` the result, build the interpolant closure;
(2) `py.detach`: pure-Rust evolve, GIL fully released, ADR-0031 preserved
byte-for-byte. The Python user passes `lambda xs: a_of(xs)` (vectorized, NumPy
in/out) — idiomatic and ~1 GIL crossing instead of ~N.

### WASM trampoline

`semiflow-wasm` exposes `Diffusion1d.newSampled(xmin, xmax, n, aFn, …)` where
`aFn: (xs: Float64Array) => Float64Array`. The batched call crosses JS↔WASM
once at construction. WASM is single-threaded (`wasm32-unknown-unknown`, no
`--features threads`), so no `Send+Sync` constraint applies. Errors as
`Result<_, JsValue>` (ADR-0028 Amendment 1; `panic=abort` profile).

### Performance note

A per-grid-node host callback costs one FFI + (Python) GIL re-acquire **per
node per Chernoff step** — ADR-0034 measured ~200× slowdown and a fully nulled
GIL-release optimization. The batched sampler costs **one host crossing total**
at construction; the hot loop then runs on a pure-Rust interpolant at native
speed (the v0.8.x SIMD `catmull_rom` path is preserved, ADR-0034 invariant (a)).
For irregular or analytic coefficients this is the recommended mitigation; for
the common case the **already-shipped array path** (`smf_varcoef_tt_evolver_new`,
ADR-0178) needs no callback at all.

## Scope

**Phase-1 (this ADR, if built):** batched scalar `a(x)` sampler for 1D
`DiffusionChernoff` only, across FFI + PyO3 + WASM, mirroring ADR-0171's
canonical-C-ABI-then-mirror discipline. One parity gate
(`G_CALLBACK_SAMPLED_PARITY`, RELEASE_BLOCKING, slow-tests): sampled `a(x)`
output byte-identical (0 ULP) across core ↔ FFI ↔ PyO3 ↔ WASM, and identical to
the pre-sampled-array (`with_closure`-over-`Vec`) path.

**Fail-loud deferral wall (named non-goals):**
- Per-node single-point callback `double(*)(double, void*)` — NOT shipped
  (feasibility §"Per-node verdict"); attempts to use it are a compile-time
  absence, not a runtime fallback.
- `b(x)` / `c(x)` for `DriftReaction`, `a/b/c` for `ShiftChernoff1D`, and the
  4th/6th-order siblings (`Diffusion4th/6th`, `TruncatedExp*`) — deferred;
  mechanical copy of phase-1 once demanded (mirrors ADR-0034 §"Composition types").
- **Time-dependent** coefficients `VarCoefMagnusGraph` (`WeightAtTime` /
  `LaplacianAtTime` = `Box<dyn Fn(t)->Vec<F>>`) — **architecturally out of
  scope**: this callback is invoked *repeatedly per time step* and returns a
  whole vector, so sample-once-at-construction cannot serve it. A time-driven
  callback ABI is a separate, harder design (the per-step crossing reintroduces
  every hazard the batched sampler avoids); deferred with a named wall, no
  attempt in this ADR.
- D-dim `smf_batch_vec_fn` for `GridFnND` / `AdjointFokkerPlanck::pair`
  functionals — designed above, deferred to a later phase; the measure-state
  `pair` functional additionally needs a particle-position ABI (ADR-0171
  `positions[n_part*D]`) and is gated behind real demand.
- Storing a callback for *repeated* invocation across multiple `evolve` calls —
  out of scope; sample-once is the only supported lifetime.

## Consequence

If implemented, this closes ADR-0034's reserved `with_a_cfunction` /
`with_a_array` escape hatches with a single batched surface that is *safer*
than the per-node sketch (no GIL-defeat, no `Send+Sync` split, no in-flight
leak). The feasibility note recommends whether to build it at all.
