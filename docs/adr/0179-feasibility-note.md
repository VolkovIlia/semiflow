# ADR-0179 Feasibility / Risk Note — is a callback ABI worth building?

**Companion to:** `docs/adr/0179-callback-abi-for-closures.md`
**Date:** 2026-06-23 · **Author:** ai-solutions-architect · **Verdict:** below.

## Honest bottom line

**Do NOT build a general per-node callback ABI. Build at most the batched
sampler — and only if a concrete user appears who needs irregular/analytic
coefficient sampling that the array path cannot express. For every use case in
the tree today, the already-shipped pre-sampled-array path is the supported,
safe, and faster way; document it as such.**

This is a value/effort + risk judgment, not a physical contradiction, so no
TRIZ resolution is forced: "expose closures vs keep the C ABI closure-free" has
a clean honest answer — sample at the boundary, pass arrays, never let a
foreign callable into the hot loop.

## Why the cheap path already wins

The premise "these APIs are unbound because C has no closures" is only half
true. The variable-coefficient surfaces that matter were **re-shaped to take
arrays precisely so they could cross the ABI**:

- `smf_varcoef_tt_evolver_new` (ADR-0178, `tt_varcoef_ffi.rs`) crosses
  per-axis `a/b/v` as CSR `*const f64` — variable coefficients, no closure.
- `VarCoefGraphHeatChernoff::new` takes `a: Vec<F>` in core itself.
- `coeff2d.rs` `closure_2d_from_array` already turns a host array into a
  pure-Rust interpolant *before* `py.detach` — the exact "host drives a Rust
  coefficient" story, solved without a callback ABI.
- `SchrodingerChernoff::new` already takes `impl Fn(F)->F` for `V(x)` and
  **pre-samples it once at construction** into `v_at_node: Vec<F>` — proof the
  sample-once idiom is the library's established pattern, not a new invention.

A host that can write `a(x)` can sample it into a NumPy array / `Float64Array`
in one line and hand it over. The library then runs at full native + SIMD
speed. The callback only adds value when the *library* must choose the sample
points (irregular grids, on-demand functionals) — a real but currently
**hypothetical** need with zero call sites demanding it.

## The real hazards (enumerated honestly)

| Hazard | Severity | Mitigation in ADR-0179 | Residual |
|---|---|---|---|
| **Unwind across FFI** — Python exception / JS throw / C++ throw propagating through Rust frames = UB | CRITICAL | Trampoline converts host error to an `int` status (`PyErr`/`Err(JsValue)` are `Result`s, not unwinds); whole body still inside `catch_panic!` (`panic=unwind` profile) | C author can still `longjmp`/C++-throw — contractual UB, undetectable. Inherent to any C callback ABI. |
| **GIL deadlock / GIL-release defeat** (ADR-0031 vs per-node callback inside `py.detach`) | HIGH | Sampling happens in pre-flight **under the GIL**, before `py.detach`; ADR-0031 byte-preserved | None for the batched path. Per-node path would re-introduce it — hence per-node is rejected. |
| **Per-node overhead** — FFI + GIL/JS crossing per node per step | HIGH | Batched: one crossing total at construction | None for batched. ADR-0034 measured 200× for per-node. |
| **Thread-safety under `parallel` feature** — `Strang2D` `std::thread::scope` needs `D: Send+Sync`; `js_sys::Function` is NOT `Send+Sync` | HIGH | Host callable never reaches worker threads (sampled-then-discarded); interpolant captures only `Arc<Vec<f64>>` → `Send+Sync`; no `with_closure_local` split needed | None — this is the structural advantage of sampling. |
| **`user_data` lifetime / dangling** | MEDIUM | Sample-once ⇒ `user_data` need only live for the constructor call (weaker than ADR-0034's "until free") | Caller can still pass a dangling pointer — standard C-ABI caveat. |
| **In-flight state leak on panic** (ADR-0034 per-node hazard) | MEDIUM | No in-flight integrator state exists during sampling; half-built handle dropped before return | None for batched. |
| **Surface / maintenance cost** — ×3 bindings × N coefficient types × parity gates | MEDIUM | Phase-1 = ONE type (scalar 1D `a`), ONE parity gate; rest fail-loud deferred | Scope creep risk if "completeness" pressure returns (cf. ADR-0028 Amendment 2 tiering). |

## Per-node verdict

The ADR-0034-sketched `double(*)(double, void*)` per-node callback is **not
worth building**: it is 200× slower, nullifies the v0.11.0 GIL-release work,
forces a `Send+Sync`/thread-local constructor split, and risks an in-flight
leak on panic — to deliver a coefficient the host could have sampled into an
array itself. Reject it as a shipped surface.

## Recommendation

1. **Default supported path (document loudly):** pre-sample the coefficient
   host-side and pass an array — `smf_varcoef_tt_evolver_new` /
   `with_closure`-over-`Vec` / `coeff2d`-style interpolant. Add a docs section
   "Variable coefficients: pass an array" to the binding READMEs and to
   ADR-0034's user-facing note. This needs **no new code**.
2. **Build the batched sampler (ADR-0179) only on demand** — when a user has a
   genuinely library-chosen sampling need (irregular nodes, analytic
   functional) that an array cannot pre-express. It is feasible and safe as
   specified; it is not urgent.
3. **Never build the per-node callback ABI.**

Phase-1 batched scalar-`a` is **feasible and safe**. Per-node callbacks under
the parallel/SIMD hot path are **NOT recommended**. The discretized-array path
already covers every real use case in the tree; the truthful answer is "don't
build the general callback ABI — here's the array path instead," and build the
narrow batched sampler only when evidence demands it.
