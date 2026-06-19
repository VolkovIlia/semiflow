# ADR-0034 — `DiffusionChernoff::with_closure` API for variable `a(x)` across FFI/PyO3/WASM

**Status**: Accepted (planning ADR for v0.12.0; prerequisite for I3/I4/I5/O-4)
**Date**: 2026-05-10
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0001 (contract-first; this ADR fixes the math-side contract
before binding code lands), ADR-0026 (`ChernoffFunction` generic-over-`F`; the
`Box<dyn Fn>` storage MUST not break the trait), ADR-0028 (FFI/PyO3/WASM crate
split; §"Out of scope" defers variable `a(x)` to v0.11.0+ — this ADR un-defers
it), ADR-0029 (v0.11.0 zero-core-diff gate — this ADR is the FIRST core-API
addition since v0.10.0 and must wait for v0.12.0), ADR-0018 (parallel
`Strang2D` requires `D: Send + Sync`; closure variant MUST inherit), ADR-0031
(PyO3 GIL release via `py.detach`; Python callbacks force GIL re-acquisition
inside the released window), constitution v1.1.0 §"Project-Specific
Principles" #2 (additive, never subtractive — `with_closure` is a sibling
constructor, not a replacement). Audit O-4 (v0.10.0): "FFI/PyO3/WASM bindings
restricted to `a = 1.0`; document the path to variable-`a` exposure" — this
ADR closes O-4 at the design level; implementation lands in v0.12.0.

## Context

`DiffusionChernoff<F: SemiflowFloat = f64>::new`
(`crates/semiflow-core/src/diffusion.rs:91-105`) takes three `fn(F) -> F`
function pointers (`a`, `a_prime`, `a_double_prime`). The same fn-ptr-only
shape repeats in `truncated_exp.rs:119`, `truncated_exp4.rs:119`,
`diffusion6.rs:162`, `drift_reaction.rs:83`. The choice was deliberate at
v0.3.0 (ADR-0008) and re-affirmed at v0.10.0 Wave A (ADR-0028 §"Out of
scope"): C ABI cannot pass closures, so `fn`-pointer was the safe baseline
that kept all five Chernoff types `#[derive(Clone, Copy)]` and
`Send + Sync` (verified by `static_assertions::assert_impl_all!` in
`crates/semiflow-py/src/send_assertions.rs:20`).

The cost of that choice surfaces at v0.12.0 scope (ROADMAP §v0.12.0 I3): the
three bindings cannot expose variable `a(x)` because Python `lambda x: x*x`,
JS `x => x*x`, and C `double f(double x, void* user_data)` all carry state
that no `fn`-pointer can hold. v0.10.0 Wave A worked around this by
hard-coding `extern "Rust" fn unit_a(_: f64) -> f64 { 1.0 }` in
`crates/semiflow-py/src/handle.rs:33-35` and the FFI/WASM mirrors of the same
file. Persona P2 ("real PDE problems have variable diffusion") cannot be
served until `DiffusionChernoff` accepts an owning closure.

The design constraint is hard: any closure-storing variant MUST preserve (a)
the v0.8.x bit-equality of the `f64` SIMD path
(`crates/semiflow-core/src/diffusion.rs:144-173` — `apply` uses
`f.sample` SIMD `catmull_rom`); (b) the `Send + Sync` invariant required by
`Strang2D` parallel path (`crates/semiflow-core/src/strang2d_parallel.rs:116`,
which uses `std::thread::scope`); (c) the `no_std + alloc` budget (constitution
§"Technology Constraints"); (d) the ≤3-direct-dep budget; (e) the ChernoffFunction
trait shape (ADR-0026, no new associated types); (f) the panic boundary
(ADR-0028 — closures called from inside the integrator MUST be wrapped at the
binding crate, not at `semiflow-core`). The 39 existing call-sites of
`DiffusionChernoff::new` (grep across `crates/semiflow-core/src/**/*.rs` and
`tests/**/*.rs`) MUST keep compiling unchanged at v0.12.0.

## Decision

Add a sibling constructor `DiffusionChernoff::with_closure` that takes
**owned** `Box<dyn Fn(F) -> F + Send + Sync + 'static>` for each of the three
coefficient fields. The struct gains a `Storage<F>` enum (private) holding
either the `fn`-pointer triplet (current path, `Copy`-able) or the heap-owned
closure triplet (new path, `Clone` via `Arc` if needed but **not** `Copy`).
The public field accessors (`self.a`, `self.a_prime`, `self.a_double_prime`)
become inline thunks `fn call_a(&self, x: F) -> F` that dispatch on the
storage variant — **one branch per coefficient evaluation**, predictable by
the branch predictor for the lifetime of the struct.

### Concrete signatures

**Core (Rust)** — `crates/semiflow-core/src/diffusion.rs`:

```rust
// EXISTING (UNCHANGED — backward compat, all 39 call-sites compile):
pub fn new(
    a: fn(F) -> F,
    a_prime: fn(F) -> F,
    a_double_prime: fn(F) -> F,
    a_norm_bound: f64,
    grid: Grid1D<F>,
) -> Self;

// NEW (additive sibling):
pub fn with_closure<A, P, D>(
    a: A,
    a_prime: P,
    a_double_prime: D,
    a_norm_bound: f64,
    grid: Grid1D<F>,
) -> Self
where
    A: Fn(F) -> F + Send + Sync + 'static,
    P: Fn(F) -> F + Send + Sync + 'static,
    D: Fn(F) -> F + Send + Sync + 'static;
// Internally: stores Box<dyn Fn(F) -> F + Send + Sync + 'static> in a
// private Storage::Closure variant; no monomorphisation explosion (the
// generic params collapse at the Box boundary).
```

The struct loses `Copy` (closures are not `Copy`); `Clone` is preserved via
`Arc<dyn Fn(F) -> F + Send + Sync>` on the `Closure` variant if cloning is
needed (it is — `Strang2D` clones `D` for the parallel path per
`strang2d_parallel.rs`). Suckless trade: `Arc` adds one `alloc::sync::Arc`
import to `semiflow-core` but no new direct dep (Arc is in `alloc`).

**Public field migration**: the current `pub a: fn(F) -> F` fields become
`pub(crate)` and are replaced by inline accessor methods
`pub fn call_a(&self, x: F) -> F` (and likewise for `a_prime`,
`a_double_prime`). The 8 in-crate call-sites that today write `(dc.a)(x)`
become `dc.call_a(x)` (mechanical rename, ≤30 lines diff total). External
callers who previously accessed the field directly (none in tree per grep)
get a one-version `#[deprecated]` shim returning the underlying `fn` ptr if
the storage is the `FnPtr` variant, panicking otherwise — but since no
external caller uses the field directly, the shim is documented and never
exercised.

**FFI** — `crates/semiflow-ffi/src/ffi.rs`:

```c
// In remizov.h (cbindgen-generated):
typedef double (*remizov_a_fn)(double x, void* user_data);

remizov_status smf_state_new_with_closure(
    double xmin, double xmax, size_t n,
    remizov_a_fn a, remizov_a_fn a_prime, remizov_a_fn a_double_prime,
    void* user_data,        /* opaque, threaded through to all 3 callbacks */
    double a_norm_bound,
    const double* u0, size_t u0_len,
    size_t n_steps,
    SemiflowState** out_state /* opaque handle, freed by smf_state_free */
);
```

Rust side wraps `(remizov_a_fn, *mut c_void)` into a `Box<dyn Fn(f64) -> f64
+ Send + Sync>` via a `move` closure that calls back through the function
pointer. The wrapper is `unsafe` (raw pointer dereference) and lives inside
the `extern "C"` `catch_unwind` boundary already mandated by ADR-0028 — if
the C callback panics or unwinds, `catch_unwind` returns
`SemiflowStatus::Panic` and the state is leaked (caller must call
`smf_state_free` on whatever handle it received, or the constructor
returns NULL on failure and the closure is dropped before any leak).

**Safety contract** (documented in `remizov.h` and `crates/semiflow-ffi/src/ffi.rs`):

1. `user_data` lifetime: caller MUST keep `user_data` valid until
   `smf_state_free(*out_state)` returns. Violation = UB. Recommendation:
   pass NULL if no state needed.
2. Thread safety: the FFI wrapper marks the closure `Send + Sync`. Caller is
   responsible for `user_data` thread-safety (e.g. read-only data is fine; a
   shared `int*` counter without atomic guards is UB).
3. Reentrancy: the callbacks may be invoked many times per `smf_evolve`
   call (one per grid node per Chernoff step). Callbacks MUST be pure (no
   global state mutation) for math correctness.
4. Panic discipline: callbacks MUST NOT panic in Rust or throw in C. C
   exceptions across an FFI boundary are UB; Rust panics are caught at the
   `extern "C"` boundary but leak the in-flight state.

**PyO3** — `crates/semiflow-py/src/state.rs`:

```rust
#[pymethods]
impl Heat1D {
    /// New constructor: variable `a(x)` via Python callable.
    #[staticmethod]
    #[pyo3(signature = (xmin, xmax, n, a, a_prime, a_double_prime,
                        u0, n_steps = 100))]
    fn with_a_function(
        py: Python<'_>,
        xmin: f64, xmax: f64, n: usize,
        a: PyObject,                 // Python callable: float -> float
        a_prime: PyObject,
        a_double_prime: PyObject,
        u0: &Bound<'_, PyAny>,
        n_steps: usize,
    ) -> PyResult<Self> { ... }
}
```

The three `PyObject` callables are wrapped in
`Box<dyn Fn(f64) -> f64 + Send + Sync>` closures that call
`Python::with_gil(|py| obj.call1(py, (x,)).extract::<f64>())` per
invocation. Per-call overhead: ~2-5 µs on CPython 3.11 (GIL acquisition +
Python frame setup + return-value extraction). For a 1024-node grid with
100 Chernoff steps, that is 1024 × 100 × 5 calls (a, a', a'', plus 5
sample points in `gamma_a_baseline`) ≈ 2.5M Python calls = ~10 s wallclock,
versus ~50 ms for the `fn`-ptr unit-`a` path — **200× slowdown**, accepted.

**GIL interaction with ADR-0031**: `Heat1D.evolve` releases the GIL via
`py.detach`. Python callbacks INSIDE `py.detach` MUST re-acquire via
`Python::with_gil`. This is supported by PyO3 0.28 (`with_gil` blocks on the
GIL but is reentrancy-safe inside `detach`). The cost: the GIL-release
optimisation is functionally **null** when Python callbacks are present (each
callback re-acquires the GIL). This is documented in the rustdoc and surfaced
in the Python-side docstring as: *"with_a_function defeats the v0.11.0 GIL
release. For maximum throughput, see "
`Heat1D.with_a_array(a_values: numpy.ndarray, ...)` [v0.13.0+ if
demanded] OR pass a `ctypes.CFUNCTYPE` to `with_a_cfunction(...)` to use the
zero-overhead C callback path."*

**Default vs opt-in (Python)**: `with_a_function` is the user-facing default
(simple `lambda x: x*x` works). The `with_a_array` and `with_a_cfunction`
escape hatches are explicitly OUT OF SCOPE for v0.12.0 — ADR-0034 reserves
their names; they ship in v0.13.0+ if persona feedback demands them. We
deliberately do NOT recommend numba/cython JIT in the v0.12.0 docs because it
adds a dependency the user controls and is outside the project's quality
boundary.

**WASM** — `crates/semiflow-wasm/src/state.rs`:

```rust
#[wasm_bindgen]
impl Heat1D {
    #[wasm_bindgen(js_name = "withAFunction")]
    pub fn with_a_function(
        xmin: f64, xmax: f64, n: usize,
        a: js_sys::Function,
        a_prime: js_sys::Function,
        a_double_prime: js_sys::Function,
        u0: &Float64Array,
        n_steps: usize,
    ) -> Result<Heat1D, JsValue> { ... }
}
```

Each `js_sys::Function` is wrapped in a Rust closure that calls
`fn.call1(&JsValue::NULL, &JsValue::from_f64(x))` and extracts the result.
Per-call overhead: ~0.5-2 µs on V8 (JS↔WASM crossing). Same magnitude
slowdown as Python. **Critical**: `js_sys::Function` is **NOT** `Send +
Sync` (JS callbacks are pinned to the JS thread via a thread-local
`JS_REGISTRY`). This means the WASM `with_a_function` path CANNOT use any
core code path that requires `Send + Sync` on the Chernoff function — but
WASM is single-threaded by default (`wasm32-unknown-unknown` has no threads
without `--features threads`), so the `Strang2D` parallel path is already
unreachable. The WASM closure wrapper uses
`Box<dyn Fn(f64) -> f64>` (no `Send + Sync`), and we add a separate
non-`Send + Sync` constructor `DiffusionChernoff::with_closure_local` for
the WASM crate to call. This duplicates the constructor (one for `Send +
Sync`, one for thread-local) — accepted as the cost of supporting both
threaded (FFI/PyO3) and single-threaded (WASM) hosts.

**Default vs opt-in (WASM)**: `withAFunction` is user-facing default. No
escape hatch needed — JS users who care about throughput pre-sample `a(x)`
into a `Float64Array` and call a future `withAArray` (out of scope for
v0.12.0).

### Composition types — sequencing strategy

`with_closure` lands ONLY on `DiffusionChernoff` in v0.12.0. The other four
fn-ptr-only types (`TruncatedExp`, `TruncatedExp4`, `Diffusion6thChernoff`,
`DriftReaction`) get **the same treatment in v0.13.0** as a separate
mechanical pass. Rationale: (1) v0.12.0 budget is bounded by the I4/I5
binding work that depends on `DiffusionChernoff::with_closure`; (2)
`TruncatedExp`/`TruncatedExp4` are 4th-order replacements for the same math
and inherit the same usage pattern — copying the v0.12.0 design is risk-free
mechanical work; (3) `Diffusion6thChernoff` (6th-order) and `DriftReaction`
(`b(x)` and `c(x)` for advection-reaction) are independent surface and
deserve their own ADR if scope grows beyond mechanical translation.

## Per-binding plan

| Binding | Mechanism | Per-call overhead | Safety contract | Default vs opt-in |
|---------|-----------|-------------------|-----------------|-------------------|
| **Rust core** | `Box<dyn Fn(F) -> F + Send + Sync + 'static>` (FFI/PyO3 path) or `Box<dyn Fn(F) -> F + 'static>` (WASM path via `with_closure_local`) | ~5 ns dyn-dispatch overhead vs `fn`-ptr direct call (Rust closures) | `'static` lifetime; thread-safety via `Send + Sync` opt-in | `with_closure` (Send+Sync) is default; `with_closure_local` is WASM-only |
| **FFI** | `Box<dyn Fn(f64) -> f64 + Send + Sync>` wrapping `(remizov_a_fn, *mut c_void)` move closure | ~10 ns dyn-dispatch + 1 indirect call through C fn-ptr ≈ ~15 ns total per coefficient evaluation | Caller owns `user_data` lifetime through `smf_state_free`; callbacks MUST be pure and panic-free | `smf_state_new_with_closure` opt-in alongside existing `smf_state_new` (which keeps `a = 1.0`) |
| **PyO3** | `Box<dyn Fn(f64) -> f64 + Send + Sync>` wrapping `PyObject` with `Python::with_gil` per call | ~2-5 µs (GIL acquisition + frame setup + extract); defeats ADR-0031 GIL release | `'static` PyObject (cloned into closure); `Send + Sync` via `Py<PyAny>::clone_ref` thread-safety guarantee | `Heat1D.with_a_function` opt-in alongside `Heat1D(...)` (which keeps `a = 1.0`) |
| **WASM** | `Box<dyn Fn(f64) -> f64>` (NOT Send + Sync — JS callbacks are thread-local) wrapping `js_sys::Function` with `.call1(&JsValue::NULL, ...)` per call | ~0.5-2 µs (JS↔WASM crossing) | Single-threaded by construction; no `Send + Sync` needed; closure pinned to JS thread | `Heat1D.withAFunction` opt-in alongside `new Heat1D(...)` (which keeps `a = 1.0`) |

## Suckless audit

- **Direct deps**: 0 added to `semiflow-core` (`Box`, `Arc` are in `alloc`).
  Budget remains 2 (`num-traits`, `libm`); ≤3 hard cap (ADR-0028) preserved.
- **`no_std + alloc`**: preserved. `Box<dyn Fn>` and `alloc::sync::Arc` are
  both `alloc`-only. No `std` import added to `semiflow-core`.
- **Function-line budget (≤50)**: `with_closure` constructor is ~12 lines
  (3 `Box::new` calls + struct literal). The new `call_a`/`call_a_prime`/
  `call_a_double_prime` accessors are 4 lines each. The internal `Storage`
  enum is ~15 lines. All within budget.
- **File-line budget (700, Override #1)**: `diffusion.rs` is currently 405
  lines; the additions push it to ~480 lines, well within the override.
- **Backward compat**: 39 existing call-sites of `DiffusionChernoff::new` in
  `crates/semiflow-core/**/*.rs` and `tests/**/*.rs` compile unchanged. The
  field-access migration (`(dc.a)(x)` → `dc.call_a(x)`) is internal-only and
  invisible to external callers.
- **`Copy` removal**: `DiffusionChernoff: Copy` is removed at v0.12.0 (closures
  are not `Copy`). Audit shows zero in-tree usages depend on `Copy` (all
  `Strang2D::new(dx, dy, ...)` uses pass `dx` by value via move, which works
  for `Clone`-only types). External callers who relied on `Copy` get a
  compile error at v0.12.0 — flagged in CHANGELOG as a SemVer MINOR-tolerable
  regression per constitution §2 ("additive, never subtractive" applies to
  surface; `Copy` is an auto-trait removal that the additive surface forces).
  Mitigation: `#[derive(Clone)]` is retained.
- **SIMD bit-equality (ADR-0018)**: `with_closure` callers use the same
  `gamma_a_baseline_f64` / `zeta_correction_f64` paths via `call_a` accessor
  thunks. The thunk dispatch is a single tag-test predicted as taken by the
  branch predictor for the lifetime of the struct (one variant chosen at
  construction). Bit-equality with v0.8.x is preserved for the `FnPtr`
  variant; the `Closure` variant is a NEW caller and has no historical
  bit-equality contract.
- **Send + Sync invariant (`send_assertions.rs:20`)**: `DiffusionChernoff<f64>:
  Send + Sync` STILL HOLDS for both `with_closure` (explicit bound) and the
  legacy `new` constructor (fn-pointers are `Send + Sync`).
- **Panic boundary (ADR-0028)**: closures never cross `extern "C"` raw —
  always wrapped in the binding crate's `catch_unwind` (FFI) or Python's
  exception propagation (PyO3 — `with_gil` returns `PyErr` on Python-side
  exception, which is caught at the `#[pymethods]` boundary by
  `catch_panic_py!`) or `Result<T, JsValue>` (WASM — JS exceptions surface
  via `wasm-bindgen` shim per ADR-0028 Amendment 1).

## Considered alternatives

1. **Generic `<C: Fn(F) -> F>` (zero-cost monomorphisation)** — rejected.
   Would require `DiffusionChernoff<F, A, P, D>` with three extra type
   params; this propagates into `Strang2D<DiffusionChernoff<f64, _, _, _>,
   ...>`, into `ChernoffSemigroup<...>`, into the FFI handle type, into
   `SemiflowStateInner`. Each binding would then need to either (a)
   monomorphise to a single closure type (impossible — Python callbacks are
   `PyObject`, not a concrete `Fn` impl) or (b) erase to a trait object at
   the boundary, defeating the zero-cost claim. Conclusion: the boundary
   layer would `Box::new` anyway, so paying the dyn-dispatch cost
   uniformly inside `semiflow-core` is simpler and keeps the public type
   `DiffusionChernoff<F = f64>` (one type parameter, same as today).
   Trade-off accepted: ~5 ns/call dyn-dispatch overhead is invisible
   compared to the 5-50 ns of `f.sample` SIMD interpolation that dominates
   each node update.

2. **Borrowed `&dyn Fn(F) -> F` with explicit lifetime parameter** —
   rejected. `DiffusionChernoff<'a, F>` adds a lifetime that infects every
   downstream type (`Strang2D<'a, ...>`, `ChernoffSemigroup<'a, ...>`,
   `SemiflowStateInner<'a>`, the FFI opaque handle, the PyO3 `#[pyclass]` —
   `pyclass` cannot hold a non-`'static` reference, fatal). The opaque-handle
   FFI pattern requires `'static` storage or `unsafe` lifetime extension at
   every boundary. The Python `Heat1D` `#[pyclass]` MUST be `'static`. The
   WASM `#[wasm_bindgen]` MUST be `'static`. There is no callable site where
   borrowing wins; owning is the only option compatible with the three
   binding shapes.

3. **"Sample upfront into `Vec<F>` of length N, pass the array"** —
   rejected as the *default*; reserved as a v0.13.0+ escape hatch
   (`with_a_array`). Loses adaptivity: `gamma_a_baseline` evaluates `a` at
   `x_pre = x + (τ/2) · a'(x)` (ADR-0008 §9.2.3.A), which is OFF the grid;
   pre-sampling forces `a(x_pre) ≈ interp(a_array, x_pre)`, introducing an
   interpolation error that is order O(dx⁴) for cubic Hermite (ADR-0005)
   but DOUBLES the interpolation budget per Chernoff step (the function
   `f` is already sampled cubic-Hermite; adding a second Hermite layer for
   `a` is correct but duplicative). For genuinely-variable `a`, this also
   defeats the math.md §9.2.3.B `ζ` τ²-correction which evaluates `a`,
   `a'`, `a''` at NODES — pre-sampling those derivatives requires the
   user to also pre-sample `a'` and `a''` arrays, which is the API we're
   trying to avoid. Conclusion: pre-sampling is a *user* optimisation, not
   a *core* default. Saved for v0.13.0 if benchmarks show the closure
   overhead matters in practice.

## Migration path

ADR-0033 (NonSeparable2D coexistence at v1.0.0 freeze) sets the precedent:
two types can coexist indefinitely without `#[deprecated]` markers when
both serve distinct caller intents. Same logic applies here:

- `DiffusionChernoff::new(fn-ptr, ...)` is the *correct, simpler* API for
  callers who have a constant or compile-time-known `a` (e.g.
  `DiffusionChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.0, grid)` — used 30+
  times in tests, doctest, benches). Forcing those callers through
  `Box::new(|_| 1.0)` is anti-suckless.
- `DiffusionChernoff::with_closure(closure, ...)` is the *correct* API for
  callers who need runtime-parameterised `a` (FFI/PyO3/WASM users, runtime
  CFL adaptation, parameter sweeps).

**Recommendation**: keep BOTH constructors at v1.0.0 freeze, no
`#[deprecated]` marker, no removal. CHANGELOG records `with_closure` as
ADDED in v0.12.0; legacy `new` keeps the same signature. Reviewer-suckless
gate at v1.0.0 verifies both constructors compile and the field-access
migration (`call_a` thunks) is mechanical with no behavioural change for
the `FnPtr` variant.

`Copy` removal is the one SemVer MINOR-tolerable regression. CHANGELOG
v0.12.0 entry: "BREAKING (downstream-only): `DiffusionChernoff` no longer
implements `Copy`. `Clone` is preserved. Affected callers: any external
code relying on `Copy` (none in tree). Migration: replace `let dc2 = dc;`
with `let dc2 = dc.clone();`."

## Out of scope

- **2D / 3D variable-coefficient bindings (I4/I5)**: this ADR designs only
  `DiffusionChernoff` (1D). 2D anisotropic `β(x,y)` and 3D tensor `a(x)/b(y)/c(z)`
  closure paths require their own ADRs (separate composition-type
  design questions: how does `NonSeparable2DAnisotropicChernoff::new`'s
  `Fn(f64, f64) -> f64` map to a Python callable? — defer to v0.13.0).
- **`b(x)` and `c(x)` for `DriftReactionChernoff`**: same mechanical pattern
  as this ADR but with two coefficients instead of three. Defer to v0.13.0
  mechanical pass (see §"Composition types — sequencing strategy").
- **`TruncatedExp{,4}`, `Diffusion6thChernoff`**: same pattern. Defer to
  v0.13.0 mechanical pass.
- **Async / yielding API (I14)**: the v0.12.0 closure runs synchronously per
  Chernoff step. Async callbacks (e.g. Python `async def a(x)`) are NOT
  supported and require a fundamentally different integrator design (Python
  `asyncio` event loop reentry from inside Rust). Reserved for a separate
  ADR if persona demand emerges.
- **Pre-sampled `with_a_array` escape hatch**: reserved name, no
  implementation in v0.12.0. Ships in v0.13.0+ if benchmarks justify.
- **`with_a_cfunction` (PyO3 ctypes path)**: reserved name, no
  implementation in v0.12.0. The recommended path for users who need
  zero-overhead variable-`a` from Python is to drop to `semiflow-ffi`
  directly with their own ctypes wrapper.
- **v1.0.0 API freeze documentation**: this ADR commits to the
  `with_closure` signature as the v1.0.0-frozen API; the freeze ADR
  itself is a separate v1.0.0 task.

---

## Amendment 1 — v0.13.0 Wave B2 storage refactor (HalfNodeCoeffCache)

**Status**: Proposed 2026-05-19 for v0.13.0.

**Context**: Original ADR-0034 introduced `DiffusionChernoff::with_closure` accepting caller `a: impl Fn(F)→F`. In `TruncatedExp4thDiffusionChernoff::apply` (truncated_exp4.rs:411-414), this closure is invoked **4 times per node per stencil application** — per ADR-0019 Amendment 2 analysis, closure indirection defeats SIMD lane utilization regardless of vector width.

**Decision**: Additive sibling API: `TruncatedExp4thDiffusionChernoff::with_cached_coefficients(a: impl Fn(F)→F, n: usize, dx: F)` constructs a `HalfNodeCoeffCache<F>` storing pre-evaluated half-node values `a_at_halfnodes: Vec<F>` (length 2N — values at x_i ± 1/2·dx and x_i ± 3/2·dx as needed by 5-point K-kernel). Hot loop accesses cache by index, no closure dispatch. Original `with_closure` API preserved as-is for caller convenience and ABI stability (per ADR-0028 v0.10.0 binding-boundary). The cache is opaque to FFI; PyO3 / WASM / C ABI bindings call the variable-a constructor unchanged and the cache is built internally.

**Consequences**: SIMD lane saturation feasible (Wave B3). Memory cost: 2N·sizeof(F) ≈ 8 KB at N=512 per `TruncatedExp4` instance (negligible vs 2.8 MB baseline; tracked by `audit-findings-v0_13_0.md` memory accounting section). Bit-equality preserved if cache populated via identical `a(x)` evaluations at half-node points. Risk: if caller's `a(x)` is non-deterministic (impure closure), cached vs runtime evaluation diverges — mitigated by ADR-0034 §"Out of scope" already requiring pure deterministic closures.
