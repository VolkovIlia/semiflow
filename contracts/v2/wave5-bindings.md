# Wave 5 Contract: Zero-Copy Bindings + Generic-over-F Composition

**Status**: NORMATIVE
**ADR**: docs/adr/0045-zero-copy-bindings-and-f32-composition.md
**Sister ADR**: docs/adr/0046-precision-policy-bands.md
**Sister contract**: contracts/v2/wave5-precision-policy.md
**Scope**: semiflow-core v2.0 Wave 5
**Depends on**: contracts/v2/wave1-scratch.md, contracts/v2/wave2-inplace-strang.md
(NORMATIVE prerequisites — Wave 5 reuses `ScratchPool<F>` and `apply_into`).

---

## §1 — `semiflow-py` `evolve_into` (zero-copy with copy fallback)

### 1.1 Surface

Two new methods on the existing `Heat2D` and `Heat3D` pyclasses. **No change to
`Heat1D`** (out of scope per ADR-0045 §"Out of Scope"). Both are siblings of
the existing `evolve(u0, tau, n_steps) -> ndarray` method — additive, never
replacing the v1.0.0 surface.

```rust
// crates/semiflow-py/src/state_2d.rs (Heat2D)
#[pyo3(signature = (buf, tau, n_steps))]
fn evolve_into<'py>(
    &self,
    py: Python<'py>,
    buf: PyReadwriteArray1<'py, f64>,
    tau: f64,
    n_steps: usize,
) -> PyResult<()>;

// crates/semiflow-py/src/state_3d.rs (Heat3D) — same signature
```

### 1.2 Dtype/stride/length probe (validate phase, under GIL)

Pseudocode for the probe inside `evolve_into`:

```rust
// Phase 1 (under GIL)
validate_evolve_2d_params(tau, n_steps)?;            // existing helper

let expected_len = self.nx * self.ny;                // 2D; 3D = nx*ny*nz
let arr: ArrayViewMut1<'_, f64> = buf.as_array_mut();
let raw_len = arr.len();
let is_contig = arr.is_standard_layout();            // numpy crate helper
let dtype_ok = true;  // PyReadwriteArray1<f64> already gates dtype at extract

if !dtype_ok || !is_contig || raw_len != expected_len {
    tracing::warn!(
        target: "semiflow::zero_copy",
        ?dtype_ok, ?is_contig, raw_len, expected_len,
        "evolve_into falling back to copy mode"
    );
    return self.evolve_into_copy_fallback(py, buf, tau, n_steps);
}

// Phase 2 (GIL released, zero-copy)
let strang = self.inner.strang.clone();
let grid = self.inner.grid;
let mut scratch = ScratchPool::<f64>::new();         // or reuse pooled
let result = py.detach(|| {
    let src = GridFn2D { values: arr.to_vec(), grid }; // see note 1.3
    compute_evolve_2d_in_place(&strang, src, tau, n_steps, &mut scratch)
});
result.map_err(|e| from_core(&e))?;
Ok(())
```

> **NOTE 1.3 — naming**: "zero-copy" here means "no double-buffer in the
> Python heap"; we still pass `&mut [f64]` into the Strang ping-pong path,
> which internally uses `ScratchPool` for its own working memory. The single
> O(N) copy avoided is the `Vec<f64>` extract that today's `evolve` does at
> `state_2d.rs:100`.
>
> The cleanest implementation: lift the existing `compute_evolve_2d` helper
> in `handle.rs` to take `&mut [f64]` instead of `Vec<f64>`, and have it call
> `Strang2D::apply_into` from ADR-0042. The closure passed to `py.detach`
> then borrows `buf`'s backing memory directly — *that* is the zero-copy
> path. The pseudocode above is approximate; the engineer pass fills in the
> exact lifetime gymnastics.

### 1.4 Copy fallback

Triggers on any probe failure:

```rust
fn evolve_into_copy_fallback(
    &self,
    py: Python<'_>,
    mut buf: PyReadwriteArray1<'_, f64>,
    tau: f64,
    n_steps: usize,
) -> PyResult<()> {
    let mut owned: Vec<f64> = buf.as_array().to_vec();   // copy in
    // ... run compute on owned ...
    let result = py.detach(|| compute_evolve_2d_owned(/* ... */, owned, /* ... */));
    let result = result.map_err(|e| from_core(&e))?;
    let mut view = buf.as_array_mut();
    let dst_slice = view.as_slice_mut()
        .ok_or_else(|| new_pyerr("OutOfDomain",
            "destination buffer is not contiguous; cannot write back"))?;
    if dst_slice.len() != result.len() {
        return Err(new_pyerr("GridMismatch",
            "result length != buffer length after fallback"));
    }
    dst_slice.copy_from_slice(&result);
    Ok(())
}
```

If the buffer is non-contiguous on the *output* side too, we cannot scatter
without iteration; in that case return `GridMismatch` rather than silently
producing wrong values.

### 1.5 Tracing warning policy

- `tracing::warn!` with `target: "semiflow::zero_copy"`.
- Fields: `dtype_ok: bool`, `is_contig: bool`, `raw_len: usize`,
  `expected_len: usize`.
- Emitted **once per call** that falls back (no per-step spam).
- Users opt-in by initialising a `tracing-subscriber` in their Python harness
  (we document a Python snippet in `docs/precision-policy.md` and in
  `crates/semiflow-py/README.md`).
- `tracing` is a new transitive dep but no new direct dep (PyO3 0.28 already
  pulls it; verify in `cargo tree -p semiflow-py` before merge).

### 1.6 Error model

Unchanged from `evolve` — all errors raise `SemiflowError` with
`.kind ∈ {"GridMismatch", "OutOfDomain", "NanInf", ...}`. The zero-copy path
adds no new error variants.

---

## §2 — `semiflow-ffi` caller-owned buffer ABI

### 2.1 New symbol (additive — v1.0.0 ABI preserved)

```c
/// Advance state in place using a caller-provided buffer.
///
/// Borrows `buf` for the duration of the call only; no ownership transfer.
/// On `Ok`, `buf[0..buf_len]` contains the evolved values. On error,
/// `buf` is left in an indeterminate state (may be partially written).
///
/// # Safety
/// - `state` must be a valid non-null pointer from `smf_state_new_*`.
/// - `buf` must be a valid pointer to `buf_len` contiguous, writable f64
///   values, well-aligned for f64.
/// - `buf_len` must equal `smf_state_size(state)`.
/// - `buf` must not alias the state's internal buffer (UB).
SemiflowStatus smf_evolve_inplace(
    SemiflowState* state,
    double*       buf,
    size_t        buf_len,
    double        tau,
    size_t        n_steps
);
```

### 2.2 Rust implementation sketch

```rust
// crates/semiflow-ffi/src/ffi.rs (new function, ~40 LoC)
#[no_mangle]
pub unsafe extern "C" fn smf_evolve_inplace(
    state: *mut SemiflowState,
    buf: *mut f64,
    buf_len: usize,
    tau: f64,
    n_steps: usize,
) -> SemiflowStatus {
    if state.is_null() || buf.is_null() {
        return SemiflowStatus::NullPtr;
    }
    catch_panic!({
        // SAFETY: caller guarantees live Box<SemiflowStateInner>.
        let inner = unsafe { &mut *state.cast::<SemiflowStateInner>() };
        if n_steps == 0 || !tau.is_finite() || tau < 0.0 {
            return SemiflowStatus::OutOfDomain;
        }
        let expected = inner.current.values.len();
        if buf_len != expected {
            return SemiflowStatus::GridMismatch;
        }
        // SAFETY: caller-validated pointer + length; non-aliasing per docstring.
        let buf_slice = unsafe { std::slice::from_raw_parts_mut(buf, buf_len) };

        // Copy caller buf -> internal state, run in-place evolve, copy back.
        // The "zero-copy" win is at the *caller* layer (their Python/C++ array
        // never copies); we still bridge once into the internal Vec to reuse
        // the v1.0.0 ChernoffSemigroup::evolve path.  An even tighter path
        // would expose ChernoffSemigroup::evolve_into and skip the bridge —
        // see §2.4 "future optimisation".
        inner.current.values.copy_from_slice(buf_slice);

        let chernoff = inner.semigroup.func.clone();
        match semiflow_core::ChernoffSemigroup::new(chernoff, n_steps) {
            Err(e) => SemiflowStatus::from(&e),
            Ok(sg) => match sg.evolve(tau, &inner.current) {
                Err(e) => SemiflowStatus::from(&e),
                Ok(next) => {
                    buf_slice.copy_from_slice(&next.values);
                    inner.current = next;
                    inner.semigroup = sg;
                    SemiflowStatus::Ok
                }
            },
        }
    })
}
```

### 2.3 cbindgen drift gate

`xtask ffi-headers` must regenerate `crates/semiflow-ffi/remizov.h` and the CI
drift gate must accept the new symbol. Add a NORMATIVE test
`tests/ffi_header_drift.rs` (already exists per v0.10.0 Wave A) — extend its
golden manifest to include `smf_evolve_inplace` and its expected C signature.

### 2.4 Future optimisation (out of Wave 5)

Wave 5 ships the additive symbol with one internal bridge copy
(`copy_from_slice` into `inner.current.values`). A v2.1 optimisation can
expose `ChernoffSemigroup::evolve_into(&mut [F])` and skip the bridge
entirely — saving the second O(N) copy. Tracked in ROADMAP under v2.1
performance follow-ups.

### 2.5 Panic and error model

Unchanged. All `SemiflowStatus` variants apply; `Panic = 99` from
`catch_unwind`; `OutOfDomain`, `NanInf`, `BoundaryFailure`, `CflViolated`,
`ConvergenceFailed` from `ChernoffSemigroup::evolve`.

### 2.6 ABI stability statement

The new symbol is **additive only**. `smf_evolve`,
`smf_state_new_heat_1d_unit`, `smf_state_new_with_closure`,
`smf_state_free`, `smf_state_values`, `smf_state_size`,
`smf_status_str`, `smf_version` are byte-for-byte unchanged. v1.0.0
ABI compatibility (ADR-0035 §"C ABI freeze") is preserved.

---

## §3 — Generic-over-F lift on Strang2D/3D parallel paths

### 3.1 Inventory of f64 monomorphism sites (post-W2/W3, verified
`grep -n "f64"` on 2026-05-20)

| File | Line | Symbol | Current | Wave 5 target |
|---|---|---|---|---|
| `strang2d.rs` | 165 | `impl<X, Y> ChernoffFunction<f64> for Strang2D<X, Y, f64>` | f64-only | `impl<X, Y, F: SemiflowFloat + Send + Sync> ChernoffFunction<F> for Strang2D<X, Y, F>` |
| `strang2d.rs` | 177 | `fn apply(tau: f64, f: &GridFn2D<f64>)` | f64 | `fn apply(tau: F, f: &GridFn2D<F>)` |
| `strang2d.rs` | 182-190 | `apply_into` parallel override | f64 | `apply_into<F>` |
| `strang2d.rs` | 302 | `apply_parallel` body | uses `f.values.clone(): Vec<f64>` and `0.5_f64` half | `Vec<F>`, `F::from(0.5)` |
| `strang3d.rs` | 168 | `impl<X, Y, Z> ChernoffFunction<f64> for Strang3D<X, Y, Z, f64>` | f64-only | `impl<X, Y, Z, F: SemiflowFloat + Send + Sync> ChernoffFunction<F> for Strang3D<X, Y, Z, F>` |
| `strang3d.rs` | 176 | `fn apply(tau: f64, f: &GridFn3D<f64>)` | f64 | `fn apply(tau: F, f: &GridFn3D<F>)` |
| `strang3d.rs` | 185-193 | `apply_into` parallel override | f64 | `apply_into<F>` |
| `strang3d.rs` | 560 | `apply_parallel` body (5-leg) | f64 | F-generic |
| `strang2d_parallel.rs` | 43 | `pub static PARALLEL_2D_POOL: RefCell<ScratchPool<f64>>` | f64 | split per-precision (see §4) |
| `strang2d_parallel.rs` | 140 | `fn parallel_x_pass(state: &mut [f64], ..., op: &X, tau: f64)` | f64 | `fn parallel_x_pass<X, F>(state: &mut [F], ..., op: &X, tau: F) where X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync` |
| `strang2d_parallel.rs` | 172-210 | `x_pass_chunk` | f64 | F-generic |
| `strang2d_parallel.rs` | 249-282 | `parallel_y_pass` | f64 | F-generic |
| `strang2d_parallel.rs` | 285-359 | `y_phase1_apply`, `y_apply_cols` | f64 | F-generic |
| `strang2d_parallel.rs` | 377-411 | `y_phase2_scatter`, `scatter_rows` | f64 | F-generic |
| `strang3d_parallel.rs` | 46 | `PARALLEL_3D_POOL: RefCell<ScratchPool<f64>>` | f64 | split per-precision |
| `strang3d_parallel.rs` | 124-181 | `parallel_x_pass_3d`, `x_pass_chunk_3d` | f64 | F-generic |
| `strang3d_parallel.rs` | 200-298 | `parallel_y_pass_3d` and helpers | f64 | F-generic |
| `strang3d_parallel.rs` | 346-444 | `parallel_z_pass_3d` and helpers | f64 | F-generic |
| `strang3d_parallel.rs` | 451 | `z_scatter` and helpers | f64 | F-generic |

**Total**: 17 inventoried sites across 4 files. All lifts are mechanical:
`f64 → F`, `0.0_f64 → F::zero()`, `2.0 → F::from(2.0).unwrap_or(F::zero())`
(better: introduce a `half<F>()` helper as already exists in `strang2d.rs:286`).

### 3.2 Mechanical lift template

```rust
// BEFORE
pub(crate) fn parallel_x_pass<X>(
    state: &mut [f64],
    gx: Grid1D,
    n_threads: usize,
    op: &X,
    tau: f64,
) -> Result<(), SemiflowError>
where
    X: ChernoffFunction<S = GridFn1D> + Clone + Send + Sync,
{ /* ... */ }

// AFTER
pub(crate) fn parallel_x_pass<X, F>(
    state: &mut [F],
    gx: Grid1D<F>,
    n_threads: usize,
    op: &X,
    tau: F,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{ /* same body; F replaces f64 */ }
```

### 3.3 Grid1D parameterisation

After ADR-0025 Wave 1, `Grid1D` is generic over F (`Grid1D<F>`). The current
`strang2d_parallel.rs` uses `Grid1D` without the F parameter because the file
is monomorphised. The lift propagates the `<F>` through.

### 3.4 Bit-equality preservation (ADR-0018)

The `tests/strang2d_parallel_bit_equal.rs` test currently asserts
`strang2d_parallel<f64>::apply == strang2d_serial<f64>::apply` byte-for-byte
on 5 grid sizes × 3 thread counts. After Wave 5, the f64 codegen path is
unchanged — `Strang2D<X, Y, f64>` still calls the same `parallel_x_pass`
body with the same f64 types instantiated. **The only change at f64 codegen
is the *signature*, not the *body*.** SHA-256 manifest of test outputs MUST
remain byte-identical to v1.0.0 (Wave 5 first commit snapshots the manifest
into `tests/fixtures/strang_parallel_bit_equal_v1.sha256` for CI gating).

For `F = f32`, a sister test `tests/strang2d_parallel_bit_equal_f32.rs`
asserts f32 serial = f32 parallel byte-for-byte (no cross-precision
comparison). See ADR-0046 §3.3.

---

## §4 — Per-precision thread-local pools

Rust's `thread_local!` macro forbids generic statics:

```rust
// FORBIDDEN — does not compile
thread_local! {
    pub static POOL<F>: RefCell<ScratchPool<F>> = ...;  // ERROR
}
```

### 4.1 Resolution — trait-dispatched per-precision pools

```rust
// crates/semiflow-core/src/strang2d_parallel.rs

thread_local! {
    #[doc(hidden)]
    pub static PARALLEL_2D_POOL_F64: RefCell<ScratchPool<f64>> =
        RefCell::new(ScratchPool::new());
    #[doc(hidden)]
    pub static PARALLEL_2D_POOL_F32: RefCell<ScratchPool<f32>> =
        RefCell::new(ScratchPool::new());
}

/// Trait dispatching the right per-precision thread-local pool.
/// Sealed — only f32 and f64 may implement.
pub(crate) trait ParallelPool2D: SemiflowFloat {
    fn with_pool<R>(f: impl FnOnce(&mut ScratchPool<Self>) -> R) -> R;
}

impl ParallelPool2D for f64 {
    fn with_pool<R>(f: impl FnOnce(&mut ScratchPool<f64>) -> R) -> R {
        PARALLEL_2D_POOL_F64.with(|cell| f(&mut cell.borrow_mut()))
    }
}

impl ParallelPool2D for f32 {
    fn with_pool<R>(f: impl FnOnce(&mut ScratchPool<f32>) -> R) -> R {
        PARALLEL_2D_POOL_F32.with(|cell| f(&mut cell.borrow_mut()))
    }
}
```

Then `x_pass_chunk<X, F>` calls `F::with_pool(|pool| pool.take_vec(nx))`
instead of `PARALLEL_2D_POOL.with(...)`.

Same pattern for 3D: `ParallelPool3D` with `PARALLEL_3D_POOL_F64`,
`PARALLEL_3D_POOL_F32`.

### 4.2 Drain hooks

```rust
pub fn drain_thread_local_pools_2d<F: ParallelPool2D>() {
    F::with_pool(|pool| *pool = ScratchPool::new());
}
```

Users call `drain_thread_local_pools_2d::<f64>()` or `<f32>()`. The Wave 2
test `tests/parallel_scratch_drain.rs` extends to cover both precisions.

### 4.3 Why not `OnceCell` + per-thread `LazyLock`

`OnceCell` does not auto-clean on thread exit; long-running processes would
leak. `thread_local!` matches the v1.0.0 design and the std-lib mechanism is
clean across `thread::scope` (the scope's spawned workers inherit nothing,
but each worker hits its own thread-local on first borrow — same as v1.0.0).

### 4.4 Memory impact

A process that uses Strang2D only on f64 has:
- `PARALLEL_2D_POOL_F64`: high-water mark per worker thread (~6 MiB on 1024²)
- `PARALLEL_2D_POOL_F32`: empty `ScratchPool` per worker thread (~24 bytes
  per thread for the `Vec<Vec<f32>>` header)

The f32 pool's idle overhead is negligible. Symmetric for f64-only Strang3D.

### 4.5 Sealing `ParallelPool2D`/`ParallelPool3D`

The traits MUST be sealed via `pub(crate)` (no `pub`) so downstream crates
cannot add precision variants. `SemiflowFloat` is already sealed per ADR-0026
§"Sealing"; this is a defensive belt-and-braces.

---

## §5 — Optional WASM SharedArrayBuffer hook

### 5.1 Feature gate

`crates/semiflow-wasm/Cargo.toml` adds:

```toml
[features]
default = []
wasm-sab = []
```

### 5.2 API surface (under `cfg(feature = "wasm-sab")`)

```rust
// crates/semiflow-wasm/src/state.rs

#[cfg(feature = "wasm-sab")]
#[wasm_bindgen]
impl Heat1D {
    /// Evolve in place into a SharedArrayBuffer-backed Float64Array.
    ///
    /// Requires `crossOriginIsolated` on the host page (COOP + COEP headers).
    /// If the underlying buffer is not a SAB, falls back to copy mode.
    pub fn evolve_sab(&mut self, buf: js_sys::Float64Array, tau: f64, n_steps: usize)
        -> Result<(), JsValue> {
        // Probe: buf.buffer() instanceof SharedArrayBuffer && contiguous
        let is_sab = js_sys::Reflect::get(&buf.buffer(), &"constructor".into())
            .ok()
            .and_then(|c| c.dyn_into::<js_sys::Function>().ok())
            .map(|c| c.name() == "SharedArrayBuffer")
            .unwrap_or(false);
        if !is_sab {
            return self.evolve_copy_fallback(buf, tau, n_steps);
        }
        // SAFETY: `view_mut_raw` is sound on SAB-backed Float64Array if no
        // concurrent JS-side write occurs during the call. Caller's
        // responsibility (mirrors the FFI caller-owned contract §2).
        let mut sab_view = unsafe { buf.view_mut_raw() };
        // ... run in-place evolve on sab_view ...
        Ok(())
    }
}
```

### 5.3 Deferred to v2.1 condition

If implementation cost exceeds 2 days (probe correctness on Safari, FF, Chrome;
COOP/COEP test harness in `wasm-bindgen-test`), engineer pass MAY defer §5
to v2.1. ADR-0045 records this as an explicit allowance.

### 5.4 Documentation

If shipped, `crates/semiflow-wasm/README.md` gains a §"SharedArrayBuffer" with
COOP/COEP header examples for Vite, Next.js, and a static `nginx.conf`
snippet.

---

## §6 — Test plan

### 6.1 `tests/python_zero_copy.rs` (NEW)

3 × 2 × 2 = 12-cell matrix:
| dtype | layout | stride | expected path |
|---|---|---|---|
| f64 | C | contiguous | **zero-copy** |
| f64 | C | `[::2]` | copy fallback + warn |
| f64 | C | `[::-1]` | copy fallback + warn |
| f64 | F (column-major) | contiguous | copy fallback + warn (numpy `is_standard_layout()` returns false) |
| f32 | * | * | reject with TypeError (Heat2D/3D only accept f64 v2.0) |
| object | * | * | reject with TypeError |

For each cell, assert:
1. Output values match reference (computed via `evolve` + `to_pyarray`).
2. Tracing event emitted iff fallback (verified via `tracing-test` subscriber).
3. No panic, no UB (Miri ignored — too slow for PyO3).

### 6.2 `tests/ffi_caller_owned.rs` (NEW)

```rust
#[test]
fn caller_owned_matches_v1_evolve() {
    // Smoke: 1D heat, N=64, n_steps=10, tau=0.001
    let mut buf_a = vec![/* gaussian init */];
    let mut buf_b = buf_a.clone();

    // Path A: v1.0.0 evolve + values
    unsafe {
        let mut state_a = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(-1.0, 1.0, 64, buf_a.as_ptr(), 64, &mut state_a);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve(state_a, 0.01, 10);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_state_values(state_a, buf_a.as_mut_ptr(), 64);
        assert_eq!(rc, SemiflowStatus::Ok);
        smf_state_free(state_a);
    }

    // Path B: caller-owned
    unsafe {
        let mut state_b = std::ptr::null_mut();
        let rc = smf_state_new_heat_1d_unit(-1.0, 1.0, 64, buf_b.as_ptr(), 64, &mut state_b);
        assert_eq!(rc, SemiflowStatus::Ok);
        let rc = smf_evolve_inplace(state_b, buf_b.as_mut_ptr(), 64, 0.01, 10);
        assert_eq!(rc, SemiflowStatus::Ok);
        smf_state_free(state_b);
    }

    // Byte-for-byte equality
    assert_eq!(buf_a, buf_b, "caller-owned must match v1.0.0 evolve byte-for-byte");
}
```

Edge cases:
- `buf == null` → `NullPtr`.
- `buf_len != state_size` → `GridMismatch`.
- `tau < 0`, `tau` NaN/Inf → `OutOfDomain`.
- `n_steps == 0` → `OutOfDomain`.
- Panic injection (via test-only inject hook) → `Panic = 99`.

### 6.3 `tests/generic_float_strang.rs` (NEW)

```rust
#[test] fn strang2d_self_convergence_f64() {
    let slope = strang2d_slope_sweep::<f64>(/* ... */);
    assert!(slope <= -1.95, "f64 Strang2D slope={slope} < -1.95");
}

#[test] fn strang2d_self_convergence_f32() {
    let slope = strang2d_slope_sweep::<f32>(/* ... */);
    assert!(slope <= -1.80, "f32 Strang2D slope={slope} < -1.80");  // band per ADR-0046
}

// Same pattern for Strang3D.
```

### 6.4 `tests/strang2d_parallel_bit_equal_f32.rs` (NEW)

Mirror of `tests/strang2d_parallel_bit_equal.rs` but with `F = f32`. Asserts
`Strang2D<X, Y, f32>::apply == strang2d_serial::<f32>::apply` byte-equal on
the 5×3 grid×thread matrix.

### 6.5 Existing regression gates re-pass

- `tests/strang2d_parallel_bit_equal.rs` (f64) — byte-identical to v1.0.0
  snapshot.
- `tests/strang_inplace_byte_equal.rs` (W2 regression) — unchanged.
- `tests/parallel_scratch_drain.rs` (W2) — extended for f32 pool.
- All 18 sympy + 6 f64 slope gates — green.

---

## §7 — Risk table

| ID | Risk | Severity | Detection | Mitigation / rollback |
|---|---|---|---|---|
| R1 | f64 byte-equality drifts after generic lift | HIGH | `tests/strang2d_parallel_bit_equal.rs` + SHA-256 manifest | Manifest committed in first Wave 5 commit; CI gates on byte-equal |
| R2 | numpy stride probe false-positive admits non-contig buf → UB | MEDIUM | `tests/python_zero_copy.rs` 12-cell matrix; PyO3 `is_standard_layout()` is canonical | Copy fallback is always safe; if probe ever returns true on non-contig, we corrupt — keep probe conservative (require strict `is_standard_layout`) |
| R3 | Per-precision pool causes ABI breakage if `SemiflowFloat` gains variant | LOW | `SemiflowFloat` is sealed per ADR-0026 §"Sealing" | Any new variant is a MAJOR bump and gets its own pool by trait impl |
| R4 | WASM SAB unavailable on Safari pre-iOS 16 | LOW | `wasm-bindgen-test` runs on Node + headless Chrome | Feature flag `wasm-sab` off by default; no-op fallback |
| R5 | f32 sympy gate VACUOUSLY SATISFIED is misread as "broken" | LOW | Rustdoc on `Strang2D::new` for f32 cites ADR-0046 explicitly | `docs/precision-policy.md` is the user-facing summary |
| R6 | `thread_local!` per-precision pool double-borrow under nested `thread::scope` | LOW | Wave 2 design already nests `thread::scope` (X then Y then X); no double-borrow because pool is borrowed-released within each pencil | `RefCell` borrow rules catch at runtime; no soundness issue |
| R7 | New `tracing` transitive dep escalates `semiflow-core` dep budget | NONE | `semiflow-core` does NOT depend on `tracing`; only `semiflow-py` does (PyO3 0.28 already pulls it) | `cargo tree -p semiflow-core` gate in CI confirms ≤3 direct deps |
| R8 | `smf_evolve_inplace` aliases internal state buffer → UB | MEDIUM | Docstring: "buf must not alias internal buffer". Practically impossible (internal `Vec<f64>` is heap, caller's `buf` is their own allocation) | If a malicious caller does pass the same pointer, they get the same UB as any other aliased-mut-borrow. Same risk-class as v1.0.0 `smf_state_values`'s `out_buf` not-aliasing requirement |
| R9 | f32 6th-order gate disabled creates "silent slow convergence" | MEDIUM | `Diffusion6thChernoff<f32>` rustdoc explicitly cites ADR-0046 disabled-gate row | Engineer pass adds compile-time `#[deprecated(note = "f32 6th-order has no asymptotic window; see ADR-0046")]` or rustdoc-only warning |

---

## §8 — Acceptance gate

Wave 5 is COMPLETE when:

1. ADR-0045, ADR-0046 land on master.
2. `contracts/v2/wave5-bindings.md` (this doc), `contracts/v2/wave5-precision-policy.md`
   (sister doc), and `docs/precision-policy.md` exist.
3. All test §6.1–§6.5 land green on stable + MSRV 1.78.
4. `tests/strang2d_parallel_bit_equal.rs` (f64) byte-identical to v1.0.0
   manifest.
5. `cargo run -p xtask -- test-full` green (release + parallel + simd + slow-tests).
6. `cargo run -p xtask -- ffi-headers` regenerates `remizov.h` with
   `smf_evolve_inplace`; drift gate green.
7. `cargo run -p xtask -- py-smoke` exercises `evolve_into` happy + fallback
   path; both paths produce byte-equal results to v1.0.0 `evolve`.
8. `cargo run -p xtask -- check-lints` passes; `unsafe_code = "deny"` holds for
   `semiflow-core`; pre-existing FFI/PyO3 carve-outs unchanged.
9. File caps respected per constitution Override #1:
   - `strang2d.rs` ≤ 500
   - `strang3d.rs` ≤ 700 (Override #1 carve-out)
   - `ffi.rs` ≤ 715 (Override #1 carve-out — gains ~40 LoC for `smf_evolve_inplace`)
   - All `semiflow-py` files ≤ 500.
10. Documentation updates landed:
    - `crates/semiflow-py/README.md` documents `evolve_into` + tracing snippet.
    - `crates/semiflow-ffi/README.md` documents `smf_evolve_inplace`.
    - `docs/precision-policy.md` exists and is linked from
      `crates/semiflow-core/src/lib.rs` crate-level rustdoc.
11. ROADMAP.md marks Wave 5 closed; release coordination note added (see §9).

---

## §9 — Release coordination follow-up (v2.0.0-rc.1)

After Wave 5 ships and acceptance gate §8 turns green, the next step is the
v2.0.0 release coordination — out of architect scope, but flagged here for
explicit handoff:

1. **Workspace bump**: `crates/semiflow-core/Cargo.toml` (and sibling crates)
   `version = "2.0.0-rc.1"`.
2. **Reviewer-suckless audit gate**: full v2.0 audit per
   `.dev-docs/v1_0_0/v0_12_0-audit-gate-report.md` pattern. Output:
   `.dev-docs/v2_0_0/v2_0_0-rc1-audit-gate-report.md`.
3. **ipo-specialist review (if applicable)**: NORMATIVE math fidelity audit
   against ADR-0045 §"Acceptance Criteria" 1–6. Output: design freeze.
4. **CHANGELOG.md**: v2.0.0 section enumerating breaking changes (only Wave 3
   `State<F>` trait split is breaking; Wave 5 itself is additive).
5. **Migration guide**: `docs/migration/v1-to-v2.md` extended with Wave 5
   §"Zero-copy paths" subsection and `docs/precision-policy.md` link.
6. **Pre-tag manual checks**: `cargo semver-checks` on all four crates;
   `cargo bench --bench evolve_bench` baseline vs v1.0.0; npm pre-flight if
   `semiflow-wasm` ships SAB.
7. **Tag**: `git tag -a v2.0.0-rc.1` after all gates green.

The bump itself, the audit, and the tag are reviewer + maintainer scope, not
architect scope. ADR-0045 + ADR-0046 + this contract are sufficient context
for the audit pass.

---

## §10 — Cross-references

- ADR-0045 (this Wave's primary ADR)
- ADR-0046 (sister: precision bands)
- ADR-0018 (parallel-strang2d bit-equality)
- ADR-0019 (SIMD intrinsics)
- ADR-0025 (Generic-over-Float)
- ADR-0026 (ChernoffFunction trait generic)
- ADR-0028 (FFI/PyO3/WASM v0.10)
- ADR-0031 (PyO3 GIL release)
- ADR-0035 (v1.0.0 API stability)
- ADR-0041 (Scratch arena)
- ADR-0042 (In-place pencil ping-pong)
- contracts/v2/wave1-scratch.md (NORMATIVE prerequisite)
- contracts/v2/wave2-inplace-strang.md (NORMATIVE prerequisite)
- contracts/v2/wave5-precision-policy.md (sister contract)
- contracts/semiflow-core.math.md §11.1.bis (Strang palindromic structure)
- docs/precision-policy.md (rustdoc-facing summary)
