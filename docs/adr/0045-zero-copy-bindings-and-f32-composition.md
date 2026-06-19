# ADR-0045: Zero-Copy Bindings + f32 Lift on Strang2D/3D Parallel Composition

**Status**: PROPOSED
**Date**: 2026-05-20
**Architect**: ai-solutions-architect
**Supersedes**: none
**Superseded by**: none
**Related**: ADR-0025 (Generic-over-Float v0.9.0), ADR-0026 (ChernoffFunction generic),
ADR-0028 (FFI/PyO3/WASM v0.10), ADR-0031 (PyO3 GIL release), ADR-0034 (with_closure
API), ADR-0035 (v1.0.0 API stability), ADR-0041 (Scratch arena), ADR-0042 (In-place
pencil ping-pong), ADR-0046 (Precision-policy bands).

---

## Context

By Wave 4, remizov v2.0 has:

1. **Generic-over-F core** (ADR-0025/0026): every `ChernoffFunction<F>` impl and
   every `GridFn{1,2,3}D<F>` parameterised over `F: SemiflowFloat ∈ {f32, f64}`.
2. **Allocation-free composition** (ADR-0041/0042): `apply_into` + `ScratchPool<F>`
   eliminates per-step `Vec` allocations; Strang2D/3D pencil ping-pong is in-place.
3. **PyO3 + FFI bindings frozen at v1.0.0** (ADR-0035): `Heat1D`, `Heat2D`, `Heat3D`
   in `semiflow-py`; 7 `extern "C"` symbols in `semiflow-ffi`; copy-in/copy-out
   semantics everywhere.
4. **Wave 2 carve-out**: `Strang2D`/`Strang3D` parallel impls remain monomorphised on
   `f64` even though their serial impls already accept `F: SemiflowFloat` (per
   ADR-0025). Sites currently locked to `f64`:
   - `strang2d.rs:165` — `impl<X, Y> ChernoffFunction<f64> for Strang2D<X, Y, f64>`
   - `strang2d.rs:182-190` — `apply_into` parallel override (f64-only)
   - `strang3d.rs:168` — `impl<X, Y, Z> ChernoffFunction<f64> for Strang3D<X, Y, Z, f64>`
   - `strang3d.rs:185-193` — `apply_into` parallel override (f64-only)
   - `strang2d_parallel.rs:43,140,210,249,256` — `state: &mut [f64]`, `tau: f64`,
     `temp: &mut Vec<f64>`, `PARALLEL_2D_POOL: ScratchPool<f64>`
   - `strang3d_parallel.rs:46,124,200,346` — same pattern, including
     `PARALLEL_3D_POOL: ScratchPool<f64>`

The Wave 2 in-place pencil ping-pong gave us per-thread `ScratchPool<f64>` pools.
Lifting the f64 lock means parameterising those pools over `F: SemiflowFloat`
without losing the SIMD bit-equality contract (ADR-0018) or the byte-equal
regression gate from Wave 2.

Separately, the Python and C boundaries pay an O(N·8) memcpy on **every**
`evolve` call: PyO3 extracts `u0: Vec<f64>` and re-emits a fresh `numpy.ndarray`;
FFI requires the caller to copy values out with `smf_state_values`. For a
1024² grid at 100 steps, this is 8 MiB × 200 marshalls = 1.6 GB of needless
bandwidth. Profiling on the v0.13.0 Python heat-2D benchmark shows
copy-in/copy-out at 22 % of wall time at the (nx=1024, ny=1024, n_steps=10)
working point.

---

## Decision

Wave 5 makes four additive changes and one breaking lift (with backward-compat
default):

### 5.1 Python zero-copy `evolve_into(buffer)`

Add `Heat2D.evolve_into(&mut self, buf, tau, n_steps)` and the analogous
`Heat3D.evolve_into`. Signature (PyO3):

```rust
fn evolve_into<'py>(
    &self,
    py: Python<'py>,
    buf: PyReadwriteArray1<'py, f64>,   // mutable borrow, contig probe
    tau: f64,
    n_steps: usize,
) -> PyResult<()>
```

**Contract (informal, full spec in `contracts/v2/wave5-bindings.md` §1)**:

1. Probe `buf` for **dtype == f64**, **C-contiguous (row-major) strides**, and
   **`len == nx*ny` (2D) or `nx*ny*nz` (3D)**.
2. If all three hold → call `as_array_mut()`, obtain `ArrayViewMut1<f64>`,
   convert to `&mut [f64]`, and pass into the in-place Strang path (ADR-0042
   pencil ping-pong). **Zero copy.**
3. If any check fails → emit `tracing::warn!(target: "semiflow::zero_copy",
   reason = "{dtype|stride|len}", ...)`, allocate an owned `Vec<f64>`, copy in,
   run the in-place path on the owned vec, copy out into `buf`. **Copy
   fallback**, preserves correctness; user sees a one-line warning.
4. The existing `evolve(u0, tau, n_steps) -> ndarray` stays unchanged
   (additive — `evolve_into` is a sibling method, never replaces `evolve`).

The 1D `Heat1D` class is **out of scope** for Wave 5 (existing `evolve` already
mutates in place; the win is sub-1 % on a 1024-node grid).

### 5.2 FFI caller-owned buffer mode

Add one new `extern "C"` symbol — additive to the v1.0.0 ABI:

```c
SemiflowStatus smf_evolve_inplace(
    SemiflowState* state,
    double* buf,           // caller-owned, mutated in place
    size_t buf_len,        // must equal smf_state_size(state)
    double tau,
    size_t n_steps
);
```

**Existing `smf_evolve` is preserved unchanged** (v1.0.0 freeze maintained
per ADR-0035 — this is the additive escape hatch promised in ADR-0028 §"Out of
scope"). The new entry point borrows `buf` for the duration of the call only;
no ownership transfer.

**Panic boundary**: same `catch_panic!` macro as the rest of `ffi.rs`; release
profile MUST stay `[profile.release-ffi]` with `panic = "unwind"` (already in
workspace per ADR-0028 §"Build requirement").

### 5.3 Lift f64 lock on `Strang2D`/`Strang3D` parallel paths

Generalise:

```rust
// BEFORE (current):
#[cfg(feature = "parallel")]
impl<X, Y> ChernoffFunction<f64> for Strang2D<X, Y, f64> { ... }

// AFTER (Wave 5):
#[cfg(feature = "parallel")]
impl<X, Y, F: SemiflowFloat + Send + Sync> ChernoffFunction<F> for Strang2D<X, Y, F>
where
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{ ... }
```

Concrete edits (specs only — implementation in Wave 5 engineer pass):

1. **`strang2d_parallel.rs`**: parameterise `parallel_x_pass`, `parallel_y_pass`,
   `x_pass_chunk`, `y_apply_cols`, `y_phase{1,2}_*`, `scatter_rows` over
   `F: SemiflowFloat + Send + Sync`. Mechanical lift: `f64` → `F`, `0.0_f64` →
   `F::zero()`, `[f64]` → `[F]`, `Vec<f64>` → `Vec<F>`.

2. **`strang3d_parallel.rs`**: same lift across all three pass families
   (X, Y, Z) and the scatter.

3. **Thread-local pool**: the existing `thread_local! PARALLEL_2D_POOL:
   RefCell<ScratchPool<f64>>` cannot itself be generic over `F` (Rust's
   `thread_local!` macro forbids generic statics). Resolution: introduce
   **per-precision pools**, one per concrete `F`:

   ```rust
   thread_local! {
       #[doc(hidden)]
       pub static PARALLEL_2D_POOL_F64: RefCell<ScratchPool<f64>> = RefCell::new(ScratchPool::new());
       #[doc(hidden)]
       pub static PARALLEL_2D_POOL_F32: RefCell<ScratchPool<f32>> = RefCell::new(ScratchPool::new());
   }

   pub(crate) trait ParallelPool<F: SemiflowFloat>: Sized {
       fn with_pool<R>(f: impl FnOnce(&mut ScratchPool<F>) -> R) -> R;
   }
   impl ParallelPool<f64> for f64 { /* uses PARALLEL_2D_POOL_F64 */ }
   impl ParallelPool<f32> for f32 { /* uses PARALLEL_2D_POOL_F32 */ }
   ```

   The trait dispatches the right pool per concrete `F` at compile time. Pool
   growth is independent (f32 work never touches f64 capacity and vice versa).
   See `contracts/v2/wave5-bindings.md` §4 for the full pseudocode and a sketch
   of an `OnceCell`-based alternative if `thread_local!` proves brittle in
   nested `thread::scope` (it should not — see §4 risk row).

4. **Drain hooks**: `drain_thread_local_pools_2d` becomes
   `drain_thread_local_pools_2d::<F>()`. Same pattern for 3D. Existing test
   `tests/parallel_scratch_drain.rs` extended with f32 cases.

5. **Strang2D/Strang3D parallel impls**: drop the `f64` monomorphism on the
   `where` clause; the `apply_parallel` body already uses `F::zero()` in its
   non-parallel sibling — extract a small helper or duplicate-and-lift.

**Bit-equality contract (ADR-0018)**: The SIMD path (`diffusion.rs` SIMD
intrinsics, ADR-0019) stays **f64-specialised** in its concrete impls. Strang2D
on f64 calls the same `parallel_x_pass<DiffusionChernoff<f64>>` body it always
did (no codegen change for the f64 monomorphisation — only the *type signature*
generalises). The existing `tests/strang2d_parallel_bit_equal.rs` MUST stay
green byte-for-byte.

### 5.4 Optional WASM SharedArrayBuffer hook (gated, may defer)

`semiflow-wasm` currently allocates a fresh `Float64Array` on every `evolve`
return. With `crossOriginIsolated` pages, callers can pass a SAB-backed
`Float64Array`; if the JS side has already done the SAB dance (COOP/COEP
headers), Rust can map it via `js_sys::Float64Array::view_mut_raw` and operate
in place.

Gated behind `cfg(feature = "wasm-sab")`. If implementation cost exceeds 2
days, **deferred to v2.1**; in that case Wave 5 ships only §5.1–§5.3.

### 5.5 f32 precision-policy bands

Generic-over-F composition raises a math fidelity question: `f32` mantissa
(23 bits) cannot match `f64` (52 bits) on slope-gate tolerances. **ADR-0046**
defines the policy in detail; ADR-0045 records the binding constraint:

- f64 path: all NORMATIVE sympy gates (T9N_*, T10N_*, T11N_*) and slope gates
  unchanged.
- f32 path: sympy gates are **VACUOUSLY SATISFIED** (sympy uses arbitrary
  precision; f32 numerical rounding cannot be probed there). Slope gates run
  on f32 with relaxed tolerances per ADR-0046 §3.
- `tests/generic_float_strang.rs` (new) gates Strang2D/Strang3D self-convergence
  on both `F = f64` and `F = f32`, using the bands from ADR-0046.

---

## Consequences

### Positive
1. **Python hot path −15…−30 %** on 2D/3D evolve (no marshalling on the typical
   `np.zeros(nx*ny, dtype=np.float64)` buffer).
2. **FFI peak memory halved** for callers that pre-allocate (no double-buffer
   in `SemiflowStateInner.values` + caller's buffer).
3. **f32 composition halves grid storage** on 2D/3D for callers willing to
   accept relaxed slope-gate tolerances (mesh refinement studies, parameter
   sweeps).
4. **No new direct deps in semiflow-core** (≤ 3 cap preserved; binding crates
   already depend on numpy/pyo3/wasm-bindgen — unchanged).
5. **Additive across the board**: `evolve` and `smf_evolve` keep their
   v1.0.0 contracts; the new `evolve_into` and `smf_evolve_inplace` are
   opt-in.

### Negative
1. **Per-precision thread-local pools**: doubles the thread-local static count.
   Each pool's idle memory is `O(max_buf_observed × pool_depth)`; on a CPU with
   16 threads and a 1024² grid, the f64 pool's high-water mark is ~96 MiB
   (already true at v1.0.0). Adding a parallel f32 pool that is never used
   costs **8 bytes** (an empty `Vec<Vec<f32>>`).
2. **f32 path has weaker math gates**: ADR-0046 documents the carve-out.
   Misuse risk: a user runs f32 in production, sees no warnings, but accepts
   convergence at a degraded slope band. Mitigation: rustdoc on
   `Strang2D::new` for `F = f32` cites ADR-0046; the new
   `tests/generic_float_strang.rs` runs both bands so CI catches drift.
3. **Stride-edge cases**: a Python user passes a non-contiguous slice
   (`buf[::2]`); we warn and fall back to copy. The warning is opt-in via
   `tracing-subscriber`; users running without a subscriber see no signal.
   Mitigation: `Heat2D.evolve_into` docstring documents the requirement;
   `pytest` integration test asserts on a known-bad input (`tests/python_zero_copy.rs`
   §6).
4. **WASM SAB**: COOP/COEP setup is a deployment hassle for downstream;
   we document but do not gate on it.

### Risks (full matrix in `contracts/v2/wave5-bindings.md` §7)
- **R1 (HIGH)**: f64 byte-equality on parallel Strang2D/Strang3D drifts after
  generic lift. Mitigation: ADR-0018 bit-equal proptest stays in CI; an
  additional `tests/strang_inplace_byte_equal_v1.rs` snapshot v1.0.0 outputs
  for 5 grid sizes × 3 thread counts.
- **R2 (MEDIUM)**: numpy stride probe accepts a buffer that fails contiguity
  at runtime → UB. Mitigation: PyO3's `PyReadwriteArray::is_contiguous()`
  helper; matrix test covers all 4 stride patterns.
- **R3 (MEDIUM)**: per-precision pool causes ABI breakage if `SemiflowFloat`
  gains a new variant. Mitigation: `SemiflowFloat` is `sealed` per ADR-0026
  §"Sealing"; only the crate itself can add variants, and any addition is a
  major version bump.
- **R4 (LOW)**: WASM SAB unavailable on Safari pre-iOS 16. Mitigation: feature
  flag, no-op fallback when not provided.

---

## Acceptance Criteria

1. All 18 NORMATIVE sympy gates (T9N_*, T10N_*, T11N_*) re-pass on f64
   composition. (sympy: no f32 carve-out.)
2. All 6 slope gates re-pass on f64 (current tolerance) and pass on f32 at
   bands from ADR-0046.
3. `tests/strang2d_parallel_bit_equal.rs` byte-identical to v1.0.0 SHA-256
   manifest (the manifest is committed in Wave 5 first commit).
4. `tests/python_zero_copy.rs`: matrix of (dtype ∈ {f32, f64}, stride ∈
   {contig, ::2, ::-1}, layout ∈ {C, F}) covers all 12 cells; only
   (f64, contig, C) takes the zero-copy path; all others warn + copy + still
   produce a correct result.
5. `tests/ffi_caller_owned.rs`: `smf_evolve_inplace` returns correct values
   byte-for-byte vs `smf_evolve` + `smf_state_values` (existing flow)
   for the smoke grid `(N=64, n_steps=10)`.
6. `tests/generic_float_strang.rs`: Strang2D and Strang3D self-convergence
   slope ≥ −1.95 on f64, ≥ −1.80 on f32 (per ADR-0046 §3).
7. `cargo run -p xtask -- test-fast` green; `test-full` green; new tests pass
   on stable + MSRV 1.78.
8. `cargo run -p xtask -- ffi-headers` shows the new symbol in `remizov.h`
   (cbindgen drift check stays green).
9. `cargo run -p xtask -- py-smoke` exercises `evolve_into` happy + fallback
   path.
10. `cargo run -p xtask -- check-lints` passes; no new `unsafe_code` outside
    pre-existing carve-outs (`simd/`, `ffi/`, `py/`).
11. File caps respected: `strang2d.rs` ≤ 500, `strang3d.rs` ≤ 700
    (constitution Override #1 carve-out), `ffi.rs` ≤ 715 (Override #1
    carve-out for ffi.rs), all binding files ≤ 500.

---

## Alternatives Considered

- **A1: Make `ScratchPool<F>` itself a non-thread-local hand-passed object**
  through the parallel path. Rejected: it would force a major API change on the
  public `apply_parallel` entry point and re-thread every `thread::scope`
  worker; the per-precision thread-local is mechanical and contained.
- **A2: Use `OnceCell` instead of `thread_local!` for the parallel pool**.
  Rejected for v2.0: `OnceCell` does not bind to the thread, so a worker
  spawning would re-allocate. `thread_local!` matches Wave 2 design and we
  have a working prototype.
- **A3: Defer f32 parallel until v2.1**. Rejected: serial Strang2D/3D is
  already f32-capable (ADR-0025) — leaving the parallel feature f64-only would
  create a permanent asymmetry. Strang*2D parallel f64 byte-equality is
  preserved by construction; lifting now is cheap.
- **A4: Lift f32 sympy gates with relaxed tolerance**. Rejected: sympy is
  arbitrary-precision; the "gate" is mathematical, not numerical. Re-deriving
  with f32 rounding would invent a fake oracle. ADR-0046 instead documents
  that f32 has no sympy gate and the slope band is the sole math guarantee.
- **A5: Single-symbol FFI with a "use_caller_buf" flag in an enum**.
  Rejected: would force re-issuing the v1.0.0 frozen `smf_evolve` symbol
  with a new signature → SemVer-major on the C ABI. Sibling additive symbol
  is the cleaner path (matches `pthread_create` vs `pthread_create_attr`
  convention).
- **A6: Wave 5 also covers `semiflow-wasm` zero-copy unconditionally**.
  Rejected: WASM zero-copy requires SharedArrayBuffer + COOP/COEP, which is a
  deployment story, not a code story. Gated behind `cfg(feature = "wasm-sab")`
  and documented as optional. Engineer may defer to v2.1 if implementation
  exceeds 2 days.

---

## Out of Scope

- **Heat1D zero-copy**: 1D evolve uses a single owned `Vec<f64>` already
  resident in `SemiflowStateInner`; the marshalling overhead is ~1 % on a
  1024-node grid. Not worth the API growth.
- **Variable-`a(x)` zero-copy**: `with_a_function` callbacks dominate runtime
  (ADR-0031 §"Performance note"); copy overhead is invisible at <0.1 % of
  the call cost.
- **Async FFI / async Python**: deferred to v2.1+ per ADR-0028 §"Out of scope".
- **f16 / bf16 composition**: ADR-0025 sealed `SemiflowFloat` at `{f32, f64}`;
  adding low-precision types requires a separate ADR and full sympy
  re-derivation.

---

## References

- ADR-0018 (parallel-strang2d), ADR-0025 (generic-over-f), ADR-0026
  (chernoff-trait-generic), ADR-0028 (ffi-pyo3-wasm-v0.10), ADR-0031
  (pyo3-gil-release), ADR-0034 (with-closure-api), ADR-0035 (v1.0.0-api-stability),
  ADR-0041 (scratch-arena), ADR-0042 (inplace-strang), ADR-0046
  (precision-policy-bands).
- `contracts/v2/wave5-bindings.md` (NORMATIVE Wave 5 contract).
- `contracts/v2/wave5-precision-policy.md` (slope-tolerance bands per
  precision per gate).
- `docs/precision-policy.md` (rustdoc-facing summary of ADR-0046).
- PyO3 0.28 docs: `PyReadwriteArray`, `Bound::extract`, `py.detach`.
- numpy crate 0.28 docs: `as_array_mut`, `is_contiguous`.
- math.md §11.1.bis (Strang palindromic structure; untouched by Wave 5).
