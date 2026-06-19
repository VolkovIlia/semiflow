# Wave 2 Contract: In-Place Pencil Ping-Pong for Strang Composition

**Status**: NORMATIVE
**ADR**: docs/adr/0042-inplace-strang-pencil-pingpong.md
**Scope**: semiflow-core v2.0 Wave 2
**Depends on**: contracts/v2/wave1-scratch.md (NORMATIVE prerequisite)

---

## §1 — `pencil.rs` API (slice-based pencil views over `GridFnXD::values`)

**New module**: `crates/semiflow-core/src/pencil.rs` (~120 LoC).

Provides stride-aware slice-iterator utilities that let `AxisLift::apply_into` and `AxisLift3D::apply_into` read/write a single pencil without allocating a `GridFn1D`.

### 1.1 X-pencil utilities (contiguous, stride = 1)

```rust
// 2D, x-fastest row-major: values[j*nx + i]
#[inline]
pub(crate) fn row_2d<F: SemiflowFloat>(values: &[F], nx: usize, j: usize) -> &[F] {
    &values[j*nx .. (j+1)*nx]
}

#[inline]
pub(crate) fn row_2d_mut<F: SemiflowFloat>(values: &mut [F], nx: usize, j: usize) -> &mut [F] {
    &mut values[j*nx .. (j+1)*nx]
}

// 3D, x-fastest row-major: values[k*nx*ny + j*nx + i]
#[inline]
pub(crate) fn pencil_x_3d<F: SemiflowFloat>(values: &[F], nx: usize, ny: usize, j: usize, k: usize) -> &[F] {
    let start = k*nx*ny + j*nx;
    &values[start .. start + nx]
}

#[inline]
pub(crate) fn pencil_x_3d_mut<F: SemiflowFloat>(values: &mut [F], nx: usize, ny: usize, j: usize, k: usize) -> &mut [F] {
    let start = k*nx*ny + j*nx;
    &mut values[start .. start + nx]
}
```

### 1.2 Y-pencil utilities (strided, stride = nx)

Y-pencils require gather/scatter; the slice API yields a strided iterator over `&[F]` with a `gather_into(slot)` and `scatter_from(slot)` pair:

```rust
/// Gather Y-pencil (i, k) into `slot` (length ny). 3D version; 2D version is k=0.
#[inline]
pub(crate) fn gather_y_3d_into<F: SemiflowFloat>(
    values: &[F], nx: usize, ny: usize, i: usize, k: usize, slot: &mut [F]
) {
    debug_assert!(slot.len() == ny);
    let base = k * nx * ny + i;
    for j in 0..ny {
        slot[j] = values[base + j*nx];
    }
}

/// Scatter `slot` back to Y-pencil (i, k).
#[inline]
pub(crate) fn scatter_y_3d_from<F: SemiflowFloat>(
    values: &mut [F], nx: usize, ny: usize, i: usize, k: usize, slot: &[F]
) {
    debug_assert!(slot.len() == ny);
    let base = k * nx * ny + i;
    for j in 0..ny {
        values[base + j*nx] = slot[j];
    }
}

/// 2D Y-pencil at column i: gather slot (length ny).
#[inline]
pub(crate) fn gather_y_2d_into<F: SemiflowFloat>(values: &[F], nx: usize, ny: usize, i: usize, slot: &mut [F]) {
    debug_assert!(slot.len() == ny);
    for j in 0..ny { slot[j] = values[j*nx + i]; }
}

#[inline]
pub(crate) fn scatter_y_2d_from<F: SemiflowFloat>(values: &mut [F], nx: usize, ny: usize, i: usize, slot: &[F]) {
    debug_assert!(slot.len() == ny);
    for j in 0..ny { values[j*nx + i] = slot[j]; }
}
```

### 1.3 Z-pencil utilities (strided, stride = nx·ny)

```rust
#[inline]
pub(crate) fn gather_z_3d_into<F: SemiflowFloat>(
    values: &[F], nx: usize, ny: usize, nz: usize, i: usize, j: usize, slot: &mut [F]
) {
    debug_assert!(slot.len() == nz);
    let base = j * nx + i;
    let stride = nx * ny;
    for k in 0..nz {
        slot[k] = values[base + k * stride];
    }
}

#[inline]
pub(crate) fn scatter_z_3d_from<F: SemiflowFloat>(
    values: &mut [F], nx: usize, ny: usize, nz: usize, i: usize, j: usize, slot: &[F]
) {
    debug_assert!(slot.len() == nz);
    let base = j * nx + i;
    let stride = nx * ny;
    for k in 0..nz {
        values[base + k * stride] = slot[k];
    }
}
```

**Visibility**: all `pub(crate)`. Not part of the v1.0.0 stable API (per ADR-0035). Frozen as `pub(crate)` so future SIMD-on-strided-pack rewrites can replace internals without breaking semver.

**Stride math (NORMATIVE)** — must match `GridFn3D::idx` (verified `grid_fn3d.rs:4` doc: `idx(i,j,k) = k*nx*ny + j*nx + i`):

| Pencil | Stride | Base | Length |
|--------|-------:|------|-------:|
| X-pencil 2D `(j)` | 1 | `j*nx` | `nx` |
| Y-pencil 2D `(i)` | `nx` | `i` | `ny` |
| X-pencil 3D `(j,k)` | 1 | `k*nx*ny + j*nx` | `nx` |
| Y-pencil 3D `(i,k)` | `nx` | `k*nx*ny + i` | `ny` |
| Z-pencil 3D `(i,j)` | `nx*ny` | `j*nx + i` | `nz` |

---

## §2 — Strang2D in-place 3-leg ping-pong

### 2.1 Buffer-parity rule

Two working buffers `A` (source) and `B` (dest), aliased to two `&mut [F]` slices over distinct `Vec<F>` backing storage. After each leg, **swap** the role: the destination of leg N becomes the source of leg N+1.

```
Leg 1: X(τ/2): src = A, dst = B  →  B holds X(τ/2) · A
Leg 2: Y(τ)  : src = B, dst = A  →  A holds Y(τ)  · B
Leg 3: X(τ/2): src = A, dst = B  →  B holds X(τ/2) · A
Final:   `dst` parameter copy-from B
```

The final copy is required because `dst: &mut GridFn2D<F>` is the user's destination buffer, which may be a third storage independent of the two scratch buffers.

### 2.2 Storage acquisition

`A` is initialised by `dst.values.clone_from_slice(&src.values)` (one O(N²) memcpy — unavoidable for entry, but does NOT allocate). `B` is borrowed from `ScratchPool<F>`.

### 2.3 Palindromic order (NORMATIVE)

`X(τ/2) → Y(τ) → X(τ/2)`. Must match `apply_strang2d_full` lines 277–279.

### 2.4 Allocation profile

- 0 fresh `Vec<F>` allocations in steady state (after first call; pool reuses `buf_b`).
- Per-pencil `Vec<F>` for Y-pass strided gather is borrowed/returned within `apply_axis_into`. Steady-state: pool serves it from free-list.
- Total per `apply_into` call: **0 allocations** after warmup.

---

## §3 — Strang3D in-place 5-leg ping-pong

### 3.1 Buffer-parity rule

```
Init: A = src.clone() (into dst.values)
Leg 1: X(τ/2): src=A, dst=B
Leg 2: Y(τ/2): src=B, dst=A
Leg 3: Z(τ)  : src=A, dst=B
Leg 4: Y(τ/2): src=B, dst=A
Leg 5: X(τ/2): src=A, dst=B
Final copy: dst.values ← B (odd count: final lives in B)
```

Because the 5-leg count is **odd**, the final value lives in `B` (not `A`). The implementation MUST copy back to `dst.values` from `B` after leg 5.

### 3.2 Palindromic order (NORMATIVE)

`X(τ/2) → Y(τ/2) → Z(τ) → Y(τ/2) → X(τ/2)`. Must match `apply_strang3d_full` lines 460–464.

### 3.3 Allocation profile

0 allocations in steady state (same argument as §2.4). Per-pencil 1D scratch borrowed from pool by `apply_axis3d_into`.

---

## §4 — `AxisLift::apply_into` override (2D)

### 4.1 `apply_into_via_view` helper

`pub(crate)` function in `grid_fn.rs` that wraps a slice-based call to a `ChernoffFunction<F, S=GridFn1D<F>>` using pool buffers. Avoids public trait API changes (ADR-0035). Pool steady-state: 0 allocations.

### 4.2 Override pattern

- **X-pass**: contiguous row slices via `pencil::row_2d_mut`; `apply_into_via_view` per row.
- **Y-pass**: strided gather/scatter per column; `core::mem::take` reclaim pattern; pool-owned `src_col` / `dst_col`.

### 4.3 Override semantics

- The leaf `self.inner.apply_into` (Wave 1 override) fires for each pencil.
- For both axes: 0 steady-state allocations.

---

## §5 — `AxisLift3D::apply_into` override (3D)

Lives in `strang3d_axislift.rs` (new file, constitution split).

### 5.1 Override pattern

- **X-pass**: contiguous pencils via `pencil::pencil_x_3d_mut`; `apply_into_via_view` per `(j,k)`.
- **Y-pass**: strided gather/scatter per `(i,k)`; pool-owned buffers.
- **Z-pass**: strided gather/scatter per `(i,j)`; stride = `nx*ny`.

### 5.2 Z-axis stride NORMATIVE check

`base = j*nx + i`, `stride = nx*ny`, `length = nz`. Matches `GridFn3D::idx = k*nx*ny + j*nx + i`.

---

## §6 — Parallel paths: per-thread `ScratchPool<f64>`

### 6.1 Thread-local storage

`thread_local!` `RefCell<ScratchPool<f64>>` in `strang2d_parallel.rs` (`PARALLEL_2D_POOL`) and `strang3d_parallel.rs` (`PARALLEL_3D_POOL`), gated on `feature="parallel"`.

### 6.2 Replacement target

Replace `vec![0.0_f64; n]` per-thread-per-pass with `pool.take_vec(n)` + `pool.return_vec(...)`. Affects `x_pass_chunk`, `y_apply_cols`, `x_pass_chunk_3d`, `y_apply_chunk`, `z_apply_chunk`.

### 6.3 Lifetime (W2 decision: option d)

Pool is per-call (std::thread::scope spawns fresh threads). Pool reuse within one `Strang*::apply` call (across legs); no cross-call reuse until W6 long-lived worker pool.

### 6.4 Test hook

`drain_thread_local_pools()` resets calling thread's pools. `#[doc(hidden)]`, `pub`.

---

## §7 — Capacity policy

Grow-only, no auto-shrink. Mirrors Wave 1 `ScratchPool` policy. `drain_thread_local_pools()` is the only reset path.

---

## §8 — `NonSeparable2DChernoff::apply_into` — Wave 1 retained verbatim

**NORMATIVE**: Do NOT modify `nonseparable2d.rs:166-185` in Wave 2.

---

## §9 — `ChernoffSemigroup<C>::evolve` integration

No changes needed in `ChernoffSemigroup::evolve` — the Wave 1 call to `apply_into` automatically picks up Wave 2 overrides.

---

## §10 — Backward compatibility (v1.0.0 surface)

All `apply` signatures unchanged. `apply_into` overrides are ADDITIVE. 15 test files + 2 bench files compile and pass unchanged.

---

## §11 — Kernel migration table

| Symbol | Strategy |
|--------|----------|
| `Strang2D::apply` (serial) | UNCHANGED |
| `Strang2D::apply_into` (NEW) | 2-buffer ping-pong, 0 allocs steady |
| `Strang3D::apply` (serial) | UNCHANGED |
| `Strang3D::apply_into` (NEW) | 2-buffer ping-pong, 0 allocs steady |
| `Strang2D::apply_parallel` | Replace per-thread `vec![0.0;n]` with pool |
| `Strang3D::apply_parallel` | Replace per-thread `vec![0.0;n]` with pool |
| `AxisLift::apply` | UNCHANGED |
| `AxisLift::apply_into` (NEW) | per-pencil scratch from pool, 0 allocs steady |
| `AxisLift3D::apply` | UNCHANGED |
| `AxisLift3D::apply_into` (NEW, new file) | per-pencil scratch from pool, 0 allocs steady |
| `NonSeparable2DChernoff::apply_into` (Wave 1) | UNCHANGED |

---

## §12 — Test plan

### 12.1 `tests/strang_inplace_byte_equal.rs` (proptest, ~120 LoC)

Proptest: `Strang2D::apply` and `Strang2D::apply_into` produce byte-identical results. Same for Strang3D, AxisLift, AxisLift3D. Min 256 cases per scenario.

### 12.2 `tests/strang_inplace_alloc_count.rs` (~90 LoC)

0 allocs/step after 3 warmup steps for `ChernoffSemigroup<Strang2D>::evolve` and `ChernoffSemigroup<Strang3D>::evolve`.

### 12.3 `tests/parallel_scratch_drain.rs` (~50 LoC, `feature="parallel"`)

Thread-local pool high-water mark settles; `drain_thread_local_pools()` resets and pool re-fills.

### 12.4 Re-run gates (no modification)

- `tests/apply_into_byte_equal.rs` (Wave 1)
- `tests/strang2d_parallel_bit_equal.rs`
- `tests/strang3d_parallel_bit_equal.rs`
- `tests/strang3d_serial_scratch_bit_equal.rs`
- `tests/simd_bit_equal.rs`
- All slope gates (G3, G5_3D, G3⁴, G3⁶-2D, G4_NS2D_aniso)
- `tests/generic_float_smoke.rs`

### 12.5 Miri

`cargo +nightly miri test --test strang_inplace_byte_equal` MUST pass.

---

## §13 — Risk table

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Palindromic leg reorder | Side-by-side proptest catches byte-mismatch |
| R2 | Y-axis stride mismatch (2D) | NORMATIVE stride table; debug_assert on slot len |
| R3 | Z-axis stride mismatch (3D) | Same as R2; `lift_z_strided` existing test |
| R4 | Per-thread pool unbounded growth | Grow-only invariant; settle test in §12.3 |
| R5 | AxisLift default-bridge fallback | 0-alloc test (§12.2) catches any fallthrough |

---

## §14 — Out of scope (deferred)

- OS1: `State<F>` trait split (W3)
- OS2: Generic `f32` parallel paths (W6)
- OS3: FFI zero-copy (W5)
- OS4: AdaptivePI scratch (W4)
- OS5: `NonSeparable2DChernoff` pencilisation (W7)
- OS6: Long-lived worker pool (W6)

---

**Designed-By**: ai-solutions-architect
**Reviewed-By**: pending
**To-be-implemented-by**: agentic-engineer
