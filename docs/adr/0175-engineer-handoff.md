# ADR-0175 — Engineer hand-off checklist (f32 first-class + SIMD)

**Companion to**: `docs/adr/0175-f32-first-class-and-simd.md`
**Scope**: implementation only. Strictly additive — never alter the `ChernoffFunction` trait
signature (56 CRITICAL dependents). Suckless: functions ≤50 lines, files ≤500 lines, no new crates
(only `core::arch`). `unsafe` ONLY inside `src/simd/x86_64.rs` + `src/simd/aarch64.rs`.

Before touching any symbol, run `mcp__gitnexus__impact({target: "<symbol>", direction: "upstream"})`
and report blast radius. Do NOT find-and-replace renames — none are needed (additive only).

---

## Phase 5a — additive `impl ChernoffFunction<f32>` on the seven leaf kernels

Each kernel already has a generic `apply_f(&self, tau: F, f: &GridFn1D<F>)` scalar path and a
concrete `impl ChernoffFunction<f64>`. Add a sibling `impl ChernoffFunction<f32>` next to the f64
one, delegating `apply_into` to the kernel's own `apply_f` (copy the f64 impl's `order()` /
`growth()` bodies, retyped to f32). Pattern reference: the retired `WrapDiff<F>` impl in
`crates/semiflow-core/tests/generic_float_strang.rs:69` (delegate to `apply_f`, copy into `dst.values`).
No new math, no SIMD in 5a.

Files to touch (one additive `impl` block each):

1. `crates/semiflow-core/src/diffusion.rs`            — `DiffusionChernoff<f32>` (sibling to impl at line 306)
2. `crates/semiflow-core/src/diffusion4.rs`           — `Diffusion4thChernoff<f32>` (sibling to line 391)
3. `crates/semiflow-core/src/diffusion6.rs`           — `Diffusion6thChernoff<f32>` (sibling to line 314)
4. `crates/semiflow-core/src/drift_reaction.rs`       — `DriftReactionChernoff<f32>` (sibling to line 213)
5. `crates/semiflow-core/src/truncated_exp.rs`        — `TruncatedExpDiffusionChernoff<f32>` (sibling to line 219)
6. `crates/semiflow-core/src/truncated_exp4.rs`       — `TruncatedExp4thDiffusionChernoff<f32>` (sibling to line 221)
7. `crates/semiflow-core/src/shift1d.rs`              — `ShiftChernoff1D<f32>` (sibling to line 236)

Then retire the shim:
8. `crates/semiflow-core/tests/generic_float_strang.rs` — delete `WrapDiff<F>` (struct + impl, lines ~63–92)
   and replace its two construction sites (`WrapDiff(DiffusionChernoff::<f64/f32>::new(...))`, ~lines 193,
   200, 241, 248) with the bare `DiffusionChernoff::<F>::new(...)` now that f32 implements the trait directly.

5a is correctness/API only — covered by the existing generic-float tests (no new gate). 5a may merge
before 5b.

---

## Phase 5b — f32 SIMD kernel wired into the hot paths

### 5b.1 — `crates/semiflow-core/src/simd/mod.rs`
- Add a crate-private `trait SimdF32x8: Copy` (AVX2) and `trait SimdF32x4: Copy` (NEON) mirroring
  `SimdF64x4` (lines 96–111): `splat`, `load_unaligned(&[f32; LANES])`, `store_unaligned(&mut [f32; LANES])`,
  lane-wise `add`/`sub`/`mul` (NO FMA), `horizontal_sum() -> f32` with a **hard-fixed addition tree**.
- Add type aliases resolving the fastest impl per arch (mirror the `F64x4` alias block, lines 41–49):
  `F32x8` → `x86_64::F32x8Avx2` (avx2), `aarch64::F32x4Neon` (neon), `scalar::F32x?Scalar` otherwise.
- The `FORCE_SCALAR` thread-local + `with_force_scalar` hook (lines 64–84) already exist and are
  float-agnostic — reuse them; do NOT add a second hook.
- Extend the scalar fallback `mod scalar` with the f32 scalar lane types so `cfg(test)` and non-SIMD
  arches resolve (mirror `F64x4Scalar`). Its `horizontal_sum` MUST use the identical tree as the
  intrinsic impls and as the per-node f32 scalar reference.

### 5b.2 — `crates/semiflow-core/src/simd/x86_64.rs`
- Add `pub(crate) struct F32x8Avx2(__m256)` impl of `SimdF32x8`, using `_mm256_set1_ps`,
  `_mm256_loadu_ps`, `_mm256_storeu_ps`, `_mm256_add_ps`, `_mm256_sub_ps`, `_mm256_mul_ps`
  (mirror `F64x4Avx2`, lines 26–80). NO `_mm256_fmadd_ps`. `horizontal_sum`: store to `[f32; 8]`,
  reduce in the fixed tree. Add a golden-vector unit test mirroring `avx2_golden_vector` (line 187).
- If a f32 stencil helper is needed, add `apply_g?_stencil_avx2_*` siblings (mirror line 98) — keep ≤50 lines/fn.

### 5b.3 — `crates/semiflow-core/src/simd/aarch64.rs`
- Add `F32x4Neon(float32x4_t)` impl of `SimdF32x4` (NEON has native 4×f32 in one register — simpler
  than the f64 case which packs 2×`float64x2_t`). NO FMA intrinsic. Mirror the f64 NEON impl structure.

### 5b.4 — `crates/semiflow-core/src/grid_cubic.rs`
- Add `catmull_rom_simd_f32` (mirror `catmull_rom_simd`, line 33) using `F32x8`/`F32x4` load+mul+sum,
  and an f32 dispatcher `catmull_rom_f32` mirroring `catmull_rom` (line 52) with the
  `cfg!(test) && FORCE_SCALAR` collapse to `catmull_rom_scalar_f32`. Keep the existing f64 `catmull_rom`
  untouched. The f32 leaf kernels' `apply_f` must route through `catmull_rom_f32` when sampling.

### 5b.5 — `crates/semiflow-core/src/diffusion6_helpers.rs`
- This file is `f64`-specific (`mod helpers_f64`). Add a parallel f32 helper module (or `diffusion6_helpers_f32.rs`
  via `#[path = ...] mod helpers_f32;`) providing `fd9_simd` / `gamma6_a_baseline` f32 siblings using `F32x8`.
  The 9-pt FD splits 8+1 into one `__m256` + a tail (vs the f64 4+4+1 split) — fixed tree, tail handled
  in the same scalar order (no padding garbage in the sum). Wire `Diffusion6thChernoff<f32>::apply_f`
  (added in 5a) to this f32 SIMD path under `feature = "simd"`, scalar otherwise.

---

## Verification commands (run from worktree root `/home/volk/vibeprojects/sf-issue-5`)

```bash
# 5a + 5b compile + fast tests (parallel+simd, opt-level=2)
cargo run -p xtask -- test-fast

# Full validation incl. the new RELEASE-BLOCKING gate SIMD_F32_BIT_EQUAL
cargo run -p xtask -- test-full
#   wraps: RUSTFLAGS="-C target-cpu=native" cargo test --workspace \
#          --features parallel,simd,slow-tests --release

# Run ONLY the new f32 gate
cargo test --features simd --test simd_bit_equal_f32

# Unsafe-scope guard (must show f32 intrinsics confined to src/simd/)
cargo run -p xtask -- check-unsafe-scope

# Confirm the AVX2 f32 path actually builds with avx2 enabled
RUSTFLAGS="-C target-feature=+avx2" cargo build -p semiflow-core --features simd

# Line/file suckless limits
cargo run -p xtask -- check-lints
```

Acceptance: `test-fast` green (5a), `SIMD_F32_BIT_EQUAL` green under `test-full` (5b),
`check-unsafe-scope` clean, `check-lints` clean (≤50-line fns / ≤500-line files).
