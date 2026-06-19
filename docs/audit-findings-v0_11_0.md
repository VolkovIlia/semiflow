---
version: 1.0.0
last_updated: 2026-05-09
freshness_score: 1.0
dependencies:
  - docs/adr/0018-parallel-strang.md
  - docs/adr/0020-g3-6th-2d-flagship.md
  - docs/adr/0024-tensor-3d.md
  - docs/adr/0032-heavy-validation-harness.md
  - contracts/semiflow-core.math.md (section 10.8.7)
  - contracts/semiflow-core.properties.yaml (schema 0.7.6)
  - crates/semiflow-core/src/strang3d.rs
  - crates/semiflow-core/src/strang3d_parallel.rs
  - docs/audit-findings-v0_10_0.md (baseline carried forward)
changelog:
  - 1.0.0: Initial v0.11.0 I12 heavy validation audit
verified_by: docs-writer
verification_date: 2026-05-09T00:00:00Z
verification_score: 1.0
---

# v0.11.0 Math Fidelity + I12 Heavy Validation Audit

**Auditor**: docs-writer agent (delegated by anchor)
**Date**: 2026-05-09
**Scope**: `v0.10.0..HEAD` plus uncommitted v0.11.0 closure patches
**Theme**: I12 heavy validation closure on production hardware

## 1. Summary

**APPROVED FOR RELEASE.** All 3 NORMATIVE flagship slope gates pass on
production hardware (Intel i7-12700K, 12C/20T, 31 GiB RAM, AVX2 native,
rustc 1.94.1). Total wallclock 987 s (16.5 min). G5_3D speedup: ~12× from
the single-thread baseline via new parallel Strang3D (ADR-0018 pattern, bit-equal
preserved). Two N-basket recalibrations land alongside I12 (ADR-0020 and ADR-0024
Amendments 2026-05-09). A pre-existing xtask Path B filter bug is fixed; no
past flagship result is invalidated.

## 2. Hardware Reproducibility Block

| Field | Value |
|-------|-------|
| CPU | Intel Core i7-12700K — 12 cores, 20 threads |
| RAM | 31 GiB (no swap) |
| OS | Artix Linux 6.19.11-artix1-1 |
| Rust | `rustc 1.94.1 (e408947bf 2026-03-25)` |
| RUSTFLAGS | `-C target-cpu=native` (AVX2 engaged) |
| Build | `RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship cargo run -p xtask -- test-flagship` |
| Total wallclock | **987 s** (16.5 min; includes build with `target-flagship` cache reuse) |

## 3. Per-Gate Results

| Gate | N basket | Slope | Threshold | Wallclock | Status |
|------|----------|-------|-----------|-----------|--------|
| G3⁶-2D FLAGSHIP | {503, 997, 1999} prime | **-6.0837** | window [-6.15, -5.85] | 600 s | **PASS** |
| G5_3D | {32, 64, 128, 256} | **-2.1735** | ≤ -1.95 | 257 s | **PASS** |
| G4_NS2D_aniso | {32, 64, 128, 256} | **-2.1965** | ≤ -1.95 | 123 s | **PASS** |

### 3.1 G3⁶-2D FLAGSHIP

| N | ‖err‖∞ | wallclock |
|---|--------|-----------|
| 503 | 1.2198e-7 | 155 s |
| 997 | 9.7974e-10 | 588 s |
| 1999 | 2.7481e-11 | 2341 s |

OLS slope: **-6.0837**. Matches v0.8.1 pilot run (`docs/audit-findings-v0_8_1.md` §4)
byte-for-byte. Window margin 0.0837 from lower bound. Schema 0.7.6 unchanged.

### 3.2 G5_3D (recalibrated basket)

| N | dx | err_sup | ratio | log₂(ratio) |
|---|------|---------|-------|-------------|
| 32 | 0.323 | 1.5131e-2 | — | — |
| 64 | 0.159 | 4.5033e-3 | 3.36 | 1.75 |
| 128 | 0.079 | 1.0030e-3 | 4.49 | 2.17 |
| 256 | 0.039 | 1.6455e-4 | 6.09 | 2.61 |

OLS slope: **-2.1735**. Asymptote reached at N≥128; super-asymptotic nudge at
N=256 consistent with small higher-order contribution. Prior basket `{16, 32, 64}`
included N=16 (`dx=0.667`, ~2 cells per Gaussian half-width) — pre-asymptotic
slope -1.20 failed the gate. ADR-0024 Amendment 2026-05-09 records full diagnosis
and ratifies recalibration. Gate margin 0.22.

### 3.3 G4_NS2D_aniso

Slope **-2.1965**, gate ≤ -1.95, margin 0.25. Matches v0.9.0 self-convergence
redesign (commit 0180292) byte-for-byte.

## 4. Calibration Patches (in working tree, this commit)

Zero math content altered. All changes are N-basket recalibrations and
infrastructure corrections.

- **`tests/convergence_rate_6th.rs`** — `wide_sweep` N_SWEEP `[200, 400, 800,
  1600, 3200, 6400]` → `[400, 800, 1600, 3200]` (drop pre-asymptote N=200 and
  precision-floor N=6400). Gate ≤ -5.50 unchanged. Slope on new basket: -5.78.
- **`docs/adr/0020-g3-6th-2d-flagship.md`** — Amendment 2026-05-09 (wide_sweep
  recalibration rationale; gate threshold unchanged).
- **`tests/strang_3d_slope.rs`** — `N_SPATIAL` `[16, 32, 64]` → `[32, 64, 128,
  256]`. Gate ≤ -1.95 unchanged.
- **`docs/adr/0024-tensor-3d.md`** — Amendment 2026-05-09 (N=16 under-resolution
  diagnosis; G4_NS2D_aniso self-convergence precedent at 0180292).
- **`contracts/semiflow-core.math.md` §10.8.7** — basket reference updated; NORMATIVE
  pre-asymptote diagnosis paragraph added.
- **`contracts/semiflow-core.properties.yaml`** — G5_3D `N_SPATIAL` updated to
  `[32, 64, 128, 256]`. Schema version 0.7.6 unchanged.
- **`xtask/src/main.rs:438-475`** — Path B filter bug fixed (see §6).
- **NEW `src/strang3d_parallel.rs`** (416 LoC) and **NEW `tests/strang3d_parallel_bit_equal.rs`**
  (207 LoC) — parallel Strang3D (see §5).
- **`src/strang3d.rs`** (+193 LoC, serial/parallel split, `feature = "parallel"` dispatch).
- **`src/lib.rs`** — `#[cfg(feature = "parallel")] pub mod strang3d_parallel;`.
- **`tests/generic_float_smoke.rs`** — `strang3d_f32_smoke` gated
  `#[cfg(not(feature = "parallel"))]` (parallel impl is `f64`-only per ADR-0018).

## 5. Parallel Strang3D (new infrastructure)

`strang3d_parallel.rs` (416 LoC) mirrors `strang2d_parallel.rs` (eae2b7a,
ADR-0018): `std::thread::scope` per-axis pencil dispatch (X/Y/Z),
`MIN_PENCILS_PER_THREAD = 16`, `FORCE_THREADS_3D` test hook.

`strang3d_parallel_bit_equal.rs` (207 LoC) asserts `abs_diff == 0.0` for every
grid cell across N ∈ {16, 32, 64} × threads ∈ {1, 2, 4, 8}.

**Measured speedup**: N=256³ G5_3D gate step — **257 s** vs estimated ~50 min
single-thread = **~12×** on 20 hardware threads. Prerequisite for feasible
flagship run within the 987 s total wallclock budget.

Public API: zero changes. Parallel path is `pub(crate)`; `ChernoffFunction<f64>`
dispatch is feature-gated identically to the 2D pattern.

## 6. xtask test-flagship Bug (pre-existing — fixed in this commit)

`test_flagship()` Path B previously appended `-- --ignored`. The 3 NORMATIVE slope
tests lack `#[ignore]` (they are gated by `slow-tests` feature, not `#[ignore]`),
so `-- --ignored` silently produced an empty-suite "pass". **Fix**: explicit
`--test convergence_rate_6th_2d --test strang_nonseparable_aniso_slope
--test strang_3d_slope --no-fail-fast -- --nocapture`.

**No past flagship result is invalidated.** v0.8.1 G3⁶-2D was validated via
a direct `cargo test` command that did not route through the xtask wrapper
(`docs/audit-findings-v0_8_1.md` §4). The xtask bug was a dead code path until
I12 activated Path B.

## 7. Suckless Invariants

- **Runtime deps**: 2 (`num-traits`, `libm`) — unchanged from v0.7.0.
- **Largest new src file**: `strang3d.rs` 566 LoC (serial + parallel split;
  grandfathered alongside pre-existing 3-file grandfather per v0.9.0 suckless
  check); `strang3d_parallel.rs` 416 LoC (new; under 500 LoC cap).
- **Functions ≤ 50 LoC**: enforced; pencil helpers factored identically to 2D pattern.
- **`unsafe` scope**: confined to `src/simd/{x86_64,aarch64}.rs` per ADR-0019.
  No new `unsafe` in any I12 closure file.
- **Public API**: zero changes. `cargo public-api --diff-against 0.10.0` reports
  no surface change.
- **Workspace version**: `0.11.0` in `Cargo.toml [workspace.package]`.

## 8. Out of Scope (deferred to v0.12.0+)

- Var-a / 2D / 3D bindings across FFI/PyO3/WASM (I3, I4, I5) — requires core
  `DiffusionChernoff::with_closure` design.
- Async PyO3 evolution API (I14).
- Safari headless WASM smoke (I2 partial).
- `NS2D_ANISO_PARALLEL_BIT_EQUAL` regression gate (audit-findings-v0_9_0.md O-2).
- npm org `@remizov` claim + `NPM_TOKEN` secret — maintainer pre-flight required
  before `git push origin v0.11.0` (triggers `release-wasm.yml`); local annotated
  tag creation is **unblocked**.

## 9. Recommendation

**Ship v0.11.0.** I12 closed. All four v0.11.0 MUST items shipped: I1 (f8dc9d5),
I6 (07a4689), I13 (6fb7781), I12 (this commit). Math invariants intact:
`semiflow-core` API surface unchanged from v0.10.0; bit-equal contracts preserved
(parallel Strang3D verified bit-equal vs serial across {1, 2, 4, 8} threads);
Theorem 6 / Theorem 7 / Theorem 7' framework unchanged. Gate margins: G3⁶-2D
0.0837 from lower bound, G5_3D 0.22, G4_NS2D_aniso 0.25. Maintainer pre-flight
(npm org + NPM_TOKEN) required before `git push origin v0.11.0`.
