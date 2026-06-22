---
version: 1.0.0
last_updated: 2026-05-10
freshness_score: 1.0
dependencies:
  - docs/api-stability.md
  - docs/perf/baseline-v1_0_0.json
  - .github/workflows/nightly.yml
changelog:
  - 1.0.0: Initial perf commitment, effective at v1.0.0 release (task S2.3)
---

# Performance Commitment — v1.0.0

> **Scope note:** This document tracks internal criterion micro-benchmarks
> (regression thresholds for throughput in isolated kernels). It is
> **distinct from competitive head-to-head comparisons**, which are
> documented in the iter-8 campaign
> (`remizov-publications/benchmarks/results/aggregate-iter8/iter8-cross-wave.md`,
> library HEAD b923777, 45 families). The iter-8 campaign found:
> memory frugality PARTIAL_SUPPORT (33/43 pairs RC < competitor); wallclock
> FALSIFIED vs adaptive/spectral solvers; parallelism FALSIFIED as universal
> claim (eta8≈0.125 for 2D; 0.908 for large 3D only). The micro-benchmarks
> below track that the library does not regress on its own prior throughput;
> they do not imply wallclock competitiveness vs external solvers.

## 1. Status

Effective from v1.0.0.

Performance characteristics are **not** part of the API stability contract
(`docs/api-stability.md` § 6): *"performance characteristics … are not
contractual."* The targets in this document are informational guides, not
enforceable guarantees.  A `PATCH` release may improve throughput without
notice; no release will knowingly introduce a regression beyond the 5 %
threshold below without a recorded rationale in the CHANGELOG.

---

## 2. What We Measure

Five criterion benchmarks are tracked at v1.0.0.  Each runs via
`cargo bench --workspace` with the `bench` profile
(`opt-level = 3`, `lto = "thin"`).

| Benchmark key | File | What it exercises |
|---|---|---|
| `semiflow_core::heat_1d` | `crates/semiflow-core/benches/heat_1d.rs` | `ShiftChernoff1D::evolve` — 1D Gaussian heat kernel, N=1000 grid points, n=100 Chernoff steps, T=1.0.  Single-threaded scalar path. |
| `semiflow_core::heat_2d` | `crates/semiflow-core/benches/heat_2d.rs` | `Strang2D<DiffusionChernoff>` — 2D isotropic heat, 400×400 grid, `n_chernoff=10`, T=1.0.  Representative slice of the v0.7.0 baseline grid. |
| `semiflow_core::advdiff_2d` | `crates/semiflow-core/benches/advdiff_2d.rs` | `Strang2D<DriftReactionChernoff>` — 2D advection-diffusion, 400×400 grid, `n_chernoff=10`, T=1.0.  Constant drift `b=0.5`, zero reaction. |
| `semiflow_core::diffusion6_simd` | `crates/semiflow-core/benches/diffusion6_simd.rs` | `Diffusion6thChernoff::apply` — single `apply` call, N=1024, variable-`a` coefficients with sinusoidal variation, `BoundaryPolicy::Reflect`.  Exercises the 9-point Fornberg FD kernel and SIMD path (when `--features simd`). |
| `semiflow::evolve_bench` | `crates/semiflow-py/benches/evolve_bench.rs` | `ChernoffSemigroup::evolve` via the pure-Rust kernel that backs `Heat1D::evolve` in the PyO3 layer.  N=1000, `n_steps=100`, T=1.0.  Measures the Rust hot path without Python GIL overhead. |

---

## 3. Targets

Numerical baselines are populated by the maintainer after running
`cargo bench --workspace` on the reference hardware (see § 6) at the v1.0.0 cut.

Current values in `docs/perf/baseline-v1_0_0.json`:

| Benchmark key | Median (ns) | Std-dev (ns) |
|---|---|---|
| `semiflow_core::heat_1d` | TBD — populated by maintainer | TBD |
| `semiflow_core::heat_2d` | TBD — populated by maintainer | TBD |
| `semiflow_core::advdiff_2d` | TBD — populated by maintainer | TBD |
| `semiflow_core::diffusion6_simd` | TBD — populated by maintainer | TBD |
| `semiflow::evolve_bench` | TBD — populated by maintainer | TBD |

Fill these by running:

```sh
RUSTFLAGS="-C target-cpu=native" cargo bench --workspace 2>&1 | tee bench-results.txt
```

Then extract the criterion median from `bench-results.txt` and update
`docs/perf/baseline-v1_0_0.json` with the actual nanosecond values.

---

## 4. Methodology

- **Harness**: criterion 0.5 (`harness = false` bench entries with
  `criterion_group!` / `criterion_main!`).  Criterion uses a bootstrap
  statistical estimator (100 resamples, 95 % CI) on wall-clock samples.
- **Warmup**: 3 criterion warmup iterations (criterion default).
- **Measurement**: 5 criterion measurement iterations (criterion default),
  each iteration repeated until `measurement_time` (5 s default) elapses.
- **Compilation flags**: `[profile.bench]` — `opt-level = 3`, `lto = "thin"`.
  SIMD: add `RUSTFLAGS="-C target-cpu=native"` and `--features simd` to
  engage AVX2 / NEON intrinsics on capable hardware.
- **Comparison**: criterion `--baseline` flag compares a saved baseline set
  against a fresh run and exits non-zero if any measurement exceeds the
  regression threshold.

To record a local baseline and compare:

```sh
# Record baseline (first run)
RUSTFLAGS="-C target-cpu=native" cargo bench --workspace -- --save-baseline v1_0_0

# Compare against it
RUSTFLAGS="-C target-cpu=native" cargo bench --workspace -- --baseline v1_0_0
```

---

## 5. Regression Policy

A nightly CI job (`.github/workflows/nightly.yml`) runs the five tracked
benchmarks and compares against `docs/perf/baseline-v1_0_0.json`.

- **> 5 % regression** on any tracked benchmark fails the CI nightly job.
- **± 2–3 %** hardware-dependent variation (CPU throttling, thermal noise) is
  tolerated by the threshold.
- **Improvements** are always welcome and never blocked.
- If a regression is intentional (e.g. a correctness fix that costs
  throughput), the maintainer records the rationale in `CHANGELOG.md` under
  `### Changed` and updates `docs/perf/baseline-v1_0_0.json` with the
  new expected values before merging.

---

## 6. Hardware Reference

The v1.0.0 baseline is collected on the maintainer's primary development
machine:

| Field | Value |
|---|---|
| CPU | Intel Core i7-12700K (12C/20T, ADL-S) |
| RAM | TBD — populated by maintainer |
| OS | Linux x86_64 (Fedora) |
| Rust toolchain | Stable (version at v1.0.0 cut) |
| `RUSTFLAGS` | `-C target-cpu=native` |
| Profile | `bench` (`opt-level = 3`, `lto = "thin"`) |

Other architectures (macOS/arm64, x86_64 without AVX2) will see different
absolute numbers.  The CI regression check runs on `ubuntu-latest` (GitHub
Actions runner) which may differ from the reference hardware; the 5 %
threshold accounts for this variance.

---

## 7. What Is Not Measured

The following are **not** currently part of the tracked commitment:

- **Startup / cold-cache latency** — first-call overhead from cold instruction
  and data caches.  Criterion's warmup phase (§ 4) discards these samples.
- **Memory bandwidth peak** — not exercised by the current bench suite.
- **Python GIL overhead** — `semiflow::evolve_bench` measures the pure-Rust
  kernel; GIL acquire/release and NumPy array copy overhead are exercised by
  the PyO3 integration smoke test (`xtask py-smoke`) but not tracked here.
- **WASM throughput** — `semiflow-wasm` has no criterion bench at v1.0.0.
- **3D and non-separable 2D operators** — `Strang3D`, `NonSeparable2DChernoff`,
  `NonSeparable2DAnisotropicChernoff`.  These may be added in a future MINOR
  release as additional tracked benchmarks.
- **Parallel speedup** — `--features parallel` (`Strang2D` thread-pool path).
  The v0.8.1 / v0.11.0 flagship gate (`G3⁶-2D`) documented this separately in
  `docs/perf-baseline-v0_7_0.md`.  A parallel-specific bench may be added at
  v1.1.0.
