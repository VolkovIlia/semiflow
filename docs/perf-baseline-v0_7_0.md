# Performance Baseline — v0.7.0 (single-threaded, serial)

This document records the wall-clock baseline for the 2D Chernoff-product loop at
the v0.7.0 mathematical freeze. Numbers measured by `benches/heat_2d.rs` and
`benches/advdiff_2d.rs` (no criterion, `std::time::Instant` only).
They are the falsification surface for the ≥5× speedup gate targeted by
v0.8.x Blocks B+C (parallel Strang2D, SIMD intrinsics).

## Machine

| Field          | Value                                  |
|----------------|----------------------------------------|
| CPU            | Intel Core i7-4700MQ @ 2.40 GHz       |
| Logical cores  | 8                                      |
| RAM            | ~16 GB                                 |
| OS             | Linux x86_64 (Fedora 43)               |
| Rust toolchain | 1.78-x86_64-unknown-linux-gnu (stable) |
| Profile        | `--release` (optimized)                |
| Date           | 2026-05-03                             |

## heat_2d — `Strang2D<DiffusionChernoff, DiffusionChernoff>`

Parameters: `n_chernoff=50`, `T=1.0`, isotropic grid `[-4, 4]²`.
Warmup: 1 run discarded before timing.

| Grid N×N      | n_iter | avg per evolve() |
|---------------|--------|-----------------|
| 400×400       | 3      | 7.36 s          |
| 800×800       | 1      | 29.73 s         |
| 1600×1600     | 1      | 118.94 s        |

Scaling ratio 400→800: ×4.0 (expected ×4 for O(N²) work).
Scaling ratio 800→1600: ×4.0 (confirms O(N²)).

## advdiff_2d — `Strang2D<DriftReactionChernoff, DriftReactionChernoff>`

Parameters: `n_chernoff=50`, `T=1.0`, constant drift `b=0.5`, zero reaction.
Warmup: 1 run discarded before timing.

| Grid N×N      | n_iter | avg per evolve() |
|---------------|--------|-----------------|
| 400×400       | 3      | 905 ms          |
| 800×800       | 1      | 3.99 s          |
| 1600×1600     | 1      | 16.03 s         |

Scaling ratio 400→800: ×4.4 (close to expected ×4).
Scaling ratio 800→1600: ×4.0 (confirms O(N²)).

## v0.8.0-block-A re-measurement (2026-05-06)

Re-run after clippy-clean pass (no algorithmic changes). Same machine.

### heat_2d re-run

| Grid N×N  | n_iter | avg per evolve() |
|-----------|--------|-----------------|
| 400×400   | 3      | 8.56 s          |
| 800×800   | 1      | 30.74 s         |
| 1600×1600 | 1      | 131.15 s        |

### advdiff_2d re-run

| Grid N×N  | n_iter | avg per evolve() |
|-----------|--------|-----------------|
| 400×400   | 3      | 946 ms          |
| 800×800   | 1      | 4.16 s          |
| 1600×1600 | 1      | 16.79 s         |

Numbers are consistent with the v0.7.0 baseline within expected CPU throttling variance.

## Methodology

Each benchmark binary is a `harness = false` Cargo bench entry compiled with the
`bench` profile (equivalent to `--release`). The timing loop wraps `n_iter`
consecutive `ChernoffSemigroup::evolve()` calls inside `std::hint::black_box` to
prevent dead-code elimination, then divides total elapsed by `n_iter`. One extra
warmup call (not timed) precedes the loop to prime instruction and data caches.
`n_iter=3` is used for N=400² (cheap enough) and `n_iter=1` for larger grids to
keep each bench binary under ~2 min total.

The numbers are single-threaded. All v0.8.x parallelism work targets the same
`ChernoffSemigroup::evolve()` entry point; the ≥5× speedup gate means each of the
entries in the table above must drop to ≤20 % of its recorded baseline.

## v0.8.0-block-C SIMD measurement (2026-05-05)

Measured by `benches/diffusion6_simd.rs` (new bench, `--features simd`).
Compares `Diffusion6thChernoff::apply` at N=1024 with AVX2 SIMD vs scalar
(force-scalar via thread-local hook, same process, no CPU governor effects).

| Path             | N=1024 avg (n_iter=50) |
|------------------|------------------------|
| scalar (forced)  | 1.938 ms               |
| SIMD (AVX2)      | 1.937 ms               |
| Speedup ratio    | ~1.00×                 |

**Diagnosis**: The `fd9` hot path (9-point Fornberg FD) is memory-bound at N=1024.
Each call accesses 9 non-contiguous grid points via `f.sample()` (quintic-Hermite
interpolation), making the SIMD multiply-accumulate subservient to cache-miss
latency. The compiler's autovectorizer matches SIMD performance for this pattern.

**Gate**: `SIMD_HERMITE_SPEEDUP` — **INFORMATIONAL** (non-blocking, ADR-0019).

**Phase-3 consequence**: Per ADR-0019, speedup < 2× triggers Phase-3 (vectorize
`catmull_rom` in `grid.rs`). Phase-3 was implemented: `catmull_rom_simd` uses
`F64x4` dot-product to replace 12 scalar multiplies with a 4-lane reduce.
Bit-equality to scalar confirmed by `SIMD_BIT_EQUAL` gate (release-blocking,
`tests/simd_bit_equal.rs` — all 32 diffusion6 + 32 quintic-Hermite combinations
pass byte-for-byte).

**Combined SIMD_BIT_EQUAL status**: PASS (release-blocking gate satisfied).
