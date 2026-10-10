# Performance overhaul (2026-10) — before / after

Measured with `cargo run --release -p semiflow --example perf_suite -- <group>`
on the same container, before (base `182a3d3`) and after the overhaul (ADRs
0204–0209). Best of 3–5 runs, wall-clock milliseconds; single-threaded. Timings
are indicative (shared CI-class host); the accuracy columns and all gate
thresholds are deterministic.

## Grid engines — bit-identical (ADR-0204)

| Workload | Before | After | Speedup |
|---|---:|---:|---:|
| `ShiftChernoff1D`, N = 1000, 100 steps | 17.9 | 6.3 | 2.8× |
| `DiffusionChernoff` const `a`, N = 1000, 100 steps | 29.2 | 9.4 | 3.1× |
| `DiffusionChernoff` `fn`-pointer `a ≡ ½` (README path) | 60.9 | 20.0 | 3.0× |
| `DiffusionChernoff` variable `a` (closure) | 73.2 | 27.9 | 2.6× |
| `Diffusion6th`, N = 1024, 50 steps | 21.9 | 6.8 | 3.2× |
| `Diffusion4th` + Chebyshev sampling, N = 256, 20 steps | 104.4 | 2.7 | 39× |
| `Strang2D` heat, 400², 10 steps | 3300 | 1140 | 2.9× |
| `Strang3D` heat, 64³, 10 steps | 8200 | 3800 | 2.2× |

Same output bits (gate `G_PLAN_BIT_EQUAL`; every `no_std` grid digest unchanged).

## Krylov / Taylor / ETD — new algorithms (ADR-0205)

Issue #16 operator (1-D Neumann Laplacian, N = 400), `tol = 1e−10`:

| Workload | Before | After | Speedup | sup error before → after |
|---|---:|---:|---:|---|
| Chebyshev, `λt ≈ 4e3` | 2.56 | 0.455 | 5.6× | |
| Lanczos `m_max = 18`, `λt ≈ 4e3` | 204 | 40.0 | 5.1× | |
| Chebyshev, `λt ≈ 4e5` | 240 | 4.34 | 55× | |
| Lanczos `m_max = 18`, `λt ≈ 4e5` | 19 637 | 5 156 | 3.8× | |
| Chebyshev, `λt ≈ 4e7` (issue #16) | 23 746 | 42.5 | 559× | 6.8e−6 → 1.4e−11 |
| `GeneralOperator` Taylor `expmv`, N = 1000, `‖A‖t ≈ 4e3` | 114.9 | 54.7 | 2.1× | |
| ETDRK4 Allen–Cahn, N = 256, 100 steps | 299.8 | 140.9 | 2.1× | |
| `phi_action` φ₁, N = 1000 | 287.7 | 241.8 | 1.2× | |
| `MatrixDiffusion` M = 3, N = 512, 100 steps | 18.5 | 17.6 | — | wrong exponential (M = 3) → exact |

Deterministic counters (gates): Chebyshev degree `≤ ⌈L/3 + √(L²/9 + 2zL)⌉ − 1`
(`G_CHEB_SQRT_COST`); TPS stack 7.35e5 → 8829 mat-vecs (`G_TPS_STIFF_STEPCOUNT`);
ETDRK4 `5·C(h/2) + 4·C(h)` → `3·C(h/2) + C(h)` generator applications
(`G_ETDRK4_SWEEPS`); Taylor shift halves `‖·‖_∞` on diffusion stencils
(`G_EXPMV_SHIFT_COST`).

## Fewer steps for the same accuracy (ADR-0207)

`ShiftChernoff1D` (order 1) on the heat equation, 96 steps in total:
plain `3.3e−4`, three Richardson levels on `n = 16` `7.2e−8` (4600×). Order
1 → 2 → 3 with 1 / 2 / 3 levels (`G_RICHARDSON_SHIFT1D`).

## Reaction–diffusion (ADR-0208/0209)

Python, CPython 3.13: Gray–Scott on 64² nodes, 2 species, 200 Strang steps:
0.94 s; a callable `K = 2` system on 401 nodes, 50 steps with two Richardson
levels: 22 ms (the callable is invoked once per RK stage, not per node).

## Known slow paths (not addressed here)

- `GridFnND::sample` (tensor-product cubic) needs all `4^D` node values per
  sample: `SmolyakD6V8.apply` on `4⁶` nodes takes ≈ 50 s (unchanged by this
  overhaul; see ROADMAP).
- Lanczos stays linear in `λt` at fixed `m_max` (restarted Krylov); prefer
  Chebyshev, or the planned rational Krylov path, for very stiff problems.
