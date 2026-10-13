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

## N-D shift engines (ADR-0214)

Constant coefficients sample through `D` whole-array axis passes with prefix
sharing instead of a `4^D` recursion per sample, bit-identically:
`SmolyakD6V8.apply` on `4⁶` nodes (CPython 3.13, release wheel) ≈ 50 s → 0.09 s per
step; `test_smolyak_v8.py` ≈ 8 min → 1.1 s with unchanged goldens. Variable
coefficients keep the per-node loop (parallel for `f64`).

## Rational Krylov (ADR-0206)

Neumann path Laplacian, `tol = 1e−10`: resolvent solves vs Chebyshev mat-vecs
(deterministic counts from `G_RATKRYLOV_SOLVE_COUNT`):

| `λ_max t` | n = 400 solves (err/‖v‖∞) | n = 4000 solves (err/‖v‖∞) | Chebyshev degree |
|---:|---:|---:|---:|
| 1e2 | 27 (1.7e−13) | — | 49 |
| 1e4 | 27 (8.2e−14) | — | 472 |
| 1e6 | 27 (3.3e−16) | 27 (3.4e−12) | 4 719 |
| 1e8 | 27 (2.2e−16) | 27 (1.4e−15) | 47 188 |
| 1e10 | 27 (9.4e−16) | — | 471 877 |

The count is the a-priori dimension of `tol` (16 / 21 / 27 / 33 solves at
`1e−6 / 1e−8 / 1e−10 / 1e−12`). Each solve on a tridiagonal pattern is one `O(N)`
`LDLᵀ` sweep; a 40×40 grid Laplacian on the PCG path (IC(0)) takes 13 … 90 CG
iterations per solve for `λ_max t = 1e2 … 1e8`.

## φ shift (ADR-0210)

Generator applications per φ sweep (deterministic, `G_PHI_SHIFT_COST`, 64-node
diffusion generators): `τ‖G‖ = 40`: 720 → 390; `τ‖G‖ = 400`: 6 840 → 3 510
(div-form 3 450). Wall clock, same session back to back (shift disabled vs
enabled; the host was loaded, so the absolute numbers are above the ADR-0205 table):

| Workload | Unshifted | Shifted | Speedup |
|---|---:|---:|---:|
| `phi_action` φ₁, N = 1000, `‖G‖t ≈ 4e3` | 300 | 145 | 2.1× |
| ETDRK4 Allen–Cahn, N = 256, 100 steps | 195 | 96 | 2.0× |

## Adaptive Krylov φ / exp (ADR-0211)

Upwind-drift tridiagonal `N = 1000`, `e^{−tA}v`, same session (agreement with the
Taylor result `≤ 1e−13`):

| `‖A‖t` | Taylor `expmv` | Krylov `m = 30` | Krylov `m = 15` |
|---:|---:|---:|---:|
| 4·10³ | 90 ms | 44 ms | 31 ms |
| 4·10⁴ | 973 ms | 311 ms | 171 ms |
| 4·10⁵ | 92 s | 3.7 s | 1.9 s |

Generator applications (deterministic, `G_PHI_KRYLOV_COST`): drifted Fokker–Planck
`n = 1000`, `τ‖G‖ = 4·10³`: 1 023 (Krylov) vs 33 960 (Taylor). One-pass Gram–Schmidt
dominates a substep on cheap stencils; `m ≈ 15` trades applications for wall clock.

## Known slow paths (not addressed here)

- Lanczos stays linear in `λt` at fixed `m_max` (restarted Krylov); prefer
  Chebyshev, or the rational Krylov path (ADR-0206), for very stiff problems.
