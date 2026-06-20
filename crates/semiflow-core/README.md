# semiflow-core

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Docs.rs](https://img.shields.io/badge/docs.rs-semiflow--core-blue)](https://docs.rs/semiflow-core)
[![Crates.io](https://img.shields.io/crates/v/semiflow-core)](https://crates.io/crates/semiflow-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

`no_std` Rust solver for linear evolution equations (`∂ₜu = Lu`) via Chernoff
approximation of operator semigroups. No matrix exponentials. No linear solves.
Embarrassingly parallel. Ships 60+ engine implementations across heat,
Schrödinger, diffusion, graph, manifold, and hypoelliptic operators.

Mathematical foundation: **Theorem 6 of I. D. Remizov (2025)**, *Vladikavkaz
Math. J.* 27(4), 124–135
([doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)).

## How it works

Each step `S(τ)` is an explicit, allocation-free contraction kernel. The
Chernoff product formula then gives:

```
(S(t/n))ⁿ f  →  e^{tL} f   as n → ∞
```

The hot loop operates entirely on `Vec<F>` buffers with reused scratch space —
steady-state evolution is zero-allocation and cache-friendly. Generic over
`F: SemiflowFloat` (default `f64`; `f32` supported; forward-mode AD via
`Dual<F>`).

## Install

```toml
[dependencies]
semiflow-core = "0.9"
```

Or:

```bash
cargo add semiflow-core
```

MSRV: **Rust 1.78**. Default feature: `simd` (AVX2 on x86_64, NEON on aarch64;
scalar fallback on all other targets). `no_std + alloc` compatible: disable
default features.

```toml
# no_std build
semiflow-core = { version = "0.9", default-features = false }
```

## Quickstart

Solve `∂ₜu = ½·∂ₓₓu` from `u₀(x) = exp(-x²)` to `t = 1`:

```rust
use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff, ChernoffSemigroup};

let grid = Grid1D::new(-10.0, 10.0, 1000)
    .expect("grid bounds and node count are valid");
let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

// a(x) = 0.5 constant; a' = 0, a'' = 0 enables the fast ζ-A path.
let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, grid);
let semi = ChernoffSemigroup::new(diff, 100)
    .expect("n >= 1 required");
let u1 = semi.evolve(1.0, &u0)
    .expect("evolve should not fail for valid inputs");
// Closed-form oracle: u(1,x) = 3^{-1/2} exp(-x²/3)
// Numerical sup-norm error ≈ 1.46e-6 at n = 100
```

## Engine catalogue

### 1D Chernoff functions

| Type | Order | Description |
|------|-------|-------------|
| `DiffusionChernoff` | 2 | 5-point formula for `a(x)∂²_x` |
| `Diffusion4thChernoff` | 4 | 4th-order spatial diffusion |
| `Diffusion6thChernoff` | 6 | 6th-order spatial diffusion |
| `Diffusion4thZeta4Chernoff` | 4 | ζ⁴ Richardson |
| `Diffusion6thZeta6Chernoff` | 6 | ζ⁶ nested-Richardson |
| `Diffusion8thZeta8Chernoff` | 8 | ζ⁸ Chebyshev M=64 |
| `DiffusionExpmvChernoff<F>` | — | Al-Mohy–Higham expmv action |
| `TruncatedExpDiffusionChernoff` | 2 | Truncated-exponential diffusion |
| `TruncatedExp4thDiffusionChernoff` | 4 | Truncated-exponential 4th-order |
| `DriftReactionChernoff` | exact | Characteristic-flow formula for `b(x)∂_x + c(x)` |
| `DriftReactionZeta4Chernoff` | 4 | ζ⁴ WITH drift via Richardson |
| `ShiftChernoff1D` | 2 | Formula (6) of Theorem 6, Remizov 2025 |

### Compositions

| Type | Description |
|------|-------------|
| `StrangSplit<D, R>` | Strang 2nd-order splitting: `D(τ/2) ∘ R(τ) ∘ D(τ/2)` |
| `AxisLift<C>` | Lift a 1D `ChernoffFunction` to a 2D grid row or column |

### 2D tensor-product

| Type | Description |
|------|-------------|
| `Strang2D<X, Y>` | Palindromic Strang: `Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2)`, global order 2 |
| `NonSeparable2DChernoff` | Non-separable isotropic 2D operator |
| `NonSeparable2DAnisotropicChernoff` | Non-separable anisotropic 2D operator |

### 3D tensor-product

| Type | Description |
|------|-------------|
| `AxisLift3D<C>` | Lift a 1D `ChernoffFunction` to a 3D grid along `Axis::X/Y/Z` |
| `Strang3D<X, Y, Z>` | 3D palindromic Strang |

### Adaptive

| Type | Description |
|------|-------------|
| `AdaptivePI<C>` | PI step-size controller wrapping any `ChernoffFunction` |

### Boundary conditions

| Type | Description |
|------|-------------|
| `BoundaryPolicy::Dirichlet { value }` | Fixed-value stencil BC |
| `BoundaryPolicy::Neumann` | Clamp-to-boundary stencil BC |
| `BoundaryPolicy::Robin { alpha, beta }` | Robin stencil BC |
| `KillingChernoff<C, R, F>` | Operator-level Dirichlet via Feynman-Kac killing |
| `Killing2ndChernoff<C, K, F>` | Order-2 soft-killing `e^{-τκ/2}·C(τ)·e^{-τκ/2}` |
| `ReflectedHeatChernoff<C, R, F>` | Neumann via Walsh 1986 image method |
| `ObstacleChernoff<C, O, F>` | Projective-splitting obstacle / variational-inequality evolver |

### Resolvent and nonautonomous

| Type | Description |
|------|-------------|
| `LaplaceChernoffResolvent<C, F>` | `(λI − A)⁻¹ g` via Gauss-Laguerre 32-pt quadrature — unique to Remizov |
| `HowlandLift<C, F>` | Nonautonomous Howland lift on `L²([0,T], X)` |
| `HdrSnapshot<F>` | NIST nearest-rank percentile library (`no_std + alloc`) |

### Riemannian manifold

| Type | Description |
|------|-------------|
| `ManifoldChernoff<M, F>` | Curvature-corrected Gaussian on T_xM; order 1 (base) or 2 (with R/12) |
| `Torus<F, D>` | Flat torus backend |
| `Sphere2<F>` | Round 2-sphere backend |
| `Hyperbolic2<F>` | Poincaré disk backend (also the SABR volatility manifold) |
| `FubiniStudyCp1` | Kähler CP¹ / Fubini-Study backend |

### High-dimensional and sparse-grid

| Type | Description |
|------|-------------|
| `AnisotropicShiftChernoffND<F, D>` | d-D Gaussian shift via Gauss-Hermite tensor quadrature |
| `AnisotropicShiftZeta2ND` | Order-2 ζ²-correction variant |
| `AnisotropicShiftAdaptiveQ` | Adaptive per-point GH quadrature (41% fewer nodes/axis) |
| `SmolyakGridND<F, const D>` | Smolyak sparse-grid backend for D ≥ 5 |

### Matrix-valued operators

| Type | Description |
|------|-------------|
| `MatrixDiffusionChernoff<F, M>` | Coupled M-component 1D diffusion; Padé[13/13] expm |
| `MatrixDiffusionChernoff2D<F, M>` | Palindromic Strang 2D for M-component diffusion |
| `MatrixDiffusionChernoff3D<F, M>` | Same for 3D |
| `MatrixDiffusionChernoffComplex<F, M>` | Complex-valued Padé[13/13] coupled diffusion |

### Quantum graphs

| Type | Description |
|------|-------------|
| `QuantumGraphHeatChernoff<F>` | Heat on quantum graphs with Kirchhoff vertex condition |
| `QuantumSchrödingerChernoff<C>` | Complex Schrödinger on quantum graphs |

### Reverse-mode AD

| Type | Description |
|------|-------------|
| `ReverseChernoff<F>` | Reverse-mode AD over `(F_θ(τ))ⁿ u₀` via binomial checkpointing |
| `CheckpointSchedule` | `O(√n)` default schedule (Griewank-Walther) |

**Scope (v9.0.0):** constant-a `DiffusionChernoff<F>` only. Variable-coefficient
kernels are deferred.

### Tensor-train carrier

| Type | Description |
|------|-------------|
| `TtChernoff<F>` | TT-Chernoff evolver; storage `O(d·n·r²)` |
| `TtState<F>` | Tensor-train state `u(i₁,…,i_d) = G₁[i₁]·…·G_d[i_d]` |

**Scope (v9.0.0):** linear diagonal-A Gaussian class. Off-diagonal A and
variable coefficients are research-track.

### Gridless / particle-ensemble

| Type | Description |
|------|-------------|
| `GridlessChernoff<F, const D>` | Particle-ensemble Chernoff; implements `ChernoffFunction<F>` |
| `ParticleReduction` | Particle cap policy: `WeightedVoronoi { cap }` or `GaussianBackground` |

### Executor

`ChernoffSemigroup<C, S>` — wraps a `ChernoffFunction` and a step count;
call `.evolve(t, &f)` to compute `(S(t/n))ⁿ f`.

## Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `simd` | on | SIMD kernels (AVX2 / NEON); scalar fallback elsewhere |
| `parallel` | off | Multi-threaded `apply` for 1D types and 2D/3D compositions (requires `std`) |
| `slow-tests` | off | Gates long parameter sweeps (CI only) |
| `linear-interp` | off | Adds `InterpKind::Linear` to grid boundary policies |

`no_std + alloc` works without `std`; libm provides `f32`/`f64` transcendentals.

The `parallel` feature uses `std::thread::scope` and is incompatible with WASM
targets. Bit-equality is guaranteed across thread counts {1, 2, 4, 8} for `f64`
monomorphisations (ADR-0018).

## Examples

| Example | Description |
|---------|-------------|
| `heat_2d_demo.rs` | 2D tensor heat equation (Gaussian oracle, convergence table) |
| `strang_advdiff_demo.rs` | Strang splitting for 1D advection-diffusion |
| `cev_european_call.rs` | CEV European option pricing vs Schroder (1989) closed form |
| `boundary_demo.rs` | Boundary policy showcase (Dirichlet, Neumann, periodic) |
| `latency_tail.rs` | HFT-style p99.9 latency benchmark |
| `resolvent_perf.rs` | `LaplaceChernoffResolvent` L-gate bench harness |
| `heston_pricer.rs` | Heston ρ→0 pricer via palindromic Strang |
| `sabr_pricer.rs` | SABR-on-H² via `ManifoldChernoff<Hyperbolic2>` |

```bash
cargo run -p semiflow-core --example heat_2d_demo
```

## Mathematical references

- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135.
  DOI [10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q).
  Primary reference; Theorem 6 (Chernoff formula) and Theorem 3 (Laplace-Chernoff resolvent).
- S. Mazzucchi, V. Moretti, I. D. Remizov, O. G. Smolyanov, *Math. Nachr.* (2023).
  Riemannian manifold Chernoff with R/12 curvature correction.
- J. S. Howland, *Trans. AMS* **207** (1974). Nonautonomous augmented-generator lift.
- Y. A. Butko, *Fract. Calc. Appl. Anal.* **21** (2018). Feynman-Kac killing for Dirichlet BC.
- J. Walsh, *Séminaire de Probabilités* **20** (1986). Image method for Neumann BC.

```bibtex
@article{Remizov2025,
  author  = {I. D. Remizov},
  title   = {Chernoff Approximations of the Solution of Linear ODE with Variable Coefficients},
  journal = {Vladikavkaz Math. J.},
  volume  = {27},
  number  = {4},
  pages   = {124--135},
  year    = {2025},
  doi     = {10.46698/a3908-1212-5385-q}
}
```

## Bindings

This crate is Rust-only. Sibling crates wrap it for other languages:

| Language | Crate / Package | Distribution |
|----------|----------------|--------------|
| C | `semiflow-ffi` (cdylib) | GitHub release artifacts |
| Python | `semiflow-pde` (PyO3, abi3-py310) | PyPI / GitHub releases |
| JavaScript | `@semiflow/wasm` (wasm-bindgen) | npm |

See the [workspace README](https://github.com/VolkovIlia/semiflow#bindings) for
cross-language status and build instructions.

## License

Dual MIT OR Apache-2.0. See [`LICENSE-MIT`](../../LICENSE-MIT) and
[`LICENSE-APACHE`](../../LICENSE-APACHE) in the workspace root.

## Author

Created and maintained by **Ilia Volkov** — ilia.volkov@outlook.com.

## Contributing

See [CONTRIBUTING.md](https://github.com/VolkovIlia/semiflow/blob/master/CONTRIBUTING.md)
and [CODE_OF_CONDUCT.md](https://github.com/VolkovIlia/semiflow/blob/master/CODE_OF_CONDUCT.md).
Security reports to ilia.volkov@outlook.com per
[SECURITY.md](https://github.com/VolkovIlia/semiflow/blob/master/SECURITY.md).
