# semiflow-core

Chernoff approximations of strongly continuous operator semigroups (Theorem 6, Remizov 2025).

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Docs.rs](https://img.shields.io/badge/docs.rs-remizov--core-blue)](https://docs.rs/semiflow-core)
[![Crates.io](https://img.shields.io/crates/v/semiflow-core)](https://crates.io/crates/semiflow-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

## What this crate does

Approximates `e^{tA} f` for a generator `A` — diffusion, drift-reaction,
advection-diffusion in 1D / 2D / 3D — via finite iterations of a contraction
`S(τ)`, per the Chernoff theorem: `(S(t/n))^n f → e^{tA} f` as `n → ∞`.
No matrix exponentials, no linear-system solves, embarrassingly parallel in `n`.

v9.0.0: 50+ `ChernoffFunction` implementations spanning 1D/2D/3D diffusion and
shift kernels (including ζ⁴/ζ⁶/ζ⁸ ladder with TRUTHFUL_ORDER gates now passing),
graph PDE on arbitrary weighted graphs, Schrödinger 1D, non-separable anisotropic 2D,
Riemannian manifold Chernoff (Sphere2/Hyperbolic2/Torus/CP¹), Laplace-Chernoff
resolvent (real + complex λ), Howland nonautonomous lift, Feynman-Kac killing,
order-2 soft-killing, Neumann via image method, adjoint backward-semigroup,
adaptive step control, obstacle/VI evolver, quantum graph Schrödinger, expmv action,
matrix-valued 2D/3D, Smolyak sparse-grid high-dimensional shift, reverse-mode AD
(binomial checkpointing), tensor-train carrier (Gaussian class), and gridless
particle-ensemble Chernoff.

Pure Rust `rlib`, `no_std + alloc` compatible. Zero unsafe in math kernels
(SIMD intrinsics isolated to `src/simd/`).

## Install

```toml
[dependencies]
semiflow-core = "9"
```

Or with Cargo:

```sh
cargo add semiflow-core
```

MSRV: Rust 1.78. Default features: `simd` (AVX2 on x86\_64, NEON on aarch64;
scalar fallback on all other targets).

## Quickstart

Solve `∂_t u = 0.5 ∂_xx u` from `u_0(x) = exp(-x²)` to `t = 1`:

```rust
use semiflow_core::{Grid1D, GridFn1D, DiffusionChernoff, ChernoffSemigroup};

let grid = Grid1D::new(-10.0, 10.0, 1000)
    .expect("grid bounds and node count are valid");
let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

// a(x)=0.5 (constant); a'=0, a''=0 enables the ζ-A fast path.
let diff = DiffusionChernoff::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.5, grid);
let semi = ChernoffSemigroup::new(diff, 100)
    .expect("n=100 satisfies the n >= 1 precondition");
let u1 = semi.evolve(1.0, &u0)
    .expect("evolve should not fail for valid inputs");
// Closed-form oracle: u(1,x) = (3)^{-1/2} exp(-x²/3)
// Numerical sup-norm error ≈ 1.46e-6 at n=100
```

## What's available

### 1D Chernoff functions

| Type | Order | Description |
|------|-------|-------------|
| `DiffusionChernoff` | 2 | 5-point formula for `a(x)∂²_x` |
| `Diffusion4thChernoff` | 4 | 4th-order spatial diffusion (ADR-0009) |
| `Diffusion6thChernoff` | 6 | 6th-order spatial diffusion |
| `Diffusion4thZeta4Chernoff` | 4 | ζ⁴ Richardson Option E (ADR-0086) |
| `Diffusion6thZeta6Chernoff` | 6 | ζ⁶ nested-Richardson (ADR-0088); `G_zeta6_TRUTHFUL_ORDER` PASS (v7.0) |
| `Diffusion8thZeta8Chernoff` | 8 | ζ⁸ Chebyshev M=64; `G_zeta8_TRUTHFUL_ORDER` PASS (v7.0) |
| `DiffusionExpmvChernoff<F>` | — | Al-Mohy–Higham expmv action; `sup_error ≈ 1.1e-15` at `τ‖A‖=62` (ADR-0121, v7.0) |
| `TruncatedExpDiffusionChernoff` | 2 | Truncated-exponential diffusion |
| `TruncatedExp4thDiffusionChernoff` | 4 | Truncated-exponential 4th-order |
| `DriftReactionChernoff` | exact | Characteristic-flow formula for `b(x)∂_x + c(x)` |
| `DriftReactionZeta4Chernoff` | 4 | ζ⁴ WITH drift via Path β Richardson; resolves §27.7 OPEN (ADR-0131, v7.0) |
| `ShiftChernoff1D` | 2 | Formula (6) of Theorem 6, Remizov 2025 |

### Compositions

| Type | Description |
|------|-------------|
| `StrangSplit<D, R>` | Strang 2nd-order splitting: `D(τ/2) ∘ R(τ) ∘ D(τ/2)` |
| `AxisLift<C>` | Lift a 1D `ChernoffFunction` to operate on a 2D grid row or column |

### 2D tensor product

| Type | Description |
|------|-------------|
| `Strang2D<X, Y>` | Palindromic Strang: `Sx(τ/2) ∘ Sy(τ) ∘ Sx(τ/2)`, global order 2 |
| `NonSeparable2DChernoff` | Non-separable isotropic 2D operator (ADR-0021) |
| `NonSeparable2DAnisotropicChernoff` | Non-separable anisotropic 2D operator (ADR-0023) |

### 3D tensor product

| Type | Description |
|------|-------------|
| `AxisLift3D<C>` | Lift a 1D `ChernoffFunction` to a 3D grid along `Axis::X/Y/Z` |
| `Strang3D<X, Y, Z>` | 3D palindromic Strang, order = min per-axis order (Lemma 10.1) |

### Adaptive

| Type | Description |
|------|-------------|
| `AdaptivePI<C>` | PI step-size controller wrapping any `ChernoffFunction` (ADR-0014) |

### Boundary conditions (v2.6+)

| Type | Description |
|------|-------------|
| `BoundaryPolicy::Dirichlet { value: F }` | Fixed-value stencil BC (v2.6, ADR-0068) |
| `BoundaryPolicy::Neumann` | Clamp-to-boundary stencil BC (v2.6, ADR-0068) |
| `BoundaryPolicy::Robin { alpha, beta }` | Robin stencil BC (v4.6, ADR-0098) |
| `KillingChernoff<C, R, F>` | Operator-level Dirichlet via Feynman-Kac killing; order-1 (Butko 2018, ADR-0068) |
| `Killing2ndChernoff<C, K, F>` | Order-2 soft-killing `e^{-τκ/2}·C(τ)·e^{-τκ/2}`; distinct from hard-wall `KillingChernoff` (ADR-0126, v7.0) |
| `ReflectedHeatChernoff<C, R, F>` | Neumann via Walsh 1986 image method; order matches inner `C` (v2.8, ADR-0072) |
| `HalfSpaceRegion<F, D>` / `BoxRegion<F, D>` / `BallRegion<F, D>` | Concrete `KillingRegion` / `ReflectingRegion` geometries |
| `ObstacleChernoff<C, O, F>` | Projective-splitting obstacle/VI evolver `Π_g(S(Δτ)Vⁿ)`; order-1 (ADR-0116, v6.3) |

### Resolvent and nonautonomous (v2.7+)

| Type | Description |
|------|-------------|
| `LaplaceChernoffResolvent<C, F>` | `(λI − A)⁻¹ g` via Gauss-Laguerre 32-pt quadrature. **Unique to Remizov.** Cite Remizov 2025 Vladikavkaz Thm 3. (ADR-0069) |
| `EvalComplex` | `LaplaceChernoffResolvent::eval_complex` — complex-λ extension; Re λ > ω required (ADR-0127, v7.0) |
| `LaplaceChernoffResolventResidual<C, F>` | Residual-verification wrapper (ADR-0083) |
| `HowlandLift<C, F>` | Nonautonomous Howland lift on L²([0,T], X). Cite Howland 1974 Trans. AMS 207. (ADR-0070) |
| `TimedChernoffFunction<F>` | Super-trait adding `apply_at(t, τ, ...)` with blanket autonomous default |
| `HdrSnapshot<F>` | NIST nearest-rank percentile lib (`no_std + alloc`). `record(ns)`, `percentile(pct)` |

### Riemannian manifold (v2.8+)

| Type | Description |
|------|-------------|
| `ManifoldChernoff<M, F>` | Curvature-corrected Gaussian on T_xM; order 1 (base) or 2 (with R/12). Cite MMRS 2023 Math. Nachr. (ADR-0071) |
| `BoundedGeometryManifold<F>` | Trait: `dim`, `injectivity_radius`, `exp_map`, `parallel_transport`, `scalar_curvature`, `volume_element_log` |
| `Torus<F, D>` | Flat torus backend (R ≡ 0) |
| `Sphere2<F>` | Round 2-sphere backend (R = 2/r², great-circle exp_x via 3D embedding) |
| `Hyperbolic2<F>` | Poincaré disk backend (R = -2/s², Möbius exp_x) — also the SABR volatility manifold |
| `FubiniStudyCp1` | Kähler CP¹ / Fubini-Study backend; isometric to S² (R=2), MMRS-2023 R/12 correction applies (ADR-0129, v7.0) |

### High-dimensional shift and sparse grids (v4.0+)

| Type | Description |
|------|-------------|
| `AnisotropicShiftChernoffND<F, D>` | d-D Gaussian shift via Gauss-Hermite tensor quadrature; order-1 (ADR-0081 + ADR-0112) |
| `AnisotropicShiftZeta2ND` | Order-2 ζ²-correction variant for `AnisotropicShiftChernoffND` (ADR-0112 AMENDMENT 2, v7.0) |
| `AnisotropicShiftAdaptiveQ` | Opt-in adaptive per-point GH quadrature; 41% fewer nodes/axis (ADR-0122, v7.0) |
| `SmolyakGridND<F, const D>` | Smolyak sparse-grid backend for D≥5; 9.2× node reduction at D=5, ℓ=8 (ADR-0123, v7.0) |

### Matrix-valued operators (v4.0+)

| Type | Description |
|------|-------------|
| `MatrixDiffusionChernoff<F, M>` | Coupled M-component diffusion 1D; Padé[13/13] expm for M≤4, M≥5 via `MatrixExpPade` (ADR-0082) |
| `MatrixDiffusionChernoff2D<F, M>` | Palindromic Strang 2D for M-component diffusion (ADR-0124, v7.0) |
| `MatrixDiffusionChernoff3D<F, M>` | Same for 3D (ADR-0124, v7.0) |
| `MatrixDiffusionChernoffComplex<F, M>` | Complex-valued Padé[13/13] + coupled diffusion over `SemiflowComplex` (ADR-0128, v7.0) |
| `MatrixExpPade<M>` | Padé[13/13] for M≥5 coupling blocks; lifts the M≥5 `Unsupported` guard (ADR-0125, v7.0) |

### Quantum graphs (v3.1+)

| Type | Description |
|------|-------------|
| `QuantumGraphHeatChernoff<F>` | Heat on quantum graphs with Kirchhoff vertex condition (ADR-0078) |
| `QuantumSchrödingerChernoff<C>` | Complex Schrödinger on quantum graphs; unitarity gate `G_QSCHROD` (ADR-0130, v7.0) |

### Reverse-mode AD (v9.0.0, ADR-0156)

| Type | Description |
|------|-------------|
| `ReverseChernoff<F>` | Reverse-mode AD over `(F_θ(τ))ⁿ u₀` via binomial checkpointing (math §51) |
| `CheckpointSchedule` | Checkpoint schedule; `sqrt_n(n)` constructor gives `O(√n)` default (Griewank-Walther) |

**NARROW scope (§51.5):** constant-a `DiffusionChernoff<F>` ONLY.
Variable-coefficient and nonlinear kernels are not supported at v9.0.0.

```rust
use semiflow_core::{CheckpointSchedule, DiffusionChernoff, Dual, Grid1D,
                   GridFn1D, InterpKind, ReverseChernoff};

let grid = Grid1D::new(-4.0, 4.0, 24).unwrap()
    .with_interp(InterpKind::CubicHermite);
let kernel = DiffusionChernoff::with_closure(|_| 0.4_f64, |_| 0.0, |_| 0.0, 0.4, grid);

let grid_dual = Grid1D::<Dual<f64>>::new_generic(
    Dual::constant(-4.0), Dual::constant(4.0), 24,
).unwrap().with_interp(InterpKind::CubicHermite);
let kernel_dual = DiffusionChernoff::<Dual<f64>>::with_closure(
    |_| Dual::variable(0.4_f64),
    |_| Dual::constant(0.0),
    |_| Dual::constant(0.0),
    0.4,
    grid_dual,
);

let rc = ReverseChernoff::new(kernel, kernel_dual, CheckpointSchedule::sqrt_n(8));
let u0     = GridFn1D::from_fn(grid, |x| (-x * x).exp());
let target = GridFn1D::from_fn(grid, |_| 0.0_f64);

let (loss, grad) = rc.value_and_grad_k1(0.05, 8, &u0, &target)
    .expect("reverse AD");
// loss: f64 — L² loss ‖(F_θ(τ))⁸ u₀ − target‖²
// grad: f64 — ∂J/∂θ (K=1 forward-mode Dual, 0-ULP vs forward-mode reference)
```

Key methods:

| Method | Signature | Notes |
|--------|-----------|-------|
| `CheckpointSchedule::sqrt_n` | `fn sqrt_n(n_steps: usize) -> Self` | Builds `O(√n)` schedule; `stride = ⌈√n⌉` |
| `CheckpointSchedule::checkpoint_count` | `fn checkpoint_count(&self) -> usize` | `(n − 1) / stride + 1` |
| `ReverseChernoff::new` | `fn new(kernel: DiffusionChernoff<F>, kernel_dual: DiffusionChernoff<Dual<F>>, schedule: CheckpointSchedule) -> Self` | |
| `ReverseChernoff::value_and_grad_k1` | `fn value_and_grad_k1(&self, tau: F, n: usize, u0: &GridFn1D<F>, target: &GridFn1D<F>) -> Result<(F, F), SemiflowError>` | K=1 gradient; 0-ULP (§51.4) |

---

### Tensor-train carrier (v9.0.0, ADR-0159)

| Type | Description |
|------|-------------|
| `TtChernoff<F>` | TT-Chernoff evolver for linear diagonal-A diffusion (Gaussian class); `step(tau, &mut state)` and `evolve(t_final, n_steps, &mut state)` |
| `TtState<F>` | Tensor-train state `u(i₁,…,i_d) = G₁[i₁]·…·G_d[i_d]`; storage `O(d·n·r²)` |

**NARROW scope (ADR-0159):** linear diagonal-A Gaussian class only.
Off-diagonal A, variable/nonlinear coefficients, non-Gaussian IC: TT rank not
algebraically bounded — research-track. No FFI/PyO3/WASM bindings at v9.0.0.

Constructor signatures:

| Method | Signature |
|--------|-----------|
| `TtChernoff::new` | `fn new(a: Vec<F>, b: Vec<F>, c: F, domain: Vec<(F, F)>, eps_round: F) -> Self` |
| `TtState::rank1_separable` | `fn rank1_separable(slices: Vec<Vec<F>>) -> Self` |
| `TtState::inner_separable` | `fn inner_separable(&self, functionals: &[Vec<F>]) -> F` |
| `TtChernoff::step` | `fn step(&self, tau: F, state: &mut TtState<F>)` |
| `TtChernoff::evolve` | `fn evolve(&self, t_final: F, n_steps: usize, state: &mut TtState<F>)` |

---

### Gridless / particle-ensemble Chernoff (v9.0.0, ADR-0155)

| Type | Description |
|------|-------------|
| `GridlessChernoff<F, const D: usize>` | Particle-ensemble Chernoff evolver; implements `ChernoffFunction<F>` |
| `ParticleReduction` | Particle cap policy: `WeightedVoronoi { cap }` or `GaussianBackground` (stub) |

**NARROW scope (ADR-0155 §50.7):** diagonal A, constant scalar coefficients,
d ≤ ~10 (d=2 validated). Off-diagonal A, variable coefficients, d > 10:
research-track. No FFI/PyO3/WASM bindings at v9.0.0.

Constructor signatures:

| Method | Signature |
|--------|-----------|
| `GridlessChernoff::new` | `fn new(a: [F; D], b: [F; D], c: F, reduction: ParticleReduction) -> Self` |
| `GridlessChernoff::isotropic` | `fn isotropic(a: F, b: F, c: F, reduction: ParticleReduction) -> Self` |
| `ChernoffFunction::apply_into` | `fn apply_into(&self, tau: F, src: &MeasureState<F, D>, dst: &mut MeasureState<F, D>, scratch: &mut ScratchPool<F>) -> Result<(), SemiflowError>` |

---

### Executor

`ChernoffSemigroup<C, S>` — wraps a `ChernoffFunction` and a step count;
call `.evolve(t, &f)` to compute `(S(t/n))^n f`.

## Feature flags

| Flag | Default | Effect |
|------|---------|--------|
| `simd` | on | SIMD kernels (AVX2 on x86\_64, NEON on aarch64); scalar fallback elsewhere |
| `parallel` | off | Multi-threaded `apply` for 1D Chernoff types and 2D/3D compositions (requires std; see below) |
| `slow-tests` | off | Gates 135-combo `Diffusion6th` parameter sweep (test-only, CI-only) |
| `linear-interp` | off | Adds `InterpKind::Linear` to grid boundary policies |

`no_std + alloc` works without `std`; libm provides `f32`/`f64` transcendentals.

### `parallel` feature details

Parallelised via `std::thread::scope`. Incompatible with WASM targets.

**1D Chernoff types** — all seven types apply across grid points for `N >= 2048`
(ADR-0036): `ShiftChernoff1D`, `DiffusionChernoff`, `Diffusion4thChernoff`,
`Diffusion6thChernoff`, `TruncatedExpDiffusionChernoff`,
`TruncatedExp4thDiffusionChernoff`, `DriftReactionChernoff`.

**2D/3D compositions** — parallelised across rows / pencils (ADR-0018, ADR-0022):
`Strang2D`, `Strang3D`, `NonSeparable2DChernoff`, `NonSeparable2DAnisotropicChernoff`.

Bit-equality is guaranteed across thread counts {1, 2, 4, 8} for `f64`
monomorphisations. Generic `F = f32` paths remain serial.

## Examples

| Example | Description |
|---------|-------------|
| `heat_2d_demo.rs` | 2D tensor heat equation (Gaussian oracle, 3-step convergence table) |
| `strang_advdiff_demo.rs` | Strang splitting for 1D advection-diffusion |
| `cev_european_call.rs` | CEV European option pricing against Schroder (1989) closed form |
| `boundary_demo.rs` | Boundary policy showcase (Dirichlet, Neumann, periodic) |
| `latency_tail.rs` | HFT-style p99.9 latency benchmark; 149× CEV advantage vs QuantLib (v2.5.1+) |
| `resolvent_perf.rs` | `LaplaceChernoffResolvent` L-gate bench harness (v2.7+) |
| `heston_pricer.rs` | Heston ρ→0 pricer via palindromic Strang; p99 ≈ 28 ns/tick (v2.7+) |
| `sabr_pricer.rs` | SABR-on-H² via `ManifoldChernoff<Hyperbolic2>`; p99 ≈ 3 ms/tick (v2.8+) |

Run any example with:

```sh
cargo run -p semiflow-core --example heat_2d_demo
```

## Mathematical reference

- I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135.
  DOI [10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q) |
  [arXiv:2301.06765](https://arxiv.org/abs/2301.06765). Primary reference;
  Theorem 6 (Chernoff formula) and Theorem 3 (Laplace-Chernoff resolvent).
- S. Mazzucchi, V. Moretti, I. D. Remizov, O. G. Smolyanov, *Math. Nachr.* (2023).
  Riemannian manifold Chernoff formula with R/12 curvature correction. (ADR-0071, math.md §24)
- J. S. Howland, *Trans. AMS* **207** (1974) Theorem 1. Augmented-generator nonautonomous
  lift. (ADR-0070, math.md §23)
- Y. A. Butko, *Fract. Calc. Appl. Anal.* **21** (2018). Feynman-Kac killing for
  operator-level Dirichlet BC. (ADR-0068, math.md §21)
- J. Walsh, *Séminaire de Probabilités* **20** (1986). Image method for Neumann BC.
  (ADR-0072, math.md §25)
- Full normative spec: `contracts/semiflow-core.math.md` in the workspace root
  (§1–§45+ as of v7.0.0, properties schema v4.0.0).

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

## Bindings (other languages)

This crate is Rust-only. Sibling crates wrap it for other languages:

| Language | Crate | Distribution |
|----------|-------|--------------|
| C | `semiflow-ffi` (cdylib) | GitHub releases |
| Python | `semiflow-pde` (PyO3, abi3-py310) | GitHub releases; PyPI pending first publish |
| JavaScript | `@semiflow/wasm` (wasm-bindgen) | npm (pending first publish); `wasm-pack build` locally |

See the workspace [README](https://github.com/VolkovIlia/semiflow#bindings) for
cross-language status and build instructions.

## License

Dual MIT OR Apache-2.0. See [`LICENSE-MIT`](../../LICENSE-MIT) and
[`LICENSE-APACHE`](../../LICENSE-APACHE) in the workspace root.

## Contributing

See [CONTRIBUTING.md](https://github.com/VolkovIlia/semiflow/blob/master/CONTRIBUTING.md)
and [CODE_OF_CONDUCT.md](https://github.com/VolkovIlia/semiflow/blob/master/CODE_OF_CONDUCT.md).
Security reports to ilia.volkov@outlook.com per
[SECURITY.md](https://github.com/VolkovIlia/semiflow/blob/master/SECURITY.md).
