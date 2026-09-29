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
| `BoundaryPolicy::OddReflect` | Odd-image stencil BC (used by `DirichletHeat2ndChernoff`) |
| `KillingChernoff<C, R, F>` | Operator-level Dirichlet via Feynman-Kac killing |
| `Killing2ndChernoff<C, K, F>` | Order-2 soft-killing `e^{-τκ/2}·C(τ)·e^{-τκ/2}` |
| `ReflectedHeatChernoff<C, R, F>` | Neumann via Walsh 1986 image method |
| `DirichletHeat2ndChernoff<C, R, F>` | Order-2 absorbing wall via odd-image method (math §21.9) |
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
| `ManifoldChernoff<M, F>` | Curvature-corrected Gaussian on `T_xM`; order 1 (base) or 2 (with R/12) |
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
| `RegionMap` | DoF-aligned region partition enabling K>1 per-region parameter sensitivity |

**Scope:** constant-a `DiffusionChernoff<F>` only. Multi-parameter (K>1)
sensitivity via `ReverseChernoff::with_region_map`. Variable-coefficient
kernels are deferred.

### Tensor-train carrier

| Type | Description |
|------|-------------|
| `TtChernoff<F>` | TT-Chernoff evolver; storage `O(d·n·r²)` |
| `TtState<F>` | Tensor-train state `u(i₁,…,i_d) = G₁[i₁]·…·G_d[i_d]` |
| `VarCoefTt<F>` | Variable-coefficient TT carrier for separable diagonal `a_j(x_j)` |

**Scope:** linear diagonal-A Gaussian class (`TtChernoff`); separable diagonal
variable coefficients (`VarCoefTt`). Non-separable variable coefficients and
off-diagonal A are research-track. `VarCoefTt::new` returns
`SemiflowError::VarCoefOutOfClass` when inputs fall outside the separable
diagonal class.

### Gridless / particle-ensemble

| Type | Description |
|------|-------------|
| `GridlessChernoff<F, const D>` | Particle-ensemble Chernoff; implements `ChernoffFunction<F>` |
| `MeasureState<F, D>` | Particle-ensemble state; exposes `first_moment`, `variance`, `variance_per_axis` diagnostics (math §38.12) |
| `ParticleReduction` | Particle cap policy: `WeightedVoronoi { cap }` or `GaussianBackground` |

### Schrödinger, graph and quantum-graph operators

| Type | Description |
|------|-------------|
| `SchrödingerChernoffComplex` / `SchrodingerChernoff` | Schrödinger equation on a native complex (or real-pair) carrier |
| `GraphHeatChernoff` | Heat semigroup of a graph Laplacian |
| `GraphKrylovChernoff`, `graph_expmv_frechet` | Depth-independent `e^{−tL}·v` (Chebyshev / Lanczos) and its edge-weight Fréchet gradient |
| `QuantumGraphHeatChernoff` / `QuantumSchrödingerChernoff` | Metric graphs with Kirchhoff vertex conditions |

### Conservative, stiff and semilinear problems

| Type | Description |
|------|-------------|
| `ConservativeDiffusionChernoff`, `assemble_conservative_csr_1d`, `assemble_conservative_csr_nd` | Divergence-form `∂ₓ(k(x)∂ₓu)` with harmonic-mean faces (sharp material interfaces, optional contact resistance) |
| `MultilayerStack`, `multilayer_evolve`, `MassWeightedConservativeChernoff` | Stiff multilayer conduction in one depth-flat Krylov action |
| `SymmetricOperator`, `MassKOperator`, `EntrySensitivity`, `mass_lumped_evolve` | Externally assembled symmetric PSD sparse operators (FEM), the `(M, K)` problem without forming `M⁻¹K`, entry-wise gradients |
| `general_operator::GeneralOperator` | Possibly non-symmetric CSR operators via Taylor `expmv` |
| `phi_action`, `phi_action_batched`, `Etdrk4`, `Nonlinearity` | Semilinear `∂ₜu = Lu + N(u)` by exponential time differencing (order 4) |
| `SubordinatedChernoff` | Bochner–Phillips subordination of any engine |
| `HypoellipticChernoff` | Kolmogorov, Heisenberg and Engel (step-2/3 Carnot) operators |

### Executor

`ChernoffSemigroup<C, S>` — wraps a `ChernoffFunction` and a step count;
call `.evolve(t, &f)` to compute `(S(t/n))ⁿ f`.

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
| `graph_par_speedup.rs` | Parallel graph-heat speedup (`--features parallel`) |
| `rough_heston_pricer.rs` | Oracle-validated risk-neutral rough-Heston pricer (`--rate`/`--price`); honest scope: solver of a 4-factor Markov approximation, not validated vs true rough-Heston |

```bash
cargo run -p semiflow --example heat_2d_demo
```
