# SemiFlow

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Docs.rs](https://img.shields.io/badge/docs.rs-semiflow--core-blue)](https://docs.rs/semiflow-core)
[![Crates.io](https://img.shields.io/crates/v/semiflow-core)](https://crates.io/crates/semiflow-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

> **Status: `0.9.0-beta`** — first public beta. The API is stabilizing toward
> `1.0`; minor versions may make breaking changes. Bug reports and feedback welcome.

**SemiFlow is a `no_std` Rust library for solving evolution equations and PDEs
(`∂ₜu = Lu`) by Chernoff approximation of operator semigroups — no matrix
exponentials, no linear solves, embarrassingly parallel.**

The method evaluates `e^{tL}f` by iterating an explicit, allocation-free step
operator `S(τ)`: `(S(t/n))ⁿ f → e^{tL}f`. Each step fits in cache; the working
set is flat and steady-state evolution is zero-allocation. The core ships as a
pure `no_std + alloc` `rlib` with only 3 runtime dependencies, plus optional C,
Python, and WebAssembly bindings with near-full engine parity.

> Mathematical foundation: **Theorem 6 of Remizov (2025)**, *Vladikavkaz Math. J.*
> 27(4), 124–135 ([doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)).

## Why SemiFlow

- **60+ evolution-equation engines** — heat, Schrödinger, diffusion, graph
  Laplacians, manifold PDEs, hypoelliptic operators, and more (see [catalogue](#engine-catalogue)).
- **`no_std` + only 3 runtime deps** — num-traits, libm, num-complex; suitable
  for embedded, WASM, and HPC environments.
- **Up to 8th-order spatial accuracy** via the ζ-ladder (Richardson extrapolation
  on the Chernoff step); convergence gated in CI against closed-form oracles.
- **Curse-of-dimensionality escape** via tensor-train (TT) carriers for the
  Gaussian class: storage `O(d·n·r²)` instead of `O(nᵈ)`.
- **Reverse-mode automatic differentiation** via binomial checkpointing at
  `O(√n)` memory overhead — compute parameter gradients through the semigroup.
- **Four language surfaces** — Rust core, C ABI (FFI / `cdylib`), Python
  (PyO3 / maturin wheels, abi3-py310), and WebAssembly (wasm-bindgen / npm),
  with a lite default WASM bundle and opt-in `full` feature for heavy-grid engines.
- **18,992-symbol codebase, 279 verified execution flows** — every numerical
  claim tested in CI.

## Who is this for

- **Numerical / applied mathematicians** solving parabolic, Schrödinger,
  hypoelliptic, or manifold PDEs and wanting a verified, matrix-free integrator.
- **Quantitative / HFT engineers** pricing diffusion models (CEV, Heston, SABR,
  rough vol) where low, predictable per-tick latency and small working sets matter.
- **Researchers** who need a reference implementation of Chernoff-based semigroup
  approximation with convergence gated against closed-form oracles.

## Install

### Rust

```bash
cargo add semiflow-core
```

`no_std` users: disable default features:

```toml
semiflow-core = { version = "0.9.0-beta", default-features = false }
```

The `simd` (AVX2 / NEON) and `parallel` features require `std`.

### Other language bindings

| Language | How to install | Distribution |
|----------|---------------|--------------|
| Python | `pip install semiflow-pde` — `import semiflow` | PyPI (published per release) |
| JavaScript / WASM | `npm install semiflow` | npm (published per release) |
| C / C++ | Download a release artifact; see [`crates/semiflow-ffi/README.md`](crates/semiflow-ffi/README.md) | Header `semiflow.h` |

## Quickstart — heat equation in 30 seconds

Solve `∂ₜu = ½·∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1`:

```rust
use semiflow_core::{Grid1D, GridFn1D, ShiftChernoff1D, ChernoffSemigroup};

// Uniform grid [-10, 10] with 1000 nodes.
let grid = Grid1D::new(-10.0, 10.0, 1000).expect("valid grid");

// Initial condition u₀(x) = exp(-x²).
let u0 = GridFn1D::from_fn(grid, |x| (-x * x).exp());

// Operator L = a·∂² + b·∂ + c with a = ½, b = c = 0.
let chernoff = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.0, grid);

// Iterate (S(t/n))ⁿ with n = 100 steps to t = 1.
let semi = ChernoffSemigroup::new(chernoff, 100).expect("n >= 1");
let u1 = semi.evolve(1.0, &u0).expect("evolve ok");
// sup-norm error vs closed-form Gaussian kernel ≈ 3.2e-4 at n=100
```

A runnable version with the error check lives in
[`docs/QUICKSTART.md`](docs/QUICKSTART.md). For a use-case-driven tour
("I want to solve …"), see the [**User Guide**](docs/USER_GUIDE.md).

## Engine catalogue

| Family | Representative types | Notes |
|--------|----------------------|-------|
| Diffusion / advection–reaction | `DiffusionChernoff`, `ShiftChernoff1D`, `DriftReactionChernoff` | 1D/2D/3D, variable coefficients |
| Higher-order accuracy (ζ-ladder) | `Diffusion4thChernoff`, `Diffusion6thZeta6Chernoff`, `Diffusion8thZeta8Chernoff` | 4th–8th order via Richardson |
| 2D/3D tensor-product splitting | `Strang2D`, `Strang3D`, `NonSeparable2DAnisotropicChernoff` | Palindromic Strang; parallel |
| Schrödinger | `SchrödingerChernoffComplex` | Native complex carrier |
| Matrix-valued systems | `MatrixDiffusionChernoff`, `MatrixDiffusionChernoff2D/3D` | Coupled M-component operators |
| Riemannian manifolds | `ManifoldChernoff` + `Sphere2`, `Hyperbolic2`, `Torus`, `FubiniStudyCp1` | R/12 curvature-corrected |
| Hypoelliptic / sub-Riemannian | `HypoellipticChernoff` (Kolmogorov, Heisenberg, Engel) | Strang–Hörmander, step-2/3 Carnot |
| Graph and quantum-graph Laplacians | `GraphHeatChernoff`, `QuantumGraphHeatChernoff`, `QuantumSchrödingerChernoff` | Kirchhoff vertex conditions |
| Boundary conditions | `KillingChernoff`, `ReflectedHeatChernoff`, `ObstacleChernoff`, `BoundaryPolicy` | Dirichlet / Neumann / Robin / obstacle |
| Resolvent and nonautonomous | `LaplaceChernoffResolvent`, `HowlandLift` | `(λI−A)⁻¹g`; Howland augmented generator |
| High-dimensional / sparse-grid | `AnisotropicShiftChernoffND`, `SmolyakGridND` | Gauss-Hermite / Smolyak, d ≥ 5 |
| Tensor-train (TT) carrier | `TtChernoff`, `TtState` | Curse-of-dimensionality escape; `O(d·n·r²)` |
| Gridless / particle ensemble | `GridlessChernoff` | Particle-ensemble Chernoff, d ≤ ~10 |
| Reverse-mode AD | `ReverseChernoff`, `CheckpointSchedule` | Binomial checkpointing, `O(√n)` memory |
| Lévy subordination | `SubordinatedChernoff` | Subordinate any engine to a Lévy process |
| Adaptive stepping | `AdaptivePI` | PI step-size controller for any `ChernoffFunction` |

Some families are experimental and gated behind cargo features (e.g. `s3-poc`,
`full` for WASM). See [`crates/semiflow-core/README.md`](crates/semiflow-core/README.md)
for the full type catalogue, constructor signatures, and feature flags.

## Bindings

| Language | Crate / Package | Key feature |
|----------|----------------|-------------|
| Rust | `semiflow-core` (crates.io) | Full engine catalogue, `no_std + alloc` |
| C / C++ | `semiflow-ffi` (cdylib; release artifacts) | `extern "C"` ABI, `catch_unwind` on every entry point, header `semiflow.h` |
| Python | `semiflow-pde` (PyPI; maturin wheel) | PyO3, abi3-py310 wheel (Python 3.10–3.13), GIL-release via `py.detach` |
| JavaScript / WASM | `@semiflow/wasm` (npm; wasm-bindgen) | Lite default bundle; `full` feature for higher-order / 2D / 3D engines |

All four surfaces target near-full engine parity. For build instructions and
cross-language examples see [docs/BINDINGS.md](docs/BINDINGS.md).

## Documentation

| If you want to… | Read |
|-----------------|------|
| Get started | [Quickstart](docs/QUICKSTART.md) · [User Guide](docs/USER_GUIDE.md) |
| Use a binding | [Bindings Guide](docs/BINDINGS.md) (C / Python / WASM) |
| Browse the API | [docs.rs/semiflow-core](https://docs.rs/semiflow-core) |
| See worked examples | [`examples/`](crates/semiflow-core/examples) ([index](crates/semiflow-core/examples/README.md)) |
| Understand accuracy / precision policy | [precision-policy.md](docs/precision-policy.md) · [api-stability.md](docs/api-stability.md) |
| Read design decisions | [`docs/adr/`](docs/adr) (Architecture Decision Records) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Accuracy and design

Every numerical claim is gated in CI against a closed-form or high-order
reference oracle (convergence-order tests and sup-norm-floor tests). Because
`GridFnXD<F>` types are `Vec<F>`-backed with reused scratch buffers, steady-state
evolution is zero-allocation — but treat memory and latency as **measured
properties of the concrete grid types, not contractual guarantees of the trait
API**. Reproducible benchmarks are published in the benchmarks repository.

Design principles: `no_std + alloc` core; only 3 runtime dependencies; SIMD
hot paths isolated to `src/simd/` (AVX2 on x86_64, NEON on aarch64, scalar
fallback elsewhere); bit-reproducible parallelism across thread counts (ADR-0018);
zero unsafe in math kernels.

## Mathematical foundation

The method is rooted in the Chernoff product formula and specifically implements
**Theorem 6 of I. D. Remizov (2025)**:

> Remizov, I. D. (2025). *Chernoff Approximations of the Solution of Linear ODE
> with Variable Coefficients.* Vladikavkaz Mathematical Journal, 27(4), 124–135.
> [doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.

## Author

SemiFlow is created and maintained by **Ilia Volkov**.

## Citation

If you use SemiFlow in academic work, please cite the software (see
[`CITATION.cff`](CITATION.cff)) and reference the underlying theorem above:

> Volkov, I. (2026). *SemiFlow* (version 0.9.0-beta) [Computer software].
> https://github.com/VolkovIlia/semiflow

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports go to
ilia.volkov@outlook.com per [SECURITY.md](SECURITY.md) — do not open public
issues for vulnerabilities.
