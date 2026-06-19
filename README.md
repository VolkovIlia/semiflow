# SemiFlow

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Docs.rs](https://img.shields.io/badge/docs.rs-semiflow--core-blue)](https://docs.rs/semiflow-core)
[![Crates.io](https://img.shields.io/crates/v/semiflow-core)](https://crates.io/crates/semiflow-core)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)

> **Status: `0.9.0-beta`** — first public beta. The API is stabilizing toward
> `1.0`; minor versions may make breaking changes. Bug reports and feedback welcome.

**SemiFlow is a Rust library for solving evolution equations `∂ₜu = Lu` by
Chernoff approximation of operator semigroups — no matrix exponentials, no
linear solves, embarrassingly parallel.**

It integrates `(S(t/n))ⁿ → e^{tL}` directly: each step `S(τ)` is an explicit,
allocation-free kernel, so memory stays flat and the hot loop fits in cache.
The core is `no_std + alloc`, dependency-light, and ships C, Python, and
WebAssembly bindings.

> The method implements **Theorem 6 of Remizov (2025)**, *Vladikavkaz Math. J.*
> 27(4), 124–135 ([doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)).

## Who is this for

- **Numerical / applied mathematicians** solving parabolic, Schrödinger,
  hypoelliptic, or manifold PDEs and wanting a verified, matrix-free integrator.
- **Quantitative / HFT engineers** pricing diffusion models (CEV, Heston, SABR,
  rough vol) where low, predictable per-tick latency and small working sets matter.
- **Researchers** who need a reference implementation of Chernoff-based semigroup
  approximation with convergence gated against closed-form oracles.

## Install

```bash
# Rust
cargo add semiflow-core
```

| Binding | Install | Status |
|---------|---------|--------|
| Rust (crates.io) | `cargo add semiflow-core` | stable |
| Python (PyPI) | `pip install semiflow-pde` — imports as `import semiflow` | published per release |
| JavaScript / WASM (npm) | `npm install semiflow` | published per release |
| C / C++ (FFI) | download a release artifact; see [`crates/semiflow-ffi/README.md`](crates/semiflow-ffi/README.md) | header `semiflow.h` |

`no_std` users: disable default features (`semiflow-core = { version = "0.9.0-beta", default-features = false }`).
The `simd` and `parallel` features require `std`.

## Quickstart — heat equation in 30 seconds

Solve `∂ₜu = ½·∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1` and compare against the
closed-form Gaussian heat kernel:

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
// sup-norm error vs the (1+2t)^{-1/2} exp(-x²/(1+2t)) oracle ≈ 3.2e-4
```

A runnable version with the error check lives in
[`docs/QUICKSTART.md`](docs/QUICKSTART.md). For a use-case-driven tour
("I want to solve …"), see the [**User Guide**](docs/USER_GUIDE.md).

## What you can solve

| Family | Representative types | Notes |
|--------|----------------------|-------|
| Diffusion / advection–reaction | `ShiftChernoff1D`, `DiffusionChernoff`, `DriftReactionChernoff` | 1D/2D/3D, variable coefficients |
| Higher-order accuracy | `Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`, `Diffusion8thZeta8Chernoff` | ζ-ladder, Richardson |
| Tensor-product splitting | `Strang2D`, `Strang3D`, `NonSeparable2DAnisotropicChernoff` | per-axis Strang, parallel |
| Schrödinger | `SchrödingerChernoffComplex` | native complex carrier |
| Manifolds | `ManifoldChernoff` + `Torus` / `Sphere2` / `Hyperbolic2` / `FubiniStudyCp1` | curvature-corrected |
| Hypoelliptic / sub-Riemannian | `HypoellipticChernoff` (Kolmogorov, Heisenberg, Engel) | Strang–Hörmander |
| Graphs | `GraphHeatChernoff`, `QuantumGraphHeatChernoff` | Kirchhoff vertex conditions |
| Boundary value problems | `KillingChernoff`, `ReflectedHeatChernoff`, `ObstacleChernoff`, `BoundaryPolicy` | Dirichlet / Neumann / Robin / obstacle |
| Resolvent / nonautonomous | `LaplaceChernoffResolvent`, `HowlandLift`, `ResolventJumpChernoff` | `(λI−A)⁻¹g`, time-dependent |
| Sensitivities (AD) | `Dual<F>: SemiflowFloat` | forward-mode Greeks at zero allocation |

Some families are experimental and gated behind cargo features (e.g. `s3-poc`).
See the per-crate [`semiflow-core/README.md`](crates/semiflow-core/README.md) for
the full type catalogue and feature flags.

## Documentation

| If you want to… | Read |
|-----------------|------|
| Get started | [Quickstart](docs/QUICKSTART.md) · [User Guide](docs/USER_GUIDE.md) |
| Use a binding | [Bindings Guide](docs/BINDINGS.md) (C / Python / WASM) |
| Browse the API | [docs.rs/semiflow-core](https://docs.rs/semiflow-core) |
| See worked examples | [`examples/`](crates/semiflow-core/examples) ([index](crates/semiflow-core/examples/README.md)) |
| Understand accuracy / performance | [precision-policy.md](docs/precision-policy.md) · [api-stability.md](docs/api-stability.md) |
| Read design decisions | [`docs/adr/`](docs/adr) (Architecture Decision Records) |
| Contribute | [CONTRIBUTING.md](CONTRIBUTING.md) |

## Accuracy & performance

Every numerical claim is gated in CI against a closed-form or high-order
reference oracle (convergence-order and sup-norm-floor tests). Because the
concrete `GridFnXD<F>` types are `Vec<F>`-backed with reused scratch buffers, the
working set is small and steady-state evolution is allocation-free — but
treat memory/latency as **measured properties of the concrete grid types, not
contractual guarantees of the trait API**. Reproducible benchmarks are being
re-run for this release; see the benchmarks repository for the current figures.

## License

Dual-licensed under either of [MIT](LICENSE-MIT) or
[Apache-2.0](LICENSE-APACHE) at your option.

## Author

SemiFlow is created and maintained by **Ilia Volkov** (Волков Илья Николаевич).

## Acknowledgments

SemiFlow implements **Theorem 6 of I. D. Remizov (2025)**. We gratefully thank
I. D. Remizov, whose work provides the mathematical foundation for the library's
numerical method:

> Remizov, I. D. (2025). *Chernoff Approximations of the Solution of Linear ODE
> with Variable Coefficients.* Vladikavkaz Mathematical Journal, 27(4), 124–135.
> [doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)

## Citation

If you use SemiFlow in academic work, please cite the software (see
[`CITATION.cff`](CITATION.cff)) and reference the underlying theorem above:

> Volkov, I. (2026). *SemiFlow* (version 0.9.0-beta) [Computer software].
> https://github.com/VolkovIlia/semiflow
