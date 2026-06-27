# SemiFlow

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Docs.rs](https://img.shields.io/badge/docs.rs-semiflow-blue)](https://docs.rs/semiflow)
[![Crates.io](https://img.shields.io/crates/v/semiflow)](https://crates.io/crates/semiflow)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](#license)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.20837851.svg)](https://doi.org/10.5281/zenodo.20837851)

> **Status: `0.10.0-beta`** — active beta. The API is stabilizing toward
> `1.0`; minor versions may make breaking changes. Bug reports and feedback welcome.

**SemiFlow is a `no_std` Rust library for solving evolution equations and PDEs
(`∂ₜu = Lu`) by Chernoff approximation of operator semigroups — no matrix
exponentials, no linear solves, flat memory footprint.**

> **Honest performance note (iter-8, HEAD b923777, 45 families):** SemiFlow's
> primary measured advantage is **memory frugality** — a flat ~3 MB working
> set across all 1D/2D/3D families, vs 50–418 MB for heavy frameworks (KIOPS,
> Dedalus, scipy-mol), with 33/43 head-to-head pairs below the 0.5× memory
> gate (76.7%). Wallclock is **not a general strength**: SemiFlow is uniformly
> slower than adaptive ODE solvers and spectral methods for matched-accuracy
> PDE solving (H-WALL FALSIFIED by iter-8; e.g. 730× slower than
> SUNDIALS-CVODE at 5e-5 accuracy; 7303× slower than QuantLib FDM-CEV at
> 1e-2 accuracy). Parallelism pays off only at large 3D grid sizes
> (eta8=0.908 for 3D fine; eta8≈0.125 for 2D). The library wins in two
> additional confirmed niches: (1) tail-latency-sensitive HFT pricing —
> `Diffusion4thChernoff` achieves **41 ns p99.9** vs QuantLib V3 CEV 9573 ns
> (**233×** clean, **284×** under DRAM stress) at matched accuracy, confirmed
> on b923777; (2) L1-resident S³ low-rank carriers — TtChernoff 0.0008%
> L1d-miss, ReverseChernoff 0.0019%, roughly 4 orders of magnitude below a
> dense 256 KB working set (80.9% L1d-miss). Both are niche results;
> neither generalizes to "SemiFlow is fast" or "SemiFlow computes in L1."

The method evaluates `e^{tL}f` by iterating an explicit, allocation-free step
operator `S(τ)`: `(S(t/n))ⁿ f → e^{tL}f`. Each step fits in cache; the working
set is flat and steady-state evolution is zero-allocation. The core ships as a
pure `no_std + alloc` `rlib` with only 3 runtime dependencies, plus optional C,
Python, and WebAssembly bindings with near-full engine parity.

> Mathematical foundation: **Theorem 6 of Remizov (2025)**, *Vladikavkaz Math. J.*
> 27(4), 124–135 ([doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)).

## Why SemiFlow

- **65+ evolution-equation engines** — heat, Schrödinger, diffusion, conservative
  variable-coefficient diffusion (harmonic-mean faces), graph Laplacians,
  manifold PDEs, hypoelliptic operators, semilinear ETD, and more (see
  [catalogue](#engine-catalogue)).
- **`no_std` + only 3 runtime deps** — num-traits, libm, num-complex; suitable
  for embedded, WASM, and HPC environments.
- **Up to 8th-order spatial accuracy** via the ζ-ladder (Richardson extrapolation
  on the Chernoff step); convergence gated in CI against closed-form oracles.
- **Curse-of-dimensionality escape** via tensor-train (TT) carriers for the
  Gaussian class: storage `O(d·n·r²)` instead of `O(nᵈ)`. Variable-coefficient
  separable diagonal diffusion now supported via `VarCoefTt` (fail-loud
  `VarCoefOutOfClass` for non-separable inputs).
- **Reverse-mode automatic differentiation** via binomial checkpointing at
  `O(√n)` memory overhead — compute parameter gradients through the semigroup.
  Multi-parameter (K>1) sensitivity supported via `RegionMap` (per-region θ).
- **Native `f32` + f32 SIMD** — all leaf 1D kernels implement `ChernoffFunction<f32>`;
  a dedicated f32x8 AVX2 / f32x4 NEON kernel is provided with no FMA dependency.
- **Order-2 Dirichlet boundary** — `DirichletHeat2ndChernoff` with
  `BoundaryPolicy::OddReflect` (odd-image method, §21.9); order-2 sibling of
  `ReflectedHeatChernoff`.
- **Gridless variance / MSE diagnostic** — `MeasureState` now exposes
  `first_moment`, `variance`, and `variance_per_axis` for particle-ensemble
  convergence diagnostics (math §38.12).
- **Depth-independent graph-semigroup action and Fréchet gradient** —
  `GraphKrylovChernoff` computes `e^{−tL}·v` with matvec count set by `ε` and
  `t‖L‖`, flat in the depth `t`; `graph_expmv_frechet` returns `∂J/∂w` for all
  edge weights in one augmented solve — both forward and backward are
  depth-independent (ADR-0185).
- **Generic symmetric-operator entry point** — `SymmetricOperator::from_csr`
  accepts any externally-assembled symmetric PSD sparse matrix (FEM stiffness,
  anisotropic conductivity); `MassKOperator` handles the generalized `(M,K)`
  eigenproblem without forming `M⁻¹K` (ADR-0186).
- **Conservative divergence-form diffusion** — `ConservativeDiffusionChernoff`
  with harmonic-mean face conductivities handles sharp material interfaces
  (k-contrast up to 3025:1) where the non-conservative pointwise expansion fails;
  optional contact resistance `R_c`; separable N-D assembler (ADR-0187).
- **Stiff multilayer conduction** — `multilayer_evolve` propagates the full
  re-entry heat-transfer problem in one depth-flat Krylov action (~28000× fewer
  operator applications than explicit CFL on the Shuttle TPS stack; ADR-0188).
- **Semilinear ETD** — `phi_action` / `Etdrk4` integrates `∂ₜu = Lu + N(u)` at
  order 4 without splitting or re-discretizing `L`; `N(u)` is a declarative
  `Nonlinearity` trait (Allen–Cahn, Burgers, Gray–Scott, KS menus at the PyO3
  surface; ADR-0189).
- **Four language surfaces** — Rust core, C ABI (FFI / `cdylib`), Python
  (PyO3 / maturin wheels, abi3-py310), and WebAssembly (wasm-bindgen / npm),
  with a lite default WASM bundle and opt-in `full` feature for heavy-grid engines.
  The three binding surfaces now reach the full operator zoo (see [Bindings](#bindings)).
- **17,848-symbol codebase, 279 verified execution flows** — every numerical
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
cargo add semiflow
```

`no_std` users: disable default features:

```toml
semiflow = { version = "0.10.0-beta", default-features = false }
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
| Conservative (divergence-form) diffusion | `ConservativeDiffusionChernoff`, `assemble_conservative_csr_1d`, `assemble_conservative_csr_nd` | Harmonic-mean faces; sharp k-jumps; optional contact resistance; separable N-D |
| Stiff multilayer conduction | `MultilayerStack`, `multilayer_evolve`, `MassWeightedConservativeChernoff` | Per-layer `(k, ρc)` stack; mass-weighted Krylov; ~28000× fewer matvecs than explicit CFL |
| Semilinear ETD | `phi_action`, `phi_action_batched`, `Etdrk4`, `Nonlinearity` | `∂ₜu = Lu + N(u)`; order-4 Cox–Matthews; φ-functions via augmented matvec |
| Generic symmetric-operator entry | `SymmetricOperator`, `MassKOperator`, `EntrySensitivity`, `mass_lumped_evolve` | Externally-assembled PSD CSR; `(M,K)` eigenproblem; entry-wise Fréchet gradient |
| Graph and quantum-graph Laplacians | `GraphHeatChernoff`, `GraphKrylovChernoff`, `QuantumGraphHeatChernoff`, `QuantumSchrödingerChernoff` | Kirchhoff vertex conditions; depth-independent Krylov + Fréchet gradient |
| Boundary conditions | `KillingChernoff`, `ReflectedHeatChernoff`, `DirichletHeat2ndChernoff`, `ObstacleChernoff`, `BoundaryPolicy` | Dirichlet (order-2) / Neumann / Robin / obstacle |
| Resolvent and nonautonomous | `LaplaceChernoffResolvent`, `HowlandLift` | `(λI−A)⁻¹g`; Howland augmented generator |
| High-dimensional / sparse-grid | `AnisotropicShiftChernoffND`, `SmolyakGridND` | Gauss-Hermite / Smolyak, d ≥ 5 |
| Tensor-train (TT) carrier | `TtChernoff`, `TtState`, `VarCoefTt` | Curse-of-dimensionality escape; `O(d·n·r²)`; variable-coef separable diagonal via `VarCoefTt` |
| Gridless / particle ensemble | `GridlessChernoff` | Particle-ensemble Chernoff, d ≤ ~10 |
| Reverse-mode AD | `ReverseChernoff`, `CheckpointSchedule` | Binomial checkpointing, `O(√n)` memory |
| Lévy subordination | `SubordinatedChernoff` | Subordinate any engine to a Lévy process |
| Adaptive stepping | `AdaptivePI` | PI step-size controller for any `ChernoffFunction` |

Some families are experimental and gated behind cargo features (e.g. `s3-poc`,
`full` for WASM). See [`crates/semiflow/README.md`](crates/semiflow/README.md)
for the full type catalogue, constructor signatures, and feature flags.

## Bindings

| Language | Crate / Package | Key feature |
|----------|----------------|-------------|
| Rust | `semiflow` (crates.io) | Full engine catalogue, `no_std + alloc` |
| C / C++ | `semiflow-ffi` (cdylib; release artifacts) | `extern "C"` ABI, `catch_unwind` on every entry point, header `semiflow.h` |
| Python | `semiflow-pde` (PyPI; maturin wheel) | PyO3, abi3-py310 wheel (Python 3.10–3.13), GIL-release via `py.detach` |
| JavaScript / WASM | `@semiflow/wasm` (npm; wasm-bindgen) | Lite default bundle; `full` feature for higher-order / 2D / 3D engines |

All three non-Rust surfaces now expose a curated mirror of the user-facing operator
zoo, including the S³ carriers (`TtState`, `TtEvolver`, `TtCoupledEvolver`,
`MeasureState`, `GridlessEvolver`) and recently added engines
(`DiffusionExpmvChernoff`, `DriftReactionZeta4Chernoff`, `Killing2ndChernoff`,
`MatrixDiffusionChernoff2D/3D`, `DirichletHeat2ndChernoff`, `VarCoefTt`).
Variable coefficients cross the binding boundary as pre-sampled arrays, not live
closures. The binding surface is a curated mirror of the user-facing operator zoo,
not a 1:1 map of internal composition types (e.g. `AxisLift`, `StrangSplit` are
not directly exposed). v0.10.0-beta adds `GraphKrylov` (PyO3 pyclass) and `graph_expmv_frechet`
(PyO3 pyfunction) for depth-independent graph-semigroup actions and edge-weight
Fréchet gradients; `sym_op_evolve`, `mass_k_evolve`, and `sym_op_entry_grad`
functions for generic symmetric operators (ADR-0185/0186). Deferred (bindings label):
conservative diffusion, multilayer, and ETD PyO3 surfaces; C-ABI for new types.
The one remaining PyO3-only deferral is `GraphAdjoint`'s time-dependent Laplacian
constructor accepting a live callback — the pre-sampled path
(`GraphAdjointPresampled` / `smf_graph_adjoint_new_presampled`) covers the common
case and is available in all three surfaces.

For build instructions and cross-language examples see [docs/BINDINGS.md](docs/BINDINGS.md).

## Documentation

| If you want to… | Read |
|-----------------|------|
| Get started | [Quickstart](docs/QUICKSTART.md) · [User Guide](docs/USER_GUIDE.md) |
| Use a binding | [Bindings Guide](docs/BINDINGS.md) (C / Python / WASM) |
| Browse the API | [docs.rs/semiflow](https://docs.rs/semiflow) |
| See worked examples | [`examples/`](crates/semiflow/examples) ([index](crates/semiflow/examples/README.md)) |
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

The current authoritative benchmark is **iter-8** (semiflow HEAD b923777,
45 families, i7-12700K, 12 reps + 3 warmup). Key verdicts: H-MEM
PARTIAL_SUPPORT (33/43 pairs RC < competitor; flat ~3 MB vs 50–418 MB for
heavy frameworks); H-WALL FALSIFIED (RC slower than adaptive/spectral solvers
at matched accuracy); H-PAR FALSIFIED as universal claim (eta8=0.908 only for
large 3D; ~0.125 for 2D); S³ capabilities (TtChernoff 524288× storage
advantage at d=4, ReverseChernoff O(√n) checkpoint slope=0.496, GridlessChernoff
flat 13.9/22.5 KB at d=1/2) all SUPPORT. Iter-8 addendum (b923777) confirms
two niche advantages: (1) HFT tail-latency 233× p99.9 clean / 284× under DRAM
stress vs QuantLib V3 CEV; (2) S³ low-rank carriers L1-resident (TtChernoff
0.0008% L1d-miss, ReverseChernoff 0.0019%). Neither generalizes beyond the
niche. Source: `remizov-publications/benchmarks/results/aggregate-iter8/`
and `benchmarks/hft-latency-tail/data/phase-e-summary.md`.

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

## How to cite

If you use SemiFlow in academic work, cite both the **software** and the
**underlying theorem**:

**Software** — use the GitHub "Cite this repository" widget (top-right of the
repo) which reads [`CITATION.cff`](CITATION.cff), or use this BibTeX entry:

```bibtex
@software{volkov2026semiflow,
  author    = {Volkov, Ilia},
  title     = {{SemiFlow}: {Chernoff} Approximation of Operator Semigroups},
  year      = {2026},
  version   = {0.10.0-beta},
  doi       = {10.5281/zenodo.20837851},
  url       = {https://doi.org/10.5281/zenodo.20837851}
}
```

DOI (concept DOI — always resolves to the latest version):
**[10.5281/zenodo.20837851](https://doi.org/10.5281/zenodo.20837851)**
See [`CITATION.cff`](CITATION.cff) for the full CFF entry.

**Underlying theorem** — always cite this alongside the software:

> Remizov, I. D. (2025). *Chernoff Approximations of the Solution of Linear ODE
> with Variable Coefficients.* Vladikavkaz Mathematical Journal, 27(4), 124–135.
> [doi:10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)

## Contributing and security

See [CONTRIBUTING.md](CONTRIBUTING.md) and
[CODE_OF_CONDUCT.md](CODE_OF_CONDUCT.md). Security reports go to
ilia.volkov@outlook.com per [SECURITY.md](SECURITY.md) — do not open public
issues for vulnerabilities.
