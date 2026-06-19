# SemiFlow User Guide

This guide is organized by **what you want to solve**. Each section names the
types to reach for and points at a runnable example. For the full type catalogue
and cargo feature flags, see
[`crates/semiflow-core/README.md`](../crates/semiflow-core/README.md); for the
API reference, see [docs.rs/semiflow-core](https://docs.rs/semiflow-core).

## Core concepts (read once)

SemiFlow approximates the semigroup `e^{tL}` of a linear operator `L` by the
Chernoff iteration `(S(t/n))ⁿ`, where `S(τ)` is a one-step kernel that satisfies
`S(0) = I` and `S'(0) = L`. You assemble three things:

1. **A grid** — `Grid1D::new(lo, hi, n)`, or `Grid2D` / `Grid3D` for tensor products.
2. **An initial condition** — `GridFn1D::from_fn(grid, |x| …)` (and 2D/3D variants).
3. **A Chernoff kernel** — e.g. `ShiftChernoff1D`, `DiffusionChernoff`,
   `ManifoldChernoff`, … — wrapped in the driver `ChernoffSemigroup::new(kernel, n)`
   (a thin form of the canonical `Evolver<C, F>`).

Then call `semi.evolve(t, &u0)` to get `u(t)`. Everything is generic over the
scalar type via the sealed `SemiflowFloat` trait (`f64` by default; `f32`, and
`Dual<F>` for automatic differentiation).

```rust
use semiflow_core::{Grid1D, GridFn1D, ShiftChernoff1D, ChernoffSemigroup};

let grid = Grid1D::new(-10.0, 10.0, 1000).expect("valid grid");
let u0   = GridFn1D::from_fn(grid, |x| (-x * x).exp());
let k    = ShiftChernoff1D::new(|_| 0.5_f64, |_| 0.0, |_| 0.0, 0.0, grid);
let u1   = ChernoffSemigroup::new(k, 100).expect("n >= 1")
            .evolve(1.0, &u0).expect("evolve ok");
```

## I want to solve a diffusion / advection–reaction equation

Operator `L = a(x)·∂² + b(x)·∂ + c(x)`. Use `ShiftChernoff1D` (formula (6) of the
theorem) or `DiffusionChernoff` / `DriftReactionChernoff`. Coefficients are
closures, so variable coefficients are free. For 2D/3D, build a `Grid2D`/`Grid3D`
and split per axis with `Strang2D` / `Strang3D`.

→ Examples: [`heat_2d_demo.rs`](../crates/semiflow-core/examples/heat_2d_demo.rs),
[`strang_advdiff_demo.rs`](../crates/semiflow-core/examples/strang_advdiff_demo.rs).

## I want higher-order accuracy

The ζ-ladder kernels raise the spatial order: `Diffusion4thZeta4Chernoff` (order 4),
`Diffusion6thZeta6Chernoff` (order 6), `Diffusion8thZeta8Chernoff` (order 8). Pick
the interpolation degree via `InterpKind` (default `OctonicHermite`). See
[precision-policy.md](precision-policy.md) for the accuracy/cost trade-offs and the
gated floors.

## I want boundary conditions

Set a `BoundaryPolicy` on the grid: `Reflect` (default, Neumann), `Dirichlet`,
`Robin { alpha, beta }`. For operator-level treatments use `KillingChernoff`
(absorbing/Dirichlet via Feynman–Kac), `ReflectedHeatChernoff` (Neumann via image
method), or `ObstacleChernoff` (obstacle / variational-inequality problems).

→ Example: [`boundary_demo.rs`](../crates/semiflow-core/examples/boundary_demo.rs).

## I want to solve a Schrödinger equation

Use `SchrödingerChernoffComplex<C>` with a complex carrier (`SemiflowComplex`).
The kinetic step is unitary by construction.

## I want PDEs on a manifold

`ManifoldChernoff` with a geometry backend: `Torus`, `Sphere2`, `Hyperbolic2`
(Poincaré disk), or `FubiniStudyCp1`. Enable the `R/12` curvature correction to
lift the order from 1 to 2. The SABR volatility model lives on `Hyperbolic2`.

→ Example: [`sabr_pricer.rs`](../crates/semiflow-core/examples/sabr_pricer.rs).

## I want hypoelliptic / sub-Riemannian operators

`HypoellipticChernoff<F, D, M>` uses a Strang–Hörmander step over step-2/step-3
Carnot groups. Backends: Kolmogorov, Heisenberg, Engel. For PDEs on metric graphs,
`QuantumGraphHeatChernoff<F>` enforces Kirchhoff vertex conditions.

## I want option pricing (CEV / Heston / SABR / rough vol)

Diffusion pricers map directly onto the kernels above. CEV is a 1D diffusion;
Heston and rough Heston are matrix/2D; SABR is manifold-based. The HFT-oriented
examples report per-tick latency.

→ Examples: [`cev_european_call.rs`](../crates/semiflow-core/examples/cev_european_call.rs),
[`heston_pricer.rs`](../crates/semiflow-core/examples/heston_pricer.rs),
[`rough_heston_pricer.rs`](../crates/semiflow-core/examples/rough_heston_pricer.rs),
[`latency_tail.rs`](../crates/semiflow-core/examples/latency_tail.rs).

## I want sensitivities / Greeks

Run any generic kernel with the scalar type `Dual<F>` (which implements
`SemiflowFloat`) to get forward-mode automatic differentiation through the whole
evolution at zero extra allocation.

## I want the resolvent or a nonautonomous problem

`LaplaceChernoffResolvent` evaluates `(λI − A)⁻¹g` as the Laplace transform of the
Chernoff semigroup. `HowlandLift` handles time-dependent generators via the
Howland augmented-generator trick. `ResolventJumpChernoff` gives O(1) large-`t`
cost for self-adjoint/sectorial generators.

→ Example: [`resolvent_perf.rs`](../crates/semiflow-core/examples/resolvent_perf.rs).

## Performance & precision

- Choose `n` (number of Chernoff steps) for accuracy; error decreases with order.
- Enable the `parallel` feature for multi-threaded `Strang2D`/`Strang3D` and the
  `simd` feature (default) for AVX2/NEON hot paths.
- Steady-state evolution is allocation-free (reused scratch buffers).
- See [precision-policy.md](precision-policy.md) and
  [api-stability.md](api-stability.md) for guarantees and gated floors.
