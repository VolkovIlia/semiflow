# SemiFlow User Guide

This guide is organized by **what you want to solve**. Each section names the
types to reach for and points at a runnable example. For the full type catalogue
and cargo feature flags, see
[`crates/semiflow/README.md`](../crates/semiflow/README.md); for the
API reference, see [docs.rs/semiflow](https://docs.rs/semiflow).

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

→ Examples: [`heat_2d_demo.rs`](../crates/semiflow/examples/heat_2d_demo.rs),
[`strang_advdiff_demo.rs`](../crates/semiflow/examples/strang_advdiff_demo.rs).

## I want higher-order accuracy

The ζ-ladder kernels raise the spatial order: `Diffusion4thZeta4Chernoff` (order 4),
`Diffusion6thZeta6Chernoff` (order 6), `Diffusion8thZeta8Chernoff` (order 8). Pick
the interpolation degree via `InterpKind` (default `OctonicHermite`). See
[precision-policy.md](precision-policy.md) for the accuracy/cost trade-offs and the
gated floors.

## I want boundary conditions

Set a `BoundaryPolicy` on the grid: `Reflect` (default, Neumann), `Dirichlet`,
`Robin { alpha, beta }`. For operator-level treatments:

- `KillingChernoff` — absorbing / Dirichlet via Feynman–Kac.
- `ReflectedHeatChernoff` — Neumann via Walsh 1986 image method.
- `DirichletHeat2ndChernoff` — absorbing / Dirichlet via the odd-image method
  (math §21.9); order 2 in the continuation region; use
  `BoundaryPolicy::OddReflect` with this kernel.
- `ObstacleChernoff` — obstacle / variational-inequality evolver.

→ Example: [`boundary_demo.rs`](../crates/semiflow/examples/boundary_demo.rs).

## I want to solve a Schrödinger equation

Use `SchrödingerChernoffComplex<C>` with a complex carrier (`SemiflowComplex`).
The kinetic step is unitary by construction.

## I want PDEs on a manifold

`ManifoldChernoff` with a geometry backend: `Torus`, `Sphere2`, `Hyperbolic2`
(Poincaré disk), or `FubiniStudyCp1`. Enable the `R/12` curvature correction to
lift the order from 1 to 2. The SABR volatility model lives on `Hyperbolic2`.

→ Example: [`sabr_pricer.rs`](../crates/semiflow/examples/sabr_pricer.rs).

## I want hypoelliptic / sub-Riemannian operators

`HypoellipticChernoff<F, D, M>` uses a Strang–Hörmander step over step-2/step-3
Carnot groups. Backends: Kolmogorov, Heisenberg, Engel. For PDEs on metric graphs,
`QuantumGraphHeatChernoff<F>` enforces Kirchhoff vertex conditions.

## I want option pricing (CEV / Heston / SABR / rough vol)

Diffusion pricers map directly onto the kernels above. CEV is a 1D diffusion;
Heston and rough Heston are matrix/2D; SABR is manifold-based. The HFT-oriented
examples report per-tick latency.

→ Examples: [`cev_european_call.rs`](../crates/semiflow/examples/cev_european_call.rs),
[`heston_pricer.rs`](../crates/semiflow/examples/heston_pricer.rs),
[`rough_heston_pricer.rs`](../crates/semiflow/examples/rough_heston_pricer.rs)
(oracle-validated 4-factor Markov approximation; see the honest two-tier validation
note in the [rough-vol section](#i-want-to-price-rough-heston--rough-vol-models)),
[`latency_tail.rs`](../crates/semiflow/examples/latency_tail.rs).

## I want sensitivities / Greeks

Run any generic kernel with the scalar type `Dual<F>` (which implements
`SemiflowFloat`) to get forward-mode automatic differentiation through the whole
evolution at zero extra allocation.

## I want the resolvent or a nonautonomous problem

`LaplaceChernoffResolvent` evaluates `(λI − A)⁻¹g` as the Laplace transform of the
Chernoff semigroup. `HowlandLift` handles time-dependent generators via the
Howland augmented-generator trick. `ResolventJumpChernoff` gives O(1) large-`t`
cost for self-adjoint/sectorial generators.

For a time-dependent graph Laplacian without a live callback, supply the Laplacian
sequence as a pre-sampled array via `MagnusGraphHeatChernoff::from_presampled` (Rust)
or `GraphAdjointPresampled` (Python / WASM) / `smf_graph_adjoint_new_presampled` (C).
The pre-sampled path works across all three binding surfaces; it expects 2·n_steps
Laplacian samples (GL4-aware). The live-callback constructor of `GraphAdjoint` that
accepts a closure is available in PyO3 only.

→ Example: [`resolvent_perf.rs`](../crates/semiflow/examples/resolvent_perf.rs).

## I want gridless (particle-ensemble) diagnostics

After each evolution step with `GridlessChernoff`, inspect the `MeasureState`
output to monitor convergence:

```rust
use semiflow_core::{MeasureState, GridlessChernoff, ChernoffSemigroup};

// ... build evolver and evolve to get `state: MeasureState<f64, 2>` ...
let mean:         [f64; 2] = state.first_moment();
let total_var:    f64      = state.variance();          // E[|x|²] − |E[x]|² (§38.12)
let per_axis_var: [f64; 2] = state.variance_per_axis(); // per-component variance
```

`variance` is the total scalar variance (§38.12). `variance_per_axis` returns each
component separately. These are diagnostic-only — they do not gate correctness of
the evolution.

## I want to use the S³ carriers from Python

The `TtEvolver`, `TtCoupledEvolver`, `TtState`, `VarCoefTtEvolver`, `MeasureState`,
and `GridlessEvolver` types are importable directly from the `semiflow` package
(Python wheel `semiflow-pde`):

```python
from semiflow import TtState, TtEvolver, TtCoupledEvolver
from semiflow import MeasureState, GridlessEvolver
from semiflow import VarCoefTtEvolver

# TtEvolver: TT-Chernoff for the diagonal-A Gaussian class
ev = TtEvolver(d=4, n=32, a_diag=[0.5, 0.5, 0.5, 0.5], n_steps=50)
state_out = ev.evolve(t=1.0, state_in=initial_tt_state)

# VarCoefTtEvolver: separable diagonal variable coefficients
# Raises SemiflowError (VarCoefOutOfClass) if coefficients are non-separable
ev2 = VarCoefTtEvolver(d=2, n=64, a_seqs=a_seqs_list, n_steps=100)
```

`TtState` holds the tensor-train grid function; `MeasureState` holds the
particle-ensemble state for `GridlessEvolver`. Both can be constructed, inspected,
and passed across the evolve boundary in Python.

## I want to price rough-Heston / rough-vol models

`examples/rough_heston_pricer.rs` is an oracle-validated risk-neutral pricer for a
4-factor Markov approximation of the rough-Heston model.

**Tiered validation claim (important):**

- `G_ROUGH_HESTON_MC_PARITY` (RELEASE-BLOCKING): the pricer matches a Monte Carlo
  oracle within the stated tolerance for the 4-factor Markov approximation.
- `A_ROUGH_HESTON_MODEL_BIAS` (ADVISORY): the Markov approximation itself is
  ~1–5% O(H)-biased from the true rough-Heston model (expected; this is an
  approximation of an approximation).

This is an **oracle-validated solver of an approximate model**, not a validated
approximation of true rough-Heston. Do not use the bias-advisory tier to claim
high accuracy for the full model.

```bash
# Risk-neutral mode (r=0.05, spot pricing)
cargo run --release -p semiflow --example rough_heston_pricer \
    -- --rate 0.05 --price

# Latency mode (r=0.0, recovers the latency demonstrator)
cargo run --release -p semiflow --example rough_heston_pricer \
    -- --rate 0.0
```

## Performance & precision

- Choose `n` (number of Chernoff steps) for accuracy; error decreases with order.
- Enable the `parallel` feature for multi-threaded `Strang2D`/`Strang3D` and the
  `simd` feature (default) for AVX2/NEON hot paths.
- Steady-state evolution is allocation-free (reused scratch buffers).
- See [precision-policy.md](precision-policy.md) and
  [api-stability.md](api-stability.md) for guarantees and gated floors.

### Honest benchmark summary (iter-8, HEAD b923777, 45 families)

SemiFlow's primary measured advantage is **memory frugality**: flat ~3 MB working
set across all 1D/2D/3D families, vs 50–418 MB for heavy frameworks (KIOPS 418 MB,
Dedalus ~146 MB, scipy-mol-3d 90 MB). 33/43 head-to-head pairs show RC using less
memory than the competitor.

**Wallclock** is honestly not a general strength. RC is uniformly slower than
adaptive ODE solvers and spectral methods at matched accuracy (e.g. 730× slower
than SUNDIALS-CVODE on 1D heat at 5e-5 accuracy; 7303× slower than QuantLib
FDM-CEV). If raw solve speed is the primary concern, an adaptive solver is likely
the better choice.

**Parallelism** (`--features parallel`) scales well only at large problem sizes:
eta8=0.908 for 3D Strang at fine resolution; eta8≈0.125 for 2D problems (Amdahl
ceiling for small-grid 2D work).

**Where SemiFlow wins:**

- Novel operator coverage (manifold, hypoelliptic, graph, S³ carriers) with no
  practical alternative.
- Tail-latency-sensitive HFT pricing (niche): `Diffusion4thChernoff` achieves
  **41 ns p99.9** vs QuantLib V3 CEV 9573 ns (**233× advantage**, 95% CI [227×,
  236×]) clean; **284×** [272×, 335×] under DRAM stress (p99.9 RC 53 ns vs QL
  15032 ns). Matched accuracy err=8e-6 < 5e-4 gate; 0 heap allocs in hot loop;
  1M ticks × 5 reps, core-0 pinned, i7-12700K. RC tail degrades only 1.29× clean
  → DRAM-stressed vs QL's 1.57× ("gold under stress"). Source:
  `benchmarks/hft-latency-tail/data/phase-e-summary.md`, `examples/latency_tail.rs`.
  **This is a niche win.** Wallclock H-WALL is FALSIFIED for general PDE solving.
- S³ flagship capabilities: TtChernoff 524288× storage advantage at d=4
  (H-CURSE SUPPORT); ReverseChernoff O(√n) checkpoint memory (slope 0.4956,
  r²=0.999); GridlessChernoff N-independent 13.9/22.5 KB working set at d=1/2.
  S³ low-rank carriers are **L1-resident**: TtChernoff 0.0008% L1d-miss,
  ReverseChernoff 0.0019% (confirmed via perf cpu_core/ counters, b923777) —
  approximately 4 orders of magnitude below a dense 256 KB working set (80.9%
  L1d-miss). This applies to the low-rank carriers only, not dense-grid engines.

Source: `remizov-publications/benchmarks/results/aggregate-iter8/iter8-cross-wave.md`.
