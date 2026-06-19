# semiflow-py

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![PyPI badge — pending
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

PyO3 Python bindings for [`semiflow-core`](../../crates/semiflow-core) —
Chernoff approximations of operator semigroups (Remizov 2025, Theorem 6).

**Aligned with `semiflow-core` v9.0.0** (ADR-0154, 2026-06-10). The Python
surface has parity with all core kernel families via ADR-0111 Waves P1–P7
plus the v9.0.0 addition of `ReverseHeat1D` (reverse-mode AD, math §51,
ADR-0156): 26 binding classes + 1 free function. Pyright errors: 0. Complete
`__init__.pyi` stubs; `py.typed` marker; GIL released in all `evolve` paths
(ADR-0031).

`TtChernoff` / `TtState` and `GridlessChernoff` / `ParticleReduction` are
**Rust-only at v9.0.0** — not exposed via PyO3 (binding design deferred).

## Installation

```sh
pip install semiflow-pde
```

> **Note**: as of v6.0 the package is not yet published to PyPI. Wheels are
> distributed via [GitHub releases](https://github.com/VolkovIlia/semiflow/releases).
> Download the `semiflow-*.whl` for your platform and install with
> `pip install semiflow-*.whl`.

Or build from source (requires Rust toolchain + maturin):

```sh
pip install maturin
maturin develop --profile release-ffi -m crates/semiflow-py/Cargo.toml
```

## Array I/O conventions

- All real-valued state arrays are `numpy.float64` (`np.float64`).
- Schrödinger and `SchrodingerComplex1D` state arrays are `numpy.complex128`.
- 2D state is flat `float64` in row-major x-fastest order: index `j*nx + i`
  corresponds to `u(x_i, y_j)`.
- 3D state is flat x-fastest: index `k*nx*ny + j*nx + i`.
- `values()` always returns a **copy** of the internal Rust state; mutations
  to the returned array do not affect the object.
- Inputs are validated for `NaN`/`Inf` at construction and before `evolve`;
  non-finite inputs raise `SemiflowError(kind='NanInf')`.
- All finite-check and grid-size errors raise `SemiflowError`.

## Error model

All semiflow-py operations raise a single exception type:

```python
from semiflow import SemiflowError
```

The `.kind` attribute (a string) identifies the error category:

| `kind` | When raised |
|--------|-------------|
| `GridMismatch` | Invalid geometry, mismatched array lengths |
| `NanInf` | Input array contains NaN or Inf |
| `OutOfDomain` | Parameter out of valid range (e.g. `t < 0`, `n < 4`) |
| `BoundaryFailure` | Unrecognised boundary policy string |
| `CflViolated` | CFL-like stability constraint exceeded |
| `ConvergenceFailed` | Magnus / adaptive integration convergence check failed |
| `Unsupported` | Unrecognised string selector (e.g. `subordinator=`) |
| `Panic` | Unrecoverable internal Rust panic (should never occur) |

## Boundary policies

All 1D/2D/3D kernels accept a keyword argument `boundary`:

| Value | Semantics |
|-------|-----------|
| `"reflect"` (default) | Mirror / zero-flux Neumann at grid boundaries |
| `"periodic"` | Periodic wrap |
| `"zero"` | Dirichlet zero at grid boundaries |
| `"linear"` | Linear extrapolation |

---

## Usage examples

### 1. Unit-diffusion 1D heat

Solve `∂_t u = ∂²_x u` on `[-10, 10]` with a Gaussian initial condition:

```python
import numpy as np
import semiflow as rp

n = 1000
xs = np.linspace(-10.0, 10.0, n)
u0 = np.exp(-(xs - 0.5)**2 / 0.01)   # narrow Gaussian at x=0.5

state = rp.Heat1D(-10.0, 10.0, n, u0)
state.evolve(t=1.0, n_steps=100)

u = state.values()    # float64 ndarray, shape (n,)
print(f"n={len(state)}, max={u.max():.6f}")
```

The GIL is released during `evolve` (ADR-0031); concurrent Python threads
make progress during long calls.

### 2. `SchrodingerComplex1D` — native complex128 wavefunction

Solve `i ψ_t = (−½∂²_x + V) ψ` and verify unitarity:

```python
import numpy as np
import semiflow as rp

n = 512
xs = np.linspace(-10.0, 10.0, n)
psi0 = np.exp(-xs**2 / 2.0).astype(np.complex128)  # normalised Gaussian
psi0 /= np.sqrt(np.trapz(np.abs(psi0)**2, xs))      # L2-normalise

sch = rp.SchrodingerComplex1D(-10.0, 10.0, n, psi0)
norm0 = sch.norm_squared()

sch.evolve(t=0.5, n_steps=200)

psi_t = sch.values()    # complex128 ndarray
assert abs(sch.norm_squared() / norm0 - 1.0) < 1e-12, "unitarity violated"
print(f"norm ratio = {sch.norm_squared() / norm0:.15f}")
```

### 3. `Manifold2D` — Riemannian manifold heat kernel

Solve `∂_t u = Δ_{S²} u` on the 2-sphere via MMRS 2023 Chernoff formula:

```python
import numpy as np
import semiflow as rp

nx, ny = 32, 64
u0 = np.zeros(nx * ny, dtype=np.float64)
u0[nx * (ny // 2) + nx // 2] = 1.0  # delta-like at chart centre

sphere = rp.Manifold2D(
    0.1, np.pi - 0.1, nx,    # theta axis
    0.0, 2 * np.pi,   ny,    # phi axis
    u0,
    manifold="sphere2",
    radius=1.0,
    curvature_correction=True,  # enables R/12 correction -> order 2
)
sphere.evolve(t=0.02, n_steps=50)

u_t = sphere.values()    # float64 ndarray, length nx*ny (row-major theta-fastest)
print(f"integral ≈ {u_t.sum() * (np.pi / nx) * (2 * np.pi / ny):.4f}")
```

Available manifolds: `"torus"` (flat T²), `"sphere2"` (S²(r)), `"hyperbolic2"`
(Poincaré disk H²(s)). The `radius` parameter sets r or s.

---

## Class reference

Classes are grouped by kernel family. All stateful classes expose at least
`evolve(t, n_steps=100)` (mutates in-place, GIL released) and `values()` →
`NDArray[np.float64]` (copy). See `__init__.pyi` for complete signatures.

### 1D diffusion family

| Class | Kernel | Order | Notes |
|-------|--------|-------|-------|
| `Heat1D` | `DiffusionChernoff` | 2 | Unit or variable-`a`; `.with_a_array` / `.with_a_function` factories |
| `Heat1D4th` | `Diffusion4thChernoff` | 4 | 4th-order temporal; `.with_a_array` |
| `Heat1D6th` | `Diffusion6thChernoff` | 6 | 6th-order temporal; `.with_a_array` |
| `Heat1DZeta4` | `Diffusion4thZeta4Chernoff` | 4 | ζ⁴ kernel; `.with_quintic_sampling()` opt-in |
| `Heat1DZeta6` | `Diffusion6thZeta6Chernoff` | 6 | ζ⁶ kernel; Quintic spatial unconditional |
| `Heat1DZeta8` | `Diffusion8thZeta8Chernoff` | 8 | ζ⁸ kernel; Chebyshev sampling default |
| `TruncatedExp1D` | `TruncatedExpChernoff` | 2 | CFL-conditional truncated-exp |
| `TruncatedExp4th1D` | `TruncatedExp4thChernoff` | 4 | 4th-order truncated-exp |
| `DriftReaction1D` | `DriftReactionChernoff` | 2 | `b(x) ∂_x u + c(x) u`; `.with_arrays` |
| `Shift1D` | `ShiftChernoff1D` | 1 | Universal `a ∂² + b ∂ + c`; `.with_arrays` |
| `Strang1D` | `StrangSplit` (diffusion + drift) | 2 | Advection-diffusion `∂²u + b ∂u`; default `b=0.5` |

### Operator splitting — multi-dimensional

| Class | Kernel | Order | Notes |
|-------|--------|-------|-------|
| `Heat2D` | `Strang2D` | 2 | Unit diffusion on 2D grid; flat x-fastest output |
| `Heat3D` | `Strang3D` | 2 | Unit diffusion on 3D grid; flat x-fastest output |
| `Heat2DVarA` | `Strang2D` + variable-a | 2 | `a_x(x) u_xx + a_y(y) u_yy`; pass `a_x`, `a_y` arrays |
| `Heat3DVarA` | `Strang3D` + variable-a | 2 | `a_x u_xx + a_y u_yy + a_z u_zz`; pass `a_x`, `a_y`, `a_z` arrays |
| `NonSeparable2D` | 5-leg palindromic | 2 | `∂²_x + ∂²_y + c·∂_x ∂_y`; scalar or `.with_beta_array` |
| `NonSeparable2DAniso` | 5-leg + position-dep. β | 2 | `∂²_x + ∂²_y + β(x,y)·∂_x ∂_y`; requires `beta_values` array |

### Schrödinger

| Class | Kernel | Notes |
|-------|--------|-------|
| `Schrodinger1D` | `SchrodingerChernoff<f64>` | Real-pair split; `values()` → `complex128` |
| `SchrodingerComplex1D` | `SchrödingerChernoffComplex` | Native `complex128` state; exact unitary (ADR-0079 Option B) |

Both support `.with_potential(v_array)` and `.norm_squared()`.

### Boundary-condition kernels

| Class | Kernel | Order | Physics |
|-------|--------|-------|---------|
| `Resolvent1D` | `LaplaceChernoffResolvent` | — | `(λI − ∂²)⁻¹ g` via GL-32 quadrature; `.eval(lambda_, g)` + `.residual(lambda_, g)` |
| `Killing1D` | `KillingChernoff` | 1 | Absorbing (Dirichlet) BC via Feynman-Kac; `lo`/`hi` kwargs |
| `Reflected1D` | `ReflectedHeatChernoff` | 2 | Neumann (reflecting) BC via Walsh 1986 image method; `origin` kwarg |
| `Robin1D` | `RobinHeatChernoff` | 1 | Robin BC `α u − β ∂_n u = 0`; `alpha`, `beta`, `origin` kwargs |

### Time-dependent and subordinated

| Class | Kernel | Notes |
|-------|--------|-------|
| `Howland1D` | `HowlandLift<DiffusionChernoff>` | Nonautonomous lift (Howland 1974); `n_t`, `t_horizon` kwargs; `.evolve()` takes no args |
| `Subordinated1D` | `SubordinatedChernoff` | Bochner-Phillips subordination (Butko 2018); backends: `"stable"`, `"gamma"`, `"inverse_gaussian"` |

### Geometry and hypoelliptic operators

| Class | Manifold / Group | Notes |
|-------|-----------------|-------|
| `Manifold2D` | Torus / S²(r) / H²(s) | MMRS 2023 formula with optional R/12 correction; `manifold=`, `radius=`, `curvature_correction=` kwargs |
| `HypoellipticChernoffKolmogorov` | Kolmogorov phase space | `∂_t p = v ∂_x p + ½ ∂²_v p`; 2D state `nx×nv` |
| `HypoellipticChernoffEngel` | Engel step-3 Carnot (ℝ⁴) | `n**4` flat state; `n` per-axis |
| `HypoellipticChernoffHeisenberg` | Heisenberg H₁ | `.kernel(h, x, y, tc)` point evaluator; `heisenberg_heat_kernel(h, x, y, tc)` free function |

### Graph PDE

| Class / Function | Role |
|-----------------|------|
| `Graph.path(n)` / `.cycle(n)` / `.from_edges(n, edges)` / `.erdos_renyi(n, p, seed)` | Graph topology builders |
| `GraphPath(n)` | Legacy path builder (use `Graph.path(n)`) |
| `Laplacian.combinatorial(graph)` / `.normalized(graph)` | Laplacian assembly |
| `GraphHeat(graph=..., laplacian=..., rho_bar=...)` | Order-1 static graph heat |
| `GraphHeat4th(graph=..., laplacian=..., rho_bar=...)` | Order-4 static |
| `GraphHeat6(graph=..., laplacian=..., rho_bar=...)` | Order-6 static |
| `MagnusGraphHeat(graph, lap_at_t, rho_bar)` | Magnus K=4 time-varying |
| `MagnusGraphHeat6(graph=..., laplacian=..., lap_at_t=..., rho_bar_max=...)` | Magnus K=6 |
| `VarCoefGraphHeat(graph, a, rho_bar)` | Variable node-conductivity |
| `VarCoefMagnusGraph(n_nodes, lap_at_t=..., a_at_t=..., rho_bar_max=..., a_sup_max=...)` | Variable-coef Magnus K=4 |
| `QuantumGraph.path(n_edges)` / `.star(n_arms)` / `.from_edges(edges)` | Metric graph (edge lengths) |
| `QuantumGraphHeat(qgraph)` | Kirchhoff-vertex heat Chernoff |
| `GraphTraj(graph, t_horizon)` | Fixed-topology graph trajectory |
| `StrangGraph.from_path(graph)` / `.from_cycle(graph)` | Palindromic Strang split on graph |

### Matrix and point-eval kernels

| Class / Function | Role |
|-----------------|------|
| `MatrixDiffusion1D(xmin, xmax, n, u0, *, a_diag, c_coupling)` | Coupled 2-component 1D diffusion; flat state length `2*n` |
| `PointEval(xmin, xmax, n)` | Pointwise evaluation via Backend A; `.eval_at(tau, u0, x, n_steps)` |
| `sample_gridfn2d(values, x0min, x0max, nx, x1min, x1max, ny, cx, cy)` | Bilinear interpolation at chart point |

### Anisotropic multi-D

| Class | Notes |
|-------|-------|
| `AnisotropicShiftND2(nx, ny, xmin, xmax, ymin, ymax, a_values, *, b_values, c_values)` | 2D anisotropic shift; order 1 (ADR-0112); `a_values` is flat `2×2×nx×ny` SPD tensor |
| `AnisotropicShiftND3(nx, ny, nz, xmin, xmax, ymin, ymax, zmin, zmax, a_values, *, b_values, c_values)` | 3D variant |

### Adjoint and adaptive wrappers

| Class | Notes |
|-------|-------|
| `Adjoint(xmin, xmax, n, u0, *, kernel="heat2", self_adjoint=False, boundary="reflect")` | Adjoint semigroup; `kernel` in `"heat2"`, `"heat4"`, `"heat6"`, `"drift"`, `"shift"` |
| `AdaptivePI(xmin, xmax, n, u0, *, kernel="heat2", tol_abs=1e-6, tol_rel=1e-4, boundary="reflect")` | PI-controller adaptive step |

### Reverse-mode AD (v9.0.0, ADR-0156)

| Class | Notes |
|-------|-------|
| `ReverseHeat1D(theta, xmin, xmax, n_grid, n_steps)` | Reverse-mode AD for constant-a 1D heat (narrow scope: constant-a `DiffusionChernoff` only, §51.5); `.value_and_grad(tau, u0, target) -> (float, float)` |

**Constructor parameters:**

| Parameter | Type | Constraint |
|-----------|------|------------|
| `theta` | `float` | Diffusivity θ > 0, finite |
| `xmin` | `float` | Left domain boundary |
| `xmax` | `float` | Right domain boundary (xmax > xmin) |
| `n_grid` | `int` | Grid nodes (>= 4) |
| `n_steps` | `int` | Chernoff steps per `.value_and_grad` call (>= 1) |

**`.value_and_grad(tau, u0, target) -> (float, float)`:**

| Parameter | Type | Notes |
|-----------|------|-------|
| `tau` | `float` | Per-step time increment (> 0, finite) |
| `u0` | `numpy.ndarray[float64]` | Initial condition, length `n_grid` |
| `target` | `numpy.ndarray[float64]` | Target state, length `n_grid` |
| returns `value` | `float` | L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²` |
| returns `grad` | `float` | `∂J/∂θ` (K=1 forward-mode Dual; 0-ULP vs core, §51.4) |

```python
import numpy as np
import semiflow as rp

n_grid = 24
xs = np.linspace(-4.0, 4.0, n_grid)

rc = rp.ReverseHeat1D(theta=0.4, xmin=-4.0, xmax=4.0, n_grid=n_grid, n_steps=8)
u0     = np.exp(-xs**2)
target = np.zeros(n_grid)

value, grad = rc.value_and_grad(tau=0.05, u0=u0, target=target)
print(f"loss={value:.6e}  ∂J/∂θ={grad:.6e}")
```

Raises `SemiflowError` with `.kind` in `{'OutOfDomain', 'GridMismatch', 'NanInf'}`.

**NARROW scope (§51.5):** constant-a `DiffusionChernoff` only; θ is the
uniform diffusivity. Variable-coefficient and nonlinear kernels are out of scope
at v9.0.0. `TtChernoff` and `GridlessChernoff` are **not** exposed in PyO3
(Rust-only at v9.0.0).

### v3 Evolver surface

| Class | Notes |
|-------|-------|
| `EvolverHeat1DUnitV3(domain_lo, domain_hi, n_grid, u0, n_chernoff)` | Zero-alloc `apply_into` hot path; `.evolve_into(t, buf)` |
| `GrowthV3` | Growth bound `(multiplier, omega)` returned by `.growth()` |

---

## Performance

GIL release follows the three-phase `py.detach` pattern (ADR-0031):
acquire → snapshot inputs → detach → Rust compute → reacquire. `Send + Sync`
is verified at compile time with `static_assertions`.

Indicative timings on i7-12700K (1000 nodes, 100 steps, `Heat1D`):

| Metric | Value |
|--------|-------|
| Throughput (criterion) | ~56.6 ms per call |
| p99.9 latency (HFT loop, N=1536) | 45 ns/tick |
| Memory footprint | 2.8 MB RSS |

For large grids or many time steps, prefer `.with_a_array` over
`.with_a_function`: the array path uses a pure-Rust `Arc<Vec<f64>>`
Catmull-Rom interpolant and never re-acquires the GIL during `evolve`.

## Type stubs

`__init__.pyi` and the `py.typed` marker ship with every wheel. Static
type checkers (mypy, pyright, pylance) pick them up automatically.
The `pyrightconfig.json` at the repo root adds `crates/semiflow-py/python`
to `extraPaths` so local development also resolves the stubs correctly
(0 `reportAttributeAccessIssue` errors).

## Mathematical reference

I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135.
DOI [10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q)

## License

MIT OR Apache-2.0 — same as `semiflow-core`.
