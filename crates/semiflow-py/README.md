# semiflow-py

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![PyPI](https://img.shields.io/pypi/v/semiflow-pde)](https://pypi.org/project/semiflow-pde/)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

PyO3 Python bindings for [`semiflow`](../../crates/semiflow) —
Chernoff approximations of operator semigroups (Remizov 2025, Theorem 6).

**Built on the `semiflow` core crate** (ADR-0154, 2026-06-10). The Python
surface has parity with all core kernel families via ADR-0111 Waves P1–P7
plus the 0.9.0-beta addition of `ReverseHeat1D` (reverse-mode AD, math §51,
ADR-0156): 26 binding classes + 1 free function. Pyright errors: 0. Complete
`__init__.pyi` stubs; `py.typed` marker; GIL released in all `evolve` paths
(ADR-0031).

Tensor-train and gridless carriers (`TtEvolver`, `TtState`, `TtCoupledEvolver`,
`VarCoefTtEvolver`, `GridlessEvolver`, `MeasureState`) are **fully exposed via
PyO3** as of 0.12.0-beta (Rust-only at 0.9.0-beta; bindings added in ADR-0171 /
ADR-0178).

## Installation

```sh
pip install semiflow-pde
```

> **Note**: `semiflow-pde` is published on [PyPI](https://pypi.org/project/semiflow-pde/).
> Wheels for common platforms are also available via
> [GitHub releases](https://github.com/VolkovIlia/semiflow/releases) if you need
> a specific build: `pip install semiflow_pde-*.whl`.

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

| Value | Semantics | Typical use |
|-------|-----------|-------------|
| `"reflect"` (default) | Mirror / zero-flux Neumann at grid boundaries | General PDEs; required by G1/G2 oracle tests |
| `"periodic"` | Periodic wrap with period `(n-1)·dx` | Periodic domains |
| `"zero"` | Extend with 0.0 outside domain | Solutions that vanish at the boundary (barriers, puts far OTM in log-space) |
| `"linear"` | Linear extrapolation from the two outermost nodes | **Asymptotically-linear far-field** (European calls, linear ramps) |

### Far-field / Dirichlet-like boundaries for finance

Users coming from classical finite-difference option pricers often look for an
inhomogeneous Dirichlet far-field ("set `V = S − K e^{−rτ}` at `S_max`") and,
finding only `"zero"`, conclude the library cannot price calls.  In fact
`boundary="linear"` is the correct and recommended closure for European call
pricing.

**Why `"linear"` works for calls:**  The Black-Scholes / CEV solution satisfies
`V(S, τ) ≈ S − K e^{−rτ}` as `S → ∞`.  This asymptote is linear in `S`, so
`V_SS → 0`.  Linear extrapolation from the two outermost grid nodes reproduces
an affine far-field exactly (to machine precision), introducing no spurious
curvature or kink.  Validated on `Shift1D.with_arrays` with `S ∈ [0, 4K]`,
`n = 1025`: ATM relative error ≈ **8.5e-5** with no boundary artifact.

**Puts in log-price space:** When a European put is priced in log-price
coordinates (`z = ln S`), the solution satisfies `u_z → 0` at both ends of a
sufficiently wide domain.  Either `"reflect"` (zero-flux Neumann) or `"linear"`
works; `"reflect"` is marginally preferable because it imposes the correct
zero-derivative condition exactly.

**Example — Black-Scholes call via `Shift1D`:**

```python
import numpy as np
import semiflow as rp

# Black-Scholes PDE: dV/dt = (sigma^2 / 2) S^2 V_SS + r S V_S - r V
# Rewrite as L V = a(S) V_SS + b(S) V_S + c(S) V with:
#   a(S) = 0.5 * sigma^2 * S^2,  b(S) = r * S,  c(S) = -r

K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
S_max = 4.0 * K      # wide enough that far-field is linear
n = 1025
S_nodes = np.linspace(0.0, S_max, n)

a_arr = 0.5 * sigma**2 * S_nodes**2
b_arr = r * S_nodes
c_arr = np.full(n, -r)

# European call payoff at maturity
u0 = np.maximum(S_nodes - K, 0.0)

state = rp.Shift1D.with_arrays(
    0.0, S_max, n,
    a_arr, b_arr, c_arr,
    c_norm_bound=r,  # upper bound on |c(x)|
    u0=u0,
    boundary="linear",   # <-- correct far-field closure for calls
)
state.evolve(t=T, n_steps=200)

V = state.values()
S_atm_idx = np.argmin(np.abs(S_nodes - K))
print(f"ATM call price (semiflow): {V[S_atm_idx]:.6f}")
# rel. error vs Black-Scholes closed form ≈ 8.5e-5 at n=1025, n_steps=200
```

**Inhomogeneous Dirichlet (future work):** The Rust core provides
`BoundaryPolicy::Dirichlet { value }` (a constant ghost-node extension), but
it is not yet wired into the Python `boundary=` string parser.  If you need to
pin `u = g_lo` at the left edge and `u = g_hi` at the right edge with distinct
values, use `"linear"` as the near-exact idiom for asymptotically-linear
payoffs.  Support for `boundary=("dirichlet", value)` is tracked in issue #20.

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
| `Heat2DVarA` | `Strang2D` + variable-a | 1 | **Non-divergence** `a_x(x) u_xx + a_y(y) u_yy`; pass `a_x`, `a_y` arrays. Order 1, not 2: the axis kernels freeze `a` at the node (ADR-0191 AM1) |
| `Heat3DVarA` | `Strang3D` + variable-a | 1 | **Non-divergence** `a_x u_xx + a_y u_yy + a_z u_zz`; same order caveat as `Heat2DVarA` |
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

### Reverse-mode AD (0.9.0-beta, ADR-0156)

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
for `ReverseHeat1D`.

### Tensor-train carriers (ADR-0171 / ADR-0178)

Curse-escaped O(d·n·r²) storage for separable diagonal-A diffusion on ℝᵈ.
State and evolvers share the `TtState` carrier; `TtEvolver` / `TtCoupledEvolver`
cover constant-coefficient cases, `VarCoefTtEvolver` covers variable-coefficient
additive-separable operators (issue #2, ADR-0178).

| Class | Constructor | Key method | Notes |
|-------|-------------|------------|-------|
| `TtState(slices)` | `slices: list[NDArray[np.float64]]` — per-axis 1-D arrays | `.ndim()`, `.n_j(j)`, `.peak_rank()`, `.storage_size()`, `.inner_separable(functionals)` | Shared carrier for all TT evolvers |
| `TtEvolver(a, b, c, dom_min, dom_max, eps_round)` | `a`, `b`: `list[float]` per-axis coeffs; `c: float`; bounds lists; `eps_round: float` | `.evolve(state, t_final, n_steps)` — mutates `TtState` in-place | Diagonal-A Gaussian class (§52) |
| `VarCoefTtEvolver(a_axis, b_axis, v_axis, domain, eps_round)` | `a_axis`, `b_axis`, `v_axis`: `list[list[float]]` per-axis nodal values; `domain: list[tuple[float,float]]`; `eps_round: float` | `.evolve(state, t_final, n_steps)` | Variable-coef additive-separable; rank-1 IC → rank-1 output (ADR-0178) |
| `TtCoupledEvolver(a, b, c, coupling, dom_min, dom_max, eps_round)` | Same as `TtEvolver` + `coupling: tuple` — `("None",)`, `("Tridiagonal", rho)`, or `("Pairs", [(j,k,rho),...])` | `.evolve(state, t_final, n_steps)` | Nearest-neighbour pair-factor coupling (§52.9) |

### Gridless / particle carriers (ADR-0171)

Sparse signed-measure ensemble on ℝ (D=1). Curse-escaped: the 3ᴰ dense tree is
never materialised; only sparse marginals and scalar observables cross the
Python boundary.

| Class | Constructor | Key method | Notes |
|-------|-------------|------------|-------|
| `MeasureState(positions, weights, dim)` | `positions`, `weights`: `NDArray[np.float64]`; `dim: int` (must be 1) | `.n_diracs()`, `.total_variation()`, `.second_moment()`, `.marginal(axis) -> (pos, wgt)` | Particle carrier; signed-weight Dirac ensemble (§50) |
| `GridlessEvolver(a, b, c, *, voronoi_cap=64, gaussian_background=False)` | `a`, `b`, `c`: `float`; optional particle-cap / background kwargs | `.evolve(state, t_final, n_steps)` — mutates `MeasureState`; `.apply(tau, src, dst)` one-step | 1-D 3-branch Chernoff kernel; `WeightedVoronoi` reduction (ADR-0155) |

### v3 Evolver surface

| Class | Notes |
|-------|-------|
| `EvolverHeat1DUnitV3(domain_lo, domain_hi, n_grid, u0, n_chernoff)` | Zero-alloc `apply_into` hot path; `.evolve_into(t, buf)` |
| `GrowthV3` | Growth bound `(multiplier, omega)` returned by `.growth()` |

### Greeks and hyper-dual AD

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `EvolverHeat1DGreeksV3` | `(domain_lo, domain_hi, n_grid, u0, n_chernoff, scale_theta=0.5)` | `.greeks(t) -> (value, delta, gamma)`, `.size()`, `.n_chernoff()` | Hyper-dual `Dual<Dual<f64>>` sweep; returns primal + ∂u/∂θ + ∂²u/∂θ² in one pass (ADR-0133 A3) |

### Adjoint Fokker-Planck (particle pushforward)

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `AdjointFokkerPlanckV8` | `(a, b, c)` — constant diffusion, drift, reaction coefficients | `.step(tau, positions, weights, n_steps=1) -> (positions, weights)`, `.total_variation(...)`, `.second_moment(...)` | Weak-* Fokker-Planck pushforward: each Dirac δ_x spawns 4 children per step (Lemma A.1, §38.3); D=1 constant-coefficient only |

### Killing and absorbing boundary conditions

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `Killing2nd1D` | `(xmin, xmax, n, u0, *, kappa=0.0, boundary="reflect")` | `.evolve(t, n_steps=100)`, `.values()`, `.order()`, `len()` | Order-2 soft-killing `∂_t u = ∂²u − κu`; palindromic Strang split `e^{−τκ/2} C(τ) e^{−τκ/2}` (ADR-0126, §21.8) |
| `KilledDirichlet1D` | `(domain_lo, domain_hi, n_grid, u0, n_chernoff)` | `.apply(t) -> NDArray`, `.size()` | Crank-Nicolson Cayley map; absorbing Dirichlet u\|∂R=0; order 2 (ADR-0135 Amdt 2, §44.ter) |
| `DirichletHeat2nd1D` | `(xmin, xmax, n, u0, *, origin=nan, boundary="reflect")` | `.evolve(t, n_steps=100)`, `.values()`, `.order()`, `len()` | Order-2 Dirichlet via odd-image method; sibling of `Reflected1D` (Neumann); higher-order than `Killing1D` (ADR-0176, §21.9) |

### Obstacle / variational-inequality family

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `ObstacleChernoff` | `(xmin, xmax, n, u0, *, a=1.0, b=0.0, c=0.0, level=nan, obstacle_array=None)` | `.evolve(t, n_steps=100) -> NDArray`, `.values()`, `.evolve_active_set_adjoint(w_fwd, lam, tau)`, `.order()`, `len()` | `V^{n+1} = Π_g(S(Δτ)Vⁿ)`; constant or array obstacle floor; order 1 globally (§44, Theorem 44.1) |
| `ObstacleGammaV8` | `(domain_lo, domain_hi, n_grid, *, level=..., obstacle_array=None)` | `.inactive_gamma(v) -> (gamma, defined, count)`, `.size()` | Inactive-set Γ = V″ primitive; `defined[i]=False` means Γ **refused** (active set / contact); D=1 only (ADR-0153 §4.1) |
| `ObstacleNDV8` | `(xmin, xmax, nx, ymin, ymax, ny, level)` | `.apply(tau, v) -> NDArray`, `.shape()` | D=2 projective-splitting obstacle `Π_g ∘ S(Δτ)`; input shape `(nx, ny)` accepted (raveled order='F' internally); output is a flat nx*ny float64 array — use `out.reshape((nx, ny), order='F')` to recover 2D layout (ADR-0153 §4.2) |

### Resolvent time-jump family (TWS contour)

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `ResolventJumpV8` | `(domain_lo, domain_hi, n_grid, m_nodes)` | `.jump(t, g) -> NDArray`, `.size()`, `.m_nodes()` | Evaluates `e^{tA}g` via TWS parabolic-contour ILT; large-t alternative to many Chernoff steps; `m_nodes >= 6` (ADR-0138, §47) |
| `ResolventJump2DV8` | `(xmin, xmax, nx, ymin, ymax, ny, m_nodes)` | `.jump(t, g) -> NDArray`, `.shape()`, `.m_nodes()` | 2D TWS contour ILT; input/output shape `(nx, ny)` Fortran-order (ADR-0153, §47.8) |
| `ResolventJump3DV8` | `(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, m_nodes)` | `.jump(t, g) -> NDArray`, `.shape()`, `.m_nodes()` | 3D TWS contour ILT; input/output shape `(nx, ny, nz)` Fortran-order (ADR-0153, §47.8) |

### Matrix diffusion (coupled 2-component)

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `MatrixDiffusion2D` | `(xmin, xmax, nx, ymin, ymax, ny, u0, *, a_diag=1.0, c_coupling=0.0)` | `.evolve(t, n_steps=100)`, `.values()`, `.order()` | 2-component coupled 2D diffusion `∂_t u = a∂²u + cu`; flat buffer length `2·nx·ny`; index `(j·nx+i)·2+component` (ADR-0124, §33.2) |
| `MatrixDiffusion3D` | `(xmin, xmax, nx, ymin, ymax, ny, zmin, zmax, nz, u0, *, a_diag=1.0, c_coupling=0.0)` | `.evolve(t, n_steps=100)`, `.values()`, `.order()` | 2-component coupled 3D; flat buffer length `2·nx·ny·nz`; index `(k·nx·ny+j·nx+i)·2+component` (ADR-0124, §33.3) |

### Wentzell dynamic boundary conditions

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `GammaFamily` | Static factories: `.constant(c)`, `.linear(a, b)`, `.exponential(rate)` | — | Ergonomic γ-schedule builder for `WentzellV8`; expands to pre-sampled float64 array (ADR-0153) |
| `WentzellV8` | `(domain_lo, domain_hi, n_grid, u0, n_steps, c_reaction, gamma_schedule)` or `.from_family(...)` | `.evolve(t, t_offset=0.0) -> NDArray`, `.size()`, `.n_steps()` | Dynamic Wentzell BC `∂_t u + γ(t)·∂_ν u + c·u = 0`; bulk-boundary Cayley Lie split; order 1; 1D half-line only (ADR-0151, §49) |

### Additional 1D kernels

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `DiffusionExpmv1D` | `(xmin, xmax, n, u0, *, boundary="reflect")` | `.evolve(t, n_steps=100)`, `.values()`, `.order()`, `len()` | Tolerance-driven expmv kernel (Al-Mohy & Higham 2011); `order()` returns `u32::MAX`; not a fixed convergence order (ADR-0121) |
| `DriftReaction4th1D` | `(xmin, xmax, n, u0, *, boundary="reflect")` | `.evolve(t, n_steps=100)`, `.values()`, `.order()`, `len()` | Order-4 `b(x)∂_x u + c(x)u` via palindromic `R_sym(τ/2) ∘ K5(τ) ∘ R_sym(τ/2)`; defaults `b=0.5, b'=0, c=0` (ADR-0127) |

### Sparse-grid and high-dimensional kernels

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `SmolyakD6V8` | `(domain_lo, domain_hi, n_per_axis)` | `.apply(tau, u0, n_steps=1) -> NDArray`, `.n_nodes()`, `.level()`, `.size()` | D=6 Smolyak sparse-grid unit-diffusion; level ℓ=D+3=9 (533 nodes); input/output flat `n_per_axis^6` (ADR-0138, ADR-0123 Amdt 1) |
| `ComplexTripleJumpV8` | `(domain_lo, domain_hi, n_per_axis)` | `.apply_real(tau, u0) -> NDArray`, `.verify_gamma_star() -> bool`, `.size()` | Order-4 complex triple-jump on filiform-N5 Carnot (D=5); returns real projection `Re(Ψ(τ)f)`; complex substeps are internal (ADR-0138, ADR-0136 Amdt 2) |

### Graph adjoint and Krylov families

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `GraphAdjoint` | `(graph=None, laplacian=None, *, lap_at_t, rho_bar, a=None, kernel="magnus_graph", convergence_check=True)` | `.evolve_state_adjoint(lambda_n, t, n_steps=100) -> NDArray`, `.n_nodes()` | Backward costate sweep `λ_0 = S⋆_1⋯S⋆_n · λ_n` via transpose Magnus K=4 map; supports `"magnus_graph"` / `"varcoef_magnus_graph"` (ADR-0115, §42) |
| `GraphAdjointPresampled` | `.from_presampled(graph, lap_at_t, rho_bar, n_steps, t_horizon, a=None, kernel="magnus_graph", convergence_check=True)` | `.evolve_state_adjoint(lambda_n, n_steps=None) -> NDArray`, `.evolve_state_adjoint_batched(lambda_cols, n_steps=None) -> NDArray`, `.n_nodes()`, `.n_steps()` | Pre-samples callbacks once at construction; adjoint sweep runs fully in `py.detach` with no per-step GIL re-entry (ADR-0180) |
| `GraphKrylov` | `(laplacian, *, path="chebyshev", tol=1e-10, m_max=18)` | `.evolve_batched(t, features_nc) -> NDArray`, `.n_nodes()` | Depth-independent `e^{-tL_G}·V` via Krylov; accepts `[N, C]` feature matrix; single GIL release (ADR-0185, §54) |

### Symmetric operator and FEM utilities

| Class | Constructor | Key methods | Notes |
|-------|-------------|-------------|-------|
| `SymmetricOperator` | `.from_csr(indptr, indices, data, n, sym_tol=1e-10)` | `.evolve_batched(t, v_nc, path="chebyshev", tol=1e-10, m_max=18, n_steps=100) -> NDArray`, `.n()`, `.lambda_max_bound()` | Externally-assembled symmetric PSD sparse operator from CSR arrays; feeds Krylov expmv and Fréchet VJP (`symmetric_op_expmv_frechet`) (§55) |
| `ConservativeDiffusionChernoff` | `.from_k_array(n, x_lo, x_hi, k_nodes, r_contact=None, boundary="neumann")` | `.to_symmetric_operator() -> SymmetricOperator`, `.n()`, `.dx()` | Order-2 FV divergence-form `∂_x(k(x)∂_x u)` with harmonic-mean face conductivities; bridge to `SymmetricOperator` Krylov path (§56) |
| `GeneralOperator` | `.from_csr(n, indptr, indices, data)` | `.evolve_batched(t, v_nc, n_steps) -> NDArray`, `.apply_transpose(v) -> NDArray`, `.n()`, `.norm_inf_bound()` | Externally-assembled **possibly non-symmetric** CSR operator; `e^{-tA}v` via the symmetry-agnostic Al-Mohy–Higham Taylor `expmv`. Drifted Fokker–Planck and inventory-ladder generators, which `SymmetricOperator` rejects. Cost is `Θ(t‖A‖_∞)` matvecs — linear in the horizon, NOT depth-flat like Lanczos (§57, ADR-0195) |
| `MassKOperator` | `.from_k_and_mass(k_op, m_dense)` | `.evolve(t, v, path="chebyshev", tol=1e-10, m_max=18, n_steps=100) -> NDArray`, `.n()` | Consistent-mass operator `Â = R⁻ᵀ K R⁻¹` where `M = RᵀR`; applies `e^{-t M⁻¹ K}` via Krylov (§55.4) |
| `Etdrk4` | `.from_symmetric_op(op, nonlinearity="allen_cahn", h=0.01)` | `.step(u) -> NDArray`, `.integrate(u0, n_steps) -> NDArray` | Cox-Matthews ETDRK4 for `u' = -Au + N(u)`; `"allen_cahn"` nonlinearity `N(u) = u − u³`; arbitrary Python callbacks NOT supported (ADR-0189, §58.3) |

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

## Changelog / Release notes

- [CHANGELOG.md](https://github.com/VolkovIlia/semiflow/blob/master/CHANGELOG.md) — full version history
- [GitHub Releases](https://github.com/VolkovIlia/semiflow/releases) — tagged release notes and wheel downloads

## License

MIT OR Apache-2.0 — same as `semiflow`.
