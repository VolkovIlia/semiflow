---
version: 1.9.0
last_updated: 2026-09-30
freshness_score: 1.0
dependencies:
  - crates/semiflow/src/lib.rs
  - crates/semiflow-py/src/
  - crates/semiflow-ffi/src/
  - crates/semiflow-wasm/src/
  - docs/adr/0028-ffi-pyo3-wasm-v0_10.md
  - docs/adr/0035-v1_0_0-api-stability.md
  - docs/adr/0059-graph-bindings.md
  - docs/adr/0061-python-parity-expansion.md
  - docs/adr/0154-v9-third-scurve-gridless-umbrella.md
  - docs/adr/0156-reverse-mode-ad-chernoff-layer.md
  - docs/adr/0159-tensor-train-chernoff.md
  - docs/adr/0162-band-split-tt-coupling-resolved.md
  - docs/adr/0169-s3-honest-scope-public-api-promotion.md
  - docs/adr/0186-symmetric-operator.md
  - docs/adr/0191-implicit-stiff-symmetric-operator.md
  - docs/adr/0191-nd-sampler-interpolation-order-and-boundary.md
  - docs/adr/0193-adaptive-and-schedules-over-variable-coefficients.md
  - docs/adr/0194-batched-multichannel-grid-evolve.md
  - docs/adr/0195-general-nonsymmetric-operator-action.md
  - docs/adr/0196-per-pencil-strang-composition.md
  - docs/adr/0197-shift1d-coefficient-field-gradients.md
changelog:
  - 1.0.0: Initial coverage matrix for v2.3.0 Python parity expansion (feat/python-parity-v2.3)
  - 1.1.0: v6.2.2 ADR-0115 additions — GraphAdjoint, edge_weight_grad, dtype kwarg, Laplacian accessors, from_edges fix
  - 1.2.0: v9.0.0 — ReverseHeat1D added to PyO3 + WASM; TtChernoff/TtState/GridlessChernoff/ParticleReduction Rust-only
  - 1.3.0: v9.1.0 — CoupledTtChernoff Rust-only (TT contraction interface design deferred)
  - 1.4.0: v9.2.0 — six S3* types (s3-poc feature) Rust-only; no new binding exposure
  - 1.5.0: v0.11.0-beta — add section 9 (SymmetricOperator / MassKOperator / mass_lumped_evolve; issue #15, ADR-0186)
  - 1.6.0: issue #16 branch — document path="implicit" kwarg on evolve_batched / mass_lumped_evolve / MassKOperator.evolve (ADR-0190)
  - 1.7.0: 0.13.0-beta issue campaign #17/#19/#21-#26 — GeneralOperator, shift1d_coeff_grad,
    Shift1D.evolve_batched / evolve_with_coefficient_schedule, AdaptivePI.with_arrays,
    Heat2DVarA.with_grid_arrays (pencil backend), ND boundary= kwarg
  - 1.8.0: 0.13.1-beta de-stale — FFI and WASM columns re-derived from include/semiflow.h and
    #[wasm_bindgen] exports; TT / gridless / GraphTraj / StrangSplitGraph rows corrected
    (all bound); internal-scheme versions replaced by public 0.x releases
  - 1.9.0: 0.14.0-beta — ReverseHeat1D importable from `semiflow` (ImportError note removed);
    no binding surface change otherwise
graph-unverified: false
---

# Python Coverage Matrix

This document tracks binding parity across the four public surfaces of the
`semiflow` workspace: the Rust crate `semiflow`, the C ABI `semiflow-ffi`, the
Python wheel `semiflow-pde` (import name `semiflow`, crate `semiflow-py`) and
the npm package `@semiflow/wasm` (crate `semiflow-wasm`). It reflects
**0.14.0-beta**. All four crates share one workspace version (ADR-0035).

The authoritative lists are generated artefacts, not this page: the C header
[`crates/semiflow-ffi/include/semiflow.h`](../crates/semiflow-ffi/include/semiflow.h),
the Python stubs `crates/semiflow-py/python/semiflow/__init__.pyi`, and the
TypeScript declarations shipped in the npm package. The tables below map Rust
types to their binding names.

**Legend**

| Symbol | Meaning |
|--------|---------|
| ✅ stable | Exposed and covered by an acceptance gate |
| 🚧 experimental | Exposed; API may change in a MINOR release |
| ❌ not exposed | Implemented in core; not surfaced in this binding |
| *(full)* | WASM only: needs a `--features full` build; not in the npm (lite) package |

All cells are for `f64` unless noted. `f32` is opt-in for PyO3 on four
kernels (`GraphHeat`, `MagnusGraphHeat`, `VarCoefGraphHeat`, `Heat1D`) via the
`dtype="f32"` kwarg (ADR-0115). FFI and WASM f32 paths remain out of
scope (ADR-0115).

---

## 1. 1D Kernels and carriers

| Rust type | FFI (`smf_…` prefix) | PyO3 class | WASM class |
|-----------|----------------------|------------|------------|
| `DiffusionChernoff` | ✅ `smf_state_new_heat_1d_unit`, `smf_state_new_with_closure` (var-a callback) | ✅ `Heat1D` (var-a via `with_a_array` / `with_a_function`) | ✅ `Heat1D` |
| `Diffusion4thChernoff` | ✅ `smf_heat1d_4th_*` | ✅ `Heat1D4th` (`with_a_array`) | ✅ `Heat1D4th` *(full)* |
| `Diffusion6thChernoff` | ✅ `smf_heat1d_6th_*` | ✅ `Heat1D6th` (`with_a_array`) | ✅ `Heat1D6th` *(full)* |
| `Diffusion4thZeta4/6thZeta6/8thZeta8Chernoff` | ✅ `smf_heat1d_zeta{4,6,8}_*` | ✅ `Heat1DZeta4/6/8` | ✅ `Heat1DZeta4/6/8` *(full)* |
| `DriftReactionChernoff` | ✅ `smf_drift_reaction_*` | ✅ `DriftReaction1D` (`with_arrays`) | ✅ `DriftReaction1D` *(full)* |
| `DriftReactionZeta4Chernoff` | ✅ `smf_drift_reaction_zeta4_*` | ✅ `DriftReaction4th1D` | ✅ `DriftReaction4th1D` *(full)* |
| `ShiftChernoff1D` | ✅ `smf_shift1d_*` | ✅ `Shift1D` (`with_arrays`) | ✅ `Shift1D` *(full)* |
| `TruncatedExpDiffusionChernoff` | ✅ `smf_trunc_exp_*` | ✅ `TruncatedExp1D` | ✅ `TruncatedExp1D` *(full)* |
| `TruncatedExp4thDiffusionChernoff` | ✅ `smf_trunc_exp4_*` | ✅ `TruncatedExp4th1D` | ✅ `TruncatedExp4th1D` *(full)* |
| `DiffusionExpmvChernoff` | ✅ `smf_expmv1d_*` | ✅ `DiffusionExpmv1D` | ✅ `DiffusionExpmv1D` *(full)* |
| `ReverseChernoff<F>` + `CheckpointSchedule` | ❌ not exposed | ✅ `ReverseHeat1D` (constant-a narrow scope, ADR-0156) | ✅ `ReverseHeat1D` |
| `TtChernoff<F>` + `TtState<F>` | ✅ `smf_tt_*`, `smf_ttstate_*` | ✅ `TtEvolver`, `TtState` | ✅ `TtEvolver`, `TtState` |
| `CoupledTtChernoff<F>` (ADR-0162) | ✅ `smf_tt_coupled_*` | ✅ `TtCoupledEvolver` | ✅ `TtCoupledEvolver` |
| `VarCoefTt<F>` (ADR-0178) | ✅ `smf_varcoef_tt_*` | ✅ `VarCoefTtEvolver` | ✅ `VarCoefTtEvolver` |
| `GridlessChernoff<F, D>` + `ParticleReduction` | ✅ `smf_gridless_*`, `smf_measurestate_*` | ✅ `GridlessEvolver`, `MeasureState` | ✅ `GridlessEvolver`, `MeasureState` (D=1) |
| `S3DriftSpectralEvolver`, `S3DenseCouplingEvolver`, `S3VarCoefEvolver`, `S3NonSepVarCoefEvolver`, `S3BurgersColeHopf`, `S3ReactionDiffusion` (`s3-poc` feature) | ❌ not exposed | ❌ not exposed | ❌ not exposed |

**Notes**

- `Heat1D` in PyO3 exposes the `boundary` kwarg (`'reflect'` / `'periodic'` /
  `'zero'` / `'linear'`; default `'reflect'`).
- Pre-sampled coefficient path (`with_a_array`) performs cubic-Hermite
  interpolation inside Rust, achieving zero GIL re-acquires during `evolve`.
  See ADR-0061 §"Pre-sampled coefficients".
- The `s3-poc` evolvers are a research-track surface behind a non-default
  feature (ADR-0169); binding design is deferred.

---

## 2. 2D / 3D Composition

| Rust type | FFI | PyO3 | WASM |
|-----------|-----|------|------|
| `Strang2D` | ✅ `smf_heat2d_*` | ✅ `Heat2D` (`boundary` kwarg) | ✅ `Heat2D` *(full)* |
| `Strang3D` | ✅ `smf_heat3d_*` | ✅ `Heat3D` (`boundary` kwarg) | ✅ `Heat3D` *(full)* |
| `Strang2D`/`Strang3D` with variable `a` | ✅ `smf_heat2d_vara_*`, `smf_heat3d_vara_*` | ✅ `Heat2DVarA`, `Heat3DVarA` | ✅ `Heat2DVarA`, `Heat3DVarA` *(full)* |
| `NonSeparableMixedChernoff` / `NonSeparable2DChernoff` | ✅ `smf_nonsep2d_*` | ✅ `NonSeparable2D` (`with_beta_array`) | ✅ `NonSeparable2D` *(full)* |
| `NonSeparable2DAnisotropicChernoff` | ✅ `smf_nonsep2d_aniso_*` | ✅ `NonSeparable2DAniso` | ✅ `NonSeparable2DAniso` *(full)* |
| `AnisotropicShiftChernoffND<F, 2/3>` | ✅ `smf_aniso_nd2_*`, `smf_aniso_nd3_*` | ✅ `AnisotropicShiftND2/3` | ✅ `AnisotropicShiftND2/3` *(full)* |
| `MatrixDiffusionChernoff` (1D/2D/3D) | ✅ `smf_matrix_diffusion_*`, `smf_matrix2d_*`, `smf_matrix3d_*` | ✅ `MatrixDiffusion1D/2D/3D` | ✅ `MatrixDiffusion1D/2D/3D` *(full)* |

**Notes**

- `Heat2D` and `Heat3D` take the `boundary` kwarg; internally wired through
  `Grid1D::new(...)?.with_boundary(...)` on each axis (there is no
  `Grid1D::new_with_policy`; boundary is a builder step).
- `NonSeparable2D` wraps the unified `NonSeparableMixedChernoff` type
  (ADR-0058). The constant-`c` path and the `with_beta_array` pre-sampled
  β(x,y) path via bilinear interpolation are both exposed. See `coeff2d.rs`.

---

## 3. Adjoint / Schrödinger / Adaptive Wrappers

| Rust type | FFI | PyO3 | WASM |
|-----------|-----|------|------|
| `AdjointChernoff` | ✅ `smf_adjoint1d_*` | ✅ `Adjoint` (enum dispatch over inner kernels) | ✅ `Adjoint1D` *(full)* |
| `SchrodingerChernoff` + `SchrodingerState` | ✅ `smf_schrodinger_*` | ✅ `Schrodinger1D` | ✅ `Schrodinger1D` *(full)* |
| `SchrödingerChernoffComplex` | ✅ `smf_schrodinger_cx_*` | ✅ `SchrodingerComplex1D` | ✅ `SchrodingerComplex1D` *(full)* |
| `AdaptivePI` | ✅ `smf_adaptive_pi_*` | ✅ `AdaptivePI` (enum dispatch; `with_arrays`) | ✅ `AdaptivePI1D` *(full)* |

**Adjoint dispatch**: the Python `Adjoint` takes a `kernel=` string selecting the
inner kernel (`"heat2"` default, `"heat4"`, `"heat6"`, …; see the stub).
Adding a new inner kernel requires extending the enum in
`crates/semiflow-py/src/adjoint.rs`; this is a known rigidity trade-off
documented in ADR-0061 §"Consequences".

**AdaptivePI**: return value is a dict
`{final_state, steps_accepted, steps_rejected, last_tau}`.

**Schrödinger**: 4 constructors — default-V, `from_parts`
(psi\_re / psi\_im), `with_potential` (pre-sampled V array), and
`with_potential_parts`. Methods: `evolve(t, n_steps=200)`, `values()` →
complex128 ndarray, `values_parts()` → (float64, float64) ndarrays,
`norm_squared()`, `__len__()`. Unitarity gate: `‖ψ‖²/‖ψ₀‖² − 1 < 1e-6` over
500 steps on the harmonic oscillator.

---

## 4. Graph PDE

| Rust type | FFI | PyO3 | WASM |
|-----------|-----|------|------|
| `Graph` | ✅ opaque `SmfGraph` (`smf_graph_path`, …) | ✅ `Graph` (and `GraphPath`) | ✅ `GraphPath` (path graphs only) |
| `Laplacian` | ✅ opaque `SmfLaplacian` (`smf_graph_laplacian_*`, `smf_laplacian_*`) | ✅ `Laplacian` | ✅ `Laplacian` *(full)* |
| `GraphHeatChernoff` | ✅ `smf_ghc_*` | ✅ `GraphHeat` | ✅ `GraphHeat` |
| `GraphHeat4thChernoff` | ✅ `smf_ghc4_*` | ✅ `GraphHeat4th` | ✅ `GraphHeat4th` *(full)* |
| `GraphHeat6thChernoff` | ✅ `smf_ghc6_*` | ✅ `GraphHeat6` | ✅ `GraphHeat6` |
| `MagnusGraphHeatChernoff` | ✅ `smf_mghc_*` | ✅ `MagnusGraphHeat` | ✅ `MagnusGraphHeat` *(full)* |
| `MagnusGraphHeat6thChernoff` | ✅ `smf_mghc6_*` | ✅ `MagnusGraphHeat6` | ✅ `MagnusGraphHeat6` *(full)* |
| `VarCoefGraphHeatChernoff` | ✅ `smf_vc_ghc_*` | ✅ `VarCoefGraphHeat(graph, a, rho_bar=…)` | ✅ `VarCoefGraphHeat` *(full)* |
| `VarCoefMagnusGraphHeatChernoff` | ✅ `smf_vc_mghc_*` | ✅ `VarCoefMagnusGraph` | ✅ `VarCoefMagnusGraph` *(full)* |
| `GraphTraj` | ✅ `smf_graph_traj_*` | ✅ `GraphTraj` | ✅ `GraphTraj` *(full)* |
| `StrangSplitGraph` | ✅ `smf_strang_graph_*` | ✅ `StrangGraph` (`from_path` / `from_cycle`) | ✅ `StrangGraph` *(full)* |
| `QuantumGraphHeatChernoff` | ✅ `smf_qgraph_*`, `smf_qgheat_*` | ✅ `QuantumGraph`, `QuantumGraphHeat` | ✅ `QuantumGraph`, `QuantumGraphHeat` *(full)* |
| `GraphKrylovChernoff`, `graph_expmv_frechet` | ❌ not exposed | ✅ `GraphKrylov`, `graph_expmv_frechet` | ❌ not exposed |

**Python `Graph` factory methods**: `Graph.path(n)`, `Graph.cycle(n)`,
`Graph.from_edges(n, edges)` where `edges` is a list of `(u, v, w)` triples or a
flat float64 ndarray of them, and `Graph.erdos_renyi(n, p, seed)`.
`GraphPath(n)` is retained as an alias for `Graph.path(n)`.

**Python `Laplacian` factory methods**: `Laplacian.combinatorial(graph)`,
`Laplacian.normalized(graph)`. Introspection: `n_nodes()`,
`is_combinatorial()`, `is_normalized()`, `spectral_bound()`.

**`MagnusGraphHeat6` callback**: the time-varying `L_G(t)` callback
accepts either a `Graph` (auto-assembled to combinatorial Laplacian) or a
`Laplacian` (used directly). This matches the Rust `LaplacianAtTime` API.

**Cross-binding sup-error gate** (ADR-0059): Python vs FFI sup-error ≤ 3 ULP
for `P_64` path graph, combinatorial Laplacian, `t = 0.5`, `n = 50`.

**Adjoint-state sensitivity** (ADR-0115):

| Rust symbol | PyO3 surface | Status | Notes |
|-------------|-------------|--------|-------|
| `MagnusGraphHeatChernoff::evolve_state_adjoint_into` | `GraphAdjoint.evolve_state_adjoint(lambda_n, t, n_steps)` | ✅ stable | kernel="magnus_graph"; math §42 T42.1 |
| `VarCoefMagnusGraphHeatChernoff::evolve_state_adjoint_into` | `GraphAdjoint(kernel="varcoef_magnus_graph")` | ✅ stable | a= callback required |
| `MagnusGraphHeatChernoff::from_presampled` | `GraphAdjointPresampled` | ✅ stable | also FFI `smf_graph_adjoint_new_presampled[_varcoef]` and WASM `GraphAdjointPresampled` *(full)* |
| `adjoint_state_gradient` + `EdgeWeightSensitivity` | `edge_weight_grad(graph, a, *, u0, dj_du_n, t, n_steps, rho_bar, params)` | ✅ stable | params: list[(i,j)] or "all_edges" |
| `GraphHeatChernoff<f32>` path | `GraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `MagnusGraphHeatChernoff<f32>` path | `MagnusGraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `VarCoefGraphHeatChernoff<f32>` path | `VarCoefGraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `DiffusionChernoff<f32>` path (1D) | `Heat1D(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `Laplacian::row_ptr` / `col_idx` / `vals` (CSR) | `Laplacian.row_ptr()` / `.col_idx()` / `.vals()` | ✅ stable | copy; frozen-topology invariant |
| `Laplacian` dense reconstruction | `Laplacian.to_dense()` | ✅ stable | O(n²) copy; raises OutOfDomain on overflow |

---

## 5. Boundary Policies

| Policy | Rust `BoundaryPolicy` | PyO3 string literal |
|--------|-----------------------|---------------------|
| Reflect (default) | `BoundaryPolicy::Reflect` | `'reflect'` |
| Periodic | `BoundaryPolicy::Periodic` | `'periodic'` |
| Zero-extend | `BoundaryPolicy::ZeroExtend` | `'zero'` |
| Linear extrapolation | `BoundaryPolicy::LinearExtrapolate` | `'linear'` |

The `boundary` kwarg is accepted by the grid-based kernels — among others
`Heat1D`, `Heat1D4th`, `Heat1D6th`, `Heat1DZeta4/6/8`, `DriftReaction1D`,
`Shift1D`, `Heat2D`, `Heat3D`, `Heat2DVarA`, `Heat3DVarA`,
`AnisotropicShiftND2/3` and the boundary-condition kernels (the stubs list the
full set). Unknown string values raise `SemiflowError` with the message
`[OutOfDomain] unknown boundary policy …` listing the accepted values.
Implemented in `crates/semiflow-py/src/boundary.rs`.

---

## 6. Variable-Coefficient Paths

| Kernel | fn-ptr `::new` | closure `with_closure` (Rust only) | Pre-sampled arrays (Python) |
|--------|---------------|------------------------------------|------------------------------------|
| `DiffusionChernoff` | ✅ Rust | ✅ Rust (`DiffusionChernoff::with_closure`) | ✅ Python (`Heat1D.with_a_array(...)`) |
| `Diffusion4thChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`Heat1D4th.with_a_array(...)`) |
| `Diffusion6thChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`Heat1D6th.with_a_array(...)`) |
| `DriftReactionChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`DriftReaction1D.with_arrays(...)`) |
| `ShiftChernoff1D` | ✅ Rust | ✅ Rust (`ShiftChernoff1D::with_closure`, `f64`) | ✅ Python (`Shift1D.with_arrays(...)`) |
| `VarCoefGraphHeatChernoff` | ✅ Rust (`new(graph, a, rho_bar)`, node array) | — | ✅ Python (`VarCoefGraphHeat(graph, a, rho_bar=…)`) |
| `NonSeparableMixedChernoff` | ✅ Rust | ✅ Rust (`nonseparable_mixed_closure::with_closure_beta` / `with_closure_c`) | ✅ Python (`NonSeparable2D.with_beta_array(...)`) |

**Performance note**: the pre-sampled array path performs cubic-Hermite (1D) or
bilinear (2D) interpolation inside Rust with zero GIL re-acquires. Measured
speedup vs the `with_a_function` Python-callback path: approximately 10× for
`n = 1000` grid, `n_steps = 200` (Phase 2 benchmark, referenced in ADR-0061).
The `with_a_function` callback API is preserved for backwards compatibility but
is not the recommended path for performance-sensitive code.

---

## 7. Reverse-mode AD (since 0.9.0-beta, ADR-0154/0156)

**`ReverseHeat1D`** is the Python (PyO3) and JavaScript (WASM) binding for
`semiflow::ReverseChernoff<f64>` with constant-a `DiffusionChernoff`. There is
no C (FFI) surface for it.

**Python (`semiflow-py`) — `ReverseHeat1D`:**

```python
import numpy as np
from semiflow import ReverseHeat1D   # see note below

n_grid = 24
x = np.linspace(-4.0, 4.0, n_grid)
rc = ReverseHeat1D(theta=0.4, xmin=-4.0, xmax=4.0, n_grid=n_grid, n_steps=8)
u0     = np.exp(-x**2)       # float64, shape (n_grid,)
target = np.zeros(n_grid)    # float64, shape (n_grid,)
value, grad = rc.value_and_grad(tau=0.05, u0=u0, target=target)
# value: float  — L² loss ‖(F_θ(τ))ⁿ u₀ − target‖²
# grad:  float  — ∂J/∂θ (scalar diffusivity gradient, K=1 forward-mode Dual, 0-ULP)
```

**WASM (`@semiflow/wasm`) — `ReverseHeat1D`:**

```js
const rc = new ReverseHeat1D(0.4, -4.0, 4.0, 24, 8);
const result = rc.valueAndGrad(0.05, u0, target);
// result: Float64Array[2] — [value, grad]
```

> **Note:** up to 0.13.1-beta `ReverseHeat1D` was registered in the native
> module but missing from the package's re-export list, so `from semiflow import
> ReverseHeat1D` raised `ImportError`. Fixed in 0.14.0-beta;
> `tests/test_public_exports.py` guards it.

**NARROW scope (§51.5, ADR-0156):** constant-a `DiffusionChernoff` ONLY.
Variable-coefficient and nonlinear kernels are out of scope.
Gradient parity: 0-ULP between PyO3 and WASM implementations
(`G_BINDING_REVERSE_AD_PARITY`).

The tensor-train and gridless carriers that shipped alongside reverse-mode AD
(`TtChernoff`, `CoupledTtChernoff`, `VarCoefTt`, `GridlessChernoff`) are bound
in all three bindings — see §1. Only the `s3-poc` evolvers remain Rust-only.

---

## 8. Generic Symmetric Operator (v0.11.0-beta, issue #15; `path="implicit"` issue #16)
| Rust type / function | PyO3 surface | Status | Notes |
|----------------------|-------------|--------|-------|
| `SymmetricOperator::from_csr` | `SymmetricOperator.from_csr(indptr, indices, data, n)` | ✅ stable | accepts CSR arrays (int32 or int64 indices) |
| `SymmetricOperator::evolve_batched` | `SymmetricOperator.evolve_batched(v, t, *, path="lanczos", n_steps=100)` | ✅ stable | `path=` kwarg selects Krylov mode |
| `mass_lumped_evolve` | `mass_lumped_evolve(K_csr, mass_diag, v, t, *, path="lanczos", n_steps=100)` | ✅ stable | lumped-mass `(M,K)` 3-liner |
| `MassKOperator::evolve` | `MassKOperator.evolve(v, t, *, path="lanczos", n_steps=100)` | ✅ stable | consistent-mass `(M,K)` via Cholesky congruence |
| `symmetric_op_expmv_frechet` | `symmetric_op_expmv_frechet(op, v, t, dj_du)` | ✅ stable | combined action + per-entry Fréchet gradient |
| `EntrySensitivity` | returned by `symmetric_op_expmv_frechet`; not a separate Python class | ✅ stable | gradient returned directly |
**`path=` values for `evolve_batched`, `mass_lumped_evolve`, `MassKOperator.evolve`:**
| `path=` | Method | When to use |
|---------|--------|-------------|
| `"lanczos"` (default) | Lanczos Krylov expmv | General-purpose; matvec count flat in `t` per sub-step |
| `"chebyshev"` | Chebyshev polynomial expmv | O(1) working vectors; slightly cheaper per matvec than Lanczos for moderate stiffness |
| `"implicit"` | PCG backward-Euler `(I+Δt·A)^{−n_steps}` (ADR-0190, issue #16) | Stiff operators with `λ_max ≳ 10⁵` where the explicit paths time out |
**`n_steps` parameter** (default 100, only used with `path="implicit"`): number of
backward-Euler sub-steps. Accuracy is O(t/n_steps); increase to tighten tolerance.
Cost per sub-step is proportional to `√κ` of the Jacobi-preconditioned system,
not strictly `λ_max`-independent (IC(0) is deferred, §59.6).
**Notes**
- GIL is released inside `evolve_batched` for all `path=` values.
- `SymmetricOperator` accepts any externally-assembled symmetric PSD sparse matrix
  (FEM stiffness with Robin BC, anisotropic conductivity, conservative diffusion via
  `to_symmetric_operator()`). The operator must be PSD; a non-PSD matrix is a
  domain error.
- FFI and WASM do not expose the symmetric-operator surface (PyO3-only; ADR-0186).
- `MassKOperator` gradient via `EntrySensitivity` covers `SymmetricOperator` only;
  `MassKOperator` differentiability is deferred.
---

## 9. 0.13.0-beta issue campaign (#17 / #19 / #21–#26)

Python surface added by the campaign. FFI and WASM are unchanged for all of it —
see ADR-0191 AMENDMENT 2 for why the ND `boundary=` kwarg in particular was NOT
mirrored (neither surface exposes boundary selection for *any* kernel, so a
two-constructor exception would create an inconsistency rather than close one).
| Surface | Rust | Python | FFI | WASM | Authority |
|---------|------|--------|-----|------|-----------|
| `GeneralOperator::from_csr` + `expmv` (non-symmetric CSR) | ✅ | ✅ `GeneralOperator` | ❌ | ❌ | ADR-0195 |
| `shift1d_coeff_gradient` (VJP w.r.t. `a`/`b`/`c` fields) | ✅ | ✅ `shift1d_coeff_grad` | ❌ | ❌ | ADR-0197 |
| Batched multi-channel 1-D evolve | ✅ `grid_batched` | ✅ `Shift1D.evolve_batched` | ❌ | ❌ | ADR-0194 |
| Full coefficient schedules (`a`/`b`/`c`, scalar or array) | ✅ | ✅ `Shift1D.evolve_with_coefficient_schedule` | ❌ | ❌ | ADR-0193 |
| `AdaptivePI` over pre-sampled coefficients | ✅ | ✅ `AdaptivePI.with_arrays` | ❌ | ❌ | ADR-0193 |
| Per-pencil 2-D composition (transverse-varying `a`) | ✅ `Strang2DPencil` | ✅ `Heat2DVarA.with_grid_arrays` | ❌ | ❌ | ADR-0196 |
| ND `boundary=` selection | ✅ (`Grid1D::with_boundary`, always) | ✅ `AnisotropicShiftND2/3(boundary=…)` | ❌ none for any kernel | ❌ none for any kernel | ADR-0191 |
| ND 2-D/3-D array state (`set_state`, `values_2d`) | n/a | ✅ | ❌ | ❌ | ADR-0191 |
Behaviour changes on the existing Python surface, not additions:
| Surface | Change | Authority |
|---------|--------|-----------|
| every `D > 1` kernel | `GridFnND::sample` now honours `InterpKind` + `BoundaryPolicy`; results change (they become correct) | ADR-0191 |
| `Heat2DVarA.order()`, `Heat3DVarA.order()` | 2 → **1**; the axis kernels freeze `a` at the node | ADR-0191 AM1 |
| `ConservativeDiffusionChernoff` | accepts `k ≥ 0` (CEV / Feller / Wright–Fisher degenerate ends) | ADR-0192 |
| `DiffusionExpmv1D`, graph Lanczos | corrected θ_m table changes the substep count | ADR-0198 |
## 10. Known Gaps and Deferred Items

The following items are not exposed through the named bindings as of
0.14.0-beta:

| Item | Missing from | Reason for gap |
|------|--------------|----------------|
| `S3*` evolvers (`s3-poc` feature) | FFI, PyO3, WASM | Research-track surface; binding design deferred (ADR-0169) |
| `ReverseChernoff` | FFI | Bound as `ReverseHeat1D` in PyO3 and WASM only |
| Symmetric-operator / `(M, K)` / general-operator / Krylov / φ-function / ETD surface (§8, §9) | FFI, WASM | PyO3-only (ADR-0186 and the 0.13.0-beta campaign ADRs) |
| `boundary=` selection | FFI, WASM | Neither surface exposes boundary selection for any kernel (ADR-0191 AMENDMENT 2) |
| `f32` paths | FFI, WASM | Out of scope (ADR-0115) |
| Heavy-grid engines in the npm package | WASM (npm) | The published package is the lite build; build with `--features full` |
| Live-callback `GraphAdjoint` | FFI, WASM | Closures are not ABI-safe; use the pre-sampled `GraphAdjointPresampled` path |
| Async / yield PyO3 API | PyO3 | Insufficient telemetry on GIL-release saturation (ADR-0034 §"Out of scope") |
