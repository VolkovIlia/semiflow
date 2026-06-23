---
version: 1.4.0
last_updated: 2026-06-19
freshness_score: 1.0
dependencies:
  - crates/semiflow/src/lib.rs
  - crates/semiflow-py/src/
  - crates/semiflow-ffi/src/
  - crates/semiflow-wasm/src/
  - docs/adr/0028-bindings-ffi-pyo3-wasm.md
  - docs/adr/0035-v1_0_0-api-stability.md
  - docs/adr/0059-graph-bindings.md
  - docs/adr/0061-python-parity-expansion.md
  - docs/adr/0154-binding-parity-v9.md
  - docs/adr/0156-reverse-ad.md
  - docs/adr/0159-tt-chernoff.md
  - docs/adr/0162-tt-coupled-spectral.md
  - docs/adr/0169-s3-honest-scope-public-api-promotion.md
changelog:
  - 1.0.0: Initial coverage matrix for v2.3.0 Python parity expansion (feat/python-parity-v2.3)
  - 1.1.0: v6.2.2 ADR-0115 additions — GraphAdjoint, edge_weight_grad, dtype kwarg, Laplacian accessors, from_edges fix
  - 1.2.0: v9.0.0 — ReverseHeat1D added to PyO3 + WASM; TtChernoff/TtState/GridlessChernoff/ParticleReduction Rust-only
  - 1.3.0: v9.1.0 — CoupledTtChernoff Rust-only (TT contraction interface design deferred)
  - 1.4.0: v9.2.0 — six S3* types (s3-poc feature) Rust-only; no new binding exposure
graph-unverified: false
---

# Python Coverage Matrix

This document tracks binding parity across the four public surfaces of the
`semiflow` workspace. It was created at **v2.3.0** and has been updated
through **v9.2.0** (ADR-0169, 2026-06-19). The Python expansion follows the
lockstep SemVer rule of ADR-0035: all four crates (`semiflow`, `semiflow-ffi`,
`semiflow-py`, `semiflow-wasm`) bump together at the final tag. See
[`docs/audit-findings-v2_3_0.md`](audit-findings-v2_3_0.md) for the companion
math-fidelity and gate report.

**Legend**

| Symbol | Meaning |
|--------|---------|
| ✅ stable | Exposed and covered by an acceptance gate |
| 🚧 experimental | Exposed; API may change in a MINOR release |
| ❌ not exposed | Implemented in core; not yet surfaced in this crate |

All cells are for `f64` unless noted. `f32` is now opt-in for PyO3 on four
kernels (`GraphHeat`, `MagnusGraphHeat`, `VarCoefGraphHeat`, `Heat1D`) via the
`dtype="f32"` kwarg (v6.2.2, ADR-0115). FFI and WASM f32 paths remain out of
scope (ADR-0115).

---

## 1. 1D Kernels

| Rust type | Rust rlib | FFI (`semiflow-ffi`) | PyO3 (`semiflow-py`) | WASM (`@semiflow/wasm`) |
|-----------|-----------|---------------------|---------------------|------------------------|
| `DiffusionChernoff` | ✅ stable | ✅ stable (`Heat1D`, unit-a) | ✅ stable (`Heat1D`, var-a via `with_a_array` / `with_a_function`) | ✅ stable (`Heat1D`) |
| `Diffusion4thChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`Heat1D4th`, `with_arrays`) | ❌ not exposed |
| `Diffusion6thChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`Heat1D6th`, `with_arrays`) | ❌ not exposed |
| `DriftReactionChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`DriftReaction1D`, `with_arrays`) | ❌ not exposed |
| `ShiftChernoff1D` | ✅ stable | ❌ not exposed | ✅ stable (`Shift1D`, `with_arrays`) | ❌ not exposed |
| `TruncatedExpDiffusionChernoff` | ✅ stable | ❌ not exposed | ❌ not exposed | ❌ not exposed |
| `TruncatedExp4thDiffusionChernoff` | ✅ stable | ❌ not exposed | ❌ not exposed | ❌ not exposed |
| `ReverseChernoff<F>` + `CheckpointSchedule` | ✅ stable | ❌ not exposed | ✅ stable (`ReverseHeat1D`, constant-a narrow scope, ADR-0156) | ✅ stable (`ReverseHeat1D`, ADR-0154) |
| `TtChernoff<F>` + `TtState<F>` | ✅ stable | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `GridlessChernoff<F, D>` + `ParticleReduction` | ✅ stable | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `CoupledTtChernoff<F>` (v9.1.0, ADR-0162) | ✅ stable | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3DriftSpectralEvolver<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3DenseCouplingEvolver<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3VarCoefEvolver<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3NonSepVarCoefEvolver<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3BurgersColeHopf<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |
| `S3ReactionDiffusion<F>` (v9.2.0, `s3-poc`) | ✅ experimental | ❌ not exposed | ❌ not exposed (Rust-only) | ❌ not exposed (Rust-only) |

**Notes**

- `Heat1D` in PyO3 exposes the `boundary` kwarg (`'reflect'` / `'periodic'` /
  `'zero'` / `'linear'`; default `'reflect'`) since Phase 1.
- Pre-sampled coefficient path (`with_a_array`) performs cubic-Hermite
  interpolation inside Rust, achieving zero GIL re-acquires during `evolve`.
  See ADR-0061 §"Pre-sampled coefficients".
- `TruncatedExpDiffusionChernoff` and `TruncatedExp4thDiffusionChernoff` remain
  Rust-only; no user demand has been expressed for these through bindings.
  Deferred to a future MINOR release.

---

## 2. 2D / 3D Composition

| Rust type | Rust rlib | FFI | PyO3 | WASM |
|-----------|-----------|-----|------|------|
| `Strang2D` | ✅ stable | ❌ not exposed | ✅ stable (`Heat2D`, `boundary` kwarg) | ❌ not exposed |
| `Strang3D` | ✅ stable | ❌ not exposed | ✅ stable (`Heat3D`, `boundary` kwarg) | ❌ not exposed |
| `NonSeparableMixedChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`NonSeparable2D`, `with_beta_array`) | ❌ not exposed |
| `NonSeparable2DChernoff` (alias) | ✅ stable | ❌ not exposed | ✅ stable (via `NonSeparable2D`) | ❌ not exposed |
| `NonSeparable2DAnisotropicChernoff` (alias) | ✅ stable | ❌ not exposed | ✅ stable (via `NonSeparable2D`) | ❌ not exposed |

**Notes**

- `Heat2D` and `Heat3D` gained the `boundary` kwarg in Phase 1; internally
  wired through `Grid1D::new_with_policy` on each axis.
- `NonSeparable2D` wraps the unified `NonSeparableMixedChernoff` type
  (ADR-0058). The constant-`c` path and the `with_beta_array` pre-sampled
  β(x,y) path via bilinear interpolation are both exposed. See `coeff2d.rs`.
- `StrangSplitGraph` (bipartite graph Strang) is Rust-only; no Python surface
  is planned for v2.3 — the expected use pattern is `GraphHeat4th` + manual
  Strang composition by the caller.

---

## 3. Adjoint / Schrödinger / Adaptive Wrappers

| Rust type | Rust rlib | FFI | PyO3 | WASM |
|-----------|-----------|-----|------|------|
| `AdjointChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`Adjoint`, 5-variant enum dispatch) | ❌ not exposed |
| `SchrodingerChernoff` + `SchrodingerState` | ✅ stable | ❌ not exposed | ✅ stable (`Schrodinger1D`) | ❌ not exposed |
| `AdaptivePI` | ✅ stable | ❌ not exposed | ✅ stable (`AdaptivePI`, 5-variant enum dispatch) | ❌ not exposed |

**Adjoint dispatch variants** (Phase 4): `Heat1D`, `Heat1D4th`, `Heat1D6th`,
`DriftReaction1D`, `Shift1D`. Adding a new inner kernel requires extending the
enum in `crates/semiflow-py/src/adjoint.rs`; this is a known rigidity trade-off
documented in ADR-0061 §"Consequences".

**AdaptivePI dispatch variants** (Phase 4): same 5 kernels as Adjoint.
Return value is a dict `{final_state, steps_accepted, steps_rejected, last_tau}`.

**Schrödinger** (Phase 3): 4 constructors — default-V, `from_parts`
(psi\_re / psi\_im), `with_potential` (pre-sampled V array), and
`with_potential_parts`. Methods: `evolve(t, n_steps=200)`, `values()` →
complex128 ndarray, `values_parts()` → (float64, float64) ndarrays,
`norm_squared()`, `__len__()`. Unitarity gate: `‖ψ‖²/‖ψ₀‖² − 1 < 1e-6` over
500 steps on the harmonic oscillator (Phase 3 acceptance gate).

---

## 4. Graph PDE

| Rust type | Rust rlib | FFI | PyO3 | WASM |
|-----------|-----------|-----|------|------|
| `Graph` | ✅ stable | ✅ stable (opaque `smf_graph_t`) | ✅ stable (`Graph` pyclass) | ✅ stable (`Graph` JS class) |
| `Laplacian` | ✅ stable | ✅ stable (opaque `smf_laplacian_t`) | ✅ stable (`Laplacian` pyclass) | ✅ stable (`Laplacian` JS class) |
| `GraphHeatChernoff` | ✅ stable | ✅ stable | ✅ stable (`GraphHeat`) | ✅ stable (`GraphHeat`) |
| `GraphHeat4thChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`GraphHeat4th`) | ❌ not exposed |
| `MagnusGraphHeatChernoff` | ✅ stable | ✅ stable | ✅ stable (`MagnusGraphHeat`) | ✅ stable (`MagnusGraphHeat`) |
| `MagnusGraphHeat6thChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`MagnusGraphHeat6`) | ❌ not exposed |
| `VarCoefGraphHeatChernoff` | ✅ stable | ❌ not exposed | ✅ stable (`VarCoefGraphHeat`, `with_beta_array`) | ❌ not exposed |
| `GraphTraj` | ✅ stable | ❌ not exposed | ❌ not exposed | ❌ not exposed |
| `StrangSplitGraph` | ✅ stable | ❌ not exposed | ❌ not exposed | ❌ not exposed |

**Python `Graph` factory methods** (Phase 5): `Graph.path(n)`, `Graph.cycle(n)`,
`Graph.from_edges(n_nodes, edges)` where `edges` is a flat float64 ndarray of
(u, v, w) triples, and `Graph.erdos_renyi(n, p, seed)`.
`GraphPath(n)` is retained as a deprecated alias for `Graph.path(n)`.

**Python `Laplacian` factory methods** (Phase 5): `Laplacian.combinatorial(graph)`,
`Laplacian.normalized(graph)`. Introspection properties: `n_nodes`,
`is_combinatorial`, `is_normalized`, `spectral_bound`.

**`MagnusGraphHeat6` callback** (Phase 5): the time-varying `L_G(t)` callback
accepts either a `Graph` (auto-assembled to combinatorial Laplacian) or a
`Laplacian` (used directly). This matches the Rust `LaplacianAtTime` API.

**Cross-binding sup-error gate** (ADR-0059): Python vs FFI sup-error ≤ 3 ULP
for `P_64` path graph, combinatorial Laplacian, `t = 0.5`, `n = 50`.

**v6.2.2 additions** (ADR-0115, Issue #2 + #1 + #3 + #5):

| Rust symbol | PyO3 surface | Status | Notes |
|-------------|-------------|--------|-------|
| `MagnusGraphHeatChernoff::evolve_state_adjoint_into` | `GraphAdjoint.evolve_state_adjoint(lambda_n, t, n_steps)` | ✅ stable | kernel="magnus_graph"; math §42 T42.1 |
| `VarCoefMagnusGraphHeatChernoff::evolve_state_adjoint_into` | `GraphAdjoint(kernel="varcoef_magnus_graph")` | ✅ stable | a= callback required |
| `adjoint_state_gradient` + `EdgeWeightSensitivity` | `edge_weight_grad(graph, a, *, u0, dj_du_n, t, n_steps, rho_bar, params)` | ✅ stable | params: list[(i,j)] or "all_edges" |
| `GraphHeatChernoff<f32>` path | `GraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `MagnusGraphHeatChernoff<f32>` path | `MagnusGraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `VarCoefGraphHeatChernoff<f32>` path | `VarCoefGraphHeat(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `DiffusionChernoff<f32>` path (1D) | `Heat1D(dtype="f32")` | ✅ stable | f64 default; f32 opt-in |
| `Laplacian::row_ptr` / `col_idx` / `vals` (CSR) | `Laplacian.row_ptr()` / `.col_idx()` / `.vals()` | ✅ stable | copy; frozen-topology invariant |
| `Laplacian` dense reconstruction | `Laplacian.to_dense()` | ✅ stable | O(n²) copy; raises OutOfDomain on overflow |
| `Graph::from_edges` (flat 3M array) | `Graph.from_edges(n, edges)` — both list and flat array | ✅ stable (fix) | was broken for flat array; error message corrected |

**Gaps (unchanged)**: `GraphTraj` and `StrangSplitGraph` are Rust-only. `GraphTraj` requires
mutable closure lifetimes that are difficult to express safely in PyO3 without a
GIL-hold; deferred to v2.4+.

---

## 5. Boundary Policies

| Policy | Rust `BoundaryPolicy` | PyO3 string literal |
|--------|-----------------------|---------------------|
| Reflect (default) | `BoundaryPolicy::Reflect` | `'reflect'` |
| Periodic | `BoundaryPolicy::Periodic` | `'periodic'` |
| Zero-extend | `BoundaryPolicy::ZeroExtend` | `'zero'` |
| Linear extrapolation | `BoundaryPolicy::LinearExtrapolate` | `'linear'` |

The `boundary` kwarg is accepted by `Heat1D`, `Heat1D4th`, `Heat1D6th`,
`DriftReaction1D`, `Shift1D`, `Heat2D`, `Heat3D`. Unknown string values raise
`SemiflowError(kind='OutOfDomain')` with the list of accepted values.
Implemented in `crates/semiflow-py/src/boundary.rs`.

---

## 6. Variable-Coefficient Paths

| Kernel | fn-ptr `::new` | closure `with_closure` (Rust only) | Pre-sampled `with_arrays` (Python) |
|--------|---------------|------------------------------------|------------------------------------|
| `DiffusionChernoff` | ✅ Rust | ✅ Rust (`DiffusionChernoff::with_closure`) | ✅ Python (`Heat1D(a=a_values, ...)`) |
| `Diffusion4thChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`Heat1D4th.with_arrays(...)`) |
| `Diffusion6thChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`Heat1D6th.with_arrays(...)`) |
| `DriftReactionChernoff` | ✅ Rust | ✅ Rust (`with_closure`) | ✅ Python (`DriftReaction1D.with_arrays(...)`) |
| `ShiftChernoff1D` | ✅ Rust | — (no separate closure variant) | ✅ Python (`Shift1D.with_arrays(...)`) |
| `VarCoefGraphHeatChernoff` | ✅ Rust | ✅ Rust (`with_closure_beta`) | ✅ Python (`VarCoefGraphHeat.with_beta_array(...)`) |
| `NonSeparableMixedChernoff` | ✅ Rust | ✅ Rust (`with_closure_beta`) | ✅ Python (`NonSeparable2D.with_beta_array(...)`) |

**Performance note**: the pre-sampled array path performs cubic-Hermite (1D) or
bilinear (2D) interpolation inside Rust with zero GIL re-acquires. Measured
speedup vs the `with_a_function` Python-callback path: approximately 10× for
`n = 1000` grid, `n_steps = 200` (Phase 2 benchmark, referenced in ADR-0061).
The `with_a_function` callback API is preserved for backwards compatibility but
is not the recommended path for performance-sensitive code.

---

## 7. Reverse-mode AD (v9.0.0, ADR-0154/0156)

**`ReverseHeat1D`** is the Python (PyO3) and JavaScript (WASM) binding for
`semiflow_core::ReverseChernoff<f64>` with constant-a `DiffusionChernoff`.

**Python (`semiflow-py`) — `ReverseHeat1D`:**

```python
from semiflow import ReverseHeat1D

rc = ReverseHeat1D(theta=0.4, xmin=-4.0, xmax=4.0, n_grid=24, n_steps=8)
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

**NARROW scope (§51.5, ADR-0156):** constant-a `DiffusionChernoff` ONLY.
Variable-coefficient and nonlinear kernels are out of scope at v9.0.0.
Gradient parity: 0-ULP between PyO3 and WASM implementations
(`G_BINDING_REVERSE_AD_PARITY`).

**v9.0.0 Rust-only types (not bound):**

| Rust type | Reason |
|-----------|--------|
| `TtChernoff<F>` + `TtState<F>` | Multi-core TT contraction and SVD-based rounding require a well-typed ND array interface; deferred pending design work |
| `GridlessChernoff<F, D>` + `ParticleReduction` | Particle ensemble API (variable-length `MeasureState`) is awkward to express safely in PyO3/WASM without a design pass; deferred |

**v9.1.0 Rust-only types (not bound):**

| Rust type | Reason |
|-----------|--------|
| `CoupledTtChernoff<F>` | TT contraction interface design for multi-core adjacent-pair coupling deferred (same design dependency as `TtChernoff`) |

**v9.2.0 Rust-only types (not bound, `s3-poc` feature only):**

| Rust type | Reason |
|-----------|--------|
| `S3DriftSpectralEvolver<F>` | S³ POC — binding design deferred; Rust-only at v9.2.0 |
| `S3DenseCouplingEvolver<F>` | S³ POC — binding design deferred |
| `S3VarCoefEvolver<F>`, `AxisCoef<F>` | S³ POC — binding design deferred |
| `S3NonSepVarCoefEvolver<F>`, `CpTerm/CpCoef/CoefRole` | S³ POC — CP-coefficient interface complex to represent in Python safely; deferred |
| `S3BurgersColeHopf<F>`, `S3ReactionDiffusion<F>`, `Reaction<F>` | S³ POC — binding design deferred |

---

## 8. Known Gaps and Deferred Items

The following items are Rust-only as of v9.2.0 and are not exposed through any
binding:

| Item | Reason for gap | Target release |
|------|----------------|----------------|
| `TruncatedExpDiffusionChernoff` | No expressed user demand | Unscheduled |
| `TruncatedExp4thDiffusionChernoff` | No expressed user demand | Unscheduled |
| `StrangSplitGraph` | Bipartite edge-set API awkward in Python; caller can compose manually | Unscheduled |
| `GraphTraj` | Mutable closure lifetimes unsafe across GIL boundary; needs design work | v2.4+ |
| `f32` Python path (FFI/WASM) | FFI and WASM f32 surfaces remain out of scope (ADR-0115) | Unscheduled |
| `CoupledTtChernoff<F>` | TT interface design pending | v9.x |
| `S3*` evolvers (`s3-poc`) | POC track — binding design deferred | v10.0+ |
| Async / yield PyO3 API | Insufficient telemetry on GIL-release saturation (ADR-0034 §"Out of scope") | Unscheduled |
| FFI/WASM surface for 4th/6th-order, Schrödinger, Adjoint, AdaptivePI | Large LoC cost; FFI/WASM callers can use Python or Rust directly | Unscheduled |
| `TtChernoff<F>` + `TtState<F>` | ND array / TT binding design deferred | Unscheduled |
| `GridlessChernoff<F, D>` + `ParticleReduction` | Particle ensemble binding design deferred | Unscheduled |
