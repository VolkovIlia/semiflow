# Changelog

All notable changes to SemiFlow are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] — binding-parity follow-up to 0.9.0-beta

### Changed — bindings

- **Near-full binding parity** across `semiflow-ffi` (C ABI), `semiflow-py`
  (PyO3), and `semiflow-wasm` (wasm-bindgen).  All three bindings now reach
  the full engine set shipped in the core: higher-order ζ-ladder, 2D/3D
  tensor-product, non-separable anisotropic, boundary conditions (Killing /
  Reflected / Robin / Resolvent / KilledDirichlet / Obstacle), Schrödinger
  (real + complex), matrix diffusion, Howland nonautonomous, subordinated,
  manifold (Torus/Sphere2/Hyperbolic2), hypoelliptic (Heisenberg/Kolmogorov/
  Engel), graph family (4th-order, Magnus-6, VarCoef, Quantum, Strang),
  sparse-grid SmolyakD6, Adjoint, AdaptivePI, and ComplexTripleJump/PointEval.
- **S³ carrier surface stabilised** (ADR-0171): `TtState/TtEvolver`,
  `TtCoupledEvolver`, and `GridlessEvolver/MeasureState` are now exposed
  across all three binding layers — `semiflow-ffi` (C opaque-handle ABI),
  `semiflow-py` (PyO3), and `semiflow-wasm` (wasm-bindgen) — and are covered
  by dedicated smoke tests (`crates/semiflow-ffi/tests/ffi_s3_smoke.rs` and
  `crates/semiflow-wasm/tests/s3_smoke.rs`).  The `s3-poc` cargo feature that
  previously guarded the six S³ POC evolvers is retired — those types are now
  part of the default core API.
  Exported C symbols: `smf_ttstate_new_separable`, `smf_ttstate_free`,
  `smf_ttstate_ndim`, `smf_ttstate_n_j`, `smf_ttstate_peak_rank`,
  `smf_ttstate_storage_size`, `smf_ttstate_inner_separable`,
  `smf_tt_evolver_new`, `smf_tt_evolver_evolve`, `smf_tt_evolver_free`
  (`SmfTtState`, `SmfTtEvolver`); `smf_tt_coupled_new`,
  `smf_tt_coupled_evolve`, `smf_tt_coupled_free` (`SmfTtCoupledEvolver`);
  `smf_measurestate_new`, `smf_measurestate_free`, `smf_measurestate_n_diracs`,
  `smf_measurestate_total_variation`, `smf_measurestate_second_moment`,
  `smf_measurestate_marginal`, `smf_gridless_new`, `smf_gridless_apply`,
  `smf_gridless_evolve`, `smf_gridless_free` (`SmfMeasureState`,
  `SmfGridlessEvolver`).  WASM JS types: `TtState`, `TtEvolver`,
  `TtCoupledEvolver`, `MeasureState`, `GridlessEvolver`.
- **WASM `full` cargo feature**: the default/"lite" WASM build stays small
  (≈ 768 KB raw, baseline 1D + graph engines); `--features full` enables all
  heavy-grid, multi-dimensional, and hypoelliptic engines (≈ 1.4 MB raw).
- **Cargo.toml description fields** updated to reflect broad engine surface
  (no longer say "1D heat, unit diffusion only").

### Fixed — PyO3 S³ wiring (issue #4, 2026-06-22)

- **`semiflow-py` S³ modules were orphaned** after ADR-0171 wiring (2026-06-20):
  `tt_py.rs`, `tt_coupled_py.rs`, and `gridless_py.rs` existed but were never
  declared (`mod`) or registered in `lib.rs`, so `from semiflow import TtState`
  raised `ImportError` at runtime.  Commit `64654b9` (feature `fe840b7`) adds
  the three `mod` declarations, `register()` calls, and `__init__.py` re-exports;
  39/39 `test_s3_engines.py` now pass.  FFI and WASM surfaces were not affected.

### Added — new type bindings (Pass 2)

- **`DirichletHeat2nd1D`** (order-2 absorbing Dirichlet BC, odd-image method,
  §21.9, ADR-0176, issue #6): exposed across all three binding layers —
  `semiflow-ffi` (`smf_dirichlet_heat2nd1d_*`, `SmfDirichletHeat2nd1D`),
  `semiflow-py` (`DirichletHeat2nd1D` pyclass), and `semiflow-wasm`
  (`DirichletHeat2nd1D` JS class, `--features full`).
  PEP 561 `.pyi` stub added.
- **`VarCoefTtEvolver`** (additive-separable variable-coefficient TT carrier,
  §52.10, ADR-0178, issue #2): exposed across all three binding layers —
  `semiflow-ffi` (`smf_varcoef_tt_evolver_*`, `SmfVarCoefTtEvolver` in
  `tt_varcoef_ffi.rs`), `semiflow-py` (`VarCoefTtEvolver` pyclass in
  `tt_varcoef_py.rs`), `semiflow-wasm` (`VarCoefTtEvolver` JS class in
  `tt_varcoef_wasm.rs`).  Operates on the same `TtState` carrier as
  `TtEvolver`; `VarCoefOutOfClass` → `OutOfDomain` on all surfaces.
  PEP 561 `.pyi` stub added.

### Added — new type bindings (bind-remaining-operators wave)

- **`DiffusionExpmv1D`** (tolerance-driven Al-Mohy & Higham expmv, ADR-0121,
  `order() = u32::MAX`): exposed across all three binding layers —
  `semiflow-ffi` (`smf_expmv1d_*`, `SmfExpmv1D`), `semiflow-py` (`DiffusionExpmv1D`
  pyclass), and `semiflow-wasm` (`DiffusionExpmv1D` JS class, `--features full`).
  Uses static unit-a / zero-drift fn-pointers; no closures.  PEP 561 `.pyi` stub added.

- **`DriftReaction4th1D`** (order-4 palindromic Strang drift-reaction, ADR-0127):
  exposed across all three binding layers — `semiflow-ffi`
  (`smf_drift_reaction_zeta4_*`, `SmfDriftReactionZeta4`), `semiflow-py`
  (`DriftReaction4th1D` pyclass), and `semiflow-wasm` (`DriftReaction4th1D`
  JS class, `--features full`).  Fixed `b=0.5`, `b'=0.0`, `c=0.0` via static
  fn-pointers (closure API is a separate architect task).  PEP 561 `.pyi` stub added.

- **`Killing2nd1D`** (order-2 soft-killing Feynman-Kac, ADR-0126): exposed across
  all three binding layers — `semiflow-ffi` (`smf_killing2nd_*`, `SmfKilling2nd`),
  `semiflow-py` (`Killing2nd1D` pyclass), and `semiflow-wasm` (`Killing2nd1D`
  JS class, `--features full`).  Constant `κ ≥ 0` via `ConstKappa`/`ConstKappaWasm`
  newtype implementing `KillingRate<f64>`.  PEP 561 `.pyi` stub added.

- **`MatrixDiffusion2D`** (coupled 2-component 2D palindromic Strang, ADR-0124):
  exposed across all three binding layers — `semiflow-ffi` (`smf_matrix2d_*`,
  `SmfMatrix2D`), `semiflow-py` (`MatrixDiffusion2D` pyclass), and `semiflow-wasm`
  (`MatrixDiffusion2D` JS class, `--features full`).  Buffer layout:
  `2*nx*ny` f64, index `(j*nx+i)*2+component`.  PEP 561 `.pyi` stub added.

- **`MatrixDiffusion3D`** (coupled 2-component 3D palindromic Strang, ADR-0124):
  exposed across all three binding layers — `semiflow-ffi` (`smf_matrix3d_*`,
  `SmfMatrix3D`), `semiflow-py` (`MatrixDiffusion3D` pyclass), and `semiflow-wasm`
  (`MatrixDiffusion3D` JS class, `--features full`).  Buffer layout:
  `2*nx*ny*nz` f64, index `(k*nx*ny+j*nx+i)*2+component`.  PEP 561 `.pyi` stub added.

### Intentionally skipped (bind-remaining-operators wave)

The following 5 candidates were classified SKIP after analysis:
- `AnisotropicShiftAdaptiveQ` / `AnisotropicShiftZeta2ND` — internal variant
  types; public surface is `AnisotropicShiftND2` / `AnisotropicShiftND3`.
- `QuantumSchrödingerChernoff` — internal builder pattern; public surface is
  `Schrodinger1D` / `SchrodingerComplex1D`.
- `TruncatedExp4WithCache` — internal optimisation shim; public surface is
  `TruncatedExp4th1D`.
- `IdentityND` — utility type not intended for direct user construction.

### Known gaps (documented, not silently omitted)

`ObstacleND`, `ObstacleGamma`, `GraphTraj`, Laplacian introspection, and
`GraphAdjoint` dense read-back remain PyO3-only deferrals (closures and
dense-matrix read-back are not expressible in a stable C / WASM ABI without
additional design work).

Cross-refs: ADR-0028 (binding split), ADR-0171 (S³ carrier C-ABI contract).

## [0.9.0-beta] — 2026-06-19

First public release of **SemiFlow** — a Rust library that solves linear
evolution equations `∂ₜu = Lu` by Chernoff approximation of operator semigroups
(Theorem 6 of Remizov 2025, *Vladikavkaz Math. J.* 27(4), 124–135). The library
was developed privately through extensive internal iteration and is published as a
`0.x` beta for community testing ahead of a stable `1.0`.

### Features

- Matrix-free semigroup evolution: `(S(t/n))ⁿ → e^{tL}`, no matrix exponentials
  or linear solves; allocation-free steady state; `no_std + alloc` core.
- Diffusion / advection–reaction kernels in 1D/2D/3D with variable coefficients
  (`ShiftChernoff1D`, `DiffusionChernoff`, `DriftReactionChernoff`, Strang
  tensor-product splitting).
- Higher-order accuracy via the ζ-ladder (`Diffusion4thZeta4Chernoff`,
  `Diffusion6thZeta6Chernoff`, `Diffusion8thZeta8Chernoff`).
- Schrödinger (`SchrödingerChernoffComplex`), manifold (`ManifoldChernoff` over
  torus / sphere / hyperbolic / Fubini–Study), hypoelliptic / sub-Riemannian
  (`HypoellipticChernoff`), and graph (`GraphHeatChernoff`,
  `QuantumGraphHeatChernoff`) operators.
- Boundary conditions: Dirichlet / Neumann / Robin / obstacle
  (`BoundaryPolicy`, `KillingChernoff`, `ReflectedHeatChernoff`,
  `ObstacleChernoff`).
- Resolvent and nonautonomous evolution (`LaplaceChernoffResolvent`,
  `HowlandLift`, `ResolventJumpChernoff`).
- Forward-mode automatic differentiation for sensitivities via `Dual<F>`.
- Generic over the scalar type (`SemiflowFloat`: `f64` / `f32` / `Dual`),
  optional `simd` (AVX2/NEON) and `parallel` features.
- Bindings: C (`semiflow-ffi`, header `semiflow.h`), Python
  (`semiflow-pde` on PyPI, `import semiflow`), and WebAssembly (`semiflow` on npm).

Every numerical claim is gated in CI against closed-form or high-order reference
oracles. As a `0.x` beta, minor releases may include breaking API changes.
