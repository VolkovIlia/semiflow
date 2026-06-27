# Changelog

All notable changes to SemiFlow are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.10.2-beta] — 2026-06-27

### Fixed
- **Lanczos OOB panic at t≥4** (graph_krylov): off-by-one wrote past
  MAX_LANCZOS_DIM=18 when the Krylov dimension maxed out at large t·λ_max
  (present since A1/ADR-0185, only triggered at deep t). Lanczos now matches
  Chebyshev/dense expm at t∈{4,8,16} (max|Δ|≤1e-9, no panic). Gate
  lanczos_large_t_no_oob. The `--no-lanczos` workaround is no longer needed.
- Clippy `cast_precision_loss` in the graph_par_speedup example (`--all-targets`).

## [0.10.1-beta] — 2026-06-27

### Performance
- **Channel-parallel graph evolve** (ADR-0184 D3): `evolve_batched` and the batched
  adjoint now split channels across cores via `std::thread::scope` (gated behind
  `parallel`, C≥2), each worker with its own `ScratchPool`. The graph Krylov ML path
  was previously single-threaded (the `parallel` feature only covered Strang2D/3D grid
  kernels). Bit-identical to serial (0-ULP; forward + ascending-index gradient
  reduction per ADR-0184 D4/D5). Measured ~5–6× on i7-12700K (612–1942% CPU). Closes
  the "speed" half of the memory↔speed contradiction for graph diffusion / SSM layers.

## [0.10.0-beta] — 2026-06-27

Five-feature wave (issues #11, #12, #13, #14 + A1 stiff fix): generic symmetric
operators, conservative divergence-form diffusion, stiff multilayer conduction,
ETD/φ semilinear integrator, and depth-independent Krylov + Fréchet for graph
semigroups. All additive; no breaking changes to the 0.9.1-beta public surface.

### Added

- **A1/A2 — Depth-independent graph-semigroup Krylov action + edge-weight Fréchet
  gradient** (`GraphKrylovChernoff`, `graph_expmv_frechet`; ADR-0185, math §54):
  `graph_expmv(L, v, t, ε)` computes `e^{−tL_G}·v` without stepping through each
  time unit — Chebyshev default (O(1) working vectors, Bessel-coefficient degree
  set by `t‖L‖` and `ε`) or Lanczos adaptive (O(m·N) basis, m ≈ 20–40, flat in `t`).
  `graph_expmv_frechet` returns `∂J/∂w` for all edge weights via one augmented
  Krylov solve (Al-Mohy & Higham 2009), closing the forward and backward speed
  ceilings exposed by v0.9.1-beta (#10). PyO3: `GraphKrylov` pyclass +
  `graph_expmv_frechet` pyfunction in one GIL-releasing call.
  Gates: `G_GRAPH_EXPMV_DENSE` ≤ 1e-10 vs dense `mat_exp_pade13`;
  `G_GRAPH_FRECHET_FD` rel-err ≤ 1e-7 (§43.6 FD oracle);
  `G_GRAPH_EXPMV_DEPTH_FLAT` matvec count flat in `t` at fixed `ε`.
  Honest limits: symmetric `L_G` only (Arnoldi for directed graphs deferred);
  time-varying `L(t)` is out of scope (use Magnus/Howland).

- **#13 — Generic externally-assembled symmetric-operator entry point**
  (`SymmetricOperator`, `MassKOperator`, `EntrySensitivity`; ADR-0186, math §55):
  `SymmetricOperator::from_csr` accepts any externally-assembled symmetric PSD
  sparse operator (FEM stiffness with Robin BC, anisotropic conductivity — not
  just zero-row-sum graph Laplacians). `MassKOperator` propagates
  `e^{−τM⁻¹K}·v` for the generalized eigenproblem `(M,K)` via the congruence
  chain `Â = R^{−T}KR^{−1}` (Cholesky factor `M = RᵀR`) — never forms `M⁻¹K`
  explicitly, stays sparse. `EntrySensitivity` provides per-entry Fréchet gradient
  for any `SymmetricOperator`. The lumped `(M,K)` path
  (`mass_lumped_evolve(K, diag(m), …)`) is a 3-liner.
  Gates: `G_SYMOP_DENSE`, `G_MASSK_LUMPED`, `G_MASSK_CONSISTENT` (all ≤ 1e-10
  vs `mat_exp_pade13`); `G_SYMOP_ENTRY_FRECHET` rel-err ≤ 1e-7.
  Honest limits: symmetric PSD required; large consistent-mass sparse Cholesky is
  caller-supplied (in-crate dense Cholesky for moderate `n`); consistent-mass
  differentiability deferred.

- **#11 — Conservative (divergence-form) variable-coefficient diffusion**
  (`ConservativeDiffusionChernoff`, `assemble_conservative_csr_1d`; ADR-0187,
  math §56): harmonic-mean face conductivities `k_{i+½} = 2k_ik_{i+1}/(k_i+k_{i+1})`
  (Patankar 1980) reproduce the series-resistance network at machine precision
  across sharp material interfaces (k-contrast 100:1 to 3025:1) where the
  non-conservative pointwise expansion gives ≥50% error.
  `assemble_conservative_csr_1d` builds `A = −L_k` (symmetric PSD CSR) directly
  consumable by §55 `SymmetricOperator::from_csr`; the Krylov action then gives
  an exact, unconditionally-stable propagator. Optional per-face contact resistance
  `R_c`. Separable N-D assembler (`assemble_conservative_csr_nd`, 5-pt/7-pt CSR).
  `ConservativeDiffusionChernoff` wraps a single-step Crank–Nicolson (`order()=2`,
  `Growth::contraction()`, one O(n) Thomas solve; A-stable, no CFL). Bridges to
  §54/§55 via `to_symmetric_operator()` for stiff stacks.
  Gates: `G_CONS_SERIES` per-layer ΔT ≤ 1e-2 AND face-flux ≤ 1e-2;
  `G_CONS_NONCONS_FAILS` (teeth — old kernel fails same test by ≥50%);
  `G_CONS_SYMOP` ≤ 1e-10; `G_CONS_ORDER` slope ≤ −1.95; `G_CONS_CONTACT` ≤ 1e-2.
  Honest limits: `k > 0`; symmetric NSD by construction; full-tensor non-separable
  `∂_x(k∂_y)` is out of scope.

- **#14 — Stiff multilayer conduction via mass-weighted Krylov**
  (`MultilayerStack`, `multilayer_evolve`, `MassWeightedConservativeChernoff`;
  ADR-0188, math §57): `MultilayerStack::from_layers` maps physical
  `[(thickness, k, ρc)]` → node arrays on a single uniform grid.
  `multilayer_evolve` propagates `e^{−τdiag(ρc)⁻¹A}·u₀` in **one** depth-flat
  Krylov action (via §55 `mass_lumped_evolve`) for any integration span —
  beats explicit CFL by ~28000× on the Shuttle LI-900/SIP/RTV/Al-2024 TPS stack
  (k-contrast ≈ 3025×, ρc-contrast ≈ 27×, λ_max(M⁻¹A) ≈ 794 s⁻¹,
  explicit needs ~1.98 M steps, Lanczos needs ~1409 matvecs).
  `MassWeightedConservativeChernoff` provides an optional A-stable CN convenience
  (O(n) Thomas, one step at a time) for O(1)-vector memory.
  Gates: `G_TPS_MASS_WEIGHT` ≤ 1e-10; `G_TPS_UNITMASS_FAILS` (teeth, ≥50% miss);
  `G_TPS_STACK_ACCEPTANCE` per-probe ≤ 2e-2 on the real TPS stack at
  t ∈ {500, 1500, 2500} s; `G_TPS_STIFF_STEPCOUNT` X/Y ≥ 100 (measured ~28000).
  Honest limits: 1-D first; one global `dx`; node-centered lumped mass only.

- **#12 — ETD φ-functions and ETDRK4 semilinear integrator**
  (`phi_action`, `phi_action_batched`, `Etdrk4`, `Nonlinearity`; ADR-0189,
  math §58): `phi_action_batched` computes φ₀…φ_p(τA)·v simultaneously via ONE
  augmented block-triangular matvec-only Taylor action (Al-Mohy & Higham 2011 §4,
  Sidje 1998) — no Padé on φ, no contour integrals, no new coefficient math;
  reuses the `expmv` `THETA_M` substepping table and `select_s_m`.
  `Etdrk4` (Cox–Matthews 2002 / Kassam–Trefethen 2005) integrates
  `∂ₜu = Lu + N(u)` at order 4 without re-discretizing or splitting `L`.
  `N(u)` is a declarative `Nonlinearity` trait (native opcode interpreter or a
  fixed enum menu `AllenCahn` / `Burgers` / `GrayScott` / `KuramotoSivashinsky`
  at the PyO3 surface — never a per-step Python callback; ADR-0179 wall
  preserved). `NonlinearityDiff` enables end-to-end adjoint `∂J/∂param` through
  one step. One unified augmented path serves 1-D and graph operators.
  Gates: `G_PHI_AUG_DENSE` ≤ 1e-10 (z ∈ [0.5, 5] non-trivial band);
  `G_ETDRK4_ORDER` slope ∈ [3.7, 4.3] (two-sided, Allen–Cahn 1-D);
  `G_ETD_ADJOINT_FD` ≤ 1e-6 (N-flowing param, §43.6 discipline).
  Honest limits: `L` 1-D divergence-form or symmetric graph only; non-symmetric
  graphs, 2-D/3-D tensor ETD, exponential Rosenbrock, and arbitrary per-step
  Python `N` are deferred.

### Fixed

- **A1 stiff operator NaN bug** (commit `9e5f557`):
  `GraphKrylovChernoff`'s Chebyshev path silently returned garbage or false-zero
  for stiff operators (`λ_max ≳ 1400`) because `z = τλ_max/2` exceeded the f64
  range, causing the pattern `0·∞ = NaN` absorbed by `f64::max` — no error
  surfaced. Fix: substep `s = ⌈z/Z_SAFE⌉` (Z_SAFE = 200), mirroring the
  Lanczos path's scaling, so each substep's `z` stays representable; plus a
  fail-loud finiteness guard returning `SemiflowError::DomainViolation`. Normalized
  graph Laplacians (`λ_max ≲ 2`) were never affected; conservative high-contrast
  diffusion (`λ_max ∼ 10⁴`) was. New test `cheb_stiff_regime` (z ≈ 6400) was red
  before, green after (sup_error 1.0e-11).

### Honest limitations (campaign-wide)

- Chebyshev stiff-substepping matvec count grows as `O(τλ_max)` when substepping
  kicks in; Lanczos is recommended for operators with large `τλ_max` (stiff stacks).
- `(M, K)` consistent-mass differentiability via `MassKOperator` is deferred; the
  entry-Fréchet covers the direct `SymmetricOperator` path only.
- ETD arbitrary per-step Python/JS `N(u)` is an explicit non-goal (ADR-0179
  wall; per-step GIL crossing reintroduces the 200× / GIL-defeat hazards).
- Non-symmetric / directed graphs, 2-D/3-D tensor ETD, and exponential Rosenbrock
  are all deferred to later issues.

---

## [0.9.1-beta] — 2026-06-26

First 0.9.1 series beta — adds the batched multi-channel graph evolve API (#10) and ships the Python wheel with SIMD+parallel.

### Added

- **Batched multi-channel evolve** (`evolve_batched`) for the graph-heat kernels
  (`GraphHeat`, `GraphHeat4th`, `GraphHeat6`, `MagnusGraphHeat`, `MagnusGraphHeat6`,
  `VarCoefGraphHeat`, `VarCoefMagnusGraph`) and batched adjoint paths
  (`evolve_state_adjoint_batched` on `GraphAdjointPresampled`, `edge_weight_grad_batched`).
  Evolves an `[N, C]` feature matrix in ONE Rust call / one GIL release (ADR-0184).
  Bit-exact (0-ULP) identical to the per-channel loop.

### Changed

- **Python wheel now compiled with `simd` + `parallel`** (previously built with
  `default-features=false, features=["std"]` — scalar-only).

### Performance (measured, i7-12700K)

- Forward batched (C=4): **2.1×–5.6× faster** than the per-channel Python loop
  (`MagnusGraphHeat` 540 µs → 96 µs via Laplacian hoisting + single PyO3 call).
- Peak Python memory (C=4): **~23× lower**.
- Adjoint state-sweep (C=4): **1.0×–1.8×** faster.
- **Honest caveat — edge-weight gradient path**: `edge_weight_grad_batched` shows
  **~1.0× (no speedup)**. This path is Rust-compute-bound — O(edges) sensitivity
  per channel — not Python-overhead-bound; batching gives correctness + fewer PyO3
  calls, not a throughput gain. No blanket gradient speedup is claimed.

## [0.9.0-beta.3] — 2026-06-25

### Fixed
- **@semiflow/wasm npm description**: corrected the stale package description and
  README (the old text referenced the pre-release crate name `semiflow-core`, the
  private repo's internal `v9.0.0` history, and falsely claimed TT/gridless are not
  exposed via WASM — they are: `TtEvolver`/`TtState`/`GridlessEvolver` etc. are
  registered in `lib.rs`). npm 0.9.0-beta.2 shipped immutably with the old text;
  this version corrects it.

### Note
- Documentation-only release — no library code change from 0.9.0-beta.2.

## [0.9.0-beta.2] — 2026-06-24

- Add PyPI long description: `pyproject.toml` now includes `readme = "README.md"` so
  the `semiflow-pde` PyPI page renders the full README instead of "no description".
- Refresh `crates/semiflow-py/README.md` after the `semiflow-core → semiflow` crate
  rename: stale `semiflow-core` references replaced with `semiflow` throughout.
- **Suckless compliance**: 24 over-budget functions/files reduced to ≤50 lines/function
  and ≤500 lines/file via additive extraction into helper functions and sibling
  `*_tests_mod.rs` include files (no public API changes, no symbol renames).
- **Stable rustfmt** (ADR-0182): removed nightly-only `imports_granularity` /
  `group_imports` from `rustfmt.toml`; CI `fmt` job now runs on stable toolchain.
- **Honest WASM Greeks parity gate** (ADR-0183): `G_BINDING_GREEKS_PARITY`
  sub-test 4 (WASM) previously asserted 0-ULP byte-equality against a "golden"
  that had been regenerated from the WASM binary's own output — a vacuous gate
  (WASM == WASM) masking a real native↔wasm32 divergence. The golden is now the
  legitimate SCALAR core hyper-dual sweep (Richardson-FD-verified, oracle
  independent of the WASM SUT), and the WASM criterion is a ≤ 1e-9 per-array
  relative-error tolerance. Root cause: native↔wasm32 libm `exp()` differs in
  the last ULP and amplifies over 32 Chernoff steps + the hyper-dual chain rule
  to ≤ 6.1e-11 relative; 0-ULP is physically unreachable (the hyper-dual path is
  not SIMD, so a scalar golden does not close the gap). FFI/PyO3 sub-tests stay
  0-ULP (native, shared libm).

## [0.9.0-beta.1] — 2026-06-24

- **Completes `semiflow-core → semiflow` rename**: the initial `0.9.0-beta`
  published the library under the new `semiflow` crate name but shipped with
  stale doctests and binding-crate re-exports that still referenced the old
  `semiflow-core` path, causing `cargo test --doc` failures and `ImportError`
  on `from semiflow import TtState`.  This patch fixes all affected doctests,
  re-exports, and smoke tests across `semiflow-ffi`, `semiflow-py`, and
  `semiflow-wasm` so the published crate is self-consistent.

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

### Added — C/WASM parity (close-c-wasm-parity wave)

- **`SmfLaplacian`** opaque type: `smf_graph_laplacian_combinatorial` /
  `smf_graph_laplacian_normalized`, introspection (`n_nodes`, `is_combinatorial`,
  `is_normalized`, `spectral_bound`), CSR getters (`row_ptr`, `col_idx`, `vals`),
  and dense read-back `smf_laplacian_to_dense` (n×n row-major).  WASM `Laplacian`
  class exposes the same surface.
- **`SmfGraphTraj`** (degenerate fixed-topology): `smf_graph_traj_new` + getters
  `n_nodes`, `n_segments`, `t_horizon`.  WASM `GraphTraj` class.
- **`SmfObstacleGamma`**: `new_const` / `new_array` + `size` +
  `inactive_gamma` (dense `(gamma, defined, count)` read-back).  WASM
  `ObstacleGammaV8` class.
- **`SmfObstacleND2`** (D=2): `new` + `shape` + `apply` (flat buffer in/out).
  WASM `ObstacleND2` class.
  Together these close all prior PyO3-only deferrals for these four types.
  46 `semiflow-ffi` tests pass; check-unsafe-scope PASS; header regenerated.
  Cross-refs: ADR-0028, ADR-0171, ADR-0179.

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

### Added — pre-sampled graph state-adjoint (ADR-0180)

- **`PreSampledLaplacianSeq<F>`** (`semiflow-core`): holds the pre-sampled CSR
  Laplacian weight sequence (`row_ptr`, `col_idx`, `vals_seq`) consumed at
  construction; `vals_seq.len() == 2 * n_steps * nnz` enforced — the factor-of-2
  reflects GL₄ Magnus K=4 sampling at both abscissae (`c₁ = (3−√3)/6`,
  `c₂ = (3+√3)/6`) per step.  One-value-per-step layout is SILENTLY WRONG at
  O(τ²) and is rejected at construction.
- **`fill_abscissa_times(t_horizon, n_steps, out)`**: fills a `2*n_steps` slice
  with the GL₄ abscissa sample times in adjoint-schedule order
  `[(step k, c₁), (step k, c₂)]` where adjoint `t_start = (n_steps−1−k)·τ`.
  Exposed on all four surfaces so callers supply exactly the right times.
- **`MagnusGraphHeatChernoff::from_presampled`** / **`PreSampledMagnusAdj<F>`**:
  pre-sampled Magnus K=4 graph state-adjoint; `evolve_state_adjoint_into` takes
  the pre-built sequence and runs the backward costate sweep without any runtime
  callback.
- **`VarCoefMagnusGraphHeatChernoff::from_presampled`** /
  **`PreSampledVarCoefAdj<F>`**: variable-coefficient variant; additionally
  accepts `a_seq` (2·n_steps scalar diffusion weights) and `a_sup_max`.
- **RELEASE_BLOCKING gate `G_GRAPH_ADJOINT_SAMPLED_PARITY`**: closure path vs
  pre-sampled path must be bit-exact (0 ULP).  2 tests PASS.
- **FFI** (`semiflow-ffi`): new `SmfGraphAdjoint` opaque type with 6 functions —
  `smf_graph_adjoint_abscissa_times`, `smf_graph_adjoint_new_presampled`,
  `smf_graph_adjoint_new_presampled_varcoef`,
  `smf_graph_adjoint_evolve_state_adjoint`, `smf_graph_adjoint_n_nodes`,
  `smf_graph_adjoint_free`.  `tau` is captured at construction
  (`t_horizon / n_steps`); `evolve` validates `n_steps` matches or returns
  `OutOfDomain`.  C header regenerated; check-unsafe-scope PASS.
- **PyO3** (`semiflow-py`): new `GraphAdjointPresampled` pyclass.  GIL policy
  (ADR-0031): `lap_at_t` callback sampled once under GIL at construction;
  `evolve_state_adjoint` runs fully in `py.detach` with no Python reattachment
  per step.  Registered alongside the existing `GraphAdjoint` class (additive).
- **WASM** (`semiflow-wasm`, `--features full`): new `GraphAdjointPresampled`
  JS class with `abscissaTimes` (static), `fromPresampled`, `evolveStateAdjoint`,
  `nNodes`, `nSteps`.  Magnus K=4 only (VarCoef deferred to a future WASM wave).
  All code is `#[cfg(feature = "full")]`-gated.

### Known gaps (documented, not silently omitted)

`ObstacleND`, `ObstacleGamma`, `GraphTraj`, and Laplacian introspection
(including dense `to_dense` read-back) are now fully exposed across FFI and
WASM — see "Added — C/WASM parity" below.

The sole remaining PyO3-only deferral is **`GraphAdjoint`'s constructor**:
its `lap_at_t` (time-dependent Laplacian) and optional weight callbacks are
Rust/Python closures that cannot cross a stable C/WASM ABI (ADR-0179).  The
`evolve_state_adjoint` method is ABI-shaped (dense vector in/out); only the
closure-accepting constructor is blocked.  Workaround: use the pre-sampled
array path; a batched-sampler API is specced for a future minor.

Cross-refs: ADR-0028 (binding split), ADR-0171 (S³ carrier C-ABI contract),
ADR-0179 (GraphAdjoint closure deferral).

### Added — production rough-Heston pricer (issue #9, ADR-0181)

- **Risk-neutral discounting** (`semiflow-core`, `examples/rough_heston_pricer.rs`):
  `c_00 = −r` in the reaction matrix — the block-CN Strang half-steps
  `exp(τC/2)` compound to `e^{−rT}` over `n = T/τ` steps via the
  Feynman-Kac equation `∂_τ u = Lu − ru` (math.md §33.9, ADR-0181 §D1).
  No post-evolution multiply by `e^{−rT}` — discount rides the existing
  matrix-exp machinery.

- **`--price` mode** (`examples/rough_heston_pricer.rs`): builds the call-payoff
  initial condition, evolves `n = T/τ` backward steps, reads component-0 at
  `x = 0`, and prints discounted call prices at `K ∈ {90, 100, 110}`.
  `--rate 0.0` recovers the pre-issue-#9 demonstrator output (regression guard).

- **RELEASE_BLOCKING gate `G_ROUGH_HESTON_MC_PARITY`**
  (`tests/rough_heston_mc_oracle.rs`, slow-tests):
  Gate I of the two-tier honesty design. Asserts that the Chernoff kernel
  (accuracy grid N=192, τ=0.01) agrees with a QE-CIR Monte-Carlo of the
  SAME linearised/frozen-V₀ 4-factor Markov model — zero model bias enters,
  so this is a pure numerical gate. Tolerance: 3·MC_stderr + δ_kernel
  (δ_kernel ≤ 0.55 price units ≈ 0.6% ATM, measured by N=48 vs N=192
  self-convergence). MC: 1M antithetic paths, n_steps=200, QE-CIR factors
  (Andersen 2008), seed PCG64(lower-64 of 0xC0FFEE_BABE_DEAD_BEEF).
  Three strikes K ∈ {90, 100, 110}, T=1, H=0.1, S₀=100, V₀=0.04, κ=1.5,
  θ=0.04, ξ=0.3, ρ=−0.7, r=0.05.

- **Discount sub-test** (`tests/rough_heston_mc_oracle.rs`): flat IC u₀≡1,
  coupling zeroed, c_00 = −r → component-0 ≈ e^{−rT} to ≤1e-6 (validates
  the discounting mechanism independently of diffusion/coupling).

- **ADVISORY record `A_ROUGH_HESTON_MODEL_BIAS`**
  (`tests/rough_heston_model_bias.rs`, slow-tests):
  Gate II of the two-tier design. Measures and reports three model-approximation
  sub-biases (frozen-V₀ vs stochastic √V_t, reaction coupling vs exact cross-term,
  3-factor GL vs N→∞ Markov). Expected aggregate O(H) ≈ 1–5% at H=0.1. Never
  fails. Reports one JSONL line per sub-bias to stdout.

- **Math §33.9** (`contracts/semiflow-core.math.md`): discounting formula and
  the two-error-source decomposition (gate I / gate II). Cites Andersen 2008
  (QE-CIR), El Euch–Rosenbaum 2019 (multifactor convergence), Carr-Cisek-Pintar
  2021 (GL 3-factor model).

- **`contracts/semiflow-core.properties.yaml`** bumped to schema_version 4.16.0:
  adds `G_ROUGH_HESTON_MC_PARITY` property, new `advisory_records:` section with
  `A_ROUGH_HESTON_MODEL_BIAS`, and `notes:` entry documenting the two-tier design.

**Honest claim**: oracle-validated solver of a documented 4-factor Markov model
(~0.6% numerical precision); itself O(H)-biased ~1–5% vs true rough-Heston at H=0.1.

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
