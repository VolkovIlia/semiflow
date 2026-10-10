# Changelog

All notable changes to SemiFlow are documented here.
Format: [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).
Versioning: [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Performance overhaul and correctness audit (2026-10)

The library was slow for algorithmic reasons, not for lack of micro-tuning.
This section records the fixes and speedups in order; every existing gate
threshold is unchanged or tightened, and every speedup that is claimed
bit-identical is proven so by a 0-ULP gate and the unchanged `no_std` digests.

#### Added — semilinear systems and extrapolation

- **Reaction–diffusion systems on every engine (ADR-0208, math §64.1).**
  `ReactionDiffusion` evolves `∂ₜu_k = L_k u_k + f_k(t, x, u)`, `K` species,
  each diffusing with its own engine (any `ChernoffFunction` on a `GridFn1D`,
  `GridFn2D`, `GridFn3D` or `GraphSignal`: any boundary policy, interpolant,
  Strang/ADI composition, graph Laplacian), Strang-split with an exact or
  RK4 (`with_reaction_substeps`) reaction flow; time-dependent kinetics and
  source terms. `Kinetics` evaluates all nodes per call (one host callback
  per RK stage). Catalogue: `FisherKpp`, `AllenCahnReaction`, `Nagumo`,
  `GrayScott`, `FitzHughNagumo`, `Brusselator`, `LinearReaction`,
  `FnReaction`; `Species<S>`, `NodalField`. Gates `G_SEMILIN_*` (exact
  fronts of Fisher–KPP and Nagumo, MMS, commuting linear system, Gray–Scott
  2-D).
- **Richardson extrapolation for any engine (ADR-0207, math §64.2).**
  `Evolver::evolve_extrapolated_into(t, levels, …)`, `extrapolate_into` for any
  `StepIntegrator` (`Evolver`, `ReactionDiffusion`), `richardson_weights`.
  Order `p` → `p + levels − 1`; `ShiftChernoff1D` at 96 steps: 3.3e-4 → 7.2e-8.
  Gates `G_RICHARDSON_*`.

#### Python

- **Python 3.15** (ADR-0209): `requires-python = ">=3.10,<3.16"`, classifiers
  3.14/3.15, CI `py-smoke` on 3.10/3.13/3.15; PyO3 and rust-numpy 0.28 → 0.29.
  The abi3-py310 wheel serves 3.10–3.15; the suite passes on 3.13.16 and
  3.15.0b4.
- **`ReactionDiffusion1D/2D/3D`**: `K` species with per-species diffusivity,
  built-in kinetics by name (`fisher_kpp`, `allen_cahn`, `nagumo`,
  `gray_scott`, `fitzhugh_nagumo`, `brusselator`, `linear`, with `params=`) or
  any NumPy-vectorised callable `f(t, x, u)` called once per RK stage (not per
  node); `evolve(t, n_steps, richardson=L)`, `values()` `(K, *grid)`, `time`,
  `species`; GIL released; callback exceptions propagate unchanged.
  `semiflow.richardson_weights`; `Heat1D.evolve(..., richardson=L)`. Stubs and
  README updated.

#### Changed (behaviour)

- Chebyshev, Lanczos and general Taylor actions (and everything built on them:
  `GraphKrylovChernoff`, Fréchet gradients, `CsrExpmvChernoff`, ETDRK4) return
  different bits — within their tolerance, and more accurately — see ADR-0205.
- `KrylovPath::Lanczos { m_max }` with an `m_max` that cannot reach `tol` at
  any substep count (`m_max = 1`) returns `DomainViolation` instead of an
  under-resolved result.
- `FrechetPlan` (unreleased) gains `chain_weight` and is `#[non_exhaustive]`.

#### Fixed

- **Issue #44 — a non-finite operator norm bound is an error, never a silent
  schedule.** Finite entries can overflow the Gershgorin / row-sum bound (2×2
  all-`1e308`). Such operators stay legal — with a diagonal mass `M⁻¹A` has a
  finite norm — but every consumer of the bound now returns `DomainViolation`:
  `CsrGenerator` without mass, `GraphKrylovChernoff::new`,
  `graph_expmv_krylov` (also for an external `SymmetricLinearOp` returning
  `∞`), the Taylor `expmv` kernels (`select_s_m` used to skip every θ row and
  return ONE substep for `∞`, zero substeps for NaN, and saturate `s` at
  `u32::MAX`) and the φ entry points (which also reject `τ < 0` / NaN / `∞`).
  Non-finite entries are rejected in `Laplacian::from_csr_parts`,
  `lumped_congruence` (no longer `.expect`s) and `Graph::from_edges`
  (overflowing weighted degree). `expmv_cost_probe` / `phi_cost_probe` return
  `(0, 0)` for such inputs. `SymNormalized` assembly no longer turns an
  overflowing `dᵢdⱼ` into a zero entry. Test helpers that folded with
  `f64::max` (which drops NaN) now assert finiteness. Gate
  `G_NONFINITE_BOUND_REJECT`.
- **`MatrixDiffusionChernoff` exponentiated the wrong matrices.** `M = 2`: a
  negative discriminant (complex eigenvalues, e.g. any rotation-like coupling)
  took `cosh/sinh` instead of `cos/sin`; `M = 3, 4`: the Taylor helper summed
  `Bᵈ/d` instead of `Bᵈ/d!` (`e^{0.5}` came out as `1.6931`);
  `MatrixDiffusionChernoffComplex`, `M ≤ 4`: scaled by the largest entry rather
  than the row sum, truncation up to `M¹³/13!`. Now a stable closed form for
  `M = 2` and Padé[13/13] for every other size. Gate `G_MATRIX_EXP_SMALL_M`
  (old errors: 5e-3, 6e-4, 9e-5, 2e-6).
- **`MassKOperator` under-estimated `λ_max(M⁻¹K)`**, so its Chebyshev series
  diverged on P1 consistent mass (relative error `1.7e11` at `n = 12`): the
  5-step inverse-power estimate of `λ_min(M)` started from an eigenvector of
  `M`'s largest eigenvalue. Replaced by a rigorous `O(n²)` bound
  `ρ̄(K)·‖R⁻¹‖₁‖R⁻¹‖_∞` (comparison matrix of the Cholesky factor). Gate
  `G_MASSK_RIGOROUS_BOUND`; `G_MASSK_CONSISTENT` tightened `1e-8 → 1e-11`
  (measured 3.5e-10 → 2.2e-13).
- **Lanczos `m_max` cap** kept the substep count of the uncapped degree
  (silently under-resolved). Gate `G_LANCZOS_M_MAX_CAP` (now against the new
  a-priori bounds, recomputed independently).
- **`DriftReactionChernoff::apply_adjoint_into`** dropped the divergence term:
  the adjoint of `b∂ₓ + c` is `−b∂ₓ + (c − b′)`. Exact before only for
  constant `b` (still bit-identical there). Gate `G16C_ADJOINT_VARIABLE_DRIFT`
  (dual-pairing defect 1.3e-3 → 2.7e-6).
- `Jacobi::build` read the diagonal through `n` unit-vector mat-vecs
  (`O(n·nnz)`, its doc said `O(nnz)`); new provided method
  `SymmetricLinearOp::diagonal_into`, overridden by the CSR operators.
  Bit-identical.
- `Diffusion4thChernoff` documented "Chebyshev wins over octonic" while the
  code (and `Diffusion8thZeta8Chernoff::with_octonic_sampling`) relies on
  octonic winning; the docs now say what the code does.

#### Performance (bit-identical)

- **Prepared sampling (ADR-0204).** Within one Chernoff step every sample reads
  the same state, but `GridFn1D::sample` rebuilt the interpolant's ghost data
  per sample: 44 `bc_value` calls for the default septic sampler, all `M + 1`
  virtual nodes for the Chebyshev sampler. The grid engines now build that
  data once per step (`sample_table`) and evaluate every sample from it with
  the same functions in the same order. The ζ-corrections of
  `Diffusion4th`/`6th` sample their shared stencil points once instead of three
  times, and hoist the per-node `libm::pow` calls. Bit-identical (gate
  `G_PLAN_BIT_EQUAL`, all `no_std` digests unchanged). Measured (this
  container, best of 5): `ShiftChernoff1D` 17.9 → 6.3 ms, `DiffusionChernoff`
  const-a 29.2 → 9.4 ms, variable-a 73.2 → 27.9 ms, `Diffusion6th` 21.9 → 6.8 ms,
  `Diffusion4th` + Chebyshev sampling 104 → 2.7 ms, `Strang2D` 400² 3.30 → 1.14 s,
  `Strang3D` 64³ 8.2 → 3.8 s.

#### Performance (new algorithms; output bits change, ADR-0205)

- **Chebyshev Krylov path: one expansion, cost `∝ √(λt)` instead of `∝ λt`.**
  The coefficients `e^{−z}Iₖ(z)` were a power series for `Iₖ` times `e^{−z}`,
  which overflows near `z ≈ 700`, so every action was split into `⌈z/200⌉`
  substeps of degree ≈ 101, each with the FULL tolerance (error grew with the
  substep count). Now exponentially scaled Bessel values by Miller's backward
  recurrence, ONE expansion of degree `≤ ⌈L/3 + √(L²/9 + 2zL)⌉ − 1`
  (`L = ln(8/tol)`, Bennett bound, gate `G_CHEB_SQRT_COST`). Issue #16 operator
  (`N = 400`, `λt ≈ 4e7`): 23.7 s → 42 ms (559×), error 6.8e-6 → 1.4e-11 at
  `tol = 1e-10` (gate `G_CHEB_STIFF_ORACLE`, `≤ tol + 1e-11` up to `λt = 1e8`).
  `λ_max = 0` / `τ = 0` no longer divide by zero.
- **Lanczos path: a-priori schedule from Lanczos theory.** Substeps came from
  the Taylor radii `θ_m` (`λh ≤ 1.09` at `m = 18`) plus a dense Padé-13 per
  substep; now the Hochbruck–Lubich and Chebyshev-interpolation (`4xᵐ/m!`)
  bounds and an `O(m²)` tridiagonal eigen-solve. 3.8–5.1× faster (gate
  `G_LANCZOS_HL_COST`); `m_max` is no longer tied to 18; an unschedulable
  `m_max` (1) is an error.
- **ETDRK4: four φ sweeps per step instead of nine.** Each stage is one
  `phi_combination` (`Σ τᵏφₖ(τL)wₖ`) sweep. Exactly `3·C(h/2) + C(h)`
  generator applications per step, 2.25–2.6× fewer (gate `G_ETDRK4_SWEEPS`);
  `G_ETDRK4_ORDER` and `G_ETD_AFFINE_EXACT` unchanged.
- **`GeneralOperator` Taylor `expmv`: trace shift.** `e^{−τA} = e^{−τμ}e^{−τ(A−μI)}`,
  `μ = tr(A)/n` when it lowers `‖·‖_∞` (Al-Mohy & Higham 2011, §3.1): half the
  mat-vecs on diffusion-dominated stencils (gate `G_EXPMV_SHIFT_COST`); new
  accessor `GeneralOperator::taylor_norm_bound`. `CsrExpmvChernoff::apply_into`
  no longer allocates (scratch pool).
- **Fréchet gradients** inherit the Chebyshev speedup in every propagator call.
  `FrechetPlan` gains `chain_weight` and the §63.7.a bound uses it (rounding
  `Σ(zᵢ/2 + mᵢ)` along a chain, Skellam variance identity) instead of
  `N_chain·m_max²`; the bound is 1.03–25× TIGHTER at every gate point
  (Amendment 4). The SpMV bound `B(ρ̄t)` is re-derived (`√(6L·N·ρ̄t) + N·2L/3`,
  never above the old one, asserted).
- `G_TPS_STIFF_STEPCOUNT` now counts what ADR-0188 defined (total mat-vecs,
  not the per-substep degree; the old kernel had `X/Y = 3.96` under that
  definition): `X/Y ≥ 300` (was 100), `Y ≤` the Bennett bound (was `2√X`,
  below the minimax degree at `tol = 1e-12`). Measured 7.35e5 → 8829 mat-vecs.
- Re-recorded once: `no_std` digests `graph_krylov_chebyshev`,
  `graph_frechet_large_t` (identical in `no_std`, `std-ref`, AVX2).

ADR-0202: an `O(N)` SPD resolvent / steady solve, operator composition, and a
φ-combination that closes the affine and ETD gaps. All additive except one
behaviour change in `phi_action` (see Changed). No ABI change; FFI and WASM are
unchanged (neither binding has `SymmetricOperator` or `GeneralOperator`).

### Added

- **`SpdResolvent` (Rust, Python).** `SymmetricOperator::resolvent(λ, mass,
  solver, tol)` factors `λ·M + A` once for repeated solves, `λ ≥ 0`, `M =
  diag(mass) > 0`. `λ = 0` is the steady state. A tridiagonal operator uses an
  exact `O(n)` LDLᵀ with a positive-pivot certificate; everything else uses PCG
  with IC(0) (reported fallback to Jacobi). `SolveReport` carries the iteration
  count and the true relative residual. Singular (`λ = 0` with a constant null
  vector), `n = 0`, overflow and stagnating systems return errors, not `Ok`. PCG
  has no multigrid: cost is `O(√κ·nnz)` per solve.
- **Operator composition.** `SymmetricOperator::with_diagonal(c)` (`A + diag(c)`,
  `c ≥ 0`, inserts missing diagonal entries), `csr()` / `to_csr()` (round-trips
  with `from_csr`), and Python `lumped_congruence`.
- **`phi_combination`** computes `Σₖ τᵏ φₖ(τG) wₖ` (`p ≤ 3`) in one augmented sweep
  (Al-Mohy–Higham 2011, Thm 2.1). With the §62.5 recipes it gives exact affine
  evolution `M u' = −A u + s`, ETD-RK2 and ETDRK4 with the nonlinearity evaluated
  by the caller. Python: `semiflow.phi_combination(op, tau, w, mass=None)`.
- **`CsrGenerator`** gives φ-functions a diagonal mass (`G = −M⁻¹A`) and makes them
  work on `GeneralOperator`. Python `phi_action` and `phi_action_batched` take
  `mass=` and accept a `GeneralOperator`; `mass=None` is bit-identical to the old
  path.
- **Python surface** `SpdResolvent` (`solve`, `solve_batched`, `solve_info`,
  `method`, `n`) and `SymmetricOperator.{resolvent, with_diagonal, to_csr,
  lumped_congruence}`; the GIL is released during solves (ADR-0031). `.pyi` stubs
  and README updated.
- **Gates.** `G_SPDR_TRIDIAG_DENSE`, `G_SPDR_PCG_DENSE`, `G_SPDR_STEADY_MMS`,
  `G_SPDR_PHI1_LIMIT`, `G_SPDR_IMPLICIT_CROSS`, `G_SPDR_REJECT`,
  `G_SYMOP_COMPOSE_EXACT`, `G_PHI_COST_V_INVARIANT`, `G_PHI_COMBINATION_DENSE`,
  `G_PHI_MASS_DENSE`, `G_PHI_GENERAL_DENSE`, `G_ETD_AFFINE_EXACT` (all
  RELEASE_BLOCKING); `G_SPDR_LINEAR_COST` is advisory. `semiflow-nostd-check`
  gains the scenarios `spdr_tridiag`, `spdr_pcg_ic0` and `phi_combination`, with
  committed digests that the `std-ref`, AVX2, NEON and QEMU jobs must reproduce.
  `tests/spdr_binding_parity.rs` pins the FNV digest the Python test checks
  bit-for-bit.
- `GeneratorSensitivity::accumulate_bilinear` (provided method; default = the
  old per-parameter loop, so external implementors keep working) with `O(nnz)`
  overrides for `EdgeWeightSensitivity`, `EntrySensitivity` and
  `NodeTimescaleSensitivity`.
- `graph_expmv_frechet_plan` / `FrechetPlan`: pure predictor of node count,
  chain length, maximum degree and SpMV bound of one Fréchet channel (the
  accuracy gates derive their thresholds from it).
- `GraphKrylovChernoff::{lambda_max_bound, tol, path}` accessors.
- Python: `tol` keyword (default `1e-12`) on `symmetric_op_expmv_frechet`.
- Gates `G_FRECHET_QUAD_CONSTANT`, `G_FRECHET_LARGE_T_ORACLE`,
  `G_FRECHET_BILINEAR_ONE_PASS`, `G_FRECHET_LARGE_T_NOSTD_DIGEST` (new
  `graph_frechet_large_t` scenario in `semiflow-nostd-check`),
  `G_PY_FRECHET_LARGE_T`, `G_FRECHET_SWEEP_EXACT_PROP` (in-crate: the library sweep
  driven by exact eigen propagators, `ρ̄t`-floor-free bound) (RELEASE_BLOCKING) and `G_FRECHET_COST_LOG_NODES`
  (ADVISORY); evidence kit `scripts/frechet_large_t_kit.py`.

### Changed

- **`phi_action` / `phi_action_batched` cost no longer depends on `‖v‖`.** The
  Taylor truncation `(s, m)` was chosen from `τ‖G‖ + ‖v‖∞ + 1`, so the substep
  count grew with the magnitude of the input (16.9M matvecs at `‖v‖ = 1e6`, 90
  after the fix). The input is now scaled by a power of two, so `(s, m)` depends
  only on `τ‖G‖` and the result is bitwise homogeneous in `v`. **Behaviour
  change at the truncation level:** results can move by at most the truncation
  tolerance. At unit norm they are unchanged (the ETDRK4 and adjoint gates are
  bit-identical). §62.4 is amended to the implemented scaling.
- `SemiflowError::ConvergenceFailed` is now returned in production (PCG iteration
  cap and gross-residual guard); its rustdoc no longer says it is reserved.

### Fixed

- The IC(0) preconditioner that §59.2 declared normative but `pcg.rs` never
  shipped is now implemented (zero fill-in, Jacobi fallback reported by
  `SpdResolvent::method()`). `ImplicitEuler` is untouched and bit-identical.
- `G_SPDR_STEADY_MMS` threshold re-derived a priori (`C* = 3.506917`, ADR-0202
  Amendment 1); PCG result semantics and the near-singular band documented
  (Amendment 2).
- **Graph / symmetric-operator Fréchet gradient: quadrature error independent of
  `λ_max·t` (ADR-0203, math §63).** `graph_expmv_frechet` (and through it
  `symmetric_op_expmv_frechet`, `EdgeWeightSensitivity`, `EntrySensitivity`,
  `NodeTimescaleSensitivity`, Python `graph_expmv_frechet` and
  `symmetric_op_expmv_frechet`) integrated the Duhamel integral with ONE 8-point
  Gauss-Legendre panel on `[0, t]`. The integrand has boundary layers of width
  `1/λ_max` at both ends, so the relative error was `1e-10` at `λ_max·t = 10`,
  `7e-4` at 53 and `O(1e-2)` beyond ~100 (measured; the propagators were exact).
  The integral is now evaluated on a two-sided graded Gauss-Legendre mesh
  (`t/2` split, ratio 3/2, nodes carried as distances; `16·(K+1)` nodes,
  `K = ⌈log_{1.5}(ρ̄t/4)⌉`), with quadrature error `≤ 1.1e-14` of the
  absolute magnitude for EVERY `λ_max·t` (a-priori bound, §63.4) and memory
  `≤ 15·N` independent of `λ_max·t`. The f64 conditioning floor
  `≈ (r+n)·u·ρ̄t` of any method that applies `L` in floating point remains
  (about `3e-9` relative at `λ_max·t = 1e6`), and the `ImplicitEuler` path keeps
  its `O(Δt)` bias. The result changes numerically (more accurate), not
  bit-identically; `graph_expmv_frechet`'s signature is unchanged and
  `G_GRAPH_FRECHET_FD`, `G_SYMOP_ENTRY_FRECHET` and `T_ADJOINT_STATE_SENSITIVITY`
  keep their thresholds. Cost: `2 + 32(K+1)` propagator calls per channel (was
  16) and `SpMV`s `≤ B(ρ̄t) = m_Z·(3ρ̄t/Z_SAFE + 2 + 32(K+1))`, which tends to
  `6×` one action as `ρ̄t → ∞` (measured `5.4–25×` one action for
  `ρ̄t = 1e6…1e3`; the old rule was `≈ 8×`). Operators must be PSD (§63.1.d;
  `SymmetricOperator::from_csr` does not check it).
- The all-parameter contraction is `O(nnz)` per node (it was `O(n·n_params)`):
  an all-edges gradient now costs the same propagator actions as a single edge.

## [0.14.0-beta] — 2026-10-05

CI, documentation and contract hygiene, plus three library changes: the
`no_std` build fix (#40), one math backend for every build so `std` and
`no_std` results are bit-identical, and `no_std` as the default build (see
Changed). No ABI change. The `G_SMOLYAK_D5` gate's slope band and measurement
method were corrected, and the advisory zeta8 Chebyshev gate's reference
sampler was realigned (see Fixed); no other gate threshold or tolerance moved.

### Changed

- **BREAKING: the default build is `#![no_std]` + `alloc` (ADR-0201); MSRV
  1.78 → 1.81.** `simd` no longer implies `std` (its intrinsics come from
  `core::arch`), so `semiflow = "…"` gives a `no_std` build with SIMD kernels.
  `std` is opt-in and needed only for `parallel`. `SemiflowError` implements
  `core::error::Error` (stable since 1.81, the same trait as
  `std::error::Error`) in every build, so `?` into `Box<dyn Error>` keeps
  working without the `std` feature. Code that relied on semiflow's default
  features to enable `num-traits/std` must enable it itself. The binding
  crates already name their features and are unaffected.

- **`std` and `no_std` builds produce identical bits (ADR-0200).** Before
  this, `std` builds computed generic transcendentals (`exp`, `powf`, `sin`, …
  via `num_traits::Float`, and `num_complex::Complex::{exp, sqrt, norm}`) with
  whichever backend num-traits resolved to — the platform library in test
  builds, CPU-dependent through glibc IFUNC — and `no_std` builds with `libm`;
  without the `simd` feature, the lane kernels (Catmull-Rom, 9-point FD,
  septic/octonic Hermite derivatives, cached G⁴ stencil) ran separate scalar
  code that summed in a different order. Now:
  - every transcendental goes through the new `SemiflowFloat::libm_*` methods
    (`libm` for `f32`/`f64`; `Dual<F>` applies them to its components) and
    complex ones through num-complex's formulas on top of them;
    `clippy::disallowed_methods` forbids the platform-dependent originals;
  - the portable lanes compile in every build, and the G⁴ scalar fallback
    mirrors the AVX2/NEON arithmetic operation for operation;
  - `crates/semiflow-nostd-check` hashes each scenario's output and CI
    requires the same committed digests from the `no_std` build, the `std` +
    `simd` build (`--features std-ref`), AVX2, aarch64 NEON and QEMU
    Cortex-M3/M4F.

  `std` results can change in the last bits compared with 0.13.1-beta. The
  `std` feature still enables `num-traits/std` and `num-complex/std`; that now
  only affects downstream code calling `Float` directly.

- **One source for every published README.** `docs/readme/` is rendered by
  `cargo xtask readme` into the GitHub, crates.io/docs.rs, PyPI and npm
  READMEs. Versions, MSRV, dependencies, feature flags, the Python
  class/function inventory and the WASM class tables (lite vs `full`) are
  computed from the manifests and sources. `cargo xtask readme --check` fails
  CI on drift. The docs.rs front page is the crate README (its Rust examples
  are doctests). The Python and Node.js examples are executed in CI
  (`test_readme_examples.py`, `xtask readme-examples-js`).

- **`doc-check` covers names and versions across user docs.** It now covers
  top-level `docs/*.md`, `SECURITY.md`, `CONTRIBUTING.md`, `CITATION.cff` and
  `.zenodo.json`. It rejects the pre-rebrand names (`semiflow_core`,
  `semiflow-core`, `remizovcore`), stale `semiflow = "…"` requirements, and
  install commands naming anything but `@semiflow/wasm` / `semiflow-pde`.

### Fixed

- **flagship-gates CI: rust-cache is keyed by runner CPU model.** `target-cpu=native`
  test binaries are no longer restored onto a CPU lacking their instructions
  (SIGILL in the scheduled RELEASE_BLOCKING runs 2026-10-03..05).

- **`no_std` is now verified by execution, not just type-checked.** The
  previous state:
  - `cargo test --no-default-features` did not compile.
  - Nothing executed the `libm` math that `no_std` users get, because test
    builds always load `std` through dev-dependencies.

  New CI jobs:
  - `no-std-test` runs the full suite with the crate built `#![no_std]`.
  - `no-std-libm` runs `crates/semiflow-nostd-check` (closed-form oracle
    scenarios plus output digests) with `num-traits` provably without `std`,
    then with `--features std-ref` and with AVX2; `no-std-libm-neon` repeats
    it on aarch64.
  - `no-std-qemu` runs the same scenarios as a bare-metal binary on Cortex-M3
    and Cortex-M4F under QEMU (`nostd-qemu/`).
  - `no-std` now builds for the host, `thumbv7em-none-eabihf`,
    `thumbv7m-none-eabi` and at MSRV.

  Targets without pointer-sized atomics (e.g. `thumbv6m`) now get a clear
  `compile_error!` instead of dozens of `alloc::sync` errors.

- **User documentation matched reality again.** Fixes include:
  - `use semiflow_core::…` in the front-page quickstart and guides.
  - Stale versions: `semiflow = "0.9"`, `"9"`, `v0.9.0-beta`.
  - `npm install semiflow` → `@semiflow/wasm`.
  - Wrong Python signatures and kernel names.
  - "Rust-only" claims about classes Python exports.
  - Stale SECURITY/CITATION/Zenodo metadata.
  - A "Production/Stable" PyPI classifier on a beta.
  - The claim that `simd` is bit-identical to the scalar path. The Catmull-Rom
    interpolant evaluates the same polynomial in a different arrangement, so
    builds with and without `simd` agree to rounding only.
  - A C build recipe using `--release` (`panic = "abort"`, which defeats the
    `catch_unwind` boundary) instead of `--profile release-ffi`.

  Executing the README examples found a JS quickstart that was a syntax error
  (`-x ** 2`).

- **Python: `SemiflowError.kind` now exists.** The class docstring, the `.pyi`
  stub and the README all document a `kind` attribute; it was only ever present
  as the `[Kind] ` message prefix (kept for compatibility).

- **Python: `ReverseHeat1D` was not importable.** It was registered in the
  native module but missing from `semiflow/__init__.py`, so
  `from semiflow import ReverseHeat1D` raised `ImportError`. The new
  `tests/test_public_exports.py` fails if any native class is not re-exported.

- **The `no_std` build (`--no-default-features`) did not compile (#40).** Plain
  `f64` receivers called `std`-only methods (`sqrt`, `ceil`, `floor`, `round`,
  `powf`, `ln`, `log2`, `sin`, `cos`, `rem_euclid`), and several modules used
  `vec!`, `Vec` or `to_owned` without importing them from `alloc`. The
  `std`-only methods now resolve through `num_traits::Float` (libm) only when
  `std` is off, so `std` builds keep calling exactly the same functions.
  `rem_euclid` became a local helper with the same arithmetic. At the then-current
  MSRV of 1.78, `f64::abs` is not in `core`, so those calls use `libm::fabs`,
  which is exact. A new `ci.yml` job, `no-std`, checks the build on the host, on
  `thumbv7em-none-eabihf`, and at the MSRV (1.78 when added; 1.81 since the
  Changed entry above). Until now no job compiled it, because
  the default `simd` feature always turns on `std`.

- **`G4_NS2D_aniso` and `G5_3D` are now `RELEASE_BLOCKING` in the contract
  (#34, ADR-0199).** They were `NORMATIVE`, while `release-process.md`, the
  `flagship-gates.yml` title and ADR-0024 all treated them as blocking. That
  also left them outside the coverage check. `check_gate_coverage.py` now strips
  YAML trailing comments from `severity:`/`test_file:`. Before this,
  `severity: RELEASE_BLOCKING  # note` silently fell out of the check.

- **`latency_tail` emitted `"library":"semiflow-core"` (#32)**, a crate name that
  no longer exists. It now emits `"semiflow"`, and `schema_version` goes from
  `0.1` to `0.2`, so a consumer pinned to 0.1 fails loudly instead of matching
  nothing. Nothing in the repo reads either field.

- **Two contract properties had no test (#29).** They now do:
  `strang_split_palindrome_consistency` and
  `truncated_exp_strang_quasi_contractivity`, in
  `tests/proptest_strang_split.rs`. Both run the library `StrangSplit` through
  `apply_into`. Their `invariant:` blocks are ported off the removed v0.3 API,
  and they now have `test_file:` pointers.

- **Gated tests below `RELEASE_BLOCKING` could run in no workflow (#31).**
  `check_gate_coverage.py --list-unnamed` computes the set: 43 binaries today.
  The new nightly job `catch-all-gated` runs it with `-- --include-ignored`. Two
  binaries are excluded, each with a reason recorded in `NEVER_RUN`: an OOM stub
  and a fixture-overwriting capture.

- **62 of the 117 `RELEASE_BLOCKING` gates were executed by no workflow at
  all.** `ci.yml` runs `cargo test --workspace --release` — no
  `--features slow-tests`, no `-- --ignored` — and `flagship-gates.yml` names
  its test binaries one by one, 24 of them in total. Everything else fell
  through. 27 of the affected files are `#![cfg(feature = "slow-tests")]`, so CI
  never compiled them; the rest carry `#[ignore]`, so CI compiled the binary,
  skipped the gate, and counted it in the `N ignored` line of a green run.

  The formal safety net was the manual `xtask test-full` + `test-ignored-gates`
  step in `docs/release-process.md`. It depends on a human remembering, and
  before `0.13.0-beta` it did not happen — which is how the gates written *for*
  that release's issue campaign shipped having never run in CI: `G_ASND_MOMENT`
  (the second-moment oracle written to catch issue #17), the five `G_CONS_*`
  (#26), `G_SHIFT1D_COEFF_FD` (#25), `G_PENCIL_ORDER2` (#21).

  `flagship-gates.yml` gains seven concern-grouped jobs covering all 62 —
  `campaign-gates`, `operator-exponential-gates`,
  `geometry-hypoelliptic-gates`, `resolvent-sampling-gates`,
  `wentzell-multilayer-ad-gates`, `nonseparable-2d-gates` — run with
  `-- --include-ignored` so coverage does not depend on which gating mechanism a
  file uses. That workflow already triggers nightly and on every `v*` tag, so
  the set is now bound to released commits as well as to the schedule. The
  manual pre-tag run on bench hardware is unchanged and still authoritative;
  what changes is that skipping it becomes visible within a day instead of
  never.

- **The gate-to-workflow mapping is now machine-checked**
  (`scripts/check_gate_coverage.py`, wired into `ci.yml`). The list of `--test`
  flags that closes the hole above is hand-maintained, and it drifted within a
  day of being written: review of this change found four binaries already
  missing from it — `sym_op_dense`, `obstacle_vi_slope`,
  `strang2d_parallel_bit_equal`, `chernoff1d_parallel_bit_equal`, 7 gates
  between them. They are added here, and the invariant is asserted on every PR
  instead of audited once.

  The same pass corrected the audit figure: **62**, not the 59 first published.
  The original script matched one regex and had three blind spots — it saw only
  the bare `#![cfg(feature = "slow-tests")]` and missed the compound
  `#![cfg(all(feature = "parallel", feature = "slow-tests"))]`; it never
  considered per-test `#[cfg(feature = "slow-tests")]`; and a trailing comment
  after an attribute broke its attribute-chain match, which is exactly how
  `sym_op_dense` read as reachable.

- **`G_SMOLYAK_D5` / `G_SMOLYAK_D6` cost far more than their headers claim.**
  Found by measuring the newly-reachable gates before wiring them up — their
  first execution, ever. `g_smolyak_d5.rs` advertises "~10-30 s on release";
  it exceeded a 40-minute cap on a 12-core host without finishing.
  `g_smolyak_d6.rs` budgets "≤ 2 min wall-clock"; it hit the same 40-minute cap.

  The `G_SMOLYAK_D6` gate is unchanged. `G_SMOLYAK_D5` was, in this release,
  rewritten (see the next entry) — its slope band and measurement method both
  changed, so "the estimates went stale" applies to its cost, not to its
  contents. ADR-0191 replaced multilinear N-D sampling with the `K^D` tensor
  stencil, so a `D = 5` sample reads 1024 nodes where it used to read 32.
  ADR-0191 measured that 32× on the `D = 5` Smolyak smoke tests (70 s → 198 s,
  which is why those were re-sized) and recorded the slope gate as "untouched",
  meaning not re-sized — its runtime was never measured, because it ran nowhere.

  The completed sweep found a third of the same kind: `strang_nonseparable_slope`
  (`G3_NS2D`, `G3_NS2D_var`) also exceeded the 40-minute cap. No estimate was
  stale there — the gate is expensive by construction, fixing N=1000 Chernoff
  steps across a spatial sweep — but for scale, its sibling
  `strang_nonseparable_aniso_slope` is 118 min on a hosted runner and this one
  is at least twice its local cost, which projects past 4 h and too near the 6 h
  job limit for the tag lane.

  All three now run in `nightly.yml` next to `ddim-d5`, which was moved out of
  the flagship workflow for the same reason — **each in its own job**, since two
  multi-hour gates sharing one runner would exceed the 6 h hosted limit and be
  killed with neither result recorded. Their `timeout-minutes` is 350, not 420:
  GitHub kills a hosted job at 6 h regardless, so a larger number would only
  misdescribe the budget. They are deliberately not on the tag lane: a hosted
  runner is ~6× slower than the measurement host. The stale budget claims in both
  file headers are corrected in place rather than deleted.

- **`G_SMOLYAK_D5` measured the wrong convergence regime (#38).** The gate
  demanded a first-order slope (`≤ −0.95`; the contract still said `≤ −1.95`),
  but this variable-coefficient tanh-coupled datum runs on the `2√τ`
  anisotropic-shift family, which is globally order ½ (ADR-0191 Amendment 3);
  Smolyak changes only the quadrature backend. Hosted-runner output measured a
  successive-difference slope ≈ −0.477. The test (`tests/g_smolyak_d5.rs`) now
  gates the OLS slope of `log‖u_n − u_2n‖∞` against `log n` over
  n ∈ {32, 64, 128} — pairwise refinement deltas, with no `n_ref = 512`
  reference run, whose floor contaminated the old estimate — and asserts the
  two-sided band `−0.75 ≤ slope ≤ −0.42`, the same band as the dense
  anisotropic D = 5 gate. The upper bound fails loudly if the kernel ever gains
  an order. The node-count (`< 3125`) and `F(0) = I` sub-checks are unchanged.
  `contracts/semiflow-core.properties.yaml` and ADR-0123 now state the same
  band; `SmolyakGridND::order() = 1` is documented as a nominal
  constructor-level value, not a promise that every variable-`A` gate slopes at
  −1; the Smolyak jobs in `nightly.yml` use `target-cpu=native` only on
  self-hosted runners.

- **Hosted-runner portability in `flagship-gates.yml`.** The "operator and
  matrix-exponential gates" job exported `RUSTFLAGS="-C target-cpu=native"`
  unconditionally, which can produce binaries a GitHub-hosted runner's CPU
  cannot execute. It now applies native tuning only on self-hosted runners and
  uses portable target-cpu defaults on hosted ones. A clippy-driven cleanup in
  `matrix_pade_tests_mod.rs` (`a[0].fill(..)`) rode along; no behaviour change.

- **zeta8 Chebyshev advisory gate compared mismatched samplers (#37).**
  `g_zeta8_var_a_slope_cheb` (`tests/zeta8_correction_slope.rs`) probed a
  Chebyshev-sampled ζ⁸ kernel against a K5 reference built without Chebyshev
  sampling, so the advisory slope near its 0.1 gate drifted with the runner.
  The K5 reference oracle now uses `with_chebyshev_sampling()` too. The gate
  stays advisory and its threshold is unchanged.

- **`clippy` had never type-checked the `slow-tests` files either.** `ci.yml`
  now runs `cargo clippy --workspace --all-targets --all-features -- -D warnings`.
  Without `--all-features` the same 27 files were invisible to the lint job too,
  so they could drift out of sync with the library — a renamed constructor or a
  changed signature would break them while every PR stayed green. clippy is a
  `check` build, so covering them costs minutes; a full
  `cargo test --no-run --release` over the same targets measured >18 min on 12
  cores, which is hours on a hosted runner.

  Clearing the backlog this exposed — **346 unique diagnostics across 31 files**
  — is part of this change: 183 auto-fixed by `cargo clippy --fix`, 10 fixed by
  hand, 153 covered by file-level `#![allow]` with an inline justification, the
  convention 130 test files in this repo already followed. Three of those
  justifications are load-bearing and deliberately not "cleanable":
  `identity_op`/`erasing_op` keep row-major index formulas readable
  (`gen[0 * nd + 0]`), `manual_clamp` preserves `max().min()` NaN behaviour
  which `f64::clamp` does not share, and `match_same_arms` keeps a spec table
  transcribed arm-for-arm. Every `#[ignore]` on a gate now states its reason in
  the attribute.

## [0.13.1-beta] — 2026-08-15

Patch release over `0.13.0-beta`. Two defects shipped in that release (a
non-portable `π^{-D/2}`, and an ADR left self-contradictory by the merge); a
third — `xtask py-smoke` reusing a cached virtualenv — has been latent since
`0.9.0-beta` and only became reachable once CI cached `target/` across a
multi-version matrix. Fixing them ran into a pre-existing deadlock between two
project gates, resolved here rather than deferred again.

No public API change. Numerically, only the N-D sampler normalisation moves, and
only for odd `D`: `D = 3` and `D = 5` shift by 1 ULP on this platform;
`D = 2, 4, 6` are bit-identical to `0.13.0-beta`. The shift is a change of
*which* value you get, not an accuracy improvement — see the per-`D` table in
ADR-0191 AMENDMENT 5.

### Fixed

- **`G_BINDING_SMOLYAK_PARITY` sub-test 2 was non-deterministic by
  construction** (ADR-0191 AMENDMENT 6, `Gate-Change-Approved-By: ilia-volkov`).
  The golden vector was captured from a Rust run whose initial condition came
  from `f64::exp`, while the Python sub-test recomputed that initial condition
  with `np.exp` — a different, runtime-CPU-dispatched implementation — and then
  asserted the two pipelines agree **bit-for-bit**. The gate's verdict depended
  on which hosted runner it drew: fail, fail, pass, fail across four CI runs
  whose only Smolyak-affecting difference was the removal of a lint attribute.
  The Python side no longer calls `np.exp`. It computes the *exponent*, which is
  exact — `linspace` reproduces `Grid1D`'s `x_i = xmin + i·dx` bit-for-bit — and
  looks the exponential up in a 16-entry table of raw bit patterns taken from
  the Rust initial condition itself (only 16 distinct exponents occur over the
  4096 points). With the input pinned and `c ≡ 0`, the compared path is `+ − × ÷`
  and `sqrt` over literal Gauss–Hermite tables, all correctly rounded by
  IEEE-754 §5.4, so 0 ULP is now required by specification instead of achieved
  by luck. This **strengthens** the gate — it previously produced accidental
  passes as readily as accidental failures, and could not distinguish "the
  marshalling is transparent" from "two libms happened to agree". No threshold,
  tolerance or skip was added to the parity assertion, and the golden output
  constants are untouched. Measured non-vacuity: a 1-ULP change to the input
  moves the output by up to 25 ULP and trips the assertion. Two guards keep the
  pinned data honest — an FNV-1a/64 checksum against the Rust side (bit-exact),
  and a ≤4-ULP tolerance check that the constants really are `exp(-Σx²)`.

- **`π^{-D/2}` was computed with `powf`, which is not portable** (ADR-0191
  AMENDMENT 5). This was found while investigating a failure of
  `G_BINDING_SMOLYAK_PARITY_SUB2_PYO3_0ULP`; it is **not** the cause of that
  failure and does not fix it (ADR-0191 AMENDMENT 6 has the diagnosis and the
  evidence — that gate remains OPEN). It is a real defect on its own terms:
  `powf` lowers to the system `pow`, which IEEE-754 does not require to be
  correctly rounded and which glibc dispatches by IFUNC per CPU; because
  `π^{-D/2}` is a global normalisation multiplying *every* output, a 1-ULP
  platform difference there moves the whole vector. Replaced by
  `float::inv_pi_pow_half` — `⌊D/2⌋` multiplications, one `sqrt` for odd `D`,
  one reciprocal, all correctly rounded by IEEE-754 §5.4 and therefore
  bit-identical across platforms by specification. `D = 2, 4, 6` are unchanged;
  `D = 3, 5` move 1 ULP. Determinism is the deliverable, not accuracy: measured
  against a 60-digit reference, the old `powf` was the correctly rounded result
  for `D = 3, 5` *on this host* and the new chain is 1 ULP above it, while at
  `D = 6` both are 1 ULP above it. Applied at all three call sites:
  `SmolyakGridND`, `AnisotropicShiftChernoffND` (the kernel behind every
  `G_DDIM` gate), and `AnisotropicShiftAdaptiveQ` — the third was found by
  grepping for the pattern rather than by a failing gate, since no parity gate
  covers the adaptive N-D kernel.
- **`xtask py-smoke` reused a cached virtualenv across Python versions** — all
  six `py-smoke` matrix cells and `py-test-fast` failed with
  `Command '[...target/py-smoke-venv/bin/python3', '-m', 'ensurepip', ...]'
  returned non-zero exit status 1`. The interpreter in that message is the
  *venv's*, not the runner's, which is the tell: `python3 -m venv DIR` on an
  existing `DIR` reuses it and bootstraps pip through the symlink recorded when
  that venv was first built. The venv lives under `target/`, which CI restores
  from the `Swatinem/rust-cache` archive, and the matrix points 3.10 and 3.13 at
  the same path — so a cache entry written by one interpreter was handed to
  another. `create_venv` now removes the directory first, matching what its own
  doc comment already claimed ("create a fresh venv"); the `py-test-fast`
  workflow step gets the same `rm -rf`.
- **ADR-0191 carried stale duplicate copies of AMENDMENTs 2 and 3** — the
  0.13.0-beta merge appended the pre-resolution drafts (`### Why this is not
  resolved here`) after the resolved versions, so the document contradicted
  itself on whether `G_DDIM` had been re-based. Duplicates removed; a
  duplicate-heading scan over `docs/adr/`, `contracts/` and the top-level docs
  found no other instance.

### Changed

- `cargo fmt --all` applied across the workspace (30 files). `fmt` had been red
  in CI since 2026-07-15 because two project gates were mutually unsatisfiable:
  rustfmt expands the packed `pub mod a; pub mod b;` lines that `check-lints`'
  500-line file budget was being kept under by packing them. Resolved by
  shrinking `lib.rs` prose (506 → 498 lines) and splitting the three functions
  that crossed the 50-line budget once reformatted (`shift1d_coeff_grad`,
  `diag_const_coeff_slope_control`, `g_shift1d_coeff_fd`) rather than by
  relaxing either gate. The other 27 files are mechanical reformatting with no
  semantic change.
- `.gitignore` now covers `target-*/`. `docs/release-process.md` instructs
  running the heavy gates under `CARGO_TARGET_DIR=target-flagship`, so following
  the documented procedure left an untracked multi-hundred-MB directory in
  `git status`.

## [0.13.0-beta] — 2026-08-15

Issue campaign #17 / #19 / #21–#26.

Wave A (correctness): #17 + #26. Wave B (capability): #19, #21, #22, #23, #24, #25.
All eight issues closed.

### Fixed

- **`GridFnND::sample` ignored interpolation order and boundary policy**
  (#17, ADR-0191, math §32.9) — **this silently corrupted every `D > 1` kernel.**
  The N-D sampler hard-coded multilinear interpolation with an index clamp,
  consulting neither the `InterpKind` nor the `BoundaryPolicy` each axis already
  carried. Because the Chernoff product resamples at off-grid quadrature feet
  every step, linear interpolation injected ≈ `dx²/6` of spurious second moment
  **per step**, accumulating *linearly in the step count* — so refining
  `n_steps`, the one knob users turn for accuracy, made the answer worse. On the
  issue's datum (`A = I`, `t = 0.5`, 96² grid) the variance gain read
  `1.2113 / 2.2449 / 4.4901` at `n_steps = 100 / 400 / 1600` against an exact
  `1.0`; it now reads `1.000000` at all three. `A = diag(1.0, 0.5)` gives
  `(1.000000, 0.500000)` and an off-diagonal `0.4` gives `dCov = 0.400000`.
  `GridND` gains `interp` (default `CubicHermite`) and `with_interp`. Gate
  `G_ASND_MOMENT`. Affects `shift_nd`, `shift_nd_zeta2`, `shift_nd_adaptive`,
  `smolyak`, `obstacle`, `obstacle_nd`, `point_eval`, `carnot_complex`,
  `carnot_stepk`, `hormander_engel` and the ND FFI/WASM surfaces.
- **`thomas_solve` accepted non-finite pivots** (ADR-0192). The guard tested
  `w == 0`, which is false for `NaN`, so a NaN pivot propagated into the state
  vector silently. This was the only path in the conservative subsystem without
  a finiteness backstop.

- **`expmv`'s θ_m table was mis-transcribed** (ADR-0198, math §45.2.bis) —
  **`expmv` and the Lanczos path were silently under-substepping by up to 8×.**
  `THETA_M` claimed to be Al-Mohy & Higham Table 3.1 but paired each degree with
  a radius from two to three rows further down it: `m = 18` carried `θ ≈ 8.84`,
  whose real owner is `m ≈ 51`, against its own `θ_18 = 1.09`. Since
  `select_s_m` takes `s = ⌈τ‖A‖/θ_m⌉`, too large a θ means too few substeps and
  a wrong answer with no runtime symptom. Forward relative error of `T_18` at
  the claimed radius: **3.8e+04**; at the correct one, 5.0e−16. The replacement
  values were recomputed from the definition in exact rational arithmetic rather
  than re-copied, and reproduce Table 3.1 at every degree. `M_MAX` raised 18 → 30
  (the old cap protected the Horner *argument*, which the wrong table was itself
  violating). `graph_krylov`'s mirror corrected in place, keeping its structural
  `m ≤ 18` cap. Found while building #24 — the tight `‖A‖_∞` there exposed what
  loose norm bounds elsewhere had been masking. New gate `G_THETA_M_TABLE`;
  `expmv_div_form_action_accuracy` improves from 1.1e−15 to 2.2e−16.
- **`Heat2DVarA` / `Heat3DVarA` claimed order 2 and are order 1**
  (ADR-0191 AMENDMENT 1, math §9.2.3.B.bis). Their axis kernels freeze `a` at the
  node, which agrees with `e^{τa∂ₓₓ}` at `O(τ)` and differs at `O(τ²)` by
  `(τ²/2)·a·(a''f'' + 2a'f''')` whenever `a` varies. Measured global order
  **1.007**. Corrected on the ADR-0112 precedent; `DiffusionChernoff::order() == 2`
  is untouched and still earned, because that path supplies `a'`/`a''`.
- **The WASM crate's `full` feature never compiled.** `graph_magnus_wasm.rs`
  declared `mod graph_magnus_wasm_helpers;` at a path that has never existed
  (Rust 2018 resolves a nested `mod` in a non-`mod.rs` file to a subdirectory),
  and `graph_adjoint_wasm.rs` carried five non-snake-case methods. No build path
  enables `full` — not `xtask wasm-build`, not CI — so ~20 gated modules,
  including the whole Magnus graph surface, had been dead and unbuildable.
  Fixed, and CI now type-checks `--features full` so it cannot rot again.

### Added

- **`k ≥ 0` in conservative divergence-form diffusion** (#26, ADR-0192,
  math §56.8.bis). Degenerate-at-the-boundary conductivities are now accepted:
  CEV `k(S) = ½σ²S^{2β}` at `S = 0`, Feller/CIR `k(v) = ½ξ²v` at `v = 0`,
  Wright–Fisher at both ends. A degenerate face carries exactly zero flux, so
  the boundary classifies itself and the operator stays symmetric PSD; the
  `max(k, ε)` floor callers were forced into is no longer needed. `k < 0` and
  non-finite `k` remain `DomainViolation`. Gate `G_CONS_DEGENERATE`.
- **`boundary=` on `AnisotropicShiftND2` / `AnisotropicShiftND3`** (#17
  secondary) — `"reflect"` (default), `"periodic"`, `"zero"`, `"linear"`,
  now actually honoured by the sampler.
- **`shift1d_coeff_grad`** (#25, ADR-0197, math §61) — gradients w.r.t. the
  per-node coefficient **fields** of `Shift1D.with_arrays`, i.e. local-vol Vega
  surfaces `∂V/∂σ(S_i)`. `EvolverHeat1DGreeksV3` differentiates only w.r.t. a
  single global diffusion scale. Mirrors `edge_weight_grad`'s contract: you
  supply the cotangent, the loss stays outside the library. Costs
  `O(n_steps · n)` — the same order as the forward solve — because the
  parameter→output coupling is diagonal, which is why `GeneratorSensitivity`
  (one parameter per call, `O(n_steps · n²)`) is deliberately not implemented.
  The adjoint's interpolation weight rows are **measured** from the sampler
  rather than hand-transposed, so they are correct for every `InterpKind` and
  `BoundaryPolicy` by construction. Gates `G_SHIFT1D_WEIGHTS_ORACLE`,
  `G_SHIFT1D_TRANSPOSE_ID`, `G_SHIFT1D_COEFF_FD`. Honest limits: `wrt='a'`
  requires `a_i > 0` strictly (the `√(τ/a)` chain factor diverges), so the
  gradient's domain is strictly smaller than the forward kernel's;
  `O(n_steps · n)` trajectory memory with no checkpointing; order 1, matching
  the kernel.
- **`Heat2DVarA.with_grid_arrays`** (#21, ADR-0196, math §60) — full-grid
  `a_x(x,y)` / `a_y(x,y)`. Each diagonal coefficient could previously vary only
  along its *own* axis, because `Strang2D` applies one shared kernel to every
  pencil — and the decorrelated Heston generator needs exactly the opposite
  (both coefficients depend on `v`). New core type
  `semiflow::strang2d_pencil::Strang2DPencil` carries one 1-D kernel per pencil.
  Gates `G_PENCIL_REDUCTION` (separable input reproduces `Strang2D` to 1e-13)
  and `G_PENCIL_ORDER2`. §60 replaces the *justification* for order 2: `Strang2D`
  argues from `[L_x, L_y] = 0`, which is false here, so order 2 now rests on the
  classical symmetric-splitting BCH residue instead. The slope survives; the
  error **constant** carries the double commutators — measured slope 2.05 / 1.91
  / 1.18 at ±10% / ±30% / ±60% transverse amplitude, with the ±60% ladder
  pre-asymptotic. Serial only, deliberately: reusing `Strang2D`'s threaded
  passes would put the ADR-0018 bit-equality contract in the blast radius.
- **`GeneralOperator`** (#24, ADR-0195) — externally-assembled **non-symmetric**
  CSR operators and their `e^{−tA}v` action. `SymmetricOperator::from_csr`
  validates symmetry, closing the whole Krylov surface to non-self-adjoint
  generators; drifted Fokker–Planck `∂_t p = ∂_x(D∂_x p) − ∂_x(μp)` and
  Cartea–Jaimungal inventory ladders were both shut out. Routed to the
  **existing** symmetry-agnostic scaled-truncated-Taylor engine rather than to
  new Arnoldi code — which is also the safer choice for the nilpotent-plus-
  diagonal ladder, where Arnoldi's residual-based stopping is least reliable.
  Arnoldi stays deferred, now with a reason. Gates `G_GENOP_DENSE`,
  `G_GENOP_NONNORMAL`, `G_GENOP_ASYM_ACCEPTED` (teeth), `G_GENOP_COST_LINEAR`
  (advisory). Honest limits: cost is `Θ(t‖A‖_∞)` — **not** depth-flat; only the
  backward error is certified; Chebyshev and Lanczos remain structurally
  unavailable and there is deliberately no `path=` argument.
- **`Shift1D.evolve_batched`** (#19, ADR-0194) — batched multi-channel evolve
  for the 1-D grid family, `[N, C]` in and out. A strike strip, bump Greeks, or
  a batch of Fokker–Planck density anchors is `C` independent solves under the
  *same* generator with only `u0` differing; that was a Python loop paying
  object construction and a GIL round-trip per column. Core entry point
  `semiflow::grid_batched::evolve_batched_1d`, generic over any
  `ChernoffFunction<F, S = GridFn1D<F>>`. Bit-identical to `C` sequential solves
  (gate `G_GRID1D_BATCH_ULP`, asserted on `f64` bit patterns). Channel-parallel
  and the pre-existing node-parallel path are mutually exclusive by
  construction, so nesting cannot oversubscribe.
- **`AdaptivePI.with_arrays`** (#22, ADR-0193) — adaptive PI stepping over a
  variable-coefficient `Shift1D` generator, `du/dt = a(x)u_xx + b(x)u_x + c(x)u`.
  The `kernel=` menu only reached constant-coefficient kernels — its `"shift"`
  arm hard-codes `a=0.5, b=0, c=0` — so Black-Scholes-type generators had no
  adaptive path and `n_steps` was hand-tuned per (grid, maturity, vol) triple.
- **`Shift1D.evolve_with_coefficient_schedule`** (#23, ADR-0193) — per-segment
  schedules for **all three** coefficients, each entry independently a scalar or
  a length-`n` array. Closes both gaps in `evolve_with_time_schedule`: no `b`/`c`
  schedules, and no space-varying coefficients inside a schedule (the
  Almgren–Chriss killing term `−γν(t)(p − ην(t))` needs both). Runs the whole
  walk in one GIL release with the state kept in Rust, ping-pongs instead of
  cloning the state per step, and leaves the object's coefficients at the final
  segment so a subsequent `evolve` continues instead of reverting.
  `evolve_with_time_schedule` is unchanged; its revert-on-next-evolve behaviour
  is now documented rather than altered.
- **`AnisotropicShiftND2.set_state` accepts a `(nx, ny)` array**, and
  **`values_2d()`** returns one. The library's flat layout is x-fastest
  (`flat[i + j*nx]` — Fortran order for shape `(nx, ny)`), and a reflexive
  C-order `ravel()` transposes the axes; this is what issue #17 reported as
  "axis mixing". Passing the 2-D array directly removes the trap.

### Changed

- **BREAKING (0.x)**: `GridND` gains a public `interp` field, so
  `GridND { axes }` struct literals no longer compile. Use `GridND::new`.
- `InterpKind::{SepticHermite, OctonicHermite, ChebyshevSpectralWithBC}` return
  `Unsupported` for `D > 1` (checked once in `GridFnND::new`). The N-D spatial
  floor is `O(dx⁴)`, not the 1-D family's `O(dx⁸)` — an honest limit, not a
  temporary one.
- The N-D tensor-product interpolant is not positivity-preserving, where
  multilinear was. Undershoot is `O(dx⁴)` and converges away (measured
  `min/peak`: `−1.8e−3` at n=8, `−3.5e−10` at n=32, `+3.9e−21` at n=64).
  `apply_into_smoke_d2` now asserts that convergence rather than strict
  non-negativity.
- `AdaptivePI`'s documented error kind for exceeding `max_substeps` corrected
  from `'CflViolated'` to `'ConvergenceFailed'` in both the rustdoc and the
  `.pyi`. The core returns `AdaptiveStepRejected` from that path and never
  `CflViolated`, so a caller following the docs caught the wrong kind.
- The `G_DDIM` D=2…5 order ladder (`N_AXIS = 8`, ADR-0112 AMENDMENT 1) was
  calibrated *around* the interpolation floor this release removes and must be
  re-measured on production hardware before tagging.
- `G_BINDING_SMOLYAK_PARITY`'s recorded golden values were regenerated — the
  sampler change moves them legitimately. Worth noting *how*: the new golden is
  symmetric (`[a, b, b, a, …]`) as the symmetric Gaussian IC on a symmetric
  domain requires, where the old one read `[a, b, c, c, …]` with `g[0]` and
  `g[3]` differing by a factor of 22. The index clamp had been breaking the
  reflection symmetry at the two ends.

- **`shift1d_vjp` no longer costs the 1-D path 70%** (ADR-0197 AMENDMENT 1).
  The regression recorded below in the previous revision was real but was not in
  that module: `boundary::bc_value` / `bc_index` / `reflect_index` flip between
  inlined and out-of-line with unrelated crate volume, and a septic sample
  resolves eight nodes through `bc_value` while `DiffusionChernoff` takes eleven
  samples per grid node — ~88 calls per node, ~2.2 M per `evolve(0.1, 100)`.
  Marking the three `#[inline(always)]` removes the fragility. Ten other
  interventions (`inline(always)` on `Grid1D::interp`, on `GridFn1D::sample`,
  outlining the samplers, hot-arm-first dispatch, resolving the coefficient
  `Storage` variant once per node, `codegen-units = 16`, `lto = "off"`, …) were
  each built and timed, changed nothing, and were reverted. Results are unchanged
  bit-for-bit. The path is now **faster than before the campaign**: Path 2
  50.7 ms → 34.2 ms, Path 1 123.8 ms → 110.4 ms, speedup 2.4× → 3.2× against the
  same `0e6d25b` baseline re-measured on the same machine.
- **`GridFnND::sample` resolves each axis's boundary policy once per sample**
  (ADR-0191 AMENDMENT 4). It used to resolve inside the collapse recursion, which
  re-visits axis `d` once per combination of the axes above it: `1364` `bc_index`
  calls per sample at `D=5, K=4`, of which `20` are distinct. `bc_value_by` is
  split into `bc_index` + `bc_value_from_hit`; `AxisStencil` carries the resolved
  hits and the axis stride. Arithmetic and summation order untouched, so results
  are bit-identical — `G_DDIM D=2`/`D=3` return exactly `−0.4676`/`−0.4766` and
  `D=4`'s successive differences match digit for digit.
  It bought **1.32×** (`D=4`: 1081 s → 820 s), not the 5× expected: the dominant
  cost is the recursion itself, not the policy resolution that was removed. That
  is enough to restore `D=5`'s `N_AXIS = 6` datum, which AMENDMENT 3 had lowered
  to 5 and which the gate rejected at slope `−0.3595`. A `D=2` probe showed why:
  on an adequate grid the slope is stable at every ladder position
  (`−0.446…−0.468`), on the coarse one it reads shallow everywhere
  (`−0.334…−0.443`) — the grid was contaminating the differences, not the ladder.
- `general_operator` uses the shared `expmv::select_s_m` again, now that the θ
  table it was avoiding is correct; its own derived criterion is deleted.

- **`G_DDIM` re-based: its estimator was contaminated and the kernel is order ½**
  (ADR-0191 AMENDMENT 3, `Gate-Change-Approved-By: VolkovIlia`). The gate compared
  each swept `n` against a single reference at `n_ref = 512` — only twice the
  largest swept `n`, so the last point measured the reference's own error.
  Holding the sweep and raising `n_ref` to 1024/2048/4096/8192 walked the reported
  slope from −0.92 to −0.54; a converged estimator would have been flat. It now
  fits the OLS slope of the reference-free successive differences
  `sup|u_2n − u_n|`, whose ratio settles on **√2** across `n ∈ [32, 16384]` and
  `N_AXIS ∈ {8, 16, 32}`: the global temporal order on the normative variable-`A`
  datum is **½**, from the `√τ` inside the Gauss–Hermite shift. Threshold
  `−0.95 → −0.45`, now two-sided (a `−0.75` ceiling fails loudly if the kernel
  ever gains an order). Measured after re-basing: D=2 −0.4676, D=3 −0.4766.
  Not a regression — the same probe on `0e6d25b` shows no clean power law and
  errors 10× larger; the ND sampler fix made the order legible.
  The ladder is also re-sized: `K^D` sampling had pushed `D=4` to **8105 s** and
  `D=5` to an extrapolated ~85 h. `D=4` now runs `{8,16,32,64}` (120 steps against
  752) and `D=5` runs `{4,8,16,32}`. Dropping a node per axis at `D=5` was tried
  and **reverted** — see the sampler entry below.

### Open

- **`shift_nd_zeta2`'s ζ² correction may lift the ND kernel's global order to 1.**
  `G_DDIM` was re-based this release after its estimator was found contaminated,
  and the kernel's true order measured at **½** (ADR-0191 AMENDMENT 3). The ζ²
  correction exists to lift exactly this kernel, and `G_AS_ZETA2_TAU2` already
  confirms its magnitude scales as a genuine `O(τ²)` — but whether that makes the
  *global* order 1 has never been measured with an uncontaminated estimator. If
  it does, `AnisotropicShiftChernoffND::order() == 1` becomes earned rather than
  inherited, and `G_DDIM`'s two-sided band will fail loudly to say so.
- **`ChernoffFunction::order()` cannot express a fractional order.** It returns
  `u32`, and the ND anisotropic shift is order ½. `order()` is left at 1 with the
  honest statement in the gate, the ADR and math §32.5; truncating to 0 would
  change adaptive step control.
- **`GridFnND::sample` is still recursion-bound.** Hoisting the boundary
  resolution out of the collapse recursion (AMENDMENT 4) bought 1.32×, enough to
  restore `D=5`'s datum but not enough to make the `D ≥ 4` gates comfortable —
  `D=5` costs ≈ 4.3 h against a 6 h runner limit. The remaining cost is the
  recursion itself: one closure call per stencil node per level plus `K^D` leaf
  reads. Expanding the tensor stencil into `K^D` flat (offset, weight) pairs and
  summing them in one loop removes it, but **changes the summation order**, so
  every N-D gate's numbers move at ULP level and all of them need re-verification.
  Deliberately not done in the same release as the correctness fix.
- **`cargo clippy --features slow-tests` has a pre-existing backlog.** CI's gate
  is `cargo clippy --workspace --all-targets -- -D warnings`, which is clean;
  the `slow-tests`-gated integration tests are outside that surface and were
  never linted. A pass over them found ~100 findings in ~10 files, mostly
  `doc_markdown` / `similar_names` / `cast_*` on math prose, plus a handful of
  real ones (`RangeInclusive::contains`, `vec![]` over push-after-new,
  `#[ignore]` without reason). Blanket-allowing twelve lints per file was tried
  and reverted: it suppresses rather than fixes, and it collided with the
  targeted allows those files already carry. Either lint them properly or add
  the feature to CI's clippy invocation — not both halfway.

- **Boundary selection is absent from the C and WASM surfaces entirely**
  (ADR-0191 AMENDMENT 2). Every FFI constructor hard-codes
  `BoundaryPolicy::Reflect` and `semiflow-wasm` never calls `with_boundary` at
  all; this predates the campaign (`Shift1D` has had `boundary=` in Python for
  longer). Mirroring the new ND `boundary=` kwarg into just those two
  constructors would create an inconsistency rather than fix one, so it is not
  done here — exposing boundary selection across both surfaces is its own change.

## [0.12.1-beta] - 2026-07-15
### Fixed
- **docs(py):** de-stale the `semiflow-pde` PyPI README (long_description).
  Fixed broken PyPI badge, removed false "not yet published" note, relabelled
  obsolete internal "v9.0.0" markers to the public 0.9.0-beta scheme, and
  corrected the false "Tt/Gridless are Rust-only, not exposed via PyO3" claim
  (TtEvolver, TtState, TtCoupledEvolver, VarCoefTtEvolver, GridlessEvolver,
  MeasureState are exposed — ADR-0171/ADR-0178; class-reference rows added).
  Docs-only patch release to refresh the PyPI project page (9bac992).

## [0.12.0-beta] - 2026-07-02

### Added (#16)

- **`path="implicit"` on `SymmetricOperator.evolve_batched`, `mass_lumped_evolve`,
  and `MassKOperator.evolve`** (ADR-0190, math §59, branch
  `feat/issue-16-implicit-symmetric-operator`): an implicit backward-Euler /
  shift-invert action for the externally-assembled symmetric-operator path.
  Computes `e^{−tA}v ≈ (I+Δt·A)^{−n_steps} v` via preconditioned Conjugate Gradient
  (PCG) with Jacobi preconditioning, dependency-free (no new crate; governance budget
  unchanged). The `+I` shift makes the per-step system `S = I + Δt·Â` symmetric
  positive-definite for any `Δt > 0` — CG is always well-posed and converges (§59.3).
  `n_steps` (default 100) is the number of backward-Euler sub-steps; accuracy is
  O(Δt) = O(t/n_steps), so increase `n_steps` to tighten tolerance.

  **When to use**: stiff FEM or Robin-BC operators where `λ_max ≳ 10⁵ /s` causes
  the explicit `path="lanczos"` / `path="chebyshev"` to sub-step O(τλ_max) times
  and time out. `scipy.sparse.linalg.expm_multiply` has the same cost ceiling.

  **Cost caveat**: governed by `√κ` of the Jacobi-preconditioned `S`, not strictly
  `λ_max`-independent. IC(0) (zero-fill incomplete Cholesky, drop-in stronger
  preconditioner) is specified as §59.6 follow-on work and deferred.

  Gates: `G_SYMOP_IMPLICIT_DENSE` ≤ 1e-9 (well-conditioned Poisson-like operator);
  `G_SYMOP_IMPLICIT_STIFF` ≤ 1e-9 (Neumann N=400 ×1e7, surviving-mode reference);
  `G_SYMOP_IMPLICIT_PCG_SPD` structural (CG non-breakdown proved by SPD shift).

  Empirical (QA, stiff Neumann N=400 ×1e7): `path="implicit"` returns in ~4 ms at
  sup_error 3.3e-12 vs the analytic surviving-mode reference; `path="lanczos"` takes
  ~56 s; `scipy.sparse.linalg.expm_multiply` does not return within 90 s. In a
  well-conditioned regime (`path="implicit"` matches `scipy.linalg.expm` to ~1.6e-9).
  Memory ≈ 32 MB, ~2× below scipy's dense path.

### Fixed (CI hardening)

- **Flagship Gates**: Resolvent argument parsing and bounded `L_RESOLVENT` tick budget
  enforcement turned green on main. Nightly miri scalar path (`src/scalar/`) validated.
- **Nightly CI**: Bench-regression detection scoped to core benchmark families +
  baseline guard to prevent false positives from external library version drift.

## [0.11.0-beta] — 2026-06-27

Python binding surface for the 0.10.0-beta feature wave (#11–#14 + A1).
Issues #11, #12, #13, #14, and the graph Krylov additions from the 0.10.x patch
series are now usable from Python via the `semiflow-pde` wheel. No breaking
changes to the existing Python surface or the Rust public API.

### Added (Python bindings, #15)

- **`SymmetricOperator.from_csr` / `SymmetricOperator.evolve_batched`** —
  externally-assembled symmetric PSD sparse operators (FEM stiffness, Robin BC)
  accepted as CSR arrays; batched time-stepping across multiple right-hand sides.
- **`symmetric_op_expmv_frechet`** — combined matrix-exponential action +
  per-entry Fréchet gradient in one call. `EntrySensitivity` is not exposed as a
  separate Python class; the gradient is returned directly from this function.
- **`MassKOperator` / `mass_lumped_evolve`** — generalized-eigenproblem `(M, K)`
  propagation via the congruence chain `Â = R⁻ᵀKR⁻¹`; lumped-mass 3-liner path.
- **`ConservativeDiffusionChernoff` / `assemble_conservative_csr_1d`** —
  conservative divergence-form diffusion with harmonic-mean face conductivities;
  `to_symmetric_operator()` bridge to the Krylov engine.
- **`phi_action` / `phi_action_batched` / `Etdrk4`** — ETD φ-functions and the
  Cox–Matthews ETDRK4 semilinear integrator. Nonlinearity is menu-based
  (`AllenCahn`, `Burgers`, `GrayScott`, `KuramotoSivashinsky`); arbitrary
  per-step Python callbacks are deferred (ADR-0179 / ADR-0189 wall preserved).
- **`Laplacian.from_csr`** — direct CSR-based Laplacian construction path.
- PEP 561 stubs updated to cover all new symbols; 16/16 smoke tests green.

### Changed

- Registry metadata: `Changelog` and `Release Notes` links now appear on the
  PyPI sidebar (`semiflow-pde` `[project.urls]`), the crates.io `semiflow` page
  (README `## Changelog` section), and the npm `@semiflow/wasm` page
  (README `## Changelog / Release notes` section).

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
