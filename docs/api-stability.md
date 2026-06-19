---
version: 1.4.0
last_updated: 2026-06-19
freshness_score: 1.0
dependencies:
  - docs/adr/0028-ffi-pyo3-wasm-v0_10.md
  - docs/adr/0033-nonseparable2d-deprecation-policy.md
  - docs/adr/0034-with-closure-api.md
  - docs/adr/0035-breaking-window-policy.md
  - docs/adr/0098-robin-bc.md
  - docs/adr/0100-ladder-rung-trait.md
  - docs/adr/0103-subordinated-chernoff.md
  - docs/adr/0104-chebyshev-breaking-redesign.md
  - docs/adr/0106-g-zeta4-gr2025-prefactor.md
  - docs/adr/0154-binding-parity-v9.md
  - docs/adr/0155-gridless-chernoff.md
  - docs/adr/0156-reverse-ad.md
  - docs/adr/0159-tt-chernoff.md
  - docs/adr/0162-tt-coupled-spectral.md
  - docs/adr/0169-s3-honest-scope-public-api-promotion.md
  - Cargo.toml
  - crates/semiflow-core/src/lib.rs
changelog:
  - 1.0.0: Initial API stability policy, effective at v1.0.0 release
  - 1.1.0: Add §11 surface changes v4.6.0–v5.0.0 (OobPolicy, ChebyshevSpectralWithBC, Grid1D::cheb_m, SubordinatedChernoff, LevySubordinator, LadderRung, Robin BC; schema 2.0.0)
  - 1.2.0: Add v9.0.0 ADDITIVE surface (ReverseChernoff, CheckpointSchedule, TtChernoff, TtState, GridlessChernoff, ParticleReduction; ReverseHeat1D bindings for PyO3 + WASM)
  - 1.3.0: Add v9.1.0 ADDITIVE surface (CoupledTtChernoff; ADR-0162)
  - 1.4.0: Add v9.2.0 ADDITIVE surface (six S3* types behind s3-poc; AxisCoef, CpTerm, CpCoef, CoefRole, Reaction promoted; ADR-0169; schema 4.14.0/4.15.0)
---

# API Stability Policy

**Effective from**: v1.0.0
**Scope**: All four public surfaces — `semiflow-core` Rust rlib, `semiflow-ffi`
C ABI, `semiflow-py` PyO3 wheel, `semiflow-wasm` npm package.
**Cross-refs**: ADR-0028 §"API stability", ADR-0033, ADR-0034, ADR-0035
(v1.0.0 freeze inventory; written concurrently).

---

## 1. Status and Scope

This document is the authoritative stability contract for the semiflow-core
workspace after the v1.0.0 tag. It supersedes the pre-1.0 "research preview"
disclaimer in `README.md` (which is removed at v1.0.0 by task S2.7).

Coverage:

| Crate | Artifact | Distribution |
|-------|----------|-------------|
| `semiflow-core` | `rlib` | crates.io |
| `semiflow-ffi` | `cdylib + staticlib + remizov.h` | GitHub Releases |
| `semiflow-py` | `abi3-py310` wheel | PyPI / GitHub Releases |
| `semiflow-wasm` | `wasm-pack` npm package | npm registry |

All four surfaces are frozen simultaneously at v1.0.0.

---

## 2. SemVer Commitment

The workspace follows [Semantic Versioning 2.0.0](https://semver.org/) for
all four crates in lockstep. A single version number applies to the workspace.

**MAJOR (X.0.0)** — breaking changes to any covered public API are allowed
and required. A MAJOR bump signals: removed or renamed public items, changed
function signatures, behaviour changes that violate documented contracts
(e.g. a different convergence guarantee, a changed error variant), or removal
of a `#[deprecated]` item.

**MINOR (1.X.0)** — strictly additive. New `pub` items, new constructors,
new methods on existing types, new `SemiflowError` variants for previously
unhandled conditions. No existing covered public item changes shape or
behaviour.

**PATCH (1.0.X)** — bug fixes, documentation corrections, performance
improvements that do not change documented behaviour. No API changes of any
kind. Panic message text may change in a PATCH.

All four crates bump in lockstep: a MINOR change to `semiflow-core` that has
no binding impact still increments the workspace version. Crates are not
independently versioned after v1.0.0.

---

## 3. MSRV Policy

**v1.0.0 MSRV**: Rust **1.78**, as declared in `[workspace.package]
rust-version = "1.78"` in the root `Cargo.toml`.

MSRV increases are **MINOR** version bumps, not MAJOR. Rationale: Rust's
compatibility model guarantees that older compilers reject code using newer
language features with a clear diagnostic; no silent runtime regression
occurs. A MINOR bump gives downstream users advance notice via semver before
the window where the old compiler would fail.

Commitment: the project supports MSRV over at least the trailing
six-month stable-release window. The window is computed from the date of
the MINOR tag, not from the date of the upstream compiler release.

MSRV will not be raised solely to access convenience syntax. If a new
feature is only available on a newer compiler, the benefit must be
substantive (e.g. a required compiler bug fix, a significant performance
improvement, or an unsafe soundness issue). The rationale is recorded in
the CHANGELOG under a `### Changed` heading.

---

## 4. Cross-Binding Synchronization

`semiflow-core` is the source of truth. Binding crates (`semiflow-ffi`,
`semiflow-py`, `semiflow-wasm`) mirror it and may lag by at most one MINOR
release; they may not lead.

Rules:

- A **MAJOR** change to `semiflow-core` triggers simultaneous MAJOR bumps
  for all four crates, regardless of whether the binding surfaces are
  directly affected. Bindings are locked to the core's API contract.

- A **MINOR** addition to `semiflow-core` that is not yet reflected in any
  binding is permitted. The binding catches up at the next MINOR bump.

- A **MINOR** addition partially mirrored across bindings is permitted
  (e.g. FFI mirrors a new constructor but PyO3 does not yet). The binding
  that lags notes the gap in its own `CHANGELOG` entry.

- Binding crates are not published to crates.io (`publish = false` in their
  `Cargo.toml`). Their stability is enforced via GitHub Releases and the
  npm/PyPI artifact, not via crates.io yanking. A MAJOR bump to the
  workspace version is the signal to consumers of all four surfaces.

Note on C ABI: `semiflow-ffi` does not make guarantees about ABI
compatibility across different Rust toolchain versions or different host
platforms. The C API types (`SemiflowStatus`, opaque handle, function
signatures in `remizov.h`) are covered by this policy; the physical ABI of
the compiled `cdylib` is a property of the Rust compiler and is not frozen
by this document.

---

## 5. Deprecation Cycle

Items targeted for removal follow this lifecycle:

1. Mark with `#[deprecated(since = "1.x.y", note = "...")]` in a MINOR
   release.
2. The deprecated item remains callable and produces no compile error (only
   a warning) for the entire v1.x.y lineage.
3. Removal is only allowed at a MAJOR bump (2.0.0 or later).

**Exception — experimental surface**: items carrying `#[doc(hidden)]` or
explicitly labelled "experimental" or "unstable" in their rustdoc are not
covered by this guarantee and may change or be removed in any MINOR release.
Such items will be listed in the release CHANGELOG.

**ADR-0033 cross-reference**: both `NonSeparable2DChernoff<X, Y, F>` and
`NonSeparable2DAnisotropicChernoff<X, Y, F>` are first-class public APIs at
v1.0.0 with no deprecation marker. ADR-0033 records the rationale; the
summary is that scalar-c is the correct, simpler API for isotropic callers
and the aniso type is an additive sibling, not a replacement. Neither will
be deprecated unless a future caller-survey at a MAJOR milestone reveals no
active users.

---

## 6. What Is Not Covered

The following are explicitly outside the stability guarantee:

- **Internal modules**: items declared `pub(crate)` or in `mod`s not
  re-exported from `lib.rs`.

- **Private fields of public structs**: e.g. the fields of `Grid1D`, which
  are accessed only via constructor and accessor methods.

- **`#[doc(hidden)]` items**: any item carrying this attribute.

- **Performance characteristics**: convergence rates, wallclock times, and
  memory usage are not contractual. `docs/perf-commitment-v1_0_0.md` (task
  S2.3) publishes reference baselines; those are informational, not
  enforceable. A PATCH may improve throughput without notice.

- **Panic message text**: `SemiflowError` variants (their discriminants and
  associated data shapes) are covered. The human-readable string returned by
  `Display` or by a Rust `panic!` on contract violation is not.

- **WASM threading assumption**: `semiflow-wasm` currently ships for
  `wasm32-unknown-unknown` (single-threaded). If `wasm32` multi-threading
  becomes standard and the WASM binding's `unsafe impl Send + Sync` for
  JS-callback types requires revision, that is a MAJOR bump.

- **Heavy-test slope gates**: the numerical slope thresholds in
  `#[ignore]` convergence tests (e.g. G3⁶-2D, G5_3D) are calibrated per
  release. Tightening a gate in response to improved hardware or algorithms
  is not a breaking change.

- **xtask binary interface**: xtask is a build helper, not a public API.
  Its subcommands may change in any release without notice.

- **`#[cfg(feature = "...")]` gating**: new feature flags may be added in
  MINOR releases. Existing feature flags will not be renamed or removed
  without a MAJOR bump.

---

## 7. Surfaces Frozen at v1.0.0

The following are covered by this policy starting at v1.0.0:

**`semiflow-core` Rust rlib** — all `pub use` re-exports in
`crates/semiflow-core/src/lib.rs` (lines 134–157 at v1.0.0), which include:

| Export | Module |
|--------|--------|
| `AdaptiveOutcome`, `AdaptivePI` | `adaptive` |
| `Axis`, `AxisLift` | `axis` |
| `ChernoffFunction`, `ChernoffSemigroup` | `chernoff` |
| `DiffusionChernoff` | `diffusion` |
| `Diffusion4thChernoff` | `diffusion4` |
| `Diffusion6thChernoff` | `diffusion6` |
| `DriftReactionChernoff` | `drift_reaction` |
| `SemiflowError` | `error` |
| `SemiflowFloat` | `float` |
| `BoundaryPolicy`, `Grid1D`, `InterpKind` | `grid` |
| `Grid2D` | `grid2d` |
| `Grid3D` | `grid3d` |
| `GridFn1D` | `grid_fn` |
| `GridFn2D` | `grid_fn2d` |
| `GridFn3D` | `grid_fn3d` |
| `TruncatedExpDiffusionChernoff` | `truncated_exp` |
| `TruncatedExp4thDiffusionChernoff` | `truncated_exp4` |
| `ShiftChernoff1D` | `shift1d` |
| `State` | `state` |
| `StrangSplit` | `strang` |
| `NonSeparable2DChernoff` | `nonseparable2d` |
| `NonSeparable2DAnisotropicChernoff` | `nonseparable2d_aniso` |
| `Strang2D` | `strang2d` |
| `AxisLift3D`, `Strang3D` | `strang3d` |

Feature-gated re-exports (`parallel`, `simd`) are covered when those
features are enabled; their gating attribute is not considered a breaking
change.

**`semiflow-ffi` C ABI** — all `extern "C"` functions in
`crates/semiflow-ffi/src/ffi.rs` and all declarations in the
cbindgen-generated `crates/semiflow-ffi/include/remizov.h`, including the
`SemiflowStatus` enum and the opaque `SemiflowState` handle.

**`semiflow-py` PyO3 wheel** — all `#[pyclass]` types and `#[pymethods]`
blocks in `crates/semiflow-py/src/`, including `Heat1D` and `SemiflowError`.

**`semiflow-wasm` npm package** — all `#[wasm_bindgen]` items in
`crates/semiflow-wasm/src/`, including the `Heat1D` JS class and
`panic_hook_init`.

Items demoted to experimental during the S2.2 audit (task S2.2, concurrent
with this document) will be listed in ADR-0035 and excluded from this table.

**Field-freeze footnote (S2.2 audit recommendation)**: Four public structs
expose their fields directly as part of the intended API contract. The *field
names* (not just the type names) of the following structs are covered by this
freeze — renaming any field is a MAJOR bump:

| Struct | Public fields covered |
|--------|-----------------------|
| `AdaptivePI` | `func`, `tol_abs`, `tol_rel`, `safety`, `alpha`, `beta`, `min_ratio`, `max_ratio`, `max_substeps` |
| `AdaptiveOutcome` | `t_final`, `steps_taken`, `substeps_total`, `converged` |
| `ShiftChernoff1D` | `a`, `b`, `c`, `c_norm_bound`, `grid` |
| `NonSeparable2DChernoff` | `x`, `y`, `c`, `c_norm_bound`, `grid` |

These fields are user-tunable knobs (or required constructor state), documented
in their respective ADRs. Accessing them directly is supported usage.

---

## 8. Surfaces Not Frozen at v1.0.0

The following are explicitly experimental or internal at v1.0.0:

- **`mod diffusion_storage`** (`pub(crate)` — the `Storage<F>` enum backing
  `DiffusionChernoff`; its layout and variants may change without notice).

- **`mod grid_cubic`** and **`mod grid_quintic`** (internal interpolation
  kernels; not re-exported from `lib.rs`; subject to algorithmic revision).

- **`mod simd`** (feature-gated, intended as a platform acceleration layer;
  intrinsic availability and the dispatch logic may change across toolchain
  versions).

- **`mod strang2d_parallel`** and **`mod strang3d_parallel`** (feature-gated
  `parallel`; the scheduling model may change with new Rust threading
  primitives; the public-facing interface exported via `Strang2D` / `Strang3D`
  is covered, but the parallel module's own items are not).

- **Any item listed as `#[doc(hidden)]`** (verified per the S2.2 audit).

- **`xtask`** (build helper binary; not part of the library API).

- **Test binaries and benchmark harnesses** (not part of the distributed
  library).

---

## 9. Pre-1.0 SemVer Record

Before v1.0.0, the workspace operated under pre-1.0 SemVer conventions where
MINOR bumps were allowed to carry breaking changes. The breaking changes made
during that period are:

- **v0.3.0**: `DiffusionChernoff::new` constructor extended from 3 to 5
  arguments (`a_prime` and `a_double_prime` inserted). Existing callers must
  pass `|_| 0.0_f64` for constant `a`. (CHANGELOG §"Changed (BREAKING —
  SemVer minor in pre-1.0)")

- **v0.7.0**: `MagnusDiffusionChernoff` renamed to
  `TruncatedExpDiffusionChernoff`; `Magnus4thDiffusionChernoff` renamed to
  `TruncatedExp4thDiffusionChernoff`; associated constants renamed (no value
  change). Clean-break rename per ADR-0013 Amendment 2. (CHANGELOG §"Changed
  — BREAKING (D2 full fix)")

- **v0.12.0** (planned, per ADR-0034): `DiffusionChernoff` loses the `Copy`
  auto-trait. `Clone` is preserved. Migration: replace implicit copy sites
  with `.clone()`. The `with_closure` sibling constructor is added
  simultaneously (additive).

All other changes in v0.x.y releases were additive or restricted to
internal implementation details. The full history of `### Changed` and
`### Removed` entries is in `CHANGELOG.md`.

---

## 10. Process for Changes After v1.0.0

**Breaking change (MAJOR)**:

1. Open an ADR (`docs/adr/NNNN-*.md`) proposing the change, the rationale,
   and the migration path for downstream callers.
2. Add a `### Changed` or `### Removed` entry to `CHANGELOG.md` describing
   what breaks and how to migrate.
3. Bump the workspace version to the next MAJOR.

**Additive change (MINOR)**:

1. Add a `### Added` entry to `CHANGELOG.md`.
2. Bump the workspace version to the next MINOR.
3. A new ADR is not required if the addition is covered by an existing ADR
   (e.g. extending the `with_closure` pattern to a new Chernoff type per
   ADR-0034 is MINOR additive work, no new ADR needed unless the design
   diverges from ADR-0034's §"Composition types — sequencing strategy").

**Bug fix / documentation / performance (PATCH)**:

1. Add a `### Fixed` entry to `CHANGELOG.md`.
2. Bump the workspace version to the next PATCH.
3. No ADR required.

**MSRV bump (MINOR)**:

1. Update `rust-version` in `Cargo.toml`.
2. Add a `### Changed` entry citing the new MSRV and the rationale.
3. Bump the workspace version to the next MINOR.

Reviewers gate every MAJOR on: (a) ADR present, (b) CHANGELOG updated,
(c) migration path documented, (d) `#[deprecated]` markers in place for the
previous MINOR if the removal is of a previously-public item.

---

## 11. Surface Changes by Release (post-v1.0.0)

### v5.0.0 — BREAKING (2026-05-29, commit 1ba9960)

**Deprecated items (removal target: v6.0.0, ADR-0035 §9):**

| Item | Kind | Deprecation date | Note |
|------|------|-----------------|------|
| `InterpKind::ChebyshevSpectral { m }` | enum variant | 2026-05-29 | Use `ChebyshevSpectralWithBC { m, oob_policy }` |

**New public items (additive, stable):**

| Item | Kind | Location | ADR |
|------|------|----------|-----|
| `OobPolicy` | `pub enum` (4 variants: `Inherit`, `ForceReflect`, `ForcePeriodic`, `ForceZero`) | `boundary.rs` / `lib.rs` re-export | ADR-0104 |
| `InterpKind::ChebyshevSpectralWithBC { m, oob_policy }` | enum variant | `boundary.rs` | ADR-0104 |
| `Grid1D::cheb_m(xmin, xmax, n, m)` | constructor | `grid.rs` | ADR-0104 |
| `out_of_domain_sample<F>` | `pub fn` | `grid_chebyshev.rs` | ADR-0104 |

**Schema version bumps:**

| Schema | Old | New |
|--------|-----|-----|
| `contracts/semiflow-core.properties.yaml` `schema_version` | 1.6.0 | 2.0.0 (MAJOR) |
| `contracts/semiflow-core.traits.yaml` `schema_version` | 2.2.0 | 2.3.0 |

**Sympy oracles (NORMATIVE at v5.0.0):**

| Gate | Result |
|------|--------|
| `T_GR_2025_THM3` | 5/5 NORMATIVE (ADR-0106 — G_zeta4 escalation RESOLVED) |
| `T_CHEBYSHEV_WEIGHTS` | 2/2 NORMATIVE (ADR-0104) |

### v4.8.0 — ADDITIVE (2026-05-29, commit 346e372)

**New public items:**

| Item | Kind | ADR |
|------|------|-----|
| `SubordinatedChernoff<C, S, F>` | `pub struct` implementing `ChernoffFunction<F>` | ADR-0103 |
| `LevySubordinator<F>` | `pub trait` | ADR-0103 |
| `StableSubordinator` | `pub struct` implementing `LevySubordinator<f64>` | ADR-0103 |
| `GammaSubordinator` | `pub struct` implementing `LevySubordinator<f64>` | ADR-0103 |
| `InverseGaussianSubordinator` | `pub struct` implementing `LevySubordinator<f64>` | ADR-0103 |

**Sympy oracle (NORMATIVE):** `T_SUBORD` 5/5 NORMATIVE (ADR-0103).

### v4.7.0 — ADDITIVE (2026-05-29, commit 05ccd0a)

**New public items:**

| Item | Kind | ADR |
|------|------|-----|
| `LadderRung<const K: usize, F>` | sealed `pub trait` super-trait | ADR-0100 |

Impls: `DiffusionChernoff` (K=2), `Diffusion4thZeta4Chernoff` (K=4),
`Diffusion6thZeta6Chernoff` (K=6), `Diffusion8thZeta8Chernoff` (K=8, experimental).

**Sympy oracle (NORMATIVE):** `T_LADDER_RUNG` 4/4 NORMATIVE (ADR-0100).

### v9.0.0 — ADDITIVE (2026-06-10)

No removals, no deprecations, no signature changes. All six new public types are
strictly additive. Cross-refs: ADR-0154 (binding parity), ADR-0155
(GridlessChernoff), ADR-0156 (ReverseChernoff), ADR-0159 (TtChernoff).

**New public items in `semiflow-core` Rust rlib:**

| Item | Kind | Module | ADR |
|------|------|--------|-----|
| `ReverseChernoff<F>` | `pub struct` | `reverse_ad` | ADR-0156 |
| `CheckpointSchedule` | `pub struct` | `reverse_ad` | ADR-0156 |
| `TtChernoff<F>` | `pub struct` implementing `step` / `evolve` | `tt_chernoff` | ADR-0159 |
| `TtState<F>` | `pub struct` | `tt_chernoff` | ADR-0159 |
| `GridlessChernoff<F, const D: usize>` | `pub struct` implementing `ChernoffFunction<F>` | `gridless` | ADR-0155 |
| `ParticleReduction` | `pub enum` (2 variants: `WeightedVoronoi { cap }`, `GaussianBackground` (stub)) | `gridless_reduce` | ADR-0155 |

**Binding surface changes (ADR-0154):**

| Binding | New item | Method(s) |
|---------|----------|-----------|
| `semiflow-py` PyO3 wheel | `ReverseHeat1D` | `.value_and_grad(tau, u0, target) -> (float, float)` |
| `semiflow-wasm` npm package | `ReverseHeat1D` | `.valueAndGrad(tau, u0, target) -> Float64Array[value, grad]` |

`TtChernoff`, `TtState`, `GridlessChernoff`, and `ParticleReduction` are
**Rust-only at v9.0.0** — not exposed via FFI, PyO3, or WASM.
`semiflow-ffi` C ABI surface is **unchanged**.

**Narrow-scope markers (normative):**

- `ReverseChernoff<F>`: constant-a `DiffusionChernoff` ONLY (§51.5).
  Variable-coefficient and nonlinear kernels are out of scope.
- `TtChernoff<F>`: linear diagonal-A Gaussian class ONLY (§52, ADR-0159).
  Off-diagonal A, variable/nonlinear coefficients, non-Gaussian IC: rank not
  algebraically capped, may blow up — research-track.
- `GridlessChernoff<F, D>`: diagonal A, constant scalar coefficients, d ≤ ~10.
  Off-diagonal Cholesky / variable coefficients / d > 10: research-track (ADR-0155 §50.7).

**Convergence gate (NORMATIVE at v9.0.0):**

| Gate | Status |
|------|--------|
| `G_BINDING_REVERSE_AD_PARITY` | 0-ULP parity between PyO3 and WASM `ReverseHeat1D` implementations |

**Schema version bumps:** none at v9.0.0 (no trait or property schema changes).

---

### v9.1.0 — ADDITIVE (2026-06-11, commit 5069d85)

No removals, no deprecations, no signature changes. One new public type; all existing
types and their methods are unchanged. Cross-ref: ADR-0162 (`CoupledTtChernoff`).

**New public items in `semiflow-core` Rust rlib:**

| Item | Kind | Module | ADR |
|------|------|--------|-----|
| `CoupledTtChernoff<F>` | `pub struct` implementing `step` / `evolve` (no solver on coupling path) | `tt_coupled` | ADR-0162 |

**Narrow-scope marker (normative):**

- `CoupledTtChernoff<F>`: constant-coefficient correlated-Gaussian / linear cross-diffusion,
  adjacent pairs only (`b=0`, constant `ρ_{jk}`, `|ρ|<1`). Drift advection (`b≠0`) and
  non-adjacent pairs are rejected fail-loud at construction. Variable-coefficient and
  nonlinear paths deferred.

**Convergence gate (NORMATIVE at v9.1.0):**

| Gate | Status |
|------|--------|
| `G_TT_COUPLED_EXACT` | ≤1e-12 vs dense `expm(τL_h^{dx})` for `d∈{3,4}` (RELEASE-BLOCKING, `slow-tests`) |
| `T_TT_BAND_SHIFT_RANK` | band-shift QTT-op-rank ≤ 3, constant in grid resolution and `d` (RELEASE-BLOCKING, `test-fast`) |

**Binding surface changes:** none (Rust-only at v9.1.0).
`semiflow-ffi`, `semiflow-py`, `semiflow-wasm` surfaces are **unchanged**.

**Schema version bumps:** none at v9.1.0 (no trait or property schema changes required
by the additive `CoupledTtChernoff` type; gates reference existing property schema
entries).

---

### v9.2.0 — ADDITIVE (2026-06-19)

No removals, no deprecations, no signature changes to existing public items. All new
public types are behind the non-default `s3-poc` cargo feature — default builds are
completely unaffected. Cross-ref: ADR-0169.

**Stability tier for `s3-poc` surface:** POC / research-track experimental. Items are
public (not `#[doc(hidden)]`) and follow MINOR/MAJOR SemVer for the `s3-poc` surface,
but carry explicit "Proven boundary" narrow-scope documentation and are expected to
evolve as the research matures. The feature flag itself will not be removed without a
MAJOR bump.

**New public items in `semiflow-core` Rust rlib (all `#[cfg(feature = "s3-poc")]`):**

| Item | Kind | Module | ADR |
|------|------|--------|-----|
| `S3DriftSpectralEvolver<F>` | `pub struct` | `tt_drift_spectral` | ADR-0164, §53.1 |
| `S3DenseCouplingEvolver<F>` | `pub struct` | `tt_dense_coupling` or `tt_dense_coupling_api` | ADR-0165, §53.2 |
| `S3VarCoefEvolver<F>` | `pub struct` | `tt_varcoef_spectral` | ADR-0166, §53.3 |
| `AxisCoef<F>` | `pub struct` (container; was `pub(crate)`) | `tt_varcoef_spectral` | ADR-0166, §53.3 |
| `S3NonSepVarCoefEvolver<F>` | `pub struct` | `tt_nonsep_varcoef` or `tt_nonsep_varcoef_api` | ADR-0167, §53.4 |
| `CpTerm<F>` | `pub struct` (container; was `pub(crate)`) | `tt_nonsep_varcoef` | ADR-0167, §53.4 |
| `CpCoef<F>` | `pub struct` (container; was `pub(crate)`) | `tt_nonsep_varcoef` | ADR-0167, §53.4 |
| `CoefRole` | `pub enum` (container; was `pub(crate)`) | `tt_nonsep_varcoef` | ADR-0167, §53.4 |
| `S3BurgersColeHopf<F>` | `pub struct` | `tt_nonlinear_spectral` | ADR-0168, §53.5 |
| `S3ReactionDiffusion<F>` | `pub struct` | `tt_nonlinear_spectral` | ADR-0168, §53.5 |
| `Reaction<F>` | `pub enum #[non_exhaustive]` (container; was `pub(crate)`) | `tt_nonlinear_spectral` | ADR-0168, §53.5 |

All constructors return `Result<Self, SemiflowError>` and validate in-class membership
at construction time. Out-of-class inputs are unconstructible by type (type wall) or
fail loud at construction. Raw `pub(crate)` free functions remain private.

**Narrow-scope markers (normative — see `## Proven boundary` rustdoc on each type):**

| Type | Proven class | Proven wall |
|------|-------------|-------------|
| `S3DriftSpectralEvolver` | Constant-coef diffusion+drift | Variable coef out-of-class |
| `S3DenseCouplingEvolver` | Rank-1-dense `D = diag(a) + λ·g·gᵀ` | Generic full-rank `D` = info-theoretic wall |
| `S3VarCoefEvolver` | Additive-separable `L = Σⱼ Lⱼ` | Non-separable `a(x,y)` unrepresentable by `AxisCoef` |
| `S3NonSepVarCoefEvolver` | Low-CP-rank fixed-`m` coefficients | Generic full-CP-rank = CP-rank wall |
| `S3BurgersColeHopf` | 1-D viscous Burgers | Higher-D or non-Burgers = mode-mixing wall |
| `S3ReactionDiffusion<F>` | Polynomial/logistic `Reaction` enum | Transcendental `f(u)` = mode-mixing wall |

**Convergence gates (NORMATIVE at v9.2.0, all RELEASE-BLOCKING under `slow-tests`,
independent of `s3-poc` — gates re-implement the algorithm locally):**

| Gate | Status |
|------|--------|
| `g_s3_drift_spectral` | exactness ≤1e-12 vs dense Padé `expm`; Δrank-preservation under drift |
| `g_s3_dense_coupling` | non-vacuous rank-2 contrast proving the info-theoretic full-rank wall |
| `g_s3_varcoef_spectral` | slope ≤ −1.95 (order-2); wrong-operator floor documents the out-of-class boundary |
| `g_s3_nonsep_varcoef` | slope ≤ −1.95 on `cos(x)sin(y)·∂²ₓ` (the §53.3 floor case) |
| `g_s3_nonlinear` | exactness gate (Cole-Hopf) + slope gate (Strang-split) |

**Binding surface changes:** none (Rust-only at v9.2.0 — the `s3-poc` surface is not
exposed via FFI, PyO3, or WASM). `semiflow-ffi`, `semiflow-py`, `semiflow-wasm` surfaces
are **unchanged**.

**Schema version bumps:**

| Schema | Old | New | Reason |
|--------|-----|-----|--------|
| `contracts/semiflow-core.traits.yaml` | 4.13.0 | 4.14.0 | MINOR — additive: six new S³ public types + `Result`-constructor contracts |
| `contracts/semiflow-core.properties.yaml` | 4.14.0 | 4.15.0 | MINOR — additive: `## Proven boundary` properties + five `g_s3_*` gate cross-references |

---

### v4.6.0 — ADDITIVE (2026-05-29, commit ea2d2a6)

**New public items:**

| Item | Kind | ADR |
|------|------|-----|
| `BoundaryPolicy::Robin { alpha: F, beta: F }` | enum variant | ADR-0098 |

**Sympy oracle (NORMATIVE):** `T_ROBIN` 4/4 PASS (ADR-0098 AMENDMENT 1).
