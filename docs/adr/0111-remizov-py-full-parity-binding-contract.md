# ADR-0111 — `semiflow-py` full-parity binding contract

**Status**: Accepted (planning ADR; gap-closing, multi-wave)
**Date**: 2026-05-30
**Authors**: ai-solutions-architect
**Cross-refs**: ADR-0028 (FFI/PyO3/WASM sibling boundary — PyO3 depends on `semiflow-core`
directly, f64-only, experimental-until-v1.0), ADR-0031 (three-phase GIL-release),
ADR-0034 (Python-callable coefficient closures), ADR-0057 (Schrödinger unitarity),
ADR-0076 (v3 additive surface), constitution v2.0.0 Override #2 (MCP WAIVED — a
binding has no runtime to introspect; this contract adds none).

## Context

`semiflow-py` is **not** a Heat1D stub: it already exposes 26 pyclasses + 1 free
function across 19 source modules, covering ~13 of `semiflow-core`'s ~55 public
kernel/type families. The maintainer wants **full parity** so Python users reach
the whole kernel surface. This is **gap-closing**, not greenfield. This ADR fixes
the *contract* (dispatch, numpy I/O, error mapping, GIL, naming, boundary) so the
remaining waves are mechanical; the per-wave sequencing lives in
`.dev-docs/reports/SEMIFLOW_PY_FULL_PARITY.md`.

## Decision

**1. Dispatch strategy — per-type pyclass mirroring, NOT a generic `Evolver` wrapper.**
The existing surface is one `#[pyclass]` per concrete kernel (`Heat1D4th`,
`Schrodinger1D`, `GraphHeat6`, …). `ChernoffFunction`/`Evolver` are Rust generic
traits with no single concrete instantiation crossable over the f64-only PyO3
boundary (ADR-0028); a generic wrapper would require type erasure that defeats the
`static_assertions` `Send+Sync` proofs in `send_assertions.rs`. New families follow
the established pattern: a private `*Inner` struct (Mutex-wrapped only when the core
kernel is `!Sync`, as in `schrodinger.rs`), a `#[pyclass(name = "…")]`, and a
module-level `pub fn register(py, m)` called from `lib.rs` `#[pymodule]` (mirror
`hormander_py::register`, `v3::register`). One module per missing family group.

**2. numpy array I/O conventions (NORMATIVE).** Input: accept any 1-D
`numpy.ndarray[float64]` or float sequence via `extract_f64_slice` (`state_1d.rs`);
2-D/3-D states accept flat row-major `float64` arrays of length `nx·ny`(`·nz`) with
x as the fast axis (existing `Heat2D`/`Heat3D` convention — DO NOT introduce 2-D
numpy shapes). Complex states accept `numpy.ndarray[complex128]` (existing
`Schrodinger1D` precedent via `numpy::Complex64`) — this is the binding pattern for
ALL complex/Schrödinger-complex families. Output: `values()` returns a **copy**
(`to_pyarray`), dtype always matches input (`float64`/`complex128`), contiguous,
length == `len(self)`. Finite-checks: every input buffer is validated finite under
the GIL in Phase 1 (reject NaN/Inf → `kind='NanInf'`); never inside the
GIL-released window.

**3. Error mapping (NORMATIVE).** All fallible ops raise the single
`SemiflowError(Exception)` with a `.kind: str` discriminator, via
`error::from_core` (`GridMismatch`/`NanInf`/`OutOfDomain`/`BoundaryFailure`/
`CflViolated`/`ConvergenceFailed`/`Unsupported`/`Panic`). No per-family exception
classes. Every pymethod body is wrapped in `catch_panic_py!{…}` so Rust panics
surface as `kind='Panic'`, never UB across the boundary.

**4. GIL-release policy (NORMATIVE — ADR-0031).** Three-phase pattern for every
`evolve`: (1) validate + copy input to owned `Vec` under GIL; (2) pure-Rust compute
inside `py.detach`; (3) write result back under GIL. The detach closure must capture
only `Send+Sync` values; add a `static_assertions::assert_impl_all!` line in
`send_assertions.rs` for each new `ChernoffSemigroup<Kernel, State>`. Python-callable
coefficients (`make_coeff_closure`, ADR-0034) re-acquire the GIL per call and DEFEAT
detach — every variable-coefficient family MUST also ship a pre-sampled-array
constructor (`with_*_array`) as the performant path, mirroring `Heat1D::with_a_array`.

**5. abi3 single-wheel constraint.** All new pyclasses MUST compile under the
existing `abi3-py310` limited API (single wheel, Python 3.10–3.13). No `#[pyo3]`
feature requiring a version-specific ABI. No new direct PyO3-side dependency:
`pyo3` + `numpy` remain the only two (the `static_assertions` dev-dep is unchanged).

**6. Naming.** pyclass `name = "…"` uses the user-facing math name without the
`Chernoff` suffix where a friendlier name exists (`Heat1D4th`, not
`Diffusion4thChernoff`); the Rust struct is prefixed `Py…` only when the bare name
collides (`PyHypoellipticChernoffHeisenberg`, `PyGraph`). Free oracle functions keep
their core name (`heisenberg_heat_kernel`). New families pick names per the matrix in
the parity report; ASCII-only Rust identifiers (the core `SchrödingerChernoffComplex`
binds as `SchrodingerComplex1D`).

**7. no_std-core / std-binding boundary (re-affirms ADR-0028).** `semiflow-py`
depends on `semiflow-core` with `std`; it MUST NOT add features to or alter
`semiflow-core`. Binding-only helpers (numpy interop, closure factories) live in
`semiflow-py/src`, never upstreamed into the `no_std + alloc` core.

## Non-bindable exclusions (honesty over false 100%)

These core exports are **intentionally excluded** with justification (recorded in the
parity report's exclusion table): trait/marker types with no concrete instantiation
(`ChernoffFunction`, `Evolver`, `State`, `HilbertState`, `Discrete`, `SemiflowFloat`,
`ApproximationSubspace`/`LadderRung`, `VectorField`, `Sampleable`,
`TimedChernoffFunction`, `LevySubordinator`, `WeightAtTime`/`LaplacianAtTime`/
`SegmentWeightFn`); `f32` monomorphisations (ADR-0028 f64-only); internal scaffolding
(`ScratchPool`/`ScratchVec`, `Growth`, `compute_rho_bar`, `MAX_GRAPH_TRAJ_SEGMENTS`,
controllers `ClassicalPI`/`H211bFilter`/`StepController` — surfaced only via
`AdaptivePI`); `#[doc(hidden)]` test hooks (`drain_thread_local_pools`,
`ApplyChernoffExt`). These are mechanism, not user-facing kernels.

## Consequences

Engineer implements the parity report's waves P1–P7 in order; each wave is a single
new module + `register()` call + `xtask py-smoke` extension, independently shippable
under the experimental (pre-v1.0) API stability promise. The contract guarantees
no two waves diverge in error/GIL/numpy conventions. Known impl-wave hazard: the
T3-GL32 duplicated-constant defect in `scripts/verify_hormander_heisenberg.py:234`
(audit-backlog T3) must be fixed in the Hörmander-extension wave's oracle before its
smoke is trusted.
