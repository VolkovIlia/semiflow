---
version: 1.0.0
last_updated: 2026-05-22
freshness_score: 1.0
dependencies:
  - docs/adr/0061-python-parity-expansion.md
  - docs/adr/0055-adjoint-chernoff.md
  - docs/adr/0057-schrodinger-unitary.md
  - docs/adr/0058-nonseparable-aniso-unification.md
  - docs/adr/0059-graph-bindings.md
  - docs/audit-findings-v2_0_0.md
  - crates/semiflow-py/tests/
  - contracts/semiflow-core.math.md
changelog:
  - 1.0.0: Initial v2.3.0 Python parity audit. Status DRAFT — Phase 7 final
    acceptance (cross-binding parity + full test suite on release build) pending.
---

# v2.3.0 Python Parity Audit

**Branch**: `feat/python-parity-v2.3`
**Parent commit (v2.2.0 tag)**: `801166c`
**Phase 5 HEAD**: `dc45181`
**Date**: 2026-05-22
**Status**: **DRAFT** — fast-test suite passes (129/0 PASS at dc45181); Phase 7
final acceptance (cross-binding parity + `test-full` on production build) pending.
**Theme**: Python parity expansion — bring `semiflow-py` to full parity with the
Rust core surface across five implementation phases.

This document satisfies the "documentation in lockstep" principle and will be
updated to **APPROVED** status after Phase 7 cross-binding parity gate passes.

---

## 1. Audit Scope

### Python surface additions (Phases 1–5)

| Phase | Commit | Additions |
|-------|--------|-----------|
| 1 — Foundation | 7fc090a | `boundary` kwarg on `Heat1D/2D/3D`; `Heat1D.with_a_array`; `boundary.rs`, `coeff.rs` |
| 2 — 1D kernels | 2308ea6 | `Heat1D4th`, `Heat1D6th`, `DriftReaction1D`, `Shift1D`; `with_arrays` staticmethods |
| 3 — Schrödinger | a63ade3 | `Schrodinger1D` with 4 constructors, `evolve`, `values`, `values_parts`, `norm_squared` |
| 4 — Composition | 52d2695 | `NonSeparable2D`, `Adjoint`, `AdaptivePI`; `coeff2d.rs`; core closure additions |
| 5 — Graph expansion | dc45181 | `Graph` factories, `Laplacian` factories, `GraphHeat4th`, `VarCoefGraphHeat`, `MagnusGraphHeat6` |

### Rust-core additive changes (Phase 2, ADR-0034 pattern)

Four new constructor overloads added to `semiflow-core` (additive, no breaking
changes): `Diffusion4thChernoff::with_closure`, `Diffusion6thChernoff::with_closure`,
`DriftReactionChernoff::with_closure`. New module `diffusion_storage.rs`
(`Storage2` / `Storage3` enums, `pub(crate)`).

### Phase 4 core additions (ADR-0058 additive)

`nonseparable_mixed_closure.rs`: `NonSeparableMixedChernoff::with_closure_c` and
`with_closure_beta` (additive, no surface change to existing types).

---

## 2. Gate Results

### 2.1 Boundary policy gates (Phase 1)

| Gate | Kernel | Boundary | Result |
|------|--------|----------|--------|
| `test_boundary_reflect` | `Heat1D` | `'reflect'` | PASS — matches default oracle |
| `test_boundary_periodic` | `Heat1D` | `'periodic'` | PASS — Fourier solution at t=1 |
| `test_boundary_zero` | `Heat1D` | `'zero'` | PASS — Dirichlet Gaussian decay |
| `test_boundary_linear` | `Heat1D` | `'linear'` | PASS — linear-extrapolation far-field |
| `test_boundary_heat2d` | `Heat2D` | `'periodic'` | PASS |
| `test_boundary_heat3d` | `Heat3D` | `'periodic'` | PASS |

Pre-sampled coefficient gate: `Heat1D.with_a_array` vs `Heat1D.with_a_function`
sup-error ≤ 3 ULP for same input (cubic-Hermite interpolation parity, Phase 1
gate `test_coeff_parity`). PASS.

### 2.2 Convergence-rate gates (Phase 2)

| Gate | Kernel | Order | Slope threshold | Measured slope | Result |
|------|--------|-------|-----------------|----------------|--------|
| `test_heat1d4th_slope` | `Heat1D4th` | 4th | ≤ −3.5 | −3.97 | PASS |
| `test_heat1d6th_slope` | `Heat1D6th` | 6th | ≤ −5.5 | −5.82 | PASS |
| `test_drift_reaction_slope` | `DriftReaction1D` | 2nd | ≤ −1.95 | −2.01 | PASS |
| `test_shift1d_slope` | `Shift1D` | 2nd | ≤ −1.95 | −1.98 | PASS |

These slope thresholds follow the same calibration methodology as the v2.2.0
gates (G12–G20): log–log regression of sup-norm error vs number of Chernoff
steps.

### 2.3 Unitarity gate (Phase 3)

| Gate | Test | Threshold | Measured | Result |
|------|------|-----------|----------|--------|
| `test_schrodinger_unitarity` | Harmonic oscillator, 500 steps, harmonic V | `|‖ψ‖²/‖ψ₀‖² − 1| < 1e-6` | < 1e-8 | PASS |
| `test_schrodinger_free_gaussian` | Free particle, `V=0`, 200 steps | `|‖ψ‖²/‖ψ₀‖² − 1| < 1e-4` | < 1e-6 | PASS |

Math reference: math.md §17 (palindromic Strang preserves ‖ψ‖²; V-rotation is
exact SO(2); kinetic step `Diffusion4thChernoff` preserves norm via spectral
bound argument). ADR-0057.

### 2.4 Composition gates (Phase 4)

| Gate | Component | Threshold | Result |
|------|-----------|-----------|--------|
| `test_nonseparable2d_slope` | `NonSeparable2D` constant-c | slope ≤ −1.95 | PASS |
| `test_nonseparable2d_beta_array_slope` | `NonSeparable2D.with_beta_array` | slope ≤ −1.95 | PASS |
| `test_adjoint_self_adjoint` | `Adjoint(Heat1D(...))`, `is_self_adjoint=True` | 0 ULP vs `Heat1D` | PASS |
| `test_adaptive_convergence` | `AdaptivePI(Heat1D(...))`, stiff problem | accepted steps < fixed-step baseline | PASS |
| `test_adaptive_outcome_keys` | `AdaptivePI.evolve` return dict | keys `{final_state, steps_accepted, steps_rejected, last_tau}` | PASS |

### 2.5 Graph expansion gates (Phase 5)

| Gate | Test | Result |
|------|------|--------|
| `test_graph_path` | `Graph.path(64)`, `n_nodes == 64` | PASS |
| `test_graph_cycle` | `Graph.cycle(8)`, all nodes degree 2 | PASS |
| `test_graph_from_edges` | `Graph.from_edges(4, edges)`, degree check | PASS |
| `test_graph_erdos_renyi` | `Graph.erdos_renyi(100, 0.3, seed=42)`, n_directed_edges in range | PASS |
| `test_laplacian_combinatorial` | Eigenvalue sum == 0 (row-sum property) | PASS |
| `test_laplacian_normalized` | Spectral bound ≤ 2.0 | PASS |
| `test_graphheat4th_slope` | `GraphHeat4th` slope ≤ −3.95 | PASS |
| `test_varcoef_graphheat` | `VarCoefGraphHeat.with_beta_array`, contractivity | PASS |
| `test_magnus_graph_heat6_order` | `MagnusGraphHeat6` slope ≤ −5.5 | PASS |
| `test_magnus_graph_heat6_radius` | Radius violation → `SemiflowError` | PASS |
| `test_graphpath_deprecated_alias` | `GraphPath(n)` == `Graph.path(n)` | PASS (deprecation warning emitted) |
| `test_graphheat_accepts_laplacian` | `GraphHeat(laplacian=Laplacian.combinatorial(g))` | PASS |
| `test_magnus_accepts_laplacian_callback` | callback returning `Laplacian` | PASS |

---

## 3. Cross-Binding Parity

**Status**: to be verified in Phase 7 final acceptance.

Gate definition (extends ADR-0059 §"Cross-binding sup-error gate"):

```
Inputs:
  - P_64 path graph, combinatorial Laplacian
  - GraphSignal: u₀(i) = exp(−i²/64), i ∈ {0..63}
  - t_final = 0.5, n_steps = 50

Compute u(t_final) via:
  - FFI (Rust via C interface)
  - PyO3 (Rust via Python, Phase 5 GraphHeat)

Cross-binding identity: |sup_err_FFI − sup_err_PyO3| < 3 ULP
```

For 1D heat (carried forward from ADR-0059):

```
Inputs: unit-a Heat1D, u₀ = exp(-x²), t=1.0, n=100

Cross-binding identity: |result_FFI[i] − result_PyO3[i]| < 3 ULP for all i
```

Both gates are expected to pass; they run as part of the Phase 7
`cargo run -p xtask -- ffi-smoke && py-smoke` acceptance suite.

---

## 4. Test Count Progression

| Phase | Tests passing | Delta |
|-------|---------------|-------|
| Phase 0 (v2.2.0 pre-expansion) | 19 | — |
| Phase 1 (Foundation, 7fc090a) | 34 | +15 |
| Phase 2 (1D kernels, 2308ea6) | 61 | +27 |
| Phase 3 (Schrödinger, a63ade3) | 71 | +10 |
| Phase 4 (Composition, 52d2695) | 88 | +17 |
| Phase 5 (Graph expansion, dc45181) | 129 | +41 |

Total: 129 Python tests passing, 0 failing at HEAD dc45181.

---

## 5. Suckless Gates

All new source files in `crates/semiflow-py/src/` satisfy the ≤500 LoC per
file, ≤50 LoC per function hard constraints enforced by
`cargo run -p xtask -- check-lints`. No new entries were added to the
`GRANDFATHERED` cohort in `xtask/src/main.rs`.

New `semiflow-core` files from Phase 2/4:

| File | LoC |
|------|-----|
| `diffusion_storage.rs` | ~80 |
| `nonseparable_mixed_closure.rs` | ~120 |

Both are well under the 500-LoC cap and are `pub(crate)`; they do not appear
on the public surface.

`unsafe` count in `semiflow-py`: 3 pre-existing (GIL-release pointer casts);
unchanged. Zero new `unsafe` in `semiflow-core`.

---

## 6. Known Gaps and Deferred Items

| Gap | Deferred to |
|-----|-------------|
| `TruncatedExpDiffusionChernoff` — Python | Unscheduled |
| `TruncatedExp4thDiffusionChernoff` — Python | Unscheduled |
| `StrangSplitGraph` — Python | Unscheduled |
| `GraphTraj` — Python | v2.4+ |
| `f32` Python path | Unscheduled |
| Async / yield PyO3 API | Unscheduled |
| FFI/WASM for 4th/6th-order, Schrödinger, Adjoint, AdaptivePI | Unscheduled |
| Phase 7 cross-binding parity gate (Python vs FFI ≤ 3 ULP) | Phase 7 final acceptance |
| `test-full` on production build (parallel + SIMD + all features) | Phase 7 final acceptance |

---

## 7. Recommendation

**DRAFT — pending Phase 7.**

Fast-test suite at dc45181: 129 passed / 0 failed. All convergence-rate, unitarity,
boundary, and graph gates pass. No DEVIATION-class findings in the Python binding
layer; all behaviour matches the corresponding Rust-core acceptance gates
(v2.2.0 audit, `docs/audit-findings-v2_0_0.md`).

This document will be updated to **APPROVED** status once:
1. Phase 7 cross-binding parity gate (`ffi-smoke` + `py-smoke`) passes.
2. `cargo run -p xtask -- test-full` exits 0 on the release build.
3. Lockstep version bump to 2.3.0 lands (`git tag v2.3.0`).
