# ADR-0065 — Python parity for v2.4 graph kernels

- **Status**: ACCEPTED (v2.5 Phase 1)
- **Date**: 2026-05-22
- **Wave**: v2.5 Phase 1 (Python parity rollup for v2.4)
- **Authors**: claude (orchestrator)
- **Depends on**: ADR-0061 (Python parity expansion v2.3), ADR-0062
  (Order-6 spatial graph heat), ADR-0063 (VarCoef × time-dep Magnus),
  ADR-0064 (v2.4 release scope rollup), ADR-0031 (Python GIL three-phase pattern).
- **Mathematical foundation**: math.md §19, §20 — unchanged. No new math
  in this ADR.

## Context

v2.4 (ADR-0064) shipped Rust core + FFI for `GraphHeat6thChernoff` and
`VarCoefMagnusGraphHeatChernoff`, plus WASM for the static `GraphHeat6`
only. The audit doc `docs/audit-findings-v2_4_0.md` §5 listed Python
bindings for both kernels as **explicitly deferred to v2.5+**. This
ADR closes that deferral.

The pattern is the established v2.3 Phase 5 PyO3 mechanism (ADR-0061
+ ADR-0059 R2): three-phase GIL pattern (`validate → py.detach →
to_pyarray`), Python callbacks wrapped via `Python::attach` at each
GL quadrature point.

## Decision

Ship two new pyclasses in a new module
`crates/semiflow-py/src/graph_v2_4.rs` (~430 LoC, under 500-LoC cap):

```rust
#[pyclass(name = "GraphHeat6")]
pub struct GraphHeat6 { laplacian: Arc<Laplacian<f64>>, graph: Arc<Graph<f64>>, n_nodes: usize }

#[pyclass(name = "VarCoefMagnusGraph")]
pub struct VarCoefMagnusGraph {
    n_nodes: usize, graph: Arc<Graph<f64>>,
    rho_bar_max: f64, a_sup_max: f64, convergence_check: bool,
    lap_callback: Py<PyAny>, a_callback: Py<PyAny>,
}
```

Both are f64-only (precedent: ADR-0056 for Magnus K=6 Python binding,
generalised to all graph kernel pyclasses). Rust core types remain
generic over `F: SemiflowFloat`.

### `compute_rho_bar` as `@staticmethod`

The free function `varcoef_magnus_graph::compute_rho_bar` is exposed
as `VarCoefMagnusGraph.compute_rho_bar(lap_at_t, a_at_t, t0, t1,
n_nodes, n_samples=32) -> (rho_bar_max, a_sup_max)`. Static-method
form avoids creating a new module-level Python function precedent
(only `version()` exists today) and gives the helper natural
discoverability as part of the kernel's public API.

### `a_at_t` callback signature

`a_at_t: Callable[[float], np.ndarray[float64]]` — symmetric with the
existing `lap_at_t: Callable[[float], Graph | Laplacian | GraphPath]`
of `MagnusGraphHeat` and `MagnusGraphHeat6`. The wrapper
`make_a_at_t_py` follows the same defensive-fallback pattern as
`make_lap_at_t_py` (`magnus6.rs:203–211`): on Python exception OR
length mismatch the wrapper returns `vec![1.0; n_nodes]` to keep the
Rust core's Magnus step deterministic.

### `apply_la_on_slice` visibility

The Rust-core helper `graph_var_coef::apply_la_on_slice` was already
`pub(crate)` from ADR-0063. No further core changes.

### `resolve_lap_and_graph` re-use

The private helper `graph_extra::resolve_lap_and_graph` is promoted
to `pub(crate)` so `GraphHeat6::new` can dispatch identically to
`GraphHeat4th::new`. One-line visibility change; no semantic impact.

## Rationale

- **Closes the v2.4 deferral.** v2.5 was promised in the audit doc.
- **Re-uses Phase 5 patterns 1:1.** Three-phase GIL, `extract_f64_vec`,
  `extract_laplacian_arc`, `catch_panic_py!`, error mapping via
  `from_core`.
- **No version-table redesign.** Both classes appear in
  `lib.rs::semiflow` module after the existing Phase 5 cluster.
- **No new failure modes.** All errors map through `from_core` ⇒
  `SemiflowStatus` strings (`OutOfDomain`, `GridMismatch`, etc.).
  `OutOfMagnusRadius` maps to `OutOfDomain` (per existing
  `error::from_core` policy).

## Consequences

- `src/graph_v2_4.rs` projected ~430 LoC; under cap.
- `python/semiflow/__init__.py` and `__init__.pyi` updated with the
  two new symbols.
- Public Python surface +2 classes + 1 staticmethod. Additive minor
  bump.
- `Cargo.toml` workspace version bump `2.4.0 → 2.5.0` (lockstep via
  `version.workspace = true`).

## Acceptance gates

- `pytest crates/semiflow-py/tests/test_graph_heat6.py` — 10 tests PASS.
- `pytest crates/semiflow-py/tests/test_varcoef_magnus.py` — 12 tests PASS.
- Full `pytest crates/semiflow-py/tests/` — 151 tests PASS (129 baseline
  + 22 new).
- `cargo run -p xtask -- check-lints` — PASS, no new grandfathered entries.
- `cargo run -p xtask -- py-build` — wheel built successfully on
  release-ffi profile.
- **Constant-a parity** (`test_constant_a_parity_vs_magnus_k4`):
  with `a(t) ≡ 1` and `L_G(t) ≡ L_G`, `VarCoefMagnusGraph.evolve`
  matches `MagnusGraphHeat.evolve` to ≤ 1e-2 sup-norm (precedent
  per `test_magnus6.py`).
- **K=6 empirical order** (`test_graph_heat6_empirical_order`):
  log-log slope ≤ −5.0 on cos IC, T=1.0, `n_steps ∈ {5, 8, 12}` vs
  reference at `n_steps=80`.

## Out of scope (v2.5 Phase 1)

- **FFI bindings for `GraphHeat4`, `MagnusGraphHeat6`, `VarCoefGraphHeat`**
  (constant-time) — these remain Python-only per ADR-0064 §"Scope
  rationale". Deferred to v2.5+.
- **WASM bindings for time-dependent Magnus** — JS callback overhead
  motivated the "pre-built schedule" approach planned for v2.5+ per
  ADR-0064 §"WASM layout".
- **Type stubs (`__init__.pyi`)** for the two new classes — minimal
  stub additions only. Comprehensive doctring-driven stubs deferred
  to v2.5 Phase 2.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | `Py<PyAny>` callback called from a non-Python thread under `py.detach` | `Python::attach` re-acquires the GIL safely; precedent in `magnus6.rs`. |
| R2 | `a_at_t` returns wrong-length vector silently | `make_a_at_t_py` length-checks; falls back to all-ones (defensive — matches `lap_at_t` fallback in ADR-0059 R2). Unit tests cover the path. |
| R3 | Constant-a parity test fails due to f64 ULP drift | Tolerance 1e-2 per `test_magnus6.py` precedent (graph kernel cross-comparison band). |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `src/graph_v2_4.rs` | ~430 |
| `python/semiflow/__init__.py` update | +4 |
| `tests/test_graph_heat6.py` | ~115 |
| `tests/test_varcoef_magnus.py` | ~185 |
| ADR-0065 (this) | ~150 |
| math.md updates | 0 (math unchanged) |
| **Total** | **~880** |

## References

- ADR-0061 (Python parity expansion v2.3) — pattern source.
- ADR-0062 / ADR-0063 — Rust-core kernels.
- ADR-0064 — release scope; v2.4 deferrals.
- ADR-0031 — three-phase GIL pattern.
- ADR-0059 (Graph bindings FFI/PyO3/WASM) §R2 — callback wrapping
  + fallback on Python error.
