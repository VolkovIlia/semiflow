---
version: 1.0.0
last_updated: 2026-05-22
freshness_score: 1.0
dependencies:
  - docs/adr/0065-py-parity-v2_4.md
  - docs/adr/0062-order-6-spatial-graph-heat.md
  - docs/adr/0063-varcoef-time-dep-graph.md
  - docs/adr/0064-v2_4-graph-completeness-release.md
  - docs/audit-findings-v2_4_0.md
changelog:
  - 1.0.0: Initial v2.5.0 Python parity audit. Status DRAFT — release
    sign-off (test-full --release + cross-binding parity smokes) pending.
---

# v2.5.0 Python Parity Audit (for v2.4 graph kernels)

**Branch**: `feat/v2.5-py-parity` (parent: `feat/v2.4-graph-completeness`)
**Date**: 2026-05-22
**Status**: **DRAFT** — implementation + Python pytest suite green
(151/0 PASS); release sign-off (test-full --release + cross-binding parity
smokes) pending.
**Theme**: close the v2.4 Python deferral — ship PyO3 bindings for
`GraphHeat6thChernoff` and `VarCoefMagnusGraphHeatChernoff` plus
`compute_rho_bar` staticmethod helper.

---

## 1. Audit Scope

### Python additions (Phase 1 of v2.5)

| Class | File | LoC | Tests | Source kernel |
|-------|------|-----|-------|---------------|
| `GraphHeat6` | `crates/semiflow-py/src/graph_v2_4.rs` | ~80 | 10 | `GraphHeat6thChernoff<f64>` (ADR-0062) |
| `VarCoefMagnusGraph` | same | ~150 | 12 | `VarCoefMagnusGraphHeatChernoff<f64>` (ADR-0063) |
| `VarCoefMagnusGraph.compute_rho_bar` | same | ~30 | 2 | `varcoef_magnus_graph::compute_rho_bar` (helper) |
| `make_lap_at_t_py`, `make_a_at_t_py`, `compute_heat6`, `compute_varcoef_magnus` | same | ~120 | — (indirectly via above) | private helpers |
| **Total** | | **~430 LoC** | **22** | |

### Wrapper-only — no Rust core / FFI / WASM changes

The Rust core, FFI, and WASM crates are byte-identical to v2.4 except for
the workspace version bump `Cargo.toml: 2.4.0 → 2.5.0`. All member crates
inherit via `version.workspace = true`.

### Build infra

- `crates/semiflow-py/python/semiflow/__init__.py` — re-exports the two
  new classes; `__all__` list extended.
- `crates/semiflow-py/src/lib.rs` — `mod graph_v2_4;` + two
  `m.add_class::<...>()` lines.
- `crates/semiflow-py/src/graph_extra.rs` — `resolve_lap_and_graph`
  visibility `fn` → `pub(crate)` (single-line change).

---

## 2. Gate Results

### 2.1 Python pytest

```
$ python3 -m pytest crates/semiflow-py/tests/ -q
.....................................................................   [ 47%]
........................................................................ [ 95%]
.......                                                                  [100%]
151 passed in 5.38s
```

| Test file | Tests | Result |
|-----------|-------|--------|
| `test_graph_heat6.py` | 10 | **10 PASS** |
| `test_varcoef_magnus.py` | 12 | **12 PASS** |
| All other Python tests (v2.3 baseline) | 129 | **129 PASS** |
| **Total** | **151** | **151 / 0 fail** |

### 2.2 Key correctness gates inside the new test files

| Test | Property verified | Result |
|------|-------------------|--------|
| `test_graph_heat6_preserves_sum` | Combinatorial Laplacian rows sum to 0 ⇒ `Σ f_i` conserved | PASS |
| `test_graph_heat6_dissipates_variance` | Delta IC's variance decreases | PASS |
| `test_graph_heat6_more_accurate_than_k4` | K=6 self-conv diff < K=4 self-conv diff at same `n_steps` | PASS |
| `test_graph_heat6_empirical_order` | Slope ≤ −5.0 vs reference at `n_steps=80` | PASS |
| `test_evolve_preserves_sum_constant_a` | Same conservation invariant for VarCoef Magnus | PASS |
| `test_radius_violation_raises` | `ρ̄·a_sup²·τ ≥ π/2` ⇒ `SemiflowError(OutOfDomain)` | PASS |
| `test_compute_rho_bar_constant` | Static rho returned exact (rho=4.0, a_sup=1.0) | PASS |
| `test_compute_rho_bar_varying` | Sinusoidal `a(t)` peaks at t=0.5; `a_sup ≈ √1.5` | PASS |
| `test_constant_a_parity_vs_magnus_k4` | `a(t) ≡ 1` ⇒ matches MagnusGraphHeat to ≤ 1e-2 | PASS |

### 2.3 Lint / size budget

```
$ cargo run -p xtask -- check-lints
NOTE (grandfathered…): 17 pre-existing
check-lints: PASS — no new violations
```

| File | LoC | Budget | Status |
|------|-----|--------|--------|
| `crates/semiflow-py/src/graph_v2_4.rs` | ~430 | 500 | OK |

No new constitution override entries.

---

## 3. Cross-language coverage matrix (v2.5 snapshot)

| Kernel | Rust | FFI | Python | WASM |
|--------|------|-----|--------|------|
| `GraphHeatChernoff` (K=1) | ✓ (v2.1) | ✓ (v2.2) | ✓ (v2.3) | ✓ (v2.2) |
| `GraphHeat4thChernoff` (K=4 spatial) | ✓ (v2.1) | — (v2.6+) | ✓ (v2.3) | — (v2.6+) |
| `GraphHeat6thChernoff` (K=6 spatial) | ✓ (v2.4) | ✓ (v2.4) | **✓ (v2.5 NEW)** | ✓ (v2.4) |
| `MagnusGraphHeatChernoff` (K=4 time-dep) | ✓ (v2.1) | ✓ (v2.2) | ✓ (v2.3) | — (v2.6+) |
| `MagnusGraphHeat6thChernoff` (K=6 time-dep, f64) | ✓ (v2.2) | — (v2.6+) | ✓ (v2.3) | — (v2.6+) |
| `VarCoefGraphHeatChernoff` (variable-a, fixed t) | ✓ (v2.1) | — (v2.6+) | ✓ (v2.3) | — (v2.6+) |
| `VarCoefMagnusGraphHeatChernoff` (variable-a × time-dep) | ✓ (v2.4) | ✓ (v2.4) | **✓ (v2.5 NEW)** | — (v2.6+) |

Net effect of v2.5: **2 new Python pyclasses (no other coverage changes)**.

---

## 4. Known gaps deferred to v2.6+

1. **FFI bindings** for the Python-only kernels (`GraphHeat4`,
   `MagnusGraphHeat6`, `VarCoefGraphHeat`). v2.4's `graph_ffi_v2_4.rs`
   covers only the **new** kernels; older v2.1/v2.3 Python-only kernels
   still lack FFI bindings.
2. **WASM time-dependent Magnus** — `MagnusGraphHeat`,
   `MagnusGraphHeat6`, `VarCoefMagnusGraph` on JS. The "pre-built
   schedule" alternative to JS callbacks (per ADR-0064 §"WASM layout")
   needs a new ADR.
3. **Comprehensive `__init__.pyi` type stubs** for `GraphHeat6` /
   `VarCoefMagnusGraph`. The existing `__init__.pyi` file is 1765 LoC of
   carefully-curated stubs; adding equivalent depth for the two new
   classes is a polish task deferred to v2.5 Phase 2 or v2.6.
4. **Variable-coefficient Magnus K=6**, **time-discontinuous `a(t)`**,
   **Order-8 Magnus**, **NS2D-aniso parallelism**, **Schrödinger 3D**,
   **`SemiflowComplex` trait** — all per v2.4 ROADMAP deferrals
   (unchanged).

---

## 5. Release sign-off checklist

Pre-release verifications still required (release engineer):

- [ ] `cargo run -p xtask -- test-full` on
      `RUSTFLAGS=-C target-cpu=native --release --features parallel,simd,slow-tests`.
- [ ] Workspace version lock-step: `grep -r 'version = "2\.5\.0"'
      Cargo.toml` returns expected lines.
- [ ] `cargo run -p xtask -- check-unsafe-scope` (no `unsafe` leaked
      into core).
- [ ] G11 / G21 / G22 byte-equality regressions remain green in release
      build.
- [ ] CHANGELOG, ROADMAP, this audit doc updated.
- [ ] Optional: Manual verification that the two new pyclasses appear
      in IPython tab-completion after `pip install` of the v2.5.0 wheel.

Once all items above PASS, this document is upgraded to **APPROVED**.
