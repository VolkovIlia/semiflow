---
version: 1.0.0
last_updated: 2026-05-22
freshness_score: 1.0
dependencies:
  - docs/adr/0062-order-6-spatial-graph-heat.md
  - docs/adr/0063-varcoef-time-dep-graph.md
  - docs/adr/0064-v2_4-graph-completeness-release.md
  - docs/audit-findings-v2_3_0.md
  - contracts/semiflow-core.math.md
changelog:
  - 1.0.0: Initial v2.4.0 Graph Completeness audit. Status DRAFT — full
    test-full release suite + cross-binding parity smokes not yet rerun
    on this branch; results reported below are from `test-fast --release`.
---

# v2.4.0 Graph Completeness Audit

**Branch**: `feat/v2.4-graph-completeness`
**Parent commit (v2.3 Phase 7)**: `8073830`
**Date**: 2026-05-22
**Status**: **DRAFT** — fast-test suite passes (graph_heat6 + varcoef_magnus
unit + integration tests + G11 byte-equality regression); cross-binding
smoke parity (`ffi-graph-smoke`, `py-graph-smoke`, `wasm-graph-smoke`) and
`test-full --release` deferred to release sign-off.
**Theme**: Graph Completeness — fill three documented gaps in the v2.3
graph stack (order-6 spatial, variable-coefficient × time-dependent,
FFI/WASM coverage).

---

## 1. Audit Scope

### Rust-core additions (Wave 2.4A + 2.4B)

| Wave | Commit | Additions |
|------|--------|-----------|
| 2.4A — Order-6 spatial | (this PR) | `GraphHeat6thChernoff<F>` (`graph_heat6.rs`, 283 LoC), `lib.rs` re-export, contracts math.md §19, ADR-0062. |
| 2.4B — VarCoef × time-dep | (this PR) | `VarCoefMagnusGraphHeatChernoff<F>` (`varcoef_magnus_graph.rs`, 479 LoC), `WeightAtTime<F>`, `compute_rho_bar` helper, `graph_var_coef.rs::apply_la_on_slice` → `pub(crate)`, contracts math.md §20, ADR-0063. |

### FFI / WASM additions (Wave 2.4C)

| Layer | File | Additions |
|-------|------|-----------|
| FFI | `crates/semiflow-ffi/src/graph_ffi_v2_4.rs` (403 LoC) | `smf_ghc6_*` (GraphHeat6), `smf_vc_mghc_*` (VarCoef Magnus). `crates/semiflow-ffi/include/remizov.h` regenerated. |
| WASM | `crates/semiflow-wasm/src/graph_wasm_hi.rs` (155 LoC) | `GraphHeat6` (order-6 static; no callbacks). Magnus variants deferred to v2.5 per ADR-0064. |

### Build / version

`Cargo.toml` workspace version bumped `2.3.0 → 2.4.0`; all member crates
inherit via `version.workspace = true`. No source migration required
(additive minor bump under v2.x).

---

## 2. Gate Results

### 2.1 Sympy gates

| Gate | Script | Result |
|------|--------|--------|
| T16N (Order-6 spatial Taylor) | `scripts/verify_graph_heat6_sympy.py` | **PASS** — `S_6(τ) − exp(−τL_G) = O(τ⁷)` on symbolic 4×4 path Laplacian; S_5 disagreement at τ⁶ sanity check PASS. |
| T17N (VarCoef Magnus K=4) | `scripts/verify_varcoef_magnus_graph_sympy.py` | **PASS** — `Ω₄_library(τ) − Ω_true(τ) = O(τ⁸)` (better than the required O(τ⁵)) on symbolic L_a(t) with degree-1 polynomial `w(t)`, `a(t)`. |

### 2.2 Slope / floor gates

| Gate | Test | Threshold | Measured | Result |
|------|------|-----------|----------|--------|
| G21 f64 | `tests/graph_g_k6_slope.rs::g21_graph_heat6_convergence_slope_f64` | ≤ −5.85 | ≈ −6.0 (5/8/12 steps, T=1.0, cos IC on P_64) | **PASS** |
| G21 f32 abs-floor | `tests/graph_g_k6_slope.rs::g21_graph_heat6_f32_absolute_floor` | `|err|_∞ ≤ 5e-6` | within budget | **PASS** |
| G22 f64 | `tests/varcoef_magnus_slope.rs::g22_varcoef_magnus_convergence_slope_f64` | ≤ −3.85 | within budget (4 steps {5,8,12,20}, T=0.5, time-varying `w(t)` + `a(t)`) | **PASS** |
| G11 byte-equality regression | `tests/g11_magnus_graph_slope.rs` (6 tests) | byte-identical to v2.3 baseline | byte-identical | **PASS** |

### 2.3 Zero-alloc steady state (R4 invariant)

| Test | Budget | Result |
|------|--------|--------|
| `graph_heat6_zero_alloc::graph_heat6_zero_alloc_steady_f64` | 0 allocs | PASS |
| `graph_heat6_zero_alloc::graph_heat6_zero_alloc_steady_f32` | 0 allocs | PASS |
| `varcoef_magnus_graph_zero_alloc::varcoef_magnus_kernel_allocs_only_for_callbacks_f64` | ≤ 2 allocs (closure-owned `Vec<f64>` from `a_at_t`) | PASS |
| `varcoef_magnus_graph_zero_alloc::varcoef_magnus_kernel_allocs_only_for_callbacks_f32` | ≤ 2 allocs | PASS |

### 2.4 Proptest

| Test | Property | Result |
|------|----------|--------|
| `varcoef_magnus_proptest::p2_zero_tau_is_identity` | `apply_into(0, src) ≈ src` for random `n ∈ [4, 32]` | PASS |
| `varcoef_magnus_proptest::p3_radius_violation_always_errors` | `ρ̄ · a_sup² · τ ≥ π/2` ⇒ `OutOfMagnusRadius` for random `(τ, a_sup)` | PASS |
| `varcoef_magnus_proptest::p4_negative_a_returns_error` | `a_i < 0` ⇒ `DomainViolation` | PASS |
| `varcoef_magnus_proptest::p4_wrong_length_a_returns_error` | `a.len() ≠ n_nodes` ⇒ `DomainViolation` | PASS |

---

## 3. Lint / size budget

`cargo run -p xtask -- check-lints` — PASS (17 grandfathered pre-existing
entries; no new violations).

| File | LoC | Budget | Status |
|------|-----|--------|--------|
| `crates/semiflow-core/src/graph_heat6.rs` | 283 | 500 | OK |
| `crates/semiflow-core/src/varcoef_magnus_graph.rs` | 479 | 500 | OK (tight) |
| `crates/semiflow-ffi/src/graph_ffi_v2_4.rs` | 403 | 500 | OK |
| `crates/semiflow-wasm/src/graph_wasm_hi.rs` | 155 | 500 | OK |
| `crates/semiflow-core/src/magnus_graph.rs` | 856 | (grandfathered) | unchanged ✓ |

No new constitution override entries (per ADR-0064 §"Override discipline").

---

## 4. Cross-language coverage matrix (v2.4 snapshot)

See ADR-0064 §"Decision" table for the full grid.

| Kernel | Rust | FFI | Python | WASM |
|--------|------|-----|--------|------|
| `GraphHeat6thChernoff` (K=6 spatial) | ✓ v2.4 | ✓ v2.4 | — v2.5+ | ✓ v2.4 |
| `VarCoefMagnusGraphHeatChernoff` | ✓ v2.4 | ✓ v2.4 | — v2.5+ | — v2.5+ |

Pre-existing kernels are unaffected by v2.4 (per ADR-0064 §"Override
discipline").

---

## 5. Known gaps deferred to v2.5+

1. **Python bindings** for `GraphHeat6thChernoff` and
   `VarCoefMagnusGraphHeatChernoff` — scaffolding pattern follows
   `crates/semiflow-py/src/graph_extra.rs::Graph` etc.; deferred to v2.5.
2. **WASM bindings** for time-dependent Magnus variants
   (`MagnusGraphHeat`, `MagnusGraphHeat6`, `VarCoefMagnusGraph`) — JS
   callback overhead motivates a "pre-built schedule" approach in v2.5
   (ADR-0064 §"WASM layout").
3. **FFI bindings** for `GraphHeat4thChernoff`, `MagnusGraphHeat6thChernoff`,
   `VarCoefGraphHeatChernoff` (constant-time) — the original v2.4 plan
   listed these; scope reduced to keep `graph_ffi_v2_4.rs` ≤500 LoC and
   to focus the release on the NEW kernels of ADR-0062 / ADR-0063.
   Future FFI rollout per ADR-0059 cross-binding parity policy.
4. **Variable-coefficient Magnus K=6** — ADR-0063 §"Out of scope (v2.4)".
5. **Time-discontinuous `a(t)`** — ADR-0063 §"Out of scope (v2.4)".

---

## 6. Release sign-off checklist

Pre-release verifications still required (run by release engineer):

- [ ] `cargo run -p xtask -- test-full` on `RUSTFLAGS=-C target-cpu=native`
      `--release --features parallel,simd,slow-tests`.
- [ ] `cargo run -p xtask -- ffi-headers --check` (no header drift).
- [ ] `cargo run -p xtask -- ffi-graph-smoke` (FFI end-to-end on K=6 + VarCoef Magnus).
- [ ] `cargo run -p xtask -- wasm-graph-smoke` (WASM end-to-end on K=6).
- [ ] Workspace version lock-step: `grep -r 'version = "2\.4\.0"' Cargo.toml` returns expected lines.
- [ ] `cargo run -p xtask -- check-unsafe-scope` (no unsafe leaked into core).
- [ ] G11 byte-equality regression re-verified in release build.
- [ ] CHANGELOG, ROADMAP, this audit doc updated.

Once all items above PASS, status of this document is upgraded to
**APPROVED**.
