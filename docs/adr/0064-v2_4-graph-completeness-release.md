# ADR-0064 — v2.4 Graph Completeness release scope and cross-binding parity rollup

- **Status**: ACCEPTED (v2.4 Release)
- **Date**: 2026-05-22
- **Wave**: v2.4 (Graph Completeness)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0062 (Order-6 spatial graph heat),
  ADR-0063 (Variable-coefficient time-dependent graph Magnus),
  ADR-0059 (Graph bindings FFI/PyO3/WASM), ADR-0061 (Python parity expansion).
- **Mathematical foundation**: math.md §19 + §20 (per ADR-0062,
  ADR-0063 respectively). No new math.

## Context

The v2.3 release shipped Python parity for the existing graph stack
(`GraphHeat4thChernoff`, `MagnusGraphHeat6thChernoff`,
`VarCoefGraphHeatChernoff`, etc.) per ADR-0061. The remaining graph
gaps are:

1. **Missing static-graph order-6 kernel** — closed by ADR-0062
   (`GraphHeat6thChernoff`).
2. **Missing variable-coefficient × time-dependent composition** —
   closed by ADR-0063 (`VarCoefMagnusGraphHeatChernoff`).
3. **FFI/WASM expose only K=1 graph kernels** — per audit of v2.3:
   K=4, K=6, Magnus K=4, Magnus K=6 are Python-only. ADR-0059 graph
   bindings policy mandates cross-binding parity; this release closes
   the FFI/WASM half.

This ADR is the **release-scope rollup**: it does not introduce new
math or new types, only ties together the three changes under a
single cross-binding parity gate.

## Decision

v2.4 ships:

| Kernel | Rust | FFI | Python | WASM |
|--------|------|-----|--------|------|
| `GraphHeatChernoff` (K=1) | ✓ (v2.1) | ✓ (v2.2) | ✓ (v2.3) | ✓ (v2.2) |
| `GraphHeat4thChernoff` (K=4 spatial) | ✓ (v2.1) | — (v2.5+) | ✓ (v2.3) | — (v2.5+) |
| `GraphHeat6thChernoff` (K=6 spatial) | **new v2.4 (ADR-0062)** | **new v2.4** | — (v2.5+) | **new v2.4** |
| `MagnusGraphHeatChernoff` (K=4 time-dep) | ✓ (v2.1) | ✓ (v2.2) | ✓ (v2.3) | — (v2.5+) |
| `MagnusGraphHeat6thChernoff` (K=6 time-dep, f64) | ✓ (v2.2) | — (v2.5+) | ✓ (v2.3) | — (v2.5+) |
| `VarCoefGraphHeatChernoff` (variable-a, time-static) | ✓ (v2.1) | — (v2.5+) | ✓ (v2.3) | — (v2.5+) |
| `VarCoefMagnusGraphHeatChernoff` (variable-a × time-dep) | **new v2.4 (ADR-0063)** | **new v2.4** | — (v2.5+) | — (v2.5+) |

**Scope rationale**: v2.4 ships the two NEW kernels (ADR-0062, ADR-0063)
plus their FFI bindings (so any FFI consumer can use the new functionality
end-to-end). WASM ships only the static `GraphHeat6` because the
time-dependent variants need JS callbacks (deferred to v2.5 per
§"WASM layout"). The Python `GraphHeat6thChernoff` and
`VarCoefMagnusGraphHeatChernoff` bindings ALSO land in v2.5+ — v2.4
focuses on Rust core + FFI parity for the new kernels.

### FFI layout (NORMATIVE)

New file `crates/semiflow-ffi/src/graph_ffi_k46.rs` houses the K=4 / K=6
/ Magnus K=6 / VarCoef Magnus C entry points. The existing
`graph_ffi.rs` (currently 748 LoC, grandfathered) is NOT expanded — see
ADR §"Override discipline" below.

Entry-point naming follows the existing `smf_<kernel>_<verb>` convention:

| Kernel | FFI prefix |
|--------|-----------|
| `GraphHeat4thChernoff` | `smf_ghc4_*` |
| `GraphHeat6thChernoff` | `smf_ghc6_*` |
| `MagnusGraphHeat6thChernoff` | `smf_mghc6_*` |
| `VarCoefGraphHeatChernoff` | `smf_vchc_*` |
| `VarCoefMagnusGraphHeatChernoff` | `smf_vc_mghc_*` |

Each Magnus variant uses the callback pattern from `smf_mghc_new`
(`graph_ffi.rs:597`): `user_data: usize` + `extern "C" fn` for
`lap_at_t` and (for VarCoef Magnus) a second callback for `a_at_t`.

### WASM layout (NORMATIVE — narrowed v2.4 scope)

New file `crates/semiflow-wasm/src/graph_wasm_hi.rs` houses ONLY the
`GraphHeat6` wasm-bindgen `#[wasm_bindgen]` type (order-6 static graph
heat, no callbacks). The existing `graph_wasm.rs` (K=1 / ζ-A only) is
NOT expanded.

**Deferred to v2.5 (out of scope for v2.4)**: `MagnusGraphHeat`,
`MagnusGraphHeat6`, and `VarCoefMagnusGraph` WASM bindings. The
time-dependent variants require `lap_at_t` / `a_at_t` JS callbacks,
which need:
1. unsafe `Send + Sync` wrappers around `js_sys::Function` (because
   WASM is single-threaded, sound but requires audit);
2. JS callback invocation per GL abscissa per τ-step (~16-30 JS-Rust
   trips per step for Magnus K=4/K=6 + VarCoef Magnus); and
3. JsValue ↔ `Vec<f64>` / `Arc<Graph<f64>>` round-trips that allocate
   per-call.

Per the v2.4 plan §"Q5", these costs were accepted-then-deferred when
the practical complexity ballooned. v2.5 will revisit with a
"pre-built schedule" approach: caller provides full `(t_i, L_G(t_i),
a(t_i))` arrays at constructor time, and the WASM kernel walks them
without JS callbacks.

## Cross-binding parity gate (NORMATIVE — inherits ADR-0059 + ADR-0061)

`cargo run -p xtask -- ffi-graph-smoke`,
`cargo run -p xtask -- py-graph-smoke`,
`cargo run -p xtask -- wasm-graph-smoke` MUST cover every kernel in
the v2.4 table above. Outputs MUST agree pairwise across
{Rust, FFI, Python, WASM} to ≤ 3 ULP on `P_64` path-graph workloads
(per ADR-0061 ULP-budget policy).

## Override discipline (NORMATIVE)

Constitution Override #1 file-list (per `.dev-docs/constitution.md`)
is NOT expanded by v2.4:

- `graph_heat6.rs` (new): projected ~250 LoC ✓ under cap.
- `varcoef_magnus_graph.rs` (new): projected ~380 LoC ✓ under cap.
- `graph_ffi_k46.rs` (new): projected ~480 LoC ✓ under cap.
- `graph_wasm_hi.rs` (new): projected ~420 LoC ✓ under cap.
- `magnus_graph.rs` operator-form refactor: ~+50 LoC; current 749 LoC
  + 50 = 799. **Already grandfathered (Override #1)**; expansion within
  the existing override allowance — no constitution amendment.

Override budget remains at 3 entries (per constitution v1.4.0).

## Rationale

- **Single release, single gate.** Bundling the three changes
  (ADR-0062, ADR-0063, FFI/WASM rollout) into v2.4 means one
  cross-binding parity run, one CHANGELOG section, one audit
  findings document, one version bump.
- **Closes the v2.3 cross-language asymmetry** documented in
  v2.3 audit-findings §"Known gaps": "FFI/WASM for 4th/6th-order
  graph kernels — Unscheduled".
- **No new theory required.** Both math.md §19 and §20 are pure
  citation + NORMATIVE library choices (per ADR-0062 / ADR-0063 §"no
  new theorems" notes).

## Consequences

- Version lock-step: workspace `Cargo.toml` and all crate
  `Cargo.toml` bump to `2.4.0`.
- New audit findings document: `docs/audit-findings-v2_4_0.md`.
- CHANGELOG: 2 new kernels + 4 new FFI bindings + 4 new WASM bindings
  + 1 operator-form refactor (compat invariant via byte-equality
  gate).
- ROADMAP: v2.4 marked SHIPPED; remaining v2.5+ items (NS2D-aniso
  parallelism, Schrödinger 3D, Order-8 Magnus, `SemiflowComplex` trait)
  unchanged.

## Acceptance gates (release-level)

- **G21** (ADR-0062) — order-6 spatial slope.
- **G22** (ADR-0063) — VarCoef Magnus slope.
- **T16N** (ADR-0062) — order-6 spatial sympy.
- **T17N** (ADR-0063) — VarCoef Magnus sympy.
- **G11 byte-equality** (ADR-0051 / ADR-0063 mitigation) — Magnus K=4
  byte-equality post operator-form refactor.
- **Cross-binding parity ≤ 3 ULP** (this ADR + ADR-0059 + ADR-0061)
  on all v2.4 kernels.
- **Workspace version lock-step** — every `Cargo.toml` reports `2.4.0`.
- **Zero-alloc steady-state** — `tests/graph_heat6_zero_alloc.rs` and
  `tests/varcoef_magnus_graph_zero_alloc.rs` PASS via `allocation_counter`.
- **`xtask test-full` (release)** — entire workspace test suite green
  on `RUSTFLAGS=-C target-cpu=native cargo test --workspace
  --features parallel,simd,slow-tests --release`.

## Out of scope (v2.4 — deferred to v2.5+)

- **NS2D-aniso parallelism** (per ADR-0060 critical-path nomination).
  Continuous PDE, not graph; orthogonal to v2.4.
- **Schrödinger 3D** (per ADR-0057 v2.3+ deferral).
- **Order-8 Magnus / `SemiflowComplex` trait** (per ADR-0056 §"out of
  scope" + ADR-0057 Option B deferral).
- **WASM struct-of-arrays prebuilt schedule** (v2.5 perf optimisation).
- **Variable-coefficient Magnus K=6** (per ADR-0063 §"out of scope").
- **Time-discontinuous a(t)** (per ADR-0063 §"out of scope").

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Magnus K=4 operator-form refactor breaks G11 byte-equality | Documented in ADR-0063 R1; fall-back path is code duplication. Refactor reverted if G11 fails. |
| R2 | WASM JS-callback overhead unacceptable for production users | Accepted for v2.4; flagged in CHANGELOG. v2.5 optimisation planned. |
| R3 | `graph_ffi_k46.rs` and `graph_wasm_hi.rs` grow beyond 500-LoC cap | Per-kernel sub-modules if necessary; ADR-0062 / ADR-0063 LoC budgets give headroom. |
| R4 | Cross-binding parity ≤ 3 ULP fails on Magnus K=6 (f64 only) due to platform exp() variation | Document as known limitation: `f64` ULP-budget for K=6 Magnus is ≤ 8 ULP on platforms without IEEE-754 strict mode (matches v2.3 audit-findings §"Cross-binding parity" tolerance band). |

## Cost (LoC estimate, release-level)

| Component | LoC |
|---|---|
| ADR-0062 + ADR-0063 implementations (per their cost tables) | ~2140 |
| `src/graph_ffi_k46.rs` | ~480 |
| `src/graph_wasm_hi.rs` | ~420 |
| `xtask/src/{ffi,py,wasm}_tasks.rs` smoke extensions | ~150 |
| `crates/semiflow-py/src/graph_extra.rs` (VarCoef Magnus pyclass) | ~120 |
| `docs/audit-findings-v2_4_0.md` | ~280 |
| CHANGELOG + ROADMAP updates | ~80 |
| ADR-0064 (this) | ~220 |
| **Total** | **~3890** |

## References

- ADR-0059 (Graph bindings FFI/PyO3/WASM) — cross-binding parity policy.
- ADR-0061 (Python parity expansion v2.3) — ULP-budget framework.
- ADR-0062 — Order-6 spatial graph heat.
- ADR-0063 — Variable-coefficient time-dependent graph Magnus.
- `docs/audit-findings-v2_3_0.md` — v2.3 known gaps explicitly closed
  by this ADR.
