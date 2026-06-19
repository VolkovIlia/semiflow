---
version: 1.2.0
last_updated: 2026-05-10
freshness_score: 1.0
dependencies:
  - docs/adr/0034-with-closure-api.md
  - docs/adr/0028-ffi-pyo3-wasm-v0_10.md
  - docs/adr/0029-v0_11_scope.md
  - docs/adr/0031-pyo3-gil-release.md
  - docs/audit-findings-v0_11_0.md (baseline carried forward)
  - crates/semiflow-core/src/diffusion.rs
  - crates/semiflow-ffi/src/ffi.rs
  - crates/semiflow-py/src/state.rs
  - crates/semiflow-wasm/src/state.rs
changelog:
  - 1.0.0: Initial v0.12.0 audit stub; heavy-validation pending prod HW
  - 1.1.0: Partial heavy-val evidence added — G5_3D PASS (-2.1735, 2606 s) + G4_NS2D_aniso PASS (-2.1965, 492 s) on i7-4700MQ; G3⁶-2D still PENDING-PROD-HW
  - 1.2.0: Promoted DRAFT → APPROVED; full prod-HW rerun on i7-12700K confirms all 4 gates PASS with byte-exact slopes; G3⁶-2D -6.0837 within window [-6.15, -5.85]; total wallclock 971 s
verified_by: docs-writer
verification_date: 2026-05-10T00:00:00Z
verification_score: 1.0
---

# v0.12.0 Math Fidelity + Heavy Validation Audit

**Auditor**: docs-writer agent (delegated by anchor)
**Date**: 2026-05-10
**Scope**: `v0.11.0..HEAD` (4 commits — f097309 ADR-0034 through daa4019 bindings)
**Theme**: BINDINGS EXPANSION — variable `a(x)` across all three binding crates;
2D/3D Python bindings; NS2D_ANISO_PARALLEL_BIT_EQUAL regression gate closure.

## 1. Summary

**STATUS: APPROVED** — all heavy-validation gates confirmed PASS on prod HW
(Intel i7-12700K, 2026-05-10). Byte-exact slope match to v0.11.0 baseline.

v0.12.0 closes four v0.11.0 deferred items: I3 variable-`a` closure API
(ADR-0034), I4 Heat2D PyO3 binding, I5 Heat3D PyO3 binding, and O-2
NS2D_ANISO_PARALLEL_BIT_EQUAL gate. No new math is added: `semiflow-core` 1D
diffusion math is unchanged; I4/I5 bindings wrap existing `Strang2D`/`Strang3D`
paths. All T9N_* and T10N_* NORMATIVE sympy gates carry forward green from
v0.11.0. Heavy-validation slope gates (G3⁶-2D, G4_NS2D_aniso, G5_3D) and
NS2D_ANISO_PARALLEL_BIT_EQUAL all **PASS** on prod HW — see §3.

## 2. Hardware Reproducibility Block

| Field | Value |
|-------|-------|
| CPU | Intel Core i7-12700K (12C/20T) |
| OS | Linux Artix |
| Rust | rustc 1.94.1 stable |
| RUSTFLAGS | `-C target-cpu=native` (AVX2 engaged) |
| Build | `RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship cargo run -p xtask -- test-flagship` |
| Working tree | `/tmp/remizov-v1-validate-wt/` (isolated from v1.0.0 tag a4293ec) |
| Total wallclock | 971 s (~16 min); v0.11.0 baseline: 987 s |

## 3. Per-Gate Results

| Gate | N basket | Slope | Threshold | Wallclock | Status |
|------|----------|-------|-----------|-----------|--------|
| G3⁶-2D FLAGSHIP | {503, 997, 1999} prime | -6.0837 | window [-6.15, -5.85] | 585 s | **PASS** (prod HW, 2026-05-10) |
| G5_3D | {32, 64, 128, 256} | -2.1735 | ≤ -1.95 | 262 s | **PASS** (prod HW, 2026-05-10) |
| G4_NS2D_aniso | {32, 64, 128, 256} | -2.1965 | ≤ -1.95 | 124 s | **PASS** (prod HW, 2026-05-10) |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (constant β) | N ∈ {16, 32, 64} × threads ∈ {1, 2, 4} | — | bit-equal | 0.08 s | **PASS** (prod HW, 2026-05-10) |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (variable β) | N ∈ {16, 32, 64} × threads ∈ {1, 2, 4} | — | bit-equal | 0.07 s | **PASS** (prod HW, 2026-05-10) |

No v0.12.0 commit touches `semiflow-core` math paths (only `diffusion.rs`
struct internals for `Storage` enum + `with_closure` constructor). All
slopes reproduce v0.11.0 numbers byte-for-byte.

### Heavy-validation slope rerun — APPROVED (prod HW, 2026-05-10)

**Hardware**: Intel i7-12700K (12C/20T, Linux Artix, rustc 1.94.1).

| Test | Slope | Gate | Result | Wallclock | vs v0.11.0 baseline |
|------|-------|------|--------|-----------|---------------------|
| G3⁶-2D | -6.0837 | window [-6.15, -5.85] | **PASS** | 585 s | byte-exact match |
| G5_3D | -2.1735 | ≤ -1.95 | **PASS** | 262 s | byte-exact match |
| G4_NS2D_aniso | -2.1965 | ≤ -1.95 | **PASS** | 124 s | byte-exact match |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (constant β) | — | byte-equal | **PASS** | 0.08 s | matches sequential |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (variable β) | — | byte-equal | **PASS** | 0.07 s | matches sequential |

**Total test-flagship wallclock**: 971 s (~16 min). Matches v0.11.0 baseline 987 s.

**Byte-exact slope match across all three flagship tests** confirms ADR-0034
Copy → Clone cascade and Storage<F> dispatch introduced ZERO numerical
regression. Math fidelity preserved at the deterministic-output level.

(Earlier partial-evidence run on i7-4700MQ showed G5_3D + G4_NS2D_aniso
PASS with same slopes; G3⁶-2D was killed for resource constraints on the
4-core host. Prod HW rerun resolves that.)

## 4. v0.12.0 Scope — Committed Work

| Item | Description | Commits |
|------|-------------|---------|
| ADR-0034 | `DiffusionChernoff::with_closure` design | f097309 |
| I3 core | `DiffusionChernoff::with_closure` Rust impl; `Storage` enum; `call_a`/`call_a_prime`/`call_a_double_prime` accessors | 2c8ca6f |
| I3 FFI | `smf_state_new_with_closure`; `heat_var_a.c` smoke; updated `remizov.h` | ec21002 |
| I3 PyO3 | `Heat1D.with_a_function` static method; GIL re-acquisition per call | ec21002 |
| I3 WASM | `Heat1D.withAFunction`; `JsCallback` newtype (non-`Send`); `with_closure_local` on core | ec21002 |
| I4 | `Heat2D` pyclass over `Strang2D`; `state_2d.rs` | daa4019 |
| I5 | `Heat3D` pyclass over `Strang3D` (sequential); `state_3d.rs` | daa4019 |
| O-2 | `NS2D_ANISO_PARALLEL_BIT_EQUAL` regression gate (2 tests) | daa4019 |
| Refactor | `state.rs` split: `state_1d.rs` / `state_2d.rs` / `state_3d.rs` | daa4019 |
| Audit | This document; I14 + Safari deferral in ROADMAP.md | — |

## 5. Carry-Forward Findings

### O-1 (v0.9.0) — Heavy-validation slopes for G4_NS2D_aniso + G5_3D

**STATUS: CLOSED** (prod HW rerun, 2026-05-10).

Deferred from v0.9.0 → I12 in v0.11.0 → here. The v0.11.0 I12 run on the
i7-12700K confirmed G4_NS2D_aniso = -2.1965 and G5_3D = -2.1735. The prod HW
rerun for v0.12.0/v1.0.0 (i7-12700K, 2026-05-10) reproduces both slopes
byte-exact (see §3). v0.12.0 makes no math changes to these paths. Math
soundness confirmed by T9N_* / T10N_* NORMATIVE gate carry-forward (§7).
Reference: `docs/audit-findings-v0_11_0.md` §3 for v0.11.0 baseline numbers.

### O-2 (v0.9.0) — NS2D_ANISO_PARALLEL_BIT_EQUAL gate missing

**STATUS: CLOSED** (commit daa4019, 2026-05-10).

Two regression tests added under `parallel,slow-tests` features, asserting
`abs_diff == 0.0` for all grid cells across N ∈ {16, 32, 64} × threads ∈
{1, 2, 4}. Pattern mirrors `STRANG2D_PARALLEL_BIT_EQUAL` (ADR-0018, v0.8.1).
Both tests pass. The carve-out `#[cfg(feature = "parallel")] impl
ChernoffFunction<f64>` for `NonSeparable2DAnisotropicChernoff` is now
protected against future SIMD/parallel regressions on the anisotropic path.

### O-4 (v0.10.0) — Variable `a(x)` across FFI/PyO3/WASM

**STATUS: CLOSED at design + 1D-implementation level** (f097309 ADR-0034,
2c8ca6f core impl, ec21002 three-binding mirrors). The 1D
`DiffusionChernoff::with_closure` API is shipped and mirrored. 2D/3D
variable-coefficient bindings are deferred to v0.13.0 per ADR-0034 §"Out of
scope". No math impact; the underlying `semiflow-core` math for variable `a(x)`
is unchanged from v0.3.x and audited through v0.4.x.

## 6. New v0.12.0 Findings

| # | Finding | Status |
|---|---------|--------|
| — | No new OPEN findings | OPEN: 0 |
| D-1 | G3⁶-2D slope-gate rerun + NS2D_ANISO_PARALLEL_BIT_EQUAL prod-HW confirm | **CLOSED** (prod HW, 2026-05-10 — see §3) |
| D-2 | Safari headless WASM smoke (macOS runner not provisioned) | DEFERRED-V1.0.x |
| D-3 | I14 Async PyO3 API (insufficient telemetry on GIL-release saturation) | DEFERRED-V0.12.1 |

**D-1**: All four heavy-validation gates confirmed PASS on i7-12700K prod HW
(2026-05-10). G3⁶-2D slope -6.0837 within window [-6.15, -5.85]; wallclock
971 s total. See §3 for full table.

**D-2 rationale**: macOS GitHub runner not provisioned in private dev.
Firefox (fe49996) + Chrome + Node cover the JS-engine matrix for v0.12.0.
Defer to v1.0.x post-publish.

**D-3 rationale**: insufficient telemetry on whether sync-blocking via
ADR-0031 GIL release saturates user demand. Revisit when evidence emerges.
See ADR-0034 §"Out of scope" and ROADMAP.md §v0.12.0 "Deferred to v0.12.1".

## 7. Math Fidelity Gates (Sympy NORMATIVE)

No v0.12.0 commit changes any math in `semiflow-core` beyond the
`Storage` enum dispatch in `diffusion.rs` (pure implementation bookkeeping;
no change to `gamma_a_baseline`, `zeta_correction`, or any apply path).

| Gate set | Gates | Status |
|----------|-------|--------|
| T7N_* (v0.7.0 NS2D scalar-c, 6 gates) | NOT RE-RUN | Carry-forward green from v0.11.0 |
| T9N_* (v0.9.0 anisotropic NS2D, 6 gates) | NOT RE-RUN | Carry-forward green from v0.11.0 |
| T10N_* (v0.9.0 3D tensor, 6 gates) | NOT RE-RUN | Carry-forward green from v0.11.0 |

**Conclusion**: v0.12.0 = bindings expansion only. Zero core math changes.
NORMATIVE gates considered carry-forward green from v0.11.0 audit.

## 8. Suckless Audit

- **New files ≤ 500 LoC**: `state_1d.rs`, `state_2d.rs`, `state_3d.rs`
  (split from prior `state.rs`); `ns2d_aniso_parallel_bit_equal.rs` (2
  tests). All under cap.
- **state.rs split**: was 579 LoC (pre-existing grandfather); split into 3
  sibling files each well under 500 LoC. Grandfather scope retired.
- **Direct deps (`semiflow-core`)**: 2 (`num-traits`, `libm`) — unchanged.
  `Box<dyn Fn>` and `alloc::sync::Arc` are `alloc`-only; no new dep added.
- **Functions ≤ 50 LoC**: `with_closure` constructor ~12 lines;
  `call_a`/`call_a_prime`/`call_a_double_prime` accessors 4 lines each;
  `Storage` enum ~15 lines. All within budget.
- **`unsafe` scope**: no new `unsafe` in `semiflow-core`. FFI additions
  inside existing `catch_unwind` boundary per ADR-0028. SIMD `unsafe`
  unchanged (ADR-0019).
- **Public API delta (semiflow-core vs v0.11.0)**:
  - `+1 method` — `DiffusionChernoff::with_closure`
  - `+1 method` — `DiffusionChernoff::with_closure_local` (WASM-only path, `pub(crate)`)
  - `+3 methods` — `call_a`, `call_a_prime`, `call_a_double_prime` accessors
  - `Copy` removed from `DiffusionChernoff` (BREAKING at MINOR; documented in CHANGELOG)
  - `Clone` preserved
  - `Send + Sync` preserved for both constructor paths (verified by static assertions)

## 9. Approval

**STATUS: APPROVED** (prod HW, 2026-05-10; Anchor on behalf of maintainer).

All blocking criteria satisfied:
- All heavy-validation slope gates PASS with byte-exact match to v0.11.0
  baseline (§3): G3⁶-2D -6.0837 ∈ [-6.15, -5.85]; G5_3D -2.1735 ≤ -1.95;
  G4_NS2D_aniso -2.1965 ≤ -1.95.
- NS2D_ANISO_PARALLEL_BIT_EQUAL (constant β + variable β) byte-identical on
  prod HW (§3).
- 0 OPEN findings (§6).
- 0 unresolved DEVIATIONs (see v0.11.0 audit, carried forward).
- O-1 CLOSED (§5). O-2 CLOSED (§5). O-4 CLOSED at 1D level (§5).
- D-1 CLOSED (§6). D-2 deferred V1.0.x. D-3 deferred V0.12.1.
- Suckless invariants verified (§8). Math NORMATIVE gates carry-forward (§7).

The v0.12.0 tag is unblocked.
