---
version: 1.2.0
last_updated: 2026-05-10
freshness_score: 1.0
dependencies:
  - docs/api-stability.md
  - docs/perf-commitment-v1_0_0.md
  - docs/perf/baseline-v1_0_0.json
  - docs/adr/0035-v1_0_0-api-stability.md
  - docs/adr/0034-with-closure-api.md
  - docs/adr/0033-nonseparable2d-deprecation-policy.md
  - docs/adr/0032-heavy-validation-harness.md
  - docs/audit-findings-v0_12_0.md (baseline carried forward)
  - docs/audit-findings-v0_11_0.md (baseline carried forward)
  - .dev-docs/verification/scripts/verify_v0_7_0_nonseparable.py
  - .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py
  - .dev-docs/verification/scripts/verify_v0_9_0_3d_tensor.py
  - .github/workflows/nightly.yml
changelog:
  - 1.0.0: Initial v1.0.0 MAJOR-release math fidelity audit (DRAFT — heavy-validation slopes pending S2.5 prod-HW rerun)
  - 1.1.0: Partial heavy-val evidence added — G5_3D PASS (-2.1735, 2606 s) + G4_NS2D_aniso PASS (-2.1965, 492 s) on i7-4700MQ; G3⁶-2D still PENDING-PROD-HW; DRAFT status unchanged
  - 1.2.0: Promoted DRAFT → APPROVED; full prod-HW rerun on i7-12700K confirms all 4 gates PASS with byte-exact slopes; G3⁶-2D -6.0837 within window [-6.15, -5.85]; total wallclock 971 s; S2.9 reviewer-suckless gate CONDITIONAL-GO promoted to GO
verified_by: researcher
verification_date: 2026-05-10T00:00:00Z
verification_score: 1.0
---

# v1.0.0 Math Fidelity Audit (MAJOR Release)

**Auditor**: researcher agent (delegated by anchor)
**Date**: 2026-05-10
**Approver**: Anchor on behalf of maintainer (heavy-val executed on prod HW, 2026-05-10)
**Status**: **APPROVED** — all gates PASS on i7-12700K prod HW; S2.9 reviewer-suckless GO
**Scope**: `v0.11.0..HEAD` (v0.12.0 commits + v1.0.0 stability deliverables)
**Theme**: API freeze, performance commitment, and the obligatory MAJOR-release
fidelity audit per ROADMAP "Math fidelity commitment" rule #5.

## 1. Summary

**STATUS: APPROVED** — all heavy-validation gates confirmed PASS on prod HW
(Intel i7-12700K, 2026-05-10). Byte-exact slope match to v0.11.0 baseline.

The three NORMATIVE sympy gate sets that seal the math-content surface area
of `semiflow-core` (`T7N_*`, `T9N_*`, `T10N_*`) have all been re-run on the
auditor host (sympy 1.14.0, Python 3, Linux). **All 18 NORMATIVE gates PASS**
(T7N: 6/6, T9N: 6/6, T10N: 6/6) — see § 3 Rule 1 below for per-gate
evidence and wallclock. Zero math drift since the v0.7.0 / v0.9.0 seal points.

v1.0.0 freezes (i) all v0.12.0 binding-expansion commits over a frozen
v0.9.0 math core, plus (ii) S2.x stability deliverables that add **zero new
math** (API stability policy, performance commitment infrastructure, and
ADR-0035 freezing the public surface). Heavy-validation slope gates
(G3⁶-2D, G4_NS2D_aniso, G5_3D, NS2D_ANISO_PARALLEL_BIT_EQUAL) **PASS** on
prod HW (S2.5 rerun, i7-12700K, 2026-05-10) — see §4.

Cite-coverage (Rule 4): all eleven math-introducing ADRs cite primary
literature (Remizov 2025 / HLW 2006 / BCO-R 2009 / Strang 1968 / Magnus
1954 / Söderlind 2002 as applicable). DEVIATIONs (Rule 3): zero unresolved
across the v0.10.0 → v0.12.0 audit chain. The two v0.6.0 DEVIATIONs (D1
order() axis mismatch, D2 Magnus naming) closed in v0.6.1 (rename
TruncatedExp* per ADR-0011 Amendment 1).

This document satisfies ROADMAP "Math fidelity commitment" rule #5
(researcher fidelity audit at every MAJOR release).

---

## 2. Hardware Reproducibility Block

| Field | Value |
|-------|-------|
| Sympy host | researcher container (Python 3, sympy 1.14.0, Linux) |
| Sympy gate wallclock | T7N 57.82 s + T9N 62.76 s + T10N 2.40 s = **123.0 s** total |
| Heavy-validation host (S2.5) | Intel Core i7-12700K (12C/20T), Linux Artix, rustc 1.94.1 stable |
| Heavy-validation working tree | `/tmp/remizov-v1-validate-wt/` (isolated from v1.0.0 tag a4293ec) |
| Heavy-validation RUSTFLAGS | `-C target-cpu=native` (AVX2 engaged) |
| Heavy-validation build | `RUSTFLAGS="-C target-cpu=native" CARGO_TARGET_DIR=target-flagship cargo run -p xtask -- test-flagship` |
| Heavy-validation total wallclock | 971 s (~16 min); v0.11.0 I12 baseline: 987 s |

---

## 3. Math Fidelity Commitment Compliance

The five rules in ROADMAP § "Math fidelity commitment" are reproduced here
verbatim and evaluated for the v1.0.0 cut.

### Rule 1 — Pass sympy verification for all NORMATIVE math.md sections (properties.yaml gates)

> *"Pass sympy verification for all NORMATIVE math.md sections (properties.yaml gates)."*

**STATUS: PASS** (18/18 gates green; zero drift since seal).

| Script | Gate set | Gates passed | Wallclock | Exit |
|--------|----------|--------------|-----------|------|
| `verify_v0_7_0_nonseparable.py` | T7N_* (NS2D scalar-`c`, v0.7.0 §10.7-bis) | **6/6** | 57.82 s | 0 |
| `verify_v0_9_0_nonseparable_aniso.py` | T9N_* (anisotropic NS2D, v0.9.0 §10.7-ter) | **6/6** | 62.76 s | 0 |
| `verify_v0_9_0_3d_tensor.py` | T10N_* (3D tensor Strang3D, v0.9.0 §10.8) | **6/6** | 2.40 s | 0 |

**Per-gate detail (verbatim verifier output):**

T7N_* (NonSeparable2DChernoff):
- `T7N_tau2`          (palindromic τ²-cancellation):     **True** [NORMATIVE]
- `T7N_tau3`          (Y₃ formula match):                **True** [NORMATIVE]
- `T7N_K2_local`      (K=2 preserves order 2):           **True** [NORMATIVE]
- `T7N_oracle`        (rotated-Gaussian solves PDE):     **True** [NORMATIVE]
- `T7N_zero-c`        (c≡0 → Strang2D path):             **True** [NORMATIVE]
- `T7N_palindrome`    (leg sequence palindromic):        **True** [NORMATIVE]

T9N_* (NonSeparable2DAnisotropicChernoff):
- `T9N_tau2`          (palindromic τ²-cancellation):     **True** [NORMATIVE]
- `T9N_tau3`          (Y₃ formula match):                **True** [NORMATIVE]
- `T9N_K2_local`      (K=2 preserves order 2):           **True** [NORMATIVE]
- `T9N_oracle`        (rotated-Gaussian solves PDE):     **True** [NORMATIVE]
- `T9N_zero-beta`     (β≡0 → Strang2D path):             **True** [NORMATIVE]
- `T9N_palindrome`    (leg sequence palindromic):        **True** [NORMATIVE]

T10N_* (Strang3D):
- `T10N_pairwise_commute`  ([A,B]=[A,C]=[B,C]=0):        **True** [NORMATIVE]
- `T10N_strang3d_collapse` (S_3D = exp(τL) exactly):     **True** [NORMATIVE]
- `T10N_palindrome`        (leg sequence palindromic):    **True** [NORMATIVE]
- `T10N_oracle`            (3D Gaussian solves PDE):      **True** [NORMATIVE]
- `T10N_zero-axis`         (C=0 → Strang2D path):         **True** [NORMATIVE]
- `T10N_order_min`         (order = min per-axis):        **True/STRUCTURAL** [NORMATIVE]

**Conclusion**: all NORMATIVE math sealed at v0.7.0 / v0.9.0 reproduces
byte-for-byte at v1.0.0. Zero math drift; no v0.10.0 / v0.11.0 / v0.12.0
commit invalidates any sealed proof.

### Rule 2 — Pass v0.5.0 regression bit-equal golden gate (no v0.5.0 surface change)

> *"Pass v0.5.0 regression bit-equal golden gate (no v0.5.0 surface change)."*

**STATUS: STANDING-GREEN** (CI-enforced; not re-run in this audit pass).

The v0.5.0 surface (`Grid2D`, `GridFn2D`, `AxisLift`, `Strang2D`, `Axis`) is
covered by:

- `tests/simd_bit_equal.rs` — SIMD path bit-equal vs scalar (ADR-0019).
- `tests/strang2d_parallel_bit_equal.rs` — parallel Strang2D bit-equal vs
  serial across thread counts (ADR-0018, v0.8.1).
- `tests/strang3d_parallel_bit_equal.rs` — parallel Strang3D bit-equal vs
  serial across thread counts (v0.11.0 I12).
- `tests/ns2d_aniso_parallel_bit_equal.rs` — parallel anisotropic NS2D
  bit-equal vs serial (v0.12.0 daa4019, closes O-2).

These are gated behind `--features parallel,slow-tests` and run as part of
`xtask test-flagship`. The standing `cargo test --workspace` matrix on each
push (CI green) provides the regression attestation. **No v1.0.0 commit
modifies any of these test files**, and no commit modifies the
v0.5.0-vintage `Strang2D` / `AxisLift` / `Grid2D` paths.

### Rule 3 — Document any new SIMPLIFICATION / APPROXIMATION / DEVIATION

> *"Document any new SIMPLIFICATION / APPROXIMATION / DEVIATION in
> docs/audit-findings-v{N}.md before tagging."*

**STATUS: 0 unresolved DEVIATIONs across the v0.10.0 → v0.12.0 audit chain.**

| Audit | DEVIATION | APPROXIMATION | SIMPLIFICATION | Notes |
|-------|-----------|---------------|----------------|-------|
| v0.6.0 | 2 | — | — | D1 (order() axis mismatch) + D2 (Magnus naming) — both CLOSED in v0.6.1 / v0.7.0 (rename `TruncatedExp*`) |
| v0.9.0 | 0 | — | — | — |
| v0.10.0 | 0 | 0 | 5 | Five SIMPLIFICATIONs all documented in v0.10.0 § 3 (binding-scope narrowing, no math impact) |
| v0.11.0 | 0 | — | — | I12 closure; ratified all v0.10.0 SIMPLIFICATIONs |
| v0.12.0 | 0 | — | — | Bindings expansion only; no new math |

**v1.0.0 introduces zero new SIMPLIFICATION / APPROXIMATION / DEVIATION**.
The v1.0.0 deliverables (api-stability.md, ADR-0035, perf-commitment-v1_0_0.md,
perf baseline JSON template) are policy / infrastructure artefacts with no
algorithmic content.

### Rule 4 — Cite original literature for every EXTENSION

> *"Cite original literature for every EXTENSION (Remizov 2025, HLW 2006,
> BCO-R 2009, Söderlind 2002, etc.)."*

**STATUS: PASS** — all eleven math-introducing ADRs cite primary literature.

| ADR | Subject | Primary citations |
|-----|---------|-------------------|
| ADR-0001 | Contract-first math library framing | Remizov 2025 (Theorem 6) |
| ADR-0006 | Strang symmetrization | Strang 1968; HLW 2006 §III.5 |
| ADR-0008 | Self-adjoint variable-`a` (option ζ) | HLW 2006 §III.5 Thm 4.1 |
| ADR-0011 | TruncatedExp / Magnus integrator (option ε) | Theorem 6 (Remizov) + HLW 2006 §III.4 + BCOR 2009 Theorem 3; Magnus 1954 (corrected naming, Amendment 1 v0.7.0) |
| ADR-0012 | Tensor-product 2D (`Strang2D`) | HLW 2006 §III.5; math.md §10 Theorem 7 |
| ADR-0013 | 4th-order spatial (Diffusion4thChernoff) | Cubic-Hermite (ADR-0005); HLW 2006 |
| ADR-0014 | Adaptive PI controller | Söderlind 2002 |
| ADR-0015 | 6th-order spatial (Diffusion6thChernoff) | Mickens 1994 §3.2 (no 6th-order Magnus); Fornberg FD weights |
| ADR-0016 | NS2D scalar-`c` (NonSeparable2DChernoff) | HLW 2006 §III.5; Theorem 6 |
| ADR-0023 | Anisotropic 2D NS (NonSeparable2DAnisotropicChernoff) | HLW 2006 §III.5.2; Theorem 7 (with M → M_β relabel) |
| ADR-0024 | 3D tensor (Strang3D) | HLW 2006 §III.5.2; Theorem 7' (math.md §10.8.2) |
| ADR-0034 | `with_closure` API for variable `a(x)` | Remizov 2025; ADR-0008 chain |

Infrastructure / policy ADRs (0002, 0003, 0004, 0005, 0007, 0009, 0010,
0017, 0018, 0019, 0020, 0021, 0022, 0025, 0026, 0027, 0028, 0029, 0030,
0031, 0032, 0033, 0035) do **not** introduce math and are out of scope for
Rule 4. No ADR introducing math lacks a primary citation.

### Rule 5 — Run a fidelity audit (researcher agent) at every MAJOR (X.0.0) release

> *"Run a fidelity audit (researcher agent) at every MAJOR (X.0.0) release."*

**STATUS: SATISFIED BY THIS DOCUMENT.** Researcher agent invoked by Anchor
at the v1.0.0 cut (S2.4); deliverable is this audit-findings file.

---

## 4. Heavy-Validation Slopes — APPROVED (prod HW, 2026-05-10)

**Hardware**: Intel i7-12700K (12C/20T, Linux Artix, rustc 1.94.1).
**Working tree**: `/tmp/remizov-v1-validate-wt/` (isolated from v1.0.0 tag a4293ec).

All four heavy-validation gates confirmed PASS. Slopes are byte-exact matches
to the v0.11.0 I12 baseline, confirming that v0.12.0's Copy → Clone cascade
and `Storage<F>` dispatch (ADR-0034) introduced zero numerical regression.
Math soundness is attested by sympy NORMATIVE gates (§ 3 Rule 1); slope
gates measure empirical convergence rate on the discretised paths.

| Gate | N basket | Slope | Threshold (window) | Wallclock | Status | vs v0.11.0 |
|------|----------|-------|--------------------|-----------|--------|------------|
| G3⁶-2D FLAGSHIP | {503, 997, 1999} prime | -6.0837 | window [-6.15, -5.85] | 585 s | **PASS** | byte-exact match |
| G4_NS2D_aniso | {32, 64, 128, 256} | -2.1965 | ≤ -1.95 | 124 s | **PASS** | byte-exact match |
| G5_3D | {32, 64, 128, 256} | -2.1735 | ≤ -1.95 | 262 s | **PASS** | byte-exact match |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (constant β) | N ∈ {16, 32, 64} × threads ∈ {1, 2, 4} | — | abs_diff == 0.0 | 0.08 s | **PASS** | matches sequential |
| NS2D_ANISO_PARALLEL_BIT_EQUAL (variable β) | N ∈ {16, 32, 64} × threads ∈ {1, 2, 4} | — | abs_diff == 0.0 | 0.07 s | **PASS** | matches sequential |

**Total test-flagship wallclock**: 971 s (~16 min). Matches v0.11.0 baseline 987 s.
**EXIT=0**.

### Earlier partial-evidence context

An autonomous session (2026-05-10, 02:00–05:00 local) on an Intel i7-4700MQ
(4C/8T, 2.4 GHz, 16 GB RAM) confirmed G5_3D (-2.1735, 2606 s) and
G4_NS2D_aniso (-2.1965, 492 s) with byte-exact slopes. G3⁶-2D was killed
after 1h26m under oversubscription on that 4-core host; the prod HW rerun
above resolves that. All slopes from the partial-evidence run are reproduced
byte-for-byte on the prod HW.

---

## 5. API Surface (S2.2 audit applied)

S2.2 audit landed at commit `d3ff8b9` (`docs/api-stability.md`) and
`9419b2c` (ADR-0035, perf infra, follow-up). Five modules are explicitly
demoted to experimental (excluded from the v1.0.0 freeze):

| Module | Justification |
|--------|---------------|
| `mod grid_cubic` | Internal interpolation kernel; not re-exported from `lib.rs`; subject to algorithmic revision |
| `mod grid_quintic` | Internal interpolation kernel; not re-exported; subject to algorithmic revision |
| `mod simd` | Feature-gated; intrinsic availability and dispatch logic may change across toolchain versions (ADR-0019) |
| `mod strang2d_parallel` | Feature-gated `parallel`; scheduling model may change with Rust threading primitives (ADR-0018) |
| `mod strang3d_parallel` | Feature-gated `parallel`; analogous rationale; the public-facing dispatch via `Strang3D` is covered (v0.11.0 I12 mirror of ADR-0018) |

Plus: `mod diffusion_storage` (`pub(crate)` — `Storage<F>` enum backing
`DiffusionChernoff`; layout may change without notice), all `#[doc(hidden)]`
items, `xtask`, and test/bench harnesses.

**Field-freeze footnote** (S2.2 recommendation, applied in §7 of
`api-stability.md`): four structs expose public fields directly
(`AdaptivePI`, `AdaptiveOutcome`, `ShiftChernoff1D`, `NonSeparable2DChernoff`)
— field names are part of the freeze.

**Cross-binding freeze**: Rust + FFI (`remizov.h` regen) + PyO3
(`semiflow` pyclass) + WASM (`semiflow-wasm` `#[wasm_bindgen]`) all freeze
simultaneously per ADR-0028 §"API stability".

---

## 6. Performance Commitment (S2.3 infrastructure in place)

S2.3 deliverables landed at `9419b2c`:

- **5 criterion benchmarks** registered:
  - `crates/semiflow-core/benches/heat_1d.rs`
  - `crates/semiflow-core/benches/heat_2d.rs`
  - `crates/semiflow-core/benches/advdiff_2d.rs`
  - `crates/semiflow-core/benches/diffusion6_simd.rs`
  - `crates/semiflow-py/benches/evolve_bench.rs`
- **Baseline template**: `docs/perf/baseline-v1_0_0.json` (median + std-dev
  per benchmark; values TBD pending maintainer prod-HW run).
- **Nightly CI bench job**: `.github/workflows/nightly.yml` —
  `bench-regression` job (>5 % regression fails; criterion baseline diff).
- **Policy doc**: `docs/perf-commitment-v1_0_0.md` (§ 1 Status: effective
  from v1.0.0; § 2 What We Measure; § 3 Targets; § 4 Methodology with
  warmup / measurement-time / compilation flags).

Performance numbers are **explicitly non-contractual** per
`api-stability.md` § 6: API freeze covers types and method signatures, not
throughput. The 5 % regression threshold in nightly CI is informational —
no PATCH release will silently regress beyond it without CHANGELOG
rationale.

---

## 7. Carry-Forward Findings Reconciliation

| Finding | Origin | Status at v1.0.0 |
|---------|--------|------------------|
| O-1 | v0.9.0 (heavy-validation slopes for G4_NS2D_aniso + G5_3D) | **CLOSED** — all slopes confirmed byte-exact on prod HW (i7-12700K, 2026-05-10); G3⁶-2D -6.0837 ∈ [-6.15, -5.85]; G4_NS2D_aniso -2.1965 ≤ -1.95; G5_3D -2.1735 ≤ -1.95; see §4. |
| O-2 | v0.9.0 (NS2D-aniso parallel bit-equal regression gate missing) | **CLOSED** at commit `daa4019` (v0.12.0); two regression tests under `parallel,slow-tests`, 6 configurations PASS |
| O-3 | v0.9.0 (`NonSeparable2D` deprecation policy at v1.0.0) | **CLOSED** by ADR-0033 — both `NonSeparable2DChernoff` (scalar-`c`, v0.7.0) and `NonSeparable2DAnisotropicChernoff` (anisotropic-`β`, v0.9.0) promoted as first-class APIs; no `#[deprecated]` cycle |
| O-4 | v0.10.0 (variable `a(x)` across FFI/PyO3/WASM) | **CLOSED for 1D** at commits `f097309` (ADR-0034 design) + `2c8ca6f` (core `with_closure`) + `ec21002` (FFI/PyO3/WASM mirrors); audit stub `e5ee774` records the v0.12.0 deferral. 2D / 3D variable-`a` deferred to **v0.13.0** per ADR-0034 § "Out of scope" — **NOT a v1.0.0 freeze blocker** (1D variable-`a` covers the documented API contract; 2D/3D extension does not retroactively break the freeze, since `with_closure` ergonomic surface is forward-compatible) |
| D-1 | v0.12.0 (slope-gate prod-HW rerun) | **CLOSED** — all four gates PASS on prod HW (i7-12700K, 2026-05-10); see §4. |
| D-2 | v0.12.0 (Safari headless WASM smoke) | **DEFERRED-V1.0.x** (macOS runner not provisioned; no math impact) |
| D-3 | v0.12.0 (I14 async PyO3 API) | **DEFERRED-V0.12.1+** (insufficient telemetry on GIL-release saturation; ADR-0034 § "Out of scope") |

**Net OPEN math findings at v1.0.0**: **0** (all open items are infrastructure
deferrals; none touch math correctness).

---

## 8. Suckless Audit Pass

Carried forward from v0.12.0 audit § 8 (no new code in v1.0.0 stability
deliverables — pure docs / ADR / config / bench infra):

- **Direct deps (`semiflow-core`)**: 2 (`num-traits`, `libm`) — unchanged
  since v0.7.0.
- **Largest src file**: `state.rs` was split in v0.12.0 (`state_1d.rs` /
  `state_2d.rs` / `state_3d.rs`); all under the 500-LoC cap.
  `strang3d.rs` 566 LoC remains under v0.11.0 grandfather.
- **Functions ≤ 50 LoC**: `with_closure` constructor ~12 lines; accessors
  ~4 lines each; `Storage` enum ~15 lines. v1.0.0 adds none.
- **`unsafe` scope**: confined to `src/simd/{x86_64,aarch64}.rs` per
  ADR-0019 + the FFI/PyO3/WASM `#![allow(unsafe_code)]` proc-macro
  expansion zones per ADR-0028. Zero new `unsafe` in v1.0.0 deliverables.
- **Public API delta vs v0.12.0**: zero (S2.2 audit only demotes / annotates
  — does not add public items). Workspace version bumps `0.12.0 → 1.0.0`.

---

## 9. Approval Criteria

All criteria satisfied (re-evaluated at S2.9 by reviewer-suckless):

- [x] All NORMATIVE sympy gates PASS — **18/18 PASS** (this audit, § 3 Rule 1).
- [x] All heavy-validation slope gates PASS within their windows —
      G3⁶-2D -6.0837 ∈ [-6.15, -5.85] **PASS**; G4_NS2D_aniso -2.1965 ≤ -1.95
      **PASS**; G5_3D -2.1735 ≤ -1.95 **PASS** (prod HW, 2026-05-10 — see §4).
- [x] NS2D_ANISO_PARALLEL_BIT_EQUAL byte-identical — **PASS** (constant β + variable β,
      prod HW, 2026-05-10 — see §4).
- [x] 0 OPEN math findings — confirmed (§ 7).
- [x] 0 unresolved DEVIATIONs — confirmed (§ 3 Rule 3).
- [x] `cargo test --workspace` green at the v1.0.0 commit — confirmed (EXIT=0, prod HW).
- [x] `cargo run -p xtask -- test-flagship` green on prod HW — confirmed (EXIT=0, 971 s).
- [x] Final reviewer-suckless gate (S2.9) — **CONDITIONAL-GO promoted to GO** (all
      blocking items resolved; deferred items D-2/D-3 non-blocking per ADR-0034).

**Status: APPROVED**. The v1.0.0 tag is unblocked.

---

## 10. Sources Cross-Referenced

- ROADMAP.md § "Math fidelity commitment" (lines 360-370) — five rules.
- ROADMAP.md § "v1.0.0" (lines 330-345) — three theme deliverables.
- `docs/audit-findings-v0_11_0.md` — v0.11.0 I12 prod-HW slope numbers
  (cited as v1.0.0 baseline reference in §4).
- `docs/audit-findings-v0_12_0.md` — v0.12.0 binding expansion; D-1/D-2/D-3
  carry-forward.
- `docs/audit-findings-v0_10_0.md` — five SIMPLIFICATIONs documented; cite-coverage
  posture for binding crates.
- `docs/audit-findings-v0_9_0.md` / `docs/audit-findings-v0_8_x.md` —
  NORMATIVE gate seal points (T7N, T9N, T10N).
- `docs/audit-findings-v0_6_0.md` — D1 / D2 DEVIATION closures (referenced
  in § 3 Rule 3 table).
- `docs/api-stability.md` (commit `d3ff8b9`) — S2.2 audit output;
  demoted-modules table; field-freeze footnote.
- `docs/perf-commitment-v1_0_0.md` (commit `9419b2c`) — S2.3 infra.
- `docs/adr/0035-v1_0_0-api-stability.md` — ADR for v1.0.0 freeze posture.
- `.dev-docs/verification/scripts/verify_v0_7_0_nonseparable.py` /
  `verify_v0_9_0_nonseparable_aniso.py` /
  `verify_v0_9_0_3d_tensor.py` — sympy NORMATIVE gate scripts.
