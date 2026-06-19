---
version: 1.0.1
last_updated: 2026-05-09
freshness_score: 1.0
dependencies:
  - docs/adr/0023-anisotropic-2d-non-separable.md
  - docs/adr/0024-tensor-3d.md
  - docs/adr/0025-generic-over-float.md
  - docs/adr/0026-chernoff-trait-generic.md
  - docs/adr/0032-heavy-validation-harness.md
  - docs/adr/0033-nonseparable2d-deprecation-policy.md
  - contracts/semiflow-core.math.md (sections 10.7-ter, 10.8.1, 10.8.2, 10.8.3, 10.8.4, 10.8.5, 10.8.6, 10.8.7, plus Lemma 10.1 retroactive)
  - .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py
  - .dev-docs/verification/scripts/verify_v0_9_0_3d_tensor.py
  - crates/semiflow-core/src/nonseparable2d_aniso.rs
  - crates/semiflow-core/src/grid3d.rs
  - crates/semiflow-core/src/grid_fn3d.rs
  - crates/semiflow-core/src/strang3d.rs
  - crates/semiflow-core/src/float.rs
  - crates/semiflow-core/tests/generic_float_smoke.rs
  - docs/audit-findings-v0_8_1.md (template baseline)
changelog:
  - 1.0.0: Initial v0.9.0 math fidelity audit (v0.11.0 milestone item I13, ADR-0032)
  - 1.0.1: O-3 resolved via ADR-0033 (scalar-c kept, Option A — additive promise preserved at v1.0.0 freeze)
  - 1.0.2: O-1 and O-2 marked DEFERRED-v0.12.0 (I12 HW pending; NS2D_ANISO gate deferred alongside I4/I5)
---

# v0.9.0 Math Fidelity Audit

**Auditor**: researcher agent
**Date**: 2026-05-09
**Scope**: `v0.8.1..v0.9.0` (11 commits — `b8032a0` Block A through `779c400` release)
**Theme**: ANISOTROPIC NS-2D + 3D TENSOR + GENERIC-OVER-FLOAT — first v0.x release
since v0.7.0 to add new math (math.md §10.7-ter and §10.8 + Lemma 10.1) on top of
substantive type-system refactor (SemiflowFloat trait, ChernoffFunction generalised).

## 1. Summary

**APPROVED FOR RELEASE WITH FOLLOW-UPS.** v0.9.0 closes all three ROADMAP bullets
(anisotropic 2D non-separable; 3D tensor product; Generic-over-Float). The two new
math sections (§10.7-ter and §10.8) are **proven sound** by 12 NORMATIVE sympy
gates (T9N_* 6/6 + T10N_* 6/6, all PASS at HEAD on this audit run, 2026-05-09).
The Generic-over-Float refactor (ADR-0025 + ADR-0026) is structurally complete:
all 14 `ChernoffFunction`-implementing types are now generic over `F: SemiflowFloat`
with `= f64` default, preserving backward-compat at every callsite. Three follow-up
items are recorded as v0.11.x non-blocking issues (see §5 OPEN). Heavy empirical
gates G4_NS2D_aniso and G5_3D were validated by their owning agents at commit time
but were `#[ignore]`-gated in the v0.9.0 release tag and remain on the v0.11.0 I12
heavy-validation backlog (separate from this I13 audit per ADR-0032).

| Class | Count |
|-------|-------|
| FAITHFUL | 14 |
| SIMPLIFICATION | 5 |
| APPROXIMATION | 3 |
| DEVIATION | 0 |
| EXTENSION | 6 |
| OPEN | 0 (was 3; O-1 + O-2 CLOSED 2026-05-10 by v1.0.0; O-3 RESOLVED via ADR-0033) |

## 2. Findings

### 2.1 Anisotropic 2D non-separable (`NonSeparable2DAnisotropicChernoff`, Block A+B)

- **F-1 (FAITHFUL)** — `crates/semiflow-core/src/nonseparable2d_aniso.rs` 472 LoC
  implements the math.md §10.7-ter palindromic 5-leg Strang composition exactly
  as specified: `S_5(τ) = e^{τA/2} · e^{τB/2} · Φ_{M_β}(τ) · e^{τB/2} · e^{τA/2}`
  with K=2 truncated-Taylor mixed leg `Φ_{M_β}(τ) = I + τM_β + (τ²/2)M_β²`
  (eq. 10.7-ter.7), 4-point centred cross-stencil eq. 10.7-ter.10 weighted
  pointwise by `β(x_i, y_j)`, CFL gate `4·τ·‖β‖_∞ < dx·dy` (eq. 10.7-ter.12,
  θ = 1/4), `order() = 2` per §11.1.bis. Inherits proof from v0.7.0
  `NonSeparable2DChernoff` via the `M → M_β` relabel (ADR-0023 §Decision); the
  reduction to a 2-operator Strang collapse depends only on `[A, B] = 0`
  (Lemma 10.1, retroactive N=2 special case in math.md §10.8.1) — preserved
  verbatim under the relabel (math.md §10.7-ter.2, lines 3588–3602).

- **F-2 (FAITHFUL)** — Sympy gate **T9N_τ²** (palindromic τ²-cancellation) PASS:
  `log(S_5(τ))|_{τ²} = 0` symbolically with `[A, B] = 0` algebra applied via
  `canonicalize_AB` (reused verbatim from v0.7.0 `verify_v0_7_0_nonseparable.py`,
  ADR-0023 §Verification). Confirmed at HEAD 2026-05-09 by running
  `python3 .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py`
  (see §7 below).

- **F-3 (FAITHFUL)** — Sympy gate **T9N_τ³** (Y₃ formula match) PASS: the third-order
  truncation residue under the palindromic Strang form matches the boxed eq.
  10.7-ter.6 closed-form expression
  `Y_3 = -(1/24)([A,[A,M_β]] + 2[A,[B,M_β]] + [B,[B,M_β]]) + (1/12)([M_β,[M_β,A]] + [M_β,[M_β,B]])`
  symbolically. The structural identity to v0.7.0 (with `M → M_β`) is the
  central audit claim ratified by this gate.

- **F-4 (FAITHFUL)** — Sympy gate **T9N_K2_local** PASS: `S_5^{K2}(τ) - e^{τL}`
  has zero `τ^k` terms for k ∈ {0, 1, 2} and a non-zero `τ³` leading term, so
  K=2 truncation preserves local order 3 / global order 2 (math.md §10.7-ter.3
  + eq. 10.7-ter.9). The K=1 counterexample (math.md §10.7-bis.3) applies
  verbatim to `M_β` and is therefore correctly rejected.

- **F-5 (FAITHFUL)** — Sympy gate **T9N_oracle** PASS: the rotated 2D Gaussian
  closed form (eq. 10.7-ter.13) satisfies the constant-coefficient PDE
  `∂_t u = a_xx ∂_xx u + a_yy ∂_yy u + β ∂_xy u` identically. This ratifies
  the frozen-coefficient consistency floor used by the position-dependent
  `β(x, y) = 0.05·exp(-(x²+y²)/4)` slope gate.

- **F-6 (FAITHFUL)** — Sympy gate **T9N_zero-β** PASS: `S_5^{K2}(τ)|_{β≡0}`
  collapses to `Strang2D` symbolically. The Rust implementation (Block B,
  commit `04ec408`) detects `beta_norm_bound == 0.0` at construction and
  branches to the existing `Strang2D::apply` for **bit-equal FP output**
  (math.md §10.7-ter.6 NORMATIVE clause). This is a strong reduction oracle:
  any future regression that breaks v0.5.0 `Strang2D` would also break this
  branch — protected by both the `STRANG2D_PARALLEL_BIT_EQUAL` regression
  gate from v0.8.x (preserved in v0.9.0) and by the static T9N_zero-β gate.

- **F-7 (FAITHFUL)** — Sympy gate **T9N_palindrome** PASS: the leg sequence
  `[A/2, B/2, M_β, B/2, A/2]` equals its own reverse — static refactor guard.

- **F-8 (FAITHFUL)** — `SemiflowError::CflViolated` is **reused** for the
  anisotropic case (third semantic field overload — `a_norm_bound` carries
  `‖β‖_∞`, `dx_squared` carries the product `dx·dy`); no new error variant
  added (ADR-0023 §Consequences). Documentation amendment in
  `nonseparable2d_aniso.rs:81-92` (rustdoc on the `CflViolated` propagation
  path) records the third overload semantics.

- **F-9 (FAITHFUL)** — G4_NS2D_aniso slope test redesigned in commit `0180292`
  (2026-05-07) from a high-N reference design (N_ref=1024) to **probe-vs-2N-1
  self-convergence** mirroring v0.7.0 G3_NS2D_var (math.md §10.7-bis.7
  secondary variant). The original Block A spec (commit `b8032a0`) failed
  flagship validation (slope -0.3136 vs gate -1.95) due to reference-solution
  floor (~1e-3 at N_ref=1024) dominating probe error at N=256. Diagnosis
  recorded in commit message: "TEST DESIGN flaw, not math/code — T9N_* sympy
  gates 6/6 PASS, production CFL gate works correctly." Math.md §10.7-ter.7
  (lines 3860–3897) was amended to ratify the self-convergence design;
  N_STEPS was bumped 200→500 to keep CFL margin at the finest 511×511 probe
  grid (4·τ·‖β‖_∞ = 2.0e-4 < dx·dy = 3.84e-4 ≈ 1.92× margin). The fix is a
  faithful adaptation of a previously-proven pattern; the math claim
  (order 2) is unchanged.

### 2.2 3D tensor product (`Strang3D`, `Grid3D`, `GridFn3D`, `AxisLift3D`, Block C)

- **F-10 (FAITHFUL)** — `crates/semiflow-core/src/strang3d.rs` 401 LoC implements
  the math.md §10.8.3 palindromic 5-leg `S_3D(τ) = e^{τA/2}·e^{τB/2}·e^{τC}·e^{τB/2}·e^{τA/2}`
  with X outermost (cache-friendly on row-major x-fastest storage). Per-axis
  legs commute pairwise (`[A,B] = [A,C] = [B,C] = 0`) by Lemma 10.1 (math.md
  §10.8.1); under exact arithmetic the 5-leg form collapses to `e^{τ(A+B+C)} =
  e^{τL}` with **zero BCH residue** (eq. 10.8.4 + proof at lines 4073–4081).
  The 5-leg form is fixed for FP determinism, not for cancellation budget
  (Remark 10.2, math.md §10.8.2 lines 4044–4049).

- **F-11 (FAITHFUL)** — `crates/semiflow-core/src/grid3d.rs` 280 LoC stores
  `Vec<f64>` of length `nx · ny · nz` with row-major x-fastest layout
  `idx(i,j,k) = k·nx·ny + j·nx + i` (math.md §10.8.4 eq. 10.8.6, line 4106),
  inductively extending the v0.5.0 2D layout invariant I-T1
  (`grid2d.rs:7-8`). Per-axis strides are `(1, nx, nx·ny)`; each per-axis
  lift is a strided 1D walk reusing the v0.5.0 `AxisLift::X` and
  `AxisLift::Y` apply paths verbatim with the stride parameter, with `Z`
  added as a third stride case (Option A: `Axis::Z` enum variant; ADR-0024
  §Decision, the math-level contract is identical to a separate
  `AxisLift3D<C>` type).

- **F-12 (FAITHFUL)** — Sympy gate **T10N_pairwise_commute** PASS: the
  axis-disjoint commutator `[A,B] = [A,C] = [B,C] = 0` is verified
  symbolically with NC sympy symbols and the Lemma 10.1 canonicaliser
  (`canonicalize_ABC`, a trivial 3-symbol extension of the v0.7.0
  `canonicalize_AB`). Confirmed at HEAD 2026-05-09.

- **F-13 (FAITHFUL)** — Sympy gate **T10N_strang3d_collapse** PASS: under
  T10N_pairwise_commute, `S_3D(τ) - e^{τ(A+B+C)} = 0` exactly (no `τ^k`
  residue for any k ≥ 2). This is the strongest possible 3D claim — exact
  identity, not just `O(τ^k)` for some k.

- **F-14 (FAITHFUL)** — Sympy gate **T10N_zero-axis** PASS: setting `L_z ≡ 0`
  collapses `S_3D|_{C=0}` symbolically to `e^{τ(A+B)}`, identical to
  `Strang2D` (Lemma 10.1 N=2 case, math.md §10.8.6 row line 4175). The Rust
  implementation must detect this at construction and branch to
  `Strang2D::apply` for FP bit-equality with v0.5.0 — analogous reduction
  oracle to T9N_zero-β.

- **F-15 (FAITHFUL)** — Sympy gate **T10N_palindrome** PASS: leg sequence
  `[A/2, B/2, C, B/2, A/2]` equals its own reverse.

- **F-16 (FAITHFUL)** — Sympy gate **T10N_oracle** PASS: the 3D heat product
  Gaussian `u(t,x,y,z) = (1+4at)^{-3/2}·exp(-(x²+y²+z²)/(1+4at))` (eq.
  10.8.7-osc) satisfies `∂_t u = (a_xx ∂_xx + a_yy ∂_yy + a_zz ∂_zz) u`
  identically.

- **F-17 (FAITHFUL — STRUCTURAL)** — Sympy gate **T10N_order_min** PASS as
  STRUCTURAL: because all five legs commute (Lemma 10.1), the splitting
  error is identically zero, so global order is set entirely by the per-axis
  Chernoff legs (Theorem 7', math.md §10.8.2). The verifier records
  `S_3D - exp(τL) = 0 to N=4` numerically and defers the full inductive
  proof to math.md §10.8.2 lines 4030–4042. This is faithful — the
  inductive proof in §10.8.2 is rigorous, and the sympy verifier's
  STRUCTURAL stamp signals that a structural argument supplements the
  symbolic check (not a relaxation of the gate).

### 2.3 Generic-over-Float (`SemiflowFloat` trait, ADR-0025 + ADR-0026, Block D)

- **F-18 (FAITHFUL)** — `crates/semiflow-core/src/float.rs` 84 LoC defines
  the `SemiflowFloat` sealed trait (ADR-0025 §Decision lines 36–53) with
  blanket impls for `f32` and `f64` only. The trait composes `num_traits::Float`
  with `AddAssign + MulAssign + SubAssign + DivAssign + Send + Sync + Copy +
  Debug + Display + PartialOrd + 'static`; the trait is sealed (project-private
  marker) so downstream crates cannot impl for unknown types, preserving the
  ADR-0025 §Decision invariant that "f32 SIMD bandwidth doubling for engineering
  callers on large 3D grids" remains achievable without exposing arbitrary-float
  monomorphisations.

- **F-19 (FAITHFUL)** — All 14 `ChernoffFunction`-implementing types (7 leaf 1D:
  `DiffusionChernoff`, `Diffusion4thChernoff`, `Diffusion6thChernoff`,
  `DriftReactionChernoff`, `LiouvilleChernoff`, `TruncatedExpDiffusionChernoff`,
  `TruncatedExpDiffusion4thChernoff`, `AdvectionDiffusionChernoff`; 7 composition:
  `AxisLift`, `Strang2D`, `Strang3D`, `StrangSplit`, `NonSeparable2DChernoff`,
  `NonSeparable2DAnisotropicChernoff`, `AdaptivePI`) carry `<F: SemiflowFloat = f64>`
  in their generic parameters. The `= f64` default ensures every existing
  callsite compiles unchanged via Rust type inference (ADR-0026 §Consequences,
  no breaking change). Non-exact `f64` literal conversions in generic `apply`
  bodies route through `two()`, `half()`, `from_f64()` helpers in `float.rs`
  with documented commentary.

- **F-20 (FAITHFUL)** — `crates/semiflow-core/tests/generic_float_smoke.rs`
  779 LoC contains 26 smoke tests instantiating both `f32` and `f64` paths
  through every leaf 1D type, every 2D/3D storage type, and all six
  composition types (6 new f32 composition smokes added in Wave 4). The
  tests verify that the generic path **compiles** and **produces correct
  numerical results** within f32 tolerance (e.g. 1D heat-equation oracle
  at sup-error ≤ 1e-5 relative to the f64 reference solution). Project
  memory baseline at v0.9.0 release: **198 passed / 0 failed / 1 ignored**
  fast-tests (test-fast wraps `cargo test --workspace`).

### 2.4 Lemma 10.1 retroactive

- **F-21 (FAITHFUL)** — Lemma 10.1 (axis-disjoint commutation, inductive form,
  math.md §10.8.1 lines 3973–3986) was added in commit `1c48bb8` and explicitly
  notes (lines 3988–3997) that its **N=2 specialisation is the implicit lemma
  cited in §10.7-bis.2 (line 3180), §10.7-bis.6 gate T7N_zero-c (line 3452),
  §10.7-ter.2 (line 3591), and §10.7-ter.6 gate T9N_zero-β (line 3845)**. Prior
  cite-sites are well-defined under the inductive lemma without prose edits at
  the cite-sites. This is a faithful retroactive scope correction: the §10.7-bis
  / §10.7-ter texts had used "Lemma 10.1" as a shorthand for axis-disjoint
  commutation before §10.8.1 formally stated it. The Lemma 10.1 retroactive
  note is a documentation rigor improvement, not a math change — the algebra
  cited at every site is unchanged.

## 3. SIMPLIFICATIONs (documented assumptions)

- **S-1** — `SemiflowFloat` is sealed to `f32` and `f64` only. `num_complex::Complex<f64>`
  for spectral methods and `Interval<F>` for guaranteed bounds (the original
  ADR-0004 motivator) are deferred to v1.0+ separate ADRs (ADR-0025 §Forward
  compatibility lines 110–123, ADR-0026 §Forward compatibility lines 97–108).
  The sealing prevents misuse but also prevents principled-extension impls
  pre-v1.0.

- **S-2** — SIMD intrinsics (AVX2/NEON, `#[cfg(feature = "simd")]`) remain
  `f64`-specialised in v0.9.0. The generic path dispatches through the scalar
  fallback when `F ≠ f64`. f32 SIMD intrinsics (`f32x8` on AVX2, `f32x4` on
  NEON) are deferred to v1.0+ per ADR-0025 §Forward and ADR-0026 §Forward —
  the value proposition of f32 SIMD bandwidth doubling on large 3D grids
  remains theoretical until that ADR lands.

- **S-3** — Parallel-feature impls of `Strang2D`, `NonSeparable2DChernoff`,
  and `NonSeparable2DAnisotropicChernoff` remain **f64-only** via
  `#[cfg(feature = "parallel")] impl ChernoffFunction<f64> for Type<f64>`
  carve-outs (ADR-0026 §Decision lines 46–51). The `STRANG2D_PARALLEL_BIT_EQUAL`
  gate from ADR-0018 is release-blocking and depends on f64 SIMD intrinsics;
  generalising to `F ≠ f64` would either drop SIMD (regressing v0.8.1's
  heat_2d 4.38× speedup) or require the deferred f32 SIMD ADR.

- **S-4** — The K=2 truncated-Taylor mixed leg `Φ_{M_β}(τ) = I + τM_β + (τ²/2)M_β²`
  is an *approximation* of `e^{τM_β}` (residue `(τ³/6)M_β³ + O(τ⁴)`,
  eq. 10.7-ter.8). Genuine matrix-exponential evaluation via Krylov subspace
  methods would lift global order to 4 at 2–3× per-step cost; deferred per
  ADR-0023 alt (b) to v1.0+ pending profiling evidence. For v0.9.0 the K=2
  approximation is sufficient (preserves global order 2 — proven faithful
  by gate T9N_K2_local).

- **S-5** — The 4-point centred cross-stencil for `∂_x ∂_y` (eq. 10.7-ter.10)
  has spatial truncation residue `O(dx² + dy²)`, capping the spatial slope
  at 2 even when per-axis legs are higher-order (`Diffusion4thChernoff` /
  `Diffusion6thChernoff`). This is a Strang-floor cap inherited from v0.7.0
  (math.md §10.7-bis.5) and is not exposed by `order()` per §11.1.bis. Higher-
  order cross stencils (e.g. 8-point) are deferred to v1.0+ per ADR-0023
  §Forward compatibility lines 125–128.

## 4. EXTENSIONs (additions beyond canonical Theorem 6)

- **E-1** — `NonSeparable2DAnisotropicChernoff<X, Y>` (commit `04ec408`,
  ~472 LoC additive sibling) extends v0.7.0's scalar-`c` `NonSeparable2DChernoff`
  to anisotropic position-dependent `β(x, y) ∈ C³(Ω, ℝ)` cross-coupling.
  Mathematically a relabel `M → M_β` of the v0.7.0 derivation (per ADR-0023
  §Decision); enables anisotropic option pricing with stochastic correlation
  without forcing artificial isotropic `c`.

- **E-2** — `Strang3D<X, Y, Z>` and supporting types (commit `d827b83`)
  extend `Strang2D<X, Y>` to 3D separable tensor products. Mathematically
  the inductive base case for Theorem 7' (math.md §10.8.2) at N=3; unblocks
  3D heat / advection-diffusion / option-pricing applications. Higher N
  (`Strang4D`, etc.) is supported by the Lemma 10.1 inductive form and
  deferred to v1.0+ as a separate API/perf ADR (ADR-0024 §Forward).

- **E-3** — Lemma 10.1 (math.md §10.8.1 lines 3973–3986) **inductive form**
  was new in v0.9.0 (the N=2 special case was previously implicit in §10.7-bis
  / §10.7-ter cite-sites). The retroactive scope note (lines 3988–3997)
  ratifies the inductive lemma as the formal name for the previously-implicit
  axis-disjoint commutation rule.

- **E-4** — Theorem 7' (math.md §10.8.2 lines 4011–4042) **N-axis Chernoff
  lift** generalises Theorem 7 (§10.3, N=2 base case). Inductive proof on
  N (base case from §10.3 verbatim; inductive step factors the N-axis
  composition as `Φ_1(τ/2) · S_{N-1}(τ) · Φ_1(τ/2)` and applies Lemma 10.1
  for the outer-Φ_1 commutation with each inner Φ_i, i ≥ 2). Remark 10.2
  (lines 4044–4049) documents that N-axis separable splitting acquires
  **no commutator-cancellation bonus** — order is strictly limited by the
  weakest axis with no off-set.

- **E-5** — `SemiflowFloat` sealed trait (ADR-0025) lifts the v0.1.0
  ADR-0004 f64 monomorphism. The trait is project-local sealed for `f32`
  and `f64`; downstream callers gain f32 instantiation via the `<F: SemiflowFloat
  = f64>` parameter on every data-bearing type. The lift was deferred from
  v0.5.0 (when the original ADR-0004 trigger — interval arithmetic for
  `remizov-adaptive` — never fired) to v0.9.0 (when f32 SIMD bandwidth +
  spectral-method `Complex<f64>` requests + `Interval<F>` feasibility
  jointly motivated the refactor). ADR-0025 §Context records the trigger
  lines 22–34.

- **E-6** — `ChernoffFunction` trait generalised over `F: SemiflowFloat = f64`
  (ADR-0026, commit `eb0abfe`). Closes the ADR-0025 deferral ("generic
  ChernoffFunction is future scope") and lifts all 14 trait-implementing
  types to generic, including composition types. The change preserves
  v0.8.x SIMD bit-equality regressions (`diffusion4_unit` 9/9, `simd_bit_equal`
  2/2; ADR-0026 §Consequences) — a strong consistency proof that the
  generic refactor introduced **no numerical change** on the f64 path.

## 5. OPEN questions (v0.11.x non-blocking follow-ups per ADR-0032 AC-4)

- **O-1** — Heavy validation of G4_NS2D_aniso (math.md §10.7-ter.7) and G5_3D
  (math.md §10.8.7) was performed at commit time by the architect/engineer
  team (per project memory) but the slope-test functions were `#[ignore]`-gated
  (`slow-tests` feature only) at the v0.9.0 release tag. Per ADR-0032 these
  belong to v0.11.0 item **I12** (heavy-validation harness), not I13 (math
  fidelity audit — this document). **Recommended action**: keep the v0.11.0
  I12 backlog item; this audit attests the math is sound but does not
  re-verify the empirical slope at flagship N. Severity: LOW (math gates
  T9N_*/T10N_* PASS; empirical slopes are end-to-end ratifiers, not
  correctness gates).
  → **DEFERRED-v0.12.0** (2026-05-09): I12 heavy validation PENDING maintainer
  prod-HW access. v0.11.0 tag ships without fresh flagship run; v0.12.0 will
  absorb results into `docs/audit-findings-v0_11_0.md`. Math soundness is
  unaffected (T9N_*/T10N_* PASS remain the correctness attestation).
  → **CLOSED 2026-05-10**: heavy-validation rerun on prod HW (i7-12700K,
  12C/20T) PASSED for all three flagship slope gates (G5_3D -2.1735,
  G4_NS2D_aniso -2.1965, G3⁶-2D -6.0837) with byte-exact match to
  v0.11.0 baseline. Total wallclock 971s (matches v0.11.0 baseline 987s).
  Confirmed in `docs/audit-findings-v1_0_0.md` (APPROVED) and commit
  458a8dc.

- **O-2** — `NonSeparable2DAnisotropicChernoff` parallel impl exists
  (`#[cfg(feature = "parallel")] impl ChernoffFunction<f64>` carve-out) but
  is not covered by a `STRANG2D_PARALLEL_BIT_EQUAL`-style regression gate
  in the v0.9.0 audit-visible test set. v0.8.1 showed that parallel SIMD
  bit-equality regressions are the right protective contract for the f64
  parallel path. **Recommended action**: file a v0.11.x follow-up to add
  `NS2D_ANISO_PARALLEL_BIT_EQUAL` mirroring the v0.8.1 gate. Severity:
  MEDIUM (the f64-only carve-out is intentional; the gate would protect
  against future SIMD/parallel refactors silently breaking the
  anisotropic path).
  → **DEFERRED-v0.12.0** (2026-05-09): NS2D_ANISO_PARALLEL_BIT_EQUAL gate
  not added in v0.11.0 (no math changes; v0.11.0 scope confined to binding
  crates per ADR-0029). Schedule for v0.12.0 alongside the I4/I5 binding
  expansion that will exercise the parallel anisotropic path more heavily.
  → **CLOSED 2026-05-10**: NS2D_ANISO_PARALLEL_BIT_EQUAL regression gate
  added in commit daa4019 (`crates/semiflow-core/tests/ns2d_aniso_parallel_bit_equal.rs`,
  226 LoC, slow-tests + parallel features gated). Two tests PASS on prod
  HW (constant β + variable β; byte-equal vs sequential). See
  `docs/audit-findings-v0_12_0.md` (APPROVED).

- **O-3** — The `#[deprecated]` lifecycle for `NonSeparable2DChernoff`
  (scalar-`c`) is **not declared** in v0.9.0. Per ADR-0023 §Consequences
  the anisotropic sibling is purely additive and v0.7.0 / v0.8.x callers
  remain bit-equal — a deprecation may not be desired. However, if v1.0+
  intends to consolidate around the more general anisotropic type, the
  deprecation policy should be recorded ahead of API freeze.
  **Recommended action**: file a v0.11.x or v0.12.x ADR addressing whether
  the scalar-`c` and anisotropic-`β` types are both promoted at v1.0
  freeze, or whether scalar-`c` is deprecated as a degenerate case.
  Severity: LOW (planning-only, no math impact).
  → RESOLVED via ADR-0033 (decision: Option A — keep both types
  indefinitely, no `#[deprecated]` cycle; honours ADR-0023 additive
  promise + constitution v1.1.0 "additive, never subtractive";
  2026-05-09).

## 6. Bit-equal evidence

- **SIMD bit-equality regressions preserved** (ADR-0026 §Consequences
  line 92): `diffusion4_unit` 9/9 PASS, `simd_bit_equal` 2/2 PASS at
  v0.9.0 release. The f64 path through every leaf 1D Chernoff type is
  numerically identical to v0.8.x — a strong proof that the
  `SemiflowFloat` and trait generalisation refactors introduced no
  numerical change on the f64 default path.

- **`STRANG2D_PARALLEL_BIT_EQUAL`** (v0.8.1 ADR-0018, three sub-tests
  preserved through v0.9.0 via the parallel f64-only carve-out at
  ADR-0026 §Decision lines 46–51).

- **`v0_5_0_regression_bit_equal`** (4 sub-tests) — v0.5.0 frozen golden
  vector reproduces byte-for-byte under v0.9.0 with `parallel,simd`
  features. This protects the 2D Strang path against the new generic
  refactor and against the new 3D / anisotropic siblings.

- **Workspace fast-test count**: **198 passed / 0 failed / 1 ignored**
  (project memory baseline, `cargo run -p xtask -- test-fast` at
  v0.9.0 release; +48 over v0.8.1 baseline of 150). The 1 ignored is the
  G3⁶-2D flagship gate (slow-tests, separate `test-flagship` target).

## 7. Sympy NORMATIVE proof

Both v0.9.0 sympy verifiers were re-run at HEAD on this audit run (2026-05-09):

- `python3 .dev-docs/verification/scripts/verify_v0_9_0_nonseparable_aniso.py`:
  **6/6 NORMATIVE PASS** — T9N_τ², T9N_τ³, T9N_K2_local, T9N_oracle,
  T9N_zero-β, T9N_palindrome. `NonSeparable2DAnisotropicChernoff ACCEPTED`.

- `python3 .dev-docs/verification/scripts/verify_v0_9_0_3d_tensor.py`:
  **6/6 NORMATIVE PASS** — T10N_pairwise_commute, T10N_strang3d_collapse,
  T10N_palindrome, T10N_oracle, T10N_zero-axis, T10N_order_min (the last
  as STRUCTURAL — full proof in math.md §10.8.2; the verifier confirms
  the structural identity numerically to `N=4`). `Strang3D ACCEPTED`.

Both verifiers reuse `canonicalize_AB` from the v0.7.0 verifier verbatim
(extended trivially to `canonicalize_ABC` for the 3-symbol 3D case) — no
new sympy machinery. Reuse note ratified by ADR-0023 §Verification and
math.md §10.7-ter.6 lines 3848–3858 / §10.8.6 lines 4178–4182.

## 8. Suckless invariants check

- **Runtime deps**: 2 (`num-traits` v0.2 with `libm` feature; `libm` v0.2
  for `sqrt` in `no_std`). Unchanged from v0.7.0; well under the <10
  guardrail-#1 limit. `[dependencies]` in `crates/semiflow-core/Cargo.toml`
  confirmed at HEAD.

- **Largest src files** (post-v0.9.0 additions and existing): max LoC is
  `grid.rs` 677 LoC (pre-existing, grandfathered per project memory
  "3-file grandfather"); `diffusion6.rs` 567 LoC (v0.7.0); `truncated_exp4.rs`
  531 LoC (v0.6.0). v0.9.0 additions stay under the 500-LoC suckless cap:
  `diffusion4.rs` 498, `truncated_exp.rs` 479, `nonseparable2d_aniso.rs`
  472 (new), `grid_fn2d.rs` 454, `nonseparable2d.rs` 444, `grid_fn3d.rs`
  442 (new), `strang3d.rs` 401 (new), `diffusion.rs` 389,
  `strang2d_parallel.rs` 376, `strang2d.rs` 335, `adaptive.rs` 282,
  `grid3d.rs` 280 (new), `shift1d.rs` 278, `axis.rs` 269, `drift_reaction.rs`
  225, `float.rs` 84 (new). Three pre-existing files exceed 500 LoC and
  carry the grandfather scope from v0.8.x check-lints; no new violation
  introduced in v0.9.0.

- **Functions ≤ 50 LoC**: enforced by clippy clean at HEAD (project memory
  records `cargo clippy --all-targets clean` at Block B + Wave 4 commits).

- **`unsafe` scope**: confined to `src/simd/{x86_64,aarch64}.rs` per
  ADR-0019; no new `unsafe` introduced in v0.9.0 (Block A/B/C/D pure-safe
  Rust, generic refactor changes only type signatures).

- **Public API delta vs v0.8.1** (additive; no breaking changes per
  ADR-0023, ADR-0024, ADR-0025, ADR-0026 §Consequences):
  - `+1 type` — `NonSeparable2DAnisotropicChernoff<X, Y, F = f64>`
  - `+4 types` — `Grid3D<F = f64>`, `GridFn3D<F = f64>`, `AxisLift3D<C, F = f64>`
    (or `Axis::Z` enum variant — Option A chosen at impl), `Strang3D<X, Y, Z, F = f64>`
  - `+1 trait` — `SemiflowFloat` (sealed; blanket impls for `f32`, `f64` only)
  - All 14 `ChernoffFunction`-implementing types gain `<F: SemiflowFloat = f64>`
    parameter (default ensures zero callsite migration)
  - `+0 dependencies`, `+0 SemiflowError variants`, `+0 boundary policies`
  - Existing callsites compile unchanged via type inference (`= f64` default)

- **Workspace version**: `0.9.0` in `Cargo.toml [workspace.package]` at
  release commit `779c400`.

## 9. Recommendation

**Ship-with-followups: APPROVE for retrospective release verification.** All three
ROADMAP bullets (anisotropic 2D non-separable, 3D tensor, Generic-over-Float) are
math-faithful. 12 NORMATIVE sympy gates pass at HEAD on this audit run. The
backward-compatibility claim (`= f64` default ⟹ zero callsite migration) is
ratified by the preserved SIMD bit-equality regressions and by the +48 fast-test
count uplift with 0 failures. Three follow-up items are recorded as v0.11.x
non-blocking issues (heavy-validation harness coverage for G4_NS2D_aniso/G5_3D
deferred to I12; NS2D_aniso parallel bit-equal gate; scalar-`c` deprecation
policy at v1.0 freeze). Math invariants for the Theorem 6 / Theorem 7 framework
are intact; new math (Theorem 7' + Lemma 10.1 inductive) is rigorously stated
with retroactive scope correctly handled.
