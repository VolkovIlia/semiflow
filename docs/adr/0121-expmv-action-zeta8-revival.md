# ADR-0121 — `expmv` action kernel REVIVES the ζ⁸ Padé path per Al-Mohy & Higham (2011); supersedes ADR-0101 TERMINAL CLOSURE with the published Wave-I-bug-bypass it required

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — `DiffusionExpmvChernoff` additive kernel)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Supersedes / refines**: ADR-0101 (Path δ TERMINAL CLOSURE, v5.0). ADR-0101 permanently deferred the *operator-level Padé* kernel `Diffusion8thZeta8PadeChernoff` and made any revival conditional on **(1)** explicit citation of ADR-0101, **(2)** PRE-FLIGHT sympy clearance, and **(3)** citation to a *published* algorithm demonstrating Wave-I-bug-bypass at the algorithmic-equivalence level. This ADR satisfies all three: it does NOT revive the Padé/Q⁻¹ kernel — it adopts a **structurally different published algorithm** (`expmv`, Al-Mohy & Higham 2011) that sidesteps every ADR-0101 failure mode by construction.
- **Depends on**: ADR-0090 (Chebyshev `Diffusion8thZeta8Chernoff` — the production ζ⁸ kernel; `expmv` is ADDITIVE, not a replacement), ADR-0118/§40.7 (`apply_div_form` divergence-form stencil — the `A·v` primitive `expmv` reuses verbatim), constitution v2.0.0 (3/3 dep cap INVIOLATE — `expmv` adds ZERO deps).
- **Mathematical foundation**: PRE-FLIGHT sympy/numpy harness `scripts/verify_expmv_preflight.py` (executed 2026-06-05). All three sub-checks PASS — see §PRE-FLIGHT below.
- **Researcher synthesis**: `.dev-docs/research/verdict-v7-pade-zeta8.md` (ANALYSIS mode, 2026-06-05) — identified `expmv` as the published structural bypass and prescribed the mandatory PRE-FLIGHT this ADR records.
- **Acceptance gates added**: `g_expmv_div_form_action_accuracy` (slow-tests-gated backward-error gate; threshold `sup_error ≤ 1e-11`, see §Gate test). NO order-slope gate (`expmv` is tolerance-driven, not fixed-order — the "slope ≤ −8" pattern is INAPPLICABLE).

## Context

The ζ⁸ operator-level Padé kernel was deferred three times (ADR-0091 Wave B v4.3, ADR-0094 Wave I v4.5, ADR-0101 Path δ v5.0 TERMINAL CLOSURE). The three documented root causes are all *rational-Padé-on-matrix* defects:

- **D1** — P₄/Q₄ coefficient transcription bug.
- **D2** — banded-LU error propagation under repeated squaring (the Q₄⁻¹ solve).
- **D3** — scalar-vs-matrix squaring mismatch (FALSIFIED by ADR-0101 PRE-FLIGHT: `R^{2^s}·v ≡ R·R···R·v` byte-identical for fixed `R`).
- **Envelope blow-up** — `τ‖A‖ ≈ 62` at N=512 ≫ Padé degree-4 radius `θ_4 = 5.4`.

ADR-0101 explicitly named the unblocking condition: *"a published algorithm that bypasses both the matrix-squaring D3-equivalence and the m=4 vs m=13 envelope identity."* The v7.0 research wave located exactly that:

> A. H. Al-Mohy and N. J. Higham, *Computing the Action of the Matrix Exponential, with an Application to Exponential Integrators*, SIAM J. Sci. Comput. 33(2):488–511, 2011, DOI 10.1137/100788860 (`expmv`).

`expmv` computes the **action** `e^{τA}b` by applying a truncated Taylor polynomial `T_m(τA/s)` to the **vector**, `s` times. It is a *different algorithm class*: **no Padé denominator** (D1/D2 cannot recur — there is no Q₄ to transcribe or LU-solve), **no matrix squaring** (D3 is moot — there is no `R` matrix to square), and the scaling `s = ⌈τ‖A‖/θ_m⌉` bounds each per-application argument below the convergence radius (the `τ‖A‖≈62` blow-up is exactly what `s`-scaling tames). The implementation reuses the crate's existing banded `A·v` primitive (`apply_div_form`) plus a compile-time `θ_m` constant table — zero new deps.

## PRE-FLIGHT sympy/numpy (executed 2026-06-05; `scripts/verify_expmv_preflight.py`)

Banded 3-point divergence-form generator `A` (mimics `apply_div_form`, Neumann BCs, variable `a(x)=1+0.3 sin(2πx/L)`), N=64, L=20, τ chosen so `τ‖A‖₁ = 62.00` — the exact blow-up regime of ADR-0101. Reference: dense `scipy.linalg.expm(τA)·b` and the independent `scipy.sparse.linalg.expm_multiply` (peer-reviewed Al-Mohy–Higham implementation).

### (a) Scaled truncated-Taylor action reproduces `e^{τA}b` in the `τ‖A‖≈62` regime — **PASS**

For each `(s,m)` with `s = ⌈τ‖A‖/θ_m⌉`, the per-step argument `(τ/s)‖A‖ ≤ θ_m` BY CONSTRUCTION, and the action error reaches round-off:

| m  | s   | mat-vecs | sup_error | per-step arg ≤ θ_m |
|----|-----|----------|-----------|--------------------|
| 5  | 431 | 2155     | 3.22e-15  | 0.144 ≤ 0.144 ✔   |
| 8  | 44  | 352      | 1.28e-15  | 1.409 ≤ 1.440 ✔   |
| 10 | 23  | 230      | 1.33e-15  | 2.696 ≤ 2.740 ✔   |
| 13 | 14  | 182      | 1.33e-15  | 4.429 ≤ 4.740 ✔   |
| 18 | 8   | 144      | 1.11e-15  | 7.750 ≤ 8.840 ✔   |

`best sup_error = 1.11e-15`; per-step argument bounded for every `(s,m)`. **PASS** (`args_bounded ∧ best_err < 1e-12`). (m=25 at arg/step=12.4 over-extends a monomial Horner past float stability → 7.2e-10; this is the empirical reason Algorithm 3.2 caps the degree, and is the engineer-spec degree-cap directive below — not a defect of the algorithm.)

### (b) Structural bypass of Wave-I — **PASS**

Symbolic (sympy): the OLD path `R(z) = P₄(z)/Q₄(z)` has a **non-trivial denominator** and the step `v ← R^{2^s}·v` performs matrix squaring; the NEW step `v ← T_m(τA/s)^s·v` is **polynomial — no R, no denominator, no squaring**. Empirical convergence-out-of-unconverged-regime probe (fixed `s`, raising truncation degree `m`):

| m | sup_error | log₂ ratio |
|---|-----------|-----------|
| 2 | 8.34e-05  | —         |
| 4 | 4.77e-08  | +10.77    |
| 6 | 2.94e-11  | +10.67    |
| 8 | 2.03e-14  | +10.50    |
| 10| 1.17e-15  | +4.12     |

Super-algebraic Taylor convergence from 8.3e-5 down to round-off — the **opposite** of Wave-I's persistent `log₂ = −0.4053` anti-convergence (which sat at errors ≫ round-off). Tail ratio at m=12 (−0.067) is floor-noise (error already 1.2e-15). **PASS** (`monotone_convergent ∧ reaches_roundoff ∧ structural`).

### (c) Reaches accuracy ≤ existing Chebyshev ζ⁸ floor — **PASS**

Chebyshev ζ⁸ effective floor (target bar, `diffusion8_zeta8.rs`): `4.17e-12`.

- Al-Mohy–Higham Algorithm 3.2 auto-selected `(s=8, m=18)` (per-step arg 7.75 ≤ θ_18=8.84): **sup_error = 1.11e-15**.
- SciPy `expm_multiply` (independent reference `expmv`): **sup_error = 1.44e-15**.

Both ≈ 3700× **below** the Chebyshev floor. **PASS** (both ≤ `4.17e-12`).

### PRE-FLIGHT verdict: (a)=PASS (b)=PASS (c)=PASS → **GO**

## Decision

**GO — implement `DiffusionExpmvChernoff`**, an ADDITIVE tolerance-driven `e^{τA}·v` evolver realising Al-Mohy & Higham (2011) `expmv` for the divergence-form generator `A = ∂_x(a ∂_x) (+ b∂_x + c)`, reusing `apply_div_form` and a baked `θ_m` const table. It is NOT a fourth attempt at the Padé kernel — it is a different kernel class whose bug-bypass is structural (the published condition ADR-0101 required). The Padé `Diffusion8thZeta8PadeChernoff` kernel REMAINS terminally deferred (ADR-0101 unchanged); this ADR revives the *ζ⁸-accuracy goal* by a different published route, NOT the Padé kernel itself.

## Rationale (≤300 words)

**`expmv` over revived Padé**: PRE-FLIGHT is dispositive. (a) The scaled action reaches 1.1e-15 in the `τ‖A‖=62` regime that gave the Padé kernel a `4.97e31` blow-up — `s`-scaling bounds each per-step argument below θ_m by construction. (b) It is structurally R-free / denominator-free / square-free (sympy-confirmed), so D1 (coefficient transcription) and D2 (banded-LU propagation) **cannot exist** and the Wave-I `−0.4053` anti-convergence is replaced by super-algebraic `+10.5` convergence. (c) It reaches ~3700× below the Chebyshev ζ⁸ floor, cross-validated against SciPy's independent peer-reviewed `expm_multiply`.

**ADR-0101 revival gate satisfied**: (1) this ADR cites ADR-0101 explicitly; (2) PRE-FLIGHT sympy clearance recorded above; (3) `expmv` is the *published* Wave-I-bug-bypass algorithm — its bypass is at the algorithmic-equivalence level (polynomial action vs rational squaring), not envelope-level. It is not an α/β/γ variant (all three were rational-Padé-on-matrix); it is the Taylor-action-on-vector paradigm ADR-0101 did not enumerate.

**Suckless / additive**: `expmv` reuses the existing `A·v` primitive, adds a ~20-float const θ_m table, needs O(1) work vectors (`y`, `w`), no recursion, no new dep — `no_std + alloc` clean, 3/3 dep cap preserved. It coexists with the Chebyshev ζ⁸ kernel (additive sibling per the third-occurrence test): `expmv` offers tolerance-driven arbitrary-τ accuracy + an independent cross-validation of Chebyshev ζ⁸ (the cross-validation ADR-0101 §Consequences listed as permanently lost — now recovered).

## Consequences

- **POSITIVE**: closes the rolling ζ⁸-Padé defer debt with a *working* published kernel; recovers the independent matrix-exp cross-validation of Chebyshev ζ⁸ (ADR-0101 listed this as the primary lost value); gives end-users a tolerance-driven, A-stable-by-scaling arbitrary-τ ζ⁸ option (Chebyshev requires bounded τ within nested-Richardson radius). Reuses `apply_div_form` (no new stencil).
- **NEUTRAL**: ADDITIVE surface only — no existing kernel changes; ADR-0101's Padé terminal closure is UNCHANGED (this ADR revives the goal, not the kernel). Constitution: 3/3 dep cap preserved; `expmv.rs` ≤ 500 LoC default file cap (no Cohort carve-out anticipated — confirm at engineer wave).
- **NEGATIVE**: `expmv` is tolerance-driven, so its gate is a backward-error bound (`sup_error ≤ tol`), NOT a fixed-order slope — it does not produce a `−8` order signature, which makes it not directly comparable to the ζ⁸ TRUTHFUL_ORDER pair-slope gates (ADR-0119). Documented in the gate spec below.
- **BREAKING**: NONE. Additive new public type behind a `.with_expmv()` builder OR a free `DiffusionExpmvChernoff` evolver (engineer chooses the minimal additive API).
- **Schema bumps**: `traits.yaml` / `properties.yaml` gain the new kernel + gate at engineer wave (MINOR). `math.md` §45 NEW (NORMATIVE algorithm). No errors.yaml change (reuses `DomainViolation`).

## Engineer spec (GO deliverables)

1. **New module** `crates/semiflow-core/src/expmv.rs` (≤ 500 LoC), `no_std + alloc`, zero new dep:
   - `pub struct DiffusionExpmvChernoff<F: SemiflowFloat = f64>` wrapping a `Diffusion4thChernoff<f64>` (the carrier of `apply_div_form` + grid + `a(x)`), plus `tol: f64` (default `2^-53`).
   - `θ_m` table as `const THETA_M: [(u32, f64); _]` (Al-Mohy–Higham 2011 Table 3.1, double-precision subset — the values in the PRE-FLIGHT harness `THETA_M_DOUBLE`). **Degree cap `M_MAX = 18`** (PRE-FLIGHT (a): monomial Horner above arg≈9 loses precision; cap the degree and push remaining argument into `s` — Al-Mohy–Higham Code Fragment 3.1 guard).
   - `‖A‖` estimate: reuse the crate's existing conservative analytic bound (`‖A‖_est ≈ 4·a_inf/dx²`, already computed for the prior kernel) — over-estimation only raises `s` (more, cheaper steps), never harms correctness. NO Higham–Tisseur block estimator needed (avoids ~80 LoC; documented trade-off).
   - `(s, m)` selection: `select_s_m` minimising `s·m` s.t. `(τ/s)·‖A‖_est ≤ θ_m`, `m ≤ M_MAX` (the corrected PRE-FLIGHT selector).
   - Action: `expmv_action` Horner-on-vector — `for i in 1..=s { w←y; for k in 1..=m { w ← (τ/s)·(A·w)/k; y ← y + w } }`. One `apply_div_form` call per inner term; SIMD/parallel banded paths (ADR-0018) reusable.
   - Implement `ChernoffFunction<f64>` (`type S = GridFn1D<f64>`); `order()` returns a documented sentinel (e.g. `u32::MAX` or a `// tolerance-driven, not fixed-order` rustdoc — engineer chooses, but it MUST NOT claim a false fixed order). `apply_into` = the action above. Validate τ via existing `validate_tau`.
2. **Public API (minimal additive)**: prefer a free `DiffusionExpmvChernoff::new(inner, tol)` + `with_tolerance(tol)` builder. Do NOT reshape any frozen trait (ADR-0073). No `.with_expmv()` on existing kernels unless trivially additive.
3. **Gate test** `crates/semiflow-core/tests/expmv_div_form_action_accuracy.rs` (`slow-tests`-gated): `g_expmv_div_form_action_accuracy` — build the canonical divergence operator, push `τ‖A‖` into the ≥30 regime, compare against a **high-`s` self-converged reference** (mirror the `G_zeta8` self-convergence pattern: reference = `expmv_action` at a degree/scaling pair giving sub-round-off accuracy). Assert `sup_error ≤ 1e-11` (one order above the Chebyshev `4.17e-12` floor for discretisation headroom; tighten to `4.17e-12` only if the in-crate operator reproduces the PRE-FLIGHT 1e-15). NO slope assertion.
4. **PRE-FLIGHT harness** `scripts/verify_expmv_preflight.py` is the permanent reproducibility artifact (already written + passing) — cite it in `properties.yaml` next to the new gate.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Keep ADR-0101 Path δ (no revival)** | ADR-0101 explicitly conditioned revival on a published Wave-I-bug-bypass; `expmv` is exactly that, PRE-FLIGHT GO. Keeping the closure would forgo a working kernel + the recovered Chebyshev cross-validation for no reason. |
| **Revive the Padé/Q⁻¹ kernel (Path α/β/γ)** | All three are rational-Padé-on-matrix (D1/D2 inherent, D3 byte-identical-equivalent per ADR-0101 PRE-FLIGHT). `expmv` avoids the entire failure class structurally. |
| **arXiv:2512.20777 (Sastre et al. 2025 dense Taylor)** | Dense matrix-matrix `O(N³)`, forms the matrix — opposite of a banded `no_std` action kernel; not action-on-vector; does not tame the norm blow-up via vector scaling. Rejected by the research verdict §"arXiv:2512.20777". |
| **Higham–Tisseur 1-norm estimator for `‖A‖`** | ~80 LoC for a *performance* nicety (tighter `s`); the conservative analytic bound already in-crate is correctness-sufficient (over-estimate → more cheap steps). Deferred; over-estimation documented. |

## Cross-references

- ADR-0101 — Padé operator-level kernel TERMINAL CLOSURE (this ADR revives the *goal* via a different published kernel; ADR-0101's Padé deferral is UNCHANGED).
- ADR-0090 — Chebyshev `Diffusion8thZeta8Chernoff` (production ζ⁸; `expmv` is additive + cross-validating).
- ADR-0118 / §40.7 — `apply_div_form` divergence-form stencil (`A·v` primitive reused).
- ADR-0018 — SIMD/parallel banded mat-vec (reusable for the `A·w` inner loop).
- math.md §45 — `expmv` action NORMATIVE algorithm (this ADR's math note).
- `scripts/verify_expmv_preflight.py` — PRE-FLIGHT harness (executed 2026-06-05, GO).
- `.dev-docs/research/verdict-v7-pade-zeta8.md` — v7.0 research verdict prescribing `expmv` + the mandatory PRE-FLIGHT.
- A. H. Al-Mohy, N. J. Higham (2011), SIAM J. Sci. Comput. 33(2):488–511, DOI 10.1137/100788860 — primary citation.
- A. H. Al-Mohy, N. J. Higham (2009/2010), SIAM J. Matrix Anal. Appl. 31(3):970–989, DOI 10.1137/09074721X — `α_p` / θ_m backward-error machinery.

ADR-0121 status: **Accepted (PRE-FLIGHT GO; `DiffusionExpmvChernoff` additive engineer wave authorised; Padé kernel ADR-0101 deferral UNCHANGED)**.
