# ADR-0091 — Diagonal Padé P₄/Q₄ for direct order-8 ζ⁸ approximation (complement to deferred nested Richardson Wave II)

- **Status**: Proposed
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Depends on**: ADR-0086 + AMENDMENT 1 (Path β K5 + Richardson + Option E hybrid gate methodology), ADR-0088 + AMENDMENT 1 + AMENDMENT 2 (ζ⁶ shipped + ζ⁸ Wave II DEFERRED via nested Richardson due to floor-cascade regime), ADR-0073 (`ApproximationSubspace<K, F>` opt-in marker), ADR-0074 (v3.0 `ChernoffFunction` cleanup + typed `Growth<F>`), ADR-0025/0026 (Generic-over-Float), ADR-0041 (`apply_into` + `ScratchPool`).
- **Mathematical foundation**: Higham 2005 *SIAM J. Matrix Anal. Appl.* 26(4), pp. 1179-1193 — "The Scaling and Squaring Method for the Matrix Exponential Revisited" (canonical scale-and-square algorithm with diagonal Padé approximant; backward error analysis at degree 4 gives order 2K=8). Hochbruck-Lubich 2010 *Acta Numerica* §3.4 (operator exponential via rational approximation; A-stability of Cayley-style maps generalises to diagonal Padé). The Padé table for $\exp(z)$ at $K=4$ is classical (Baker-Graves-Morris 1996 *Padé Approximants*, Cambridge §1.2).
- **Researcher synthesis**: `.dev-docs/research/verdicts/verdict-v4-3-research-waves.md` Q2 Option α RECOMMENDED (`HIGH leverage / LOW risk / no math creation`); raw findings `.dev-docs/research/raw-findings-romberg-2d.md` §"Query 7: diagonal Padé operator exponential matrix Higham" (P₄/Q₄ order-8 confirmed industry standard).
- **Acceptance gates added**: 3 new gates (Option E hybrid mirroring ADR-0086 AMENDMENT 1): `G_zeta8_pade_const_a_richardson` + `G_zeta8_pade_var_a_slope` + `T_ZETA8_PADE` (sympy). Wave I delivers single new kernel type `Diffusion8thZeta8PadeChernoff<F>` alongside (NOT replacing) the DEFERRED nested-Richardson `Diffusion8thZeta8Chernoff` research artifact.

## Context

ADR-0088 AMENDMENT 2 deferred `Diffusion8thZeta8Chernoff` (nested Richardson over Quintic-K5) to v4.3+ after the Wave II calibration measured Richardson ratio log₂(err_1/err_2) = 3.067 — below the 4.0 hard-stop threshold — at the canonical N=512, T=0.5, n-pair {1, 2} setup. Engineer + architect diagnosed a **floor-cascade regime**: nesting a 4th Richardson rung on top of a 3-rung tower at the same pre-asymptotic τ amplifies floor effects monotonically with K rather than cancelling temporal error (full diagnosis in `.dev-docs/research/zeta8-wave-ii-deferred.md`). Researcher Wave 2 (`.dev-docs/research/raw-findings-romberg-2d.md`) identifies **diagonal Padé P₄/Q₄** as the canonical published alternative to nested cascade: order 2K=8 directly in a single operator-polynomial step, no recursive nesting, no Richardson floor contamination across rungs. Higham 2005 SIAM is the industry standard (MATLAB `expm`, Julia, NumPy `scipy.linalg.expm` all use it). The verdict file (`verdict-v4-3-research-waves.md`) ranks this Option α as `HIGH leverage / LOW risk / no math creation`, parallel and independent of Path ε spatial work (ADR-0089-pending Quintic / future ADR-0090 candidate Chebyshev spectral).

## Decision

Ship **a single new kernel type** `Diffusion8thZeta8PadeChernoff<F>` (~200 LoC source target; HARD cap 500 LoC per default suckless guardrail) as a **complement** (NOT replacement) to the existing ζ-ladder. The new type achieves order-8 temporal convergence via diagonal Padé approximation `Q_4(τA)^{-1} P_4(τA) ≈ exp(τA) + O((τ‖A‖)^9)` applied to the K5 base operator A = divergence-form stencil `∂_x(a(x)∂_x)` reused verbatim from `apply_div_form` in `diffusion4_zeta4.rs` (pub(crate) helper already exists). The algorithm computes RHS `rhs = P_4(τA) · src` via 4 sequential A-applications + scalar accumulation, then solves `Q_4(τA) · dst = rhs` via explicit assembly of the 9-banded matrix Q_4(τA) (bandwidth 2K+1 = 9 for K=4) followed by LAPACK-style banded LU with partial pivoting (mirroring v4.1 `matrix_strang.rs` block-Thomas pattern at the scalar level). Existing ζ⁴ + ζ⁶ nested Richardson rungs ship UNCHANGED; existing DEFERRED ζ⁸ nested-Richardson research artifact (`.dev-docs/research/zeta8-wave-ii-deferred.md`) is preserved verbatim as historical record. Gates follow Option E hybrid: const-a + analytic Gaussian-heat oracle BLOCKING; var-a + K5-reference ADVISORY (mirrors ADR-0086 AMENDMENT 1 + ADR-0088 ζ⁶ Wave I calibration). NEW T23N-style sympy oracle T_ZETA8_PADE proves Padé identity P_4(z)/Q_4(z) = exp(z) + O(z^9) symbolically (4 sub-checks: Padé coefficient identity / Hermite-eigenfunction tangency / rate-constant bound / inverse-existence guard).

## Rationale (complement vs replace; banded LU vs Cramer vs Krylov)

- **Complement (NOT replace) ζ⁴/ζ⁶ existing rungs**: ζ⁴ + ζ⁶ Richardson rungs ship at calibrated thresholds (3.5 / 3.5 per ADR-0086 + ADR-0088 AMENDMENT 1 hybrid gate); they are peer-reviewable, mathematically validated (T23N + T23N_zeta6 sympy PASS 4/4 each), and add zero new tech-debt. Replacing them with Padé P₂/Q₂ (order 4) and P₃/Q₃ (order 6) would require ~600 LoC additional rewrites for zero observable accuracy gain and would invalidate v4.1 + v4.2 byte-equality tests. Suckless principle: third-occurrence rule. Padé is added ONLY for the rung where nested Richardson failed (ζ⁸); not as a uniform replacement.
- **Banded LU chosen over alternatives**: At N=512 with K=4, Q_4(τA) is a 512×512 9-banded matrix (bandwidth 2K+1 = 9). Three solver options were considered:
  - **Banded LU with partial pivoting** (CHOSEN): ~250 LoC including band assembly + LU factor + back-substitution; cost O(N · K²) factor + O(N · K) per RHS; reuses v4.1 `matrix_strang.rs` block-Thomas algorithmic pattern at scalar level (sub/main/sup → 9-banded forward sweep); deterministic; no_std-tractable (no LAPACK dependency); numerically stable per Golub-Van Loan §4.5.
  - **Cramer-style polynomial inverse on full 512×512** (REJECTED): O(N³) flops + 2 MB storage; ~50× more flops than banded LU at N=512; wastes the band structure that the polynomial-of-tridiagonal naturally produces.
  - **Krylov subspace iteration (Arnoldi)** (REJECTED for v4.3): published alternative (Hochbruck-Lubich 2010 §3.4); preserves matrix-free property; but requires ~400 LoC for orthogonalisation + reorthogonalisation + Krylov tolerance budget; iterative convergence harder to gate. Defer to v4.4+ if banded LU runtime profiles dominate.
- **Math creation: ZERO**. P₄/Q₄ coefficients are classical from the Padé table (Baker-Graves-Morris 1996 §1.2 verbatim). Higham 2005 algorithm is battle-tested industry standard. T_ZETA8_PADE sympy oracle verifies the identity `P_4(z) - exp(z) · Q_4(z) = O(z^9)` symbolically as a sanity check (sympy `series(expr, z, 0, 10)` returns the residual leading-term).
- **Skip scaling-and-squaring at v4.3**: Higham 2005's full algorithm includes scaling `τA → τA/2^s` to bound spectral radius, then squaring `R(τA)^{2^s}` to recover `exp(τA)`. For our v4.3 use case at N=512 with τ ≤ 0.5 and K5 stencil spectral radius ρ ≈ 3916, `τ · ρ ≈ 1958` — far above the Padé degree-4 convergence radius (rule of thumb: `τ · ρ < 5.4` for direct P₄/Q₄ accuracy at machine precision per Higham 2005 Table 2.1). v4.3 ships **without explicit scaling** because the caller controls outer-step size n and inner-step τ/n; the Chernoff iteration outer loop `(F(τ/n))^n f` for n ≥ 32 gives effective τ_inner = τ/32 ≈ 0.016 and τ_inner · ρ ≈ 62 — still requires implicit scaling. Add scaling-and-squaring infrastructure ONLY if `G_zeta8_pade_const_a_richardson` measurement fails the Wave I gate (calibrated 5.5 + empirical adjustment per ADR-0088 AMENDMENT 1 rule). Documented in spec AC8 as conditional Wave II.
- **Cost-vs-benefit (Wave I scope)**: ~200 LoC source + ~150 LoC banded LU helper module + ~200 LoC test + ~200 LoC sympy = ~750 LoC NEW total. 1-2 engineer waves (~5-8 working days). Compared to nested Richardson cascade which DEFERRED at 9× K5 cost per step and floor-cascade failure, Padé has per-step cost ~12-16 K5-equivalent applications (4 A-stencil applications for P_4 + 4 A-stencil applications during Q_4 assembly + one banded LU solve at O(N·K²) ≈ 8200 flops at K=4, N=512). Comparable per-step cost; bypasses floor-cascade entirely (no recursive Richardson nesting).
- **Grid compatibility**: Padé works at any spatial discretisation where A is well-defined (uniform Grid1D, future Chebyshev per ADR-0090-candidate, future SepticHermite per `.dev-docs/research/zeta8-wave-ii-deferred.md` direction 1). Wave I ships against the same `Diffusion4thChernoff` K5 base as ζ⁴/ζ⁶ — uniform Grid1D with 7-point Fornberg-stencil A operator (via `apply_div_form` 3-point divergence-form helper). Spatial floor is independent of Padé vs nested Richardson choice; both measure against the same Catmull-Rom/Quintic floor.

## Algorithm (NORMATIVE, per math §27.quart NEW section)

```text
Diagonal Padé P_4/Q_4 for direct order-8 ζ⁸ approximation (Higham 2005):

  Padé coefficients (Baker-Graves-Morris 1996 §1.2, classical):
    P_4(z) = 1 + z/2 + (3/28)·z² + (1/84)·z³ + (1/1680)·z⁴
    Q_4(z) = 1 − z/2 + (3/28)·z² − (1/84)·z³ + (1/1680)·z⁴

  Identity (Padé approximation theorem):
    Q_4(z)^{-1} · P_4(z) = exp(z) + O(z^9)
    ⇒ algorithm order 2K = 8 directly (no Richardson cascade).

  Per outer τ-step (NORMATIVE algorithm for apply_into(τ, src, dst, scratch)):
    1. Compute P_4(τA) · src via 4 sequential A-applications:
         A_src     = A · src              (= apply_div_form helper)
         A2_src    = A · A_src
         A3_src    = A · A2_src
         A4_src    = A · A3_src
         rhs[i]    = src[i] + (τ/2)·A_src[i]
                            + (3·τ²/28)·A2_src[i]
                            + (τ³/84)·A3_src[i]
                            + (τ⁴/1680)·A4_src[i]
    2. Assemble Q_4(τA) as a 9-banded matrix B of size N×N (bandwidth 2K+1 = 9):
         B = I − (τ/2)·A_mat + (3·τ²/28)·A_mat² − (τ³/84)·A_mat³ + (τ⁴/1680)·A_mat⁴
       where A_mat is the 3-banded stencil of A; B inherits band 2·4+1 = 9 from
       the polynomial of a 3-banded matrix.
    3. Solve B · dst = rhs via banded LU with partial pivoting (mirror block-Thomas
       at scalar level; sub/sup arrays of length N − k for k = 1..4; main of length N).
    4. Return dst.

  Stability: unconditionally stable for any τ > 0 when A is dissipative (real
  spectrum ≤ 0 — divergence-form heat with Neumann BCs satisfies this);
  Padé diagonal preserves A-stability per Hochbruck-Lubich 2010 §3.4.

  Per-step cost (counted in K5-base-equivalent units):
    - 4× A-stencil applications (P_4(τA) · src) ≈ 4× K5 cost   (K5 also = 1 A-application + interp)
    - 4× A^k stencil applications during B assembly             ≈ 4× K5 cost
    - One 9-banded LU factor + back-solve: O(N · K²) + O(N · K) ≈ 8200 flops at N=512
                                                                  ≈ ~0.5× K5 cost
    - Total: ~8-9× K5 cost per outer τ-step
    - vs DEFERRED nested-Richardson ζ⁸ Wave II: 27× K5 cost per step (R⁴ = 27 K5 calls)
    ⇒ Padé is ~3× CHEAPER per step at K=4.
```

## Implementation spec (engineer Wave I) — see `.dev-docs/specs/pade-zeta8-wave.md`

Concrete file-level deliverables, acceptance criteria (AC1–AC8), test plan, file touch list, Padé coefficient const-array, and properties.yaml YAML scaffold externalised to spec file. Wave I: single new source file `crates/semiflow-core/src/diffusion8_zeta8_pade.rs` + sibling banded-LU helper `crates/semiflow-core/src/banded_lu.rs` (~150 LoC; reusable for future v4.4+ implicit kernels) + test + sympy oracle + properties.yaml MINOR bump. Wave I is a single engineer wave; no Wave II planned unless calibration fails (then scaling-and-squaring deferred to follow-up ADR).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Replace ζ⁴/ζ⁶ existing Richardson rungs with Padé P₂/Q₂ + P₃/Q₃** | Existing rungs ship at calibrated thresholds with peer-reviewable Option E hybrid gates; replacement adds ~600 LoC for zero accuracy gain and invalidates v4.1/v4.2 byte-equality. Suckless: don't churn working code. |
| **Cramer-style full N×N polynomial inverse** | O(N³) flops + 2 MB storage; ~50× more flops than banded LU at N=512; wastes the band structure that polynomial-of-tridiagonal naturally produces. |
| **Krylov subspace iteration (Arnoldi)** | ~400 LoC for orthogonalisation + tolerance budget; iterative convergence harder to gate at order-8; defer to v4.4+ as performance optimisation only if banded LU runtime profiles dominate. |
| **Scaling-and-squaring infrastructure at v4.3** | Adds ~150 LoC squaring loop + spectral-norm estimation; v4.3 outer-loop Chernoff iteration `(F(τ/n))^n f` for n ≥ 32 implicitly scales τ_inner = τ/n; calibration measurement first, then add scaling ONLY if gate fails. |
| **Defer Padé entirely to v4.4+** | Researcher Q2 verdict ranks this `HIGH leverage / LOW risk / no math creation` — postponing accumulates tech-debt without research benefit. Plus the parallel Path ε spatial work (ADR-0089 + future ADR-0090) is independent; Padé does not block on it and vice versa. |
| **Replace DEFERRED nested-Richardson ζ⁸ research artifact** | The nested-Richardson Wave II remains valuable for future investigation under SepticHermite or Chebyshev spectral spatial bases (different floor regime); preserve verbatim per `.dev-docs/research/zeta8-wave-ii-deferred.md`. Padé is a parallel direct path, not a supersession. |

## Consequences

- **POSITIVE**: closes the ζ⁸ rung at v4.3+ via published canonical algorithm (Higham 2005 industry standard); ~3× cheaper per outer step than the DEFERRED nested-Richardson Wave II (8-9× K5 vs 27× K5); bypasses floor-cascade regime entirely (single-step Padé has no recursive Richardson nesting → no nested floor amplification); banded LU helper `banded_lu.rs` is reusable for future v4.4+ implicit kernels (matrix Crank-Nicolson, exponential integrators, etc.); adds 3 acceptance gates with same Option E hybrid methodology as ζ⁴/ζ⁶ (uniform gate structure across the rung family).
- **NEUTRAL**: per-τ-step cost ~8-9× K5 (4 A-applications for P_4 + 4 A-applications during B assembly + ~0.5× K5 banded LU); comparable to ζ⁶ (9× K5) and dramatically cheaper than DEFERRED ζ⁸ nested Richardson (27× K5). Working set adds N×9 = 4608 doubles (~37 KB at N=512) for banded matrix B storage plus N=512 doubles (~4 KB) for rhs scratch — well within `ScratchPool` capacity per ADR-0041.
- **NEGATIVE**: regularity contract `f ∈ D(A^8)` ≈ `f ∈ H^{16}(Ω)` for 1D divergence-form with `a ∈ C^8_b` is strict; documented in rustdoc as caller-asserted invariant via `a_kth_bound: Some(c)` and `ApproximationSubspace<8, F>` witness (mirror DEFERRED nested-Richardson ζ⁸ contract). Direct Padé without scaling-and-squaring may fail at large τ (τ·ρ > 5.4 per Higham 2005 Table 2.1); calibration measurement is the gate. If calibration fails, follow-up ADR adds scaling-and-squaring infrastructure (~150 LoC).
- **BREAKING**: NONE. One new type added (`Diffusion8thZeta8PadeChernoff<F>`); one new pub(crate) module (`banded_lu`); no existing API touched. Constitution principle #2 (additive surface, never subtractive) satisfied.
- **Schema bumps**: `properties.yaml` MINOR bump (e.g. `1.1.0 → 1.2.0`; 3 new gate entries added). `traits.yaml` unchanged. `math.md` amended (§27.quart appended after §27.tris; no edit to §27 / §27 AMENDMENT / §27 AMENDMENT 2 / §27.bis / §27.tris). `errors.yaml` unchanged (existing `DomainViolation` covers all new failure modes: NaN τ, negative τ, malformed `a_kth_bound`, banded LU singular pivot).
- **Constitution unchanged**: this ADR adds 2 source files (`diffusion8_zeta8_pade.rs` ~200 LoC + `banded_lu.rs` ~150 LoC) — both within default 500-LoC cap per file; no Cohort expansion needed. Override count remains 3/3.
- **Bench-track HFT example**: not in scope for this ADR; if Wave I gates pass, an `examples/zeta8_pade_smoke.rs` 3-rung comparison (ζ⁴ vs ζ⁶ vs ζ⁸-Padé vs analytic) could ship as opt-in side-track (mirrors `examples/heston_pricer.rs` + `examples/sabr_pricer.rs` pattern).

## Migration

End-user impact is ADDITIVE (no API surface change to existing types):

```rust
// v4.2 (current): order 4 + order 6 only via nested Richardson
use semiflow_core::{Diffusion4thChernoff, Diffusion4thZeta4Chernoff,
                   Diffusion6thZeta6Chernoff, Grid1D};
let grid = Grid1D::new(-10.0, 10.0, 512)?;
let k5    = Diffusion4thChernoff::new(a_fn, a_prime, a_double_prime, 2.5, grid);
let zeta4 = Diffusion4thZeta4Chernoff::new(k5, Some(2.5_f64))?;          // order 4 (ADR-0086)
let zeta6 = Diffusion6thZeta6Chernoff::new(zeta4, Some(2.5_f64))?;       // order 6 (ADR-0088 Wave I)

// v4.3+ (ADR-0091 Wave I): direct order-8 via Padé P_4/Q_4
use semiflow_core::Diffusion8thZeta8PadeChernoff;
let zeta8_pade = Diffusion8thZeta8PadeChernoff::new(k5, Some(2.5_f64))?; // order 8 (Padé direct on K5)
assert_eq!(zeta8_pade.order(), 8);
// Note: constructor takes K5 directly (NOT zeta6) — Padé bypasses the
// nested-Richardson cascade entirely.
```

Brief migration doc snippet may be added to `docs/migration/v4.2-to-v4.3.md` (~30 LoC) noting the additive new type + recommended usage table (ζ⁴ for `f ∈ D(A^4)`, ζ⁶ for `f ∈ D(A^6)`, ζ⁸-Padé for `f ∈ D(A^8)`).

## Cross-references

- ADR-0001 — contract-first; this ADR amends the v4.x contract via the math §27.quart NEW section.
- ADR-0013 — v0.6.0 `Diffusion4thChernoff` (K5 base; the leaf node Padé operates on directly).
- ADR-0035 §9 — deprecation cycles (no deprecations in this ADR; Padé is additive).
- ADR-0041 — `apply_into` + `ScratchPool` (new kernel uses scratch for rhs + banded matrix B working set).
- ADR-0073 — `ApproximationSubspace<K, F>` (new kernel impls `ApproximationSubspace<8, F>`).
- ADR-0074 — v3.0 typed `Growth<F>` (preserved; new kernel inherits K5's growth bound at multiplier 1.0).
- ADR-0086 + AMENDMENT 1 — Path β + Option E hybrid gate methodology (template for `G_zeta8_pade_*` gates).
- ADR-0088 + AMENDMENT 1 + AMENDMENT 2 — ζ⁶ shipped + ζ⁸ nested Richardson DEFERRED (the cascade alternative Padé complements).
- ADR-0089 — Path ε QuinticHermite spatial sample upgrade (independent; Padé works at any spatial discretisation).
- ADR-0090 — TBD (Chebyshev spectral collocation per Q1 Option B in researcher verdict; independent of Padé).
- math.md §27.quart — NEW NORMATIVE section appended after §27.tris (Padé P_4/Q_4 algorithm spec).
- `.dev-docs/research/verdicts/verdict-v4-3-research-waves.md` §Q2 Option α — researcher verdict ranking this `HIGH leverage / LOW risk / no math creation`.
- `.dev-docs/research/raw-findings-romberg-2d.md` §"Query 7: diagonal Padé operator exponential matrix Higham" — primary-source synthesis.
- `.dev-docs/research/zeta8-wave-ii-deferred.md` — research artifact preserving the DEFERRED nested-Richardson Wave II (Padé is the parallel direct path; the nested approach stays as historical record for future SepticHermite or Chebyshev-spectral spatial work).
- `.dev-docs/specs/pade-zeta8-wave.md` — engineer Wave I spec (AC1–AC8, file touch list, test plan, Padé coefficient table, properties.yaml YAML scaffold).
- N. J. Higham (2005) — *The Scaling and Squaring Method for the Matrix Exponential Revisited*, **SIAM J. Matrix Anal. Appl.** 26(4), pp. 1179-1193 — canonical scale-and-square algorithm (we use the Padé component without scaling at v4.3 baseline).
- M. Hochbruck, C. Lubich (2010) — *Exponential integrators*, **Acta Numerica** 19, pp. 209-286, §3.4 (A-stability of Cayley-style rational approximations; generalises to diagonal Padé).
- G. A. Baker, P. Graves-Morris (1996) — *Padé Approximants*, **Cambridge** §1.2 (classical Padé table for `exp(z)`; P_4/Q_4 coefficients verbatim).
- G. H. Golub, C. F. Van Loan (1996) — *Matrix Computations* 3rd ed., **Johns Hopkins** §4.5 (banded LU with partial pivoting; algorithm template for `banded_lu.rs` sibling module).

---

### AMENDMENT 1 (2026-05-29) — Wave B Padé DEFERRED v4.4+ pending scaling-and-squaring architecture; Wave A Chebyshev `Diffusion8thZeta8Chernoff` (ADR-0090) ships ζ⁸ at v4.3 alone

**Trigger**: Following Wave B engineer landing (`diffusion8_zeta8_pade.rs` 434 LoC + `banded_lu.rs` 615 LoC + test + sympy), bug-fixer corrected two numerical bugs (sign in `build_a_tridiag`; LU storage 2K+1 → 3K+1 LAPACK convention). T_ZETA8_PADE sympy oracle **PASSED 4/4 sub-checks** (algebraic identity is correct; math is sound). The post-fix calibration measurement of `G_zeta8_pade_const_a_richardson_ratio` returned `log₂(err_4/err_8) = 7.87` (above the 7.0 placeholder threshold) — BUT the absolute errors were `err_4 ≈ err_8 ≈ 4.97e31` (numerical garbage). `G_zeta8_pade_var_a_temporal_slope` returned slope `+5.58` (gate ≤ −6.5; sign-inverted from theoretical −8). The const-a "PASS" arises because both `err_4` and `err_8` are dominated by the same `1/‖A‖²⁸ · τ⁸` polynomial blow-up term, so the ratio is correct algebraically while absolute values are meaningless.

**Diagnosis (bug-fixer verbatim, architecturally correct)**: "Padé P_4/Q_4 is only accurate when `τ‖A‖ = O(1)`. For N=512 on `[−10, 10]`, the operator norm `‖A‖₂ ≈ 260 000`, and `τ = T/n = 0.5/4 = 0.125`, giving `τ‖A‖ ≈ 32 500`. At this scale, `P_4(z)/Q_4(z) → 1` instead of `exp(z)` for large negative z. ADR-0091's Padé design assumption that the operator norm is bounded independently of N is incorrect for diffusion on a grid." The ADR §"Scaling-and-squaring deferred at v4.3 baseline" paragraph anticipated this conditionally: at `n ≥ 32`, `τ_inner · ρ ≈ 62` — STILL above the ~5.4 convergence radius (Higham 2005 Table 2.1); the calibration gate is the canonical falsifier and it fired. Higham 2005's full algorithm — scaling `τ → τ/2^s` then squaring `R(τA)^{2^s}` — is the canonical industry remedy; the v4.3 baseline elision per the original "Skip scaling-and-squaring at v4.3" rationale was incorrect.

**Decision (Option γ — DEFER Wave B Padé to v4.4+ ADR-0091 follow-up)**: revert all Wave B Padé uncommitted artifacts; preserve as research artifact under `.dev-docs/research/zeta8-pade-wave-b-deferred.md` (mirror `zeta8-wave-ii-deferred.md` exemplar). v4.3 ships ζ⁸ via Wave A Chebyshev `Diffusion8thZeta8Chernoff` (ADR-0090, nested-Richardson on Quintic-K5 with default-ON Chebyshev floor lift) ALONE — the suckless single-kernel principle requires we not parallel-ship two ζ⁸ kernels when one already satisfies the contract. The order-8 deliverable is closed for v4.3. Padé becomes a deferred alternative kernel whose v4.4+ scope is "Higham 2005 full scaling-and-squaring with backward error estimator", not the truncated v4.3 baseline that omitted the scaling step.

**Why Option γ (DEFER) over Option α (add scaling-and-squaring at v4.3) or Option β (domain-restrict + gate redesign)**:
1. **Single-kernel suckless principle**: Wave A Chebyshev `Diffusion8thZeta8Chernoff` already ships order-8 ζ⁸. Adding a second order-8 kernel via Padé costs +~750 LoC source + test + sympy + 2 acceptance gates + 2 properties.yaml entries for *zero* observable accuracy gain over Wave A. Option α (scaling-and-squaring) adds ~150 LoC more (squaring loop + spectral-radius estimator + new const-array) for the same parallel-kernel outcome. Cost-vs-benefit fails the third-occurrence test.
2. **Mirror precedent (ADR-0088 AMENDMENT 2 + Item 3 step-k Carnot)**: STILL_OPEN closure with research-artifact preservation is the established project pattern for ζ-ladder rungs that hit architectural ceilings post-implementation. Wave B Padé takes the same shape: the algorithm is mathematically sound (T_ZETA8_PADE 4/4 PASS), the implementation is correct (bug-fixer-validated), the *application regime* (`τ‖A‖` for grid-discretised diffusion) is the binding constraint, and the architectural lift (Higham 2005 full algorithm) is a future-ADR-pending advance.
3. **Option α numerical risk surface**: scaling-and-squaring adds spectral-radius estimation (typically 1-norm estimator per Higham 2002 *Accuracy and Stability of Numerical Algorithms* §15.3 — itself ~80 LoC + ~50 LoC iteration), backward-error-based scaling parameter selection (Higham 2005 Algorithm 2.3), and a squaring loop with error-accumulation guards. The combined diff is ~250 LoC engineering plus a re-calibration cycle for both const-a and var-a gates. At v4.3 close-out this exceeds the marginal-benefit envelope versus simply shipping Chebyshev nested-Richardson alone.
4. **Option β UX regression**: forcing the caller to pre-split τ into `τ ≤ τ_safe(N)` is a usability cliff inconsistent with the Chernoff `apply_into` contract used by every other kernel (any positive τ). The gate redesign (`n_ref ≥ 65000` reference) also bloats test runtime ~10× without removing the underlying degeneration.

**Impact on Wave B uncommitted artifacts (revert list — engineer/bug-fixer action)**:
- DELETE `crates/semiflow-core/src/diffusion8_zeta8_pade.rs` (434 LoC)
- DELETE `crates/semiflow-core/src/banded_lu.rs` (615 LoC; reusability for future v4.4+ Padé scaling-and-squaring is preserved via inline-paste in the research artifact)
- DELETE `crates/semiflow-core/tests/zeta8_pade_correction_slope.rs` (341 LoC)
- DELETE `scripts/verify_zeta8_pade.py` (349 LoC; preserved via inline-paste in research artifact)
- REVERT in `crates/semiflow-core/src/lib.rs`: remove `pub mod diffusion8_zeta8_pade;` and `diffusion8_zeta8_pade::Diffusion8thZeta8PadeChernoff` re-export
- REVERT in `crates/semiflow-core/src/lib.rs`: remove `pub mod banded_lu;` declaration (was added with Wave B)
- REVERT in `contracts/semiflow-core.properties.yaml`: remove `G_zeta8_pade_const_a_richardson_ratio` (lines 6126-6145) + `G_zeta8_pade_var_a_temporal_slope` (lines 6146-6165) + `T_ZETA8_PADE` (if added). Do NOT touch the ADR-0090 Wave A gates (`G_PATH_EPS_CHEB_FLOOR` / `G_zeta8_const_a_richardson_cheb` / `G_zeta8_var_a_slope_cheb` / `T_CHEB`); these stay BLOCKING for v4.3.
- CREATE `.dev-docs/research/zeta8-pade-wave-b-deferred.md` (~250 LoC): inline-paste reverted source + helper + test + sympy as fenced code blocks; include verbatim measurement record (const-a log₂(ratio)=7.87 with err ≈ 4.97e31; var-a slope +5.58); include bug-fixer's `τ‖A‖ = 32 500` diagnosis paragraph; include this AMENDMENT 1 verbatim; include the 3 v4.4+ ADR-0091-followup candidate directions (full Higham 2005 scaling-and-squaring; Krylov subspace Arnoldi per Hochbruck-Lubich 2010 §3.4; rational Padé table at lower degree K=2/3 only at small τ regimes); mark `RESEARCH-ARTIFACT-ONLY, NOT-BUILT, NOT-TESTED, PENDING-v4.4+-ARCHITECTURE`.

**v4.3 release scope (ζ⁸ rung status)**: closed via Wave A Chebyshev `Diffusion8thZeta8Chernoff` (ADR-0090). Padé out of v4.3 scope. CHANGELOG entry: "ζ⁸ closure shipped via ADR-0090 Wave A (Chebyshev spectral floor + nested-Richardson). ADR-0091 Wave B (direct Padé P_4/Q_4) deferred to v4.4+ pending Higham 2005 full scaling-and-squaring architecture (operator norm `τ‖A‖ ≈ 32 500` at N=512 exceeds Padé convergence radius ~5.4; bug-fixer-validated diagnosis preserved in `.dev-docs/research/zeta8-pade-wave-b-deferred.md`)."

**v4.4+ ADR-0091 follow-up directions (3 mutually-compatible candidates, ordered by architectural complexity)**:
1. **Full Higham 2005 scaling-and-squaring** (~250 LoC: 1-norm estimator + Algorithm 2.3 scaling-parameter selection + squaring loop + accumulator guards) — canonical industry-standard remedy; reuses `banded_lu.rs` verbatim.
2. **Krylov subspace Arnoldi** (Hochbruck-Lubich 2010 §3.4) — ~400 LoC: orthogonalisation + reorthogonalisation + Krylov tolerance budget — matrix-free; iterative convergence harder to gate.
3. **Lower-degree Padé regimes only** (P_2/Q_2 order 4 + P_3/Q_3 order 6 with smaller `τ‖A‖` envelope) — would duplicate ζ⁴/ζ⁶ surface for negligible accuracy gain; lowest priority.

ADR-0091 status: **"Accepted (Amendment 1: Wave B Padé DEFERRED to v4.4+; ζ⁸ rung closed for v4.3 via ADR-0090 Wave A Chebyshev alone)"**.

**Cross-references for AMENDMENT 1**: ADR-0090 §"Decision" + §"ζ⁸ Wave II RESURRECTION" (the parallel Chebyshev path that ships ζ⁸ at v4.3); ADR-0088 AMENDMENT 2 (the precedent STILL_OPEN closure with research artifact for the prior nested-Richardson Wave II — exact pattern Wave B Padé follows); `~/.claude/projects/.../memory/project_step_k_carnot_open.md` (the Item 3 STILL_OPEN closure pattern for architectural-ceiling deferrals); `.dev-docs/research/zeta8-pade-wave-b-deferred.md` (NEW research artifact, engineer creates from reverted source + bug-fixer measurement); math.md §27.quart AMENDMENT 1 (the math-side annotation marking `Diffusion8thZeta8PadeChernoff` algorithm as v4.4+-deferred at the operator level while preserving its scalar Padé identity as CITATION mathematics). N. J. Higham (2002) — *Accuracy and Stability of Numerical Algorithms* 2nd ed., **SIAM** §15.3 (1-norm estimator algorithm for spectral-radius-based scaling parameter selection in v4.4+ Higham-2005 full architecture).
