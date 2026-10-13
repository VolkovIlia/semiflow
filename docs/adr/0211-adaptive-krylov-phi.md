# ADR-0211 — Adaptive Krylov (Arnoldi) φ-combination and exponential action

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: none (additive; the number was reserved in ADR-0205).
- **Contract**: math §68 (`contracts/semiflow-core.math-phi-krylov.md`); gates
  `G_PHI_KRYLOV_DENSE`, `G_PHI_KRYLOV_COST`, `G_PHI_KRYLOV_TOL`,
  `G_ETDRK4_KRYLOV_ORDER`.

## Context

The φ-actions (`phi_action`, `phi_combination`, ETDRK4, ADR-0202/0205/0210) and the
`GeneralOperator` Taylor `expmv` (ADR-0195/0205) cost `Θ(τ‖G‖)` generator
applications. For a stiff non-symmetric generator — a drifted Fokker–Planck operator,
an inventory ladder — at `τ‖G‖ = 4·10³` that is 33 960 applications per
`phi_combination` (after the ADR-0210 shift, which does not help a drift-dominated
hull), and the cost grows linearly with the horizon. The symmetric Krylov paths
(Chebyshev, Lanczos, rational) do not apply: the spectrum is complex.

## Decision

1. **Expokit `expv` on the augmented operator.** `Σ_{k≤p} τᵏφ_k(τG)w_k` is the first
   block of `e^{B̃_η}·[w₀; 0; …; 1/η]` for `B̃_η = [[τG, ηW], [0, J_p]]` (§62.4, the
   η-scaled operator of the Taylor path). `y(1) = e^{B̃_η}y(0)` is advanced over
   `t ∈ [0, 1]` with Sidje's adaptive substeps (ACM TOMS 24 (1998)): per substep an
   `m`-step Arnoldi basis of `B̃_η` from the current `y` (one modified Gram–Schmidt
   pass, as Expokit, phipm and KIOPS), the `(m+2)`-augmented Hessenberg whose extra
   rows carry the `φ₁`/`φ₂` corrector terms, local error
   `err = φ₂` if `φ₁ > 10φ₂`, `φ₁φ₂/(φ₁ − φ₂)` if `φ₁ > φ₂`, else `φ₁`, acceptance
   `err ≤ 1.2·Δt·tol·‖y(0)‖₂`, rejection with the same basis (≤ 10 per substep), and
   `Δt ← round₂(0.9·Δt·(Δt·tol/err)^{1/m})`. A happy breakdown
   (`h_{j+1,j} ≤ 1e−12·max(1, ‖B̃_η‖)`) takes the whole remaining time exactly.
2. **Only `e^{tH}e₁`.** Every quantity used (the coefficients and the two corrector
   entries) is in the first column of the Hessenberg exponential, so it is computed
   as a vector action: `s` scaled Taylor steps of degree `m_T` with `(s, m_T)` from
   the Al-Mohy–Higham selector of the generator-level `expmv` on `‖tH‖₁` —
   `O(s·m_T·(m+2)²)` per trial step instead of a dense matrix exponential.
3. **Determinism.** Fixed orders everywhere (the inner products use four interleaved
   partial sums in a fixed order); no timing or randomness: the result is
   reproducible bit for bit.
4. **API** (additive): `phi_combination_krylov(op, τ, w, out, opts, scratch) →
   KrylovPhiReport` for any `GeneratorAction`, `expmv_krylov(&GeneralOperator, τ, v,
   out, opts, scratch)` (`e^{−τA}v`), `KrylovPhiOptions { tol = 1e−10, m = 30,
   max_substeps = 100 000 }`, `KrylovPhiReport { matvecs, substeps, rejected,
   est_error }`, `PhiSolver { Taylor (default), Krylov(opts) }` and
   `Etdrk4::with_phi_solver`. All new types are `#[non_exhaustive]`. Python:
   `phi_combination(..., method="krylov", tol, m)`,
   `GeneralOperator.evolve_batched(..., method="krylov", tol, m)`,
   `Etdrk4.from_symmetric_op(..., phi="krylov", tol)`.

## Consequences

- Generator applications, drifted Fokker–Planck `n = 1000`, `τ‖G‖ = 4·10³`,
  `p = 3` combination (`G_PHI_KRYLOV_COST`, exact counter): 1 023 against 33 960
  for the Taylor path actually used (33×), agreement `9e−13` relative.
- Wall clock, upwind-drift tridiagonal `N = 1000`, `e^{−tA}v`, same session (the
  `perf_suite` row and a sweep over `t`):

  | `‖A‖t` | Taylor `expmv` | Krylov `m = 30` | Krylov `m = 15` |
  |---:|---:|---:|---:|
  | 4·10³ | 90 ms | 44 ms | 31 ms |
  | 4·10⁴ | 973 ms | 311 ms | 171 ms |
  | 4·10⁵ | 92 s | 3.7 s | 1.9 s |

  Differences against the Taylor result `≤ 1e−13`.
- ETDRK4 keeps order 4 with the Krylov φ (`G_ETDRK4_KRYLOV_ORDER`, slope 4.08) and
  matches the Taylor driver to `1e−12`.

## Honest limits

- **Orthogonalisation dominates cheap generators.** Each substep costs `O(m²n)` for
  Gram–Schmidt against `O(m·nnz)` for the applications; on a tridiagonal stencil a
  substep with `m = 30` is mostly orthogonalisation. The default `m = 30` (Expokit's)
  minimises applications; `m ≈ 15` is 1.4–2× faster in wall clock on sparse
  stencils. Below `τ‖G‖ ≈ 10³` the Taylor path is competitive, and it keeps exact,
  input-independent application counts (`G_ETDRK4_SWEEPS`); it stays the default.
- `est_error` is Expokit's estimate (the sum over substeps of
  `max(local estimate, ‖B̃_η‖·ε·‖y‖)`, relative to `‖y(0)‖₂`), not a bound; for a strongly non-normal `G` the
  forward error is not certified. `tol` is relative to `‖y(0)‖₂` of the augmented
  start vector `[w₀; 0; …; 1/η]`.
- The basis vectors are allocated per substep (not zero-alloc).
- The Arnoldi coefficients are `f64`: `F = f32` runs at `f32` accuracy; float types
  wider than `f64` (forward-mode `Dual`) are rejected with `Unsupported`.

## Gates

- `G_PHI_KRYLOV_DENSE` (`tests/g_phi_krylov.rs`): `φ_k` (`k ≤ 3`) and a full
  combination for a drifted Fokker–Planck and an inventory ladder (`n = 12`, with and
  without a lumped mass) against the dense augmented exponential, `tol = 1e−12`,
  `τ‖G‖ ∈ {1, 30, 300}`, `m = 30` (the whole space: exact projection) and `m = 6`
  (adaptive substeps, asserted `> 1` at `τ‖G‖ ≥ 30`): `≤ 1e−10·max(1, ‖want‖∞)`.
- `G_PHI_KRYLOV_COST`: the counts above, `≥ 20×` fewer applications, agreement
  `≤ 1e−8` relative.
- `G_PHI_KRYLOV_TOL`: `τ‖G‖ = 4·10²`, `p = 1`, `tol ∈ {1e−6, 1e−9, 1e−12}`: error
  against the Taylor result `≤ 10·tol·‖[w₀; w₁]‖₂`, and a tighter `tol` never costs
  fewer applications (217 / 248 / 310).
- `G_ETDRK4_KRYLOV_ORDER`: slope `∈ [3.7, 4.3]` on the Allen–Cahn problem of
  `G_ETDRK4_ORDER`, Taylor agreement `≤ 1e−12`.
