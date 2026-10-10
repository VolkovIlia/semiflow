# ADR-0205 — Krylov and Taylor actions: cost by the right asymptotics, bounds that are bounds

- **Status**: Accepted
- **Date**: 2026-10-10
- **Supersedes / amends**: ADR-0185 §54.3/§54.4 (Chebyshev substepping, Lanczos
  θ-table schedule), ADR-0186 §55.4 (`MassKOperator` bound), ADR-0188
  (`G_TPS_STIFF_STEPCOUNT` definition), ADR-0203 §63.6.c/§63.7.a/§63.8 (SpMV
  bound, rounding term of the Fréchet bound, planner fields), ADR-0082/0128
  (small-`M` matrix exponentials). Fixes issue #44.
- **Contract**: `semiflow-core.errors.yaml` (#44 entries);
  `semiflow-core.math-frechet-large-t.md` §63.7.a amendment; gates
  `G_NONFINITE_BOUND_REJECT`, `G_LANCZOS_M_MAX_CAP`, `G_MATRIX_EXP_SMALL_M`,
  `G_MASSK_RIGOROUS_BOUND`, `G16C_ADJOINT_VARIABLE_DRIFT`, `G_CHEB_SQRT_COST`,
  `G_CHEB_STIFF_ORACLE`, `G_LANCZOS_HL_COST`, `G_ETDRK4_SWEEPS`, `G_EXPMV_SHIFT_COST`;
  `G_MASSK_CONSISTENT` tightened;
  `G_TPS_STIFF_STEPCOUNT` redefined and tightened (Decision 7).

## Context

Measured on the issue #16 operator (1-D Neumann Laplacian, `N = 400`):

| `λ_max·t` | Chebyshev | Lanczos (`m_max = 18`) |
|---|---|---|
| 4e3 | 2.6 ms | 204 ms |
| 4e5 | 240 ms | 19.6 s |
| 4e7 | 23.7 s, error 6.8e-6 at `tol = 1e-10` | — |

**Chebyshev.** The coefficients `e^{−z}I_k(z)` were computed as `e^{−z}` times a
power series for `I_k(z)`, which overflows near `z ≈ 700`; every action with
`z = τλ/2 > 200` was therefore split into `⌈z/200⌉` substeps of degree ≈ 101.
The cost became linear in `z`, where one expansion needs a degree
`≈ √(2z·ln(1/tol))`; the per-substep tolerance was not divided by the substep
count (hence 6.8e-6 at `tol = 1e-10`); and the series was re-summed per
coefficient per substep.

**Lanczos.** The substep length came from the Taylor backward-error radii `θ_m`
(`λh ≤ 1.09` at `m = 18`), a property of the truncated Taylor series rather than
of Lanczos, plus a dense Padé-13 of a zero-padded 18×18 matrix per substep, which
cost more than the `m` mat-vecs it served. `m` was capped at `m_max` after the
search, keeping the substep count of the larger degree.

**Bounds that were not bounds.** Issue #44: a norm bound can overflow for finite
entries, and the selectors then picked a silently wrong schedule. Separately,
`MassKOperator` estimated `λ_max(M⁻¹K)` from below (5-step inverse power started
on an eigenvector of `M`'s largest eigenvalue), so its Chebyshev series diverged
on P1 consistent mass.

## Decision

1. **One Chebyshev expansion.** `cheb_coeffs` computes `cₖ = e^{−z}Iₖ(z)` by the
   power series (`z ≤ 1`) or Miller's backward recurrence normalised by
   `c₀ + 2Σcₖ = 1` (`z > 1`); the scaled values never overflow. Degree: smallest
   `m` with `2Σ_{k>m} cₖ ≤ tol/4`, which bounds the truncation of the matrix series
   by `tol/4·‖v‖` because `‖Tₖ(B)‖₂ ≤ 1`. One series for any `z` up to
   `MAX_CHEB_DEGREE = 2²⁰` (`z ≈ 1.5·10¹⁰`); beyond, `z` is halved into equal
   substeps with `tol/s` each. `L = 0` / `τ = 0` returns the input (the old code
   divided by `λ_max = 0`).
2. **Lanczos by a-priori polynomial bounds.** Saad's lemma bounds the `m`-step
   error by `2·E_{m−1}(h)`, the best uniform approximation error of `e^{−hλ}` on
   the spectrum, which is nondecreasing in `h`. Two bounds of it are used, each
   valid for every shorter step: Hochbruck–Lubich (SIAM J. Numer. Anal. 34,
   1997, Thm 2; `ρ = λ_max/4`, `x = ρh`) `10e^{−m²/(5x)}` (`√(4x) ≤ m ≤ 2x`) or
   `(10/x)e^{−x}(ex/m)^m` (`m ≥ 2x`), and Chebyshev interpolation
   `4xᵐ/m!` (sharper for short steps, the only one for `m ≤ 2`).
   `(s, m ≤ m_max)` minimises `s·m` subject to the per-step bound `≤ tol/s`.
   `e^{−hT_m}e₁` by an implicit-QL eigen-decomposition of `T_m`, in f64. A priori
   and deterministic, so `graph_expmv_matvec_count` and the Fréchet planner stay
   exact predictors. `m_max` is no longer tied to 18; `m_max = 1` (no schedule
   reaches `tol`) is an error.
3. **Issue #44**: construction accepts operators whose bound overflows (legal with
   a mass, `G_SPDR_OVERFLOW_ROBUST`); every consumer of the bound rejects it.
4. **`MassKOperator`**: `λ_max(Â) ≤ ρ̄(K)·‖R⁻¹‖₁‖R⁻¹‖_∞`, both norms bounded in
   `O(n²)` through the comparison matrix of the Cholesky factor (Higham, ASNA
   §8.3) — rigorous, exact for bidiagonal `R`.
5. **Small-`M` matrix exponentials**: stable closed form for `M = 2`
   (`e^μ[C(δ)I + S(δ)(A − μI)]`, cos/sin branch for complex eigenvalues, series
   for coalescing ones), Padé[13/13] for every other size, real and complex.

6. **Fréchet bound and planner (amends ADR-0203 §63.6.c/§63.7.a).** The rounding
   term `2·N_chain·(r+3)·m_max²·u` assumed `m ≤ 101`; with one expansion it would
   grow like `z` per evaluation. The forward-recurrence analysis gives
   `(r+3)·u·Σ_{k≥1}c_k k(k+1) = (r+3)·u·(z/2 + Σc_k k)` per evaluation (Skellam
   variance identity `Σ_{k≥1}c_k k² = z/2`), so `FrechetPlan` gains
   `chain_weight = max over chains of Σ(⌈zᵢ/2⌉ + mᵢ)` (`mᵢ²` for Lanczos) and
   `η = 2·(N_chain·tol + (r+3)·W·u) + ε_skip + (r+n)·u·ρ̄t`. The new `η` is below
   the old one at every gate point (ratio 0.92–0.97 at `λ_max t = 1`, 0.04–0.06 at
   `1e6`); the SpMV bound `B(ρ̄t) = √(6L·N·ρ̄t) + N·2L/3` replaces
   `m_Z(3ρ̄t/200 + …)` and is asserted never to exceed it.
7. **`G_TPS_STIFF_STEPCOUNT` measures what ADR-0188 defined.** Its table defines
   `Y` as the stable path's mat-vec count; the test read the per-substep degree
   (101, with 7277 substeps). Under the ADR's definition the substep kernel had
   `X/Y = 3.96` and failed the economy clause. `Y` is now `s·m`; the economy
   clause is tightened `100 → 300` (measured 329.7, a deterministic count); the
   clause `Y ≤ 2√X` — below the minimax degree `Θ(√(X ln(1/tol)))` of any
   polynomial approximating `e^{−x}` on `[0, X]` to `tol = 1e−12` — is replaced by
   the derived `Y ≤ ⌈L/3 + √(L²/9 + 2zL)⌉ − 1` (9309). The test asserts the
   substep kernel's `X/Y < 100`.

8. **ETDRK4 in four sweeps.** Each stage of Cox–Matthews ETDRK4 is one
   combination `Σ τᵏφₖ(τL)wₖ`, which `phi_combination` evaluates in one
   augmented sweep (`a`, `b`, `c` at `τ = h/2`; the update at `τ = h` with
   `w = [u, N(u), combo2/h, combo3/h²]`). The step made nine single-φ sweeps.

9. **Trace shift in the general Taylor kernel.** `CsrExpmvChernoff` runs on
   `A − μI`, `μ = tr(A)/n`, kept only when it lowers `‖·‖_∞`, and scales by
   `e^{−τ_sμ}` per substep (Al-Mohy & Higham 2011, §3.1). With `μ = 0` the
   arithmetic is the unshifted kernel's. The φ kernels keep the unshifted
   augmented operator (`φ_k` does not factor through a shift).

Rejected: keeping substeps with a larger `Z_SAFE` (cost stays linear in `z`);
Clenshaw summation with on-the-fly coefficients (O(1) memory, but its error in
the matrix case grows like `Σ|aₖ|k²`, worse than the forward recurrence near the
spectrum ends); a-posteriori (Saad) step control for Lanczos (cheaper on smooth
data but not predictable, which the Fréchet planner and its derived gate bounds
need).

## Consequences

Measured with `examples/perf_suite.rs krylov` (release, this container, best of
runs; the issue #16 operator, 1-D Neumann Laplacian `N = 400`, `tol = 1e−10`):

| workload | before | after | speedup | sup error before → after |
|---|---:|---:|---:|---|
| Chebyshev `λt ≈ 4e3` | 2.56 ms | 0.455 ms | 5.6× | |
| Lanczos `m_max = 18`, `λt ≈ 4e3` | 204 ms | 40.0 ms | 5.1× | |
| Chebyshev `λt ≈ 4e5` | 240 ms | 4.34 ms | 55× | |
| Lanczos `m_max = 18`, `λt ≈ 4e5` | 19.6 s | 5.16 s | 3.8× | |
| Chebyshev `λt ≈ 4e7` (issue #16) | 23.7 s | 42.5 ms | 559× | 6.8e−6 → 1.4e−11 |

`G_TPS_STIFF_STEPCOUNT`: 7.35e5 → 8829 mat-vecs (83×). ETDRK4: `3·C(h/2) + C(h)`
generator applications per step instead of `5·C(h/2) + 4·C(h)` (2.25–2.6×). Fréchet gradients inherit
the Chebyshev speedup through every propagator call. Output bits of the
Chebyshev and Lanczos paths change (re-recorded digests: ADR-0205 note in
`semiflow-nostd-check`); no API is removed; `FrechetPlan` gains `chain_weight`
(unreleased type).

## Honest limits

- Chebyshev still needs `λ_max` from Gershgorin; on irregular graphs it can
  over-estimate by up to 2×, costing up to √2× more degree. A tighter estimate
  (Lanczos + residual) is not a rigorous bound, so it is not used.
- Lanczos stays linear in `τλ` at fixed `m_max` (restarted Krylov is); it is the
  right choice when `m_max` can grow like `√(τλ)` or when `v` is smooth.
- The stiff regime `τλ ≫ 1e6` is better served by the rational Krylov path
  (ADR-0206), whose cost does not depend on `λ`.

## Gate

See the gate list above; numbers in `contracts/semiflow-core.properties.yaml`.
