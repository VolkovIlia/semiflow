# ADR-0197 — The Al-Mohy–Higham θ_m table was mis-transcribed

- **Status**: Proposed (found while implementing #24; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-14
- **Amends**: ADR-0121 (`expmv`), ADR-0185 (graph Krylov), ADR-0189 (φ-action).
- **Contract**: `contracts/semiflow-core.math.md` §45.2; gate `G_THETA_M_TABLE`.

## Context

`expmv.rs::THETA_M` and its mirror `graph_krylov.rs::THETA_M` both claimed to be
"Al-Mohy & Higham (2011) Table 3.1, double-precision subset". θ_m is the largest
argument for which a degree-`m` truncated Taylor exponential is accurate to
double-precision **backward** error; the selector `select_s_m` uses it to choose
the substep count `s = ⌈τ‖A‖ / θ_m⌉`. Too large a θ means too few substeps and a
silently inaccurate answer.

Implementing #24 (`CsrExpmvChernoff`) surfaced this: on a deliberately
non-symmetric operator with a **tight** `‖A‖_∞`, the shared selector produced
`(s, m) = (1, 18)` at `τ‖A‖ = 7.02` and the result was wrong at `1.6e−4` where
double precision was claimed. Existing callers do not see it because they pass
loose norm bounds, which inflate `s` and accidentally restore accuracy.

## The defect

The shipped pairs, against Table 3.1:

| shipped | shipped θ | correct θ_m | the shipped value is actually θ of |
|---|---|---|---|
| m = 1 | 2.29e−16 | 2.29e−16 | m = 1 ✓ |
| m = 2 | 2.58e−8 | 2.58e−8 | m = 2 ✓ |
| m = 4 | 3.40e−3 | 3.40e−4 | — (10× the correct value) |
| m = 5 | 1.44e−1 | 2.40e−3 | m = 10 |
| m = 8 | 1.44 | 5.00e−2 | m = 20 |
| m = 10 | 2.74 | 1.44e−1 | m ≈ 27 |
| m = 13 | 4.74 | 4.00e−1 | m = 35 |
| m = 18 | 8.84 | 1.09 | m ≈ 51 |

Degrees were paired with radii from rows two to three times further down the
table. The first two entries are right, which is why the table reads as
plausible.

The correct values were **recomputed from the definition** rather than
re-copied, in exact rational arithmetic: for each `m`, expand
`log(e^{−x}·T_m(x))` as a power series, take `h_{m+1}(x) = Σ_{k>m} |c_k| x^k`,
and solve `h_{m+1}(θ)/θ = u = 2^{−53}`. The result reproduces Table 3.1 to three
significant figures at every degree (`2.220e−16, 2.581e−8, 1.386e−5, 3.397e−4,
2.401e−3, 9.066e−3, 2.384e−2, 4.991e−2, 8.958e−2, 1.442e−1, …, 1.091, 1.260,
1.438, 2.429, 3.540`), which is independent confirmation that the published
table — not the recomputation — is what the code failed to copy.

Direct evidence of the consequence, forward relative error of `T_m(−x)` against
`exp(−x)`:

| x | m | relative error |
|---|---|---|
| 8.84 (shipped θ_18) | 18 | **3.8e+04** |
| 7.02 (the #24 datum) | 18 | **8.2e+01** |
| 4.74 (shipped θ_13) | 13 | **2.9e+00** |
| 1.09 (correct θ_18) | 18 | 5.0e−16 ✓ |

## Decision

**1. Replace both tables with the recomputed values**, at every degree
`m = 1…20` plus `25` and `30`, rather than the sparse subset. `select_s_m`
minimises `s·m` over the table, so a denser table is strictly cheaper as well as
correct.

**2. Raise `expmv::M_MAX` from 18 to 30.** The old cap cited "above arg ≈ 9 a
plain monomial Horner loses precision" — a constraint on the *argument*, not the
degree, and one that the wrong table was violating (it fed arg up to 8.84 into
`m = 18`). With correct radii the per-substep argument is now at most
`θ_30 = 3.54`, comfortably inside that limit at every degree. Keeping `M_MAX = 18`
would have cost `s = 7` substeps where `s = 2` at `m = 30` suffices.

**3. `graph_krylov::THETA_M` keeps its `m ≤ 18` cap**, which is structural —
`MAX_LANCZOS_DIM` sizes the `[[F; 18]; 18]` tridiagonal arrays — so only the
radii change there.

**4. `phi_action::PHI_NORM_TIGHTEN` is retained.** It compensates for the
φ-extraction needing a *tighter* argument than the plain exponential backward
error, which is a real effect independent of this bug.

## Consequences

Correct results cost more matvecs: at `τ‖A‖ = 8.84` the selector now returns
`(s, m) = (3, 30)`, cost 90, where it previously returned `(1, 18)`, cost 18 —
but that cheaper answer carried four digits of error. Callers passing loose norm
bounds are largely unaffected, because their inflated `s` was already doing the
work the table should have demanded.

`expmv_div_form_action_accuracy` passes before and after: its reference is a
self-convergence run at per-step argument ≤ 1.0, which is inside the *correct*
θ_18 = 1.09, so the reference was always sound; and its datum is a strongly
decaying symmetric operator, where a large relative error hides inside a tiny
absolute one. That is precisely why the defect survived: the existing gate
measures absolute sup-error on a decaying solution.

## Honest limits

- The recomputation assumes the standard scalar backward-error model, i.e. the
  bound is in terms of `‖A‖`. Al-Mohy & Higham's sharper `‖A^k‖^{1/k}` estimates
  are **not** implemented; with a loose `‖A‖` the selector remains conservative,
  which is the safe direction.
- `general_operator.rs` previously carried its own hand-derived criterion
  (`Z_MAX = 1.1894` from `(u·19!)^{1/19}`) specifically to avoid the shared
  selector. With the table corrected that duplication is removed and it uses
  `expmv::select_s_m`; the forward-error radius it used (1.1894) and the
  backward-error radius (1.091) agree to within 10%, so its gates are unaffected.
- Nothing here changes the Chebyshev path, whose degree comes from Bessel decay
  and never consulted this table.

## Gate

`G_THETA_M_TABLE` (`crates/semiflow/tests/theta_m_table.rs`): for every
`(m, θ)` in both tables, the degree-`m` truncated Taylor series evaluated at `±θ`
must agree with `exp(±θ)` to a relative error `≤ 1e-13`. This falsifies a
mis-paired radius directly and would have caught the original defect at every one
of the six wrong entries.
