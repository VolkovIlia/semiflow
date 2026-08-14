# ADR-0192 — Conservative diffusion admits degenerate conductivity `k = 0`

- **Status**: Proposed (Issue #26; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: amends ADR-0187 §"Honest limits" and
  `contracts/semiflow-core.math.md` §56.8 (`k > 0` required → `k ≥ 0` required).
  Purely widening — no existing accepted input changes behaviour.
- **Contract**: `contracts/semiflow-core.math.md` §56.8.bis (new NORMATIVE
  subsection, Proposition 56.9); gate `G_CONS_DEGENERATE`.

## Context

`ConservativeDiffusionChernoff::from_k_array` and `assemble_conservative_csr_1d`
rejected any node with `k ≤ 0`. That guard is stronger than the scheme needs, and
the inputs it rejects are not exotic: CEV in price space has
`k(S) = ½σ²S^{2β} → 0` at `S = 0`; the Feller/CIR variance process has
`k(v) = ½ξ²v → 0` at `v = 0` — the Heston volatility axis; Wright–Fisher-type
domains vanish at both ends. Callers were floored to `max(k, ε)`, which works but
is an artificial change to the model, and it moves quantiles measurably when
density mass sits near the degenerate end — exactly the regime those models are
used to study.

The harmonic-mean face conductivity already produces the correct answer there:
the harmonic mean of a conductor and an insulator is an insulator, so a
degenerate node gets `T = 0` and no flux crosses it. The degenerate end
classifies itself, which is the discrete analogue of the Feller boundary
classification and is why no extra boundary condition is needed.

## Decision

**1. Accept `k ≥ 0`; keep rejecting `k < 0` and non-finite.** The asymmetry is
deliberate. Zero is safe (Proposition 56.9): the energy identity
`⟨u, L_k u⟩ = −Σ T(u_{i+1}−u_i)²` needs only `T ≥ 0`, so `A = −L_k` stays
symmetric PSD with `diag ≥ 0` and the §55 carrier and §54 Krylov contraction are
unaffected. Negative is not: it flips the sign of a term in that sum and destroys
PSD, and **none of `from_csr`'s three checks would catch it** — finiteness passes,
symmetry passes, and the assembled diagonal `(T_{i−½}+T_{i+½})/dx` can stay
non-negative while individual faces are negative.

**2. Branch explicitly on zero rather than relying on IEEE arithmetic.** For a
single degenerate node bare arithmetic already gives the right answer
(`0·k_r/(0+k_r) = 0`, then `dx/0 = ∞`, `1/(∞+R_c) = 0`). For **two adjacent**
degenerate nodes it does not: `0·0/(0+0)` is `0/0 = NaN`. Adjacent zeros are
precisely the Wright–Fisher case, and on the
`ConservativeDiffusionChernoff → cn_step → thomas_solve` path there is no
finiteness backstop — the `assemble_conservative_csr_1d` route is saved by
`SymmetricOperator::from_csr`'s `check_finite`, but the CN/Thomas route is not, so
the state vector would go silently NaN. One comparison in `harmonic_mean` and one
in `face_transmissibility` makes the degenerate case total.

**3. No `allow_degenerate` opt-in flag.** The issue offered one as a fallback. It
is not worth the surface: with the branch in place `k = 0` is not a dangerous
input requiring opt-in, it is the mathematically natural one, and a flag would
imply the library is unsure which behaviour is correct.

**4. Make `thomas_solve` reject non-finite pivots.** The existing guard tested
`w == F::zero()`, which is false for `NaN`, so a NaN pivot passed straight through
and poisoned the solve. This is independent of the `k` relaxation — it closes the
only path in the conservative subsystem without a finiteness backstop, and it is
what makes claim 2's failure mode unreachable rather than merely unlikely.

## Rationale

The guard was protecting against a real hazard (`NaN` from `0/0`) with a blunt
instrument that also excluded the well-behaved majority of degenerate cases. Two
comparisons remove the hazard exactly, at the point where it arises, and let the
scheme accept the inputs its own face formula already handles correctly. Widening
a precondition is only honest when the wider domain is genuinely supported;
Proposition 56.9 establishes that it is, and `G_CONS_DEGENERATE` measures it.

## Consequences

- `from_k_array`, `from_k_closure` and `assemble_conservative_csr_1d` accept
  `k = 0`. No previously-accepted input changes behaviour: the new branches only
  fire on `k ≤ 0`, which used to be rejected outright.
- `validate_k_positive` is renamed `validate_k_nonneg`; the error message changes
  from "strictly positive and finite" to "non-negative and finite".
- `thomas_solve` returns `DomainViolation` on a non-finite pivot instead of
  propagating it. A caller who was somehow relying on receiving a NaN state now
  receives an error — the intended change.
- ADR-0187's "Honest limits: `k > 0`" line and math.md §56.8's first bullet are
  superseded; ROADMAP's §"#11" honest-limits entry needs the same correction.

## Honest limits

- **Order degrades at a degenerate end.** The scheme stays consistent and
  conservative, but `k → 0` makes the operator degenerate-parabolic there. The
  §56.6 interior order claim does not extend to the degenerate node and no order
  gate is asserted for it.
- **No positivity guarantee is added.** Zero flux across the degenerate face
  stops mass leaking *through* it; it does not make the scheme
  positivity-preserving.
- **`to_symmetric_operator` still emits a Neumann carrier** regardless of the
  object's configured `BoundaryPolicy` (`conservative.rs`), and
  `assemble_conservative_csr_1d` still ignores its `_boundary` argument. That
  divergence predates this ADR and is documented in the rustdoc; it is left
  unchanged here rather than turned into an error, because doing so would break
  the existing `Dirichlet → to_symmetric_operator` flow for a reason unrelated to
  issue #26. It remains a wart worth closing separately.
