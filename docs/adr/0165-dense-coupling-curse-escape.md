# ADR-0165 — Dense / non-adjacent (all-pairs) coupling: curse-escape holds for fixed-rank-off-diagonal `D`, fail-loud for generic dense `D` (S³ POC)

**Status:** PROPOSED (proof-of-concept; design-only this round). **Date:** 2026-06-16.
**Branch:** `experiment/triz-s3-curse-escape`. **Builds on:** ADR-0164 (complex-spectral
drift), ADR-0162 (v9.1 real spectral pair-factor).
**Design:** `.dev-docs/specs/s3-dense-coupling.md` (full TRIZ + proof + boundary).
**Probes:** `.dev-docs/specs/probe_dense_rank.py`, `probe_offdiag_rank.py`,
`probe_rank1_dense.py`, `probe_triz_confirm.py`.

**Decision.** Resolve the loudest open S³ fail-loud case — the v9.1 coupled solver rejecting
**non-adjacent / dense all-pairs** coupling at construction ("adjacent pairs only; SPD-checked
non-adjacent panics") — by applying ADR-0164's complex Fourier symbol `exp(τσ(k))`,
`σ(k) = −kᵀDk + i·bᵀk`, to the **full symmetric coupling matrix `D`** (cross terms
`∂²/∂x_i∂x_j` for *every* pair, not only neighbors), with drift `b ≠ 0`. The symbol is a
single diagonal multiplier and is **exact for any `D`** (closed-form Gaussian semigroup);
`D`'s structure changes only the TT-rank of `exp(τσ(k))`, never the accuracy. The TRIZ
resolution (АП→ТП→ФП→ИКР→решение, design §1) separates the conflicting properties **in matrix
structure, not in space or time**: density (number of nonzeros) and rank (spectrum) are
independent. The make-or-break probe confirms the rank-driver is the **numerical rank `ρ` of
`D`'s off-diagonal block across a cut**, NOT the nonzero count.

**Honest S³ claim (POSITIVE — the only one proven).** A constant, symmetric, **fully dense**
diffusion matrix of the form `D = diag(a) + λ·g·gᵀ` (rank-1-dense: *every* axis pair coupled,
`d(d−1)` nonzeros, e.g. 56 at d=8), with drift `b ≠ 0` on every axis, evolves on the TT
carrier at **bounded TT-rank that SATURATES at `d≥7`** — exactly
(≤1e-12 vs an independent dense Padé `expm`), solver-free (Theorem-6 R2: FFT + elementwise
complex multiply + take-real; no LU, no dense `expm` in the evolver), and runs where the dense
reference is un-formable (d=8,10, >1 TB). The honest evidence (probe `probe_rank1_dense.py`,
max-over-all-bonds TT-SVD rank): the operator-symbol Δrank over the diagonal-only baseline is
**`4,5,6,7,7,7` for d=3..8** — it grows one-per-`d` while the single bilinear cross-generator
ladder fills (d≤6), then **saturates flat at d≥7** (max-bond rank pins at 8 = `f(ρ=1,n=5)`,
independent of `d`; the predicted curse-escape ceiling). The earlier "`5,6,7,8,8,8` saturates"
line conflated the absolute operator rank (which does plateau at 8) with the registered Δrank
metric and was measured on a non-dominant half-cut; the corrected metric is max-over-all-bonds
Δrank, saturating at the tail d≥7 (gate window MUST reach d=8 to see it). Bounded rank as
`d→∞` — climbing to the ceiling before plateauing — IS curse-escape. This strictly extends the
v9.1 *adjacent-only* carrier to a *non-adjacent / all-pairs* class. The mechanism generalizes
to rank-`m`-dense (`m` fixed in `d`; factor-model diffusion) by the same argument.

**Sharp fail-loud BOUNDARY (NEGATIVE — proven, not assumed).** Curse-escape **fails** for
*generic* dense `D` whose off-diagonal block numerical rank grows with `d`: the bond rank
explodes (probe `5→21→24→61` over d=3..6 for random dense; rank-2-dense `23→41→61` over
d=4,6,8). This is information-theoretically unavoidable — a generic dense coupling carries
`O(d²)` independent cross-parameters through a cut. The boundary is **escape ⇔ off-diagonal
block of `D` has fixed numerical rank in `d`**, and the gate PROVES it is real by a contrast
experiment (assert N-): rank-2-dense `D` must *exceed* a rank cap that rank-1-dense stays
under, at the same `d` — so the rank gate is non-vacuous and the boundary is not an artifact.

**Anti-vacuous discipline (per the prior milestone's rejected-then-fixed gate).** The rank
claim is stated as **Δrank-preservation at fixed eps on the FULL max-over-all-bonds TT-SVD
rank of the operator symbol** (difference over the diagonal-only baseline cancels the SVD
knife-edge; the full sweep — NOT a single half-cut — prevents a non-dominant bond plateauing
early from masking a still-growing true max-bond), swept over eps∈{1e-8…1e-14} × d∈{3,4,5,6,7,8}
(window reaches d=8 so tail-saturation Δ=…,7,7 is inside it); the generic-LCG-IC state arm is
**deleted** as a ceiling artifact (a full-rank state pins at `min(n^cut,n^{d−cut})` for both
operators ⇒ state-Δ identically 0 at any n; the operator-symbol Δrank is the knife-edge defense
on its own); paired with EXACTNESS ≤1e-12 vs an independent dense `expm`, a
load-bearing assert (`‖U(dense)−U(diag)‖/‖U(diag)‖ ≥ 0.05`), the rank-2 contrast (boundary is
real), operational cost-scaling at d=8,10, and a no-solver source grep. **Rejected
alternative:** orthogonal eigen-rotation `D=QΛQᵀ` to diagonalize the symbol — fatal because
the dense rotation `Q` breaks tensor-product grid alignment (off-grid resampling = the curse)
unless `Q` is itself rank-structured, in which case it reduces to the structure we exploit
directly while forfeiting exactness (design §5). **Out of scope (deferred):** variable-coef,
nonlinear, public API, rank-`m` with `m` growing in `d`. The POC ships ONE evolver variant
(`DenseCouplingSpectral`) + ONE gate (`G_S3_DENSE_COUPLING`).
