# ADR-0167 — Non-separable variable-coefficient curse-escape: order-2 convergent for low-CP-rank `a(x),b(x),v(x)` (fixes the ADR-0166 boundary), fail-loud for generic full-rank `a(x)` (S³ POC)

**Status:** PROPOSED (proof-of-concept; design-only this round). **Date:** 2026-06-17.
**Branch:** `experiment/triz-s3-curse-escape`. **Builds on:** ADR-0166 (additive-separable
variable coef — this ADR fixes its fail-loud boundary), ADR-0165 (dense all-pairs coupling —
the "rank of the structured object" analogue), ADR-0164 (complex-spectral drift), v9.1.
**Design:** `.dev-docs/specs/s3-nonsep-varcoef.md` (full TRIZ + CP-rank mechanism + boundary).
**Probe:** `.dev-docs/specs/probe_s3_nonsep_varcoef.py` (make-or-break + ablation + rank + reduction).

**Decision.** Resolve the loudest open fail-loud case left by ADR-0166 — **non-separable**
variable coefficients, the case 0166's per-axis additive split sends to the WRONG operator
(`0.25·cos(x)·sin(y)·∂²ₓ`: slope 0, floor `9.53e-3`) — by recognising that the coefficient
**FUNCTION** `a(x)` on the tensor grid is itself a `d`-way tensor whose **CP/TT-rank** is the
structured quantity (the exact analogue of ADR-0165's coupling-matrix off-block rank). ADR-0166
FROZE `a(x)` to a scalar mean `a₀` and kept only an additive per-axis tridiagonal residual,
discarding all cross-structure. ADR-0167 instead keeps the **FULL residual generator**
`R = L − a₀·Lap` and realises it as a **low-CP-rank TT operator**: for
`a(x) = a₀ + Σ_{r=1}^{m} ∏ⱼ a_{r,j}(xⱼ)` (CP-rank `m` fixed; **`m=1` is the non-separable product
`f(x)g(y)`**), `diag(a(x))·core` is a rank-`m` TT operator
(`Σ_r ⊗ⱼ (diag·tridiag)`), so the Chernoff factor `P₂(s)=I+sR+s²/2·R²` is `2m` cheap TT mat-vecs
— **no LU, no dense `expm`** (Theorem-6 R2). The TRIZ resolution (АП→ТП→ФП→ИКР→решение, design §1)
separates the conflicting properties **in the CP-structure of the coefficient tensor** — "non-
separable in the VALUES of `a(x)`, cheap in the CP-RANK of `a(x)`" — not in space or time.

**Honest S³ claim (POSITIVE — order-2, ALL coefficient roles).** The step
`P₂(τ/2)·k(τ)·P₂(τ/2)` with `k=exp(τa₀Lap)` const-leading spectral (FFT-diagonal, no solve) and
`R` the full non-separable residual (rank-`m` TT operator) is **ORDER-2 in τ** and **solver-free**
for **leading-DIFFUSION, DRIFT, and POTENTIAL** roles alike. The make-or-break is met: the **exact
ADR-0166 fail-loud boundary** `cos(x)sin(y)·∂²ₓ` now **converges at slope +2.0000, floor `1.03e-9`**
(probe, leading-diffusion; errors in `1e-6..1e-9`, a real regime, vs an independent dense Padé
`expm` of the FULL non-separable generator). Per step: one d-D FFT pair + `4m` TT mat-vecs =
`O(d·n log n + m·d·n²)`, `O(d·n)` storage, **flat in `d`** — runs where dense `expm` OOMs (d=8,10).
On the additive sub-case the new residual is **provably identical** to 0166's (`‖R−R₀₁₆₆‖=2.22e-16`)
⇒ faithful superset of the 0166 mechanism. There is **no fake exactness claim** — variable LEADING
diffusion is provably not a single Fourier multiplier, so the result is honestly order-2 (a slope
gate, not an exactness gate), measured against an independent LU-Padé reference.

**Sharp fail-loud BOUNDARY (NEGATIVE — proven, and on a DIFFERENT axis than 0166).** Curse-escape
**fails** for **generic full-CP-rank** `a(x)`. The crucial honesty: 0166's boundary was an **ORDER**
failure (wrong operator, slope 0); 0167's boundary is a **COST/RANK** failure. The order-2 tangency
of `P₂·k·P₂` holds for ANY `R` (separable or not) — so generic `a(x)` STILL converges to the right
operator — but its residual operator has TT-rank `= n` (full; probe `gate_generic`: op-rank `[5,5,5]`
at d=2,3,4 vs **1** for rank-1, `2` for rank-2, `3` for rank-3), so the `P₂` mat-vec costs `O(n^d)`:
the **escape** is forfeit, not the convergence. The boundary is **escape ⇔ coefficient field is
low-CP-rank** (CP-rank bounded in `d`) — the variable-coef analogue of ADR-0165's "escape ⇔ off-
block rank fixed in `d`," applied to the rank of the coefficient FUNCTION. It is **enforced by
construction**: the API accepts only a fixed-`m` list of CP-terms `[(c_{r,0},…,c_{r,d-1})]`, making
generic `a(x)` unrepresentable (fail-loud by type). Truncating a generic `a(x)` to rank `m` is
REJECTED (the truncation bias is not τ-controllable ⇒ a wrong-operator floor) — `a(x)` must be
EXACTLY low-CP-rank.

**Anti-vacuous discipline (per the rejected-then-fixed prior milestones).** The convergence claim
is a **τ-slope gate** (log-log OLS slope ≥ 1.9 over an nsteps sweep vs an independent dense Padé
`expm` of the FULL non-separable generator) — NOT a single-τ error, NOT a spectral self-comparison,
and measured in a REAL error regime (`1e-6..1e-9`, orders above float floor). The headline assert is
the **make-or-break: the exact 0166 boundary case `cos(x)sin(y)` now converges at slope ≤ −1.9**.
Paired with: a **LOAD-BEARING ABLATION** (drop the cross part of `R` = freeze to mean = 0166 →
asserted to FLOOR at slope > −1.0 AND floor > 1e-4, reproducing the recorded `9.53e-3`; probe gives
slope `−0.0000`, floor `9.548e-3`) — this is the central load-bearing honesty: the SAME scheme minus
the cross term collapses to the proven 0166 boundary; the **REAL max-over-all-bonds operator-TT-rank**
of `R` = CP-rank of `a(x)` (rank-1→1, rank-2→2, rank-3→3, generic→`n`=full), flat in `d` (NO half-cut,
the M2 vacuity); the **generic-`a(x)` boundary contrast** (op-rank = full ⇒ curse cost — makes the
escape claim non-vacuous); **operational cost-scaling** at d=8,10 (absolute, tolerance-free);
**reduction invariant** (additive `a(x)` ⇒ `R` bit-identical to 0166, ≤2.2e-16; and CP-terms all
const ⇒ bit-identical to ADR-0164 const-coef, 0 ULP); and a **no-solver source grep**
(`tt_nonsep_varcoef.rs` contains no `lu_solve_inplace(`/`dense_expm(`). **No exactness gate** (variable
leading diffusion is honestly order-2; a slope gate, like 0166). **Rejected alternatives:** mean-
freeze (= the ablated 0166 boundary); rank-`m` truncation of generic `a` (uncontrollable bias);
Magnus-K4 residual (commutator rank cost — deferred); eigen-rotation (breaks tensor-grid alignment,
ADR-0165 §5). **Out of scope (deferred):** generic full-rank `a(x)`, Magnus-4, nonlinear, time-
dependent coef, public API/bindings. The POC ships ONE evolver variant (`NonSepVarCoefSpectral`) +
ONE gate (`G_S3_NONSEP_VARCOEF`). **Relation to 0166:** 0166 handles "sum of per-axis terms (any
count, commuting)"; 0167 handles "fixed count `m` of product terms (each spanning all axes, non-
separable)"; their UNION is the honest escape frontier, generic full-rank `a(x)` is the proven wall.
