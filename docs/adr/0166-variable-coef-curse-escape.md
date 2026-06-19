# ADR-0166 — Variable-coefficient curse-escape: order-2 convergent for additive-separable `a(x),b(x),v(x)`, fail-loud for non-separable `a(x)` (S³ POC)

**Status:** PROPOSED (proof-of-concept; design-only this round). **Date:** 2026-06-17.
**Branch:** `experiment/triz-s3-curse-escape`. **Builds on:** ADR-0164 (complex-spectral
drift), ADR-0165 (dense all-pairs coupling), v9.1 (const-coef spectral pair-factor).
**Design:** `.dev-docs/specs/s3-variable-coef.md` (full TRIZ + two-layer proof + boundary).
**Probes:** `.dev-docs/specs/probe_s3_varcoef_final.py` (definitive), `probe_s3_varcoef.py`,
`probe_s3_varcoef_diffusion.py`.

**Decision.** Resolve the LAST and hardest open S³ fail-loud case — variable coefficients
`a(x), b(x)`, which every prior carrier (v9.1, ADR-0164/0165) rejects because the exact
Fourier-multiplier `exp(τσ(k))` requires CONSTANT coefficients (the generator must be diagonal
in `k`-space) — by **operator-splitting in matrix STRUCTURE plus a Chernoff-tangent factor in
τ**, not by a single multiplier. For the ADDITIVE-SEPARABLE class `a(x)=a₀+Σⱼαⱼ(xⱼ)`,
`b(x)=(bⱼ(xⱼ))ⱼ`, `v(x)=Σⱼvⱼ(xⱼ)`, the generator decomposes as `L=ΣⱼLⱼ` with each `Lⱼ` acting
on a single axis. The TRIZ resolution (АП→ТП→ФП→ИКР→решение, design §1) separates the
conflicting properties **in the operator's axis-structure (operator-splitting) and in time
(Chernoff tangency)**, not in space or time of one operator.

**Honest S³ claim (POSITIVE — TWO layers, the only one proven).**
- **Layer-1 (inter-axis) is EXACT.** Disjoint-axis generators commute (`[Lⱼ,Lₖ]=0`; probe:
  `‖[Lⱼ,Lₖ]‖=0`, `‖exp(τΣLⱼ)−∏exp(τLⱼ)‖=3.3e-16`), so `exp(τL)=∏ⱼexp(τLⱼ)` with ZERO
  splitting error; each `exp(τLⱼ)` is a **rank-1 TT operator** (`Eⱼ⊗I^{⊗(d−1)}`; probe:
  operator-TT-rank=1). This layer carries the curse and solves the entire `d`-dependence
  exactly (`n^d`→`d·n`).
- **Layer-2 (intra-axis) is ORDER-2 in τ and SOLVER-FREE.** Each 1-D variable-coef factor is
  `P₂(τ/2)·k(τ)·P₂(τ/2)` with `k=exp(τa₀Lap)` const-coef spectral (FFT-diagonal, no solve) and
  `P₂(s)=I+sRⱼ+s²/2·Rⱼ²` a polynomial Chernoff factor for the tridiagonal residual `Rⱼ=Lⱼ−a₀Lap`
  (PURE mat-vecs: NO `lu_solve_inplace`, NO `dense_expm`). Probe: rel_err `1.24e-7→1.21e-10`
  over nsteps 4..128, **slope 2.0000** vs an independent dense Padé `expm`.

Net: rank-1 TT operator per axis, `O(d·(n log n + n))` per step, **order-2 in τ**, fully
solver-free (Theorem-6 R2), runs where the dense reference OOMs (d=8,10; dense `expm` needs
1221 GB / 763 PB respectively). The escape itself (the `n^d`→`d·n` reduction) is EXACT; only
the per-axis step is order-2. This strictly extends the const-coef carrier to a variable-coef
class. There is **no fake exactness claim** — variable LEADING diffusion is provably not a
single multiplier (probe: a multiplier model converges to the wrong operator, floor `2.4e-2`),
so the result is honestly order-p (p=2), measured by a slope gate, not an exactness gate.

**Sharp fail-loud BOUNDARY (NEGATIVE — proven, not assumed).** Curse-escape **fails** for
**non-separable / cross-dependent** `a(x)` (any coefficient not a sum of per-axis functions).
Two proven failure modes: (1) **wrong-operator floor** — a per-axis additive split has no
factor for the cross term and converges to the wrong operator (probe `gate_boundary`: rel_err
plateaus at `9.53e-3`, slope `0`, does NOT →0 with τ); (2) **multiplier rank explosion** — a
non-separable potential `exp(s·W(x))` is full-rank (probe `gate_rank`: max-bond TT-rank
`5→22→24` over d=3,4,5 vs **1** for additive). The boundary is **escape ⇔ coefficient field is
additive-separable** (generator = sum of single-axis operators) — the variable-coef analogue of
ADR-0165's "escape ⇔ off-block rank fixed in d." It is **enforced by construction**: the API
accepts only per-axis coefficient arrays `aⱼ:[F;n], bⱼ:[F;n]`, making non-separable `a(x,y)`
unrepresentable (fail-loud by type, like v9.1's non-adjacent panic).

**Anti-vacuous discipline (per the two prior rejected-then-fixed milestones).** The convergence
claim is a **τ-slope gate** (log-log OLS slope ≥ 1.9 over an nsteps sweep vs an independent
dense Padé `expm`), mirroring the repo's `G3_NS2D_var`/`G4_NS2D_aniso` self-convergence slope
gates — NOT a single-τ error and NOT a spectral self-comparison. Paired with: a **load-bearing
assert** (`‖u(a_var)−u(a_const-mean)‖/‖u(a_const)‖ ≥ 0.02` AND `max αⱼ − min αⱼ > 0.1` so the
coefficient genuinely varies — anti-lesson #2/degenerate-params); the **wrong-operator-floor
boundary contrast** (non-separable `a(x)` slope <1.0 AND floor >1e-4 — makes the order gate
non-vacuous, the variable-coef analogue of ADR-0165's rank-2 contrast); **operator-TT-rank=1**
of the lifted axis factor (the rank-1 carrier, the curse-escape backbone); **operational
cost-scaling** at d=8,10 (absolute, tolerance-free); **reduction invariants** (αⱼ=0 ⇒
bit-identical to ADR-0164 const-coef spectral, 0 ULP); and a **no-solver source grep**
(`tt_varcoef_spectral.rs` contains no `lu_solve_inplace(`/`dense_expm(`). **No exactness gate**
(variable LEADING diffusion is honestly order-2, not exact — an exactness gate would be a false
claim; the slope gate is the correct truncation-bearing gate). **Rejected alternatives:** fold
variable diffusion as a position multiplier (WRONG operator — only `v(x)` may be a multiplier);
Magnus-K4 intra-axis (order-4 but commutator build cost unjustified for a POC — deferred);
eigen-rotation (breaks tensor-grid alignment, ADR-0165 §5). **Out of scope (deferred):**
non-separable `a(x)`, Magnus-4, nonlinear, time-dependent coef, public API/bindings. The POC
ships ONE evolver variant (`VarCoefSplitSpectral`) + ONE gate (`G_S3_VARCOEF_SPECTRAL`).
