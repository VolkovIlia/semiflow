# ADR-0168 — Nonlinear curse-escape: EXACT-in-time for Cole-Hopf-integrable Burgers + order-2 for low-degree-polynomial reaction-diffusion, fail-loud for generic mode-mixing nonlinearity (S³ POC)

**Status:** PROPOSED (proof-of-concept; design-only this round). **Date:** 2026-06-17.
**Branch:** `experiment/triz-s3-curse-escape`. **Builds on:** ADR-0164 (complex-spectral const-coef
factor — the EXACT linear-heat carrier reused verbatim, `b=0`), ADR-0166/0167 (variable-coef Strang
split — the Seam-B mechanism is its nonlinear sibling), v9.1 (TT-carrier curse-escape backbone).
**Design:** `.dev-docs/specs/s3-nonlinear.md` (full TRIZ + dual mechanism + rank analysis + boundary).
**Probe:** `.dev-docs/specs/probe_s3_nonlinear.py` (make-or-break + rank-over-time + reduction + wall).

**Decision.** Attack THE fundamental open case — **nonlinearity**, where Chernoff's theorem and
operator semigroups (built on linear `e^{tL}`) are inapplicable and a pointwise nonlinear map blows
up TT-rank — by recognising that the resolvable conflict is **not** "nonlinear vs linear" but
**"per-step nonlinear rank-growth vs per-step diffusion rank-damping."** The TRIZ resolution
(АП→ТП→ФП→ИКР→решение, design §1) separates the conflicting properties **into the super-system**
(Seam A) and **in time** (Seam B), not by compromise. **Seam A:** viscous Burgers `u_t=ν u_xx−u u_x`
is mapped EXACTLY by the **Cole-Hopf** transform `φ=exp(−Ψ/2ν)`, `u=−2ν φ_x/φ` to the **linear heat
equation** `φ_t=ν φ_xx` — the nonlinearity is carried entirely in the cheap invertible change of
variable, the dynamics is curse-escaped linear heat (ADR-0164 `b=0`), **EXACT in time** (heat
semigroup), and **rank-1** for a separable Cole-Hopf potential. **Seam B:** reaction-diffusion
`u_t=ν Δu+f(u)` with a **low-degree-polynomial** reaction is Strang-split into
`react(τ/2)·heat(τ)·react(τ/2)`, where `react` is the **EXACT closed-form pointwise flow** of the
scalar ODE `du/ds=f(u)` (no semigroup needed; logistic `u e^{rs}/(1−u+u e^{rs})`) and `heat` is the
ADR-0164 spectral factor — **order-2** and **eff-rank-bounded** because the degree-2 reaction lifts
rank `1→2` per sub-step and the **rank-attracting diffusion holds it there**. Both are **solver-free**
(FFT + pointwise maps; no LU, no dense `expm` — Theorem-6 R2).

**Honest S³ claim (DUAL POSITIVE).** **(A) Cole-Hopf Burgers — EXACT in time, rank-1.** Probe: the
heat-semigroup property is verified (`1-shot == 8-step` to `3.16e-11` — there is structurally no
time-splitting error), the separable-potential `φ=⊗ⱼexp(−ψⱼ/2ν)` stays **rank-1 for all time**
(`[1]→[1]→[1]`), and the spatial back-map floor (`2e-4` at `n=256`) is explicitly attributed and
`→0` as `n` grows (order-2 in `dx`, by an `n`-sweep), measured vs an INDEPENDENT direct-PDE
spectral-RK4 Burgers integrator (no Cole-Hopf). **(B) Polynomial reaction-diffusion — order-2,
eff-rank-bounded.** Probe: slope `+2.0002` in a real regime (`4.3e-5→4.2e-8`) vs an INDEPENDENT
dense-FD-Laplacian + RK4 reference (different algorithm; RK4 self-converged to `3.3e-15`), with the
**make-or-break met**: a rank-1 IC stays at **effective TT-rank 2** for the entire evolution
(rank-attracting), while `max|u|` grows physically `0.8847→0.9730` (growth `1.0998`, ≈ +10%, at the
normative IC center `0.91`; gate bar `growth ≥ 1.08` — the honest reproduced value, not the earlier
center-`0.87`-derived `0.78→0.97`/`≥20%` figure), so the reaction is genuinely load-bearing (and
independently confirmed by the reaction-on ≠ reaction-off ablation, assert 5). On `f≡0` the scheme
is **bit-identical (`0 ULP`)** to ADR-0164 spectral heat —
faithful superset. There is **no fake exactness claim** for the reaction (honestly order-2, a slope
gate); only Seam A is exact-in-time, gated by the semigroup property, not a forged assert.

**Sharp fail-loud BOUNDARY (NEGATIVE — proven, on the RANK/COST axis).** Curse-escape **fails** for
**generic nonlinearity**. The crucial honesty: the Strang scheme is order-2 for ANY `f` and
Cole-Hopf is exact for ANY Cole-Hopf-integrable equation — so generic nonlinearity still CONVERGES
to the right answer — but its TT-carrier rank EXPLODES. Probe `gate_generic_rank`: the SAME rank-1
IC under a pointwise `du/ds=sin(25·u)` reaction (same Strang loop, same diffusion) saturates
effective TT-rank to `11 ≈ n/2` (full) within 5 steps — diffusion cannot re-truncate what a
mode-mixing nonlinearity excites, so the carrier mat-vec costs `O(n^d)` and the escape is forfeit.
Likewise Seam A fails for a **non-separable** Cole-Hopf potential: `φ=exp(−Ψ_gen/2ν)` has full
TT-rank `=n` (probe). The boundary is **escape ⇔ the nonlinearity's per-step rank growth is bounded
AND diffusion-truncatable** — realized by (A) Cole-Hopf-separable advection and (B)
low-degree-polynomial reaction; generic / high-degree / mode-mixing nonlinearity is the wall. It is
the nonlinear analogue of ADR-0165/0167's "escape ⇔ the structured object is low-rank," applied to
the rank-growth-vs-rank-damping balance. **Enforced by construction:** Seam A accepts only a
separable Cole-Hopf potential (`d` per-axis arrays); Seam B accepts only a `Reaction` enum of
closed-form-flow polynomial reactions — a generic transcendental `f` is unrepresentable (fail-loud
by type). Carleman linearization and per-step Newton/Picard (which would reintroduce a linear solve)
are REJECTED.

**Anti-vacuous discipline (per the rejected-then-fixed prior milestones).** The reaction claim is a
**τ-slope gate** (log-log OLS slope ≥ 1.9 over an nsteps sweep vs an independent dense-FD + RK4
reference) in a REAL error regime (`4e-5..4e-8`, orders above float floor) — NOT a single-τ error,
NOT a self-comparison. The Cole-Hopf exactness claim is gated by the **semigroup property**
(`1-shot == multi-step`, an algebraic invariant) plus an `n`-sweep attributing the residual to space
— NOT a forged exactness assert. The **make-or-break rank gate** uses the operative
**eff-TT-rank(1e-6) max-over-ALL-bonds** (NO half-cut, the M2 vacuity) — a probe bug that reported
`~n/2` flat for everything (incl. rank-1) was caught and FIXED (verified rank-1→`[1,1]`,
rank-2→`[2,2]`, generic→`[20,20]`); and a second artifact (`np.clip` of a kron product silently
lifting rank `1→11`) was caught and removed. Paired with: a **LOAD-BEARING ABLATION** (`f≡0` →
collapses to pure ADR-0164 heat, `0 ULP`, and the recorded `max|u|` growth proves the reaction is
active — without it "RD converges" could be the reference matching anything); the **generic-`f`
rank-explosion contrast** (eff-rank `→11` — makes the escape claim non-vacuous by exhibiting the
case the SAME scheme cannot escape); **reduction invariant** (`f≡0` bit-identical to ADR-0164); and
a **no-solver source grep** (`tt_nonlinear_spectral.rs` contains no `lu_solve_inplace(`/`dense_expm(`).
**No exactness gate for the reaction** (honestly order-2, a slope gate, like 0166). **Rejected
alternatives:** generic TT-cross + re-truncation (uncontrollable rank for mode-mixing `f`); per-step
Newton/Picard (reintroduces a solve); Carleman (infinite lift, fixed truncation bias). **Out of
scope (deferred):** generic/mode-mixing nonlinearity, non-separable Cole-Hopf potential, Carleman,
nonlinear Schrödinger, higher-degree-reaction calibration, public API/bindings. The POC ships ONE
module (`NonlinearSpectral`: `burgers_cole_hopf_evolve` + `strang_rd_evolve`) + ONE gate
(`G_S3_NONLINEAR`). **Relation to prior milestones:** 0164–0167 escaped the curse for LINEAR
generators of growing structural complexity; 0168 crosses into NONLINEAR dynamics for two structured
classes (Cole-Hopf-integrable advection, low-degree-polynomial reaction), with the proven wall at
generic mode-mixing nonlinearity — the honest frontier of the entire S³ program.
