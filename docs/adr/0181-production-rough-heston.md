# ADR-0181 — Production rough-Heston pricer: discounting + two-tier oracle (issue #9)

- **Status**: Accepted (design only — engineer Wave delegated separately, ADR-0181 hand-off)
- **Date**: 2026-06-23
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0082 (`MatrixDiffusionChernoff<F, M>` + §33.7 AMENDMENT 2 block-CN Strang),
  ADR-0008 (ζ-A scalar diffusion baseline), ADR-0041 (`apply_into` + `ScratchPool`),
  the CEV ncx² oracle precedent (`tests/cev_high_lam_oracle.rs` + L_CEV_PTICK).
- **Supersedes / amends**: none. STRICTLY ADDITIVE. The existing
  `examples/rough_heston_pricer.rs` capability/L-gate mode (latency demonstrator) is
  PRESERVED. Discounting is added as a parameterised reaction-matrix entry; the new
  oracle test + property gate are new files.
- **Mathematical foundation**: math.md §33 (matrix-valued operators; §33.7 AMENDMENT 2
  block-CN Strang — the per-step reaction-matrix exponential `exp(τC(x)/2)` is where the
  discount factor enters), NEW §33.9 (this ADR — discounting + the two-error-source
  decomposition).
- **Acceptance gates added**: `G_ROUGH_HESTON_MC_PARITY` (RELEASE_BLOCKING — gate I,
  kernel-vs-MC self-consistency on the SAME 4-factor Markov SDE);
  `A_ROUGH_HESTON_MODEL_BIAS` (ADVISORY record — gate II, 4-factor Markov vs high-factor
  reference). One slow MC test file + one advisory record. NO new dependency.

## Context

`examples/rough_heston_pricer.rs` (v4.0 Wave H) is a **documented latency demonstrator**:
it times one `MatrixDiffusionChernoff<f64,4>` backward step per market tick. It is NOT a
validated risk-neutral pricer. Issue #9 asks to promote it so a *price* claim — not just a
speed claim — is defensible. Three gaps:

1. **No discounting.** The spot-component reaction `c_00 = 0` (`fill_c_ij`, line 206) →
   the backward operator carries no `−r u` term → the output is a forward-ish value, not a
   discounted risk-neutral price.
2. **No accuracy oracle.** Rough-Heston has no closed form and QuantLib ships no engine.
   There is no analogue of the CEV Schroder-ncx² oracle (`tests/cev_high_lam_oracle.rs`).
3. **Documented approximations are prose, not quantities.** Frozen-V₀ spot diffusion
   (`a_00 = ½V₀`, line 181), leading-order Markov coupling (`c_{0,k} = ρξw_k`, line 211),
   O(H) characteristic-function error (3-factor Gauss-Laguerre, lines 84-89) are described
   but not measured.

**The honesty crux.** There are TWO distinct error sources that MUST stay separate:

- **(I) Numerical / kernel error** — does `MatrixDiffusionChernoff<4>` correctly solve ITS
  OWN 4-factor Markov PDE? Validated by an MC of the SAME 4-factor Markov SDE. Zero model
  assumptions enter the comparison → can be a TIGHT RELEASE gate.
- **(II) Model-approximation error** — how far is the 4-factor Markov approximation
  (+ frozen-V₀ + leading-order coupling) from TRUE rough-Heston? Quantified against a
  high-factor El Euch–Rosenbaum 2019 multifactor MC. This is a documented bias, **likely
  larger than (I)** and O(H)-scaling at H=0.1 (expect >1%). ADVISORY, never a hard gate.

Conflating (I) and (II) into one vague tolerance either overclaims ("validated price")
or yields no gate (tolerance loosened to swallow the model bias). The resolution
(TRIZ structural separation, contradiction-scan applied) is to give each its own oracle
and its own tier. The defensible claim is then explicit:

> "The kernel solves its 4-factor Markov rough-Heston model to tolerance X (gate I,
> RELEASE), and that model approximates true rough-Heston to Y (gate II, ADVISORY,
> documented)."

## Decision

### D1 — Discounting via `c_00 = −r` (parameterised, additive)

Set the spot-component reaction to `c_00 = −r`. Make `r` a parameter, not a frozen
constant, so the demonstrator's capability mode still runs (default `r = 0.05` matches the
existing `_R` constant, line 66; a `--rate` CLI flag and a `RoughHestonParams { r, .. }`
struct expose it).

**Why this discounts (against ADR-0082 §33.7 AMENDMENT 2 + math §33).** The Chernoff step
is the palindromic block-CN Strang `F(τ) = exp(τC(x)/2) ∘ DiffStep_CN(τ) ∘ exp(τC(x)/2)`
(math §33.7 AMENDMENT 2, eq. 33.2). The reaction matrix `C(x)` enters ONLY through the
per-grid-point matrix exponential `exp(τC(x)/2)` (two half-steps per `apply_into`). With
`C = diag(−r, −γ₁, −γ₂, −γ₃) + (off-diagonal coupling)`, the spot component's self-reaction
contributes a scalar factor on component 0:

```
exp(τC/2)_{00} ⊇ exp(−rτ/2)                    (the c_00 = −r diagonal entry)
```

Composed over the two Strang half-steps in one `apply_into`:

```
exp(−rτ/2) · exp(−rτ/2) = exp(−rτ)             (per backward step)
```

and over `n = T/τ` backward steps the spot density carries

```
∏_{step} exp(−rτ) = exp(−r·n·τ) = exp(−rT)     (exact risk-neutral discount)
```

This is the standard Feynman-Kac discount: the backward PDE `∂_τ u = L u − r u` has
solution `u(0) = e^{−rT} · 𝔼[payoff]` (for `c_00 = −r` constant in x). Because `c_00` is a
DIAGONAL entry it commutes with the rest of `C`, so the factorisation `exp(τC/2)_{00} =
exp(−rτ/2)·[exp of the coupling part]_{00}` is exact at the spot self-term — the discount
is reproduced to machine precision regardless of the off-diagonal coupling
`c_{0,k} = ρξw_k`. **No kernel change is needed** beyond the closure value: `fill_c_ij`
sets `mat[0][0] = -r` instead of `0.0`. The matrix-exp machinery (Cayley-Hamilton for
M=4) already exponentiates the full `C`.

**Sanity sub-check (engineer)**: with all coupling zeroed and pure discount
(`c_{0,k}=0`, flat IC `u₀≡1` on component 0), after `n` steps the spot component must equal
`exp(−rT)` to ≤1e-12. This isolates the discount factor from diffusion/coupling.

### D2 — MC reference scheme (gate I oracle): multifactor-Markovian MC of the SAME model

The oracle is an MC of the **exact same 4-factor Markov SDE** the kernel discretises — NOT
of true rough-Heston. This is the TRIZ resource: the SDE already in the topology gives a
zero-model-bias reference for gate I.

**The 4-factor Markov SDE (El Euch–Rosenbaum 2019 multifactor-Markovian form, restricted to
the 3 Gauss-Laguerre factors of the example).** Risk-neutral log-spot `X = log(S/S₀)` and
3 CIR-like variance factors `V₁,V₂,V₃`; aggregate instantaneous variance
`V = Σ_k w_k V_k` (the same `GL_WEIGHTS`):

```
dX_t   = (r − ½ V_t) dt + √V_t dW_t                       # risk-neutral drift carries r
dV_k,t = (κ(θ − V_k,t) − γ_k V_k,t) dt + ξ √(w_k V_k,t) dW^v_t ,   k=1,2,3
d⟨W, W^v⟩_t = ρ dt                                         # spot-vol correlation
V_0,k = θ  (or w_k·V_0 to match the example IC)
```

Note: the kernel uses the **frozen-V₀, leading-order-coupling LINEARISED** form of this SDE
(`a_00 = ½V₀` constant, `c_{0,k} = ρξw_k` reaction-approximated). For gate I to have ZERO
model bias, **the MC must simulate the SAME linearised/frozen approximation the kernel
encodes**, not the full nonlinear SDE above. Concretely the MC integrates the kernel's PDE
generator:

```
dX_t   = (r − ½ V_0) dt + √V_0 dW_t                       # frozen-V₀ spot diffusion (matches a_00, b_00)
dV_k,t = (κ(θ − w_k V_0) − γ_k V_k,t) dt + ξ√(w_k V_0) dW^v_t   # matches a_kk, b_kk, c_kk
                       + (leading-order coupling term ρξw_k as encoded in c_{0,k})
```

This is the **honesty pivot**: gate I compares two discretisations of one operator
(Chernoff PDE step vs Euler/QE SDE step), so any disagreement is pure numerical error.
The MC of the FULL nonlinear SDE belongs to gate II (model bias), not gate I.

**Discretisation**:
- **Scheme**: Euler–Maruyama on `X`; **QE (Andersen 2008 Quadratic-Exponential)** on the
  CIR variance factors `V_k` to keep them non-negative without the Euler full-truncation
  bias (Andersen, *Efficient simulation of the Heston stochastic volatility model*, J.
  Comp. Finance 11:3, 2008). Correlated Brownian increments via Cholesky of the 2×2
  `[[1,ρ],[ρ,1]]` block.
- **`n_steps = 200`** per year (matches the kernel τ-grid resolution `TAU = 0.025` →
  40 steps/yr is the kernel's; MC uses finer 200 steps to drive SDE-discretisation error
  well below MC stderr so the comparison is MC-stderr-dominated).
- **`n_paths = 1_000_000`** with **antithetic** variance reduction (pair each path with its
  Brownian sign-flip) → effective stderr ≈ `σ_payoff / √(2·N_pairs)`.
- **Fixed seed**: `PCG64(0xC0FFEE_BABE_DEAD_BEEF)` — the project-canonical deterministic
  seed (matches L_CEV_PTICK.canonical_input.seed, ADR-0082 G_MATRIX seed). Reproducible
  across CI + reviewer audits.
- **Estimator**: discounted European call `Ĉ = e^{−rT} · mean_paths[(S_T − K)_+]`, with
  reported `MC_stderr = e^{−rT} · std_paths[(S_T − K)_+] / √N_eff`. Put via put-call parity
  cross-check.

**Where it lives**:
- **Rust oracle test** `crates/semiflow-core/tests/rough_heston_mc_oracle.rs`, mirroring
  `tests/cev_high_lam_oracle.rs` (self-contained, deterministic, the `G_ROUGH_HESTON_MC_PARITY`
  gate). `#[cfg_attr(not(feature="slow-tests"), ignore)]` (1M paths is a slow test).
- **Python cross-check** `scripts/verify_rough_heston_mc.py` (numpy/scipy QE-MC; the
  language-independent pre-flight reference, prints a `PASS` line — mirrors
  `scripts/verify_graphadjoint_sampled.py`).

### D3 — Gate I: `G_ROUGH_HESTON_MC_PARITY` (RELEASE_BLOCKING)

Mirror the CEV oracle pattern. Assert kernel price ≈ MC price of the SAME model:

```
|C_chernoff − C_mc| ≤ tol ,   tol = k · MC_stderr + δ_kernel
```

- `k = 3` (3σ band on the MC reference — false-fail probability ≈ 0.3%).
- `δ_kernel` = stated kernel space+time discretisation margin. The kernel is order-2
  (ADR-0082 §33.7 AMENDMENT 2 block-CN Strang) on a coarse `N_GRID=48`, `TAU=0.025` grid;
  `δ_kernel` is the kernel's own truncation at that resolution, **measured** by self-
  convergence (kernel at N=48 vs N=192) and recorded as a constant in the test
  (initial estimate `δ_kernel ≈ 5e-3 · S₀`; the engineer fits it from the self-convergence
  run and the value becomes the literal in the test + this ADR amendment).

**Test points** (parameters H=0.1, r=0.05, v0=0.04, κ=1.5, θ=0.04, ξ=0.3, ρ=−0.7,
S₀=100, T=1.0 — the example's canonical set):
- Strikes `K ∈ {90, 100, 110}` (ITM / ATM / OTM call).
- Maturity `T = 1.0` (single maturity for the RELEASE gate; multi-T deferred to advisory).

**Tolerance numbers (target, to be confirmed at rc.1 by the engineer)**:
- MC_stderr at N_eff = 2M (1M antithetic pairs) for an ATM call at these params:
  `σ_payoff ≈ 8` (price units), `MC_stderr ≈ 8/√2e6 ≈ 5.7e-3`.
- `tol = 3 · 5.7e-3 + δ_kernel ≈ 1.7e-2 + 5e-3·100`-scaled term. **Gate I target:**
  `|C_chernoff − C_mc| ≤ 0.55` price units (≈ 0.6% of a ~9.0 ATM call). The dominant
  contributor is `δ_kernel` (coarse grid), NOT MC noise; tightening the grid tightens the
  gate. This is RELEASE-grade because it measures ONLY the kernel.

RELEASE_BLOCKING if achievable at these numbers; if the engineer's rc.1 measurement shows
`δ_kernel` cannot be driven below ~1% at the demonstrator's coarse grid without breaking
the latency story, the gate ships RELEASE_BLOCKING on a **dedicated accuracy grid**
(`N_GRID=192`, `TAU=0.01`) while the latency demonstrator keeps its coarse grid — the two
modes are separated (accuracy mode vs capability/latency mode), see hand-off.

### D4 — Gate II: `A_ROUGH_HESTON_MODEL_BIAS` (ADVISORY record, not a gate)

Quantify the model-approximation error (II) as a documented number, NOT a CI gate:

- **Frozen-V₀ bias**: the kernel freezes spot diffusion at `a_00 = ½V₀`. At H=0.1 the
  variance moves over `[0,T]`; the frozen-V₀ vs stochastic-V spot-variance discrepancy
  drives a documented bias. Measured by an MC with **stochastic** `√V_t` spot diffusion
  (full SDE of D2) vs the frozen-V₀ MC.
- **Leading-order Markov coupling bias**: `c_{0,k} = ρξw_k` is a reaction-term
  approximation of the true `ρ√(V_k) ∂_x` cross-term. Measured by an MC with the exact
  correlated cross-term vs the reaction-approximated one.
- **O(H) characteristic-function error**: the 3-factor Gauss-Laguerre approximation of the
  fractional kernel `K(t) = t^{H−½}/Γ(H+½)` has O(H) CF error (lines 84-86). Measured by a
  **high-factor** El Euch–Rosenbaum MC (e.g. 20 factors) vs the 3-factor MC, both at H=0.1.

**Recorded as** `A_ROUGH_HESTON_MODEL_BIAS` in properties.yaml under a NEW `advisory_records:`
top-level section (sibling to `latency_gates:`; advisory by construction — never exits 1).
Each sub-bias reports a measured price-difference in % of the ATM call. **Honest expectation
(stated up front, to be confirmed):** the aggregate (II) bias at H=0.1 is **O(H) ≈ 1–5%**,
materially LARGER than gate I's ~0.6%. The advisory record makes this inspectable; the
price-claim language (below) is bounded by it.

### D5 — Honest acceptance (the defensible claim)

| Tier | Gate | Error source | Reference | Target tolerance | Severity |
|------|------|--------------|-----------|------------------|----------|
| I | `G_ROUGH_HESTON_MC_PARITY` | numerical/kernel (I) | MC of the SAME 4-factor Markov model | ≤ 0.55 price units (~0.6% ATM) | RELEASE_BLOCKING |
| II | `A_ROUGH_HESTON_MODEL_BIAS` | model approximation (II) | high-factor ER-2019 MC + stochastic-V MC | measured, expected 1–5% (O(H)) | ADVISORY |

**Defensible price-claim**: *"`rough_heston_pricer` computes a discounted risk-neutral
price of its 4-factor Markov rough-Heston model, validated against an independent
Monte-Carlo of that same model to ~0.6% (gate I, RELEASE). That 4-factor Markov model is
itself an O(H)-biased approximation of true rough-Heston; the bias is documented at
~1–5% at H=0.1 (gate II, ADVISORY). The pricer is therefore an oracle-validated solver of
a documented approximate model — NOT a validated approximation of true rough-Heston."*

This is a truthful bounded claim. The latency demonstrator capability is preserved; the new
artefact is an honestly-tiered price-claim, not an overclaim.

## Rationale

- **Why separate the two error sources** (vs one tolerance): a single tolerance must either
  swallow the ~1–5% model bias (and then proves nothing about the kernel) or be tight and
  always fail against true rough-Heston (no reference exists). Structural separation gives
  a tight, honest RELEASE gate AND an honest documented bias — both properties at once
  (contradiction resolved, not split).
- **Why MC of the SAME model for gate I** (vs MC of true rough-Heston): the kernel's
  contract is "solve THIS 4-factor Markov PDE." An MC of that exact SDE has zero model bias
  relative to the contract, so disagreement is pure numerical error — the only thing a
  RELEASE gate on the kernel should measure. Mirrors how the CEV gate measures the kernel
  against the ncx² closed form of the SAME CEV model.
- **Why QE for the CIR factors** (vs Euler full-truncation): Euler on CIR injects an O(√Δt)
  positivity-bias that would contaminate `δ` and force a looser `tol`. QE (Andersen 2008) is
  the standard low-bias CIR scheme; keeps the MC reference MC-stderr-dominated.
- **Why `c_00 = −r` (vs adding a separate discount multiply after evolution)**: the reaction
  matrix already exponentiates per step; a diagonal `−r` rides the existing machinery and
  reproduces `e^{−rT}` exactly with ZERO new code. A post-hoc `× e^{−rT}` multiply would be
  a second code path that the engineer could desync from the PDE.
- **Why advisory (not blocking) for gate II**: the model bias is a property of the
  approximation choice (3 factors, frozen V₀), not of the implementation; blocking on it
  would gate the release on a modelling decision, which is the user's to make. Documenting
  it keeps the claim honest without freezing the model.
- **Why parameterise `r` and the approximations** (vs frozen constants): issue #9 gap 3
  asks to make the approximations inspectable; a `RoughHestonParams` struct + flags let the
  advisory bias-measurement vary them and report magnitudes, and let the discount be turned
  off (`r=0`) to recover the old demonstrator behaviour exactly.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| One tolerance vs true rough-Heston | No closed form, no engine; gate would be untestable or loosened to meaninglessness. Two-tier separation is the only honest design. |
| Post-evolution `× e^{−rT}` discount multiply | Second code path divergeable from the PDE; `c_00 = −r` rides existing matrix-exp exactly. |
| Euler full-truncation CIR MC | O(√Δt) positivity bias contaminates the kernel-vs-MC δ; QE is the standard low-bias scheme. |
| Make gate II RELEASE_BLOCKING | Gates the release on a modelling choice (factor count, frozen-V₀), not on implementation correctness. Advisory keeps the model free + the claim honest. |
| Drop the price-claim, keep capability-only (the non-goal escape) | The user chose full implementation; a defensible bounded claim IS achievable via the two-tier design. Escape is valid only if (I) were also untestable — it is not. |
| Single ATM strike for gate I | Three strikes (90/100/110) cheaply cover ITM/ATM/OTM smile shape; the MC paths are shared across strikes (one path set, three payoff reductions). |

## Consequences

- **Demonstrator preserved.** Capability/L-gate mode runs unchanged (`r=0.05` default;
  `--rate 0.0` recovers the pre-#9 forward-ish behaviour).
- **New gate** `G_ROUGH_HESTON_MC_PARITY` (RELEASE_BLOCKING, slow-tests) in
  `tests/rough_heston_mc_oracle.rs`.
- **New advisory record** `A_ROUGH_HESTON_MODEL_BIAS` under a NEW `advisory_records:`
  section in properties.yaml (advisory by construction).
- **New oracle** `scripts/verify_rough_heston_mc.py` (numpy/scipy; pre-flight reference).
- **math.md §33.9 NEW** — discounting (`c_00=−r` → `e^{−rT}`) + the two-error-source
  decomposition with magnitudes.
- **Dependency count unchanged** (Rust: no new dep; Python oracle uses numpy/scipy already
  in scripts/ convention).
- **`examples/rough_heston_pricer.rs`** gains a `RoughHestonParams` struct + `--rate` flag +
  `c_00 = −r` in `fill_c_ij`; ~30 LoC, additive, keeps fns ≤50 lines.

## Cross-references

- ADR-0082 — `MatrixDiffusionChernoff<F, M>`; §33.7 AMENDMENT 2 block-CN Strang (reaction
  matrix entry point for the discount).
- `tests/cev_high_lam_oracle.rs` + L_CEV_PTICK — the oracle/gate pattern this mirrors.
- math.md §33 (matrix operators), NEW §33.9 (this ADR's companion).
- Andersen 2008, *Efficient simulation of the Heston SV model*, J. Comp. Finance 11:3 — QE
  CIR scheme.
- El Euch, Rosenbaum 2019, *The characteristic function of rough Heston models*, Math.
  Finance 29:1 — multifactor-Markovian rough-Heston (gate II high-factor reference).
- Carr, Cisek, Pintar 2021 — the 3-factor Gauss-Laguerre Markov approximation (the kernel's
  model; source of the O(H) CF error in gate II).
- `docs/adr/0181-engineer-handoff.md` — file checklist + verification commands.
