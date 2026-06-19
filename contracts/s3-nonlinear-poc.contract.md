# Contract — S³ nonlinear POC (`NonlinearSpectral`: Cole-Hopf Burgers + Strang-split RD)

**Scope:** proof-of-concept ONLY. ONE new module (two evolver variants) + ONE gate test. No public
API churn, no migration, no library-wide rollout. Suckless: ≤500 LoC/file, ≤50 LoC/fn, no new dep
(reuses `tt_drift_spectral.rs` `apply_drift_spectral_axis` for the heat factor + its 1-D DFT
helpers + `SemiflowFloat` `exp/sin/cos`; adds small spectral first-derivative/antiderivative + the
pointwise reaction-flow / Cole-Hopf maps).
**Design:** `.dev-docs/specs/s3-nonlinear.md` (TRIZ + dual mechanism + rank analysis + boundary).
**Probe (truth):** `.dev-docs/specs/probe_s3_nonlinear.py` (make-or-break: Seam-A Cole-Hopf EXACT-in-
time `1-shot==8-step→3.16e-11`, φ rank-1 for separable potential; Seam-B Strang logistic RD slope
`+2.0002` real regime `4.3e-5..4.2e-8`, eff-TT-rank bounded at `2`; WALL: generic `sin(25u)` reaction
eff-rank `→11≈n/2`; reduction `f≡0` → ADR-0164 heat `0 ULP`).
**Builds on:** `crates/semiflow-core/src/tt_drift_spectral.rs` (ADR-0164 `b=0` linear heat) and the
ADR-0166/0167 Strang-split discipline — the ONLY new ingredients are the **Cole-Hopf forward/back
maps** (Seam A) and the **exact pointwise polynomial reaction flow** (Seam B).

---

## 1. New types / functions (Rust, `no_std` + `alloc`, generic over `F: SemiflowFloat`)

All live in a NEW module `crates/semiflow-core/src/tt_nonlinear_spectral.rs` (keeps the POC isolated;
≤500 LoC). Reuse `apply_drift_spectral_axis(line, n, dx, ν, 0, τ)` (ADR-0164, `b=0`) for every heat
factor. NO LU, NO dense `expm` anywhere in the evolvers.

### 1.1 Reaction enum (the construction-time fail-loud boundary for Seam B)

```rust
/// Closed-form-flow polynomial reactions ONLY. Generic / transcendental f is UNREPRESENTABLE
/// by construction (the wall: design §4). Each variant carries the EXACT pointwise flow of
/// du/ds = f(u). debug_assert checks u stays in the admissible range for Logistic (0,1).
pub(crate) enum Reaction<F: SemiflowFloat> {
    Logistic { r: F },                 // f(u)=r u(1-u);  flow u e^{rs}/(1-u+u e^{rs})
    Linear   { c: F },                 // f(u)=c u;       flow u e^{cs}  (reduction sanity)
    Quadratic{ a: F, b: F, c: F },     // f(u)=a u^2+b u+c; flow via closed form (small a)
}

/// Apply the EXACT pointwise reaction flow Phi_f(u, s) ELEMENTWISE to a flat n^d state.
/// NO solve, NO semigroup — the exact scalar-ODE flow map (closed form per variant).
pub(crate) fn react_flow<F: SemiflowFloat>(u: &mut [F], reaction: &Reaction<F>, s: F);
```

### 1.2 Seam B — Strang-split reaction-diffusion evolver (order-2, eff-rank-bounded)

```rust
/// Evolve u0 (flat n^d real) by u_t = nu*Lap u + f(u) via the symmetric Strang sandwich
///   react(tau/2) . heat(tau) . heat applied PER AXIS via apply_drift_spectral_axis(nu,0,tau) .
///   react(tau/2)
/// heat = ADR-0164 spectral factor (b=0), exact-in-time, solver-free; react = exact pointwise
/// polynomial flow. Order-2 in tau. NO lu_solve_inplace, NO dense_expm (Theorem-6 R2).
pub(crate) fn strang_rd_evolve<F: SemiflowFloat>(
    u0: &[F], n: usize, d: usize, dx: F, nu: F, reaction: &Reaction<F>, tau: F, nsteps: usize,
) -> Vec<F>;
```

### 1.3 Seam A — spectral first derivative + antiderivative helpers (Cole-Hopf maps)

```rust
/// Spectral first derivative d/dx of a 1-D periodic real line: ifft(i k * fft(line)).real.
/// Reuses the 1-D DFT helpers from tt_spectral/tt_drift_spectral. NO solve.
pub(crate) fn spectral_deriv_1d<F: SemiflowFloat>(line: &[F], n: usize, dx: F) -> Vec<F>;

/// Spectral periodic antiderivative Psi with Psi' = u (zero mean enforced on u; Psi_hat[0]=0):
///   Psi_hat[k] = u_hat[k] / (i k), k != 0. NO solve. Requires sum(u)=0 for single-valuedness.
pub(crate) fn spectral_antideriv_1d<F: SemiflowFloat>(u: &[F], n: usize, dx: F) -> Vec<F>;
```

### 1.4 Seam A — Cole-Hopf Burgers evolver (EXACT-in-time, rank-1 for separable potential)

```rust
/// Evolve viscous Burgers u_t = nu*u_xx - u*u_x (1-D periodic, zero-mean u0) EXACTLY in time:
///   1. Psi = spectral_antideriv_1d(u0 - mean(u0))           [forward Cole-Hopf]
///   2. phi = exp(-Psi / (2 nu))                              [pointwise]
///   3. phi <- apply_drift_spectral_axis(phi, n, dx, nu, 0, T) [LINEAR heat, exact-in-time, no solve]
///   4. u = -2 nu * spectral_deriv_1d(phi) / phi              [back Cole-Hopf, pointwise divide]
/// EXACT in time (heat semigroup): for ANY split of T into substeps the result is identical
/// (gated by the semigroup invariant, contract assert A1). NO solve, NO expm in the evolver.
pub(crate) fn burgers_cole_hopf_evolve<F: SemiflowFloat>(
    u0: &[F], n: usize, dx: F, nu: F, t_final: F,
) -> Vec<F>;
```

### 1.5 Reduction invariants (NORMATIVE — Gate sub-checks)

(a) **`f≡0` reduction:** `strang_rd_evolve` with `Reaction::Linear{c:0}` (flow = identity) MUST equal
a pure per-axis `apply_drift_spectral_axis(ν,0,τ)` heat evolution to **`0 ULP`** (probe `0.000e+00`).
Proves Seam B is a faithful superset of the ADR-0164 const-coef heat path.
(b) **Cole-Hopf semigroup exactness:** `burgers_cole_hopf_evolve` evolving `T` in ONE shot vs `k`
substeps (re-fold φ each substep) MUST agree to ≤`1e-9` (probe `3.16e-11`). Proves the time-exactness
(the heat semigroup carries the nonlinearity with no splitting error) — this is the Seam-A exactness
gate (NOT a forged exactness-vs-reference assert).

---

## 2. The ONE gate that proves S³ (`G_S3_NONLINEAR`)

`crates/semiflow-core/tests/g_s3_nonlinear.rs`, RELEASE-BLOCKING-class but gated
`#[cfg_attr(not(feature = "slow-tests"), ignore)]` (fine-RK4 reference is expensive). HARD asserts.

**References (independent, NO shared code with the evolver):**
- **Seam B:** assemble the **dense periodic FD Laplacian** (Kronecker-lifted per-axis tridiagonals —
  NOT the spectral factor) and integrate `u_t = ν·Lap_h·u + f(u)` by **RK4** at fine `dt`
  (`fine=40000`). Different ALGORITHM (real-space FD + explicit RK, not FFT + closed-form flow) ⇒
  genuine independence. The evolver is ALSO re-implemented locally in the gate (zero reuse of the
  production module).
- **Seam A:** integrate Burgers **directly** (spectral `u_xx`,`u_x` + RK4 in time, NO Cole-Hopf) at
  fine `dt`. Different algorithm ⇒ independence. The Cole-Hopf time-exactness is gated SEPARATELY by
  the semigroup property (assert A1) so it does not depend on the reference.

**Frozen params (§A pre-registration):**
- Seam B order: `n=24`, `d=2`, `dx=2π/n`, `ν=0.10`, `Reaction::Logistic{r=6.0}`, `T=0.40`,
  separable IC `u₀=⊗(0.3+0.25cos)`, nsteps sweep `{4,8,16,32,64,128}`.
- Seam B rank: `n=20`, `d=3`, `ν=0.10`, `Reaction::Logistic{r=4.0}`, `T=0.40`, nsteps `40`,
  rank-1 IC `u₀=⊗(0.91+0.05cos)` (product stays in `(0,1)`, NO clip).
- Seam A: `ν=0.10`, `T=0.30`, IC `u₀=sin(x)`, `n` sweep `{32,64,128,256}`.
- Generic-`f` wall: `n=20`, `d=3`, pointwise `du/ds=sin(25u)` (RK4 sub-stepped) in the SAME Strang
  loop.

### Asserts (all HARD)

1. **MAKE-OR-BREAK A — Cole-Hopf Burgers is EXACT IN TIME (headline #1).** Evolve `T=0.30` via
   `burgers_cole_hopf_evolve` in ONE shot and (re-folding φ) in `8` substeps; assert
   `max|u_1 − u_8| ≤ 1e-9`. [Probe: `3.16e-11`.] Then assert convergence vs the INDEPENDENT direct-PDE
   spectral-RK4 reference IMPROVES with `n` at order ≈2 in `dx`: rel_err `{n=64}/{n=128} ≥ 3.5` and
   `{n=128}/{n=256} ≥ 3.5` (probe `3.22e-3 → 8.07e-4 → 2.02e-4`, ratio ≈4).
   *Anti-vacuous:* the time-exactness is an ALGEBRAIC invariant (semigroup: any substepping gives the
   same answer), not a tolerance match — it proves the nonlinearity is carried with zero time-splitting
   error. The `n`-sweep ATTRIBUTES the residual vs the independent reference to SPACE (back-map FD),
   not time, so the exactness claim is honest (no forged exactness-vs-reference assert).

2. **MAKE-OR-BREAK B — low-degree-polynomial RD converges at order-2 (headline #2).** On `d=2`,
   `Reaction::Logistic{r=6}`, measure `rel_l2(strang_rd_evolve(nsteps), u_ref)` over the nsteps sweep
   vs the INDEPENDENT dense-FD + RK4 reference; log-log OLS slope on the asymptotic tail (drop the 2
   coarsest). Pass ⇔ **slope ≤ −1.9** (order ≥ 1.9; probe `+2.0002`) AND **real regime** (coarsest
   err > 1e-5, finest < 1e-5, all ≥ 100× float floor). [Probe: errs `4.3e-5 → 4.2e-8`, slope `+2.0002`.]
   *Anti-vacuous:* a τ-slope gate in a real error regime vs an independent different-algorithm
   reference (real-space FD + RK4, not FFT + closed-form flow) — not a single τ, not a self-comparison.
   Proves the Strang scheme is genuinely order-2 to the TRUE nonlinear flow.

3. **MAKE-OR-BREAK RANK (the central S³ assert) — structured nonlinearity keeps eff-TT-rank
   BOUNDED.** On `d=3`, `Reaction::Logistic{r=4}`, rank-1 IC, evolve `40` Strang steps; at steps
   `{1,5,20,40}` read the **eff-TT-rank(1e-6) max-over-ALL-bonds** (NO half-cut — the M2 vacuity).
   Assert: initial eff-rank `= [1,1]`; max eff-rank over the evolution `≤ 3` (probe stays at `2`);
   AND `max|u|` grows by **≥ 8%** over the run (`growth = max|u|_final / max|u|_initial ≥ 1.08`),
   proving the reaction is LOAD-BEARING, not inert. **The growth bar is HONEST:** at the normative
   IC center `0.91` (L113, probe L211) the reproduced evolution is `max|u| 0.8847 → 0.9730`, i.e.
   `growth = 1.0998` (≈ +10%); the `≥ 1.08` threshold is the real value with a small margin, NOT a
   number the IC cannot meet. (A prior `≥ 20% / 0.78→0.97` figure was a center-`0.87`-derived
   inconsistency vs the `0.91` IC and is corrected here.) The growth bar is also **not the sole**
   load-bearing check: assert 5 independently proves reaction-on materially differs from reaction-off
   (`f≡0`), so the reaction's load-bearing role does not rest on the growth bar alone.
   [Probe (IC center 0.91): `[1]→[2]→[2]→[2]`, `max|u| 0.8847→0.9730`, growth `1.0998`.]
   *Anti-vacuous:* this is the literal make-or-break — does the pointwise nonlinear map blow TT-rank?
   For low-degree polynomial it does NOT (rank `1→2`, diffusion rank-attracts and holds it). The
   eff-rank(1e-6) max-over-all-bonds is the operative curse-escape quantity (what a TT solver keeps),
   read with the honest M3/M4 metric (NOT the M2 half-cut). The `max|u|` growth assert (real `+10%`,
   gated `≥ 1.08`) blocks the degenerate "reaction is inert so of course rank is preserved" vacuity,
   and assert 5 backs it independently.

4. **NEGATIVE BOUNDARY — generic mode-mixing nonlinearity BLOWS rank (the wall).** With the SAME
   rank-1 IC and SAME Strang loop but a pointwise `du/ds = sin(25u)` reaction (RK4 sub-stepped),
   evolve and read eff-TT-rank(1e-6) at steps `{1,5,20,40}`. Assert eff-rank SATURATES:
   max eff-rank `≥ 8` (≈ `n/2`; probe `→ 11`), strictly larger than the structured case's `≤3`.
   Also assert (Seam A wall) that `φ₀ = exp(−Ψ_gen/2ν)` for a GENERIC random `Ψ_gen` (`n` grid) has
   eff-TT-rank `= n` (full; probe `[n]` at `n=32`), vs rank-1 for a separable `Ψ`.
   *Anti-vacuous:* makes the escape claim non-vacuous by exhibiting the case where the SAME scheme
   cannot escape (rank explosion). The boundary is on the RANK/COST axis: the Strang scheme is still
   order-2 for generic `f` (it converges to the right answer) — what fails is the curse-ESCAPE
   (carrier rank `→ full ⇒ O(n^d)` mat-vec). Distinct, honest boundary.

5. **LOAD-BEARING ABLATION + REDUCTION (`f≡0`).** Assert `strang_rd_evolve` with
   `Reaction::Linear{c:0}` (flow = identity) equals a pure per-axis `apply_drift_spectral_axis(ν,0,τ)`
   heat evolution to **`0 ULP`** (`to_bits()` equality; probe `0.000e+00`). Then assert the SAME
   evolver with `Reaction::Logistic{r=6}` DIFFERS from the `f≡0` result by a physically large margin
   (rel diff ≥ 1e-2 at `T=0.40`). [Probe: `max|u|` grows under reaction; `f≡0` is pure heat.]
   *Anti-vacuous:* (a) the `0 ULP` reduction proves Seam B is a faithful superset of ADR-0164 (not a
   relabel); (b) the "reaction-on differs materially from reaction-off" assert is the load-bearing
   ablation — it proves the nonlinearity does real work (without it, "RD converges to the reference"
   could be the reference matching the linear part).

6. **OPERATIONAL cost-scaling (absolute, tolerance-FREE escape).** The Seam-B evolver produces a
   finite, real state for the rank-1 structured class at `d ∈ {8,10}` (`n=5`, nsteps=8) using only a
   bounded-rank carrier + per-axis FFTs + pointwise reaction, whereas a dense `n^d`-state (8-byte
   `f64`) explicit integrator needs an un-formable state. The dense byte-count is evaluated at the
   **fixed `n = 27`** so the literal `> 1 TB` contract bar holds UNIFORMLY for both `d` with NO
   per-`d` threshold split: `27^8·8 = 2.26 TB` (`d=8`) and `27^10·8 = 1647 TB` (`d=10`), both
   `> 1 TB`. (The evolver-runs-finite check uses its own `n=5`; only the byte-count assert is pinned
   to `n=27`.) State bytes are `n^d · 8` (a single dense `f64` state on the `n^d` grid). Pass ⇔ the
   evolver runs (finite, real) at `d=8,10` AND the static byte-count `27^d·8 > 1 TB` holds for both
   `d` (uniform `> 1 TB`, no split).
   *Anti-vacuous:* an absolute resource statement with no tolerance knob, matching the literal
   contract bar at a single fixed `n` — the real curse the scheme escapes for the structured
   nonlinear class.

7. **NO-SOLVER audit.** A source-level grep asserts `tt_nonlinear_spectral.rs` contains NO
   `lu_solve_inplace(` and NO `dense_expm(` call-sites (Theorem-6 R2; dense FD-RK4 / direct-PDE-RK4
   live ONLY in the gate references). The Cole-Hopf maps use only FFT + pointwise `exp`/divide; the
   reaction uses only the closed-form pointwise flow; the heat uses only ADR-0164's spectral factor.
   *Anti-vacuous:* solver-free proof — the evolver never forms or solves an `n^d` linear system.

### 2.4 Why the gate satisfies the audit (NORMATIVE cross-reference)

The surviving audit counter-probes from prior milestones are honoured: (a) **knife-edge** — the
reaction headline (assert 2) is a **τ-slope** gate (order across a τ-sweep, not a single
SVD-tolerance rank read), robust to the SVD knife-edge; the Cole-Hopf headline (assert 1) is an
ALGEBRAIC semigroup invariant; the rank asserts (3,4) use **eff-TT-rank(1e-6) max-over-ALL-bonds**
(NOT the M2 half-cut) and contrast structured (`≤3`) vs generic (`≥8`), a rank-DIFFERENCE structure
that is knife-edge-robust; (b) **degenerate / rigged params** — the reaction is the LITERAL Fisher-KPP
logistic `r·u(1−u)` (genuinely nonlinear), the LOAD-BEARING checks (assert 3 `max|u|` growth, assert
5 reaction-on≠reaction-off) prove it does real work, and a probe artifact (`np.clip` of a kron product
silently lifting rank) was caught and removed; (c) **wrong-answer masking** — the references are
INDEPENDENT (Seam B: real-space FD + RK4; Seam A: direct-PDE spectral RK4), different algorithms with
zero shared code, measured in a real error regime; the Cole-Hopf exactness is gated by the semigroup
invariant, not a forged assert; (d) **honest boundary** — assert 4 proves the wall (generic
mode-mixing nonlinearity blows rank) is real and DISTINCT from an order failure (the scheme still
converges for generic `f` — only the escape fails). **Reconciliation with the S³ chain:** 0164 is the
`f=0` linear heat (exact); 0166/0167 are linear variable-coef (order-2); 0168 crosses into NONLINEAR
for two structured classes (Cole-Hopf-integrable advection — exact-in-time; low-degree-polynomial
reaction — order-2), with generic mode-mixing nonlinearity the proven wall.

**No exactness gate for the reaction** (Strang splitting of two non-commuting flows is order-2, not
exact; an exactness assert would be a FALSE claim — matches the ADR-0166 inversion: order-p ⇒ slope
gate). Only Seam A is exact-in-time, gated by the semigroup property.

## 3. Build / run (single command, suckless)

```bash
# unit + reduction invariants (fast):
cargo test -p semiflow-core tt_nonlinear_spectral
# the S³ proof gate (fine-RK4 reference, slow-tests):
cargo test -p semiflow-core --features slow-tests g_s3_nonlinear
```

## 4. Out of scope (FAIL-LOUD — do NOT implement in this POC)

Generic / mode-mixing / transcendental nonlinearity (proven un-escapable by assert 4 — eff-rank
explodes; enforced unrepresentable by §1.1's `Reaction` enum), non-separable Cole-Hopf potential
(`φ` full rank — design §4), Carleman linearization (infinite lift + fixed truncation bias —
rejected, design §5), per-step Newton/Picard (reintroduces a linear solve — violates Theorem-6 R2),
nonlinear Schrödinger (no Cole-Hopf, gauge nonlinearity — separate milestone), higher-degree-reaction
calibration (`Quadratic` ships but its order/rank are only spot-checked), Magnus/higher-order
splitting, public API / bindings (FFI/PyO3/WASM), FFT perf (keep O(n²) direct DFT). The POC proves
*existence in principle* of a NONLINEAR order-2/exact curse-escaping scheme for two structured classes
(Cole-Hopf-integrable Burgers; low-degree-polynomial reaction-diffusion), plus its exact rank/cost
boundary at generic mode-mixing nonlinearity, nothing more.

## v9.2.0 — public surface (ADR-0169)

The POC evolver and its container types are promoted to the crate public API
behind the non-default `s3-poc` cargo feature. Three honesty layers apply:

1. **Type wall** — boundary-as-type wrapper constructor accepts only in-class
   arguments; out-of-class operators are unconstructible at the type level.
2. **Feature gate** — all tokens are `#[cfg(feature = "s3-poc")]`;
   a default build sees none of them.
3. **Rustdoc stanza** — every public type carries a normative
   `## Proven boundary` section citing the RELEASE-BLOCKING gate.

Previous line "Out of scope: public API / bindings" is amended: **public API is
NOW IN SCOPE** as the curated wrapper surface described above. FFI/PyO3/WASM
remain out of scope for v9.2.0.
