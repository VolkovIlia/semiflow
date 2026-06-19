# Contract — S³ non-separable variable-coefficient POC (`NonSepVarCoefSpectral`)

**Scope:** proof-of-concept ONLY. ONE new evolver variant + ONE gate test. No public API churn,
no migration, no library-wide rollout. Suckless: ≤500 LoC/file, ≤50 LoC/fn, no new dep (reuses
`tt_drift_spectral.rs` 1-D DFT helpers + `SemiflowFloat` `exp/sin/cos`; the const-leading spectral
factor is ADR-0164's, the residual factor is pure low-rank TT mat-vecs).
**Design:** `.dev-docs/specs/s3-nonsep-varcoef.md` (TRIZ + CP-rank mechanism + boundary).
**Probe (truth):** `.dev-docs/specs/probe_s3_nonsep_varcoef.py` (make-or-break: 0166 boundary
`cos(x)sin(y)` slope +2.0000 all 3 roles; ablation reproduces 0166 floor 9.548e-3; op-rank ==
CP-rank flat in d; generic → full op-rank; reduction to 0166 = 2.22e-16).
**Builds on:** `crates/semiflow-core/src/tt_drift_spectral.rs` (ADR-0164: const-coef spectral
factor) and the ADR-0166 split idea — the ONLY new ingredient is the FULL low-CP-rank residual
operator `R` (vs 0166's mean-frozen per-axis residual).

---

## 1. New types / functions (Rust, `no_std` + `alloc`, generic over `F: SemiflowFloat`)

All live in a NEW module `crates/semiflow-core/src/tt_nonsep_varcoef.rs` (keeps the POC isolated;
≤500 LoC). Reuse the d-D spectral const-coef factor (ADR-0164 `apply_drift_spectral_axis`/FFT
helpers) for `k(τ)`. The evolver is a single symmetric Chernoff sandwich `P₂(τ/2)·k(τ)·P₂(τ/2)`;
the residual `R` is a sum of `m` rank-1 TT operators applied as TT mat-vecs.

### 1.1 CP-term coefficient container (the construction-time fail-loud boundary)

```rust
/// One CP-term of a low-CP-rank coefficient: c_r(x) = prod_j factor[j](x_j).
/// `factor[j]` is a length-n grid of the per-axis factor on axis j.
pub(crate) struct CpTerm<F: SemiflowFloat> {
    pub factor: Vec<Vec<F>>,   // d × n  (factor[j] = c_{r,j}(x_j))
}

/// Low-CP-rank coefficient field c(x) = c0 + sum_{r<m} prod_j factor_r[j](x_j), attached to
/// ONE differential `core` (diffusion | drift | potential). m = terms.len() is FIXED in d.
/// Non-separable yet low-rank (m=1 => f(x)g(y)...). Generic full-rank a(x) is UNREPRESENTABLE
/// by construction (the API cannot express an arbitrary n^d field — design §4). debug_assert:
/// the reconstructed c(x) > 0 on a probe set for the diffusion role (parabolicity).
pub(crate) struct CpCoef<F: SemiflowFloat> {
    pub c0: F,                       // constant leading part (a0 for diffusion; 0 for drift/pot)
    pub terms: Vec<CpTerm<F>>,       // m CP-terms (m fixed in d)
    pub role: CoefRole,              // Diffusion | Drift | Potential
}

pub(crate) enum CoefRole { Diffusion, Drift, Potential }
```

### 1.2 Per-axis const-coef tridiagonal `core` builders (D2 / D1c / I)

```rust
/// Build the 3 diagonals (sub, main, super) of the periodic const-coef 1-D core on axis j:
///   Diffusion -> FD Laplacian d2/dx2 ;  Drift -> centred d/dx ;  Potential -> identity.
/// NO solve, NO expm — pure coefficient assembly. (length n each, periodic.)
pub(crate) fn core_tridiag<F: SemiflowFloat>(role: CoefRole, dx: F, n: usize)
    -> (Vec<F>, Vec<F>, Vec<F>);
```

### 1.3 Rank-`m` TT residual application `R u` (pure TT mat-vec, ZERO solve/expm)

```rust
/// Apply R·u where R = sum_{r<m} [ diag(factor_r[j0]) · core_1d ]_{axis j0} (x) prod_{j!=j0} diag(factor_r[j])
/// for the coefficient's `core` axis j0. `u` is a flat n^d state; the application is m rank-1
/// TT contractions (per-axis diagonal scalings along non-core axes + a 1-D tridiagonal mat-vec
/// along the core axis). Cost O(m·d·n^d-as-stride) — NO n^{2d} matrix, NO LU, NO dense_expm.
pub(crate) fn apply_residual<F: SemiflowFloat>(
    u: &[F], out: &mut [F], n: usize, d: usize, dx: F, coef: &CpCoef<F>,
);
```

### 1.4 Polynomial residual factor `P₂(s)·u = u + s·R u + (s²/2)·R(R u)` (2 TT mat-vecs)

```rust
/// 2nd-order Chernoff factor for exp(s·R). PURE TT mat-vecs (2 applications of apply_residual):
/// NO lu_solve_inplace, NO dense_expm (Theorem-6 R2). `R` aggregates ALL CP-coefficient terms.
pub(crate) fn p2_apply<F: SemiflowFloat>(
    u: &mut [F], scratch: &mut [F], n: usize, d: usize, dx: F, coefs: &[CpCoef<F>], s: F,
);
```

### 1.5 const-leading spectral factor `k(τ) = exp(τ·a₀·Σⱼ Lap_j)` (FFT-diagonal, NO solve)

```rust
/// exp(tau·a0·sum_j Lap_j) via d-D FFT multiplier (reuse ADR-0164 spectral machinery, b=0).
/// Returns max|imag residue| (< 1e-12 expected). Solver-free. a0 = leading-diffusion mean.
pub(crate) fn k_spectral<F: SemiflowFloat>(
    u: &mut [F], n: usize, d: usize, dx: F, a0: F, tau: F,
) -> F;
```

### 1.6 d-D non-separable evolver (single Chernoff sandwich; rank-m TT; solver-free)

```rust
/// Evolve `u0` (flat n^d real) by exp(τ·L), L = a0·Lap + sum_role diag(c_role(x))·core_role,
/// via the symmetric Chernoff sandwich  P₂(τ/2)·k(τ)·P₂(τ/2)  (R = full non-separable residual,
/// a rank-m TT operator carrying ALL cross structure). Order-2 in τ; solver-free.
/// NO lu_solve_inplace, NO dense_expm (Theorem-6 R2). Returns (evolved flat n^d, max|imag|).
pub(crate) fn nonsep_evolve<F: SemiflowFloat>(
    u0: &[F], n: usize, d: usize, dx: F, a0: F, coefs: &[CpCoef<F>], tau: F, nsteps: usize,
) -> (Vec<F>, F);
```

### 1.7 Reduction invariants (NORMATIVE — Gate sub-checks)

(a) **Const-coef reduction:** with every CP-term's factors constant (and only a Potential/Drift
role, c0 carrying the diffusion), the residual `R=0` ⇒ `P₂=I` ⇒ the step equals ADR-0164's
const-coef spectral to ≤1e-12 (0 ULP target). Proves faithful superset of the const-coef path.
(b) **Additive reduction to 0166:** for an ADDITIVE leading diffusion `a(x)=a0+Σⱼαⱼ(xⱼ)` encoded
as `d` single-axis CP-terms, the assembled residual operator `R` MUST equal the ADR-0166 per-axis
residual `Σⱼ(Lⱼ−a0·Lapⱼ)` to ≤1e-12 (probe `gate_reduction`: `2.22e-16`). Proves 0167 is a faithful
superset of the 0166 mechanism (anti-lesson #1).

---

## 2. The ONE gate that proves S³ (`G_S3_NONSEP_VARCOEF`)

`crates/semiflow-core/tests/g_s3_nonsep_varcoef.rs`, RELEASE-BLOCKING-class but gated
`#[cfg_attr(not(feature = "slow-tests"), ignore)]` (dense `expm` control). HARD asserts.

**Reference (independent, NO spectral/split code):** assemble the dense `n^d × n^d` centred-FD
generator with the FULL non-separable coefficient `L_h = a0·Lap_h + diag(a(x))·core_h` (per-axis
const-coef tridiagonals lifted by Kronecker; `diag(a(x))` is the full n^d coefficient vector) and
compute `u_ref = expm(T·L_h)·u₀` via an in-test scaling-and-squaring Padé[6/6] (`expm_l`, COPIED
from the ADR-0165 gate; NO production reuse, NO new dep). Different ALGORITHM (LU-Padé) than the
FFT+TT-matvec scheme under test — genuine independence (anti-lesson #3). The evolver is ALSO
re-implemented locally in the gate (zero reuse of `tt_nonsep_varcoef.rs`).

**Frozen params (§A pre-registration):** `n=7` (order asserts), `n=5` (rank/cost asserts),
`dx=2π/n`, `T=0.10`, `a0=0.5`, **non-separable rank-1 coefficient `a(x,y)=0.25·cos(x)·sin(y)`**
(the EXACT ADR-0166 boundary case), IC `u₀=⊗(cos+0.3)`; nsteps sweep `{4,8,16,32,64,128}`
(order), `{16,32,64,128,256}` (ablation).

### Asserts (all HARD)

1. **MAKE-OR-BREAK — the 0166 boundary now CONVERGES (headline).** On `d=2`, role =
   **DIFFUSION** (`diag(a)·∂²ₓ`, the hardest role, the literal 0166 boundary), measure
   `rel_l2(nonsep_evolve(nsteps), u_ref)` over the nsteps sweep; log-log OLS slope of rel_err vs
   `τ=T/nsteps` on the asymptotic tail (drop the 2 coarsest). Pass ⇔ **slope ≤ −1.9** (order ≥ 1.9;
   probe +2.0000) AND **errs are in a real regime** (coarsest err > 1e-7, finest < 1e-7, all
   ≥ 100× float floor). [Probe: errs `1.06e-6 → 1.03e-9`, slope `+2.0000`.]
   *Anti-vacuous:* this is the EXACT case ADR-0166 sends to the wrong operator (slope 0, floor
   9.53e-3). Converging it at order 2 vs an INDEPENDENT dense Padé `expm` of the FULL non-separable
   generator (not a self-comparison) is the make-or-break: it proves the scheme is Chernoff-tangent
   to the TRUE non-separable semigroup. A SLOPE gate in a real error regime (not a single τ, not in
   the noise) is the correct truncation-bearing gate.

2. **ALL-ROLES ORDER-2.** Repeat assert 1 for role = **POTENTIAL** (`diag(a)·I`) and role =
   **DRIFT** (`diag(a)·∂ₓ`), same non-separable `a(x,y)`. Pass ⇔ each slope ≤ −1.9. [Probe:
   potential `+2.0000` floor `1.18e-9`; drift `+2.0000` floor `9.73e-10`.]
   *Anti-vacuous:* proves the fix is not role-specific — the full-R mechanism captures non-
   separable cross-structure for every coefficient role, including variable LEADING diffusion.

3. **LOAD-BEARING ABLATION (the central honesty assert).** With the SAME `d=2` diffusion case,
   run TWO evolvers over nsteps `{16,32,64,128,256}`: (i) FULL-R (the scheme) and (ii) MEAN-FROZEN-R
   (replace `diag(a(x,y))` by its scalar grid-mean `ā·∂²ₓ`, i.e. drop the cross structure = the
   ADR-0166 reduction). Assert FULL-R slope ≤ −1.9 AND floor < 1e-4; assert MEAN-FROZEN slope > −1.0
   AND floor > 1e-4. [Probe `gate_ablation`: FULL `+2.0000`/`2.58e-10`; MEAN-FROZEN `−0.0000`/
   `9.548e-3` — reproduces the recorded 0166 boundary `9.53e-3` to 3 digits.]
   *Anti-vacuous:* this is the load-bearing proof that the cross term in `R` does the work —
   ablating it collapses the SAME scheme back to the proven 0166 wrong-operator boundary. Without
   this, "non-separable converges" could be a property of the reference matching anything. The
   ablation is the variable-coef analogue of ADR-0165's rank-2 contrast.

4. **OPERATOR-TT-RANK = CP-RANK (curse-escape backbone; REAL max-over-all-bonds).** Build the
   residual operator `R = diag(a(x))·∂²ₓ₀` for coefficients of CP-rank `1, 2, 3` and GENERIC
   (full-rank random), `n=5`, and read the **max-over-ALL-bonds** operator-TT-rank (NO half-cut —
   the M2 vacuity). Assert: CP-rank-1 → op-rank 1; CP-rank-2 → 2; CP-rank-3 → 3 (each **flat over
   d=2,3,4**); GENERIC → op-rank = `n` = 5 (full). [Probe `gate_rank`: `[1,1,1]`, `[2,2,2]`,
   `[3,3,3]`, generic `[5,5,5]`.]
   *Anti-vacuous:* proves the residual is a rank-`m` TT operator (mat-vec cost `O(m·d·n)`, flat in
   d) for low-CP-rank `a`, and FULL (curse cost `O(n^d)`) for generic `a`. The max-over-all-bonds
   metric (not a single non-dominant half-cut) is the honest rank read (M2 anti-lesson). This is
   the algebraic curse-escape statement and the boundary in ONE assert.

5. **NEGATIVE BOUNDARY — generic full-rank `a(x)` forfeits the escape.** With a GENERIC random
   `a(x)` (full-rank tensor, `n=5`, d=2,3,4), assert the residual operator-TT-rank = `n` (full,
   not bounded in CP-rank) — so `R u` costs `O(n^d)` and the escape is forfeit. [Probe
   `gate_generic_boundary`: op-rank `5 = n` at d=2,3,4.] Assert (documentation) that the order-2
   tangency of `P₂·k·P₂` STILL holds for generic `R` (no order failure) — the boundary is on the
   COST/RANK axis, not the order axis.
   *Anti-vacuous:* makes the escape claim non-vacuous by exhibiting the case where the SAME order-2
   scheme cannot escape (full residual rank). This is the precise, honest boundary distinct from
   0166's order boundary: 0167 fixes 0166's ORDER boundary for low-CP-rank `a`, and its OWN boundary
   is the curse-COST of full-rank `a`.

6. **OPERATIONAL cost-scaling (absolute, tolerance-FREE escape).** The evolver produces a finite,
   real (`max|imag| < 1e-9`) state for the rank-1 non-separable class at `d ∈ {8,10}` (`n=5`,
   nsteps=16) using only `O(m·d·n)` coefficient storage + rank-m TT mat-vecs + d-D FFTs, whereas the
   dense-`expm` reference needs an `n^{2d}`-entry matrix (`1221 GB` at d=8, `763 PB` at d=10) that is
   un-formable. Pass ⇔ evolver runs (finite, real) at d=8,10 AND a static byte-count confirms
   `n^{2d}·8 > 1 TB`.
   *Anti-vacuous:* an absolute resource statement with no tolerance knob — the real curse the scheme
   escapes for the non-separable low-CP-rank class.

7. **REDUCTION + NO-SOLVER audit.** (a) **Const-coef:** with all CP-term factors constant (R=0 ⇒
   P₂=I), the step equals ADR-0164's `apply_drift_spectral_axis(a0,0,τ)` to ≤1e-12. (b) **Additive
   → 0166:** an additive leading diffusion `a0+Σⱼαⱼ(xⱼ)` encoded as `d` single-axis CP-terms yields
   a residual operator equal to the ADR-0166 per-axis residual `Σⱼ(Lⱼ−a0·Lapⱼ)` to ≤1e-12 [probe
   `gate_reduction`: `2.22e-16`]. (c) A source-level grep asserts `tt_nonsep_varcoef.rs` does NOT
   contain `lu_solve_inplace(` or `dense_expm(` call-sites (Theorem-6 R2; dense `expm` lives ONLY in
   the reference).
   *Anti-vacuous:* faithful-superset proof (of BOTH const-coef AND the 0166 additive mechanism) +
   solver-free proof.

### 2.4 Why the gate satisfies the audit (NORMATIVE cross-reference)

The surviving audit counter-probes from prior milestones are honoured: (a) **knife-edge** —
the headline (asserts 1–3) is a **τ-slope** gate (order across a τ-sweep, not a single SVD-tolerance
rank read), a robust regression immune to the SVD knife-edge; the rank assert (4) uses **max-over-ALL-
bonds** TT-SVD (NOT the M2 half-cut) and reads the rank DIFFERENCE structure (CP-rank → op-rank) that
is knife-edge-robust by construction; (b) **degenerate / rigged params** — the coefficient is the
LITERAL recorded 0166 boundary `0.25·cos(x)·sin(y)` (genuinely non-separable, not a relabel), and the
ABLATION (assert 3) proves the cross term is load-bearing by collapsing to the 0166 floor when removed;
(c) **wrong-operator masking** — the reference is an INDEPENDENT LU-Padé `expm` of the FULL non-
separable generator (different algorithm, zero shared code), measured in a real error regime, so order-2
cannot be an artifact of self-comparison; (d) **honest boundary** — assert 5 proves the new boundary
(generic full-rank `a(x)` forfeits the escape on the COST/RANK axis) is real and DISTINCT from 0166's
order boundary. **Reconciliation:** const-coef is the `R=0` case (ADR-0164, exact); additive is the
`R = Σⱼ` per-axis case (ADR-0166, `‖R−R₀₁₆₆‖=2.2e-16`); non-separable low-CP-rank is the `R = Σ_{r≤m} ⊗ⱼ`
case (this POC); generic full-rank is the boundary.

**No exactness gate** (variable leading diffusion is provably NOT a single Fourier multiplier — honestly
order-2; an exactness gate would be a FALSE claim). This matches the ADR-0166 inversion (order-p ⇒ slope
gate, no exactness gate) — the honest signature of the approximate variable-coef regime.

## 3. Build / run (single command, suckless)

```bash
# unit + reduction invariants (fast):
cargo test -p semiflow-core tt_nonsep_varcoef
# the S³ proof gate (dense expm control, slow-tests):
cargo test -p semiflow-core --features slow-tests g_s3_nonsep_varcoef
```

## 4. Out of scope (FAIL-LOUD — do NOT implement in this POC)

Generic full-rank `a(x)` (proven un-escapable by assert 5 — residual op-rank = full; enforced
unrepresentable by §1.1), rank-`m` truncation of a generic `a(x)` (uncontrollable bias ⇒ wrong-
operator floor — design §5), Magnus-K4 residual (order-4 but commutator rank cost unjustified for a
POC — deferred), time-dependent coefficients `a(x,t)` (Howland lift, separate milestone), nonlinear,
public API / bindings (FFI/PyO3/WASM), FFT perf (keep O(n²) direct DFT). The POC proves *existence in
principle* of a NON-separable (low-CP-rank) variable-coef order-2 curse-escaping class that fixes the
ADR-0166 boundary, plus its exact new (cost/rank) boundary, nothing more.

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
