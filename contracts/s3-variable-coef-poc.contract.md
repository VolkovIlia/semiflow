# Contract — S³ variable-coefficient POC (`VarCoefSplitSpectral`)

**Scope:** proof-of-concept ONLY. ONE new evolver variant + ONE gate test. No public API
churn, no migration, no library-wide rollout. Suckless: ≤500 LoC/file, ≤50 LoC/fn, no new dep
(reuses `tt_drift_spectral.rs`/`tt_spectral.rs` 1-D DFT helpers + `SemiflowFloat`
`exp/sin/cos`; the const-coef spectral factor is ADR-0164's, the residual factor is pure
tridiagonal mat-vecs).
**Design:** `.dev-docs/specs/s3-variable-coef.md` (TRIZ + two-layer proof + boundary).
**Probes (truth):** `.dev-docs/specs/probe_s3_varcoef_final.py` (definitive two-layer: L1
exact, L2 order-2 slope 2.0000, rank-1 op, boundary floor), `probe_s3_varcoef.py` (additive
potential order/rank/cost), `probe_s3_varcoef_diffusion.py` (variable-diffusion crux + boundary).
**Builds on:** `crates/semiflow-core/src/tt_drift_spectral.rs` (ADR-0164: 1-D spectral const-coef
factor `apply_drift_spectral_axis`; the ONLY new ingredients are the per-axis ADDITIVE split +
the polynomial residual factor `P₂`).

---

## 1. New types / functions (Rust, `no_std` + `alloc`, generic over `F: SemiflowFloat`)

All live in a NEW module `crates/semiflow-core/src/tt_varcoef_spectral.rs` (keeps the POC
isolated; ≤500 LoC). Reuse the 1-D DFT helpers from `tt_spectral.rs`
(`dft_1d_real_to_cplx`, `idft_1d_cplx`) and ADR-0164's `axis_symbols_drift` pattern. The
evolver is a per-axis Strang of 1-D variable-coef factors; each 1-D factor is the order-2
Chernoff sandwich `P₂(τ/2)·k(τ)·P₂(τ/2)`.

### 1.1 Per-axis coefficient container (the construction-time fail-loud boundary)

```rust
/// Per-axis ADDITIVE-separable coefficients. Non-separable a(x,y) is UNREPRESENTABLE
/// by construction (the boundary is enforced by the type — design §4). `a_axis[j]` and
/// `b_axis[j]` are length-n grids of the diffusion / drift coefficient on axis j.
/// debug_assert: every a_axis[j][i] > 0 (parabolicity).
pub(crate) struct AxisCoef<F: SemiflowFloat> {
    pub a_axis: Vec<Vec<F>>,   // d × n, leading diffusion a_j(x_j) > 0
    pub b_axis: Vec<Vec<F>>,   // d × n, drift b_j(x_j)
    pub v_axis: Vec<Vec<F>>,   // d × n, reaction/potential v_j(x_j) (may be empty ⇒ 0)
}
```

### 1.2 Tridiagonal residual operator `Rⱼ = Lⱼ − a₀·Lap_fd` (built once, applied as mat-vec)

```rust
/// Build the 1-D divergence-form FD generator L_j for axis j, then the residual
/// R_j = L_j − a0_j·Lap_fd  (a0_j = mean(a_axis[j])). Returned as 3 diagonals
/// (sub, main, super) of the periodic tridiagonal operator (n entries each).
/// L_j[i] = (a_{i+1/2}(u_{i+1}−u_i) − a_{i−1/2}(u_i−u_{i−1}))/dx² + b_i(u_{i+1}−u_{i−1})/2dx + v_i·u_i.
/// NO solve, NO expm — pure coefficient assembly.
pub(crate) fn residual_tridiag<F: SemiflowFloat>(
    a: &[F], b: &[F], v: &[F], dx: F, a0: F,
) -> (Vec<F>, Vec<F>, Vec<F>);   // (sub, main, super), each length n, periodic
```

### 1.3 Polynomial residual factor `P₂(s) = I + s·R + s²/2·R²` (pure mat-vec, ZERO solve/expm)

```rust
/// Apply P₂(s)·u = u + s·(R u) + (s²/2)·R(R u) for periodic tridiagonal R (3 diagonals).
/// 2nd-order Chernoff factor for exp(s·R). PURE tridiagonal mat-vecs (2 of them):
/// NO lu_solve_inplace, NO dense_expm (Theorem-6 R2). Acts on ONE 1-D line.
pub(crate) fn p2_apply_tridiag<F: SemiflowFloat>(
    line: &mut [F], sub: &[F], main: &[F], sup: &[F], s: F,
);
```

### 1.4 1-D variable-coef Chernoff factor `exp(τ Lⱼ) ≈ P₂(τ/2)·k(τ)·P₂(τ/2)`

```rust
/// One 1-D variable-coef step on a single line. k(τ)=exp(τ·a0·Lap) via 1-D spectral
/// (reuse ADR-0164 `apply_drift_spectral_axis` with a=a0, b=0), sandwiched by P₂(τ/2).
/// Returns max|imag residue| from the spectral factor (< 1e-12 expected). Solver-free.
pub(crate) fn varcoef_axis_step<F: SemiflowFloat>(
    line: &mut [F], n: usize, dx: F,
    a: &[F], b: &[F], v: &[F], tau: F,
) -> F;   // max|imag residue|
```

### 1.5 d-D additive-split evolver (per-axis Strang; rank-1 per axis; solver-free)

```rust
/// Evolve `u0` (flat n^d real) by exp(τ·L), L = Σ_j L_j (additive-separable), via a
/// symmetric per-axis Strang: half-sweep j=0..d, then j=d-1..0, each axis a `varcoef_axis_step`.
/// Layer-1 (inter-axis) is EXACT (commuting); Layer-2 (intra-axis) is order-2.
/// NO lu_solve_inplace, NO dense_expm (Theorem-6 R2). Returns (evolved flat n^d, max|imag|).
pub(crate) fn varcoef_evolve<F: SemiflowFloat>(
    u0: &[F], n: usize, d: usize, dx: F, coef: &AxisCoef<F>, tau: F, nsteps: usize,
) -> (Vec<F>, F);
```

### 1.6 Reduction invariants (NORMATIVE — Gate sub-checks)

(a) **Const-coef reduction:** with `a_axis[j] = [a0; n]` constant, `b_axis[j] = [b0; n]`
constant, `v=0`, the per-axis factor with nsteps=1 MUST equal ADR-0164's
`apply_drift_spectral_axis(a0, b0, τ)` to ≤1e-12 (the residual `R=0` ⇒ `P₂=I` ⇒ pure spectral)
— proves the new code is a faithful SUPERSET of the const-coef path (anti-lesson #1).
(b) **Zero-residual identity:** with `a_axis[j]` constant, `residual_tridiag` returns the three
diagonals all `≈0` (≤1e-13) so `P₂(s)=I` exactly.

---

## 2. The ONE gate that proves S³ (`G_S3_VARCOEF_SPECTRAL`)

`crates/semiflow-core/tests/g_s3_varcoef_spectral.rs`, RELEASE-BLOCKING-class but gated
`#[cfg_attr(not(feature = "slow-tests"), ignore)]` (dense `expm` control). HARD asserts.

**Reference (independent, NO spectral/split code):** assemble the dense `n^d × n^d`
**divergence-form** variable-coef FD generator `L_h = Σ_j [D_div(a_j) + b_j·D1c + diag(v_j)]`
(per-axis tridiagonal lifted by Kronecker over ALL axes) and compute `u_ref = expm(T·L_h)·u₀`
via an in-test scaling-and-squaring Padé[6/6] (`expm_l`, COPIED from the ADR-0165 gate; NO
production reuse, NO new dep). Different ALGORITHM (LU-Padé) than the FFT+matvec scheme under
test — genuine independence (anti-lesson #3). The split evolver is ALSO re-implemented locally
in the gate (zero reuse of `tt_varcoef_spectral.rs`).

**Asserts (all HARD). Frozen params (§A pre-registration):** `n=7` (order/exactness asserts),
`n=5` (rank/cost asserts), `dx=L/n` with `L=2π`, `T=0.15`, `a0_j=mean`, additive
`a_axis[j][i]=0.5+0.2·cos(x_i+0.4·j)` (so `a∈[0.3,0.7]`, genuinely variable, parabolic),
`b_axis[j][i]=0.3·sin(x_i+0.2·j)`, `v=0`; nsteps sweep `{4,8,16,32,64,128}`.

1. **ORDER-2 CONVERGENCE (headline).** On `d=3` (dense `expm` tractable at `n=7`,
   `7³=343`), measure `rel_l2(varcoef_evolve(nsteps), u_ref)` over the nsteps sweep; compute
   the log-log OLS slope of rel_err vs `τ=T/nsteps` on the asymptotic tail (drop the 2
   coarsest). Pass ⇔ **slope ≤ −1.9** (order ≥ 1.9; probe gives 2.0000). [Probe
   `gate_layer2_order` + composite: errs `1.24e-7→1.21e-10`, slope 2.0000.]
   *Anti-vacuous:* the reference is an independent LU-Padé `expm` of the FULL divergence-form
   variable-coef generator — not a spectral self-comparison. A SLOPE gate (not a single-τ
   error) is the correct truncation-bearing gate for an order-p (not exact) scheme; it mirrors
   the repo's `G3_NS2D_var`/`G4_NS2D_aniso` self-convergence slope gates. Order ≥ 1.9 proves
   the scheme is Chernoff-tangent to the TRUE variable-coef semigroup (not a wrong operator).

2. **LAYER-1 EXACTNESS (inter-axis split, the curse-carrying layer).** Build the lifted
   per-axis generators `L_j ⊗ I` for `d=3`, assert `max‖[L_j,L_k]‖ ≤ 1e-12` over all pairs
   AND `‖expm(τΣL_j) − ∏_j expm(τL_j)‖_max ≤ 1e-12` at `τ=0.03`. [Probe `gate_layer1_exact`:
   `‖[L_j,L_k]‖=0`, split residue `3.3e-16`.] *Anti-vacuous:* proves the `n^d`→`d·n` reduction
   (the actual curse-escape) is EXACT, so the order-2 of assert 1 lives ENTIRELY in the 1-D
   sub-step, not in a lossy inter-axis split. This is the load-bearing honesty of the milestone:
   the escape is exact even though the step is order-2.

3. **RANK-1 TT OPERATOR per axis (curse-escape backbone).** Build the lifted axis-0 factor
   `E₀ ⊗ I^{⊗(d−1)}` (`E₀ = expm(τ·L₀)`, `n=4`, `d=3`), matricise across the axis0|rest
   operator cut, assert operator-TT-rank `= 1` at eps=1e-12. [Probe `gate_layer1_rank`: rank=1.]
   *Anti-vacuous:* proves each axis factor never entangles the axes it does not act on → TT-rank
   grows ≤1 per axis → bounded independent of `d`. This is the algebraic curse-escape statement
   (the variable-coef analogue of ADR-0165's bounded-bond assert).

4. **NEGATIVE BOUNDARY — non-separable `a(x)` wrong-operator floor (makes assert 1 non-vacuous).**
   Build a TRUE generator with a non-separable cross-diffusion term
   `0.25·cos(x)·sin(y)·∂²_x` added to the additive part (`d=2`, `n=7`), compute its dense
   `expm` reference, and run the per-axis additive split (which has NO cross factor) over the
   nsteps sweep `{16,32,64,128,256}`. Assert the rel_err **slope > −1.0** (does NOT converge at
   order ≥1) AND a **nonzero floor > 1e-4** (converges to the WRONG operator). [Probe
   `gate_boundary`: slope `0.0`, floor `9.53e-3`.] *Anti-vacuous:* without this, "additive
   converges at order 2" could be a property of ALL variable coef (it is not). Exhibiting a
   non-separable `a(x)` where the SAME scheme converges to the wrong operator proves (i) the
   boundary is real, (ii) the order gate can detect non-convergence, (iii) assert 1's order-2
   is a genuine property of the additive structure, not an artifact. This is the variable-coef
   analogue of ADR-0165's rank-2 contrast — the load-bearing honesty assert.

5. **LOAD-BEARING variable coefficient (makes asserts 1–3 non-vacuous).** With the same `d=3`
   regime, assert `‖u(a_var) − u(a_const)‖ / ‖u(a_const)‖ ≥ 0.02` where `a_const` is the
   per-axis MEAN (same `a0`, ZERO spatial variation) AND `max_i a_axis[0][i] − min_i a_axis[0][i]
   > 0.1` (the coefficient genuinely varies). [Probe `gate_load_bearing`: rel `0.18`, var-amp
   `3.19` — but with the milder gate-frozen params expect rel ≥ 0.02, var-amp ≥ 0.39.]
   *Anti-vacuous:* proves the variable part is present and load-bearing, so "order-2 + rank-1"
   means "variable coef carried at bounded cost," NOT "coefficient does nothing / a₀ in
   disguise" (anti-lesson: degenerate params).

6. **OPERATIONAL cost-scaling (absolute, tolerance-FREE escape).** The evolver produces a
   finite, real (`max|imag| < 1e-9`) state for the additive class at `d ∈ {8,10}` (`n=5`,
   nsteps=16) using only `O(d·n)` coefficient storage + per-axis tridiagonal mat-vecs + 1-D
   FFTs, whereas the dense-`expm` reference needs an `n^{2d}`-entry matrix
   (`1221 GB` at d=8, `763 PB` at d=10) that is un-formable. Pass ⇔ evolver runs (finite, real)
   at d=8,10 AND a static byte-count confirms `n^{2d}·8 > 1 TB`. [Probe `gate_cost`: runs,
   imag ~1e-16; dense `1221 GB`/`763 PB`.] *Anti-vacuous:* an absolute resource statement with
   no tolerance knob — the real curse the scheme escapes.

7. **REDUCTION + NO-SOLVER audit.** (a) With `a_axis[j]=[a0;n]`, `b_axis[j]=[b0;n]` constant,
   `v=0`, the per-axis factor (nsteps=1) equals ADR-0164's `apply_drift_spectral_axis(a0,b0,τ)`
   to ≤1e-12 (residual `R=0` ⇒ `P₂=I` ⇒ pure spectral; §1.6a) AND `residual_tridiag` returns
   diagonals all ≤1e-13. (b) A source-level grep asserts `tt_varcoef_spectral.rs` does NOT
   contain `lu_solve_inplace(` or `dense_expm(` call-sites (Theorem-6 R2; the dense `expm`
   lives ONLY in the reference). *Anti-vacuous:* faithful-superset proof + solver-free proof.

### 2.4 Why the gate satisfies the audit (NORMATIVE cross-reference)

The surviving audit counter-probes from the prior milestones are honoured: (a) **knife-edge** —
N/A here because the headline is a **τ-slope** gate (order is measured across a τ-sweep, not a
single SVD-tolerance rank read; the slope is a robust regression, immune to the SVD knife-edge
that plagued the earlier rank gates); (b) **degenerate params** — assert 5 proves the
coefficient genuinely varies (var-amp > 0.1) AND is load-bearing (rel ≥ 0.02), so the order-2
is not a const-coef result in disguise; (c) **wrong-operator masking** — assert 4 (the
non-separable boundary contrast) proves the same scheme FAILS (wrong-operator floor) off the
additive class, so assert 1's order-2 is a genuine property of additivity, not of the scheme
trivially matching any reference. **The boundary (design §4) is explicitly NOT overclaimed:**
non-separable `a(x)` escape is FALSE and is *proven* false by assert 4 (wrong-operator floor)
and by the multiplier-rank explosion (probe `gate_rank`: 5→22→24); the positive claim is scoped
to additive-separable coefficients and **enforced by construction** (§1.1 — the API cannot
express `a(x,y)`). **Reconciliation with v9.1/0164/0165:** const-coef is the `R=0` case of this
scheme (`P₂=I` ⇒ pure ADR-0164 spectral, exact); variable-coef adds the order-2 residual factor
and delimits non-separability as the boundary.

**No exactness gate** (variable LEADING diffusion is provably NOT a single Fourier multiplier —
the only truncation-free factor is the const-coef `k`; the variable residual is honestly
order-2). An exactness gate would be a FALSE claim; the slope gate is the correct gate. This
INVERTS the ADR-0164/0165 decision (those were exact ⇒ no slope gate; this is order-p ⇒ slope
gate, no exactness gate) — and that inversion is itself the honest signature that variable-coef
is a strictly harder, strictly-approximate regime.

## 3. Build / run (single command, suckless)

```bash
# unit + reduction invariants (fast):
cargo test -p semiflow-core tt_varcoef_spectral
# the S³ proof gate (dense expm control, slow-tests):
cargo test -p semiflow-core --features slow-tests g_s3_varcoef_spectral
```

## 4. Out of scope (FAIL-LOUD — do NOT implement in this POC)

Non-separable / cross-dependent `a(x)` (off-axis coupling — proven un-escapable by assert 4 +
the multiplier-rank explosion; enforced unrepresentable by §1.1), Magnus-K4 intra-axis (order-4
but commutator build cost unjustified for a POC — deferred to a production wave),
time-dependent coefficients `a(x,t)` (Howland lift, separate milestone), nonlinear, public API
/ bindings (FFI/PyO3/WASM), FFT perf (keep O(n²) direct DFT), variable-coef with VARIABLE
leading symbol per cell folded as a potential (rejected — wrong operator, design §5). These are
explicitly deferred; the POC proves *existence in principle* of a variable-coef order-2
curse-escaping class plus its exact boundary, nothing more.

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
