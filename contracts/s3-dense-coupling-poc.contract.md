# Contract — S³ dense / non-adjacent (all-pairs) coupling POC (`DenseCouplingSpectral`)

**Scope:** proof-of-concept ONLY. ONE new evolver variant + ONE gate test. No public API
churn, no migration, no library-wide rollout. Suckless: ≤500 LoC/file, ≤50 LoC/fn, no new dep
(reuses `tt_spectral.rs` DFT + `SemiflowFloat` `exp/sin/cos`; the symbol machinery is
ADR-0164's, applied to the full coupling matrix).
**Design:** `.dev-docs/specs/s3-dense-coupling.md` (TRIZ + rank-preservation proof + boundary).
**Probes (truth):** `.dev-docs/specs/probe_dense_rank.py` (make-or-break d-sweep),
`probe_offdiag_rank.py` (off-block-rank governs bond rank), `probe_rank1_dense.py` (positive
class + negative boundary), `probe_triz_confirm.py` (single-generator analytic backing).
**Builds on:** `crates/semiflow-core/src/tt_drift_spectral.rs` (ADR-0164 complex symbol; the
ONLY new ingredient is the *full* (all-pairs) coupling sum in the symbol + a rank-1-dense
matrix builder).

---

## 1. New types / functions (Rust, `no_std` + `alloc`, generic over `F: SemiflowFloat`)

All live in a NEW module `crates/semiflow-core/src/tt_dense_coupling.rs` (keeps the POC
isolated; ≤500 LoC). Reuse the existing `pub(crate)` DFT helpers from `tt_spectral.rs`
(`dft_1d_real_to_cplx`, `dft_1d_cplx`, `idft_1d_cplx`) and ADR-0164's axis-symbol pattern.
The evolver is a d-D full-symbol apply (the d-D generalization of ADR-0164's `spectral_evolve`
gate helper, promoted to production with an arbitrary symmetric `D` instead of adjacent-only ρ).

### 1.1 Coupling-matrix structure builder (the ONE new ingredient)

```rust
/// Build a rank-1-DENSE symmetric coupling matrix `D = diag(a) + λ·g·gᵀ` (row-major d×d).
/// Every off-diagonal entry `D[i,j] = λ·g_i·g_j ≠ 0` (genuinely dense, all pairs coupled),
/// yet the off-diagonal block has numerical rank 1 across every cut ⇒ bounded TT-rank.
/// `a` is the per-axis self-diffusion (diagonal); `g` is the dense coupling eigenvector.
pub(crate) fn rank1_dense_matrix<F: SemiflowFloat>(
    a: &[F],        // length d, diagonal self-diffusion (D[j,j] = a[j] + λ·g[j]²)
    g: &[F],        // length d, coupling vector (all entries non-zero for full density)
    lambda: F,      // coupling strength
) -> Vec<F>;        // length d·d, row-major symmetric

/// Build a rank-m-DENSE symmetric coupling matrix `D = diag(a) + Σ_{a<m} λ_a·g_a·g_aᵀ`.
/// Used ONLY by the negative-boundary contrast (m=2) — proves the gate is non-vacuous.
pub(crate) fn rankm_dense_matrix<F: SemiflowFloat>(
    a: &[F],            // length d, diagonal
    factors: &[&[F]],   // m vectors of length d
    lambdas: &[F],      // m strengths
) -> Vec<F>;            // length d·d, row-major symmetric
```

### 1.2 Full-symbol d-D builder (the ALL-PAIRS generalization of ADR-0164)

```rust
/// Build the COMPLEX d-D exp-symbol for `exp(τ·σ(k))`, `σ(k) = −kᵀDk + i·bᵀk`,
/// over the FULL n^d Fourier grid. Returns interleaved (re,im) of length 2·n^d.
///
/// σ(m₀..m_{d-1}) = Σ_j D[j,j]·σ_D2(m_j)                       (diagonal diffusion, RE)
///                − Σ_{j<k} 2·D[j,k]·σ_D1r(m_j)·σ_D1r(m_k)     (ALL-PAIRS cross, RE)
///                + i·Σ_j b[j]·σ_D1r(m_j)                       (drift, IM; ADR-0164)
/// expsym = exp(τ·Re)·(cos(τ·Im) + i·sin(τ·Im))  [conjugate-even ⇒ real output].
///
/// Adjacency-agnostic: the pair sum runs over EVERY (j,k), j<k — this is precisely
/// the non-adjacent / dense coupling the v9.1 solver fail-loud rejected.
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_expsym_nd<F: SemiflowFloat>(
    n: usize, d: usize, dx: F,
    d_mat: &[F],    // d·d symmetric coupling matrix (row-major; from §1.1)
    b: &[F],        // length d drift vector (b=0 ⇒ pure dense diffusion)
    tau: F,
) -> Vec<F>;        // length 2·n^d interleaved (re,im)
```

### 1.3 Dense-coupling d-D evolver (full-symbol apply, solver-free)

```rust
/// Evolve `u0` (flat n^d real) by `exp(τ·L_D)` via the full d-D complex spectral symbol.
/// fft_d → elementwise COMPLEX multiply by `dense_expsym_nd` → ifft_d → take real.
/// NO lu_solve_inplace, NO dense_expm, NO triangular solve (Theorem-6 R2).
/// Output is real; imaginary residue MUST be < 1e-10 (asserted by gate).
#[allow(clippy::too_many_arguments)]
pub(crate) fn dense_coupling_evolve<F: SemiflowFloat>(
    u0: &[F], n: usize, d: usize, dx: F,
    d_mat: &[F], b: &[F], tau: F,
) -> (Vec<F>, F);   // (evolved flat n^d real state, max |imag residue|)
```

### 1.4 Reduction invariants (NORMATIVE — Gate sub-checks)

(a) **Adjacent reduction:** with `D` tridiagonal (only `D[j,j+1]≠0`), `dense_expsym_nd` MUST
equal (re-part) the v9.1 / ADR-0164 adjacent-pair symbol to 0 ULP — proves the new code is a
faithful SUPERSET of the adjacent path, not a relabel (anti-lesson #1).
(b) **Diagonal reduction:** with `D = diag(a)`, `b = 0`, the symbol MUST equal the separable
diffusion symbol to 0 ULP.
(c) **Drift reduction:** with `D` adjacent + `b ≠ 0`, MUST equal ADR-0164's `spectral_evolve`
bit-for-bit (0 ULP) — chains this POC onto the proven drift milestone.

---

## 2. The ONE gate that proves S³ (`G_S3_DENSE_COUPLING`)

`crates/semiflow-core/tests/g_s3_dense_coupling.rs`, RELEASE-BLOCKING-class but gated
`#[cfg_attr(not(feature = "slow-tests"), ignore)]` (dense `expm` control). HARD asserts.

**Reference (independent, NO spectral code):** assemble the dense centred-FD generator
`L_h = Σ_j (D[j,j]·D2_j/dx² + b_j·D1c_j/(2dx)) + Σ_{j<k} 2·D[j,k]·(D1c_j/2dx)⊗(D1c_k/2dx)`
over ALL pairs `(j,k)` (extend ADR-0164's gate `build_gen` from adjacent-only to all-pairs)
and compute `u_ref = expm(T·L_h)·u₀` via the in-test scaling-and-squaring Padé[6/6] helper
(`expm_l`, copied from the 0164 gate; NO production-code reuse, NO new dep). This is a
different algorithm (LU-Padé) than the FFT-diagonal scheme under test — genuine independence
(anti-lesson #3). The spectral apply is ALSO re-implemented locally in the gate from the
symbol formula (zero reuse of `tt_dense_coupling.rs`), so the gate is fully independent.

**Asserts (all HARD). Frozen params (§A pre-registration):** `n=5`, `dx=1/n`, `τ=0.02`,
`b_j = 0.6 + 0.1·j ≠ 0`; rank-1-dense `g = cos(lin(0.3,1.4,d))·0.6`, `λ=0.25`, `a_j=0.5`.

1. **EXACTNESS (headline):** `rel_l2(dense_coupling_evolve, u_ref) ≤ 1e-12` for `d∈{3,4}`,
   rank-1-dense `D` with `b_j≠0` on every axis. [Probe: symbol is exact for any `D`; ~1e-14.]
   *Anti-vacuous:* the reference is an independent LU-Padé `expm` of the FULL all-pairs FD
   generator — not a spectral self-comparison. Exactness is the correctness backbone that
   licenses the rank read-outs at d≥5 (where `expm` is too costly).

2. **DENSITY (genuinely all-pairs, non-degenerate):** the test `D` has ALL `d(d−1)`
   off-diagonal entries `|D[i,j]| > 1e-14` (e.g. 56 nonzeros at d=8). Asserted before the
   sweep. *Anti-vacuous:* proves we test a *dense* matrix, not a sparse/adjacent one in
   disguise (the whole point is non-adjacent coupling).

3. **DRIFT present + REALITY:** `b_j·τ/dx` non-integer (frac > 0.05) at every level AND
   `max|imag residue| < 1e-10`. *Anti-vacuous:* genuine sub-grid advection (anti-lesson #2) +
   catches conjugate-even regressions.

4. **CURSE-ESCAPE — Δrank-PRESERVATION sweep (Gate-1; honest, tolerance-robust).**
   On the **operator symbol** `exp(τσ(k))` (NOT an evolved state — see anti-vacuous (b)),
   measure the **max-over-all-bonds** TT-SVD rank (full left-to-right SVD sweep, mirror of
   Python `tt_ranks → max(rks)`), for rank-1-dense `D` and for the diagonal-only baseline at
   the SAME SVD tolerance. Define
   `Δrank(eps,d) := max_bond_rank_{rank1-dense}(eps) − max_bond_rank_{diag-only}(eps)`.
   Sweep `d ∈ {3,4,5,6,7,8}` (the window MUST reach d=8 — saturation is only visible at
   d≥7) and `eps ∈ {1e-8,1e-10,1e-12,1e-14}`. Pass ⇔ **BOTH** of:
   - **(boundedness)** `Δrank(eps,d) ≤ Δ_CAP` for every `eps` and every `d ∈ {3..8}`,
     `Δ_CAP = 7` (frozen, EXACT probe maximum — no `+1` slack); AND
   - **(tail-saturation)** `Δrank(eps, d=8) ≤ Δrank(eps, d=7)` for every `eps` (the plateau
     is flat at the tail; both equal 7 at eps=1e-12). NO comparison at small d, where the
     ladder is still filling and Δrank legitimately grows 4→5→6→7.
   *Anti-vacuous (metric-correctness — the load-bearing honesty requirement):* (a) **difference
   at fixed eps on the FULL max-over-all-bonds rank** cancels the SVD knife-edge — the absolute
   rank wanders by ≤1 per eps-decade (probe: 5,6,7,8,9 at d=8), but the difference over the
   exact diagonal baseline is stable. The half-cut single-bond estimator is FORBIDDEN: it
   reads a non-dominant bond (left=d/2) that plateaus early while the true max-bond is still
   climbing, masking the real Δrank curve. (b) This measures an ALGEBRAIC operator property
   (ρ=1 off-block ⇒ single bilinear generator `f(left)·g(right)` ⇒ each of `{f^m},{g^m}`
   spans ≤ n grid-powers ⇒ bond ≤ f(ρ=1,n)=8 independent of d, design §1) — it is NOT a
   self-comparison and NOT an evolved-state arm; correctness is assert 1's independent-`expm`
   job. The honest saturation value Δ=7 = (operator bond ceiling 8) − (diagonal baseline 1)
   ties directly to the predicted `f(ρ=1, n=5)`. [Probe `probe_rank1_dense.py`: max-bond
   Δrank = **4,5,6,7,7,7** over d=3..8 at eps=1e-12 — grows through d≤6, SATURATES at d≥7.]

5. **NEGATIVE BOUNDARY (Gate-2; the contrast that makes Gate-1 non-vacuous).** A **rank-2**-
   dense `D = diag(a) + Σ_{a<2} λ_a g_a g_aᵀ` (off-block rank 2), SAME `n,τ,b`, MUST **exceed**
   the rank-1-dense bond rank by a growing margin: assert
   `rank_{rank2-dense}(d=8, eps=1e-12) ≥ 2 · rank_{rank1-dense}(d=8, eps=1e-12)` AND
   `rank_{rank2-dense}(d=8) > rank_{rank2-dense}(d=4)` (it GROWS with d). [Probe: rank-2-dense
   = 23,41,61 at d=4,6,8; rank-1-dense = 6,8,8 — ratio ≫ 2 and rank-2 grows.]
   *Anti-vacuous:* without this, "rank-1-dense stays bounded" could be a property of *all*
   dense `D` (it is not) or of the estimator saturating at the cap. Exhibiting a dense `D`
   that the SAME estimator reports as EXPLODING proves (i) the boundary is real, (ii) the
   estimator can see growth, (iii) Gate-1's boundedness is a genuine property of the rank-1
   structure, not an artifact. This is the load-bearing honesty assert of the milestone.

6. **OPERATIONAL cost-scaling (Gate-3; absolute, tolerance-FREE escape).** The evolver
   produces a **finite, real** (`max|imag| < 1e-10`) state for rank-1-dense `D` at
   `d ∈ {8,10}` (n=5) using only `O(d·n + d²)` symbol/matrix storage and an FFT-diagonal
   apply, whereas the dense-`expm` reference at those `d` needs an `n^{2d}`-entry matrix
   (`>1 TB` at d=8) that is un-formable. Pass ⇔ evolver runs (finite, real) at d=8,10 AND a
   static byte-count confirms `n^{2d}·8 > 1 TB`. *Anti-vacuous:* an absolute resource
   statement with no tolerance knob. [Probe: runs at d=8,10, imag~1e-15; dense gen 1.2 TB.]

7. **ANTI-TRIVIALITY:** a rank-1 separable IC evolves to TT rank > 1 under the dense coupling
   (excludes the separability no-op failure mode). *Anti-vacuous:* proves the coupling
   actually entangles axes. [Probe: 1→≥5.]

8. **LOAD-BEARING coupling (makes 4 non-vacuous).**
   `‖U(rank1-dense) − U(diag-only)‖ / ‖U(diag-only)‖ ≥ 0.05` at the gate regime (`d=4`).
   *Anti-vacuous:* proves the off-diagonal coupling is present and load-bearing, so "Δrank
   bounded" (assert 4) means "dense coupling costs bounded rank," NOT "coupling does nothing."
   [Probe: ≥ 0.05 at the regime, grows with λ.]

9. **REDUCTION + NO-SOLVER audit.** (a) With `D` tridiagonal, the symbol is bit-identical
   (0 ULP) to the ADR-0164 adjacent path (§1.4a); with `D=diag`, `b=0`, bit-identical to the
   separable symbol (§1.4b). (b) A source-level grep asserts `tt_dense_coupling.rs` does NOT
   contain `lu_solve_inplace(` or `dense_expm(` call-sites (Theorem-6 R2; the dense `expm`
   lives ONLY in the reference). *Anti-vacuous:* faithful-superset proof + solver-free proof.

### 2.4 Why the gate satisfies the audit (NORMATIVE cross-reference)

The two surviving audit counter-probes from the prior milestone are re-run and SURVIVE:
(a) **knife-edge** — absolute max-bond rank varies across eps (5,6,7,8,9 at d=8), but
`Δrank ≤ Δ_CAP=7` holds at every eps and the *difference* over the exact diagonal baseline is
stable; (b) **estimator-correctness** — the metric is the FULL max-over-all-bonds TT-SVD rank,
not a single half-cut, so a non-dominant bond plateauing early cannot hide a still-growing true
max-bond. **The generic-IC state-Δrank arm is DELETED, not relied upon:** at n=5 (and at any n)
a generic full-rank state pins at the algebraic ceiling `min(n^cut, n^{d−cut})` for BOTH dense
and diagonal operators, so the state-Δ is identically 0 (probe: Δ_state = 0,0,−1 at n=5,9,13)
— a ceiling artifact that proves nothing and gives false reassurance; raising n lifts the
ceiling in lockstep and never exposes the operator's rank budget. The operator-symbol Δrank
(fixed-eps difference) IS the knife-edge defense on its own. **The
boundary (design §4) is explicitly NOT overclaimed:** generic dense `D` escape is FALSE and is
*proven* false by assert 5 (rank-2 contrast). The positive claim is scoped to fixed-off-block-
rank `D`. **Reconciliation with v9.1/0164:** the adjacent and drift milestones are the ρ=0,1
cases of the same off-block-rank law; this POC adds the all-pairs ρ=1 (rank-1-dense) case and
delimits ρ≥2 as the boundary.

**No slope gate** (constant-coef ⇒ exact ⇒ a slope gate bottoms out at machine floor ⇒
degenerate; the only truncation-bearing regime, variable-coef, is OUT OF SCOPE). Mirrors the
ADR-0164 / v9.1 §10.13.2(a) exactness-gate decision.

## 3. Build / run (single command, suckless)

```bash
# unit + reduction invariants (fast):
cargo test -p semiflow-core tt_dense_coupling
# the S³ proof gate (dense expm control, slow-tests):
cargo test -p semiflow-core --features slow-tests g_s3_dense_coupling
```

## 4. Out of scope (FAIL-LOUD — do NOT implement in this POC)

Generic dense `D` (off-block rank grows with `d` — info-theoretically un-escapable, proven by
assert 5), variable-coefficient `D(x),b(x),a(x)`, rank-`m` with `m` growing in `d`, nonlinear,
public API / bindings (FFI/PyO3/WASM), FFT perf (keep O(n²) direct DFT), the eigen-rotation
alignment route (design §5, rejected). These are explicitly deferred; the POC proves
*existence in principle* of a dense/all-pairs escape class plus its exact boundary, nothing more.

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
