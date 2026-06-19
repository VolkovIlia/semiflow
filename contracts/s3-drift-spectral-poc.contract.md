# Contract — S³ complex-spectral drift POC (`DriftSpectralPairFactor`)

**Scope:** proof-of-concept ONLY. ONE new evolver variant + ONE gate test. No public API churn, no migration, no library-wide rollout. Suckless: ≤500 LoC/file, ≤50 LoC/fn, no new dep (reuses `tt_spectral.rs` DFT + `SemiflowFloat` `exp/sin/cos`).
**Design:** `.dev-docs/specs/s3-triz-general-curse-escape.md` (Amendment 1, post-audit). **Probes (truth):** `.dev-docs/specs/probe_s3_drift_spectral.py` (exactness/drift core) + `.dev-docs/specs/probe_s3_honest_curse_escape.py` (the honest Δrank + cost-scaling gate, replaces vacuous assert 4).
**Builds on:** `crates/semiflow-core/src/tt_spectral.rs` (v9.1 §11.3 real spectral pair-factor — the ONLY change is widening the symbol from real to complex + one drift term).

---

## 1. New types / functions (Rust, `no_std` + `alloc`, generic over `F: SemiflowFloat`)

All live in a NEW module `crates/semiflow-core/src/tt_drift_spectral.rs` (keeps the POC isolated; ≤500 LoC). Reuse the existing private DFT helpers from `tt_spectral.rs` (promote them to `pub(crate)` if not already — they are: `dft_1d_real_to_cplx`, `dft_1d_cplx`, `idft_1d_cplx`).

### 1.1 Complex symbol builder (the ONE mathematical novelty)

```rust
/// Build the COMPLEX expsym diagonal for `exp(τ·(L_diff + L_drift + L_cross))`
/// over a single (j,k) pair panel. Returns interleaved `(re, im)` of length `2·n_j·n_k`.
///
/// symbol(mj,mk) = cj·σ_D2(mj) + ck·σ_D2(mk)            (real, diffusion)
///               + i·bj·σ_D1r(mj) + i·bk·σ_D1r(mk)      (imaginary, DRIFT — the new term)
///               − 2·r·σ_D1r(mj)·σ_D1r(mk)              (real, cross; i·i = −1)
/// where σ_D2(m) = (2cos ω−2)/dx²,  σ_D1r(m) = sin ω/dx,  ω = 2π m/n.
/// expsym = exp(τ_eff · symbol)  — COMPLEX (re = e^{τ·Re}·cos(τ·Im), im = e^{τ·Re}·sin(τ·Im)).
#[allow(clippy::too_many_arguments)]
pub(crate) fn drift_pair_expsym_cplx<F: SemiflowFloat>(
    n_j: usize, n_k: usize,
    dx_j: F, dx_k: F,
    cj: F, ck: F,          // diffusion coefficients (a_j/#pairs(j), as v9.1)
    bj: F, bk: F,          // DRIFT coefficients (NEW; bj=bk=0 ⇒ reduces to pair_expsym_real)
    r_cross: F,            // cross-diffusion coupling
    tau_eff: F,
) -> Vec<F>;               // length 2·n_j·n_k, interleaved (re,im)
```

### 1.2 Complex-expsym apply (mirror of `apply_spectral_pair_to_panel`, complex multiply)

```rust
/// Apply `exp(τ·L_pair)` (with drift) to a flat n_j×n_k real panel via complex spectral.
/// fft2 → elementwise COMPLEX multiply by `expsym_cplx` → ifft2 → take real part.
/// NO lu_solve, NO dense_expm, NO triangular solve (Theorem-6 R2).
/// Output is real; imaginary residue MUST be < 1e-12 (asserted by gate).
pub(crate) fn apply_drift_spectral_pair_to_panel<F: SemiflowFloat>(
    panel: &mut [F],       // flat n_j·n_k row-major (modified in place; real in, real out)
    n_j: usize, n_k: usize,
    expsym_cplx: &[F],     // 2·n_j·n_k interleaved (re,im) from drift_pair_expsym_cplx
) -> F;                    // returns max |imag residue| for the reality assertion
```

### 1.3 Separable (no-cross) 1D drift apply — proves the rank-1 / O(d·n) escape leg

```rust
/// Apply per-axis `exp(τ(a·∂² + b·∂))` to a 1D real line via complex spectral symbol.
/// The separable d-D evolver is the tensor product of these — TT-op-rank 1 (Probe B).
pub(crate) fn apply_drift_spectral_axis<F: SemiflowFloat>(
    line: &mut [F], n: usize, dx: F, a: F, b: F, tau: F,
) -> F;                    // returns max |imag residue|
```

### 1.4 Reduction invariant (NORMATIVE — Gate sub-check)

`drift_pair_expsym_cplx(.., bj=0, bk=0, ..)` MUST equal (re-part) the existing
`tt_spectral::pair_expsym_real(..)` to 0 ULP (the imaginary part is exactly zero).
This proves the new code is a faithful superset, NOT a relabel (anti-lesson #1).

## 2. The ONE gate that proves S³ (`G_S3_DRIFT_SPECTRAL`)

`crates/semiflow-core/tests/g_s3_drift_spectral.rs`, RELEASE-BLOCKING-class but gated
`#[cfg_attr(not(feature = "slow-tests"), ignore)]` (dense `expm` control). HARD asserts.

**Reference (independent, no spectral code):** assemble the dense centred-FD generator
`L_h = Σ_j (a_j·D2_j/dx² + b_j·D1_j/(2dx)) + Σ_pairs 2r·(D1_j/2dx)⊗(D1_k/2dx)` and compute
`u_ref = expm(T·L_h)·u₀` via an in-test scaling-and-squaring Padé[6/6] helper (reuse
`tt_dense_expm::dense_expm_pub`; NO new dep). This is a different algorithm (LU-Padé) than
the FFT-diagonal scheme under test — genuine independence (anti-lesson #3).

**Asserts (all HARD), for `d∈{3,4}`, with `b_j≠0` on every axis, `ρ=0.6`:**
1. **EXACTNESS (headline):** `rel_l2(drift_spectral_evolved, u_ref) ≤ 1e-12` on the valid
   regime `τ ∼ 0.35·dx²`, `n∈{7,9,11,13}` (d=3) / `{7,9,11}` (d=4). [Probe A/C: ~1e-16.]
2. **DRIFT PRESENT, non-degenerate:** `b_j·τ/dx` non-integer with frac>0.05 at every level
   (anti-lesson #2 — genuine sub-grid advection, not a lattice multiple).
3. **REALITY:** `max|imag residue| < 1e-12` (output real; catches conjugate-even regressions).
4. **CURSE-ESCAPE — HONEST, tolerance-robust (REPLACES the vacuous original; see §2.4).**
   The original assert 4 ("evolved-state rank constant ≤8, does NOT grow with `d`") was
   **retracted as VACUOUS** by adversarial audit (SVD-tolerance knife-edge / IC-rank confound
   / generic-input full-rank — design §6.5). It is replaced by **two** sub-asserts, 4a + 4b,
   neither tolerance-soft, both isolating the drift contribution. (Asserts 1,2,3,5,6,7 unchanged.)

   **4a — Δrank-PRESERVATION sweep (Gate-1; the honest rank statement).** For the SAME IC and
   the SAME SVD tolerance, the evolved-state first-cut TT-rank with `b_j≠0` EQUALS that with
   `b_j=0`:
   `Δrank(eps) := rank_{b≠0}(eps) − rank_{b=0}(eps) == 0`, asserted for **every**
   `eps ∈ {1e-8, 1e-10, 1e-12, 1e-14}` AND **every** `d ∈ {3,4,5,6}`, AND for **BOTH**
   a smooth IC (`g=cos x+0.3`, rank-1 tensor) AND a **generic random IC** (`rng(seed=12345)`,
   evolves to full rank). Pass ⇔ all `Δrank == 0`. Because it is a **difference at fixed eps**
   it is **robust to the knife-edge** (the tolerance cancels); the **generic-IC arm** defeats the
   IC-confound (drift adds zero rank even to a full-rank state). *(Not a self-comparison: 4a is a
   difference of two SPECTRAL runs measuring an ALGEBRAIC operator property — drift = rank-1
   phase — not a correctness claim; correctness is assert 1/E's independent-`expm` job. The
   spectral apply is independently validated exact at d≤4, so its rank read-out at d=5,6 is
   faithful while avoiding the `n^d` `expm` cost.)* [Probe `probe_s3_honest_curse_escape.py`
   Gate-1: `Δrank=[0,0,0,0,0]` across the full sweep, both ICs; n=5 (d-sweep) and n=13 (anti-saturation,
   baseline rank 3–5 ≪ cap with strong coupling r=0.4).] **Algebraic backing (4a-alg):** the
   per-axis drift multiplier `exp(τ b σ_D1)` is unit-modulus (`‖·‖∈[1,1]`) and the 2-axis
   drift-only exp-symbol has `sv₂/sv₁ < 1e-12` ⇒ **TT-operator-rank 1** ⇒ cannot increase state
   rank (design §6.6 proof sketch). [Probe: `sv₂/sv₁ = 8.2e-17`.]

   **4b — OPERATIONAL cost-scaling (Gate-2; the absolute, tolerance-FREE escape).** The evolver
   NEVER assembles the dense `n^d` generator: assert it produces a **finite, real**
   (`max|imag| < 1e-10`) state at `d ∈ {8,10}` (n=5) using only `O(d·n)` symbol storage and an
   FFT-diagonal apply, whereas the independent dense-`expm` reference at those `d` requires an
   `n^{2d}`-entry matrix (`>1 TB` at d=8) that is **un-formable** — THAT intractability is the
   curse the scheme escapes. Pass ⇔ evolver runs (finite, real) at d=8,10 AND a static check
   confirms no `n^d` dense object is built in the evolver path. [Probe Gate-2: runs at d=8,10,
   imag~1e-16; dense expm needs 1.2 TB / 763 PB.] **Pairing (NORMATIVE):** 4a and 4b are a
   conjunction — 4a bounds the state rank `r` (drift inherits the v9.1 `b=0` rank, adds zero),
   4b bounds the cost given `r`. Neither alone is the S³ claim; together they are. Crucially the
   POC does **NOT** assert an absolute generic-input curse-escape (info-theoretically false — no
   TT method compresses random tensors; design §6.8).
5. **ANTI-TRIVIALITY:** a rank-1 separable IC evolves to TT rank > 1 under coupling
   (excludes the v9.0 separability no-op failure mode). [Probe C: 1→4.]
6. **REDUCTION:** with `b_j=0`, the scheme is bit-identical (0 ULP) to the v9.1
   `CoupledTtChernoff` spectral path (Gate-C-style invariant).
7. **NO-SOLVER audit:** a source-level grep assert in the test comment / a CI check that
   `tt_drift_spectral.rs` does NOT reference `lu_solve_inplace` or `dense_expm`
   (Theorem-6 R2; the dense `expm` lives ONLY in the reference, never in the evolver).
8. **LOAD-BEARING drift (makes 4a non-vacuous; NEW, criterion L of design §6.7).**
   `‖U(b≠0) − U(b=0)‖ / ‖U(b=0)‖ ≥ 0.05` at the gate regime (`d=4`, `τ∼0.35·dx²`). This
   proves the drift is **present and load-bearing**, so "Δrank=0" (4a) is "drift costs zero
   rank" — NOT "drift does nothing." Without this, a hostile reviewer could call 4a vacuous.
   [Probe: 0.33 at the gate regime; 0.05→0.45 as τ grows.]

### 2.4 Why the new assert 4 satisfies the audit (NORMATIVE cross-reference)

The three audit counter-probes are **re-run against the new gate** in
`.dev-docs/specs/probe_s3_honest_curse_escape.py` (`audit_counterprobes()`), all SURVIVE:
(a) knife-edge — absolute rank varies across eps, but `Δrank=[0,0,0,0,0]` at every eps;
(b) IC-confound — claim is "Δrank=0," not "rank is low," so a low-rank baseline is fine;
(c) generic input — full-rank state (`n`), yet `Δrank=[0,0,0,0,0]`. The honest boundary
(§6.8): absolute generic-input curse-escape is **NOT** claimed (info-theoretically false).
**Reconciliation with v9.1:** the v9.1 `b=0` scheme shipped the same `[5,5,5,5]` absolute
headline, which is **equally tolerance-soft**; the Δrank framing is robust **regardless**,
because it is a difference relative to that exact baseline (design §6.5–§6.6).

**No slope gate** (the constant-coef scheme is exact ⇒ a slope gate bottoms out at machine
floor ⇒ degenerate; the only truncation-bearing regime, variable-coef, is OUT OF SCOPE — §8
of the design). This mirrors the v9.1 §10.13.2(a) exactness-gate decision.

## 3. Build / run (single command, suckless)

```bash
# unit + reduction invariants (fast):
cargo test -p semiflow-core tt_drift_spectral
# the S³ proof gate (dense expm control, slow-tests):
cargo test -p semiflow-core --features slow-tests g_s3_drift_spectral
```

## 4. Out of scope (FAIL-LOUD — do NOT implement in this POC)

Variable-coefficient (`a(x),b(x)`), non-adjacent / dense all-pairs coupling, nonlinear,
public API / bindings (FFI/PyO3/WASM), FFT perf (keep O(n²) direct DFT), perf benchmarks.
These are explicitly deferred; the POC proves *existence in principle*, nothing more.

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
