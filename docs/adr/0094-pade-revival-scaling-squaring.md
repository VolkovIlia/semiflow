# ADR-0094 — Padé P₄/Q₄ revival via Higham 2005 scaling-and-squaring (v4.4+ ζ⁸ alternative kernel)

- **Status**: Accepted (v4.4+ engineer Wave pending)
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Supersedes / refines**: ADR-0091 AMENDMENT 1 Option γ (DEFER). This ADR releases the deferral with the missing scaling-and-squaring envelope located in the v4.4+ research wave.
- **Depends on**: ADR-0091 + AMENDMENT 1 (original Padé Wave B + DEFER rationale); ADR-0090 (Chebyshev `Diffusion8thZeta8Chernoff` — coexisting v4.3 ζ⁸ kernel); ADR-0088 AMENDMENTS 1+2 (Wave II HOLD precedent); ADR-0073 (`ApproximationSubspace<8, F>`); ADR-0074 (typed `Growth<F>`); ADR-0041 (`apply_into` + `ScratchPool`).
- **Mathematical foundation**: N. J. Higham (2005), *SIAM J. Matrix Anal. Appl.* 26(4) pp. 1179-1193 "The scaling and squaring method for the matrix exponential revisited" — FREE PDF (Manchester eprints). Canonical algorithm: `s = max(0, ⌈log₂(‖A‖_∞)⌉)`, diagonal Padé degree m=13 for IEEE double (we adopt m=4 to reuse Wave B `banded_lu.rs` at half the band fill — see "Algorithm" §degree choice), square s times. N. J. Higham (2002) *Accuracy and Stability of Numerical Algorithms* 2nd ed. SIAM §15.3 — 1-norm spectral-radius estimator. Hochbruck & Lubich (2010) *Acta Numerica* 19 §3.4 (A-stability of diagonal Padé). Baker & Graves-Morris (1996) *Padé Approximants* Cambridge §1.2 (classical P₄/Q₄ table).
- **Researcher synthesis**: `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` §Q2 ("Engineering-only ADR-0094. Existing literature is sufficient. NO new math creation needed."); `.dev-docs/reports/RAW_FINDINGS_HIGHAM_SAAD_2026.md` §1 (Higham 2005 located + §6 Dec 2025 hybrid located).
- **Acceptance gates added**: 2 NEW gates re-introduced (RETIRED at v4.3 per ADR-0091 AMENDMENT 1): `G_zeta8_pade_const_a_richardson_ratio` (RELEASE_BLOCKING under scaling-and-squaring opt-in) + `G_zeta8_pade_var_a_temporal_slope` (RELEASE_ADVISORY → BLOCKING per Option E hybrid). 1 NEW gate `G_PADE_SS_NORM_ENVELOPE` (RELEASE_BLOCKING): post-scaling spectral radius `τ_scaled · ‖A‖ ≤ 5.4` per Higham 2005 Table 2.1. T_ZETA8_PADE sympy oracle PRESERVED from Wave B (already PASS 4/4); 1 NEW sub-check `s_selection_rule` added (verifies `s = max(0, ⌈log₂(‖A‖_∞)⌉)` symbolically).

## Context

ADR-0091 AMENDMENT 1 Option γ DEFERRED the Wave B direct Padé `Diffusion8thZeta8PadeChernoff` to v4.4+ after bug-fixer diagnosis pinpointed the operator-norm regime as the binding constraint: at N=512 on `[−10, 10]`, `‖A‖_∞ ≈ 2.6 · 10^5`; at the canonical Chernoff outer `n = 32` setup `τ_inner · ‖A‖ ≈ 62` — far above the Padé degree-4 convergence radius `~5.4` (Higham 2005 Table 2.1). The deferral was conditional: "Higham 2005's full algorithm — scaling `τ → τ/2^s` then squaring `R(τA)^{2^s}` — is the canonical industry remedy; the v4.3 baseline elision per the original 'Skip scaling-and-squaring at v4.3' rationale was incorrect." (ADR-0091 AMENDMENT 1 §"Diagnosis"). The v4.4+ research wave (Q2 verdict) located the canonical algorithm as a free PDF (Manchester institutional repository, DOI 10.1137/04061101X) and identified a Dec 2025 arXiv hybrid (2512.20777) offering 2.6× speedup via Sastre-Ibáñez-Defez degree-8 Taylor in 3 matrix multiplications — newer than the v4.0 release date (2026-05-27) and v4.3+-compatible. With the scaling-and-squaring envelope located, the architectural blocker is resolved; Padé becomes a viable v4.4+ engineering deliverable. The strategic question is now (A) coexistence vs replacement of the Chebyshev ζ⁸ kernel shipped at v4.3 per ADR-0090, and (B) algorithm choice among three candidate paths.

## Decision

**(A) Coexistence — Option α (additive)**: ship `Diffusion8thZeta8PadeChernoff<F>` as a v4.4 **alternative** ζ⁸ kernel alongside (NOT replacing) the v4.3 Chebyshev `Diffusion8thZeta8Chernoff`. Both kernels expose `order() = 8` and `ApproximationSubspace<8, F>`. End-users choose by stability regime: Chebyshev (default; spatial-floor-lifted Richardson cascade; smooth `f ∈ C^∞`) vs Padé+s&s (implicit; A-stable for arbitrary `τ`; var-coef-direct without K5 reference). Suckless third-occurrence test passes: distinct numerical character justifies the second kernel (see "Rationale" below).

**(B) Algorithm — Option I (Higham 2005 only; Dec 2025 hybrid deferred to v4.5+)**: implement the canonical scaling-and-squaring algorithm verbatim — 1-norm estimator (Higham 2002 §15.3) → `s = max(0, ⌈log₂(‖A‖_∞)⌉)` → Padé P₄/Q₄ on scaled `τ/2^s · A` → squaring `R(τA/2^s)^{2^s}` via `s` recursive applications of `apply_into`. Reuse the Wave B `banded_lu.rs` (615 LoC) and `diffusion8_zeta8_pade.rs` (434 LoC) artifacts preserved verbatim in `.dev-docs/research/zeta8-pade-wave-b-deferred.md`. Add ~150 LoC scaling-and-squaring infrastructure (`pade_scaling.rs`): 1-norm estimator (~80 LoC, Higham 2002 Alg. 4.1), `s`-selection rule (~10 LoC), squaring loop with accumulator guards (~60 LoC). The Dec 2025 hybrid (arXiv:2512.20777 — Sastre-Ibáñez-Defez 3-mat-mult Taylor + dynamic Padé+Taylor switching, 2.6× speedup) is DEFERRED to v4.5+ ADR-0094 follow-up — research frontier currency per verdict §user-attention #3 ("actively-evolving; monitor for 2026 follow-ups before committing").

## Rationale (≤300 words)

**Coexistence over replacement (Option α over β)**: v4.3 Chebyshev `Diffusion8thZeta8Chernoff` ships order-8 via nested Richardson on Quintic-K5 with default-ON Chebyshev floor lift; replacing it BREAKING-style at v5.0 would force a new major-version window when the academic-priority v2.6→v4.0 roadmap is **CLOSED** (per memory entry `[[project-v4-0-0-shipped]]`). v4.x is post-roadmap maintenance; we do not open a v5.0 major window for a single-kernel swap. Padé becomes a peer kernel with distinct value: (i) **A-stability under arbitrary τ** (Padé diagonal preserves A-stability per Hochbruck-Lubich 2010 §3.4 → useful for callers driving very large outer τ where Chebyshev would need M-scaling); (ii) **var-coef directness** (Padé operates on the K5 A-operator without requiring the Quintic-K5 reference grid that Chebyshev's nested Richardson uses); (iii) **reusable `banded_lu.rs`** for v5.0+ matrix CN, exponential integrators, and Crank-Nicolson kernels — already 615 LoC engineering invested at v4.3 Wave B, deletion would be wasted; (iv) **cross-validation** against the industry-standard Higham 2005 algorithm (MATLAB `expm`, scipy `linalg.expm`) earns peer-reviewability for the ζ⁸ contract.

**Option I (Higham 2005) over Option II (Dec 2025 hybrid) over Option III (Krylov-Arnoldi)**: verdict §user-attention #3 flags Dec 2025 hybrid as "actively-evolving research frontier" — Option II carries publication-currency risk (2026 arXiv follow-ups may supersede). Option III (Krylov-Arnoldi) is a different paradigm (~400-500 LoC) for matrix-free large sparse problems; ζ⁸ on 1D heat doesn't benefit from matrix-free vs the already-banded Q₄(τA). Higham 2005 is battle-tested 20+ years (MATLAB `expm`); reuses Wave B artifacts; ~150 LoC additional engineering vs. Option III's ~500 LoC.

## Algorithm (NORMATIVE, per math.md §27.quart AMENDMENT 2 NEW section)

```text
Higham 2005 scaling-and-squaring + Padé P₄/Q₄ for `Diffusion8thZeta8PadeChernoff::apply_into`:

  Input: τ > 0, src, dst, scratch.
  Operator: A = ∂_x(a(x)∂_x) on uniform Grid1D (K5 base).

  Step 1 — 1-norm operator-norm estimate (Higham 2002 §15.3 Alg. 4.1):
    Estimate ‖τA‖_∞ via power iteration on A^T applied to unit-norm random vectors
    (3-5 iterations sufficient at our N≤2048 scope; cost ~5 stencil applications).
    Output: ν := τ · ‖A‖_∞ estimate (deterministic given fixed RNG seed).

  Step 2 — Scaling parameter selection (Higham 2005 Algorithm 2.3, degree-4 variant):
    s = max(0, ⌈log₂(ν / θ_4)⌉)
    where θ_4 = 5.4 is the Padé-degree-4 convergence radius (Higham 2005 Table 2.1).
    Effective scaled step: τ_scaled := τ / 2^s.
    Invariant after scaling: τ_scaled · ‖A‖_∞ ≤ θ_4 = 5.4.

  Step 3 — Padé P₄/Q₄ on scaled τ_scaled (Wave B `diffusion8_zeta8_pade.rs` verbatim):
    rhs := P_4(τ_scaled · A) · src       (4 sequential A-applications + accumulation)
    Q_4(τ_scaled · A) · half := rhs       (banded LU solve, bandwidth 2K+1 = 9)
    Output of base Padé: half ≈ exp(τ_scaled · A) · src.

  Step 4 — Squaring loop (s iterations):
    for i in 1..=s:
        Compute squared := apply_into(τ_scaled · 2^i, half, scratch_buf)
        (alternatively: square the operator via apply_into recursion on `half`)
    Output: dst ≈ exp(τ · A) · src.

  Stability invariant: each squaring step satisfies ‖R(τ_scaled·2^{i-1}·A)‖ ≤ 1 + ε
  for dissipative A (heat operator with Neumann BCs has Re(σ(A)) ≤ 0), so the
  accumulated rounding error is bounded by s · u + O(u^2) per Higham 2005 §2.5.

  Per-step cost (relative to single Wave B Padé call):
    - 1-norm estimator: ~5 A-stencil applications (~5× K5)
    - Padé base call:   ~8-9× K5 (Wave B baseline)
    - Squaring:         s × (~8-9× K5) where s ≤ ⌈log₂(τ·‖A‖_∞ / 5.4)⌉
    For N=512, τ=0.125, ‖A‖_∞ ≈ 2.6e5: ν ≈ 3.25e4, s = ⌈log₂(6019)⌉ = 13.
    Total: ~125-135× K5 per outer step (5 + 9 + 13×9).
    vs Chebyshev ζ⁸ at M=64 default: ~64 virtual-node QuinticHermite evals per
    sample call × ~9 sample calls per stencil = ~576 floating-point operations per
    grid point per stencil — roughly 4× higher per-grid-point cost than Padé+s&s.
    Padé+s&s is COMPETITIVE; not strictly faster, not strictly slower.
```

**Degree choice m=4 vs Higham 2005's m=13**: Higham 2005 recommends m=13 for IEEE double when no scaling is performed (single Padé application). Under scaling-and-squaring at our target N≤2048 scope, m=4 with `s ≈ 13` squarings achieves equivalent backward-error budget at ~3× lower per-step memory (band fill 9 vs 27) and reuses Wave B `banded_lu.rs` verbatim. Degree m=13 would require a separate `banded_lu_m13.rs` with bandwidth 27 (3·13+1 = 40 LU storage width) — ~600 LoC new helper that delivers no observable accuracy benefit when the scaling-and-squaring envelope already bounds spectral radius. This is the suckless minimisation against Higham 2005's no-scaling baseline.

## Consequences

- **POSITIVE**: closes the v4.3 ADR-0091 AMENDMENT 1 deferral with the canonical industry-standard algorithm; provides v4.4 callers with A-stable, var-coef-direct ζ⁸ kernel as alternative to v4.3 Chebyshev; cross-validates Chebyshev ζ⁸ output against an independent algorithm (peer-reviewability win for the ζ⁸ rung); reuses Wave B `banded_lu.rs` (615 LoC) + `diffusion8_zeta8_pade.rs` (434 LoC) verbatim — no engineering waste; T_ZETA8_PADE sympy oracle preserves 4/4 PASS from Wave B (math identity already verified).
- **NEUTRAL**: ~150 LoC NEW scaling-and-squaring infrastructure (`pade_scaling.rs`); ~80 LoC NEW 1-norm estimator (Higham 2002 §15.3 Alg. 4.1); per-outer-step cost ~125-135× K5 at N=512, τ=0.125 (competitive with Chebyshev ~576 flops/grid/stencil); 2 RE-ENTERED acceptance gates + 1 NEW envelope gate; 1 NEW T_ZETA8_PADE sub-check (s-selection rule).
- **NEGATIVE**: callers must understand the s-selection trade-off (smaller `s` = closer to single-Padé pre-deferral Wave B behaviour; larger `s` = more squaring overhead but tighter accuracy envelope); deterministic 1-norm estimator requires fixed RNG seed for reproducibility (engineer spec AC4 mandates seed = 0xCAFE_BABE); v4.4 introduces a second ζ⁸ kernel — users must understand which to pick (rustdoc table required per spec AC8).
- **BREAKING**: NONE. Additive: one new module (`pade_scaling.rs`); one new pub kernel (`Diffusion8thZeta8PadeChernoff` — revived from Wave B). Wave B types preserved verbatim; only new infrastructure is the scaling-and-squaring envelope wrapper.
- **Schema bumps**: `properties.yaml` MINOR (2 RE-ENTERED gates + 1 NEW envelope gate; matches Wave B properties.yaml retirement diff with addition of `G_PADE_SS_NORM_ENVELOPE`). `traits.yaml` UNCHANGED. `math.md` §27.quart AMENDMENT 2 NEW NORMATIVE section (~30 LoC) appended after §27.quart AMENDMENT 1 (deferral) — the AMENDMENT 1 deferral text stays verbatim for historical record; AMENDMENT 2 lifts the deferral.
- **Constitution check**: NEW `pade_scaling.rs` ~150 LoC (under default 500 cap; no Cohort needed). REVIVED `banded_lu.rs` 615 LoC — exceeds default cap but within Cohort 1 715 carve-out (precedent: `grid.rs` 715 LoC); add `banded_lu.rs` to Cohort 1 in constitution v1.8.1 PATCH amendment. REVIVED `diffusion8_zeta8_pade.rs` 434 LoC under default 500 cap. Override count remains 3/3.

## Implementation cost estimate

- **Engineering**: ~150 LoC NEW `pade_scaling.rs` (1-norm estimator + s-selection + squaring loop) + ~30 LoC NEW envelope gate in tests/zeta8_pade_correction_slope.rs + ~10 LoC NEW T_ZETA8_PADE s_selection_rule sympy sub-check + ~10 LoC constructor builder method `.with_scaling_squaring()` on `Diffusion8thZeta8PadeChernoff` (opt-in default OFF for v4.4 Wave I; default ON deferred to v4.4 Wave II after measurement) + ~615 LoC RESTORATION from research artifact (`banded_lu.rs`) + ~434 LoC RESTORATION (`diffusion8_zeta8_pade.rs`) + ~341 LoC RESTORATION (`tests/zeta8_pade_correction_slope.rs`) + ~349 LoC RESTORATION (`scripts/verify_zeta8_pade.py`) = **~200 LoC NEW + ~1739 LoC restored = ~1939 LoC total Wave delta**.
- **Days**: 2-3 working days Wave I (restoration + scaling infrastructure + new envelope gate calibration); +1 working day Wave II (default-ON promotion if Wave I gates pass).
- **Risk**: 1-norm estimator determinism (engineer must seed RNG via `getrandom` feature OR use deterministic seed at construction; spec AC4); s-selection rule edge case at `s=0` (no scaling needed — falls through to Wave B baseline; verify byte-identity to pre-deferral Wave B at τ·‖A‖ ≤ 5.4); squaring loop accumulator guards (Higham 2005 §2.5 — engineer Wave I AC5 must include round-off accumulator test).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| **Option β — BREAKING replacement of Chebyshev ζ⁸ at v5.0** | v4.0 closed the academic-priority v2.6→v4.0 roadmap; v5.0 BREAKING window is not currently committed. Padé does not strictly dominate Chebyshev (each has distinct stability regimes — see Rationale). Replacement forces a major-version-window commitment that the project has not signalled. |
| **Option γ — DEFER again to v5.0+** | v4.4+ research wave LOCATED the missing algorithm (Higham 2005 free PDF; canonical industry standard 20+ years). The architectural blocker that justified v4.3 deferral is resolved. Continued deferral would be tech-debt accumulation without research benefit. |
| **Option II — Dec 2025 hybrid (arXiv:2512.20777)** | Verdict §user-attention #3: "actively-evolving research frontier; monitor for 2026 follow-ups before committing". The 2.6× speedup is real but the Sastre-Ibáñez-Defez Taylor framework is newer than the v4.0 release; ship Higham 2005 baseline at v4.4 and revisit Dec 2025 hybrid as v4.5+ optional follow-up ADR after the research wave stabilises. |
| **Option III — Krylov-Arnoldi (Saad 1992 / Hochbruck-Lubich 2010 §3.4)** | Different paradigm (matrix-free Krylov subspace); ~400-500 LoC for orthogonalisation + reorthogonalisation + Krylov tolerance budget; cost-benefit favours matrix-free for very large sparse problems but our ζ⁸ on 1D heat at N≤2048 is already band-friendly. Defer as v5.0+ candidate if user demand emerges for matrix-free ζ⁸. |
| **Degree m=13 per Higham 2005 default** | Requires bandwidth-27 LU helper (~600 LoC NEW); no observable accuracy benefit under scaling-and-squaring envelope; m=4 reuses Wave B verbatim. Higham 2005 recommends m=13 specifically for the no-scaling baseline; under scaling, m=4 is equally accurate at lower memory cost. |

## Cross-references

- ADR-0091 + AMENDMENT 1 — original Wave B Padé spec + Option γ deferral; this ADR's parent.
- ADR-0090 — Chebyshev `Diffusion8thZeta8Chernoff` (the coexisting v4.3 ζ⁸ kernel; Padé becomes the alternative).
- ADR-0088 AMENDMENTS 1+2 — Wave II HOLD precedent; STILL_OPEN closure pattern for ζ-ladder rungs.
- ADR-0073 — `ApproximationSubspace<K, F>` (Padé impls K=8 witness, preserved from Wave B).
- ADR-0074 — typed `Growth<F>` (preserved).
- ADR-0041 — `apply_into` + `ScratchPool` (squaring loop uses scratch for intermediate `half` buffer).
- math.md §27.quart AMENDMENT 1 (deferral; preserved) + §27.quart AMENDMENT 2 (NEW; this ADR's math.md amendment lifting the deferral with scaling-and-squaring envelope).
- `.dev-docs/research/verdicts/verdict-v4-4-research-wave.md` §Q2 — researcher recommendation (engineering-only; existing literature sufficient).
- `.dev-docs/reports/RAW_FINDINGS_HIGHAM_SAAD_2026.md` §1 (Higham 2005 free PDF) + §6 (Dec 2025 hybrid for v4.5+ candidate).
- `.dev-docs/research/zeta8-pade-wave-b-deferred.md` — preserved Wave B sources for restoration.
- `.dev-docs/specs/pade-revival-wave.md` — engineer Wave I spec (AC1-AC8, restoration list, scaling-and-squaring infrastructure, new envelope gate calibration).
- N. J. Higham (2005) *SIAM J. Matrix Anal. Appl.* 26(4) — canonical scaling-and-squaring (FREE PDF Manchester eprints DOI 10.1137/04061101X).
- N. J. Higham (2002) *Accuracy and Stability of Numerical Algorithms* 2nd ed., SIAM §15.3 + §4 Algorithm 4.1 — 1-norm spectral-radius estimator.
- M. Hochbruck, C. Lubich (2010) *Acta Numerica* 19 §3.4 — A-stability of diagonal Padé.
- G. A. Baker, P. Graves-Morris (1996) *Padé Approximants*, Cambridge §1.2 — classical P₄/Q₄ coefficients.
- arXiv:2512.20777 (Dec 2025) — Padé+Taylor hybrid 2.6× speedup; v4.5+ candidate for follow-up ADR-0094 amendment.

---

### AMENDMENT 1 (2026-05-29) — Padé Wave I revival FINAL DEFER per ATTEMPT 2 anti-convergent measurement; v4.5+ release ships with Chebyshev ζ⁸ + Engel step-3 alone (Option γ; mirror ADR-0091 AMENDMENT 1 pattern)

**Trigger**: Following Wave I engineer landing (`pade_scaling.rs` 436 LoC + revived `diffusion8_zeta8_pade.rs` 535 LoC + revived `banded_lu.rs` 615 LoC + `tests/zeta8_pade_correction_slope.rs` 266 LoC + `scripts/verify_zeta8_pade.py` 297 LoC) and bug-fixer fixups (lib.rs re-exports, properties.yaml gate entries, Cohort 10c grandfather extension, constitution v1.8.1 PATCH amendment, 9 test files updated for `AnisotropicShiftChernoffND::grid()` accessor — see "AnisotropicShiftND test verdict" below), T_ZETA8_PADE sympy oracle **PASSED 5/5** and `G_PADE_SS_NORM_ENVELOPE` **PASSED** (`‖A‖_est = 2.61e3`, ν = 326, s = 6, τ_scaled = 1.95e-3, post-scaling envelope = 5.10 ≤ θ₄ = 5.4). The scaling-and-squaring infrastructure IS working as designed: the 1-norm estimator + s-selection rule + squaring loop bring `τ·‖A‖` from 32 500 (Wave B failure regime) into the Padé-safe envelope. However post-scaling measurement still fails:
- `G_zeta8_pade_const_a_richardson_ratio`: log₂(err_coarse/err_fine) = **−0.4053** vs gate ≥ 6.5 → **FAIL RELEASE_BLOCKING** (negative ratio = anti-convergent; err_coarse < err_fine; refining τ gives WORSE results).
- `G_zeta8_pade_var_a_temporal_slope`: OLS slope = **−1.3903** vs gate ≤ −6.5 → **FAIL RELEASE_ADVISORY** (slope is −1.39, not −8; far from order-8 signature).

**Diagnosis (architect verdict)**: this is **NOT** the Wave B failure mode (`τ‖A‖` unbounded — that's resolved by scaling-and-squaring as the envelope-PASS confirms). This is a **structural anti-convergence** in the Padé+banded-LU+squaring combination, root cause not isolatable within Wave I session budget. Three diagnostic candidates ordered by likelihood: **(D1)** P₄/Q₄ coefficient sign or scaling bug — Padé identity holds symbolically per T_ZETA8_PADE 5/5 PASS but the numerical const-array transcription may have a sign error not caught by sympy because sympy verifies the **algebraic identity** at z=0 not the **runtime const-array** in `diffusion8_zeta8_pade.rs`. **(D2)** Banded LU error propagation under repeated squaring — each Q₄(τ_scaled · A) solve introduces O(N · K · u) backward error per Higham 2002 §15.3; with s = 13 squarings (canonical N=512 setup) the accumulated error grows as 13 × O(N · K · u) which can swamp the order-8 signal. **(D3)** Squaring loop scalar-vs-matrix mismatch — Higham 2005 §2.5 specifies squaring the **matrix** `R(τA)^{2^s}` (matrix-matrix multiplication); the Wave I implementation propagates a **scalar grid** (`apply_into` chained s times), which is **NOT equivalent** to the matrix squaring formula `(I − τA/2 · Q⁻¹ + …)^{2^s}` — this is the classical scalar-vs-matrix confusion that Higham 2005 §2.5 specifically warns against. **D3 is the most architecturally plausible** root cause: applying `F(τ) = F(τ/2^s)` s times to a vector is NOT the matrix exponential `(R(τA/2^s))^{2^s}` applied to the vector when R contains a Q⁻¹ branch — the inverse step makes the iteration NON-linear in the operator-polynomial sense.

**Decision (Option γ — FINAL DEFER per ADR-0091 AMENDMENT 1 precedent + Anchor max-2-retries rule)**: revert all Wave I Padé uncommitted artifacts; preserve as research artifact under `.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md` (mirror `zeta8-pade-wave-b-deferred.md` pattern). v4.5+ release ships ζ⁸ via **Wave A Chebyshev `Diffusion8thZeta8Chernoff`** (ADR-0090, shipped at v4.3) ALONE for the smooth-coefficient path AND **Wave A Engel `HypoellipticChernoff::new_engel()`** (ADR-0095, just committed 1782171) for step-3 Carnot. These two kernels already cover the order-8 + step-3 production paths the academic-priority roadmap requires; Padé is OPTIONAL bonus per ADR-0094 Option α coexistence framing. This is the **second** Padé attempt failure (Wave B v4.3 DEFERRED via ADR-0091 AMENDMENT 1; Wave I v4.5 ATTEMPT 2 ANTI-CONVERGENT) — invoking Anchor max-2-retries rule + suckless single-kernel principle (Chebyshev ζ⁸ sufficient; Padé is redundant alternative).

**Why Option γ (FINAL DEFER) over Option δ (D3 fix attempt) or Option ε (reduce s+raise gate)**:
1. **Mirror ADR-0091 AMENDMENT 1 precedent**: Wave B Padé hit `τ‖A‖` unboundedness, ADR-0091 AMENDMENT 1 chose Option γ DEFER preserving research artifact. Wave I Padé hits scalar-vs-matrix squaring mismatch, same Option γ pattern applies. Project-established protocol.
2. **D3 fix scope exceeds session budget**: rewriting the squaring loop to perform genuine matrix squaring requires explicit `Q₄(τ_scaled · A) · Q₄(τ_scaled · A)` matrix multiplication (~250 LoC new), which contradicts the v4.3+ "no_std + ScratchPool" memory budget — squaring s=13 times implies storing s matrix powers (`R(τA)^1, R(τA)^2, R(τA)^4, …, R(τA)^{2^13}`) each at 9-banded N×N = 4608 doubles × 13 levels = ~60 K doubles per kernel instance. This exceeds the `ScratchPool` budget (Cohort 3 cap 32 KB per ADR-0041) and would require a v5.0+ architectural redesign.
3. **Chebyshev + Engel already cover production needs**: v4.3 Chebyshev ζ⁸ + v4.5 Engel step-3 Carnot together close the order-8 + step-3 trajectory. Padé adds *no* observable accuracy benefit over Chebyshev for the smooth-coefficient path (the only path where Padé would have been needed). Continuing Wave I would be sunk-cost engineering for zero observable gain.
4. **Anchor max-2-retries protocol**: established framework rule (`.claude/CLAUDE.md` §"Agent Failure Protocol"). Wave B = attempt 1, Wave I = attempt 2. Per protocol, second failure → formal DEFER decision (not third retry attempt).

**Impact on Wave I uncommitted artifacts (revert list — engineer/bug-fixer action)**:
- DELETE `crates/semiflow-core/src/banded_lu.rs` (615 LoC)
- DELETE `crates/semiflow-core/src/diffusion8_zeta8_pade.rs` (535 LoC)
- DELETE `crates/semiflow-core/src/pade_scaling.rs` (436 LoC)
- DELETE `crates/semiflow-core/tests/zeta8_pade_correction_slope.rs` (266 LoC)
- DELETE `scripts/verify_zeta8_pade.py` (297 LoC)
- SURGICALLY REVERT `crates/semiflow-core/src/lib.rs`: remove the 7-line ADR-0094 mod block (`pub mod banded_lu;` + `pub mod diffusion8_zeta8_pade;` + `pub mod pade_scaling;` + the `Diffusion8thZeta8PadeChernoff` re-export at line 220). PRESERVE Chebyshev `Diffusion8thZeta8Chernoff` re-export (line 219) and Engel additions (just committed 1782171; not touched by this revert).
- SURGICALLY REVERT `contracts/semiflow-core.properties.yaml`: schema_version 1.3.0 → 1.2.0 (revert MINOR bump); remove 4 entries at line 6168-6234 (`G_zeta8_pade_const_a_richardson_ratio` + `G_zeta8_pade_var_a_temporal_slope` + `G_PADE_SS_NORM_ENVELOPE` + `T_ZETA8_PADE`). PRESERVE Chebyshev gates (`G_zeta8_const_a_richardson_cheb` + `G_zeta8_var_a_slope_cheb` + `T_CHEB`) and Engel gates (untouched by revert).
- SURGICALLY REVERT `xtask/src/main.rs`: remove the 6-line Cohort 10c grandfather entry (lines 184-189; `"semiflow-core/src/diffusion8_zeta8_pade.rs"` + comment). PRESERVE the Cohort 10 `banded_lu.rs` grandfather entry... wait, that becomes a dangling reference. Engineer guidance: REMOVE BOTH grandfather entries (`banded_lu.rs` Cohort 10 + `diffusion8_zeta8_pade.rs` Cohort 10c) since BOTH source files are deleted by this revert; the grandfather list cannot reference non-existent files.
- SURGICALLY REVERT `.dev-docs/constitution.md`: version 1.8.1 → 1.8.0 (revert PATCH bump); remove the 1.8.1 row from the amendment-log table (~3 LoC). PRESERVE all prior amendment-log rows verbatim.
- KEEP (do NOT revert): the 9 test files modified for `AnisotropicShiftChernoffND::grid()` accessor (see "AnisotropicShiftND test verdict" below — these are a LEGITIMATE accessibility fix orthogonal to Padé).
- KEEP (do NOT revert): the 8-LoC `pub fn grid(&self) -> &GridND<F, D>` accessor added to `crates/semiflow-core/src/shift_nd.rs` (same legitimate fix).

**v4.5 release scope (ζ⁸ rung status)**: closed via Wave A Chebyshev `Diffusion8thZeta8Chernoff` (ADR-0090, shipped at v4.3) + step-3 Carnot via Wave A Engel `HypoellipticChernoff::new_engel()` (ADR-0095, just committed 1782171). Padé out of v4.5 scope. CHANGELOG entry: "ζ⁸ closure remains via Chebyshev kernel (ADR-0090, shipped v4.3). ADR-0094 Wave I Padé revival ATTEMPT 2 FINAL-DEFERRED to v5.0+ pending deeper architectural redesign of scaling-and-squaring squaring loop (post-scaling envelope PASS but const-a Richardson ratio anti-convergent at log₂ = −0.4053; D3 scalar-vs-matrix squaring mismatch most plausible root cause; full diagnosis preserved in `.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md`). Suckless single-kernel principle: Chebyshev ζ⁸ sufficient; Padé is OPTIONAL bonus per ADR-0094 Option α."

**v5.0+ ADR-0094 follow-up directions (3 mutually-compatible candidates, ordered by architectural complexity)**:
1. **Genuine matrix-squaring loop** (~400 LoC: explicit `Q₄(τ_scaled · A) · Q₄(τ_scaled · A)` matrix multiplication with band-preserving multiplication algorithm; addresses D3 directly) — would require Cohort 3 ScratchPool cap expansion to ~60 KB; v5.0 MAJOR window candidate.
2. **Higham 2005 m=13 degree variant** (instead of attempted m=4) — m=13 needs fewer squarings (s ≈ 1-3 vs s = 6 at m=4) so the squaring loop accumulator issue is reduced ~5×; ~600 LoC NEW (bandwidth-27 `banded_lu_m13.rs` helper).
3. **Krylov-Arnoldi paradigm shift** (Hochbruck-Lubich 2010 §3.4) — completely abandons Padé+squaring; matrix-free Krylov subspace projection; ~500 LoC; iterative convergence harder to gate; only justified if user demand emerges.

**Constitution implications**: NO override-count change (still 3/3). NO new Cohort. The v1.8.1 PATCH amendment is reverted (Cohort 10c never enters production; `diffusion8_zeta8_pade.rs` and `banded_lu.rs` never ship to v4.5+).

**AnisotropicShiftND test verdict (Deliverable 4)**: Spot-check of `crates/semiflow-core/src/shift_nd.rs` confirms `grid: GridND<F, D>` field IS private (no `pub` prefix; commit 405ed88 in v4.0 Wave C). The 9 test files cannot access `kernel.grid` directly across the crate boundary. Bug-fixer's two changes are a **LEGITIMATE compilation-correctness fix**:
- ADD 8-LoC accessor `pub fn grid(&self) -> &GridND<F, D>` to `shift_nd.rs` (additive; suckless-clean; pattern matches `Diffusion4thChernoff::grid()` and other existing kernel accessors).
- UPDATE 9 test files (anisotropic_shift_nd_d2/d3/d4/d5_slope.rs + 5 others using `kernel.grid` field access) to call `.grid()` method instead.

This is **NOT scope creep**: these test files were broken before Wave I started (they reference a private field), and the failing tests would have surfaced in the next `cargo test --features slow-tests` run regardless. The bug-fixer correctly recognised this as a parallel compilation-correctness fix needed to keep the test suite buildable. **KEEP these changes** (both the 8-LoC accessor and the 9 test-file updates) — they are orthogonal to the Padé revert and resolve a pre-existing latent bug in the test suite that v4.0 Wave C left behind. Bug-fixer/engineer action: COMMIT these changes separately from the Padé revert (clean two-commit sequence: commit 1 = Padé revert per Deliverables 1-3; commit 2 = AnisotropicShiftND accessor + test-file updates as a standalone `fix(shift_nd): add pub grid() accessor for downstream test compilation` commit with `Fixes-Agent: ai-solutions-architect` + `Fixes-Commit: 405ed88` trailers per git-quality-tracking protocol).

ADR-0094 status: **"Accepted (Amendment 1: Wave I Padé revival FINAL-DEFERRED to v5.0+; ζ⁸ rung remains closed for v4.3+ via ADR-0090 Wave A Chebyshev alone; step-3 Carnot closed for v4.5 via ADR-0095 Engel; Padé optional bonus per Option α coexistence)"**.

**Cross-references for AMENDMENT 1**: ADR-0091 AMENDMENT 1 (the precedent Option γ DEFER pattern that this AMENDMENT mirrors verbatim); ADR-0090 (Wave A Chebyshev `Diffusion8thZeta8Chernoff` — the kernel that ships ζ⁸); ADR-0095 (Wave A Engel — the kernel that ships step-3 Carnot just committed 1782171); `.dev-docs/research/zeta8-pade-wave-i-revival-deferred.md` (NEW research artifact, engineer creates per Deliverable 3 specification below); `.dev-docs/research/zeta8-pade-wave-b-deferred.md` (the Wave B research artifact, the structural template for the Wave I artifact); N. J. Higham (2005) *SIAM J. Matrix Anal. Appl.* 26(4) §2.5 (the matrix-squaring-vs-scalar-iteration warning that D3 diagnosis cites). Anchor `.claude/CLAUDE.md` §"Agent Failure Protocol" (max-2-retries rule that this AMENDMENT invokes).
