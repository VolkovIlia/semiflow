# ADR-0087 — Heisenberg Hypoelliptic Backend + `G_HORM_HEISENBERG` Gate (B5)

- **Status**: Accepted
- **Date**: 2026-05-28
- **Authors**: ai-solutions-architect
- **Decision-maker**: ai-solutions-architect
- **Related**: ADR-0077 (v3.1 base Hörmander spec — `VectorField<F, D>` + `HypoellipticChernoff<F, D, M>` + `KolmogorovPhaseSpace`); ADR-0078 (v3.1 B7 quantum graphs); v1.7.1 constitution Cohort 6 (`hormander.rs` 800-LoC HARD LIMIT carve-out).
- **Mathematical foundation**: math.md §28 AMENDMENT (this ADR) — Gaveau-Hulanicki real integral form for the Heisenberg sub-Laplacian heat kernel; closed-form pure-`f64` via 32-pt Gauss-Legendre on truncated symmetric interval.
- **Acceptance gates added**: `G_HORM_HEISENBERG` (RELEASE_BLOCKING — palindromic Strang-Hörmander slope ≤ −1.95 against Gaveau-Hulanicki oracle on `ℍ¹`); `T_HORM_HEISENBERG` (NORMATIVE sympy — 4 sub-checks on the integral closed form).
- **Schema bumps**: `properties.yaml` MINOR (+2 gate entries); `traits.yaml` UNCHANGED (no new public type — additive backend re-uses the v3.1 `HeisenbergGroup<F>` + `HypoellipticChernoff<F, 3, 2>` surfaces).

## Context

v3.1 (ADR-0077 §"Decision") shipped `HeisenbergGroup<F>` (`X₁ = ∂_x − ½y∂_t`, `X₂ = ∂_y + ½x∂_t`, `X₀ = 0`) as a CONSTRUCTIVE step-2 Carnot instance: it compiles, passes the bracket-rank checker (`[X₁, X₂] = ∂_t` spans the missing direction), but is **NOT** gated. The deferral rationale at v3.1 sign-off (math.md §28.4.B + §28.7 + §28.8) was: *"the Heisenberg sub-Laplacian fundamental solution involves complex-valued integrals (Beals-Gaveau-Greiner 1997); validation deferred to v4.0+ when SemiflowComplex is available"*.

`.dev-docs/research/verdicts/verdict-heisenberg.md` (Campaign 2, analysis mode, 2026-05-28) **falsifies** this assumption. The Gaveau-Hulanicki integral form for the Heisenberg heat kernel `p_h(x, y, t)` is *pure-real* — the `e^{iλt/h}` Fourier factor reduces to `cos(λt/h)` by even-odd symmetry under `λ ↔ −λ`; no `Complex<f64>` is required. The integrand decays super-exponentially (`λ/sinh(λh) ~ 2λe^{−|λ|h}`) and is captured at ≤ 10⁻¹⁰ by a **32-pt Gauss-Legendre quadrature on `[−25/h, +25/h]`** — the same engineering pattern already validated in v2.7 `resolvent_quad.rs` (Cohort N/A; 75-LoC quadrature-tables module). The Gaveau-Hulanicki formula is canonically cited in Beals-Gaveau-Greiner 1997 *Bull. Sci. Math.* 121 §3 (Part I, equation 3.1), originally derived independently by Hulanicki ~1976 and Gaveau 1977 (Beals-Greiner 1988 *Calculus on Heisenberg Manifolds* AMS Vol. 119 reproduces the explicit form; arxiv:math/0401243v2 Krötz-Thangavelu-Xu 2005 §1 gives an accessible online statement).

Closing the Heisenberg gate is a **net-positive paper-track artefact**: `docs/papers/hormander-paper-draft.md` §3 currently cites Kolmogorov + B7 quantum graphs as the validated step-2 Carnot instances; adding Heisenberg makes Heisenberg the **second numerically validated step-2 Carnot heat kernel** in a Rust library and strengthens the Festschrift §3 narrative.

## Decision

Ship `HeisenbergHypoelliptic` as an *additive* backend on the existing `HypoellipticChernoff<F, 3, 2>` surface in v3.x (post-v3.1.x PATCH or v3.2.0 MINOR — engineer Wave timing per release planner). The decision has three operative parts.

**Part 1 — `HypoellipticChernoff<F, 3, 2>` constructor on `HeisenbergGroup<F>` fields**. Add a NEW `impl<F: SemiflowFloat> HypoellipticChernoff<F, 3, 2> { pub fn new_heisenberg(...) -> Result<Self, SemiflowError> }` (or extend the existing `HypoellipticChernoff::<f64, 3, 2>::new` to accept the Heisenberg pair). Re-use the existing `HeisenbergX<F>` + `HeisenbergY<F>` + step-2 bracket-rank checker (`[X₁, X₂] = ∂_t`). NO change to the `VectorField<F, D>` trait. NO change to `HypoellipticChernoff` struct layout.

**Part 2 — Closed-form Gaveau-Hulanicki heat-kernel oracle in a NEW sibling module `heisenberg_kernel.rs`** (~150-200 LoC; stays well under the default 500-LoC cap — NO new Cohort needed). Public surface:
```rust
/// Gaveau-Hulanicki heat kernel for ℍ¹ at (x, y, t) at time h > 0.
///
/// Evaluates the canonical real-valued integral
///   p_h(x, y, t) = (1/(16π²h²)) · ∫_{-∞}^{∞} (λ/sinh(λh))
///                  · exp(−r²·λ·coth(λh)/(4h)) · cos(λt/h) dλ
/// where r² = x² + y². Truncated to [−25/h, +25/h] and evaluated via
/// 32-pt Gauss-Legendre quadrature; tail truncation error O(e^{−25}) ≈ 1e-11.
pub fn heisenberg_heat_kernel<F: SemiflowFloat>(h: F, x: F, y: F, t: F) -> F;
```
Internally uses `const GAUSS_LEGENDRE_32_NODES: [f64; 32]` + `const GAUSS_LEGENDRE_32_WEIGHTS: [f64; 32]` arrays (mirror `resolvent_quad.rs` `GL32_NODES` + `GL32_WEIGHTS` from v2.7); all `no_std + libm`-compatible (only `sinh`, `cosh`, `exp`, `cos` needed — all in `libm`). Pre-computed nodes/weights ship as `f64` constants; generic-over-F via `from_f64::<F>(...)` per ADR-0025.

**Part 3 — NEW RELEASE_BLOCKING gate `G_HORM_HEISENBERG`** (slope ≤ −1.95 against the Gaveau-Hulanicki oracle on a 3D Gaussian initial datum on `ℍ¹ = ℝ³`, sweep `n ∈ {16, 32, 64, 128}`) + NEW NORMATIVE sympy gate `T_HORM_HEISENBERG` (4 sub-checks: PDE residual, real-valuedness, normalization, step-2 Lie bracket). Test file: `crates/semiflow-core/tests/hormander_heisenberg_slope.rs` (NEW, ~100 LoC, mirror G28 structure verbatim). Sympy script: `scripts/verify_hormander_heisenberg.py` (NEW, ~250 LoC, mirror `verify_hormander_kolmogorov.py` + extend `lie_bracket_kit.py` for `[X₁, X₂] = ∂_t`).

## Consequences

**POSITIVE**: (a) closes the v3.1-deferred Heisenberg gate without v4.0 `SemiflowComplex`; (b) makes Heisenberg the second numerically validated step-2 Carnot heat kernel in `semiflow-core`; (c) strengthens `docs/papers/hormander-paper-draft.md` §3 (Festschrift §3 narrative); (d) re-uses the validated v2.7 `resolvent_quad.rs` quadrature-tables pattern (engineering risk: LOW); (e) ~250-300 LoC additive on a NEW sibling module + ~80 LoC on `hormander.rs` (well under Cohort 6 800-LoC cap; `hormander.rs` currently 597 LoC + ~80 = ~677 LoC, 123 LoC headroom remaining).

**NEGATIVE**: (a) adds 1 dependency on `libm::sinh/cosh/cos` (already in `semiflow-core`'s build budget — no new dep); (b) ~150-200 LoC quadrature-tables (NEW `heisenberg_kernel.rs`); (c) NEW sympy script (~250 LoC; same maintenance class as 5 existing `verify_*.py`).

**BREAKING**: NONE. Strictly additive on the public surface. `properties.yaml` schema bump is MINOR (additive gates). `traits.yaml` UNCHANGED. v3.1.0 `HeisenbergGroup<F>` rustdoc updates from "un-gated constructive instance" → "gated step-2 Carnot heat kernel" — non-breaking doc-only edit.

**Constitution impact**: NO change to v1.7.1 Cohort 6 cap (800 LoC `hormander.rs`). NEW sibling `heisenberg_kernel.rs` stays under the default 500-LoC cap (target ~200 LoC). The v3.2 split trigger recorded in v1.7.1 PATCH ("if hormander.rs exceeds 800 LoC at engineer Wave B/C completion, the backends WILL be split into hormander_kolmogorov.rs / hormander_heisenberg.rs") REMAINS in force; this ADR does NOT trigger it because the heat-kernel oracle goes to a sibling, not into `hormander.rs`.

## Implementation spec

See `.dev-docs/specs/heisenberg-wave.md` for the engineer Wave spec (acceptance criteria, file touch list, test plan, ~1-week engineering runway).

## AMENDMENT 1 (2026-05-28) — formula transcription correction

**Trigger**: First engineer Wave at commit `02ab970` (since reverted) carried over the AMENDMENT 1 Gaveau-Hulanicki formula from math.md §28 verbatim. `T_HORM_HEISENBERG.pde_residual` FAILED on the first probe `(h=0.1, x=y=t=0)` with `|∂_h p − Lp| ≈ 1.9 × 10⁴` (`dh ≈ -1.25e4`, `Lp ≈ -3.15e4`) — the kernel itself is off by ~100× scale. `G_HORM_HEISENBERG` slope = 0.0797 (essentially flat, NOT order-2) is downstream of the same root cause.

**Diagnosis**: Diagnostic agent (architect, 2026-05-28) cross-referenced the AMENDMENT 1 formula against Krötz-Thangavelu-Xu 2005 arxiv math/0401243 §2.2 + §4 (PDF extracted via pdftotext) and Boggess-Raich 2007 arxiv 0711.4117 Corollary 1.4. Verdict: AMENDMENT 1 formula contains **four independent transcription errors** (B1 prefactor `1/(16π²h²)` vs canonical `1/(4π²)`; B2 missing factor-of-2 in `sinh(λh)` → `sinh(2λh)`; B3 spurious `1/h` in exponent `coth(λh)/(4h)` → `coth(2λh)·(1/2)`; B4 spurious `1/h` in `cos(λt/h)` → `cos(λt)`). Source of error traced to `verdict-heisenberg.md` Q2.A, where the researcher quoted a half-finished re-derivation as "canonical form" without citing a verbatim primary-source equation number. Full bug analysis in `.dev-docs/research/heisenberg-formula-diagnostic.md` (§"Bug identification" + §"On-diagonal scale sanity check").

**Corrected formula** (math.md §28 AMENDMENT 2; Hulanicki form for the symmetric convention rescaled to the math.md operator-prefactor $L = \tfrac{1}{2}(X^2+Y^2)$):
```
p_h(x, y, t) = (1/(2π)²) · ∫_{-∞}^{+∞}
  (λ/sinh(λh/2)) · exp(−(λ/4)·coth(λh/2)·(x²+y²)) · cos(λt) dλ
```
**Derivation**: KTX 2005 arxiv math/0401243 eq (4.1.2) gives the kernel for $L_{\text{full}} = X^2+Y^2$ (no $\tfrac{1}{2}$) as `(1/(2π)²) ∫ (λ/sinh(λh)) exp(-(λ/4)coth(λh)r²) cos(λt) dλ`. For math.md's $L = \tfrac{1}{2}L_{\text{full}}$, we have $e^{hL} = e^{(h/2)L_{\text{full}}}$, so substituting $h \to h/2$ in the $L_{\text{full}}$ formula gives the boxed expression. Verified by mpmath at 4-digit accuracy:
- On-diagonal: $p_h(0,0,0) = 1/(2h^2)$ exactly.
- Off-diagonal heat equation: $\partial_h p / [\tfrac{1}{2}(X^2+Y^2) p] = 1.0001$ at $(h, x, y, t) = (0.1, 0.5, 0.5, 0)$.

**Primary citation**: Beals-Greiner 1988 *Calculus on Heisenberg Manifolds* AMS Studies 119 Theorem 5.18 (reproduces Hulanicki 1976 *Studia Math.* 56:165-173 explicit form for symmetric algebra; the operator-prefactor handling via $h \to h/2$ is standard for canonical-Hulanicki vs canonical-KTX formulations). The two cross-check sources (KTX 2005 and Boggess-Raich 2007) provide the asymmetric-algebra formula for the kernel-only verification.

**Impact on engineer Wave spec**:
- AC2 (closed-form heat kernel oracle): formula in `crates/semiflow-core/src/heisenberg_kernel.rs` MUST use the corrected formula. **Truncation cutoff changes**: from $\Lambda = 25/h$ (AMENDMENT 1, both under-samples bulk AND has wrong formula) to $\Lambda = 16/h$ (4-digit accuracy across the full probe set, including off-diagonal `cos(λt)` oscillation). 32-pt GL node count is unchanged. The `λ=0` removable-singularity guard value changes from `(1/h)·exp(-r²/(4h²))` (AMENDMENT 1, wrong) to `(2/h)·exp(-r²/(2h))` (AMENDMENT 2, correct: $\lambda/\sinh(\lambda h/2) \to 2/h$ and $(\lambda/4)\coth(\lambda h/2) \to 1/(2h)$ as $\lambda \to 0$).
- AC4 (sympy gate): `scripts/verify_hormander_heisenberg.py` `_gh_integrand` and `_gh_kernel` helpers MUST be updated per the corrected formula; `T_HORM_HEISENBERG PASS` is the BLOCKING precondition for re-introduction of the Rust `heisenberg_kernel.rs`.
- AC3 (G_HORM_HEISENBERG slope gate): structure unchanged; once the oracle is correct, the gate measures genuine Strang-Hörmander order-2 tangency (mirror G28 Kolmogorov).
- AC1/AC5/AC6 (constructor wiring, schema bumps, Cohort 6 budget check): UNCHANGED.

**Process gap identified**: no architect-side `verify_*.py` execution gate ran between the math.md §28 AMENDMENT 1 commit and the engineer-Wave delegation. Pipeline-level fix: for any future ADR introducing a closed-form kernel oracle, the architect MUST run the corresponding sympy `verify_*.py` script (with the proposed formula) and confirm PASS BEFORE delegating engineer Wave. This is hereby promoted from implicit-best-practice to NORMATIVE in math.md §28 AMENDMENT 2 (§"Re-execution requirement").

**Re-validation path**:
1. Engineer Wave 2 updates `scripts/verify_hormander_heisenberg.py` per `.dev-docs/specs/heisenberg-wave.md` Wave-2 amendment.
2. Confirm `python3 scripts/verify_hormander_heisenberg.py` prints `T_HORM_HEISENBERG PASS` (currently FAILS at `pde_residual` first probe).
3. Re-introduce `crates/semiflow-core/src/heisenberg_kernel.rs` + `hormander.rs` constructor extension + `tests/hormander_heisenberg_slope.rs` per AMENDMENT-2 formula.
4. Confirm `G_HORM_HEISENBERG` slope ≤ -1.95 in full-suite.
5. Confirm `wc -l crates/semiflow-core/src/hormander.rs` ≤ 800 (Cohort 6 budget unchanged).

## References

- `.dev-docs/research/verdicts/verdict-heisenberg.md` (researcher analysis-mode synthesis, 2026-05-28 — A-level evidence; decisive finding §"Statistical summary")
- `.dev-docs/research/raw-findings-heisenberg.md` (Campaign 2 raw data; 47 search results, 11 URLs)
- Beals, R., Gaveau, B., Greiner, P.C., "Complex Hamiltonian mechanics and parametrices for subelliptic Laplacians," *Bull. Sci. Math.* **121** (1997), Parts I-III pp. 1-36, 97-149, 195-259. — Canonical modern treatment (Part I §3 contains the explicit integral form)
- Hulanicki, A. (~1976), "The distribution of energy in the Brownian motion in the Gaussian field…," *Studia Math.* — Original explicit integral derivation (parallel to Gaveau)
- Gaveau, B., "Principe de moindre action, propagation de la chaleur, et estimées sous-elliptiques sur certains groupes nilpotents," *Acta Math.* **139** (1977), 95-153. — Independent original derivation
- Beals, R., Greiner, P.C., *Calculus on Heisenberg Manifolds*, Annals of Math. Studies **119**, Princeton University Press, 1988. — Comprehensive monograph reproducing the explicit form
- Krötz, B., Thangavelu, S., Xu, Y., arxiv:math/0401243v2 (2005), "The heat kernel transform for the Heisenberg group." — Accessible online statement (integral form in §1)
- Folland, G. B., "Subelliptic estimates and function spaces on nilpotent Lie groups," *Ark. Mat.* **13** (1975), 161-207. — Cited from `HeisenbergGroup<F>` rustdoc (left-invariant field structure)
- ADR-0077 (v3.1 base Hörmander spec — `VectorField<F, D>`, `HypoellipticChernoff<F, D, M>`)
- ADR-0069 (v2.7 Laplace-Chernoff resolvent — established the 32-pt Gauss-Laguerre / Gauss-Legendre quadrature pattern in `resolvent_quad.rs`)
- v1.7.1 PATCH constitution amendment — Cohort 6 `hormander.rs` 800-LoC HARD LIMIT (mirror Cohort 5 manifold.rs pattern)
