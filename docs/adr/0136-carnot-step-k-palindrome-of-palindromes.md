# ADR-0136 — F4 Step-k Carnot closure via recursive palindrome-of-palindromes

**Status:** ACCEPTED · **Date:** 2026-06-06 · **Shipped:** 2026-06-08 · **Branch:** `feat/v8.0.0-planning`
**Note (Amendment 2, operative):** STRONG-GO via complex-time order-4; supersedes Amendment 1 NARROW order-2 scope as the F4 deliverable; Amendment 1 order-2 baseline also ships.
**Theme:** v8.0.0 — Differentiable Chernoff (F4, RESEARCH-TRACK; order-4 complex-time CLOSURE of the order-≥3 case)
**Gate:** `G_CARNOT_CPLX3` (order-4 complex triple-jump, Amendment 2) + `G_CARNOT_STEP4` (order-2 real baseline, Amendment 1)
**Parent:** ADR-0132

## Context

The hypoelliptic `HypoellipticChernoff` kernel (v3.1.0, ADR-0077) achieves a palindromic Strang product formula for step-2 Carnot groups (Kolmogorov, Heisenberg). The Festschrift §3 problem (constructive hypoelliptic product formula for step-k ≥ 3) was declared OPEN at that time. The Engel group (step-3, depth-4 Lie algebra) was partially closed by the v4.5+ research wave (2026-05-29): a sympy-verified recursive construction achieves super-algebraic convergence (slope −43.95) for the specific Engel case, confirming that the palindrome idea extends beyond step-2. Contradiction C4: the step-k formula requires controlling depth-k brackets, which expand combinatorially and appear to demand a different product structure at each level — no known closed-form constructive formula exists for general k.

## Decision

Pursue the recursive "palindrome-of-palindromes" hypothesis: at step-k, treat any depth-2 iterated bracket `[Y, Z]` as a new effective vector field `W` (locally, in the exponential chart), then `[X, W]` is again a step-2 structure over {X, W} and the existing palindromic product applies recursively. The induction base is the Engel step-3 construction (verified). Each level of recursion adds one palindrome layer around the inner-level product; by TRIZ-7 (nesting / matryoshka), the depth-k product is the depth-(k−1) product embedded as a single "step" inside the next palindrome. TRIZ-1 (segmentation — factor the bracket hierarchy into separable depth-2 sub-problems) + TRIZ-7 (nesting) + recursion/self-similarity. The existing `VectorField<F, D>` trait and `HypoellipticChernoff` composition infrastructure serve as the implementation substrate. Gate `G_CARNOT_STEP4`: sympy-verified step-4 Carnot group (first non-Engel depth-5 case) self-convergence, slope ≤ −1.95, confirming the palindrome-of-palindromes tangency order is preserved under one level of induction. A proof sketch that nested palindromes preserve the tangency order must be recorded in math.md §28.bis before the gate is declared RELEASE_BLOCKING.

## Consequences

If the hypothesis holds, closes a recognized OPEN problem in sub-Riemannian analysis (journal-grade contribution to Russ. J. Math. Phys. or equivalent). Unlocks arbitrary sub-Riemannian / Carnot PDE: high-order kinetic Fokker-Planck, sub-elliptic control systems, stochastic mechanics on nilpotent groups. Highest-prestige direction in the v8.0 roadmap; also highest-risk — the induction step (that each palindrome-embedding preserves the full tangency order without leaving a residual BCH term at the next depth) requires a non-trivial Lie-algebra estimate and may fail for certain bracket configurations. If failed for general k, the result still publishes as a constructive theorem for specific nilpotent families (Engel, free step-3, etc.).

## Amendment 1 — F4 math-prerequisite spike ruling: NARROW (2026-06-07)

**Author:** ai-solutions-architect · **Witness:** `scripts/carnot_step4_kit.py` (`T_CARNOT_STEP4` PASS) · **Math:** math.md §28.bis.7

The v8.0.0 Phase-2 math-prerequisite spike for F4 ran a step-4 sympy oracle and reached a **NARROW** ruling. The original Decision/Consequences above are PRESERVED as the audit trail of the hypothesis under test; this amendment is the operative ruling.

**Crux findings (two independent symbolic witnesses):**
- **Witness A (step-independent, decisive).** Free-associative-algebra BCH of the palindromic Strang product `Ψ(τ)=e^{(τ/2)A} e^{τB} e^{(τ/2)A}`: the τ² coefficient of `log Ψ` is identically **0** (palindromic cancellation), and the τ³ coefficient is the standard non-zero combo `−1/24[A,[A,B]] + 1/12[B,[B,A]]`. Order-2 tangency is therefore an **algebraic identity for any A,B**, hence holds for sub-Laplacians of **any** Carnot step. *The Carnot step does not enter the order.*
- **Witness B (concrete step-4 filiform N=5).** Bracket chain `X₃=[X₁,X₂]`, `X₄=[X₁,X₃]`, `X₅=[X₁,X₄]`, filiform termination, Hörmander rank 5 — genuine step 4. The exact operator residual `F₅(τ)f − e^{(τ/2)L₅}f` on a generic degree-5 polynomial jet first disagrees at **τ³** (e.g. `−5/48·x₁x₃ − 11/48·x₄ − 1/24`), confirming global order **exactly 2**, slope **−2**.

**Ruling: NARROW.**
1. **The order-2 gate is achievable — by the FLAT horizontal palindromic Strang, not by nesting.** `G_CARNOT_STEP4` is declared RELEASE_BLOCKING **only** for the order-2 flat-horizontal palindromic Strang `F₅(τ)=e^{τX₁²/4}∘e^{τX₂²/2}∘e^{τX₁²/4}` on the filiform step-4 (N=5) group, self-convergence slope `≤ −1.95`.
2. **The "palindrome-of-palindromes" recursion is unnecessary for the order-2 gate** and is therefore NOT the mechanism the gate validates. The depth-k brackets are never exponentiated as legs; palindromic symmetry cancels their leading effect at τ² for any step. One level of nesting trivially preserves order 2 (composing order-2 methods) but is not a route to higher order.
3. **The general-k / order-≥3 closure stays OPEN.** The genuine Festschrift §3 open problem is a constructive order-≥3 hypoelliptic approximant. The τ³ coefficient (Witness A) is the obstruction; it is not cancelled by any extra palindromic wrap of the two existing legs. Reaching order ≥3 requires correction legs along bracket directions (Suzuki–Yoshida style) — that induction step is unproven for general k. **Escalated, mirroring the G_zeta4 precedent (ADR-0075); NOT declared closed.**
4. **v8.0.0 "all-blocking F4" scope is reduced** to the order-2 filiform step-4 family. The higher-order step-k direction is removed from the blocking scope and recorded as OPEN research.

**Honesty note.** The previously recorded Engel slope of ≈ −43.95 ("super-exponential") is an **artifact** of an origin-centred Gaussian IC (the polynomial flow coefficients `x₁, x₁²/2, x₁³/6` are small near the origin, so the τ³ residual nearly vanishes). The spike reproduced this artifact with a low-degree probe (residual ≡ 0 through τ⁴) then dispelled it with a generic degree-5 jet (residual ≠ 0 at τ³). Genuine order is **2**. Carnot self-convergence gates MUST use a generic, sufficiently-high-degree, non-origin-symmetric probe; otherwise they over-report the order. `G_HORM_ENGEL` (and `G_CARNOT_STEP4`) probe data should be reviewed against this note.

## Engineer spec (NARROW family — order-2 filiform step-4, N=5)

**File:** `crates/semiflow-core/src/carnot_stepk.rs` (NEW; do NOT author in this spike — engineer Wave). Strictly additive sibling; mirror `hormander_engel.rs` structure verbatim.

**Substrate (reuse, no trait changes):** `VectorField<F, 5>` + `HypoellipticChernoff<F, 5, 2>` + the §28.3/§28.bis flat palindromic Strang composition. Requires `GridND`/`GridFnND` at D=5 (already generic per `grid_nd.rs` / `grid_fn_nd.rs` — confirm D=5 instantiation compiles; if a fixed-D path is needed, mirror the Engel D=4 helpers).

**Public API (additive):**
- `Filiform5X1<F>`, `Filiform5X2<F>` implementing `VectorField<F,5>` per (28.bis.7a). X₁ flow trivial; X₂ flow polynomial: `(x₁, x₂+s, x₃+s·x₁, x₄+s²·x₁/2 + s·x₃, x₅+s³·x₁/6 + s²·x₃/2 + s·x₄)` — derive the exact integral curve in the engineer Wave and unit-test it against the field (a `T_FILIFORM5_FLOW` sympy sub-check is recommended; the flow is the only delicate piece, 5-coordinate coupling).
- `HypoellipticChernoff::<f64,5,2>::new_filiform5()` constructor (mirror `new_engel()`).
- Diffusive legs `exp(σXₖ²)` via 32-pt Gauss-Hermite quadrature (reuse Engel/Heisenberg GH32 constants).

**Gate `G_CARNOT_STEP4` (RELEASE_BLOCKING, narrow):** `crates/semiflow-core/tests/carnot_step4_slope.rs` (NEW; feature `slow-tests`; mirror `hormander_engel_slope.rs`). Probe-vs-2N self-convergence, `T=0.5`, sweep `n ∈ {16,32,64,128}`, OLS slope `≤ −1.95`. **IC requirement (per Amendment 1 honesty note): use a generic, NON-origin-symmetric, ≥degree-4 probe** (e.g. an off-centre anisotropic bump or a polynomial-times-Gaussian), NOT a centred isotropic Gaussian — otherwise the gate over-reports order. Grid: N=24 per axis (24⁵ ≈ 8M points × 8 B ≈ 64 MB/state; or N=16 → 8 MB if memory-constrained). Record observed slope verbatim; if slope ≫ −2, re-check the probe is non-degenerate (Amendment 1 artifact).

**Sympy CI:** add `scripts/carnot_step4_kit.py` (`T_CARNOT_STEP4`) to the sympy gate sweep (NORMATIVE, PASS at architect time). The architect-side PASS is the precondition for the engineer Wave (mirror `T_HORM_ENGEL_BRACKETS`).

**Out of scope (OPEN):** any order-≥3 construction; any recursive palindrome-of-palindromes nesting; any general-k claim. These remain the escalated open problem.

**Contracts deltas (engineer Wave):** `traits.yaml` MINOR (+`Filiform5X1/X2`, `new_filiform5`); `properties.yaml` MINOR (+`G_CARNOT_STEP4` narrow-scoped, +`T_CARNOT_STEP4`). Constitution: new `carnot_stepk.rs` under default 500-LoC cap (target ~280 LoC).

## Amendment 2 — F4 INVENTIVE re-spike ruling: STRONG-GO (complex-time order 4) (2026-06-07)

**Status:** STRONG-GO (operative ruling — supersedes Amendment 1's order-2-only scope as the F4 deliverable; Amendment 1's order-2 `carnot_stepk.rs` is retained as the cheaper real-state baseline). **Gate:** `G_CARNOT_CPLX3` (RELEASE_BLOCKING). **Author:** ai-solutions-architect (with `triz-inventive-solver`, mandatory). **Witness:** `scripts/carnot_complex_order3_kit.py` (`T_CARNOT_CPLX3` PASS 16/16). **Math:** math.md §28.bis.8. **Full verdict + ARIZ chain:** `.dev-docs/research/verdicts/verdict-v8-f4-complex-order3.md`.

**Context.** The user rejected settling for order-2 Strang and directed an inventive (TRIZ/ARIZ) re-spike. The Amendment-1 NARROW ruling correctly closed the *real-coefficient* "palindrome-of-palindromes" hypothesis as order-2 only. This amendment records the *complex-coefficient* escape that reaches order 4.

**The contradiction and its resolution (ARIZ).** ТП: a real splitting of order ≥ 3 must contain a negative substep (Sheng–Suzuki barrier), which runs the hypoelliptic semigroup backward-in-time — unbounded/ill-posed; but a bounded (all-forward) real splitting is capped at order 2. ФП: the substep duration must be simultaneously *forward/bounded* (Re > 0) and *order-raising* (carry the cancellation a negative step would). Resolution (надсистема — extend the coefficient field to ℂ; разрешение в структуре комплексного числа): split the duration `c = Re(c) + i·Im(c)` with **Re(c) > 0** (bounded, since the Carnot sub-Laplacian is analytic/sectorial) and **Im(c) ≠ 0** (supplies the cancellation). The decisive resource (ВПР) is **already in the library**: the `SemiflowComplex` substrate + complex Cayley + `GridFnND<C>` (ADR-0079/0127/0130). This is the Castella–Chartier–Descombes–Vilmart (2009) / Hansen–Ostermann (2009) escape, instantiated on a hypoelliptic generator.

**Decision.** Adopt the **complex triple-jump** `Ψ(τ) = S(γ⋆τ) ∘ S((1−2γ⋆)τ) ∘ S(γ⋆τ)`, where `S` is the shipped order-2 horizontal Strang and `γ⋆ = 0.32439640402017117 ∓ 0.1345862724908067·i` is the complex root of `2γ³+(1−2γ)³=0` with **Re(γ⋆)=0.324 > 0 AND Re(1−2γ⋆)=0.351 > 0** (both substeps forward-in-time). Symbolic crux (T_CARNOT_CPLX3): free-algebra τ²,τ³,τ⁴ coefficients all ≡ 0 at γ⋆ (order 4 via the symmetric even-order theorem); on the genuine step-4 filiform N=5 sub-Laplacian the operator residual first disagrees at τ⁵ (global order 4, slope −4) on a generic degree-10 jet, vs τ³ (order 2) for the real Strang control on the identical probe. **Step-independent** in the free algebra ⇒ applies to step-2/3/4/general-k via the corresponding shipped `S`.

**Consequences.** Constructively closes (at order 4, exceeding the order-≥3 threshold) the Festschrift §3 open problem for the analytic-hypoelliptic case, with all substeps bounded — a journal-grade contribution and the headline v8.0 F4 result. Costs: complex-valued grid state (2× memory, complex arithmetic) recovered to a real order-4 result via `Re(·)`; rides the existing complex substrate (zero new deps). Conditional convergence theorem (inherits the Galkin–Remizov tangency framework's complex-time/analytic-semigroup extension — CCDV 2009; gated empirically by `G_CARNOT_CPLX3`). The order-4 claim is a tangency-order statement verified symbolically; honesty caveats in math.md §28.bis.8 and the verdict.

**Gate `G_CARNOT_CPLX3` (RELEASE_BLOCKING).** Filiform N=5 complex triple-jump self-convergence, `T=0.5`, `n∈{8,16,32,64}`, generic non-origin high-degree IC, OLS slope `≤ −3.80` (order-4, 2.5% margin). Sympy precondition `T_CARNOT_CPLX3` PASS (architect 2026-06-07).

### Engineer spec (STRONG-GO — complex triple-jump, order 4)

**File:** `crates/semiflow-core/src/carnot_complex.rs` (NEW; additive; ~320 LoC; do NOT author this spike).

**Substrate (reuse, no trait changes):** `SemiflowComplex` (`complex.rs`, ADR-0079; `num_complex::Complex<f64>`); `GridFnND<C>` at `D=5, C=Complex<f64>` (confirm the ND generic instantiates for complex; if a hot path is f64-specialised, mirror Schrödinger Option B's complex grid wiring); the filiform-N5 horizontal Strang `S(s)` — the inner order-2 building block. Each complex leg `e^{c·s·X_k²}` is a **Gaussian convolution with complex variance** `c·s`, evaluable by the existing 32-pt Gauss–Hermite quadrature with the convolution weight `G_{cσ}(s)=(4π cσ)^{−1/2}·exp(−s²/(4cσ))` using `SemiflowComplex::sqrt`/`exp` (only the scalar `c` becomes complex vs the real kernel).

**Public API (additive):**
- `const GAMMA_STAR: Complex<f64> = Complex::new(0.32439640402017117, -0.1345862724908067)` — derive from the exact algebraic root of `2γ³+(1−2γ)³=0`; unit-test `2γ³+(1−2γ)³` ≈ 0 to ≤ 1e-14.
- `ComplexTripleJump<K>` adapter over an inner order-2 symmetric Chernoff map `K` (the filiform-N5 Strang). `apply_into(τ,…)` = `K.apply(γ⋆τ) ∘ K.apply((1−2γ⋆)τ) ∘ K.apply(γ⋆τ)` — **table-driven** fold over the const scale array `[γ⋆, 1−2γ⋆, γ⋆]` (no deep if/else; ~15-line fn).
- Real-output wrapper: `into_real()` returns `GridFnND<f64>` via `Re(·)` (the conjugate-pair average ⇒ real, order-4).

**Gate test:** `crates/semiflow-core/tests/carnot_cplx3_slope.rs` (NEW; `slow-tests`; mirror `hormander_engel_slope.rs`). Probe-vs-2N, `T=0.5`, `n∈{8,16,32,64}`, generic non-origin high-degree IC (NOT centred isotropic Gaussian — Amendment-1 honesty note), `N=16` per axis (≈16 MB/complex-state; `N=12` if memory-bound), OLS slope `≤ −3.80`. Failure ladder: `≤−3.80` PASS; `(−3.80,−2.85]` ship `experimental` order-3; `>−2.85` escalate.

**Build the order-2 real kernel first** (`carnot_stepk.rs`, Amendment 1) — it IS the inner `S` the triple-jump composes — then wrap with `ComplexTripleJump`. Gates `G_CARNOT_STEP4` (order-2 real) and `G_CARNOT_CPLX3` (order-4 complex) are independent.

**Contracts deltas (engineer Wave):** `traits.yaml` MINOR (+`ComplexTripleJump<K>`, +`GAMMA_STAR`, +real-output wrapper); `properties.yaml` MINOR (+`G_CARNOT_CPLX3`, +`T_CARNOT_CPLX3`). Suckless: functions ≤50 LoC, file ≤500 (target ~320), zero new deps (`num-complex` already direct, 3/3 cap). Blast radius: additive only (no existing symbol modified; d=1 = ∅).
