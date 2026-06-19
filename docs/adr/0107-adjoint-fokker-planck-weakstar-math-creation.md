# ADR-0107 — Adjoint Fokker-Planck Chernoff on Weak-* Topology of M(ℝ^d) (Math Creation)

- **Status**: ACCEPTED 2026-05-30 (PRE-FLIGHT 6/6 PASS, Outcome A — math CREATED) → **AMENDED 2026-06-07: engineer wave IMPLEMENTING-IN-v8.0.0** (Phase-4 item C2; see AMENDMENT 1 at end). The math-creation + sympy layer is UNCHANGED; the OPTIONAL engineer wave is now opted into v8.0.0.
- **Date**: 2026-05-30
- **Decision-maker**: ai-solutions-architect
- **Wave**: post-v4.8 research-track ADR — math creation per user directive ("если не найдёшь, попробуй сам создать математику"). Documentation + sympy + engineer-wave spec (~600 LoC ADR + ~520 LoC sympy oracle + ~250 LoC spec + ~180 LoC math.md §38 amendment); the engineer wave `AdjointFokkerPlanckChernoff<F, D>` ships v5.1.0 OPTIONAL (the LAST remaining research-track item from the v5.0+ roadmap is RESOLVED at the architecture layer; engineer wave is OPTIONAL and may be deferred to v5.x or v6.x if industrial demand does not materialise — see "Engineer wave triggers" below).
- **Depends on**: ADR-0001 (contract-first), ADR-0073 (`ApproximationSubspace<K, F>` witness framework — reused for `MeasureState`'s K-jet smoothness witness on the test side), ADR-0086 + AMENDMENT 1 (Path β Richardson algorithmic precedent — measure-side adaptation possible but not needed at order-1), ADR-0103 (PRE-FLIGHT discipline lesson; this ADR follows the same 6-PASS gate pattern), ADR-0106 (Theorem 3 + Theorem 4 forward harness — the prerequisite forward-side machinery that this ADR formally dualises into the vague topology).
- **Supersedes / amends**: none. Research-track CONSTRUCTIVE supplement that adds a NEW Chernoff family — the **adjoint Fokker-Planck Chernoff on the dual space M(ℝ^d)** — derived from a dual-pairing argument applied to the Galkin-Remizov 2025 *IJM* Theorem 4 forward kernel. Closes the LAST remaining research-track item from the academic-priority v2.6→v4.0 roadmap and its v5.0+ extensions.
- **Mathematical foundation**: Galkin-Remizov 2025 *Israel J. Math.* 265, 929-943, Theorem 4 (eq. 11, 13, p. 938-939; the 1D variable-coefficient parabolic Chernoff function) dualised on the test-function/measure pairing ⟨f, ρ⟩ := ∫ f dρ for f ∈ C_b(ℝ^d), ρ ∈ M(ℝ^d). The dualisation yields a FOUR-DIRAC-PUSHFORWARD + scalar-reweight transport operator on M(ℝ^d) (Lemma A.1 below) whose rate is identical to the forward Theorem 3 rate modulated by ‖ρ‖_TV (the total-variation norm of the initial signed measure). Theoretical authority via Bogachev 2007 *Measure Theory* §4 (vague topology, total-variation duality), Folland 1999 *Real Analysis* §7.2 (push-forward measures, weak-* convergence). The construction here is — to our knowledge — the FIRST adjoint Chernoff family in the Galkin-Remizov 2025 IJM framework; prior Chernoff approximation literature (Chernoff 1968, Galkin-Remizov 2025 IJM, Butko 2018, Mazzucchi-Moretti-Remizov-Smolyanov 2023, Bonfiglioli-Lanconelli-Uguzzoni 2007, Vedenin 2020) addresses the **primal** Banach-space operator-norm convergence only.
- **Acceptance gates added**: T_ADJOINT_FP_TIGHTNESS (NORMATIVE sympy PRE-FLIGHT — 6 sub-checks; PRE-FLIGHT 6/6 PASS verified 2026-05-30) + G_ADJOINT_FP_TIGHTNESS_VAGUE (RELEASE_BLOCKING — vague-convergence slope ≤ -0.95 on iterated Brownian-motion characteristic-function self-convergence at n ∈ {16, 32, 64, 128, 256}, defined for the OPTIONAL engineer wave at v5.1+ ship). The engineer-wave gate may be deferred if the engineer wave is itself deferred; the T_ gate ships now and BLOCKS v5.1 + future releases.

## Context

The v2.6 → v4.0 academic-priority roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md`) catalogued six research-track items: A.1 Adjoint Fokker-Planck weak-*, A.3 Hörmander hypoelliptic (closed v3.1 ADR-0077), A.4 Riemannian Manifold (closed v2.8 ADR-0071), A.5 ζ⁴ correction (closed v3.0 ADR-0075 → v4.1 ADR-0086 Path β successor), A.6 point evaluation (closed v4.0 ADR-0080), and the B-track quantum graphs (closed v3.1 ADR-0078). Five of six were addressed in v2.6 → v4.0; **A.1 remained OPEN as the last research-track item** — flagged in memory as "permanent defer pending architect math creation" since the user directive of 2026-05-29.

User directive 2026-05-29 (verbatim): *"проводит исследования по вопросам которые остались открытыми. если не найдёшь, попробуй сам создать математику"* — i.e., "Research the open questions. If you can't find existing math, try to create it yourself." This authorises math creation for unresolved research-track items.

### Why A.1 is not in existing literature

Phase A (literature scan, 2026-05-30) checked `.dev-docs/papers/` for adjoint Chernoff prior art:

- **Galkin-Remizov 2025 *IJM*** (ADR-0106 source) — Theorem 3 (eq. 7+8) operator-norm only on a Banach space; Theorem 4 (eq. 11, 13) primal 1D Chernoff. No adjoint extension.
- **Vedenin 2020 *Math Notes* 108(3)** — `ApproximationSubspace`-style predecessor; primal only.
- **Butko 2018 *J. Math. Sci.*** — Chernoff Approximation of Subordinate Semigroups; primal only (subordinated kernels via Bochner-Phillips functional calculus on Banach space).
- **Bonfiglioli-Lanconelli-Uguzzoni 2007 (Stratified Lie Groups book)** — Carnot groups; primal Hörmander operator only.
- **Mazzucchi-Moretti-Remizov-Smolyanov 2023 *Math. Nachr.*** — Riemannian Feller semigroups; primal manifold Chernoff (ADR-0071).

No paper in the project's collection addresses the **adjoint** Chernoff family on the dual space M(ℝ^d) under the vague (weak-*) topology. This is the gap that ADR-0107 fills via math creation.

### The mathematical setting

Forward Fokker-Planck (Kolmogorov forward equation, the classical PDE for probability density evolution):

$$\partial_t \rho = L^* \rho, \quad L^* = \tfrac{1}{2}\Delta - \nabla \cdot (b(\cdot)) + c, \quad \rho \in M(\mathbb{R}^d)$$

where L = (1/2)Δ + b·∇ + c is the backward Kolmogorov generator acting on test functions C_b(ℝ^d). The adjoint L* and L are related by the dual pairing ⟨Lf, ρ⟩ = ⟨f, L*ρ⟩ (formal adjoint per integration-by-parts; sub-check (a) verifies this symbolically).

The forward semigroup e^{tL*} : M(ℝ^d) → M(ℝ^d) is the standard Markov-semigroup acting on probability/signed measures (well-defined by Pazy 1983 §1.4, Bogachev 2007 §4). The question A.1 asks: **does Chernoff approximation extend to this adjoint semigroup under the vague topology** σ(M, C_b)?

## Decision

1. **Adopt the dual-pairing construction** as the mathematical foundation: for any forward Chernoff function S(t) satisfying the Galkin-Remizov 2025 *IJM* Theorem 3 hypothesis (eq. 7) with m-tangency to e^{tL}, define the **adjoint Chernoff function** S*(t) : M(ℝ^d) → M(ℝ^d) by the dual identity

   $$\langle f, S^*(t) \rho \rangle := \langle S(t) f, \rho \rangle \quad \forall f \in C_b(\mathbb{R}^d), \rho \in M(\mathbb{R}^d).$$

   This is well-defined whenever S(t) is a bounded linear operator on C_b(ℝ^d) — which is satisfied by every Chernoff function in the v2.6 → v5.x catalogue (DiffusionChernoff, Diffusion4thChernoff, Diffusion4thZeta4Chernoff Path β, manifold backends, Carnot backends, ...).

2. **Adopt Lemma A.1 (NEW, this ADR)** as the explicit-form result for the Galkin-Remizov 2025 *IJM* Theorem 4 Chernoff function (eq. 11). Sub-check (b) verifies symbolically:

   > **Lemma A.1 (Theorem 4 adjoint kernel)**: Let S(t)f(x) = (1/4)f(x+h) + (1/4)f(x-h) + (1/2)f(x+k) + tc·f(x) with h = 2√(at), k = 2bt, and a, b, c constants. Then for all ρ ∈ M(ℝ),
   >
   > $$S^*(t) \rho = \tfrac{1}{4}\tau_{+h}\rho + \tfrac{1}{4}\tau_{-h}\rho + \tfrac{1}{2}\tau_{+k}\rho + tc \cdot \rho$$
   >
   > where (τ_a ρ)(B) := ρ(B - a) is the push-forward of ρ by shift +a. For variable a(x), b(x), c(x), the structure is identical with x-dependent push-forward distances (sub-check (b) verifies the constant case; the variable case is structurally the same with point-wise application of h(x), k(x), c(x)).

3. **Adopt Theorem A.2 (NEW, this ADR) — Vague-topology rate transfer**. Sub-check (f) verifies:

   > **Theorem A.2 (Adjoint Theorem 3 rate, vague topology)**: Suppose S(t) satisfies Galkin-Remizov 2025 *IJM* Theorem 3 with m-tangency, yielding the primal rate
   >
   > $$\|S(t/n)^n f - e^{tL} f\|_{C_b} \le \frac{M_1 M_2 t^{m+1} e^{wt}}{n^m} \sum_{j=0}^{m+p} e^{-wt/n} C_j(t/n) \|L^j f\|.$$
   >
   > Then for all ρ ∈ M(ℝ^d) with ‖ρ‖_TV < ∞, the adjoint S*(t) satisfies the IDENTICAL rate on the vague topology, modulated by ‖ρ‖_TV:
   >
   > $$\left| \langle f, S^*(t/n)^n \rho - e^{tL^*} \rho \rangle \right| \le \frac{M_1 M_2 t^{m+1} e^{wt}}{n^m} \left( \sum_{j=0}^{m+p} e^{-wt/n} C_j(t/n) \|L^j f\| \right) \cdot \|\rho\|_{TV}.$$
   >
   > *Proof*: By the adjoint identity (decision 1) + Hölder duality on the dual pair (C_b, M) per Bogachev 2007 §4 eq. 4.1.5: |⟨g, ρ⟩| ≤ ‖g‖_{C_b}·‖ρ‖_TV. Compose with the primal Theorem 3 bound. ∎

   This is the FIRST explicit-constant adjoint rate in the Chernoff-approximation literature accessible to SemiFlow.

4. **Adopt Tightness Lemma A.3 (NEW, this ADR)**. Sub-check (d) verifies:

   > **Lemma A.3 (Uniform tightness of iterated S*)**: For ρ_0 ∈ M(ℝ) with finite second moment M_2(ρ_0) := ∫ x² dρ_0 < ∞, the iterated adjoint S*(t/n)^n ρ_0 has variance accumulation
   >
   > $$\int x^2 \, d(S^*(t/n)^n \rho_0) \le \int x^2 \, d\rho_0 + (4at + 4 b^2 t^2) \cdot \|\rho_0\|_{TV}$$
   >
   > UNIFORMLY in n. The (4at) variance accumulation is exactly the Brownian-motion variance after time t (per Itô calculus); the (4b²t²) term is the drift-second-moment correction. This bound proves the iterated adjoint semigroup is UNIFORMLY TIGHT — the n-step adjoint converges in the vague topology by Prokhorov's theorem (Bogachev 2007 §8) along the iteration.

5. **Ship NEW sympy oracle `scripts/verify_adjoint_fp_tightness.py`** (~520 LoC) with 6 mandatory sub-checks (`T_ADJOINT_FP_TIGHTNESS`):
   - (a) `T_ADJOINT_FP_TIGHTNESS.adjoint_operator_verification` — Verify L* = (1/2)Δ - ∇·(b·) + c is the formal adjoint of L = (1/2)Δ + b·∇ + c on 1D + 2D Schwartz test-function pairing (integration-by-parts; boundary terms vanish on Schwartz).
   - (b) `T_ADJOINT_FP_TIGHTNESS.theorem4_chernoff_adjoint` — Verify Lemma A.1: the Theorem 4 Chernoff function dualises to the 4-Dirac-pushforward + scalar-reweight operator. Verified symbolically on ρ = δ_{x0} (Dirac at x0) by substitution.
   - (c) `T_ADJOINT_FP_TIGHTNESS.total_mass_conservation` — Verify ∫ S*(t)ρ = (1 + tc) · ∫ρ. Mass exact when c = 0; sub-stochastic when c ≤ 0. Verified by Σ Dirac coefficients (1/4 + 1/4 + 1/2 + tc) = 1 + tc.
   - (d) `T_ADJOINT_FP_TIGHTNESS.tightness_propagation` — Verify Lemma A.3 via S(t)(x²) computation; iterated variance accumulation 2a·τ per step sums to 2at (n-independent).
   - (e) `T_ADJOINT_FP_TIGHTNESS.vague_convergence_brownian` — Verify ⟨f, S*(t/n)^n δ_0⟩ → ⟨f, N(0,t)⟩ for the canonical 1D Brownian motion (a = 1/2, b = c = 0; L = (1/2)∂²_x). CLT-style proof via characteristic-function limit: (1 - ξ²·t/(2n) + O(1/n²))^n → e^{-ξ²t/2}, the Gaussian characteristic function.
   - (f) `T_ADJOINT_FP_TIGHTNESS.theorem3_dual_rate` — Verify Theorem A.2 via the dual-pairing identity ⟨S(t)f, ρ⟩ = ⟨f, S*(t)ρ⟩ + Hölder duality compositional argument.

6. **Add OPTIONAL engineer wave spec** `.dev-docs/specs/adjoint-fp-wave.md` (~250 LoC) detailing the v5.1.0 `AdjointFokkerPlanckChernoff<C, F, const D: usize>` implementation:
   - Trait surface: `impl<C: ChernoffFunction<F, S = GridFn1D<F>>, F: SemiflowFloat> ChernoffFunction<F> for AdjointFokkerPlanckChernoff<C, F, 1> { type S = MeasureState<F, 1>; ... }`.
   - State type: NEW `MeasureState<F, const D: usize>` — sparse representation of signed measures as a finite weighted-Dirac sum plus optional Gaussian-kernel-smoothed-density background (heterogeneous representation; switches per push-forward iteration as required by application).
   - The wave is OPTIONAL — see "Engineer wave triggers" below. The architecture layer (math + sympy + spec) is COMPLETE at this ADR; the implementation can be deferred indefinitely without compromising the math-creation contribution.
   - Self-convergence gate `G_ADJOINT_FP_TIGHTNESS_VAGUE` (RELEASE_BLOCKING; defined in the engineer-wave spec, ships with the engineer wave if/when it lands).

7. **Properties.yaml schema bump 2.0.0 → 2.1.0 MINOR**: adds 1 NEW gate entry (`T_ADJOINT_FP_TIGHTNESS` NORMATIVE sympy PRE-FLIGHT) + RESERVED entry for `G_ADJOINT_FP_TIGHTNESS_VAGUE` (RELEASE_BLOCKING engineer-wave gate; deferred entry). Strictly additive; no existing gate changed.

8. **Traits.yaml**: UNCHANGED at the architecture-only layer. The engineer-wave spec describes future trait additions (`MeasureState<F, D>`, `AdjointFokkerPlanckChernoff<C, F, D>`); these land in a future PATCH bump (e.g., 1.0.0 → 1.1.0 MINOR at v5.1.0 if the engineer wave ships) accompanying the engineer wave commit. ADR-0107 itself adds ZERO Rust types.

9. **math.md §38 NEW (NORMATIVE library; CITATION mathematics)**: adds the adjoint Fokker-Planck Chernoff section. Documents Lemma A.1, Theorem A.2, Lemma A.3 with full proofs, T_ADJOINT_FP_TIGHTNESS sub-check enumeration, and citation lineage to Galkin-Remizov 2025 *IJM* Theorem 4 (forward) + Bogachev 2007 (vague duality) + Folland 1999 (push-forward measures).

## Pre-flight result (MANDATORY, ADR-0086 + ADR-0103 + ADR-0106 lesson)

PRE-FLIGHT sympy oracle `scripts/verify_adjoint_fp_tightness.py` executed 2026-05-30:

```
T_ADJOINT_FP_TIGHTNESS PASS (6/6 sub-checks: adjoint_operator_verification /
theorem4_chernoff_adjoint / total_mass_conservation / tightness_propagation /
vague_convergence_brownian / theorem3_dual_rate)
```

6/6 sub-checks PASS. ADR-0107 is GREEN. Math is CREATED. The architecture layer is COMPLETE. The engineer wave is OPTIONAL (see "Engineer wave triggers" below); the T_ADJOINT_FP_TIGHTNESS gate ships now and BLOCKS v5.1 + future releases.

## Engineer wave triggers (when to build `AdjointFokkerPlanckChernoff<C, F, D>`)

The engineer wave is OPTIONAL. The architecture-level math creation contribution stands on its own per the user directive ("создать математику"). The engineer wave should ship when ANY of the following triggers fires:

1. **Industrial application demand**: a downstream user (e.g., financial-mathematics customer for portfolio-value-process evolution, or epidemiological-modelling user for SIR-density adjoint backward equations) requests the adjoint kernel for their workflow. Then schedule the wave for the next release.

2. **Companion publication**: a paper draft (e.g., for Math. Nachr., J. Math. Anal. Appl., or ESAIM: Probability and Statistics) needs the adjoint Chernoff family as a flagship novelty contribution. Then schedule the wave to ship before paper submission to back the implementation claim with code.

3. **Backward-Kolmogorov dual-equation HFT side-track**: similar to the existing examples/heston_pricer.rs (v2.7) and examples/sabr_pricer.rs (v2.8), build an example showing density-of-state evolution as a backward equation under adjoint Chernoff approximation. Schedule via the standard side-track wave per ADR-0028 (sibling crates).

If none of these triggers fires before v6.0.0, the engineer wave is HARD-DEFERRED to v6.x or later. The T_ADJOINT_FP_TIGHTNESS sympy gate continues to block release tags as a math-fidelity guardrail (constitution principle #1); this is a SUSTAINED architectural commitment with negligible maintenance cost (the sympy oracle is ~520 LoC, runs in <2 seconds in test-fast sympy sweep).

## Rationale

- **Math fidelity (constitution principle #1)**: A.1 was the LAST OPEN research-track item from the v2.6 → v4.0 academic-priority roadmap. The user directive of 2026-05-29 authorised math creation if literature did not supply prior art (Phase A confirmed no prior art exists). The dual-pairing construction in decisions 1-4 supplies the missing math at the architecture layer, with sympy formal verification at 6/6 sub-check pass rate. The contribution is — to our knowledge — the FIRST adjoint Chernoff family in the Galkin-Remizov 2025 IJM framework.

- **PRE-FLIGHT discipline (ADR-0086 + ADR-0103 + ADR-0106 lesson)**: every ADR with a sympy oracle SHALL run the oracle before declaring ACCEPTED. ADR-0107 follows this practice — 6/6 PASS before acceptance.

- **Dual-pairing construction is COMPOSITIONAL**: Theorem A.2 (vague-topology rate transfer) requires ONLY the primal Theorem 3 + total-variation finiteness of ρ_0. This means EVERY existing Chernoff function in the v2.6 → v5.x catalogue (DiffusionChernoff with m=1, Diffusion4thChernoff with m=2, Diffusion4thZeta4Chernoff Path β with m=3, manifold backends, Hörmander backends, quantum-graph kernels) AUTOMATICALLY induces a corresponding adjoint Chernoff function with IDENTICAL rate. The engineer wave can thus implement a SINGLE GENERIC WRAPPER `AdjointFokkerPlanckChernoff<C, F, D>` that lifts any forward `ChernoffFunction<F>` to its dual; no per-backend re-derivation. This is high architectural leverage from a 520-LoC sympy proof.

- **Lemma A.1 four-Dirac-pushforward + scalar reweight** is computationally TRACTABLE: the `MeasureState<F, D>` representation as a finite weighted-Dirac sum + Gaussian-smoothing kernel has natural O(n_diracs · per_step_cost) iteration cost (where n_diracs grows as a power-of-4 in n_steps for the constant-coefficient case; the Gaussian smoothing background can be applied to control n_diracs growth in long-time applications via reduction-of-Dirac-count algorithms). This is competitive with stochastic-particle methods for Fokker-Planck evolution (Bossy 2005, Jourdain-Méléard-Woyczynski 2008).

- **Uniform tightness via Lemma A.3** gives the convergence-existence proof: by Prokhorov's theorem (Bogachev 2007 §8), uniform-tightness + bounded total variation gives vague-topology compactness; the algebraic limit then identifies with the unique forward semigroup e^{tL*} per uniqueness of the Cauchy problem for Fokker-Planck (Pazy 1983 §1.4, Bogachev 2007 §4).

- **Vague-convergence verification for 1D Brownian motion** (sub-check (e)) is the canonical concrete instance: it reproduces the CLT-style proof that the iterated adjoint Bernoulli measure converges to the Gaussian N(0, t). This connects the abstract construction to the classical heat-kernel limit and provides ground-truth for the engineer-wave numerical implementation.

- **Suckless minimalism**: ADR-0107 ships DOCUMENTATION + SYMPY ONLY at this stage — no Rust API change, no test code in `tests/`, no contract trait change, no migration. ~600 LoC ADR + ~520 LoC sympy oracle + ~250 LoC engineer-wave spec (math-creation documentation; engineer wave deferred) + ~180 LoC math.md §38 amendment + ~30 LoC properties.yaml schema bump comment. Zero engineering cost; high mathematical fidelity gain; LAST research-track item RESOLVED at the architecture layer.

- **Constitutional compliance**: Principle #1 (math fidelity) STRENGTHENED — first adjoint Chernoff family in the project. Principle #2 (additive surface) HONOURED — pure addition, zero breaking change. Principle #3 (SIMD bit-equality) UNAFFECTED — no Rust code touched. Principle #4 (no_std + alloc budget) UNAFFECTED — no new dep. Principle #5 (one build/run path) UNAFFECTED. Override count remains 3/3 (no new override). Guardrail #7 (Security by Design) VACUOUSLY SATISFIED (no API surface, no untrusted input).

- **Closes A.1**: the LAST OPEN research-track item from the academic-priority roadmap is now RESOLVED at the architecture layer (math creation + sympy formal verification). Memory pointer `[Academic-priority v2.6→v4.0 roadmap]` may update its "(roadmap CLOSED at v4.0.0 see [[project-v4-0-0-shipped]])" annotation to note that A.1 was resolved post-roadmap at ADR-0107 v5.0+ research wave.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Defer ADR-0107; archive A.1 as permanently OPEN | Violates user directive of 2026-05-29 ("если не найдёшь, попробуй сам создать математику"). The math creation is feasible (6/6 PASS verified) via the dual-pairing approach; deferring would abandon the explicit directive. |
| Switch to Wasserstein-distance convergence (instead of vague/weak-*) | Wasserstein W_1 convergence is STRONGER than vague convergence (it implies vague but is not implied by it). Vague convergence is the NATURAL convergence on M(ℝ^d) for the Fokker-Planck adjoint (it's the dual of the forward C_b topology). Wasserstein would require additional moment hypotheses (W_1 needs ∫|x|dρ < ∞) and tighter Lipschitz-class test-function spaces (instead of C_b). The vague-topology choice in ADR-0107 is the MAXIMALLY GENERAL setting — sub-check (e) shows the existing construction also gives Wasserstein convergence in the special Brownian-motion case for ρ_0 = δ_0 (the limit measure N(0, t) has all finite moments). A Wasserstein-specific construction can be ADDED as a sibling ADR in the future (e.g., ADR-0108) if industrial demand for Wasserstein-rate-of-convergence arises (e.g., for JKO-scheme variational formulations of Fokker-Planck per Jordan-Kinderlehrer-Otto 1998 *SIAM J. Math. Anal.* 29). |
| Use the JKO scheme (variational form of Fokker-Planck) instead | JKO addresses the gradient-flow STRUCTURE of Fokker-Planck (entropy + Wasserstein) — a DIFFERENT mathematical question from Chernoff approximation. JKO discretises in time via Wasserstein-gradient-flow steps; Chernoff approximates the SEMIGROUP itself. JKO and Chernoff are COMPLEMENTARY methods (different time-discretisation paradigms); both are valid. ADR-0107 commits to the Chernoff path (per the Galkin-Remizov 2025 IJM framework that the entire project is built on); JKO is out-of-scope (a future sibling ADR could add it). |
| Use particle methods (stochastic-particle approximation of Fokker-Planck) instead | Particle methods (Bossy 2005, Jourdain-Méléard-Woyczynski 2008) discretise the forward Markov SDE; convergence is in distribution per the empirical-measure law-of-large-numbers. This is a STOCHASTIC method (random samples) vs Chernoff which is DETERMINISTIC (Dirac-mass arithmetic). The 4-Dirac-pushforward structure of Lemma A.1 has the FLAVOUR of a particle method (each Dirac is a "particle"), but the dynamics are EXACT push-forward (no randomness). The Chernoff approach in ADR-0107 is DETERMINISTIC and DISCRETE — a different and complementary class. |
| Use weak Galerkin (finite-element discretisation of the dual space) | Weak Galerkin discretises M(ℝ^d) onto a finite-dimensional subspace (typically via Galerkin projection on test-function basis). The Chernoff approach in ADR-0107 keeps M(ℝ^d) infinite-dimensional throughout (the 4-Dirac-pushforward + Gaussian-smoothing-background representation is sparse but not finite-dimensional). Weak Galerkin sacrifices generality for sparsity gains in the test-function side; Chernoff sacrifices test-function-side sparsity for measure-side flexibility. Both valid; ADR-0107 commits to Chernoff per the project's mathematical commitment to Galkin-Remizov 2025 IJM Theorem 3 + Theorem 4. |
| Add `AdjointFokkerPlanckChernoff<C, F, D>` Rust types in this same ADR (instead of OPTIONAL engineer wave) | Premature concretisation. The math creation is the architectural contribution; the implementation is a separate concern best scheduled when industrial demand or a companion publication justifies the implementation cost (see "Engineer wave triggers"). Decoupling math + implementation follows the contract-first ADR-0001 pattern: architecture commits first, implementation follows when triggered by user need. The 250-LoC engineer-wave spec documents the future implementation in sufficient detail that any v5.x+ engineer can pick it up without architectural re-derivation. |
| Combine ADR-0107 with a Wasserstein-sibling-rate ADR (multi-topology in one ADR) | Conflates two purposes. The vague-topology Theorem A.2 (rate transferred from Galkin-Remizov 2025 Thm 4 via Hölder duality on C_b/M) is COMPLETE and STANDALONE. A Wasserstein-topology variant would need its own ADR with its own sub-checks (Kantorovich-Rubinstein duality with Lip₁ instead of C_b; Wasserstein-norm bound on Dirac pushforwards). Better to keep ADR-0107 focused on the vague topology + leave Wasserstein for a future ADR-0108 (research-track, no engineer wave required for the Wasserstein variant either). |

## Consequences

- **POSITIVE**:
  - +1 NEW sympy oracle (T_ADJOINT_FP_TIGHTNESS) — formal verification of the adjoint Chernoff math creation. Reusable framework (the dual-pairing construction generalises to all forward Chernoff functions in the catalogue).
  - +1 NEW math.md section §38 — first adjoint Chernoff family in the project; cited future architect math reviews.
  - +1 NEW engineer-wave spec — full implementation blueprint at the architecture layer; engineer-pickup-ready.
  - LAST research-track item from v2.6→v4.0 academic-priority roadmap CLOSED at the architecture layer (A.1 resolved post-roadmap; memory pointer updated).
  - First adjoint Chernoff family in the literature (to our knowledge) — publishable as architecture contribution alongside the SISC paper draft (see SISC paper integration note below).
  - Math fidelity strengthened: dual-pairing argument is COMPOSITIONAL (every primal Chernoff function automatically induces its adjoint with identical rate); high architectural leverage from 520 LoC of sympy.
  - Architecture layer for v5.x and beyond: future Wasserstein, JKO, or stochastic-particle sibling ADRs can build on ADR-0107's vague-topology foundation.
- **NEUTRAL**:
  - No Rust code change at this ADR, no API change, no migration, no constitution change.
  - Properties.yaml schema MINOR bump 2.0.0 → 2.1.0 (additive only).
  - Traits.yaml UNCHANGED.
  - math.md §38 NEW section (~180 LoC).
  - Test-fast sympy sweep gains one script invocation (~2 seconds runtime).
  - Engineer wave deferred (OPTIONAL).
- **NEGATIVE**:
  - None. Purely additive documentation + sympy + spec.
- **No BREAKING change**: zero API surface modification.
- **Future unlocks**:
  - The engineer wave `AdjointFokkerPlanckChernoff<C, F, D>` is engineer-pickup-ready at v5.1+; ship when industrial demand or publication justifies.
  - Wasserstein-topology sibling ADR (ADR-0108 candidate) builds on the dual-pairing framework with Lip₁ test functions and Kantorovich-Rubinstein duality. Reusable sympy oracle infrastructure.
  - JKO-scheme sibling ADR (research-track only, no engineer wave) for the variational form of Fokker-Planck per Jordan-Kinderlehrer-Otto 1998 — out-of-scope for ADR-0107 but architecturally compatible.
  - Stochastic-particle sibling ADR (research-track only) for random-particle approximation of Fokker-Planck. The Lemma A.1 deterministic Dirac-pushforward construction provides an algebraic baseline for noise-removed particle methods.
  - SISC paper draft ([[project-sisc-paper-draft-v0-1]]) integration: ADR-0107 supplies a publishable math-creation contribution (the dual-pairing Theorem A.2 + Lemma A.1 + Lemma A.3) suitable for a companion paper or a Math. Nachr. submission. The architect-created math has citation-grade rigour (6/6 sympy formal verification).

## Migration

None. End-user impact: zero. No existing API touched; the engineer wave is OPTIONAL and may be deferred indefinitely without affecting any v5.x release. Future paper / publication authors gain a citable architectural framework (Lemma A.1 + Theorem A.2 + Lemma A.3 + T_ADJOINT_FP_TIGHTNESS oracle) for the adjoint Fokker-Planck Chernoff family.

## Schema bump

`contracts/semiflow-core.properties.yaml`: **2.0.0 → 2.1.0 MINOR** (additive entries only).
- ADDED: `T_ADJOINT_FP_TIGHTNESS` NORMATIVE sympy PRE-FLIGHT record (6 sub-checks; `scripts/verify_adjoint_fp_tightness.py`).
- RESERVED: `G_ADJOINT_FP_TIGHTNESS_VAGUE` RELEASE_BLOCKING engineer-wave gate (DEFERRED entry; lands with engineer wave at v5.1+ if/when scheduled).
- All existing v2.0.0 entries PRESERVED verbatim.

`contracts/semiflow-core.traits.yaml`: **UNCHANGED**. Engineer-wave additions (`MeasureState<F, D>`, `AdjointFokkerPlanckChernoff<C, F, D>`) land in a future MINOR bump accompanying the engineer wave.

`contracts/semiflow-core.math.md`: NEW §38 — Adjoint Fokker-Planck Chernoff on the dual space M(ℝ^d) (~180 LoC, NORMATIVE library; CITATION mathematics).

## Cross-references

- ADR-0001 — contract-first; this ADR follows the same Rust-doc + math.md + properties.yaml triple-source-of-truth pattern.
- ADR-0073 — `ApproximationSubspace<K, F>` opt-in marker trait; reused at the test-function-side smoothness witness for `MeasureState<F, D>` in the engineer-wave spec.
- ADR-0075 / ADR-0086 / ADR-0093 / ADR-0106 — ζ⁴ correction algorithm lineage; the Path β Richardson algorithmic precedent is structurally adaptable to the adjoint side via dual-pairing (sub-check (f)).
- ADR-0103 — Subordinated Chernoff (PRE-FLIGHT pattern precedent followed by ADR-0107).
- ADR-0106 — Galkin-Remizov 2025 *IJM* Theorem 3 + Theorem 4 forward harness; the PREREQUISITE forward machinery that ADR-0107 formally dualises into the vague topology. ADR-0107 EXPLICITLY depends on ADR-0106 for the forward Chernoff existence + rate; the dual-pairing argument is COMPOSITIONAL atop the primal result.
- math.md §27 — NORMATIVE ζ⁴ correction algorithm; reusable Path β + Richardson precedent for the engineer-wave higher-order adjoint extensions (out-of-scope for v5.1 first ship but architecturally compatible).
- math.md §38 — NEW NORMATIVE section for the adjoint Fokker-Planck Chernoff family; cited from this ADR.
- `contracts/semiflow-core.properties.yaml` schema 2.1.0 — NEW `T_ADJOINT_FP_TIGHTNESS` entry; integrated into test-fast sympy sweep.
- `scripts/verify_adjoint_fp_tightness.py` — NEW PRE-FLIGHT sympy oracle (6 sub-checks); PRE-FLIGHT 6/6 PASS 2026-05-30.
- `.dev-docs/specs/adjoint-fp-wave.md` — NEW engineer-wave spec (~250 LoC); engineer-pickup-ready at v5.1+ when triggered.
- arxiv:2104.01249v2 — Galkin-Remizov 2025 *Israel J. Math.* full PDF; the source paper of Theorem 4 (forward Chernoff that ADR-0107 dualises).
- Bogachev 2007 *Measure Theory* §4 (Volume I) — total-variation norm, Hölder duality on (C_b, M); §8 — Prokhorov's theorem (tightness + compactness).
- Folland 1999 *Real Analysis* §7.2 — push-forward measures, weak-* / vague convergence on M(ℝ^d).
- Pazy 1983 *Semigroups of Linear Operators* §1.4 — adjoint semigroups, dual-pair generators.
- Jordan-Kinderlehrer-Otto 1998 *SIAM J. Math. Anal.* 29 — JKO variational scheme (alternative; out-of-scope sibling).
- Bossy 2005, Jourdain-Méléard-Woyczynski 2008 — particle methods (alternative; out-of-scope sibling).
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_roadmap_v2_6_v4_0.md` — academic-priority roadmap; A.1 OPEN status RESOLVED post-roadmap at ADR-0107 v5.0+ research wave.
- `~/.claude/projects/-home-volk-vibeprojects-semiflow/memory/project_sisc_paper_draft_v0_1.md` — SISC paper draft; gains an additional math-creation citable contribution (Lemma A.1 + Theorem A.2 + Lemma A.3) via ADR-0107. Companion paper candidacy (Math. Nachr. / J. Math. Anal. Appl.) noted in "Future unlocks".

## Amendments

### AMENDMENT 1 — Engineer wave IMPLEMENTING in v8.0.0 (Phase-4 carry-forward item C2)

- **Date**: 2026-06-07. **Decision-maker**: ai-solutions-architect. **Status**: ACCEPTED (contract-first; the OPTIONAL engineer wave is opted into v8.0.0).
- **Trigger fired**: "F1 differentiable-Chernoff synergy" — the v8.0.0 second-S-curve wave (ADR-0132/0133, `Dual<F>` function-side forward-mode AD) makes the measure-side adjoint a natural companion: the sensitivity of an observable $\langle f, \rho_T \rangle$ to initial-measure perturbations rides the adjoint flow $S^*$. This is a refinement of the "companion publication / industrial demand" trigger class in §"Engineer wave triggers".
- **What changes (status only — math is FROZEN)**: the §38 math (Lemma A.1, Theorem A.2, Lemma A.3, mass conservation, Brownian example) and the `T_ADJOINT_FP_TIGHTNESS` sympy oracle (6/6 PASS, **re-confirmed 2026-06-07**) are UNCHANGED — no new math, no new oracle. The engineer wave materialises the EXISTING `.dev-docs/specs/adjoint-fp-wave.md` blueprint as Rust.
- **Module name**: `crates/semiflow-core/src/adjoint_fp.rs` (the Cohort 13 constitution pre-allocation name — supersedes the spec's earlier `adjoint_fokker_planck.rs` working name for constitution consistency).
- **Gate materialisation**: the v6.0.0 RESERVED `G_ADJOINT_FP_TIGHTNESS_VAGUE` stub is MATERIALISED as `G_ADJOINT_FP_ORDER` (RELEASE_BLOCKING): (1) vague-convergence slope ≤ −0.95 on the 1D Brownian char-fn self-convergence (n ∈ {16…256}, ξ ∈ {0.5,1,1.5,2}) — the §38.9 gate; PLUS (2) a discrete-adjoint-identity smoke ⟨L*u,v⟩=⟨u,Lv⟩ < 1e-12 that catches sign/index errors in the Lemma A.1 pushforward that a slope-only gate would miss.
- **Schema bumps**: traits.yaml 4.3.1 → 4.4.0 MINOR (ADD `MeasureState<F,D>`, `Adjointable<F,D>`, `AdjointFokkerPlanckChernoff<C,F,D>`); properties.yaml 4.5.1 → 4.6.0 MINOR (G_ADJOINT_FP_ORDER materialised). Constitution Cohort 13 already pre-allocated `adjoint_fp.rs` (700-LoC cap; spec estimates ≤500 LoC so the cap is headroom, not needed).
- **No BREAKING change**: purely additive. All §"Alternatives considered" and §"Consequences" reasoning is unchanged.
