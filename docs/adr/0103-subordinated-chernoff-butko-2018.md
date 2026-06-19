# ADR-0103 — A.5 SubordinatedChernoff via Butko 2018

- **Status**: ACCEPTED 2026-05-29 (PRE-FLIGHT 5/5 PASS)
- **Decision-maker**: ai-solutions-architect
- **Target release**: v4.8.0 MINOR (orthogonal track; ~2026-09)
- **Related ADRs**: ADR-0069 (v2.7 Laplace-Chernoff Resolvent — Gauss-Laguerre 32-pt quadrature pattern REUSED); ADR-0071 (v2.8 Manifold sibling pattern — `trait + concrete-impls + generic wrapper` REUSED); ADR-0072 (v2.8 ReflectingRegion + RegionFamily — sub-trait composition REUSED); ADR-0098 (v4.6 Robin BC — partial-additive shipping pattern REUSED); ADR-0078 (v3.1 quantum-graph standalone kernel — separate-module REUSED to avoid base-Chernoff contract entanglement).
- **Mathematical foundation**: math.md §37 NEW (`LevySubordinator<F>` + `SubordinatedChernoff<C, S, F>` semantics; Bochner-Phillips subordination + Butko 2018 §4 Theorem 2.1 order-1 Chernoff tangency + 3 concrete CBF subordinator backends). CITATIONS: Butko 2018 *Stochastics and Dynamics* §4 Theorem 2.1; Bochner 1949 *PNAS* 35 (5) "Diffusion equation and stochastic processes"; Sato 1999 *Lévy Processes and Infinitely Divisible Distributions* §30; Schilling-Song-Vondraček 2012 *Bernstein Functions: Theory and Applications* (de Gruyter) §13 (CBF criterion); Abramowitz-Stegun 1964 Table 25.9 (Gauss-Laguerre 32-pt).
- **Acceptance gates added**: T_SUBORD (NORMATIVE sympy oracle — 5 sub-checks; PRE-FLIGHT 5/5 PASS verified 2026-05-29 — see "Pre-flight result" below); G_SUBORD_ORDER1 (RELEASE_BLOCKING — self-convergence slope ≤ −0.95 for ≥ 2 of 3 subordinator backends, scalar-A heat-like contraction setup).
- **Constitution change**: NONE required. The new file `crates/semiflow-core/src/subordinated.rs` targets ≤ 500 LoC and uses no Cohort carve-out. The shared GL32 table is REUSED from existing `resolvent_quad.rs` (ADR-0069) — no new constant file.

## Context

The v4.6+ academic-priority backlog flagged subordination (Bochner 1949 / Phillips 1952 functional calculus) as item **A.5** — a genuinely orthogonal track to all other open backlog (Padé, Engel step-3, ζ⁸, Robin, Hörmander). Butko 2018 *Stochastics and Dynamics* §4 establishes that for any Chernoff approximation `F(τ)` of a base semigroup `(T_t)_{t≥0}` with generator `A`, and any one-dimensional Lévy subordinator `(S_t)_{t≥0}` (non-decreasing càdlàg with values in `[0, ∞)`) with Laplace exponent `φ`, the subordinate Chernoff family

$$F^\varphi(\tau) f \;:=\; \int_0^\infty F(s)\, f \;\mu^\varphi_\tau(\mathrm{d}s) \tag{0103.1}$$

is order-1 Chernoff-tangent to the subordinate semigroup `T^φ_t := \mathbb{E}[T_{S_t}]` whose generator is `−φ(−A)` (Phillips functional calculus). This unlocks a large class of fractional / tempered / inverse-Gaussian semigroups that are otherwise inaccessible without analytic continuation in λ (the v2.7 resolvent track) or operator-theoretic spectral resolution.

**Concrete motivations**:

1. **α-stable subordination** — `φ_α(λ) = λ^α, α ∈ (0,1)`. When `A = Δ`, the subordinate semigroup is `exp(-t·(-Δ)^α)` — the *Riesz fractional heat semigroup*. Used in nonlocal diffusion, jump-process Feynman-Kac, and anomalous transport modelling.
2. **Gamma subordination** — `φ_c(λ) = log(1 + λ/c)`. Tempered exponential variance; used in variance-gamma option pricing (Madan-Carr-Chang 1998).
3. **Inverse-Gaussian subordination** — `φ_c(λ) = sqrt(c² + 2λ) − c`. NIG (Normal Inverse Gaussian) Lévy process foundation; used in foreign-exchange returns modelling.

**Why NOT shipping in v2.7 (Laplace-Chernoff Resolvent)**: The resolvent computes `(λI − A)^{-1}` via *time-domain* Laplace transform; subordination computes `T^φ_t` via *subordinator-measure* averaging. The two integrals live in different domains (resolvent: time; subordinated: subordinator state); the GL32 *table* is REUSED but the kernel and integration variable are distinct. A shared abstraction `(quadrature: GL32, kernel: φ-specific)` would be premature — Butko 2018 quadrature for α-stable uses a *shifted* GL32 weight (per §4 eq 4.3) that the time-domain Laplace integrand does NOT use.

**Why standalone module `subordinated.rs` rather than extending `resolvent.rs`** — mirrors ADR-0078 (quantum-graph) precedent: a separate kernel class avoids composition unknowns and preserves the v2.7 resolvent contract semantics. Extending `LaplaceChernoffResolvent` to take a generic `LevySubordinator` would either silently weaken the Hille-Yosida contract (which requires `Re(λ) > ω`) or require a runtime branch with load-bearing behaviour difference. A separate `SubordinatedChernoff<C, S, F>` keeps both contracts crisp and patch-friendly.

## Pre-flight result (MANDATORY, ADR-0086 lesson)

PRE-FLIGHT sympy oracle `scripts/verify_subordinated_chernoff.py` executed 2026-05-29:

```
[PASS] T_SUBORD.bernstein_laplace_exponents (3 subordinators × k=1..4)
[PASS] T_SUBORD.alpha_stable_moment_match (α=1/2 Lévy density Laplace)
[PASS] T_SUBORD.gauss_laguerre_node_agreement (GL32 ∫s^6: rel err 3.158e-16 ≤ 5e-9)
[PASS] T_SUBORD.order1_chernoff_residual (Taylor coeffs 0..2 match)
[PASS] T_SUBORD.gamma_subordinator_closed_form (φ = log(1+λ) match)
T_SUBORD PASS
```

5/5 sub-checks PASS. ADR is GREEN. Engineer wave authorized to begin against the spec at `.dev-docs/specs/subordinated-wave.md`.

## Decision

Ship the following additive public-surface items in v4.8.0:

### Item 1 — NEW module `crates/semiflow-core/src/subordinated.rs` (target ≤ 400 LoC; under default 500-LoC cap; no Cohort carve-out)

```rust
//! A.5 — SubordinatedChernoff via Butko 2018 (math.md §37, ADR-0103).
//!
//! `LevySubordinator<F>` trait + 3 closed-form backends:
//!   - `StableSubordinator<F> { alpha }`            — φ(λ) = λ^α, α ∈ (0,1)
//!   - `GammaSubordinator<F> { c }`                  — φ(λ) = log(1 + λ/c)
//!   - `InverseGaussianSubordinator<F> { c }`        — φ(λ) = sqrt(c² + 2λ) − c
//!
//! `SubordinatedChernoff<C, S, F>` wraps any base Chernoff `C: ChernoffFunction<F>`
//! and any subordinator `S: LevySubordinator<F>`, computing
//!   F^φ(τ) f := Σ_k w_k · C(s_k) f       (Gauss-Laguerre 32-pt; REUSES resolvent_quad.rs)
//!
//! ## Citations
//! - Butko 2018 *Stochastics and Dynamics* §4 Theorem 2.1
//!   ("Chernoff Approximation of Subordinate Semigroups").
//! - Bochner 1949 *PNAS* 35 (5) (diffusion equation and stochastic processes).
//! - Sato 1999 *Lévy Processes and Infinitely Divisible Distributions* §30.
//! - Schilling-Song-Vondraček 2012 *Bernstein Functions* (de Gruyter) §13.

use core::marker::PhantomData;

use crate::{
    chernoff::{ChernoffFunction, Growth},
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    resolvent_quad::{GL32_NODES, GL32_WEIGHTS},
    scratch::ScratchPool,
    state::State,
};

/// One-dimensional Lévy subordinator interface for the Chernoff approximation
/// of the subordinate semigroup `T^φ_t := E[T_{S_t}]`. See math.md §37.
///
/// **Admissibility (Bernstein / CBF requirement)**: `laplace_exponent(λ)` MUST be
/// a Complete Bernstein Function (Schilling-Song-Vondraček 2012 §13):
///   φ(0) = 0, φ' ≥ 0, (-1)^{k+1} φ^{(k)} ≥ 0 for all k ≥ 1.
/// The three concrete backends (Stable, Gamma, InverseGaussian) are CBF by
/// construction; user-defined impls SHOULD include their own sympy CBF screen.
///
/// **Object-safety**: `Send + Sync + 'static` (mirrors `BoundedGeometryManifold`).
///
/// **Default `F = f64`**: matches the trait family convention (v0.9 ADR-0026).
pub trait LevySubordinator<F: SemiflowFloat = f64>: Send + Sync + 'static {
    /// Laplace exponent `φ(λ) := -log E[exp(-λ S_1)]`.
    ///
    /// MUST be a Complete Bernstein Function. Used in tests / sympy oracles
    /// to symbolically verify the subordinator's defining identity.
    fn laplace_exponent(&self, lambda: F) -> F;

    /// Quadrature for `E[h(S_τ)] ≈ Σ_k w_k h(s_k)`, sized `n_nodes`.
    ///
    /// Implementations SHOULD return the GL32 nodes/weights from
    /// `resolvent_quad.rs` shifted/scaled for the subordinator-specific
    /// density (Butko 2018 §4 eq 4.3 for α-stable; closed-form Gamma /
    /// inverse-Gaussian densities for the other two).
    ///
    /// Allocation: implementations MAY return `Vec<F>` for v4.8 ergonomics;
    /// a zero-alloc `quadrature_into(nodes: &mut [F], weights: &mut [F], τ)`
    /// is deferred to v5.x along with the wider scratch-pool refactor.
    fn quadrature(&self, tau: F, n_nodes: usize) -> (alloc::vec::Vec<F>, alloc::vec::Vec<F>);
}

// ─── Concrete subordinator backends ──────────────────────────────────────────

/// α-stable subordinator: `φ_α(λ) = λ^α` for `α ∈ (0, 1)`.
///
/// Subordinated semigroup is `exp(-τ · (-A)^α)` — Riesz fractional heat when
/// `A = Δ`. CBF: `φ_α` is the canonical example (Schilling-Song-Vondraček 2012
/// Example 14.3).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StableSubordinator<F: SemiflowFloat = f64> {
    /// Stability index; MUST be in `(0, 1)`. Validated in `new`.
    pub alpha: F,
}

impl<F: SemiflowFloat> StableSubordinator<F> {
    /// Construct with validated `α ∈ (0, 1)`.
    pub fn new(alpha: F) -> Result<Self, SemiflowError> { /* ... */ }
}

impl<F: SemiflowFloat> LevySubordinator<F> for StableSubordinator<F> { /* ... */ }

/// Gamma subordinator: `φ_c(λ) = log(1 + λ/c)` for `c > 0`.
///
/// Density: `p_t(s) = c^t s^{t-1} exp(-c·s) / Γ(t)`. CBF: Schilling-Song-
/// Vondraček 2012 Example 14.4.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GammaSubordinator<F: SemiflowFloat = f64> {
    /// Rate parameter; MUST be > 0. Validated in `new`.
    pub c: F,
}

impl<F: SemiflowFloat> GammaSubordinator<F> { pub fn new(c: F) -> Result<Self, SemiflowError> { /* ... */ } }
impl<F: SemiflowFloat> LevySubordinator<F> for GammaSubordinator<F> { /* ... */ }

/// Inverse-Gaussian subordinator: `φ_c(λ) = sqrt(c² + 2λ) − c` for `c > 0`.
///
/// CBF: Schilling-Song-Vondraček 2012 Example 14.5. Foundation of NIG Lévy
/// processes (Barndorff-Nielsen 1997).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InverseGaussianSubordinator<F: SemiflowFloat = f64> {
    /// Drift / location parameter; MUST be > 0. Validated in `new`.
    pub c: F,
}

impl<F: SemiflowFloat> InverseGaussianSubordinator<F> { pub fn new(c: F) -> Result<Self, SemiflowError> { /* ... */ } }
impl<F: SemiflowFloat> LevySubordinator<F> for InverseGaussianSubordinator<F> { /* ... */ }

// ─── SubordinatedChernoff<C, S, F> generic wrapper ──────────────────────────

/// Subordinated Chernoff approximation per Butko 2018 Theorem 2.1.
///
/// Wraps a base Chernoff `C` and a Lévy subordinator `S`; computes
///   `F^φ(τ) src := Σ_k w_k · C.apply_into(s_k, src, tmp)`
/// where `(s_k, w_k)` are produced by `S.quadrature(τ, n_nodes)`.
///
/// **Order**: 1 (per Butko 2018 Thm 2.1; the Chernoff product `F^φ(τ/n)^n` is
/// then norm-convergent to `T^φ_τ` by the Trotter-Kato-Chernoff theorem).
/// Higher-order subordinated Chernoff is RESEARCH-OPEN (Butko 2018 §5 lists
/// it as future work); v4.8 ships order-1 only.
///
/// **Growth bound**: inherited from the base Chernoff's `growth()`; the
/// subordinator-averaged operator norm satisfies `‖F^φ(τ)‖ ≤ ∫ ‖C(s)‖ μ^φ_τ(ds)`
/// which is bounded by `exp(τ · φ(ω))` for any quasi-contractive base
/// (`‖C(s)‖ ≤ exp(s·ω)`) — Phillips functional calculus, math §37.4.
#[derive(Debug, Clone)]
pub struct SubordinatedChernoff<C, S, F = f64>
where
    C: ChernoffFunction<F>,
    S: LevySubordinator<F>,
    F: SemiflowFloat,
{
    pub base: C,
    pub subordinator: S,
    /// Number of quadrature nodes; DEFAULT 32 (matches resolvent_quad.rs GL32).
    pub n_nodes: usize,
    _phantom: PhantomData<F>,
}

impl<C, S, F> SubordinatedChernoff<C, S, F>
where C: ChernoffFunction<F>, S: LevySubordinator<F>, F: SemiflowFloat,
{
    /// Construct with default `n_nodes = 32` (GL32 tables from resolvent_quad.rs).
    pub fn new(base: C, subordinator: S) -> Self { /* ... */ }

    /// Construct with explicit node count `n_nodes` (v4.8 supports n_nodes ≤ 32
    /// only; higher Gauss-Laguerre orders require shipping a wider table —
    /// deferred to v5.x per ADR-0069 §"Future" notes).
    pub fn with_n_nodes(base: C, subordinator: S, n_nodes: usize)
        -> Result<Self, SemiflowError> { /* ... */ }
}

impl<C, S, F> ChernoffFunction<F> for SubordinatedChernoff<C, S, F>
where
    C: ChernoffFunction<F>,
    C::S: Clone,                       // needed to accumulate Σ w_k · C(s_k) src
    S: LevySubordinator<F>,
    F: SemiflowFloat,
{
    type S = C::S;

    fn apply_into(&self, tau: F, src: &Self::S, dst: &mut Self::S)
        -> Result<(), SemiflowError>
    {
        // ─── Pseudocode (engineer wave fills in) ───────────────────────────
        //   1. Validate tau finite, tau ≥ 0 (DomainViolation if not).
        //   2. Get nodes/weights from self.subordinator.quadrature(tau, self.n_nodes).
        //   3. Initialize dst := 0  (in-place zero on the State trait).
        //   4. For each (s_k, w_k):
        //        tmp := C.apply_into(s_k, src, tmp_dst)
        //        dst += w_k · tmp                  (AXPY-style accumulation)
        //   5. Return Ok(()).
        // ─── Hot-path: tmp_dst lives in ScratchPool (REUSED, no allocation) ─
        todo!("engineer wave per spec at .dev-docs/specs/subordinated-wave.md")
    }

    fn order() -> usize { 1 }  // Butko 2018 Theorem 2.1 — subordinated Chernoff is order-1.

    fn growth(&self) -> Growth<F> {
        // Per math §37.4: ‖F^φ(τ)‖ ≤ exp(τ · φ(ω_base)).
        // Conservative bound: pass through base growth with φ-amplification.
        // Exact formula in engineer-wave spec; placeholder here.
        self.base.growth()
    }
}
```

### Item 2 — `contracts/semiflow-core.math.md` §37 (NORMATIVE; ~80 LoC)

Add §37 "Subordination via Bochner / Butko 2018" with:
- §37.1 Lévy subordinators and CBF admissibility (Schilling-Song-Vondraček 2012 §13);
- §37.2 Bochner-Phillips functional calculus statement (`T^φ_t = ∫_0^∞ T_s μ^φ_t(ds)`, generator `−φ(−A)`);
- §37.3 Butko 2018 Theorem 2.1 verbatim (order-1 Chernoff tangency for subordinate semigroups);
- §37.4 Growth bound `‖F^φ(τ)‖ ≤ exp(τ · φ(ω_base))`;
- §37.5 Three concrete CBF backends (`StableSubordinator`, `GammaSubordinator`, `InverseGaussianSubordinator`) with Laplace exponent + density formulas;
- §37.6 Quadrature: Gauss-Laguerre 32-pt REUSED from `resolvent_quad.rs` (ADR-0069); per-backend kernel transformation;
- §37.7 Implementation map → `crates/semiflow-core/src/subordinated.rs`;
- §37.8 References.

### Item 3 — `contracts/semiflow-core.properties.yaml` (schema bump 1.4.0 → 1.5.0)

Add two new records: `T_SUBORD` (NORMATIVE sympy, 5 sub-checks; PRE-FLIGHT GATE) and `G_SUBORD_ORDER1` (RELEASE_BLOCKING; self-convergence slope ≤ -0.95 for ≥ 2 of 3 backends).

### Item 4 — `contracts/semiflow-core.traits.yaml` (schema bump 2.2.0 → 2.3.0)

Add five new records: `LevySubordinator` (trait), `StableSubordinator` / `GammaSubordinator` / `InverseGaussianSubordinator` (concrete CBF impls), `SubordinatedChernoff<C, S, F>` (generic wrapper impl of `ChernoffFunction<F>`).

### Item 5 — `scripts/verify_subordinated_chernoff.py` (PRE-FLIGHT sympy oracle; ~250 LoC SHIPPED 2026-05-29)

Already shipped as a precondition of this ADR per the ADR-0086 PRE-FLIGHT lesson. Five sub-checks, all PASS — see "Pre-flight result" above. This script becomes the `T_SUBORD` test record's `invocation` and ships in the engineer wave's xtask sympy-sweep stanza.

### Item 6 — Engineer wave spec `.dev-docs/specs/subordinated-wave.md` (separate file; ~150 LoC)

NOT inline in this ADR. The wave spec details:
- Concrete validation logic for `StableSubordinator::new(α)` / `GammaSubordinator::new(c)` / `InverseGaussianSubordinator::new(c)`;
- Gauss-Laguerre transformation per backend (α-stable shifted kernel per Butko 2018 §4 eq 4.3; Gamma direct density evaluation; inverse-Gaussian Pinsky 1986 inversion);
- AXPY-style accumulation loop (with `ScratchPool` use for `tmp_dst`);
- Order-1 chernoff residual gate `G_SUBORD_ORDER1` setup (scalar A = -μ, 3 subordinator backends, sweep n ∈ {16, 32, 64, 128}, slope ≤ -0.95);
- LoC budget: subordinated.rs ≤ 400 LoC + tests/subordinated_*.rs ≤ 300 LoC each;
- Sympy regression: extend `scripts/verify_subordinated_chernoff.py` ONLY if new CBF backend ships in a future minor (forward-compatibility note).

## Consequences

**Public surface additions** (5 surface items; strictly additive — no v4.x BREAKING):

1. `pub trait LevySubordinator<F>` (one trait)
2. `pub struct StableSubordinator<F>` + `impl LevySubordinator<F>` (one concrete + impl)
3. `pub struct GammaSubordinator<F>` + `impl LevySubordinator<F>` (one concrete + impl)
4. `pub struct InverseGaussianSubordinator<F>` + `impl LevySubordinator<F>` (one concrete + impl)
5. `pub struct SubordinatedChernoff<C, S, F>` + `impl ChernoffFunction<F>` (generic wrapper + impl)

**Forward compatibility for `ApproximationSubspace<K, F>` (ADR-0073)**: `SubordinatedChernoff` is order-1 and therefore NOT a `LadderRung<K, F>` impl (ADR-0100 catalogue is closed); it MAY impl `ApproximationSubspace<2, F>` in a future minor if order-2 subordinated Chernoff ships per Butko 2018 §5 future work — that decision is deferred and out-of-scope for v4.8.

**No v3.0 v2_compat impact** (ADR-0084): `SubordinatedChernoff` did not exist in v2.x. No deprecation shim required.

**FFI / PyO3 / WASM impact** (ADR-0028 / ADR-0076): `SubordinatedChernoff` is generic over `C, S, F`; binding surface follows the v3.0 Approach A pattern — a single concrete `EvolverSubordinatedHeat1DStable_v4_8` per binding (FFI cdylib + PyO3 pyclass + WASM JS class) shipping ONLY the α-stable Heat1D combination at v4.8 (Gamma + InverseGaussian backends remain Rust-only at v4.8). Per-binding wave is OUT OF SCOPE for this ADR — see ADR-0028 amendment cadence; the binding wave is targeted for v4.9 or later contingent on user demand.

**Dependency budget** (constitution Override #3): 3/3 direct deps in `semiflow-core` (`num-traits`, `num-complex`, `static_assertions`). **No new deps**. Gauss-Laguerre tables ship as inline `const [f64; 32]` literals REUSED from `resolvent_quad.rs` (zero allocation, zero new deps).

**LoC inventory delta** (v4.8.0 NEW module): `subordinated.rs` target ≤ 400 LoC (well under 500 default); zero existing-file changes (additive only); zero Cohort impact.

**Risk profile**: LOW. Pure additive surface; PRE-FLIGHT 5/5 PASS; standalone module pattern proven by ADR-0078 / ADR-0098; GL32 table REUSE proven by ADR-0069. Primary risk vector is `G_SUBORD_ORDER1` engineer-wave slope failure for one or both Gamma / InverseGaussian backends — this risk is BOUNDED by the ≥ 2 of 3 acceptance threshold per the spec, allowing α-stable + one of {Gamma, IG} to suffice for v4.8 release while flagging the third for a v4.9+ calibration follow-up.

## Out of scope (deferred)

- Order-2+ subordinated Chernoff (Butko 2018 §5 open problem; would need a new family of `OrderKLevySubordinator` traits — out-of-scope until research clarifies).
- Per-cell varying subordinators (e.g., space-dependent `α(x)`-stable) — would require a `LevySubordinatorField<F>` super-trait; out-of-scope for v4.8.
- Multi-dimensional vector-valued subordinators (matrix-stable processes — Meerschaert-Sikorskii 2012) — out-of-scope until A.5 stabilizes.
- Subordinated Chernoff under bindings (FFI / PyO3 / WASM) — out-of-scope per "FFI / PyO3 / WASM impact" note above; tracked separately for v4.9+.
- Quadrature node count > 32 (would need wider Abramowitz-Stegun table) — out-of-scope per ADR-0069 §"Future" cross-reference.

## References

- Butko 2018 — *Chernoff Approximation of Subordinate Semigroups*, **Stochastics and Dynamics** 18 (3), §4 Theorem 2.1.
- Bochner 1949 — *Diffusion equation and stochastic processes*, **PNAS** 35 (5), pp. 368–370.
- Phillips 1952 — *On the generation of semigroups of linear operators*, **Pacific J. Math.** 2, pp. 343–369.
- Sato 1999 — *Lévy Processes and Infinitely Divisible Distributions*, **Cambridge Studies in Advanced Math.** 68, §30.
- Schilling-Song-Vondraček 2012 — *Bernstein Functions: Theory and Applications*, **de Gruyter Studies in Math.** 37, §13.
- Abramowitz-Stegun 1964 — *Handbook of Mathematical Functions*, **National Bureau of Standards** 55, Table 25.9 (Gauss-Laguerre nodes/weights).
- ADR-0069 — v2.7 Laplace-Chernoff Resolvent (GL32 quadrature pattern REUSED).
- ADR-0071 — v2.8 Manifold sibling pattern (`trait + concretes + wrapper` REUSED).
- ADR-0078 — v3.1 quantum-graph standalone kernel (separate-module pattern REUSED).
- ADR-0098 — v4.6 Robin BC partial-additive port (PRE-FLIGHT + standalone-kernel pattern REUSED).
- ADR-0100 — v5.0 `LadderRung<K, F>` sealed catalogue (consequence cross-reference; subordinated is NOT a ladder rung).
- math.md §37 (this ADR's contract authority).
- `~/.claude/plans/roadmap-reflective-biscuit.md` — v4.6+ academic-priority backlog item A.5.

---

## ADR-0103 Amendment 1 — Subordinated Chernoff φ-linearization correctness fix (v6.2.4)

- **Status**: Accepted. Supersedes the "generator-scaled GL32" quadrature of ADR-0103.
- **Decision-maker**: ai-solutions-architect
- **Buggy commit**: 346e372
- **Fix release**: v6.2.4 PATCH

The original SubordinatedChernoff quadrature `s_k = GL32_NODES[k]·φ(1)·τ`, `w_k = GL32_WEIGHTS[k]` linearizes the Bernstein function φ to the constant slope φ(1): per eigenvalue it yields F^φ(τ) = I − φ(1)·τ·A + O(τ²), converging to the rescaled plain semigroup exp(−Tφ(1)A) instead of the subordinated exp(−Tφ(A)). It is correct ONLY at base eigenvalue λ=1, where φ(1)·λ = φ(λ). The shipped gate G_SUBORD_ORDER1 tested a single eigenvalue (μ=1) and could not detect this. Fix: quadrature the subordinator's subordination density f_τ(s) (Bochner-Phillips exact identity exp(−τφ(A)) = ∫₀^∞ e^{−sA} f_τ(s) ds) rather than a generator-scaled Laplace transform. Gamma uses generalized Gauss-Laguerre with shape parameter τ−1 (exact in s); InverseGaussian uses a truncated standardized-IG Gauss rule; Stable (no closed-form density) uses a generalized-Gauss-Laguerre(−α) rule on the regularized Lévy integrand (1−e^{−sλ})/s, achieving order-1 tangency to exp(−τλ^α) for ALL λ. The eigensolver/quadrature infrastructure (Golub-Welsch symmetric-tridiagonal + generalized Gauss-Laguerre) is extracted to `gen_quadrature.rs`. No public API change (PATCH). G_SUBORD_ORDER1 redesigned to a 5-eigenvalue sweep λ∈{4,8,16,32,48} asserting BOTH order-1 slope AND correct limit exp(−φ(λ_i)) per mode (correct-limit ratio 0.2), plus an always-on `subordinated_does_not_linearize_phi` regression test, so φ-linearization can never again hide. **InverseGaussian KNOWN LIMITATION (deferred): its density quadrature over-decays for small per-step τ (Pinsky 1986 s^{−3/2} head), converging to the wrong limit; the IG backend is NOT yet correct in v6.2.4 — the gate passes via Stable+Gamma (≥2/3) and IG is explicitly flagged KNOWN-FAILING pending a future fix.** gitnexus upstream impact: LOW (0).
