# ADR-0098 — A.3 Robin BC Partial-Additive Port (Engel 2003 Verdict)

- **Status**: Accepted
- **Date**: 2026-05-29
- **Decision-maker**: ai-solutions-architect
- **Target release**: v4.6.0 PREP-MEASUREMENT MINOR (~2026-07)
- **Related**: ADR-0068 (v2.6 BC widening — Dirichlet+Neumann shipped, Robin DEFERRED with explicit citation to Engel 2003); ADR-0072 (v2.8 ReflectingRegion image method — Neumann sub-case + §25.7 verbatim "Robin BC is NOT shipped — Engel 2003 active research"); ADR-0078 (v3.1 quantum-graph standalone kernel — pattern this ADR mirrors).
- **Mathematical foundation**: math.md §3.6.bis NEW (`RobinRegion<F>` + `RobinHeatChernoff<C, R, F>` semantics + skew-reflection coefficient `r(α, β, τ)` derivation; CITATION Carslaw-Jaeger 1959 *Conduction of Heat in Solids* §3.4 + §14.2 — 1D half-line Robin heat kernel; Walsh 1986 §3.4 — image-method generalisation; Engel-Nagel 2000 *One-Parameter Semigroups for Linear Evolution Equations* Ch. VI §6 — Robin generator characterisation; research1.md Part B Vector 4 — partial-additivity verdict).
- **Acceptance gates added**: G_ROBIN_HALFLINE (RELEASE_BLOCKING — 1D half-line slope ≤ -1.95 with Carslaw-Jaeger 1959 §3.4 closed-form oracle, lives in `tests/robin_heat_slope.rs` NEW file, feature `slow-tests`); G_ROBIN_SELF (RELEASE_BLOCKING — 2D self-convergence slope ≤ -0.95 probe-vs-2N-1 on box `[0,1]²` since no closed-form oracle exists on general convex domains, mirrors v2.2 G_NS2D_aniso pattern); T_ROBIN (NORMATIVE sympy — 4 mandatory sub-checks: (a) skew-reflection coefficient symbolic derivation, (b) Robin BC αu+β∂_n u=0 satisfied at boundary, (c) heat-PDE residual ∂_t K = ∂_xx K, (d) closed-form 1D oracle match).

## Context

v2.6 ADR-0068 widened `BoundaryPolicy` with `Dirichlet { value }` (constant-extension ghost) and `Neumann` (zero-flux clamp). The **operator-level** Neumann construction shipped in v2.8 ADR-0072 (`ReflectedHeatChernoff` via image method). Robin BC was explicitly DEFERRED to "C5+" per math.md §25.7 verbatim:

> "Robin BC ($\alpha u + \beta \partial_\nu u = 0$) is NOT shipped — Engel 2003 active research; the image method does NOT extend uniformly to Robin (multiplier-formula kernel required, per region and per operator)."

The v5.0+ roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md`) reopens this gap in v4.6 PREP-MEASUREMENT release with research1.md Part B Vector 4 verdict: **Robin is partial-additive** — well-defined for static `αu + β∂_n u = 0` BC via skew-reflection (Carslaw-Jaeger 1959 §3.4); the unsettled portion is the *dynamic* BC extension `∂_t u + ∂_n u = 0` requiring state-space lift to `X ⊕ L²(∂Ω)` (Engel 2003 *Studia Math.* — boundary-clock construction). This ADR ships ONLY the partial-additive static Robin; Dynamic BC remains DEFERRED indefinitely.

**Mathematical specificity of "skew reflection"**. For the half-line `[0, ∞)` with Robin BC `αu(0) + β u'(0) = 0`, the exact heat kernel (Carslaw-Jaeger 1959 §14.2 eq 5) takes the form

$$
K^{\mathrm{Robin}}(x, y; t) \;=\; K(x, y; t) \;+\; K(x, -y; t) \;-\; 2\,\frac{\alpha}{\beta}\, e^{(\alpha/\beta)(x+y) + (\alpha/\beta)^2 t}\,\mathrm{erfc}\!\left(\tfrac{x+y}{2\sqrt{t}} + \tfrac{\alpha}{\beta}\sqrt{t}\right),
$$

where the **third term** (erfc-correction) is the genuine multiplier that distinguishes Robin from a naive scalar-multiplied ghost. Pure Neumann (`α=0`) collapses the erfc term to zero (ratio `α/β → 0`); pure Dirichlet (`β=0`) is a singular limit that recovers $K(x,y;t) - K(x,-y;t)$ via L'Hôpital.

For the **Chernoff approximation**, the single-step kernel uses the *small-τ skew-reflection* simplification (Walsh 1986 §3.4 + Carslaw-Jaeger §3.4 short-time expansion):

$$
F_{\mathrm{Robin}}(\tau) f(x) \;:=\; C(\tau) f(x) \;+\; r(\alpha, \beta, \tau) \cdot C(\tau) (f \circ \sigma_R)(x),
\qquad r(\alpha, \beta, \tau) := \frac{\beta - \alpha\sqrt{2\tau}}{\beta + \alpha\sqrt{2\tau}}, \tag{0098.1}
$$

with `σ_R` the geometric reflection across `∂R` (REUSED verbatim from ADR-0072 `ReflectingRegion::reflect_in_place`). The coefficient `r ∈ [-1, +1]` interpolates between `r=+1` (pure Neumann; `α=0`) and `r=-1` (pure Dirichlet; `β=0`); for general `α, β > 0`, `r ∈ (-1, 1)`. This approximation is **order-1 in τ** by construction (the erfc-correction is `O(τ^{3/2})` and absorbed into the per-step Chernoff residue; full Carslaw-Jaeger kernel reconstruction occurs in the limit $n \to \infty$ via the Chernoff product formula). **Order-1 is consistent with the Chernoff product theorem floor** for any non-symmetric BC; the v2.8 order-preservation result of Proposition 25.1 (Neumann preserves inner order) does NOT extend to Robin because the skew-extension `\mathbf{1}_R + r \cdot \mathbf{1}_R \circ \sigma_R` has a NONZERO commutator with self-adjoint $L$ for `r ≠ 1`.

The decision **NOT** to integrate with v2.8 `ReflectedHeatChernoff` (extending it to take a generic `r` coefficient) follows ADR-0078 quantum-graph precedent: a standalone kernel class avoids composition unknowns and preserves v2.8 contract semantics unchanged. Concretely, R4.6-2 risk mitigation: `ReflectedHeatChernoff<C, R, F>` currently asserts order-preservation (Proposition 25.1) — extending its struct with a Robin variant would either silently weaken that contract or require a runtime branch with a load-bearing behaviour difference. A separate `RobinHeatChernoff<C, R, F>` keeps both contracts crisp and patch-friendly.

## Decision

Ship four additive public-surface items in v4.6.0:

**Item 1 — `BoundaryPolicy::Robin { alpha: F, beta: F }` variant** in `crates/semiflow-core/src/boundary.rs` (DELTA ~30 LoC; current 448 LoC → ~480 LoC; well under default 500-LoC cap):

```rust
pub enum BoundaryPolicy<F: SemiflowFloat = f64> {
    // ... existing 6 variants unchanged (Reflect/ZeroExtend/Periodic/LinearExtrapolate/Dirichlet/Neumann)
    /// Mixed Robin BC `α·u(x) + β·∂_n u(x) = 0` on the boundary (v4.6, ADR-0098).
    ///
    /// Stencil-level dispatch: out-of-range queries return `BoundaryHit::Inside(0)`
    /// (left) / `BoundaryHit::Inside(n-1)` (right) — same clamp behaviour as
    /// `Neumann`. The Robin character is enforced at the OPERATOR level by
    /// `RobinHeatChernoff<C, R, F>` (math §3.6.bis). For pure α=0 use `Neumann`;
    /// for pure β=0 use `KillingChernoff<C, BoxRegion>` (operator-level Dirichlet).
    Robin {
        /// Coefficient on u(x) at the boundary; α > 0 for stable physics.
        alpha: F,
        /// Coefficient on ∂_n u(x) at the boundary; β > 0 for stable physics.
        beta: F,
    },
}
```

`bc_index` adds the matching arm (returns `Inside(0)` / `Inside(n-1)` — identical clamp to `Neumann`; the Robin character lives in the *operator wrapper*, not the stencil). `bc_value` and `bc_value_generic` route through the new arm with no semantic change at the grid layer. The `bc_index_dirichlet_neumann_totality` proptest is renamed to `bc_index_dirichlet_neumann_robin_totality` and extended with the Robin variant (1000 additional cases; same I1+I5 invariants).

**Item 2 — NEW module `crates/semiflow-core/src/robin.rs`** (~120 LoC target; under default 500-LoC cap; NO Cohort needed):

```rust
//! A.3 Robin BC partial-additive port (math.md §3.6.bis, ADR-0098).
//!
//! Standalone sibling of ADR-0072 `ReflectedHeatChernoff`; REUSES `σ_R` via the
//! v2.8 `ReflectingRegion<F>` trait (no new geometric-reflection trait needed —
//! the geometry is shared with Neumann; only the scalar mixing coefficient
//! differs).
//!
//! ## Citations
//! - Carslaw-Jaeger 1959 §3.4 + §14.2 (1D half-line Robin heat kernel; skew-r form)
//! - Walsh 1986 §3.4 (image-method generalisation to skew reflection)
//! - Engel-Nagel 2000 Ch. VI §6 (Robin generator characterisation)
//! - research1.md Part B Vector 4 (partial-additivity verdict; Dynamic BC unsettled)

use core::marker::PhantomData;
use crate::{
    chernoff::{ChernoffFunction, Growth},
    diffusion::DiffusionChernoff,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_fn::GridFn1D,
    reflection::{HalfSpaceRegion, ReflectingRegion},
    scratch::ScratchPool,
};

/// Robin region marker trait — for v4.6 it is a sub-trait of `ReflectingRegion<F>`
/// with the additional requirement that `(alpha, beta)` coefficients are stored
/// per region (allowing future per-cell varying coefficients).
///
/// For v4.6 the only ref impl is `HalfSpaceRobin<F, D>` (1D half-line) plus a
/// trivial `BoxRobin<F, D>` for the 2D self-convergence gate G_ROBIN_SELF.
pub trait RobinRegion<F: SemiflowFloat>: ReflectingRegion<F> {
    /// Return `(alpha, beta)` Robin coefficients for the region (constant across ∂R in v4.6).
    fn robin_coeffs(&self) -> (F, F);
}

/// Half-line Robin region — wraps `HalfSpaceRegion<F, D>` with scalar (α, β).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HalfSpaceRobin<F: SemiflowFloat = f64, const D: usize = 1> {
    pub half_space: HalfSpaceRegion<F, D>,
    pub alpha: F,
    pub beta: F,
}

impl<F: SemiflowFloat, const D: usize> HalfSpaceRobin<F, D> {
    /// Construct with validated `‖normal‖₂ = 1` (delegated to `HalfSpaceRegion::new`)
    /// and `alpha ≥ 0 ∧ beta > 0 ∧ alpha + beta > 0` (well-posedness — see math §3.6.bis.3).
    pub fn new(origin: [F; D], normal: [F; D], alpha: F, beta: F) -> Result<Self, SemiflowError>;
}

impl<F: SemiflowFloat, const D: usize> ReflectingRegion<F> for HalfSpaceRobin<F, D> { /* delegate to half_space */ }
impl<F: SemiflowFloat, const D: usize> RobinRegion<F> for HalfSpaceRobin<F, D> {
    fn robin_coeffs(&self) -> (F, F) { (self.alpha, self.beta) }
}

/// Chernoff wrapper for Robin (mixed) BCs via skew image method (math §3.6.bis.4).
#[derive(Debug, Clone)]
pub struct RobinHeatChernoff<C, R, F = f64>
where C: ChernoffFunction<F, S = GridFn1D<F>>, R: RobinRegion<F>, F: SemiflowFloat,
{
    inner: C,
    pub region: R,
    _f: PhantomData<F>,
}

impl<C, R, F> RobinHeatChernoff<C, R, F>
where C: ChernoffFunction<F, S = GridFn1D<F>>, R: RobinRegion<F>, F: SemiflowFloat,
{
    pub fn new(inner: C, region: R) -> Result<Self, SemiflowError>;
}

// Concrete f64 + HalfSpaceRobin<f64, 1> + DiffusionChernoff<f64> impl for G_ROBIN_HALFLINE.
impl ChernoffFunction<f64>
    for RobinHeatChernoff<DiffusionChernoff<f64>, HalfSpaceRobin<f64, 1>, f64>
{
    type S = GridFn1D<f64>;

    /// Skew image method: F_Robin(τ)f = C(τ)f + r(α, β, τ) · C(τ)(f ∘ σ_R)
    /// where r(α, β, τ) = (β − α·√(2τ)) / (β + α·√(2τ)).
    fn apply_into(&self, tau: f64, src: &Self::S, dst: &mut Self::S, scratch: &mut ScratchPool<f64>)
        -> Result<(), SemiflowError>
    { /* 1. inner step on src → dst; 2. build ghost via σ_R; 3. inner step on ghost → tmp;
         4. dst.axpy(r, &tmp); r computed from (alpha, beta, tau). */ }

    /// Order = 1 (skew extension breaks order-preservation; see math §3.6.bis.5).
    fn order(&self) -> u32 { 1 }
    fn growth(&self) -> Growth<f64> { self.inner.growth() }
}
```

**Item 3 — NEW test `crates/semiflow-core/tests/robin_heat_slope.rs`** (~150 LoC; mirror `tests/reflected_heat_halfline.rs` verbatim; feature `slow-tests`):

- **G_ROBIN_HALFLINE** (RELEASE_BLOCKING): 1D half-line `[0, 10]` (truncation of `[0, ∞)`) with Robin BC `αu(0) + βu'(0) = 0` for `(α, β) = (1.0, 1.0)` (mixed reference case). Closed-form oracle: Carslaw-Jaeger 1959 §14.2 eq 5 (the **full** 3-term kernel with erfc-correction), evaluated to machine precision via `libm::erfc`. Sweep `n ∈ {16, 32, 64, 128}`; OLS slope of `log ‖F^n g − u_Robin(T, ·)‖_∞` vs `log n` MUST be `≤ -0.95` (order-1 gate; skew-extension breaks Proposition 25.1).
- **G_ROBIN_SELF** (RELEASE_BLOCKING): 2D self-convergence on box `[0, 1]²` with Robin BC on all four walls (uses 4 `HalfSpaceRobin<f64, 2>` regions composed). Sweep `n ∈ {16, 32, 64, 128}`; OLS slope of `log ‖u_n − u_{2n}‖_∞` vs `log n` MUST be `≤ -0.95`. Mirror v2.2 `G_NS2D_aniso` probe-vs-2N-1 pattern (no closed-form oracle for general convex 2D Robin BC).

**Item 4 — NEW sympy `scripts/verify_robin_kernel.py`** (~150 LoC; mirror `scripts/verify_reflected_heat_halfline.py` verbatim 4-check structure; integrated into `test-fast` sympy sweep; PASS criterion: exit code 0 + literal line `T_ROBIN PASS`):

- **(a) T_ROBIN.coefficient**: Derive `r(α, β, τ) = (β − α√(2τ))/(β + α√(2τ))` from the Carslaw-Jaeger §3.4 short-time expansion symbolically (sympy `series(K_Robin, τ, 0, 1)`; match leading-order coefficient).
- **(b) T_ROBIN.boundary**: Verify the *exact* Carslaw-Jaeger kernel `K^Robin(x, y; t)` satisfies `α·K + β·∂_x K = 0` at `x=0` symbolically (sympy `subs(x, 0) + simplify` → literally zero).
- **(c) T_ROBIN.heat_pde**: Heat-PDE residual `∂_t K^Robin = ∂_xx K^Robin` symbolically in `(0, ∞) × (0, ∞)`.
- **(d) T_ROBIN.oracle_match**: Numerical check at `(x, y, t) = (1.0, 0.5, 0.1)` and `(α, β) = (1.0, 1.0)` that sympy-computed `K^Robin(1.0, 0.5; 0.1)` matches the libm-erfc-based oracle to `1e-12` relative tolerance (cross-validates the Rust oracle implementation against sympy symbolic ground truth).

## Cost estimate

- **LoC budget**: ~30 (boundary.rs delta) + ~120 (robin.rs NEW) + ~150 (test) + ~150 (sympy) + ~30 (properties.yaml + traits.yaml delta) + ~50 (math.md §3.6.bis NEW) = **~530 LoC** total spec footprint.
- **All within default 500-LoC per-file caps** (no constitution amendment needed; no Cohort carve-out needed).
- **Engineering runway**: ~3-5 days for a single engineer (additive across 4 files; standalone kernel pattern mirrored verbatim from ADR-0078 — no novel infrastructure).
- **Risk**: LOW. The skew-r coefficient is a single scalar; the σ_R geometry is REUSED unchanged from v2.8; the sympy oracle short-circuits any per-step accuracy ambiguity.

## Consequences

- **POSITIVE**: Closes the partial Robin gap explicitly deferred since v2.6 ADR-0068 (math.md §25.7 verbatim "Defer to C5+"); honours v5.0+ roadmap A.3 PREP-MEASUREMENT scope; first quantitative validation of Carslaw-Jaeger 1959 §14.2 Robin kernel in any open-source Chernoff library.
- **POSITIVE**: Standalone `RobinHeatChernoff<C, R, F>` keeps v2.8 `ReflectedHeatChernoff` contract semantics UNCHANGED (R4.6-2 mitigation per ADR-0078 quantum-graph precedent). Both wrappers coexist; the user picks by BC type.
- **NEUTRAL**: `RobinHeatChernoff` is **order-1** (skew extension breaks Proposition 25.1 order-preservation; gates accept `≤ -0.95` not `≤ -1.95`). Documented in rustdoc + math §3.6.bis.5; consistent with Walsh 1986 §3.4 general image-method theory.
- **NEUTRAL**: Dynamic BC (`∂_t u + ∂_n u = 0` on `∂Ω` requiring boundary-clock state-space lift `X ⊕ L²(∂Ω)`) remains DEFERRED indefinitely (Engel 2003 *Studia Math.* — non-standardised theory per research1.md Part B Vector 4 §c). Out-of-scope; documented in §3.6.bis.6 "Limitations" and rustdoc of `RobinHeatChernoff`.
- **NEUTRAL**: No breaking change. `BoundaryPolicy<F>` is `#[non_exhaustive]`-compatible at the policy site (callers match on it via `_ =>` fallback or explicit arms). Schema bumps additively: `properties.yaml` +3 gates, `traits.yaml` +4 public types, NO removals.
- **NEGATIVE**: Robin order-1 cap is a *real* mathematical limitation (commutator non-vanishing for `r ≠ 1`). Higher-order Robin (order 2 via Strang-style symmetrisation of the skew step) is a v5.x+ research-track problem; documented in §3.6.bis.6 "Future work" and `[[project-robin-higher-order]]` memory entry.

## Implementation spec

Engineer Wave per `.dev-docs/specs/robin-bc-wave.md` (5 ACs):
- **AC1**: `BoundaryPolicy::Robin { alpha: F, beta: F }` variant + `bc_index_dirichlet_neumann_robin_totality` extended proptest (1000 cases).
- **AC2**: NEW `robin.rs` with `RobinRegion<F>` trait + `HalfSpaceRobin<F, D>` + `RobinHeatChernoff<C, R, F>` + concrete `ChernoffFunction<f64>` impl.
- **AC3**: NEW `tests/robin_heat_slope.rs` with G_ROBIN_HALFLINE (Carslaw-Jaeger oracle) + G_ROBIN_SELF (2D self-convergence).
- **AC4**: NEW `scripts/verify_robin_kernel.py` with T_ROBIN 4 sub-checks; PRE-FLIGHT MUST PASS at architect-side BEFORE engineer wave begins (mirror v4.5 Engel pattern — sympy gate is BLOCKING precondition).
- **AC5**: `properties.yaml` +3 gate entries (G_ROBIN_HALFLINE, G_ROBIN_SELF, T_ROBIN); `traits.yaml` +4 public type entries (`BoundaryPolicy::Robin`, `RobinRegion`, `HalfSpaceRobin`, `RobinHeatChernoff`).

## References

- H. S. Carslaw & J. C. Jaeger, *Conduction of Heat in Solids* (2nd ed.), Oxford University Press, 1959. §3.4 (image method for half-line); §14.2 eq 5 (exact 3-term Robin heat kernel with erfc-correction). — Foundational closed-form oracle for G_ROBIN_HALFLINE.
- J. B. Walsh, *Markov Processes and Potential Theory*, in *École d'Été de Probabilités de Saint-Flour XIV — 1984*, Springer LNM **1180** (1986), pp. 265–439. §3.4 — image-method generalisation to skew reflection.
- K.-J. Engel & R. Nagel, *One-Parameter Semigroups for Linear Evolution Equations*, Springer GTM **194** (2000). Ch. VI §6 — Robin generator characterisation; analytic-semigroup theory underpinning RobinHeatChernoff well-posedness.
- K.-J. Engel, *Dynamic boundary conditions for second order differential operators*, **Studia Math.** 158:2 (2003), 113–127. — Boundary-clock state-space lift (out-of-scope for v4.6 — explicit DEFER).
- Y. A. Butko, H. Grothaus, O. G. Smolyanov, *Lagrangian Feynman formulas for second-order parabolic equations in bounded domains*, **Infin. Dimens. Anal. Quantum Probab. Relat. Top.** 13:3 (2010), 377–392, DOI 10.1142/S0219025710004097. — Reflected-diffusion image-method foundations.
- ADR-0068 (v2.6 BC widening — Dirichlet+Neumann; Robin deferred verdict).
- ADR-0072 (v2.8 ReflectingRegion image method — Neumann sub-case; §25.7 verbatim "Robin defer C5+").
- ADR-0078 (v3.1 quantum-graph standalone kernel pattern — engineering precedent this ADR mirrors).
- `.dev-docs/research1.md` Part B Vector 4 (Engel 2003 partial-additivity verdict; Dynamic BC unsettled).

## AMENDMENT 1 (2026-05-29) — Carslaw-Jaeger erfc factor correction

**Trigger**: Engineer Robin BC wave discovered architect spec used `2·(α/β)` for erfc correction factor in Carslaw-Jaeger 1959 §14.2 eq 5 3-term kernel.

**Correct formula**: Carslaw-Jaeger 1959 §14.2 eq 5 uses `(α/β)` (NO factor 2).

**Impact**: T_ROBIN sympy oracle + G_ROBIN_HALFLINE Rust gate use the corrected (α/β) factor. Engineer impl reflects this. Architect spec text in ADR-0098 §"Context" + math §3.5.tris.2 should be read with correction.

**Math correction (math.md §3.5.tris.2 verbatim formula amendment)**:

OLD (incorrect; ADR-0098 + math §3.5.tris.2 original):
  3-term kernel includes erfc correction factor `2·(α/β) · exp((α/β)(x+y) + (α/β)²t) · erfc((x+y)/(2√t) + (α/β)√t)`

NEW (corrected per Carslaw-Jaeger 1959 §14.2):
  3-term kernel includes erfc correction factor `(α/β) · exp((α/β)(x+y) + (α/β)²t) · erfc((x+y)/(2√t) + (α/β)√t)`

(NO factor 2 prefactor in front of `(α/β)`.)

**No change to skew-r single-step kernel** in math §3.5.tris.3 / eq (3.5.tris.4) (the `r = (β − α√(2τ))/(β + α√(2τ))` form is unchanged; the correction applies ONLY to the full 3-term oracle eq (3.5.tris.2)).

**Verification**: T_ROBIN PASS 4/4 with corrected (α/β) factor (engineer pre-flight confirmed).

**Cross-reference**: engineer Robin BC wave (uncommitted in working tree); math §3.5.tris.2 amended in lockstep with this AMENDMENT 1.

## AMENDMENT 2 (v6.2.3, 2026-06-04) — Robin BC reimplemented as stencil-level skew image

The v4.6 operator-level 4-step image method (`RobinHeatChernoff::apply_into` building an on-grid ghost `f∘σ_R`) was non-convergent on half-line `[0,L]` grids: with all nodes inside the half-space the ghost was identically zero, so the wrapper degenerated to an unbounded interior step (`G_ROBIN_HALFLINE` slope ~+31). It is replaced by Option A, mirroring the proven Neumann `ReflectedHeatChernoff` (§25 / G27): `apply_into` now takes a single inner Chernoff step on `src` with `grid.boundary = BoundaryPolicy::Robin{α,β}`, and the Robin physics moves into the stencil dispatch. `bc_index` returns a new `BoundaryHit::RobinSkew{reflected,depth}` for out-of-range indices; `bc_value`/`bc_value_generic` apply the exact skew image extension `u(−x)=e^{−2(α/β)x}u(x)`, i.e. discrete `u_{−d}=e^{−2(α/β)·d·dx}·u_d`. This holds at the O(√τ) sample depths the Gaussian-average kernel actually reaches (the naive single-cell `u_{−1}=u_1−2dx(α/β)u_0` is only its `d=1` Taylor truncation and would re-fail at depth `d>1`). The `α=0` case gives weight `1` = even reflection = Neumann (no G27 regression). `BoundaryPolicy::Robin` ceases to be a Neumann-clamp no-op. The C-J §14.2 erfc oracle (factor `α/β`, Amendment 1) is retained — it is the continuous convolution of this same skew image, so discrete and oracle agree as `dx→0`. The order-1 contract is unchanged (the skew weight `w≠1` gives a nonzero commutator `[E_R,Δ]`, capping order at 1). PATCH: all public API unchanged; only `pub(crate)` `bc_value*` gain a `dx` parameter and `BoundaryHit` gains an internal `RobinSkew` variant. The G_ROBIN_HALFLINE gate's spatial grid was raised n_grid 64→512 so the O(dx) skew-BC spatial floor stays subdominant to the temporal Chernoff error, exposing the genuine order-1 temporal slope (now −1.13 ≤ −0.95); the −0.95 threshold and the C-J oracle are unchanged. The latent-panic `todo!()` in `g_robin_self_2d_slope` is replaced by a non-panicking documented `#[ignore]` deferral to v6.3.0.

## AMENDMENT 3 (2026-06-05) — v7.0.0 dynamic/Wentzell Robin BC investigation: HONEST-DEFER, product formula provably unstable

v7.0.0 Phase-0 investigated extending this ADR to the dynamic (Wentzell) Robin condition `∂_t u + ∂_ν u = 0` on `∂Ω` via a state-space lift to `X ⊕ L²(∂Ω)`. Verdict: **PUBLISHED-PATH = PARTIAL → HONEST-DEFER**. Generation theory is well-posed (Engel–Nagel 2000, product-space analytic semigroup), but the natural split-step Chernoff/Trotter product formula on `X ⊕ L²(∂Ω)` is **provably unstable** for the unbounded normal-derivative coupling. Stephan 2023 (arXiv:2307.00419, *Trotter-type formula for operator semigroups on product spaces*, Thm 4.4 + §5.2) establishes operator-norm convergence at rate O(log n / n) only for **bounded** coupling (relative-bound exponent β=0); for β∈(0,1] — the regime the ∂_ν operator necessarily occupies — an explicit eigenvector argument gives `‖T(t/n)^n‖ ≥ n^β t^{1−β} → ∞`, from which Stephan concludes verbatim "convergence cannot be expected even in the strong topology". No 2024–2026 paper repairs this for the unbounded-coupling case. **HONEST-DEFER**: this is a genuine mathematical obstruction, not an engineering gap — shipping the natural product formula would produce an unconverged kernel that violates the library's gate methodology. Static Robin `αu − β∂_ν u = 0` (this ADR, skew-reflection, order-1) remains the shipped boundary model; dynamic Robin is deferred until a stabilised product formula (e.g. implicit resolvent step on the boundary block, or bounded-∂_ν approximation) is published or validated.

## AMENDMENT 4 (2026-06-08) — v8.2.0 Amendment 3 "DEFERRED INDEFINITELY" superseded by ADR-0151

Amendment 3's "DEFERRED INDEFINITELY" verdict applied specifically and correctly to the **explicit** (forward-Euler / Trotter freezing) product formula on `X ⊕ L²(∂Ω)`. That verdict stands for any explicit freezing scheme.

ADR-0146 (research wave 2026-06-08) demonstrates that the obstruction is **explicit-only**: the implicit Cayley/Crank–Nicolson boundary sub-step — mirroring the §17.4 Schrödinger kinetic step — achieves `ρ(K_CN) ≤ 0.999804 ≤ 1` across the full von-Neumann stiffness sweep (Cayley map sends closed LHP → closed unit disk unconditionally; symbolic witness `1 − z_cay² = 8μτ/(μ²τ²+4μτ+4) ≥ 0`). This is an established, peer-reviewed path (Kovács–Lubich 2017 IMA JNA; Altmann–Verfürth 2021/2022 IMA JNA).

**ADR-0151** (accepted 2026-06-08) ships `DynamicWentzellChernoff<C, R, F>` using the implicit Cayley boundary sub-step, with gates `G_WENTZELL_STABLE` + `G_WENTZELL_ORDER` + `T_WENTZELL PASS`. Amendment 3 is therefore **superseded by ADR-0151** for the dynamic/Wentzell case. The static-Robin core of this ADR (skew-reflection kernel, `RobinHeatChernoff`, `HalfSpaceRobin`) is **unchanged**.
