# ADR-0071 — Riemannian Manifold Chernoff (A4)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v2.8 (first math pillar of the Manifold Pillar release; additive minor)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float with `F = f64` default), ADR-0026 (`ChernoffFunction` trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0068 (v2.6 BC widening — `BallRegion` reuse), ADR-0070 (v2.7 `TimedChernoffFunction` — explicitly *not* extended to manifold by default; the timed manifold variant is a v3.x research item).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW trait class `BoundedGeometryManifold<F>` (no prior abstraction; no impacted existing impls).
- **Mathematical foundation**: math.md §24 (NORMATIVE library — `ManifoldChernoff` semantics + curvature correction; CITATION Mazzucchi-Moretti-Remizov-Smolyanov 2023 *Math. Nachr.* Theorem 1 for the Gaussian-on-tangent-space Chernoff approximation; Bismut 1984 *Large Deviations and Stochastic Mechanics* for the geometric framework).
- **Acceptance gates added**: G26 (RELEASE_BLOCKING — Sphere-S² slope, two sub-tests for base + R/12 promotion), T21N (NORMATIVE sympy — flat torus eigenmode exact identity).

## Context

For a Riemannian manifold $(M, g)$ of bounded geometry (uniformly bounded sectional curvature plus strictly positive injectivity radius), the Laplace-Beltrami operator $\Delta_M$ generates the *heat semigroup* $\{S(t)\}_{t \ge 0}$ on $L^2(M)$. The classical Chernoff approximation `S(t) ≈ (F(t/n))^n` requires a Chernoff function $F(\tau)$ that is order-consistent with $\Delta_M$ on a dense core.

Mazzucchi-Moretti-Remizov-Smolyanov (2023, *Math. Nachr.* — *Operator semigroups and Chernoff approximations on Riemannian manifolds*) construct exactly such a Chernoff function by integrating against a *Gaussian-on-tangent-space* kernel and pulling back to $M$ via the Riemannian exponential map:

```
F(τ) f(x)  :=  (4πτ)^{-d/2} ∫_{T_x M} exp(-‖v‖²_{g_x} / (4τ)) · f(exp_x(v)) · [1 + (τ/12) · R(x)] dv
```

where $d = \dim M$, $\exp_x: T_x M \to M$ is the Riemannian exponential at $x$, $\|\cdot\|_{g_x}$ is the inner-product norm on $T_x M$, and $R(x)$ is the **scalar curvature** at $x$. The trailing factor $[1 + (\tau/12) R(x)]$ is the *first-order curvature correction* — it cancels the leading-order curvature contribution to the Chernoff residual, lifting the global rate from order 1 (the bare Gaussian-on-tangent kernel) to order 2 (matching Theorem 1 of MMRS 2023 §4).

The library has NO prior abstraction for Riemannian manifolds: all existing impls (`DiffusionChernoff`, `Strang2D`, `NonSeparable2DAniso`, `MagnusGraphHeat`, etc.) operate on flat $\mathbb{R}^d$ or on discrete graphs. v2.8 ships a new trait class `BoundedGeometryManifold<F>` plus three closed-form backends (`Torus<F, D>`, `Sphere2<F>`, `Hyperbolic2<F>`) plus the generic wrapper `ManifoldChernoff<M, F>`.

This is **scoped** for v2.8: closed-form $\exp_x$ in the three reference backends only. User-defined manifolds (with numerically-integrated geodesic flow) COMPILE via the trait but are not gated by any acceptance test — deferred to v3.x. The pillar pairs with B4 Neumann via image method (ADR-0072) — together they extend the v2.6/v2.7 boundary infrastructure from flat $\mathbb{R}^d$ to curved manifolds and reflecting domains.

## Decision

Ship five additive public-surface items in v2.8:

- **`pub trait BoundedGeometryManifold<F: SemiflowFloat = f64>`** — new trait class. Required methods:
  ```rust
  fn dim(&self) -> usize;                                            // d = manifold dimension
  fn injectivity_radius(&self) -> F;                                 // inf_{x ∈ M} inj(x); MUST be > 0
  fn exp_map(&self, x: &[F], v: &[F], out: &mut [F]) -> Result<(), SemiflowError>;
                                                                     // out := exp_x(v); writes d coords
  fn parallel_transport(&self, x: &[F], y: &[F], v: &[F], out: &mut [F])
      -> Result<(), SemiflowError>;                                   // out := P_{x→y}(v)
  fn scalar_curvature(&self, x: &[F]) -> F;                          // R(x); MUST be uniformly bounded
  fn volume_element_log(&self, x: &[F]) -> F;                        // log √det g(x); for the measure on M
  ```
  Each method validates input dimension via `debug_assert_eq!(coord.len(), self.dim())` and returns `DomainViolation` from `exp_map` / `parallel_transport` if the input contains NaN/Inf or violates the bounded-geometry hypothesis (e.g., $\|v\|_{g_x} > \mathrm{inj}(x)$ takes the exponential past the cut locus — implementation MUST either clamp to the cut locus or return `DomainViolation`; documented per backend).

- **`pub struct ManifoldChernoff<M, F>`** where `M: BoundedGeometryManifold<F>` — generic wrapper that implements `ChernoffFunction<F>`. Constructor:
  ```rust
  pub fn new(manifold: M, with_curvature_correction: bool) -> Self;
  ```
  When `with_curvature_correction == false`, ships the bare Gaussian-on-tangent Chernoff (order 1). When `true`, includes the $[1 + (\tau/12) R(x)]$ multiplicative factor and order is 2. `order()` returns `1` or `2` accordingly. `growth()` returns `(1.0, 0.0)` (the heat semigroup on a compact manifold is a contraction; on non-compact bounded-geometry manifolds the bound holds in the $\sup$-norm on the core).

- **Three closed-form reference backends** — each implements `BoundedGeometryManifold<F>`:
  - **`pub struct Torus<F, const D: usize>`** — flat $D$-torus (any dimension). $R \equiv 0$, $\exp_x(v) = x + v$ in normal coordinates (identity on the universal cover); parallel transport is the identity. The Chernoff function reduces to the standard heat kernel on $\mathbb{T}^D$ — FFT-based oracle for validation in G26 prep + T21N.
  - **`pub struct Sphere2<F>`** — unit-radius 2-sphere. Constant scalar curvature $R = 2$ (sectional curvature 1, 2D so $R = 2 \cdot 1 = 2$). $\exp_x(v)$ via the great-circle formula in stereographic projection chart (uniform on the chart's domain; cut locus is the antipode, which the bounded-geometry assumption avoids if $\|v\| < \pi$).
  - **`pub struct Hyperbolic2<F>`** — Poincaré disk model with constant negative curvature $-1$ (so scalar curvature $R = 2 \cdot (-1) = -2$). $\exp_x(v)$ via the Möbius transformation in the conformal chart.

  All three backends store their parameters as plain const-or-`F` fields (no allocation; `Copy` where appropriate). Each ships a sympy oracle script in `scripts/verify_manifold_curvature.py` validating the scalar curvature and exp-map formulas symbolically (Wave A).

File layout: `crates/semiflow-core/src/manifold.rs` (~700 LoC target — math co-location for the trait + 3 backends + `ManifoldChernoff` impl + curvature correction; HARD LIMIT 800 LoC, see Override #1 Cohort 5). Module added to `traits.yaml` `modules:` list with `budget_lines: 800`.

Schema bumps: `properties.yaml` 0.9.0 → 0.10.0 (new "manifold-aware convergence" gate category + new sympy test class for geometric oracles); `traits.yaml` 0.7.0 → 0.8.0 (new trait class + multiple new types).

## Rationale

- **Why port Mazzucchi-Moretti-Remizov-Smolyanov 2023 first (vs the older Hörmander Lie-bracket framework)?** The MMRS 2023 paper is the most *mathematically mature* result in the research1.md+research2.md pillar evaluation: closed-form Gaussian-on-tangent kernel, explicit $R/12$ curvature correction with proof, three closed-form backends within scope (sphere, hyperbolic plane, torus), and direct application to SABR-on-H² (the C3 industrial showcase). It banks a *port paper* (the v2.8 release notes acknowledge the MMRS port with full citation) before tackling Hörmander Lie-bracket sub-Riemannian Chernoff (an OPEN research problem deferred to v3.1+ A3). The sympy curvature oracle infrastructure shipped with this ADR becomes a shared `scripts/manifold_curvature_kit.py` reused by v3.1 A3 — banking infrastructure is a strict positive for sequencing.
- **Why a new trait `BoundedGeometryManifold<F>` rather than extending `Grid1D` / `Grid2D` / `Grid3D`?** Existing grid types are *flat-$\mathbb{R}^d$* by construction: their `x_at(i)`, `bc_index`, `dx()` methods presume linear coordinates. Bolting a manifold structure on top would either require breaking the grid API (rejected — see ADR-0008 precedent for grid breakage cost) or duplicating the geometric methods on a per-grid basis. A new trait class for Riemannian structure is the suckless choice: composition over inheritance; the three backends are tiny structs that own only their geometric parameters (no grid coupling), and `ManifoldChernoff<M, F>` glues them to the Chernoff machinery via the standard `ChernoffFunction<F>` interface.
- **Why three reference backends (Torus, Sphere2, Hyperbolic2) and not just Sphere2?** Each backend has a distinct mathematical role: (a) Torus is the *zero-curvature* sanity check — the Chernoff function reduces to the standard heat kernel, giving an exact symbolic oracle for T21N (`F(τ)φ_k = exp(-τ k²) φ_k` for Fourier eigenmodes); (b) Sphere2 is the *positive-curvature* benchmark — G26 sub-test 1 (base, no R/12) and sub-test 2 (with R/12) measure the order promotion empirically; (c) Hyperbolic2 is the *negative-curvature* test — needed for SABR-on-H² (the C3 industrial showcase shipped in Wave D `examples/sabr_pricer.rs`). Cutting any of the three breaks a downstream story (oracle, gate, or industrial showcase). The three together cost ~150 LoC each in `manifold.rs` (closed-form $\exp_x$, scalar curvature constant, parallel transport closed form) — well within the 700-LoC budget.
- **Why `with_curvature_correction: bool` (boolean) rather than two separate types `ManifoldChernoff` and `ManifoldChernoffCorrected`?** Order is a runtime property of the wrapper, not a compile-time type-level distinction. The boolean flag is the suckless choice: one type, one impl, one set of tests. The two-types alternative would force users to convert between them and would double the rustdoc surface. G26 sub-tests select the variant via the same constructor with different booleans — clean, minimal.
- **Why closed-form $\exp_x$ only in v2.8 (no numerical ODE for geodesics on user-defined manifolds)?** Numerical geodesic flow (Runge-Kutta integration of the geodesic equation $\ddot{\gamma} + \Gamma_{ij}^k \dot{\gamma}^i \dot{\gamma}^j = 0$) requires either (a) symbolic Christoffel symbols (sympy at construction time, runtime evaluation expensive), or (b) a per-manifold compute graph (premature abstraction). v2.8 ships the three closed-form backends to validate the framework; user-defined manifolds remain *constructable* (the trait compiles for any user impl that provides the four geometric methods) but are not gated by acceptance tests. v3.x will revisit numerical geodesic flow once concrete use cases inform the API.
- **Why `scalar_curvature` (not full Riemann curvature tensor)?** Only $R(x)$ enters the MMRS 2023 first-order correction $[1 + (\tau/12) R(x)]$. The full Riemann tensor $R_{ijkl}$ is needed for higher-order corrections (order-3 from $R^2$, order-4 from $\nabla^2 R$, etc.) which the library does NOT ship in v2.8. A scalar-curvature method has minimal trait surface (1 scalar per query, no allocation) and matches the math.md §24 spec exactly.
- **Why `volume_element_log` (not `volume_element` directly)?** Numerically, $\sqrt{\det g(x)}$ for Sphere2 / Hyperbolic2 / Torus can underflow or overflow in $f32$ near coordinate singularities (e.g., the Poincaré disk metric blows up at $|x| \to 1$); operating in log-space gives ULP-stable arithmetic. The Chernoff integrand on $T_xM$ multiplies by $\exp(\text{vol\_log}(x) + \text{vol\_log}(y))$ for change-of-variables; log-space avoids the catastrophic-cancellation regime in the marginal chart.
- **Why explicitly NOT extend `TimedChernoffFunction<F>` (ADR-0070) to `ManifoldChernoff`?** The time-dependent manifold case ($L(t) = \Delta_{M(t)}$ on a *moving* metric $g(t)$) is a research problem of its own — the MMRS 2023 paper handles only static $g$. Coupling `ManifoldChernoff` to `TimedChernoffFunction` would require either (a) defining the time derivative of the curvature correction (open math), or (b) bridging to `apply_into` and silently freezing the metric at $t=0$ (footgun). The autonomous v2.8 surface ships as `impl ChernoffFunction<F> for ManifoldChernoff<M, F>` only; users wanting nonautonomous manifold evolution will need to wait for v3.x research crystallisation.
- **Why a NEW Override #1 Cohort 5 carve-out for `manifold.rs` (~700 LoC)?** The three backends each carry 30-50 lines of NORMATIVE rustdoc citing MMRS 2023 §4 equation numbers + Bismut 1984 §3 + the math.md §24 sympy oracle correspondence; the closed-form $\exp_x$ formulas need to be inline with the citations so reviewers can verify the formula vs the cited equation in one read. Splitting the file (e.g., one backend per file) would scatter the citation cluster and break the math.md §24 → `manifold.rs:LINE` cross-references that the v2.7 contract-first discipline requires. The 700 LoC budget mirrors the v2.0/v2.2 carve-outs (Cohort 2/3/4 file caps) — same math-co-location class. If the engineer reports >800 LoC at Wave B/C completion, the backends WILL be split into `manifold_sphere.rs` / `manifold_hyperbolic.rs` / `manifold_torus.rs` (v2.9 architecture review trigger).

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Implement only Sphere2 (no Torus, no Hyperbolic2) | Sphere2 alone has no exact symbolic oracle (Fourier coefficients of $\exp(t \Delta_{S^2})$ involve Legendre polynomial expansions — sympy-heavy, fragile gate). Torus provides the flat-eigenmode oracle (T21N — exact); Hyperbolic2 underwrites the C3 SABR-on-H² industrial showcase. Cutting either breaks a downstream story. |
| Direct grid-based discretisation of $\Delta_M$ (finite element on a triangulated manifold) | Per-manifold ad-hoc; no closed-form $\exp_x$; introduces a mesh dependency (rejected by ADR-0001 contract-first — no mesh primitives in `semiflow-core`). Also: the Mazzucchi-Moretti-Remizov-Smolyanov 2023 framework is mesh-FREE by construction (Gaussian on $T_xM$, pulled back via $\exp_x$); FEM would defeat the whole point of the port. |
| Brownian motion on the manifold (stochastic Monte-Carlo sampler) | Stochastic, not deterministic — breaks the project's bit-equality / SIMD-bit-equality contract (Override #1 math fidelity). MC error is $O(1/\sqrt{N_{\mathrm{samples}}})$ which is unacceptable for the gate-blocking convergence slope tests. |
| Numerical Christoffel-symbol-based geodesic flow (Runge-Kutta on the geodesic ODE) for the three backends | Defeats the point of closed-form backends. Closed-form $\exp_x$ is exact and zero-allocation; numerical RK introduces tolerance-tunable error that contaminates the convergence-slope gates. |
| Boolean flag `with_curvature_correction` → two distinct types `ManifoldChernoff` and `CorrectedManifoldChernoff` | Doubles the rustdoc surface. Order is a runtime property; the bool is the suckless choice. |
| Force ALL existing `ChernoffFunction` impls onto a flat-vs-manifold trait hierarchy | Breaks the v0.1.0 simple `ChernoffFunction<F>` contract. Manifold is opt-in; flat (the vast majority of existing impls) stays as-is. |
| Couple `ManifoldChernoff` to `TimedChernoffFunction<F>` (ADR-0070) by default | Time-dependent manifolds (`g = g(t)`) is open math. Forcing the time-dimension on every manifold construction would either compile-fail or silently freeze the metric — both footguns. Strict autonomy in v2.8; nonautonomous manifold is v3.x research. |
| Couple `ManifoldChernoff` to `LaplaceChernoffResolvent` (ADR-0069) by default | Resolvent on manifolds is the *Stokes phenomenon* domain (complex-$\lambda$ heavy). Forces the v4.0 B6 SemiflowComplex deferral one release early — premature. v2.8 ships autonomous + real-$\lambda$ only; resolvent-on-manifold is composable at the user's discretion (`LaplaceChernoffResolvent<ManifoldChernoff<Sphere2<f64>>>` will compile because both wrappers are generic over any `ChernoffFunction<F>`). |
| Generic over $\dim$ via const-generic `D` parameter on `ManifoldChernoff` | Two of the three v2.8 backends have fixed dimension (Sphere2 is 2D, Hyperbolic2 is 2D — natural mathematical objects); only Torus is generic. Pushing const-generic D up to `ManifoldChernoff` would force `Sphere2` and `Hyperbolic2` into `D: 2` const-generic carriers — boilerplate without semantic gain. The manifold's `dim()` method is the natural runtime accessor. |
| Override #1 carve-out budget of 500 LoC (no expansion) | Three backends × 150 LoC each = 450 LoC; plus the trait + `ManifoldChernoff` impl + curvature-correction code = ~250 LoC overhead = ~700 LoC total. Splitting into multiple files breaks the citation co-location justification. The 700-LoC budget is the suckless minimum; engineer flag if >800 at impl time. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing trait or struct is modified.
- **New module `crates/semiflow-core/src/manifold.rs`** (~700 LoC budget; HARD LIMIT 800 LoC per Override #1 Cohort 5 expansion). Constitution amendment v1.6.2 → v1.6.3 records the new Cohort 5 entry.
- **New trait class `BoundedGeometryManifold<F>`** — independent of `KillingRegion<F>` / `ReflectingRegion<F>` (B4); manifold is a *geometric structure on the entire space*, region is a *subset of $\mathbb{R}^d$ or $M$*. The two trait classes COULD be composed in v2.9+ (e.g., `KillingChernoff<ManifoldChernoff<Sphere2>, SphereCapRegion>` for a Dirichlet-on-spherical-cap problem) but no such composition is shipped in v2.8.
- **Dependency count unchanged** at 2/3 budget (still `num-traits`, `libm`). The closed-form $\exp_x$ formulas use `libm::sin` / `libm::cos` / `libm::cosh` / `libm::sinh` already available.
- **Schema bumps**: `properties.yaml` 0.9.0 → 0.10.0 (new "manifold-aware convergence" gate category G26 + new sympy gate class T21N for geometric oracles); `traits.yaml` 0.7.0 → 0.8.0 (new trait class `BoundedGeometryManifold<F>` + `ManifoldChernoff<M, F>` struct + 3 backend structs `Torus<F, D>` / `Sphere2<F>` / `Hyperbolic2<F>`). math.md is append-only (§24 NEW).
- **New gates**: G26 (RELEASE_BLOCKING — Sphere-S² slope, two sub-tests: base ≤ -0.95 and R/12-corrected ≤ -1.95 on $n \in \{16, 32, 64, 128\}$); T21N (NORMATIVE sympy — flat-torus eigenmode exact identity, 4 sub-checks).
- **No L-gate for `ManifoldChernoff` in v2.8.** The SABR-on-H² industrial showcase (Wave D `examples/sabr_pricer.rs`) needs a latency gate eventually, but per-tick latency calibration requires one RC cycle of hardware-profile data (the v2.6 → v2.7 promotion ladder). Defer L_SABR_PTICK to v2.8.0-rc.1 advisory or v2.9.
- **CITATIONs added to math.md §24**: Mazzucchi-Moretti-Remizov-Smolyanov 2023 *Math. Nachr.* — Theorem 1 (the Gaussian-on-tangent-space Chernoff approximation with $R/12$ correction); Bismut 1984 *Large Deviations and Stochastic Mechanics*, Birkhäuser (the geometric framework for stochastic semigroups on manifolds). The CITATION-only mathematics (proof of Theorem 1 in MMRS 2023 §4) is referenced; the library reproduces only the *formula*.

## Migration

None for end-users. v2.7 binaries / crates link against v2.8 without recompilation. The new trait + types are additive; nobody is forced to depend on them.

The Wave D `examples/sabr_pricer.rs` (SABR-on-H² industrial showcase) is opt-in; it constructs `ManifoldChernoff<Hyperbolic2<f64>>` and validates against published SABR closed-form bench numbers but is not part of the test gate.

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; manifold backends use closed-form `libm` trigs/hyperbolics, no allocation.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `BoundedGeometryManifold<F>`, `ManifoldChernoff<M, F>`, all 3 backends.
- ADR-0026 — `ChernoffFunction<F>` super-trait; `ManifoldChernoff<M, F>` implements it.
- ADR-0041 — `apply_into` + `ScratchPool`; `ManifoldChernoff::apply_into` uses the scratch pool for the per-tangent-vector integration buffer.
- ADR-0068 — v2.6 BC widening (`KillingRegion<F>` trait pattern reused for the v2.8 `ReflectingRegion<F>` sibling design, ADR-0072); `BallRegion<F, D>` reused for spherical caps in future v2.9+ composition.
- ADR-0070 — Howland nonautonomous lift (v2.7); `ManifoldChernoff` is explicitly NOT a `TimedChernoffFunction` by default — nonautonomous manifold is v3.x research.
- ADR-0072 — Neumann via image method (B4, v2.8 companion ADR; shared release window; `ReflectingRegion<F>` sibling to `KillingRegion<F>`).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v2.8 (Manifold Pillar) — release-level roadmap.
- math.md §24 (NEW v2.8) — Riemannian manifold Chernoff normative spec.
- `.dev-docs/research/research1.md` — Mazzucchi-Moretti-Remizov-Smolyanov 2023 pillar selection rationale.
- `.dev-docs/constitution.md` v1.6.3 (NEW) — Override #1 Cohort 5 carve-out for `manifold.rs`.

## Amendments

(none at acceptance time)
