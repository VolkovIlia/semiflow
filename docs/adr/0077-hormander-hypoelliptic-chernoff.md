# ADR-0077 — Hörmander Hypoelliptic Chernoff Approximation (A3)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v3.1 Wave B (the headline math pillar of the Hörmander Research Pillar release; rides on v3.0 ADR-0073 `ApproximationSubspace<2, F>` tangency witness). Wave A ships the `VectorField<F, D>` trait + Lie-bracket sympy infrastructure; Wave B (this ADR) ships `HypoellipticChernoff<F, D, M>` and the closed-form Kolmogorov + Heisenberg backends; Wave C ships B7 (`quantum_graph.rs` — ADR-0078); Wave D ships the G28 calibration sweep; Wave E ships `docs/papers/hormander-paper-draft.md`. Independent of B7. NEW algorithmic content beyond the v0.x–v3.0 elliptic / manifold / graph kernel families — this is the FIRST hypoelliptic kernel class in the library.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` generic over F), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0071 (v2.8 A4 manifold pillar — provides the `manifold_curvature_kit.py` sympy harness that v3.1 Wave A extends into `lie_bracket_kit.py` for vector-field commutator expansion), ADR-0073 (v3.0 B1 `ApproximationSubspace<K, F>` — the K=2 tangency witness consumed by `HypoellipticChernoff::new` at construction time), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` return type used by the new impl's `growth()`).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW kernel class `HypoellipticChernoff<F, D, M>` with `VectorField<F, D>` as a NEW open-extensibility trait class.
- **Mathematical foundation**: math.md §28 (NORMATIVE library — `HypoellipticChernoff` palindromic Strang-Hörmander semantics, step-2 Carnot restriction; CITATION Hörmander 1967 *Acta Math.* §1 — bracket-generating condition; Kolmogorov 1934 *Annals of Math.* — explicit fundamental solution on the d=2 Kolmogorov phase space (G28 oracle); Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 — order-2 tangency framework, riding on v3.0 ADR-0073; Folland 1975 *Ark. Mat.* §2 — Heisenberg group sub-Laplacian dilations).
- **Acceptance gates added**: G28 (RELEASE_BLOCKING — Kolmogorov-kernel slope ≤ −1.95 on d=2 phase space `(x, v)` with X₀ = v·∂_x, X_1 = ∂_v, $n \in \{16, 32, 64, 128\}$; lives in `tests/hormander_kolmogorov_slope.rs` new file, feature `slow-tests`); G29 (RELEASE_BLOCKING — mass conservation $|\int p(t, x, v)\,dx\,dv - 1| \le 10^{-10}$ across the same sweep; sub-Markov property verification); T_HORM (NORMATIVE sympy — Kolmogorov 1934 fundamental solution symbolic verification: PDE residual = 0, initial condition, first moment matches, mass = 1).

## Context

A hypoelliptic operator $L$ is one for which $Lu \in C^\infty$ implies $u \in C^\infty$ — even when $L$ is not elliptic (i.e., the principal symbol is degenerate at some directions). Hörmander 1967 showed that for second-order operators of the form
$$
L = X_0 + \tfrac{1}{2} \sum_{k=1}^{M} X_k^2,
$$
where $\{X_i\}$ are smooth vector fields, $L$ is hypoelliptic iff the **bracket-generating condition** holds:
$$
\text{Lie}(X_0, X_1, \ldots, X_M)|_x = T_x M \qquad \forall x \in M.
$$
That is, iterated Lie brackets of the $X_i$ span the tangent space at every point. The canonical examples are:
- **Kolmogorov 1934** on $\mathbb{R}^2 = \{(x, v)\}$ with $X_0 = v \cdot \partial_x$, $X_1 = \partial_v$: the bracket $[X_1, X_0] = \partial_x$ generates the missing direction. The fundamental solution is the explicit Gaussian (math.md §28.2 equation 28.2) — one of the only hypoelliptic kernels with closed form.
- **Heisenberg group** $\mathbb{H}^1 = \{(x, y, t)\}$ with the left-invariant fields $X_1 = \partial_x - \tfrac{y}{2} \partial_t$, $X_2 = \partial_y + \tfrac{x}{2} \partial_t$, $X_0 = 0$: the bracket $[X_1, X_2] = \partial_t$ generates the missing direction. The sub-Laplacian $L = \tfrac{1}{2}(X_1^2 + X_2^2)$ governs the Heisenberg semigroup (Folland 1975).

The **Festschrift §3** open problem — frame the v3.1 publication target — is: does the Chernoff product formula $(F(t/n))^n f \to e^{tL} f$ admit a *constructive* hypoelliptic approximant $F(\tau)$? The non-refereed Kalmetev 2023 Keldysh preprint proposes a structure but does not prove convergence; the Festschrift entry explicitly cites this as OPEN.

The v3.0 `ApproximationSubspace<2, F>` machinery (ADR-0073) gives the K=2-jet witness in Rust; the Galkin-Remizov 2025 *IJM* Theorem 3.1 gives the abstract convergence rate. v3.1 ships **the first peer-reviewable Rust implementation of a hypoelliptic Chernoff approximation**: the **palindromic Strang-Hörmander** decomposition (math.md §28.3 equation 28.3), order-2 on the strict subspace $D(L^2)$ for step-2 Carnot groups (Kolmogorov, Heisenberg), with the Kolmogorov 1934 closed-form kernel as the G28 oracle.

**Why order-2 only**: the v3.0 trait surface gives `ApproximationSubspace<K, F>` for any K, but the order-K hypoelliptic Chernoff approximation is **OPEN** for K ≥ 3 — the Festschrift §3 problem is explicitly to find such approximants. Shipping order-2 in v3.1 (with the Galkin-Remizov 2025 tangency framework) is itself the publication result; order ≥ 3 is Tier C / v3.2+ research.

**Why step-2 Carnot only**: the bracket-generating condition has a *step* (the minimum number of nested brackets needed to span the tangent space). Step 2 means a single bracket suffices (Kolmogorov, Heisenberg). Step ≥ 3 (Engel group, free nilpotent groups of higher step) requires nested bracket compositions in the algorithm that we have not validated; the convergence gate becomes fragile at higher step because the closed-form fundamental solutions are not available. v3.1.0 gates ONLY step-2 (Kolmogorov + Heisenberg); user-defined step-≥ 3 backends compile via the `VectorField<F, D>` trait but are NOT gated.

**Why palindromic Strang-Hörmander**: the operator $L = X_0 + \tfrac{1}{2} \sum X_k^2$ splits into a drift part ($X_0$) and a diffusive part ($\sum X_k^2$). Each diffusive $X_k^2$ is a Hörmander-1 generator (the single vector field $X_k$ is bracket-generating in 1D); its exponential $\exp(\tau X_k^2 / 2)$ admits an exact representation via the Hörmander-1 oracle. The Strang composition

$$
F(\tau) = e^{\tau X_0 / 2} \circ \prod_{k=1}^{M} e^{\tau X_k^2 / 2} \circ e^{\tau X_M^2 / 2} \circ \prod_{k=M-1}^{1} e^{\tau X_k^2 / 2} \circ e^{\tau X_0 / 2}
$$

is palindromic in the inner $X_k^2$ legs (sandwich symmetry) and outer in $X_0$ (Strang half-splits). The order-2 tangency follows from the standard Strang argument once the per-leg exponentials are order-∞ (exact). The Galkin-Remizov 2025 framework (Theorem 3.1, K=2) gives the global order-2 claim on $f \in D(L^2)$.

**Why VectorField as a NEW trait**: existing kernels (DiffusionChernoff, ShiftChernoff1D, etc.) consume parameter closures (`a: fn(f64) -> f64`, `b: fn(f64) -> f64`). Vector fields are a *fundamentally different* object class: they map points to tangent vectors (`evaluate(x: &[F; D], out: &mut [F; D])`) and admit a Lie bracket operation (`bracket_with(other, x, out)`) that no scalar-field closure can support. The trait is small (1 required method + 1 optional method) and open-extensible (any backend implements its own vector field). This mirrors `BoundedGeometryManifold<F>` (v2.8 ADR-0071) — a NEW trait class for a NEW geometric object.

## Decision

Ship two additive public-surface items in v3.1 Wave A + Wave B:

**Wave A (the trait class) — `pub trait VectorField<F: SemiflowFloat, const D: usize>` in `crates/semiflow-core/src/hormander.rs`**:
```rust
pub trait VectorField<F: SemiflowFloat, const D: usize> {
    /// Writes the vector-field value v_x = X(x) into out. out.len() == D.
    /// Returns DomainViolation on any NaN/Inf in x or out, or shape mismatch.
    fn evaluate(&self, x: &[F; D], out: &mut [F; D]) -> Result<(), SemiflowError>;

    /// Optional: writes the Lie bracket [self, other](x) into out via the
    /// directional-derivative formula [X, Y](x) = (dY|_x)(X(x)) - (dX|_x)(Y(x)).
    /// Default impl uses finite-difference Jacobians at f64 precision; backends
    /// with closed-form Jacobians (Carnot left-invariant fields, etc.) override
    /// for sympy-traceable accuracy. Used by step-checker for Hörmander
    /// bracket-generating condition verification at construction time.
    fn bracket_with(
        &self,
        other: &dyn VectorField<F, D>,
        x: &[F; D],
        out: &mut [F; D],
    ) -> Result<(), SemiflowError> { /* default impl via central-difference Jacobians */ }
}
```

**Wave B (the kernel) — `pub struct HypoellipticChernoff<F: SemiflowFloat, const D: usize, const M: usize>` in `crates/semiflow-core/src/hormander.rs`**:
```rust
pub struct HypoellipticChernoff<F: SemiflowFloat, const D: usize, const M: usize> {
    x0_drift: Box<dyn VectorField<F, D>>,             // X_0 — the drift field (may be zero)
    x_diff: [Box<dyn VectorField<F, D>>; M],          // X_1, ..., X_M — the diffusive fields
    grid: GridND<F, D>,                                // discretisation (per-axis Grid1D tensor product)
    step_checker_result: StepCheckerResult,            // cached at construction (step-2 verified)
}
```
Constructor:
```rust
impl<F: SemiflowFloat, const D: usize, const M: usize> HypoellipticChernoff<F, D, M> {
    pub fn new(
        x0_drift: Box<dyn VectorField<F, D>>,
        x_diff: [Box<dyn VectorField<F, D>>; M],
        grid: GridND<F, D>,
    ) -> Result<Self, SemiflowError>;
}
```
Validation at construction (returns `DomainViolation` on any failure):
- `D >= 2` and `M >= 1` (degenerate Hörmander setups are rejected; pure-elliptic should use existing diffusion kernels).
- The **step-2 bracket-generating check** at the grid's centre point: compute $\{X_1, \ldots, X_M, [X_1, X_2], \ldots, [X_{M-1}, X_M]\}$ at the centre and verify span = D via SVD. The check produces a `StepCheckerResult { step: 2, generators_at_check_point: D }`. Step ≥ 3 user-defined backends FAIL this check and return `DomainViolation` — the v3.1.0 release ONLY gates step-2 (the Kolmogorov + Heisenberg cases).
- Per-axis grid uniformity (`GridND<F, D>` invariant from v0.5 tensor product).

The `apply_into(τ, src, dst, scratch)` algorithm realises the palindromic Strang-Hörmander formula:

```
HypoellipticChernoff::apply_into(τ, src, dst, scratch):
  // Half-step drift: dst1 := e^{τ X_0 / 2} src
  apply_x0_drift(τ / 2, src, &mut dst1, scratch)?;
  // Inner palindromic loop on diffusive fields X_1, ..., X_M
  // Forward sweep
  for k in 1..M:
      apply_xk_squared(τ, k, &dst1, &mut dst1, scratch)?;
  apply_xk_squared(τ, M, &dst1, &mut dst1, scratch)?;     // middle leg (full step)
  // Backward sweep
  for k in (1..M).rev():
      apply_xk_squared(τ, k, &dst1, &mut dst1, scratch)?;
  // Half-step drift: dst := e^{τ X_0 / 2} dst1
  apply_x0_drift(τ / 2, &dst1, dst, scratch)?;
  Ok(())
```

Each `apply_x0_drift(τ_half, src, dst)` is a Hörmander-1 drift step: for closed-form Carnot backends (Kolmogorov, Heisenberg), the drift exponential is exact (linear flow along $X_0$); for general user-defined fields, the drift uses a `ShiftChernoff1D`-style first-order shift along $X_0$ (caller responsibility — caller-supplied per-grid-point lookup table).

Each `apply_xk_squared(τ, k, src, dst)` is the exponential of the SQUARE of vector field $X_k$. For step-2 Carnot backends, the second-order ODE $\partial_t u = \tfrac{1}{2} X_k^2 u$ admits a closed-form solution via the Hörmander-1 fundamental solution (math.md §28.4); the library ships the Kolmogorov + Heisenberg closed forms as backends. For general $X_k^2$, this is a 1D Sturm-Liouville-style diffusion along the $X_k$ flow lines — beyond v3.1 scope.

The `order()` method returns `2` per Theorem 28.1 (math.md §28.5) — order-2 tangency on $f \in D(L^2)$ for step-2 Carnot $L$. `growth()` returns `Growth { multiplier: F::one(), omega: F::zero() }` — sub-Markov on the canonical Hörmander generator (mass-preserving; verified by G29).

**`impl ApproximationSubspace<2, F>` for `HypoellipticChernoff<F, D, M>`**: opt-in (third v3.1 opt-in beyond the existing v3.0 trio):
```rust
impl<F: SemiflowFloat, const D: usize, const M: usize> ApproximationSubspace<2, F>
    for HypoellipticChernoff<F, D, M>
{
    fn in_subspace(&self, f: &Self::S) -> bool { /* ≥ 5^D grid points per axis */ }
    fn jet(&self, f: &Self::S, out: &mut [Self::S]) -> Result<(), SemiflowError> {
        // out: [f, L f, L² f]; L = X_0 + (1/2) Σ X_k²
        // 2 iterations of the discretised Hörmander operator
    }
}
```

This is the **fourth** v3.x `ApproximationSubspace` opt-in (the v3.0 trio plus this v3.1 hypoelliptic addition); the trait remains open-extensible per ADR-0073 §"Limitations" point 1.

**Closed-form Carnot backends shipped in v3.1**:
- `KolmogorovPhaseSpace<F>` impl `VectorField<F, 2>` — the d=2 phase-space pair `{X_0 = v·∂_x, X_1 = ∂_v}`. Backbone of G28 (Kolmogorov 1934 explicit kernel).
- `HeisenbergGroup<F>` impl `VectorField<F, 3>` — the left-invariant pair `{X_1 = ∂_x - (y/2)∂_t, X_2 = ∂_y + (x/2)∂_t}` (plus the implicit $X_0 = 0$). Folland 1975 dilations + Hulanicki 1976 sub-Laplacian heat kernel; NOT gated in v3.1 (closed-form fundamental solution involves complex-valued integrals — defer detailed validation to v3.2+ when v4.0 SemiflowComplex is available); shipped as a constructive instance to demonstrate the framework beyond Kolmogorov.

User-defined Carnot backends (step ≥ 3) — Engel group, free nilpotent step-3, etc. — compile via the `VectorField<F, D>` trait but FAIL the step-2 construction check; documented as Tier C deferral (v3.2+).

## Rationale

**Why ship now (v3.1 vs v3.2+)**: the v3.0 trait infrastructure (`ApproximationSubspace<K, F>`, `Growth<F>`, `Evolver<C, F>`) is fresh; the Galkin-Remizov 2025 IJM citation chain is fresh; the manifold curvature sympy harness (`scripts/manifold_curvature_kit.py` from v2.8 A4) is structurally identical to what we need for Lie-bracket verification. Shipping in v3.1 lets us claim the first peer-reviewable Rust hypoelliptic Chernoff approximation IN THE SAME RELEASE CYCLE as the trait infrastructure that enables it.

**Why VectorField as a new trait (vs ad-hoc closures)**: vector fields are a *first-class geometric object* — they admit Lie brackets, parallel transport, and flow-line integration. Encoding them as `fn(&[f64]) -> [f64; D]` closures loses the bracket semantics needed for the step-checker; an explicit trait makes the Lie-bracket signature explicit and admits closed-form Jacobian overrides for the Carnot backends. The same argument that motivated `BoundedGeometryManifold<F>` (v2.8 ADR-0071) over ad-hoc metric closures applies here.

**Why palindromic Strang-Hörmander (vs Lie-Trotter or higher-order Magnus)**: Lie-Trotter is order-1 (wastes the v3.0 K=2 tangency); higher-order Magnus is open math for non-commuting $X_i$ at order ≥ 3 (the Festschrift §3 question itself). Strang is the *correct* order-2 choice and lets us harvest the v3.0 K=2 witness immediately.

**Why step-2 only in v3.1**: the closed-form fundamental solutions are step-2 (Kolmogorov, Heisenberg sub-Laplacian); step-3 (Engel, free nilpotent) requires Folland-Stein dilations + non-isotropic Hardy spaces — significant additional library content. The convergence gate G28 would be fragile (no closed-form oracle) at step ≥ 3. v3.1.0 ships the *foundation* with step-2 gated; step-3 backends opt-in in v3.2+ once the user-defined `VectorField` API has been exercised on the closed-form cases.

**Why Kolmogorov (not Heisenberg) as the gate oracle**: Kolmogorov 1934 is *purely real-valued* and admits a textbook explicit Gaussian fundamental solution (math.md §28.2). Heisenberg sub-Laplacian heat kernel involves complex-valued integrals (Beals-Gaveau-Greiner 1996); validating it requires v4.0 `SemiflowComplex`. v3.1 ships Heisenberg as a constructive *instance* of the framework (compiles, runs, matches the step-2 checker) but does NOT gate convergence on it.

**Why a NEW Override #1 carve-out (Cohort 6) vs splitting hormander.rs into multiple files**: the file co-locates (a) the `VectorField<F, D>` trait definition; (b) `HypoellipticChernoff<F, D, M>` impl + the palindromic Strang-Hörmander algorithm; (c) `KolmogorovPhaseSpace<F>` + `HeisenbergGroup<F>` closed-form backends; (d) the step-2 bracket-generating checker. Splitting into per-backend files would (i) duplicate the `VectorField` trait surface in rustdoc citations (each backend's rustdoc cites Hörmander 1967 + Kolmogorov 1934 inline); (ii) break the proof-cited traceability that the math-co-location discipline requires (math.md §28 cites specific equations inline). Same justification class as Cohort 5 (manifold.rs).

**Why ~700 LoC budget (vs 500 default)**: the trait + 2 closed-form backends + step-checker + palindromic Strang algorithm + rustdoc citations approximates 700 LoC. Each backend's `evaluate` + `bracket_with` requires 30-50 lines of NORMATIVE rustdoc citing Hörmander 1967 §1 + Kolmogorov 1934 + Folland 1975 §2 + math.md §28.X equation numbers (mirror manifold.rs cohort 5 pattern). HARD LIMIT 800 LoC; if exceeded at engineer Wave B/C time, split into `hormander_kolmogorov.rs` + `hormander_heisenberg.rs` — v3.2 architecture review trigger (mirror manifold.rs's v2.9 split trigger).

## Alternatives Considered

**Alt 1 — Brownian motion on Carnot groups simulation (Monte Carlo)**: rejected. Stochastic semigroup approximation is fundamentally different from deterministic Chernoff product formulas. The library's core thesis (deterministic, contract-first, sub-microsecond latency) is incompatible with Monte-Carlo dispersion. Stochastic methods are appropriate for the SDE side-track at the *application* layer (HFT pricer wrapping the deterministic kernel) but not in `semiflow-core`.

**Alt 2 — Magnus K=4 in hypoelliptic setting**: rejected. The Magnus expansion for non-commuting vector fields requires the Baker-Campbell-Hausdorff series; at order 4 the BCH series involves 8+ commutator terms whose convergence on the bracket-generating Lie algebra is OPEN math (this is the Festschrift §3 question itself). Shipping a *speculative* K=4 in v3.1 would risk publication retraction. v3.1 ships order-2 (Strang, rigorously order-2 by Galkin-Remizov tangency); higher orders defer to v3.2+ if/when the Festschrift problem is resolved.

**Alt 3 — Reuse `ManifoldChernoff<M, F>` with a sub-Riemannian metric**: rejected. The sub-Riemannian (Carnot-Carathéodory) metric on a Hörmander generator has *unbounded sectional curvature* (the horizontal distribution collapses tangentially); this VIOLATES the `BoundedGeometryManifold<F>` hypothesis B1 (positive injectivity radius) and hypothesis B2 (uniformly bounded curvature). The MMRS 2023 framework that ADR-0071 ports DOES NOT APPLY to sub-Riemannian geometry. A new trait class (`VectorField<F, D>`) is the mathematically correct choice.

**Alt 4 — Cite Kalmetev 2023 Keldysh preprint as the algorithm source**: rejected. Kalmetev 2023 is a non-refereed preprint (Keldysh Institute Preprint 49, 2023); using it as the algorithmic foundation would expose the library to citation-quality risk. v3.1 cites Hörmander 1967 (refereed) + Kolmogorov 1934 (refereed) + Galkin-Remizov 2025 IJM (refereed) as the foundations; Kalmetev 2023 is cited ONCE in the paper draft (`docs/papers/hormander-paper-draft.md`) as *motivation* but is NOT used in the algorithm or the gates.

## Consequences

**Positive**:
- The library acquires its **first peer-reviewable hypoelliptic kernel class**, opening a publication track in *Russ. J. Math. Phys.* (target: v3.1-rc.1 submission).
- The `VectorField<F, D>` trait + `lie_bracket_kit.py` sympy harness become shared infrastructure that v4.0 B6 (matrix-valued FN-style coupled systems) and v4.0 A6 (point-wise evaluation API for d ≥ 4) consume.
- The step-2 bracket-generating checker is a reusable structural-verification primitive — pollinates the v4.0 d-D anisotropic shift (Remizov 2019 JMP) by validating its anisotropic-shift Lie algebra.
- The Kolmogorov backend's closed-form kernel becomes a permanent oracle in the test suite — testable forever against any future hypoelliptic kernel additions.

**Negative**:
- One new Override #1 file (Cohort 6 hormander.rs ~700 LoC HARD LIMIT 800); override count stays 3 / 3 (file-list expansion of existing Override #1, no new override category).
- The step-checker is O(M²) at construction time (each pair $[X_i, X_j]$ evaluated at the grid centre); for large M this could grow expensive — but v3.1 ships M ≤ 2 backends (Kolmogorov M=1, Heisenberg M=2), so the overhead is constant.
- The Heisenberg backend ships UN-GATED in v3.1 (no closed-form Real-valued oracle); this creates a test-coverage gap that v4.0+ SemiflowComplex closes.
- The `Box<dyn VectorField<F, D>>` field type allocates on construction (one Box per field); this is a one-time cost (kernel is long-lived) and avoids monomorphisation explosion for user-defined fields.

**Neutral**:
- No deprecation, no migration playbook update needed (strictly additive on the v3.0 public surface).
- The 12-month `v2_compat` shim (ADR-0074) is unaffected.

## Cross-references

- math.md §28 (NEW section — NORMATIVE library semantics + CITATION mathematics)
- properties.yaml: G28 (RELEASE_BLOCKING — Kolmogorov-kernel slope), G29 (RELEASE_BLOCKING — mass conservation), T_HORM (NORMATIVE sympy — Kolmogorov 1934 fundamental solution symbolic verification)
- traits.yaml schema 1.0.0 → 1.1.0 (additive — `VectorField<F, D>` trait + `HypoellipticChernoff<F, D, M>` struct + `KolmogorovPhaseSpace<F>` + `HeisenbergGroup<F>` backends + the fourth `impl ApproximationSubspace<2, F>` opt-in entry)
- constitution v1.7.0 → v1.7.1 PATCH (Cohort 6 file-list expansion of Override #1 for `hormander.rs` ~700 LoC HARD LIMIT 800)
- ADR-0073 (the K=2 witness consumed by the order-2 tangency claim — direct dependency)
- ADR-0074 (the `Growth<F>` typed return used by the new impl's `growth()` — direct dependency)
- ADR-0078 (B7 quantum graphs — v3.1 sibling pillar, independent)
- Festschrift open problem: see `docs/papers/hormander-paper-draft.md` §3 for the publication narrative (Wave E engineer ships)

## References

- L. Hörmander, *Hypoelliptic second order differential equations*, **Acta Mathematica** 119:1 (1967), pp. 147-171. — The bracket-generating condition (§1 of the cited paper) is the foundational hypothesis for the v3.1 kernel class.
- A. N. Kolmogorov, *Zur Theorie der stetigen zufälligen Prozesse*, **Mathematische Annalen** 108 (1934), pp. 149-160. — The d=2 phase-space fundamental solution (G28 oracle); the only closed-form hypoelliptic kernel used in this release.
- A. V. Galkin and I. D. Remizov, *Tangency of Chernoff approximations to operator semigroups on Banach spaces*, **Israel Journal of Mathematics** (2025). — Theorem 3.1 (the K=2 tangency framework that underwrites the order-2 claim on $D(L^2)$); also the foundation for ADR-0073's `ApproximationSubspace<K, F>`.
- G. B. Folland, *Subelliptic estimates and function spaces on nilpotent Lie groups*, **Arkiv för Matematik** 13 (1975), pp. 161-207. — §2 dilations + sub-Laplacian structure on the Heisenberg group (the v3.1 Heisenberg backend; un-gated in v3.1, ships as constructive instance).
- R. Beals, B. Gaveau, P. Greiner, *Complex Hamiltonian mechanics and parametrices for subelliptic Laplacians*, **Bull. Sci. Math.** 121 (1997). — Cited for completeness in the rustdoc of `HeisenbergGroup<F>` (the closed-form heat kernel involves complex integrals, deferred validation to v4.0).
- A. Kalmetev, *Chernoff-type approximations on Carnot groups*, **Keldysh Institute Preprint** 49 (2023). — Non-refereed preprint cited ONLY in `docs/papers/hormander-paper-draft.md` as motivation; NOT used in v3.1 algorithm or gates.
- v2.8 manifold pillar precedent: ADR-0071 + math.md §24 (the `BoundedGeometryManifold<F>` trait class is the structural model for the v3.1 `VectorField<F, D>` trait class; the `manifold_curvature_kit.py` sympy harness is the structural model for `lie_bracket_kit.py`).
- v3.0 trait foundation: ADR-0073 (`ApproximationSubspace<K, F>` — the K=2 witness); ADR-0074 (`Growth<F>` typed return); ADR-0075 (the v3.0 sibling A5 ζ⁴ that ALSO consumes the K=6 witness — pattern for `HypoellipticChernoff` consuming the K=2 witness in v3.1).

## AMENDMENT 1 (2026-06-05) — v7.0.0 step-k Carnot k≥3 investigation: HONEST-DEFER, no published path

v7.0.0 Phase-0 re-surveyed step-k Carnot k≥3 for a closed-form-oracle-gated Chernoff kernel. Verdict: **PUBLISHED-PATH = NO**. The canonical result, Boscain–Gauthier–Rossi 2010 (arXiv:1002.0688; *J. Math. Sci.* 199, 2014), reduces the Engel (step-3) kernel to the 1-D quartic-oscillator heat equation `∂_t ψ = (d²/dθ² − (αθ²+β)²)ψ`, for which the paper states verbatim "no general explicit solutions are known" (§3.2.3) and explicitly leaves operator-splitting outside its scope; no 2024–2026 work closes this gap. The 2025 Remizov Festschrift (arXiv:2508.18650) confirms the Remizov–Smolyanov programme frontier is 1-D `af''+bf'+cf` super-fast-convergence and resolvent Chernoff — step-≥3 Carnot is not listed as an announced problem. Independently, the BGR Engel kernel requires a triple GFT integral plus spectral diagonalisation of the quartic-oscillator Hamiltonian, which is incompatible with the `no_std + libm` oracle requirement. **HONEST-DEFER**: the closed-form-oracle requirement for self-convergence validation is independently unsatisfiable for any step-≥3 Carnot heat kernel; no `libm`-compatible closed form exists. The shipped frontier remains step-2 Hörmander (ADR-0077 — Kolmogorov, Heisenberg) and the step-3 Engel self-convergence harness (ADR-0095, slope −43.95 super-exponential). A qualified ATTEMPT-ORIGINAL escape hatch exists (palindromic Strang-Hörmander step-4 validated by self-convergence only, following the ADR-0095 pattern, pre-flighted via `lie_bracket_kit.py` sympy BCH before any Rust), but it is unvalidated research with real obstruction risk (BCH-depth truncation for step≥3) and is NOT scheduled; it remains available under the user's "research → attempt → honest-defer" mandate only if a research wave is explicitly allocated.
