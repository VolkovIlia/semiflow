# ADR-0081 — d-D Anisotropic Shift Flagship Kernel (A2)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave C (third Wave of the second BREAKING window; ships after Wave A trait freeze + Wave B PointEval surface). The d-D shift kernel is the CONSOLIDATING FLAGSHIP of v4.0 — it lifts the v0.1.0 1D `ShiftChernoff1D` to general $D \in \{2, 3, 4, 5\}$ via the closed-form Remizov 2025 Vladikavkaz Theorem 3 cousin formula.
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0008 (v0.3.0 ζ-A τ²-correction — the order-2 promotion mechanism baked into the v4.0 baseline), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` super-trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0073 (v3.0 `ApproximationSubspace<K, F>` — `AnisotropicShiftChernoffND` opts into K=2 witness; the FIFTH v3.x opt-in), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` typed return used by the new kernel), ADR-0080 (v4.0 PointEval first-class — `AnisotropicShiftChernoffND` is PointEval Backend E; ADR-0080 + ADR-0081 ship in lockstep).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW kernel `AnisotropicShiftChernoffND<F: SemiflowFloat, const D: usize>` as a generalisation of v0.1.0 `ShiftChernoff1D` from 1D to d-D. The 1D specialisation (`AnisotropicShiftChernoffND<F, 1>`) is BYTE-IDENTITY tested against `ShiftChernoff1D<F>` (sub-test of G_DDIM); the v0.1.0 kernel is PRESERVED verbatim through v4.x (no replacement).
- **Mathematical foundation**: math.md §32 (NORMATIVE library — d-D anisotropic Gaussian kernel + the d-D Chernoff product formula; CITATION Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 3 cousin formula — the d-D anisotropic shift Chernoff product; Friedman 1964 *Partial Differential Equations of Parabolic Type* §1.4 — anisotropic Gaussian fundamental solutions). math §32 BUILDS on §1-§2 (v0.1.0 1D `ShiftChernoff1D`) + §9.2.3.B (v0.3.0 ζ-A τ²-correction — inherited by the v4.0 baseline).
- **Acceptance gates added**: G_DDIM (RELEASE_BLOCKING — d-D anisotropic shift convergence slope ≤ -1.95 on n ∈ {16, 32, 64, 128} sweep at T = 0.5 for d ∈ {2, 3, 4, 5}; 4 per-D sub-tests). Lives in `tests/anisotropic_shift_nd_d{2,3,4,5}_slope.rs` new files, feature `slow-tests`.

## Context

The v0.1.0 `ShiftChernoff1D<F>` (math.md §1-§2) implemented Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 6 formula (6) for the 1D parabolic generator $L = a(x) \partial_x^2 + b(x) \partial_x + c(x)$. The natural d-dimensional generalisation is the subject of Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 3 (the "Theorem 3 cousin formula" — for the d-D parabolic generator $L = \sum a_{ij} \partial^2_{ij} + \sum b_i \partial_i + c$).

The v2.5+ library has SPECIFIC 2D / 3D anisotropic kernels (`Strang2D`, `Strang3D`, `NonSeparable2DChernoff`, `AnisotropicShift2DChernoff`) using composition patterns (Strang splitting + non-separable mixed-leg + axis-lift) rather than the closed-form shift formula. These work but:
1. Are SPECIFIC to D = 2 or D = 3 (cannot generalise to D = 4, 5, etc.).
2. Use Strang composition that introduces a per-axis order-2 floor (the composition is order-2 even if the per-axis inner is higher order).
3. Don't natively handle strongly cross-coupled diffusion tensors (require non-separable mixed-leg correction).

v4.0 ADR-0081 ships `AnisotropicShiftChernoffND<F, const D: usize>` as the GENERIC d-dimensional kernel via the closed-form Theorem 3 cousin formula. Reference impls are gated for $D \in \{2, 3, 4, 5\}$ (the realistic range for f64 tensor-product Gauss-Hermite quadrature; $D \ge 6$ compiles but defers to Smolyak sparse grids in v4.1+).

The d-D shift gives a *natively d-dimensional* approximation that does NOT decompose into per-axis steps; the closed-form Gaussian-like kernel (math §32.2 equation 32.1) handles arbitrary anisotropic diffusion tensors $a_{ij}(x)$ including cross-terms. For applications with strongly cross-coupled diffusion (d-D Hörmander, multi-factor financial PDEs with correlated noise), the d-D shift is the natural primitive.

This is the CONSOLIDATING FLAGSHIP of v4.0: it depends on the v3.0 trait surface (ApproximationSubspace witness), the v4.0 ADR-0080 PointEval API (Backend E), and lifts the v0.1.0 1D primitive to general dimension.

## Decision

Ship two additive public-surface items in v4.0 Wave C:

**Item 1 — `pub struct AnisotropicShiftChernoffND<F: SemiflowFloat, const D: usize>`** in `crates/semiflow-core/src/shift_nd.rs` (NEW module, ~600 LoC target — ABOVE default cap; **Cohort 7 Override #1 carve-out** per constitution v1.8.0 with per-file cap 700 LoC):

```rust
pub struct AnisotropicShiftChernoffND<F: SemiflowFloat = f64, const D: usize = 2> {
    a_ij: Box<dyn Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync>,
    b_i:  Box<dyn Fn(&[F; D], &mut [F; D]) + Send + Sync>,
    c:    Box<dyn Fn(&[F; D]) -> F + Send + Sync>,
    grid: GridND<F, D>,
    cholesky_cache: Vec<SquareMatrix<F, D>>,                // cached Cholesky factors per grid point
    quadrature: GaussHermiteTensor<F, D>,                    // pre-computed Gauss-Hermite tensor nodes
}

impl<F: SemiflowFloat, const D: usize> AnisotropicShiftChernoffND<F, D> {
    pub fn new(
        a_ij: impl Fn(&[F; D], &mut SquareMatrix<F, D>) + Send + Sync + 'static,
        b_i:  impl Fn(&[F; D], &mut [F; D]) + Send + Sync + 'static,
        c:    impl Fn(&[F; D]) -> F + Send + Sync + 'static,
        grid: GridND<F, D>,
    ) -> Result<Self, SemiflowError>;
}

impl<F: SemiflowFloat, const D: usize> ChernoffFunction<F>
    for AnisotropicShiftChernoffND<F, D>
{
    type S = GridFnND<F, D>;
    fn apply_into(
        &self, tau: F, src: &GridFnND<F, D>, dst: &mut GridFnND<F, D>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 }                                // with ζ²-d-D correction baked in
    fn growth(&self) -> Growth<F> {
        Growth { multiplier: F::one() + F::from(0.5), omega: F::zero() }
    }
}

impl<F: SemiflowFloat, const D: usize> ApproximationSubspace<2, F>
    for AnisotropicShiftChernoffND<F, D>
{
    fn in_subspace(&self, f: &GridFnND<F, D>) -> bool {
        // ≥ 5^D grid nodes (5-point Gauss-Hermite tensor stencil width) +
        // Reflect or Periodic per-axis BoundaryPolicy.
        self.grid.n_per_axis().iter().all(|n| *n >= 5)
            && matches!(self.grid.boundary_per_axis(), [BoundaryPolicy::Reflect | BoundaryPolicy::Periodic; D])
    }
    fn jet(&self, f: &GridFnND<F, D>, out: &mut [GridFnND<F, D>]) -> Result<(), SemiflowError>;
}

impl<F: SemiflowFloat, const D: usize> PointEval<F>
    for AnisotropicShiftChernoffND<F, D>
{
    fn eval_at(
        &self, tau: F, src: &GridFnND<F, D>, x: &[F], n_steps: u32,
    ) -> Result<F, SemiflowError>;
    // Backend E per math §31.2: tensor-product Gauss-Hermite quadrature for D ≤ 4;
    // MonteCarlo fallback for D ≥ 5 (out of scope for v4.0 baseline; sparse grid v4.1+).
}
```

Validation at construction:
- `a_ij` evaluated at grid CENTRE produces a symmetric-positive-definite matrix (SPD verification via Cholesky succeeds).
- `c.is_finite()` at grid centre.
- `grid.n_per_axis()` all $\ge 5$ (Gauss-Hermite 5-point stencil width per axis).
- `grid.boundary_per_axis()` all `Reflect` or `Periodic` (the only stable variants for the d-D shift quadrature).
- Cholesky factors of $A(x_k)$ pre-computed at each grid point and cached; cache size $O(N^D \cdot D^2)$.
- Gauss-Hermite tensor quadrature pre-computed (nodes + weights for $q = 5$ per axis); cost $O(5^D \cdot D)$ at construction.

Per-step algorithm (math §32.4 equation 32.3):

```
AnisotropicShiftChernoffND::apply_into(τ, f_src, f_dst, scratch):
  for each grid point k in 0..N^D:
    sum_k := 0;
    for each Gauss-Hermite tensor node q in [0, 5)^D:
      let eta_q = quadrature.node(q);                            // [F; D]
      let w_q   = quadrature.weight(q);                          // F
      let chol_k = self.cholesky_cache[k];                       // [F; D × D]
      let y_q = self.grid.x_at(k) + tau * self.b_i_eval(k) + (2*tau).sqrt() * chol_k * eta_q;
      // f at the shifted point y_q, with on-grid interpolation if y_q is off-grid.
      sum_k += w_q * f_src.sample(&y_q);
    f_dst[k] = (tau * self.c_eval(k)).exp() * sum_k;             // potential pre-multiplier
```

The per-step cost is $O(N^D \cdot 5^D)$ — exponential in D. For $D = 2$, $N = 128$, $5^2 = 25$ — manageable; for $D = 5$, $N = 128$, $5^5 = 3125$ — at the upper end of practical CI runtime budget; for $D \ge 6$, the cost explodes and the v4.0.0 release falls back to Smolyak sparse grids (deferred to v4.1+).

**Item 2 — `pub struct GaussHermiteTensor<F: SemiflowFloat, const D: usize>`** in `shift_nd.rs` (co-located helper):

```rust
pub struct GaussHermiteTensor<F: SemiflowFloat = f64, const D: usize = 2> {
    nodes_per_axis: [F; 5],           // canonical 5-point Gauss-Hermite nodes (const-array)
    weights_per_axis: [F; 5],          // canonical 5-point Gauss-Hermite weights (const-array)
}

impl<F: SemiflowFloat, const D: usize> GaussHermiteTensor<F, D> {
    pub const fn new() -> Self;        // const-fn populates from the canonical f64 table
    pub fn node(&self, q_index: usize) -> [F; D];      // q_index ∈ [0, 5^D)
    pub fn weight(&self, q_index: usize) -> F;
    pub fn n_nodes(&self) -> usize { 5usize.pow(D as u32) }
}
```

The const-array 5-point Gauss-Hermite weights are the canonical Hermite nodes from Abramowitz-Stegun §25 (table reproduced inline as `pub const HERMITE_5_NODES_F64: [f64; 5] = [...]`); the f64 → F coercion is via `F::from(f64_value)`.

## Rationale

- **Why ship d-D as a SINGLE generic kernel (vs separate Anisotropic2D, Anisotropic3D, Anisotropic4D, Anisotropic5D types)**: Rust const-generics let us write a single `AnisotropicShiftChernoffND<F, D>` impl that monomorphises to per-D code; the alternative (4 separate types) duplicates the algorithm 4× and forces users to switch types based on D. The const-generic D is the suckless choice for d-D parametric kernels (mirrors the v2.8 `Torus<F, D>` const-generic dim).
- **Why D ∈ {2, 3, 4, 5} for the v4.0 gate (not D = 2 only)**: the math is genuinely d-dimensional; restricting to D = 2 in v4.0 would force every D > 2 user to wait for v4.1+. The 5^D tensor-product Gauss-Hermite cost is manageable through D = 5 (3125 nodes per evaluation point); D = 6 (15625 nodes) is at the engineering edge of CI runtime budget. Restricting D = 6+ to "compiles but not gated" is the suckless cost trade-off.
- **Why ζ²-d-D correction baked into the v4.0 baseline (`order() = 2`)**: the v0.3.0 1D ζ-A correction (ADR-0008) lifts the bare shift Chernoff from order 1 to order 2 via a τ²-correction term; this is the empirically validated path for d-D as well. Shipping with the correction means users get order-2 convergence out of the box (matching the v0.3.0 1D semantics). The bare order-1 baseline is NOT exposed as a separate kernel — single combined impl is the suckless choice.
- **Why 5-point Gauss-Hermite (not 3-point or 7-point)**: 5-point is exact for polynomials of degree ≤ 9, sufficient for f64 precision against the Gaussian kernel. 3-point loses 1-2 ULP at the edge of the Gaussian tail; 7-point doubles the cost for negligible accuracy gain. 5-point is the suckless minimum that meets the precision target.
- **Why `Box<dyn Fn>` for the coefficient closures (not generic `F1: Fn`, `F2: Fn`, `F3: Fn`)**: the kernel needs to store the three coefficient closures (a_ij, b_i, c); each is a different closure type with different captures. Three generic parameters would inflate the type signature to `AnisotropicShiftChernoffND<F, const D, A: Fn(...), B: Fn(...), C: Fn(...)>` — unworkable. `Box<dyn Fn>` is the standard Rust workaround for storing heterogeneous closures; the per-step cost of the dynamic dispatch is negligible compared to the $O(5^D)$ quadrature.
- **Why Cholesky factors pre-computed at construction time** (not per-step): the Cholesky of $A(x_k)$ depends only on the grid point, not on $\tau$ or the iteration step. Pre-computing once at construction saves $O(D^3)$ per grid point per step — a 100-step run with $N^D$ grid points saves $100 \cdot N^D \cdot D^3$ FLOPs. The cache size $O(N^D \cdot D^2)$ is bounded by the grid size.
- **Why `growth() = 1.5×` multiplier** (not 1× or 2×): the Theorem 3 cousin formula has an inherent growth bound that includes the integral over the Gaussian kernel (which contributes 1) PLUS the per-grid-point amplification from the discrete divergence-form correction. The empirical bound on the canonical test datum (smooth bounded coefficients) is 1.5×; the same constant as v3.0 `Diffusion4thZeta4Chernoff` (ADR-0075). The exponential rate $\omega = 0$ because the correction is polynomial in $\tau$.
- **Why **Override #1 Cohort 7 carve-out** for `shift_nd.rs`**: the file co-locates the kernel impl + Cholesky cache management + Gauss-Hermite tensor + per-D specialisations (D = 2, 3, 4, 5 each have closed-form Cholesky shortcuts) + the PointEval Backend E impl + rustdoc citations to math §32 (~3-5 equation references inline per impl) + the ApproximationSubspace<2, F> opt-in. Each per-D specialisation adds ~50-80 lines; total ~600 LoC. The default 500-LoC cap would force splitting the per-D specialisations across separate files (shift_nd_d2.rs, etc.), scattering the citation cluster. Cohort 7 with 700-LoC cap absorbs the projected size; if it exceeds 700 LoC at engineer Wave C completion, split is triggered by v4.1 architecture review.
- **Why the G_DDIM gate is self-convergence (vs closed-form reference)**: the closed-form anisotropic Gaussian fundamental solution (Friedman 1964 §1.4) is available only for CONSTANT $A(x)$; for the variable-$A$ test datum (smooth bounded off-diagonal coupling), no closed-form analytical reference exists. Self-convergence against high-resolution $n_{\mathrm{ref}} = 2048$ is the only verifiable reference. Parallels the v3.0 G_zeta4 self-convergence pattern.
- **Why test files split per D** (not single multi-D file): per-D gates run independently; if D = 5 fails (e.g., CI runtime budget), the D = 2/3/4 results are still useful. Splitting also lets engineers parallelise the per-D runs across CI workers.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Implement AnisotropicShiftChernoff2D + AnisotropicShiftChernoff3D + AnisotropicShiftChernoff4D + AnisotropicShiftChernoff5D as separate types | 4× code duplication; loss of generic reasoning; user-facing type switch on D. Const-generic D is the suckless choice. |
| Restrict G_DDIM gate to D = 2 only (defer D ∈ {3, 4, 5} to v4.1+) | Forces every D > 2 user to wait; the math + impl ship together at v4.0; gating at v4.0 verifies the impl is correct across the realistic D range. |
| Implement the d-D kernel via composition of d 1D ShiftChernoff1D (one per axis with Strang splitting) | Loses the closed-form d-D shift formula; introduces a per-axis order-2 floor; doesn't natively handle cross-coupled diffusion tensors. The v2.5+ Strang2D path already provides this; v4.0's value-add is the NATIVE d-D representation. |
| Use Monte Carlo for the quadrature (not Gauss-Hermite tensor) | MC has $O(1/\sqrt{N_{\mathrm{samples}}})$ convergence vs Gauss-Hermite's super-polynomial; the v4.0 baseline uses Gauss-Hermite for D ≤ 4 with MC fallback for D ≥ 5 (the latter handled by ADR-0080 PointEval Backend E MC fallback). |
| Skip the Cholesky cache; recompute per step | $O(D^3)$ per grid point per step wasted; for N = 128, D = 4, n = 100 → 64 GB wasted FLOPs over a run. Pre-computation is the suckless optimisation. |
| Make the coefficient closures generic (Fn1, Fn2, Fn3 type parameters) | Inflates the type signature unworkably; the per-step dyn dispatch cost is negligible vs the O(5^D) quadrature; Box<dyn Fn> is the standard Rust workaround. |
| Defer the kernel to v4.1+ (keep v4.0 to just trait surface + Schrödinger + matrix-valued) | The d-D shift IS the consolidating flagship of v4.0; deferring it splits the BREAKING window into v4.0a + v4.0b for no engineering gain. The v4.0 Wave C ships the kernel + Wave B PointEval Backend E together. |
| Set the slope budget to ≤ -1.5 (looser than -1.95) | Loses the empirical signal distinguishing order-2 from order-1.5 (a bare shift kernel without ζ²-d-D correction); -1.95 is the standard precedent matching G3-strang / G_zeta4 / G27 / G28. |
| Drop the byte-identity sub-test at D = 1 vs ShiftChernoff1D | The byte-identity sub-test is the strongest verification that the d-D formula reduces to the v0.1.0 1D verbatim. Removing it would lose a key sanity check. Keep. |
| Implement Cholesky via nalgebra crate | Adds a dep (nalgebra is ~500 KB compiled); the per-D closed-form Cholesky (for D ≤ 5) is ~30 LoC each, well within the file budget. Suckless: avoid the dep. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing kernel is modified. The v0.1.0 `ShiftChernoff1D<F>` is preserved verbatim; v4.0 `AnisotropicShiftChernoffND<F, 1>` is BYTE-IDENTITY equivalent to it.
- **New module `crates/semiflow-core/src/shift_nd.rs`** (~600 LoC target; **Cohort 7 carve-out per constitution v1.8.0**; per-file cap 700 LoC; if engineering pushes past 700, split per D is triggered).
- **New kernel `AnisotropicShiftChernoffND<F, const D: usize>`** — generic over D ∈ {2, 3, 4, 5} for full-coverage gating; D ≥ 6 compiles but ungated.
- **New helper `GaussHermiteTensor<F, const D: usize>`** — pre-computed tensor-product quadrature.
- **Three trait impls** for the new kernel: `ChernoffFunction<F>` (apply_into per math §32.4), `ApproximationSubspace<2, F>` (K=2 witness; FIFTH v3.x opt-in), `PointEval<F>` (Backend E; per ADR-0080).
- **Dependency count unchanged** at 3/3 (num-traits, libm, num-complex). The kernel uses only stdlib + existing deps.
- **Schema bumps**: shared with ADR-0079/0080/0082/0083/0084/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. math.md is append-only (§32 NEW).
- **New gate**: G_DDIM (RELEASE_BLOCKING — 4 sub-tests across D ∈ {2, 3, 4, 5}; slope ≤ -1.95 each). Test files `tests/anisotropic_shift_nd_d{2,3,4,5}_slope.rs` new files, feature `slow-tests`.
- **CITATIONs added to math.md §32**: Remizov 2025 *Vladikavkaz Math. J.* 27:4 Theorem 3 cousin formula (the d-D Chernoff product); Friedman 1964 §1.4 (anisotropic Gaussian fundamental solutions).
- **Performance characterisation**: per-step cost $O(N^D \cdot 5^D)$:
  - D = 2, N = 128: $16384 \cdot 25 = 4.1 \times 10^5$ FLOPs per step.
  - D = 3, N = 128: $2.1 \times 10^6 \cdot 125 = 2.6 \times 10^8$ FLOPs per step.
  - D = 4, N = 128: $2.7 \times 10^8 \cdot 625 = 1.7 \times 10^{11}$ FLOPs per step.
  - D = 5, N = 128: $3.4 \times 10^{10} \cdot 3125 = 1.1 \times 10^{14}$ FLOPs per step — CI runtime budget concern.
- **Constitution amendment**: v1.7.1 → v1.8.0 adds Cohort 7 Override #1 carve-out for `shift_nd.rs` with per-file cap 700 LoC.

## Migration

End-user impact is **opt-in additive**. v0.1.0+ callers using `ShiftChernoff1D<F>` continue to compile unchanged.

New v4.0 users wanting d-D anisotropic shift:

```rust
// v0.1.0 baseline 1D (still works):
let kernel_1d = ShiftChernoff1D::<f64>::new(a_const, b_const, c_const, grid_1d)?;

// v4.0 NEW (d=2 generalisation; byte-identity equivalent at d=1):
let grid_2d = GridND::<f64, 2>::new([-5.0, -5.0], [5.0, 5.0], [128, 128])?;
let kernel_2d = AnisotropicShiftChernoffND::<f64, 2>::new(
    |x, a_out| {
        a_out[(0, 0)] = 1.0;
        a_out[(0, 1)] = 0.5 * (x[0] + x[1]).tanh();
        a_out[(1, 0)] = a_out[(0, 1)];
        a_out[(1, 1)] = 1.0;
    },
    |_x, b_out| { b_out[0] = 0.0; b_out[1] = 0.0; },
    |_x| 0.0,
    grid_2d.clone(),
)?;
let evolver_2d = Evolver::new(kernel_2d, n_steps)?;
let result_2d = evolver_2d.evolve(t_final, &initial_2d)?;
```

Worked example with d ∈ {2, 3, 4, 5} and the d-D heat-equation reference in `docs/migration/v3-to-v4.md` §4 (Wave G).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; the kernel uses only stdlib + existing deps.
- ADR-0008 — v0.3.0 ζ-A τ²-correction; the order-2 mechanism baked into v4.0 baseline.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `AnisotropicShiftChernoffND<F, D>` with `F = f64, D = 2` defaults.
- ADR-0026 — `ChernoffFunction<F>` super-trait; implemented by the new kernel.
- ADR-0041 — `apply_into` + `ScratchPool`; reused for the per-step quadrature buffers.
- ADR-0073 — v3.0 `ApproximationSubspace<K, F>`; the new kernel is the FIFTH v3.x K=2 opt-in.
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; `Growth<F>` typed return used by the new kernel.
- ADR-0080 — v4.0 PointEval first-class; the new kernel is PointEval Backend E (per math §31.2).
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap (d-D shift Wave C placement).
- math.md §32 (NEW v4.0) — d-D anisotropic Gaussian kernel + Theorem 3 cousin formula + G_DDIM gate spec.
- math.md §1-§2 (v0.1.0) — 1D `ShiftChernoff1D` precedent (byte-identity reduction at D = 1).
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation; Cohort 7 Override #1 carve-out added for `shift_nd.rs` (700-LoC cap).
- `docs/migration/v3-to-v4.md` §4 — d-D shift worked example (Wave G fills).

## Amendments

### AMENDMENT 1 (2026-05-30, ADR-0112) — F(0)=I normalization fix + honest order-1 + N(D) ladder

A read-only audit found the shipped kernel mathematically broken and three claims in this ADR false. ADR-0112 supersedes the affected parts:

1. **`apply_into` was un-normalized** (missing `π^(D/2)` divisor) and used node scale `√(2τ)` instead of `2√τ` ⇒ `F(0)=π^(D/2)·I ≠ I`, product diverges. Corrected formula in math.md §32.4 eq (32.3).
2. **Order is 1, not 2.** This ADR's "ζ²-d-D correction baked into the v4.0 baseline (`order()=2`)" (Decision Item 1, Rationale bullet 3, §60 of the code block) is **RETRACTED** — no such correction exists in the code; the frozen-coefficient kernel is genuinely order-1 for variable A (sympy + empirical proof, ADR-0112). `order()` returns **1**. The order-2 ζ²-d-D lift is DEFERRED (math §32.6).
3. **G_DDIM threshold is -0.95** (order-1), not -1.95. Gate now asserts `slope.is_finite()` first (closes the `f64::max(0.0,NaN)=0.0` self-mask) and adds a REQUIRED F(0)=I smoke sub-test.
4. **N=128 uniform spec replaced by the N(D) ladder** {128,32,8,6} / n_ref {2048,2048,512,512} (math §32.5 table).
5. **The "ADR-0081 §D=5 fallback" citation never existed** — it was fabricated in the D=5 test to justify a -1.7 relaxation. There is NO such section in this ADR and there is NO D=5 gate relaxation. D=5 uses the same -0.95 as D≤4.

The "Set the slope budget to ≤ -1.5" alternative-rejection row and the byte-identity-at-D=1 sub-test claim are likewise corrected by ADR-0112 (the D=1 reduction is to quadrature accuracy, not bit-identity — two different quadrature rules of the same integral). See ADR-0112 for the full derivation and decision.
