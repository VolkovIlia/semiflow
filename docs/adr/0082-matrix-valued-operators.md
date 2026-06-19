# ADR-0082 — Matrix-Valued Coupled-Component Operators (B5)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave D (fourth Wave of the second BREAKING window; ships after Wave A trait freeze + Wave B PointEval + Wave C d-D shift). Independent of Wave A/B/C; sibling to ADR-0081 (d-D shift) in the v4.0 BREAKING window. The matrix-valued kernel is the SIXTH v3.x ApproximationSubspace<2, F> opt-in (after the v3.0 trio + v3.1 HypoellipticChernoff + v4.0 AnisotropicShiftChernoffND).
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0008 (v0.3.0 ζ-A baseline — scalar diffusion that the matrix-valued kernel generalises at M = 1), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` super-trait), ADR-0041 (`apply_into` + `ScratchPool`), ADR-0043 (v2.0 `State<F>` 3-layer trait hierarchy — `MatrixGridFn1D<F, M>` is a new State<F> impl per the hierarchy), ADR-0073 (v3.0 `ApproximationSubspace<K, F>` — matrix kernel opts into K=2 witness; SIXTH v3.x opt-in), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` typed return).
- **Supersedes / amends**: none — strictly additive on the public surface. Establishes a NEW kernel `MatrixDiffusionChernoff<F: SemiflowFloat, const M: usize>` for coupled-component diffusion + a NEW state type `MatrixGridFn1D<F: SemiflowFloat, const M: usize>`. v0.3.0 scalar `DiffusionChernoff<F>` is the M = 1 SPECIALISATION (byte-identity test); v0.3.0 kernel is PRESERVED verbatim.
- **Mathematical foundation**: math.md §33 (NORMATIVE library — coupled matrix-valued state + matrix exponential per step; CITATION Pazy 1983 §3.3 — matrix semigroups and their generators; Higham 2008 *Functions of Matrices* §10 — matrix exponential via Padé approximation; Hilhorst-Mimura 1992 *Manuscripta Math.* — reaction-diffusion application class). math §33 BUILDS on §9.2.3.B (v0.3.0 scalar diffusion baseline).
- **Acceptance gates added**: G_MATRIX (RELEASE_BLOCKING — coupled-diffusion slope ≤ -1.95 on n ∈ {16, 32, 64, 128} sweep at T = 0.5 for M ∈ {2, 3, 4}; PLUS M = 1 byte-identity vs scalar DiffusionChernoff). 4 sub-tests; all RELEASE_BLOCKING.

## Context

The v0.1.0 → v3.1 library covers SCALAR-valued state functions: $f : \mathbb{R}^D \to \mathbb{R}$ (or $\mathbb{C}$ at v4.0 ADR-0079). For COUPLED-COMPONENT PDE systems where the state is $u : \mathbb{R}^D \to \mathbb{R}^M$ with each of $M$ components coupled to the others via the operator $L$, the v2.x library has no native support. Users must either (a) hand-encode the coupling via composition of $M$ scalar kernels with off-trait coupling steps (verbose, error-prone), or (b) treat the multi-component state as a single block-vector at the cost of losing the per-component grid structure (loses domain knowledge).

The canonical applications are:
- **Reaction-diffusion systems** with M coupled species (Hilhorst-Mimura 1992; the FitzHugh-Nagumo neurone model has M = 2).
- **Multi-population kinetic models** with M population types coupled through cross-collision terms.
- **Multi-asset financial PDEs** with M underlying asset types coupled through cross-correlation.
- **Quantum spin systems** with M spin components coupled through Pauli matrices (M = 2 for spin-1/2).

v4.0 ADR-0082 ships `MatrixDiffusionChernoff<F, const M: usize>` for the linear coupled-diffusion generator $(L u)_i = \sum_j a_{ij}(x) \partial_x^2 u_j + \sum_j b_{ij}(x) \partial_x u_j + \sum_j c_{ij}(x) u_j$ on $\mathbb{R}$ (1D base) with $u \in \mathbb{R}^M$.

The kernel is the first matrix-valued / vector-valued representation in the library. The state type `MatrixGridFn1D<F, M>` is the first multi-component state type (paralleling v3.1 `QuantumGraphSignal<F>` which is multi-edge, but per-edge scalar; the matrix state is per-grid-point M-vector).

## Decision

Ship two additive public-surface items in v4.0 Wave D:

**Item 1 — `pub struct MatrixGridFn1D<F: SemiflowFloat, const M: usize>`** in `crates/semiflow-core/src/matrix_system.rs` (NEW module, ~450 LoC target, default 500-LoC cap):

```rust
pub struct MatrixGridFn1D<F: SemiflowFloat = f64, const M: usize = 2> {
    grid: Grid1D<F>,
    values: Vec<F>,                              // row-major [F; N * M], component i at grid k stored at k*M + i
}

impl<F: SemiflowFloat, const M: usize> MatrixGridFn1D<F, M> {
    pub fn from_fn(grid: &Grid1D<F>, f: impl Fn(F) -> [F; M]) -> Self;
    pub fn from_components(grid: Grid1D<F>, components: [Vec<F>; M]) -> Result<Self, SemiflowError>;
    pub fn component_view(&self, i: usize) -> &[F];        // strided slice of length N
    pub fn component_view_mut(&mut self, i: usize) -> &mut [F];
    pub fn point_view(&self, k: usize) -> [F; M];          // M-vector at grid point k
    pub fn set_point(&mut self, k: usize, val: &[F; M]);
}

impl<F: SemiflowFloat, const M: usize> State<F> for MatrixGridFn1D<F, M> {
    // axpy: per-component axpy on each of the M strided slices.
    // scale: per-component scaling.
    // norm_sup: max over components of per-component sup-norm.
    // zeroed_like: zero MatrixGridFn1D of the same grid + same M.
}
```

**Item 2 — `pub struct MatrixDiffusionChernoff<F: SemiflowFloat, const M: usize>`** in `crates/semiflow-core/src/matrix_system.rs` (co-located with the state type):

```rust
pub struct MatrixDiffusionChernoff<F: SemiflowFloat = f64, const M: usize = 2> {
    a_ij_field: Box<dyn Fn(F, &mut SquareMatrix<F, M>) + Send + Sync>,
    b_ij_field: Box<dyn Fn(F, &mut SquareMatrix<F, M>) + Send + Sync>,
    c_ij_field: Box<dyn Fn(F, &mut SquareMatrix<F, M>) + Send + Sync>,
    grid: Grid1D<F>,
    c_norm_bound: Option<F>,                     // ‖C(x)‖_∞ bound for growth(); None = caller-asserted
    scratch_a_kx: Vec<SquareMatrix<F, M>>,       // pre-allocated a_ij(x_k) cache (filled per step)
    scratch_c_kx: Vec<SquareMatrix<F, M>>,       // pre-allocated c_ij(x_k) cache
    scratch_exp_c: Vec<SquareMatrix<F, M>>,      // pre-allocated exp(τ C(x_k)) cache
}

impl<F: SemiflowFloat, const M: usize> MatrixDiffusionChernoff<F, M> {
    pub fn new(
        a_ij_field: impl Fn(F, &mut SquareMatrix<F, M>) + Send + Sync + 'static,
        b_ij_field: impl Fn(F, &mut SquareMatrix<F, M>) + Send + Sync + 'static,
        c_ij_field: impl Fn(F, &mut SquareMatrix<F, M>) + Send + Sync + 'static,
        grid: Grid1D<F>,
    ) -> Result<Self, SemiflowError>;

    pub fn with_c_norm_bound(self, c_norm_bound: F) -> Self;
}

impl<F: SemiflowFloat, const M: usize> ChernoffFunction<F>
    for MatrixDiffusionChernoff<F, M>
{
    type S = MatrixGridFn1D<F, M>;
    fn apply_into(
        &self, tau: F, src: &MatrixGridFn1D<F, M>, dst: &mut MatrixGridFn1D<F, M>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
    fn order(&self) -> u32 { 2 }
    fn growth(&self) -> Growth<F> {
        Growth {
            multiplier: F::one(),
            omega: self.c_norm_bound.unwrap_or(F::infinity()),
        }
    }
}

impl<F: SemiflowFloat, const M: usize> ApproximationSubspace<2, F>
    for MatrixDiffusionChernoff<F, M>
{
    fn in_subspace(&self, f: &MatrixGridFn1D<F, M>) -> bool {
        f.grid.n() >= 5 && self.c_norm_bound.is_some()
    }
    fn jet(&self, f: &MatrixGridFn1D<F, M>, out: &mut [MatrixGridFn1D<F, M>]) -> Result<(), SemiflowError>;
}
```

Validation at construction:
- All three coefficient closures evaluated at `grid.x_at(grid.n() / 2)` produce finite SquareMatrix entries.
- `a_ij_field` at grid centre is symmetric-positive-definite (Cholesky succeeds).
- `c_ij_field` at grid centre is symmetric (if not, returns DomainViolation).
- `grid.n() >= 5` (5-point stencil width).

Per-step algorithm (math §33.3):

```
MatrixDiffusionChernoff::apply_into(τ, u_src, u_dst, scratch):
  // Phase 1 — per-component coupled discrete diffusion.
  // For each (i, j) pair, apply the (i, j) component of the coupled diffusion to u_src[j].
  // Accumulate into intermediate[i] = Σ_j (a_ij·∂² + b_ij·∂)(u_src[j])
  let mut intermediate = scratch.borrow_matrix_state(u_src);
  for i in 0..M {
    for j in 0..M {
      // Apply scalar diffusion with coefficients a_ij(x) and b_ij(x) to u_src.component_view(j).
      scalar_diffusion_apply(τ, &self.a_ij_at, &self.b_ij_at,
                              i, j, u_src.component_view(j),
                              &mut intermediate.component_view_mut(i));
    }
  }

  // Phase 2 — pointwise matrix exponential of reaction matrix.
  for k in 0..N {
    let mut c_k = self.scratch_c_kx[k];                  // reaction matrix at grid k
    (self.c_ij_field)(self.grid.x_at(k), &mut c_k);
    let exp_c_k = matrix_exp(τ * c_k);                    // Cayley-Hamilton for M ≤ 4; Padé for M ≥ 5
    let intermediate_k = intermediate.point_view(k);
    let new_u_k = matrix_vec_mul(&exp_c_k, &intermediate_k);
    u_dst.set_point(k, &new_u_k);
  }
```

The matrix exponential `matrix_exp(τ * C_k)` uses:
- **M ∈ {1, 2}**: closed-form via Cayley-Hamilton + scalar exp.
- **M ∈ {3, 4}**: Cayley-Hamilton with Newton's identities for the characteristic polynomial.
- **M ≥ 5**: Padé approximation per Higham 2008 §10 — order [13/13] with scaling-and-squaring. Cost $O(M^3 \log_2 \|\tau C\|)$ per grid point.

## Rationale

- **Why a NEW kernel `MatrixDiffusionChernoff<F, M>` (not a generalisation of v0.3.0 DiffusionChernoff via const M)**: v0.3.0 DiffusionChernoff has `type S = GridFn1D<F>` — a scalar state. Adding a const-generic M to it would BREAK the v0.3.0 type signature (M = 1 would still be different from the scalar case because the state type changes from `GridFn1D<F>` to `MatrixGridFn1D<F, 1>`). Sibling kernel design preserves v0.3.0 compatibility AND lets the M = 1 byte-identity sub-test verify the reduction.
- **Why a NEW state type `MatrixGridFn1D<F, M>` (not reuse [Vec<F>; M] or similar)**: a tuple of M GridFn1D loses the State<F> impl (would need M separate axpy / scale / norm_sup calls); a flat `Vec<F>` of length N*M loses the M-vector point-view ergonomics. The dedicated newtype with row-major layout is the suckless choice — single State<F> impl, fast point views, clear semantics.
- **Why M ∈ {2, 3, 4} gated (not M = 2 only)**: the math is genuinely matrix-valued; restricting to M = 2 would force every M > 2 user to wait. The Cayley-Hamilton closed-form covers M ≤ 4 efficiently (M = 4 has ~30 LoC; M = 5+ requires Padé scaling-and-squaring). Gating through M = 4 is the suckless coverage.
- **Why M ≥ 5 NOT gated (compiles but uses Padé)**: Padé matrix exponential is correct but slower; G_MATRIX gate runtime budget would explode at M = 5+. The Padé code path is validated by a separate unit test on a known 4×4 matrix exponential (verifying the Padé converges to the Cayley-Hamilton result at M = 4) — high confidence carries to M ≥ 5 without per-M gating.
- **Why `c_norm_bound: Option<F>`** (not required `F`): the bound is caller-asserted (the library cannot verify $\|C(x)\|_\infty \le c$ without sampling). `Option<F>` lets the caller opt into the unchecked path (`None`) for performance hot loops where the bound has been verified off-line. Mirrors the v3.0 `Diffusion4thZeta4Chernoff::a_kth_bound` pattern (ADR-0075).
- **Why M = 1 byte-identity sub-test** (vs slope sub-test for M = 1): the M = 1 reduction is mechanical (the matrix becomes scalar; the matrix exponential becomes scalar exp; the matrix-vector mul becomes scalar mul). Byte-identity at f64 is the strongest possible verification that the reduction is correct. Slope at M = 1 would be redundant with v0.3.0 G3-strang.
- **Why no Override #1 expansion for matrix_system.rs (~450 LoC)**: target ~450 LoC, well under the default 500-LoC cap. The state type (~120 LoC) + the kernel impl (~250 LoC) + Cayley-Hamilton helpers (~80 LoC) fit comfortably. If engineering pushes past 500 LoC (e.g., Padé impl is more involved), a v1.8.x PATCH would add it; not budgeting for it pre-emptively.
- **Why deterministic PCG64 seed (0xC0FFEE_BABE_DEAD_BEEF) for the SPD coupling matrices in G_MATRIX**: reproducibility — the gate must produce identical test data across CI runs and across reviewer-suckless audits. The seed is the canonical project-wide deterministic-seed constant (matches v2.5.1 HFT-latency-tail baseline; mirrors L_CEV_PTICK.canonical_input.seed).
- **Why pre-allocate the Cayley-Hamilton scratch matrices** (vs allocate per step): the per-step alloc cost would dominate at M = 4 with N = 128 (16384 matrix allocations per step × 100 steps = ~10^7 allocations — heap thrashing). Pre-allocation at construction time keeps per-step alloc count at 0.
- **Why `apply_into` with single-buffer scratch** (vs per-component buffers): the State<F> impl gives us per-component view access; the scratch buffer for the intermediate state is itself a MatrixGridFn1D, allocated once per kernel invocation. Mirrors the v2.0 `apply_into` zero-alloc pattern.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Add const-generic M to v0.3.0 `DiffusionChernoff<F, const M>` (in-place generalisation) | BREAKING signature change for all v0.3+ callers. Sibling kernel preserves v0.3.0 compatibility. |
| Use a fixed-size array `[GridFn1D<F>; M]` as the state type | Tuple-style; loses State<F> impl unification; M separate axpy calls per step. The newtype is the suckless choice. |
| Implement matrix exponential via nalgebra (drop the Cayley-Hamilton + Padé code) | Adds nalgebra dep (~500 KB); the per-M closed-forms are ~30 LoC each (well within file budget). Suckless: avoid the dep. |
| Restrict G_MATRIX to M = 2 only (defer M ∈ {3, 4} to v4.1+) | Forces every M > 2 user to wait; Cayley-Hamilton is cheap to ship through M = 4. |
| Implement the kernel as a composition of M scalar DiffusionChernoff (no native matrix support) | Loses the coupled-component representation; user must hand-code the M-component coupling at every step; verbose; error-prone. The dedicated kernel is the suckless choice. |
| Use generic Fn types for the closures (`A: Fn(...), B: Fn(...), C: Fn(...)`) | Inflates the type signature unworkably (4 type parameters); Box<dyn Fn> is the standard Rust workaround for storing heterogeneous closures. |
| Implement the Cayley-Hamilton for M = 2, 3 only; force Padé from M ≥ 4 | Cayley-Hamilton through M = 4 is ~30 LoC additional; the perf win (Cayley-Hamilton is ~5× faster than Padé at M = 4) justifies the LoC. |
| Skip pre-allocation of scratch matrices (allocate per step) | Per-step heap alloc kills the per-tick latency story; pre-allocation is the suckless optimisation. |
| Make c_norm_bound required (not optional) | Forces every caller (including hot loops with off-line-verified bounds) to pay the construction-time bound check. Optional opt-out matches v3.0 Diffusion4thZeta4Chernoff. |
| Defer the entire matrix-valued track to v4.1+ (ship v4.0 with just trait surface + Schrödinger + d-D shift) | The matrix-valued use cases (reaction-diffusion, FitzHugh-Nagumo, basket pricing) are mature; v4.0 BREAKING window is the right place to ship; deferring loses momentum. |
| Skip the M = 1 byte-identity sub-test | The byte-identity is the strongest verification that the reduction is correct; skipping it loses a key sanity check. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; v0.3.0 `DiffusionChernoff<F>` is preserved verbatim.
- **New module `crates/semiflow-core/src/matrix_system.rs`** (~450 LoC target; default 500-LoC cap; NO Override expansion).
- **New kernel `MatrixDiffusionChernoff<F, const M: usize>`** — gated for M ∈ {1, 2, 3, 4}; M ≥ 5 compiles via Padé but ungated.
- **New state type `MatrixGridFn1D<F, const M: usize>`** — first multi-component state type in the library. Implements `State<F>` per ADR-0043 hierarchy.
- **Two trait impls** for the new kernel: `ChernoffFunction<F>` (apply_into per math §33.3), `ApproximationSubspace<2, F>` (SIXTH v3.x opt-in).
- **Dependency count unchanged** at 3/3.
- **Schema bumps**: shared with ADR-0079/0080/0081/0083/0084/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. math.md is append-only (§33 NEW).
- **New gate**: G_MATRIX (RELEASE_BLOCKING — 4 sub-tests: M = 1 byte-identity + M ∈ {2, 3, 4} slope ≤ -1.95). Test files `tests/matrix_diffusion_m{1,2,3,4}_*.rs` new files, feature `slow-tests`.
- **CITATIONs added to math.md §33**: Pazy 1983 §3.3 (matrix semigroups); Higham 2008 *Functions of Matrices* §10 (matrix exponential Padé); Hilhorst-Mimura 1992 (reaction-diffusion application class).

## Migration

End-user impact is **opt-in additive**. v0.3.0+ scalar diffusion callers continue to compile unchanged.

New v4.0 users wanting coupled-component diffusion:

```rust
// v0.3.0 baseline scalar (still works):
let scalar_kernel = DiffusionChernoff::<f64>::new(a_fn_scalar, grid_1d.clone())?;

// v4.0 NEW (M = 2 coupled FitzHugh-Nagumo-like reaction-diffusion):
let matrix_kernel = MatrixDiffusionChernoff::<f64, 2>::new(
    |x, a_mat| { /* a_ij(x) symmetric-positive-definite */ a_mat[(0,0)] = 1.0; a_mat[(0,1)] = 0.0; a_mat[(1,0)] = 0.0; a_mat[(1,1)] = 0.1; },
    |_x, b_mat| { /* skew-symmetric drift, zero here */ for i in 0..2 { for j in 0..2 { b_mat[(i,j)] = 0.0; } } },
    |x, c_mat| { /* symmetric reaction; FitzHugh-Nagumo-like nonlinearity */ c_mat[(0,0)] = -0.5; c_mat[(0,1)] = 1.0; c_mat[(1,0)] = -0.5; c_mat[(1,1)] = -0.1; },
    grid_1d.clone(),
)?.with_c_norm_bound(1.0);
let evolver = Evolver::new(matrix_kernel, n_steps)?;
let u0 = MatrixGridFn1D::<f64, 2>::from_fn(&grid_1d, |x| {
    [ (-x*x).exp(),                // component 0: activator
      0.5 * (-x*x*2.0).exp() ]     // component 1: inhibitor
});
let result = evolver.evolve(t_final, &u0)?;
let activator_final  = result.component_view(0);
let inhibitor_final  = result.component_view(1);
```

Worked example with the FitzHugh-Nagumo travelling-wave reference in `docs/migration/v3-to-v4.md` §5 (Wave G).

## Cross-references

- ADR-0001 — contract-first.
- ADR-0003 — no_std + alloc.
- ADR-0008 — v0.3.0 ζ-A baseline scalar diffusion; M = 1 reduction reference.
- ADR-0025 — Generic-over-Float defaulting; `MatrixDiffusionChernoff<F, M>` with `F = f64, M = 2` defaults.
- ADR-0026 — `ChernoffFunction<F>` super-trait.
- ADR-0041 — `apply_into` + `ScratchPool`.
- ADR-0043 — v2.0 `State<F>` 3-layer trait hierarchy; `MatrixGridFn1D<F, M>` is a new State<F> impl.
- ADR-0073 — v3.0 `ApproximationSubspace<K, F>`; matrix kernel is SIXTH v3.x K=2 opt-in.
- ADR-0074 — v3.0 ChernoffFunction trait cleanup.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap (matrix-valued Wave D placement).
- math.md §33 (NEW v4.0) — coupled matrix-valued state + matrix exponential per step + G_MATRIX gate spec.
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation; matrix_system.rs under default 500-LoC cap (no Override expansion).
- `docs/migration/v3-to-v4.md` §5 — matrix-valued FitzHugh-Nagumo worked example (Wave G fills).

## Amendments

### AMENDMENT (2026-05-28) — Strang-symmetric order-2 restoration; G_MATRIX threshold reverted -0.80 → -1.95

- **Status**: Original v4.0 order-1 deferral REVOKED. Order-2 Strang-symmetric kernel restored as the canonical `MatrixDiffusionChernoff` algorithm; G_MATRIX RELEASE_BLOCKING threshold reverted to the original ADR-0082 §"Acceptance gates added" specification (slope ≤ -1.95).
- **Date**: 2026-05-28 (v4.x maintenance window; engineer Wave delegated separately).
- **Authors**: ai-solutions-architect.
- **Supersedes**: the "**Gate threshold AMENDED -1.95 → -0.80**" engineering note recorded in CHANGELOG [4.0.0] §Added "B5" + §Changed "properties.yaml" + §Tested "G_MATRIX" + §Notes "3 amendments accumulated". The order-1 explicit-Euler kernel shipped in v4.0 commit `82d1177` is replaced atomically in the v4.x engineer Wave.

**Trigger** — post-v4.0 researcher Campaign 3 (`.dev-docs/research/verdicts/verdict-matrix-strang.md`, synthesising `raw-findings-matrix-strang.md` over 9 queries / 37+ peer-reviewed sources) gathered 15 A-level sources confirming the standard Marchuk-Strang split `F(τ) = exp(τC/2) ∘ DiffusionStep(τ) ∘ exp(τC/2)` is order-2 globally for matrix-valued parabolic reaction-diffusion `∂_t u = D(x)∂_xx u + b(x)∂_x u + C(x)u`. The v4.0 Wave D CHANGELOG line — "research-level for M ≥ 2; not closed-form like scalar case" — misread "no single-source turnkey Python tutorial" as "research-level math". Primary citations: Auzinger-Herfort-Koch-Thalhammer 2016 (arXiv 1604.01190, *The BCH-formula and order conditions for splitting methods*) Theorem 4.1 for the BCH order-conditions; Cheng-Wang-Wise 2021 (arXiv 2108.11254, *Stability and convergence of Strang splitting. Part II: tensorial Allen-Cahn equations*); arXiv 2602.07437 (2025, *Convergence of a Low-Rank Strang Splitting for Stiff Matrix Differential Equations*); Hochbruck-Lubich 2010, *Acta Numerica* — *Exponential integrators*.

**New algorithm** (REPLACES the v4.0 explicit-Euler `apply_into` body verbatim; public API surface preserved):

```
MatrixDiffusionChernoff::apply_into(τ, u_src, u_dst, scratch):
  // Phase 1 — half-step reaction (per grid point, pointwise matrix exp).
  for k in 0..N:
    let c_k = c_ij_field(x_k);                                   // M × M symmetric
    let half_exp_k = matrix_exp(τ * c_k / 2.0);                  // reuse v4.0 matrix_exp_m{1,2,3,4} helpers
    let u1_k = matrix_vec_mul(&half_exp_k, &u_src.point_view(k));
    u1.set_point(k, &u1_k);

  // Phase 2 — full-step coupled diffusion (REUSES the v4.0 spatial step verbatim).
  for i in 0..M:
    for j in 0..M:
      scalar_diffusion_apply(τ, &a_ij_at, &b_ij_at, i, j,
                              u1.component_view(j),
                              &mut u2.component_view_mut(i));

  // Phase 3 — half-step reaction (same matrix_exp call; can cache from Phase 1 since c_k unchanged).
  for k in 0..N:
    let u_dst_k = matrix_vec_mul(&half_exp_k_cached[k], &u2.point_view(k));
    u_dst.set_point(k, &u_dst_k);
```

Order-2 global convergence on smooth `D, b, C, u` follows from the BCH expansion at depth ≥2 (Auzinger 2016 Theorem 4.1): the Strang-symmetric composition `exp(τR/2) ∘ exp(τL) ∘ exp(τR/2)` produces local error of order `τ³` with leading commutator term `(τ³/12)·(2[R,[R,L]] − [L,[L,R]])`; the parabolic semigroup smoothing absorbs the residual into a global `τ²` rate even for `[L, R] ≠ 0`. The BCH theory is dimension-independent — closure is NOT blocked by matrix dimension M (v4.0 engineer's "research-level for M ≥ 2" caveat is demonstrably false; cf. Q5.B in verdict-matrix-strang.md).

**G_MATRIX gate restoration**: properties.yaml `G_MATRIX.gate` threshold REVERTS `-0.80 → -1.95` (math.md §33.4 original spec). Per-M test files `tests/matrix_diffusion_m{2,3,4}_slope.rs` assertion constant reverts identically. M=1 byte-identity vs scalar `DiffusionChernoff` preserved verbatim. The pattern mirrors the v3.1 G29 amendment-reversal (mass-conservation threshold relaxed at test time, then re-tightened in v3.2 once the algorithm matched the architect spec) — except here the amendment-reversal happens within the same major version (v4.x maintenance) because the algorithm fix is mechanical, not research-level.

**Out of scope** (preserve simplicity per suckless guardrail #1):
- **Magnus K=2 alternative** (`F(τ) = exp(τL + τR + (τ²/2)[L,R])`): deferred indefinitely — Strang-symmetric is sufficient for the stated order-2 claim, and the explicit `[L,R]` commutator computation would inflate `matrix_system.rs` without benefit at the stated gate.
- **M ≥ 5 Padé / Krylov matrix-exp**: stays deferred per §33.5 original (the BCH theory does not restrict M; the closed-form Cayley-Hamilton `matrix_exp_m{1,2,3,4}` helpers do — v4.x maintenance does NOT widen the closed-form ladder).
- **2D / 3D matrix-valued kernels** (`MatrixDiffusionChernoff2D/3D`): stay deferred per §33.5 — composition with `Strang2D/3D` is the v4.x+ path.
- **FitzHugh-Nagumo cross-diffusion benchmark** (verdict-matrix-strang.md §"Threshold restoration"): documented as a recommended future unit test in `.dev-docs/specs/matrix-strang-wave.md` §"Optional follow-on" but NOT gated in this Wave (the recursive G_MATRIX slope sweep on PCG64-seeded SPD `D, b, C` already exercises the cross-diffusion code path on `[D, C] ≠ 0`).

### Engineer Wave (delegated to agentic-engineer)

See `.dev-docs/specs/matrix-strang-wave.md` for the full Wave specification with acceptance criteria, test plan, and file touch list. Summary: rewrite `MatrixDiffusionChernoff::apply_into` as palindromic Strang per the algorithm above; revert G_MATRIX threshold across 3 slope tests + properties.yaml + CHANGELOG; AMEND math.md §33 (this amendment's companion); ~30 LoC growth in `matrix_system.rs` (493 → ~523 — file split into `matrix_strang.rs` sibling if budget exceeded). Optional sympy sub-check T_MATRIX_STRANG verifies the BCH order-conditions for the leading `[L, R]` term at M=2. Estimated effort: 1-2 weeks per verdict §"Engineering runway".

### AMENDMENT 2 (2026-05-28) — Phase 2 must be order-≥2 sub-flow; block Crank-Nicolson chosen

- **Status**: ratifies AMENDMENT 1's three-phase Strang skeleton; CORRECTS Phase 2 from "reuse v4.0 spatial step verbatim" (explicit Euler, order 1) to **block Crank-Nicolson via block-Thomas solve** (Cayley map of the discrete diffusion operator, order 2, unconditionally stable). G_MATRIX RELEASE_BLOCKING threshold stays at $-1.95$ per AMENDMENT 1.
- **Date**: 2026-05-28 (same v4.x maintenance window; engineer Wave NOT YET delegated — this amendment SUPERSEDES the spec text of `.dev-docs/specs/matrix-strang-wave.md` AC1 Phase 2 before delegation).
- **Authors**: ai-solutions-architect.
- **Supersedes within ADR-0082**: AMENDMENT 1's Phase 2 line "REUSES the v4.0 spatial step verbatim" — replaced by the block-CN algorithm below.

**Trigger** — engineer pre-flight implementation (faithful to AMENDMENT 1 spec) measured G_MATRIX M=2 slope $-1.1208$ (FAIL gate $\le -1.95$; matches v4.0 order-1 baseline). Root cause: the v4.0 "spatial step" is `acc[i] = u1[i] + τ·Σ_j (a[i][j]·d²u_j + b[i][j]·d u_j)` — explicit forward Euler in time. Strang composition `R(τ/2) ∘ D(τ) ∘ R(τ/2)` is order-2 IFF $D(τ) = \exp(τ L_{\text{diff}})$ is exact (or at least order-≥2). The Auzinger-Herfort-Koch-Thalhammer 2016 Theorem 4.1 cited in AMENDMENT 1 and verdict-matrix-strang.md Q5.A explicitly assumes both sub-flows are exact or have local error $O(τ^3)$. With explicit Euler in Phase 2, the global Strang error decomposes as $O(τ^2)_{\text{BCH}} + O(τ \cdot \text{err}_D) = O(τ^2) + O(τ \cdot τ) = O(τ)$ — engineer's $-1.12$ measurement confirms.

**Chosen option — block Crank-Nicolson (Option A)**. For each grid point $k$ and component pair $(i, j)$, the discrete diffusion operator is the $NM \times NM$ block-tridiagonal matrix $L^h$ acting on the flattened state $u \in \mathbb{R}^{NM}$. Phase 2 becomes the Cayley map:

```
(I − τ/2 · L^h) · u^(2) = (I + τ/2 · L^h) · u^(1)
```

Solved via **block-Thomas algorithm** (block-tridiagonal LU with $M \times M$ block inversions — closed-form for $M \in \{1, 2, 3, 4\}$ via the existing `matrix_exp_m{1,2,3,4}` companion helpers; the M×M inverse is mechanical Cramer's rule plus determinant). Cost per Phase 2: $O(N \cdot M^3)$ (block-Thomas forward + back sweep). Stability: unconditionally A-stable (Cayley map is unitary on the imaginary axis, contractive on the left half-plane — covers all SPD `D(x)` and all skew-symmetric `b(x)` discretisations). Order: time-2 (Cayley map), space-2 (3-point stencil for $\partial^2$, 3-point centred for $\partial$).

**Why Option A over alternatives**:

| Option | Why rejected |
|---|---|
| **B — per-component Diffusion4thChernoff** | Requires `D(x)` diagonal (i.e. `a[i][j] = δ_ij · a_i(x)`); the G_MATRIX PCG64-seeded SPD test matrices are FULL (non-diagonal). Adding a runtime `is_diagonalizable()` check + per-eigenchannel decomposition adds an LAPACK-like dependency, conflicts with `no_std + alloc` budget. The cross-diffusion class (FitzHugh-Nagumo with $D_{12} \ne 0$, Hilhorst-Mimura 1992) is the canonical motivating use case — rejecting it would gut ADR-0082's stated scope. |
| **C — matrix Magnus K=2** | Requires symbolic commutators $[D \partial^2, C]$ which generate first-derivative cross-terms $D' \cdot C \cdot \partial$ needing additional FD stencils; ~250 LoC and high-risk for sign/coefficient errors. The order-2 win is identical to block-CN for ~70% more LoC. Block-CN reuses the established schrodinger.rs Cayley-map pattern. |
| **D — accept order-1, revert AMENDMENT 1** | Contradicts the 15 A-level sources cited in verdict-matrix-strang.md; would force `MatrixDiffusionChernoff::order()` from 2 to 1, invalidating the v4.0 `ApproximationSubspace<2, F>` opt-in (SIXTH v3.x K=2 witness); regenerates the v4.0 "research-level" debt the verdict explicitly retired. |
| **E — diagonalization assumption** | Same blocker as Option B: G_MATRIX inputs are random SPD, not diagonal. Practical user code may have diagonal `D` (Brusselator, classical Allen-Cahn) but the gate must cover the cross-diffusion case to be honest about the kernel's contract. |

**Why block-CN is well-precedented**: `crates/semiflow-core/src/schrodinger.rs` Phase 2 (the `K(τ)` kinetic step) uses precisely the Cayley map `(I − τ/2 [[0, L], [-L, 0]])^{-1} (I + τ/2 [[0, L], [-L, 0]])` solved by a 2-block Thomas variant. Block-CN for `MatrixDiffusionChernoff` is the same pattern with $M$-block Thomas instead of 2-block. The schrodinger.rs precedent is ~200 LoC; block-CN here is estimated at ~150-200 LoC.

**Revised G_MATRIX impact**: threshold stays at $-1.95$ (AMENDMENT 1 specification unchanged). The expected post-implementation slope is $-2.0 \pm 0.1$ (time order 2 + space order 2 at the gate's $n \in \{16, 32, 64, 128\}$ sweep). M=1 byte-identity test against scalar `DiffusionChernoff` is PRESERVED only if v0.3.0 `DiffusionChernoff` is ALSO upgraded to block-CN (it currently uses explicit-Euler-style ζ-A baseline per ADR-0008) — **OR** the M=1 byte-identity test is REPLACED with an M=1 slope sub-test at the same $-1.95$ threshold. The engineer Wave spec MUST resolve this; the architect's preference is the SLOPE replacement (preserves v0.3.0 verbatim; new test verifies M=1 reduction via convergence, not byte-equality).

**Implementation cost** — ~200 LoC new (block-Thomas solver `block_thomas_solve_m::<F, M>` in `matrix_strang.rs` sibling + Cayley-map phase 2 in `matrix_system.rs`), ~30 LoC modified (`apply_into` body). File `matrix_system.rs` 493 → ~520 (still under 500-LoC cap if the block-Thomas solver lives in the existing `matrix_strang.rs` sibling created by AMENDMENT 1's spec). Effort estimate: **1-2 weeks** (unchanged from AMENDMENT 1; block-Thomas is textbook Golub-Van Loan §4.5.1, not research).

**Out of scope** (preserves suckless guardrail #1):
- **Krylov / Padé matrix-exp for $M \ge 5$**: stays deferred per §33.5; block-CN is closed-form via the same M ≤ 4 ladder.
- **Adaptive time-stepping inside Phase 2**: out of scope — fixed-τ Cayley map matches the v4.0 contract.
- **GPU / SIMD block-Thomas**: out of scope — sequential Thomas sweep on $N \le 128$ for the gate is sub-millisecond in scalar Rust.

**Additional citations** (Amendment-2-specific): G. H. Golub, C. F. Van Loan, *Matrix Computations*, 4th ed., Johns Hopkins UP 2013 — §4.5.1 (block-tridiagonal Thomas / block-LU). J. D. Lambert, *Numerical Methods for Ordinary Differential Systems*, Wiley 1991 — §6.6 (Cayley map for stiff IVPs; A-stability proof). Hochbruck-Lubich 2010 *Acta Numerica* §3.4 — explicit confirmation that "the implicit midpoint rule [= Cayley map] is the only A-stable, time-symmetric, order-2 Runge-Kutta scheme; it is the canonical Phase 2 for parabolic Strang splittings". `crates/semiflow-core/src/schrodinger.rs` (intra-project precedent for Cayley-map Phase 2 + Thomas-style solve).

The companion math.md §33.7 AMENDMENT 2 paragraph (≤20 LoC) updates Theorem 33.2 to reflect block-CN as the canonical Phase 2 sub-flow. The companion `.dev-docs/specs/matrix-strang-wave.md` REVISES AC1 Phase 2 + AC3 M=1 sub-test handling per this amendment before delegation to agentic-engineer.
