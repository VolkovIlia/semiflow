# ADR-0186 — Generic externally-assembled symmetric-operator entry point (`SymmetricOperator`, `MassKOperator`, entry-Fréchet)

- **Status**: Proposed (design only — Issue #13; branch `issue-13-generic-symmetric-L`)
- **Date**: 2026-06-26
- **Supersedes**: none — purely ADDITIVE over ADR-0185 (A1 `graph_expmv` Krylov action,
  A2 `graph_expmv_frechet`), ADR-0180 (`Laplacian::from_csr_parts`), ADR-0115/§43
  (`GeneratorSensitivity`). No existing kernel, gate, public signature, or 0-ULP
  scope is changed. A1/A2 source is touched in exactly one mechanical way (lift one
  concrete bound to the trait it already satisfies — see D2).
- **Contract**: `contracts/semiflow-core.math.md` §55 (new NORMATIVE section);
  gate quad `G_SYMOP_DENSE`, `G_MASSK_LUMPED`, `G_MASSK_CONSISTENT`,
  `G_SYMOP_ENTRY_FRECHET`.

## Context

A1 (`GraphKrylovChernoff`, §54) applies `e^{−τL}·v` for the **combinatorial /
symmetric-normalized graph Laplacian** — a symmetric PSD operator with *zero row
sums*. But the action machinery never uses that structure: the Chebyshev and Lanczos
helpers (`graph_krylov_helpers.rs`) touch the operator **only** through
`Laplacian::apply_into_slice` (CSR SpMV) and `spectral_radius_bound` (Gershgorin).
A2 (`graph_expmv_frechet`) is **already generic** over the perturbation provider
`P: GeneratorSensitivity`; the edge-weight rank-1 stencil (§43.2) is just one impl.

So the issue's restriction is incidental, not essential. Issue #13 asks to drop the
zero-row-sum constraint and accept **any externally-assembled symmetric PSD sparse
operator** (CSR): FEM stiffness `K` with Robin/convective boundary terms (row sums
≠ 0), anisotropic / spatially-varying conductivity. It also asks for the
**generalized `(M, K)` path** — propagate `e^{−τ M⁻¹K}` for a consistent mass
matrix `M` — and to keep the differentiable adjoint `∂J/∂entries`.

The gap is threefold: (1) there is no clearly-named, **symmetry-validated** generic
CSR carrier (`from_csr_parts` validates dimensions/range only, and is named
`Laplacian`); (2) `M⁻¹K` is non-symmetric, so the symmetric Krylov path cannot be
applied to it directly; (3) the Fréchet provider has no single-entry stencil.

**Design tension (resolved, not compromised).** We want the consistent-`(M,K)` path
to reuse the *sparse* Krylov action AND leave A1/A2 essentially untouched AND add no
heavy linear-algebra (no LAPACK, suckless). АРИЗ chain: АП = "M⁻¹K is non-symmetric
so symmetric Krylov can't run on it." ТП = make the operator symmetric (lose
sparsity if formed densely) **or** keep it sparse (lose symmetry). ФП = the operator
the Krylov step consumes must be *symmetric* AND *never materialised*. **Resolution
by separation in structure + super-system:** `M⁻¹K` is *similar* to the symmetric
`Â = R^{−T} K R^{−1}` (`M = RᵀR`); we never form `Â` — its matvec is the **chain**
`x ↦ R^{−T}(K(R^{−1}x))`, three sparse/triangular applies, so `K` stays sparse AND
the consumed operator is symmetric. The Krylov helpers already abstract the operator
behind a matvec, so the only essential change is to lift their incidental
`&Laplacian` bound to the `SymmetricLinearOp` trait they already behave against
(D2). The *factorisation* `M = RᵀR` — the only heavy, fill-in-prone step — is pushed
to the super-system: the caller/binding supplies `R` (scipy/`sksparse`), or a small
in-crate dense Cholesky serves moderate `n`. No property is split down the middle.

## Decision

Four additive pieces. **Symmetric `L` only** (non-symmetric ⇒ Arnoldi, explicitly
out of scope); **PSD is a documented precondition** (the `e^{−τL}` contraction /
`Growth::contraction()` rests on it).

### D1 — `SymmetricOperator<F>`: validated generic symmetric-CSR carrier

A newtype over `Arc<Laplacian<F>>` whose **only** constructor validates symmetry.
It reuses `Laplacian::from_csr_parts` (dimension/range checks + cached Gershgorin
bound) under a new documentation-only kind `LaplacianKind::GeneralSymmetric`, then
feeds A1 verbatim. The *direct generic-`L`* path and the *lumped-`(M,K)`* path both
go through this — i.e. through unmodified `GraphKrylovChernoff`.

```rust
// crates/semiflow/src/symmetric_operator.rs  (new, ≤220 lines)

/// Minimal interface the Krylov action needs: matvec + spectral bound + size.
/// (Lifts the incidental `&Laplacian` bound of the §54 helpers to what they
///  actually use.) Object-safe; matvec dominates cost so a vtable is free.
pub trait SymmetricLinearOp<F: SemiflowFloat> {
    fn n(&self) -> usize;
    fn lambda_max_bound(&self) -> F;                 // upper bound on ρ(op)
    fn apply_into_slice(&self, src: &[F], dst: &mut [F]); // dst ← op · src
}

impl<F: SemiflowFloat> SymmetricLinearOp<F> for Laplacian<F> { /* delegate */ }

/// Externally-assembled symmetric PSD sparse operator (CSR). PSD is a
/// PRECONDITION (cheap necessary check: all diagonal entries ≥ 0).
pub struct SymmetricOperator<F: SemiflowFloat = f64> {
    inner: alloc::sync::Arc<Laplacian<F>>, // kind = GeneralSymmetric
}

impl<F: SemiflowFloat> SymmetricOperator<F> {
    /// Build from externally-assembled symmetric CSR (rows sorted ascending).
    /// # Errors
    /// `DomainViolation` on: CSR shape/range (via `from_csr_parts`); asymmetry
    /// `|L[i,j] − L[j,i]| > sym_tol`; non-finite value; negative diagonal entry.
    pub fn from_csr(
        n: usize, row_ptr: Vec<usize>, col_idx: Vec<u32>, vals: Vec<F>, sym_tol: F,
    ) -> Result<Self, SemiflowError>;

    pub fn n(&self) -> usize;
    pub fn lambda_max_bound(&self) -> F;
    pub fn apply_into_slice(&self, src: &[F], dst: &mut [F]); // CSR SpMV (reused)

    /// A1 action over THIS operator (Chebyshev default / Lanczos). Zero new code:
    /// hands the inner `Arc<Laplacian>` to the existing `GraphKrylovChernoff`.
    pub fn krylov(&self, path: KrylovPath, tol: F)
        -> Result<GraphKrylovChernoff<F>, SemiflowError>;

    /// Lumped-mass congruence  Â = D^{−1/2} K D^{−1/2}  for diagonal `M = D`,
    /// returned as a SymmetricOperator (CSR entry scaling; stays sparse).
    /// # Errors  `DomainViolation` if any `masses[i] ≤ 0` or `len != n`.
    pub fn lumped_congruence(&self, masses: &[F]) -> Result<Self, SemiflowError>;
}
```

The **lumped `(M,K)`** action is a 3-liner around `lumped_congruence` + `krylov`
(pre-scale `w0 = √m ⊙ v`, evolve `Â`, post-scale `u = w / √m`), exposed as:

```rust
/// u(τ) = e^{−τ M⁻¹K} v for DIAGONAL (lumped) mass `M = diag(masses)`.
pub fn mass_lumped_evolve<F: SemiflowFloat>(
    k: &SymmetricOperator<F>, masses: &[F], tau: F,
    v: &[F], out: &mut [F], path: KrylovPath, tol: F, scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>;
```

### D2 — One mechanical A1 generalisation (lift `&Laplacian` → `&impl SymmetricLinearOp`)

The §54 private helpers (`chebyshev_action`, `lanczos_action`, `chebyshev_accumulate`,
`lanczos_iterate`, `lanczos_step_inner`) change their operator parameter from
`lap: &Laplacian<F>` to `op: &impl SymmetricLinearOp<F>`. The bodies are **unchanged**
(they already call only `apply_into_slice`). `GraphKrylovChernoff::apply_into` passes
`&*self.laplacian` (works via the `impl` in D1). `GraphKrylovChernoff`'s struct,
public signature, `order()`, `growth()`, and **A2 entirely** are untouched — so
ADR-0184 D5 0-ULP scope and §54 gates are preserved bit-for-bit.

### D3 — `MassKOperator<F>`: consistent (non-diagonal) `(M,K)` via the symmetric chain

Holds `K` (sparse) + the upper-triangular Cholesky factor `R` of `M = RᵀR`; its
matvec is the symmetric chain `x ↦ R^{−T}(K(R^{−1}x))`, so `Â = R^{−T}KR^{−1}` is
**never formed**. The factor is caller-supplied (binding path) or built by a small
in-crate dense Cholesky for moderate `n` (no LAPACK; the heavy sparse factorisation
is out of scope — large consistent mass ⇒ supply `R` or use lumped).

```rust
// crates/semiflow/src/mass_operator.rs  (new, ≤260 lines)

/// Upper-triangular factor R (dense, row-major) with O(n²) tri-solves.
pub struct TriangularFactor<F: SemiflowFloat = f64> { n: usize, r: Vec<F> }
impl<F: SemiflowFloat> TriangularFactor<F> {
    /// Small in-crate dense SPD Cholesky M = RᵀR (moderate n; O(n³), no deps).
    /// # Errors `DomainViolation` if M is not numerically SPD (non-positive pivot).
    pub fn dense_cholesky_spd(m_dense: &[F], n: usize) -> Result<Self, SemiflowError>;
    /// Adopt a caller-supplied upper-triangular R (binding / sparse-chol path).
    pub fn from_upper(n: usize, r_upper: Vec<F>) -> Result<Self, SemiflowError>;
    pub fn solve_r (&self, x: &[F], out: &mut [F]);   // R u = x   (back-sub)
    pub fn solve_rt(&self, x: &[F], out: &mut [F]);   // Rᵀu = x   (fwd-sub)
    pub fn apply_r (&self, x: &[F], out: &mut [F]);   // out = R x  (for w0 = R v)
}

/// e^{−τ M⁻¹K} for CONSISTENT SPD mass M = RᵀR. Default path: Lanczos
/// (adaptive; needs only a loose λmax for substep count).
pub struct MassKOperator<F: SemiflowFloat = f64> {
    k: alloc::sync::Arc<Laplacian<F>>,   // K (GeneralSymmetric inner of a SymmetricOperator)
    r: TriangularFactor<F>,
    lambda_max_bound: F,                 // gershgorin(K) / mu_min(M)
}
impl<F: SemiflowFloat> SymmetricLinearOp<F> for MassKOperator<F> { /* chain matvec */ }

impl<F: SemiflowFloat> MassKOperator<F> {
    /// # Errors `DomainViolation` on dimension mismatch.
    pub fn new(k: &SymmetricOperator<F>, r: TriangularFactor<F>)
        -> Result<Self, SemiflowError>;

    /// u(τ) = e^{−τ M⁻¹K} v = R⁻¹ e^{−τÂ} (R v),  Â = R^{−T}KR^{−1} (never formed).
    pub fn evolve(
        &self, tau: F, v: &[F], out: &mut [F],
        path: KrylovPath, tol: F, scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError>;
}
```

### D4 — `EntrySensitivity<F>`: generic single-entry Fréchet (reuses A2 verbatim)

A new `GeneratorSensitivity` impl whose `apply_param_deriv(k, ·)` applies the
single-entry symmetric stencil for the k-th tracked entry `(i,j)`:
`∂L/∂L_{ij}` acts as `v ↦ (e_i e_jᵀ + e_j e_iᵀ)v` for `i≠j` (one symmetric DOF),
and `v ↦ e_i e_iᵀ v` for `i=j`; negated for `∂A = −∂L`. The existing
`graph_expmv_frechet` (GL8 Duhamel, §54.5) consumes it **unchanged** — exact for
non-commuting directions. The action is built via `SymmetricOperator::krylov`, so the
whole differentiable path is `SymmetricOperator` + existing A2 + this 1 stencil.

```rust
// crates/semiflow/src/entry_sensitivity.rs  (new, ≤90 lines)
pub struct EntrySensitivity { pub entries: Vec<(usize, usize)>, pub n_nodes: usize }
impl<F: SemiflowFloat> GeneratorSensitivity<F> for EntrySensitivity { /* §55.4 stencil */ }
```

### D5 — Bindings sketch (`bindings` label; scipy-CSR in / gradient out)

PyO3 (`crates/semiflow-py`) consumes a scipy `csr_matrix` as `(indptr, indices, data,
n)` numpy arrays and a numpy `v`; the Rust side is the monomorphic `f64` surface.

```rust
// crates/semiflow-py — signatures only
#[pyfunction] fn sym_op_evolve(            // e^{−τL}·v, generic symmetric CSR
    indptr: PyReadonlyArray1<usize>, indices: PyReadonlyArray1<u32>,
    data: PyReadonlyArray1<f64>, n: usize, v: PyReadonlyArray1<f64>,
    tau: f64, tol: f64) -> PyResult<Py<PyArray1<f64>>>;

#[pyfunction] fn mass_k_evolve(            // e^{−τ M⁻¹K}·v; mass="lumped"|"consistent"
    k_csr: (…), m: PyReadonlyArray1<f64> /*diag or R-upper*/, mass_kind: &str,
    n: usize, v: …, tau: f64, tol: f64) -> PyResult<Py<PyArray1<f64>>>;

#[pyfunction] fn sym_op_entry_grad(        // ∂J/∂entries for tracked (i,j) list
    k_csr: (…), n: usize, u0: …, dj: …, n_cols: usize, t: f64,
    entries: PyReadonlyArray2<usize>) -> PyResult<Py<PyArray1<f64>>>;
```

(Consistent-mass binding: caller passes `R` from `scipy`/`sksparse`; the core never
factorises a large sparse `M`.) C-ABI exposure follows the carrier-handle precedent
(`semiflow-ffi.s3-carrier-handle.yaml`) and is deferred — not required by #13.

### D6 — Acceptance gates (NORMATIVE; all RELEASE_BLOCKING, `feature_gate: slow-tests`)

| Gate | Definition | Threshold | Oracle (REUSE) |
|------|-----------|-----------|----------------|
| `G_SYMOP_DENSE` | `SymmetricOperator::krylov` action `e^{−τL}v` vs dense `mat_exp_pade13(−τL)`, **non-zero-row-sum** Robin-BC 1-D stiffness `L` (`N ≤ 12`), `τ‖L‖` in the ≥10 regime | `sup_error ≤ 1e-10` | dense `mat_exp_pade13` (§54.7 `_DENSE` pattern; **no sympy**) |
| `G_MASSK_LUMPED` | `mass_lumped_evolve` vs dense `expm(−τ M⁻¹K)v` for diagonal `M` (random `m_i∈[0.5,2]`) + Robin `K` (`N ≤ 12`) | `sup_error ≤ 1e-10` | dense `mat_exp_pade13` of `−τ D⁻¹K` |
| `G_MASSK_CONSISTENT` | `MassKOperator::evolve` vs dense `expm(−τ M⁻¹K)v` for **non-diagonal** consistent FEM mass `M = (h/6)·tridiag(1,4,1)` + Robin `K` (`N ≤ 12`) | `sup_error ≤ 1e-9` | dense `mat_exp_pade13` of `−τ M⁻¹K` (M⁻¹K via in-crate dense Cholesky solve) |
| `G_SYMOP_ENTRY_FRECHET` | `⟨∂J/∂entries, δ⟩` (A2 + `EntrySensitivity`) vs central FD `(J(L+εδ)−J(L−εδ))/(2ε)`; tracked set includes ≥1 **off-diagonal** entry on a **non-zero-row-sum** `L` | rel-err `≤ 1e-7` | §43.6 FD pattern / `T_ADJOINT_STATE_SENSITIVITY` (**no new oracle**) |

**Non-vacuity is asserted inside each gate** (a degenerate shortcut cannot pass):
`G_SYMOP_DENSE`/`_ENTRY_FRECHET` assert at least one row sum `≠ 0`;
`G_MASSK_CONSISTENT` asserts `M[i,i+1] ≠ 0` (genuinely non-diagonal, so a lumped
shortcut fails the tolerance).

## Consequences

- **Reuse, near-zero disturbance.** New behaviour = one trait + one `impl` for
  `Laplacian` + a mechanical bound-lift on 5 private helpers + three small new files
  + one new `GeneratorSensitivity` impl. A1's public type/`order()`/`growth()`,
  A2 in full, all §54 gates, and ADR-0184 0-ULP scope are unchanged.
- **One new public `LaplacianKind::GeneralSymmetric` variant** ⇒ 2 exhaustive
  `match` arms in `graph.rs` (`fill_laplacian_rows`, `append_diagonal_entries`) gain
  an `unreachable!()` arm (those paths are graph-assembly-only; `GeneralSymmetric`
  flows solely through `from_csr_parts`). Pre-1.0 (0.9.x): acceptable, ADR-recorded.
- **Honest boundaries (documented, not hidden):** symmetric `L` only; PSD assumed
  (cheap necessary diag-≥0 check only); consistent-mass dense Cholesky is O(n³) for
  moderate `n` — large sparse consistent mass requires a caller-supplied `R` or the
  lumped path; consistent-`(M,K)` **differentiability is out of scope** for #13 (the
  entry-Fréchet covers the direct symmetric-`L` path).
- **Suckless:** zero new dependencies; no LAPACK; every new file ≤ limits
  (`symmetric_operator.rs` ≤220, `mass_operator.rs` ≤260, `entry_sensitivity.rs` ≤90);
  all fns ≤50 lines; one build path unchanged.

### Implementation ordering (for the engineer)

1. **Carrier + trait** — `SymmetricLinearOp<F>`, `impl … for Laplacian`,
   `LaplacianKind::GeneralSymmetric` (+ 2 `unreachable!()` arms),
   `SymmetricOperator::{from_csr, krylov}` with symmetry/finiteness/diag-≥0
   validation. Gate: **`G_SYMOP_DENSE`**.
2. **A1 reuse generalisation** — lift the 5 §54 helpers to `&impl SymmetricLinearOp`
   (bodies untouched); re-run §54 gates to confirm bit-identical.
3. **Lumped `(M,K)`** — `lumped_congruence` + `mass_lumped_evolve`.
   Gate: **`G_MASSK_LUMPED`**.
4. **Consistent `(M,K)`** — `TriangularFactor` (tri-solves + small dense Cholesky),
   `MassKOperator` (chain matvec + `lambda_max_bound`), `evolve`.
   Gate: **`G_MASSK_CONSISTENT`**.
5. **Entry-Fréchet** — `EntrySensitivity` (§55.4 stencil); drive the **existing**
   `graph_expmv_frechet`. Gate: **`G_SYMOP_ENTRY_FRECHET`**.
6. **Bindings (`bindings` label)** — `sym_op_evolve` / `mass_k_evolve` /
   `sym_op_entry_grad` over the f64 surface; scipy-CSR in, numpy out.

## References

- ADR-0185 / §54 — A1 Krylov action + A2 Fréchet (the reused mechanism).
- ADR-0180 — `Laplacian::from_csr_parts` (reused CSR ingest + Gershgorin).
- ADR-0115 / §43 — `GeneratorSensitivity` trait + §43.6 FD oracle (reused).
- ADR-0125 — `mat_exp_pade13` (reused dense oracle).
- A. H. Al-Mohy, N. J. Higham (2011/2009) — action / Fréchet of `expm` (§54.8).
- Y. Saad (1992) — Lanczos action error bound (§54.2).
- G. Strang, G. J. Fix, *An Analysis of the Finite Element Method* — consistent vs
  lumped mass; congruence `R^{−T}KR^{−1}` of the generalized eigenproblem.
