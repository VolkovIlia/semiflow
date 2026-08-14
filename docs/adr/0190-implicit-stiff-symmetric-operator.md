# ADR-0190 — Implicit / shift-invert stiff action for the externally-assembled `SymmetricOperator` path (dependency-free PCG)

- **Status**: Proposed (design only — Issue #16; branch `issue-16-implicit-symmetric-operator`)
- **Date**: 2026-07-02
- **Supersedes**: none — purely ADDITIVE. Reuses ADR-0186/§55
  (`SymmetricOperator::from_csr`, `SymmetricLinearOp`, `lumped_congruence`,
  `mass_lumped_evolve`, `MassKOperator`), ADR-0185/§54 (the explicit A1 Krylov
  action, LEFT UNTOUCHED), ADR-0125/§45 (`mat_exp_pade13` dense oracle).
  It is the **general sparse-CSR** sibling of ADR-0188/§57.4, whose A-stable
  Crank–Nicolson solve is **tridiagonal-only** (Thomas pass) and therefore does
  NOT cover an arbitrary externally-assembled FEM stiffness. No existing kernel,
  gate, public signature, or 0-ULP scope is changed.
- **Contract**: `contracts/semiflow-core.math.md` §59 (new NORMATIVE section);
  API in `contracts/semiflow-core.implicit-symmetric-api.md`; gates
  `G_SYMOP_IMPLICIT_DENSE`, `G_SYMOP_IMPLICIT_STIFF`, `G_SYMOP_IMPLICIT_PCG_SPD`.

## Decision (≤1 paragraph)

Add an **implicit backward-Euler / shift-invert** action for `SymmetricOperator`
(and the lumped `(M,K)` path) that computes `e^{−tA}v ≈ (I+Δt·A)^{−n_steps} v`
by repeatedly solving the shifted system `S x = b`, `S = I + Δt·Â`, with a
**dependency-free preconditioned Conjugate Gradient (PCG)** solver that reuses the
existing `SymmetricLinearOp::apply_into_slice` matvec — **no new crate, no
governance amendment**. The shift makes this unconditionally well-posed: because
`Â` is symmetric PSD (`σ(Â) ⊂ [0, λ_max]`), `S` is symmetric **positive-definite**
for every `Δt > 0` even when `A` is singular (`σ(S) ⊂ [1, 1+Δt·λ_max] ⊂ (0,∞)`), so
CG never breaks down and converges. The single reusable "factorization" is the
**preconditioner** `P ≈ S`, built once for the fixed `Δt` and reused across all
`n_steps` sub-steps **and** all channels; v1 uses the Jacobi (diagonal) `P`, with
`IC(0)` (incomplete Cholesky, zero fill-in — keeps exactly the sparsity of `S`,
so no fill-in blow-up on 3-D FEM) specified as the drop-in stronger `P`. The
implicit path is surfaced as `path="implicit"` on the existing PyO3
`evolve_batched` / `mass_lumped_evolve` and as a new `KrylovPath::ImplicitEuler
{ n_steps }` arm in `graph_krylov.rs`; the mass-weighting is handled by the
UNCHANGED §55.3 √μ congruence (the solver only ever sees the symmetric `Â`).

## Context

The `SymmetricOperator.evolve_batched` / `mass_lumped_evolve` / `MassKOperator.evolve`
path (0.11.0-beta) computes `e^{−τA}v` by the explicit §54 Chebyshev/Lanczos expmv,
which sub-steps `s = ⌈τ·λ_max / Z_SAFE⌉` (`Z_SAFE=200`) with `m_max` matvecs per
sub-step. Its cost is `O(λ_max·t)`. For a real stiff FEM heat operator (M5-bolt
cooling, `λ_max ≈ 5.14e7 /s`) this is `~10^7–10^9` matvecs and **times out at t=1 s**;
`m_max` is only a per-sub-step cap, not a global work cap. scipy's `expm_multiply`
times out identically (its cost is likewise `∝ ‖tA‖₁`), so an implicit /
factorization method is a **genuine capability, not scipy parity**. ADR-0188/§57.4
already ships an implicit A-stable step, but its Thomas solve assumes a **tridiagonal**
`A` (1-D nodal conduction); an externally-assembled 3-D FEM stiffness is general
sparse CSR, so §57.4 does not apply. The gap is exactly the general-CSR implicit solve.

## Dependency decision (the delegated choice) — recommend (ii), dependency-free

The dep budget is **3/3 saturated** (`num-traits`, `libm`, `num-complex`;
constitution v7.0.0) and the **override slots are 3/3 saturated**. Option (i) —
adding `faer`/`nalgebra-sparse` — therefore requires a permanent constitution
amendment (raise the ≤3 dep cap or retire an override) and maintainer sign-off.
Option (ii) needs none.

| Criterion | (i) external Cholesky/LU crate | (ii-a) hand-rolled sparse Cholesky | **(ii-b) PCG, matrix-free (CHOSEN)** |
|-----------|-------------------------------|-------------------------------------|--------------------------------------|
| New dependency | **yes** (`faer`/`nalgebra-sparse`) | no | **no** |
| Governance gate | **yes** — dep 4/4 + override amendment + maintainer sign-off | no | **no** |
| λ_max-independence | full (direct factor) | full (direct factor) | √κ (Jacobi) → near-constant (IC(0)); NOT strictly constant |
| Fill-in on 3-D FEM | handled (crate reordering) | **catastrophic** without AMD/nested-dissection (self-owned) | **none** (matrix-free; IC(0) keeps `S` sparsity) |
| Code to own/verify | ~small wrapper | **~250–350 LoC** (symbolic + numeric + ordering) | **~70 LoC** (v1 Jacobi) / +40 (IC(0)) |
| SPD-after-shift guarantee needed | yes | yes | yes (proven §59.3) |
| Suckless risk | new heavy dep; audit surface | large self-owned numerics; fill risk | **lowest** |
| Reuse-one-factorization | the Cholesky factor | the Cholesky factor | the preconditioner `P` (built once, reused ∀ sub-steps, channels) |

**Recommendation: (ii-b) dependency-free PCG.** It resolves the contradiction
without compromise (see TRIZ note): unconditional implicit stability **and** suckless
zero-dep **and** zero fill-in, by exploiting resources already in the topology (the
existing matvec + the SPD-lift of the `+I` shift). **Governance flag: NONE required.**
(If a future hard requirement for strict λ_max-independence emerges, an in-crate
sparse Cholesky or option (i) becomes a separate ADR with the maintainer sign-off
recorded there.)

## TRIZ note (the contradiction is resolved by a topology resource, not a compromise)

**АП**: the stiff FEM operator needs an implicit solve whose cost is set by the low
modes. **ТП**: a direct factorization is λ_max-independent but is either a new dep
(governance) or a self-owned sparse Cholesky that fills catastrophically on 3-D FEM
(violates suckless) / a matrix-free method is suckless and fill-free but "has no
factorization to reuse". **ФП**: the solver must be **a reusable factorization** AND
**not a factorization** (matrix-free, zero fill, ≤80 LoC). **Resolution in structure**:
split `S⁻¹` into a matrix-free PCG iteration (implements `S⁻¹` via the existing
matvec — zero fill) **plus** a preconditioner `P` that IS the single reusable
"factorization" (Jacobi diagonal, or IC(0) with zero fill-in). The `+I` shift is a
free super-system resource: it lifts `σ(Â)⊂[0,λ_max]` to `σ(S)⊂[1,1+Δt·λ_max]⊂(0,∞)`,
making `S` SPD for any `Δt>0` even when `A` is singular, so CG is well-posed and
cannot break down. The system is thus **simultaneously** a reused factorization (`P`)
and a matrix-free non-factorization (PCG) — both properties at once, no golden mean.

## Consequences

- **Additive.** New `KrylovPath::ImplicitEuler { n_steps }` arm + a new `pcg` /
  `implicit_action` helper module in the core; the explicit Chebyshev/Lanczos arms,
  their substep logic, and all §54/§55/§57 gates are untouched.
- **Blast radius (fail-loud).** Adding the enum variant makes every `match KrylovPath`
  site fail to compile until it gains the new arm (`graph_krylov.rs::action`,
  `substep`/degree selection, PyO3 `krylov_path`). These are the only WILL-BREAK
  (d=1) sites; they are compile-time-enforced and few (≤4). The explicit hot path
  is not re-entered by the implicit arm.
- **Errors as values.** CG non-convergence within `max_iter` → `ConvergenceFailed`
  (existing kind → PyO3 `ConvergenceFailed`); a non-positive IC(0) pivot → fall back
  to Jacobi (never surfaces) or, if forced, `DomainViolation` (→ `OutOfDomain`). **No
  new `SemiflowError` kind** — reuses the existing taxonomy.
- **Accuracy.** Backward-Euler is O(Δt), L-stable (damps stiff modes — unlike A-stable
  Crank–Nicolson, which does not). Order-2 is available L-stably via single-γ SDIRK2
  reusing the same `P`; deferred to keep v1 minimal (§59.5).
- **Mass cases.** Lumped `M⁻¹A` and consistent `(M,K)` reuse the UNCHANGED §55.2/§55.3
  symmetric congruence `Â`; the implicit solver never touches the non-symmetric
  `M⁻¹A`. Consistent-mass implicit is a trivial reuse (same `SymmetricLinearOp` via the
  `MassKOperator` R-solve chain), scoped as a follow-on.

## References

- Issue #16 (M5-bolt stiff FEM heat; explicit + scipy `expm_multiply` both time out).
- Y. Saad, *Iterative Methods for Sparse Linear Systems*, 2nd ed., SIAM 2003 — CG,
  preconditioning, IC(0).
- G. H. Golub & C. F. Van Loan, *Matrix Computations*, 4th ed. — SPD CG convergence
  `≈ ½√κ ln(2/ε)`; the shift `I+Δt·A` conditioning.
- E. Hairer & G. Wanner, *Solving ODEs II*, Springer — backward-Euler / SDIRK
  L-stability for stiff systems.
- §55 (`SymmetricOperator`, `SymmetricLinearOp`, congruence — reused verbatim), §57.4
  (tridiagonal-only CN — the boundary this ADR extends), §45 (`mat_exp_pade13` oracle).
- ADR-0186 (§55 authority), ADR-0188 (§57 authority), ADR-0185 (§54 explicit action).
