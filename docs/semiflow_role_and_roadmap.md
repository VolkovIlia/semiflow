# SemiFlow: Backend Role and Roadmap Narrative

> Last updated: 2026-06-27 (v0.10.0-beta, issue campaign #11/#12/#13/#14 + A1 stiff fix)

## The "BLAS/cuDNN for Semigroups" Positioning

SemiFlow occupies the same structural role in the operator-semigroup stack that
BLAS occupies in linear algebra, or cuDNN in GPU neural-network inference:

- **BLAS** provides the fundamental matrix-multiply primitives (`dgemm`, `dsymv`…)
  that every higher-level framework (LAPACK, scipy, PyTorch) composes.
- **cuDNN** provides verified, hardware-aware convolution and attention kernels
  that deep learning frameworks call into directly, treating them as correct black boxes.
- **SemiFlow** provides the verified, exact, memory-frugal semigroup evaluation
  primitives — `e^{τL}·v`, `φ_k(τL)·v`, `∂J/∂w` through a semigroup —
  that higher-level systems (`revssm`, specialized ML layers, simulation codes)
  call into as correct black boxes, without needing to implement or maintain the
  underlying numerics.

The analogy is structural, not performance-based. SemiFlow is NOT faster than
adaptive ODE solvers for general PDE problems (H-WALL FALSIFIED by iter-8).
What the analogy captures is:

1. **Primitives are correct and verified.** Every public kernel is gated in CI
   against a closed-form or high-order oracle. Callers do not inherit numerical
   uncertainty from reimplementing semigroup actions.
2. **The primitive boundary is explicit.** The library ships math primitives
   (operator application, gradient, φ-functions). Training loops, autograd tape,
   optimization algorithms, and state-space-model wrappers stay in the caller
   (`revssm` boundary reaffirmed in ADR-0115 and every subsequent ADR that touches
   graph/ML interfaces).
3. **Composable, not monolithic.** Semigroup engines are thin `ChernoffFunction`
   impls. A caller assembles a pipeline (`SymmetricOperator` → Krylov action →
   `Etdrk4` step → gradient via `GeneratorSensitivity`) from verified pieces rather
   than writing bespoke numerical code per use case.

## What Is Covered (as of v0.10.0-beta)

### Linear evolution: `∂ₜu = Lu`

The core of every primitive. `L` may be:

| Class | Representative API | Notes |
|-------|--------------------|-------|
| Constant / smooth variable-coefficient | `DiffusionChernoff`, `ShiftChernoff1D`, `Diffusion4thChernoff` | 1–3D, `f32`/`f64` |
| **Conservative (divergence-form)** | `ConservativeDiffusionChernoff`, `assemble_conservative_csr_1d` | Harmonic-mean faces; sharp k-jumps; contact resistance `R_c` |
| Graph Laplacian (standard stepping) | `GraphHeatChernoff`, `MagnusGraphHeatChernoff`, `VarCoefGraphHeatChernoff` | Per-channel, batched, adjoint |
| **Graph semigroup (depth-independent)** | `GraphKrylovChernoff` | Chebyshev O(1)-vector / Lanczos O(m·N); matvec count flat in `t` |
| **Generic symmetric PSD** | `SymmetricOperator`, `SymmetricLinearOp` | FEM stiffness, anisotropic conductivity, any externally-assembled symmetric CSR |
| **Generalized eigenproblem `(M,K)`** | `MassKOperator`, `mass_lumped_evolve` | Consistent mass via Cholesky congruence; lumped mass via pre-scaling |
| **Stiff multilayer conduction** | `MultilayerStack`, `multilayer_evolve` | Per-layer `(k, ρc)`; one depth-flat Krylov action for the whole interval |
| High-dimensional (TT carrier) | `TtChernoff`, `VarCoefTt` | Curse-escape for diagonal-A Gaussian class |
| Manifold / hypoelliptic / graph-quantum | `ManifoldChernoff`, `HypoellipticChernoff`, `QuantumGraphHeatChernoff` | See engine catalogue in README |

### Semilinear evolution: `∂ₜu = Lu + N(u)` (NEW in v0.10.0-beta)

`Etdrk4` (Cox–Matthews 2002 / Kassam–Trefethen 2005) composes with ANY linear
engine via the `GeneratorAction` adapter. It treats `L` exactly — never re-discretizes
or operator-splits it — and quadratures the Duhamel term via φ-functions.

`phi_action` / `phi_action_batched` compute φ₀…φ_p(τL)·v simultaneously via one
augmented block-triangular matvec-only Taylor action (Al-Mohy & Higham 2011 §4),
reusing the existing `THETA_M` substepping. No Padé on φ, no contour integrals,
no new dependencies.

`N(u)` is a declarative `Nonlinearity` trait:
- **Native Rust:** any `eval(u, out)` impl (Allen–Cahn, Gray–Scott, KS, Burgers
  are pre-built); supports JVP/VJP via `NonlinearityDiff` for end-to-end adjoint.
- **PyO3 surface:** fixed enum menu crossed once at construction — `py.detach`
  preserved, zero per-step GIL crossing.
- **Per-step arbitrary Python/JS callback:** explicit non-goal (ADR-0179 wall;
  per-step crossing reintroduces 200× / GIL-defeat hazards).

### Gradient / adjoint (forward and backward both depth-independent)

Before v0.10.0-beta, the edge-weight gradient `∂J/∂w` scaled as
`O(edges · C · n_steps)` — linear in the step count. `graph_expmv_frechet` closes
this: it seeds the **augmented Krylov action** (Al-Mohy & Higham 2009) once with
the upstream cotangent and reads out `∂J/∂w` for all edges simultaneously.
Cost = one augmented Krylov solve ≈ two forward-depth-flat actions.

The `revssm` Semigroup Layer (Track B) therefore gets:
- Forward: `GraphKrylovChernoff` — one call, depth-flat, O(m·N) working memory.
- Backward: `graph_expmv_frechet` — one augmented call, depth-flat, per-step
  trajectory storage eliminated.

For generic `SymmetricOperator`, `EntrySensitivity` provides per-entry Fréchet
gradient reusing the same augmented mechanism.

For semilinear `Etdrk4`, `NonlinearityDiff` delivers the adjoint through one step
(chain rule through φ-actions and N; gated by `G_ETD_ADJOINT_FD`).

## Honest Scope and Limits

The following are explicit non-goals or deferred items as of v0.10.0-beta:

- **Non-symmetric / directed graphs:** Arnoldi required (stores full Hessenberg);
  deferred. Symmetric `L` only for Krylov/Chebyshev paths.
- **Time-varying `L(t)` in Krylov:** not a semigroup — Magnus/Howland (already in
  core) cover the augmented-generator path; Krylov action is for fixed `L`.
- **Chebyshev stiff-substep cost:** scales as `O(τλ_max)` when substepping is
  active; use Lanczos path for stiff operators.
- **`(M,K)` consistent-mass differentiability:** the entry-Fréchet covers
  `SymmetricOperator` only; `MassKOperator` gradient is deferred.
- **2-D/3-D tensor ETD, exponential Rosenbrock, per-step Python N:** deferred.
- **Full-tensor non-separable conservative diffusion `∂_x(k∂_y)`:** out of scope;
  separable axes only.

## Relationship to Downstream Systems (`revssm` / ML Layers)

SemiFlow enforces a hard boundary (ADR-0115, reaffirmed through ADR-0189):

- **Core ships:** computation primitives — semigroup action, φ-function, Fréchet
  gradient, conservative assembly, multilayer plumbing, ETD stage logic.
- **Core does NOT ship:** autograd tape, training loop, optimizer state, model
  parameters, learned `N(u)` arbitrary per-step callbacks.

This boundary is what makes the "BLAS/cuDNN" analogy valid: the primitive layer
stays thin, verified, and independently releasable; the downstream system composes
it into a learning or simulation pipeline without inheriting the numerical risk of
reimplementing semigroup numerics.

## Roadmap Pointers

| Topic | ADR | Math section |
|-------|-----|--------------|
| Depth-independent graph Krylov + Fréchet | ADR-0185 | §54 |
| Generic symmetric-operator entry | ADR-0186 | §55 |
| Conservative divergence-form diffusion | ADR-0187 | §56 |
| Stiff multilayer (mass-weighted Krylov) | ADR-0188 | §57 |
| ETD φ-functions + ETDRK4 | ADR-0189 | §58 |
| Graph adjoint state + sensitivity (v6.2.2) | ADR-0115 | §42–§43 |
| Batched multi-channel (v0.9.1-beta, #10) | ADR-0184 | §43.4 |
| Reverse-mode AD (v9.0.0) | ADR-0156 | §51 |

Open issues tracked in the ROADMAP and ADRs: non-symmetric graph Krylov (Arnoldi),
consistent-mass Fréchet, 2-D/3-D ETD, exponential Rosenbrock, per-step learned
`N(u)` via native C fn-ptr (the only cross-language arbitrary-N path).
