# SemiFlow: Backend Role and Roadmap Narrative

> Covers 0.15.0-beta. Install commands, feature flags and the full engine
> catalogue are in the [project README](../README.md).

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

## What Is Covered (as of 0.15.0-beta)

### Linear evolution: `∂ₜu = Lu`

The core of every primitive. `L` may be:

| Class | Representative API | Notes |
|-------|--------------------|-------|
| Constant / smooth variable-coefficient | `DiffusionChernoff`, `ShiftChernoff1D`, `Diffusion4thChernoff` | 1–3D, `f32`/`f64` |
| **Conservative (divergence-form)** | `ConservativeDiffusionChernoff`, `assemble_conservative_csr_1d` | Harmonic-mean faces; sharp k-jumps; contact resistance `R_c` |
| Graph Laplacian (standard stepping) | `GraphHeatChernoff`, `MagnusGraphHeatChernoff`, `VarCoefGraphHeatChernoff` | Per-channel, batched, adjoint |
| **Graph semigroup (depth-independent)** | `GraphKrylovChernoff` | Chebyshev O(1)-vector / Lanczos O(m·N); matvec count independent of the step count: Chebyshev `O(√(λ_max·t))`, Lanczos linear in `λ_max·t` at fixed `m_max` (ADR-0205) |
| **Generic symmetric PSD** | `SymmetricOperator`, `SymmetricLinearOp` | FEM stiffness, anisotropic conductivity, any externally-assembled symmetric CSR; `path="implicit"` (PCG backward-Euler) for stiff operators (ADR-0190) |
| **Generalized eigenproblem `(M,K)`** | `MassKOperator`, `mass_lumped_evolve` | Consistent mass via Cholesky congruence; lumped mass via pre-scaling; `path="implicit"` available (ADR-0190) |
| **Stiff multilayer conduction** | `MultilayerStack`, `multilayer_evolve` | Per-layer `(k, ρc)`; one depth-flat Krylov action for the whole interval |
| **Very stiff SPD (shift-and-invert)** | `RationalKrylov`, `RationalKrylovChernoff`, `path="rational"` | `e^{−tM⁻¹A}v` from one factor of `M/γ + A`; solve count independent of `λ_max·t` (27 at `tol = 1e−10`) (ADR-0206) |
| **Non-symmetric (general CSR)** | `GeneralOperator`, `expmv_krylov` | Taylor `expmv` (default) or adaptive Arnoldi `e^{−τA}v` for large `τ‖A‖` (ADR-0211) |
| High-dimensional (TT carrier) | `TtChernoff`, `VarCoefTt` | Curse-escape for diagonal-A Gaussian class |
| Manifold / hypoelliptic / graph-quantum | `ManifoldChernoff`, `HypoellipticChernoff`, `QuantumGraphHeatChernoff` | See engine catalogue in README |

**Resolvent / steady state.** `SpdResolvent` (`SymmetricOperator::resolvent`,
ADR-0202) factors `λM + A` (`λ ≥ 0`; `λ = 0` is the steady state) once for
repeated solves: exact `O(N)` LDLᵀ for tridiagonal operators, PCG with IC(0) or
Jacobi otherwise (`O(√κ·nnz)` per solve, no multigrid).

### Semilinear evolution: `∂ₜu = Lu + N(u)` (NEW in v0.10.0-beta)

`Etdrk4` (Cox–Matthews 2002 / Kassam–Trefethen 2005) composes with ANY linear
engine via the `GeneratorAction` adapter. It treats `L` exactly — never re-discretizes
or operator-splits it — and quadratures the Duhamel term via φ-functions.

`phi_action` / `phi_action_batched` compute φ₀…φ_p(τL)·v simultaneously via one
augmented block-triangular matvec-only Taylor action (Al-Mohy & Higham 2011 §4),
reusing the existing `THETA_M` substepping. No Padé on φ, no contour integrals,
no new dependencies.

`phi_combination` (ADR-0202) computes `Σₖ τᵏφₖ(τG)wₖ` (`p ≤ 3`) in one augmented
sweep: exact affine evolution `Mu' = −Au + s`, and each ETDRK4 stage. For large
`τ‖G‖`, `phi_combination_krylov` / `expmv_krylov` use adaptive Arnoldi
(Expokit-style, ADR-0211) on any `GeneratorAction`, non-symmetric included, and
`Etdrk4::with_phi_solver(PhiSolver::Krylov(..))` keeps order 4; the Taylor path
stays the default (competitive below `τ‖G‖ ≈ 10³`).

`N(u)` is a declarative `Nonlinearity` trait:
- **Native Rust:** any `eval(u, out)` impl (Allen–Cahn, Gray–Scott, KS, Burgers
  are pre-built); supports JVP/VJP via `NonlinearityDiff` for end-to-end adjoint.
- **PyO3 surface:** fixed enum menu crossed once at construction — `py.detach`
  preserved, zero per-step GIL crossing.
- **Per-step arbitrary Python/JS callback:** explicit non-goal for `Etdrk4`
  (ADR-0179 wall; per-step crossing reintroduces 200× / GIL-defeat hazards). The
  one batched exception is `ReactionDiffusion` (below): a vectorised Python
  kinetics callable is called once per stage with all nodes (ADR-0209); per-node
  callbacks stay excluded.

### Semilinear systems: `∂ₜu_k = L_k u_k + f_k(t, x, u)` (NEW in v0.15.0-beta)

`ReactionDiffusion` (ADR-0208) evolves `K` species, each diffusing with its own
engine — any `ChernoffFunction` on a 1-D/2-D/3-D grid or a graph, with its own
boundary policy, interpolant and Strang/ADI composition — Strang-split with the
reaction, which is evaluated for all nodes in one batched call (catalogue:
Fisher–KPP, Allen–Cahn, Nagumo, Gray–Scott, FitzHugh–Nagumo, Brusselator, linear,
closure). Python: `ReactionDiffusion1D/2D/3D`.

- **Stiff kinetics:** exact flow or RK4 substeps (default), or adaptive Rosenbrock
  ROS2 (ADR-0212): L-stable, order 2 for any Jacobian (analytic or forward
  differences), per-node `K×K` LU, embedded error control.
- **Cross-diffusion** with a constant `K×K` matrix (ADR-0213): exact modal
  splitting `D = PΛP⁻¹`, verified; complex, negative and (nearly) defective `D` are
  rejected. Every engine is reused, with every boundary policy except a non-zero
  `Dirichlet` value.
- **Richardson extrapolation for any engine** (ADR-0207): any `StepIntegrator`
  (`Evolver`, `ReactionDiffusion`) goes from order `p` to `p + L − 1` with `L`
  levels; tolerance-driven kernels are rejected.

### Gradient / adjoint (forward and backward both depth-independent)

Before v0.10.0-beta, the edge-weight gradient `∂J/∂w` scaled as
`O(edges · C · n_steps)` — linear in the step count. `graph_expmv_frechet` closes
this: it evaluates the exact Duhamel integral with depth-flat Krylov propagators
and reads out `∂J/∂w` for all edges at once — an all-edges gradient costs the same
propagator actions as one edge. Since v0.15.0-beta the quadrature is a two-sided
graded Gauss–Legendre mesh (ADR-0203): its error is independent of `λ_max·t` and
memory is `≤ 15·N`; cost `5.4–25×` one forward action (measured, `ρ̄t = 1e6…1e3`).

The `revssm` Semigroup Layer (Track B) therefore gets:
- Forward: `GraphKrylovChernoff` — one call, depth-flat, O(m·N) working memory.
- Backward: `graph_expmv_frechet` — one call, depth-flat, per-step trajectory
  storage eliminated.

For generic `SymmetricOperator`, `EntrySensitivity` provides per-entry Fréchet
gradient reusing the same mechanism.

For semilinear `Etdrk4`, `NonlinearityDiff` delivers the adjoint through one step
(chain rule through φ-actions and N; gated by `G_ETD_ADJOINT_FD`).

## Honest Scope and Limits

The following are explicit non-goals or deferred items as of 0.15.0-beta:

- **Non-symmetric / directed graphs:** `GraphKrylovChernoff`, the
  Chebyshev/Lanczos/rational paths and the Fréchet gradient are symmetric-only.
  Non-symmetric generators (`GeneralOperator`, any `GeneratorAction`) get the
  Taylor or adaptive Arnoldi actions (ADR-0211).
- **Time-varying `L(t)` in Krylov:** not a semigroup — Magnus/Howland (already in
  core) cover the augmented-generator path; Krylov action is for fixed `L`.
- **Stiff SPD operators:** the polynomial paths cost `O(√(λ_max·t))` mat-vecs
  (`path="chebyshev"`) or grow linearly in `λ_max·t` at fixed `m_max`
  (`"lanczos"`) (ADR-0205). `path="rational"`
  (ADR-0206) needs a `λ_max·t`-independent number of solves, but its factor is an
  exact `O(N)` LDLᵀ only for tridiagonal operators; other patterns use PCG (IC(0) or
  Jacobi, no multigrid, `O(√κ·nnz)` per solve). `path="implicit"` (ADR-0190) stays
  first order in `t/n_steps`.
- **`(M,K)` consistent-mass differentiability:** the entry-Fréchet covers
  `SymmetricOperator` only; `MassKOperator` gradient is deferred.
- **2-D/3-D tensor ETD, exponential Rosenbrock, per-step Python N for `Etdrk4`:**
  deferred (the ROS2 of ADR-0212 is a classical Rosenbrock reaction step for
  `ReactionDiffusion`, not an exponential integrator).
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
| Graph adjoint state + sensitivity | ADR-0115 | §42–§43 |
| Batched multi-channel (v0.9.1-beta, #10) | ADR-0184 | §43.4 |
| Reverse-mode AD (0.9.0-beta) | ADR-0156 | §51 |

Open issues tracked in the ROADMAP and ADRs: non-symmetric graph Krylov (Arnoldi),
consistent-mass Fréchet, 2-D/3-D ETD, exponential Rosenbrock, per-step learned
`N(u)` via native C fn-ptr (the only cross-language arbitrary-N path for `Etdrk4`).
