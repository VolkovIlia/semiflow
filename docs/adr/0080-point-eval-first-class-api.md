# ADR-0080 — `PointEval<F>` First-Class API (A6)

- **Status**: Accepted
- **Date**: 2026-05-27
- **Wave**: v4.0 Wave B (second Wave of the second BREAKING window; ships immediately after Wave A trait freeze). Independent of ADR-0079 (SemiflowComplex) and ADR-0081 (d-D shift); the PointEval trait surface and its 5 opt-in impls land before the d-D shift backend uses it (Backend E in math §31.2).
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0001 (contract-first), ADR-0003 (no_std + alloc), ADR-0025 (Generic-over-Float defaulting), ADR-0026 (`ChernoffFunction<F>` super-trait generic over `F`), ADR-0041 (`apply_into` + `ScratchPool` zero-alloc pattern — the bound-stack iterative respects the same zero-alloc discipline), ADR-0070 (v2.7 `TimedChernoffFunction<F>` opt-in super-trait — same opt-in pattern reused for PointEval; PointEval has NO default-bridged impl), ADR-0071 (v2.8 ManifoldChernoff — primary use case Backend C), ADR-0073 (v3.0 ApproximationSubspace<K, F> — sibling opt-in marker trait pattern), ADR-0074 (v3.0 ChernoffFunction trait cleanup — `Growth<F>` typed return preserved), ADR-0077 (v3.1 HypoellipticChernoff — primary use case Backend D), ADR-0081 (v4.0 AnisotropicShiftChernoffND — primary use case Backend E; this ADR's PointEval surface enables Backend E's pointwise eval for d ≥ 5).
- **Supersedes / amends**: v3.0 STUB `PointEval<F>` trait (which was a placeholder returning `Unsupported` from every method per ADR-0073 §"Out of scope" / §"Future work"). The v3.0 stub is PROMOTED to a first-class trait with bound-stack iterative pointwise evaluation per math.md §31.2 Algorithm 31.1.
- **Mathematical foundation**: math.md §31 (NORMATIVE library — `PointEval<F>` semantics + bound-stack iterative algorithm; CITATION Remizov 2025 *Vladikavkaz Math. J.* 27:4 "Remark 9" — the original observation that pointwise evaluation admits a bound-stack iterative algorithm avoiding the $O(q^n)$ tree; Folland 1975 *Ark. Mat.* §2.3 — graded-tangent-space fundamental solutions for step-2 Carnot (Backend D)). math §31 BUILDS on §24 (v2.8 manifold heat kernel localisation; Backend C) + §28 (v3.1 Hörmander hypoelliptic phase-space; Backend D) + §32 (v4.0 d-D anisotropic shift; Backend E).
- **Acceptance gates added**: G_POINTEVAL (RELEASE_BLOCKING — byte-identity at f64 between `kernel.eval_at(τ, src, x, n)` and `(apply_into^n src).sample(x)` on FIVE PointEval backends: DiffusionChernoff, ShiftChernoff1D, ManifoldChernoff (Sphere2), HypoellipticChernoff (KolmogorovPhaseSpace), AnisotropicShiftChernoffND (d=2)). Lives in `tests/point_eval_byte_identity.rs` new file, feature `slow-tests`.

## Context

The v0.1.0 → v3.1 library API consistently materialises the full Chernoff iterate $(F(\tau/n))^n f$ as a fully-discretised state. For users querying only single point values (e.g., Monte-Carlo importance sampling of the heat kernel, Bayesian inference at a single tangent-space point), this is wasteful: $O(N)$ storage + $O(N \cdot n)$ compute per query regardless of how few points are queried.

ADR-0073 §"Out of scope" / §"Future work" included a STUB `PointEval<F>` trait returning `Unsupported` from every method — placeholder for v4.0+ once concrete use cases drove the API design. Two use cases are now mature:

1. **v2.8 manifold Chernoff (ADR-0071)** — pointwise heat-kernel evaluation at tangent-space points on a Riemannian manifold. Bayesian inference (e.g., posterior sampling on $S^2$ for directional statistics) needs single-point eval.
2. **v3.1 Hörmander hypoelliptic (ADR-0077)** — pointwise fundamental-solution evaluation in phase space for Kolmogorov 1934 step-2 Carnot. Path-integral Monte Carlo for kinetic Fokker-Planck.

v4.0 ADR-0080 promotes the stub to a first-class trait with a bound-stack iterative algorithm (math §31.2 Algorithm 31.1) and ships FIVE opt-in impls (DiffusionChernoff, ShiftChernoff1D, ManifoldChernoff, HypoellipticChernoff, AnisotropicShiftChernoffND). The bound-stack iterative exploits the Markov property of the kernel representation $K(x, y; \tau)$ to reduce the naive $O(q^n)$ tree to $O(n \cdot q)$ per query.

The algorithm + the trait surface are co-designed: the trait method `eval_at(τ, src, x, n)` returns a SCALAR (not a state); the implementation per backend is the kernel-specific bound-stack iterator.

## Decision

Ship four additive public-surface items in v4.0 Wave B:

**Item 1 — `pub trait PointEval<F: SemiflowFloat>`** in `crates/semiflow-core/src/point_eval.rs` (NEW module, ~350 LoC target, default 500-LoC cap):

```rust
pub trait PointEval<F: SemiflowFloat>: ChernoffFunction<F> {
    /// Evaluate (F(τ))^n applied to source, sampled at point x.
    /// Bound-stack iterative — O(n · q) compute, O(q) auxiliary storage,
    /// where q is the per-step quadrature order (kernel-specific).
    ///
    /// `x` is the query point in the kernel's spatial domain. For 1D kernels,
    /// x.len() == 1; for d-D kernels, x.len() == D; for ManifoldChernoff,
    /// x.len() == manifold.dim().
    ///
    /// Byte-identity contract (G_POINTEVAL gate, math §31.3): the returned
    /// scalar MUST be byte-identical (f64::to_bits equality) to
    /// `(F.apply_into^n src).sample(x)`.
    ///
    /// Returns `Unsupported` if the kernel does not have a kernel-representation
    /// form (e.g., grid-only kernels like Strang2D, Diffusion4thChernoff).
    fn eval_at(
        &self, tau: F, src: &Self::S, x: &[F], n_steps: u32,
    ) -> Result<F, SemiflowError>;
}
```

The trait is generic over `F: SemiflowFloat = f64` (default per ADR-0025). NO default impl — the v2.7 `TimedChernoffFunction<F>` default-bridge pattern is REJECTED here for the same reason as ADR-0073: a default-bridged `eval_at` that returns `Unsupported` is a silent-false-witness footgun. Impls explicitly opt in by writing `impl<F: SemiflowFloat> PointEval<F> for X<F> { ... }` blocks; kernels without a kernel-representation form (Strang2D, Magnus*, Diffusion4thChernoff) DO NOT implement PointEval and queries against them return a compile error (PointEval is a super-trait of ChernoffFunction; not all ChernoffFunctions implement PointEval).

**Item 2 — Five v4.0 opt-in `PointEval<F>` impls** in `point_eval.rs` (file-layout per ADR-0073 — co-located with the trait, not in their respective kernel modules):

```rust
impl<F: SemiflowFloat> PointEval<F> for DiffusionChernoff<F> { /* Backend A, math §31.2 */ }
impl<F: SemiflowFloat> PointEval<F> for ShiftChernoff1D<F>    { /* Backend B, math §31.2 */ }
impl<M: BoundedGeometryManifold<F>, F: SemiflowFloat>
    PointEval<F> for ManifoldChernoff<M, F>                 { /* Backend C, math §31.2 */ }
impl<F: SemiflowFloat, const D: usize, const M: usize>
    PointEval<F> for HypoellipticChernoff<F, D, M>          { /* Backend D, math §31.2 */ }
impl<F: SemiflowFloat, const D: usize>
    PointEval<F> for AnisotropicShiftChernoffND<F, D>       { /* Backend E, math §31.2 */ }
```

Each per-backend impl implements the kernel-specific bound-stack iterative per math §31.2:

- **Backend A (DiffusionChernoff)**: 5-point Gauss-Hermite per step; aux state is 5 node values.
- **Backend B (ShiftChernoff1D)**: identical to Backend A with constant $a$.
- **Backend C (ManifoldChernoff)**: $(q_{\mathrm{tangent}})^{D_M}$-point tangent-space quadrature on $T_{x_0}M$; per-step parallel transport.
- **Backend D (HypoellipticChernoff)**: graded-tangent quadrature on $\mathbb{R}^D$; per-step Strang-Hörmander palindromic update.
- **Backend E (AnisotropicShiftChernoffND)**: tensor-product Gauss-Hermite on $\mathbb{R}^D$ (closed-form $D \le 4$); $D \ge 5$ falls back to `MonteCarlo` path.

The per-backend O(n · q) compute is the bound-stack iterative; the byte-identity contract (math §31.3 Proposition 31.1) is verified by G_POINTEVAL on all five backends.

**Item 3 — Module file layout** `crates/semiflow-core/src/point_eval.rs` (~350 LoC target, default 500-LoC cap, NO Override expansion). The 5 per-backend impl blocks are CO-LOCATED in the single module (NOT distributed across kernel files) for the same reason as ADR-0073 §"Decision" file-layout rationale: (a) the iterator logic is independent of the kernel's `apply_into`; (b) Wave B can add or remove PointEval impls without touching kernel files; (c) future PointEval impls (v4.1+ for new kernels) accumulate in one place.

**Item 4 — STUB removal**: the v3.0 `PointEval<F>` stub in `approximation.rs` (if any was placed there per ADR-0073 §"Future work") is REPLACED by the new module. If the v3.0 stub did not yet exist (i.e., the trait name was only mentioned in rustdoc), this is a NEW trait surface; either way, v4.0 ships the trait at first-class status.

## Rationale

- **Why a SUPER-TRAIT of ChernoffFunction<F>** (vs a standalone trait): the bound-stack iterative needs to dispatch back into the kernel's per-step operator action for the kernel-specific `kernel_step_backward` (math §31.2 Algorithm 31.1 step 2). Making PointEval a super-trait ensures every PointEval impl already has a ChernoffFunction impl available, so the iterator can compose without separate dispatch. Trait inheritance is the suckless tool.
- **Why NO default-bridged impl** (unlike ADR-0070 TimedChernoffFunction): per ADR-0073 §"Rationale" — silent-false-witness footgun. A default impl returning `Unsupported` would let callers query `kernel.eval_at(...)` on an impl that hasn't opted in and get `Unsupported` thinking the kernel doesn't support PointEval, when actually the impl just hasn't declared. Explicit opt-in matches the v3.0 ApproximationSubspace pattern.
- **Why byte-identity (not ≤ 4 ULP) for the gate G_POINTEVAL**: the bound-stack iterative computes EXACTLY the same floating-point reductions as the full-grid path SAMPLED at the query point — when implemented correctly. Any deviation indicates an implementation bug; byte-identity is the verification mode that catches it.
- **Why FIVE backends in v4.0** (not all 26+ ChernoffFunctions): the five are the use-case-driven set:
  - Backend A (DiffusionChernoff): the v0.3.0 baseline; foundational for HFT extensions.
  - Backend B (ShiftChernoff1D): the v0.1.0 simplest; foundational for path-integral Monte Carlo.
  - Backend C (ManifoldChernoff): the v2.8 manifold use case.
  - Backend D (HypoellipticChernoff): the v3.1 Hörmander use case.
  - Backend E (AnisotropicShiftChernoffND): the v4.0 ADR-0081 use case + future high-d MC fallback.
  Other v3.x kernels (Strang2D, Diffusion4thChernoff, Magnus*, NonSeparable2D, QuantumGraphHeatChernoff, etc.) are grid-only or lack the kernel-representation form; they DO NOT implement PointEval in v4.0. SchrödingerChernoffComplex is excluded because the Cayley map is not a kernel-representation form (it's a unitary rational approximation; pointwise eval defers to v4.1+).
- **Why `eval_at` takes `&[F]` for the query point** (not `&Self::S::Point` or similar): the simplest abstraction that works for all five backends. 1D backends use `x.len() == 1`; d-D backends use `x.len() == D`; manifold backends use `x.len() == manifold.dim()`. Per-backend impls validate the slice length at the start of `eval_at`. The alternative (a `Point<F, const D>` newtype) would add unnecessary const-generic complexity to the trait signature.
- **Why `n_steps: u32`** (not `usize` or `i32`): the Chernoff iterate count is non-negative and bounded (a billion-step iterate is unreasonable for any production use; u32::MAX = 4.3 billion is more than enough). u32 is also more compact in the trait surface (fewer bits in serialised representations like FFI). Mirrors the v2.0 `Evolver::new(_, n: usize)` pattern but at u32 for the trait method — minor consistency wart documented but accepted (the wart is documented in the rustdoc; not worth a v5.0 break to fix).
- **Why does the byte-identity gate use 5 backends** (not 3 or 10): the 5 backends span the full kernel-representation lattice — pointwise (Backend B), variable-coefficient (Backend A), curved (Backend C), graded (Backend D), high-dimensional (Backend E). Three would miss high-dim; ten would inflate test runtime for marginal coverage. Five is the suckless minimum.
- **Why NO `eval_at` for grid-only kernels** (Strang2D, Magnus*, Diffusion4thChernoff): these kernels are defined directly on the discretised grid via finite-difference stencils; they lack a kernel representation $K(x, y; \tau)$ — the bound-stack iterative is not applicable. Users querying grid-only kernels MUST materialise the full grid and call `.sample(x)`. Documented in math §31.5.
- **Why NO `eval_at_batch(x_queries: &[F])` in v4.0**: batch eval is an incremental performance optimisation orthogonal to the core correctness story. Defer to v4.1+ if benchmark evidence demands; the single-query form is the canonical surface.

## Alternatives considered

| Alternative | Why rejected |
|---|---|
| Add `eval_at` as a default method on `ChernoffFunction<F>` (forcing every impl to override) | 26+ existing impls would need explicit overrides; grid-only kernels (Strang2D, Magnus*) would override with `Unsupported`. Inflates every impl block by 5+ lines for grid-only kernels that can't use PointEval. Opt-in super-trait is the suckless choice. |
| Implement bound-stack iterative as a generic free function `point_eval<C, F>(C, tau, src, x, n) -> F` with `where C: ChernoffFunction<F>` | Forces every kernel-representation impl into the same generic implementation; loses per-kernel specialisation (e.g., ManifoldChernoff needs parallel-transport cache; HypoellipticChernoff needs graded-tangent quadrature). Per-kernel opt-in via trait method is the suckless choice. |
| Return `Result<Option<F>, SemiflowError>` from `eval_at` (Some on supported, None on unsupported) | Doubles the failure path; callers must check both Result + Option. Returning `Err(Unsupported)` for unsupported kernels is the standard Rust idiom. |
| Make PointEval a const-generic trait `PointEval<F, const D: usize>` with the dimension parameter on the trait | Forces every impl to declare a single D; manifolds have intrinsic dim, 1D kernels have D=1, 2D kernels have D=2 — this is just `Self::S::DIM` (Self::S knows its own dimension). Add the const-generic at trait level would be over-abstracted. The `&[F]` slice with per-impl length validation is simpler. |
| Ship the 5 backend impls in their respective kernel modules (each in its own file) | Scatters the bound-stack iterative pattern across 5 modules; harder to maintain consistency; harder to add a 6th backend without finding the right file. Co-location in `point_eval.rs` mirrors the v3.0 ADR-0073 file-layout choice. |
| Defer PointEval to v4.1+ (ship v4.0 with the v3.0 stub unchanged) | The v2.8 manifold and v3.1 Hörmander use cases are mature now; v4.0 is the right window to ship. Delaying loses momentum and forces a v4.1 BREAKING (since PointEval as super-trait is a trait-surface change). |
| Ship PointEval as a trait-object-friendly `dyn PointEval` interface (no generic associated types) | The per-backend `eval_at` impls use kernel-specific quadrature tables that don't fit the trait-object boxing model. Static dispatch is the suckless choice; dyn-friendliness is not a requirement for the use cases. |
| Add SchrödingerChernoffComplex as a 6th PointEval backend in v4.0 | The Cayley map is not a kernel-representation form; the bound-stack iterative doesn't apply naturally. Pointwise eval for unitary semigroups defers to v4.1+ pending a separate algorithm design. |
| Implement Backend E (AnisotropicShiftChernoffND) for arbitrary D via Monte Carlo from v4.0 release | The MC path is an OK fallback but the closed-form quadrature is correct for D ≤ 4 and tested via byte-identity. Shipping MC-only would lose the byte-identity property. v4.0 ships closed-form D ≤ 4 with MC fallback for D ≥ 5. |

## Consequences

- **Pre-existing call-sites compile unchanged.** Strictly additive surface; no existing trait or struct is modified. The v3.0 stub (if any) is REPLACED by the first-class trait; the replacement is BREAKING for any caller that depended on the `Unsupported` return — but no caller did (the stub was a placeholder per ADR-0073 §"Future work").
- **New module `crates/semiflow-core/src/point_eval.rs`** (~350 LoC target; default 500-LoC cap; NO Override expansion). The 5 per-backend impls + the trait + rustdoc fit comfortably under the cap.
- **New trait `PointEval<F>`** with 5 opt-in impls (DiffusionChernoff, ShiftChernoff1D, ManifoldChernoff, HypoellipticChernoff, AnisotropicShiftChernoffND).
- **Dependency count unchanged** at 3/3 (the trait + impls use only existing kernels + their existing dep on libm/num-traits).
- **Schema bumps**: shared with ADR-0079/0081/0082/0083/0084/0085 — `traits.yaml` 1.1.0 → **2.0.0 MAJOR**; `properties.yaml` 0.12.0 → **1.0.0 MAJOR**. math.md is append-only (§31 NEW).
- **New gate**: G_POINTEVAL (RELEASE_BLOCKING — 5 sub-tests across 5 backends; byte-identity at f64). Test file `tests/point_eval_byte_identity.rs` new file, feature `slow-tests`.
- **CITATIONs added to math.md §31**: Remizov 2025 *Vladikavkaz Math. J.* 27:4 "Remark 9" (the original bound-stack iterative observation); Folland 1975 *Ark. Mat.* §2.3 (Backend D graded-tangent fundamental solutions).
- **Performance characterisation**: the bound-stack iterative cost is $O(n \cdot q)$ where $q$ is the per-step quadrature order:
  - Backend A: $q = 5$; n = 100 → 500 FLOPs per query.
  - Backend B: $q = 5$; n = 100 → 500 FLOPs per query.
  - Backend C: $q = q_{\mathrm{tangent}}^{D_M} = 25$ for $D_M = 2$; n = 100 → 2500 FLOPs per query.
  - Backend D: $q = 25$ for $D = 2$; n = 100 → 2500 FLOPs per query.
  - Backend E: $q = 5^D$ tensor; for $D = 2$, 25 nodes; for $D = 5$, 3125 nodes; n = 100 at $D = 5$ → 312500 FLOPs per query.
  For Monte-Carlo-style usage with $10^6$ queries, the savings vs full-grid materialisation are 4-6 orders of magnitude.
- **Migration note**: callers using `(F.apply_chernoff^n f0).sample(x0)` for pointwise queries CAN MIGRATE to `F.eval_at(τ, &f0, &[x0], n)` for the 5 supported backends — same return value (byte-identical), drastically lower cost.

## Migration

End-user impact is **opt-in additive**. The v3.0 stub PointEval (if any was placed in `approximation.rs` per ADR-0073 §"Future work") is REPLACED by the v4.0 first-class trait; callers that depended on the `Unsupported` return path (none currently exist; the stub was a placeholder) MUST migrate to the new trait surface.

New v4.0 users wanting pointwise eval:

```rust
// v3.x baseline (still works; full-grid materialisation):
let evolver = Evolver::new(kernel.clone(), n_steps)?;
let final_state = evolver.evolve(t_final, &initial_condition)?;
let value_at_x = final_state.sample(&x_query);    // O(N · n) compute, O(N) storage

// v4.0 NEW (bound-stack iterative — for the 5 supported backends):
let value_at_x = kernel.eval_at(
    t_final / (n_steps as f64),
    &initial_condition,
    &[x_query],
    n_steps,
)?;
// Byte-identical to the v3.x baseline; O(n · q) compute, O(q) storage.
// For DiffusionChernoff at n=100, q=5: ~10000× cheaper than full-grid.
```

Worked example with Monte-Carlo importance sampling in `docs/migration/v3-to-v4.md` §3 (Wave G).

## Cross-references

- ADR-0001 — contract-first; this ADR adds new contracts before any Rust impl ships.
- ADR-0003 — no_std + alloc; the trait + impls use only stdlib + existing deps.
- ADR-0025 — Generic-over-Float `F = f64` defaulting; reused for `PointEval<F>` with `F = f64` default.
- ADR-0026 — `ChernoffFunction<F>` super-trait; `PointEval<F>` is a super-trait of it.
- ADR-0041 — `apply_into` + `ScratchPool`; the bound-stack iterative respects the same zero-alloc discipline.
- ADR-0070 — v2.7 `TimedChernoffFunction<F>` opt-in super-trait pattern; PointEval follows the same opt-in pattern but with NO default-bridge.
- ADR-0071 — v2.8 ManifoldChernoff; Backend C primary use case.
- ADR-0073 — v3.0 `ApproximationSubspace<K, F>` opt-in marker trait — the file-layout precedent for co-locating opt-in impls in a single module is reused here.
- ADR-0074 — v3.0 ChernoffFunction trait cleanup; preserved verbatim.
- ADR-0077 — v3.1 HypoellipticChernoff; Backend D primary use case.
- ADR-0081 — v4.0 AnisotropicShiftChernoffND (sibling ADR in v4.0); Backend E primary use case; the PointEval surface enables Backend E's pointwise eval for $D \ge 5$ via the MC fallback path.
- `~/.claude/plans/roadmap-reflective-biscuit.md` §v4.0 — release-level roadmap (PointEval first-class promotion).
- math.md §31 (NEW v4.0) — PointEval bound-stack iterative algorithm + per-backend specifications + G_POINTEVAL gate spec.
- `.dev-docs/constitution.md` v1.8.0 (NEW v4.0) — MAJOR re-evaluation; no Override modification for `point_eval.rs` (under default 500-LoC cap).
- `docs/migration/v3-to-v4.md` §3 — PointEval pointwise eval worked example (Wave G fills).

## Amendments

(none at acceptance time)
