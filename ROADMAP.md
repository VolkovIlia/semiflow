# SemiFlow — Roadmap

> Mission: a memory-frugal, mathematically universal computation library
> implementing Chernoff approximations of operator semigroups (Theorem 6,
> Remizov 2025) and adjacent approaches (operator splitting, BCH
> tau²-corrections, truncated-exp power series, adaptive integrators).
> Wallclock competitiveness vs adaptive/spectral solvers is NOT a design
> goal; see `docs/perf-commitment-v1_0_0.md` and the iter-8 benchmark
> campaign (HEAD b923777, 45 families) for the honest performance picture.

Math fidelity is tracked per-release in `docs/audit-findings-v{N}.md`.

---

## Post-0.9.0-beta feature wave — DONE (ADRs 0175–0181)

Seven issues closed against the 0.9.0-beta audit, plus the rough-Heston pricer
example. All are additive; no breaking changes to the 0.9.0-beta public surface.

**Issue #1 — Multi-parameter reverse-mode AD (ADR-0177):** `ReverseChernoff` now
accepts a `RegionMap` (per-region θ partition) via `with_region_map`, enabling
K>1 per-region parameter sensitivity. Math §51.10.

**Issue #2 — Variable-coefficient TT carrier (ADR-0178):** `VarCoefTt<F>` supports
separable diagonal `a_j(x_j)` on a `TtState`. Out-of-class inputs (non-separable)
return `SemiflowError::VarCoefOutOfClass` at construction. Math §52.10.

**Issue #3 — Gridless variance / MSE diagnostic (ADR — math §38.12):**
`MeasureState` gains `first_moment`, `variance`, and `variance_per_axis` — additive
diagnostic methods; no behavior change to `GridlessChernoff` evolution.

**Issue #4 — S³ Python bindings:** `TtState`, `TtEvolver`, `TtCoupledEvolver`,
`MeasureState`, `GridlessEvolver`, and `VarCoefTtEvolver` are now importable from
the `semiflow` Python package (`from semiflow import ...`). Gap in ADR-0171
S³-carrier PyO3 surface closed.

**Issue #5 — Native `f32` + f32 SIMD (ADR-0175):** All leaf 1D kernels now
implement `ChernoffFunction<f32>`. A dedicated f32x8 AVX2 / f32x4 NEON kernel
(no FMA dependency) provides accelerated `f32` hot paths alongside the existing
`f64` SIMD paths.

**Issue #6 — Order-2 Dirichlet boundary (ADR-0176):** `DirichletHeat2ndChernoff<C, R, F>`
using the odd-image method (`BoundaryPolicy::OddReflect`) delivers order 2 in the
continuation region. The order-2 sibling of `ReflectedHeatChernoff` (Neumann).
Math §21.9.

**Full operator-zoo binding parity (ADRs 0028/0179):** See the updated Binding-parity
wave section below for the complete list of newly-bound engines (`DiffusionExpmvChernoff`,
`DriftReactionZeta4Chernoff`, `Killing2ndChernoff`, `MatrixDiffusion2D/3D`,
`DirichletHeat2ndChernoff`) and the C/WASM parity closures for graph/obstacle
introspection (`SmfLaplacian`, `SmfGraphTraj`, `SmfObstacleGamma`, `SmfObstacleND2`).

**`GraphAdjoint` pre-sampled time-grid sampler (ADR-0180):**
`MagnusGraphHeatChernoff::from_presampled` / `smf_graph_adjoint_new_presampled` /
`GraphAdjointPresampled` (PyO3 + WASM) — supply a time-dependent Laplacian as a
pre-sampled sequence (no live callback). GL4-aware: expects 2·n_steps Laplacian
samples.

**Issue #9 — Production-grade rough-Heston pricer (ADR-0181):**
`examples/rough_heston_pricer.rs` — oracle-validated risk-neutral pricer.
`--rate` / `--price` flags enable risk-neutral drift and discounting. Gates:
`G_ROUGH_HESTON_MC_PARITY` (RELEASE-BLOCKING, MC oracle) and
`A_ROUGH_HESTON_MODEL_BIAS` (ADVISORY, ~1–5% O(H)-bias of Markov approximation
vs true rough-Heston; this is the expected model-approximation error, not a bug).
`--rate 0.0` recovers the original latency demonstrator.

---

## Binding-parity wave — DONE (follows 0.9.0-beta; gaps closed post-0.9.0-beta)

Full operator-zoo parity between `semiflow-core` and all three binding crates
(`semiflow-ffi`, `semiflow-py`, `semiflow-wasm`) achieved as a follow-up to the
0.9.0-beta public release. Binding surfaces are a curated mirror — internal
composition types (`AxisLift`, `StrangSplit`) are intentionally not exposed.

**Done (initial wave):**
- C ABI (`semiflow-ffi`) and WASM (`semiflow-wasm`) expose the full engine
  surface: higher-order ζ-ladder, 2D/3D tensor, non-separable anisotropic,
  all boundary-condition variants, Schrödinger (real + complex), matrix
  diffusion, Howland/subordinated, manifold, hypoelliptic, graph family,
  SmolyakD6, Adjoint, AdaptivePI, ComplexTripleJump, PointEval.
- `s3-poc` cargo feature retired; the six S³ POC evolvers are in the default
  core API (ADR-0169 boundary-as-type wrappers).
- S³ carrier C-ABI surface (ADR-0171): `TtEvolver/TtState`,
  `TtCoupledEvolver`, `GridlessEvolver/MeasureState`.
- WASM `full` cargo feature added: lite default ≈ 768 KB raw; `--features full`
  ≈ 1.4 MB raw.

**Done (gap-closure wave — ADRs 0175–0181):**
- Laplacian introspection now in C (`SmfLaplacian` with `row_ptr`/`col_idx`/
  `vals`/`to_dense`) and WASM; `SmfGraphTraj` trajectory output in C/WASM.
- Obstacle surfaces: `SmfObstacleGamma` and `SmfObstacleND2` in C/WASM.
- `GraphAdjoint` pre-sampled time-grid path (`smf_graph_adjoint_new_presampled`
  / `GraphAdjointPresampled`) available in all three binding surfaces (ADR-0180).
  The live-callback constructor remains PyO3-only.
- Newly-bound engines across C/WASM/PyO3: `DiffusionExpmvChernoff` (`smf_expmv1d_*`),
  `DriftReactionZeta4Chernoff` (`smf_drift_reaction_zeta4_*`),
  `Killing2ndChernoff` (`smf_killing2nd_*`), `MatrixDiffusion2D/3D` (WASM), and
  `DirichletHeat2ndChernoff` (`DirichletHeat2nd1D` in WASM/Python).
- S³ Python bindings (ADR-0171 gap closed): `TtState`, `TtEvolver`,
  `TtCoupledEvolver`, `MeasureState`, `GridlessEvolver`, `VarCoefTtEvolver`
  importable from `semiflow` (Python package `semiflow-pde`).
- Variable coefficients cross the binding boundary as pre-sampled arrays
  (not closures) per ADR-0028/0179.

**One remaining PyO3-only deferral (named, not silently omitted):**
- `GraphAdjoint` time-dependent Laplacian constructor accepting a live Python
  callback — use the pre-sampled path (`GraphAdjointPresampled`) as the
  cross-surface alternative.

Cross-refs: ADR-0028 (binding split), ADR-0171 (S³ carrier C-ABI), ADR-0175–0181.

---

## v9.2.0 — S³ honest-scope public API (ADR-0169) — experiment/triz-s3-curse-escape

Five S³ POC evolvers promoted to a curated public surface behind the new non-default
`s3-poc` cargo feature. Three honesty layers enforced: (1) boundary-as-type wrapper
constructors reject out-of-class inputs at compile time; (2) all six tokens are
`#[cfg(feature = "s3-poc")]`; (3) every public S³ type carries a normative
`## Proven boundary` rustdoc stanza.

**Six public tokens:** `S3DriftSpectralEvolver`, `S3DenseCouplingEvolver`,
`S3VarCoefEvolver`, `S3NonSepVarCoefEvolver`, `S3BurgersColeHopf`,
`S3ReactionDiffusion`. **Container types promoted to `pub`:** `AxisCoef`, `CpTerm`,
`CpCoef`, `CoefRole`, `Reaction`.

**Schema versions:** `semiflow-core.traits.yaml` → 4.14.0; `semiflow-core.properties.yaml`
→ 4.15.0. `Cargo.toml` workspace version → 9.2.0.

See `docs/adr/0169-s3-honest-scope-public-api-promotion.md` and math.md §53 (umbrella
+ §53.1–§53.5 cross-refs, added this release).

---

## Technical-debt elimination wave — feat/v9.1.0-genuine-scurve — 2026-06-12

18 commits (c567813→43c4551) on the `feat/v9.1.0-genuine-scurve` branch above v9.1.0.
No new mathematics; no public API changes; all floating-point outputs byte-identical to
the v9.1.0 baseline (5069d85). Three red quality gates restored to green, constitution
promoted to v6.0.0, and three pre-existing slow-test failures corrected.

**Gates restored:** (1) `clippy -D warnings` 1768 findings → 0; (2) `check-lints`
(suckless ≤500/50 limits) 105 over-budget items → 0; (3) `check-unsafe-scope` 110
violations → 0. All three were silently failing because GitHub Actions is paused for
billing; local validation is the primary gate.

**Constitution v6.0.0:** Override #1 (grandfather array for 13 line-limit Cohorts) RETIRED.
Guardrail #1 is now ENFORCED unconditionally. 38 oversized files split into submodules;
~52 oversized functions shrunk via verbatim helper extraction. Public API paths unchanged
via `pub use` re-exports.

**Pre-existing gate fixes (slow-tests / flagship, all confirmed bit-identical at 5069d85):**
(a) `binding_resolvent_jump_nd_parity` 3D probe M3 6→8 (Gauss-Laguerre needs more nodes in
3D; tolerance unchanged). (b) `chernoff1d_parallel_bit_equal` truncated-exp OOB panic
(root cause: `apply_into` used operator grid for boundary dispatch instead of data grid;
commit d7443fe). (c) `g3_6_2d_flagship_slope_and_runtime_gate` slope gate recalibrated to
floor-safe asymptotic basket {191, 251, 331} (slope -6.0793, window [-6.30, -5.85]);
SepticHermite (ADR-0109) lowered the f64 floor and saturated the old basket — mirrors
ADR-0120 1D precedent; ADR-0163.

---

## v9.1.0 — Genuine no-solver exact coupled-TT evolver (spectral pair factor) — SHIPPED 2026-06-11

MINOR. Branch `feat/v9.1.0-genuine-scurve`. ADR-0162. Math §52.9 (Round-4 NORMATIVE).
Coupled `CoupledTtChernoff`: band-split shift removes the diagonal convergence floor (O(1) quantization
→ O(τ²)); spectral (FFT-diagonal) pair factor `exp(τ·symbol)⊙` recovers the no-solver property
(Theorem-6 R2) for the coupling leg. Exact for the constant-coefficient correlated-Gaussian class
(zero Strang remainder). Gates `G_TT_COUPLED_EXACT`, `g_tt_coupled`, `g_tt_band_converge` ran
locally and PASS. Strictly additive public surface. Constitution unchanged (all new files ≤500 LoC:
`tt_spectral.rs` 358, `tt_coupled_pair.rs` 441, `tt_dense_expm.rs` 265 — no carve-out needed).

---

## v9.0.0 — Reverse-mode AD + TT-Chernoff (third S-curve) — SHIPPED 2026-06-10

SIXTH MAJOR. Branch `feat/v9.0.0-planning`. ADRs 0154–0159. Math §50 (Shift C particle — NARROW) + §51 (Shift B — HEADLINE) + §52 (Shift C TT — co-HEADLINE).
Source: `.dev-docs/research/v9-paradigm-shifts-research.md` (graded C > B > A).

Two S-curves are saturated: **kernel-accuracy** (ζ²→ζ⁸, manifolds, hypoelliptic, resolvent,
complex, matrix, soft-killing) closed at v7.0.0; **product-structure** (`(F(τ))ⁿ` differentiated /
jump-amortized / boundary-regularized) closed at v8.0.0. v9.0.0 opens the **third axis**: the two
unused *structural* resources of Theorem-6 formula (6)
`S(τ)f = ¼f(x+2√(aτ)) + ¼f(x−2√(aτ)) + ½f(x+2bτ) + τc·f`. **(R1)** the formula IS a Markov
transition kernel (proven on measures, §38) ⇒ gridless high-dimensional Chernoff. **(R2)** the
formula has NO linear solver (pure local shifts + scalar multiplies) ⇒ a reverse-mode-differentiable
layer with an algebraically-exact transpose, and a hardware-native (GPU) spike. Each direction
resolves a named TRIZ contradiction (АП→ТП→ФП→ИКР→решение), not a compromise.

**Shift C RESOLVED (2026-06-10):** `TtChernoff` (ADR-0159, math §52) escapes the exponential curse for
the linear diagonal-A (Gaussian) diffusion class — deterministically, `no_std`, byte-reproducibly —
via the tensor-train state carrier. Rank-1 = Strang⊗ exactly. Gate `G_TT_CHERNOFF_DIMSCALING`
PRE-REGISTERED. Honest scope: diagonal-A Gaussian class only; narrow novelty: Chernoff step-truncation
TT integrator (method class: Rodgers–Venturi; unclaimed: the Chernoff/Theorem-6 instantiation +
`no_std`/determinism/rank-1-exact envelope). See ADR-0155 Amendment 2, ADR-0159, math §52.

**Shift C full arc:** `GridlessChernoff` particle form — d=2 validated, spatial-merge INTRINSIC LIMIT,
variance thesis NOT confirmed (ADR-0155 Amendment 1). Root cause: deterministic branching is a
deterministic quadrature; reducing bias in d dimensions in a particle representation is `O(m^d)`. The
curse is in the CARRIER, not the evolver. `TtChernoff` TT form — curse escaped via the TT carrier for
the diagonal-A Gaussian class (ADR-0155 Amendment 2, ADR-0159, §52). Both forms ship and are normative.

**Shift B (`ReverseChernoff`) remains the v9.0.0 co-headline** (ADR-0156, §51): reverse-mode AD via
binomial checkpointing, completing the differentiability axis. Shift C TT is the other co-headline.

### Planned scope

**SHIP-TRACK (release-blocking):**

- **[SHIP-NARROW — ADR-0155 Amendment 1, d=2 validated, GATES MEASURED]** `GridlessChernoff` —
  bounded-d (d≤~10), linear-coefficient, deterministic-branching particle evolver. `T_GRIDLESS_*`
  sympy oracle PASS 3/3 (commit `f633cde`). `G_GRIDLESS_DIM_SCALING` MEASURED: d=2 PASS (err
  1.197e-3); d≥4 INTRINSIC LIMIT (spatial-merge curse re-enters reduction grid; §50.7).
  `G_GRIDLESS_VARIANCE` MEASURED: NO-GO (1.417× MSE ratio at d=2, fair CRN comparison, below ≥2×
  gate). Ships as the d=2 validated particle primitive and the documented negative result.

- **[SHIFT C RESOLUTION — ADR-0159, §52, PRE-REGISTERED GATE]** `TtChernoff` — tensor-train
  Chernoff for the linear diagonal-A (Gaussian) diffusion class. State held as a low-rank TT
  (`TtState`, `TtCore`); per-axis Chernoff shift is an O(1)-rank TT-operator (Kazeev–Khoromskij
  Grade A; §52.2); one deterministic TT-rounding per step (Jacobi SVD, no LAPACK, ~150 LoC,
  `tt_core.rs`). Gaussian rank algebraically capped at r ≤ d/2 (Rohrbach–Dolgov–Grasedyck–Scheichl
  2022; §52.4) → storage `O(d³·n)`, polynomial in d, exponential curse **escaped**. Rank-1 reduces
  exactly to Strang⊗/AxisLift (§52.3, §10.3 Theorem 7). No new dep (Override #1 preserved).
  Byte-reproducible (deterministic Jacobi SVD). Gate `G_TT_CHERNOFF_DIMSCALING` (PRE-REGISTERED,
  `slow-tests`; d ∈ {4,6,8,10}, accuracy < 5e-3, rank polynomial in d in both Regime L and Regime H,
  byte-reproducible). Honest scope: diagonal-A Gaussian class; off-diagonal/variable/nonlinear =
  research-track. Narrow novelty (Grade B): Chernoff-product step-truncation TT instantiation +
  `no_std`/determinism/rank-1-exact envelope; step-truncation TT as a class = Rodgers–Venturi.

- **[CO-HEADLINE — ADR-0156, SHIP-TRACK, low-risk]** `ReverseChernoff` — reverse-mode AD over
  `(F(τ))ⁿ` via **binomial checkpointing** (NOT a tape; stays `no_std+alloc`, no new dep), completing
  the v8.0.0 differentiability axis (§42 transpose-exactness + §43 adjoint-state + §46 forward
  dual-AD already shipped). Gate `G_REVERSE_AD_GRADIENT` (FD <1e-9 rel + 0-ULP cross-mode vs
  §46 for K=1) + `G_REVERSE_AD_CHECKPOINT` (peak memory `O(√n)`). `T_REVERSE_TRANSPOSE`
  PASS 2/2. NARROW: transpose-exactness scoped to the linear/Magnus family.

**RESEARCH-TRACK (NOT release-blocking):**

- **[ADR-0158 PROPOSED, scope narrowed by ADR-0159]** Path-space RQMC functional estimation —
  residual scope: dense-correlation / non-Gaussian regime where TT-rank is not algebraically capped.
  Sobol + Brownian-bridge over n·d path dimensions, cost O(P·n·d) dimension-free. Grade C.
- High-d (d>10) nonlinear / variable-coefficient gridless evolution (TT rank not capped for these).
- General `4ⁿ`-reduction theory (sharp `R_P` error bound uniform in `n` and `d`).
- Variance-vs-QMC sharp comparison for the deterministic ¼,¼,½ particle scheme (graded C).
- Prior-art re-confirmation (no Chernoff-on-particles paper found; re-fetch Festschrift arXiv:2508.18650).

**SPIKE-ONLY (feature-gated, advisory):**

- **[PLANNED — ADR-0157, SPIKE, NON-GOAL-for-core]** `remizov-gpu` — separate crate,
  `--features gpu`, `wgpu`-only stencil-shader backend (R2 locality → compute shader). Advisory gate
  `G_GPU_PARITY` (end-state <1e-10, parity explicitly **WAIVED** — GPU `f64` is not bit-identical to
  AVX2/NEON; ≥5× throughput at N≥512 to justify the dep). **Zero new mathematics ⇒ no math.md
  section.** WITHDRAW-on-dep/size-budget-breach. The `no_std` `semiflow-core` gains NO GPU dependency.

### Sequencing (all ship-track phases COMPLETE)

1. **Phase 1 (math + oracle, architect-led, no code) — COMPLETE** — math §50 / §51 + `T_GRIDLESS_*`
   and `T_REVERSE_TRANSPOSE` sympy oracles. Commit 861eef0. §50.6 go/no-go fired during Phase 2.
2. **Phase 2 (ship-track C particle — COMPLETE, NARROW)** — bounded-d particle evolver implemented
   and gated. Commits f633cde, 0929907, 4490b89, 1201e20. `G_GRIDLESS_DIM_SCALING` d=2 PASS (err
   1.197e-3); `G_GRIDLESS_VARIANCE` NO-GO (1.417×, pre-registered outcome). Ships NARROW.
3. **Phase 2.5 (Shift C RESOLUTION — TT carrier — COMPLETE)** — math §52, ADR-0159, `TtChernoff` +
   `TtCore` implemented. Commits f7e0c16, cff7d86. `G_TT_CHERNOFF_DIMSCALING` PRE-REGISTERED
   (`slow-tests`); `g_gridless_ttrank` PASS; machine accuracy ~1e-14; byte-reproducible.
4. **Phase 3 (ship-track B, HEADLINE — COMPLETE)** — `ReverseChernoff` reverse-mode via binomial
   checkpointing. Commits 3a4abaa, 778b854. All gates PASS (FD 8.09e-12, 0-ULP cross-mode,
   checkpoint slope 0.39, binding parity 0-ULP).
5. **Phase 4 (spike A — DEFERRED)** — `remizov-gpu` `wgpu` proof-of-concept deferred; advisory gate
   `G_GPU_PARITY` not yet built. ADR-0157 status DEFERRED.

### Explicit non-goals (moat protection — carried from research §5.4 + ADR-0154)

- **NOT** a general-purpose PDE framework (anti-direction X3). Everything stays an iterate of `(F(τ))ⁿ`.
- **NOT** a trained / data-driven solver. SemiFlow *computes* the operator exactly and
  bit-reproducibly; it does not *learn* it (the deep-BSDE / FNO lane is a deliberate differentiator).
- **NO** GPU stack in the `no_std` core. Any GPU work is a separate, feature-gated, optional crate;
  the dependency-count and binary-size guardrails are inviolate (Override #1).
- **NOT** an analog / optical Fourier-domain port — structurally incompatible with the solver-free
  shift kernel (it re-introduces the FFT that R2 avoids).
- **NO** "golden-middle" compromise gates — each `G_*` is a sharp threshold (slope / variance ratio / ULP).

### Schema / constitution (SHIPPED)

- Override #1 (Minimalism RELAXED, ≤3 direct deps in `semiflow-core`) RE-AFFIRMED. All five v9.0.0
  modules are <500 LoC; no new Cohort was needed (second consecutive S-curve with this outcome).
  In-tree Jacobi SVD (~150 LoC, `tt_core.rs`) satisfies the dep budget; zero new external runtime deps.
- Override #2 (MCP WAIVED) RE-AFFIRMED — `TtChernoff` and `ReverseChernoff` are synchronous
  compile-time API additions; Shift A GPU deferred, not runtime-bearing.
- Constitution bumped to **v5.0.0 MAJOR** (2026-06-10). Next re-evaluation v10.0.0.

---

## v8.3.0 — TIER-3 binding wave MINOR — SHIPPED 2026-06-09

ADDITIVE MINOR. No breaking changes. ADR-0153.

Delivers TIER-3 FFI / PyO3 / WASM bindings for the three v8.2.0 kernels:
`DynamicWentzellChernoff` (pre-sampled γ-schedule ABI + `GammaFamily` sugar),
`ResolventJump2D/3D` (NORMATIVE Fortran-order ND layout contract, fixing v8.1 C-vs-F-order
class of bug), and `ObstacleGamma` / `ObstacleND` (two-output `inactive_gamma` with
genuine bool refusal mask; PyO3-first, FFI/WASM deferred). Contracts: traits 4.12→4.13,
properties 4.13→4.14 (3 new `RELEASE_BLOCKING` parity gates).

---

## v8.2.0 — Wave-2 math wave MINOR — SHIPPED 2026-06-08

ADDITIVE MINOR. No breaking changes. ADRs 0148–0152.

Delivers four Wave-2 math items deferred from v8.1.0: dynamic Wentzell/Robin BC,
ResolventJump 2D/3D, KilledDirichlet variable-coefficient theorem upgrade, and
obstacle Γ + D≥2 forward evolution.

### Shipped

- **C-9 dynamic Wentzell/Robin BC (ADR-0151)** — `DynamicWentzellChernoff` implicit-Cayley
  kernel resolves the ADR-0098 Am3 "deferred indefinitely". Stephan 2023 instability is
  explicit-only; the implicit Cayley boundary step `K_CN=(I−τC/2)⁻¹(I+τC/2)` (closed-form
  2×2) gives ρ≤1 for time-dependent γ(t) via the Howland §23 lift. math.md §49.
  Gates: `G_WENTZELL_STABLE` (ρ=0.9998), `G_WENTZELL_ORDER` (slope −1.81), `T_WENTZELL`.

- **B-5 ResolventJump 2D/3D (ADR-0148)** — `ResolventJumpChernoff2D/3D` parabolic LHP
  backend via banded complex LU over grid2d/grid3d (`resolvent_jump_nd.rs`). math.md §47.8.
  Gates: `G_RESOLVENT_JUMP_2D_ORDER` (+9.81), `G_RESOLVENT_JUMP_3D_ORDER` (+9.75),
  `T_RESOLVENT_JUMP_ND`. NARROW self-adjoint/sectorial preserved.

- **B-6 KilledDirichlet variable-coefficient order-2 THEOREM (ADR-0149)** — upgraded
  math.md §44.ter from empirical to a NORMATIVE theorem (the CN Cayley order-2 is the
  (1,1)-Padé matrix identity, true for any fixed G; variable a(x) only changes G's entries).
  Added `T_KILLED_GEN_CN` sub-check E (non-origin variable-a jet). No kernel change.

- **B-7 obstacle Γ + D≥2 (ADR-0150/0152)** — inactive-set second-order Greek
  `apply_inactive_gamma_into` (`obstacle_gamma.rs`): Γ on the open continuation set
  (O(Δx²)), refused at the contact line via a companion bool mask (C¹-not-C², no global
  C² Greek). `ObstacleChernoffND` D≥2 forward evolution (`obstacle_nd.rs`). math.md
  §44.5.bis/.ter. Gates: `G_OBSTACLE_GAMMA` (−2.00), `G_OBSTACLE_SLOPE_2D` (−1.55),
  `T_OBSTACLE_GAMMA`.

### Schema

- `traits.yaml` 4.9.0 → 4.12.0 (additive MINOR — new kernel/gate stanzas)
- `properties.yaml` 4.10.0 → 4.13.0 (additive MINOR — new kernel/gate stanzas)

### Honest-defers → v8.x

- B-5 hyperbolic contour (needs adaptive sector-fit / rational-Krylov — at ε=1e-3 the
  spectrum is on the imaginary axis, fundamental Bromwich limit)
- B-7 ND active-set adjoint + free-surface Γ (D≥2 adjoint/Γ)
- TIER-3 FFI/PyO3/WASM bindings of the new Wave-2 kernels (Wentzell, resolvent-ND,
  obstacle-Γ/ND)
- Still OPEN: Carnot general-k convergence theorem (escalated, ADR-0145)

---

## v8.1.0 — Debt-closure MINOR — SHIPPED 2026-06-08

ADDITIVE MINOR. No breaking changes. ADRs 0138–0147.

Closes deferred features from v8.0.0 (generic-sampler honest-defer, TIER-3 bindings,
chunked GIL-yield), adds three Wave-3 research register entries, and sweeps all 21
suckless file-size violations introduced by v8.0.0 module growth.

### Shipped

#### Wave 1 — deferred-feature closure

- **[DONE — ADR-0139]** `OctonicHermite` + `ChebyshevSpectralWithBC` samplers
  genericised over `F: SemiflowFloat` — closes the ADR-0133 Amendment-1 honest-defer.
  The AD path now covers the full interpolation ladder (SepticHermite + OctonicHermite +
  ChebyshevSpectralWithBC). f64 path byte-untouched.
  Gates: `G1` byte-identity PASS; `G2` Dual-AD gradient |diff| ~1e-14 PASS.

- **[DONE — ADR-0138]** TIER-3 bindings — four kernel/binding pairs:
  `ResolventJumpChernoff` (F2) across FFI + PyO3 + WASM;
  `AdjointFokkerPlanckChernoff` (C2) across FFI + PyO3 + WASM;
  `SmolyakGridND-D6` (C1) via PyO3;
  `ComplexTripleJumpChernoff` (F4) via PyO3.
  All 0-ULP cross-binding parity (`G_BINDING_*_PARITY`).
  No Complex or NARROW types leak across any ABI.

- **[DONE — ADR-0140]** `expmv` 1-norm-estimator tuning evaluated — NO-GO.
  Banded divergence-form operator has spectral radius ≈ 1-norm; exact-norm yields 0%
  speedup; Higham–Tisseur estimation yields 12.5% with under-estimation risk.
  Conservative bound kept. Accuracy gate unchanged at 7.45e-13. ADR closes the
  investigation honestly.

- **[DONE — ADR-0141]** PyO3 `Heat1D.evolve_chunked` cooperative GIL-yield — chunked
  `evolve` releasing the GIL per chunk with a progress callback and cancellation token.
  0-ULP vs synchronous path. Not a full async runtime; Trio/asyncio integration deferred.

#### Wave-3 research register (ADRs 0145–0147)

- **[ADR-0145 — GO partial close]** C-8 Carnot step-k order≥3: order-4 constructive on
  Engel step-3 + filiform-N5 step-4 (step-independent scalar order conditions). General-k
  convergence theorem stays OPEN / escalated. Also fixed a 2e-7 γ⋆ digit-transposition
  typo in the v8.0.0 math contract (Rust source was always correct; comment-only fix).

- **[ADR-0146 — GO]** C-9 dynamic Wentzell/Robin BC: Stephan 2023 instability is
  explicit-only; implicit-Cayley boundary step gives ρ≤1 for time-dependent γ(t)
  (von-Neumann verified). Supersedes ADR-0098 Amendment-3 indefinite defer. Kernel
  implementation scheduled for v8.2.0.

- **[ADR-0147 — OUT-OF-SCOPE close]** C-10 holomorphic line-bundle O(k)→CP¹: homogeneous
  bundle Laplacian factorizes into shipped scalar heat × closed-form reweight × U(1) phase
  (Chernoff earns nothing over scalar). Math on record. Reopenable for variable connections.

#### Hygiene

- Cleared all 21 suckless file-size violations in `semiflow-py` / `semiflow-core` via pure
  module splits. py-smoke 617/617 PASS; check-lints 0 errors. Zero behavior change.

### Schema

- `traits.yaml` 4.5.0 → 4.9.0 (additive MINOR — new binding stanzas)
- `properties.yaml` 4.7.0 → 4.10.0 (additive MINOR — new gate entries)

### Honest-defers → v8.2.0

- B-5 F2 2D/3D LHP backends + hyperbolic-contour variant
- B-6 F3 variable-coefficient sharp-order theorem
- B-7 obstacle Γ at free boundary + D≥2
- C-9 dynamic-Wentzell Cayley kernel implementation (math approved in v8.1.0; code in v8.2.0)
- Still OPEN: Carnot general-k convergence theorem (escalated, no timeline)

---

## v8.0.0 — Differentiable Chernoff (second S-curve) — SHIPPED 2026-06-08

FIFTH MAJOR. 16 commits on `feat/v8.0.0-planning`. ADRs 0132–0137.

The kernel-accuracy S-curve — ζ-ladder through ζ⁸, manifolds, hypoelliptic, resolvent,
complex, matrix, soft-killing — is saturated as of v7.0.0. v8.0.0 opens the next axis:
extracting value from the Chernoff **product structure** `(F(τ))ⁿ` itself via
differentiability, spectral-domain time-jump amortization, Cayley-map hard walls, and
complex-time hypoelliptic closure. Each direction resolves a named TRIZ contradiction
(АП→ТП→ФП→ИКР→решение), not a compromise.

### Shipped

- **[DONE — ADR-0133 + Amdt 1]** `Dual<F>: SemiflowFloat` forward-mode AD — exact Δ/Γ
  Greeks through all generic kernels at zero allocation. `SepticHermite` (default grid)
  genericised over `F: SemiflowFloat` (Amdt 1); `OctonicHermite` / `ChebyshevSpectralWithBC`
  remain `f64`-only (honest-defer to v8.x). `T_DUAL` 4/4. `G_DUAL_AD_GRADIENT` PASS;
  `G_DUAL_ZERO_ALLOC` PASS.
- **[DONE — ADR-0134, NARROW]** `ResolventJumpChernoff` TWS parabolic-contour Laplace
  inversion — large-T cost decoupled from `n ∝ T·‖A‖` to M=O(1) resolvent solves.
  `T_RESOLVENT_JUMP` 4/4. `G_RESOLVENT_JUMP_ORDER` slope +9.86 (gate ≥1.95) PASS.
  NARROW: self-adjoint / sectorial generators only; 2D/3D and hyperbolic-contour deferred.
- **[DONE — ADR-0135 Amdt 2, NARROW]** `KilledDirichletChernoff` Cayley-map hard absorbing
  wall — order-2 in the continuation region; BC baked into `L^R` domain (TRIZ structural
  separation; the Amendment-1 resolvent-mask route was NO-GO per §44.bis obstruction).
  `T_KILLED_GEN_CN` sub-checks A/B/C/D PASS. `G_HARD_WALL_ORDER2` PASS.
  NARROW: global free-boundary rate stays `O(√τ)` (structural; documented honest).
- **[DONE — ADR-0136 Amdt 2]** `ComplexTripleJumpChernoff` — order-4 hypoelliptic Chernoff
  on the filiform-N5 step-4 Carnot group via complex-time substeps (escapes Sheng–Suzuki
  barrier). `T_CARNOT_CPLX3` 16/16 PASS. `G_CARNOT_CPLX3` PASS (order-4).
  `G_CARNOT_STEP4` PASS (order-2 real baseline, Amdt 1).
  HONEST SCOPE: the order≥3 general-k Festschrift §3 open problem remains OPEN / escalated;
  only the verified order-4 filiform-N5 family ships.
- **[DONE — ADR-0123 Amdt 1]** `SmolyakGridND` extended to D=6; `G_SMOLYAK_D6` PASS.
- **[DONE — ADR-0107 Amdt 1]** `AdjointFokkerPlanckChernoff` weak-* adjoint evolver;
  `G_ADJOINT_FP_ORDER` corrected to genuine forward-vs-adjoint pairing (commit `ef48937`).
- **[DONE — ADR-0133 Phase-5]** F1 Dual-AD Greeks (value/delta/gamma) across FFI / PyO3 /
  WASM; F3 `KilledDirichlet1D` TIER-2 PyO3 binding. `G_BINDING_GREEKS_PARITY` 0-ULP
  cross-binding PASS (commit `a8ddbcc`).

### Withdrawn before tag

- **[WITHDRAWN — ADR-0137]** `EigenrotatedAnisotropicChernoff` F5 eigenbasis-rotation —
  the `D·q` node-reduction claim was falsified at review (commit `ac6853b`): the
  implementation achieved full `q^D` tensor quadrature plus per-node resample overhead,
  making it more expensive than its §32 sibling. Genuine `D·q` is blocked by the discrete
  eigen-grid resample. Per "no crutches, strong result": WITHDRAWN. ADR-0137 status:
  EXPLORED / DEFERRED. `anisotropic_eigenrotated.rs` does not ship.

### Honest-defers (carry forward to v8.x)

- **TIER-3 bindings** for F2/F4/C1/C2 — FFI/PyO3/WASM exposure of `ResolventJumpChernoff`,
  `ComplexTripleJumpChernoff`, `SmolyakGridND-D6`, `AdjointFokkerPlanckChernoff` deferred.
- **F2 2D/3D LHP backends + hyperbolic-contour variant** (math.md §47.6).
- **F3 variable-coefficient sharp order theorem** (empirically gated only in §44.ter).
- **`OctonicHermite` + `ChebyshevSpectralWithBC` generic samplers** for the AD path
  (ADR-0133 Amdt 1 honest-defer; no headline-blocking user).
- **F4 order≥3 general-k closure** — the Festschrift §3 open problem is escalated research;
  not shipped, not claimed.
- **Step-k Carnot general-k / order-≥3** — OPEN (escalated from v3.1/v4.5+ wave).
- **Dynamic Wentzell Robin BC** (ADR-0098 AMENDMENT 3, Stephan 2023 instability proof).
- **Holomorphic line-bundle sections over CP¹** (deferred from ADR-0129).

### Breaking changes

None. All v8.0.0 additions are strictly additive to the v7.0.0 public surface.

### Schema / constitution

- `traits.yaml` 4.0.0 → 4.2.0 MINOR. `properties.yaml` 4.0.0 → 4.2.0 MINOR.
- Constitution 3.0.0 → 3.1.0 MINOR: Cohort 13 (`dual.rs`, `resolvent_jump.rs`,
  `killed_dirichlet.rs`, `carnot_stepk.rs`, `adjoint_fp.rs`; 700-LoC cap each;
  all 3 overrides RE-AFFIRMED).

### Heavy validation (prod HW — deferred)

The following tests were validated on dev hardware with all blocking gates passing
in their `test-fast` / bounded-N configurations. The authoritative full-resolution
figures require the i7-12700K bench host (`~/bench-work/` or equivalent):

```bash
# All v8 blocking gates (fast, bounded N — run on any hardware)
cargo run -p xtask -- test-ignored-gates --features slow-tests

# Flagship sweep (full resolution — bench host, ~1000 s)
cargo run -p xtask -- test-flagship
```

Specific v8 gates for which the dev-gate N is small and the full-resolution run is
recommended on prod HW:

- `G_CARNOT_CPLX3` — dev gate uses N=4 temporal self-convergence on the filiform-N5
  group (spatial N=16 per axis); re-run at `n ∈ {8, 16, 24}` to confirm the order-4
  figure at higher N (5D grid, memory budget permitting).
- `G_RESOLVENT_JUMP_ORDER` — dev gate sweep `M ∈ {6,8,10,12,14}`, `t=100`, `N=64`;
  re-run at `N=256` for the authoritative large-grid cost-decoupling figure.
- `G_SMOLYAK_D6` — 6D sparse grid; confirm at N=20 per axis on prod HW (D=6, 64 GB RAM
  recommended for the dense reference oracle at N=20).

What was validated on dev (all pass, binary-size pass):
`G_DUAL_AD_GRADIENT`, `G_DUAL_ZERO_ALLOC`, `G_RESOLVENT_JUMP_ORDER` (N=64),
`G_HARD_WALL_ORDER2`, `G_CARNOT_CPLX3` (N=4), `G_CARNOT_STEP4`, `G_SMOLYAK_D6` (N=12),
`G_ADJOINT_FP_ORDER`, `G_BINDING_GREEKS_PARITY`.

Mirror pattern: v0.9.0 and v0.11.0 "Heavy validation" subsections.

---

## v7.0.0 — BREAKING Window #4: QuinticHermite Removal + ζ⁶/ζ⁸ TRUTHFUL_ORDER CLOSED (SHIPPED 2026-06-06)

FOURTH BREAKING window. 28 commits on `feat/v7.0.0-debt-closure`. ADRs 0117–0131.

### Shipped

- **[DONE — ADR-0117]** `InterpKind::OctonicHermite` (degree-9 Hermite, `O(dx¹⁰)`, virtual-node
  floor ≈ 9.1e-16). `.with_octonic_sampling()` on ζ-ladder kernels. `T_OCTONIC_HERMITE` 5/5.
- **[DONE — ADR-0118]** 6th-order divergence-form stencil `apply_div_form_6th` (7-point,
  O(dx⁶), conservation form). `T_DIV_STENCIL_4TH` 4/4 + `T_DIV_STENCIL_6TH` 4/4.
- **[DONE — ADR-0119 + AM1 + AM2]** `G_zeta6_TRUTHFUL_ORDER` PASS (finest-pair slope,
  N=8192, L=±32, T=10). `G_zeta8_TRUTHFUL_ORDER` PASS. Long-standing DEFER from v6.0 CLOSED.
- **[DONE — ADR-0120]** `Diffusion6thChernoff` genuine order-6 confirmed; floor-free basket
  recalibration; `g_zeta4_const_a_richardson_ratio` 3.5→3.1; `path_eps` preconditions fixed.
- **[DONE — ADR-0121]** `DiffusionExpmvChernoff` — Al-Mohy–Higham `expmv` action kernel,
  revives ζ⁸-accuracy goal via a structurally different published algorithm. ADR-0101 Padé
  deferral UNCHANGED; this is additive. `g_expmv_div_form_action_accuracy` PASS.
- **[DONE — ADR-0122]** `AnisotropicShiftChernoffND::with_adaptive_q(tol)` — 41% node
  reduction at equal accuracy. `G_ADAPTIVE_Q` PASS.
- **[DONE — ADR-0123]** `SmolyakGridND<F, D>` sparse-grid backend for D≥5 (9.2× node
  reduction at D=5). `G_SMOLYAK_D5` PASS.
- **[DONE — ADR-0112 AM2]** Order-2 ζ² `AnisotropicShiftChernoffND` correction.
  `G_AS_ZETA2_DDIM` (halving ratio ≈ 4.0) PASS.
- **[DONE — ADR-0124]** `MatrixDiffusionChernoff2D/3D<F, M>` palindromic Strang.
  `G_MATRIX_2D`, `G_MATRIX_3D` PASS.
- **[DONE — ADR-0125]** `MatrixExpPade<M>` — Padé[13/13] lifts M≥5 cap. `G_MATRIX_PADE_M5` PASS.
- **[DONE — ADR-0126]** `Killing2ndChernoff<C, K, F>` soft-killing order-2. `G_KILLING_ORDER2` PASS.
- **[DONE — ADR-0127]** Complex-λ Laplace-Chernoff resolvent. `G_CPLX_RES` PASS.
- **[DONE — ADR-0128]** `MatrixDiffusionChernoffComplex<F, M>`. `G_CPLX_MATRIX` PASS.
- **[DONE — ADR-0129]** `FubiniStudyCp1` Kähler backend. `G_KAHLER_CURV` PASS.
- **[DONE — ADR-0130]** `QuantumSchrödingerChernoff<C>`. `G_QSCHROD` PASS.
- **[DONE — ADR-0131]** `DriftReactionZeta4Chernoff` — resolves §27.7 OPEN via Path β.
  `G_DR_ZETA4_TRUTHFUL_ORDER` PASS.
- **[DONE — ADR-0109 12-month clock]** `InterpKind::QuinticHermite` + `legacy-quintic`
  feature REMOVED. `grid_quintic.rs` deleted. `with_quintic_sampling()` builder family
  removed from Rust and Python bindings. Migration guide `docs/migration/v6-to-v7.md`.
- **[DONE]** `eval_at_batch` (math.md §31).
- **[DONE]** WASM G_binding_parity sub-test 5 closed (commit 73de563).
- **[DONE]** `test-ignored-gates` xtask runner added (commit bb7899f).
- **[DONE]** 3 doctest scope bugs repaired (commit 8067ec8).

### BREAKING

- `InterpKind::QuinticHermite` REMOVED.
- `legacy-quintic` Cargo feature REMOVED.
- Associated builder methods removed (see CHANGELOG §[7.0.0] BREAKING section for full list).

### Honest-defers (carry forward)

- **`AdjointFokkerPlanckChernoff` implementation** (ADR-0107 engineer wave) — math.md §38
  NORMATIVE spec shipped; Rust impl pending industrial demand.
- **Step-k Carnot k≥3** — ADR-0077 AMENDMENT 1 HONEST-DEFER. Step-3 Engel slope −43.95
  verified (super-algebraic); no kernel ships until a stable discretisation is found.
- **Dynamic Wentzell Robin BC** — ADR-0098 AMENDMENT 3 HONEST-DEFER. Stephan 2023
  instability proof precludes shipping a kernel without a stable semi-discretisation.
- **D≥6 Smolyak** — construction identical to D=5; only D=5 gated in v7.0.0 (CI budget).
  V8.0 gating candidate.
- **Holomorphic line-bundle sections over CP¹** — deferred from Kähler ADR-0129 (scalar
  heat backend ships; line-bundle state requires separate math development).
- **`expmv` performance tuning** (Higham–Tisseur 1-norm estimator for `‖A‖`) — deferred
  per ADR-0121 rationale (conservative analytic bound is correctness-sufficient).

### Schema / constitution

- `properties.yaml` 3.0.0 → 4.0.0 MAJOR. `traits.yaml` 3.0.0 → 4.0.0 MAJOR.
- Constitution v2.0.0 → v3.0.0 MAJOR (all 3 overrides RE-AFFIRMED; Cohorts 11–12 added).

---

## v6.0.0 — BREAKING Window #3: SepticHermite + ζ⁶/ζ⁸ HONEST DEFER (SHIPPED 2026-05-30)

THIRD BREAKING window. 4 commits since v5.1.0 tag at f3129a9.

### Shipped

- **B.3 SepticHermite BREAKING per ADR-0109** (commit d6ae464, 49 files, 1832+):
  - NEW `InterpKind::SepticHermite` (7-pt degree-7 Hermite; Birkhoff-Garabedian-Lorentz).
  - `sample_chebyshev_1d` default: QuinticHermite → SepticHermite; floor 1e-10 → 1.89e-12 (67×).
  - `Grid1D::cheb_m` default flip (ADR-0099 reschedule resolved).
  - `grid_chebyshev_septic.rs` NEW (Cohort 10; 432 LoC ≤ 500).
  - `InterpKind::QuinticHermite` DEPRECATED (legacy-quintic shim; 12-month → v7.0.0).
  - `InterpKind::ChebyshevSpectral { m }` REMOVED (ADR-0104 12-month clock fulfilled).
  - `G_SEPTIC_HERMITE_FLOOR` RELEASE_BLOCKING gate ≤ 5e-12 (measured 1.89e-12).
- **ADR-0110 truthful_order framework** (commit c2a9203):
  - ζ⁴ TRUTHFUL_ORDER SHIPPED: `G_zeta4_TRUTHFUL_ORDER` gate ≤ -3.5 (measured -3.6573 PASS).
  - ζ⁶/ζ⁸ HONEST DEFER to v7.0+ (ADR-0110 AMENDMENT 1; T_ZETA_TRUTHFUL_ORDER_AMENDMENT1 4/4 PASS).
- **ChebyshevSpectral { m } REMOVAL** — ADR-0104 12-month clock from v5.0.0 fulfilled.
- **ADR-0099 `Grid1D::new` default flip** — reschedule resolved in v6.0 BREAKING wave.
- **Migration guide** `docs/migration/v5-to-v6.md` + `legacy-quintic` shim.
- **math.md §40 + §40.5.bis + §41** NORMATIVE: SepticHermite, three-regime taxonomy, pre-asymp order framework.
- **6 PRE-FLIGHT sympy oracles** 30/30 sub-checks PASS (T_SEPTIC_HERMITE 6/6 + T_ZETA_TRUTHFUL_ORDER_AMENDMENT1 4/4 + T_ZETA_CONST_A 6/6 + T_ZETA_TRUTHFUL_ORDER 6/6 + T_CHEBYSHEV_SLOPE_LIMIT 5/5 + T_GR_2025_THM3 5/5).
- **Schema bumps**: properties.yaml 2.2.0 → 3.0.0 MAJOR; traits.yaml 2.3.0 → 3.0.0 MAJOR; constitution v1.9.1 → v2.0.0 MAJOR.

### Architecture pivots

- THIRD MAJOR (v3.0=Window#1, v5.0=Window#2, v6.0=Window#3).
- TWO AMENDMENTs landed mid-release:
  - **ADR-0109 AMENDMENT 1** (efb810f): const-a thresholds REVERT to v5.0.0 baselines per pre-asymp-temporal regime (math.md §40.5.bis).
  - **ADR-0110 AMENDMENT 1** (87c4fb4): ζ⁶/ζ⁸ HONEST DEFER; ζ⁴ threshold -3.95 → -3.5.
  - Both AMENDMENTs diagnose the same §39.2 mis-application family.
- User-authorized honest math defer: "никаких костылей и никаких хитростей мы за чистую эффективность, точность и математику".
- math.md §40.5.bis three-regime taxonomy NORMATIVE: saturated / signal-dominated / pre-asymp-temporal-transition.
- v3 API migration cascade absorbed across ~30 test files.

### Open after v6.0.0 (post-v7.0.0 status)

- **[DONE v7.0.0 — ADR-0117/0118/0119]** OCTONIC-Hermite + 6th-order divergence stencil shipped;
  ζ⁶/ζ⁸ TRUTHFUL_ORDER gates PASS.
- **[DONE v7.0.0]** `InterpKind::QuinticHermite` REMOVED; `legacy-quintic` feature REMOVED.
- **[HONEST-DEFER]** `AdjointFokkerPlanckChernoff` implementation (ADR-0107) — pending
  industrial demand; carry forward.
- **[HONEST-DEFER]** `g3_6_slope_gate` spatial slope deficit (−4.5788 vs gate −5.85) —
  resolved at floor-measurement level by ADR-0120 (floor-free basket recalibration; the
  gate now passes). Investigation closed.
- **PyO3 parity P1–P7 (ADR-0111) — COMPLETE** (commits `3fb238f` + `de917f6`):
  all 7 waves shipped; 25 binding classes + 1 free function; pyright 40 → 0 errors.
  P7 unblocked by ADR-0112 normalization fix. No parity deferrals remain.

### v6.3.0 — Variational inequalities / obstacle evolver (2026-06-03, ADR-0116)

Additive MINOR. First nonlinear post-projection kernel family. Zero breaking changes.

- **`ObstacleChernoff<C, O, F>` + `Obstacle<F>` trait** — projective-splitting iterate
  `V^{n+1} = Π_g(S(Δτ)Vⁿ)`, `Π_g(W) = max(W, g)`. Wraps any library `ChernoffFunction`.
  `ConstantObstacle` and `ClosureObstacle` implementations. `order()=1` declared honest;
  `growth()` reports inner homogeneous part (affine `‖g⁺‖_∞` offset documented).
  D=1 only. Gates: `G_OBSTACLE_STATIONARY` (sup-err 5.55e-3 ≤ 2.5e-2, RELEASE_BLOCKING);
  `G_OBSTACLE_SLOPE_SMOOTH` (slope −0.997); `G_OBSTACLE_SLOPE_AMERICAN` (slope −0.825);
  `T_OBSTACLE_PROJECTION` 6/6; `T_OBSTACLE_ADJOINT` 1.03e-12.
- **`apply_active_set_adjoint_into`** — active-set adjoint primitive
  `λ = S*(Δτ)[diag(𝟙[W>g]) · λ]`; does NOT implement `AdjointApply<F>` (honesty
  boundary mirrors ADR-0115 §42).
- **math.md §44** NORMATIVE (7 subsections). Viscosity convergence: Barles–Souganidis
  1991. Projection identity: Crandall–Liggett 1971 / Brezis–Pazy 1972.

**Open directions after v6.3.0 (§44.7):**
- Variable-coefficient sharp order theorem (empirically gated only in §44.4).
- Degenerate / rough generators (Heston, Lévy) — comparison principle out of scope.
- Multi-asset / manifold obstacles (`D ≥ 2`, `CoordinateState` generalization).
- Second-order greeks (Γ) at the free boundary — OPEN (requires mollified `g_ε`).
- Penalty / PSOR as out-of-core alternatives (reference oracle only per §44.1).

See CHANGELOG.md §[6.3.0] for the full entry.

### v6.2.2 — External-prototype integration batch (2026-05-31, ADR-0115)

Additive PATCH driven by the `revssm` external prototype. Zero breaking changes.

- **Issue #2 — Graph state-adjoint** (math §42 Theorem 42.1): `GraphAdjoint`
  Python class + `evolve_state_adjoint_into` on `MagnusGraphHeatChernoff` /
  `VarCoefMagnusGraphHeatChernoff`. Gate: `T_MAGNUS_TRANSPOSE` 5/5.
- **Issue #1 — Adjoint-state parameter-sensitivity** (math §43): `edge_weight_grad`
  free function; `GeneratorSensitivity` trait; `adjoint_state_gradient`. §43.2 formula
  corrected pre-merge by `T_ADJOINT_STATE_SENSITIVITY` oracle (5/5).
- **Issue #3 — `dtype="f32"` opt-in** on `GraphHeat`, `MagnusGraphHeat`,
  `VarCoefGraphHeat`, `Heat1D`. Default `"f64"` unchanged.
- **Issue #5 — `Laplacian` introspection**: `to_dense()`, `row_ptr()`, `col_idx()`,
  `vals()` accessors (copies; frozen-topology invariant preserved).
- **Issue #4 — `Graph.from_edges` fix**: accepts both `(M,3)` list and flat
  `(3M,)` array; misleading error message corrected.

See CHANGELOG.md §[6.2.2] for the full entry.

### Post-v6.0.0 maintenance sweep (2026-05-30, 6 commits da1d3e5..de917f6)

Critical correctness fixes and CI hardening landed as the first post-v6.0.0 maintenance batch:

- **ADR-0112 — `AnisotropicShiftChernoffND` CRITICAL normalization fix** (commit ca70329):
  `F(0)=I` now holds; honest **order-1** (was falsely claimed order-2 since v4.0);
  gate −1.95 → −0.95; self-masking NaN behaviour resolved.
- **ADR-0111 — PyO3 full-parity Waves P1–P7 COMPLETE** (commits `de917f6` + `3fb238f`):
  Heat1DZeta8, TruncatedExp1D, TruncatedExp4th1D, Strang1D (Wave P1);
  SchrodingerComplex1D (Wave P2); Resolvent1D, Killing1D, Reflected1D, Robin1D (Wave P3);
  Howland1D, Subordinated1D (Wave P4); Manifold2D, HypoellipticChernoffKolmogorov,
  HypoellipticChernoffEngel (Wave P5); QuantumGraph, QuantumGraphHeat, MatrixDiffusion1D,
  PointEval, sample_gridfn2d, GraphTraj, StrangGraph (Wave P6); AnisotropicShiftND2,
  AnisotropicShiftND3, NonSeparable2DAniso, Heat2DVarA, Heat3DVarA (Wave P7).
  25 classes + 1 free function; pyright 40 → 0 errors; oracle-validated smoke suites PASS.
- **Quadrature constant fixes** (commit dfd4354): GH5 manifold outer node corrected;
  GL32 Heisenberg weight corrected; both fix-in-place (no gate disturbed).
- **`check-unsafe-scope` now token-aware + CI-wired** (commits dfd4354 + 45182d3):
  false-positive noise eliminated; genuine unsafe sites tracked.
- **`flagship-gates.yml` CI workflow** (commit 45182d3): RELEASE_BLOCKING gates
  (flagship + ζ truthful-order + latency) now run on nightly/manual dispatch —
  closes the systemic CI enforcement gap that allowed the G_DDIM bug to ship green.
- **Schrödinger PyO3 dispersion-sign fix** (commit e0f7dee).
- **Documentation accuracy sweep** (this commit): math.md §41.4/§41.6/§10.8.7/§41.2
  corrected; rustdoc in diffusion4_zeta4.rs / diffusion8_zeta8.rs corrected;
  zeta4_correction_slope.rs / zeta8_correction_slope.rs / zeta4_truthful_order.rs
  comments corrected; fabricated ADR-0081 §"D=5 fallback" citation removed.

### Heavy validation deferred (mirror prior pattern)

```
RUSTFLAGS="-C target-cpu=native" cargo test -p semiflow-core --release \
  --features slow-tests \
  zeta4_correction_slope_cheb zeta6_correction_slope_cheb zeta8_correction_slope \
  zeta4_truthful_order \
  -- --ignored --nocapture
```

Also applicable: G_SUBORD_ORDER1 slow-tests (v4.8.0 backlog).
`g3_6_slope_gate` requires `--features legacy-quintic`.

### Cumulative statistics post-v6.0.0

- **11 release tags** (v4.0.0 → v6.0.0).
- **25 ADRs** (0086 → 0110) + 12+ AMENDMENTs.
- **6 PRE-FLIGHT sympy oracles** in v5.0+ chain: 30/30 sub-checks PASS.
- **Constitution v2.0.0** MAJOR (3 overrides RE-AFFIRMED; Cohort 10 `grid_chebyshev_septic.rs`).
- **~286 fast-bins** (post-v6.0 engineer wave consolidation).
- **TWO MID-RELEASE AMENDMENTs** in v6.0 chain (§39.2 mis-application pattern documented).

Commits: c2a9203 (ADR-0109+0110 PRE-FLIGHT), efb810f (ADR-0109 AMENDMENT 1), 87c4fb4 (ADR-0110 AMENDMENT 1), d6ae464 (engineer wave).
See CHANGELOG.md §[6.0.0].

---

## v5.1.0 — ADR-0108 Chebyshev saturation formula NORMATIVE (SHIPPED 2026-05-30)

Docs-only MINOR. 3 commits since v5.0.0 tag at efe99b9.

### Shipped

- **ADR-0107** A.1 Adjoint Fokker-Planck weak-* math creation (commit 680fd48; ai-solutions-architect; 6/6 PRE-FLIGHT PASS; Outcome A — post-v5.0.0 additive).
- **doc-sweep-v5.0.0** README + api-stability + precision-policy + release-process update (commit 29e00bf; docs-writer).
- **ADR-0108** ζ⁴/ζ⁶/ζ⁸ Chebyshev slope deficit diagnosis (commit d586080; ai-solutions-architect):
  - H-F CONFIRMED PRIMARY: measured slopes {3.226, 3.870, 3.067} are the mathematical ceiling of QuinticHermite-bound sampler at N=512.
  - NORMATIVE saturation formula `slope_eff(N) = log₂((c·τ^{m+1} + φ) / (c·(τ/2)^{m+1} + φ))` shipped as math.md §39 (7 subsections, +97 LoC).
  - `T_CHEBYSHEV_SLOPE_LIMIT` NORMATIVE PRE-FLIGHT sympy oracle 5/5 PASS (`scripts/verify_chebyshev_slope_limit.py`); BLOCKS v5.1+ releases per ADR-0086 mandate. Sibling-complement to T_GR_2025_THM3.
  - `properties.yaml` schema 2.1.0 → 2.2.0 MINOR additive (+T_CHEBYSHEV_SLOPE_LIMIT record).
  - ADR-0104 rev-prediction inconsistency CLOSED.

### Architecture pivots

- v5.0.0 BREAKING thresholds {≥3.1, ≥3.8, ≥3.0} CODIFIED as truthful saturation-bounded measurements (NOT regressions; closes ADR-0104 rev-prediction gap).
- v6.0.0 BREAKING Window #3 plan committed: SepticHermite virtual-node sampler replacing QuinticHermite inside `sample_chebyshev_1d`; predicted φ ≈ 1e-13; predicted ζ⁴ ≥ 4.8 / ζ⁶ ≥ 5.6 / ζ⁸ ≥ 6.0 (exceeds ADR-0104 original rev-prediction). Bundles 3 BREAKING items: ADR-0099 Grid1D::new default flip + `ChebyshevSpectral { m }` REMOVAL (12-month clock) + NEW SepticHermite primitive.

### Open after v5.1.0

- v5.x prep MINORs: SepticHermite Waves R1–R4 architectural research (validate sampler before v6.0 BREAKING commit).
- v6.0.0 BREAKING Window #3 (~2027-05-29; 12-month from v5.0.0): SepticHermite + ChebyshevSpectral REMOVAL + Grid1D::new default flip.
- ADR-0107 engineer wave: `AdjointFokkerPlanckChernoff` implementation (deferred pending industrial demand).

### Heavy validation deferred

```
RUSTFLAGS="-C target-cpu=native" cargo test -p semiflow-core --release \
  --features slow-tests \
  zeta4_correction_slope_cheb zeta6_correction_slope_cheb zeta8_correction_slope \
  -- --ignored --nocapture
```

G_SUBORD_ORDER1 slow-tests (v4.8.0 backlog still applicable).

### Cumulative statistics post-v5.1.0

- **10 release tags** (v4.0.0 → v5.1.0; v5.0.0 LOCAL; v5.1.0 LOCAL — push pending maintainer).
- **23 ADRs** (0086 → 0108) + 11+ AMENDMENTs since v4.0.0.
- **5 PRE-FLIGHT sympy oracle series** in v5.0+ cycle (T_SUBORD 5/5 + T_CHEBYSHEV_WEIGHTS 2/2 + T_GR_2025_THM3 5/5 + T_ADJOINT_FP_TIGHTNESS 6/6 + T_CHEBYSHEV_SLOPE_LIMIT 5/5 = 23/23 PASS).
- Constitution v1.9.1 (3/3 overrides preserved).
- 240+ fast-bins.

Commits: 680fd48 (ADR-0107 A.1 adjoint FP math), 29e00bf (doc-sweep-v5.0.0), d586080 (ADR-0108 saturation formula).
See CHANGELOG.md §[5.1.0].

---

## v5.0.0 — BREAKING Window #2: Chebyshev Redesign + G_zeta4 Escalation RESOLVED (SHIPPED 2026-05-29)

SECOND BREAKING window of the post-v4.0 roadmap. 2 commits since v4.8.0 tag at 11c3b6a.

### Shipped

- **B.3 Chebyshev BREAKING redesign** (ADR-0104 Outcome A; commit 1ba9960, 19 files, 1198+/231−):
  - H3 PRIMARY defect fixed: `sample_chebyshev_1d` now dispatches `OobPolicy` on off-grid queries; eliminates Runge divergence cascade ζ⁴/ζ⁶/ζ⁸.
  - H4 SECONDARY defect fixed: rustdoc floor "≤ 1e-15" corrected to "≈ 1e-10 (QuinticHermite-bound; ADR-0104 H4)" in 4 diffusion kernels.
  - NEW `OobPolicy` enum (4 variants: Inherit/ForceReflect/ForcePeriodic/ForceZero).
  - NEW `Grid1D::cheb_m` convenience constructor.
  - NEW `InterpKind::ChebyshevSpectralWithBC { m, oob_policy }` variant.
  - DEPRECATED `InterpKind::ChebyshevSpectral { m }` — 12-month clock → removal v6.0.0.
  - 6 gate thresholds recalibrated TRUTHFUL; `properties.yaml` schema 1.6.0 → 2.0.0 MAJOR.
  - 11 new regression tests (`grid_chebyshev_bc_dispatch.rs`); migration guide `docs/migration/v4-to-v5.md`.
- **ADR-0106 Theorem 3 prefactor harness** (commit 1268198): Galkin-Remizov 2025 IJM Theorem 3 explicit-constant bound adopted as formal verification target. G_zeta4 escalation CLOSED — Path β Richardson permanently validated; BCH-only incompatibility confirmed symbolically by T_GR_2025_THM3 sub-check 3. 5/5 sympy PASS.

### Architecture pivots

- v3.0.0 was BREAKING Window #1 (ChernoffFunction trait cleanup); v5.0.0 is Window #2 (Chebyshev sampler redesign). Authorization: user directive "у библиотеки нет пользователей; можешь менять api".
- Path δ Padé permanently DEFERRED v6.0+ per ADR-0101 TERMINAL CLOSURE (not part of v5.0.0).
- ADR-0099 `Grid1D::new` Chebyshev DEFAULT flip RE-SCHEDULED to v6.0 (separate BREAKING decision; freed v5.0 budget for Chebyshev sampler fix).
- ADR-0106 Theorem 3 framework is machinery foundation for v5.x A.1 Adjoint Fokker-Planck research track.

### Heavy validation deferred (mirror v4.7.0 + v4.8.0 pattern)

Chebyshev-mode slope gates (slow-tests; post-fix calibration stable but lower than ADR-0104 prediction):

```
RUSTFLAGS="-C target-cpu=native" cargo test -p semiflow-core --release \
  --features slow-tests \
  zeta4_correction_slope_cheb zeta6_correction_slope_cheb zeta8_correction_slope_cheb \
  -- --ignored --nocapture
```

### Open after v5.0.0

- v5.1+ optional optimization ADR — if architect/user want +4.0 ζ⁴ slope vs current ≥ 3.1 measured threshold.
- v5.x research-track A.1 Adjoint Fokker-Planck weak-* math creation — ADR-0106 Theorem 3 prefactor harness (T_GR_2025_THM3) available as machinery foundation.
- v6.0 BREAKING: `Grid1D::new` Chebyshev DEFAULT flip (ADR-0099 reschedule).
- v6.0 BREAKING: `ChebyshevSpectral { m }` REMOVAL (12-month deprecation clock started 2026-05-29).

### Cumulative statistics post-v5.0.0

- **9 release tags** pushed origin (v4.0.0 → v4.8.0; v5.0.0 LOCAL — push pending maintainer).
- **21 ADRs** (0086–0106) + 8+ AMENDMENTs since v4.0.0.
- **Constitution v1.9.1** (3/3 overrides; Cohort 9 boundary.rs 600 LoC cap).
- 230+ fast-bins (post-v4.8 baseline; v5.0.0 +11 dispatch tests).
- Path β Richardson validated as G_zeta4 permanent closure (ADR-0106 Theorem 3).

Commits: 1268198 (ADR-0106 prefactor harness), 1ba9960 (B.3 Chebyshev BREAKING impl).
See CHANGELOG.md §[5.0.0].

---

## v2.8.0 — Manifold Pillar (SHIPPED 2026-05-27)

Third MINOR of the academic-priority v2.6 → v4.0 roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md`). First publishable math item — port Mazzucchi-Moretti-Remizov-Smolyanov 2023 Math. Nachr. + Neumann via image method + SABR-on-H² academic-industry synergy. All additive; no breaking changes.

- A4 Riemannian Manifold Chernoff — curvature-corrected Gaussian on T_xM via 3 closed-form backends (Torus/Sphere2/Hyperbolic2); BoundedGeometryManifold<F> trait; MMRS 2023 port (ADR-0071, math.md §24)
- B4 Neumann via image method — ReflectedHeatChernoff<C, R, F>; ReflectingRegion sibling to v2.6 KillingRegion (ADR-0072, math.md §25)
- G26 RELEASE_BLOCKING gates: sphere-S² slope -2.0408 (base) + -2.0406 (R/12) ≤ -0.95 + -1.95
- G27 RELEASE_BLOCKING gates: half-line residual 1.4459e-7 ≤ 1e-6 + slope -2.1343 ≤ -0.95
- T21N + T22N + T_MANIFOLD_CURVATURE sympy oracles PASS
- HFT side-track: examples/sabr_pricer.rs — SABR's vol process lives on H² → ManifoldChernoff<Hyperbolic2> IS the SABR pricer (academic+industry win in one release)
- Constitution v1.6.2 → v1.6.3 PATCH (Cohort 5 manifold.rs 800-LoC HARD LIMIT carve-out per ADR-0071)
- 149 bins / 701 fast-tests PASS / 0 failures (+29 tests over v2.7 baseline)

Commits: 43311ae, c28fdbe, 219c0f7, f7e778a, 9e3e0c3, plus sign-off commit.
See CHANGELOG.md §[2.8.0].

---

## v3.0.0 — BREAKING Window #1 (SHIPPED 2026-05-27)

Per `~/.claude/plans/roadmap-reflective-biscuit.md` v3.0 §. First MAJOR of the academic-priority v2.6 → v4.0 roadmap. FIRST BREAKING window. Six commits.

- B1 ApproximationSubspace<const K, F> opt-in marker super-trait — Galkin-Remizov 2025 IJM Theorem 3.1 (ADR-0073, math.md §26)
- ChernoffFunction trait BREAKING cleanup — apply removed (apply_chernoff inherent), Clone bound dropped, growth → Growth<F> struct, ChernoffSemigroup→Evolver rename (ADR-0074)
- A5 Diffusion4thZeta4Chernoff ζ⁴ correction kernel — Galkin-Remizov k=2 tangency (ADR-0075, math.md §27); EXPERIMENTAL — G_zeta4 RELEASE_BLOCKING slope DEFERRED to v3.1 calibration
- FFI/PyO3/WASM v3 additive surface (ADR-0076 Approach A): smf_*_v3 + EvolverHeat1DUnitV3 + GrowthV3 across all 3 bindings; v2 surface UNCHANGED for 12-month deprecation per ADR-0035 §9
- v2_compat shim (default-on feature; HARD REMOVED at v4.0): ChernoffSemigroup alias + ApplyChernoffExt blanket apply method
- Constitution v1.6.3 → v1.7.0 MAJOR per Override re-evaluation schedule (all 3 overrides RE-AFFIRMED)
- G_AS_K + T23N + G_binding_parity (sub-tests 1+3) RELEASE_BLOCKING gates PASS
- 153 bins / 719 fast-tests PASS / 0 failures

Commits: ce0cbf9, 815422d, d3e393b, 85adfe3, 4a44a15, ced48f1, plus Wave G sign-off.
See CHANGELOG.md §[3.0.0].

---

## v3.1.0 — Hörmander Research Pillar (SHIPPED 2026-05-27)

Fifth release of academic-priority v2.6 → v4.0 roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md`). Festschrift §3 open-problem ATTACK on hypoelliptic Chernoff approximations + quantum graphs Kirchhoff vertex condition + Hörmander paper draft for peer-review track. MINOR additive (no BREAKING beyond v3.0). Constitution Cohort 6 carve-out for hormander.rs.

- A3 HypoellipticChernoff palindromic Strang-Hörmander — step-2 Carnot (Kolmogorov, Heisenberg); order-2 via Galkin-Remizov 2025 IJM tangency (ADR-0077, math.md §28). Higher orders OPEN per Festschrift §3.
- B7 Quantum graphs Kirchhoff vertex condition — combined-domain Phase 1 + Kirchhoff projection (ADR-0078, math.md §29).
- G28 RELEASE_BLOCKING Kolmogorov slope -2.22 + G29 mass 3.88e-5 (threshold relaxed 1e-10 → 5e-5 per discretisation floor) + T_HORM 5/5 sympy PASS.
- G30 RELEASE_BLOCKING quantum graph eigenmode max_err 5.4e-4 + T_QG 3/3 sympy PASS.
- lie_bracket_kit.py REUSABLE sympy infrastructure.
- Hörmander paper draft (`docs/papers/hormander-paper-draft.md`) — *Russ. J. Math. Phys.* peer-review track collaboration artifact.
- **G_zeta4 calibration ESCALATED to v3.2+ architect math review** — engineer's numerical falsification proved BCH alone gives order-2 not order-4; Diffusion4thZeta4Chernoff stays experimental.
- Constitution v1.7.0 → v1.7.1 PATCH (Cohort 6 hormander.rs HARD LIMIT 800).
- 154 bins / 737 fast-tests PASS / 0 failures (+18 tests over v3.0).

Commits: 896a00a, 4c90979, 0babd03 (fmt sweep), bc864c3, aaca4f1, plus this sign-off SHA.
See CHANGELOG.md §[3.1.0].

---

## v4.0.0 — BREAKING Window #2 / FLAGSHIP (SHIPPED 2026-05-27)

EIGHTH and FINAL release of academic-priority v2.6 → v4.0 roadmap (`~/.claude/plans/roadmap-reflective-biscuit.md`). SECOND BREAKING window closing the trajectory. 9 commits.

- B6 SemiflowComplex trait + SchrödingerChernoffComplex Option B (ADR-0079, math §30); G_SCHROD_B Δ=1.35e-14 ≤ 1e-12
- A6 PointEval first-class API + 5 backend impls (ADR-0080, math §31); G_POINTEVAL 5/5 byte-identity PASS
- A2 AnisotropicShiftChernoffND<F, const D> flagship + GridFnND<F, D> (ADR-0081, math §32); G_DDIM v4.0 claim RETRACTED — CRITICAL normalization bug fixed by ADR-0112 (post-v6.0.0; see [Unreleased] section)
- B5 MatrixDiffusionChernoff<F, const M> matrix-valued (ADR-0082, math §33); G_MATRIX 4/4 PASS at order-1 baseline (threshold AMENDED -1.95→-0.80 per Wave D engineering decision)
- C6 LaplaceChernoffResolventResidual wrapper (ADR-0083, math §34); G_RES_RES 1.57e-4 ≤ 1e-3 (6.4× margin)
- Wave F Diffusion4thZeta4Chernoff order 4→2 correction per ADR-0085 Option B DEFERRAL; EXPERIMENTAL rustdoc
- Wave G v2_compat HARD REMOVAL + 11-file no_std hygiene fix (per ADR-0084; 12-month deprecation cycle complete)
- Wave H rough_heston_pricer.rs HFT side-track (Carr-Cisek-Pintar 2021 Markov approximation; M=4; p99 ≈ 13 µs/tick)
- Constitution v1.7.1 → v1.8.0 MAJOR (Cohort 7 shift_nd.rs HARD LIMIT 700; num-complex direct dep; next re-eval DEFERRED)
- 167 bins / 785 fast-tests PASS / 0 failures (+48 over v3.1)

Commits: ef1b1d8, 6d5e3eb, 59f1a69, 405ed88, 295ac7a, 67271c5, 21cf422 (fix), feadc52, plus Wave H + this sign-off SHAs.
See CHANGELOG.md §[4.0.0].

---

## Academic-priority v2.6 → v4.0 Roadmap — CLOSED

The 8-release academic-priority trajectory planned at `~/.claude/plans/roadmap-reflective-biscuit.md` has SHIPPED in full:

- v2.6.0 (bfada77 / tag v2.6.0) — Infrastructure Foundation
- v2.7.0 (0b2a64f / tag v2.7.0) — Resolvent + Nonautonomous Lift
- v2.8.0 (43311ae / tag v2.8.0) — Manifold Pillar
- v3.0.0 (ce0cbf9 / tag v3.0.0) — BREAKING Window #1 (trait redesign)
- v3.1.0 (896a00a / tag v3.1.0) — Hörmander Research Pillar
- v4.0.0 (THIS RELEASE / tag v4.0.0) — BREAKING Window #2 (flagship)

No further releases planned in this roadmap. Future architecture per user discretion. Outstanding research items:

- G_zeta4 architect math review (per ADR-0085 §"Path forward"; full 4-term Taylor expansion algorithmic redesign OR 6-monomial sympy port)
- Heisenberg group numerics gate G_HORM_HEISENBERG (Beals-Greiner 1988 / Cygan 1981 heat kernel oracle)
- Step-k Carnot groups for k ≥ 3 (Hörmander research extension)
- B8 ζ⁴/ζ⁶/ζ⁸ truncated-exp ladder (pending G_zeta4 resolution)
- Order-2 Strang-symmetric MatrixDiffusionChernoff (G_MATRIX threshold restoration)
- schrodinger_complex.rs 504 LoC refactor (governance cleanup)

Total roadmap delivery: 6 MINORs + 2 MAJORs = 8 releases across ~3 months development. Library state at v4.0.0: ~167 test bins / ~785 fast-tests / 0 failures; full no_std-buildable; cleanly versioned v4.x stable surface.

---

## v4.1.0 — Post-v4.0 Tech-Debt Sweep (SHIPPED 2026-05-28)

Closure of 6 backlog items identified at v4.0.0 sign-off. 12 commits since v4.0.0 tag (11 tech-debt sweep + 1 README refresh).

| Item | Status | SHA(s) |
|------|--------|--------|
| 1 G_zeta4 Path β Richardson Option E hybrid (closes 4-deferral ADR-0085) | SHIPPED | 74fe63e (c86d337 ADR contracts; 02ab970 AMENDMENT 1 re-design) |
| 2 Heisenberg hypoelliptic backend | SHIPPED | 8a62d97 (cf514c1 formula correction) |
| 3 step-k Carnot k≥3 (Festschrift OPEN) | STILL_OPEN closure | research-track artifact only; deferred v5.x+ |
| 4 ζ⁶ rung Diffusion6thZeta6Chernoff | SHIPPED | 0c18cea (f6d8ed1 ADR-0088) |
| 4 ζ⁸ rung Diffusion8thZeta8Chernoff | Wave II Phase B pending | conditional architect decision |
| 5 Matrix Strang block CN Cayley map | SHIPPED | a59f6f6 (02ab970 AMENDMENT 2) |
| 6 schrodinger_complex governance refactor | SHIPPED | 2751e79 |

Path ε QuinticHermite spatial sample upgrade infrastructure shipped via ADR-0089 + AMENDMENT 1 (10758cd + 671d3b4). Per-kernel `.with_quintic_sampling()` opt-in builder + direct K5 wiring for ζ⁶ bypasses ζ⁴ pre-asymptotic regression.

README refresh at a534718 updates current-highlights to v3.0–v4.0 academic-priority roadmap CLOSED state.

13 architect ADRs / AMENDMENTs land this release: ADRs 0086, 0087, 0088, 0089 NEW; ADR-0082 AMENDMENT 2; each of ADR-0086/0087/0088/0089 + AMENDMENT 1.

Constitution v1.8.0 (from v4.0.0) preserved; Cohort 8 added for `diffusion4_zeta4.rs` HARD LIMIT 600.

Total fast-test count: 250+ / 0 failures (was v4.0.0: 167 bins / 785 fast-tests). New RELEASE_BLOCKING gates: G_zeta4_const_a_richardson (3.5582 ≥ 3.5) + G_zeta6_const_a_richardson (3.8679 ≥ 3.8) + G_HORM_HEISENBERG (slope −43.82) + G_MATRIX (slope −2.029, restored −1.95 threshold) + G_PATH_EPS_FLOOR.

See CHANGELOG.md §[4.1.0].

---

## v4.2.0 — Tech-Debt Sweep Final Closure (SHIPPED 2026-05-29)

Final closure of post-v4.0 tech-debt sweep. ζ⁸ Wave II DEFERRED v4.3+; Python parity for v4.1 APIs shipped; full docs refresh.

| Scope | Status | SHA |
|-------|--------|-----|
| ζ⁸ Wave II DEFER (ADR-0088 AMENDMENT 2) | Architect closure | 445c893 |
| Python parity v4.1 APIs (3 PyO3 wrappers + tests + stubs) | SHIPPED | Phase C bundle |
| Docs refresh (README + Python README + migration) | SHIPPED | Phase C bundle |
| pyrightconfig.json (stub-coverage fix) | SHIPPED | Phase C bundle |

Post-v4.0 tech-debt sweep CLOSED at this release:
- Items 1, 2, 5, 6: SHIPPED v4.1.0
- Item 4: ζ⁶ SHIPPED v4.1.0; ζ⁸ DEFERRED v4.3+ this release
- Item 3 step-k Carnot: STILL_OPEN closure v4.1.0 (research-track)
- Path ε QuinticHermite: Wave I SHIPPED v4.1.0; v5.0 global default promotion preview opens

Next: v4.3+ ADR-0090 candidate (spatial floor / Romberg-2D / v5.0 BREAKING redesign) OR v5.0 planning per ADR-0035 §9 12-month deprecation cycle.

---

## v4.3.0 — Research Wave Closure + Chebyshev Spectral (SHIPPED 2026-05-29)

Research wave on 3 open questions identified at v4.2.0 sign-off.

| Question | Outcome | SHA |
|----------|---------|-----|
| Q1 Spatial floor / Path ε successor | SHIPPED Chebyshev spectral collocation (M=64 default; floor ≤1e-15) | Wave A bundle |
| Q2 Direct ζ⁸ / skip nested cascade | SHIPPED Diffusion8thZeta8Chernoff via Chebyshev DEFAULT; Padé DEFERRED v4.4+ | Wave A + AMENDMENT 1 |
| Q3 step-k Carnot k≥3 | STILL_OPEN UNCHANGED (research-track artifact only; Path ε prerequisite confirmed) | n/a |

Architect Research Wave produced:
- ADR-0090 Chebyshev spectral collocation (d45ffcf architect; Wave A this release)
- ADR-0091 Diagonal Padé P_4/Q_4 + AMENDMENT 1 DEFER (d45ffcf architect; AMENDMENT 1 this release)
- ADR-0092 Romberg-2D NOVEL math creation (Outcome B negative result; sympy-verified)

Engineer Wave A SHIPPED: Diffusion8thZeta8Chernoff + Chebyshev infrastructure + 4 gates + migration guide.
Engineer Wave B DEFERRED: Padé impl preserved as research artifact (Option γ Suckless single-kernel principle).

Tech-debt sweep + research wave CLOSURE STATUS:
- v4.1.0: 4.5/6 backlog items SHIPPED + Path ε Wave I
- v4.2.0: Python parity + docs refresh + ζ⁸ nested DEFERRED
- v4.3.0: ζ⁸ via Chebyshev RESURRECTED + Padé DEFERRED + Romberg-2D NEGATIVE
- v4.4+ candidates: Padé scaling-and-squaring; v5.0 global default promotion; step-k Carnot if Kalmetev access OR new external advance

---

## v4.6.0 — PREP-MEASUREMENT MINOR (SHIPPED 2026-05-29)

Robin BC partial-additive port (A.3 ADR-0098) + Chebyshev opt-in composition RED verdict (B.3 ADR-0097 AMENDMENT 1). v5.0 B.2 promotion ABORTED; plan contracted to 2-item BREAKING window.

- **A.3 Robin BC SHIPPED** (ADR-0098): `BoundaryPolicy::Robin{α, β}` variant + `RobinHeatChernoff<C, R, F>` standalone kernel (~288 LoC). Skew-r reflection per Carslaw-Jaeger 1959 §14.2. ADR-0098 AMENDMENT 1: erfc factor 2·(α/β) → (α/β). T_ROBIN 4/4 PASS. G_ROBIN_HALFLINE + G_ROBIN_SELF BLOCKING gates.
- **B.3 Chebyshev RED verdict** (ADR-0097 AMENDMENT 1): ζ⁴/ζ⁶ Chebyshev opt-in composition catastrophic fail (anti-convergent / overflow). v5.0 B.2 global Chebyshev default promotion ABORTED. T_CHEB_ZETA4/6 4/4 PASS preserved.
- **Plan revision**: v5.0 BREAKING window 3→2 items (A.6 LadderRung + B.1 Padé DECISION; B.2 removed). v5.1+ B.2 conditional revival: 3 sub-paths (direct-kernel restriction / composition fix / hybrid default).
- traits.yaml schema 2.0.0 → 2.1.0 MINOR; properties.yaml schema 1.2.0 → 1.3.0 MINOR.
- 304 fast tests / 0 failed; check-lints PASS (26 grandfathered; 0 new violations).
- Pyright: 3 unused-assignment violations resolved (rename to `_` convention).

Commits: ea2d2a6 (Robin BC + ADR-0098 AMENDMENT 1), 8c95e1e (B.3 RED verdict + ADR-0097 AMENDMENT 1), sign-off SHA.
See CHANGELOG.md §[4.6.0].

---

## v4.8.0 — A.5 SubordinatedChernoff + B.3 Chebyshev BREAKING proposed (SHIPPED 2026-05-29)

A.5 SubordinatedChernoff ships strictly additive via Butko 2018 GL32-quadrature Bochner subordination. ADR-0102 closes Robin BC post-v4.7 reverification (negative outcome — all gates PASS). ADR-0104 reopens B.3 Chebyshev: two architectural defects identified; v5.0.0 BREAKING redesign proposed.

- **A.5 SubordinatedChernoff SHIPPED** (ADR-0103; commits a9f774f + 346e372): `LevySubordinator<F>` trait + 3 CBF backends (StableSubordinator α-stable / GammaSubordinator tempered / InverseGaussianSubordinator) + `SubordinatedChernoff<C, S, F>` generic wrapper; GL32 tables REUSED from v2.7 resolvent_quad.rs; subordinated.rs 469 LoC ≤ 500 budget; math §37 NORMATIVE (Bochner 1949 / Butko 2018 / Schilling-Song-Vondraček 2012 §13).
- **G_SUBORD_ORDER1** RELEASE_BLOCKING gate 3/3 PASS: α-stable −0.9894 / Gamma −0.9913 / InverseGaussian −0.9910 (all ≤ −0.95).
- **T_SUBORD** PRE-FLIGHT sympy 5/5 PASS (`scripts/verify_subordinated_chernoff.py`).
- **ADR-0102** Robin BC post-v4.7 reverification: NEGATIVE outcome — all gates PASS; closure documented.
- **ADR-0104** B.3 Chebyshev RE-OPENED: PRIMARY defect = no BoundaryPolicy branch; SECONDARY defect = false 1e-10 spectral floor; BREAKING_REDESIGN_PROPOSED for v5.0.0 (engineer wave spec at `.dev-docs/specs/b3-chebyshev-redesign-wave.md`; ~250 LoC source + ~120 LoC test).
- **T_CHEBYSHEV_WEIGHTS** PRE-FLIGHT sympy 2/2 PASS (`scripts/verify_chebyshev_spectral_weights.py`).
- traits.yaml schema 2.2.0 → 2.3.0 MINOR; properties.yaml schema 1.4.0 → 1.5.0 MINOR.
- Pyright cleanup: 2 unused variables removed in verify scripts (commits b6f2f4d).

### Architecture pivots
- ADR-0104 Outcome A: Chebyshev sampler missing BoundaryPolicy branch (all Dirichlet data silently extrapolated) + false 1e-10 spectral floor hardcoded regardless of M. v5.0.0 BREAKING window now has a concrete engineer wave spec: ChebyshevSpectralWithBC variant + `Grid1D::cheb_m` + `OobPolicy` enum.
- v5.0.0 properties.yaml schema 1.5.0 → 2.0.0 MAJOR planned (truthful Chebyshev thresholds: ζ⁴ ≥ 3.5, ζ⁶ ≥ 5.0, ζ⁸ ≥ 4.0 measured-not-predicted; v4.3 6.5/−6.5 values were PREDICTED).
- ADR-0099 Grid1D::new flip rescheduled to v6.0 (was v5.0; frees v5.0 BREAKING budget for Chebyshev).

### Heavy validation deferred
- G_SUBORD_ORDER1 slow-tests slope gate (3/3 PASS in fast mode; formal slow-test requires):
  `RUSTFLAGS="-C target-cpu=native" cargo test -p semiflow-core --release --features slow-tests subord_order1 -- --ignored --nocapture`

Commits: bfaaed8 (ADR-0102), a9f774f (ADR-0103), 347ac28 (ADR-0104), 346e372 (A.5 impl), b6f2f4d (lint).
See CHANGELOG.md §[4.8.0].

---

## v4.7.0 — MINOR re-version: A.6 LadderRung SHIPPED + Padé TERMINAL CLOSURE (SHIPPED 2026-05-29)

A.6 LadderRung<K, F> sealed super-trait (ADR-0100) ships strictly additive. B.1 Padé v5.0 third-attempt PRE-FLIGHT FAILED for all 3 candidate paths (Path α/β/γ); ADR-0101 Path δ TERMINAL CLOSURE — permanent DEFER v6.0+. Re-versioned v5.0 → v4.7 per ADR-0035 §"MAJOR-bump barrier" (A.6 alone is additive, no BREAKING change). v5.0.0 RESERVED.

- **A.6 LadderRung SHIPPED** (ADR-0100): sealed super-trait + 4 impls K=2/4/6/8; `const PREDECESSOR_K: Option<usize>` encodes K → K-2 invariant; T_LADDER_RUNG sympy 4/4 PASS; math §36 NORMATIVE (+83 LoC); Pyright-clean (0 errors post-cleanup).
- **B.1 Padé TERMINAL CLOSURE** (ADR-0101): Path α (genuine matrix squaring) FALSIFIED by sympy — R^(2^s)·v BYTE-IDENTICAL to scalar iteration; Path β (Higham m=13) ZERO envelope advantage; Path γ (Krylov-Arnoldi) circular dependency on α correctness. All 3 fail → permanent DEFER v6.0+. math §27.quart AMENDMENT 3 downgrades AMENDMENT 2 algorithm to CITATION mathematics.
- **v5.0 → v4.7 re-version**: per ADR-0035 §"MAJOR-bump barrier", MAJOR requires BREAKING; A.6 alone is additive → MINOR. v5.0.0 RESERVED for actual future BREAKING.
- traits.yaml schema 2.1.0 → 2.2.0 MINOR (LadderRung + 4 impl markers); properties.yaml schema 1.3.0 → 1.4.0 MINOR (T_LADDER_RUNG entry).
- 176 fast tests / 0 failed; check-lints PASS (26 grandfathered; 0 new); Pyright 0 errors.

Commits: 800b960 (architect bundle), 05ccd0a (A.6 engineer wave), sign-off SHA.
See CHANGELOG.md §[4.7.0].

---

## v2.7.0 — Resolvent + Nonautonomous Lift (SHIPPED 2026-05-27)

Second MINOR of the academic-priority v2.6 → v4.0 roadmap
(`~/.claude/plans/roadmap-reflective-biscuit.md`). All additive; no breaking changes.

- A1 Laplace-Chernoff Resolvent — (λI−A)⁻¹g via Gauss-Laguerre 32-pt; UNIQUE TO REMIZOV
  (no Trotter-Kato analog); Remizov 2025 Vladikavkaz Thm 3 (ADR-0069, math.md §22)
- B2 Howland nonautonomous lift — L²([0,T], X) augmented generator −∂_s + L(s);
  `TimedChernoffFunction` super-trait (12 leaf markers); continuous analog of
  `MagnusGraphHeat` (ADR-0070, math.md §23)
- G24 RELEASE_BLOCKING gates: residual 1.569e-4 ≤ 1e-3 + rate 3.91 ≥ 1.0
  (variable-a self-convergence)
- G25 RELEASE_BLOCKING gate: Howland slope −1.17 ≤ −0.95
- T19N + T20N sympy oracles PASS
- L-gate harness PROMOTED advisory → blocking (`L_CEV_PTICK` first blocking L-gate;
  `L_RESOLVENT_N64_P99` + `L_HESTON_PTICK` advisory throughout v2.7-rc)
- HFT side-tracks: `examples/resolvent_perf.rs` + `examples/heston_pricer.rs`
  (ρ→0 limit; ρ ≠ 0 deferred to v2.7.x/v2.8)
- 147 bins / 672 fast-tests PASS / 0 failures

Commits: 0b2a64f, 6723161, 753f4f4, b1a8cb1, b702783.
See CHANGELOG.md §[2.7.0].

---

## v2.6.0 — Infrastructure Foundation (SHIPPED 2026-05-27)

First MINOR of the academic-priority v2.6 → v4.0 roadmap
(`~/.claude/plans/roadmap-reflective-biscuit.md`). All additive; no breaking changes.

- **BoundaryPolicy widened**: `Dirichlet { value: F }` + `Neumann` variants (ADR-0068
  Track 1, math.md §3.5.bis). Enum becomes `BoundaryPolicy<F = f64>`. Existing
  call-sites compile unchanged.
- **L-gate harness infrastructure**: `properties.yaml` `latency_gates:` schema
  (v0.7.6 → v0.8.0); `xtask latency-gate` subcommand (advisory, exit 0);
  `HdrSnapshot` NIST nearest-rank lib (no_std+alloc, `hdr.rs`);
  `examples/latency_tail.rs --format=jsonl` (ADR-0068 Track 2, math.md §3.6.bis).
- **B3 Dirichlet via killing**: `KillingChernoff<C, R, F>` wrapper +
  `BoxRegion<F, D>` / `BallRegion<F, D>` reference impls (D=1 fully tested;
  D=2,3 deferred to v2.7). Order-1 globally per Butko 2018 §3. (ADR-0068
  Track 3, math.md §21.)
- **G23 RELEASE_BLOCKING gate**: slope **−1.058** ≤ −0.95. PASS.
- **T18N sympy oracle**: PASS (`scripts/verify_killing_dirichlet.py`).
- **144 bins / 646 fast-tests PASS / 0 failures**.
- **check-lints**: 18 grandfathered / 0 new violations.
- Constitution v1.6.1 → v1.6.2 PATCH: `grid.rs` removed from Override #1 (499 LoC
  after `boundary.rs` extraction); `varcoef_magnus_graph.rs` added Cohort 4
  (rustfmt 476 → 520 LoC; ADR-0068).

Commits: bfada77 (contracts), e9287d8 (Wave A), 1ff4f03 (Wave B), cfd2e66 (Wave C).
See CHANGELOG.md §[2.6.0].

---

## Released

- **v0.1.0** — `ShiftChernoff1D` (Theorem 6 formula 6, Gaussian heat-kernel
  verification, no_std + alloc)
- **v0.2.x** — Strang composition (`StrangSplit`, `DiffusionChernoff`,
  `DriftReactionChernoff`); `BoundaryPolicy`; RK2 drift upgrade (HLW §III.5)
- **v0.3.x** — zeta-A tau²-correction for variable `a(x)` (BCH-derived);
  CEV option pricing example; empirical validation beta in {0.3, 0.5, 0.7}
- **v0.4.x** — TruncatedExp K=4 power series (`TruncatedExpDiffusionChernoff`,
  formerly `MagnusDiffusionChernoff`), variable-a O(tau²); log-space ncx2
  oracle hardening (v0.4.1)
- **v0.5.0** — 2D tensor product (`Grid2D`, `GridFn2D`, `AxisLift`, `Strang2D`),
  Theorem 7; G3-2D spatial slope -2.056
- **v0.6.0** — 4th-order spatial (`Diffusion4thChernoff`,
  `TruncatedExp4thDiffusionChernoff`, formerly `Magnus4thDiffusionChernoff`);
  `AdaptivePI` Söderlind PI controller; G3⁴-2D slope -3.99
- **v0.7.0** — naming rationalization (`Magnus*` → `TruncatedExp*`, ADR-0013);
  6th-order spatial (`Diffusion6thChernoff`, 9-point Fornberg + quintic-Hermite, ADR-0015);
  non-separable 2D (`NonSeparable2DChernoff`, 5-leg palindromic BCH, ADR-0016)
- **v0.7.1** — patch: G3⁶-2D flagship deferred to v0.8.1 (tile-scratch perf prerequisite)
- **v0.8.0** — performance: parallel `Strang2D::apply` (`std::thread::scope`, ADR-0018);
  manual SIMD (`AVX2`/`NEON`, f64-only, ADR-0019); 2D benchmark suite
- **v0.8.1** — G3⁶-2D FLAGSHIP gate closed (slope -6.0837 over N∈{503,997,1999},
  wallclock 3090s ≤ 3300s budget); tile-scratch prerequisite (ADR-0022, 4.38× heat_2d)
- **v0.9.0** — 3D tensor product (`Strang3D`, `Grid3D`, Theorem 7' inductive, ADR-0024);
  anisotropic non-separable 2D (`NonSeparable2DAnisotropicChernoff`, ADR-0023);
  Generic-over-Float (`SemiflowFloat` trait, ADR-0025/0026)
- **v0.10.0** — language bindings: C ABI (`semiflow-ffi`, Wave A, ADR-0028),
  PyO3 wheels (`semiflow-py`, Wave B, abi3-py310), WASM (`semiflow-wasm`, Wave C,
  wasm-bindgen+wasm-pack); cross-binding sup-error identity 1.46e-6
- **v0.11.0** — bindings polish: npm publish workflow (`release-wasm.yml`,
  ADR-0030), PyO3 GIL release in `Heat1D.evolve` (ADR-0031), PEP 561 type
  stubs + `py.typed` marker, Firefox cross-engine CI; I12 heavy validation
  pending maintainer prod-HW; v0.9.0+v0.10.0 math fidelity audit (ADR-0032);
  I3/I4/I5/I14 deferred to v0.12.0+
- **v2.0.0** — State<F>/HilbertState/Discrete 3-layer trait promotion (BREAKING,
  ADR-0043); scratch arena + `apply_into` (W1); in-place Strang2D/3D ping-pong
  (W2); generic `AdaptivePI<C,F,K>` + H211b (W4); zero-copy FFI/PyO3/WASM
  evolve_into + f32 composition path (W5); H-MEM 0-alloc steady-state enforced;
  math fidelity audit APPROVED. Migration guide: `docs/migration/v1-to-v2.md`.
- **v2.1.0** — Graph PDE: `Graph<F>`, `Laplacian<F>`, `GraphSignal<F>`,
  `GraphHeatChernoff` (order-1), `GraphHeat4thChernoff` (ζ⁴, order-4),
  `StrangSplitGraph` (bipartite Strang), `MagnusGraphHeatChernoff` (Magnus K=4,
  time-dependent `L_G(t)`); G7–G11 slope gates PASS; T12 sympy PASS;
  `OutOfMagnusRadius` convergence-radius guard. ADR-0047–0051.
  Tagged 2026-05-21 (heavy validation deferred to prod HW per v0.11.0 precedent).
- **v2.2.0 (SHIPPED 2026-05-21)** — Advanced semigroups & graph time-dependence.
  Three waves: A — variable topology / variable-a / time-discontinuous graph
  Laplacian (ADR-0052/53/54); B — adjoint wrapper + Magnus K=6 + real-only
  Schrödinger (ADR-0055/56/57); C — `NonSeparableMixedChernoff` unification
  (ADR-0058 SUPERSEDES ADR-0033) + graph FFI/PyO3/WASM bindings (ADR-0059).
  Phase 5 parallel benchmarks (ADR-0060). math.md §14–§18 NEW. Sympy gates
  T11N–T15N PASS. Slope gates G12–G20 + cross-binding graph identity PASS.
- **v2.3.0 (SHIPPED 2026-05-22)** — Python parity expansion: bring `semiflow-py` to
  full parity with the Rust core surface (Phases 1–7). ADR-0061.
- **v2.4.0 (SHIPPED 2026-05-22)** — Graph Completeness: `GraphHeat6thChernoff`
  (order-6 static, ADR-0062), `VarCoefMagnusGraphHeatChernoff` (variable-a ×
  time-dependent Magnus K=4, ADR-0063), FFI / WASM expansion (ADR-0064).
- **v2.5.0 (SHIPPED 2026-05-22)** — Python parity for v2.4 graph kernels:
  `GraphHeat6` and `VarCoefMagnusGraph` pyclasses + `compute_rho_bar` staticmethod.
  Closes `audit-findings-v2_4_0.md` §5 Python deferral. ADR-0065.
- **v2.5.1** — Cache-residency + HFT-latency benchmarks (tracking-alloc feature,
  latency_tail example, #[inline] audit on hot-loop entry points). L1d-resident CEV
  hot loop; 149× p99.9 advantage vs QuantLib V3 Schroder ncx2 at matched accuracy
  (5e-4 gate). ADR-0066/0067. Test suite unchanged 198/0.
- **v2.6.0 (SHIPPED 2026-05-27)** — Infrastructure Foundation: `BoundaryPolicy`
  widening (Dirichlet+Neumann), `KillingChernoff` B3 Dirichlet via Feynman–Kac
  killing, L-gate harness infrastructure (`latency_gates:` schema, `xtask latency-gate`,
  `HdrSnapshot`). G23 RELEASE_BLOCKING slope −1.058. T18N PASS. ADR-0068.
  646 fast-tests PASS. Constitution v1.6.2. See CHANGELOG §[2.6.0].

---

## v2.5.0 — Python parity for v2.4 graph kernels (SHIPPED 2026-05-22)

**Theme**: close the v2.4 Python deferral. f64-only PyO3 bindings for the
two new v2.4 kernels.
**SemVer**: MINOR (additive). Two new pyclasses + one staticmethod.
**Branch**: `feat/v2.5-py-parity` (parent v2.4 graph completeness).
**ADR**: 0065.
**Status**: SHIPPED 2026-05-22. 151 Python tests PASS.

### Scope (Phase 1 complete)

- [x] `GraphHeat6` pyclass — order-6 spatial heat (ADR-0062), f64-only.
- [x] `VarCoefMagnusGraph` pyclass — variable-a × time-dep Magnus K=4
  (ADR-0063), f64-only, dual JS-like callback `(lap_at_t, a_at_t)`.
- [x] `VarCoefMagnusGraph.compute_rho_bar` staticmethod — caller-side
  estimator for `(rho_bar_max, a_sup_max)` over a time interval.
- [x] 22 new pytest cases (10 GraphHeat6 + 12 VarCoefMagnusGraph).
- [x] Constant-a parity check vs `MagnusGraphHeat` (K=4 time-dep), 1e-2
  tolerance.

### Deferred to v2.6+

- FFI bindings for `GraphHeat4`, `MagnusGraphHeat6`, `VarCoefGraphHeat`
  (Python-only today).
- WASM time-dependent Magnus (pre-built schedule API).
- Comprehensive `__init__.pyi` type stubs for `GraphHeat6` /
  `VarCoefMagnusGraph`.

---

## v2.4.0 — Graph Completeness (SHIPPED 2026-05-22)

**Theme**: close three documented gaps in the v2.3 graph stack — order-6 static
graph kernel, variable-coefficient × time-dependent composition, and FFI/WASM
coverage of the new kernels.
**SemVer**: MINOR (additive). No breaking changes.
**Branch**: `feat/v2.4-graph-completeness` (parent `8073830` = v2.3 Phase 7).
**ADRs**: 0062, 0063, 0064.
**Status**: SHIPPED 2026-05-22. See CHANGELOG §[2.4.0].

### Scope (Waves 2.4A/B/C, all complete)

- [x] **Wave 2.4A — Order-6 spatial graph heat** (ADR-0062, math.md §19).
  `GraphHeat6thChernoff<F>` in `crates/semiflow-core/src/graph_heat6.rs` —
  degree-6 operator-Taylor of `e^{-τL_G}`, 6 SpMV / step, 2 ping-pong
  scratch buffers (zero heap alloc steady-state), generic `<F: SemiflowFloat>`.
  Gates: G21 f64 slope ≤ −5.85; G21 f32 absolute-floor `|err|_∞ ≤ 5e-6`;
  T16N sympy.
- [x] **Wave 2.4B — Variable-coefficient × time-dependent Magnus K=4**
  (ADR-0063, math.md §20). `VarCoefMagnusGraphHeatChernoff<F>` in
  `crates/semiflow-core/src/varcoef_magnus_graph.rs`. Closure-driven
  sampling of `(a(t), L_G(t))` at GL₂ abscissae. `compute_rho_bar` helper.
  G22 f64 slope ≤ −3.85; T17N sympy; G11 byte-equality regression PASS.
- [x] **Wave 2.4C — FFI / WASM expansion** (ADR-0064).
  - FFI: `smf_ghc6_*` + `smf_vc_mghc_*` in
    `crates/semiflow-ffi/src/graph_ffi_v2_4.rs`. `remizov.h` regenerated.
  - WASM: `GraphHeat6` in `crates/semiflow-wasm/src/graph_wasm_hi.rs`.

### Deferred to v2.5+

- **WASM time-dependent Magnus** (Magnus K=4 / K=6 / VarCoef Magnus on JS) —
  JS callback overhead; "pre-built schedule" path planned in v2.5.
- **Python bindings** for `GraphHeat6` and `VarCoefMagnusGraph`.
- **FFI bindings** for `GraphHeat4`, `MagnusGraphHeat6`, `VarCoefGraphHeat` —
  scope reduced from original plan to keep `graph_ffi_v2_4.rs` ≤500 LoC.
- Order-8 Magnus, VarCoef Magnus K=6, time-discontinuous `a(t)` — out of
  scope per ADR-0062 / ADR-0063 §"Out of scope".

---

## v2.3.0 — Python parity expansion

**Theme**: bring `semiflow-py` to full parity with the Rust core surface.
**SemVer**: MINOR (additive). No breaking changes. FFI/WASM changelogs will be
version-bump-only. Lockstep version bump to 2.3.0 deferred to Phase 7 final
acceptance per ADR-0035.
**Branch**: `feat/python-parity-v2.3` (parent `801166c` = v2.2.0).
**ADR**: 0061.
**Status**: In-progress. Phase 5 HEAD: dc45181. 129 Python tests passing.

### Scope (Phases 1–5, all complete)

- [x] **Phase 1 — Foundation** (7fc090a): `boundary` kwarg on `Heat1D/2D/3D`
  (`'reflect'` / `'periodic'` / `'zero'` / `'linear'`; default `'reflect'`);
  `Heat1D.with_a_array(a_values)` pre-sampled coefficient path (cubic-Hermite
  interpolation, zero GIL re-acquires); `boundary.rs`, `coeff.rs` helper modules.

- [x] **Phase 2 — 1D kernels** (2308ea6): `Heat1D4th` (4th-order, slope ≤ −3.5),
  `Heat1D6th` (6th-order, slope ≤ −5.5), `DriftReaction1D`, `Shift1D`. Each
  with scalar constructor + `with_arrays` staticmethod. Core additions:
  `Diffusion4thChernoff::with_closure`, `Diffusion6thChernoff::with_closure`,
  `DriftReactionChernoff::with_closure` (ADR-0034 pattern, additive).

- [x] **Phase 3 — Schrödinger** (a63ade3): `Schrodinger1D` pyclass wrapping
  `SchrodingerChernoff<f64>` + `SchrodingerState<f64>` (ADR-0057, math.md §17).
  Four constructors; `evolve`, `values`, `values_parts`, `norm_squared`.
  Unitarity gate `|‖ψ‖²/‖ψ₀‖² − 1| < 1e-6` over 500 steps. ~~Deferred~~.

- [x] **Phase 4 — Composition** (52d2695): `NonSeparable2D` (constant-c +
  `with_beta_array`; ADR-0058), `Adjoint` (5-variant enum dispatch; ADR-0055),
  `AdaptivePI` (5-variant enum dispatch, PI.4.7; ADR-0044). Core additions:
  `nonseparable_mixed_closure.rs`. ~~Deferred~~.

- [x] **Phase 5 — Graph expansion** (dc45181): `Graph` factory pyclass
  (`path`, `cycle`, `from_edges`, `erdos_renyi`), `Laplacian` pyclass
  (`combinatorial`, `normalized`), `GraphHeat4th`, `VarCoefGraphHeat`,
  `MagnusGraphHeat6`. Existing `GraphHeat`/`MagnusGraphHeat` extended to accept
  `Laplacian` directly. `GraphPath` deprecated. ~~Deferred~~.

### Pending (Phase 6–7)

- [x] **Phase 6 — Documentation** (this wave): `docs/python-coverage.md`,
  `docs/audit-findings-v2_3_0.md`, `docs/adr/0061-python-parity-expansion.md`,
  CHANGELOG / ROADMAP / README updates, `crates/semiflow-py/README.md` quickstart.

- [ ] **Phase 7 — Final acceptance**: `test-full` on production build; cross-binding
  parity gate (Python vs FFI ≤ 3 ULP); `ffi-smoke`, `py-smoke`, `wasm-test`,
  `ffi-headers --check` all green; lockstep version bump to 2.3.0.

### Out of scope (explicit)

- No `semiflow-core` breaking changes.
- No FFI/WASM surface expansion (they are already at parity for the exposed items).
- `TruncatedExp*` kernels: Rust-only (no user demand). Deferred.
- `StrangSplitGraph`: Rust-only (caller can compose manually). Deferred.
- `GraphTraj`: Rust-only (mutable closure lifetimes; deferred to v2.4+).
- `f32` Python path: no user demand (ADR-0046).
- Async / yield PyO3 API: no telemetry (ADR-0034 §"Out of scope").

---

## v0.6.1 (SHIPPED 2026-05-02)

**Theme**: Audit-derived fixes from `docs/audit-findings-v0_6_0.md`.
**SemVer**: PATCH (no public type/trait additions; `order()` numeric value
changes; `AdaptiveOutcome` step counts will differ for 4th-order inner types).

- [x] **D1 fix** — changed `Diffusion4thChernoff::order()` and
      `Magnus4thDiffusionChernoff::order()` to return `2` (was `4`).
- [x] **D1 verification** — G_PI gate re-run; `AdaptivePI` tol contract restored.
- [x] **D2 partial** — expanded rustdoc on Magnus types; cited BCO-R 2009.
- [x] Updated `CHANGELOG.md` with audit follow-up notes.

---

## v0.7.0 (SHIPPED 2026-05-02, major, naming rationalization + 6th-order spatial + non-separable 2D)

**Theme**: naming rationalization (D2 audit fix, clean break) + genuine 6th-order
spatial extension + non-separable 2D operators. Three blocks shipped together;
v0.7.0 absorbed the scope originally planned for v0.8.0.
**SemVer**: MAJOR (type renames are breaking).

- [x] **Block A — D2 full fix** (commit 8822469) — clean-break rename (NO
      deprecation aliases):
      `MagnusDiffusionChernoff` → `TruncatedExpDiffusionChernoff`,
      `Magnus4thDiffusionChernoff` → `TruncatedExp4thDiffusionChernoff`.
      Constants `MAGNUS4_CFL_NUMER` → `CFL_NUMER`, `MAGNUS4_CFL_DENOM` →
      `CFL_DENOM`, `MAGNUS_TRUNC_ORDER` → `TRUNC_ORDER`. ADR-0011
      Amendment 1, ADR-0013 Amendment 2. See `docs/audit-findings-v0_6_0.md` D2.

- [x] **Block B — 6th-order spatial** (commit 772f802) — `Diffusion6thChernoff`
      via quintic-Hermite interpolant (O(dx⁶)) + 7-point K-kernel (Fourier symbol
      matched through ξ⁶) + 9-point Fornberg FD (f', f'', f''' at 8/8/6-order).
      Caller invariant: `a ∈ C⁷(ℝ)`. ADR-0015. Empirical G3⁶ slope:
      **−6.2886** ≤ −5.85 gate (margin 0.44). 16 sympy NORMATIVE gates exit 0.

- [x] **Block C — Non-separable 2D operators** (commit 317c2ba) —
      `NonSeparable2DChernoff<X, Y>` for `L = Lx + Ly + c(x,y)·∂x∂y` via
      5-leg palindromic BCH splitting. τ-order 2. CFL: `4τ‖c‖∞ < dx·dy`.
      ADR-0016. Empirical G3_NS2D slope: **−1.9529** ≤ −1.95 (const c, tight);
      **−2.0394** ≤ −1.95 (var c). T7N_tau2 and T7N_zero-c NORMATIVE gates pass.

---

## v0.8.0 (performance theme)

**Theme**: speed up the core kernels without sacrificing accuracy.
Note: 6th-order spatial and non-separable 2D shipped in v0.7.0; v0.8.0
advances to the performance agenda originally planned for v0.9.0.
**SemVer**: MINOR (additive).

- [x] **Production parallel `Strang2D::apply`** — port the test-only parallel
      kernel from `tests/heat_2d_oracle_4th.rs` into `src/strang2d.rs` behind
      a `parallel` feature flag. Use `std::thread::scope` only (no rayon dep
      per suckless dep-count rule). Pre-allocated state buffer; no per-step
      allocator contention. Target: linear scaling to 8 threads at N >= 1600².
      (commits 300054f + 3a33e98; ADR-0018; STRANG2D_PARALLEL_BIT_EQUAL 3/3 pass)
- [x] **SIMD vectorization** — manual `#[target_feature]` paths for cubic-Hermite
      `f.sample`, quintic-Hermite `f.sample`, and 7/9-point FD stencils.
      x86_64 AVX2 + aarch64 NEON, with f64 stdlib fallback gated by `cfg`.
      No unsafe outside simd modules.
      (commits a927aac + 7d39938 + 3091c12; ADR-0019; SIMD_BIT_EQUAL 2/2 pass,
      SIMD_BIT_EQUAL_PARALLEL 3/3 pass)
- [x] **Benchmark suite expansion** — 2D heat and advection-diffusion benchmarks
      in `benches/`; target >= 5× speedup vs v0.6.0 at N=1600².
      (commits 1399e7f + a1d8c78; ADR-0017; baselines in docs/perf-baseline-v0_7_0.md)
- [x] **G3⁶-2D FLAGSHIP** — 2D 6th-order spatial slow-test gate (deferred from
      v0.7.0 per plan; planned as v0.7.1 hotfix, now rolled into v0.8.0 scope).
      > Status: CLOSED in v0.8.1 (2026-05-07). Gate
      > `g3_6_2d_flagship_slope_and_runtime_gate` PASSES with slope=-6.0837
      > (window [-6.15, -5.85]) and wallclock=3090s (budget 3300s) under
      > `RUSTFLAGS="-C target-cpu=native" cargo test --release --features
      > parallel,simd,slow-tests`. Recalibration to 3-point prime-N basket
      > {503, 997, 1999} per ADR-0020 Amendment 3 (the 4-point basket
      > {503, 997, 1999, 3989} mirroring the 1D 5-point pattern was empirically
      > infeasible at ≈3.4h wallclock — physical N²-cost amplification at the
      > largest entry). Block A tile-scratch perf (commit eae2b7a, ADR-0022) is
      > a prerequisite. ROADMAP item 3 (≥5× combined speedup) validated
      > separately by `benches/heat_2d` (4.38× at N=1600²) and
      > `benches/advdiff_2d` (3.87×) — gate's role is mathematical correctness,
      > benchmarks' role is perf evidence.
      >
      > Historical: Two consecutive calibration failures (1st: slope -4.30 at
      > 388 s under schema 0.7.3; 2nd: slope -5.3945 at 772 s under schema 0.7.4
      > + RUSTFLAGS + cfg-tightening). Block A/B/C independently green in v0.8.0.
      > Test infrastructure shipped in v0.8.0 as `#[ignore]`; gate activated in
      > v0.8.1 after Block A tile-scratch perf landed.
- [x] **Clippy pre-existing warnings** — address the 66 pre-existing warnings
      (numerical variable naming conventions, doc backticks, 51-line function
      in adaptive.rs). Deferred from v0.7.0.
      (commits 1399e7f + a1d8c78; clippy --all-targets -D warnings exits 0)
- [x] Math fidelity audit (researcher agent) at MINOR milestone.
      (2026-05-06; APPROVED — see docs/audit-findings-v0_8_0.md)

---

## v0.9.0 (3D + universality)

**Theme**: extend the tensor product to three dimensions; lift the f64 monomorphism.
Note: performance theme moved to v0.8.0; 3D/universality advances to v0.9.0.
**SemVer**: MINOR.

- [x] **3D tensor product** — `Grid3D`, `GridFn3D`, `AxisLift3D`, `Strang3D`
      via Theorem 7 inductive extension to three commuting axes. Math contract:
      ADR for 3D Strang error accumulation.
      (2026-05-07; commits 1c48bb8 [math foundation, ADR-0024, math.md §10.8 with Theorem 7' + Lemma 10.1 formal introduction] + d827b83 [Rust impl: Grid3D, GridFn3D, AxisLift3D, Strang3D; Option A: Axis::Z extension])
- [x] **Generic-over-Float `F: Float`** — type-parameterize core types over
      `f32`/`f64` (and optionally complex via `num_complex`). Use `num_traits::Float`
      (already transitive dep). Lifts ADR-0004 monomorphic restriction. API
      surface expands; existing `f64` callsites remain source-compatible via
      type inference.
      (2026-05-07–08; commits 166b7d8 [pilot: SemiflowFloat trait, Grid1D/GridFn1D/DiffusionChernoff/State generic, ADR-0025] + e895864 [Wave 1: 6 1D Chernoff types] + 9a9dbed [Wave 2: Grid2D/GridFn2D] + cba3c26 [Wave 3: Grid3D/GridFn3D] + eb0abfed [Wave 4: ChernoffFunction trait generic + composition types AxisLift/Strang2D-3D/NonSeparable2D*/AdaptivePI/StrangSplit, ADR-0026]; parallel impls stay f64-only per ADR-0018 SIMD bit-equality contract)
- [x] **Anisotropic non-separable 2D** — `NonSeparable2DAnisotropicChernoff<X,Y>` for `L = L_x + L_y + β(x,y)·∂_x∂_y`. Block A (math foundation; ADR-0023, math.md §10.7-ter, sympy gates T9N_*) → Block B (Rust impl, ~250 LoC additive sibling, slope gate G4_NS2D_aniso). Closes math.md §10.7-bis preamble deferral.
      (2026-05-07; commits b8032a0 [Block A: math foundation, ADR-0023, math.md §10.7-ter, sympy gates T9N_*] + 04ec408 [Block B: Rust impl, NonSeparable2DAnisotropicChernoff<X,Y>] + 0180292 [G4_NS2D_aniso slope test redesign to self-convergence after empirical reference-floor failure; slope -2.1965 ≤ -1.95 ✓])

### Heavy validation (deferred to production hardware)

The slow-tests slope gates require production-grade compute and are not run
in CI / `test-fast`. Validate empirically on a high-core-count machine with
native CPU SIMD enabled:

```bash
# Full slow-tests slope gate suite (G3⁶-2D FLAGSHIP, G4_NS2D_aniso, G5_3D)
RUSTFLAGS="-C target-cpu=native" cargo run -p xtask -- test-flagship
# Expected wallclock: ~1 hour total (G3⁶-2D ~52 min, G4_NS2D_aniso ~7 min, G5_3D ~few min)

# Or run individual gates in isolation (e.g. for re-validation after a code change):
CARGO_TARGET_DIR=target-flagship RUSTFLAGS="-C target-cpu=native" \
  cargo test --workspace --features parallel,simd,slow-tests --release \
  --test convergence_rate_6th_2d -- --ignored --nocapture
# G3⁶-2D FLAGSHIP (slope ≤ -5.85, NORMATIVE per math.md §10.6.6)

CARGO_TARGET_DIR=target-flagship RUSTFLAGS="-C target-cpu=native" \
  cargo test --workspace --features parallel,simd,slow-tests --release \
  --test strang_nonseparable_aniso_slope -- --nocapture
# G4_NS2D_aniso self-convergence (slope ≤ -1.95, math.md §10.7-ter.7)

CARGO_TARGET_DIR=target-flagship RUSTFLAGS="-C target-cpu=native" \
  cargo test --workspace --features parallel,simd,slow-tests --release \
  --test strang_3d_slope -- --nocapture
# G5_3D closed-form 3D Gaussian oracle (slope ≤ -1.95, math.md §10.8.7)
```

The `CARGO_TARGET_DIR=target-flagship` is recommended to avoid invalidating
the `target/` cache used by `test-fast` during ongoing development.

If any slope gate empirically fails:
- G3⁶-2D: review math.md §10.6.6 calibration parameters; v0.8.1 historical run = -6.0837 / 3090s wallclock
- G4_NS2D_aniso: should slope ≈ -2.20 (self-convergence on β = 0.05·exp(-(x²+y²)/4))
- G5_3D: should slope ≈ -2.0 (closed-form 3D Gaussian; no reference-floor risk)

---

## v0.10.0 (FFI / Python / WASM)

**Theme**: real-user reach via FFI bindings.
**SemVer**: MINOR (additive).
**Status**: Released — 2026-05-09 (commit 92bd484, tag v0.10.0).

- [x] **`semiflow-ffi` C ABI** (Wave A shipped) — opaque-handle pattern, panic-boundary
      `catch_unwind` on every `extern "C"` function, status-code enum mapping
      `SemiflowError`. 7 entry points (1D heat with `a=1` only). Build matrix:
      `x86_64-unknown-linux-gnu`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`.
      32 integration tests pass; smoke `sup_error = 1.46e-6 << 5e-4`. ADR-0028.
- [x] **`semiflow-py` PyO3 wheels** — `cibuildwheel` for
      `manylinux_2_28`, `macos-13`/`macos-14`, `windows-msvc`,
      Python 3.10–3.13. End-to-end heat-equation smoke (`tests/test_heat.py`).
      ADR-0028.
- [x] **`semiflow-wasm` WASM bindings** (Wave C shipped) — `wasm-bindgen` +
      `wasm-pack`, `wasm32-unknown-unknown` `--target web` and
      `--target nodejs`, with `wasm-bindgen-test` smoke. 4/4 tests pass on
      Node; `gaussian_smoke` measured `sup_error = 1.460302e-6` matching
      Wave A's `1.46e-6` reading (sub-ULP cross-boundary identity).
      Profile divergence from Wave A/B `[profile.release-ffi]`: Wave C uses
      workspace `[profile.release]` (panic=abort) per ADR-0028 Amendment 1
      because wasm-bindgen routes panics through `__wbindgen_throw` natively;
      `console_error_panic_hook` provides dev-time isolation. CI: 3 jobs
      (`wasm-build` 3-OS, `wasm-test-node` 3-OS, `wasm-test-chrome` Linux).
      npm publish + var-a + 2D/3D + cross-engine browser deferred to v0.11.0.
- [ ] **API stability**: v0.10.0 surfaces ship as **experimental**;
      v1.0.0 freezes them.

---

## v0.11.0 (bindings-polish milestone)

**Theme**: distribution, concurrency, and validation closure for the v0.10.0
bindings — no new math, no new core API. Per **ADR-0029**, every change in
v0.11.0 is confined to the three binding crates (`semiflow-ffi`, `semiflow-py`,
`semiflow-wasm`), CI workflow files, and `docs/`. The reviewer-suckless gate
blocks the tag if `git diff v0.10.0..v0.11.0 -- crates/semiflow-core/` is
non-empty.

**SemVer**: MINOR (additive only).
**Status**: Released — 2026-05-09. I12 heavy validation pending maintainer
prod-HW access (v0.12.0 will absorb fresh data); I3/I4/I5/I14 deferred to
v0.12.0+.

### MUST set (per ADR-0029)

- [x] **npm publish for `semiflow-wasm`** — `@semiflow/wasm` published to
      npmjs.org via `release-wasm.yml` (tag-triggered `wasm-pack publish`,
      NPM_TOKEN custody, idempotency guards). ADR-0030 + Amendment 1 (SLSA).
- [x] **PyO3 GIL release** — `py.allow_threads` around `Heat1D::evolve` inner
      loop; `Send + Sync` verified by `static_assertions`; ≤2 % overhead in
      single-thread case. ADR-0031. DONE v0.11.0.
- [ ] **Heavy validation on production hardware** — G3⁶-2D FLAGSHIP
      (slope ≤ −5.85), G4_NS2D_aniso (≤ −1.95), G5_3D (≤ −1.95) — run on
      maintainer-local prod hw (~1 h wallclock); results in
      `docs/audit-findings-v0_11_0.md` with HW reproducibility block. ADR-0032.
      **PENDING** maintainer prod-HW access; v0.12.0 will absorb results.
- [x] **Math fidelity audit for v0.9.0 + v0.10.0** — researcher-agent reports
      land as `docs/audit-findings-v0_9_0.md` and `docs/audit-findings-v0_10_0.md`.
      Both APPROVED, zero DEVIATION-class findings. ADR-0032.

### SHOULD set (ships if effort permits, else slip to v0.11.x)

- [x] **Cross-engine WASM smoke** — Firefox + Safari headless coverage via
      `wasm-bindgen-test --firefox` / `--safari`.
      **Status: DONE (Firefox) — see Added section above. Safari headless deferred to v0.12.0+ (macOS-only; cost/value defer).**
- [x] **Python type stubs** — `.pyi` for `Heat1D` + `SemiflowError` + module
      functions; `py.typed` marker shipped in wheel.
      **Status: DONE — see Added section above.**
- [ ] **pyo3 / numpy / wasm-bindgen / js-sys lockstep bumps** — coordinated
      minor bumps if upstream releases land before v0.11.0 cut.

### COULD set (nice-to-have, no commitment)

- Bundler integration smokes (Webpack 5 + Vite 5 + Bun).
- `f32` ABI variant in bindings (research user request).
- `build_heat_unit` deduplication across FFI/Py (cosmetic).

### Deferred to v0.12.0+ (WON'T for v0.11.0)

- Safari headless WASM smoke (macOS-13 runner, deferred from v0.11.0 I2).
- Variable `a(x)` coefficient support across FFI/PyO3/WASM — gated by core
  `DiffusionChernoff::with_closure` (or `Box<dyn Fn>` field), which is itself
  a v0.12.0 design open question.
- 2D bindings (`Heat2D` wrappers around `Strang2D`) — premature without I3.
- 3D bindings (`Heat3D` wrappers around `Strang3D`) — premature without I3.
- Async / yield primitives in PyO3 binding — re-evaluate after I6 telemetry.

---

## v0.12.0 (bindings expansion + heavy validation)

**Theme**: close v0.11.0 deferred items and absorb I12 heavy-validation results.
**SemVer**: MINOR (additive).
**Status**: Planned.

- [x] **I3 — Variable `a(x)` in bindings** — `DiffusionChernoff::with_closure`
      core API (commit 2c8ca6f, ADR-0034); mirrored to FFI
      (`remizov_state_new_with_closure` + `heat_var_a.c` smoke), PyO3
      (`Heat1D.with_a_function`), and WASM (`Heat1D.withAFunction`,
      `JsCallback` newtype) in commit ec21002.
- [x] **I4 — 2D Python bindings** (`Heat2D` pyclass over `Strang2D`) —
      shipped in commit daa4019.
- [x] **I5 — 3D Python bindings** (`Heat3D` pyclass over `Strang3D`,
      sequential) — shipped in commit daa4019.
- [x] **I12 — Heavy validation results** — absorbed as
      `docs/audit-findings-v0_12_0.md` (DRAFT); slope-gate rerun on
      prod HW pending before v0.12.0 tag.
- [x] **NS2D_ANISO_PARALLEL_BIT_EQUAL gate** (audit O-2 from v0.9.0) —
      added in commit daa4019; 2 regression tests pass with
      `parallel,slow-tests` features.

### Deferred to v0.12.1 — WON'T for v0.12.0

- **I14 — Async / yield API for PyO3** — Insufficient telemetry on whether
  the v0.11.0 GIL-release optimisation (ADR-0031) saturates user demand for
  sync-blocking; revisit when evidence emerges. See ADR-0034 §"Out of scope".

### Deferred to v1.0.x post-publish

- **Safari headless WASM smoke** — Requires macOS GitHub runner, not
  provisioned during private dev. Firefox (fe49996, v0.11.0) + Chrome +
  Node coverage sufficient for v0.12.0.

---

## v1.0.0 (stability commitment)

**Theme**: API freeze, performance commitment, math fidelity audit.
**SemVer**: MAJOR.

- [ ] **API freeze** — public types and method signatures stable through v1.x
      for Rust + FFI + Python + WASM surfaces frozen simultaneously
      (per ADR-0028 §API stability). Deprecation cycle required for any removal.
      Both `NonSeparable2DChernoff` (scalar-`c`, v0.7.0) and
      `NonSeparable2DAnisotropicChernoff` (anisotropic-`β`, v0.9.0) are
      promoted as first-class APIs at v1.0.0 per **ADR-0033**
      (no `#[deprecated]` cycle; resolves audit O-3 v0.9.0).
- [ ] **Performance commitment** — published benchmark targets in `benches/`;
      regressions are CI-blocking (criterion `--baseline` comparison).
- [ ] **Math fidelity audit (researcher agent)** — obligatory at every MAJOR
      (X.0.0) release per "Math fidelity commitment" rule #5; record findings
      in `docs/audit-findings-v1_0_0.md`.

---

## v0.14.0 (architectural perf wrap-up + v2.0.0 spike)

**Theme**: close v0.13.0 perf waves + validate v2.0.0 Trait State abstraction.
**SemVer**: MINOR (additive).
**Status**: Planned.

- [x] **Wave A2 — Strang3D serial scratch** — shipped 2026-05-19 (52bac39, ADR-0022 Amendment 1).
- [x] **Wave B — TruncatedExp4 perf**:
  - [x] B1 (sample bypass, free 1.3-1.5×, bit-equal) — e0a993c
  - [x] B2 (HalfNodeCoeffCache, ADR-0034 Amendment 1) — 28d3134
  - [x] B3 (SIMD AVX2/NEON, ADR-0019 Amendment 2) — 9e93d6e
  - Cumulative: F2 65× → ~15-20× residual (algorithmic Krylov floor per ADR-0037).
- [x] **Wave C — parallel threshold + small-N fused-axis**:
  - [x] C1 (REMIZOV_PARALLEL_THRESHOLD env override) — 6a1273d
  - [✗] C2 (apply_fused fast path) — ABORTED, slope test failed; see ADR-0039
  - [✓] C3 (tile-scratch in 3D parallel) — SKIP confirmed (already present v0.8.1)
- [x] **Wave D — const-a fast path + cold-start polish**:
  - [x] D1 (type-state DiffusionChernoff const-a, ADR-0040) — cf430ce
  - [x] D2 (release-wasm profile, ADR-0040) — 56ce23b
  - [x] D3 (CI binary-size gate, ADR-0040) — 9b2df63
- [ ] **Bench iter-4** на host bestfriend — re-run F1-sec, F2, F5, F7. Success: ≥30% gap closure. PENDING hardware access.
- [x] **Graph-Laplacian spike** (`crates/remizov-graph-spike/`, NOT published) — 86a1f00 + 35318e6. 5/5 tests pass, 3-trait shape VALID with corrections (5 findings for v2.0.0 ADR).

---

## v1.0.0 (stability commitment) — see above

Concrete API frozen. State trait + ChernoffFunction::type S marked `#[doc(hidden)]` experimental per **ADR-0038**.

---

## v2.0.0 (Trait State / Banach-space generic core)

**Theme**: promote State/HilbertState/Discrete trait layer to central abstraction
plus scratch-arena zero-alloc, in-place Strang pencil ping-pong, generic
AdaptivePI, and zero-copy bindings.
**SemVer**: MAJOR (State<F> trait breaking change; concrete types stay).
**Status**: SHIPPED — 2026-05-21 (commits d7443fe → c8bd333, tag pending push).
ADR-0041 / ADR-0042 / ADR-0043 / ADR-0044 / ADR-0045 / ADR-0046.

- [x] **Trait State refactor** (Wave 3, ADR-0043) — three-layer hierarchy promoted
  from v1.x experimental `State<F>` stub to stable public API:
  - `State<F>` (Banach: len, axpy_into, copy_from, zero_into, norm_sup, scale_into).
  - `HilbertState<F>: State<F>` — dot, norm_sq, norm_l2 (ℓ² inner product).
  - `Discrete<F>: State<F>` — get, set, indices, neighbours (graph/manifold index).
  Breaking-change allowlist in `.cargo/semver-checks-allowlist.toml`.
  Migration guide: `docs/migration/v1-to-v2.md`.
- [x] **Scratch-arena + `apply_into`** (Wave 1, ADR-0041) — `ScratchPool<F>` free-list
  allocator; `ChernoffFunction::apply_into` default (clone bridge) + leaf overrides;
  R4 zero-alloc invariant in steady state.
- [x] **In-place Strang2D/3D pencil ping-pong** (Wave 2, ADR-0042) — eliminates
  3–5 intermediate `GridFnXD` allocations per τ-step; bit-equal gates 7/7 PASS.
- [x] **Generic `AdaptivePI<C, F, K>`** (Wave 4, ADR-0044) — `ClassicalPI` (byte-equal
  with v1.x) and optional `H211bFilter` advisory controller. `adaptive_classical_bit_equal` 4/4 PASS.
- [x] **Zero-copy bindings + f32 composition** (Wave 5, ADR-0045/0046) —
  `remizov_state_apply_into` (FFI), `Heat1D.evolve_into` (PyO3),
  `Heat1D.evolveInto` (WASM); `Strang2D/3D/NonSeparable2D<f32>` compile and run.
- [x] **H-MEM enforcement** — `cap-allocator` test harness asserts zero allocation
  per step for all concrete `GridFnXD`/`GraphSignal` impls; `zero_alloc_steady` 3/3 PASS.
- [x] **SIMD specialisation preserved** — `<GridFnXD<f64>>` retains AVX2/NEON fast
  path per ADR-0018/0019; trait-generic path is a scalar fallback.
- [x] **Math fidelity audit** (`docs/audit-findings-v2_0_0.md`) — APPROVED, zero
  DEVIATION-class findings; 393/393 tests pass at HEAD.
- [x] **Migration guide** — `docs/migration/v1-to-v2.md`; v1.x → v2.0 decision tree.

### Post-release pending

- Iter-4 benchmark sweep on host bestfriend (agentic-qa in flight; hardware-dependent).
- `cargo semver-checks` PASS confirmed locally: major bump 1.0.0 → 2.1.0-rc.1
  satisfies all breaking changes per tool output "no semver update required" with
  allowlist in `.cargo/semver-checks-allowlist.toml`.
- Push to origin pending Anchor delegation to git-workflow (docs-writer + agentic-qa
  must both complete first).

### Deferred to v2.2+ (now ABSORBED INTO v2.2 — see `## v2.2.0` below)

- Variable topology `L_G(t)` (row_ptr/col_idx changes in t). **→ v2.2 Wave A (ADR-0052).**
- Variable-a graph ζ-A. **→ v2.2 Wave A (ADR-0053).**
- Graph bindings (FFI/PyO3/WASM). **→ v2.2 Wave C (ADR-0059).**
- Vector 3 unification (`NonSeparableAniso<S: Discrete<F>>`). **→ v2.2 Wave C (ADR-0058).**
- Vector 4a Adjoint-Chernoff (backward semigroup via HilbertState). **→ v2.2 Wave B (ADR-0055).**

---

## v2.1 — Graph-Laplacian semigroup series (Waves 2.1A/B/C)

**Theme**: time-independent and time-dependent graph heat semigroups as
first-class `ChernoffFunction` implementations, extending Theorem 6 to the
graph setting.
**Status**: All three waves SHIPPED (Wave A: 828f7bb, Wave B: 68f17d8,
Wave C: 03e9374; reviewer prep: b96fa51). Release-candidate pending
iter-4 bench on host bestfriend before v2.1.0 tag.
**SemVer**: MINOR (additive; no breaking changes to existing API).

### Wave 2.1A — `GraphHeatChernoff` + `GraphHeat4thChernoff` (order-1 and order-2)

- [x] `GraphHeatChernoff<F>` — order-1 graph heat Chernoff, frozen `L_G`. (`src/graph_heat.rs`)
- [x] `GraphHeat4thChernoff<F>` — order-2 graph heat Chernoff with ζ-correction. (`src/graph_heat4.rs`)
- [x] `GraphSignal<F>` — CSR-backed graph signal state (`src/graph_signal.rs`, `impl State<F>`)
- [x] G7/G8/G9/G10 slope gates.

### Wave 2.1B — `StrangSplitGraph` (palindromic Strang on graphs)

- [x] `StrangSplitGraph<A, B, F>` — palindromic Strang `A(τ/2) ∘ B(τ) ∘ A(τ/2)` for
      graph heat semigroup operators. ADR-0050. (`src/strang_graph.rs`)
- [x] Bipartite edge-split constructor (`split_laplacians_by_edge`).
- [x] G10/G10-strang slope gates.
- [x] math.md §12.8 (Strang split on graphs).

### Wave 2.1C — `MagnusGraphHeatChernoff` (Magnus K=4, time-dependent `L_G(t)`) ✓ COMPLETE

- [x] `MagnusGraphHeatChernoff<F>` — order-4 Magnus K=4 for time-varying edge weights.
      GL4 nodes `c₁=(3−√3)/6`, `c₂=(3+√3)/6`. ADR-0051. (`src/magnus_graph.rs`)
- [x] `LaplacianAtTime<F>` type alias (`Box<dyn Fn(F) -> Arc<Laplacian<F>> + Send + Sync + 'static>`).
- [x] `apply_into_at(t_start, tau, …)` extension for correct absolute-time GL4 sampling.
- [x] `SemiflowError::OutOfMagnusRadius { tau, rho_estimate }` — convergence-radius check
      (`ρ̄_max · τ < π/2`, 50% safety margin). Additive, non-breaking.
- [x] **G11 slope gate** — f64 slope **−4.05** ≤ −3.95 ✓; f32 slope **−4.60** ≤ −3.50 ✓.
      (`tests/g11_magnus_graph_slope.rs`)
- [x] **T12_magnus_consistency sympy gate** — 3/3 PASS (GL4 abscissae, √3/12 coefficient,
      Ω₄ matches Magnus τ⁰–τ⁴). (`.dev-docs/verification/scripts/verify_v2_1c_magnus_consistency.py`)
- [x] **math.md §12.9** — CITATION + NORMATIVE library policy. (`contracts/semiflow-core.math.md`)
- [x] **ADR-0051 ACCEPTED** (`docs/adr/0051-graph-magnus.md`).
- [x] **errors.yaml updated** — `OutOfMagnusRadius` entry with emission table.

### Deferred to v2.2+ (now ABSORBED INTO v2.2 — see `## v2.2.0` below)

- Variable topology (`row_ptr`/`col_idx` changes in `t`). **→ v2.2 Wave A (ADR-0052).**
- Time-discontinuous `L_G(t)`. **→ v2.2 Wave A (ADR-0054).**
- Order-6 Magnus (nested commutators). **→ v2.2 Wave B (ADR-0056).**
- Self-adjoint / unitary (Schrödinger) variant. **→ v2.2 Wave B (ADR-0057).**

---

## v2.2.0 — Advanced Semigroups & Graph Time-Dependence (SHIPPED 2026-05-21)

**Theme**: close the 8 deferred items from v2.0 and v2.1 across three coordinated
waves. Wave A adds time-varying graph generators (variable topology, variable-a,
time-discontinuous). Wave B adds three new advanced-semigroup types (Adjoint
wrapper, order-6 Magnus, real-only Schrödinger). Wave C unifies the
`NonSeparable2D{,Anisotropic}Chernoff` pair under a generic and extends the
v0.10.0 bindings to graph PDE.
**SemVer**: MINOR (additive; type aliases preserve v2.1 surface; ADR-0033
SUPERSEDED but kept as alias per ADR-0058).
**Status**: SHIPPED 2026-05-21. All MUST items completed. ADR-0052–0060 ACCEPTED.
**ADRs**: 0052–0060.
**Math additions**: math.md §14 (NORMATIVE), §15–§18 (NEW).
**Sympy gates**: T11N–T15N. All PASS.
**Slope gates**: G12–G20 + cross-binding graph identity. All PASS.
**Constitution**: v1.5.0 — Override #1 file-list expanded per ADR-0056/0058.

### Wave 2.2A — Graph time-dependence (SHIPPED)

**Scope**: items 1, 2, 3 from the v2.0+v2.1 deferred list.
**ADRs**: 0052 (`GraphTraj<F>`), 0053 (`VarCoefGraphHeatChernoff`), 0054
(`evolve_with_traj`).

#### MUST

- [x] `GraphTraj<F>` — piecewise-smooth graph trajectory data structure
      (ADR-0052). Snapshot semantics, K ≤ 65_535 segments, CSR-immutability
      type-enforced. (`src/graph_traj.rs`).
- [x] `VarCoefGraphHeatChernoff<F>` — variable-coefficient `L_a` Chernoff with
      ζ-A τ²-correction (ADR-0053). (`src/graph_var_coef.rs`).
- [x] `MagnusGraphHeatChernoff::evolve_with_traj` + `evolve_with_traj_into` —
      jump-aware trajectory evolution (ADR-0054).
- [x] **G12 slope gate** — 3-segment `P_64` trajectory; slope ≤ −3.95 (f64). PASS.
- [x] **G13 slope gate** — var-a `P_n` self-convergence; slope ≤ −1.95 (f64)
      / ≤ −1.50 (f32). PASS.
- [x] **G14 jump-resolution gate** — `S_4` discontinuous-weight. PASS.
- [x] **T11N sympy gate** — 3-segment `P_4` matrix-exponential product. PASS.
- [x] **T12N sympy gate** — discrete ζ-A on `P_4`. PASS.
- [x] math.md §14 NEW (§14.1, §14.2, §14.3 NORMATIVE; §14.4 references).
- [x] ADR-0052/53/54 transitioned PROPOSED → ACCEPTED.

#### SHOULD

- [x] Migration guide `docs/migration/v2.1-to-v2.2.md` with `GraphTraj`
      onboarding examples + `fixed_topology` constructor for v2.1 callers.
- [x] CHANGELOG.md entry — Wave 2.2A.

#### MAY

- [x] `GraphTraj::segment_index` binary-search optimisation.

### Wave 2.2B — Advanced semigroups (SHIPPED)

**Scope**: items 4, 5, 6 from the v2.0+v2.1 deferred list.
**ADRs**: 0055 (Adjoint), 0056 (Magnus K=6), 0057 (Schrödinger — Option A).

#### MUST

- [x] `AdjointChernoff<C, F>` — backward-semigroup wrapper (ADR-0055).
      `new_general` + `new_self_adjoint` constructors. (`src/adjoint.rs`).
- [x] `MagnusGraphHeat6thChernoff<F>` — order-6 Magnus K=6 on time-dependent
      graph Laplacian via GL₆ (3-point Gauss-Legendre) + 4 commutator terms
      (ADR-0056). **f64 ONLY**. (`src/magnus6_graph.rs`, Override #1 carve-out).
- [x] `SchrodingerChernoff<F>` + `SchrodingerState<F>` — real-only Schrödinger
      unitary semigroup, Option A picked (ADR-0057). (`src/schrodinger.rs`).
- [x] **G15** — self-adjoint identity gate, 0 ULP. PASS.
- [x] **G16** — dual-pairing gate (drift-reaction). PASS.
- [x] **G17** — Magnus K=6 slope ≤ −5.85 (f64 only). PASS.
- [x] **G18** — Schrödinger unitarity. PASS.
- [x] **G19** — harmonic-oscillator period-revival oracle (f64). PASS.
- [x] **T13N** — adjoint consistency on asymmetric `P_4`. PASS.
- [x] **T14N** — Magnus K=6 residual through τ⁶. PASS.
- [x] **T15N** — V-rotation unitarity. PASS.
- [x] math.md §15, §16, §17 NEW.
- [x] ADR-0055/56/57 transitioned PROPOSED → ACCEPTED.

#### SHOULD

- [x] `AdjointChernoff::detect_self_adjointness` probabilistic developer-tool helper.
- [x] CHANGELOG.md entry — Wave 2.2B.

#### MAY (deferred to v2.3)

- [ ] Order-8 Magnus — explicitly OUT of v2.2 per ADR-0056 §"Out of scope".
- [ ] Schrödinger 3D variant — out of v2.2 per ADR-0057 §"Out of scope".
- [ ] Complex-valued `SemiflowComplex` trait sketch — deferred to v2.3 per
      ADR-0057 (Option A locked in).

### Wave 2.2C — Refactor + bindings (SHIPPED)

**Scope**: items 7, 8 from the v2.0 deferred list, plus parallel benchmarks.
**ADRs**: 0058 (NonSeparable unification), 0059 (graph bindings), 0060 (parallel bench).

#### MUST

- [x] `NonSeparableMixedChernoff<X, Y, F, S = GridFn2D<F>>` — generic unifier
      of `NonSeparable2DChernoff` and `NonSeparable2DAnisotropicChernoff`.
      v2.1 constructors preserved as type-aliased shims. (`src/nonseparable_mixed.rs`).
- [x] **G20 alias-identity gate** — 0 ULP between v0.7/v0.9 paths and generic. PASS.
- [x] Graph bindings — `crates/remizov-{ffi,py,wasm}/src/graph_*.rs`. f64-only.
- [x] **G_cross_binding_graph_identity gate** — cross-binding sup-error ≤ 3 ULP. PASS.
- [x] **G_FFI_smoke_graph, G_PyO3_smoke_graph, G_WASM_smoke_graph** — 3-OS. PASS.
- [x] ADR-0033 transitioned to SUPERSEDED-BY-0058.
- [x] ADR-0058/59 transitioned PROPOSED → ACCEPTED.
- [x] math.md §18 NEW.
- [x] CI updated: `release-ffi.yml`, `release-wheels.yml`, `release-wasm.yml`.

#### SHOULD

- [x] Migration guide `docs/migration/v2.1-to-v2.2.md` with `NonSeparable*`
      alias note.
- [x] Constitution v1.4.0 → v1.5.0 amendment (Override #1 file-list expansion).
- [x] CHANGELOG.md entry — Wave 2.2C + v2.2.0 release notes.

#### MAY

- [x] Parallel benchmark suite (ADR-0060). Shipped in Phase 5.

---

## v2.3 — Planned (deferred from v2.2)

**Theme**: parallelism for non-separable operators, higher-dimensional
Schrödinger, order-8 Magnus, and complex-valued state.
**SemVer**: MINOR (additive).
**Status**: Planned — no implementation started.

Items carried forward from v2.2 MAY lists:

- **NS2D-aniso parallel** — `AxisLift::apply_parallel` + parallel `apply_five_leg`
  for `NonSeparableMixedChernoff`. Blocked on adding a parallel `AxisLift` impl
  (serial fraction = 100% of hot path in v2.2; per ADR-0060 Amendment
  recommendation and `docs/perf/scaling-v2_2_0.md`).
- **Schrödinger 3D** — `SchrodingerState<F>` backed by `GridFn3D<F>` + Strang3D
  kinetic step. Deferred per ADR-0057 §"Out of scope".
- **Order-8 Magnus** — `MagnusGraphHeat8thChernoff` via GL₈ (4-point
  Gauss-Legendre) + 6 commutator terms. Design notes deferred per ADR-0056.
- **Complex-valued `SemiflowComplex` trait** — f32/f64 complex number support via
  `num_complex::Complex<F>`. Prerequisite for Schrödinger Option B (native
  complex). Deferred per ADR-0057 (Option A locked in for v2.2).

---

## Out of scope (deferred indefinitely or to other libraries)

- AMR / FFT-spectral exponential — out of scope per ADR-0012 (separate library)
- GPU acceleration — different design space; no current plan
- Stochastic PDEs — different mathematical framework
- Fully-implicit schemes — Chernoff approach is explicit by design
- MCP introspection server — withdrawn per ADR-0027 (no runtime to introspect;
  rustdoc + cargo cover what MCP would have provided).

---

## Math fidelity commitment

Every release must:

1. Pass sympy verification for all NORMATIVE math.md sections (properties.yaml gates).
2. Pass v0.5.0 regression bit-equal golden gate (no v0.5.0 surface change).
3. Document any new SIMPLIFICATION / APPROXIMATION / DEVIATION in
   `docs/audit-findings-v{N}.md` before tagging.
4. Cite original literature for every EXTENSION
   (Remizov 2025, HLW 2006, BCO-R 2009, Söderlind 2002, etc.).
5. Run a fidelity audit (researcher agent) at every MAJOR (X.0.0) release.

---

## How to contribute

Deferred until v1.0.0; project is single-author research preview through 0.x.
