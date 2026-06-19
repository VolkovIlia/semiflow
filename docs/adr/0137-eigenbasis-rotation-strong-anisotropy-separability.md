# ADR-0137 — F5 (minor) Eigenbasis-rotation separability for strong anisotropy at D≥5

**Status:** EXPLORED — DEFERRED (withdrawn from v8.0.0 ship; true D·q blocked by the discrete eigen-grid resample) · **Date:** 2026-06-06 (proposed) / 2026-06-07 (accepted, architect spike) / 2026-06-07 (DEMOTED, C2 review finding) · **Branch:** `feat/v8.0.0-planning`
**Theme:** v8.0.0 — MINOR cost optimization (NOT SHIPPED) · **Parent:** ADR-0132 · **Gates:** none active (`T_EIGENBASIS` retained as a continuous-operator oracle only; `G_EIGENBASIS_ANISO_DDIM` withdrawn) · **Math:** math.md §48 (re-scoped to "explored, deferred")

> **DEMOTION NOTICE (2026-06-07, review finding C2 — false performance claim).**
> The shipped `EigenrotatedAnisotropicChernoff` did NOT deliver the `D·q` node
> reduction that is the entire reason F5 exists. The implementation loops the full
> `q^D` tensor quadrature (`n_quad = q.pow(D)`) and back-rotates **each** node
> point-wise through `Q` (`q_rot.mat_vec` per node), then samples via the `2^D`-corner
> multilinear `GridFnND::sample` — making it **more expensive than its §32 sibling**
> (full tensor quadrature AND a per-node resample). Results are numerically correct
> but the headline `125×` / `D·q` claim is false. Per the v8.0.0 directive ("no
> crutches, strong result"), F5 is **withdrawn from the v8.0.0 ship** and this ADR is
> demoted to EXPLORED / DEFERRED. See the new **"Demotion analysis"** section below for
> why the genuine `D·q` algorithm is blocked discretely, and why §32 already covers
> correctness. An engineer removes the code per the removal spec at the end of this ADR.

## Context (original — retained for audit; the "recovering D·q" premise is corrected by the Demotion analysis below)

The ζ² `AnisotropicShiftChernoffND` order-2 correction (ADR-0112 AM2, SHIPPED v7.0.0) and the Smolyak sparse-grid backend (ADR-0123, D=5 gated) together cover the dominant cost regimes for anisotropic D≥5 problems. A distinct remaining gap: when the diffusion tensor is NOT axis-aligned (off-diagonal `a_{ij} ≠ 0` with a large anisotropy ratio λ_max/λ_min ≫ 1), the tensor-product quadrature lattice aligns with the coordinate axes rather than the principal eigendirections, so `q^D` nodes are required even after Smolyak reduction. Rotating to the eigenbasis makes the operator axis-aligned by construction, recovering the Smolyak/adaptive-q node count `D·q` instead of `q^D`.

## Decision (ORIGINAL — REVERSED by the 2026-06-07 demotion; retained for audit trail only)

> **REVERSED.** The "Ship `EigenrotatedAnisotropicChernoff`... Node count drops from `q^D` to
> `D·q`... 125×" decision below is FALSE as implemented and unattainable as specced (see
> Demotion analysis). F5 is WITHDRAWN from v8.0.0. Read this section as the rejected proposal.

ACCEPTED for v8.0.0 as a MINOR, additive **cost optimization** (not a new accuracy class). Ship `EigenrotatedAnisotropicChernoff<F, D>` (new file `crates/semiflow-core/src/anisotropic_eigenrotated.rs`, math.md §48), a sibling to §32 `AnisotropicShiftChernoffND`. For a **constant** symmetric-positive-definite diffusion tensor `A` with off-diagonal entries, diagonalize `A = QΛQᵀ` **once at construction** (not per step — `A` is constant), then per step apply rotate-in → per-axis diagonal step → rotate-out (eq. 48.4). Node count drops from `q^D` to `D·q` (`D=5, q=5 ⇒ 3125 → 25`, 125×). This is distinct from Smolyak (sparsity in a *fixed* frame, ADR-0123) and from the ζ² correction (post-quadrature in the *original* frame, ADR-0112 AMENDMENT 2).

Three spike clarifications recorded as NORMATIVE (math §48):
1. **Rotation-exactness (Theorem 48.1).** `Q` is a *constant* orthogonal isometry that commutes with the time-stepping, so the kernel inherits the inner per-axis order with **zero** rotation error — certified symbolically by `T_EIGENBASIS` (D∈{3,5}, incl. the exact `e^{τA}=Qe^{τΛ}Qᵀ` identity; PASS 2026-06-07).
2. **Honest scope: constant-`A` ONLY** (§48.5). Spatially-varying `A(x)` makes `Q(x)` x-dependent, leaving a first-order `[Q(x),∂]` commutator residual that breaks exactness — explicitly DEFERRED, out of scope for v8.0.0. Variable-`A(x)` off-diagonal users keep using §32 (Cholesky-in-original-frame, correct at `q^D`). The constructor takes a *constant* `SquareMatrix<F,D>` (not a closure), making the scope explicit in the type.
3. **§32 already handles off-diagonal *correctness*** via Cholesky `A=LLᵀ`; F5's contribution is purely the node-count reduction by trading correlated `q^D` tensor quadrature for separable `D·q` per-axis quadrature in the principal frame.

**Gate decision (revising the original "no new gate"):** the correctness claim is symbolic and deterministic, so the load-bearing gate is `T_EIGENBASIS` (RELEASE_BLOCKING, symbolic, runs in `test-fast`). The numerical self-convergence slope gate `G_EIGENBASIS_ANISO_DDIM` (D≥5, slope ≤ −1.95, `slow-tests`) is declared **ADVISORY** — it is a regression sentinel for the per-axis interpolation floor (the same coarse-grid `O(n·dx²)` floor that forced the §32.5 G_DDIM AMENDMENT-1 ladder), not the accuracy proof. A floor-dominated flattening should warn, not block a MINOR cost optimization; the gate is promotable to RELEASE_BLOCKING in a later PATCH once the coarse-grid ladder is calibrated for the eigen-rotated stencil.

**Dependency decision:** NO new dependency. The `D×D` symmetric eigendecomposition is a hand-rolled cyclic Jacobi rotation (~40 LoC, `no_std`, reusing the in-crate `SquareMatrix<F,D>` from `shift_nd.rs`), not LAPACK/`nalgebra` — consistent with the library's zero-heavy-dep posture and the existing in-crate Cholesky precedent.

## Demotion analysis (2026-06-07 — why true D·q is blocked, and why §32 suffices)

**1. The Theorem 48.1 proof is a *continuous-operator* statement; the algorithm is *discrete*.**
§48.4 proves `e^{τL} = R_{Qᵀ}⁻¹ e^{τL̃} R_{Qᵀ}` exactly via the spectral mapping theorem,
treating `R_{Qᵀ}` as an exact isometry "removed identically on both sides." That holds for
the operator. The kernel, however, runs on a standard-axis **tensor grid**, and §48.3 itself
admits `R_{Qᵀ}` must "resample the grid function into eigen-coordinates `y = Qᵀx`." The
eigen-coordinates of a generic `Q` do **not** land on grid points.

**2. The genuine separable algorithm requires a grid-wide resample that breaks the exactness.**
To evaluate `∏_i S_i` as `D` independent 1D convolutions (the only way to reach `D·q` nodes),
the state must live on an **eigen-aligned tensor grid**. Producing that grid is a `D`-dimensional
resample: each eigen-node needs the `2^D`-corner multilinear `GridFnND::sample` — an `O(N·2^D)`
sweep over the **whole** grid (`N` = grid size), repeated for rotate-out. Multilinear
interpolation on a rotated grid is **not** an isometry (it is a smoothing contraction) and is
only `O(dx²)` accurate. So the discrete `R_{Qᵀ}` is NOT the exact isometry of the proof: the
rotate-in/rotate-out pair injects an `O(dx²)` per-step interpolation floor. **The "zero rotation
error" headline is unreachable on a tensor grid.**

**3. The node-count win survives, but at a NEW accuracy cost §32 does not pay.**
Cost-wise the true algorithm `O(N·(D·q + 2·2^D))` does beat the standard-frame
`O(N·q^D)` quadrature (the `O(N·2^{D+1})` resample is cheap vs. the `q^D−D·q` quadrature
saving). But §32 (`AnisotropicShiftChernoffND`) keeps the **original** grid, shifts the GH
stencil via Cholesky `A=LLᵀ`, and samples **once per node** with no rotate-in/out — so it has
**no resample floor**. The genuine D·q F5 trades §32's `q^D` quadrature cost for a resampling
`O(dx²)` accuracy penalty. This is a real (bounded) speedup, but it is a *cost-vs-accuracy*
trade, **not** the exact, free, 125× win that ADR-0137 / §48 specced.

**4. Current code is strictly dominated.** The shipped implementation does the **worst of both**:
full `q^D` tensor quadrature AND a `2^D` resample per node (`q^D·2^D` corner-ops/point), with no
D·q saving at all. It is slower than §32 for zero correctness benefit.

**Ruling (Option B — honest drop).** Because (a) §32 + its §32.8 ζ² order-2 lift already deliver
off-diagonal **correctness** with no resample floor, and (b) the true D·q algorithm is blocked
*as specced* — the zero-error claim is false discretely, and the honest version carries a new
`O(dx²)` resample floor plus two grid-wide interpolations per step — the F5 kernel as shipped
adds nothing and overclaims. We **withdraw** it from v8.0.0 rather than ship a dominated,
overclaimed kernel ("no crutches, strong result"). This is an honest defer: the genuine D·q
direction (with its resample-floor caveat) is named and left OPEN for a future minor; it is NOT
re-scoped into a face-saving "keep-but-correct" (Option C was rejected precisely because the only
honest non-D·q version of F5 is dominated by §32).

## Consequences (WITHDRAWN — superseded by the Demotion analysis above)

> The text below is the original (rejected) ship rationale, retained for audit trail only.
> F5 is NOT shipped in v8.0.0; an engineer removes the code per the removal spec.

~~Additive (new file + new public type; no mutation of existing kernels — §32 `AnisotropicShiftChernoffND` and §44 Smolyak are preserved verbatim). Reduces quadrature cost from `q^D` to `D·q` for **constant** strongly-anisotropic off-diagonal diffusion tensors at D≥5 (125× at D=5).~~ **FALSE — see Demotion analysis: code never delivered D·q.** Zero new dependency (in-crate Jacobi eigensolver). Scope was constant-`A` only. ~~Two new gates: `T_EIGENBASIS` (RELEASE_BLOCKING) and `G_EIGENBASIS_ANISO_DDIM` (ADVISORY).~~ **WITHDRAWN — see "Gate disposition" below.** Contract bumps reverted: traits.yaml and properties.yaml EigenrotatedAnisotropicChernoff/gate entries removed (re-scoped, see below).

## Gate disposition (post-demotion)

- **`G_EIGENBASIS_ANISO_DDIM`** — WITHDRAWN. There is no shipped kernel to gate.
- **`T_EIGENBASIS`** — retained ONLY as a standalone *continuous-operator* sympy oracle
  (`scripts/eigenbasis_rotation_kit.py`): it correctly certifies the spectral identity
  `e^{τA}=Qe^{τΛ}Qᵀ`, which remains true and useful as the math foundation for any future
  D·q kernel. It is **no longer a RELEASE_BLOCKING library gate** — it does not test the
  (now-removed) Rust kernel, and the discrete algorithm it would gate does NOT satisfy the
  identity (the resample floor breaks it). Demoted from RELEASE_BLOCKING to an informational
  symbolic note.

## Engineer removal spec (Option B execution — CODE, not docs; do NOT run here)

1. **Delete the kernel file** `crates/semiflow-core/src/anisotropic_eigenrotated.rs` in full
   (incl. the in-crate `jacobi_eigen`/`jacobi_cyclic`/`jacobi_2x2` eigensolver and the
   `qt_mul`/`accumulate_quad`/`build_node`/`flat_to_x` helpers — all are F5-only).
2. **Remove the module declaration and re-export** of `EigenrotatedAnisotropicChernoff` and
   `jacobi_eigen`/`qt_mul` from `crates/semiflow-core/src/lib.rs` (grep `eigenrotated`,
   `Eigenrotated`, `jacobi_eigen`, `qt_mul`).
3. **Delete the gate test** `tests/g_eigenbasis_aniso_ddim.rs`.
4. **Verify `qt_mul` / `SquareMatrix::mat_vec` have no other callers** before removal of
   `qt_mul` (it is `pub(crate)` and F5-introduced; `mat_vec` belongs to `shift_nd.rs` and is
   shared — KEEP `mat_vec`, remove only the F5-local `qt_mul`).
5. **Keep** `scripts/eigenbasis_rotation_kit.py` (now an informational continuous-operator
   oracle per "Gate disposition"); remove its RELEASE_BLOCKING wiring from any CI/test runner
   manifest that treats `T_EIGENBASIS` as blocking.
6. Build/test (`cargo run -p xtask -- test-fast`) AFTER the concurrent engineer's edits land —
   confirm no dangling `EigenrotatedAnisotropicChernoff` references remain and the sympy sweep
   no longer requires `T_EIGENBASIS` to pass for release.
7. Contract YAML and math §48 doc edits are ALREADY done by the architect (this ruling) — the
   engineer touches only Rust + test wiring.
