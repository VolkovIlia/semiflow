# ADR-0124 — 2D/3D matrix-valued kernels via per-axis matrix-Strang lift (`MatrixDiffusionChernoff2D/3D`); separability + palindromic symmetry yield order-2 even with non-commuting reaction coupling

- **Status**: Accepted (PRE-FLIGHT GO; engineer wave authorised — additive `MatrixDiffusionChernoff2D/3D<F, M>` + `MatrixAxisLift`)
- **Date**: 2026-06-05
- **Decision-maker**: ai-solutions-architect
- **Backlog item**: v7.0.0 freeze item #24 (`.dev-docs/v7.0.0-backlog-freeze.md`). NOTE: the freeze *predicted* ADR-0131; actual sequential allocation lands this at **0124** (0120–0123 already consumed, same divergence the freeze anticipated for the 012x band).
- **Depends on**: ADR-0082 (`MatrixDiffusionChernoff<F, M>` 1D matrix kernel — the per-axis ingredient), ADR-0012 (`Strang2D`/`AxisLift` tensor pattern), ADR-0024 (`Strang3D`/`Axis::Z` inductive lift), constitution v2.0.0 (3/3 dep cap INVIOLATE — adds ZERO deps).
- **Mathematical foundation**: `scripts/verify_matrix_2d3d_preflight.py` (executed 2026-06-05). All three checks PASS — see §PRE-FLIGHT.
- **Acceptance gates added**: `G_MATRIX_2D`, `G_MATRIX_3D` (slow-tests-gated self-convergence slope; threshold `slope ≤ −0.80` per the documented matrix-kernel lower-slope, freeze item #24).

## Context

`MatrixDiffusionChernoff<F, M>` (ADR-0082) ships coupled-component diffusion in 1D, applying `exp(τC(x))` per grid point via Cayley-Hamilton (M≤4) inside a palindromic reaction-diffusion-reaction Strang step. math §33.5 declared 2D/3D "no new mathematical content … via Strang composition" but deferred it. The engineering reality is sharper: the existing `Strang2D<X,Y,F>`/`AxisLift<C,F>` are hardwired to `S = GridFn1D<F>` (scalar fields), whereas the matrix kernel has `S = MatrixGridFn1D<F, M>`. A new `MatrixAxisLift` (component-aware pencil gather/scatter on a `MatrixGridFn2D`) is required; the composition's order-2 claim must be verified, NOT assumed — particularly because the per-axis reaction matrices `C_x`, `C_y` do **not** commute in general.

## Decision

Ship additive `MatrixDiffusionChernoff2D<F, M>` and `MatrixDiffusionChernoff3D<F, M>` as dedicated types (mirroring the `Strang2D`/`Strang3D` "dedicated, do-not-generalise" decision of ADR-0012/0024). Composition is palindromic Strang over per-axis `MatrixAxisLift`s. **Verified order-2** rests on two facts the PRE-FLIGHT confirms: (1) the spatial-diffusion lifts commute exactly across axes (separability holds for M-component lifts — Kronecker-factor disjointness), and (2) the palindromic symmetry annihilates the τ² BCH term even when `[C_x, C_y] ≠ 0`, so reaction non-commutation does NOT break order-2. The reaction half-step reuses the existing M×M matrix-exponential machinery verbatim. 3D follows inductively (math §10.8 Thm 7'). Zero new dependencies.

## PRE-FLIGHT sympy (executed 2026-06-05; `scripts/verify_matrix_2d3d_preflight.py`)

Finite tensor model (N=2/axis, M=2): Kronecker-lifted `Lx = Dx⊗Iy⊗I_M + Ix⊗Iy⊗Cx`, `Ly = Ix⊗Dy⊗I_M + Ix⊗Iy⊗Cy`, with distinct stencils `Dx,Dy` and distinct **non-commuting** coupling `Cx` (skew), `Cy` (symmetric), `[Cx,Cy]=diag(4,−4)≠0`.

- **C1a** `[Lx_diff, Ly_diff] = 0` exactly — separability holds for M-component lifts.
- **C2** `Φ(τ) − e^{τ(Lx+Ly)}` has τ¹ and τ² coefficient matrices **exactly zero**; τ³ nonzero (canonical Strang local error) ⟹ global order-2 **despite** non-commuting reaction.
- **C3** reaction-only Strang `e^{τ/2 Cx} e^{τ Cy} e^{τ/2 Cx}` matches `e^{τ(Cx+Cy)}` to O(τ²) — per-axis matrix-exp reuse is order-2-consistent.

VERDICT: **GO** — order-2 sympy-verified; 3D inductive.

## Consequences

- **Engineer wave** builds `MatrixAxisLift` + `MatrixGridFn2D`/`3D` + the two evolvers; gates `G_MATRIX_2D`/`G_MATRIX_3D` self-convergence `slope ≤ −0.80`. M≤4 only (M≥5 is ADR-0125).
- **Honest scope**: the `≤ −0.80` (not `−1.95`) threshold is the documented matrix-kernel lower slope (freeze #24) — the matrix reaction half-step's per-point exp cost and the cross-axis reaction non-commutation degrade the *measured* asymptotic constant, not the order class; the τ²-zero proof above guarantees the order class is 2.
- **No coupling to scalar Strang2D**: dedicated types keep stable v0.5.0+ tensor code untouched.

## Amendment 1 (2026-10) — binding coupling split per axis

**Defect.** The Python (`MatrixDiffusion2D/3D`), C FFI (`smf_matrix2d/3d_*`) and WASM
bindings document `∂ₜu = a∂²u + C·u` but built EVERY axis kernel with the full coupling
`C`. Each axis kernel integrates its reaction half-steps over a total time τ per step
(`Lx(τ/2)·Ly(τ)·Lx(τ/2)`: x twice at τ/2, y once at τ; the 3-D five-leg composition
likewise), so the composition applied `Σ_axes C = d·C`: `e^{2tC}` in 2-D, `e^{3tC}` in 3-D.
The core types were correct — they take per-axis kernels whose reactions sum to the
total (`C_x + C_y` in the PRE-FLIGHT model above); only the binding builders were wrong.
No test covered `c ≠ 0` in 2-D/3-D.

**Fix.** Each binding passes `C/d` to every axis kernel (`d` = 2 or 3). Results for
`c_coupling ≠ 0` change (they were wrong); `c_coupling = 0` is bit-identical.

**Gate `G_MATRIX_ND_COUPLING_EXACT`.** A uniform field with Neumann ends is a fixed
point of the block-CN diffusion step (up to Thomas rounding), so the result must be
`e^{tC}u₀` at every node: `‖u(t) − e^{tC}u₀‖∞ ≤ 1e−12·‖u₀‖∞`, `C = [[0, 1], [1, 0]]`,
`t = 0.5`, 2-D 9×7 and 3-D 7×6×5. FFI (`semiflow-ffi/tests/g_matrix_nd_coupling_exact_ffi.rs`),
WASM (native unit test, run by `cargo test -p semiflow-wasm --features full --lib` in CI)
and Python (`test_new_operators.py::test_matrix_nd_coupling_exact`). Without the fix the
error is 0.74 (2-D) and 2.03 (3-D).
