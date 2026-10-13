# ADR-0213 — Cross-diffusion with a constant matrix: modal splitting

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: amends ADR-0208 (the diffusion half of
  `ReactionDiffusion`); additive (the number was reserved in ADR-0205).
- **Contract**: math §70 (`contracts/semiflow-core.math-cross-diffusion.md`); gates
  `G_CROSSDIFF_GRAPH_EXACT`, `G_CROSSDIFF_STRANG_ORDER`, `G_CROSSDIFF_FOURIER_GRID`,
  `G_CROSSDIFF_REJECT`.

## Context

`ReactionDiffusion` (ADR-0208) diffuses every species on its own:
`∂ₜu_k = D_k Δu_k + f_k`. Coupled transport — multicomponent (Maxwell–Stefan
linearised) diffusion, electrolytes, chemotaxis-type linearisations, cross-diffusion
population models — has a full constant matrix `∂ₜu = DΔu + f(t, x, u)`. The only
coupled engine, `MatrixDiffusionChernoff`, is a 1-D block Crank–Nicolson with
Neumann hard-coded: no 2-D/3-D, no periodic or graph geometry, no septic
interpolation.

## Decision

1. **Modal splitting, exact in `D`.** For `D = PΛP⁻¹` with real `λ_k ≥ 0`
   (diagonalisable), pointwise mixing commutes with any scalar linear operator `L`
   applied identically to every species, so `e^{tD⊗L} = (P ⊗ I)·diag_k(e^{tλ_k L})·
   (P⁻¹ ⊗ I)`. A diffusion step mixes the species into modes `m = P⁻¹u`, applies
   mode `k`'s engine (`≈ e^{τλ_k L}`, built by the caller; the identity for
   `λ_k = 0`) and mixes back `u = Pm`; the reaction still acts on the physical
   species, so the Strang splitting (and Richardson extrapolation) of ADR-0208 is
   unchanged. Every existing engine (1-D/2-D/3-D, all interpolants and boundary
   policies, Strang/ADI compositions, graph Laplacians) is reused as is.
2. **`CrossDiffusion<F>`** (`src/cross_diffusion.rs`). `new(d: &[f64])` (row-major
   `K×K`, `1 ≤ K ≤ 16`; `D` is `f64` data for every scalar type — a constant of the
   step, no tangent to drop): Hessenberg reduction and Francis double-shift QR
   (`elmhes`/`hqr`, Wilkinson & Reinsch) for the eigenvalues, eigenvectors as null
   spaces of `D − λ̄I` by complete-pivoting elimination per eigenvalue cluster
   (relative width `1e−6`), whose dimension must equal the multiplicity. The result
   is VERIFIED before use: `cond₁(P) ≤ 10⁸` and `‖PΛP⁻¹ − D‖_max ≤ 10⁻¹⁰‖D‖_max`.
   Complex eigenvalues (`|Im| > 1e−6‖D‖`), negative ones (backward diffusion,
   ill-posed) and defective or nearly defective `D` are a `DomainViolation` naming
   the reason. `from_modes(p, λ)` takes a user decomposition (conditioning checked);
   `eigenvalues()`, `modes()`, `inverse_modes()`.
3. **`ReactionDiffusion::with_cross_diffusion(cd) → Result`** (additive; `K` must
   match). The engines become the mode engines (`engines[k]` for
   `λ_k = cd.eigenvalues()[k]`, ascending).
4. **Boundary admissibility.** Mixing commutes with `L` only when every species
   carries the same nodes, interpolant and homogeneous linear boundary policy (a
   non-zero Dirichlet value would be scaled by `P⁻¹`). `NodalField` gains the
   provided method `mixes_with(&self, other) -> bool` (default `false`: an
   unverifiable field is rejected): grids compare geometry, interpolant and policy
   per axis and refuse `Dirichlet { value ≠ 0 }`; graph signals need the same
   `Arc<Graph>`. Checked at every run; the `NodalField` impls move to
   `reaction_diffusion_fields.rs`.
5. **Python.** `ReactionDiffusion1D/2D/3D(…, diffusivity=[[…], …])`: a `K×K` nested
   list (or 2-D array) selects cross-diffusion; the binding builds the mode engines
   from the eigenvalues (`Frozen` for `λ = 0`). Every Python boundary policy is
   homogeneous linear. Engine builders move to `rd_engines_py.rs`.

## Consequences

- The modal step is exact in `D`: the cross-diffusion error IS the scalar engines'
  error on the modes, bounded by `max_i Σ_k |P_ik||(P⁻¹c)_k|·err_k` (gated, tight to
  rounding). Cost per step: `K` scalar engine applications plus two `K×K` mixings per
  node (`O(K²N)`).
- Default behaviour (no cross-diffusion) is bit-identical: the gather/scatter of the
  reaction step is the same code moved into two helpers.

## Honest limits

- Constant `D` only; `D(u)` (SKT-type nonlinear cross-diffusion) is not covered.
- `D` must be diagonalisable with real non-negative eigenvalues. Real `D` with
  complex eigenvalues is well-posed when the real parts are positive, but the modal
  route would need complex engines — rejected with the reason.
- A conditioning of `P` near the `10⁸` limit amplifies the mode engines' errors by
  up to `cond(P)` (the gate bound shows the factor).
- Species on different grids or with different boundary policies cannot be mixed.

## Gates

- `G_CROSSDIFF_GRAPH_EXACT` (`tests/g_cross_diffusion.rs`): ring graph (48 nodes),
  discrete Fourier mode, exact Chebyshev graph engines (`tol 1e−15`), `D` ∈
  {non-symmetric `2×2`, singular `2×2` (a non-diffusing species driven by the other's
  Laplacian), non-normal `3×3`}: result `= e^{−TμD}c` to `≤ 1e−12` relative
  (measured `≤ 2.0e−15`).
- `G_CROSSDIFF_STRANG_ORDER`: the same with a linear reaction `u′ = Cu` that does not
  commute with `D`: order 2 to `e^{T(−μD + C)}c` (slopes `∈ [1.9, 2.1]` over
  `n = 40, 80, 160`; measured 2.00–2.06). A reaction applied to the modes instead of
  the species converges to a different limit and fails.
- `G_CROSSDIFF_FOURIER_GRID`: periodic `sin(2πx)·c` (1-D, 64 nodes) and
  `sin(2πx)sin(2πy)·c` (2-D, `32²`) with `DiffusionChernoff` (septic, Strang across
  axes) against the closed form: error `≤` the modal bound from each scalar mode
  engine's own error (`+1e−14`), bound `≤ 1e−4`, slopes `∈ [1.9, 2.1]` over 20, 40,
  80 steps (measured 1.99–2.02). Mixing with `P` in place of `P⁻¹` fails all three
  accuracy gates.
- `G_CROSSDIFF_REJECT`: complex, negative, defective and nearly defective `D`
  rejected with the reason; non-zero Dirichlet values, species on different grids or
  policies, and a species-count mismatch rejected at run time.

## Found while gating (not changed here)

`BoundaryPolicy::Periodic` documents a period of `(n − 1)·dx`, and the coordinate
wrap of `sample` follows it, but the ghost-node index wrap of the engines uses
`n` nodes (period `n·dx`): a function of period `xmax − xmin` sampled on all `n`
nodes diffuses with an `O(dx)` error (`2.2e−2` at `n = 65` for `sin(2πx)`), while
period `n·dx` grids are accurate (`4.7e−6`). The gates use the engines' convention
(`[0, 1 − 1/n]`). Unifying the two conventions changes bits of periodic engines and
needs its own ADR (ROADMAP).
