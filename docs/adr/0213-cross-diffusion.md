# ADR-0213 — Cross-diffusion with a constant matrix: modal splitting

- **Status**: Accepted
- **Date**: 2026-10-13
- **Supersedes / amends**: amends ADR-0208 (the diffusion half of
  `ReactionDiffusion`); additive (the number was reserved in ADR-0205).
- **Contract**: math §70 (`contracts/semiflow-core.math-cross-diffusion.md`); gates
  `G_CROSSDIFF_GRAPH_EXACT`, `G_CROSSDIFF_STRANG_ORDER`, `G_CROSSDIFF_FOURIER_GRID`,
  `G_CROSSDIFF_REJECT`.
- **Amendment 1 (review)**: fixed-rank null spaces with adaptive clustering,
  power-of-two normalisation, `f64` verification for every `F`, reasons asserted
  by the gates (Decision 2, Gates).

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
   unchanged. Every existing engine (1-D/2-D/3-D, all interpolants, every
   homogeneous linear boundary policy — all except `Dirichlet { value ≠ 0 }`, see
   4 — Strang/ADI compositions, graph Laplacians) is reused as is.
2. **`CrossDiffusion<F>`** (`src/cross_diffusion.rs`). `new(d: &[f64])` (row-major
   `K×K`, `1 ≤ K ≤ 16`; `D` is `f64` data for every scalar type — a constant of the
   step, no tangent to drop). The iteration runs on the exactly normalised
   `2^{−e}D` (`‖D‖_max = m·2^e`, `frexp`/`scalbn`), so it is scale-invariant over
   the whole `f64` range: Hessenberg reduction and Francis double-shift QR
   (`elmhes`/`hqr`, Wilkinson & Reinsch) for the eigenvalues; a non-finite one is
   reported as non-convergence. `|Im λ| > 1e−6·s` is rejected as complex,
   `λ < −1e−6·s` as negative; smaller ones are dropped/clamped, and a
   reconstruction failure they cause is reported as complex/negative.
   Eigenvectors: consecutive eigenvalues within `w·s` form a cluster of size `m`
   whose basis is the `m`-dimensional null space of `D − λ̄I` — exactly `K − m`
   steps of complete-pivoting elimination (no pivot threshold); a cluster whose
   basis is not an eigenbasis to `1e−10·s/K` is split at the next width
   `w ∈ {1e−6, 1e−9, 1e−12, 1e−14, 0}`. The `f64` decomposition is VERIFIED before
   it is rounded to `F`: `cond₁(P) ≤ min(10⁸, 10⁻²/ε_F)` and
   `‖PΛP⁻¹ − D‖_max ≤ 10⁻¹⁰·‖D‖_max`; a defective `D` (a Jordan block keeps a
   non-eigen basis and fails the reconstruction) or a nearly defective one
   (`cond₁(P)` too large) is a `DomainViolation` naming the reason.
   `from_modes(p, λ)` takes a user decomposition (conditioning checked);
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
  error on the modes, bounded by `max_i Σ_k |P_ik||(P⁻¹c)_k|·err_k` (gated; attained
  to rounding when the worst row's terms share a sign, as for the `2×2` gate
  matrices; 2–4 % below it for the non-normal `3×3`, whose terms have mixed signs). Cost per step: `K` scalar engine applications plus two `K×K` mixings per
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
- `G_CROSSDIFF_REJECT` (`tests/g_cross_diffusion_reject.rs`): every rejection
  asserts its reason — complex, negative (also within the clamp tolerance),
  defective (`does not reproduce D`), nearly defective (`cond₁(P) = 2e10`:
  `ill-conditioned`), `from_modes` singular / ill-conditioned; at run time a non-zero
  Dirichlet value and species that differ in exactly ONE of policy, `xmin`, `xmax`,
  interpolant (1-D), one axis (2-D) or the graph, and a species-count mismatch.
  Unit tests: well-conditioned non-normal and close-spectrum `D` accepted, `c·D`
  for `c ∈ [1e−300, 1e300]`, `f32` = rounded `f64` decomposition.

## Found while gating

`BoundaryPolicy::Periodic` was documented as wrapping `x` with period `(n − 1)·dx`.
The code does not: `GridFn1D::sample` (Hermite and linear interpolants) wraps the
node index modulo `n` (`bc_index`), so the period is `n·dx` (node `n` is node `0`),
and the grid engines share that convention; only the `ChebyshevSpectralWithBC`
out-of-domain path (`wrap_periodic`) wraps with `xmax − xmin`. A function of period
`xmax − xmin` sampled on all `n` nodes therefore duplicates the seam node and
diffuses with an `O(dx)` error (`2.2e−2` at `n = 65` for `sin(2πx)`), while
period-`n·dx` grids are accurate (`4.7e−6`). The doc comment is corrected here; the
gates use the code's convention (`[0, 1 − 1/n]`). Whether the Chebyshev path should
follow `n·dx` is bit-changing and left to its own ADR (ROADMAP).
