# ADR-0206 — Rational (shift-and-invert) Krylov action for stiff SPD operators

- **Status**: Accepted
- **Date**: 2026-10-11
- **Supersedes / amends**: none (additive; the number was reserved in ADR-0205).
- **Contract**: math §66 (`contracts/semiflow-core.math-rational-krylov.md`); gates
  `G_RATKRYLOV_NEUMANN_ORACLE`, `G_RATKRYLOV_SOLVE_COUNT`, `G_RATKRYLOV_MASS_DENSE`,
  `G_RATKRYLOV_ISSUE16`; no_std scenario `ratkrylov_stiff`.

## Context

The polynomial Krylov paths (single-expansion Chebyshev, Lanczos; ADR-0205) need
`O(√(λ_max t))` mat-vecs: 47 188 at `λ_max t = 1e8`, 471 877 at `1e10`. Stiff SPD
problems — fine meshes, large diffusivities, long horizons, the issue #16 operator
(`λ_max ≈ 4·10⁷`) — sit exactly there. The implicit-Euler path (§59) is cheap but
first-order in `τ = t/n_steps`: a mode with `λτ = O(1)` is damped by `(1 + λτ)^{−n}`
instead of `e^{−λt}`.

## Decision

1. **Shift-and-invert Lanczos** (van den Eshof & Hochbruck, SIAM J. Sci. Comput. 27
   (2006) 1438–1457) on `Z = (M/γ + A)⁻¹M`, self-adjoint in `⟨x, y⟩_M = xᵀMy`
   (`M = diag(mass)`, default `I`), spectrum `θ = 1/(1/γ + λ) ∈ (0, γ]`. Result
   `y = ‖v‖_M·Q_m f(T_m)e₁` with `f(θ) = e^{t/γ − t/θ}` (`= e^{−tλ}`), `θ ≤ 0 → 0`.
   The stiff modes all map to `θ ≈ 0` where `f` is flat, so the number of solves is
   independent of `λ_max`. Default `γ = t/10` (within one solve of the best ratio
   `t/γ` for `tol ∈ [1e−10, 1e−6]`, within three at `1e−12`).
2. **One factor, many solves; cancellation-free on tridiagonal M-matrices.** The
   factor of `S = M/γ + A` is built once and reused by every Krylov step and every
   vector. When `A` is tridiagonal with non-positive off-diagonals (every path-graph
   Laplacian) and every row excess `sᵢ = mᵢ/γ + Σⱼ aᵢⱼ` is `≥ 0`, the `LDLᵀ`
   pivots are formed as `dᵢ = |βᵢ| + eᵢ`, `e₀ = s₀`,
   `eᵢ = sᵢ + |βᵢ₋₁|·eᵢ₋₁/dᵢ₋₁` — algebraically the textbook
   `dᵢ = δᵢ − βᵢ₋₁²/dᵢ₋₁`, but a sum of non-negative terms, so each excess has
   relative error `O(iε)`; the row sums use error-free transformations (TwoSum), so
   the factor is accurate for the operator as stored. The textbook form subtracts two
   `O(‖A‖)` numbers to obtain an excess of size `O(1/γ)` and loses
   `≈ log₁₀(λ_max t)` digits of the slow eigenvalues (`4.9e−9` at `λ_max t = 1e10`
   on the gate carrier, the floor an earlier draft of this ADR attributed to every
   shift-and-invert method). Every other pattern, a positive off-diagonal, a
   negative excess, or a forced `SpdSolver::Pcg`, uses `SpdResolvent` (tridiagonal
   `LDLᵀ`, or PCG with IC(0)/Jacobi at inner tolerance `0.1·tol`) through its
   checked `solve_into`.
3. **Stopping rule: a-priori dimension.** For self-adjoint `Z` with spectrum in
   `[0, γ]`, `‖f(Z)v − y_m‖_M ≤ 2·E_{m−1}(f; [0, γ])·‖v‖_M` (Saad 1992; Hochbruck &
   Lubich 1997) and `E_{m−1} ≤ Σ_{k≥m}|c_k|` for the Chebyshev coefficients `c_k` of
   `f`, which depend only on `r = t/γ`. `RationalKrylov::new` computes `m_min`, the
   smallest `m` with `2·Σ_{k≥m}|c_k| ≤ tol` (interpolant at `2·m_max + 65` Chebyshev
   points, phases reduced exactly; coefficients below `64ε` are rounding noise and are
   not summed), and the iteration runs exactly `m_min` solves — 16 / 21 / 27 / 33 at
   `tol = 1e−6 / 1e−8 / 1e−10 / 1e−12` with `γ = t/10` — or stops earlier on an
   invariant subspace (`β ≤ ε·γ`, or `m = n`). `m_min > m_max` (and `n > m_max`) →
   `ConvergenceFailed { last_residual: a-posteriori estimate, max_iter: m_max }`.
   An a-posteriori rule (two consecutive small coefficient differences, the first
   draft) is unsafe: when every early Ritz value is stiff, `f ≈ 0` on all of them,
   the differences are tiny, and a small slow component that has not yet entered the
   Krylov space is dropped (start vector `cos(0.97πi) + 10⁻⁶`: error `1e−6` after 4
   solves). Full `M`-reorthogonalisation; `f(T)e₁` is evaluated once at the end with
   the QL code of the Lanczos path (`tridiag_fn_e1`; the old `tridiag_exp_e1`
   delegates to it, bit-identical). `est_error` reports `‖y_m − y_{m−1}‖_M/‖v‖_M`
   as a diagnostic.
4. **Scaling.** `v` is prescaled by an exact power of two (`‖v‖∞ ∈ [1, 2)`): the
   result is bitwise homogeneous under `v ↦ 2ʲv` and `‖v‖_M` cannot under- or
   overflow. `t·‖M⁻¹A‖ < ε` (including `t = 0`) is the identity without a factor;
   `1/γ` must be finite.
5. **API** (additive): `RationalKrylov<F>` (`new`, `apply_into → RationalReport`,
   `n`, `t`, `gamma`), `RationalOptions` (`tol`, `gamma`, `m_max`, `solver`;
   `with_*` setters), `RationalReport` (`solves`, `krylov_dim`, `est_error`,
   `inner_iterations`), `rational_krylov_expmv`, and `RationalKrylovChernoff`
   (a `ChernoffFunction` on `GraphSignal`; factor built for the construction `τ`, a
   different `τ` builds a temporary factor). New structs are `#[non_exhaustive]`;
   the published `KrylovPath` enum is not extended. Python:
   `SymmetricOperator.rational_krylov(t, mass=None, tol, gamma, m_max, solver,
   precond) → RationalKrylov` (`apply`, `apply_batched`, `t`, `gamma`, `n`),
   `evolve_batched(path="rational")`, `mass_lumped_evolve(path="rational")` (uses `M`
   directly, no congruence).

## Consequences

Neumann path Laplacian, `tol = 1e−10` (gate carriers, `n = 400` and `4000`,
`λ_max t ∈ [1e2, 1e10]`): 27 solves everywhere, error `≤ 3.4e−12·‖v‖∞`
(`9.4e−16` at `λ_max t = 1e10`), against 49 … 471 877 Chebyshev mat-vecs. The issue
#16 operator (×1e7, `t = 0.7`): `3.3e−14·‖v‖∞` in 27 solves. A 40×40 grid
Laplacian on the PCG path (IC(0)): `≤ 1.8e−13·‖v‖∞` up to `λ_max t = 1e8`, about
`13 … 90` CG iterations per solve.

## Honest limits

- **Fixed cost.** The a-priori count is a worst case over `[0, γ]`: 27 solves at
  `tol = 1e−10` even where fewer would do (the unsafe a-posteriori draft stopped
  after 6–22). The count does not grow with `λ_max t` or `n`.
- **Stored row sums.** When the stored diagonal of `A` is rounded (non-dyadic
  weights), the operator's own null space is perturbed by `O(ε‖A‖)`, which
  `e^{−tA}` amplifies to `O(ε·λ_max t)` against the ideal operator: a property of
  the data, not of the method (the factor is exact for the stored matrix).
- **Tolerance plateau.** Chebyshev coefficients below `64ε` are not summed, so a
  `tol` below `≈ 1e−13` is met only to that plateau.
- **PCG path.** Accuracy follows the inner tolerance `0.1·tol`; CG iterations per
  solve grow like `√(1 + γλ_max)`. A singular `A` with a pattern that is not a
  tridiagonal M-matrix has no cancellation-free factor; no floor was measured on
  the grid carrier, but none is gated beyond `G_RATKRYLOV_MASS_DENSE`.
- `m_max` basis vectors are held (`64·N` values by default).
- The projected coefficients and the bound are `f64` scalars: `F = f32` runs at
  `f32` accuracy; float types wider than `f64` (forward-mode `Dual`) are rejected
  with `Unsupported` rather than returning tangents without the coefficient
  derivatives.
- Only symmetric (self-adjoint in the `M`-inner product) operators; non-symmetric
  generators use the Arnoldi path (ADR-0211).

## Gates

- `G_RATKRYLOV_NEUMANN_ORACLE` / `G_RATKRYLOV_SOLVE_COUNT`
  (`tests/g_rational_krylov.rs`): cosine-eigenbasis oracle, `n = 400` at
  `λ_max t ∈ {1e2, 1e4, 1e6, 1e8, 1e10}` and `n = 4000` at `{1e6, 1e8}` (slow modes
  alive at ≥ 100× the allowance), error `≤ (tol + 1e−11)·‖v‖∞` at every stiffness;
  the adversarial start vector `cos(0.97πi) + 10⁻⁶` at `λ_max t ∈ {1e4, 1e6}` to the
  same allowance; ≤ 30 solves everywhere; Chebyshev degree ≥ 100× the solve count at
  `λ_max t ≥ 1e6`.
- `G_RATKRYLOV_MASS_DENSE`: `n = 12`, lumped mass, tridiagonal and PCG paths,
  `t ∈ {0.01, 0.3, 2, 40}` against `M^{−½}e^{−tÂ}M^{½}v`: `≤ 1e−10‖v‖∞`.
- `G_RATKRYLOV_ISSUE16`: singular Neumann ×1e7, `n = 400`, `t = 0.7`, `O(1)`
  constant component: `≤ 1e−12‖v‖∞`.
- no_std `ratkrylov_stiff`: `n = 64`, ×1e4, `λ_max t = 1e3` (slow modes alive),
  `≤ (tol + 1e−11)‖v‖∞`, digest.
