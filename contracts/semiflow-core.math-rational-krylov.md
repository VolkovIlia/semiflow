# §66 — Rational (shift-and-invert) Krylov action (ADR-0206, NORMATIVE library)

## §66.1 — Operator and projection

`A` symmetric positive semidefinite (CSR, §55), `M = diag(m) > 0` (default `I`),
`t ≥ 0`, `γ > 0` (default `t/10`). `Z = (M/γ + A)⁻¹M` is self-adjoint in
`⟨x, y⟩_M = xᵀMy` with eigenpairs `(θ_j, u_j)`, `θ_j = 1/(1/γ + λ_j) ∈ (0, γ]`, where
`M⁻¹A u_j = λ_j u_j`. Hence `e^{−tM⁻¹A} = f(Z)` with
`f(θ) = exp(t/γ − t/θ)` (`θ ≤ 0 ↦ 0`).

Lanczos in `⟨·,·⟩_M` from `q_1 = v/‖v‖_M` with full re-orthogonalisation gives
`Z Q_m = Q_m T_m + β_m q_{m+1}e_mᵀ`; the approximation is
`y_m = ‖v‖_M Q_m f(T_m) e₁`.

## §66.2 — Stopping rule: a-priori dimension (NORMATIVE)

For `Z` self-adjoint with spectrum in `[0, γ]` and Lanczos exact on polynomials of
degree `< m`, `‖f(Z)v − y_m‖_M ≤ 2·E_{m−1}(f; [0, γ])·‖v‖_M` with `E_{m−1}` the best
uniform polynomial approximation error (Saad 1992; Hochbruck & Lubich 1997), and
`E_{m−1} ≤ Σ_{k≥m}|c_k|` where `f(γ(x + 1)/2) = Σ_k c_k T_k(x)` on `[−1, 1]`, i.e.
`f = exp(r(1 − 2/(x + 1)))`, `r = t/γ`. Define

`m_min(tol, r) = min{m ≥ 1 : 2·Σ_{k≥m, |c_k| > 64ε} |c_k| ≤ tol}`,

with `c_k` from the interpolant at `N = 2·m_max + 65` Chebyshev points (phases
`(2i + 1)kπ/(2N)` reduced exactly mod `4N`); coefficients `≤ 64ε` are the rounding
plateau of the transform and are not summed. The iteration returns `y_m` at
`m = m_min`, or earlier when `β_m ≤ ε·γ` or `m = n` (invariant subspace: exact in
exact arithmetic). `m_min` undefined within `m_max`, and `n > m_max` →
`ConvergenceFailed` at `m = m_max`. Reference values for `r = 10` (`γ = t/10`, the
default): `m_min = 16, 21, 27, 33` at `tol = 1e−6, 1e−8, 1e−10, 1e−12`.

`est_error = ‖y_m − y_{m−1}‖_M/‖v‖_M = ‖c_m − [c_{m−1}; 0]‖₂` (diagnostic only). A
stop on small a-posteriori differences is not allowed: while every Ritz value is
stiff, `f ≈ 0` on all of them and a small slow component not yet in the Krylov
space is lost.

## §66.3 — Factor of `S = M/γ + A` (NORMATIVE)

If `A` is tridiagonal with every stored off-diagonal `≤ 0` and every row excess
`sᵢ = mᵢ/γ + aᵢᵢ + βᵢ₋₁ + βᵢ ≥ 0` (`βᵢ = a_{i,i+1}`, duplicates summed; `sᵢ` formed
with TwoSum compensation), then `S = LDLᵀ` with `l_i = βᵢ₋₁/dᵢ₋₁`,
`dᵢ = −βᵢ + eᵢ`, `e₀ = s₀`, `eᵢ = sᵢ + (−βᵢ₋₁)·eᵢ₋₁/dᵢ₋₁` (all terms `≥ 0`).
Identity: `dᵢ = δᵢ − βᵢ₋₁²/dᵢ₋₁` with `δᵢ = sᵢ − βᵢ₋₁ − βᵢ`. A non-positive or
non-finite pivot, or any other pattern, uses `SpdResolvent` (§62.2; PCG inner
tolerance `0.1·tol`). `v` is prescaled by `2^{−e}`, `e = ⌊log₂‖v‖∞⌋`, and the result
by `2^{e}` (bitwise homogeneity); `t·‖M⁻¹A‖ < ε` returns `v`.

## §66.4 — Accuracy

Away from rounding the error is bounded by §66.2, uniformly in `λ_max`. On the
§66.3 factor the slow eigenvalues of `S` carry relative error `O(nε)`, so the result
has no `λ_max t`-dependent rounding floor relative to the stored operator (gate
carriers: `≤ 3.4e−12·‖v‖∞` up to `λ_max t = 1e10`). The textbook `LDLᵀ` loses
`≈ log₁₀(λ_max t)` digits (`4.9e−9` at `1e10`). Rounded stored row sums perturb the
operator itself by `O(ε‖A‖)`, which `e^{−tA}` turns into `O(ε·λ_max t)` against the
ideal operator.

## §66.5 — Gates

`G_RATKRYLOV_NEUMANN_ORACLE`, `G_RATKRYLOV_SOLVE_COUNT`, `G_RATKRYLOV_MASS_DENSE`,
`G_RATKRYLOV_ISSUE16` (ADR-0206 §Gates).
