# §68 — Adaptive Krylov (Arnoldi) φ-combination (ADR-0211, NORMATIVE library)

## §68.1 — Problem

For a generator `G` (any `GeneratorAction`), `τ ≥ 0` and `w_0 … w_p` (`p ≤ 3`),
`y = Σ_{k≤p} τᵏφ_k(τG)w_k` is the first block of `e^{B̃_η}y₀` with
`B̃_η = [[τG, ηW], [0, J_p]]` and `y₀ = [w₀; 0; …; 1/η]` (§62.4, the η-scaling of the
Taylor path). `expmv_krylov` is the case `p = 0`, `G = −A`.

## §68.2 — Substep (NORMATIVE)

With `y` the current vector and `β = ‖y‖₂`: one-pass modified Gram–Schmidt Arnoldi
`B̃_η V_m = V_m H_m + h_{m+1,m} v_{m+1}e_mᵀ`; augmented
`Ĥ = [[H_m, 0, 0], [h_{m+1,m}e_mᵀ, 0, 0], [0, 1, 0]] ∈ ℝ^{(m+2)²}`;
`f = e^{ΔtĤ}e₁` by `s` scaled Taylor steps of degree `m_T`, `(s, m_T)` the
Al-Mohy–Higham selector on `Δt‖Ĥ‖₁`; `φ₁ = β|f_m|`, `φ₂ = β|f_{m+1}|·‖B̃_η v_{m+1}‖₂`;
`err = φ₂` (`φ₁ > 10φ₂`), `φ₁φ₂/(φ₁ − φ₂)` (`φ₂ < φ₁ ≤ 10φ₂`), else `φ₁`. Accept iff
`err ≤ 1.2·Δt·tol·‖y₀‖₂`; then `y ← β·V_m f_{0..m}`. Reject:
`Δt ← round₂(0.9·Δt·(Δt·tol·‖y₀‖₂/err)^{1/m})` (exponent `1/(m−1)` in the third case),
at most 10 times. Next step: `round₂(0.9·Δt·(Δt·tol·‖y₀‖₂/err)^{1/m})` from the
accepted `err` (the remaining time when `err = 0`). Happy breakdown iff
`h_{j+1,j} ≤ 1e−12·max(1, τ‖G‖ + 2)` AND (`β·h_{j+1,j} ≤ tol·‖y₀‖₂` or
`j + 1 = n + p`): `f = e^{(1−t)H_j}e₁`, error `β·h_{j+1,j}·(1 − t)`, done; otherwise
the iteration continues with `v_{j+1} = z/h_{j+1,j}`. `y₀` is prescaled by `2^{−e}`
(`‖y₀‖∞ ∈ [1, 2)`) and the result by `2^{e}`.

## §68.3 — Accuracy

Each accepted substep has estimated local error `≤ 1.2·Δt·tol·‖y₀‖₂`; the estimates
sum to `≤ 1.2·tol·‖y₀‖₂` over `t ∈ [0, 1]`. Estimates, not bounds (Sidje 1998). The
reported `est_error` is `Σ max(err, (τ‖G‖ + 2)·ε·β)/‖y₀‖₂` (Expokit's round-off
floor), which can exceed `tol` at large `τ‖G‖`.

## §68.4 — Gates

`G_PHI_KRYLOV_DENSE`, `G_PHI_KRYLOV_COST`, `G_PHI_KRYLOV_TOL`,
`G_ETDRK4_KRYLOV_ORDER` (ADR-0211 §Gates).
