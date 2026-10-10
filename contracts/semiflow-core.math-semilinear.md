# §64 — Semilinear reaction–diffusion splitting and Richardson extrapolation (ADR-0207, ADR-0208, NORMATIVE library)

## §64.1 — Strang splitting of `∂ₜu = Lu + f(t, x, u)` (NORMATIVE)

State: `K` species `u_k` on common nodes `x_i`, `i < N` (`Species<S>`). Linear part
`L = diag(L_1, …, L_K)`, each `L_k` approximated by an engine `C_k(τ)` of order
`p_k` (`‖C_k(τ)f − e^{τL_k}f‖ = O(τ^{p_k+1})`). Reaction flow `R(t, h)`: the exact
flow of the ODE `u′ = f(t, x_i, u)` at every node over `[t, t + h]` when
`Kinetics::exact_flow` returns it, else `r` classical RK4 steps of `h/r`.

```text
U(t + τ) = R(t + τ/2, τ/2) ∘ L(τ) ∘ R(t, τ/2) U(t)                         [§64.1.a]
```

`evolve_into(t0, t, n)` composes `n` steps with `τ = t/n` and merges the adjacent
half-steps: `R(t0, τ/2) L R(t0 + τ/2, τ) L … L R(t0 + t − τ/2, τ/2)` — the same
operator for exact flows, a different RK4 discretisation of the same order
otherwise. Global error `O(τ^{min(2, p)})`, `p = min_k p_k` (Strang; the
splitting error is `O(τ²)` per unit time for smooth `f`, Hairer–Lubich–Wanner
GNI §III.5).

**Commuting case (NORMATIVE identity).** If `C_k = C` for all `k` and
`f(t, x, u) = Cu` with a constant matrix `C`, then `R` and `L` commute exactly
(`L` acts identically and linearly on every species, `R` is a constant pointwise
linear map), so §64.1.a equals `e^{tC}` applied pointwise after the pure
diffusion `C(τ)ⁿ` — the
gate `G_SEMILIN_LINEAR_SYSTEM_EXACT` checks it to `1e−13`.

**Batched kinetics (NORMATIVE interface).** `Kinetics::eval_batch(t, N, x, u, du)`
evaluates all nodes in one call: `u`, `du` species-major (`u[k·N + i]`), `x`
axis-major (`x[d·N + i]`, `d < dim`, empty on graphs). Per step: `4r` calls with
RK4, at most one with an exact flow.

**Exact flows.** Logistic `u′ = ru(1 − u/K)`: `u(h) = Ku/(u + (K − u)e^{−rh})`.
Allen–Cahn `u′ = κ(u − u³)`: `u(h) = u/√(u² + (1 − u²)e^{−2κh})`. Linear `u′ = Cu`:
`e^{hC}` by scaling and squaring of the degree-18 Taylor polynomial with
`‖hC/2ˢ‖₁ ≤ ½` (truncation `< 2^{−19}/19! < 1e−22`).

**Oracles (verified symbolically, `sympy`).** Fisher–KPP `u_t = u_xx + u(1 − u)`:
`u = (1 + e^{(x − ct)/√6})^{−2}`, `c = 5/√6` (Ablowitz & Zeppetella 1979).
Nagumo `u_t = u_xx + u(1 − u)(u − a)`: `u = 1/(1 + e^{(x − ct)/√2})`,
`c = √2(½ − a)`. MMS: `u = e^{−t}e^{−x²}` for `u_t = Au_xx + f`,
`f = −(4Ax² − 2A + 1)e^{−t}e^{−x²}`.

## §64.2 — Richardson extrapolation in the step count (NORMATIVE)

For an integrator with global error `Σ_{q ≥ p} c_q n^{−q}`, runs with
`n_j = (j+1)·n`, `j < L ≤ 6`, are combined with weights solving

```text
Σ_j w_j = 1,     Σ_j w_j (j+1)^{−q} = 0,   q = p, …, p + L − 2,              [§64.2.a]
```

giving error `O(n^{−(p+L−1)})` at cost `L(L+1)/2` base runs. Closed forms:
`(p, L) = (1, 2)`: `(−1, 2)`; `(1, 3)`: `(½, −4, 9/2)`; `(2, 2)`: `(−⅓, 4/3)`.
Rounding and spatial errors are amplified by `Σ|w_j|`. Not defined for
tolerance-driven kernels (`order() = u32::MAX`).
