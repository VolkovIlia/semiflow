# ADR-0134 — F2 Resolvent time-jump amortization

**Status:** ACCEPTED (NARROW — restricted to self-adjoint / sectorial generators) · **Date:** 2026-06-06 · **Shipped:** 2026-06-08 · **Branch:** `feat/v8.0.0-planning`
**Theme:** v8.0.0 — Differentiable Chernoff (F2, RESEARCH-TRACK)
**Gate:** `G_RESOLVENT_JUMP_ORDER` (RELEASE_BLOCKING), `T_RESOLVENT_JUMP` (NORMATIVE oracle)
**Parent:** ADR-0132
**Math:** math.md §47 (NEW — NORMATIVE). **Oracle:** `scripts/resolvent_jump_kit.py` (4/4 PASS 2026-06-07).

## Context

Long time horizons `T` and implicit / steady-state targets require `n = T/τ → ∞` Chernoff steps, each costing `O(N)`, so total work is `O(T·N/τ)`. Making τ large hits the `‖A‖·τ ≳ 1` regime where short-time Chernoff approximation degrades. Contradiction C2 (declared PERMANENTLY infeasible): operator-level Padé scaling-and-squaring in the time domain is impossible because `‖A‖∞ ∝ 1/dx² → ∞` for PDE generators — this ceiling was struck twice (ADR-0094 defer, ADR-0101 final defer) and is physically immovable. The resolvent `(λI − A)⁻¹` shipped in v2.7.0 (ADR-0069, Vladikavkaz Thm 3, Gauss-Laguerre GL₃₂) provides the escape: by Laplace inversion `e^{tA} = (1/2πi) ∫ e^{λt} (λI−A)⁻¹ dλ`, a large time-step equals a WEIGHTED SUM of bounded resolvents. The resolvent norm `‖(λI−A)⁻¹‖ ≤ 1/Re(λ)` is independent of `‖A‖`, so the Padé-in-time ceiling disappears under a coordinate change.

## Decision

Investigate `ResolvantJumpChernoff`: a rational-Krylov / numerical Laplace inversion scheme that approximates `e^{tA}f` as a weighted quadrature sum over the already-shipped `LaplacianChernoffResolvent` backend, enabling large time-steps `t ∝ 1/‖A‖²` at cost decoupled from `n` to a constant number of resolvent quadrature nodes. The TRIZ resolution is: TRIZ-13 (inversion — substitute the temporal expansion with a spectral one) + TRIZ-28 (field substitution — temporal domain → spectral/Laplace domain) + reuse of the GL₃₂ resolvent infrastructure in `resolvent.rs` + `resolvent_quad.rs`. This is NOT operator-Padé (ADR-0101 deferral UNCHANGED — no squaring of an infinite-norm matrix is attempted); it is a coordinate change that removes the `‖A‖` dependence by working in the spectral domain where the resolvent is bounded. Gate `G_RESOLVENT_JUMP_ORDER`: self-convergence of a large-step resolvent-jump approximation vs a many-small-step Chernoff reference, slope ≤ −1.95 in `Δτ`, demonstrating that the scheme is at least order-2 in the effective step size.

## SPIKE finding (2026-06-07) — NARROW GO

The Phase-2 math SPIKE (`scripts/resolvent_jump_kit.py`, 4/4 PASS) RESOLVES the MEDIUM-HIGH risk. The contour design is SOUND and the gate passes with enormous margin, but with ONE structural restriction:

- **Contour:** Trefethen–Weideman–Schmelzer (2006) optimised **parabolic** contour, scaled `λ(θ) = (M/t)(0.1309 − 0.1194θ² + 0.25iθ)`, midpoint rule (math.md §47.2). Converges **geometrically** (−0.442 decades/node; `f64` floor at `M ≈ 28`).
- **Cost decoupling CONFIRMED:** the `M/t` scaling makes the node count `t`-INDEPENDENT — err-vs-`M` at `t ∈ {1, 20, 100}` coincides within `1.83×`. Large-`T` cost is `M = O(1)` resolvent solves vs `n = T·‖A‖` Chernoff steps.
- **Gate:** `G_RESOLVENT_JUMP_ORDER` G24-convention slope `d log(err)/d log(1/M) = +9.86 ≥ 1.95` (≫ order-2). Declared **RELEASE_BLOCKING** (`slow-tests`).
- **THE NARROW RESTRICTION (math.md §47.4):** the optimal contour places **most nodes in the LEFT half-plane** (`16/24` at `t=20, M=24`). The shipped GL₃₂ `LaplaceChernoffResolvent::eval` / `eval_complex` is the Laplace transform `∫₀^∞ e^{−λt}S(t)g\,dt`, which **DIVERGES for `Re λ ≤ ω`** (§22.9 SPEC TRAP). A right-half-plane-confined contour does NOT recover the method (`e^{λt}` blows up; PRE-FLIGHT err `≥ 10¹³`). Therefore the engineer **REUSES the resolvent abstraction but NOT the GL₃₂ quadrature** — `(λ_k I − A)⁻¹ g` is evaluated with a **left-half-plane-capable direct complex solve** (Thomas `O(N)` for the tridiagonal divergence-form `(λI − A)`). NARROW is benign: every shipped diffusion / divergence-form generator is self-adjoint negative-semidefinite (sectorial), so Theorem 47.1 holds for the whole family. Non-self-adjoint / advection-dominated generators are OUT of scope for v8.0.0.

## Consequences

Large-T / steady-state / implicit-grade stability with a CONSTANT node count, decoupled from `‖A‖` (the ADR-0101 operator-Padé ceiling is bypassed by a coordinate change, NOT by squaring an infinite-norm matrix — that deferral is UNCHANGED). Publishable (numerical-Laplace-inversion acceleration of Chernoff semigroups over the shipped resolvent abstraction is a novel combination). The single new numerical primitive is the LHP complex tridiagonal solve (~40 LoC); the contour loop, TWS coefficients, and weight assembly are pure scalar arithmetic over the reused resolvent concept. Residual risk is LOW (gate margin `≈ 5×` in slope; the LHP solve is unconditionally stable off-spectrum for sectorial `A`). 2D/3D LHP backends and the hyperbolic-contour / rational-Krylov variant for non-sectorial generators are deferred to v8.x (math.md §47.6).

## Engineer spec (file `crates/semiflow-core/src/resolvent_jump.rs` — additive, NON-BREAKING)

**Do NOT modify `resolvent.rs` / `resolvent_complex.rs`.** New module only.

1. **Public type** `ResolventJumpChernoff<C, F = f64>` with fields `{ inner: C, m_nodes: usize }`, `C: ChernoffFunction<F>` (the `inner` carries the operator geometry — grid + coefficients — so the LHP solve can assemble `(λI − A)`). `new(inner, m_nodes) -> Result<Self, SemiflowError>` validates `m_nodes ≥ 6` (below that the geometric regime is not yet reached).
2. **Method** `pub fn jump(&self, t: F, g: &C::S) -> Result<C::S, SemiflowError> where C::S: Clone` — implements math.md §47.3:
   - validate `t.is_finite() && t > 0`;
   - `acc := zeroed_like(g)`; for `k ∈ [0, m_nodes)`: `θ_k = −π + (k+½)(2π/m_nodes)`; `λ_k = (m_nodes/t)(0.1309 − 0.1194θ² + 0.25 i θ)`; `λ'_k = (m_nodes/t)(−2·0.1194·θ + 0.25 i)`; `r_k = resolve_lhp(λ_k, g)`; `acc += Re[ e^{λ_k t} · r_k · λ'_k ] · (1/(2π))` (the `1/(2πi)` and `2π/M` combine; the result is real because conjugate-pair nodes cancel imaginary parts).
   - return `acc`.
   - Keep `jump` ≤ 50 lines; extract the contour-node computation and `resolve_lhp` into helpers (suckless).
3. **LHP resolvent backend** `fn resolve_lhp(&self, lambda: Complex<F>, g: &C::S) -> Result<C::S, SemiflowError>` — a **complex tridiagonal Thomas solve** of `(λI − A) r = g` where `A` is the divergence-form Laplacian reconstructed from `self.inner`'s grid/coefficients (mirror the FD stencil used by `LaplaceChernoffResolventResidual::verify_residual` in `resolvent.rs:447`). Use `SemiflowComplex` (ADR-0079) for the complex arithmetic; the state is real `g` lifted to `GridFnComplex1D`, solved, real part taken. NO guard on `Re λ` (the LHP IS the valid domain here — this is the inverse of the §22.9 guard).
4. **Gate harness** `pub struct ResolventJumpOrderGate` mirroring `LaplaceChernoffResolventResidual` (test-only, NOT a `ChernoffFunction`): sweeps `M ∈ {6,8,10,12,14}` at `t = 100`, `N = 64`, Gaussian `g`, computes the `log`-`log` OLS slope vs `1/M` against an `expm`-or-many-step reference, asserts `slope ≥ 1.95`. Test file `tests/resolvent_jump_order.rs` (feature `slow-tests`).
5. **Sympy/numeric oracle** already authored: `scripts/resolvent_jump_kit.py` (`T_RESOLVENT_JUMP`, 4/4 PASS). Wire it into the `xtask test-fast` sympy sweep next to `verify_complex_resolvent.py`.
6. **Constraints:** additive (no public-surface change to existing types); reuse `ScratchPool`, `State::axpy_into`, `GridFnComplex1D`, `SemiflowComplex` (no new deps — `num-complex` is already 3/3 budget); functions ≤ 50 lines, file ≤ 500 lines; `no_std`-safe (`Complex` arithmetic over `SemiflowFloat`, const TWS coefficients as `const [f64; 3]`).
