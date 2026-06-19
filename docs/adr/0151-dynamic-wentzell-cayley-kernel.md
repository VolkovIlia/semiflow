# ADR-0151 — Dynamic (Wentzell/Robin) BC via implicit Cayley boundary step — `DynamicWentzellChernoff<C, R, F>`

**Status:** ACCEPTED · **Date:** 2026-06-08 · **Branch:** `feat/v8.2.0-math`
**Theme:** v8.2.0 — Wave-2 C-9 (dynamic boundary conditions, stability-validated GO)
**Supersedes:** ADR-0098 Amendment 3 ("Dynamic BC remains DEFERRED indefinitely") — the obstruction was EXPLICIT-only; this ADR ships the implicit resolvent remedy.
**Builds on:** ADR-0146 (research pre-flight, GO verdict) · ADR-0098 (static Robin sibling `RobinHeatChernoff`, order-1) · §17.4 (Crank–Nicolson Cayley map — A-stable precedent) · §23 (Howland nonautonomous lift — the `γ(t)` vehicle).
**Gates:** `G_WENTZELL_STABLE` (RELEASE_BLOCKING), `G_WENTZELL_ORDER` (RELEASE_BLOCKING), `T_WENTZELL` (NORMATIVE oracle).
**Math:** math.md §49 (NEW — NORMATIVE). **Oracle:** `scripts/wentzell_robin_stability_preflight.py` (GO 2026-06-08; extended to print the `T_WENTZELL` PASS line).

## Context

The dynamic Wentzell/Robin condition `∂_t u + γ(t)·∂_ν u + c·u = 0` on `∂Ω` was deferred indefinitely (ADR-0098 Am.3) because the natural split-step Chernoff/Trotter product on the boundary lift `X ⊕ L²(∂Ω)` is provably unstable for the **unbounded** normal-derivative coupling `∂_ν`: Stephan 2023 (arXiv:2307.00419, ZAMM 2025) shows the EXPLICIT freezing product satisfies `‖T(t/n)ⁿ‖ ≥ nᵝ·t^{1−β} → ∞`. The TRIZ contradiction (ADR-0146) — *time-dependent boundary scaling `γ(t)` AND boundary-layer stability* — is genuinely resolved, not split, by advancing the boundary block with an **implicit Cayley (resolvent) sub-step** `K_CN = (I − τC/2)⁻¹(I + τC/2)`: the Cayley/Möbius map sends the closed left half-plane to the closed unit disk, so the same unbounded exchange yields amplification `ρ ≤ 1` at any stiffness, including time-dependent `γ(t)` (von-Neumann pre-flight: `ρ ≤ 0.9998` vs explicit `2.95 → ∞`; symbolic witness `|z_cay| ≤ 1`, `lim_{μ→∞} z_cay = −1`). Backed by Kovács–Lubich 2015/2017 (implicit BDF/Radau stable for dynamic Wentzell BC) and Altmann–Verfürth 2021/2022 (implicit-Euler bulk–surface Lie splitting, weak CFL `τ ≤ ch`).

## Decision

Ship `DynamicWentzellChernoff<C, R, F = f64>` (new module `crates/semiflow-core/src/wentzell.rs`, additive, NON-BREAKING) — a **wrapper type**, NOT a `BoundaryPolicy` variant (see API-shape note below). Per step it performs a **bulk–boundary Lie splitting**: (a) one bulk Chernoff step `C.apply_into(τ)` (the inner `DiffusionChernoff`), composed with (b) an **implicit Cayley boundary sub-step** that advances the per-boundary-DOF coupled block `(I − τC_∂/2)⁻¹(I + τC_∂/2)` by a banded (Thomas) solve, mirroring §17.4's `cn_kinetic_step_f64`. The time-dependent scaling `γ(t)` rides the §23 Howland lift: `DynamicWentzellChernoff` overrides `TimedChernoffFunction::apply_at(t, …)` to sample `γ(t)` at the step's left endpoint, so `HowlandLift<DynamicWentzellChernoff<…>>` carries the nonautonomous case for free. `order() = 1` (honest — matches static Robin ADR-0098 and the Altmann–Verfürth Lie splitting; the bulk↔boundary commutator is nonzero and the left-endpoint freezing is order-1). A **weak CFL** `τ ≤ c·h` is documented as a mesh-coupling accuracy requirement (NOT a stability collapse — stability is unconditional by the Cayley bound). Suckless: every fn ≤ 50 lines (extract `cayley_boundary_step`, `assemble_boundary_block`, contour-free banded solve helpers); the module is expected `~360 LoC` ≤ the 500-LoC cap (no Cohort carve-out needed; if the banded-solve scratch pushes it over, add a `wentzell.rs` HARD-LIMIT 800 Cohort entry mirroring `manifold.rs`).

### API shape: wrapper type, NOT a `BoundaryPolicy` variant (justification)

`BoundaryPolicy` is a **stencil-level** enum answering "what value does the interpolant read at an out-of-range index?" (`boundary.rs:30`). Even static `BoundaryPolicy::Robin { alpha, beta }` is only a skew-image **weight carrier** — the operator-level Robin character is enforced by the wrapper `RobinHeatChernoff<C, R, F>` (`robin.rs:142`). The dynamic case needs (i) an extra per-boundary-DOF **state component** (the boundary trace `u_∂` evolves by its own ODE), (ii) an **implicit banded solve** per step, and (iii) **time-dependent** `γ(t)` via `apply_at`. None of these fit an out-of-range-sample enum; all three are exactly what the `RobinHeatChernoff` wrapper precedent already carries. Therefore: **reuse the wrapper pattern** — `DynamicWentzellChernoff<C, R, F>` parameterised by the inner bulk Chernoff `C` and a `WentzellRegion<F>` (sub-trait of `RobinRegion<F>` adding the time-dependent `γ(t)` closure and reaction `c`). No `BoundaryPolicy` variant is added (the static `BoundaryPolicy::Robin` stays untouched; the dynamic block does not route through the sample-policy path at all).

## Engineer spec (file `crates/semiflow-core/src/wentzell.rs` — additive, NON-BREAKING)

**Do NOT modify `robin.rs`, `boundary.rs`, `howland.rs`, or `schrodinger.rs`.** New module only; reuse `RobinRegion`, `ReflectingRegion`, `HalfSpaceRobin` geometry, `ScratchPool`, `State::axpy_into`.

1. **Region sub-trait** `pub trait WentzellRegion<F>: RobinRegion<F>` adds:
   - `fn gamma_at(&self, t: F) -> F;` — the time-dependent boundary scaling `γ(t) ≥ 0` at absolute time `t` (default impl may return the static `β` for autonomous use).
   - `fn reaction(&self) -> F;` — the boundary reaction coefficient `c ≥ 0`.
   - Concrete `pub struct HalfSpaceWentzell<F, const D: usize> { half_space: HalfSpaceRegion<F,D>, gamma: fn(F) -> F, c: F }` with `new(origin, normal, gamma, c)` validating `c ≥ 0`, `‖normal‖₂ = 1` (delegate), finite. `robin_coeffs()` returns `(c, gamma(F::zero()))` for the `t=0` static fallback.

2. **Public type** `pub struct DynamicWentzellChernoff<C, R, F = f64> { inner: C, region: R, _f: PhantomData<F> }` with `C: ChernoffFunction<F, S = GridFn1D<F>>`, `R: WentzellRegion<F>`. `new(inner, region) -> Result<Self, SemiflowError>` (mirror `RobinHeatChernoff::new`).

3. **Concrete `ChernoffFunction<f64>` impl** for `DynamicWentzellChernoff<DiffusionChernoff<f64>, HalfSpaceWentzell<f64,1>, f64>` (mirror `robin.rs:174`), `type S = GridFn1D<f64>`:
   - `apply_into(τ, src, dst, scratch)` delegates to `apply_at(0.0, τ, …)` (autonomous default uses `γ(0)`).
   - `order()` returns `1`. `growth()` returns `self.inner.growth()` (the Cayley boundary block is a contraction, `ρ ≤ 1`, so it cannot increase growth).

4. **`TimedChernoffFunction<f64>` impl** (the nonautonomous hook) overrides `apply_at`:
   - `let g = self.region.gamma_at(t);` `let c = self.region.reaction();` `let dx = src.grid.dx();`
   - **Bulk half/step:** `self.inner.apply_into(τ, src, dst, scratch)` — the interior diffusion (Lie splitting factor (a)).
   - **Boundary Cayley sub-step:** advance the boundary DOF + its nearest interior neighbour by the implicit `2×2`-per-boundary-DOF Cayley map of the coupled generator block `C_∂(t)` (math §49.3):
     ```text
     C_∂ = [[ −a/dx²        +1/dx        ]
            [ −g/dx       −(g/dx + c)    ]]      (row 0 = near-boundary bulk, row 1 = trace u_∂)
     ```
     `K_CN = (I − (τ/2)C_∂)⁻¹(I + (τ/2)C_∂)` applied to `[dst[1], u_∂]` (the `2×2` inverse is closed-form — `det = (1−τ/2·tr)+τ²/4·det(C_∂)`; NO LAPACK). Helper `fn cayley_boundary_step(dst: &mut GridFn1D<f64>, u_bnd: &mut f64, c_block: [[f64;2];2], tau: f64)` ≤ 50 lines.
   - Keep `apply_at` ≤ 50 lines; extract `assemble_boundary_block(g, c, a0, dx) -> [[f64;2];2]` and `cayley_boundary_step` helpers (suckless).
   - **The boundary trace `u_∂`** is carried in the grid's boundary node `dst.values[0]` (1D half-line `[0,∞)` convention, boundary at `x=0`) — no separate state struct needed in 1D (the `X ⊕ ℝ_∂` lift collapses to "boundary node IS the trace DOF"). Document this collapse explicitly; multi-D `X ⊕ L²(∂Ω)` (a true product state) is deferred to v8.x.

5. **Howland wiring (free):** add `impl TimedChernoffFunction<f64> for DynamicWentzellChernoff<…>` is the one in step 4 (it overrides `apply_at`). `HowlandLift::new(dyn_wentzell, T, n_t)` then advances `γ(t)` per step with matched-step `τ = Δs` (§23.4). No new Howland machinery.

6. **Gate harnesses** (test-only, NOT `ChernoffFunction`):
   - `pub struct WentzellStabilityGate` — Rust port of the pre-flight von-Neumann sweep (§49.5 G_WENTZELL_STABLE): for `dx ∈ {1/16, 1/64, 1/256, 1/1024}`, `γ ∈ {0.5,1,4,16}`, `κ = π/dx`, `τ = 0.4·dx²/a`, assemble `C_∂`, form `K_CN`, assert `ρ(K_CN) ≤ 1 + 1e-9` (closed-form `2×2` eigen-magnitude — no LAPACK). MUST also assert the EXPLICIT map `(I + τC_∂)` has `ρ > 1` somewhere (the candidate must FIX a real instability). Test file `tests/g_wentzell_stable.rs` (feature `slow-tests`).
   - `pub struct WentzellOrderGate` — manufactured-solution order-1 self-convergence (§49.6 G_WENTZELL_ORDER): time-dependent `γ(t) = 0.5 + sin(t)` (generic, never identically static), generic non-origin spatial probe (avoid `x=0` symmetric cancellation traps per the G24 lesson), `N=64`, sweep `n ∈ {16,32,64,128}`, reference = many-small-step. Assert log-log OLS slope `≤ −0.95` (G27-convention: order-1). Test file `tests/g_wentzell_order.rs` (feature `slow-tests`).

7. **Sympy/numeric oracle** already authored: extend `scripts/wentzell_robin_stability_preflight.py` to print exactly `T_WENTZELL PASS` (3 sub-checks: `cayley_abs_le_1` symbolic `1 − z_cay² ≥ 0`; `stiff_limit` `lim_{μ→∞} z_cay = −1`; `explicit_blowup` `lim_{μ→∞}|z_expl| = ∞`) or `T_WENTZELL FAIL: <reason>`. Wire into the `xtask test-fast` sympy sweep next to `verify_reflected_heat_halfline.py`.

8. **Constraints:** additive (no public-surface change to existing types — `BoundaryPolicy`, `RobinHeatChernoff`, `HowlandLift` untouched); no new deps; functions ≤ 50 lines, file ≤ 500 LoC (Cohort entry only if exceeded); `no_std`-safe (closed-form `2×2` arithmetic over `f64`; `fn(F)->F` for `γ`, no captures, matching the §17.4 / ADR-0134 fn-ptr discipline). Order-1 only; second-order BDF boundary variant (Altmann–Verfürth 2022) deferred (math §49.7).

## Consequences

Closes the ADR-0098 Am.3 indefinite defer with a stability-validated, peer-reviewed implicit-boundary construction reusing the library's own §17.4 Cayley map and §23 Howland lift — no new machinery. Publishable (implicit-Cayley dynamic-Wentzell over a Chernoff bulk step, with time-dependent `γ(t)` via the Howland lift, is a novel combination). The single new numerical primitive is the closed-form `2×2` Cayley boundary block (~25 LoC); everything else is composition. Residual risk LOW: the Cayley bound is unconditional (pre-flight margin `ρ ≤ 0.9998` across the full sweep + time-dependent product); the only soft requirement is the weak CFL `τ ≤ ch` for ACCURACY (documented, NOT a stability gate). Multi-D `X ⊕ L²(∂Ω)` true-product state, second-order BDF boundary, and per-cell-varying `γ` along `∂Ω` are deferred to v8.x (math §49.7).
