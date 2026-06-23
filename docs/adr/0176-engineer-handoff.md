# ADR-0176 — Engineer Handoff: Order-2 Dirichlet via Odd Image

Design refs: ADR-0176, math.md §21.9 (kernel `K^D`, Prop 21.9.1, gates).
**Additive only. `KillingChernoff` (§21.3) and G23 stay UNCHANGED.** No commit.

## Investigation result — existing `BoundaryPolicy::Dirichlet` is NOT the odd image (a NEW variant IS required)

Read `crates/semiflow-core/src/boundary.rs`. Findings:

- **`BoundaryPolicy::Dirichlet { value: F }`** (lines 65–68, 268) is a **constant-extension** ghost: out-of-range index → `BoundaryHit::Dirichlet(value)` → `bc_value` returns the literal `value` (lines 318, 360). It is NOT a reflection at all — it is the order-1 stencil clamp for "ghost nodes hold a constant" (math §3.5.bis.1). It does NOT negate any interior value and does NOT realise the odd extension (21.9.1). **Do NOT reuse it.**
- **`BoundaryPolicy::Reflect`** (lines 52, 249) is the EVEN image: out-of-range → `Inside(reflect_index(n, idx))` → `bc_value` returns `+values[reflected]`. This is exactly what `ReflectedHeatChernoff` uses on a half-line to get the Neumann (even) extension (`reflection.rs` lines 303–304).
- **Conclusion**: the odd extension needs `ghost = −(mirrored interior value)` — i.e. the `Reflect` index-fold but with a **sign flip on the returned value**. The existing `Dirichlet` (constant) and `Reflect` (even, +) variants neither do this. **A new `BoundaryPolicy::OddReflect` variant is justified and required.**

## File checklist

### 1. `crates/semiflow-core/src/boundary.rs` — add `OddReflect` variant

Mirror how `Reflect` is implemented (it folds the index, then `bc_value` returns the interior value). `OddReflect` folds the **same way** but negates the returned value.

- Add to `enum BoundaryPolicy<F>` (after `Reflect`, near line 52): variant `OddReflect` (unit variant, like `Reflect` / `Neumann` — no payload). Doc it: "Odd (antisymmetric) image: out-of-range ghost = `−(mirrored interior value)`. Realises the Dirichlet odd extension (math §21.9 (21.9.1)); the minus-sign mirror of `Reflect`. Consumed by `DirichletHeat2ndChernoff`."
- Add a `BoundaryHit` carrier for the negated reflection. Cleanest mirror of the existing `RobinSkew { reflected, depth }` (lines 117–122): add `OddReflected { reflected: usize }` (no weight, just the mirror index). In `bc_index` (line 248 `match`), add arm:
  `BoundaryPolicy::OddReflect => BoundaryHit::OddReflected { reflected: reflect_index(n, idx) }`.
- In BOTH `bc_value` (line 308) and `bc_value_generic` (line 349), add arm:
  `BoundaryHit::OddReflected { reflected } => -values[reflected]` (use `F::zero() - values[reflected]` in the generic path for the `F` bound).
- The in-range fast-path (line 244) already returns `Inside(idx)` for all policies — interior nodes are unaffected, identical to `Reflect`. Good.
- `derive(PartialEq)` on both enums already covers a unit variant; no `Eq`-payload issue (unit variant carries no float).

### 2. `crates/semiflow-core/src/killing_order2.rs` (NEW) — `DirichletHeat2ndChernoff<C, R, F>`

Mirror `reflection.rs` `ReflectedHeatChernoff` structure (lines 215–320), but use the ODD half-line trick instead of the even one. D=1 half-line single inner step:

- Reuse the `ReflectingRegion<F>` trait + `HalfSpaceRegion<F, 1>` from `reflection.rs` (the geometry — `is_inside`, `σ_R` — is identical; only the wrapper's sign differs). Do NOT define a new region trait.
- `pub struct DirichletHeat2ndChernoff<C, R, F = f64>` with fields `inner: C`, `pub region: R`, `_f: PhantomData<F>`; bounds `C: ChernoffFunction<F, S = GridFn1D<F>>`, `R: ReflectingRegion<F>`, `F: SemiflowFloat`. `pub fn new(inner, region) -> Result<Self, SemiflowError>`.
- `impl ChernoffFunction<f64> for DirichletHeat2ndChernoff<DiffusionChernoff<f64>, HalfSpaceRegion<f64, 1>, f64>` (concrete D=1 case, exactly as `reflection.rs` line 285). In `apply_into`: clone `src`, set `src_odd.grid = src_odd.grid.with_boundary(BoundaryPolicy::OddReflect)`, then ONE `self.inner.apply_into(tau, &src_odd, dst, scratch)`. This is the odd-image equivalent of `reflection.rs` lines 303–307 (which uses `BoundaryPolicy::Reflect`); the odd ghost at `x=0` makes the stencil see `−f(δ)` at the mirror node, forcing `u(0)=0`.
- `fn order(&self) -> u32 { self.inner.order() }` (Prop 21.9.1; order-2 for `DiffusionChernoff`).
- `fn growth(&self) -> Growth<f64> { self.inner.growth() }` (single contraction; no mass doubling).
- Tests in `#[cfg(test)] mod tests`: (a) `order()` == 2 (mirror `reflected_heat_chernoff_order_matches_inner`); (b) smoke `apply_into` finite. **Do NOT add a non-negativity test** — the odd ghost subtracts mass (ADR-0176 limits; §21.9). This is the key divergence from `reflection.rs` (drop its `nonneg_preserved` test).

### 3. `crates/semiflow-core/src/lib.rs` — register module

After `pub mod reflection;` (line 255): add `pub mod killing_order2;` and a pub re-export of `DirichletHeat2ndChernoff` (mirror how `reflection::ReflectedHeatChernoff` is surfaced — check whether reflection re-exports at crate root or is reached via `semiflow_core::reflection::…`; the G27 test imports `reflection::{HalfSpaceRegion, ReflectedHeatChernoff}`, so a `pub mod` is sufficient. Match that.)

### 4. `crates/semiflow-core/tests/g_dirichlet_order2.rs` (NEW) — `G_DIRICHLET_ORDER2`, RELEASE_BLOCKING, slow-tests

Mirror `tests/reflected_heat_halfline.rs` structure (OLS slope harness, gate constants), with these changes:

- `#![cfg(feature = "slow-tests")]`.
- Operator: `DiffusionChernoff` with `a(x) ≡ 1/2` (so `L = ½∂_xx`), `a'≡0`, `a''≡0`. Region: `HalfSpaceRegion<f64,1>` origin `[0.0]` normal `[1.0]`. (The eigenmode oracle below uses the `(0,1)` Dirichlet eigenbasis; restrict the grid to `[0, 1]` and use the odd wall at `x=0`. If a two-sided `(0,1)` wall is needed, the engineer may run on `[0,1]` with `OddReflect` at both ends — document the choice; a one-sided half-line on `[0, L]` with a single Dirichlet eigenmode `sin(πx/L)` is an acceptable simpler oracle if the two-sided setup is awkward, as long as the slope gate is met.)
- Oracle: `u(t,x) = Σ_{k=1}^{8} a_k sin(kπx) e^{-(kπ)² t/2}` on `(0,1)`. Pick fixed `a_k` (e.g. `a_k = 1/k`). Each mode is an exact Dirichlet eigenfunction of `½∂_xx`, vanishing at `x ∈ {0,1}`.
- IC: `u_0(x) = Σ a_k sin(kπx)`.
- `SLOPE_GATE: f64 = -1.95` (order-2 — NOT −0.95; the §25 Neumann inner there was order-1, here `DiffusionChernoff` is order-2). Margin vs theoretical −2.0.
- Sweep `n ∈ {16, 32, 64, 128}` (or finer if floor interferes); sup-norm error on interior nodes; OLS `log(err)` vs `log(n)`; assert `slope ≤ SLOPE_GATE`.
- Print `G_DIRICHLET_ORDER2 PASS` on success. **Keep G23 (`tests/killing_dirichlet_slope.rs`) unchanged.**

### 5. `scripts/verify_dirichlet_order2.py` (NEW) — `T_DIRICHLET_ORDER2`

Mirror `scripts/verify_reflected_heat_halfline.py` (T22N) with the MINUS sign:

- `K(x,y,t) = (4πt)^{-1/2} exp(-(x-y)²/(4t))`; `K^D = K(x,y,t) − K(x,−y,t)` (note **minus**, vs T22N's plus).
- Sub-check 1 (`heat_pde`): `simplify(∂_t K^D − ∂_xx K^D) == 0`.
- Sub-check 2 (`dirichlet_boundary`): `simplify(K^D.subs(x, 0)) == 0` (the two terms cancel at `x=0` — the odd kernel's defining BC; mirror of T22N's `neumann_boundary` which instead checks `∂_x K^N|_{x=0}=0`). Odd ⇒ value (not derivative) vanishes on the boundary.
- Print exactly `T_DIRICHLET_ORDER2 PASS` on success, `T_DIRICHLET_ORDER2 FAIL: <reason>` on failure; exit 1 on failure. Pure symbolic, no library runtime.

## Verification commands

```bash
cargo run -p xtask -- test-fast            # unit + sympy sweep (fast)
cargo run -p xtask -- test-ignored-gates   # slow-tests order-2 gate (G_DIRICHLET_ORDER2)
python3 scripts/verify_dirichlet_order2.py # T_DIRICHLET_ORDER2 PASS
```

## Suckless / impact

- Run `mcp__gitnexus__impact` on `BoundaryPolicy`, `bc_index`, `bc_value`, `bc_value_generic` before editing — adding an enum variant forces every exhaustive `match` on `BoundaryPolicy`/`BoundaryHit` to gain an arm (compiler will flag non-exhaustive matches; fix each). Report blast radius.
- `killing_order2.rs` is a NEW file — mirror `reflection.rs`; target well under the 500-LoC cap, no Override needed.
- `order()` = `inner.order()`; no separate order logic. Errors as values (`Result<_, SemiflowError>`). Functions ≤50 lines.
