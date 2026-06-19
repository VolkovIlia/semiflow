# ADR-0146 — Dynamic (Wentzell) Robin BC: TRIZ research, stability verdict GO via implicit Cayley boundary step

- **Status**: Proposed (research / design — no implementation in this wave)
- **Date**: 2026-06-08
- **Decision-maker**: researcher (Wave-3 register C-9, TRIZ-gated)
- **Related**: ADR-0098 Amendment 3 (the HONEST-DEFER this ADR reopens); ADR-0098 (static Robin, skew-reflection, order-1); §3.5.tris (static Robin math); §17.4 (Crank–Nicolson Cayley map — the A-stable precedent reused); §22 (Laplace-Chernoff resolvent — the resolvent infrastructure a future impl can lean on); §23 (Howland nonautonomous lift — the time-dependent-γ(t) vehicle).
- **Constraint note**: research-only wave under a concurrent `test-full` build — no `crates/` edits, no `cargo`. Deliverables are a sympy/numpy stability preflight (`scripts/wentzell_robin_stability_preflight.py`) and this ADR.

## TRIZ analysis (АП → ТП → ФП → ИКР → решение)

**НЭ (нежелательный эффект)**: the dynamic Wentzell/Robin condition `∂_t u + γ(t) ∂_ν u + c u = 0` on `∂Ω` is deferred indefinitely (ADR-0098 Am.3) because the natural split-step Chernoff/Trotter product formula on the lift `X ⊕ L²(∂Ω)` is *provably unstable*: the bulk↔boundary coupling is the **unbounded** normal-derivative `∂_ν`, and Stephan 2023 shows the freezing product then satisfies `‖T(t/n)^n‖ ≥ n^β t^{1−β} → ∞` (relative-bound exponent `β ∈ (0,1]`).

**АП**: устранить расходимость продукт-формулы на динамической границе, сохранив динамику границы.

**ТП**: *инструмент* = шаг продукт-формулы по граничному блоку; *изделие* = граничный пограничный слой (boundary layer).
- ТП-1: если граничный блок продвигать **явно** (freezing, как у Stephan) → дёшево и совместимо с существующей Chernoff-машиной, но появляется вред: при неограниченном `∂_ν` коэффициент усиления `ρ → ∞` (амплификация `2.95 > 1`, witness ниже), слой неустойчив.
- ТП-2: если граничный блок продвигать так, чтобы при сколь угодно жёсткой связи усиление оставалось `≤ 1` → слой устойчив, но «наивная» явная формула это не даёт.
Выбрана половина ТП-2 (сохраняет главную полезную функцию — сходящийся kernel).

**Ресурсы (ВПР)**: библиотека уже содержит A-устойчивый шаг — **Cayley/Crank–Nicolson** §17.4 (`K_CN(τ) = (I − A)⁻¹(I + A)`, `‖K_CN‖₂ = 1` точно), и **резольвентную** инфраструктуру §22 (`(λI − A)⁻¹` через Gauss–Laguerre), и **Howland-лифт** §23 для нестационарного `γ(t)`. ОЗ = граничный DOF `u_∂`; ОВ = каждый под-шаг τ; поле = диссипативность генератора в весовом `L²(Ω) ⊕ L²(∂Ω)` скалярном произведении.

**ФП**: Х-элемент (шаг по границе) должен быть **«жёстко-связанным»** (передавать неограниченный `∂_ν`-обмен бульк↔граница) И **«усиление ≤ 1»** (не раскачивать слой при любой жёсткости).
Разрешение **переходом в структуру / физическим эффектом (преобразование Кэли)**: явный шаг `z_expl(λ) = 1 + τλ` посылает левую полуплоскость в неограниченную область (`|z|→∞` при `λ→−∞`); **неявный Cayley-шаг** `z_cay(λ) = (1 + τλ/2)/(1 − τλ/2)` — это дробно-линейное отображение, переводящее **замкнутую левую полуплоскость в замкнутый единичный круг**. Тот же неограниченный обмен теперь даёт `|z| ≤ 1` при любой жёсткости. Связь полностью сохранена, вред исчез.

**ИКР**: граничный блок **сам** держит усиление `≤ 1` при любом (в т.ч. зависящем от времени) `γ(t)`, без ограничения на шаг τ (без CFL-удушения слоя), используя уже имеющийся в библиотеке Cayley-механизм.

**Решение (класс)**: динамическую границу продвигать **неявным резольвентным (Cayley / backward-Euler) под-шагом**, а не явным freezing; нестационарность `γ(t)` нести Howland-лифтом §23. Это и есть механизм A-устойчивых bulk–surface расщеплений Альтманна–Ферфюрта–Циммера (литература ниже).

## Literature key findings (does Stephan's result preclude implicit treatment? — NO)

1. **Stephan 2023** (arXiv:2307.00419; pub. ZAMM 105(9), 2025). The divergence is **a property of the EXPLICIT "freezing" Trotter product**: the main convergence theorem (Thm 4.4, rate `O(log n / n)`) holds **only for BOUNDED off-diagonal coupling**; Section 5 discusses the unbounded case where convergence "cannot be expected even in the strong topology". The paper analyses *only* the explicit freezing scheme — it makes **no claim** that an implicit/resolvent boundary step diverges. So the obstruction is **explicit-only**, not intrinsic to the lift `X ⊕ L²(∂Ω)`.
2. **Kovács–Lubich 2015/2017** (arXiv:1501.01882, *IMA J. Numer. Anal.*). The heat equation with dynamic Wentzell BC fits the standard abstract parabolic framework with the `L²(Ω) ⊕ L²(∂Ω)` inner product; **stability and convergence of standard IMPLICIT integrators (backward Euler, BDF, algebraically-stable Radau IIA implicit Runge–Kutta) extend from Dirichlet/Neumann to dynamic BC**. Implicit ⇒ A-stable ⇒ unconditional stability — exactly the regime Stephan's explicit obstruction excludes.
3. **Altmann–Verfürth, "Bulk-surface Lie splitting" 2021** (arXiv:2108.08147, *IMA J. Numer. Anal.* 2023). An **operator-splitting** scheme for dynamic-BC parabolic problems: the boundary is reformulated as a coupled PDAE second dynamic equation; **each sub-step is solved by IMPLICIT (backward) Euler**, so the unbounded normal-derivative coupling does **not** destabilize the scheme — first-order convergence under a *weak* CFL `τ ≤ c·h` (mesh-coupling, not a stability collapse). This is the direct constructive remedy to Stephan's explicit divergence.
4. **Altmann–Verfürth, second-order variant 2022** (arXiv:2209.07835, *IMA J. Numer. Anal.* 2023): BDF / 3-step decoupling, second-order, under a weak CFL — implicit boundary treatment again.

**Conclusion of the literature scan**: Stephan's instability is **explicit-freezing-only**. Implicit / resolvent / Cayley boundary updates are an **established, peer-reviewed** stable path for dynamic Wentzell BC. The door ADR-0098 Am.3 left open ("until a stabilised product formula — e.g. implicit resolvent step on the boundary block — is published or validated") is, in fact, already open in the literature.

## Stability VERDICT: **GO** (amplification ≤ 1 demonstrated)

`scripts/wentzell_robin_stability_preflight.py` (sympy + numpy) performs a von-Neumann / 2×2 Fourier-symbol amplification analysis of the worst-case (highest-wavenumber) bulk+boundary coupled block, with the discrete normal-derivative coupling `−γ/dx` (the `O(1/dx) → ∞` unbounded operator), across a refinement sweep `dx ∈ {1/16 … 1/1024}` and scalings `γ ∈ {0.5, 1, 4, 16}`, step `τ = 0.4 dx²/a`.

| Scheme | max amplification over sweep | time-dependent γ(t) product (n=200) |
|--------|------------------------------|--------------------------------------|
| **Explicit freezing** (Stephan) | `ρ ≈ 2.9478 > 1` (every row) | **→ ∞** (overflow) |
| **Implicit Cayley** (candidate) | `ρ ≤ 0.999804 ≤ 1` | bounded, max `0.9756`, decays to `0.0071` |
| **Implicit backward-Euler** | `ρ ≤ 0.999805 ≤ 1` | (A-stable, same class) |

Symbolic witness (sympy), eigenvalue `λ = −μ`, stiff limit `μ → ∞`:
- Cayley: `1 − z_cay² = 8μτ/(μ²τ² + 4μτ + 4) ≥ 0 ⇒ |z_cay| ≤ 1`, and `lim_{μ→∞} z_cay = −1` (marginal, never exceeds 1).
- Explicit: `lim_{μ→∞} |z_expl| = ∞` (the Stephan blow-up).

The candidate amplification is `≤ 1` **unconditionally** (every refinement level, including time-dependent γ(t)) **exactly where the explicit scheme diverges**. Generator dissipativity in the weighted `diag(1, dx)` inner product is confirmed (`λ_max(C_sym^w) ≤ 0`). **VERDICT = GO; the contradiction is genuinely resolved (not split): the boundary carries a time-dependent scaling AND stays boundary-layer stable, via an implicit resolvent step.**

## Recommendation (for a later implementation wave — NOT this wave)

Ship a **stable dynamic-Robin/Wentzell kernel** built on the implicit-resolvent boundary step, reusing existing library infrastructure rather than inventing new machinery:

1. **Lift** to `X ⊕ ℝ_∂` (1D) / `X ⊕ L²(∂Ω)` (multi-D): bulk Chernoff step (existing `DiffusionChernoff`) split from a **boundary-block resolvent/Cayley sub-step** — mirror §17.4 `K_CN(τ) = (I − τC/2)⁻¹(I + τC/2)` on the 2×2-per-boundary-DOF coupled block (pentadiagonal/banded solve, as the Schrödinger kinetic step already does in `schrodinger.rs`).
2. **Time-dependent `γ(t)`** rides the §23 **Howland nonautonomous lift** (`TimedChernoffFunction`), so the boundary scaling may vary per step without re-deriving stability (the preflight confirms the per-step-frozen Cayley product stays `≤ 1`).
3. **Order-1** is the honest expected order (matching ADR-0098 static Robin and the Altmann–Verfürth Lie splitting); a second-order BDF/Strang boundary variant (arXiv:2209.07835) is a later research item.
4. **Gates** (when implemented): a `G_WENTZELL_STABLE` von-Neumann/amplification gate (Rust port of this preflight's `ρ ≤ 1` check) + a `T_WENTZELL` sympy gate (Cayley-map `|z| ≤ 1` symbolic identity + dissipativity). A **weak CFL** `τ ≤ c·h` is acceptable (it is mesh-coupling, not the Stephan stability collapse) and must be documented.
5. **Supersede ADR-0098 Am.3's "DEFERRED INDEFINITELY"** with "DEFERRED to a dedicated implicit-boundary wave; stability obstruction is explicit-only and resolved by the resolvent step (ADR-0146)."

This ADR makes **no `crates/` change and ships no kernel**; it converts the ADR-0098 Am.3 indefinite defer into a *scoped, stability-validated GO design* for a future wave. The HONEST-DEFER was correct for the explicit product formula; it does **not** apply to the implicit resolvent formulation, which this preflight demonstrates is A-stable.

## References

- A. Stephan, *Trotter-type formula for operator semigroups on product spaces*, arXiv:2307.00419 (2023); ZAMM **105**(9) (2025). — Thm 4.4 (bounded-coupling convergence); §5 (unbounded coupling: explicit freezing diverges, `‖T(t/n)^n‖ ≥ n^β`). **The obstruction is explicit-only.**
- B. Kovács, C. Lubich, *Numerical analysis of parabolic problems with dynamic boundary conditions*, arXiv:1501.01882; *IMA J. Numer. Anal.* **37**(1) (2017). — implicit backward-Euler / BDF / Radau IIA stable for dynamic Wentzell BC on `L²(Ω) ⊕ L²(∂Ω)`.
- R. Altmann, C. Verfürth, *Bulk-surface Lie splitting for parabolic problems with dynamic boundary conditions*, arXiv:2108.08147; *IMA J. Numer. Anal.* (2023). — implicit-Euler sub-step splitting, unconditional sub-step stability, weak CFL `τ ≤ ch`.
- R. Altmann, C. Verfürth, *A second-order bulk–surface splitting for parabolic problems with dynamic boundary conditions*, arXiv:2209.07835; *IMA J. Numer. Anal.* (2023). — second-order BDF decoupling.
- A. Iserles, *A First Course in the Numerical Analysis of Differential Equations* (Cambridge 2009), §8 — Cayley map (the library's §17.4 A-stable precedent).
- ADR-0098 + Amendment 3 (static Robin shipped; dynamic Robin HONEST-DEFER under the explicit product formula).
- math.md §17.4 (Crank–Nicolson Cayley map), §22 (Laplace-Chernoff resolvent), §23 (Howland lift) — reusable infrastructure for a future implementation.
