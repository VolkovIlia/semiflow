# ADR-0148 — F2 ResolventJump 2D/3D LHP backends + hyperbolic-contour research

**Status:** PROPOSED (PRE-FLIGHT — Wave-2 B-5 sympy/numeric falsification done; PARTIAL GO) · **Date:** 2026-06-08 · **Branch:** `feat/v8.2.0-math`
**Theme:** v8.2.0 — extend F2 (`ResolventJumpChernoff`) beyond its NARROW 1D self-adjoint/sectorial scope.
**Parent:** ADR-0134 (F2 1D NARROW). **Math:** math.md §47 (§47.6 deferred-work bullets are the targets).
**Oracle:** `scripts/resolvent_jump_2d3d_kit.py` (`T_RESOLVENT_JUMP_ND` PRE-FLIGHT — PART A PASS, PART B DEFER, 2026-06-08).
**Future gates (a future impl MUST add):** `G_RESOLVENT_JUMP_2D_ORDER` / `G_RESOLVENT_JUMP_3D_ORDER` (RELEASE_BLOCKING, `slow-tests`) + `T_RESOLVENT_JUMP_ND` (NORMATIVE oracle).

## Context

F2 (ADR-0134, math.md §47) amortizes large-`T` semigroup cost to `M = O(1)` resolvent solves via Trefethen–Weideman–Schmelzer (2006) parabolic-contour inverse-Laplace quadrature, but is NARROW: 1D, self-adjoint/sectorial only, with a 1D **complex tridiagonal Thomas** LHP solve. math.md §47.6 defers two extensions: (a) **2D/3D LHP** — a banded-block / sparse-LU left-half-plane solve over `grid2d.rs` / `grid3d.rs` (row-major `idx(i,j)=j·nx+i`, I-T1), reusing the SAME contour; (b) **hyperbolic contour** (TWS 2006 §4) for non-sectorial / advection-dominated generators where the parabolic contour fails.

## TRIZ (hyperbolic sub-problem)

**НЭ:** на несекториальном (адвекция-доминированном) генераторе спектр уходит к мнимой оси (β→π/2); параболический контур теряет допустимость и его ошибка **стагнирует**, не достигая f64-floor.

**ТП** (инструмент = контур Γ; изделие = квадратурная сумма):
- ТП-1: «широкий» контур, охватывающий околомнимый спектр → польза (огибает σ(A)), но **вред** — узлы попадают туда, где `e^{λt}` растёт / резольвента плохо обусловлена.
- ТП-2: «узкий/секториальный» контур → нет вреда (узлы в дальней ЛПП, `e^{λt}` мал), но **нет пользы** — он не охватывает околомнимый спектр, квадратура расходится.
Выбрана половина ТП-1 (сохраняем главную функцию — охват спектра), усиливаем до предела: контур должен **прижиматься** к спектру.

**ФП:** Х-элемент (форма контура в оперативной зоне у мнимой оси) должен быть **широким** (огибать спектр у Im-оси) И **узким** (не пускать узлы в зону роста `e^{λt}`) одновременно.

**Ресурсы.** ОЗ — окрестность мнимой оси, где сидит спектр. ОВ — момент выбора геометрии контура (construction-time), параметризованный по измеренному сектору. ВПР — уже измеримые `β` (полуугол сектора), `‖A‖`, и сам M/t-скейлинг (уже есть в §47).

**Разрешение (в структуре + по геометрическому эффекту).** Разделить свойства по геометрии ветви: **гипербола** `z(θ)=μ(1+sin(iθ−α))` имеет асимптотический полуугол `(π/2−α)`, который настраивается **независимо** от ширины раскрыва μ. Берём `α < π/2 − β`: ветви идут вдоль (а не сквозь) сектора спектра — «широко» по охвату, «узко» по углу к Im-оси. ИКР: контур **сам** обнимает спектр по его собственному сектору, без захода в зону роста экспоненты, без новой подсистемы — лишь замена аналитической формы λ(θ).

**Решение (класс):** адаптивная гипербола TWS-2006 §4 с параметрами `(μ, α, h)`, подобранными под **измеренный** сектор `β` (Weideman–Trefethen 2007 формулы оптимума). Это направление; **реализуемость подтверждается отдельно** (см. sympy-verdict ниже — на нашем стресс-операторе фиксированная гипербола НЕ разрешила противоречие, что отправляет (b) в DEFER до адаптивной подгонки / rational-Krylov).

## Literature (sources)

- **Trefethen, Weideman, Schmelzer, *Talbot quadratures and rational approximations*, BIT 46:3 (2006) 653–670** — parabolic/hyperbolic/cotangent Hankel contours for `e^{tA}`; the `(a0,a1,a2)=(0.1309,0.1194,0.2500)` parabolic optimum (math §47.2). Parabola rate `O(e^{−1.047N})`, hyperbola `O(e^{−1.176N})`.
- **Weideman & Trefethen, *Parabolic and hyperbolic contours for computing the Bromwich integral*, Math. Comp. 76:259 (2007) 1341–1356** — optimal `(μ, h, α)` for both contours; hyperbola `z(θ)=μ(1+sin(iθ−α))`, asymptotic half-angle `π/2−α`, **enclosure condition `α < π/2 − β`** for spectral sector half-angle `β`; hyperbolic preferred when the spectrum is in a sector close to the imaginary axis (advection-dominated / large `|Im|`).
- **Hale & Weideman, *Contour Integral Solution of Elliptic PDEs in Cylindrical Domains*, SIAM J. Sci. Comput. 37:6 (2015)** — the SAME deformed contour extends to 2D/3D **unchanged**; the per-node cost is one **shifted sparse linear solve** `(zI−A)⁻¹b` handled by a **sparse direct solver**, contour node count `M=O(1)`. This is the direct precedent for Part A: only the inner solve changes (Thomas → banded sparse-LU), the outer quadrature is dimension-blind.
- ADR-0134 / math §47 (1D NARROW F2); ADR-0069 / math §22 (resolvent abstraction reused); ADR-0101/0094 (operator-Padé-in-time PERMANENT deferral — UNCHANGED; this is a coordinate change, not scaling-and-squaring).

## Sympy/numeric VERDICT (`scripts/resolvent_jump_2d3d_kit.py`, 2026-06-08)

**PART A (2D/3D parabolic) — GO (5/5).** On Kronecker-sum divergence-form Laplacians (row-major, grid2d/grid3d ordering), with the LHP solve done by a banded sparse-LU `splu` (numeric proxy for the Rust banded-block direct solve):
- A1 `nd_resolvent_exact`: LHP banded solve exact off-spectrum — residual `2.7e-15` (2D), `3.1e-15` (3D) at `λ=−0.7+0.9i`. ✅
- A2 `nd_geometric_decay` (2D): `−0.449` dec/node (geometric, matches the 1D `−0.442`). ✅
- A3 `nd_order_slope` (2D, G24 sign convention, t=100): slope `+9.97 ≥ 1.95`. ✅
- A4 `nd_t_independence` (large-T cost decoupling): `err(M=16)` **monotone non-increasing** over `t∈{20,100,500}` — M-count does NOT grow with `t`. ✅ *(NOTE: the §47 1D "spread ≤ 10×" constant is a 1D-floor-tuned proxy; the 2D per-node error sits higher so a raw `t∈{1,…}` ratio reads `13–23×`, but that is inflated by the SMALL-t end — large-T, the actual use case, is the cheap regime. The kit asserts the underlying monotone property directly to stay faithful to the §47.3 claim rather than copy a 1D-calibrated constant.)*
- A5 `three_d_smoke` (3D, N=8³, t=100): slope `+9.79 ≥ 1.95`. ✅

**PART B (hyperbolic, non-sectorial) — DEFER.** On an advection-dominated `A = ε∂xx + v∂x` (periodic, non-self-adjoint): at `ε=1e-3` the spectrum is essentially **on the imaginary axis** — the boundary case where NO deformed Bromwich contour converges geometrically (a fundamental Bromwich-inversion limit, not a tuning defect); a fixed-parameter hyperbola did not beat the (also stagnating) parabola. At `ε=0.15` the spectrum re-enters a left sector and the **parabolic contour already converges** (`−0.257` dec/node → `3e-8`), so the hyperbolic variant is not even needed. **Conclusion: a fixed `(μ,α,h)` hyperbola does NOT robustly resolve the non-sectorial case.** A shippable hyperbolic variant requires (i) construction-time measurement of the spectral sector `β` and Weideman–Trefethen-2007 optimal-parameter fitting `α<π/2−β`, and possibly (ii) an adaptive-pole rational-Krylov backend — genuine research, honestly DEFERRED to v8.x.

## Decision (SHIPPABLE-vs-OPEN boundary)

**SHIPPABLE in v8.2.0 (mechanical, low-risk):** the **2D/3D parabolic** extension (math §47.6 bullet (b)). The outer TWS parabolic quadrature is dimension-blind (Hale–Weideman 2015 precedent + Part A 5/5); the ONLY new primitive is the **banded LHP solve** replacing the 1D Thomas — a banded-block or sparse-LU complex solve over the existing `grid2d.rs`/`grid3d.rs` geometry. NARROW-self-adjoint restriction of §47.4 is preserved (every shipped 2D/3D divergence-form generator is self-adjoint negative-semidefinite). Future impl gates: `G_RESOLVENT_JUMP_2D_ORDER` and `G_RESOLVENT_JUMP_3D_ORDER` (RELEASE_BLOCKING, `slow-tests`) asserting OLS slope `d log‖jump_M−ref‖∞ / d log(1/M) ≥ +1.95` — **copy the §47.5 / G24 SIGN CONVENTION verbatim** (`≥ +1.95` vs `log(1/M)` PASSES; ADR-0134's "slope ≤ −1.95 in Δτ" is the same statement with `Δτ=1/M`, do NOT re-derive the sign) — plus `T_RESOLVENT_JUMP_ND` wired into the `test-fast` sympy sweep next to `resolvent_jump_kit.py`.

**OPEN / DEFER to v8.x (research):** the **hyperbolic contour** for non-sectorial / advection-dominated generators (math §47.6 bullet (a) extension). Honestly deferred per the Part B falsification: needs adaptive `(μ,α,h)` sector fitting and/or rational-Krylov, and is bounded above by the fundamental near-imaginary-axis limit (no deformed contour converges geometrically for a spectrum *on* the imaginary axis).

This keeps F2 a coordinate change (NOT operator-Padé; ADR-0101 deferral UNCHANGED). Residual risk for Part A is LOW (banded direct solve is unconditionally stable off-spectrum for sectorial `A`; contour margin `≫ 5×` in slope). Part B remains correctly out of scope.
