# ADR-0150 — Obstacle evolver second-order Greeks (Γ) + multi-asset D≥2 (research / pre-flight)

- Status: Proposed (RESEARCH PRE-FLIGHT — no `crates/` edits, no `cargo`)
- Date: 2026-06-08
- Designed-By: researcher
- Extends: ADR-0116 (projective-splitting `ObstacleChernoff`), math.md §44.5.
- Authority for: the future `G_OBSTACLE_GAMMA` / `T_OBSTACLE_GAMMA` (7a) and
  `G_OBSTACLE_SLOPE_2D` (7b) acceptance gates.

## Decision (≤1 paragraph)

For **7a (Γ at the free boundary)** the answer is **GO via inactive-set
restriction, with mollification as a SHOULD-only option, and an HONEST DOCUMENTED
LIMIT that no classical global Γ exists at/across the contact line.** The value
function of the obstacle problem is C¹ across the free boundary `x*` (smooth-fit:
value and slope continuous — Peskir 2005; Jaillet–Lamberton–Lapeyre 1990) but its
second derivative Γ = V″ is **discontinuous** there (Γ = 0 on the active/stopping
set where `V ≡ g` is linear, Γ > 0 just inside the inactive/continuation set).
sympy verified this exactly for the canonical closed form (perpetual American put,
`q = 0`): `V = A·S^{−γ}`, `γ = 2r/σ²`, `S* = γ/(γ+1)·K`, with a strictly positive
jump `Γ(S*⁺) = (2r+σ²)²/(2Krσ²) > 0` while `Γ(S*⁻) = 0`. Therefore a single global
Γ is **intrinsically ill-posed at the contact line** — the TRIZ resolution is not
to compute Γ there at all (resolve the contradiction *in space*): expose Γ **only
on the open inactive set** `{W > g}`, refusing/masking it on the active set and a
guard band around `x*`. The numeric kit confirms this restricted Γ converges to the
analytic Γ at `O(Δx²)` with **no mollification**, matching the literature finding
that L-stable schemes recover Γ in a region-of-interest well inside the
continuation set without smoothing (Casabán et al. 2024; Forsyth–Vetzal report
spurious oscillations only *near* the boundary). Mollification `g → g_ε` (softplus)
is a legitimate SHOULD: Γ_ε stays bounded everywhere and is faithful on the open
inactive set, but it **cannot manufacture a classical Γ in the O(ε) boundary
layer** — it tracks the mid-kink subdifferential value there (≈ jump/2), which is
the same ill-posedness restated, not a cure. For **7b (D≥2 multi-asset)** the
answer is **GO by construction (mechanical)**: the projection `Π_g(W)=max(W,g)` and
the active-set mask `𝟙[W>g]` are **elementwise** and carry no dimension
assumption; `grid_nd.rs` (`GridND`/`GridFnND<F,D>` with flat `values: Vec<F>` +
`x_at(d,k)`) and `grid2d.rs` already expose exactly the flat-storage + coordinate
surface the D=1 kernel uses, so generalizing `ObstacleChernoff<C,O,F>` from
`GridFn1D` to `GridFnND` is a low-risk type/loop generalization.

## TRIZ (7a — Γ through a kink)

- **АП (НЭ):** need the second derivative (Γ) of a value function whose first
  derivative is *kinked* at the contact line, where a naive Γ of the projected
  iterate `Π_g(W)=max(W,g)` is non-differentiable and produces noisy/oscillating
  estimators.
- **ТП:** *инструмент* = the Γ-estimator (differentiator), *изделие* = the value
  field `V`.
  - ТП-1: a **global** Γ-estimator covers the whole grid (польза: one uniform Γ
    field) but is **dishonest at `x*`** (вред: Γ jumps; FD oscillates — Forsyth–
    Vetzal).
  - ТП-2: a **restricted** Γ-estimator avoids `x*` (no вред) but **leaves Γ
    undefined on the active set** (loses польза there).
  - Chosen half: **ТП-2**, sharpened — keep Γ exactly where it is mathematically
    real (the *open* continuation set) and *declare* it absent elsewhere.
- **ФП:** the Х-element (the Γ field) must be **present** (classical, bounded) on
  the inactive set AND **absent** (refused, not faked) on the active set / at `x*`.
- **Ресурсы:** ОЗ = the contact line `x*` and its O(Δx) neighbourhood; ОВ = the
  same forward step that already computes the active-set mask; ВПР = the
  **already-existing `apply_active_set_adjoint_into` indicator mask**
  `diag(𝟙[W_fwd > g])` — the exact diagonal that separates inactive from active.
- **Разрешение (in space) + ИКР:** resolve the physical contradiction *by space*
  — Γ is present on `{W>g}` and absent on `{W≤g}`, using the mask already present
  in the kernel. **ИКР:** "the obstacle evolver itself reports Γ where Γ exists and
  marks it absent where it does not — at zero extra cost, reusing its own active-set
  mask." This is a genuine resolution, not a golden-middle compromise: we do **not**
  average a fake Γ across the kink; we give the *full* classical Γ where it lives
  and an honest *absent* flag where it does not.

## Literature findings (sources)

- **Smooth-fit is C¹, not C²** — value and Delta continuous across `x*`, Gamma
  generically discontinuous (Peskir 2005, *On the American Option Problem*; Jaillet–
  Lamberton–Lapeyre 1990). Free-boundary regularity (e.g. C^∞ for convex payoffs,
  several assets) *requires* continuity of V″ to be *proven*, and it holds on each
  side separately, not across `x*` (Wiley CPA, *Regularity of the free boundary of
  an American option on several assets*).
- **Numerical Γ near the boundary oscillates** — Forsyth–Vetzal spurious Gamma
  oscillations near the early-exercise boundary; cured by L-stable time-stepping
  (DIRK/Lobatto) and by **restricting the region-of-interest to well inside the
  continuation region**, *without* mollification (Casabán, Company, Egorova,
  Jódar, *A note on the numerical approximation of Greeks for American-style
  options*, arXiv:2401.13361, 2024). This is the direct external endorsement of the
  inactive-set route.
- **Mollification / penalty bias** — penalty and mollified-payoff approximations
  give boundary-layer width `δ = σ√ε` with value `O(1/ρ)=O(ε)` and derivative
  `O(ρ^{−1/2})=O(√ε)`; the *second* derivative degrades further in the layer
  (Howison–Reisinger–Witte 2013; *The Effect of Non-Smooth Payoffs on the Penalty
  Approximation of American Options*, arXiv:1008.0836). Mollified-obstacle solutions
  and gradients converge a.e. to the original (irregular time-dependent obstacles,
  arXiv:1011.1901), but the **second derivative is exactly where convergence is
  lost in the layer** — consistent with our kit.

## sympy / numeric verdict (`scripts/obstacle_gamma_kit.py`)

5/5 sub-checks PASS → **OVERALL VERDICT: GO**.
1. `closed_form` — perpetual-put `V` solves the continuation ODE; C¹ smooth-fit
   (value+slope) holds; `Γ(S*⁺)=(2r+σ²)²/(2Krσ²)>0`, `Γ(S*⁻)=0` ⇒ C¹-not-C². PASS.
2. `gamma_jump` — numeric jump `= 4.90 > 0` (single global Γ ill-posed at `x*`). PASS.
3. `inactive_restrict` (**ROUTE A, primary**) — central-FD Γ at an interior
   continuation node converges to analytic Γ at **observed order ≈ 2.00**, no
   mollification. PASS.
4. `mollified_eps` (**ROUTE B, SHOULD**) — softplus-mollified Γ_ε is faithful on
   the fixed inactive set (`|Γ_ε−Γ| ≤ 1e-3`) and **bounded everywhere** (no
   blow-up), but in the O(ε) layer it tracks the mid-kink value (≈2.45 ≈ jump/2)
   and does **not** recover a classical Γ — the honest limit. PASS.
5. `d2_mechanical` — `Π_g` and `𝟙[W>g]` are elementwise on a flat array of any
   length (emulated D=3, len 60): `Π_g(W)=W·𝟙[W>g]+g·𝟙[W≤g]`, idempotent ⇒ 7b
   mechanical. PASS.

## Gates a future implementation needs

- **7a — `T_OBSTACLE_GAMMA`** (sympy PRE-FLIGHT oracle, RELEASE_BLOCKING): the
  five sub-checks above (closed-form C¹-not-C², jump>0, inactive-set O(Δx²)
  convergence, mollified bounded+faithful, D-agnostic mask). Script:
  `scripts/obstacle_gamma_kit.py`.
- **7a — `G_OBSTACLE_GAMMA`** (Rust slope/correctness gate, RELEASE_BLOCKING):
  Γ computed by the new primitive on the **open inactive set** (guard band ≥1 node
  off `x*`) vs the perpetual-put analytic Γ — OLS log-log slope in Δx **≤ −1.95**
  (O(Δx²)); MUST refuse / flag-absent Γ on the active set and across `x*` (no
  fabricated value). Honesty boundary mirrors ADR-0116: the Γ primitive is a
  **separate inherent method** (e.g. `apply_inactive_gamma_into`) reusing the
  active-set mask, NOT a `ChernoffFunction` Greek; `ObstacleChernoff` MUST NOT claim
  a global C² Greek.
- **7b — `G_OBSTACLE_SLOPE_2D`** (Rust slope gate, RELEASE_BLOCKING): the D=2
  `ObstacleChernoff<C,O,GridFnND<F,2>>` projected-scheme self-convergence slope in
  Δτ on a 2-asset stationary-membrane / smooth oracle — slope ≤ −0.95 (smooth,
  order-1 ceiling) mirroring `G_OBSTACLE_SLOPE_SMOOTH`, plus a 2D stationary
  correctness check mirroring `G_OBSTACLE_STATIONARY`.

## Shippable in v8.2.0 (recommendation)

- **7b (D≥2) — SHIP.** Mechanical, low-risk, GO by construction; the `grid_nd`
  path exists. Generalize `ObstacleChernoff` storage `GridFn1D → GridFnND` and add
  `G_OBSTACLE_SLOPE_2D` + a 2D stationary oracle.
- **7a (Γ) — SHIP only the inactive-set primitive + honest limit, OR defer.**
  The classical Γ exists and is gateable **only on the open inactive set** (the
  mollified global Γ is provably non-classical in the layer). A v8.2.0 ship would
  add `apply_inactive_gamma_into` (reusing the active-set mask) gated by
  `T_OBSTACLE_GAMMA`/`G_OBSTACLE_GAMMA`, with the contact-line Γ **documented as an
  open/absent quantity** (C¹-not-C²), exactly as ADR-0116 §44.5 already flags. No
  global C² Greek may be claimed.

## References

Peskir 2005; Jaillet–Lamberton–Lapeyre 1990; Howison–Reisinger–Witte 2013
(doi:10.1137/090776089); Casabán–Company–Egorova–Jódar 2024 (arXiv:2401.13361);
arXiv:1008.0836 (non-smooth payoff penalty); arXiv:1011.1901 (mollified obstacles);
Wiley CPA (free-boundary regularity, several assets); Forsyth–Vetzal (spurious Γ
oscillations); ADR-0116 / math.md §44.5.
