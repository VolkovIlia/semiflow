# §71 — τ-keyed step plans (ADR-0215, NORMATIVE library)

## §71.1 — Hook

`ChernoffFunction::plan_len(τ, proto) → Option<(data, work)>`,
`fill_plan(τ, proto, data)`, `apply_planned_into(τ, data, work, src, dst, scratch)`.
Contract: for every `src`, `apply_planned_into(τ, data, …)` with `data` filled by
the same engine for `(τ, proto)` writes exactly (bitwise) what `apply_into(τ, src, …)`
writes; when `(τ, src)` does not match the plan's header the engine runs
`apply_into`. `fill_plan` fails exactly when `apply_into` would fail for `τ` and
the proto's grid on account of `τ` or the coefficients.

## §71.2 — Planned sample (NORMATIVE)

For a sample position `x` on grid `g` (`PreparedGridFn` layout of `g.interp`):
`(idx, s) = cell_of(g, x)`; covered iff the cells the layout reads lie in
`[−MARGIN, n − 1 + MARGIN]`; then `offset = lo + MARGIN` and the weights are the
layout's weight vector at `s` (septic `septic_weights(s)`, octonic
`octonic_weights(s)`, Catmull–Rom `s` itself); otherwise `offset = −1` and the
sample is evaluated by `PreparedGridFn::sample(x)`. A planned sample combines the
stored weights with the step's ghost table in the layout's combination order.

## §71.3 — `DiffusionChernoff` plan

Header `[tag, τ, len, engine, xmin, xmax, n, boundary(3), interp(2)]`, then
`[stride, τ²]`, then per node `i` (`x = engine.grid.x_at(i)`): constant `a` —
planned samples at `x, x ± 2√(aτ), x ± 2√(3aτ)`; variable `a` — the six ζ scalars
`2δ, δ², 2δ³, a·a′, a·a″/2, a′·a″/4` (`δ = max(√τ, 2dx)`) and planned samples at the
five γ-A positions and `x, x ± δ, x ± 2δ`, every quantity by the expression of the
unplanned kernel.

## §71.4 — Gates

`G_STEP_PLAN_BIT_EQUAL`, `G_STEP_PLAN_ZERO_ALLOC` (ADR-0215).
