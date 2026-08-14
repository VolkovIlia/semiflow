# ADR-0196 — Per-pencil 2-D Strang composition for transverse-varying coefficients

- **Status**: Proposed (Issue #21; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: none — purely ADDITIVE. `Strang2D`, `AxisLift` and
  `strang2d_parallel.rs` are **untouched**, so the ADR-0018 bit-equality
  contract is not in scope for re-verification.
- **Contract**: `contracts/semiflow-core.math.md` §60 (new NORMATIVE section —
  the order argument genuinely changes); gates `G_PENCIL_REDUCTION`,
  `G_PENCIL_ORDER2`.

## Context

`Heat2DVarA` takes `a_x(x)` and `a_y(y)`: each diagonal coefficient may vary only
along **its own** axis. The reason is structural — `Strang2D` holds one `X`
kernel and one `Y` kernel, and `AxisLift`'s pencil loops apply that same `X` to
every row.

The Heston generator is what makes this bite. After the standard log-price +
decorrelation transform `z = x − (ρ/ξ)v` it is cross-term-free,

```
∂_τ u = ½v(1−ρ²)·u_zz + ½ξ²v·u_vv + b_z(v)·u_z + κ(θ−v)·u_v − r·u
```

but **both** diagonal coefficients depend on `v` — each varies along the *other*
axis. That single restriction is what stood between the library and a
one-kernel order-2 Heston solve; the issue's workaround is a hand-rolled Strang
composition of per-row/per-column `Shift1D` pencils in Python.

## Decision

**1. A sibling composition type, not a widening of `Strang2D`.** `Strang2DPencil`
holds `Vec<X>` (one per row) and `Vec<Y>` (one per column) and dispatches by
pencil index, reusing the existing `pencil::{row_2d, gather_y_2d_into, …}`
helpers and `grid_fn::apply_into_via_view`. The considered alternative — a
`PencilSource` trait dispatched from inside `AxisLift`, with the `pub(crate)`
bounds in `strang2d.rs` and `strang2d_parallel.rs` widened to match — is more
elegant and avoids a second palindromic loop, but it puts the release-blocking
ADR-0018 bit-equality machinery in the blast radius of a feature addition. A
~90-line self-contained loop is the cheaper risk.

**2. Serial only, deliberately.** The parallel X/Y passes in
`strang2d_parallel.rs` are exactly the code that carries the bit-equality
contract, and threading the pencil index through them is what the rejected
alternative required. `Strang2DPencil` has no parallel path; that is an honest
limit, recorded below, not an oversight.

**3. `order()` is `min` over pencils capped at 2; `growth()` is the sup.** A
per-pencil kernel of order 4 does not lift the composition above the τ² of
symmetric splitting, and a growth bound has to hold for the worst pencil, not
the first.

**4. The order *argument* changes, and §60 says so.** `Strang2D`'s module doc
justifies order 2 by `[L_x ⊗ I, I ⊗ L_y] = 0`, which makes palindromic Strang
exact at the BCH level. **With transverse-varying coefficients that premise is
false.** Order 2 is retained by the classical symmetric-splitting argument
instead: the τ² term of `e^{τA/2}e^{τB}e^{τA/2}` vanishes for arbitrary
non-commuting `A`, `B`, leaving `(τ³/24)([B,[B,A]] − 2[A,[A,B]]) + O(τ⁴)`. The
slope is unchanged; the error **constant** now carries those double commutators.

**5. The gate datum is calibrated, and the calibration is published.** Measured
slope on the `n_steps ∈ {20, 40, 80}` reference-free Richardson ladder:

| transverse amplitude | t = 0.005 | t = 0.02 |
|---|---|---|
| 0.1 | 2.871 | **2.053** |
| 0.3 | 2.723 | **1.913** |
| 0.6 | 1.774 | 1.175 |

At ±60% the ladder has not reached the asymptotic regime — the commutator term
still dominates at `τ = 2.5e−4`. That is decision 4 made concrete, not a defect,
and it is the same pre-asymptotic phenomenon ADR-0110 built a framework for. The
gate uses ±30%, which reaches order 2 at feasible step counts while remaining a
genuinely transverse field. Publishing the whole table rather than only the
passing row is the point: a reader can see where the method stops being usable.

**6. `a' ≡ 0` is carried over unchanged from `Heat2DVarA`.** This entry point
changes *which coefficients are expressible*, not *which operator is
discretised*. Mixing the two would confound the open question recorded at
`anisotropic_nd3.rs::build_axis_diff` with a feature addition.

## Rationale

The measurements did the deciding. A first attempt gated ±60% amplitude and read
slope 1.175; the constant-coefficient control then read 2.052 for **both**
`Strang2D` and `Strang2DPencil`, which localised the shortfall to the datum
rather than the composition, and the amplitude sweep showed the asymptotic
regime arriving exactly where the commutator argument predicts. Gating the
passing datum without publishing the sweep would have been the same number with
none of the information.

## Consequences

- New `crates/semiflow/src/strang2d_pencil.rs` (~215 lines) and
  `crates/semiflow-py/src/heat2d_pencil_py.rs`.
- `Heat2DVarA` gains `with_grid_arrays(...)` and an internal backend enum; the
  per-axis constructor and its results are unchanged.
- Memory: `ny + nx` kernels, each holding an `Arc<Vec<f64>>`. Total coefficient
  storage is the two `nx·ny` arrays the caller already supplies — the honest
  floor for a genuinely full-grid coefficient, and the same order as the state.

## Honest limits

- **Serial only.** No parallel path; `Strang2D`'s threaded X/Y passes are not
  reused (decision 2). For large grids the per-axis constructor remains faster.
- **No mixed derivative.** `∂_z∂_v` must be removed by the decorrelation
  transform before this API applies. The one-kernel order-2 Heston solve is
  available only *after* that transform, and this ADR does not ship the
  transform.
- **The error constant is commutator-driven.** The slope stays 2, but the
  practical τ range narrows as transverse variation grows — measured above.
  Users should verify the slope on their own coefficient field.
- **`NonSeparableMixedChernoff` is not upgraded**, and 3-D (`Strang3D`) is out of
  scope: the Z-axis pencil count becomes `nx·ny` and the memory argument must be
  re-derived.
- **No positivity or maximum-principle guarantee is added.** Per-pencil `a > 0`
  is validated; the splitting itself is unchanged.
