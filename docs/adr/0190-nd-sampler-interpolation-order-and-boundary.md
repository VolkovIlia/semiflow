# ADR-0190 — `GridFnND::sample`: honour interpolation order and boundary policy

- **Status**: Proposed (Issue #17; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Severity**: major — corrects a silently wrong result in every `D > 1` kernel family.
- **Supersedes / amends**: closes the `grid_nd.rs` v4.0 deferral ("Cubic Hermite
  per-axis interp deferred to v4.x"); amends ADR-0112 AMENDMENT 1 (the coarse
  `N_AXIS = 8` D-dimensional order ladder was calibrated *around* the defect this
  ADR removes) and supplies the mechanism ADR-0112 AMENDMENT 2 named as the
  implementation ceiling on the ζ² lift.
- **Contract**: `contracts/semiflow-core.math.md` §32.9 (new NORMATIVE subsection);
  gate `G_ASND_MOMENT`.

## Context

`GridFnND::sample` — the off-grid sampler every `D > 1` kernel calls — hard-coded
multilinear interpolation over `2^D` corners with an index clamp
(`grid_nd.rs:205-242`). It consulted neither the `InterpKind` nor the
`BoundaryPolicy` that each axis's `Grid1D` already carries, and the clamp silently
overrode `Reflect` (the default), `Periodic`, `ZeroExtend` and every other policy.
The 1-D family was never affected: `GridFn1D::sample` dispatches through
`Grid1D::interp_generic` to Catmull-Rom / Septic / Octonic / Chebyshev with full
boundary handling.

The consequence is not a small accuracy loss. Linear interpolation at cell fraction
`s` replaces a point mass with two masses, adding `s(1−s)·dx²` of second moment;
averaged over the quadrature feet this is ≈ `dx²/6` **per Chernoff step**. Because
the Chernoff product applies `n` steps, the spurious variance accumulates as
`n·dx²/6` — *linear in the step count*. Refining `n_steps`, the one knob a user
turns to improve accuracy, made the answer monotonically worse. Issue #17 measured
exactly this on `AnisotropicShiftND2` with `A = I`, `t = 0.5`, 96² grid: `dVar =
1.2113 / 2.2449 / 4.4901` at `n_steps = 100 / 400 / 1600` against an exact `1.0`.
A standalone numpy reimplementation of the §32 kernel reproduces those three
numbers to four decimals when — and only when — the sampler is multilinear, and
returns `1.0000` at every step count with Catmull-Rom.

No test caught it. `F(0) = I`, constant-preservation, and temporal
self-convergence are all blind to a per-step interpolation floor: it is the
identity at `τ = 0`, it preserves constants exactly (weights sum to 1), and the
scheme self-converges perfectly well to its own wrong answer. ADR-0112
AMENDMENT 1 had already *measured* the floor from the outside — it is why the
`G_DDIM` order gates were forced onto a deliberately COARSE `N_AXIS = 8` grid,
"so the temporal-truncation signal dominates the interpolation floor" — and
ADR-0112 AMENDMENT 2 named it again as the reason the ζ² lift's global rate is
"capped near O(τ¹) by interpolation/FD noise at off-grid GH nodes". Both
amendments described the symptom correctly and treated it as inherent. It was not.

The reported second defect — "axis mixing for diagonal tensors" — is **not** a
kernel defect. The library's flat layout is x-fastest (`flat = ix + iy·nx`,
`grid_nd.rs:6-7`), consistent with `coeff2d.rs` and the rest of the surface; the
report used a C-order `ravel()`. That is a documentation failure, addressed
separately (the codebase calls the layout "row-major", which is the opposite of
what numpy means for a `(nx, ny)` array, and `AnisotropicShiftND2` documented it
nowhere). It is recorded here only so the issue's two claims are not conflated.

## Decision

**1. `GridFnND::sample` evaluates a genuine tensor-product interpolant.** Per axis
it takes the nodal offsets and weights for the selected kind and resolves each
out-of-range node through that axis's own `BoundaryPolicy`, accumulating
`Σ_{k∈K^D} (Π_d w^{(d)}_{k_d})·f[idx + o_k]`. Boundary resolution goes through
`bc_value_by` — `bc_value_generic` refactored to take a node accessor instead of a
slice — so the 1-D and N-D paths share one implementation of the seven policies
and cannot disagree. The affine policies (`LinearExtrapolate`, `Dirichlet`,
`Robin`, `OddReflect`) are handled correctly because the collapse is recursive
per axis, not a flat index map; a flat map cannot express a policy that returns a
combination of three interior nodes or an additive constant.

**2. Weights live in one place.** `interp_stencil(kind, s)` returns
`(k, offsets, weights)`. `catmull_rom_scalar_generic` is deliberately **not**
rewritten in terms of it: reordering that arithmetic would perturb the f32 and
`Dual` scalar paths bit-for-bit, and this ADR must not touch the ADR-0018
bit-equality surface. The two forms are bound instead by
`catmull_rom_weights_match_the_generic_sampler`, which calls the real sampler.

**3. The N-D default is `CubicHermite`, on a single grid-level knob.** `GridND`
gains `pub interp: InterpKind` and `with_interp`. Per-axis `InterpKind` is
deliberately not consulted: `Grid1D::new` stamps every axis with `SepticHermite`,
so there is no way to distinguish a deliberate choice from an inherited default,
and silently promoting every N-D kernel to an `8^D` stencil is not a change this
ADR should make by accident. Catmull-Rom is sufficient on the merits — it removes
the defect entirely (measured residual ~1e-9, and flat in the step count, versus
septic's `8^D` cost for no measurable gain on this failure mode) — and `4^D` is
4× the previous work at `D = 2`, 8× at `D = 3`.

**4. Unsupported kinds fail loudly, once.** `SepticHermite`, `OctonicHermite` and
`ChebyshevSpectralWithBC` have no `D > 1` stencil: their nodal weights are a
composition of the Birkhoff–Garabedian–Lorentz polynomials (§40.3) with the
central-FD derivative stencils (§40.2), and extracting them means rewriting
samplers that carry a release-blocking bit-equality contract. The check runs once
in `GridFnND::new`, not per sample, because the hot kernels call `sample` inside
per-node loops and several of them discard the `Result` (`unwrap_or(F::zero())`)
in helper functions that return `F`. Validating at state construction makes the
error reachable without threading a `Result` through those helpers and without
leaving a silent-zero path.

**5. `apply_into_smoke_d2`'s non-negativity assertion is replaced, not relaxed.**
It asserted `v >= 0.0`, which held only because multilinear interpolation is
positivity-preserving — the very property that made it diffusive. Catmull-Rom, in
common with every high-order interpolant including the `SepticHermite` the 1-D
family has defaulted to since v6.0, has negative lobes. The replacement asserts
finiteness plus the stronger, measured statement that undershoot is an
under-resolution artifact which converges away: on the smoke grid the Gaussian is
resolved by ~1.4 nodes at `n = 8`, and `min/peak` runs `−1.8e−3` (n=8), `−1.4e−4`
(n=16), `−3.5e−10` (n=32), `+3.9e−21` (n=64). This is a strictly stronger claim
than the one it replaces, and it is recorded here because changing a test
assertion alongside the fix it guards otherwise reads as gate-weakening.

## Rationale

The library's own 1-D path had the right answer the whole time; the N-D path
simply never got wired to it. Preferring a self-contained N-D reimplementation
would have duplicated seven boundary policies and four interpolants, and the
duplication is precisely how the two paths came to disagree in the first place.
Reusing `bc_index`/`bc_value_by` and factoring the weights once is the smaller
change *and* the one that makes the class of bug structurally harder to
reintroduce. Choosing `CubicHermite` over `SepticHermite` for the default is the
honest reading of the evidence rather than a reflex toward the highest available
order: the measurement says cubic already saturates the fix.

## Consequences

- Results change — they become correct — for every `D > 1` kernel family:
  `shift_nd`, `shift_nd_zeta2`, `shift_nd_adaptive`, `smolyak`, `obstacle`,
  `obstacle_nd`, `point_eval`, `carnot_complex`, `carnot_stepk`,
  `hormander_engel`, and the ND surfaces in FFI and WASM. No signature changes.
- Per-step cost rises by `(K/2)^D` (4× at D=2, 8× at D=3).
- `GridND` gains a public field; `GridND { axes }` struct literals no longer
  compile. Acceptable in a `0.x` beta and recorded in `CHANGELOG.md`.
- The three flagship slope gates (G3⁶-2D, G4_NS2D_aniso, G5_3D) use
  Strang2D/NonSeparable2D and are unaffected. The `anisotropic-ddim-gates` job
  (`G_DDIM` D=2…5) is affected: its `N_AXIS = 8` ladder was calibrated against the
  removed floor and MUST be re-measured, not carried forward. ADR-0112
  AMENDMENT 1's reasoning is superseded to that extent; its record of *why* the
  coarse ladder was adopted stays, since it is the audit trail for this fix.
- `G_ASND_MOMENT` is added — the oracle whose absence let the defect ship.
  Measured after the fix (96² grid, `t = 0.5`): `A = I` gives `dVar = 1.000000`
  at `n_steps ∈ {100, 400, 1600}`; `A = diag(1.0, 0.5)` gives
  `(1.000000, 0.500000)`; `A = [[1,0.4],[0.4,1]]` gives `dCov = 0.400000`.

- `G_BINDING_SMOLYAK_PARITY`'s embedded golden values are regenerated. The
  regeneration is not cosmetic: the canonical datum is a symmetric Gaussian on a
  symmetric domain, so entries 0..3 (traversing `x₀ ∈ {−2, −⅔, +⅔, +2}`) must
  satisfy `g[0] == g[3]` and `g[1] == g[2]`. The new golden does; the old read
  `[1.98e−9, 4.26e−8, 4.34e−8, 4.34e−8, …]`, with `g[0]` and `g[3]` differing by
  a factor of 22. The index clamp had been breaking the reflection symmetry
  asymmetrically at the two ends, and the fix restores it.

## Related finding — left OPEN, deliberately

`Heat2DVarA` / `Heat3DVarA` build their per-axis kernels by passing `a' ≡ 0` and
`a'' ≡ 0` to `DiffusionChernoff` (`anisotropic_nd3.rs::build_axis_diff`). Since
`DiffusionChernoff` is documented as the ζ-A kernel for the **divergence form**
`∂_x(a(x)·∂_x)`, genuinely consumes both derivatives, and its own module doc
calls `|_| 0.0` the *constant-`a`* migration path, this reads as a bug of the
same family as the one this ADR fixes — a variable coefficient silently losing
the terms that make the discretisation the advertised one.

It is **not** changed here, on two grounds. First, all three binding surfaces —
`anisotropic_nd3.rs`, `semiflow-ffi/src/strang_nd_2d_ffi.rs`,
`semiflow-wasm/src/strang_nd_wasm.rs` — consistently advertise the
*non-divergence* operator `∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u`. Supplying the
derivatives would switch three public APIs to a different PDE without anyone
asking. Second, the evidence does not support the change: a 1-D A/B on
`Heat1D.with_a_array` (variable `a = 1 + ½sin(πx)`, `N = 129`, self-convergence
over `n_steps ∈ {20, 40, 80}` against a 1280-step reference) measured slopes
1.089 / 1.144 with zeroed derivatives and 1.029 / 1.089 with derived ones, with
*larger* absolute error in the derived case. Both sit at the O(τ¹) global
ceiling that ADR-0112 AMENDMENT 2 and math.md §9.2.3.B already document for
variable `a`, so a self-convergence measurement cannot discriminate between the
two candidate operators at all.

What would settle it: an analytic oracle for each candidate (`∂_x(a ∂_x u)` and
`a·u_xx`) on a manufactured solution, plus an architect decision on which
operator `Heat2DVarA` is contractually meant to be. Recorded in the rustdoc at
the call site and as a skipped test carrying this evidence, so the question stays
visible rather than being silently resolved either way.

## Honest limits

- `SepticHermite` / `OctonicHermite` / `ChebyshevSpectralWithBC` are unavailable
  for `D > 1` and return `Unsupported`. The N-D spatial floor is therefore
  `O(dx⁴)`, not the `O(dx⁸)` the 1-D family reaches. This bounds how far the
  `G_DDIM` gates can be pushed on a fine grid, and it is a real ceiling, not a
  temporary one — lifting it means composing the septic weight tables with their
  embedded FD stencils across all seven boundary policies.
- The tensor-product interpolant is not positivity-preserving. Kernels that need
  a non-negative state (density evolution near a degenerate boundary) must either
  accept `O(dx⁴)` undershoot in under-resolved regions or select
  `InterpKind::Linear` explicitly and pay the variance floor this ADR removes.
- Cost is `K^D` node reads per sample, and at high `D` that is not a rounding
  error. At `D = 5` `CubicHermite` reads 1024 nodes against multilinear's 32 — a
  32× interpolation cost. Measured: the two `D = 5` Smolyak plumbing smoke tests
  went from ~70 s to 198 s on an `n = 8` grid, which is why they were re-sized to
  the smallest legal grid (they assert finiteness and `F(0) = I`, neither of which
  depends on grid size; the `slow-tests` `G_SMOLYAK_D5` slope gate is untouched).
  This partially offsets what Smolyak sparse grids buy back on the quadrature
  side, and callers at `D ≥ 4` who have *measured* that the moment floor does not
  matter for their run can select `InterpKind::Linear` explicitly. What this ADR
  will not do is make that choice for them silently: the defect is
  dimension-independent, so a `D ≥ 4` default of `Linear` would quietly leave the
  bug in place exactly where it is hardest to notice.
- This ADR does not change the kernel formula, the quadrature, or the order
  claim. `AnisotropicShiftChernoffND::order()` remains 1 (ADR-0112 §Decision 2).
