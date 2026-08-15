# ADR-0191 — `GridFnND::sample`: honour interpolation order and boundary policy

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

## AMENDMENT 1 (2026-08-14) — the `Heat2DVarA` `a' ≡ 0` question, settled

The original text of this section left the `a' ≡ 0` / `a'' ≡ 0` closures in
`anisotropic_nd3.rs::build_axis_diff` recorded as an OPEN question: a
self-convergence measurement could not discriminate between the two candidate
operators, because both sit at the `O(τ¹)` global ceiling that ADR-0112
AMENDMENT 2 and math.md §9.2.3.B already document for variable `a`. What was
missing was an analytic oracle. It has now been built, and the question is
closed.

### The oracle

On a periodic grid (`N = 128`, `a(x) = 1 + ½sin 2πx`, `t = 0.02`) the two
candidate generators were assembled as dense matrices — `diag(a)·D₂` for the
non-divergence form `a(x)·u_xx`, and the conservative arithmetic-face operator
for `∂_x(a ∂_x)` — and exponentiated by scaling-and-squaring. They differ by
`7.8e−2` in sup norm, so the comparison is not vacuous. `Heat1D.with_a_array`
accepts explicit `a_prime` / `a_double_prime`, which lets the *same* kernel be
run in both configurations:

| kernel configuration | `‖u − a·u_xx‖` | `‖u − (a u_x)_x‖` |
|---|---|---|
| `a' = a'' = 0` (the `Heat2DVarA` per-axis kernel) | **6.8e−3** | 7.4e−2 |
| `a'`, `a''` from FD (the shipped 1-D path) | 8.2e−2 | **6.7e−3** |

Both residuals are flat in `n_steps` from 200 to 12800, i.e. they are the spatial
discretisation error, not a temporal one. Each configuration is an order of
magnitude closer to one reference than to the other.

### Verdict

The zeroed closures are **correct**, and produce exactly the operator all three
binding surfaces advertise. Zeroing them collapses the kernel to the
frozen-coefficient stencil — nodes at `x`, `x ± 2√(a(x)τ)`, `x ± 2√(3a(x)τ)`
with the Gauss–Hermite weights, and the entire ζ-A correction identically zero
because every one of its terms carries a factor `a'` or `a''` — which is a
consistent discretisation of `a(x)·∂_xx`. No API changes PDE.

### What *was* wrong: the order claim

`Heat2DVarA::order()` and `Heat3DVarA::order()` returned **2**. They do not earn
it. The Strang composition is second-order, but composition order is capped by
its axis kernels, and the frozen-coefficient stencil expands as

```text
  S(τ)f      = f + τ·a f'' + (τ²/2)·a² f'''' + …
  e^{τa∂ₓₓ}f = f + τ·a f'' + (τ²/2)·a (a f'')'' + …
```

which agree at `O(τ)` and differ at `O(τ²)` by `(τ²/2)·a·(a'' f'' + 2a' f''')`
whenever `a` varies. Consistency order is therefore 1. Measured on the same
datum by self-convergence over `n_steps ∈ {80, 160, 320, 640}` against a
40960-step reference: **slope −1.007** for the zeroed kernel, against −1.459 for
the divergence-form path.

Both now report `order() == 1`, on the ADR-0112 precedent ("correct `order()` to
1 rather than keep an unearned claim"). `DiffusionChernoff::order() == 2` is
untouched and remains correct: that order is earned when `a'`/`a''` are actually
supplied.

### Gates

`G_FROZEN_COEFF_NONDIV` and `G_FROZEN_COEFF_ORDER1`
(`crates/semiflow/tests/frozen_coeff_operator.rs`, Pattern B). The first carries
the dense-reference comparison, including the non-vacuity check that the two
references genuinely differ; the second pins the order-1 slope, and fails loudly
if the kernel ever becomes order 2, since `order()` would then be wrong again in
the other direction.

Note what this closes that no previous test could: every existing oracle for this
kernel — `F(0) = I`, constant preservation, temporal self-convergence — is
satisfied by *both* candidate operators, and the one quantitative test used
`a ≡ 1`, exactly the case where they coincide.

## AMENDMENT 2 (2026-08-14) — the FFI/WASM `boundary` mirror is not a parity gap

The original plan called for mirroring the new `boundary=` kwarg on
`AnisotropicShiftND2/3` into the C ABI and the WASM surface, per the bindings
parity rule. That turned out to rest on a false premise, so it is **not** done,
and the reason is recorded here rather than left as a silent omission.

Neither surface exposes boundary selection **for any kernel**. Every FFI
constructor hard-codes `.with_boundary(BoundaryPolicy::Reflect)` — `adaptive_ffi`,
`cdr_ffi`, `diffusion_hi_zeta_ffi`, `expmv_ffi`, `drift_reaction_zeta4_ffi` and
the rest — and `semiflow-wasm` never calls `with_boundary` at all, so every WASM
kernel runs on the `Reflect` default. This predates the campaign: `Shift1D` has
carried `boundary=` in Python since before it, against an FFI `Shift1D` that
cannot express anything but `Reflect`.

Adding the parameter to exactly the two ND constructors would therefore not
restore parity; it would create a new inconsistency inside the C ABI, where one
constructor of forty takes a boundary and the rest silently do not. Exposing
boundary selection across the C and WASM surfaces is a coherent piece of work —
one shared parser, one enum in the header, one sweep of the constructors, and an
ABI decision about whether to extend existing entry points or add `_bc` variants
— and it belongs in its own change, not as a two-function exception here.

## AMENDMENT 3 (2026-08-14) — `G_DDIM` fails, and the reason is that its estimator was measuring the wrong thing

**Status**: RESOLVED. The estimator is replaced, the threshold is re-based to
the measured order, and the ladder is re-sized to be runnable. Approved by the
architect; the commit carries `Gate-Change-Approved-By`.

`G_DDIM D=2` (`RELEASE_BLOCKING`, `anisotropic_shift_nd_d2_slope.rs`) now reads
`slope = −0.9249` against a `≤ −0.95` gate. It read `≈ −1.03` before this ADR.
Nothing about the gate changed; what changed is the sampler underneath it. The
investigation below says the gate was never measuring what it claimed, and the
kernel's true temporal order on its own normative datum is **½**, not 1.

### The estimator is contaminated

The gate is a self-convergence test: reference at `n_ref = 512`, sweep
`n ∈ {32, 64, 128, 256}`. The largest swept `n` is **half** the reference, so the
last point's "error" is dominated by the reference's own remaining temporal
error. Holding the sweep fixed and raising only `n_ref` shows it directly:

| `n_ref` | 512 | 1024 | 2048 | 4096 | 8192 |
|---|---|---|---|---|---|
| slope, `N_AXIS = 8` | −0.925 | −0.718 | −0.625 | −0.574 | −0.543 |
| slope, `N_AXIS = 16` | −0.934 | −0.727 | −0.634 | −0.583 | −0.552 |

A converged estimator would be flat in `n_ref`. This one drifts monotonically
toward ≈ −0.55, i.e. the `−0.95` the gate used to see was an artefact of a
reference only 2× finer than the datum.

### The order is ½

Successive differences `sup|u_{2n} − u_n|` need no reference and cannot be
contaminated. Their ratio is the convergence factor per halving of τ:

| `n → 2n` | 32→64 | 64→128 | 128→256 | 256→512 | 512→1k | 1k→2k | 2k→4k | 4k→8k | 8k→16k |
|---|---|---|---|---|---|---|---|---|---|
| ratio, `N_AXIS = 8` | — | 1.374 | 1.383 | 1.391 | 1.398 | 1.402 | 1.406 | 1.408 | 1.410 |
| ratio, `N_AXIS = 16` | — | 1.397 | 1.398 | 1.401 | 1.404 | 1.407 | 1.409 | 1.410 | 1.411 |
| ratio, `N_AXIS = 32` | — | 0.917 | 1.294 | 1.371 | 1.384 | 1.393 | 1.399 | 1.404 | 1.407 |

The ratio settles on **√2 = 1.414**, stably across three grid resolutions and
nine octaves of `n`. Order 2 would give 4, order 1 would give 2. This is
`O(τ^{1/2})`.

The mechanism is consistent with the kernel's own structure. The Gauss–Hermite
shift is `2√(a τ)·η`, and for a *variable* `A` the coefficient is frozen at the
node while the sample is taken at the shifted point, so the operator mismatch is
`A′·2√(aτ)·η` — a relative `O(√τ)` on the leading `O(τ)` term, i.e. local
`O(τ^{3/2})` and global `O(τ^{1/2})`. ADR-0112's "honest order-1" accounted for a
variable-`A` per-step mismatch of `O(τ²)`; the `√τ` inside the shift makes it
`O(τ^{3/2})`.

### It is not a regression

The same probe on the pre-campaign baseline (`0e6d25b`) gives ratios of
1.28 / 1.35 / 1.45 / 1.66 / 1.90 / 1.73 — noisy, no clean power law — with
absolute differences an order of magnitude larger (`1.9e−2` against `2.2e−3` at
`n = 32→64`, `N_AXIS = 8`). So the order was never 1. What this ADR changed is
that the kernel became ~10× more accurate and its convergence became clean
enough to read.

### Resolution

**1. The estimator is replaced.** The gate no longer compares against a single
reference run. It fits the OLS slope of the **successive differences**
`sup|u_{2n} − u_n|`, which need no reference and therefore cannot be
contaminated. This is strictly stronger than what it replaced — the old reading
of ≈ −1.0 was an artefact of a reference only 2× finer than the datum.

**2. The threshold is re-based to the measured order.** `−0.95 → −0.45`, with a
two-sided band: a `SLOPE_CEILING` of `−0.75` fails the gate if the kernel ever
becomes genuinely order-1, so the correction is caught rather than silently
passing. Measured after re-basing: **D=2 −0.4676, D=3 −0.4766**, against the
theoretical `−0.5` that the ladder approaches from above.

**3. `order()` is left at 1 for now.** `ChernoffFunction::order()` returns `u32`
and cannot express ½; truncating to 0 would change adaptive step control. The
honest statement lives in the gate, in this ADR and in math §32.5 rather than in
a value the type cannot hold. Whether `shift_nd_zeta2`'s ζ² correction lifts the
*global* order to 1 — which would make `order() == 1` earned — is the follow-up,
and must be measured with the uncontaminated estimator introduced here.

### The ladder had also become unrunnable

ADR-0191 made a sample read `K^D` nodes. At `D = 4` that is `4⁴ = 256` against
multilinear's `2⁴ = 16`, and combined with the `5^D` Gauss–Hermite nodes the old
`n_ref = 512` configuration measured **8105 s at D = 4** — up from ~6 min — and
extrapolates to ~85 h at `D = 5`, past any runner limit.

The reference-free ladder fixes most of that by needing far fewer steps:

| D | old (sweep + n_ref) | new ladder | steps | grid |
|---|---|---|---|---|
| 2 | {32,64,128,256} + 512 | {32,64,128,256,512} | 992 | `N_AXIS = 8` |
| 3 | {32,64,128,256} + 512 | {32,64,128,256,512} | 992 | `N_AXIS = 8` |
| 4 | {16,32,64,128} + 512 | {8,16,32,64} | 120 | `N_AXIS = 8` |
| 5 | {16,32,64,128} + 512 | {4,8,16,32} | 60 | `N_AXIS = 6 → 5` |

`D = 5` additionally drops one node per axis (`6⁵ = 7776 → 5⁵ = 3125`, 2.5×),
because the step reduction alone leaves it hours long. That is a genuine
weakening of the `D = 5` spatial datum and is stated as such rather than buried:
the gate still runs the real kernel on a real anisotropic `A`, but on a coarser
grid than the `N(D)` ladder nominally specifies.

The underlying cause is worth its own change: `GridFnND::sample` currently costs
~16–21 ns per node read, because `collapse` recurses per axis through a closure
and resolves every node through `bc_value_by`. For the index-mapping boundary
policies (`Reflect`, `Periodic`, `ZeroExtend`) the per-axis stencils could be
folded into flat offsets once and summed in a strided loop, which should be
several times faster and would let `D = 5` keep its `N_AXIS = 6` datum. Not done
here — it is a hot-path rewrite, and this ADR is already a correctness change.

## AMENDMENT 4 (2026-08-14) — hoisting the boundary resolution out of the N-D sampler

AMENDMENT 3 lowered the `D = 5` gate's grid from `N_AXIS = 6` to `5` to buy back
the `K^D` sampling cost this ADR introduced. That was wrong, and the gate said
so: it failed at slope `−0.3595` against a `−0.45` threshold.

### Why the coarser grid failed

A `D = 2` probe — 1.3 s, against the 3 h the `D = 5` run costs — separates the
two candidate explanations. Successive-difference slope by ladder position:

| grid | {4,8,16,32} | {8,16,32,64} | {16,32,64,128} | {32,…,256} | {32,…,512} |
|---|---|---|---|---|---|
| `N_AXIS = 8` | −0.4532 | −0.4459 | −0.4531 | −0.4631 | −0.4676 |
| `N_AXIS = 5` | −0.3337 | −0.3749 | −0.4086 | −0.4341 | −0.4431 |

On an adequate grid the slope is stable at every ladder position. On the coarse
one it reads shallow *everywhere* and only creeps toward the right answer as the
ladder rises. So the shallow reading is the **grid** contaminating the
differences, not pre-asymptotic ladder placement — the spatial datum is exactly
what a successive-difference estimator cannot afford to trade away. `D = 5` is
restored to `N_AXIS = 6`.

### Where the time actually went

`GridFnND::sample` resolves each axis's boundary policy *inside* the collapse
recursion, and the recursion re-visits axis `d` once per combination of the axes
above it. At `D = 5, K = 4` that is `4 + 16 + 64 + 256 + 1024 = 1364` `bc_index`
calls per sample, of which only `5 × 4 = 20` are distinct.

`bc_value_by` is now split into `bc_index` + `bc_value_from_hit`, and
`AxisStencil` carries the resolved `BoundaryHit`s and the axis stride, computed
once per sample. The arithmetic and the summation order are untouched, so the
result is bit-identical — confirmed: `G_DDIM D = 2` and `D = 3` return exactly
`−0.4676` and `−0.4766`, and `D = 4`'s three successive differences
(`2.0218e−3 / 1.4595e−3 / 1.0512e−3`) match digit for digit.

### The honest part: it bought 1.32×, not 5×

`D = 4` went 1081 s → 820 s. The mechanism was correctly identified but its
*weight* was over-estimated: the dominant cost is the recursion itself — one
closure call per stencil node per level, plus `K^D` leaf reads — not the policy
resolution that was removed.

`D = 5` at the restored `N_AXIS = 6` therefore costs ≈ 4.3 h (extrapolated from
the measured `D = 4`: ×1.9 grid points, ×5 quadrature nodes, ×4 reads per
sample, ×0.5 steps). That fits a 6 h runner, without much margin.

Going faster means expanding the tensor stencil into `K^D` flat
(offset, weight) pairs and summing them in one loop, which removes the recursion
entirely. It also **changes the summation order**, so every N-D gate's numbers
move at ULP level and all of them need re-verification. That is a deliberate
non-goal here: this ADR is a correctness change, and re-baselining the whole N-D
gate set on top of it would make the two indistinguishable if something went
wrong.

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

---

## AMENDMENT 5 (2026-08-15) — `π^{-D/2}` was a non-portable operation in the N-D normalisation

> **CORRECTION (2026-08-15, same day, before release).** As originally written
> this amendment claimed that the `powf` substitution *fixed*
> `G_BINDING_SMOLYAK_PARITY_SUB2_PYO3_0ULP`, and that NumPy had been ruled out.
> Both claims are withdrawn — see AMENDMENT 6, which has the evidence. The
> substitution below is still correct and still worth keeping (a global
> normalisation must not go through `pow`), but it is **not** the cause of that
> gate's failure and did not fix it. What follows is the portability argument
> only; read the causal claim as retracted.

CI's `py-smoke` matrix cell `(ubuntu-latest, 3.13)` failed
`G_BINDING_SMOLYAK_PARITY_SUB2_PYO3_0ULP` with `max ULP diff = 2`; the other
five cells passed. That failure is what prompted the audit below. (The failure
itself is diagnosed in AMENDMENT 6; the audit still found a real defect.)

Walking the arithmetic, one operation in the path is not a correctly
rounded IEEE-754 primitive:

```rust
from_f64::<F>(core::f64::consts::PI).powf(from_f64::<F>(-(D as f64) / 2.0))
```

`powf` lowers to the system `pow`, which IEEE-754 does not require to be
correctly rounded and which glibc dispatches by IFUNC to CPU-dependent
implementations. Everything else is exact-by-construction: the Gauss–Hermite
nodes and weights are literal tables, the parity datum has `c ≡ 0` so
`exp(0) = 1` exactly, and the interpolation weights are `+`, `−`, `×` only.

The severity comes from *where* it sits. `π^{-D/2}` is a global normalisation
multiplying every output value, so a 1-ULP difference in it perturbs the entire
vector.

That last property is also what should have falsified the causal claim
immediately, and did not: the observed failure had `first-8 ULP diff = 0` and
`last-4 ULP diff = 2`. A global multiplier cannot move the tail while leaving
the head bit-identical. The signature was in the failure log from the first
occurrence and contradicted the diagnosis being written about it. See
AMENDMENT 6.

**Decision.** Compute it from correctly rounded operations only, in
`float::inv_pi_pow_half`: `⌊D/2⌋` multiplications by `π`, one `sqrt(π)` when `D`
is odd, one reciprocal. Multiplication, division and `sqrt` are correctly
rounded by IEEE-754 §5.4.1 and §5.4.2 on every conforming platform, so the
result is bit-identical everywhere by specification rather than by luck.

Measured against the old expression on this host (glibc x86-64), with the
correctly rounded reference computed at 60 decimal digits:

| `D` | old `powf` vs exact | new chain vs exact | old vs new |
|-----|---------------------|--------------------|------------|
| 2 | 0 ULP | 0 ULP | identical |
| 3 | 0 ULP | +1 ULP | 1 ULP |
| 4 | 0 ULP | 0 ULP | identical |
| 5 | 0 ULP | +1 ULP | 1 ULP |
| 6 | +1 ULP | +1 ULP | identical |

**This substitution is not an accuracy improvement, and must not be described as
one.** On this host `pow` happened to return the correctly rounded result for
`D = 3, 5` and the multiplication chain is 1 ULP above it; at `D = 6` both are
1 ULP above it. What changes is *which* value you get: the chain returns the
same bits on every conforming platform because each of its operations is
correctly rounded by IEEE-754 §5.4.1/§5.4.2, whereas `pow` returns whatever the
local libm returns — correctly rounded here, 1 ULP off on the failing runner.
Determinism is the deliverable; at ±1 ULP the accuracy is a wash. Any future
claim that one is "more accurate" than the other needs a per-`D`, per-platform
measurement of exactly this kind, not an appeal to operation count.

The parity golden is a `D = 6` vector, where old and new agree, so it is
unchanged and still valid. The substitution is applied at all three sites that
carried the expression:

| Site | Type | How it was found |
|------|------|------------------|
| `smolyak.rs` (apply + weight-sum check) | `SmolyakGridND` | the failing gate |
| `shift_nd.rs` | `AnisotropicShiftChernoffND` (behind every `G_DDIM`) | inspection of the same formula |
| `shift_nd_adaptive.rs` | `AnisotropicShiftAdaptiveQ` | grep for `PI.powf`, after the first two were fixed |

The third is worth naming explicitly: **no gate would have caught it.** The
adaptive N-D kernel has no parity gate, so it could have kept the non-portable
prefactor indefinitely while `G_BINDING_SMOLYAK_PARITY` stayed green. When a
defect class is found by a gate, the gate proves the instance, not the class —
the class has to be swept by hand.

**Honest limits.** This makes the *normalisation* portable, not the whole
library. Any kernel whose formula genuinely needs `exp`, `pow`, `sin` or `cos`
of a data-dependent argument remains subject to the platform's libm, and the
0-ULP contract (ADR-0018) is claimed only for the specific kernels that carry a
parity gate. What this amendment establishes is that a gate promising bit
equality must not have a transcendental in its scaling path.

Two deliberate non-changes, recorded so a later sweep does not re-litigate them:
`controller.rs`'s `powf` is a documented ≤2-ULP deviation with a trajectory-level
proof that it changes no accept/reject decision (ADR-0044,
`adaptive_classical_bit_equal.rs`), and `dual_helpers.rs`'s `powf` must track
the user's own `powf` to be a correct derivative. Neither is a scaling prefactor.

## AMENDMENT 6 (2026-08-15) — `G_BINDING_SMOLYAK_PARITY` sub-test 2 is non-deterministic by construction

AMENDMENT 5 attributed this gate's failure to `powf` and treated one green CI
run as confirmation. That was wrong on both counts. The gate failed again, with
an identical signature, on the release candidate.

### The evidence that settles it

| # | Fact | How established |
|---|------|-----------------|
| 1 | Failure signature is `first-8 = 0 ULP`, `last-4 = 2 ULP` — before *and* after the `powf` change | CI job logs on `488a397` and on `8403b49` |
| 2 | A global normalisation cannot produce that signature | it multiplies every entry, including the first 8 |
| 3 | The only Smolyak change between the green run and the red one is the removal of an `#[allow(...)]` attribute | `git diff f094d52..8403b49 -- crates/semiflow/src/smolyak.rs` |
| 4 | Both the green and the red `(ubuntu, 3.13)` job installed the *same* NumPy 2.5.2 manylinux x86-64 wheel | pip output in both job logs |
| 5 | Rust's `simd` feature is not the variable: output is bit-identical with and without it | built both ways locally, dumped raw bits |
| 6 | The gate's input comes from `np.exp(...)`; the golden's input came from Rust `f64::exp` (`gaussian()` in `binding_smolyak_parity.rs`) | source of both files |

Facts 3 and 4 together mean the repository content that produces this vector did
not change between a pass and a fail. The variable is the runner.

### Why the gate cannot be trusted as written

Fact 6 is the defect. The golden vector was captured from a Rust run whose
initial condition was computed with the platform's scalar `exp`. The Python
sub-test recomputes the initial condition with `np.exp`, a *different*
implementation — vectorised, and dispatched at run time by CPU feature. The gate
then asserts the two agree **bit-for-bit** across the whole pipeline.

Two independent `exp` implementations agreeing to the last bit on all 4096 grid
points is not something either library promises. When they agree the gate
passes; when the runner's CPU selects a NumPy kernel that differs by 1 ULP on a
few corner points, the perturbation propagates and the gate fails. The observed
history — fail, fail, pass, fail, with no relevant code change — is what a
coin-flip looks like, not what a regression looks like.

The amplification is measured, not assumed: perturbing every entry of the input
by exactly 1 ULP moves the output by **up to 25 ULP**. A 1-ULP disagreement on a
handful of points is therefore more than sufficient to produce the observed
2-ULP drift, and the kernel does not need to be doing anything wrong for it to
happen — 533 Smolyak weights, some negative, summed per output point is simply a
conditioning that turns last-bit input noise into several bits of output noise.
That number also establishes the gate is not vacuous: it does detect a 1-ULP
change in what crosses the boundary, which is exactly the marshalling defect it
exists to catch.

Note the shape of the original mistake, since it is the more useful lesson: the
"NumPy ruled out" measurement in AMENDMENT 5 was taken on a machine where the
gate *passes*. A hypothesis about why a failure occurs cannot be tested on a
configuration that does not exhibit the failure. That inference was invalid
independently of the conclusion, and it is what let a contradicted diagnosis
(fact 2) survive.

### Resolution

**RESOLVED.** Approved by the maintainer (`Gate-Change-Approved-By: ilia-volkov`)
after the diagnosis above was presented; not self-approved by the agent that
tripped over it.

The second `exp` implementation is removed from the comparison. The Python
sub-test no longer calls `np.exp`: it computes the *exponent* — which is exact,
and which was verified bit-identical to the Rust path, `linspace` reproducing
`Grid1D`'s `x_i = xmin + i·dx` exactly — and looks the exponential up in a
pinned table.

The table costs 16 constants, not 4096, because only 16 distinct exponents occur
over the grid. They are the exact bit patterns the golden was computed from,
extracted from the Rust IC itself. The reconstruction is pinned end-to-end by an
FNV-1a/64 checksum printed on the Rust side, so a wrong ravel order or a
truncated table fails loudly rather than silently producing a different vector.

Why this is **stronger**, not weaker — the distinction matters, since relaxing a
gate to make it green is the failure mode this project guards against:

- Before: `exp_A(input) → kernel → compare bits with golden(exp_B(input))`. Two
  `exp` implementations inside a bit-exactness assertion. Green meant "the
  marshalling is transparent **and** two libms happened to agree", and the gate
  could not tell those apart. It produced accidental **passes** as readily as
  accidental failures.
- After: `pinned_input → kernel → compare bits with golden(same pinned_input)`.
  With `c ≡ 0` the compared path is `+ − × ÷` and `sqrt` over literal
  Gauss–Hermite tables, all correctly rounded by IEEE-754 §5.4, so bit equality
  is now required by specification on every conforming platform. Green means
  exactly one thing, and red is now always a real defect.

Two guards keep the pinned data honest: `test_ic_reconstruction_matches_rust_checksum`
(bit-exact, against the Rust checksum) and `test_ic_table_is_the_documented_gaussian`
(tolerance ≤4 ULP against `exp(-Σx²)`, so the constants cannot silently drift
away from the canonical §1.3 parameters). The second is deliberately *not* a bit
check — demanding bit equality with `np.exp` there would reintroduce the exact
flake this amendment removes, one test to the left.

The golden *output* constants are untouched, and no threshold, tolerance or
skip was added to the parity assertion itself.
