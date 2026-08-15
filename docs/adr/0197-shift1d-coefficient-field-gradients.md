# ADR-0197 — Gradients w.r.t. `Shift1D` coefficient fields

- **Status**: Proposed (Issue #25; branch `fix/issue-campaign-17-26`)
- **Date**: 2026-08-13
- **Supersedes / amends**: none — purely ADDITIVE.
  `EvolverHeat1DGreeksV3`, `ReverseChernoff` and `AdjointChernoff` are untouched.
- **Contract**: `contracts/semiflow-core.math.md` §61; gates
  `G_SHIFT1D_WEIGHTS_ORACLE`, `G_SHIFT1D_TRANSPOSE_ID`, `G_SHIFT1D_COEFF_FD`.

## Context

`EvolverHeat1DGreeksV3` is the only turnkey-AD Greek path in the Python surface,
and it differentiates w.r.t. a **single global** diffusion scale θ of the
unit-heat kernel — one model (Bachelier, via `θ = σ_N²/2`). What calibration
needs is `∂J/∂a_i` for the per-node coefficient arrays of `Shift1D.with_arrays`:
local-vol Vega surfaces, `∂V/∂σ(S_i)`.

## Decision

**1. `edge_weight_grad`'s contract, not `ReverseChernoff`'s.** The caller
supplies the cotangent `∂J/∂u_n`; the loss stays outside the library; the
gradient buffer is caller-owned, zeroed and length-checked by the core. This is
the ADR-0115 boundary — no autograd hook, no framework types in core.
`ReverseChernoff::value_and_grad` bakes in `½‖u − target‖²` and is
`RegionMap`-parameterised over `DiffusionChernoff`; neither the loss nor the
parameterisation fits.

**2. A new driver, because both existing ones are hard-bound.**
`adjoint_state_gradient` is wired to `MagnusGraphHeatChernoff` + `GraphSignal`
and `ReverseChernoff` to `DiffusionChernoff`. The *pattern* is reused —
store the forward trajectory, sweep the adjoint backward, accumulate
`∂J/∂θ += ⟨λ_{k+1}, (∂S_k/∂θ)u_k⟩` — the code is not.

**3. `GeneratorSensitivity` is deliberately NOT implemented.** Its
`apply_param_deriv(k, …)` computes one parameter at a time, which
`adjoint_state_gradient` turns into `O(n_steps × n_params)`. Here `n_params = n`,
so that is `O(n_steps · n²)` — `10⁹` stencil evaluations at `n = 1024`,
`n_steps = 1000`. Because the parameter→output coupling is **diagonal**
(coefficients are read only at `x = grid.x_at(i)`), all `n` derivatives come from
one `O(n)` pass and the driver is `O(n_steps · n)`. Implementing the trait would
advertise a path `n×` slower than the one shipped.

**4. Weight rows are measured, not hand-transposed — the load-bearing choice.**
The adjoint is a scatter of interpolation weights. Deriving those rows by hand
for `SepticHermite` means composing eight Birkhoff–Garabedian–Lorentz
polynomials with three central-FD stencils and folding the result through seven
boundary policies; a sign slip in any one produces plausible-but-wrong gradients,
which is the worst failure mode this feature could have. Since the interpolant is
linear in the node values, `w_j(y) = sample(e_j, y)`: probing the real sampler
over its compact support is `O(1)` per foot and correct by construction for every
`InterpKind` and `BoundaryPolicy`, including any added later.

This is not a claim, it is gated. `G_SHIFT1D_WEIGHTS_ORACLE` compares
`Σ_j w_j·u_j` against the sampler across `{CubicHermite, SepticHermite,
OctonicHermite} × {Reflect, ZeroExtend, Periodic, LinearExtrapolate}`, sweeping
well outside the domain. **It caught a real bug on the first run**: the probe was
centred on the raw cell index, but `Reflect`/`Periodic` fold a far-outside query
onto interior nodes arbitrarily distant from it, so those weights were silently
missed (sampler 2.297 vs weights 0.840). The fix resolves each raw stencil index
through the same `bc_index` the sampler uses. Without this gate the error would
have surfaced as a small, plausible gradient bias near the boundary.

**5. Restricting to `CubicHermite` was rejected.** `Grid1D::new` defaults to
`SepticHermite`, so a cubic-only VJP would differentiate a *different model* than
the one being solved — an `O(dx⁴)` gradient of an `O(dx⁸)` forward. The probing
design makes the restriction unnecessary.

**6. `∂/∂a` requires `a_i > 0` strictly.** `∂h_i/∂a_i = √(τ/a_i)` diverges as
`a_i → 0⁺`. The forward kernel admits `a_i ≥ 0`, so the gradient's domain is
**strictly smaller** than the forward one, and the code says so with a
`DomainViolation` rather than returning an infinity.

## The 23 s hyper-dual complaint — what this does and does not address

The issue also reports `EvolverHeat1DGreeksV3` taking 23 s where two
bump-and-revalue solves take under 2 s. Stated plainly:

**Addressed.** Gradients w.r.t. coefficient *fields*, where `n_params = n`.
Reverse mode costs ≈ 2 forward solves for *all* `n` parameters; hyper-dual and
bump-and-revalue both cost `Θ(n_params)` solves. At `n = 1024` that is a
structural win of ~3 orders, and it is the case `Shift1D.with_arrays` users
actually have.

**Not addressed, and not a goal.** The 23 s itself. That measurement is
*second*-order Greeks w.r.t. a handful of *scalar* parameters — the case
hyper-duals exist for. At `K ≈ 2` bump-and-revalue is asymptotically optimal and
no reverse-mode design beats it; the 23 s is a constant-factor problem
(`Dual<Dual<f64>>` arithmetic defeats the SIMD interpolation path), which belongs
to ADR-0133, not here. Per the README's standing position, wallclock parity is
not chased. The guidance that follows — bump-and-revalue for `K ≲ 10` scalars,
this VJP for per-node fields, hyper-duals only for genuine second derivatives —
is documented rather than implied.

## Consequences

- New `crates/semiflow/src/shift1d_vjp.rs` and
  `crates/semiflow-py/src/shift1d_vjp_py.rs`.
- PyO3: free function `shift1d_coeff_grad(..., wrt="a"|"b"|"c")`, mirroring
  `edge_weight_grad` being a free function. PyO3-only, inheriting the ADR-0115
  asymmetry.
- The forward step is re-implemented in this module rather than calling
  `ShiftChernoff1D::apply_into`, so the gradient is provably of the same node
  formula it differentiates. `G_SHIFT1D_TRANSPOSE_ID` ties the two together.

## Honest limits

- **`∂/∂a` is undefined at `a_i = 0`** (decision 6). The gradient domain is
  strictly smaller than the forward domain.
- **Gradients are of the discrete kernel**, including its interpolation and
  boundary folding — not of the continuous PDE solution operator. The two agree
  only to the discretisation order.
- **Order 1.** `ShiftChernoff1D` has consistency order 1, so this is a
  first-order-accurate gradient of a first-order-accurate solve.
- **Full-trajectory memory `O(n_steps · n)`** — 8.2 MB at `n = 1024`,
  `n_steps = 1000`. `CheckpointSchedule::sqrt_n` (§51.3) would reduce it to
  `O(√n_steps · n)` at ~2× forward cost; not wired up here.
- **One field per call.** All three costs three backward sweeps; a fused variant
  sharing the trajectory is possible and is not shipped.
- **f64 only**, and `ChebyshevSpectralWithBC` grids are unsupported (its
  virtual-node construction is transposable in principle; deferred).

## AMENDMENT 1 (2026-08-14) — a performance regression this ADR exposed, and its cause

**Status**: RESOLVED. The regression was real; its cause was **not** in this
module, and the fix leaves the affected path faster than it was before the
campaign.

### What was observed

Adding `shift1d_vjp` to the core crate slowed the **unrelated** 1-D pre-sampled
coefficient path by ~70%, measured by `test_path2_faster_than_path1`
(`crates/semiflow-py/tests/test_coeff.py`), which asserts that
`Heat1D.with_a_array` (pre-sampled, pure Rust) is ≥2× faster than
`Heat1D.with_a_function` (Python callback):

| tree | Path 1 | Path 2 | speedup |
|---|---|---|---|
| baseline `0e6d25b` | 123.8 ms | 50.7 ms | **2.4×** |
| through #21 (`249434d`) | 121.6 ms | 48.4 ms | **2.5×** |
| with #25 (`d2ded9e`) | 167.8 ms | 83.0 ms | **1.9×** ✗ |
| #25, `pub mod shift1d_vjp` removed | 122.0 ms | 48.8 ms | **2.5×** |
| **after the fix below** | **110.4 ms** | **34.2 ms** | **3.2×** ✓ |

Bisected commit-by-commit; removing the module — changing nothing else —
restored the speed, so it looked like the module's *presence*.

### What it actually was

It was not the module. Isolating it further, by rebuilding the module one
function at a time, put the trigger on the gradient driver — and then a control
run showed the *same* stub-driver tree had become slow after unrelated edits
elsewhere. That is the signature of a codegen threshold, not of a cause.

Disassembling the extension module (`objdump` on an unstripped
`release-ffi` build) located it precisely. Every symbol in the two builds is
byte-for-byte the same size except one: `diffusion::apply_at_node_f64`. Inside
it, the boundary-resolution primitives — `boundary::bc_value`, and through it
`bc_index` / `reflect_index` — flip between inlined and out-of-line depending on
how much other code happens to be in the crate.

The leverage is arithmetic. A `SepticHermite` sample resolves **eight** nodes
through `bc_value`, and `DiffusionChernoff::apply_at_node_f64` takes **eleven**
samples per grid node (six in the γ-A baseline, five in the ζ-A correction) —
about **88 `bc_value` calls per grid node**, ~2.2 M per `evolve(0.1, 100)` on a
256-node grid. At ~15 ns of call overhead that is ~33 ms, which is exactly the
regression that was measured.

### Interventions that did nothing

Recorded so the next person does not repeat them. Each was built and timed:

| intervention | result |
|---|---|
| f64-monomorphise the module onto `Grid1D::interp` | no change (kept: independent correctness fix) |
| `#[inline(always)]` on `Grid1D::interp` | no change |
| `#[inline(always)]` on `GridFn1D::sample` | no change |
| `#[inline(never)]` on the septic/octonic/Chebyshev samplers | no change |
| hot-arm-first dispatch + out-of-line `interp_rare` | no change |
| `#[inline(always)]` on `call_a` / `call_a_prime` / `call_a_double_prime` | no change |
| resolve the coefficient-`Storage` variant once per node | no change |
| `#[inline(never)]` on `gamma_a_baseline_f64` / `zeta_correction_f64` | no change |
| `codegen-units = 16` | 83 ms → 76 ms; ratio still 2.0× |
| `lto = "off"` | no change |

All were reverted.

### Fix

`#[inline(always)]` on `boundary::bc_value`, `boundary::bc_index` and
`boundary::reflect_index`. These are the leaf primitives of every off-grid
sample in the library; at ~88 invocations per grid node their call overhead is a
first-order term, and leaving the decision to a cost model that cannot see that
multiplier is what made an unrelated module able to move the number by 70%.

Results are unchanged bit-for-bit — inlining performs no floating-point
reassociation, and `test-fast` including the ADR-0018 parallel bit-equality
gates passes unchanged.

The path ends up **faster than before the campaign**: 50.7 ms → 34.2 ms on
Path 2 (1.5×) and 123.8 ms → 110.4 ms on Path 1, against the same
`0e6d25b` baseline re-measured on the same machine.

### Honest limit

This removes the specific 88×-multiplier fragility; it does not make the kernel's
codegen insensitive in general. `apply_at_node_f64` remains a large function
whose shape moves with crate-wide code volume. A perf gate stated as a *ratio
between two paths* also cannot distinguish "Path 2 got slower" from "Path 1 got
faster" — worth revisiting if this recurs.
