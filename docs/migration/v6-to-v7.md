# Migration Guide: v6.x → v7.0

## QuinticHermite removed (ADR-0109 12-month removal clock)

`InterpKind::QuinticHermite` was deprecated at v6.0 (ADR-0109) with a 12-month removal
clock. v7.0 is a BREAKING window and removes it ahead of the original 2027-05-30 target
as a deliberate MAJOR-window call.

### What was removed

| Symbol | Removed | Replace with |
|--------|---------|--------------|
| `InterpKind::QuinticHermite` | Yes | `InterpKind::SepticHermite` (O(dx⁸), v6.0+ default) or `InterpKind::OctonicHermite` (O(dx¹⁰), v7.0) |
| `Diffusion4thChernoff::with_quintic_sampling()` | Yes | no-op — SepticHermite is already the default |
| `Diffusion4thChernoff::with_cubic_sampling()` | Yes | `Diffusion4thChernoff::without_chebyshev_sampling()` if you need CubicHermite explicitly |
| `Diffusion4thZeta4Chernoff::with_quintic_sampling()` | Yes | no-op — default path already uses CubicHermite (stable) |
| `Diffusion4thZeta4Chernoff::without_quintic_sampling()` | Yes | no-op |
| `Diffusion4thZeta4Chernoff::new_cubic()` | Yes | `Diffusion4thZeta4Chernoff::new()` (defaults to CubicHermite) |
| `Diffusion6thZeta6Chernoff::without_quintic_sampling()` | Yes | no-op |
| `Heat1DZeta4::with_quintic_sampling()` (Python) | Yes | no-op — default path is already stable |
| `Heat1DZeta4::with_cubic_sampling()` (Python) | Yes | no-op |
| `legacy-quintic` Cargo feature | Yes | remove from `Cargo.toml` / `--features` |
| `crates/semiflow-core/src/grid_quintic.rs` | Yes (file deleted) | — |

### Why the removal is safe

At v6.0, ADR-0109 promoted `SepticHermite` to the default for `Grid1D::new()`.
The `with_quintic_sampling()` flag (Path ε) was already routing to `SepticHermite`
internally since that point — it was a naming holdover from when the flag genuinely
used `QuinticHermite`. The removal eliminates dead naming, a deleted module, and
a feature flag that had no runtime effect.

### Migration examples

**Before (v6.x)**:
```rust
use semiflow_core::InterpKind;
let grid = Grid1D::new(-5.0, 5.0, 512)?
    .with_interp(InterpKind::QuinticHermite);  // compile error at v7.0
```

**After (v7.0)**:
```rust
// SepticHermite is the default — no call needed:
let grid = Grid1D::new(-5.0, 5.0, 512)?;

// Or be explicit:
let grid = Grid1D::new(-5.0, 5.0, 512)?
    .with_interp(InterpKind::SepticHermite);

// Or use OctonicHermite for higher precision (v7.0 new):
let grid = Grid1D::new(-5.0, 5.0, 512)?
    .with_interp(InterpKind::OctonicHermite);
```

**Before (v6.x, ζ⁴ opt-in)**:
```rust
let zeta4 = Diffusion4thZeta4Chernoff::new(inner, None)?
    .with_quintic_sampling();  // compile error at v7.0
```

**After (v7.0)**:
```rust
// Remove the builder call — the effect (SepticHermite default) is already active.
let zeta4 = Diffusion4thZeta4Chernoff::new(inner, None)?;
```

**Before (v6.x, Python)**:
```python
kern = rp.Heat1DZeta4(xmin, xmax, n, u0)
kern.with_quintic_sampling()  # AttributeError at v7.0
```

**After (v7.0)**:
```python
kern = rp.Heat1DZeta4(xmin, xmax, n, u0)  # no-op needed
```

**Cargo.toml feature**:
```toml
# BEFORE:
semiflow-core = { features = ["legacy-quintic"] }  # unknown feature at v7.0

# AFTER: remove the feature entry
semiflow-core = {}
```

## ChebyshevSpectral removed (reference — already done at v6.0)

`InterpKind::ChebyshevSpectral { m }` was removed at v6.0. Use
`InterpKind::ChebyshevSpectralWithBC { m, oob_policy }` instead.
See `docs/migration/v5-to-v6.md`.

## Behavior changes (non-breaking, but worth noting)

### Gate thresholds recalibrated (ADR-0120)

`g_zeta4_const_a_richardson_ratio` threshold changed from 3.5 to 3.1. The prior
threshold was calibrated against the QuinticHermite floor; under the Septic default
the re-measurement gives 3.226, and the gate passes at 3.1. If you maintain a fork
of the property contracts with a hardcoded 3.5 value, lower it to 3.1.

### `path_eps` gate control-arm precondition set explicitly (ADR-0120)

The `CubicHermite` control arm in the `path_eps` gate tests now calls
`.with_interp(InterpKind::CubicHermite)` explicitly. Previously it silently inherited
the SepticHermite default (a v6.0 regression), causing the control's
`baseline > 1e-7` precondition to fail. If you derive test baselines from the
`path_eps` gates, re-measure after updating.

### `G_KILLING_ORDER2` vs `KillingChernoff` — distinct operators (ADR-0126)

`Killing2ndChernoff` (new v7.0) and `KillingChernoff` (v2.6) are NOT faster/slower
versions of the same operator. `KillingChernoff` implements hard absorbing walls
(Dirichlet; order-1; irreducibly). `Killing2ndChernoff` implements soft continuous
killing via a rate field κ(x) ≥ 0 (order-2; a reaction term −κu). Do not substitute
one for the other.

### `DiffusionExpmvChernoff::order()` sentinel (ADR-0121)

`DiffusionExpmvChernoff::order()` returns `u32::MAX` (or an equivalent documented
sentinel) because the kernel is tolerance-driven, not fixed-order. Gate logic that
calls `order()` and compares numerically must handle this sentinel value.

### `MatrixDiffusionChernoff{2D,3D}` M≤4 / M≥5 dispatch (ADR-0124, ADR-0125)

For `MatrixDiffusionChernoff<F, M>` with M≤4 the per-grid-point reaction-block
exponential continues to use the existing Cayley-Hamilton closed-form paths (byte-identical
with prior releases). For M≥5 it now routes to Padé[13/13] instead of returning
`Err(Unsupported)`. The M≤4 numeric outputs are unchanged.

## New public types at v7.0.0 (all additive, no migration required)

| Type | Module | ADR |
|------|--------|-----|
| `InterpKind::OctonicHermite` | `boundary.rs` | 0117 |
| `DiffusionExpmvChernoff<F>` | `expmv.rs` | 0121 |
| `AnisotropicShiftChernoffND::with_adaptive_q(tol)` | `shift_nd.rs` | 0122 |
| `SmolyakGridND<F, const D>` | `smolyak.rs` | 0123 |
| `MatrixDiffusionChernoff2D<F, M>` | `matrix_2d3d.rs` | 0124 |
| `MatrixDiffusionChernoff3D<F, M>` | `matrix_2d3d.rs` | 0124 |
| `MatrixAxisLift` | `matrix_2d3d.rs` | 0124 |
| `MatrixExpPade<M>` (internal dispatch) | `matrix_pade.rs` | 0125 |
| `Killing2ndChernoff<C, K, F>` | `killing_soft.rs` | 0126 |
| `KillingRate<F>` trait | `killing_soft.rs` | 0126 |
| `LaplaceChernoffResolvent::eval_complex` | `resolvent_complex.rs` | 0127 |
| `MatrixDiffusionChernoffComplex<F, M>` | `matrix_system_complex.rs` | 0128 |
| `FubiniStudyCp1<F>` | `manifold_kahler.rs` | 0129 |
| `QuantumSchrödingerChernoff<C>` | `quantum_schrodinger.rs` | 0130 |
| `DriftReactionZeta4Chernoff<F>` | `drift_reaction_zeta4.rs` | 0131 |
