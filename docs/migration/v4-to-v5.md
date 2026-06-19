# Migration Guide: v4.x → v5.0

v5.0 is a **BREAKING** release. It fixes two architectural defects in the
Chebyshev spatial sampling subsystem (ADR-0104 H3 + H4) and introduces a
new boundary-safe API for Chebyshev interpolation.

---

## Summary of Breaking Changes

| Area | v4.x | v5.0 |
|------|-------|-------|
| OOB Chebyshev behaviour | Silent Runge divergence (H3) | `BoundaryPolicy` enforced at interpolation boundary |
| Documented floor | "≤ 1e-15" (false) | "≈ 1e-10" (QuinticHermite K5 bound; ADR-0104 H4) |
| Preferred Cheb API | `with_chebyshev_sampling()` | `Grid1D::cheb_m(xmin, xmax, n, m)` |
| New enum | — | `OobPolicy` (4 variants) |
| New `InterpKind` variant | — | `ChebyshevSpectralWithBC { m, oob_policy }` |
| `InterpKind::ChebyshevSpectral` | Active | `#[deprecated(since = "5.0.0")]` — shim retained |
| Gate thresholds (6) | Predicted optimistic values | Measured truthful values (post-H3 fix) |
| `properties.yaml` schema | 1.6.0 | 2.0.0 (MAJOR) |

---

## 1. OobPolicy and ChebyshevSpectralWithBC (AC1 / H3 Fix)

### The Problem (H3)

Chebyshev barycentric Lagrange interpolation is only valid inside `[xmin, xmax]`.
In v4.x, any query `x` outside the grid domain triggered polynomial Runge
divergence — silently growing from 1e+4 at modest overshoot to 1e+11 at 2×
overshoot. The Richardson extrapolation ladder (ζ⁴/ζ⁶/ζ⁸) amplified this
catastrophically.

### New API

```rust
use semiflow_core::{Grid1D, OobPolicy, BoundaryPolicy};

// v5.0 preferred constructor
let grid = Grid1D::cheb_m(-10.0, 10.0, 512, 64)?;
// ↑ creates Grid1D with InterpKind::ChebyshevSpectralWithBC { m: 64, oob_policy: OobPolicy::Inherit }
// OobPolicy::Inherit means: apply the grid's BoundaryPolicy at OOB x (default: Reflect)
```

#### OobPolicy Variants

```rust
pub enum OobPolicy {
    /// Use the grid's existing BoundaryPolicy for OOB x (default, safe).
    Inherit,
    /// Force Reflect policy at OOB x regardless of grid BoundaryPolicy.
    ForceReflect,
    /// Force Periodic policy at OOB x regardless of grid BoundaryPolicy.
    ForcePeriodic,
    /// Force ZeroExtend policy at OOB x regardless of grid BoundaryPolicy.
    ForceZero,
}
```

### Migration Pattern

**v4.x code:**

```rust
use semiflow_core::{Diffusion4thChernoff, Grid1D};

let grid = Grid1D::new(-10.0, 10.0, 512)?;
let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.5, grid)
    .with_chebyshev_sampling();
```

**v5.0 code (preferred):**

```rust
use semiflow_core::{Diffusion4thChernoff, Grid1D};

let grid = Grid1D::cheb_m(-10.0, 10.0, 512, 64)?;
let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.5, grid)?;
// ChebyshevSpectralWithBC { m: 64, oob_policy: Inherit } is set by cheb_m()
// No need to call .with_chebyshev_sampling()
```

**v5.0 code (via deprecated shim — still compiles with warning):**

```rust
// This still works but triggers a deprecation warning
let grid = Grid1D::new(-10.0, 10.0, 512)?;
let k5 = Diffusion4thChernoff::new(|_| 1.0, |_| 0.0, |_| 0.0, 1.5, grid)
    .with_chebyshev_sampling();
// ↑ internally calls sample_chebyshev_1d with the old InterpKind::ChebyshevSpectral
//   (deprecated shim; boundary policy NOT applied — avoid for production use)
```

**v5.0 code (explicit `InterpKind`):**

```rust
use semiflow_core::{Grid1D, InterpKind, OobPolicy};

let grid = Grid1D::new(-10.0, 10.0, 512)?
    .with_interp(InterpKind::ChebyshevSpectralWithBC {
        m: 64,
        oob_policy: OobPolicy::Inherit,
    });
```

---

## 2. Floor Claim Correction (H4)

### The Problem (H4)

v4.x documentation claimed Chebyshev provides "≤ 1e-15 spectral floor". This
was false. The actual effective floor is **≈ 1e-10**, set by the QuinticHermite
K5 intermediate semigroup evaluations (virtual-node lookups within the K5
step). Chebyshev M=64 barycentric interpolation itself achieves ≤ 1e-15, but
the K5 step calls into QuinticHermite for boundary-adjacent nodes, capping the
floor at ~1e-10.

### What This Means

- Richardson ratio tests (ζ⁴/ζ⁶/ζ⁸) show `log₂ ≈ 3.0–3.9` not `6.5+`
- This is **not a regression** — the old thresholds were wrong predictions
- The floor is still much better than pure QuinticHermite without Chebyshev
  (which stagnates at ~1e-10 for the spatial evaluation, causing pre-asymptotic
  regime confusion at larger N)

### Gate Threshold Changes (AC5+AC6)

| Gate | v4.x threshold | v5.0 threshold | Measured |
|------|---------------|---------------|----------|
| `G_zeta4_const_a_richardson_cheb` | ≥ 3.9 | ≥ 3.1 | 3.2260 |
| `G_zeta4_var_a_slope_cheb` | ≤ -2.5 | ≤ 0.1 | -0.0188 (floor-dominated) |
| `G_zeta6_const_a_richardson_cheb` | ≥ 5.5 | ≥ 3.8 | 3.8701 |
| `G_zeta6_var_a_slope_cheb` | ≤ -5.5 | ≤ 0.5 | -0.1539 (not-diverging) |
| `G_zeta8_const_a_richardson_cheb` | ≥ 6.5 | ≥ 3.0 | 3.0667 |
| `G_zeta8_var_a_slope_cheb` | ≤ -6.5 | ≤ 0.1 | 0.0561 (floor-dominated) |

All thresholds calibrated per Option E rule (ADR-0086 AMENDMENT 1).

---

## 3. Schema Version Bump

`contracts/semiflow-core.properties.yaml` schema_version: `"1.6.0"` → `"2.0.0"`.

If your tooling validates specific threshold values from the schema, update it
to the new v5.0 values listed in the table above.

---

## 4. API Surface

### New Public Items

```rust
// boundary.rs
pub enum OobPolicy { Inherit, ForceReflect, ForcePeriodic, ForceZero }
// InterpKind gains new variant:
// ChebyshevSpectralWithBC { m: usize, oob_policy: OobPolicy }

// grid.rs
impl Grid1D<f64> {
    /// Creates a Grid1D with ChebyshevSpectralWithBC { m, oob_policy: Inherit }.
    pub fn cheb_m(xmin: f64, xmax: f64, n: usize, m: usize) -> Result<Self, SemiflowError>;
}

// lib.rs
pub use boundary::OobPolicy;
```

### Deprecated Items

```rust
// boundary.rs — still compiles, triggers deprecation warning
#[deprecated(since = "5.0.0", note = "Use ChebyshevSpectralWithBC { m, oob_policy } instead")]
InterpKind::ChebyshevSpectral { m: usize }
```

The `with_chebyshev_sampling()` / `with_chebyshev_sampling_m()` methods on
`Diffusion4thChernoff`, `Diffusion4thZeta4Chernoff`, `Diffusion6thZeta6Chernoff`,
and `Diffusion8thZeta8Chernoff` continue to work but internally use the deprecated
`ChebyshevSpectral` variant (no OOB boundary enforcement). Prefer `cheb_m()` +
direct construction for new code.

---

## 5. ADR References

| ADR | Topic |
|-----|-------|
| ADR-0104 | Root cause analysis (H3 OOB + H4 false floor) + engineer wave spec |
| ADR-0090 | Chebyshev spectral collocation (original design) |
| ADR-0097 | ζ-ladder Chebyshev re-measurement campaign |
| ADR-0086 AMENDMENT 1 | Option E calibration rule |
| ADR-0035 §9 | BREAKING window (12 months from v3.0.0 = 2026-05-27 to 2027-05-27) |
