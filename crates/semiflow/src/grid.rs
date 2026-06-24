//! Uniform 1-D grid and sub-grid interpolation.
//!
//! `Grid1D` owns only geometry. Function values live in [`crate::GridFn1D`].
//! Boundary policies and interpolation kinds are pluggable adapters (Hurd-translator
//! pattern; see `.dev-docs/adapters/registry.md`).
//!
//! ## v2.6 module relocation (ADR-0068)
//!
//! `BoundaryPolicy`, `BoundaryHit`, `InterpKind`, `bc_index`, `bc_value`,
//! `bc_value_generic`, and `reflect_index` MOVED to `crates/semiflow-core/src/boundary.rs`.
//! Re-exported here so that existing imports (`use crate::grid::BoundaryPolicy`, etc.)
//! continue to compile unchanged.
//!
//! ## Generic-over-Float (ADR-0025, v0.9.0 Block D pilot)
//!
//! `Grid1D` is now generic: `Grid1D<F: SemiflowFloat = f64>`.  The `= f64`
//! default keeps all existing call-sites compiling unchanged.  The f64
//! interpolation path (including SIMD `catmull_rom`) is used when `F = f64`;
//! an all-scalar generic path is used for other `SemiflowFloat` types.

use alloc::vec::Vec;

use num_traits::{float::FloatCore, Float};

// Re-export pub(crate) helpers for internal consumers in this file.
pub(crate) use crate::boundary::{bc_value, bc_value_generic};
// Re-export public boundary types so `use crate::grid::BoundaryPolicy` (etc.) still compile.
pub use crate::boundary::{BoundaryPolicy, InterpKind, OobPolicy};
use crate::{error::SemiflowError, float::SemiflowFloat, grid_cubic::catmull_rom};

// ---------------------------------------------------------------------------
// Grid1D
// ---------------------------------------------------------------------------

/// Uniform 1-D grid `[xmin, xmax]` with `n` nodes.
///
/// Node positions: `x_i = xmin + i · dx`, `dx = (xmax − xmin) / (n − 1)`,
/// `i = 0..n`. Minimum size: `n >= 4` (Catmull-Rom stencil).
///
/// The grid owns the geometry and hosts both adapters (boundary policy and
/// interpolation kind). Function values reside in [`crate::GridFn1D`].
///
/// ## Generic-over-Float (ADR-0025)
///
/// `Grid1D<F: SemiflowFloat = f64>` — the `= f64` default keeps all existing
/// call-sites compiling unchanged.  The f64 path (including SIMD `catmull_rom`)
/// is used when `F = f64`; a fully-scalar generic path is used otherwise.
///
/// `SepticHermite`, `OctonicHermite`, and `ChebyshevSpectralWithBC` are all
/// supported for `F: SemiflowFloat` via generic scalar samplers
/// (§46.5.bis, v8.1+, ADR-0133/ADR-0139).
///
/// # Example
///
/// ```rust
/// use semiflow::{Grid1D, BoundaryPolicy, InterpKind};
/// let grid = Grid1D::new(-1.0, 1.0, 100).unwrap();
/// assert_eq!(grid.n, 100);
/// assert!((grid.dx() - 2.0 / 99.0).abs() < 1e-15);
/// assert_eq!(grid.boundary, BoundaryPolicy::Reflect);
///
/// // Opt-in to ZeroExtend and CubicHermite
/// let grid2 = grid
///     .with_boundary(BoundaryPolicy::ZeroExtend)
///     .with_interp(InterpKind::CubicHermite);
/// assert_eq!(grid2.boundary, BoundaryPolicy::ZeroExtend);
/// ```
#[derive(Debug, Clone, Copy)]
#[allow(clippy::module_name_repetitions)]
pub struct Grid1D<F: SemiflowFloat = f64> {
    /// Left endpoint. Invariant: `xmin < xmax`.
    pub xmin: F,
    /// Right endpoint. Invariant: `xmin < xmax`.
    pub xmax: F,
    /// Number of nodes. Invariant: `n >= 4`.
    pub n: usize,
    /// Out-of-grid lookup policy. Default: [`BoundaryPolicy::Reflect`].
    pub boundary: BoundaryPolicy<F>,
    /// Sub-grid interpolation strategy. Default: [`InterpKind::CubicHermite`].
    pub interp: InterpKind,
}

impl<F: SemiflowFloat> Grid1D<F> {
    /// Construct a `Grid1D<F>` for non-`f64` scalar types.
    ///
    /// For `F = f64`, use the backward-compatible `Grid1D::new(f64, f64, usize)`
    /// on the concrete `impl Grid1D<f64>` block — it preserves type inference
    /// at existing call-sites (e.g. `Grid1D::new(-10.0, 10.0, 1000)`).
    ///
    /// Defaults: `boundary = Reflect`, `interp = SepticHermite` (v8.0+, §46.5.bis).
    /// Matches the `Grid1D::new` default so that kernels at `F = Dual<f64>`
    /// compose with the same spatial interpolant as the `f64` production path.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `xmin` or `xmax` is non-finite.
    /// - [`SemiflowError::DomainViolation`] if `xmin >= xmax`.
    /// - [`SemiflowError::DomainViolation`] if `n < 4` (minimum Catmull-Rom
    ///   stencil).
    pub fn new_generic(xmin: F, xmax: F, n: usize) -> Result<Self, SemiflowError> {
        if !xmin.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "xmin must be finite",
                value: xmin.to_f64().unwrap_or(f64::NAN),
            });
        }
        if !xmax.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "xmax must be finite",
                value: xmax.to_f64().unwrap_or(f64::NAN),
            });
        }
        if xmin >= xmax {
            return Err(SemiflowError::DomainViolation {
                what: "xmin must be < xmax",
                value: xmin.to_f64().unwrap_or(f64::NAN),
            });
        }
        if n < 4 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "n must be >= 4 (cubic Hermite stencil)",
                value: n as f64,
            });
        }
        Ok(Self {
            xmin,
            xmax,
            n,
            boundary: BoundaryPolicy::Reflect,
            // v8.0 ADR-0133 §46.5.bis: default changed CubicHermite → SepticHermite
            // to match Grid1D::new (f64) so Dual<f64> composes with the default grid.
            // To restore pre-v8 behaviour: .with_interp(InterpKind::CubicHermite).
            interp: InterpKind::SepticHermite,
        })
    }

    /// Grid spacing `dx = (xmax − xmin) / (n − 1)`.
    #[must_use]
    pub fn dx(&self) -> F {
        let n_minus_one = F::from(self.n - 1).unwrap_or_else(F::one);
        (self.xmax - self.xmin) / n_minus_one
    }

    /// Coordinate of the `i`-th node: `xmin + i · dx`.
    ///
    /// For `i >= n` the result is outside the domain; callers should use
    /// `sample` for out-of-domain queries.
    #[must_use]
    pub fn x_at(&self, i: usize) -> F {
        let i_f = F::from(i).unwrap_or_else(F::zero);
        self.xmin + i_f * self.dx()
    }

    /// Node coordinates as a `Vec<F>` (length `n`).
    #[must_use]
    pub fn nodes(&self) -> Vec<F> {
        (0..self.n).map(|i| self.x_at(i)).collect()
    }

    /// Builder: replace the boundary policy.
    #[must_use]
    pub fn with_boundary(mut self, policy: BoundaryPolicy<F>) -> Self {
        self.boundary = policy;
        self
    }

    /// Builder: replace the interpolation kind.
    #[must_use]
    pub fn with_interp(mut self, kind: InterpKind) -> Self {
        self.interp = kind;
        self
    }

    /// Evaluate the function given by `values` at arbitrary `x`.
    ///
    /// For `F = f64`, prefer `interp_f64` which uses the SIMD `catmull_rom` and the
    /// SIMD `SepticHermite`. This generic version uses scalar-only paths.
    ///
    /// `SepticHermite`, `OctonicHermite`, and `ChebyshevSpectralWithBC` are all
    /// supported for `F: SemiflowFloat` via generic scalar samplers
    /// (§46.5.bis, v8.1+, ADR-0139).
    ///
    /// # Errors
    /// - [`SemiflowError::Unsupported`] for `InterpKind::Linear` without the
    ///   `linear-interp` feature.
    /// - [`SemiflowError::DomainViolation`] for `InterpKind::ChebyshevSpectralWithBC`
    ///   if `m` is not in the supported set {8,16,32,64,128,256,512}.
    pub fn interp_generic(&self, values: &[F], x: F) -> Result<F, SemiflowError> {
        match self.interp {
            InterpKind::CubicHermite => Ok(cubic_hermite_at_generic(values, self, x)),
            InterpKind::Linear => {
                #[cfg(feature = "linear-interp")]
                {
                    Ok(linear_at_generic(values, self, x))
                }
                #[cfg(not(feature = "linear-interp"))]
                {
                    Err(SemiflowError::Unsupported {
                        feature: "linear-interp (enable with --features linear-interp)",
                    })
                }
            }
            InterpKind::SepticHermite => Ok(
                crate::grid_chebyshev_septic::sample_septic_1d_generic(values, self, x),
            ),
            InterpKind::OctonicHermite => Ok(
                crate::grid_chebyshev_octonic::sample_octonic_1d_generic(values, self, x),
            ),
            InterpKind::ChebyshevSpectralWithBC { m, oob_policy } => {
                let effective_grid = match oob_policy {
                    OobPolicy::Inherit => *self,
                    OobPolicy::ForceReflect => self.with_boundary(BoundaryPolicy::Reflect),
                    OobPolicy::ForcePeriodic => self.with_boundary(BoundaryPolicy::Periodic),
                    OobPolicy::ForceZero => self.with_boundary(BoundaryPolicy::ZeroExtend),
                };
                crate::grid_chebyshev::sample_chebyshev_spectral_1d_generic(
                    values,
                    &effective_grid,
                    x,
                    m,
                )
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Concrete impl for Grid1D<f64> — backward-compatible API + SIMD paths
// ---------------------------------------------------------------------------

impl Grid1D<f64> {
    /// Construct a `Grid1D<f64>` (backward-compatible signature).
    ///
    /// This is the historic `Grid1D::new(xmin: f64, xmax: f64, n: usize)` API.
    /// It takes concrete `f64` arguments, preserving type inference at existing
    /// call-sites (e.g. `Grid1D::new(-10.0, 10.0, 1000)`).
    ///
    /// For non-`f64` types, use `Grid1D::<F>::new_generic(xmin, xmax, n)`.
    ///
    /// Defaults: `boundary = Reflect`, `interp = SepticHermite` (v6.0+, ADR-0109).
    ///
    /// # Migration (v5 → v6)
    ///
    /// The default interp changed from `CubicHermite` to `SepticHermite`.
    /// If you need the pre-v6 behaviour, call `.with_interp(InterpKind::CubicHermite)`.
    /// See `docs/migration/v5-to-v6.md` for details.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `xmin` or `xmax` is non-finite.
    /// - [`SemiflowError::DomainViolation`] if `xmin >= xmax`.
    /// - [`SemiflowError::DomainViolation`] if `n < 4`.
    pub fn new(xmin: f64, xmax: f64, n: usize) -> Result<Self, SemiflowError> {
        if !xmin.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "xmin must be finite",
                value: xmin,
            });
        }
        if !xmax.is_finite() {
            return Err(SemiflowError::DomainViolation {
                what: "xmax must be finite",
                value: xmax,
            });
        }
        if xmin >= xmax {
            return Err(SemiflowError::DomainViolation {
                what: "xmin must be < xmax",
                value: xmin,
            });
        }
        if n < 4 {
            #[allow(clippy::cast_precision_loss)]
            return Err(SemiflowError::DomainViolation {
                what: "n must be >= 4 (septic Hermite stencil)",
                value: n as f64,
            });
        }
        Ok(Self {
            xmin,
            xmax,
            n,
            boundary: BoundaryPolicy::Reflect,
            // v6.0 BREAKING: default interp changed from CubicHermite → SepticHermite (ADR-0109).
            // To restore pre-v6 behaviour: .with_interp(InterpKind::CubicHermite).
            interp: InterpKind::SepticHermite,
        })
    }

    /// Convenience constructor: `Grid1D<f64>` with `ChebyshevSpectralWithBC` interp.
    ///
    /// Defaults: `boundary = Reflect`, `oob_policy = OobPolicy::Inherit`.
    /// Equivalent to:
    /// ```rust,ignore
    /// # use semiflow::{Grid1D, InterpKind, OobPolicy};
    /// # let (xmin, xmax, n, m) = (-1.0_f64, 1.0, 16, 64);
    /// Grid1D::new(xmin, xmax, n).unwrap().with_interp(InterpKind::ChebyshevSpectralWithBC {
    ///     m,
    ///     oob_policy: OobPolicy::Inherit,
    /// });
    /// ```
    ///
    /// # Migration (v4 → v5)
    ///
    /// Replace `Grid1D::new(xmin, xmax, n)?` + `with_interp(ChebyshevSpectral { m })`
    /// with `Grid1D::cheb_m(xmin, xmax, n, m)?`.
    ///
    /// # Errors
    ///
    /// Propagates [`SemiflowError::DomainViolation`] from `Grid1D::new`.
    pub fn cheb_m(xmin: f64, xmax: f64, n: usize, m: usize) -> Result<Self, SemiflowError> {
        let grid = Self::new(xmin, xmax, n)?;
        Ok(grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
            m,
            oob_policy: OobPolicy::Inherit,
        }))
    }

    /// Evaluate at `x` using f64-specific SIMD `catmull_rom` and `SepticHermite`.
    ///
    /// This is the historic `Grid1D::interp` from v0.8.x — now renamed to
    /// `interp_f64` to distinguish it from the generic `interp_generic`.
    /// All existing code that called `self.grid.interp(...)` on a `Grid1D<f64>`
    /// should use this method.
    ///
    /// # Errors
    /// - [`SemiflowError::Unsupported`] for `InterpKind::Linear` without the
    ///   `linear-interp` feature.
    pub fn interp(&self, values: &[f64], x: f64) -> Result<f64, SemiflowError> {
        match self.interp {
            InterpKind::SepticHermite => Ok(crate::grid_chebyshev_septic::sample_septic_1d(
                values, self, x,
            )),
            InterpKind::OctonicHermite => Ok(crate::grid_chebyshev_octonic::sample_octonic_1d(
                values, self, x,
            )),
            InterpKind::CubicHermite => Ok(cubic_hermite_at(values, self, x)),
            InterpKind::Linear => {
                #[cfg(feature = "linear-interp")]
                {
                    Ok(linear_at(values, self, x))
                }
                #[cfg(not(feature = "linear-interp"))]
                {
                    Err(SemiflowError::Unsupported {
                        feature: "linear-interp (enable with --features linear-interp)",
                    })
                }
            }
            InterpKind::ChebyshevSpectralWithBC { m, oob_policy } => {
                // Build a grid with the effective BoundaryPolicy based on oob_policy.
                let effective_grid = match oob_policy {
                    OobPolicy::Inherit => *self,
                    OobPolicy::ForceReflect => self.with_boundary(BoundaryPolicy::Reflect),
                    OobPolicy::ForcePeriodic => self.with_boundary(BoundaryPolicy::Periodic),
                    OobPolicy::ForceZero => self.with_boundary(BoundaryPolicy::ZeroExtend),
                };
                crate::grid_chebyshev::sample_chebyshev_1d(values, &effective_grid, x, m)
            }
        }
    }
}

// Boundary helpers (bc_index, bc_value, bc_value_generic, reflect_index) moved to
// boundary.rs in v2.6 (ADR-0068). Re-exported above via `pub use crate::boundary::*`.

// ---------------------------------------------------------------------------
// Interpolation kernels — f64-specific (SIMD-capable)
// ---------------------------------------------------------------------------

/// 2-point linear interpolation with boundary conditions (f64).
#[allow(dead_code)] // only compiled with linear-interp feature
fn linear_at(values: &[f64], grid: &Grid1D<f64>, x: f64) -> f64 {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = FloatCore::floor(t_frac);
    // t_floor is a whole number; cast to i64 is safe for any grid x.
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor as i64;
    let step = t_frac - t_floor;

    let p0 = bc_value(grid.boundary, values, grid.n, idx, dx);
    let p1 = bc_value(grid.boundary, values, grid.n, idx + 1, dx);
    (1.0 - step) * p0 + step * p1
}

/// Catmull-Rom 4-point cubic Hermite interpolation with boundary conditions
/// (f64, SIMD-capable via `catmull_rom` dispatcher).
///
/// Given parameter `s ∈ [0, 1)` within the `[x_i, x_{i+1}]` interval, uses
/// control points `p_{-1}, p_0, p_1, p_2`:
///
/// ```text
/// result = 0.5 * (
///   2·p_0
///   + (−p_{-1} + p_1) · s
///   + (2·p_{-1} − 5·p_0 + 4·p_1 − p_2) · s²
///   + (−p_{-1} + 3·p_0 − 3·p_1 + p_2) · s³
/// )
/// ```
#[allow(clippy::many_single_char_names)]
fn cubic_hermite_at(values: &[f64], grid: &Grid1D<f64>, x: f64) -> f64 {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = FloatCore::floor(t_frac);
    // t_floor is a whole number; cast to i64 is safe for any grid x.
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor as i64;
    let s = t_frac - t_floor;

    // Fetch the four control points, applying BC for out-of-range indices.
    // bc_value is total (no Result) in v0.2.1 — all 4 policies fully handled.
    let bnd = grid.boundary;
    let n = grid.n;
    let pm1 = bc_value(bnd, values, n, idx - 1, dx);
    let p0 = bc_value(bnd, values, n, idx, dx);
    let p1 = bc_value(bnd, values, n, idx + 1, dx);
    let p2 = bc_value(bnd, values, n, idx + 2, dx);

    catmull_rom(pm1, p0, p1, p2, s)
}

// ---------------------------------------------------------------------------
// Interpolation kernels — generic (scalar, any SemiflowFloat)
// ---------------------------------------------------------------------------

/// Catmull-Rom scalar kernel for `SemiflowFloat`.
///
/// `result = 0.5 * (2·p0 + (−pm1+p1)·s + (2·pm1−5·p0+4·p1−p2)·s²
///                        + (−pm1+3·p0−3·p1+p2)·s³)`.
///
/// This is the scalar-only version of `catmull_rom` in `grid_cubic`; SIMD
/// is not used here so that `Grid1D<f32>` (and any future `SemiflowFloat`
/// types) have a correct, clean scalar path.
#[allow(clippy::many_single_char_names)]
#[inline]
fn catmull_rom_scalar_generic<F: SemiflowFloat>(pm1: F, p0: F, p1: F, p2: F, s: F) -> F {
    let two = crate::float::two::<F>();
    let three = F::from(3.0_f64).unwrap_or_else(F::zero);
    let four = F::from(4.0_f64).unwrap_or_else(F::zero);
    let five = F::from(5.0_f64).unwrap_or_else(F::zero);
    let half = crate::float::half::<F>();
    let s2 = s * s;
    let s3 = s2 * s;
    half * ((two * p0)
        + (-pm1 + p1) * s
        + (two * pm1 - five * p0 + four * p1 - p2) * s2
        + (-pm1 + three * p0 - three * p1 + p2) * s3)
}

/// 2-point linear interpolation with boundary conditions (generic).
#[allow(dead_code)] // only compiled with linear-interp feature
fn linear_at_generic<F: SemiflowFloat>(values: &[F], grid: &Grid1D<F>, x: F) -> F {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = Float::floor(t_frac);
    // t_floor is a whole number; cast to i64 is safe for any grid x.
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor.to_i64().unwrap_or(0);
    let step = t_frac - t_floor;

    let one = crate::float::one::<F>();
    let p0 = bc_value_generic(grid.boundary, values, grid.n, idx, dx);
    let p1 = bc_value_generic(grid.boundary, values, grid.n, idx + 1, dx);
    (one - step) * p0 + step * p1
}

/// Catmull-Rom 4-point cubic Hermite interpolation (generic, scalar-only).
///
/// Uses `bc_value_generic` for boundary dispatch and `catmull_rom_scalar_generic`
/// for the Catmull-Rom kernel.  For `F = f64`, callers should prefer the
/// `cubic_hermite_at` / `interp_f64` path which uses the SIMD `catmull_rom`.
#[allow(clippy::many_single_char_names)]
fn cubic_hermite_at_generic<F: SemiflowFloat>(values: &[F], grid: &Grid1D<F>, x: F) -> F {
    let dx = grid.dx();
    let t_frac = (x - grid.xmin) / dx;
    let t_floor = Float::floor(t_frac);
    // t_floor is a whole number; cast to i64 is safe for any grid x.
    #[allow(clippy::cast_possible_truncation)]
    let idx = t_floor.to_i64().unwrap_or(0);
    let s = t_frac - t_floor;

    let bnd = grid.boundary;
    let n = grid.n;
    let pm1 = bc_value_generic(bnd, values, n, idx - 1, dx);
    let p0 = bc_value_generic(bnd, values, n, idx, dx);
    let p1 = bc_value_generic(bnd, values, n, idx + 1, dx);
    let p2 = bc_value_generic(bnd, values, n, idx + 2, dx);

    catmull_rom_scalar_generic(pm1, p0, p1, p2, s)
}

// ---------------------------------------------------------------------------
// G8 — `bc_index` property tests moved to boundary.rs (v2.6, ADR-0068).
// Tests for the 4 original variants + 2 new variants live in boundary::tests.
// Proptest property tests extracted to sibling file per ≤500-line cap.
// ---------------------------------------------------------------------------
#[cfg(test)]
#[path = "grid_tests.rs"]
mod bc_index_props;
