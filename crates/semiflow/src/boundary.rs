//! Boundary policies, hit records, and index/value dispatch helpers.
//!
//! Extracted from `grid.rs` in v2.6 per ADR-0068 §"Consequences".
//! `grid.rs` re-exports all public types via `pub use crate::boundary::*`
//! to preserve v2.5.1 import paths unchanged.
//!
//! ## v2.6 additions (ADR-0068)
//!
//! - `BoundaryPolicy` becomes generic: `BoundaryPolicy<F: SemiflowFloat = f64>`.
//!   Two new variants: `Dirichlet { value: F }` (constant-extension) and
//!   `Neumann` (clamp-to-boundary / zero-flux). See math.md §3.5.bis.
//! - `BoundaryHit` gains `Dirichlet(F)` payload variant. `Eq` derive DROPPED
//!   (f64/f32 payloads are not `Eq`).
//! - `bc_index`, `bc_value`, `bc_value_generic` handle the two new variants.
//!
//! ## Stencil-BC vs operator-level Dirichlet
//!
//! `BoundaryPolicy::Dirichlet { value }` is a *stencil-level* ghost-node policy
//! (answers "what value does the interpolant read at an out-of-range index?").
//! For **operator-level** Dirichlet (absorbing-boundary semigroup via
//! Feynman–Kac killing), use `KillingChernoff<C, BoxRegion>` (math §21).
//! See math.md §3.5.bis for the discriminating policy-selection guide.

use crate::float::SemiflowFloat;

// ---------------------------------------------------------------------------
// BoundaryPolicy (v2.6: generic over F, two new variants)
// ---------------------------------------------------------------------------

/// Policy for [`crate::GridFn1D::sample`] when the query point `x` falls
/// outside `[xmin, xmax]`.
///
/// ## v2.5.1 variants (unchanged)
///
/// - `Reflect` — mirror `x` at boundaries (default). Required for G1/G2 oracles.
/// - `ZeroExtend` — return scalar zero for `x` outside `[xmin, xmax]`.
/// - `Periodic` — wrap `x` with period `(n-1)·dx`.
/// - `LinearExtrapolate` — affine continuation from boundary nodes.
///
/// ## v2.6 variants (ADR-0068, math.md §3.5.bis)
///
/// - `Dirichlet { value }` — constant-extension to `value` for out-of-range
///   indices. Ghost-node BC only; **not** an absorbing-boundary semigroup.
/// - `Neumann` — clamp-to-boundary-node (zero-flux, zero-derivative ghost
///   extension). Unconditionally C⁰.
///
/// The generic parameter `F` defaults to `f64`; all v2.5.1 call-sites that
/// elide the type parameter continue to compile unchanged.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum BoundaryPolicy<F: SemiflowFloat = f64> {
    /// Mirror `x` at the boundaries (default). Required for G1/G2 heat oracle.
    Reflect,
    /// Return scalar zero for `x` outside `[xmin, xmax]`.
    ZeroExtend,
    /// Wrap `x` with period `(n − 1)·dx` (standard periodic boundary).
    Periodic,
    /// Linear (affine) extrapolation from the two outermost boundary nodes.
    ///
    /// When the query index falls outside `[0, n)`, the value is extrapolated
    /// as a straight line through the two nearest in-range nodes.  Concretely,
    /// for an index `d` steps beyond the *left* boundary:
    ///
    /// ```text
    /// v ≈ v[0]  − d · (−3·v[0] + 4·v[1] − v[2]) / 2
    /// ```
    ///
    /// and symmetrically at the *right* boundary.  The formula is
    /// second-order accurate in the grid spacing.
    ///
    /// ## Recommended use: asymptotically-linear far-field payoffs
    ///
    /// `LinearExtrapolate` is the **intended far-field closure for
    /// asymptotically-linear payoffs** such as European calls in Black-Scholes
    /// or CEV models.  For a call priced on a finite grid `S ∈ [0, S_max]`:
    ///
    /// - In the deep in-the-money region the payoff asymptote is
    ///   `V(S, τ) ≈ S − K e^{−rτ}`, a linear function of `S`.
    /// - Therefore `V_SS → 0` as `S → ∞`, so linear extrapolation is
    ///   **exact to leading order** beyond `S_max`: the boundary introduces
    ///   no spurious curvature or numerical kink.
    /// - Validated: `Shift1D.with_arrays` on `S ∈ [0, 4K]`, n = 1025,
    ///   attains ATM rel. error ≈ 8.5e-5 with no visible boundary artifact.
    ///
    /// **Contrast with `ZeroExtend`** (`"zero"`): zero-extension is exact only
    /// when the solution genuinely vanishes at the boundary (e.g. barrier
    /// options or puts in log-price space far below the strike).  For calls
    /// it forces `V = 0` at `S_max`, which is *wrong* and introduces a
    /// boundary layer that degrades accuracy.
    ///
    /// ## Puts in log-price space
    ///
    /// When a put is priced in log-price space (`z = ln S`), the solution
    /// satisfies `u_z → 0` at both ends of a sufficiently wide domain.
    /// In that regime, `Reflect` (`BoundaryPolicy::Reflect`) is the natural
    /// zero-flux closure and is equally accurate; `LinearExtrapolate` also
    /// works (both policies converge to the interior solution as the domain
    /// widens, but reflect guarantees the correct conservation property).
    ///
    /// ## Kernel-level vs stencil-level
    ///
    /// This is a *stencil-level ghost-node* policy: it controls what value
    /// the interpolation stencil reads when it reaches outside `[0, n)`.
    /// For an *operator-level* absorbing Dirichlet condition (u = 0 at a
    /// barrier), use `KillingChernoff<C, BoxRegion>` (math §21) instead.
    LinearExtrapolate,
    /// Constant-extension to `value` for out-of-range indices (v2.6, ADR-0068).
    ///
    /// `bc_index` returns `Inside(j)` for `j ∈ [0, n)`, else
    /// `BoundaryHit::Dirichlet(value)`. Downstream `bc_value` returns `value`.
    /// **NOT for absorbing-boundary semigroups** — use `KillingChernoff<C, BoxRegion>`
    /// (math §21) for operator-level Dirichlet.
    Dirichlet {
        /// The ghost-node constant value.
        value: F,
    },
    /// Zero-flux clamp-to-boundary-node extension (v2.6, ADR-0068).
    ///
    /// `bc_index` returns `Inside(0)` for `j < 0`, `Inside(n-1)` for `j >= n`,
    /// `Inside(j)` in range. Unconditionally C⁰; implements reflecting / insulated
    /// boundary. Non-parameterized (unit variant).
    Neumann,
    /// Mixed Robin BC `α·u(x) + β·∂_n u(x) = 0` (v4.6, ADR-0098 / math §3.5.tris).
    ///
    /// Stencil-level: out-of-range clamps like `Neumann`; Robin character enforced
    /// at operator level by `RobinHeatChernoff`. α=0 → `Neumann`; β=0 → `KillingChernoff`.
    Robin {
        /// α ≥ 0 (coefficient on u at boundary).
        alpha: F,
        /// β > 0 (coefficient on ∂_n u at boundary).
        beta: F,
    },
    /// Odd (antisymmetric) image: out-of-range ghost = `−(mirrored interior value)`.
    ///
    /// Realises the Dirichlet odd extension (math §21.9 (21.9.1)); the minus-sign
    /// mirror of `Reflect`. `bc_index` folds the index identically to `Reflect`, but
    /// `bc_value`/`bc_value_generic` negate the reflected value. At any boundary
    /// node `x₀ ∈ ∂R` where `σ_R(x₀) = x₀`, the ghost equals `−value(x₀)` and the
    /// antisymmetric extension forces `u(x₀) = 0` automatically (Dirichlet BC).
    ///
    /// Consumed by `DirichletHeat2ndChernoff` (`killing_order2.rs`, §21.9, ADR-0176).
    /// **NOT for non-self-adjoint operators** — use `KillingChernoff` (§21.3) instead.
    OddReflect,
}

// ---------------------------------------------------------------------------
// BoundaryHit (v2.6: generic F, new Dirichlet(F) variant, Eq DROPPED)
// ---------------------------------------------------------------------------

/// Internal result of `bc_index(boundary, n, idx)`.
///
/// Encodes whether the requested index folds into the grid (`Inside`), is out
/// of range under `ZeroExtend` (`Zero`), out of range under `LinearExtrapolate`
/// (`OutsideLeft`/`OutsideRight`), under `Dirichlet { value }` (`Dirichlet(F)`),
/// or under `Robin` (skew image with exponential weight, v6.2.3 ADR-0098 Am.2).
///
/// This enum is `pub(crate)` — not part of the public API.
///
/// **v2.6**: `Eq` derive DROPPED (f64/f32 payloads are not `Eq`). If you need
/// equality in tests, match structurally.
#[derive(Debug, Clone, Copy, PartialEq)]
pub(crate) enum BoundaryHit<F: SemiflowFloat = f64> {
    /// Index folded/clamped to `[0, n)`. Universal for all policies in range;
    /// also used by `Neumann` for out-of-range clamp.
    Inside(usize),
    /// `ZeroExtend` out-of-range: downstream returns scalar zero.
    Zero,
    /// `LinearExtrapolate`, `idx < 0`. Carries `d = (-idx) as u32 >= 1`.
    OutsideLeft(u32),
    /// `LinearExtrapolate`, `idx >= n`. Carries `d = (idx-(n-1)) as u32 >= 1`.
    OutsideRight(u32),
    /// `Dirichlet { value }` out-of-range: downstream returns `value` (v2.6).
    Dirichlet(F),
    /// Robin skew image (v6.2.3, ADR-0098 Am.2 / math §3.5.tris).
    /// `bc_value` computes `exp(-2(α/β)·depth·dx) · values[reflected]`.
    RobinSkew {
        /// Mirror index in `[0, n)`.
        reflected: usize,
        /// Integer distance beyond boundary (d ≥ 1).
        depth: u32,
    },
    /// Odd-image mirror (ADR-0176, math §21.9): `bc_value` = `−values[reflected]`.
    /// `reflected` is the same fold index produced by `reflect_index` (identical
    /// to `Reflect`). Only the sign of the returned value differs.
    OddReflected {
        /// Mirror index in `[0, n)`.
        reflected: usize,
    },
}

// ---------------------------------------------------------------------------
// InterpKind (moved verbatim from grid.rs)
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// OobPolicy (ADR-0104 H3 fix — Chebyshev out-of-domain policy)
// ---------------------------------------------------------------------------

/// Out-of-domain sample policy for [`InterpKind::ChebyshevSpectralWithBC`].
///
/// Controls what `sample_chebyshev_1d` returns when `x ∉ [xmin, xmax]`.
/// The barycentric Lagrange formula diverges outside `[xmin, xmax]`
/// (Berrut-Trefethen 2004 §3.2); this enum selects the fallback strategy.
///
/// ## Relationship to `BoundaryPolicy`
///
/// `Inherit` delegates to the grid's `BoundaryPolicy` (recommended default).
/// The `Force*` variants let callers override the grid BC for Chebyshev
/// sampling specifically, without changing the underlying grid policy.
///
/// Introduced in v5.0 (ADR-0104 H3 fix). Applied only by
/// `InterpKind::ChebyshevSpectralWithBC`; has no effect on `CubicHermite`,
/// `QuinticHermite`, or Linear paths (those use `BoundaryPolicy` directly).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OobPolicy {
    /// Use the grid's `BoundaryPolicy` to handle out-of-domain `x` (recommended).
    Inherit,
    /// Force mirror-reflection into `[xmin, xmax]` regardless of grid BC.
    ForceReflect,
    /// Force periodic wrap into `[xmin, xmax]` regardless of grid BC.
    ForcePeriodic,
    /// Force return of `0.0` for `x ∉ [xmin, xmax]` regardless of grid BC.
    ForceZero,
}

// ---------------------------------------------------------------------------
// InterpKind
// ---------------------------------------------------------------------------

/// Sub-grid interpolation strategy for [`crate::GridFn1D::sample`].
///
/// Set via [`crate::Grid1D::with_interp`]. Default: [`InterpKind::SepticHermite`].
///
/// `SepticHermite` (degree-7, ADR-0109 / v6.0 default), `OctonicHermite` (degree-9,
/// ADR-0117 / v7.0 KEYSTONE), `CubicHermite` (Catmull-Rom, C¹, ADR-0005), `Linear`
/// (feature-gated), `ChebyshevSpectralWithBC` (v5.0 boundary-aware spectral, ADR-0104 H3 fix).
///
/// `QuinticHermite` was removed at v7.0 (ADR-0109 12-month removal clock fulfilled).
/// Migrate: use [`InterpKind::SepticHermite`] (the v6.0+ default) or
/// [`InterpKind::OctonicHermite`] for higher precision. See `docs/migration/v6-to-v7.md`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InterpKind {
    /// Septic-Hermite degree-7 interpolant (O(dx⁸) leading residue, ADR-0109). **Default** (v6.0+).
    ///
    /// Matches f, f', f'', f''' at both cell endpoints; floor ≈ 1.49e-12 at N=512.
    /// f64-only; non-f64 callers receive [`crate::error::SemiflowError::Unsupported`].
    SepticHermite,
    /// Octonic-Hermite degree-9, O(dx¹⁰), ADR-0117 (v7.0 ADDITIVE). Floor ≈ 9.1e-16 at N=512.
    /// f64-only; non-f64 callers receive [`crate::error::SemiflowError::Unsupported`].
    OctonicHermite,
    /// Catmull-Rom 4-point cubic Hermite (C¹, O(dx⁴) leading error).
    ///
    /// Pre-v6.0 default. Lightweight option when O(dx⁴) accuracy suffices.
    CubicHermite,
    /// Linear 2-point interpolation (O(dx²) leading error). Feature-gated.
    Linear,
    /// Chebyshev spectral collocation with boundary policy (v5.0, ADR-0104).
    ///
    /// Exponential accuracy for smooth `f ∈ C^∞`; effective spatial floor
    /// ≈ 1.49e-12 at N=512 (`SepticHermite` virtual-node dominated, ADR-0109).
    /// `m` ∈ {8, 16, 32, 64, 128, 256, 512}; default M=64 via `Grid1D::cheb_m()`.
    /// f64-only; non-f64 callers receive [`crate::error::SemiflowError::Unsupported`].
    ///
    /// `oob_policy` controls out-of-domain `x` handling (ADR-0104 H3 fix).
    /// Use `OobPolicy::Inherit` (default) to delegate to the grid's `BoundaryPolicy`.
    ChebyshevSpectralWithBC {
        /// Number of Chebyshev-Lobatto intervals; M+1 virtual nodes are used.
        m: usize,
        /// Out-of-domain sample policy (ADR-0104 H3 fix).
        oob_policy: OobPolicy,
    },
}

// Accessor-based value resolution lives in `boundary_value.rs` (500-line cap);
// re-exported here so `crate::boundary::bc_value_by` keeps resolving.
pub(crate) use crate::boundary_value::{bc_value_by, bc_value_from_hit};

// ---------------------------------------------------------------------------
// reflect_index (moved verbatim from grid.rs)
// ---------------------------------------------------------------------------

/// Reflect a signed index into `[0, n)`.
#[inline(always)]
#[allow(clippy::inline_always)] // see bc_value
pub(crate) fn reflect_index(n: usize, idx: i64) -> usize {
    #[allow(clippy::cast_possible_wrap)]
    let n_i = n as i64;
    let period = 2 * (n_i - 1);
    let mut r = idx.rem_euclid(period);
    if r >= n_i {
        r = period - r;
    }
    #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
    let result = r as usize;
    result
}

// ---------------------------------------------------------------------------
// bc_index (widened with Dirichlet + Neumann arms)
// ---------------------------------------------------------------------------

/// Map a possibly out-of-range signed index to a [`BoundaryHit`] via the
/// chosen boundary policy.
///
/// **Total** (never errors) for all six policies and all valid `n >= 2`.
/// In-range fast-path → `Inside(idx as usize)` (I5). v2.6 additions:
/// `Dirichlet` → `Dirichlet(value)` out-of-range; `Neumann`/`Robin` → clamp/skew.
#[inline(always)]
#[allow(clippy::inline_always)] // see bc_value
pub(crate) fn bc_index<F: SemiflowFloat>(
    boundary: BoundaryPolicy<F>,
    n: usize,
    idx: i64,
) -> BoundaryHit<F> {
    #[allow(clippy::cast_possible_wrap)]
    let n_i64 = n as i64;
    // Fast-path: in-range → Inside for all policies (invariant I5).
    if idx >= 0 && idx < n_i64 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        return BoundaryHit::Inside(idx as usize);
    }
    match boundary {
        BoundaryPolicy::Reflect => BoundaryHit::Inside(reflect_index(n, idx)),
        BoundaryPolicy::ZeroExtend => BoundaryHit::Zero,
        BoundaryPolicy::Periodic => {
            #[allow(clippy::cast_possible_wrap)]
            let r = idx.rem_euclid(n as i64);
            #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
            BoundaryHit::Inside(r as usize)
        }
        BoundaryPolicy::LinearExtrapolate => {
            if idx < 0 {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let d = (-idx) as u32;
                BoundaryHit::OutsideLeft(d)
            } else {
                #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
                let d = (idx - (n_i64 - 1)) as u32;
                BoundaryHit::OutsideRight(d)
            }
        }
        BoundaryPolicy::Dirichlet { value } => BoundaryHit::Dirichlet(value),
        BoundaryPolicy::Neumann => {
            if idx < 0 {
                BoundaryHit::Inside(0)
            } else {
                // idx >= n_i64
                BoundaryHit::Inside(n - 1)
            }
        }
        BoundaryPolicy::Robin { alpha: _, beta: _ } => robin_skew_hit(n, n_i64, idx),
        BoundaryPolicy::OddReflect => BoundaryHit::OddReflected {
            reflected: reflect_index(n, idx),
        },
    }
}

// Skew image for Robin BC: α=0 ⟹ weight 1 = even reflection = Neumann (see math §3.5.tris).
#[inline]
fn robin_skew_hit<F: crate::float::SemiflowFloat>(
    n: usize,
    n_i64: i64,
    idx: i64,
) -> BoundaryHit<F> {
    if idx < 0 {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let d = (-idx) as u32;
        BoundaryHit::RobinSkew {
            reflected: reflect_index(n, idx),
            depth: d,
        }
    } else {
        #[allow(clippy::cast_sign_loss, clippy::cast_possible_truncation)]
        let d = (idx - (n_i64 - 1)) as u32;
        BoundaryHit::RobinSkew {
            reflected: reflect_index(n, idx),
            depth: d,
        }
    }
}

// ---------------------------------------------------------------------------
// bc_value (f64-specific, widened)
// ---------------------------------------------------------------------------

/// Resolve `bc_index(idx)` and compute the corresponding value (f64-specific).
/// `Dirichlet(v)` → `v`; `RobinSkew` → `exp(-2(α/β)·depth·dx)·values[reflected]`
/// (v6.2.3, math §3.5.tris). `dx` used only for `RobinSkew`.
///
/// `#[inline(always)]`: a septic sample resolves eight nodes through this
/// function and `DiffusionChernoff` takes eleven samples per grid node — ~88
/// calls per node. Whether the cost model inlined it moved the 1-D kernel by
/// ~70% across unrelated edits (ADR-0197 AM 1).
#[inline(always)]
#[allow(clippy::inline_always)] // ~88 calls per grid node; see the note above
pub(crate) fn bc_value(
    boundary: BoundaryPolicy<f64>,
    values: &[f64],
    n: usize,
    idx: i64,
    dx: f64,
) -> f64 {
    match bc_index(boundary, n, idx) {
        BoundaryHit::Inside(i) => values[i],
        BoundaryHit::Zero => 0.0,
        BoundaryHit::Dirichlet(v) => v,
        BoundaryHit::OutsideLeft(d) => {
            let f0 = values[0];
            let f1 = values[1];
            let f2 = values[2];
            let slope_combo = -3.0 * f0 + 4.0 * f1 - f2;
            f0 - f64::from(d) * 0.5 * slope_combo
        }
        BoundaryHit::OutsideRight(d) => {
            let fnm1 = values[n - 1];
            let fnm2 = values[n - 2];
            let fnm3 = values[n - 3];
            let slope_combo = 3.0 * fnm1 - 4.0 * fnm2 + fnm3;
            fnm1 + f64::from(d) * 0.5 * slope_combo
        }
        BoundaryHit::RobinSkew { reflected, depth } => {
            let BoundaryPolicy::Robin { alpha, beta } = boundary else {
                // Unreachable: RobinSkew is only produced by Robin policy.
                return values[reflected]; // even-reflect fallback, no panic
            };
            libm::exp(-2.0 * (alpha / beta) * f64::from(depth) * dx) * values[reflected]
        }
        // Odd-image: negate the mirrored interior value (ADR-0176, math §21.9).
        BoundaryHit::OddReflected { reflected } => -values[reflected],
    }
}

// ---------------------------------------------------------------------------
// bc_value_generic (generic F, widened)
// ---------------------------------------------------------------------------

/// Generic version of `bc_value` over `SemiflowFloat` (math §3.5.tris / ADR-0098 Am.2).
/// Same Robin skew-image formula as `bc_value`; `dx` used only for `RobinSkew`.
pub(crate) fn bc_value_generic<F: SemiflowFloat>(
    boundary: BoundaryPolicy<F>,
    values: &[F],
    n: usize,
    idx: i64,
    dx: F,
) -> F {
    bc_value_by(boundary, |i| values[i], n, idx, dx)
}

// ---------------------------------------------------------------------------
// bc_index_dirichlet_neumann_totality property tests (math.md §3.5.bis.3)
// ---------------------------------------------------------------------------
#[cfg(test)]
mod tests {
    include!("boundary_tests.rs");
}
