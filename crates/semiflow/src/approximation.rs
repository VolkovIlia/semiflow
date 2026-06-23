//! [`ApproximationSubspace<const K, F>`] — opt-in marker super-trait (v3.0 B1, ADR-0073).
//! [`LadderRung<const K, F>`] — sealed super-trait for ζ-ladder rungs (v5.0 A.6, ADR-0100).
//!
//! Codifies "this datum and this Chernoff function jointly support order-K convergence"
//! per Galkin-Remizov 2025 *Israel J. Math.* Theorem 3.1 (order-K tangency).
//! Mathematical foundation: math.md §26 (NORMATIVE) + §36 (NORMATIVE, v5.0).
//!
//! ## References
//!
//! - Galkin, Remizov (2025) *IJM* — *Tangency of Chernoff approximations to operator
//!   semigroups on Banach spaces* Theorem 3.1 + Example 4.2 (multi-K co-witness).
//! - Vedenin, Smolyanov, Voskresenskaya (2020) *Math. Notes* — original D(A^K) core
//!   characterisation.
//! - Remizov (2025) *Vladikavkaz Math. J.* 27:4 Theorem 6 — foundational formula.
//!
//! ## v3.0 opt-in impls shipped (per ADR-0073 §"Decision")
//!
//! - `ApproximationSubspace<2, F>` for [`crate::DiffusionChernoff<F>`]
//! - `ApproximationSubspace<4, F>` for [`crate::Diffusion4thChernoff<F>`]
//! - `ApproximationSubspace<6, F>` for [`crate::TruncatedExp4thDiffusionChernoff<F>`]
//!
//! ## v5.0 sealed catalogue shipped (per ADR-0100 §"Decision")
//!
//! - `LadderRung<2, F>` for [`crate::Diffusion4thChernoff<F>`]   (K=2 base; `PREDECESSOR_K` = None)
//! - `LadderRung<4, F>` for [`crate::Diffusion4thZeta4Chernoff<F>`] (`PREDECESSOR_K` = Some(2))
//! - `LadderRung<6, F>` for [`crate::Diffusion6thZeta6Chernoff<F>`] (`PREDECESSOR_K` = Some(4))
//! - `LadderRung<8, F>` for [`crate::Diffusion8thZeta8Chernoff<F>`] (`PREDECESSOR_K` = Some(6))

// Const generic `K` and length counters cast to f64 for error reporting; values ≪ 2^52.
#![allow(clippy::cast_precision_loss)]

use crate::{
    chernoff::ChernoffFunction, error::SemiflowError, float::SemiflowFloat, grid_fn::GridFn1D,
    grid_fn2d::GridFn2D,
};

// ---------------------------------------------------------------------------
// ApproximationSubspace trait
// ---------------------------------------------------------------------------

/// Opt-in marker super-trait: kernel declares order-K approximation subspace.
///
/// Witnesses the condition `f ∈ D(A^K)` required by Galkin-Remizov 2025 *IJM*
/// Theorem 3.1 for order-K Chernoff convergence. A type may implement this
/// for MULTIPLE values of K simultaneously (Example 4.2 multi-K co-witness).
///
/// ## Design
///
/// - NOT default-bridged (unlike `TimedChernoffFunction`). An impl must explicitly
///   write the `in_subspace` / `jet` bodies — silently returning `false` would be
///   a worse footgun than no impl at all (ADR-0073 §"Rationale").
/// - `in_subspace` returns `bool` (minimal predicate surface).
/// - `jet` writes to `&mut [Self::S]` (zero-alloc slice, per ADR-0041 pattern).
pub trait ApproximationSubspace<const K: usize, F: SemiflowFloat = f64>:
    ChernoffFunction<F>
{
    /// Returns `true` if `f` lies in the order-K approximation subspace D(A^K).
    ///
    /// Witnessing K-jet membership is a non-trivial claim; implementations
    /// should check datum validity (finite values, sufficient grid resolution)
    /// as a minimum. A `true` return licenses the order-K convergence claim
    /// per Galkin-Remizov 2025 §3.1.
    fn in_subspace(&self, f: &Self::S) -> bool;

    /// K-jet operator: writes `[A^0 f, A^1 f, ..., A^K f]` into `out`.
    ///
    /// `out` MUST have length `K + 1`; `out[0] = f` (identity); `out[K] = A^K f`.
    ///
    /// # Errors
    /// - [`SemiflowError::DomainViolation`] if `out.len() != K + 1`.
    /// - Any operator evaluation error propagates unchanged.
    fn jet(&self, f: &Self::S, out: &mut [Self::S]) -> Result<(), SemiflowError>;
}

// ---------------------------------------------------------------------------
// assert_in_subspace helper
// ---------------------------------------------------------------------------

/// Module-level convenience: verify datum membership in D(A^K).
///
/// Returns `Ok(())` when `chernoff.in_subspace(f)` is `true`.
/// Returns `DomainViolation` otherwise. Used by gate `G_AS_K` tests.
///
/// # Errors
/// - [`SemiflowError::DomainViolation`] when `f ∉ D(A^K)`.
pub fn assert_in_subspace<C, F, const K: usize>(chernoff: &C, f: &C::S) -> Result<(), SemiflowError>
where
    C: ApproximationSubspace<K, F>,
    F: SemiflowFloat,
{
    if chernoff.in_subspace(f) {
        Ok(())
    } else {
        Err(SemiflowError::DomainViolation {
            what: "f is not in D(A^K) (ApproximationSubspace witness failed)",
            value: K as f64,
        })
    }
}

// ---------------------------------------------------------------------------
// LadderRung<K, F> sealed super-trait (v5.0 A.6, ADR-0100)
// ---------------------------------------------------------------------------

/// Private sealed marker — prevents downstream crates from implementing
/// [`LadderRung<K, F>`] without going through an ADR + impl block inside
/// `semiflow-core` (rustls / thiserror sealed-trait pattern).
mod sealed {
    pub trait Sealed {}
    impl<F: crate::float::SemiflowFloat> Sealed for crate::diffusion4::Diffusion4thChernoff<F> {}
    impl<F: crate::float::SemiflowFloat> Sealed
        for crate::diffusion4_zeta4::Diffusion4thZeta4Chernoff<F>
    {
    }
    impl<F: crate::float::SemiflowFloat> Sealed
        for crate::diffusion6_zeta6::Diffusion6thZeta6Chernoff<F>
    {
    }
    impl<F: crate::float::SemiflowFloat> Sealed
        for crate::diffusion8_zeta8::Diffusion8thZeta8Chernoff<F>
    {
    }
}

/// Sealed sibling super-trait of `ApproximationSubspace<K, F>` codifying formal
/// membership in the ζ-ladder of nested-Richardson Chernoff rungs.
///
/// See math.md §36 for the K → K-2 invariant and Galkin-Remizov 2025 *IJM*
/// Theorem 3.1 specialisation. Sealed via `mod sealed` private marker — downstream
/// crates CANNOT impl `LadderRung<K, F>` without ADR + sealed impl block inside
/// `semiflow-core`.
///
/// ## v5.0 catalogue (4 rungs)
///
/// - `LadderRung<2, F>` for [`crate::Diffusion4thChernoff<F>`] (K=2 base; `PREDECESSOR_K` = None)
/// - `LadderRung<4, F>` for [`crate::Diffusion4thZeta4Chernoff<F>`] (`PREDECESSOR_K` = Some(2))
/// - `LadderRung<6, F>` for [`crate::Diffusion6thZeta6Chernoff<F>`] (`PREDECESSOR_K` = Some(4))
/// - `LadderRung<8, F>` for [`crate::Diffusion8thZeta8Chernoff<F>`] (`PREDECESSOR_K` = Some(6))
///
/// ## References
///
/// - ADR-0073 — `ApproximationSubspace<K, F>` super-bounded by this trait.
/// - ADR-0086/0088/0090 — populate the catalogue at K=2/4/6/8 respectively.
/// - math.md §36 — NORMATIVE typing surface spec + 4-rung catalogue.
pub trait LadderRung<const K: usize, F: SemiflowFloat = f64>:
    ApproximationSubspace<K, F> + sealed::Sealed
{
    /// Predecessor rung index, or `None` for the K=2 ladder base.
    ///
    /// `Some(K - 2)` for every rung K ≥ 4; the K → K-2 invariant is verified
    /// symbolically by the `T_LADDER_RUNG` sympy oracle (`RELEASE_BLOCKING` per ADR-0100).
    const PREDECESSOR_K: Option<usize>;
}

// ---------------------------------------------------------------------------
// LadderRung impls: K=2 base (Diffusion4thChernoff)
// ---------------------------------------------------------------------------

/// K=2 rung: `Diffusion4thChernoff<f64>` is the palindromic Strang K5 base.
///
/// `PREDECESSOR_K` = None (unique base case; verified by `T_LADDER_RUNG` sub-check (b)).
/// Requires the K=2 `ApproximationSubspace<2, f64>` precondition impl below (AC2).
impl LadderRung<2, f64> for crate::diffusion4::Diffusion4thChernoff<f64> {
    const PREDECESSOR_K: Option<usize> = None;
}

// ---------------------------------------------------------------------------
// LadderRung impls: K=4 (Diffusion4thZeta4Chernoff)
// ---------------------------------------------------------------------------

/// K=4 rung: Richardson on the K=2 base. `PREDECESSOR_K` = Some(2).
impl LadderRung<4, f64> for crate::diffusion4_zeta4::Diffusion4thZeta4Chernoff<f64> {
    const PREDECESSOR_K: Option<usize> = Some(2);
}

// ---------------------------------------------------------------------------
// LadderRung impls: K=6 (Diffusion6thZeta6Chernoff)
// ---------------------------------------------------------------------------

/// K=6 rung: Richardson on the K=4 rung. `PREDECESSOR_K` = Some(4).
impl LadderRung<6, f64> for crate::diffusion6_zeta6::Diffusion6thZeta6Chernoff<f64> {
    const PREDECESSOR_K: Option<usize> = Some(4);
}

// ---------------------------------------------------------------------------
// LadderRung impls: K=8 (Diffusion8thZeta8Chernoff — v4.3 Chebyshev kernel)
// ---------------------------------------------------------------------------

/// K=8 rung: Richardson on the K=6 rung. `PREDECESSOR_K` = Some(6).
impl LadderRung<8, f64> for crate::diffusion8_zeta8::Diffusion8thZeta8Chernoff<f64> {
    const PREDECESSOR_K: Option<usize> = Some(6);
}

// ---------------------------------------------------------------------------
// Opt-in impl: DiffusionChernoff — K=2
// ---------------------------------------------------------------------------

/// K=2 approximation subspace for the ζ-A diffusion kernel.
///
/// `in_subspace`: f lies in D(A^2) when the grid has ≥5 points (required
/// for the 5-point central-difference stencil) and all values are finite.
/// This is a pragmatic Rust check; the full `C^k_b` condition from math.md §26
/// is verified up to finite-difference proxy — flagged as v3.x refinement.
///
/// `jet`: computes `[f, Af, A²f]` via 3-point (f') and 5-point (f'') central FD
/// applied iteratively through the discrete generator. Returns `DomainViolation`
/// if `out.len() != 3`.
impl ApproximationSubspace<2, f64> for crate::diffusion::DiffusionChernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        // Minimum 5 points for central-difference 2nd-derivative stencil.
        f.values.len() >= 5 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 3 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=2 requires out.len() == 3",
                value: out.len() as f64,
            });
        }
        // out[0] = f (A^0 f = f)
        out[0].values.clone_from(&f.values);
        // out[1] = Af = ∂_x(a(x)·∂_x f) via 3-point central FD
        apply_divergence_form_op(self, f, &mut out[1])?;
        // out[2] = A²f = A(Af) — apply op again
        let af = out[1].clone();
        apply_divergence_form_op(self, &af, &mut out[2])?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opt-in impl: Diffusion4thChernoff — K=2 (LadderRung base precondition, ADR-0100)
// ---------------------------------------------------------------------------

/// K=2 approximation subspace for the palindromic Strang K5 kernel.
///
/// Added v5.0 as a strict precondition for `LadderRung<2, f64>` (ADR-0100 AC2).
/// The K5 palindromic Strang is order-2 on D(A²) per math §13 (symmetric semigroup
/// splitting theorem); this impl witnesses that tangency formally.
///
/// `in_subspace`: requires ≥9 grid points (same as K=4 below — 5-point stencil
/// applied twice to produce A²f) and all values finite. Pragmatic FD proxy for
/// `C²_b(Ω)` condition from math.md §26.
///
/// `jet`: writes `[f, Af, A²f]` via 2 iterations of `apply_divergence_form_op_d4`.
/// Returns `DomainViolation` if `out.len() != 3`.
impl ApproximationSubspace<2, f64> for crate::diffusion4::Diffusion4thChernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        // 9 points: 5-point central-difference stencil iterated twice (K=2 jet).
        f.values.len() >= 9 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 3 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=2 (Diffusion4thChernoff) requires out.len() == 3",
                value: out.len() as f64,
            });
        }
        // out[0] = A^0 f = f
        out[0].values.clone_from(&f.values);
        // out[1] = Af via 5-point divergence-form stencil
        apply_divergence_form_op_d4(self, f, &mut out[1])?;
        // out[2] = A²f = A(Af)
        let af = out[1].clone();
        apply_divergence_form_op_d4(self, &af, &mut out[2])?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opt-in impl: Diffusion4thChernoff — K=4
// ---------------------------------------------------------------------------

/// K=4 approximation subspace for the ζ⁴ 4th-order diffusion kernel.
///
/// `in_subspace`: requires ≥9 grid points (5-point stencil iterated 4×)
/// and all values finite. Pragmatic check; v3.x refinement per math.md §26.
///
/// `jet`: computes `[f, Af, A²f, A³f, A⁴f]` via repeated 5-point divergence-form
/// applications. Returns `DomainViolation` if `out.len() != 5`.
impl ApproximationSubspace<4, f64> for crate::diffusion4::Diffusion4thChernoff<f64> {
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        f.values.len() >= 9 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 5 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=4 requires out.len() == 5",
                value: out.len() as f64,
            });
        }
        out[0].values.clone_from(&f.values);
        apply_divergence_form_op_d4(self, f, &mut out[1])?;
        for k in 1..4usize {
            let prev = out[k].clone();
            apply_divergence_form_op_d4(self, &prev, &mut out[k + 1])?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opt-in impl: TruncatedExp4thDiffusionChernoff — K=6
// ---------------------------------------------------------------------------

/// K=6 approximation subspace for the truncated-exp K=4 diffusion kernel.
///
/// Though the type name contains "4th" (for the truncation order K=4 per ADR-0011
/// backward-compat naming), the *subspace* it witnesses is K=6 per math.md §27.
/// This is the witness the ζ⁴ correction (ADR-0075) consumes.
///
/// `in_subspace`: requires ≥13 grid points (7-point stencil iterated 6×)
/// and all values finite.
///
/// `jet`: computes `[f, Af, ..., A⁶f]` via 6 repeated 5-point divergence-form
/// applications on the same discrete generator. Returns `DomainViolation` if
/// `out.len() != 7`.
impl ApproximationSubspace<6, f64>
    for crate::truncated_exp4::TruncatedExp4thDiffusionChernoff<f64>
{
    fn in_subspace(&self, f: &GridFn1D<f64>) -> bool {
        f.values.len() >= 13 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(&self, f: &GridFn1D<f64>, out: &mut [GridFn1D<f64>]) -> Result<(), SemiflowError> {
        if out.len() != 7 {
            return Err(SemiflowError::DomainViolation {
                what: "jet K=6 requires out.len() == 7",
                value: out.len() as f64,
            });
        }
        out[0].values.clone_from(&f.values);
        apply_divergence_form_op_te4(self, f, &mut out[1])?;
        for k in 1..6usize {
            let prev = out[k].clone();
            apply_divergence_form_op_te4(self, &prev, &mut out[k + 1])?;
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Opt-in impl: HypoellipticChernoff — K=2 (Wave B, ADR-0077)
// ---------------------------------------------------------------------------

/// K=2 approximation subspace for the Kolmogorov hypoelliptic Chernoff kernel.
///
/// `in_subspace`: f lies in the K=2 domain when the 2D grid has ≥4 nodes on
/// each axis (required for Catmull-Rom interpolation) and all values are finite.
/// Full hypoelliptic C²-smoothness verification is a v3.x research item
/// (deferred per ADR-0077 §"Decision").
///
/// `jet`: 2-jet computation for hypoelliptic operators requires
/// bracket-aware second derivatives — deferred to v3.x. Returns
/// `Unsupported` if called; v3.1 shippers satisfy the K=2 tangency
/// proof via the palindromic Strang-Hörmander construction (math.md §28.3).
impl ApproximationSubspace<2, f64> for crate::hormander::KolmogorovHypoelliptic<f64> {
    fn in_subspace(&self, f: &GridFn2D<f64>) -> bool {
        let nx = f.grid.nx();
        let ny = f.grid.ny();
        nx >= 4 && ny >= 4 && f.values.iter().all(|v| v.is_finite())
    }

    fn jet(&self, _f: &GridFn2D<f64>, _out: &mut [GridFn2D<f64>]) -> Result<(), SemiflowError> {
        // 2-jet for hypoelliptic operators requires bracket-aware second derivatives.
        // Deferred to v3.x per ADR-0077 §"Decision" (research item).
        Err(SemiflowError::Unsupported {
            feature: "hypoelliptic-jet-K2",
        })
    }
}

// ---------------------------------------------------------------------------
// Private helpers: discrete divergence-form generator A f = ∂_x(a(x)·∂_x f)
// ---------------------------------------------------------------------------

/// 3-point divergence-form operator using a callable `a_fn: &dyn Fn(F) -> F`.
///
/// `(Af)_i = [a(x_{i+½})(f_{i+1}-f_i) - a(x_{i-½})(f_i-f_{i-1})] / dx²`.
/// Boundary: zero-Neumann (f_{-`1}=f_0`, `f_N=f`_{N-1}).
fn apply_div_form_fn<F: SemiflowFloat>(
    a_fn: &impl Fn(F) -> F,
    grid: crate::grid::Grid1D<F>,
    f: &GridFn1D<F>,
    out: &mut GridFn1D<F>,
) -> Result<(), SemiflowError> {
    let n = f.values.len();
    if n < 3 {
        return Err(SemiflowError::DomainViolation {
            what: "divergence-form jet op requires >= 3 grid points",
            value: n as f64,
        });
    }
    let dx = grid.dx();
    let dx2 = dx * dx;
    let half = crate::float::half::<F>();
    out.values.resize(n, F::zero());
    for i in 0..n {
        let x_i = grid.x_at(i);
        let a_pos = a_fn(x_i + dx * half);
        let a_neg = a_fn(x_i - dx * half);
        let f_pos = if i + 1 < n {
            f.values[i + 1]
        } else {
            f.values[n - 1]
        };
        let f_neg = if i > 0 { f.values[i - 1] } else { f.values[0] };
        let f_i = f.values[i];
        out.values[i] = (a_pos * (f_pos - f_i) - a_neg * (f_i - f_neg)) / dx2;
    }
    Ok(())
}

/// Dispatcher for `DiffusionChernoff<F>` (uses `call_a` for storage dispatch).
fn apply_divergence_form_op<F: SemiflowFloat>(
    dc: &crate::diffusion::DiffusionChernoff<F>,
    f: &GridFn1D<F>,
    out: &mut GridFn1D<F>,
) -> Result<(), SemiflowError> {
    let grid = dc.grid;
    apply_div_form_fn(&|x| dc.call_a(x), grid, f, out)
}

/// Dispatcher for `Diffusion4thChernoff<F>` (uses fn-ptr field `a` directly).
fn apply_divergence_form_op_d4<F: SemiflowFloat>(
    dc: &crate::diffusion4::Diffusion4thChernoff<F>,
    f: &GridFn1D<F>,
    out: &mut GridFn1D<F>,
) -> Result<(), SemiflowError> {
    let grid = dc.grid;
    let a_fn = dc.a;
    apply_div_form_fn(&|x| a_fn(x), grid, f, out)
}

/// Dispatcher for `TruncatedExp4thDiffusionChernoff<F>` (fn-ptr field `a`).
fn apply_divergence_form_op_te4<F: SemiflowFloat>(
    dc: &crate::truncated_exp4::TruncatedExp4thDiffusionChernoff<F>,
    f: &GridFn1D<F>,
    out: &mut GridFn1D<F>,
) -> Result<(), SemiflowError> {
    let grid = dc.grid;
    let a_fn = dc.a;
    apply_div_form_fn(&|x| a_fn(x), grid, f, out)
}
