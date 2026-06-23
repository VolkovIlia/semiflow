//! [`State`], [`HilbertState`], and [`Discrete`] traits — 3-layer vector
//! interface for Chernoff iteration (Wave 3, ADR-0043, v2.0 MAJOR).
//!
//! ## Layer 1 — [`State<F>`]
//!
//! Zero-allocation Banach-vector primitives sufficient for forward Chernoff
//! iteration (Theorem 6, Remizov 2025). **`Clone` is NOT a supertrait** — see
//! ADR-0043 for rationale. Concrete `GridFn{1,2,3}D<F>` retain `Clone` via
//! `#[derive(Clone)]` and also retain the v1.x inherent methods (`axpy`,
//! `scale`, `zeroed_like`) for source-level backward compatibility.
//!
//! ## Layer 2 — [`HilbertState<F>`]
//!
//! Inner-product extension for L²-flavoured error estimators (Wave 4 adaptive
//! PI controller foreshadow) and adjoint-Chernoff schemes. Counting-measure ℓ²:
//! `dot(a,b) = Σᵢ aᵢ·bᵢ`. Default impls for `norm_sq` and `norm_l2`.
//!
//! ## Layer 3 — [`Discrete<F>`]
//!
//! Graph/manifold/lattice extension. Uses GATs (stable Rust 1.65, MSRV 1.78)
//! to eliminate `Box<dyn Iterator>` per the v0.14.0 spike finding. Tensor
//! grid types (`GridFn{1,2,3}D`) do **not** implement `Discrete<F>`.
//!
//! ## Macro
//!
//! [`impl_state_for_gridfn!`] generates identical `State<F>` + `HilbertState<F>`
//! impls for all three tensor grid-fn types without duplicating code.

use crate::float::SemiflowFloat;
use num_traits::Float;

// ---------------------------------------------------------------------------
// Layer 1 — State<F>
// ---------------------------------------------------------------------------

/// Minimal zero-allocation Banach-vector interface for Chernoff iteration.
///
/// **Breaking change vs v1.x**: `Clone` is no longer a supertrait. Concrete
/// `GridFn{1,2,3}D<F>` retain `Clone` via `#[derive(Clone)]` and provide v1.x
/// inherent shims (`axpy`, `scale`, `zeroed_like`). Generic code that previously
/// relied on `T: State<F>` implying `T: Clone` must add an explicit `+ Clone`
/// bound.
///
/// ## Algebraic laws (modulo f64 rounding)
///
/// - `axpy_into(F::zero(), &x)` is a no-op on `self`.
/// - After `copy_from(&src)`: `self` is node-wise equal to `src`; `len()` unchanged.
/// - After `zero_into()`: `norm_sup() == F::zero()`; `len()` unchanged.
/// - `len()` is invariant under `axpy_into`, `copy_from`, `zero_into`.
/// - `norm_sup()` returns `NaN` iff `self` contains any `NaN`.
///
/// ## When to implement
///
/// Implement `State<F>` for any function-on-a-grid type you want to evolve
/// through a [`crate::ChernoffFunction`]. See [`HilbertState`] and [`Discrete`]
/// for optional extensions.
///
/// ## Example (v2.0 style)
///
/// ```rust
/// use semiflow::{Grid1D, GridFn1D, State};
/// let grid = Grid1D::new(-1.0, 1.0, 8).unwrap();
/// let mut u = GridFn1D::from_fn(grid, |x| x);
/// let v     = GridFn1D::from_fn(grid, |x| x * x);
/// // Wave 3: axpy_into (zero-alloc)
/// <GridFn1D<f64> as State<f64>>::axpy_into(&mut u, 2.0, &v);
/// assert!(u.norm_sup() > 0.0);
/// // v1.x source-compat inherent method still works:
/// u.axpy(1.0, &v);
/// ```
pub trait State<F: SemiflowFloat = f64> {
    /// Number of degrees of freedom (grid nodes, graph nodes, etc.).
    fn len(&self) -> usize;

    /// Returns `true` if this state has no nodes.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// In-place AXPY: `self ← self + alpha · src`. No allocation.
    ///
    /// # Panics (debug)
    /// `debug_assert_eq!(self.len(), src.len())` — shape mismatch.
    fn axpy_into(&mut self, alpha: F, src: &Self);

    /// Node-wise copy: `self ← src`. No allocation.
    ///
    /// # Panics (debug)
    /// `debug_assert_eq!(self.len(), src.len())` — shape mismatch.
    fn copy_from(&mut self, src: &Self);

    /// Zero all nodes: `self[i] ← 0` for all `i`. `len()` unchanged.
    fn zero_into(&mut self);

    /// Sup-norm `‖self‖_∞ = max_i |self[i]|`.
    ///
    /// Returns `NaN` if any node is `NaN`.
    fn norm_sup(&self) -> F;

    /// In-place scale: `self ← k · self`.
    ///
    /// **Must be overridden.** The default panics with `unimplemented!`.
    /// All concrete types in `semiflow-core` override this. Generic code should
    /// use concrete types' inherent `scale(k)` or override this method.
    fn scale_into(&mut self, k: F) {
        let _ = k;
        unimplemented!(
            "State::scale_into: implementors must override this method. \
             Concrete GridFnXD<F> provide an inherent scale(k) method."
        )
    }
}

// ---------------------------------------------------------------------------
// Layer 2 — HilbertState<F>
// ---------------------------------------------------------------------------

/// Hilbert-space extension: inner product and L²-norm.
///
/// Extends [`State<F>`] for adjoint-Chernoff and L²-error estimators.
/// Wave 4 will use `dot` + `norm_sq` in the zero-alloc Richardson error
/// estimator and H211b adaptive step controller.
///
/// **Counting-measure ℓ²**: `dot(a,b) = Σᵢ aᵢ·bᵢ`. For weighted L²-norms,
/// callers multiply by `√(dx·dy·dz)` outside the trait.
///
/// ## Algebraic laws
///
/// - `dot(a, b) == dot(b, a)` (symmetric, real-valued).
/// - `dot(a, a) == norm_sq(a)`.
/// - `norm_sq(a) ≥ 0.0` for finite `a`.
/// - After `zero_into()`: `dot(s, anything) == 0.0`.
pub trait HilbertState<F: SemiflowFloat = f64>: State<F> {
    /// Inner product `⟨self, other⟩ = Σᵢ self[i] · other[i]`.
    ///
    /// # Panics (debug)
    /// `debug_assert_eq!(self.len(), other.len())`.
    fn dot(&self, other: &Self) -> F;

    /// `‖self‖² = ⟨self, self⟩`. Default impl provided.
    #[inline]
    fn norm_sq(&self) -> F {
        self.dot(self)
    }

    /// `‖self‖₂ = √‖self‖²`. Provided convenience method.
    #[inline]
    fn norm_l2(&self) -> F
    where
        F: Float,
    {
        self.norm_sq().sqrt()
    }
}

// ---------------------------------------------------------------------------
// Layer 3 — Discrete<F>
// ---------------------------------------------------------------------------

/// Discrete-domain state for graph, manifold, and lattice geometries.
///
/// Extends [`State<F>`] with indexed node access and GAT-based neighbour
/// iteration. **Tensor types (`GridFn{1,2,3}D`) do NOT implement this trait**
/// — there is no canonical neighbour set for a tensor grid, and tensor
/// pipelines use slice pencils (Wave 2).
///
/// The GAT `type Neighbours<'a>` eliminates the `Box<dyn Iterator>` allocation
/// flagged by the v0.14.0 spike (stable since Rust 1.65; MSRV is 1.78).
///
/// ## Boundary conditions
///
/// Implicit Dirichlet zero: returning an empty iterator from `neighbours` for
/// boundary nodes encodes zero-value BC. Explicit BC enum is deferred to v2.x.
pub trait Discrete<F: SemiflowFloat = f64>: State<F> {
    /// Node index type. Must be `Copy + Eq + Hash` for use in maps.
    type Idx: Copy + Eq + core::hash::Hash;

    /// Iterator type for `(neighbour_idx, edge_weight)` pairs.
    type Neighbours<'a>: Iterator<Item = (Self::Idx, F)>
    where
        Self: 'a;

    /// Value at node `idx`.
    fn get(&self, idx: Self::Idx) -> F;

    /// Set value at node `idx` to `val`.
    fn set(&mut self, idx: Self::Idx, val: F);

    /// Iterate over all node indices. Order is stable across calls
    /// when `self` is unchanged.
    fn indices(&self) -> impl Iterator<Item = Self::Idx> + '_;

    /// Iterate over `(neighbour_idx, edge_weight)` pairs for node `idx`.
    ///
    /// Empty iterator = Dirichlet-zero boundary condition.
    fn neighbours(&self, idx: Self::Idx) -> Self::Neighbours<'_>;
}

// ---------------------------------------------------------------------------
// Shared macro helpers — used by impl_state_for_gridfn! (extracted batch H9b)
// ---------------------------------------------------------------------------

/// `axpy_into` body shared by all `GridFn{1,2,3}D<F>` — exact float op order preserved.
#[inline]
pub fn gridfn_axpy_into_slice<F: SemiflowFloat>(dst: &mut [F], alpha: F, src: &[F]) {
    debug_assert_eq!(dst.len(), src.len(), "axpy_into: shape mismatch");
    for (s, &x) in dst.iter_mut().zip(src.iter()) {
        *s += alpha * x;
    }
}

/// `copy_from` body shared by all `GridFn{1,2,3}D<F>`.
#[inline]
pub fn gridfn_copy_from_slice<F: SemiflowFloat>(dst: &mut [F], src: &[F]) {
    debug_assert_eq!(dst.len(), src.len(), "copy_from: shape mismatch");
    dst.copy_from_slice(src);
}

/// `norm_sup` body shared by all `GridFn{1,2,3}D<F>` — exact float op order preserved.
#[inline]
pub fn gridfn_norm_sup_slice<F: SemiflowFloat>(vals: &[F]) -> F {
    vals.iter().fold(F::zero(), |acc, &v| {
        let av = <F as num_traits::Float>::abs(v);
        if av > acc {
            av
        } else {
            acc
        }
    })
}

/// `dot` body shared by all `GridFn{1,2,3}D<F>` — exact float op order preserved.
#[inline]
pub fn gridfn_dot_slice<F: SemiflowFloat>(a: &[F], b: &[F]) -> F {
    debug_assert_eq!(a.len(), b.len(), "dot: shape mismatch");
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |acc, (&ai, &bi)| acc + ai * bi)
}

// ---------------------------------------------------------------------------
// Shared macro for GridFn{1,2,3}D — State<F> + HilbertState<F>
// ---------------------------------------------------------------------------

/// Generate `State<F>` and `HilbertState<F>` impls for a grid-fn type.
///
/// Used by `GridFn1D`, `GridFn2D`, and `GridFn3D` to keep the implementations
/// textually identical while staying under the 500-LoC file cap for each.
///
/// The generated impl assumes the type has `pub values: Vec<F>` and `pub grid`.
// Scanner note: `impl_state_for_gridfn` contains the substring `fn ` in its name.
// The macro body is compact (≤50 lines) so no #[allow] is needed (batch H9b).
#[macro_export]
macro_rules! impl_state_for_gridfn {
    ($Ty:path) => {
        impl<F: $crate::float::SemiflowFloat> $crate::state::State<F> for $Ty {
            #[inline]
            fn len(&self) -> usize {
                self.values.len()
            }
            #[inline]
            fn axpy_into(&mut self, alpha: F, src: &Self) {
                $crate::state::gridfn_axpy_into_slice(&mut self.values, alpha, &src.values);
            }
            #[inline]
            fn copy_from(&mut self, src: &Self) {
                $crate::state::gridfn_copy_from_slice(&mut self.values, &src.values);
            }
            #[inline]
            fn zero_into(&mut self) {
                for v in &mut self.values {
                    *v = <F as num_traits::Zero>::zero();
                }
            }
            #[inline]
            fn norm_sup(&self) -> F {
                $crate::state::gridfn_norm_sup_slice(&self.values)
            }
            #[inline]
            fn scale_into(&mut self, k: F) {
                for v in &mut self.values {
                    *v *= k;
                }
            }
        }
        impl<F: $crate::float::SemiflowFloat> $crate::state::HilbertState<F> for $Ty {
            #[inline]
            fn dot(&self, other: &Self) -> F {
                $crate::state::gridfn_dot_slice(&self.values, &other.values)
            }
        }
    };
}
