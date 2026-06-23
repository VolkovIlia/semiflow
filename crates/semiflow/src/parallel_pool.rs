//! Per-precision thread-local scratch pools for 2D and 3D parallel Strang.
//!
//! Rust's `thread_local!` macro forbids generic statics, so Wave 5 (ADR-0045
//! §5.3) introduces one concrete `ScratchPool<f64>` and one `ScratchPool<f32>`
//! static per dimensionality class (2D, 3D), dispatched via sealed traits.
//!
//! ## Design (ADR-0045 §4)
//!
//! ```text
//! thread_local! {
//!     PARALLEL_2D_POOL_F64: RefCell<ScratchPool<f64>>
//!     PARALLEL_2D_POOL_F32: RefCell<ScratchPool<f32>>
//!     PARALLEL_3D_POOL_F64: RefCell<ScratchPool<f64>>
//!     PARALLEL_3D_POOL_F32: RefCell<ScratchPool<f32>>
//! }
//! ```
//!
//! `ParallelPool2D`/`ParallelPool3D` are sealed `pub(crate)` traits; only `f32`
//! and `f64` may implement them. The dispatch is zero-cost (monomorphisation
//! selects the right `thread_local!` at compile time).
//!
//! The f32 pool is **idle** in the common case (unit overhead: ~24 bytes per
//! thread for the empty `Vec<Vec<f32>>` header). See ADR-0045 §4.4.

use core::cell::RefCell;

use crate::{float::SemiflowFloat, scratch::ScratchPool};

// ---------------------------------------------------------------------------
// 2D thread-local pools (one per concrete F)
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread 2D scratch pool for `F = f64` (Wave 2, ADR-0042).
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static PARALLEL_2D_POOL_F64: RefCell<ScratchPool<f64>> =
        RefCell::new(ScratchPool::new());

    /// Per-thread 2D scratch pool for `F = f32` (Wave 5, ADR-0045).
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static PARALLEL_2D_POOL_F32: RefCell<ScratchPool<f32>> =
        RefCell::new(ScratchPool::new());
}

// ---------------------------------------------------------------------------
// 3D thread-local pools (one per concrete F)
// ---------------------------------------------------------------------------

thread_local! {
    /// Per-thread 3D scratch pool for `F = f64` (Wave 2, ADR-0042).
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static PARALLEL_3D_POOL_F64: RefCell<ScratchPool<f64>> =
        RefCell::new(ScratchPool::new());

    /// Per-thread 3D scratch pool for `F = f32` (Wave 5, ADR-0045).
    ///
    /// Not part of the stable public API; not frozen at v1.0.0.
    #[doc(hidden)]
    pub static PARALLEL_3D_POOL_F32: RefCell<ScratchPool<f32>> =
        RefCell::new(ScratchPool::new());
}

// ---------------------------------------------------------------------------
// Sealed 2D pool trait
// ---------------------------------------------------------------------------

/// Sealed trait dispatching the correct 2D per-precision thread-local pool.
///
/// `pub(crate)` prevents downstream crates from adding new implementations.
/// Only `f32` and `f64` may implement this trait (ADR-0045 §4.5).
pub(crate) trait ParallelPool2D: SemiflowFloat {
    /// Borrow the calling thread's 2D scratch pool for the duration of `f`.
    fn with_pool_2d<R>(f: impl FnOnce(&mut ScratchPool<Self>) -> R) -> R;
}

impl ParallelPool2D for f64 {
    fn with_pool_2d<R>(f: impl FnOnce(&mut ScratchPool<f64>) -> R) -> R {
        PARALLEL_2D_POOL_F64.with(|cell| f(&mut cell.borrow_mut()))
    }
}

impl ParallelPool2D for f32 {
    fn with_pool_2d<R>(f: impl FnOnce(&mut ScratchPool<f32>) -> R) -> R {
        PARALLEL_2D_POOL_F32.with(|cell| f(&mut cell.borrow_mut()))
    }
}

// ---------------------------------------------------------------------------
// Sealed 3D pool trait
// ---------------------------------------------------------------------------

/// Sealed trait dispatching the correct 3D per-precision thread-local pool.
///
/// `pub(crate)` prevents downstream crates from adding new implementations.
/// Only `f32` and `f64` may implement this trait (ADR-0045 §4.5).
pub(crate) trait ParallelPool3D: SemiflowFloat {
    /// Borrow the calling thread's 3D scratch pool for the duration of `f`.
    fn with_pool_3d<R>(f: impl FnOnce(&mut ScratchPool<Self>) -> R) -> R;
}

impl ParallelPool3D for f64 {
    fn with_pool_3d<R>(f: impl FnOnce(&mut ScratchPool<f64>) -> R) -> R {
        PARALLEL_3D_POOL_F64.with(|cell| f(&mut cell.borrow_mut()))
    }
}

impl ParallelPool3D for f32 {
    fn with_pool_3d<R>(f: impl FnOnce(&mut ScratchPool<f32>) -> R) -> R {
        PARALLEL_3D_POOL_F32.with(|cell| f(&mut cell.borrow_mut()))
    }
}

// ---------------------------------------------------------------------------
// Drain helpers (test hooks)
// ---------------------------------------------------------------------------

/// Drain the calling thread's 2D parallel scratch pool for precision `F`.
///
/// Clears all free buffers; capacity released. Used by
/// `tests/parallel_scratch_drain.rs`. Not part of the stable public API.
#[doc(hidden)]
#[allow(private_bounds)]
pub(crate) fn drain_thread_local_pools_2d_for<F: ParallelPool2D>() {
    F::with_pool_2d(|pool| *pool = ScratchPool::new());
}

/// Drain the calling thread's 3D parallel scratch pool for precision `F`.
///
/// Not part of the stable public API.
#[doc(hidden)]
#[allow(private_bounds)]
pub(crate) fn drain_thread_local_pools_3d_for<F: ParallelPool3D>() {
    F::with_pool_3d(|pool| *pool = ScratchPool::new());
}
