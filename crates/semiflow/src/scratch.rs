//! Grow-only scratch-buffer pool for allocation-free Chernoff hot loops (Wave 1, ADR-0041).
//!
//! ## Design
//!
//! [`ScratchPool<F>`] owns a free-list of `Vec<F>` buffers. When [`ScratchPool::borrow_vec`]
//! is called it pops the first buffer with sufficient capacity (or allocates a fresh one),
//! fills it to `len` with `F::zero()`, and returns a RAII handle [`ScratchVec<'_, F>`].
//!
//! On [`Drop`], `ScratchVec` calls `buf.clear()` (sets `len = 0`, keeps `capacity`) then
//! pushes the buffer back onto the free-list. This makes re-use zero-cost for the common
//! steady-state: every `apply_into` call borrows, fills, and returns the same buffer.
//!
//! ## Aliasing safety
//!
//! `borrow_vec` requires `&'a mut ScratchPool<F>`, so the borrow checker enforces
//! single-borrow exclusivity. No `RefCell`, no `unsafe`.
//!
//! ## `no_std` + `alloc`
//!
//! Uses only `alloc::vec::Vec` — no `std` imports.

use alloc::vec::Vec;

use crate::float::SemiflowFloat;

// ---------------------------------------------------------------------------
// ScratchPool
// ---------------------------------------------------------------------------

/// Grow-only pool of reusable `Vec<F>` scratch buffers.
///
/// Call [`Self::borrow_vec`] to obtain a `ScratchVec<'_, F>` whose lifetime is
/// tied to the pool borrow, statically preventing double-borrow.
///
/// # Thread safety
///
/// `ScratchPool<F>` is `Send` (can move to another thread) but not `Sync`
/// (exclusively owned; single-threaded use only). The `&mut self` requirement on
/// `borrow_vec` enforces this at compile time.
///
/// # Example
///
/// ```rust
/// use semiflow::scratch::ScratchPool;
/// let mut pool: ScratchPool<f64> = ScratchPool::new();
/// {
///     let mut v = pool.borrow_vec(8);
///     v.iter_mut().enumerate().for_each(|(i, x)| *x = i as f64);
///     assert_eq!(v[3], 3.0);
/// } // ScratchVec dropped here → buffer returned to pool
/// // pool can lend it out again
/// let v2 = pool.borrow_vec(4);
/// assert_eq!(v2.len(), 4);
/// ```
pub struct ScratchPool<F: SemiflowFloat> {
    free: Vec<Vec<F>>,
    /// Largest capacity ever borrowed from this pool.
    pub high_water_capacity: usize,
}

impl<F: SemiflowFloat> Default for ScratchPool<F> {
    fn default() -> Self {
        Self::new()
    }
}

impl<F: SemiflowFloat> ScratchPool<F> {
    /// Create an empty pool with no pre-allocated buffers.
    #[must_use]
    pub fn new() -> Self {
        Self {
            free: Vec::new(),
            high_water_capacity: 0,
        }
    }

    /// Create a pool pre-populated with `count` buffers, each with `cap` capacity.
    ///
    /// Suitable as the initial scratch for `ChernoffSemigroup::evolve` when the
    /// node count is known ahead of time.
    #[must_use]
    pub fn with_capacity(count: usize, cap: usize) -> Self {
        let mut pool = Self::new();
        for _ in 0..count {
            pool.free.push(Vec::with_capacity(cap));
        }
        if cap > 0 {
            pool.high_water_capacity = cap;
        }
        pool
    }

    /// Borrow a `Vec<F>` of length `len`, filled with `F::zero()`.
    ///
    /// Selects the first free buffer with sufficient capacity; allocates a new
    /// one if none fit. The returned [`ScratchVec`] is tied to this pool's
    /// lifetime — the borrow checker prevents two live handles on the same pool.
    ///
    /// # Panics
    ///
    /// Does not panic. All allocations use the global allocator.
    pub fn borrow_vec(&mut self, len: usize) -> ScratchVec<'_, F> {
        if len > self.high_water_capacity {
            self.high_water_capacity = len;
        }
        // Find first buffer with enough capacity (smallest-fits-first scan).
        let pos = self.free.iter().position(|b| b.capacity() >= len);
        let mut buf = match pos {
            Some(idx) => self.free.swap_remove(idx),
            None => Vec::with_capacity(len),
        };
        // Resize to exactly `len`, zero-filling new slots (or truncating).
        buf.resize(len, F::zero());
        ScratchVec { buf, pool: self }
    }

    /// Take ownership of a scratch `Vec<F>` of length `len`, filled with `F::zero()`.
    ///
    /// Unlike [`borrow_vec`], the returned `Vec` is fully owned by the caller.
    /// The caller MUST return it via [`Self::return_vec`] to reclaim capacity.
    /// Use this when multiple buffers must be live simultaneously (e.g. the
    /// K=4 g-grid chain in `TruncatedExp` `apply_into`).
    ///
    /// # Panics
    ///
    /// Does not panic. All allocations use the global allocator.
    pub fn take_vec(&mut self, len: usize) -> Vec<F> {
        if len > self.high_water_capacity {
            self.high_water_capacity = len;
        }
        let pos = self.free.iter().position(|b| b.capacity() >= len);
        let mut buf = match pos {
            Some(idx) => self.free.swap_remove(idx),
            None => Vec::with_capacity(len),
        };
        buf.resize(len, F::zero());
        buf
    }

    /// Return a previously taken `Vec` to the pool (capacity preserved).
    ///
    /// Pair with [`Self::take_vec`]. After this call the `Vec` must not be used.
    /// Calling this with a buffer that was never taken is harmless (it just adds
    /// a free buffer to the pool).
    pub fn return_vec(&mut self, mut buf: Vec<F>) {
        buf.clear();
        self.free.push(buf);
    }

    /// Number of buffers currently in the free list.
    ///
    /// Useful for tests that verify pool drain/refill semantics.
    /// Not part of the stable v1.0.0 public API; subject to change.
    #[doc(hidden)]
    #[must_use]
    pub fn free_len(&self) -> usize {
        self.free.len()
    }

    // -----------------------------------------------------------------------
    // Graph-signal arena (R4 mitigation — Wave 2.1B)
    // -----------------------------------------------------------------------

    /// Take ownership of an `N`-length `Vec<F>` for use as a graph-signal
    /// scratch buffer (capacity `n`).
    ///
    /// Separate from the regular `take_vec` / `return_vec` pool so that the
    /// caller can distinguish "scalar scratch" from "graph-signal scratch"
    /// (useful for zero-alloc tests). Returns the first free buffer with
    /// sufficient capacity, or allocates a fresh one.
    ///
    /// Pair with [`Self::return_graph_buf`] to reclaim capacity.
    pub fn take_graph_buf(&mut self, n: usize) -> Vec<F> {
        if n > self.high_water_capacity {
            self.high_water_capacity = n;
        }
        // Reuse a free buf that already has enough capacity.
        let pos = self.free.iter().position(|b| b.capacity() >= n);
        let mut buf = match pos {
            Some(idx) => self.free.swap_remove(idx),
            None => Vec::with_capacity(n),
        };
        buf.resize(n, F::zero());
        buf
    }

    /// Return a graph-signal scratch buffer to the pool (capacity preserved).
    ///
    /// Pair with [`Self::take_graph_buf`].
    pub fn return_graph_buf(&mut self, mut buf: Vec<F>) {
        buf.clear();
        self.free.push(buf);
    }
}

// ---------------------------------------------------------------------------
// ScratchVec
// ---------------------------------------------------------------------------

/// RAII handle for a buffer borrowed from a [`ScratchPool`].
///
/// Derefs to `&[F]` / `&mut [F]`. On drop, the buffer is returned to the
/// pool (capacity preserved, length zeroed).
pub struct ScratchVec<'a, F: SemiflowFloat> {
    buf: Vec<F>,
    pool: &'a mut ScratchPool<F>,
}

impl<F: SemiflowFloat> core::ops::Deref for ScratchVec<'_, F> {
    type Target = [F];

    #[inline]
    fn deref(&self) -> &[F] {
        &self.buf
    }
}

impl<F: SemiflowFloat> core::ops::DerefMut for ScratchVec<'_, F> {
    #[inline]
    fn deref_mut(&mut self) -> &mut [F] {
        &mut self.buf
    }
}

impl<F: SemiflowFloat> Drop for ScratchVec<'_, F> {
    fn drop(&mut self) {
        // Swap out the buffer so we can push it back without a clone.
        let mut buf = Vec::new();
        core::mem::swap(&mut buf, &mut self.buf);
        buf.clear(); // len → 0; capacity preserved
        self.pool.free.push(buf);
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn borrow_return_cycle() {
        let mut pool: ScratchPool<f64> = ScratchPool::new();
        {
            let v = pool.borrow_vec(8);
            assert_eq!(v.len(), 8);
            assert!(v.iter().all(|&x| x == 0.0));
        }
        // Buffer should be back in the pool now.
        assert_eq!(pool.free.len(), 1);
        assert!(pool.free[0].capacity() >= 8);
    }

    #[test]
    fn reuse_preserves_capacity() {
        let mut pool: ScratchPool<f64> = ScratchPool::with_capacity(1, 64);
        let cap_before = pool.free[0].capacity();
        {
            let mut v = pool.borrow_vec(64);
            v[0] = 42.0;
        }
        // Reuse same buffer.
        let v2 = pool.borrow_vec(32);
        assert!(v2.len() == 32);
        // All zero after borrow (resize fills with zero).
        assert!(v2.iter().all(|&x| x == 0.0));
        drop(v2);
        assert!(pool.free[0].capacity() >= cap_before);
    }

    #[test]
    fn with_capacity_preallocates() {
        let pool: ScratchPool<f64> = ScratchPool::with_capacity(3, 128);
        assert_eq!(pool.free.len(), 3);
        assert!(pool.free.iter().all(|b| b.capacity() >= 128));
    }

    #[test]
    fn high_water_mark_tracks() {
        let mut pool: ScratchPool<f64> = ScratchPool::new();
        {
            let _v = pool.borrow_vec(16);
        }
        assert_eq!(pool.high_water_capacity, 16);
        {
            let _v = pool.borrow_vec(32);
        }
        assert_eq!(pool.high_water_capacity, 32);
        {
            let _v = pool.borrow_vec(8); // smaller — no change
        }
        assert_eq!(pool.high_water_capacity, 32);
    }
}
