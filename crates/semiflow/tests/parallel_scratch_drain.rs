//! ADR-0042 acceptance criterion 7: thread-local parallel scratch-pool
//! behaviour under `feature = "parallel"`.
//!
//! Design note: `std::thread::scope` spawns fresh OS threads per `apply` call;
//! those threads each start with an empty thread-local pool (one allocation per
//! pencil-type per pass, then immediate return).  The thread-local pools serve
//! persistent worker threads that process many steps in a loop — they reduce
//! the per-step allocation count from `O(pencils)` to `O(threads)` within each
//! call.  The tests here verify the pool API contract:
//!
//! 1. High-water mark on the *same* thread settles within 4 borrows.
//! 2. `drain_thread_local_pools()` resets pools to zero capacity.
//! 3. Pool re-fills on the next borrow after a drain.
//!
//! The pool API is tested via `PARALLEL_2D_POOL` / `PARALLEL_3D_POOL` directly
//! on the test thread (the same code path used by every worker thread).

#![cfg(feature = "parallel")]

use semiflow::{
    drain_thread_local_pools,
    strang2d_parallel::{drain_thread_local_pools_2d, PARALLEL_2D_POOL},
    strang3d_parallel::{drain_thread_local_pools_3d, PARALLEL_3D_POOL},
};

// ---------------------------------------------------------------------------
// Helper: take + return a buffer of `n` from a pool and check high-water mark
// ---------------------------------------------------------------------------

fn take_return_2d(n: usize) {
    PARALLEL_2D_POOL.with(|cell| {
        let buf = cell.borrow_mut().take_vec(n);
        cell.borrow_mut().return_vec(buf);
    });
}

fn take_return_3d(n: usize) {
    PARALLEL_3D_POOL.with(|cell| {
        let buf = cell.borrow_mut().take_vec(n);
        cell.borrow_mut().return_vec(buf);
    });
}

fn hwm_2d() -> usize {
    PARALLEL_2D_POOL.with(|cell| cell.borrow().high_water_capacity)
}

fn hwm_3d() -> usize {
    PARALLEL_3D_POOL.with(|cell| cell.borrow().high_water_capacity)
}

fn free_count_2d() -> usize {
    PARALLEL_2D_POOL.with(|cell| cell.borrow().free_len())
}

fn free_count_3d() -> usize {
    PARALLEL_3D_POOL.with(|cell| cell.borrow().free_len())
}

// ---------------------------------------------------------------------------
// 1. High-water mark settles within 4 borrows
// ---------------------------------------------------------------------------

#[test]
fn pool_2d_hwm_settles_within_4_borrows() {
    // Reset to known state.
    drain_thread_local_pools_2d();
    assert_eq!(hwm_2d(), 0, "hwm should be 0 after drain");

    let n = 128_usize;

    // 4 borrow/return cycles — hwm reaches `n` on the first and stays there.
    for call in 0..4 {
        take_return_2d(n);
        let hwm = hwm_2d();
        assert_eq!(
            hwm, n,
            "2D pool HWM should be {n} after call {call}, got {hwm}"
        );
    }

    // After 4 borrows, HWM is stable at n.
    let final_hwm = hwm_2d();
    assert_eq!(
        final_hwm, n,
        "2D pool HWM must be stable at {n} (was {final_hwm})"
    );
}

#[test]
fn pool_3d_hwm_settles_within_4_borrows() {
    drain_thread_local_pools_3d();
    assert_eq!(hwm_3d(), 0, "hwm should be 0 after drain");

    let n = 64_usize;

    for call in 0..4 {
        take_return_3d(n);
        let hwm = hwm_3d();
        assert_eq!(
            hwm, n,
            "3D pool HWM should be {n} after call {call}, got {hwm}"
        );
    }

    let final_hwm = hwm_3d();
    assert_eq!(
        final_hwm, n,
        "3D pool HWM must be stable at {n} (was {final_hwm})"
    );
}

// ---------------------------------------------------------------------------
// 2. drain clears capacity (free list is empty, hwm is 0 after drain)
// ---------------------------------------------------------------------------

#[test]
fn drain_2d_clears_pool() {
    // First fill the pool.
    take_return_2d(256);
    assert!(hwm_2d() >= 256, "pool should have capacity after borrow");
    assert_eq!(free_count_2d(), 1, "one buffer should be free after return");

    // Drain.
    drain_thread_local_pools_2d();
    assert_eq!(hwm_2d(), 0, "HWM must be 0 after drain");
    assert_eq!(free_count_2d(), 0, "free list must be empty after drain");
}

#[test]
fn drain_3d_clears_pool() {
    take_return_3d(256);
    assert!(hwm_3d() >= 256, "pool should have capacity after borrow");
    assert_eq!(free_count_3d(), 1, "one buffer should be free after return");

    drain_thread_local_pools_3d();
    assert_eq!(hwm_3d(), 0, "HWM must be 0 after drain");
    assert_eq!(free_count_3d(), 0, "free list must be empty after drain");
}

#[test]
fn combined_drain_clears_both() {
    take_return_2d(128);
    take_return_3d(128);
    assert_eq!(free_count_2d(), 1);
    assert_eq!(free_count_3d(), 1);

    drain_thread_local_pools();

    assert_eq!(hwm_2d(), 0, "2D pool must be 0 after combined drain");
    assert_eq!(hwm_3d(), 0, "3D pool must be 0 after combined drain");
    assert_eq!(free_count_2d(), 0);
    assert_eq!(free_count_3d(), 0);
}

// ---------------------------------------------------------------------------
// 3. Pool re-fills on next borrow after drain
// ---------------------------------------------------------------------------

#[test]
fn pool_2d_refills_after_drain() {
    drain_thread_local_pools_2d();
    assert_eq!(free_count_2d(), 0, "empty after drain");

    // First borrow after drain must allocate a new buffer.
    take_return_2d(32);
    assert_eq!(hwm_2d(), 32, "HWM re-established after re-fill");
    assert_eq!(free_count_2d(), 1, "buffer returned to pool");
}

#[test]
fn pool_3d_refills_after_drain() {
    drain_thread_local_pools_3d();
    assert_eq!(free_count_3d(), 0, "empty after drain");

    take_return_3d(32);
    assert_eq!(hwm_3d(), 32, "HWM re-established after re-fill");
    assert_eq!(free_count_3d(), 1, "buffer returned to pool");
}
