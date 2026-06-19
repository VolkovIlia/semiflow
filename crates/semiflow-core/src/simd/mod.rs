//! v0.8.0 Block C — SIMD intrinsics scoped module (ADR-0019).
//!
//! `#![allow(unsafe_code)]` at MODULE scope — the crate root has
//! `#![deny(unsafe_code)]` (`lib.rs` line 69). `forbid` cannot be
//! relaxed by inner `#[allow]` per Rust lint hierarchy, so v0.8.0
//! Block C downgraded the crate-level lint to `deny`. The unsafe
//! blast radius remains bounded to `src/simd/{x86_64,aarch64}.rs`
//! by lint enforcement (no `#[allow(unsafe_code)]` outside this
//! module — verified by CI grep). The only `unsafe` in
//! this module sits inside `x86_64.rs` and `aarch64.rs` (intrinsic shims).
//!
//! Cross-ref: ADR-0019, contracts/semiflow-core.tensor.yaml `simd` block.

#![allow(unsafe_code)]

// ---------------------------------------------------------------------------
// Arch-specific implementations (only one compiles per target).
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
mod x86_64;

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
mod aarch64;

// Scalar is compiled on other arches AND under cfg(test) on all arches,
// AND on x86_64 without avx2, AND on aarch64 without neon.
#[cfg(any(
    test,
    not(any(
        all(target_arch = "x86_64", target_feature = "avx2"),
        all(target_arch = "aarch64", target_feature = "neon")
    ))
))]
mod scalar;

// ---------------------------------------------------------------------------
// Type alias: F64x4 resolves to the fastest impl for the target arch.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use aarch64::F64x4Neon as F64x4;
#[cfg(not(any(
    all(target_arch = "x86_64", target_feature = "avx2"),
    all(target_arch = "aarch64", target_feature = "neon")
)))]
pub(crate) use scalar::F64x4Scalar as F64x4;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use x86_64::F64x4Avx2 as F64x4;

// ---------------------------------------------------------------------------
// Wave B3: G⁴ stencil SIMD kernels re-exported for truncated_exp4_cached.
// ---------------------------------------------------------------------------

#[cfg(all(target_arch = "aarch64", target_feature = "neon"))]
pub(crate) use aarch64::apply_g4_stencil_neon_4nodes;
#[cfg(all(target_arch = "x86_64", target_feature = "avx2"))]
pub(crate) use x86_64::apply_g4_stencil_avx2_4nodes;

// ---------------------------------------------------------------------------
// Test-hook: thread-local flag to force scalar path even on x86_64/aarch64.
// ---------------------------------------------------------------------------

thread_local! {
    /// When `true`, hot-path SIMD call sites use `F64x4Scalar`.
    ///
    /// The `cfg!(test)` guard in consumer call sites makes this zero-cost in
    /// release builds (branch is eliminated by the optimizer).
    ///
    /// Exposed as `pub` (not `pub(crate)`) so integration tests in `tests/`
    /// can set/clear it. Not part of the stable public API.
    pub static FORCE_SCALAR: core::cell::Cell<bool> =
        const { core::cell::Cell::new(false) };
}

/// Run `closure` with the SIMD force-scalar flag active; resets afterwards.
///
/// Integration-test hook. Not part of the stable API.
pub fn with_force_scalar<T, F: FnOnce() -> T>(closure: F) -> T {
    FORCE_SCALAR.with(|c| c.set(true));
    let result = closure();
    FORCE_SCALAR.with(|c| c.set(false));
    result
}

// ---------------------------------------------------------------------------
// Trait — crate-private, 4-lane f64 SIMD.
// ---------------------------------------------------------------------------

/// 4-lane f64 SIMD trait, crate-private.
///
/// Determinism contract (ADR-0019 §`determinism_contract`):
/// every method is bit-equal to the corresponding scalar f64 op.
/// FMA is FORBIDDEN — `mul` and `add` are SEPARATE rounding steps.
#[allow(dead_code)] // full trait surface mandated by contract; not all methods used in v0.8.0
pub(crate) trait SimdF64x4: Copy {
    /// Broadcast scalar to all 4 lanes.
    fn splat(x: f64) -> Self;
    /// Load 4 contiguous f64 values (no alignment requirement).
    fn load_unaligned(src: &[f64; 4]) -> Self;
    /// Store 4 lanes to contiguous memory (no alignment requirement).
    fn store_unaligned(self, dst: &mut [f64; 4]);
    /// Lane-wise add. NO fused multiply-add.
    fn add(self, rhs: Self) -> Self;
    /// Lane-wise subtract.
    fn sub(self, rhs: Self) -> Self;
    /// Lane-wise multiply. NO fused multiply-add.
    fn mul(self, rhs: Self) -> Self;
    /// Reduce 4 lanes via deterministic order `((l0 + l1) + l2) + l3`.
    fn horizontal_sum(self) -> f64;
}
