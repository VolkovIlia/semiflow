//! Panic boundary macro for FFI entry points.
//!
//! Any Rust panic crossing an FFI boundary is undefined behaviour.  Every
//! `extern "C"` function in `ffi.rs` wraps its body in `catch_panic!` to
//! convert panics into [`SemiflowStatus::Panic`] instead of unwinding through C.
//!
//! **Requirement**: build with `--profile release-ffi` so that
//! `panic = "unwind"` is active; with `panic = "abort"` (the workspace
//! release profile), `catch_unwind` is a no-op and this macro cannot catch
//! anything.

/// Execute `$body` inside `std::panic::catch_unwind`.
///
/// On panic, returns [`crate::status::SemiflowStatus::Panic`].
/// The body expression must evaluate to a [`crate::status::SemiflowStatus`].
macro_rules! catch_panic {
    ($body:expr) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(s) => s,
            Err(_) => $crate::status::SemiflowStatus::Panic,
        }
    }};
}
