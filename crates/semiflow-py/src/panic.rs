//! Panic boundary macro for `PyO3` entry points.
//!
//! Any Rust panic propagating through a `PyO3` call boundary can cause
//! undefined behaviour when the Python runtime cannot unwind through Rust
//! frames.  Every `#[pymethods]` function wraps its body in `catch_panic_py!`
//! to convert panics into [`crate::error::SemiflowError`] `PyErr` instead.
//!
//! **Requirement**: build with `--profile release-ffi` (`panic = "unwind"`).
//! With `panic = "abort"` (the workspace release profile), `catch_unwind` is
//! a no-op and this macro cannot catch anything.

/// Execute `$body` inside `std::panic::catch_unwind`.
///
/// On panic, returns `Err(SemiflowError("Panic"))`.
/// `$body` must evaluate to a `PyResult<T>`.
macro_rules! catch_panic_py {
    ($body:expr) => {{
        match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| $body)) {
            Ok(result) => result,
            Err(_) => Err($crate::error::new_panic_pyerr()),
        }
    }};
}

pub(crate) use catch_panic_py;
