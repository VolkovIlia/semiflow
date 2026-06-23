//! `check-unsafe-scope` subcommand — enforce bounded unsafe across all crates.
//!
//! Policy (directory-level):
//!
//! **semiflow** — STRICT.  `unsafe` is permitted only in:
//! - `crates/semiflow/src/simd/`  — AVX2/NEON intrinsics (ADR-0019)
//! - `crates/semiflow/examples/cev_european_call.rs`  — GlobalAlloc instrumentation
//! - `crates/semiflow/examples/latency_tail.rs`       — GlobalAlloc instrumentation
//! - `crates/semiflow/tests/`  — prose in test comments references the word "unsafe"
//!
//! **Binding crates** — ALLOWED crate-wide (ADR-0028).  Each binding crate carries a
//! crate-level `#![allow(unsafe_code)]` because every file is an FFI / PyO3 /
//! wasm-bindgen boundary:
//! - `crates/semiflow-ffi/`   — C ABI `extern "C"` + `catch_unwind` boundaries
//! - `crates/semiflow-py/src/` + `tests/`  — PyO3 proc-macro expansion
//! - `crates/semiflow-wasm/src/` + `tests/` — wasm-bindgen proc-macro expansion
//!
//! All other `.rs` files under `crates/*/src/` must not contain `unsafe`
//! or `#[allow(unsafe_code)]`.

use anyhow::{bail, Result};
use walkdir::WalkDir;

/// Directory-level prefixes (relative to workspace root) where `unsafe` is
/// permitted in source files.
///
/// semiflow is STRICT: only the simd/ subtree, two GlobalAlloc examples,
/// and tests/ (prose references) are allowed (ADR-0019).
///
/// Binding crates carry a crate-level `#![allow(unsafe_code)]` and are allowed
/// crate-wide because every file is an FFI / PyO3 / wasm-bindgen boundary
/// (ADR-0028).
const ALLOWED_PREFIXES: &[&str] = &[
    // semiflow: STRICT — unsafe permitted only in SIMD intrinsics (ADR-0019),
    // two GlobalAlloc instrumentation examples, and test prose.
    "crates/semiflow/src/simd/",
    "crates/semiflow/examples/cev_european_call.rs",
    "crates/semiflow/examples/latency_tail.rs",
    "crates/semiflow/tests/",
    // Binding crates are FFI/PyO3/wasm-bindgen boundaries — unsafe is expected
    // crate-wide (each carries crate-level #![allow(unsafe_code)] per ADR-0028).
    "crates/semiflow-ffi/",
    "crates/semiflow-py/src/",
    "crates/semiflow-py/tests/",
    "crates/semiflow-wasm/src/",
    "crates/semiflow-wasm/tests/",
];

/// Walk `crates/*/src/**/*.rs` and fail if any file outside the allowlist
/// contains the `unsafe` keyword or `#[allow(unsafe_code)]`.
pub fn check_unsafe_scope() -> Result<()> {
    let root = crate::workspace_root()?;
    let crates_dir = root.join("crates");
    let mut violations: Vec<String> = Vec::new();

    for entry in WalkDir::new(&crates_dir)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        if is_allowed(&root, path) {
            continue;
        }
        let rel = path
            .strip_prefix(&root)
            .unwrap_or(path)
            .to_string_lossy()
            .into_owned();
        let src = std::fs::read_to_string(path)?;
        check_for_unsafe_usage(&rel, &src, &mut violations);
    }

    if violations.is_empty() {
        println!("check-unsafe-scope: PASS — unsafe bounded to allowlist");
        Ok(())
    } else {
        for v in &violations {
            eprintln!("UNSAFE SCOPE VIOLATION: {v}");
        }
        bail!("{} unsafe scope violation(s) found", violations.len());
    }
}

/// Return true if `path` matches any allowed prefix (normalised relative to `root`).
fn is_allowed(root: &std::path::Path, path: &std::path::Path) -> bool {
    let rel = match path.strip_prefix(root) {
        Ok(r) => r.to_string_lossy().replace('\\', "/"),
        Err(_) => return false,
    };
    ALLOWED_PREFIXES
        .iter()
        .any(|prefix| rel.starts_with(prefix) || rel == prefix.trim_end_matches('/'))
}

/// Check one file for unsafe keyword or allow(unsafe_code) annotations.
///
/// Skips:
/// - Crate-level lint-policy attributes (`#![deny/allow(unsafe_code)]`).
/// - Comment-only lines: lines whose trimmed content starts with `//`, `///`,
///   `//!`, `/*`, or `*` (continuation lines inside block comments).
pub fn check_for_unsafe_usage(rel: &str, src: &str, violations: &mut Vec<String>) {
    for (lineno, line) in src.lines().enumerate() {
        let trimmed = line.trim();
        // Skip crate-level lint-policy attributes (not unsafe sites).
        if trimmed == "#![deny(unsafe_code)]" || trimmed == "#![allow(unsafe_code)]" {
            continue;
        }
        // Skip pure comment lines — they mention "unsafe" in prose, not code.
        if trimmed.starts_with("//") || trimmed.starts_with("/*") || trimmed.starts_with('*') {
            continue;
        }
        let has_unsafe = trimmed.contains("unsafe") || trimmed.contains("allow(unsafe_code)");
        if has_unsafe {
            violations.push(format!("{}:{}: {}", rel, lineno + 1, trimmed));
        }
    }
}
