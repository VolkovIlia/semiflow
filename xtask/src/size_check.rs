//! `binary-size-check` subcommand (D3, v0.13.0).
//!
//! Reports sizes of built binding artefacts vs target budgets.  Fails (exit 1)
//! if any existing artefact exceeds its budget — compatible with CI gating.
//!
//! ## Targets (per ADR-0028 Amendment 2)
//!
//! | Artefact | Budget |
//! |----------|--------|
//! | `semiflow-wasm/pkg-web/*.wasm`  | 500 KB |
//! | `semiflow-wasm/pkg-node/*.wasm` | 500 KB |
//! | `semiflow-ffi` cdylib (Linux)   | 1.5 MB |
//! | `semiflow-py` cdylib            |   5 MB |
//!
//! ## CI integration
//!
//! This check intentionally does NOT build artefacts — it only measures what
//! is already on disk.  CI jobs that produce artefacts (wasm-build, ffi-build,
//! py-build) must run before this check to produce meaningful output.
//!
//! Artefacts that do not exist on disk are reported as "NOT BUILT (skip)" and
//! do not count as failures, so the check can be run locally without building all
//! targets first.

use std::{fmt, fs, path::Path};

use anyhow::Result;

// ---------------------------------------------------------------------------
// Budget table
// ---------------------------------------------------------------------------

/// One size-budget entry.
struct Budget {
    /// Human-readable artefact label.
    label: &'static str,
    /// Glob suffix to locate the artefact (relative to workspace root).
    pattern: &'static str,
    /// Maximum allowed file size in bytes.
    limit_bytes: u64,
}

const BUDGETS: &[Budget] = &[
    Budget {
        label: "WASM web bundle",
        pattern: "crates/semiflow-wasm/pkg-web/semiflow_wasm_bg.wasm",
        limit_bytes: 500 * 1024,
    },
    Budget {
        label: "WASM node bundle",
        pattern: "crates/semiflow-wasm/pkg-node/semiflow_wasm_bg.wasm",
        limit_bytes: 500 * 1024,
    },
    Budget {
        label: "FFI cdylib (release-ffi, Linux)",
        pattern: "target/release-ffi/libsemiflow_ffi.so",
        // Operator-zoo bindings grew the stripped cdylib to ~1.24 MB; 1.5 MB keeps
        // a meaningful ceiling to catch future bloat while reflecting actual size.
        limit_bytes: 1536 * 1024,
    },
    Budget {
        label: "FFI cdylib (release-ffi, macOS)",
        pattern: "target/release-ffi/libsemiflow_ffi.dylib",
        // Same rationale as Linux entry above.
        limit_bytes: 1536 * 1024,
    },
    Budget {
        label: "PyO3 cdylib (release-ffi, Linux)",
        pattern: "target/release-ffi/libsemiflow_py.so",
        limit_bytes: 5 * 1024 * 1024,
    },
    Budget {
        label: "PyO3 cdylib (release-ffi, macOS)",
        pattern: "target/release-ffi/libsemiflow_py.dylib",
        limit_bytes: 5 * 1024 * 1024,
    },
];

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Run the binary size check.  Returns `Ok` if all present artefacts are
/// within budget; returns `Err` if any present artefact exceeds its limit.
pub fn binary_size_check() -> Result<()> {
    let root = crate::workspace_root()?;
    let mut found = 0usize;
    let mut violations = 0usize;

    for b in BUDGETS {
        let path = root.join(b.pattern);
        match measure(&path) {
            None => {
                println!("  SKIP     {:<38}  (not built)", b.label);
            }
            Some(sz) => {
                found += 1;
                let ok = sz <= b.limit_bytes;
                let mark = if ok { "OK  " } else { "FAIL" };
                let pct = (sz as f64 / b.limit_bytes as f64 * 100.0) as u64;
                println!(
                    "  {mark}     {:<38}  {} / {} ({pct}%)",
                    b.label,
                    Bytes(sz),
                    Bytes(b.limit_bytes)
                );
                if !ok {
                    violations += 1;
                }
            }
        }
    }

    if found == 0 {
        println!(
            "binary-size-check: no artefacts found on disk — \
             build with wasm-build, ffi-build, py-build first"
        );
        return Ok(());
    }

    if violations > 0 {
        anyhow::bail!(
            "binary-size-check: {violations} artefact(s) exceed size budget — \
             try building with [profile.release-wasm] (`cargo xtask wasm-build --size`)"
        );
    }

    println!("binary-size-check: PASS ({found} artefact(s) checked, 0 violations)");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return file size in bytes, or `None` if the path does not exist.
fn measure(path: &Path) -> Option<u64> {
    fs::metadata(path).ok().map(|m| m.len())
}

/// Display helper: pretty-print byte counts (B / KB / MB).
struct Bytes(u64);

impl fmt::Display for Bytes {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let n = self.0;
        if n < 1024 {
            write!(f, "{n} B")
        } else if n < 1024 * 1024 {
            write!(f, "{:.1} KB", n as f64 / 1024.0)
        } else {
            write!(f, "{:.2} MB", n as f64 / (1024.0 * 1024.0))
        }
    }
}
