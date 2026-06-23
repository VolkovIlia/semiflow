//! `T_RESOLVENT_JUMP_ND` — sympy/numeric oracle for F2 `ResolventJump` 2D/3D.
//!
//! Wires `scripts/resolvent_jump_2d3d_kit.py` into the `xtask test-fast` sympy
//! sweep (alongside `T_RESOLVENT_JUMP`, `T_HORM`, `T_QG`). Invokes the Python oracle
//! via `std::process::Command` and asserts the output contains
//! `T_RESOLVENT_JUMP_ND PASS`.
//!
//! Gate (NORMATIVE, ADR-0148, §47.8):
//!   PART A 5/5 PASS: `nd_resolvent_exact`, `nd_geometric_decay`, `nd_order_slope`,
//!   `nd_t_independence`, `three_d_smoke`.
//!   PART B is honestly DEFERRED (hyperbolic contour); its DEFER verdict does NOT
//!   block this gate — only PART A is required for v8.2.0.
//!
//! ## Skip condition
//!
//! If `python3` is not on PATH or numpy/scipy are not installed the test is
//! SKIPPED (not FAILED), matching the conventions for Python-dependent oracle
//! checks in this test suite. Set `SKIP_PYTHON_ORACLE=1` to force skip.

// Integration test/bench: allows for numerical patterns.
#![allow(clippy::manual_let_else, clippy::too_many_lines)]

use std::{env, process};

/// Path to the oracle kit relative to the workspace root.
const KIT: &str = "scripts/resolvent_jump_2d3d_kit.py";
/// Expected success marker in the kit's stdout.
const PASS_MARKER: &str = "T_RESOLVENT_JUMP_ND PASS";

/// Locate the workspace root: walk up from `CARGO_MANIFEST_DIR` until Cargo.lock.
fn workspace_root() -> Option<std::path::PathBuf> {
    let start = std::path::PathBuf::from(
        env::var("CARGO_MANIFEST_DIR").unwrap_or_else(|_| ".".to_string()),
    );
    let mut dir = start.as_path();
    loop {
        if dir.join("Cargo.lock").exists() {
            return Some(dir.to_path_buf());
        }
        dir = dir.parent()?;
    }
}

/// `T_RESOLVENT_JUMP_ND`: run the 2D/3D parabolic oracle and check PASS marker.
///
/// Skipped if `SKIP_PYTHON_ORACLE=1` or python3/scipy unavailable.
#[test]
fn t_resolvent_jump_nd() {
    // Allow CI / offline environments to skip cleanly.
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_RESOLVENT_JUMP_ND: skipped (SKIP_PYTHON_ORACLE=1)");
        return;
    }

    // Check python3 available.
    let python_check = process::Command::new("python3").arg("--version").output();
    if python_check.is_err() {
        eprintln!("T_RESOLVENT_JUMP_ND: skipped (python3 not on PATH)");
        return;
    }

    let root = if let Some(r) = workspace_root() {
        r
    } else {
        eprintln!("T_RESOLVENT_JUMP_ND: skipped (workspace root not found)");
        return;
    };

    let kit_path = root.join(KIT);
    if !kit_path.exists() {
        eprintln!(
            "T_RESOLVENT_JUMP_ND: skipped (kit not found at {})",
            kit_path.display()
        );
        return;
    }

    // Check scipy importable (fast pre-check before the heavy run).
    let scipy_check = process::Command::new("python3")
        .args(["-c", "import scipy, numpy"])
        .output();
    if scipy_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_RESOLVENT_JUMP_ND: skipped (numpy/scipy not importable)");
        return;
    }

    // Run the oracle.
    let out = process::Command::new("python3")
        .arg(kit_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_RESOLVENT_JUMP_ND: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_RESOLVENT_JUMP_ND stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_RESOLVENT_JUMP_ND stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_RESOLVENT_JUMP_ND: python3 {KIT} exited with status {}. \
         stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_RESOLVENT_JUMP_ND: expected '{PASS_MARKER}' in output. \
         Got:\n{stdout}"
    );

    println!("T_RESOLVENT_JUMP_ND PASS");
}
