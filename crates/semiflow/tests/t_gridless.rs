//! `T_GRIDLESS` — sympy oracle for Gridless particle-ensemble Chernoff evolver (math §50).
//!
//! Wires `scripts/gridless_kit.py` into the `xtask test-fast` sympy sweep
//! alongside `T_RESOLVENT_JUMP_ND`, `T_OBSTACLE_GAMMA`, `T_REVERSE_TRANSPOSE`.
//! Invokes the Python oracle via `std::process::Command` and asserts the
//! output contains `T_GRIDLESS PASS`.
//!
//! Gate (NORMATIVE, ADR-0155, math §50.6):
//!   3/3 sub-checks: `push_forward_exactness`, `mass_conservation`,
//!   `voronoi_moment_match`. All must pass.
//!
//! ## Skip condition
//!
//! Skipped if `SKIP_PYTHON_ORACLE=1`, python3 unavailable, or sympy not
//! importable — matching the conventions of `T_RESOLVENT_JUMP_ND`.

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(clippy::manual_let_else, clippy::too_many_lines)]

use std::{env, process};

/// Path to the oracle kit relative to the workspace root.
const KIT: &str = "scripts/gridless_kit.py";
/// Expected success marker in the kit's stdout.
const PASS_MARKER: &str = "T_GRIDLESS PASS";

/// Walk up from `CARGO_MANIFEST_DIR` until Cargo.lock is found.
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

/// `T_GRIDLESS`: run the particle-ensemble push-forward oracle and check PASS marker.
///
/// 3 sub-checks: `push_forward_exactness`, `mass_conservation`, `voronoi_moment_match`.
/// All must pass (ADR-0155, math §50.6). `RELEASE_BLOCKING` for v9.0.0+.
#[test]
fn t_gridless() {
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_GRIDLESS: skipped (SKIP_PYTHON_ORACLE=1)");
        return;
    }

    let python_check = process::Command::new("python3").arg("--version").output();
    if python_check.is_err() {
        eprintln!("T_GRIDLESS: skipped (python3 not on PATH)");
        return;
    }

    let root = if let Some(r) = workspace_root() {
        r
    } else {
        eprintln!("T_GRIDLESS: skipped (workspace root not found)");
        return;
    };

    let kit_path = root.join(KIT);
    if !kit_path.exists() {
        eprintln!(
            "T_GRIDLESS: skipped (kit not found at {})",
            kit_path.display()
        );
        return;
    }

    // Check sympy importable.
    let import_check = process::Command::new("python3")
        .args(["-c", "import sympy"])
        .output();
    if import_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_GRIDLESS: skipped (sympy not importable)");
        return;
    }

    let out = process::Command::new("python3")
        .arg(kit_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_GRIDLESS: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_GRIDLESS stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_GRIDLESS stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_GRIDLESS: python3 {KIT} exited with status {}. stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_GRIDLESS: expected '{PASS_MARKER}' in output. Got:\n{stdout}"
    );

    println!("T_GRIDLESS PASS");
}
