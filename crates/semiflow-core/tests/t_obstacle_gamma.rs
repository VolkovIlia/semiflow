//! `T_OBSTACLE_GAMMA` — sympy/numpy oracle for inactive-set Γ (math §44.5.bis).
//!
//! Wires `scripts/obstacle_gamma_kit.py` into the `xtask test-fast` sweep
//! alongside `T_OBSTACLE_PROJECTION`, `T_RESOLVENT_JUMP`, `T_WENTZELL`.
//! Invokes the Python oracle via `std::process::Command` and asserts the
//! output contains `T_OBSTACLE_GAMMA PASS`.
//!
//! Gate (NORMATIVE, ADR-0152, §44.5.bis):
//!   5/5 sub-checks: `closed_form`, `gamma_jump`, `inactive_restrict`, `mollified_eps`,
//!   `d2_mechanical`. All must pass.
//!
//! ## Skip condition
//!
//! Skipped if `SKIP_PYTHON_ORACLE=1`, python3 unavailable, or numpy/sympy not
//! importable — matching the conventions of `T_RESOLVENT_JUMP_ND`.

// Integration test/example: allows for numerical patterns.
#![allow(clippy::manual_let_else, clippy::too_many_lines)]

use std::{env, process};

/// Path to the oracle kit relative to the workspace root.
const KIT: &str = "scripts/obstacle_gamma_kit.py";
/// Expected success marker in the kit's stdout.
const PASS_MARKER: &str = "OVERALL VERDICT: GO";

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

/// `T_OBSTACLE_GAMMA`: run the inactive-set Γ oracle and check PASS marker.
///
/// 5 sub-checks: `closed_form`, `gamma_jump`, `inactive_restrict`, `mollified_eps`,
/// `d2_mechanical`. All must pass (ADR-0152, math §44.5.bis).
#[test]
fn t_obstacle_gamma() {
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_OBSTACLE_GAMMA: skipped (SKIP_PYTHON_ORACLE=1)");
        return;
    }

    let python_check = process::Command::new("python3").arg("--version").output();
    if python_check.is_err() {
        eprintln!("T_OBSTACLE_GAMMA: skipped (python3 not on PATH)");
        return;
    }

    let root = if let Some(r) = workspace_root() {
        r
    } else {
        eprintln!("T_OBSTACLE_GAMMA: skipped (workspace root not found)");
        return;
    };

    let kit_path = root.join(KIT);
    if !kit_path.exists() {
        eprintln!(
            "T_OBSTACLE_GAMMA: skipped (kit not found at {})",
            kit_path.display()
        );
        return;
    }

    // Check numpy + sympy importable.
    let import_check = process::Command::new("python3")
        .args(["-c", "import numpy, sympy"])
        .output();
    if import_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_OBSTACLE_GAMMA: skipped (numpy/sympy not importable)");
        return;
    }

    let out = process::Command::new("python3")
        .arg(kit_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_OBSTACLE_GAMMA: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_OBSTACLE_GAMMA stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_OBSTACLE_GAMMA stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_OBSTACLE_GAMMA: python3 {KIT} exited with status {}. stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_OBSTACLE_GAMMA: expected '{PASS_MARKER}' in output. Got:\n{stdout}"
    );

    println!("T_OBSTACLE_GAMMA PASS");
}
