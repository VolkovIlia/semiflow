//! `T_REVERSE_TRANSPOSE` — sympy oracle for reverse-mode AD transpose over the
//! Chernoff product (math §51/§51.9, extends §42 `T_MAGNUS_TRANSPOSE` to full trajectory).
//!
//! Wires `scripts/reverse_transpose_kit.py` into the `xtask test-fast` sympy sweep
//! alongside `T_GRIDLESS`, `T_RESOLVENT_JUMP_ND`, `T_OBSTACLE_GAMMA`.
//! Invokes the Python oracle via `std::process::Command` and asserts the
//! output contains `T_REVERSE_TRANSPOSE PASS (3/3`.
//!
//! Gate (NORMATIVE, ADR-0156 Amendment 1, math §51.6/§51.9):
//!   3/3 sub-checks: `product_transpose_factorisation`, `per_step_transpose_structure`,
//!   `vjp_adjoint_identity`.  All must pass.
//!
//! ## Skip condition
//!
//! Skipped if `SKIP_PYTHON_ORACLE=1`, python3 unavailable, or sympy not
//! importable — matching the conventions of `T_RESOLVENT_JUMP_ND`.

// Integration test/example: allows for numerical patterns.
#![allow(clippy::manual_let_else, clippy::too_many_lines)]

use std::{env, process};

/// Path to the oracle kit relative to the workspace root.
const KIT: &str = "scripts/reverse_transpose_kit.py";
/// Expected success marker in the kit's stdout.
/// Must contain "(3/3" to guard against a 2/2 partial pass from an older oracle.
const PASS_MARKER: &str = "T_REVERSE_TRANSPOSE PASS (3/3";

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

/// `T_REVERSE_TRANSPOSE`: run the product-transpose / per-step / VJP-adjoint oracle.
///
/// 3 sub-checks: `product_transpose_factorisation`, `per_step_transpose_structure`,
/// `vjp_adjoint_identity`.  All must pass (ADR-0156 Amendment 1, math §51.6/§51.9).
/// `RELEASE_BLOCKING` for v9.1.0+.  Fails if oracle reports fewer than 3/3 sub-checks.
#[test]
fn t_reverse_transpose() {
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_REVERSE_TRANSPOSE: skipped (SKIP_PYTHON_ORACLE=1)");
        return;
    }

    let python_check = process::Command::new("python3").arg("--version").output();
    if python_check.is_err() {
        eprintln!("T_REVERSE_TRANSPOSE: skipped (python3 not on PATH)");
        return;
    }

    let root = if let Some(r) = workspace_root() {
        r
    } else {
        eprintln!("T_REVERSE_TRANSPOSE: skipped (workspace root not found)");
        return;
    };

    let kit_path = root.join(KIT);
    if !kit_path.exists() {
        eprintln!(
            "T_REVERSE_TRANSPOSE: skipped (kit not found at {})",
            kit_path.display()
        );
        return;
    }

    // Check sympy importable.
    let import_check = process::Command::new("python3")
        .args(["-c", "import sympy"])
        .output();
    if import_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_REVERSE_TRANSPOSE: skipped (sympy not importable)");
        return;
    }

    let out = process::Command::new("python3")
        .arg(kit_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_REVERSE_TRANSPOSE: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_REVERSE_TRANSPOSE stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_REVERSE_TRANSPOSE stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_REVERSE_TRANSPOSE: python3 {KIT} exited with status {}. stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_REVERSE_TRANSPOSE: expected '{PASS_MARKER}' in output. Got:\n{stdout}"
    );

    println!("T_REVERSE_TRANSPOSE PASS");
}
