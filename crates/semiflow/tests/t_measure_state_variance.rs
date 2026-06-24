//! `T_GRIDLESS_VARIANCE` — sympy oracle for `MeasureState` variance diagnostic (§38.12).
//!
//! Wires `scripts/verify_measure_state_variance.py` into the `xtask test-fast`
//! sympy sweep alongside `T_GRIDLESS`, `T_ADJOINT_FP_TIGHTNESS`.
//! Invokes the Python oracle via `std::process::Command` and asserts the
//! output contains `T_GRIDLESS_VARIANCE PASS`.
//!
//! Gate (NORMATIVE, math §38.12):
//!   4/4 sub-checks: `closed_form_var`, `per_axis_var`,
//!   `var_equals_sum_per_axis`, `zero_mass_guard`. All must pass.
//!
//! ## Skip condition
//!
//! Skipped if `SKIP_PYTHON_ORACLE=1`, python3 unavailable, or sympy not
//! importable — matching the conventions of `T_GRIDLESS`.

use std::{env, process};

const SCRIPT: &str = "scripts/verify_measure_state_variance.py";
const PASS_MARKER: &str = "T_GRIDLESS_VARIANCE PASS";

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

/// Check all skip conditions; return `Some(script_path)` or skip.
fn check_preconditions() -> Option<(std::path::PathBuf, std::path::PathBuf)> {
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_GRIDLESS_VARIANCE: skipped (SKIP_PYTHON_ORACLE=1)");
        return None;
    }
    if process::Command::new("python3")
        .arg("--version")
        .output()
        .is_err()
    {
        eprintln!("T_GRIDLESS_VARIANCE: skipped (python3 not on PATH)");
        return None;
    }
    let root = workspace_root()?;
    let script_path = root.join(SCRIPT);
    if !script_path.exists() {
        eprintln!(
            "T_GRIDLESS_VARIANCE: skipped (script not found at {})",
            script_path.display()
        );
        return None;
    }
    let import_check = process::Command::new("python3")
        .args(["-c", "import sympy"])
        .output();
    if import_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_GRIDLESS_VARIANCE: skipped (sympy not importable)");
        return None;
    }
    Some((root, script_path))
}

#[test]
fn t_gridless_variance() {
    let Some((root, script_path)) = check_preconditions() else {
        return;
    };

    let out = process::Command::new("python3")
        .arg(script_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_GRIDLESS_VARIANCE: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_GRIDLESS_VARIANCE stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_GRIDLESS_VARIANCE stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_GRIDLESS_VARIANCE: python3 {SCRIPT} exited with status {}. stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_GRIDLESS_VARIANCE: expected '{PASS_MARKER}' in output. Got:\n{stdout}"
    );

    println!("T_GRIDLESS_VARIANCE PASS");
}
