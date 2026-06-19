//! `T_TT_COUPLED_RANK` — sympy+numeric oracle for the per-step TT-rank bound of
//! the genuine coupled Chernoff evolver (math §52.9, ADR-0159 Amendment 1).
//!
//! Wires `scripts/tt_coupled_kit.py` into the `xtask test-fast` sympy sweep
//! alongside `T_GRIDLESS`, `T_RESOLVENT_JUMP_ND`, `T_OBSTACLE_GAMMA`, `T_REVERSE_TRANSPOSE`.
//! Invokes the Python oracle via `std::process::Command` and asserts the
//! output contains `T_TT_COUPLED_RANK PASS (3/3`.
//!
//! Gate (NORMATIVE, `RELEASE_BLOCKING`, ADR-0159 Amendment 1, math §52.9):
//!   3/3 sub-checks:
//!     (a) `pair_operator_mode_rank`  — D1⊗D1 TT-op bond rank = 1; (I+τρ·D1⊗D1) ≤ 2.
//!     (b) `pre_round_rank_growth`    — one coupling pair inflates rank by ≤ min(2r,n) pre-round;
//!                                    r≤2 case satisfies the §52.9 additive bound r+2.
//!     (c) `post_round_rank_analytic` — full evolver (40 Euler steps) post-round rank bounded ≤ n;
//!                                    O(1) for local coupling, ≤ n for dense coupling.
//!   All must pass. Fails if oracle reports fewer than 3/3 or prints FAIL.
//!
//! This oracle closes the §52.9 audit gap: `g_gridless_ttrank` measured analytic rank
//! but NEVER instantiated an evolver. `T_TT_COUPLED_RANK` proves the rank bound for the
//! EVOLVER operator (`D1_j⊗D1_k`), not just the analytic precision-matrix picture.
//!
//! ## Skip condition
//!
//! Skipped if `SKIP_PYTHON_ORACLE=1`, python3 unavailable, or sympy/numpy not
//! importable — matching the conventions of `T_REVERSE_TRANSPOSE` / `T_GRIDLESS`.

// Integration test/example: allows for numerical patterns.
#![allow(clippy::manual_let_else, clippy::too_many_lines)]

use std::{env, process};

/// Path to the oracle kit relative to the workspace root.
const KIT: &str = "scripts/tt_coupled_kit.py";

/// Expected success marker in the kit's stdout.
/// Must contain "(3/3" to guard against a 2/2 partial pass from an older oracle.
const PASS_MARKER: &str = "T_TT_COUPLED_RANK PASS (3/3";

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

/// `T_TT_COUPLED_RANK`: run the per-step rank-bound oracle for `CoupledTtChernoff`.
///
/// 3 sub-checks:
///   (a) `pair_operator_mode_rank`  — TT-operator bond rank ≤ 2 for I+τρ·D1⊗D1.
///   (b) `pre_round_rank_growth`    — pre-round rank bounded by min(2r, n); r+2 for r ≤ 2.
///   (c) `post_round_rank_analytic` — post-round rank bounded (not 4^n); O(1) local, ≤ n dense.
///
/// NORMATIVE `RELEASE_BLOCKING` for v9.1.0 (math §52.9, ADR-0159 Amendment 1).
/// Fails if oracle reports fewer than 3/3 sub-checks or exits non-zero.
#[test]
fn t_tt_coupled_rank() {
    if env::var("SKIP_PYTHON_ORACLE").as_deref() == Ok("1") {
        eprintln!("T_TT_COUPLED_RANK: skipped (SKIP_PYTHON_ORACLE=1)");
        return;
    }

    let python_check = process::Command::new("python3").arg("--version").output();
    if python_check.is_err() {
        eprintln!("T_TT_COUPLED_RANK: skipped (python3 not on PATH)");
        return;
    }

    let root = if let Some(r) = workspace_root() {
        r
    } else {
        eprintln!("T_TT_COUPLED_RANK: skipped (workspace root not found)");
        return;
    };

    let kit_path = root.join(KIT);
    if !kit_path.exists() {
        eprintln!(
            "T_TT_COUPLED_RANK: skipped (kit not found at {})",
            kit_path.display()
        );
        return;
    }

    // Check sympy importable.
    let import_check = process::Command::new("python3")
        .args(["-c", "import sympy; import numpy"])
        .output();
    if import_check.map_or(true, |o| !o.status.success()) {
        eprintln!("T_TT_COUPLED_RANK: skipped (sympy or numpy not importable)");
        return;
    }

    let out = process::Command::new("python3")
        .arg(kit_path.as_os_str())
        .current_dir(&root)
        .output()
        .expect("T_TT_COUPLED_RANK: failed to spawn python3");

    let stdout = String::from_utf8_lossy(&out.stdout);
    let stderr = String::from_utf8_lossy(&out.stderr);

    println!("T_TT_COUPLED_RANK stdout:\n{stdout}");
    if !stderr.is_empty() {
        eprintln!("T_TT_COUPLED_RANK stderr:\n{stderr}");
    }

    assert!(
        out.status.success(),
        "T_TT_COUPLED_RANK: python3 {KIT} exited with status {}. stderr: {stderr}",
        out.status
    );
    assert!(
        stdout.contains(PASS_MARKER),
        "T_TT_COUPLED_RANK: expected '{PASS_MARKER}' in output. Got:\n{stdout}"
    );

    println!("T_TT_COUPLED_RANK PASS");
}
