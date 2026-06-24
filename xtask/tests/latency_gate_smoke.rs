//! Smoke tests for `xtask latency-gate` using mock JSONL fixture.
//!
//! These tests avoid running the full `latency_tail` bench (1M-tick, ~30s).
//! Instead, the `--mock-input` flag substitutes the bench output with a
//! pre-recorded fixture file from `xtask/tests/fixtures/`.
//!
//! ADR-0068 Track 2 — L-gate advisory harness (v2.6).

use std::{path::PathBuf, process::Command};

/// Resolve the workspace root by walking up from the manifest dir.
fn workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points to `xtask/` — parent is workspace root.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .parent()
        .expect("xtask has parent")
        .to_path_buf()
}

fn fixture(name: &str) -> PathBuf {
    // CARGO_MANIFEST_DIR is `xtask/`; fixtures are under `xtask/tests/fixtures/`.
    let manifest = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .join("tests")
        .join("fixtures")
        .join(name)
}

/// Run `cargo run -p xtask -- latency-gate <args>` and return (stdout, stderr, exit_code).
fn run_latency_gate(args: &[&str]) -> (String, String, i32) {
    let root = workspace_root();
    let mut cmd = Command::new("cargo");
    cmd.arg("run")
        .arg("-p")
        .arg("xtask")
        .arg("--")
        .arg("latency-gate");
    for a in args {
        cmd.arg(a);
    }
    cmd.current_dir(&root);

    let out = cmd.output().expect("spawn cargo run -p xtask");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    let stderr = String::from_utf8_lossy(&out.stderr).into_owned();
    let code = out.status.code().unwrap_or(-1);
    (stdout, stderr, code)
}

#[test]
fn smoke_mock_all_pass_exits_zero() {
    // The mock fixture has all metrics at or below the i7-12700K floors.
    let fix = fixture("lcev_ptick_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (stdout, stderr, code) = run_latency_gate(&[
        "L_CEV_PTICK",
        "--hardware-profile",
        "i7-12700K",
        "--mock-input",
        &fix_str,
    ]);

    assert_eq!(code, 0, "exit code must be 0 (advisory); stderr: {stderr}");
    // All 4 metrics should PASS since mock values are at or below the floors.
    let pass_count = stdout.lines().filter(|l| l.contains("L-GATE PASS")).count();
    assert!(
        pass_count >= 1,
        "expected at least one PASS line; stdout: {stdout}"
    );
}

#[test]
fn smoke_mock_advisory_profile_skips() {
    // m2-pro is an advisory placeholder (p999: null) — should skip gracefully.
    let fix = fixture("lcev_ptick_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (_stdout, stderr, code) = run_latency_gate(&[
        "L_CEV_PTICK",
        "--hardware-profile",
        "m2-pro",
        "--mock-input",
        &fix_str,
    ]);

    assert_eq!(code, 0, "advisory profile must exit 0; stderr: {stderr}");
}

#[test]
fn smoke_unknown_gate_exits_nonzero() {
    let (_, _, code) = run_latency_gate(&["L_DOES_NOT_EXIST"]);
    assert_ne!(code, 0, "unknown gate must exit non-zero");
}

#[test]
fn smoke_help_exits_zero() {
    let (stdout, _, code) = run_latency_gate(&["--help"]);
    assert_eq!(code, 0);
    assert!(
        stdout.contains("latency-gate"),
        "help should mention 'latency-gate'"
    );
}

#[test]
fn smoke_all_flag_mock() {
    // --all runs all gates; only L_CEV_PTICK has real floors for i7-12700K;
    // L_RESOLVENT_N64_P99 is a TBD placeholder — should emit SKIP.
    let fix = fixture("lcev_ptick_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (_stdout, _stderr, code) = run_latency_gate(&[
        "--all",
        "--hardware-profile",
        "i7-12700K",
        "--mock-input",
        &fix_str,
    ]);

    assert_eq!(code, 0, "advisory mode must always exit 0");
}

// ---------------------------------------------------------------------------
// v2.7 blocking-semantics integration tests (ADR-0069 + math.md §3.6.bis.7)
//
// These tests exercise the binary-level behavior of the advisory/blocking
// enforcement introduced in v2.7. They use the real properties.yaml and
// substitute bench output with mock fixture files.
//
// NOTE: L_CEV_PTICK (advisory=false in properties.yaml) still has
// severity=ADVISORY (not RELEASE_BLOCKING), so it never triggers exit 1.
// L_RESOLVENT_N64_P99 has severity=RELEASE_BLOCKING but advisory=true (rc.1).
// The unit tests in xtask/src/latency_gate.rs cover the blocking code path
// directly via check_floors(). These integration tests verify the binary-level
// advisory plumbing remains backward-compatible.
// ---------------------------------------------------------------------------

/// L_RESOLVENT_N64_P99 is RELEASE_BLOCKING but advisory=true in rc.1 — mock
/// breach must exit 0 (warn-only), not exit 1.
#[test]
fn resolvent_advisory_true_breach_exits_zero() {
    let fix = fixture("blocking_breach_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (_stdout, stderr, code) = run_latency_gate(&[
        "L_RESOLVENT_N64_P99",
        "--hardware-profile",
        "i7-12700K",
        "--mock-input",
        &fix_str,
    ]);

    // advisory=true in rc.1 → exit 0 even with RELEASE_BLOCKING + breach.
    assert_eq!(
        code, 0,
        "RELEASE_BLOCKING + advisory=true must exit 0; stderr: {stderr}"
    );
}

/// L_RESOLVENT_N64_P99 with values within budget exits zero cleanly.
#[test]
fn resolvent_within_budget_exits_zero() {
    let fix = fixture("within_budget_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (_stdout, _stderr, code) = run_latency_gate(&[
        "L_RESOLVENT_N64_P99",
        "--hardware-profile",
        "i7-12700K",
        "--mock-input",
        &fix_str,
    ]);

    // No floors set yet (null TBD-rc.1) → SKIP or PASS; always exit 0.
    assert_eq!(code, 0, "no floors configured → must exit 0");
}

/// L_HESTON_PTICK is ADVISORY severity throughout v2.7 — any breach exits 0.
#[test]
fn heston_advisory_severity_breach_exits_zero() {
    let fix = fixture("blocking_breach_mock.jsonl");
    let fix_str = fix.to_string_lossy();

    let (_stdout, stderr, code) = run_latency_gate(&[
        "L_HESTON_PTICK",
        "--hardware-profile",
        "i7-12700K",
        "--mock-input",
        &fix_str,
    ]);

    // severity=ADVISORY → exit 0 regardless of breach or advisory flag.
    assert_eq!(
        code, 0,
        "ADVISORY severity must always exit 0; stderr: {stderr}"
    );
}
