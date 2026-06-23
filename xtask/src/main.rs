//! xtask — build/lint helpers for semiflow workspace.
//!
//! Subcommands:
//!   check-lints        — walk `crates/*/src/**/*.rs`, check ≤50 lines/fn, ≤500 lines/file
//!                        optional: `--crate NAME` to scope to one crate
//!   check-unsafe-scope — enforce that `unsafe` appears only in allowed files (ADR-0019)
//!   gen-stubs          — emit placeholder Rust stubs from contracts/semiflow-core.traits.yaml
//!   bench-baseline     — run cargo bench and capture criterion baseline
//!   bench-parallel     — run the 4 parallel benches with features parallel,simd (ADR-0060)
//!   test-fast          — cargo test --workspace --features parallel,simd (multi-core + SIMD)
//!   test-full          — RUSTFLAGS="-C target-cpu=native" cargo test --workspace
//!                        --features parallel,simd,slow-tests --release
//!   test-flagship      — same flags as test-full but runs only 3 named binaries
//!   test-ignored-gates — same flags as test-full but passes `-- --ignored`
//!                        (runs all heavy RELEASE_BLOCKING `#[ignore]` tests)
//!   ffi-headers        — generate crates/semiflow-ffi/include/semiflow.h via cbindgen
//!   ffi-smoke          — build cdylib, compile heat.c, run C smoke test
//!   ffi-graph-smoke    — build cdylib, compile graph_heat.c, run Graph PDE smoke test
//!   py-build           — build semiflow-py wheel via maturin (--profile release-ffi)
//!   py-bench           — cargo bench -p semiflow-py --profile release-ffi (ADR-0031)
//!   py-smoke           — create venv, maturin develop, run pytest smoke suite
//!   py-graph-smoke     — create venv, maturin develop, run pytest smoke_graph.py (ADR-0059)
//!   wasm-build         — build semiflow-wasm for both `web` and `nodejs` targets
//!                        pass `--size` to use [profile.release-wasm] (opt-level=z, D2)
//!   wasm-test          — run wasm-bindgen-test smoke (Node by default, --chrome for headless Chrome)
//!   wasm-graph-smoke   — build semiflow-wasm (nodejs) + run graph PDE gate (G_WASM_smoke_graph, ADR-0059)
//!   wasm-pack-npm      — build semiflow-wasm with TS decls and assemble publishable dist/npm/ package
//!   binary-size-check  — report sizes of built binding artefacts vs targets (D3, v0.13.0)
//!   latency-gate       — run L-gate latency harness from contracts/semiflow-core.properties.yaml
//!                        (ADR-0068 Track 2; advisory in v2.6, blocking in v2.7)

use std::{
    env,
    path::{Path, PathBuf},
    process,
};

use anyhow::{bail, Result};

mod ffi_tasks;
mod latency_gate;
mod py_tasks;
mod size_check;
mod unsafe_scope;
mod wasm_npm;
mod wasm_tasks;

fn main() {
    let mut args = env::args().skip(1);
    let cmd = match args.next() {
        Some(c) => c,
        None => {
            eprintln!(
                "Usage: cargo xtask \
                 <check-lints|check-unsafe-scope|gen-stubs|bench-baseline|\
                 test-fast|test-full|test-flagship|test-ignored-gates|\
                 ffi-headers|ffi-smoke|ffi-graph-smoke|\
                 py-build|py-bench|py-smoke|py-graph-smoke|\
                 wasm-build [--size]|wasm-test|wasm-graph-smoke|wasm-pack-npm|\
                 binary-size-check|latency-gate>"
            );
            process::exit(1);
        }
    };

    let rest: Vec<String> = args.collect();
    let result = match cmd.as_str() {
        "check-lints" => {
            // Optional `--crate <name>` to scope to a single crate directory.
            let scope = rest
                .iter()
                .position(|a| a == "--crate")
                .and_then(|i| rest.get(i + 1))
                .cloned();
            check_lints(scope.as_deref())
        }
        "check-unsafe-scope" => unsafe_scope::check_unsafe_scope(),
        "gen-stubs" => gen_stubs(),
        "bench-baseline" => bench_baseline(),
        "bench-parallel" => bench_parallel(),
        "test-fast" => test_fast(),
        "test-full" => test_full(),
        "test-flagship" => test_flagship(),
        "test-ignored-gates" => test_ignored_gates(),
        "ffi-headers" => ffi_tasks::ffi_headers(rest.contains(&"--check".to_owned())),
        "ffi-smoke" => ffi_tasks::ffi_smoke(),
        "ffi-graph-smoke" => ffi_tasks::ffi_graph_smoke(),
        "py-build" => py_tasks::py_build(),
        "py-bench" => py_tasks::py_bench(),
        "py-smoke" => py_tasks::py_smoke(),
        "py-graph-smoke" => py_tasks::py_graph_smoke(),
        "wasm-build" => wasm_tasks::wasm_build_with_args(&rest),
        "wasm-test" => wasm_tasks::wasm_test(&rest),
        "wasm-graph-smoke" => wasm_tasks::wasm_graph_smoke(),
        "wasm-pack-npm" => wasm_npm::wasm_pack_npm(),
        "binary-size-check" => size_check::binary_size_check(),
        "latency-gate" => latency_gate::run(&rest),
        other => {
            eprintln!("Unknown subcommand: {other}");
            eprintln!(
                "Available: check-lints, check-unsafe-scope, gen-stubs, \
                 bench-baseline, bench-parallel, \
                 test-fast, test-full, test-flagship, test-ignored-gates, \
                 ffi-headers, ffi-smoke, ffi-graph-smoke, \
                 py-build, py-bench, py-smoke, py-graph-smoke, \
                 wasm-build [--size], wasm-test, wasm-graph-smoke, wasm-pack-npm, \
                 binary-size-check, latency-gate"
            );
            process::exit(1);
        }
    };

    if let Err(e) = result {
        eprintln!("xtask error: {e:#}");
        process::exit(1);
    }
}

// ---------------------------------------------------------------------------
// check-lints: verify suckless line budgets (G10).
// ---------------------------------------------------------------------------

// Intentionally empty: all line-limit debt eliminated (tech-debt campaign, 2026-06).
// Override #1 retired (constitution v6.0.0); file cap restored to ENFORCED 500.
const GRANDFATHERED: &[&str] = &[];

// Intentionally empty: all function-length debt eliminated (tech-debt campaign, 2026-06).
// Override #1 retired (constitution v6.0.0); function cap restored to ENFORCED 50.
const GRANDFATHERED_FNS: &[(&str, usize)] = &[];

/// Walk `crates/*/src/**/*.rs` (excluding `tests/` and `examples/`) and
/// enforce ≤50 lines/function, ≤500 lines/file.
///
/// Pass `crate_filter = Some("semiflow-ffi")` to scope to one crate.
/// Test files (`crates/*/tests/`) and examples (`crates/*/examples/`) are
/// excluded: integration tests regularly require longer setup functions, and
/// the suckless budget applies to library source code.
fn check_lints(crate_filter: Option<&str>) -> Result<()> {
    use walkdir::WalkDir;

    let crates_dir = workspace_root()?.join("crates");
    let walk_root = if let Some(name) = crate_filter {
        crates_dir.join(name)
    } else {
        crates_dir.clone()
    };
    let mut violations: Vec<String> = Vec::new();
    let mut grandfathered: Vec<String> = Vec::new();

    for entry in WalkDir::new(&walk_root)
        .into_iter()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "rs"))
    {
        let path = entry.path();
        let rel_norm = path
            .strip_prefix(&crates_dir)
            .unwrap_or(path)
            .to_string_lossy()
            .replace('\\', "/");
        // Skip test and example directories (not subject to function budget).
        if rel_norm.contains("/tests/") || rel_norm.contains("/examples/") {
            continue;
        }
        let src = std::fs::read_to_string(path)?;
        let file_gf = GRANDFATHERED.iter().any(|p| rel_norm.starts_with(p));
        // File budget: route to grandfathered list when file is in Override #1.
        {
            let file_target = if file_gf {
                &mut grandfathered
            } else {
                &mut violations
            };
            check_file_budget(&rel_norm, &src, file_target);
        }
        // Function budgets: collect into temporaries then merge.
        // When the whole file is grandfathered, function findings are also notes.
        let mut fn_v: Vec<String> = Vec::new();
        let mut fn_gf: Vec<String> = Vec::new();
        check_function_budgets(&rel_norm, &src, &mut fn_v, &mut fn_gf);
        if file_gf {
            grandfathered.extend(fn_v);
        } else {
            violations.extend(fn_v);
        }
        grandfathered.extend(fn_gf);
    }

    for v in &grandfathered {
        eprintln!("NOTE (grandfathered per constitution Override #1): {v}");
    }

    if violations.is_empty() {
        if grandfathered.is_empty() {
            println!("check-lints: PASS — no suckless violations found");
        } else {
            println!(
                "check-lints: PASS — no new violations \
                 ({} grandfathered pre-existing)",
                grandfathered.len()
            );
        }
        Ok(())
    } else {
        for v in &violations {
            eprintln!("VIOLATION: {v}");
        }
        bail!(
            "{} suckless violation(s) found (+ {} grandfathered)",
            violations.len(),
            grandfathered.len()
        );
    }
}

/// Check that a file does not exceed 500 lines.
fn check_file_budget(rel: &str, src: &str, violations: &mut Vec<String>) {
    let n = src.lines().count();
    if n > 500 {
        violations.push(format!("{rel}: file has {n} lines (budget: 500)"));
    }
}

/// Heuristic function-line check: detect `fn ` and count brace depth.
///
/// Handles both single-line signatures (`fn foo(...) {`) and multi-line
/// signatures (`fn foo(\n    ...\n) {`) by scanning forward from the `fn`
/// keyword line to find the opening `{`.  The function body length is
/// measured from the `fn` keyword line to the matching closing `}`.
///
/// Functions annotated with `#[allow(clippy::too_many_lines)]` on the
/// immediately preceding non-blank line are routed to `fn_grandfathered`
/// rather than `violations`.  Entries in `GRANDFATHERED_FNS` are treated
/// the same way.
///
/// **Limitations**: detects top-level `fn` items and direct method items
/// only; macro-generated code and closures are not detected.
fn check_function_budgets(
    rel: &str,
    src: &str,
    violations: &mut Vec<String>,
    fn_grandfathered: &mut Vec<String>,
) {
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if !is_fn_keyword_line(trimmed) {
            i += 1;
            continue;
        }
        let fn_start = i;
        // Find the line that contains the opening `{` of the function body.
        // For single-line signatures this is `fn_start`; for multi-line
        // signatures it may be several lines further down.
        let brace_line = find_opening_brace(&lines, fn_start);
        let depth_start =
            count_char(lines[brace_line], '{') as i32 - count_char(lines[brace_line], '}') as i32;
        let fn_end = find_fn_end(&lines, brace_line, depth_start);
        let fn_lines = fn_end.saturating_sub(fn_start) + 1;
        if fn_lines > 50 {
            let msg = format!(
                "{rel}:{}: function has {fn_lines} lines (budget: 50)",
                fn_start + 1
            );
            // Path A: respect `#[allow(clippy::too_many_lines)]` on the
            // immediately preceding non-blank line before `fn_start`.
            let has_allow = (0..fn_start)
                .rev()
                .find(|&j| !lines[j].trim().is_empty())
                .map(|j| lines[j].trim().contains("allow(clippy::too_many_lines)"))
                .unwrap_or(false);
            // Path B: static function-level grandfather list.
            let in_gf_list = GRANDFATHERED_FNS
                .iter()
                .any(|&(p, ln)| rel.starts_with(p) && fn_start + 1 == ln);
            if has_allow || in_gf_list {
                fn_grandfathered.push(msg);
            } else {
                violations.push(msg);
            }
        }
        i = fn_end + 1;
    }
}

/// Returns true if this line begins a `fn` item (not a comment or use-statement).
///
/// Matches both single-line signatures (`fn foo() {`) and the opening line of
/// multi-line signatures (`fn foo(`).  Does NOT require `{` on the same line.
fn is_fn_keyword_line(trimmed: &str) -> bool {
    trimmed.contains("fn ") && !trimmed.starts_with("//") && !trimmed.starts_with("use ")
}

/// Walk forward from `start` to find the first line that contains `{`.
///
/// Returns `start` itself when the opening brace is on the `fn` line (single-line
/// signature).  For multi-line signatures, scans up to 40 continuation lines
/// before giving up (returns `start` as a safe fallback — will not count the fn).
fn find_opening_brace(lines: &[&str], start: usize) -> usize {
    // Fast path: brace already on the fn line.
    if lines[start].contains('{') {
        return start;
    }
    // Multi-line signature: scan forward to the line with `{`.
    for (j, line) in lines.iter().enumerate().skip(start + 1).take(39) {
        if line.contains('{') {
            return j;
        }
    }
    // Fallback: treat as single-line (won't fire a violation).
    start
}

/// Count occurrences of `ch` on a line.
fn count_char(line: &str, ch: char) -> usize {
    line.chars().filter(|&c| c == ch).count()
}

/// Walk lines from `start`, tracking brace depth, until the function closes.
fn find_fn_end(lines: &[&str], start: usize, initial_depth: i32) -> usize {
    let mut depth = initial_depth;
    let mut i = start + 1;

    while i < lines.len() {
        let open = count_char(lines[i], '{') as i32;
        let close = count_char(lines[i], '}') as i32;
        depth += open - close;
        if depth <= 0 {
            return i;
        }
        i += 1;
    }
    lines.len().saturating_sub(1)
}

// ---------------------------------------------------------------------------
// gen-stubs: emit placeholder from contracts/semiflow-core.traits.yaml.
// ---------------------------------------------------------------------------

/// Read the traits YAML and emit a placeholder skeleton.
///
/// Full codegen (YAML → Rust) is **NOT YET IMPLEMENTED** for v0.1.0.
/// The engineer (Stage 6) fills in actual implementations by hand from
/// `contracts/semiflow-core.traits.yaml`.
fn gen_stubs() -> Result<()> {
    let root = workspace_root()?;
    let contract = root.join("contracts").join("semiflow-core.traits.yaml");

    if !contract.exists() {
        bail!("Contract not found: {}", contract.display());
    }

    let stubs_dir = root
        .join("crates")
        .join("semiflow")
        .join("src")
        .join("_stubs");
    std::fs::create_dir_all(&stubs_dir)?;

    let placeholder = build_stub_text(&contract)?;
    let out = stubs_dir.join("contract_stubs.rs");
    std::fs::write(&out, placeholder)?;

    println!("gen-stubs: wrote placeholder to {}", out.display());
    eprintln!("NOTE: Full codegen NOT YET IMPLEMENTED for v0.1.0.");
    eprintln!("Engineer (Stage 6): implement stubs by hand from:");
    eprintln!("  {}", contract.display());
    Ok(())
}

/// Parse name: lines from the YAML and build a placeholder Rust source file.
fn build_stub_text(contract: &Path) -> Result<String> {
    let src = std::fs::read_to_string(contract)?;

    let mut trait_names: Vec<String> = Vec::new();
    let mut struct_names: Vec<String> = Vec::new();
    let mut in_traits = false;
    let mut in_structs = false;

    for line in src.lines() {
        let stripped = line.trim();
        if stripped == "traits:" {
            in_traits = true;
            in_structs = false;
        } else if stripped == "structs:" {
            in_structs = true;
            in_traits = false;
        } else if let Some(rest) = stripped.strip_prefix("- name: ") {
            let name = rest.trim().trim_matches('"').to_owned();
            if in_traits {
                trait_names.push(name);
            } else if in_structs {
                struct_names.push(name);
            }
        }
    }

    let mut out = String::new();
    out.push_str("// AUTO-GENERATED PLACEHOLDER — `cargo xtask gen-stubs`\n");
    out.push_str("// Full codegen NOT YET IMPLEMENTED (v0.1.0).\n");
    out.push_str("// Engineer (Stage 6): write implementations by hand from:\n");
    out.push_str("//   contracts/semiflow-core.traits.yaml\n\n");

    for t in &trait_names {
        out.push_str(&format!(
            "// TODO: trait {t} — see contracts/semiflow-core.traits.yaml\n"
        ));
        out.push_str(&format!(
            "// pub trait {t} {{ /* unimplemented!() */ }}\n\n"
        ));
    }

    for s in &struct_names {
        out.push_str(&format!(
            "// TODO: struct {s} — see contracts/semiflow-core.traits.yaml\n"
        ));
        out.push_str(&format!(
            "// pub struct {s} {{ /* unimplemented!() */ }}\n\n"
        ));
    }

    Ok(out)
}

// ---------------------------------------------------------------------------
// bench-baseline: run cargo bench and archive criterion output.
// ---------------------------------------------------------------------------

/// Run `cargo bench -p semiflow` and copy criterion output to bench/baseline.json.
fn bench_baseline() -> Result<()> {
    let root = workspace_root()?;

    let status = process::Command::new("cargo")
        .args(["bench", "-p", "semiflow"])
        .current_dir(&root)
        .status()?;

    if !status.success() {
        bail!("cargo bench failed");
    }

    let criterion_dir = root
        .join("target")
        .join("criterion")
        .join("heat_1d_placeholder")
        .join("base");

    let baseline_out = root.join("bench").join("baseline.json");

    if criterion_dir.exists() {
        std::fs::create_dir_all(baseline_out.parent().unwrap())?;
        let estimates = criterion_dir.join("estimates.json");
        if estimates.exists() {
            std::fs::copy(&estimates, &baseline_out)?;
            println!("bench-baseline: saved to {}", baseline_out.display());
        } else {
            eprintln!(
                "WARN: estimates.json not found at {}",
                criterion_dir.display()
            );
        }
    } else {
        eprintln!("WARN: criterion output dir not found — benches may not be implemented yet");
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// bench-parallel: run the 4 thread-scaling benches (ADR-0060).
// ---------------------------------------------------------------------------

/// Run the four parallel thread-scaling criterion benches.
///
/// Equivalent to:
/// ```text
/// RUSTFLAGS="-C target-cpu=native" cargo bench -p semiflow \
///   --features parallel,simd \
///   --bench strang2d_parallel \
///   --bench strang3d_parallel \
///   --bench ns2d_aniso_parallel \
///   --bench graph_heat
/// ```
///
/// Pass `-- --quick` at the end to run each bench for ~1 s instead of 10 s:
/// ```sh
/// cargo xtask bench-parallel -- --quick
/// ```
///
/// Results land in `target/criterion/` per-bench subdirectories.
fn bench_parallel() -> Result<()> {
    let root = workspace_root()?;
    let cmd_args = [
        "bench",
        "-p",
        "semiflow",
        "--features",
        "parallel,simd",
        "--bench",
        "strang2d_parallel",
        "--bench",
        "strang3d_parallel",
        "--bench",
        "ns2d_aniso_parallel",
        "--bench",
        "graph_heat",
    ];
    eprintln!(
        "$ RUSTFLAGS=\"-C target-cpu=native\" cargo {}",
        cmd_args.join(" ")
    );
    let status = process::Command::new("cargo")
        .args(cmd_args)
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!(
            "bench-parallel failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// test-fast: cargo test --workspace --features parallel,simd using the
// optimised profile.test.
// ---------------------------------------------------------------------------

/// Run `cargo test --workspace --features parallel,simd`.
///
/// Speed comes from two sources:
/// - `[profile.test]` opt-level=2 in `Cargo.toml` (no RUSTFLAGS needed).
/// - `parallel` feature: engages `std::thread::scope` multi-core paths in
///   Strang2D/3D etc. Bit-identical to serial per ADR-0018 regression tests.
/// - `simd` feature: engages AVX2/NEON hot paths.
///
/// Debug assertions are preserved. For a pure no_std serial run use
/// `cargo test --workspace` (bare) directly.
fn test_fast() -> Result<()> {
    let root = workspace_root()?;
    let cmd_args = ["test", "--workspace", "--features", "parallel,simd"];
    eprintln!("$ cargo {}", cmd_args.join(" "));
    let status = process::Command::new("cargo")
        .args(cmd_args)
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!("test-fast failed (exit {})", status.code().unwrap_or(-1));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// test-full: release-mode full validation sweep (all features, slow tests).
// ---------------------------------------------------------------------------

/// Run the full validation sweep with native CPU tuning and every feature.
///
/// Equivalent to:
/// `RUSTFLAGS="-C target-cpu=native" cargo test --workspace
///  --features parallel,simd,slow-tests --release`
///
/// - `target-cpu=native`: engages AVX2 / NEON SIMD paths.
/// - `parallel` feature: uses all available cores.
/// - `slow-tests` feature: includes oracle sweeps and G3⁶-2D flagship gate.
/// - `--release`: fastest path; debug assertions still present in tests
///   because Cargo applies test-binary rustc flags separately from dep flags.
fn test_full() -> Result<()> {
    let root = workspace_root()?;
    let cmd_args = [
        "test",
        "--workspace",
        "--features",
        "parallel,simd,slow-tests",
        "--release",
    ];
    eprintln!(
        "$ RUSTFLAGS=\"-C target-cpu=native\" cargo {}",
        cmd_args.join(" ")
    );
    let status = process::Command::new("cargo")
        .args(cmd_args)
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!("test-full failed (exit {})", status.code().unwrap_or(-1));
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// test-flagship: run the 3 normative slope-gate test binaries.
// ---------------------------------------------------------------------------

/// Run the 3 normative flagship slope-gate tests (G3⁶-2D, G4_NS2D_aniso, G5_3D).
///
/// These tests live in `#![cfg(feature = "slow-tests")]`-gated integration
/// test files and are NOT marked `#[ignore]`, so the previous `-- --ignored`
/// filter silently produced `running 0 tests` for each binary.  This command
/// targets only the three flagship binaries by name.
///
/// Equivalent to:
/// ```text
/// RUSTFLAGS="-C target-cpu=native" cargo test --workspace \
///   --features parallel,simd,slow-tests --release \
///   --test convergence_rate_6th_2d \
///   --test strang_nonseparable_aniso_slope \
///   --test strang_3d_slope \
///   --no-fail-fast -- --nocapture
/// ```
///
/// Use this for manual gate verification and v0.x calibration runs.
fn test_flagship() -> Result<()> {
    let root = workspace_root()?;
    let cmd_args = [
        "test",
        "--workspace",
        "--features",
        "parallel,simd,slow-tests",
        "--release",
        "--test",
        "convergence_rate_6th_2d",
        "--test",
        "strang_nonseparable_aniso_slope",
        "--test",
        "strang_3d_slope",
        "--no-fail-fast",
        "--",
        "--nocapture",
    ];
    eprintln!(
        "$ RUSTFLAGS=\"-C target-cpu=native\" cargo {}",
        cmd_args.join(" ")
    );
    let status = process::Command::new("cargo")
        .args(cmd_args)
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!(
            "test-flagship failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// test-ignored-gates: run all heavy `#[ignore]` RELEASE_BLOCKING gates.
// ---------------------------------------------------------------------------

/// Run every `#[ignore]`-marked test in the workspace using the full release
/// profile + native SIMD + slow-tests feature flag.
///
/// Equivalent to:
/// ```text
/// RUSTFLAGS="-C target-cpu=native" cargo test --workspace \
///   --features parallel,simd,slow-tests --release \
///   -- --ignored
/// ```
///
/// This is the companion to `test-flagship` (which targets 3 named slow-tests
/// binaries that are NOT `#[ignore]`-marked). `test-ignored-gates` covers the
/// heavy gates that ARE `#[ignore]`-marked, e.g.:
///
/// - g17_magnus6_slope, g18_schrodinger_unitarity
/// - hormander_kolmogorov_slope, hormander_heisenberg_slope, hormander_engel_slope
/// - robin_heat_slope, subordinated_order1_slope
/// - zeta4_truthful_order, diff_scipy, capture_trace_v1
///
/// Do NOT run this in CI on pull requests — each gate takes minutes to hours.
/// Use it locally and on production hardware as part of the release checklist.
fn test_ignored_gates() -> Result<()> {
    let root = workspace_root()?;
    let cmd_args = [
        "test",
        "--workspace",
        "--features",
        "parallel,simd,slow-tests",
        "--release",
        "--",
        "--ignored",
    ];
    eprintln!(
        "$ RUSTFLAGS=\"-C target-cpu=native\" cargo {}",
        cmd_args.join(" ")
    );
    let status = process::Command::new("cargo")
        .args(cmd_args)
        .env("RUSTFLAGS", "-C target-cpu=native")
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!(
            "test-ignored-gates failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Resolve workspace root: CARGO_MANIFEST_DIR (xtask crate) → parent.
pub(crate) fn workspace_root() -> Result<PathBuf> {
    let manifest = env::var("CARGO_MANIFEST_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(|_| PathBuf::from("."));
    Ok(manifest.parent().unwrap_or(&manifest).to_path_buf())
}
