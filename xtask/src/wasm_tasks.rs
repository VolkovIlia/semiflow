//! `wasm-build` and `wasm-test` subcommands for the `semiflow-wasm` crate.
//!
//! ## wasm-build
//!
//! Runs `wasm-pack build crates/semiflow-wasm` for both `web` (ESM) and
//! `nodejs` (CJS) targets.  Requires `wasm-pack` on PATH; prints install hint
//! if missing.  Outputs land in `crates/semiflow-wasm/pkg-web/` and
//! `crates/semiflow-wasm/pkg-node/`.
//!
//! Pass `--size` to build with `[profile.release-wasm]` (opt-level=z, strip=true)
//! for a smaller bundle.  Default uses `[profile.release]` for reproducible CI.
//! See ADR-0028 Amendment 1 and D2 (v0.13.0).
//!
//! ### wasm-opt and bulk-memory (D3 fix, v0.13.x)
//!
//! wasm-pack 0.15.0 ignores unknown `package.metadata.wasm-pack.profile.*` keys
//! (warns "unknown key and will be ignored").  When `--profile release-wasm` is
//! passed, wasm-pack cannot find a matching metadata entry and falls back to its
//! default wasm-opt invocation (`-O` only), which rejects the bulk-memory opcodes
//! (`memory.copy` / `memory.fill`) that the Rust toolchain emits for memcpy/memset
//! intrinsics.  The fix: pass `--no-opt` so wasm-pack skips its bundled wasm-opt
//! for the `--size` path.  Size optimisation already comes from the `release-wasm`
//! cargo profile (opt-level=z, lto=fat, codegen-units=1, strip=true); the
//! `package.metadata.wasm-pack.profile.release` entry configures wasm-opt with
//! `--enable-bulk-memory` for the default `--release` (non-size) path.
//!
//! ## wasm-test
//!
//! Runs `wasm-pack test crates/semiflow-wasm --node` by default.
//! Pass `--chrome` (anywhere in `args`) to run headless Chrome instead.
//! Pass `--firefox` (anywhere in `args`) to run headless Firefox instead.

use std::process;

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// wasm-build
// ---------------------------------------------------------------------------

/// Build `semiflow-wasm` via wasm-pack for both `web` and `nodejs` targets.
///
/// Pass `--size` anywhere in `args` to use `[profile.release-wasm]` (opt-level=z,
/// strip=true) instead of the default `[profile.release]`.  Size profile targets
/// <500 KB per bundle (D2, v0.13.0); default profile is used for reproducible CI.
pub fn wasm_build_with_args(args: &[String]) -> Result<()> {
    let root = crate::workspace_root()?;
    let crate_dir = root.join("crates").join("semiflow-wasm");
    let use_size_profile = args.iter().any(|a| a == "--size");
    let profile_args: &[&str] = if use_size_profile {
        // --no-opt: skip wasm-pack's bundled wasm-opt for the release-wasm path.
        // wasm-pack 0.15.0 ignores the `release-wasm` metadata key and falls back
        // to running wasm-opt without --enable-bulk-memory, causing a validator
        // error.  Size is already optimised by the release-wasm rustc profile.
        &["--profile", "release-wasm", "--no-opt"]
    } else {
        &[]
    };
    if use_size_profile {
        eprintln!("wasm-build: using [profile.release-wasm] (opt-level=z, strip=true)");
    }

    ensure_wasm_pack()?;
    build_one_target(&crate_dir, &root, "web", "pkg-web", profile_args)?;
    build_one_target(&crate_dir, &root, "nodejs", "pkg-node", profile_args)?;
    Ok(())
}

/// Run a single `wasm-pack build` invocation for one target.
fn build_one_target(
    crate_dir: &std::path::Path,
    root: &std::path::Path,
    target: &str,
    out_dir: &str,
    profile_args: &[&str],
) -> Result<()> {
    let mut cmd_args: Vec<&str> = vec![
        "build",
        crate_dir.to_str().unwrap(),
        "--target",
        target,
        "--out-dir",
        out_dir,
        "--no-typescript",
    ];
    cmd_args.extend_from_slice(profile_args);
    let cmd_display = cmd_args.join(" ");
    eprintln!("$ wasm-pack {cmd_display}");
    let status = process::Command::new("wasm-pack")
        .args(&cmd_args)
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!(
            "wasm-pack build --target {target} failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    println!("wasm-build: {out_dir}/ written ({target} bundle)");
    Ok(())
}

// ---------------------------------------------------------------------------
// wasm-test
// ---------------------------------------------------------------------------

/// Run the graph PDE WASM smoke gate (`G_WASM_smoke_graph`, ADR-0059).
///
/// Builds `semiflow-wasm` for the `nodejs` target via `wasm-pack` and runs
/// only the `graph_heat` test binary using `wasm-bindgen-test`.
///
/// Equivalent to:
/// ```text
/// wasm-pack build crates/semiflow-wasm --target nodejs --release
/// wasm-pack test crates/semiflow-wasm --node --release
/// ```
///
/// Gate: `sup_error < 5e-4` for P_64 path graph, `u₀(i) = exp(−i²/64)`,
/// `t_final = 0.5`, `n_steps = 50` (ADR-0059 §2.5).
///
/// Cross-binding note: the WASM binding drives the identical Rust core as the
/// FFI and PyO3 bindings; the expected sup_error is within 3 ULP of the FFI
/// smoke value (~1.46e-6 for the 1D heat baseline).
pub fn wasm_graph_smoke() -> Result<()> {
    let root = crate::workspace_root()?;
    let crate_dir = root.join("crates").join("semiflow-wasm");

    ensure_wasm_pack()?;

    // Build first (--target nodejs for Node smoke).
    let build_args = [
        "build",
        crate_dir.to_str().unwrap(),
        "--target",
        "nodejs",
        "--release",
        "--no-typescript",
    ];
    eprintln!("$ wasm-pack {}", build_args.join(" "));
    let build_status = process::Command::new("wasm-pack")
        .args(build_args)
        .current_dir(&root)
        .status()?;
    if !build_status.success() {
        bail!(
            "wasm-graph-smoke: wasm-pack build failed (exit {})",
            build_status.code().unwrap_or(-1)
        );
    }

    // Run wasm-bindgen-test on Node (runs all tests including graph_heat).
    run_wasm_pack_test(
        &crate_dir,
        &root,
        &["--node", "--release"],
        "Node (graph smoke)",
    )
}

/// Run wasm-bindgen-test smoke suite.
///
/// Default: `wasm-pack test crates/semiflow-wasm --node`.
/// Pass `--chrome` in `args` for headless Chrome (Linux CI only).
/// Pass `--firefox` in `args` for headless Firefox (Linux CI only).
pub fn wasm_test(args: &[String]) -> Result<()> {
    let root = crate::workspace_root()?;
    let crate_dir = root.join("crates").join("semiflow-wasm");

    ensure_wasm_pack()?;

    let use_chrome = args.iter().any(|a| a == "--chrome");
    let use_firefox = args.iter().any(|a| a == "--firefox");

    if use_chrome {
        run_wasm_pack_test(
            &crate_dir,
            &root,
            &["--chrome", "--headless"],
            "headless Chrome",
        )
    } else if use_firefox {
        run_wasm_pack_test(
            &crate_dir,
            &root,
            &["--firefox", "--headless"],
            "headless Firefox",
        )
    } else {
        run_wasm_pack_test(&crate_dir, &root, &["--node"], "Node")
    }
}

/// Invoke `wasm-pack test` with the given browser/runtime flags and label.
fn run_wasm_pack_test(
    crate_dir: &std::path::Path,
    root: &std::path::Path,
    browser_args: &[&str],
    label: &str,
) -> Result<()> {
    let flag_str = browser_args.join(" ");
    eprintln!("$ wasm-pack test {flag_str} crates/semiflow-wasm");
    let mut cmd = process::Command::new("wasm-pack");
    // wasm-pack <=0.9.1 requires flags BEFORE the crate path.
    cmd.arg("test")
        .args(browser_args)
        .arg(crate_dir)
        .current_dir(root);
    let status = cmd.status()?;
    if !status.success() {
        bail!(
            "wasm-pack test {flag_str} failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    println!("wasm-test: PASS ({label})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helper
// ---------------------------------------------------------------------------

/// Verify that `wasm-pack` is on PATH; bail with an install hint if not.
fn ensure_wasm_pack() -> Result<()> {
    let ok = process::Command::new("wasm-pack")
        .arg("--version")
        .stdout(process::Stdio::null())
        .stderr(process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        bail!(
            "wasm-pack not found on PATH.\n\
             Install with: cargo install wasm-pack --locked"
        );
    }
    Ok(())
}
