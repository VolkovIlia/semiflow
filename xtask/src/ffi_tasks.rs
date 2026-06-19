//! `ffi-headers` and `ffi-smoke` subcommands for the `semiflow-ffi` crate.
//!
//! ## ffi-headers
//!
//! Generates `crates/semiflow-ffi/include/semiflow.h` via cbindgen.
//! Pass `--check` to verify the committed header matches what cbindgen would
//! generate; exits 1 on drift.
//!
//! ## ffi-smoke
//!
//! 1. Builds `semiflow-ffi` with `--profile release-ffi` (requires `panic = "unwind"`).
//! 2. Compiles `examples/heat.c` against the cdylib.
//! 3. Runs the binary and parses `sup_error=<float>` from stdout.
//! 4. Fails if `sup_error >= 5e-4`.

use std::{path::PathBuf, process};

use anyhow::{bail, Context, Result};

// ---------------------------------------------------------------------------
// ffi-headers
// ---------------------------------------------------------------------------

/// Generate (or check) `crates/semiflow-ffi/include/semiflow.h`.
pub fn ffi_headers(check: bool) -> Result<()> {
    let root = crate::workspace_root()?;
    let crate_dir = root.join("crates").join("semiflow-ffi");
    let header_path = crate_dir.join("include").join("semiflow.h");

    let generated = generate_header(&crate_dir)?;

    if check {
        let committed = std::fs::read_to_string(&header_path)
            .with_context(|| format!("reading committed header {}", header_path.display()))?;
        if generated == committed {
            println!("ffi-headers --check: PASS — header is up-to-date");
            return Ok(());
        }
        bail!(
            "ffi-headers --check: FAIL — header drift detected.\n\
             Run `cargo run -p xtask -- ffi-headers` and commit the result."
        );
    }

    std::fs::create_dir_all(header_path.parent().unwrap())?;
    std::fs::write(&header_path, &generated)?;
    println!("ffi-headers: wrote {}", header_path.display());
    Ok(())
}

/// Build the header string via cbindgen.
fn generate_header(crate_dir: &std::path::Path) -> Result<String> {
    let config = cbindgen::Config::from_file(crate_dir.join("cbindgen.toml"))
        .map_err(|e| anyhow::anyhow!("reading cbindgen.toml: {e}"))?;

    let bindings = cbindgen::Builder::new()
        .with_crate(crate_dir)
        .with_config(config)
        .generate()
        .context("cbindgen::Builder::generate failed")?;

    let mut buf = Vec::new();
    bindings.write(&mut buf);
    String::from_utf8(buf).context("cbindgen output is not valid UTF-8")
}

// ---------------------------------------------------------------------------
// ffi-smoke
// ---------------------------------------------------------------------------

/// Build the cdylib, compile smoke examples, run them, check `sup_error`.
///
/// Runs three smoke binaries:
/// 1. `examples/heat.c` — unit-a path (existing, ADR-0028 Wave A).
/// 2. `examples/heat_var_a.c` — variable-a callback path (ADR-0034 S1.2).
/// 3. `examples/greeks.c` — hyper-dual Greeks path (ADR-0028 Amendment 2, ADR-0133 A1).
pub fn ffi_smoke() -> Result<()> {
    let root = crate::workspace_root()?;
    build_cdylib(&root)?;
    let binary = compile_heat_c(&root)?;
    run_heat_binary(&root, &binary, "sup_error")?;
    let binary_var = compile_c_example(&root, "heat_var_a")?;
    run_heat_binary(&root, &binary_var, "sup_error_var_a")?;
    let binary_greeks = compile_c_example(&root, "greeks")?;
    run_greeks_binary(&root, &binary_greeks)
}

/// Run the Greeks smoke binary; check both delta and gamma FD-agreement.
fn run_greeks_binary(root: &std::path::Path, binary: &std::path::Path) -> Result<()> {
    let lib_dir = root.join("target").join("release-ffi");
    eprintln!("$ {}", binary.display());
    let mut cmd = process::Command::new(binary);
    set_library_path(&mut cmd, &lib_dir);
    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    if !output.status.success() {
        bail!("greeks binary failed.\nstdout: {stdout}\nstderr: {stderr}");
    }
    let sup_delta = parse_keyed_f64(&stdout, "sup_greeks_delta")?;
    let sup_gamma = parse_keyed_f64(&stdout, "sup_greeks_gamma")?;
    println!("ffi-smoke: sup_greeks_delta = {sup_delta:.3e}");
    println!("ffi-smoke: sup_greeks_gamma = {sup_gamma:.3e}");
    if sup_delta >= 1e-6 {
        bail!("ffi-smoke: FAIL — sup_greeks_delta {sup_delta:.3e} >= 1e-6");
    }
    if sup_gamma >= 1e-4 {
        bail!("ffi-smoke: FAIL — sup_greeks_gamma {sup_gamma:.3e} >= 1e-4");
    }
    println!("ffi-smoke: PASS (greeks)");
    Ok(())
}

/// Run `cargo build -p semiflow-ffi --profile release-ffi`.
fn build_cdylib(root: &std::path::Path) -> Result<()> {
    eprintln!("$ cargo build -p semiflow-ffi --profile release-ffi");
    let status = process::Command::new("cargo")
        .args(["build", "-p", "semiflow-ffi", "--profile", "release-ffi"])
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("cargo build -p semiflow-ffi --profile release-ffi failed");
    }
    Ok(())
}

/// Compile `examples/heat.c` against the cdylib, return the binary path.
fn compile_heat_c(root: &std::path::Path) -> Result<PathBuf> {
    compile_c_example(root, "heat")
}

/// Compile a named C example (`examples/{name}.c`) against the cdylib.
fn compile_c_example(root: &std::path::Path, name: &str) -> Result<PathBuf> {
    let lib_dir = root.join("target").join("release-ffi");
    let (cc, link_args) = detect_c_toolchain(&lib_dir)?;
    let include_dir = root.join("crates").join("semiflow-ffi").join("include");
    let src_name = format!("{name}.c");
    let src = root
        .join("crates")
        .join("semiflow-ffi")
        .join("examples")
        .join(&src_name);
    let bin_name = if cfg!(target_os = "windows") {
        format!("{name}.exe")
    } else {
        name.to_owned()
    };
    let out = lib_dir.join(&bin_name);

    eprintln!("$ {cc} {src:?} -I{include_dir:?} {link_args:?} -o {out:?}");
    let status = process::Command::new(&cc)
        .arg(&src)
        .arg(format!("-I{}", include_dir.display()))
        .args(&link_args)
        .arg("-lm")
        .arg("-o")
        .arg(&out)
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("C compilation of {src_name} failed (cc={cc})");
    }
    Ok(out)
}

/// Returns `(cc_binary, link_args)` where `link_args` are individual arguments
/// to pass to the C compiler (NOT a single shell string).
fn detect_c_toolchain(lib_dir: &std::path::Path) -> Result<(String, Vec<String>)> {
    // cdylib names depend on OS.  On Windows, cl.exe needs the import lib
    // (.dll.lib or .lib), not the .dll itself.  Rust names it semiflow_ffi.dll.lib.
    #[cfg(target_os = "windows")]
    let (import_lib, cc) = ("semiflow_ffi.dll.lib", "cl.exe");
    #[cfg(target_os = "macos")]
    let (_import_lib, cc) = ("libsemiflow_ffi.dylib", "cc");
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    let (_import_lib, cc) = ("libsemiflow_ffi.so", "cc");

    // Prefer clang if available on Linux.
    let cc = if cfg!(not(any(target_os = "windows", target_os = "macos"))) {
        if process::Command::new("clang")
            .arg("--version")
            .output()
            .is_ok()
        {
            "clang".to_owned()
        } else {
            cc.to_owned()
        }
    } else {
        cc.to_owned()
    };

    #[cfg(not(target_os = "windows"))]
    let link_args = vec![
        format!("-L{}", lib_dir.display()),
        "-lsemiflow_ffi".to_owned(),
    ];
    #[cfg(target_os = "windows")]
    let link_args = vec![lib_dir.join(import_lib).to_string_lossy().into_owned()];

    Ok((cc, link_args))
}

/// Run a heat smoke binary, parse `{key}=<f64>` from stdout, fail if >= 5e-4.
fn run_heat_binary(root: &std::path::Path, binary: &std::path::Path, key: &str) -> Result<()> {
    let lib_dir = root.join("target").join("release-ffi");
    eprintln!("$ {}", binary.display());

    let mut cmd = process::Command::new(binary);
    set_library_path(&mut cmd, &lib_dir);

    let output = cmd.output()?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);

    if !output.status.success() {
        bail!(
            "{} binary failed.\nstdout: {stdout}\nstderr: {stderr}",
            binary.display()
        );
    }

    let sup_err = parse_keyed_f64(&stdout, key)?;
    println!("ffi-smoke: {key} = {sup_err:.3e}");
    if sup_err >= 5e-4 {
        bail!("ffi-smoke: FAIL — {key} {sup_err:.3e} >= 5e-4");
    }
    println!("ffi-smoke: PASS ({key})");
    Ok(())
}

/// Set `LD_LIBRARY_PATH` / `DYLD_LIBRARY_PATH` / `PATH` for the run.
fn set_library_path(cmd: &mut process::Command, lib_dir: &std::path::Path) {
    #[cfg(target_os = "macos")]
    cmd.env("DYLD_LIBRARY_PATH", lib_dir);
    #[cfg(target_os = "windows")]
    {
        let existing = std::env::var("PATH").unwrap_or_default();
        cmd.env("PATH", format!("{};{existing}", lib_dir.display()));
    }
    #[cfg(not(any(target_os = "windows", target_os = "macos")))]
    cmd.env("LD_LIBRARY_PATH", lib_dir);
}

// ---------------------------------------------------------------------------
// ffi-graph-smoke
// ---------------------------------------------------------------------------

/// Build the cdylib, compile `examples/graph_heat.c`, run it, check `sup_error`.
///
/// Validates the Graph PDE C ABI entry points (v2.2 Wave C, ADR-0059).
/// Threshold: `sup_error < 5e-4` (same as Heat1D gate).
pub fn ffi_graph_smoke() -> Result<()> {
    let root = crate::workspace_root()?;
    build_cdylib(&root)?;
    let binary = compile_c_example(&root, "graph_heat")?;
    run_heat_binary(&root, &binary, "sup_error")
}

/// Parse `{key}=<float>` from a line in the binary's stdout.
fn parse_keyed_f64(stdout: &str, key: &str) -> Result<f64> {
    let needle = format!("{key}=");
    for line in stdout.lines() {
        if let Some(pos) = line.find(&needle) {
            let rest = &line[pos + needle.len()..];
            let token = rest.split_whitespace().next().unwrap_or(rest);
            return token
                .parse::<f64>()
                .with_context(|| format!("parsing {key} token: {token:?}"));
        }
    }
    bail!("{key} not found in binary output:\n{stdout}");
}
