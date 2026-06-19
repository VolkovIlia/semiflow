//! `py-build`, `py-smoke`, `py-bench`, and `py-graph-smoke` subcommands for `semiflow-py`.
//!
//! ## py-build
//!
//! Runs `maturin build --profile release-ffi` for `crates/semiflow-py/Cargo.toml`.
//! Produces a wheel in `target/wheels/`.  Requires `maturin` on `PATH`.
//!
//! ## py-smoke
//!
//! 1. Creates a venv at `target/py-smoke-venv`.
//! 2. Installs `maturin`, `numpy`, and `pytest` into the venv.
//! 3. Runs `maturin develop` to install the extension module in dev mode.
//! 4. Runs `pytest crates/semiflow-py/tests/`.
//! 5. Fails if pytest exits non-zero.
//!
//! ## py-bench
//!
//! Runs `cargo bench -p semiflow-py --profile release-ffi`.
//! Results land in `target/criterion/`.  On first run the baseline is saved;
//! subsequent runs compare against it.  ADR-0031 budget: ≤2% regression.
//!
//! ## py-graph-smoke
//!
//! Runs the graph PDE Python smoke gate:
//! 1. Creates a venv at `target/py-graph-smoke-venv`.
//! 2. Installs `maturin`, `numpy`, and `pytest`.
//! 3. Runs `maturin develop --profile release-ffi`.
//! 4. Runs `pytest crates/semiflow-py/tests/smoke_graph.py -v`.
//!
//! Gate: `smoke_graph.py` includes a cross-validation test verifying that
//! `GraphHeat.evolve` on a 1-D chain graph matches `Heat1D.evolve` to ≤3 ULP
//! (ADR-0059 cross-binding identity gate).

use std::{path::Path, process};

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// py-build
// ---------------------------------------------------------------------------

/// Build the `semiflow-py` wheel with maturin.
pub fn py_build() -> Result<()> {
    let root = crate::workspace_root()?;
    let manifest = root.join("crates").join("semiflow-py").join("Cargo.toml");

    ensure_maturin()?;

    eprintln!(
        "$ maturin build --profile release-ffi -m {}",
        manifest.display()
    );
    let status = process::Command::new("maturin")
        .args([
            "build",
            "--profile",
            "release-ffi",
            "-m",
            manifest.to_str().unwrap(),
        ])
        .current_dir(&root)
        .status()?;

    if !status.success() {
        bail!(
            "maturin build failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    println!("py-build: wheel written to target/wheels/");
    Ok(())
}

// ---------------------------------------------------------------------------
// py-smoke
// ---------------------------------------------------------------------------

/// Create venv, install deps, `maturin develop`, run pytest.
pub fn py_smoke() -> Result<()> {
    let root = crate::workspace_root()?;
    let venv = root.join("target").join("py-smoke-venv");
    let manifest = root.join("crates").join("semiflow-py").join("Cargo.toml");
    let tests_dir = root.join("crates").join("semiflow-py").join("tests");

    create_venv(&venv)?;
    install_deps(&venv, &root)?;
    maturin_develop(&venv, &manifest, &root)?;
    run_pytest(&venv, &tests_dir, &root)
}

/// Create a fresh Python virtual environment.
fn create_venv(venv: &Path) -> Result<()> {
    eprintln!("$ python3 -m venv {}", venv.display());
    let status = process::Command::new("python3")
        .args(["-m", "venv", venv.to_str().unwrap()])
        .status()?;
    if !status.success() {
        bail!("python3 -m venv failed");
    }
    Ok(())
}

/// Install maturin, numpy, pytest into the venv.
fn install_deps(venv: &Path, root: &Path) -> Result<()> {
    let pip = venv_bin(venv, "pip");
    eprintln!("$ {pip} install maturin numpy pytest");
    let status = process::Command::new(&pip)
        .args(["install", "maturin", "numpy", "pytest"])
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("pip install maturin numpy pytest failed");
    }
    Ok(())
}

/// Run `maturin develop` to install the extension module into the venv.
///
/// `VIRTUAL_ENV` must be set so maturin can locate the venv to install into.
fn maturin_develop(venv: &Path, manifest: &Path, root: &Path) -> Result<()> {
    let maturin = venv_bin(venv, "maturin");
    eprintln!(
        "$ VIRTUAL_ENV={} {maturin} develop --profile release-ffi -m {}",
        venv.display(),
        manifest.display()
    );
    let status = process::Command::new(&maturin)
        .args([
            "develop",
            "--profile",
            "release-ffi",
            "-m",
            manifest.to_str().unwrap(),
        ])
        .env("VIRTUAL_ENV", venv)
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("maturin develop failed");
    }
    Ok(())
}

/// Run pytest against the test directory.
fn run_pytest(venv: &Path, tests_dir: &Path, root: &Path) -> Result<()> {
    let pytest = venv_bin(venv, "pytest");
    eprintln!("$ {pytest} {}", tests_dir.display());
    let status = process::Command::new(&pytest)
        .arg(tests_dir)
        .arg("-v")
        .current_dir(root)
        .status()?;
    if !status.success() {
        bail!("pytest failed");
    }
    println!("py-smoke: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// py-bench
// ---------------------------------------------------------------------------

/// Run `cargo bench -p semiflow-py --profile release-ffi`.
///
/// Results land in `target/criterion/`.  On first run this establishes the
/// baseline; subsequent runs compare against it.
///
/// ADR-0031 performance budget: ≤2% single-thread regression vs v0.10.0.
pub fn py_bench() -> Result<()> {
    let root = crate::workspace_root()?;
    let criterion_dir = root.join("target").join("criterion");

    eprintln!("$ cargo bench -p semiflow-py --profile release-ffi");
    let status = process::Command::new("cargo")
        .args(["bench", "-p", "semiflow-py", "--profile", "release-ffi"])
        .current_dir(&root)
        .status()?;

    if !status.success() {
        bail!("cargo bench failed (exit {})", status.code().unwrap_or(-1));
    }

    println!("py-bench: PASS — results in {}", criterion_dir.display());
    println!("py-bench: open target/criterion/Heat1D_evolve/report/index.html for details");
    Ok(())
}

// ---------------------------------------------------------------------------
// py-graph-smoke
// ---------------------------------------------------------------------------

/// Graph PDE Python smoke gate (ADR-0059).
///
/// Builds the wheel in dev mode, then runs `pytest smoke_graph.py -v`.
/// The test file exercises [`GraphPath`], [`GraphHeat`], [`MagnusGraphHeat`]
/// and asserts cross-binding identity against [`Heat1D`] (≤3 ULP).
pub fn py_graph_smoke() -> Result<()> {
    let root = crate::workspace_root()?;
    let venv = root.join("target").join("py-graph-smoke-venv");
    let manifest = root.join("crates").join("semiflow-py").join("Cargo.toml");
    let smoke = root
        .join("crates")
        .join("semiflow-py")
        .join("tests")
        .join("smoke_graph.py");

    create_venv(&venv)?;
    install_deps(&venv, &root)?;
    maturin_develop(&venv, &manifest, &root)?;

    let pytest = venv_bin(&venv, "pytest");
    eprintln!("$ {pytest} {} -v", smoke.display());
    let status = process::Command::new(&pytest)
        .arg(&smoke)
        .arg("-v")
        .current_dir(&root)
        .status()?;
    if !status.success() {
        bail!("py-graph-smoke: pytest failed");
    }
    println!("py-graph-smoke: PASS");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Return the path to a binary inside the venv.
fn venv_bin(venv: &Path, name: &str) -> String {
    #[cfg(target_os = "windows")]
    let bin = venv.join("Scripts").join(format!("{name}.exe"));
    #[cfg(not(target_os = "windows"))]
    let bin = venv.join("bin").join(name);
    bin.display().to_string()
}

/// Verify that `maturin` is on PATH (or discoverable).
fn ensure_maturin() -> Result<()> {
    let ok = process::Command::new("maturin")
        .arg("--version")
        .stdout(process::Stdio::null())
        .stderr(process::Stdio::null())
        .status()
        .map(|s| s.success())
        .unwrap_or(false);
    if !ok {
        bail!(
            "maturin not found on PATH.\n\
             Install with: pip install maturin"
        );
    }
    Ok(())
}
