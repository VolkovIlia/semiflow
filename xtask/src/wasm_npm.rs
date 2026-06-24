//! `wasm-pack-npm` subcommand — assemble a publishable npm package for `semiflow-wasm`.
//!
//! ## Overview
//!
//! Runs `wasm-pack build` for both `web` and `nodejs` targets **with** TypeScript
//! declarations enabled (unlike the dev-loop `wasm-build` subcommand which passes
//! `--no-typescript`).  Merges the two build outputs into `dist/npm/` laid out as:
//!
//! ```text
//! dist/npm/
//!   package.json       (rendered from crates/semiflow-wasm/npm/package.json.tmpl)
//!   README.md
//!   LICENSE-MIT
//!   LICENSE-APACHE
//!   web/
//!     semiflow_wasm.js
//!     semiflow_wasm_bg.wasm      (wasm-bindgen emits _bg suffix; JS glue references this name)
//!     semiflow_wasm_bg.wasm.d.ts (TS declarations for the bg wasm exports)
//!     semiflow_wasm.d.ts
//!   node/
//!     semiflow_wasm.cjs          (renamed from semiflow_wasm.js produced by --target nodejs)
//!     semiflow_wasm_bg.wasm      (nodejs glue loads via __dirname/semiflow_wasm_bg.wasm)
//!     semiflow_wasm_bg.wasm.d.ts
//! ```
//!
//! The caller (CI workflow) runs `npm pack` and `npm publish` afterwards.

use std::{fs, path::Path, process};

use anyhow::{bail, Context, Result};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Build and assemble the npm package in `dist/npm/`.
///
/// Steps:
/// 1. Build `--target web` with TypeScript declarations.
/// 2. Build `--target nodejs` with TypeScript declarations.
/// 3. Assemble `dist/npm/` from both outputs.
/// 4. Render `package.json.tmpl` → `dist/npm/package.json`.
/// 5. Copy `README.md`, `LICENSE-MIT`, `LICENSE-APACHE`.
/// 6. Verify layout is complete.
pub fn wasm_pack_npm() -> Result<()> {
    let root = crate::workspace_root()?;
    ensure_wasm_pack()?;

    let pkg_web = root.join("crates").join("semiflow-wasm").join("pkg-web");
    let pkg_node = root.join("crates").join("semiflow-wasm").join("pkg-node");
    let dist = root.join("dist").join("npm");

    run_wasm_pack_release("web", "pkg-web", &root)?;
    run_wasm_pack_release("nodejs", "pkg-node", &root)?;

    assemble_dist(&dist, &pkg_web, &pkg_node, &root)?;

    let version = read_workspace_version(&root)?;
    render_template(&version, &dist, &root)?;

    copy_ancillary(&dist, &root)?;
    verify_layout(&dist)?;

    println!("wasm-pack-npm: dist/npm/ assembled (version {version})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 1+2: run wasm-pack with TypeScript declarations
// ---------------------------------------------------------------------------

/// Run a single `wasm-pack build` with TS declarations enabled.
///
/// Unlike `wasm-build`, this does NOT pass `--no-typescript`.
fn run_wasm_pack_release(target: &str, out_dir: &str, root: &Path) -> Result<()> {
    let crate_dir = root.join("crates").join("semiflow-wasm");
    eprintln!(
        "$ wasm-pack build crates/semiflow-wasm --target {target} --out-dir {out_dir} --release"
    );
    let status = process::Command::new("wasm-pack")
        .args([
            "build",
            crate_dir.to_str().unwrap(),
            "--target",
            target,
            "--out-dir",
            out_dir,
            "--release",
        ])
        .current_dir(root)
        .status()
        .context("failed to spawn wasm-pack")?;

    if !status.success() {
        bail!(
            "wasm-pack build --target {target} failed (exit {})",
            status.code().unwrap_or(-1)
        );
    }
    println!("wasm-pack-npm: {out_dir}/ built ({target})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 3: assemble dist/npm/ directory
// ---------------------------------------------------------------------------

/// Copy build artifacts from `pkg-web/` and `pkg-node/` into `dist/npm/`.
///
/// Layout:
/// - `dist/npm/web/` ← `pkg-web/{js,_bg.wasm,_bg.wasm.d.ts,d.ts}` files
/// - `dist/npm/node/` ← `pkg-node/{.js→.cjs,_bg.wasm,_bg.wasm.d.ts}` files
///
/// wasm-bindgen emits `semiflow_wasm_bg.wasm` (not `semiflow_wasm.wasm`).
/// The JS glue hard-codes this `_bg` name in both targets, so we must
/// preserve it exactly — renaming would break the JS→wasm link at runtime.
fn assemble_dist(dist: &Path, pkg_web: &Path, pkg_node: &Path, _root: &Path) -> Result<()> {
    let web_dir = dist.join("web");
    let node_dir = dist.join("node");
    fs::create_dir_all(&web_dir).context("create dist/npm/web/")?;
    fs::create_dir_all(&node_dir).context("create dist/npm/node/")?;

    // Web: copy JS glue, the _bg wasm (referenced by URL in the glue), and TS types.
    copy_dir_matching(
        pkg_web,
        &web_dir,
        &[
            "semiflow_wasm.js",
            "semiflow_wasm_bg.wasm",
            "semiflow_wasm_bg.wasm.d.ts",
            "semiflow_wasm.d.ts",
        ],
    )?;

    // Node build: rename .js → .cjs (CommonJS convention).
    // The .cjs still loads `${__dirname}/semiflow_wasm_bg.wasm`, so keep that name.
    let node_js = pkg_node.join("semiflow_wasm.js");
    if !node_js.exists() {
        bail!("expected pkg-node/semiflow_wasm.js — was --target nodejs build successful?");
    }
    fs::copy(&node_js, node_dir.join("semiflow_wasm.cjs"))
        .context("copy semiflow_wasm.js → node/semiflow_wasm.cjs")?;

    // Copy the _bg wasm and its TS declarations for the node target.
    copy_dir_matching(
        pkg_node,
        &node_dir,
        &["semiflow_wasm_bg.wasm", "semiflow_wasm_bg.wasm.d.ts"],
    )?;
    Ok(())
}

/// Copy specific filenames from `src_dir` to `dst_dir`, skipping missing files.
fn copy_dir_matching(src_dir: &Path, dst_dir: &Path, names: &[&str]) -> Result<()> {
    for name in names {
        let src = src_dir.join(name);
        if src.exists() {
            fs::copy(&src, dst_dir.join(name)).with_context(|| format!("copy {name}"))?;
        } else {
            bail!(
                "expected {src} — was the web build successful?",
                src = src.display()
            );
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 4: render package.json from template
// ---------------------------------------------------------------------------

/// Substitute `__VERSION__` in `npm/package.json.tmpl` and write to `dist/npm/package.json`.
fn render_template(version: &str, dist: &Path, root: &Path) -> Result<()> {
    let tmpl_path = root
        .join("crates")
        .join("semiflow-wasm")
        .join("npm")
        .join("package.json.tmpl");
    let tmpl = fs::read_to_string(&tmpl_path)
        .with_context(|| format!("read template {}", tmpl_path.display()))?;
    let rendered = tmpl.replace("__VERSION__", version);
    let out = dist.join("package.json");
    fs::write(&out, rendered).context("write dist/npm/package.json")?;
    println!("wasm-pack-npm: package.json rendered (version {version})");
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 5: copy ancillary files
// ---------------------------------------------------------------------------

/// Copy `README.md`, `LICENSE-MIT`, `LICENSE-APACHE` into `dist/npm/`.
fn copy_ancillary(dist: &Path, root: &Path) -> Result<()> {
    let wasm_dir = root.join("crates").join("semiflow-wasm");
    let readme = wasm_dir.join("README.md");
    if readme.exists() {
        fs::copy(&readme, dist.join("README.md")).context("copy README.md")?;
    } else {
        eprintln!("WARN: crates/semiflow-wasm/README.md not found; skipping");
    }
    for lic in ["LICENSE-MIT", "LICENSE-APACHE"] {
        let src = root.join(lic);
        if src.exists() {
            fs::copy(&src, dist.join(lic)).with_context(|| format!("copy {lic}"))?;
        } else {
            eprintln!("WARN: {lic} not found at workspace root; skipping");
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Step 6: verify layout
// ---------------------------------------------------------------------------

/// Assert that `dist/npm/` contains all required files.
fn verify_layout(dist: &Path) -> Result<()> {
    let required: &[&str] = &[
        "package.json",
        "web/semiflow_wasm.js",
        "web/semiflow_wasm_bg.wasm",
        "web/semiflow_wasm.d.ts",
        "node/semiflow_wasm.cjs",
        "node/semiflow_wasm_bg.wasm",
    ];
    let mut missing: Vec<&str> = Vec::new();
    for rel in required {
        if !dist.join(rel).exists() {
            missing.push(rel);
        }
    }
    if !missing.is_empty() {
        bail!("dist/npm/ is missing required files: {:?}", missing);
    }
    println!("wasm-pack-npm: layout verified OK");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Read the workspace version from `[workspace.package] version` in `Cargo.toml`.
fn read_workspace_version(root: &Path) -> Result<String> {
    let cargo_toml = root.join("Cargo.toml");
    let src = fs::read_to_string(&cargo_toml).context("read workspace Cargo.toml")?;
    let version = parse_workspace_version(&src).ok_or_else(|| {
        anyhow::anyhow!("could not parse [workspace.package] version from Cargo.toml")
    })?;
    Ok(version)
}

/// Extract `version = "..."` from the `[workspace.package]` table.
fn parse_workspace_version(src: &str) -> Option<String> {
    let mut in_ws_pkg = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.package]" {
            in_ws_pkg = true;
            continue;
        }
        if in_ws_pkg && trimmed.starts_with('[') {
            break; // left the section
        }
        if in_ws_pkg {
            if let Some(rest) = trimmed.strip_prefix("version") {
                let rest = rest.trim_start_matches([' ', '=']);
                let ver = rest.trim().trim_matches('"');
                if !ver.is_empty() {
                    return Some(ver.to_owned());
                }
            }
        }
    }
    None
}

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
             Install with: cargo install wasm-pack --locked\n\
             Then re-run: cargo run -p xtask -- wasm-pack-npm"
        );
    }
    Ok(())
}
