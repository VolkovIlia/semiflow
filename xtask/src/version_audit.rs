//! Layer 2: online cross-platform version audit (advisory, never a PR gate).
//!
//! Compares the workspace `[workspace.package] version` against the latest
//! version published on PyPI, crates.io, and npm.  Network failures degrade
//! to `unknown` — this tool NEVER hard-fails on registry or network errors.
//!
//! Exit codes:
//!   0  — default (advisory); used even on `behind`/`unknown`
//!   2  — `--strict` only: at least one registry is `behind` or `ahead`
//!
//! HTTP is shell-out to `curl` (already a CI dependency); JSON is extracted
//! by a hand-rolled scalar search — no serde_json, no ureq.

use anyhow::Result;

use crate::workspace_root;

// ---------------------------------------------------------------------------
// Registry table
// ---------------------------------------------------------------------------

struct Registry {
    label: &'static str,
    package: &'static str,
    url: &'static str,
    /// Field path: first key to find, then optionally a second key within the
    /// value of the first (used for npm `.dist-tags.latest`).
    key: &'static str,
    key2: Option<&'static str>,
    /// Extra curl flags (e.g. User-Agent for crates.io).
    extra_flags: &'static [&'static str],
}

const REGISTRIES: &[Registry] = &[
    Registry {
        label: "pypi",
        package: "semiflow-pde",
        url: "https://pypi.org/pypi/semiflow-pde/json",
        key: "version",
        key2: None,
        extra_flags: &[],
    },
    Registry {
        label: "crates.io",
        package: "semiflow",
        url: "https://crates.io/api/v1/crates/semiflow",
        key: "newest_version",
        key2: None,
        extra_flags: &["-H", "User-Agent: semiflow-version-audit"],
    },
    Registry {
        label: "npm",
        package: "@semiflow/wasm",
        url: "https://registry.npmjs.org/@semiflow/wasm",
        key: "dist-tags",
        key2: Some("latest"),
        extra_flags: &[],
    },
];

// ---------------------------------------------------------------------------
// Classification
// ---------------------------------------------------------------------------

#[derive(Debug, PartialEq, Eq)]
enum Class {
    InSync,
    Behind,
    Ahead,
    Unknown,
}

impl Class {
    fn label(&self) -> &'static str {
        match self {
            Class::InSync => "in-sync",
            Class::Behind => "BEHIND",
            Class::Ahead => "AHEAD",
            Class::Unknown => "UNKNOWN",
        }
    }
}

// ---------------------------------------------------------------------------
// Canonical version for comparison
// ---------------------------------------------------------------------------

/// Normalise a version string so PyPI's `0.12.1b0` == repo's `0.12.1-beta`.
///
/// Canonical form: `(major, minor, patch, pre_rank, pre_n)` where
///   pre_rank: 0=alpha, 1=beta, 2=rc, 3=release
///   pre_n:    numeric suffix (0 if absent)
fn normalize(v: &str) -> (u32, u32, u32, u32, u32) {
    // Strip leading 'v'
    let v = v.strip_prefix('v').unwrap_or(v);
    // Split off pre-release: look for '-', 'a', 'b', 'rc' after the patch digit
    let (base, pre) = split_pre(v);
    let mut parts = base.splitn(3, '.');
    let major = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    let minor = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    let patch = parts.next().unwrap_or("0").parse::<u32>().unwrap_or(0);
    let (pre_rank, pre_n) = parse_pre(pre);
    (major, minor, patch, pre_rank, pre_n)
}

/// Split `"0.12.1-beta"` → `("0.12.1", "beta")` and `"0.12.1b0"` → `("0.12.1", "b0")`.
fn split_pre(v: &str) -> (&str, &str) {
    // Hyphen separator: e.g. "0.12.1-beta", "0.12.1-rc1"
    if let Some(pos) = v.find('-') {
        return (&v[..pos], &v[pos + 1..]);
    }
    // PEP 440 inline: e.g. "0.12.1b0", "0.12.1a1", "0.12.1rc2"
    // Find first occurrence of 'a', 'b', or "rc" after the last digit of patch
    let bytes = v.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        let c = bytes[i] as char;
        if c == 'a' || c == 'b' {
            return (&v[..i], &v[i..]);
        }
        if i + 1 < bytes.len() && &v[i..i + 2] == "rc" {
            return (&v[..i], &v[i..]);
        }
        i += 1;
    }
    (v, "")
}

/// Map a pre-release string to `(rank, n)`.
/// release → rank=3, rc → 2, beta/b → 1, alpha/a → 0
fn parse_pre(pre: &str) -> (u32, u32) {
    if pre.is_empty() {
        return (3, 0); // release
    }
    let pre_lower = pre.to_lowercase();
    let (tag, rest) = if pre_lower.starts_with("alpha") || pre_lower.starts_with('a') {
        let rest = pre_lower
            .trim_start_matches("alpha")
            .trim_start_matches('a');
        (0u32, rest)
    } else if pre_lower.starts_with("beta") || pre_lower.starts_with('b') {
        let rest = pre_lower.trim_start_matches("beta").trim_start_matches('b');
        (1u32, rest)
    } else if pre_lower.starts_with("rc") {
        let rest = pre_lower.trim_start_matches("rc");
        (2u32, rest)
    } else {
        return (3, 0); // unknown pre-release tag → treat as release
    };
    let n = rest.trim_start_matches('.').parse::<u32>().unwrap_or(0);
    (tag, n)
}

fn classify(repo: &str, registry: &str) -> Class {
    let r = normalize(repo);
    let g = normalize(registry);
    match r.cmp(&g) {
        std::cmp::Ordering::Equal => Class::InSync,
        std::cmp::Ordering::Greater => Class::Behind, // repo > registry → registry behind
        std::cmp::Ordering::Less => Class::Ahead,     // repo < registry → registry ahead
    }
}

// ---------------------------------------------------------------------------
// HTTP via curl
// ---------------------------------------------------------------------------

/// Fetch URL body via curl (silent, fail-soft, 15s timeout).
/// Returns None on any curl error or non-zero exit.
fn fetch(url: &str, extra_flags: &[&str]) -> Option<String> {
    let mut cmd = std::process::Command::new("curl");
    cmd.args(["-sSf", "--max-time", "15"]);
    for &flag in extra_flags {
        cmd.arg(flag);
    }
    cmd.arg(url);
    let out = cmd.output().ok()?;
    if !out.status.success() {
        return None;
    }
    String::from_utf8(out.stdout).ok()
}

// ---------------------------------------------------------------------------
// JSON scalar extractor (hand-rolled, ~15 lines of logic)
// ---------------------------------------------------------------------------

/// Find the first occurrence of `"key":"value"` (with optional whitespace)
/// and return `value`.
fn extract_scalar(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    // Skip optional whitespace and ':'
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    if let Some(inner) = after_colon.strip_prefix('"') {
        // String value: read until the closing quote (skip escaped quotes)
        let end = inner.find('"')?;
        return Some(inner[..end].to_owned());
    }
    // Might be a nested object (e.g. dist-tags → { "latest": "x" })
    if after_colon.starts_with('{') {
        return None; // caller handles nested lookup separately
    }
    None
}

/// For npm: first find the `dist-tags` object, then extract `latest` within it.
fn extract_nested(json: &str, key: &str, key2: &str) -> Option<String> {
    let needle = format!("\"{}\"", key);
    let pos = json.find(&needle)?;
    let after_key = &json[pos + needle.len()..];
    let after_colon = after_key.trim_start().strip_prefix(':')?.trim_start();
    // Find the end of the nested object
    let open = after_colon.find('{')? + 1;
    let close = after_colon.find('}')?;
    let inner = &after_colon[open..close];
    extract_scalar(&format!("{{{}}}", inner), key2)
}

// ---------------------------------------------------------------------------
// Version from Cargo.toml
// ---------------------------------------------------------------------------

fn workspace_version() -> Result<String> {
    let root = workspace_root()?;
    let cargo = std::fs::read_to_string(root.join("Cargo.toml"))?;
    for line in cargo.lines() {
        let t = line.trim();
        if t.starts_with("version") && t.contains('=') {
            if let Some((_, rhs)) = t.split_once('=') {
                let v = rhs.trim().trim_matches('"').to_owned();
                if !v.is_empty() {
                    return Ok(v);
                }
            }
        }
    }
    anyhow::bail!("version not found in workspace Cargo.toml")
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

/// Print per-registry status table and exit per L2.3 contract.
pub fn run(args: &[String]) -> Result<()> {
    let strict = args.iter().any(|a| a == "--strict");
    let repo_ver = workspace_version()?;

    println!("version-audit: repo = {repo_ver}");

    let mut n_behind = 0usize;
    let mut n_ahead = 0usize;
    let mut n_unknown = 0usize;

    for reg in REGISTRIES {
        let (reg_ver_opt, class) = query_registry(reg, &repo_ver);
        let reg_ver_str = reg_ver_opt.as_deref().unwrap_or("—");

        match class {
            Class::Behind => n_behind += 1,
            Class::Ahead => n_ahead += 1,
            Class::Unknown => n_unknown += 1,
            Class::InSync => {}
        }

        println!(
            "  {:<12} {:<18} {:<14} {}",
            reg.label,
            reg.package,
            reg_ver_str,
            class.label()
        );

        if matches!(class, Class::Behind) {
            eprintln!(
                "  hint: {} shows {} — may be a publish lag or failed publish",
                reg.label, reg_ver_str
            );
        }
        if matches!(class, Class::Ahead) {
            eprintln!(
                "  WARN: {} shows {} > repo {} — anomaly (manual publish or repo rollback?)",
                reg.label, reg_ver_str, repo_ver
            );
        }
        if matches!(class, Class::Unknown) {
            eprintln!(
                "  note: {} unreachable or unparseable — classified unknown",
                reg.label
            );
        }
    }

    println!(
        "version-audit: {} behind, {} ahead, {} unknown",
        n_behind, n_ahead, n_unknown
    );

    if strict && (n_behind > 0 || n_ahead > 0) {
        anyhow::bail!(
            "--strict: {} registry/registries behind or ahead",
            n_behind + n_ahead
        );
    }
    Ok(())
}

/// Query one registry and return (published_version, classification).
fn query_registry(reg: &Registry, repo_ver: &str) -> (Option<String>, Class) {
    let body = match fetch(reg.url, reg.extra_flags) {
        Some(b) => b,
        None => return (None, Class::Unknown),
    };
    let ver_opt = if let Some(k2) = reg.key2 {
        extract_nested(&body, reg.key, k2)
    } else {
        extract_scalar(&body, reg.key)
    };
    match ver_opt {
        Some(ver) => {
            let class = classify(repo_ver, &ver);
            (Some(ver), class)
        }
        None => (None, Class::Unknown),
    }
}
