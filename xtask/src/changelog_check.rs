//! `changelog-check` subcommand — CHANGELOG ↔ workspace-version gate.
//!
//! Reads the workspace version from `[workspace.package] version` in the root
//! `Cargo.toml` and asserts that `CHANGELOG.md` contains a non-empty section
//! for that exact version.
//!
//! ## Rules
//!
//! 1. A `## [<version>]` heading must exist (date separator ` - ` or ` — ` are both accepted).
//! 2. The section body (lines until the next `## [` heading) must contain at least one
//!    non-blank, non-heading content line.
//!
//! ## Exit codes
//!
//! | Exit | Meaning |
//! |------|---------|
//! | `0`  | Section present and non-empty |
//! | `1`  | Missing or stub-only section |

use std::{fs, path::PathBuf};

use anyhow::Result;

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

pub fn run() -> Result<()> {
    let root = crate::workspace_root()?;
    let version = read_workspace_version(&root)?;
    let changelog_path = root.join("CHANGELOG.md");
    let changelog = read_changelog(&changelog_path)?;

    match find_version_section(&changelog, &version) {
        SectionResult::NotFound => {
            anyhow::bail!(
                "changelog-check: CHANGELOG.md has no '## [{}]' heading — \
                 add release notes before publishing",
                version
            );
        }
        SectionResult::EmptyStub => {
            anyhow::bail!(
                "changelog-check: CHANGELOG.md has a '## [{}]' heading but its \
                 section body is empty or stub-only — add release notes before publishing",
                version
            );
        }
        SectionResult::NonEmpty => {
            println!("changelog-check: PASS — ## [{version}] section is non-empty");
            Ok(())
        }
    }
}

// ---------------------------------------------------------------------------
// Version extraction
// ---------------------------------------------------------------------------

/// Read `[workspace.package] version = "..."` from root `Cargo.toml`.
fn read_workspace_version(root: &std::path::Path) -> Result<String> {
    let cargo_toml = root.join("Cargo.toml");
    let src = fs::read_to_string(&cargo_toml)
        .map_err(|e| anyhow::anyhow!("changelog-check: cannot read Cargo.toml: {e}"))?;
    parse_workspace_version(&src)
        .ok_or_else(|| anyhow::anyhow!(
            "changelog-check: [workspace.package] version not found in Cargo.toml"
        ))
}

/// Parse `version = "X.Y.Z"` that follows `[workspace.package]`.
fn parse_workspace_version(src: &str) -> Option<String> {
    let mut in_workspace_pkg = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed == "[workspace.package]" {
            in_workspace_pkg = true;
            continue;
        }
        // Stop at the next `[` section header.
        if in_workspace_pkg && trimmed.starts_with('[') {
            break;
        }
        if in_workspace_pkg {
            if let Some(ver) = extract_version_field(trimmed) {
                return Some(ver);
            }
        }
    }
    None
}

/// Extract `version = "VALUE"` from a TOML line.
fn extract_version_field(line: &str) -> Option<String> {
    let rest = line.strip_prefix("version")?.trim();
    let rest = rest.strip_prefix('=')?.trim();
    let inner = rest.strip_prefix('"')?.strip_suffix('"')?;
    Some(inner.to_owned())
}

// ---------------------------------------------------------------------------
// CHANGELOG parsing
// ---------------------------------------------------------------------------

fn read_changelog(path: &PathBuf) -> Result<String> {
    fs::read_to_string(path)
        .map_err(|e| anyhow::anyhow!("changelog-check: cannot read CHANGELOG.md: {e}"))
}

#[derive(Debug, PartialEq)]
enum SectionResult {
    /// No `## [<version>]` heading found.
    NotFound,
    /// Heading found but section body has no content lines.
    EmptyStub,
    /// Heading found and section has at least one content line.
    NonEmpty,
}

/// Find the `## [version]` heading and check that its body is non-empty.
///
/// Accepts both ` - ` and ` — ` as date separators after the version.
fn find_version_section(src: &str, version: &str) -> SectionResult {
    let target_prefix = format!("## [{version}]");
    let mut in_section = false;
    let mut has_content = false;

    for line in src.lines() {
        if !in_section {
            // Match heading: must start with `## [<version>]`
            // Accept optional ` - DATE` or ` — DATE` suffix.
            if line.starts_with(target_prefix.as_str()) {
                let rest = &line[target_prefix.len()..];
                // Must be end-of-heading or a separator (space, dash, em-dash)
                if rest.is_empty() || rest.starts_with(" -") || rest.starts_with(" —") {
                    in_section = true;
                }
            }
        } else {
            // End of section: another `## [` heading.
            if line.starts_with("## [") {
                break;
            }
            // Content line: non-blank and not a `#` heading.
            let trimmed = line.trim();
            if !trimmed.is_empty() && !trimmed.starts_with('#') {
                has_content = true;
                break;
            }
        }
    }

    if !in_section {
        SectionResult::NotFound
    } else if !has_content {
        SectionResult::EmptyStub
    } else {
        SectionResult::NonEmpty
    }
}

// ---------------------------------------------------------------------------
// Unit tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_workspace_version() {
        let toml = r#"
[workspace]
members = ["crates/*"]

[workspace.package]
version = "0.12.1-beta"
edition = "2021"
"#;
        assert_eq!(
            parse_workspace_version(toml),
            Some("0.12.1-beta".to_owned())
        );
    }

    #[test]
    fn test_section_found_non_empty() {
        let log = "## [0.12.1-beta] - 2026-07-15\n### Fixed\n- Some fix\n## [0.12.0-beta]\n";
        assert_eq!(
            find_version_section(log, "0.12.1-beta"),
            SectionResult::NonEmpty
        );
    }

    #[test]
    fn test_section_found_em_dash_separator() {
        let log = "## [1.0.0] — 2026-01-01\nSome content here\n";
        assert_eq!(
            find_version_section(log, "1.0.0"),
            SectionResult::NonEmpty
        );
    }

    #[test]
    fn test_section_not_found() {
        let log = "## [0.11.0-beta] - 2026-01-01\n- old stuff\n";
        assert_eq!(
            find_version_section(log, "0.12.1-beta"),
            SectionResult::NotFound
        );
    }

    #[test]
    fn test_section_empty_stub() {
        let log = "## [0.12.1-beta] - 2026-07-15\n## [0.12.0-beta]\n- old\n";
        assert_eq!(
            find_version_section(log, "0.12.1-beta"),
            SectionResult::EmptyStub
        );
    }
}
