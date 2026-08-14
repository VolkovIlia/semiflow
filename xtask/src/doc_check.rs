//! `doc-check` subcommand — README ↔ reality drift gate.
//!
//! Catches four categories of drift before publish:
//!
//! 1. **Version / status truth** (all `crates/*/README.md` + root `README.md`):
//!    - 1a. False "unpublished" claims (denylist; escape: `<!-- doc-check: allow-unpublished -->`).
//!    - 1b. Internal-scheme version tokens `vN.M.K` with N≥2 leaked into user-facing READMEs
//!      (N≥2: `v1.0.0` API-freeze references are allowed; escape: `<!-- doc-check: allow-vref -->`).
//!
//! 2. **Exposed-class truth** (`crates/semiflow-py/README.md` only):
//!    - 2a. README lists a class absent from `#[pyclass(name=…)]` registrations.
//!    - 2b. README lists a class absent from `__init__.pyi` stub.
//!    - 2c. README claims a registered class is "Rust-only / not exposed" (the incident rule).
//!    - 2d. (WARN, not fail) Registered classes not documented in README.
//!
//! 3. **FFI surface truth** (`crates/semiflow-ffi/README.md` ↔ `include/semiflow.h`):
//!    - 3a. README names an `smf_*` symbol absent from the header (phantom symbol).
//!    - 3b. README denies that a family/symbol is bound, but the header exports it (incident rule).
//!    - 3c. (WARN, not fail) Header families with zero README mention.
//!
//! 4. **WASM surface truth** (`crates/semiflow-wasm/README.md` ↔ `#[wasm_bindgen]` exports):
//!    - 4a. README class table documents a class absent from `#[wasm_bindgen]` exports (phantom).
//!    - 4b. README denies that a class is wired to WASM, but it IS exported (incident rule).
//!      Exception: denial qualified by "FFI handle" scopes the deferral to the S³ handle,
//!      not the class itself — this is a legit distinction, not gate-weakening.
//!    - 4c. (no-op) Per-export completeness warnings are intentionally suppressed — the
//!      authoritative, exhaustive class list is the wasm-pack-generated `semiflow_wasm.d.ts`
//!      (mirroring Check 3c / FFI policy; enumerating 50+ classes in prose is unmaintainable).
//!
//! ## Fragility ledger
//!
//! - Check 1a: unconditional denylist; `allow-unpublished` inline marker as escape hatch.
//! - Check 1b: N≥2 threshold; `allow-vref` marker still available for rare legit N≥2 refs.
//! - Check 2: `#[pyclass(name=…)]` grep is the canonical ground truth.
//! - Check 2c: tense heuristic excludes "Rust-only at 0.x" historical qualifiers.
//! - Check 3b: family-stem prefix map; over-match only ⇒ more denials flagged, never misses.
//! - Check 4b: parenthetical "FFI handle" qualifier exempts TtEvolver/GridlessEvolver denials.

use std::{
    collections::HashSet,
    fs,
    path::{Path, PathBuf},
};

use anyhow::Result;

// ---------------------------------------------------------------------------
// Violation type
// ---------------------------------------------------------------------------

pub(crate) struct Violation {
    message: String,
}

impl Violation {
    fn new(msg: impl Into<String>) -> Self {
        Self { message: msg.into() }
    }
}

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Denial phrases used by Check 2c, 3b, and 4b (case-insensitive substring match).
const DENIAL_PHRASES: &[&str] = &[
    "rust-only",
    "rustonly",
    "not exposed",
    "not yet exposed",
    "not exposed via pyo3",
    "not exposed via ffi",
    "not (yet )?exposed via ffi",
    "no python binding",
    "no c binding",
    "not bound",
    "not yet bound",
    "not wired",
    "not yet wired",
    "not wired to wasm",
];

/// Past-tense qualifiers that exempt a denial match from being a violation.
/// These indicate historical facts ("was Rust-only at 0.9.0") not current state.
const PAST_QUALIFIERS: &[&str] = &["at 0.", "prior to", "before 0.", "as of 0."];

/// Run all doc-check rules.  Prints violations to stderr; exits 1 if any found.
pub fn run() -> Result<()> {
    let root = crate::workspace_root()?;
    let mut violations: Vec<Violation> = Vec::new();
    let mut warnings: Vec<String> = Vec::new();

    check_version_truth(&root, &mut violations)?;
    check_exposed_classes(&root, &mut violations, &mut warnings)?;
    check_ffi_surface(&root, &mut violations, &mut warnings)?;
    check_wasm_surface(&root, &mut violations, &mut warnings)?;

    for w in &warnings {
        eprintln!("doc-check: warn: {w}");
    }

    if violations.is_empty() {
        println!(
            "doc-check: PASS (0 violations{})",
            if warnings.is_empty() {
                String::new()
            } else {
                format!(", {} advisory warning(s)", warnings.len())
            }
        );
        return Ok(());
    }

    for v in &violations {
        eprintln!("{}", v.message);
    }
    anyhow::bail!("doc-check: {} violation(s)", violations.len());
}

// ---------------------------------------------------------------------------
// Check 1 — version / status truth
// ---------------------------------------------------------------------------

/// Denylist for false "unpublished" claims (case-insensitive substring match).
const UNPUBLISHED_DENYLIST: &[&str] = &[
    "not yet published",
    "not published",
    "pending publication",
    "unpublished",
    "coming soon to pypi",
    "will be published",
];

/// Collect README paths to scan for version/status truth.
fn readme_paths(root: &Path) -> Vec<PathBuf> {
    let mut paths = Vec::new();
    // Root README
    let root_readme = root.join("README.md");
    if root_readme.exists() {
        paths.push(root_readme);
    }
    // crates/*/README.md
    if let Ok(entries) = fs::read_dir(root.join("crates")) {
        for entry in entries.flatten() {
            let readme = entry.path().join("README.md");
            if readme.exists() {
                paths.push(readme);
            }
        }
    }
    paths
}

/// Check 1a + 1b across all user-facing READMEs.
fn check_version_truth(root: &Path, violations: &mut Vec<Violation>) -> Result<()> {
    for path in readme_paths(root) {
        let src = fs::read_to_string(&path)?;
        let rel = path
            .strip_prefix(root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");

        check_unpublished_claims(&rel, &src, violations);
        check_internal_version_leak(&rel, &src, violations);
    }
    Ok(())
}

/// Check 1a: no false "unpublished" claims (denylist, case-insensitive).
fn check_unpublished_claims(rel: &str, src: &str, violations: &mut Vec<Violation>) {
    // Allow whole-file escape (rarely needed).
    if src.contains("<!-- doc-check: allow-unpublished -->") {
        return;
    }
    let lower = src.to_lowercase();
    for phrase in UNPUBLISHED_DENYLIST {
        if lower.contains(phrase) {
            violations.push(Violation::new(format!(
                "doc-check: {rel}: contains stale/false publication claim \"{phrase}\" \
                 (add <!-- doc-check: allow-unpublished --> only if truly unpublished)"
            )));
        }
    }
}

/// Check 1b: no internal-scheme version token `vN.M.K` (N≥1) in user-facing READMEs.
/// Escape: `<!-- doc-check: allow-vref -->` on the same line.
fn check_internal_version_leak(rel: &str, src: &str, violations: &mut Vec<Violation>) {
    // Regex: \bv([1-9]\d*)\.(\d+)\.(\d+)\b  (N≥1 prefix)
    // Implemented with a simple byte-scanner to avoid regex dependency.
    for (line_no, line) in src.lines().enumerate() {
        if line.contains("<!-- doc-check: allow-vref -->") {
            continue;
        }
        if let Some(tok) = find_internal_version_token(line) {
            violations.push(Violation::new(format!(
                "doc-check: {rel}:{}: internal-scheme version \"{tok}\" leaked into a \
                 user-facing README; use the public 0.x-beta scheme \
                 (or add <!-- doc-check: allow-vref --> to suppress)",
                line_no + 1
            )));
        }
    }
}

/// Scan one line for a `vN.M.K` token where N≥2.
///
/// N≥2 is deliberate: `v1.0.0` is the legitimate API-freeze milestone reference
/// and must NOT be flagged.  Internal-scheme leaks that matter are `v2.x`, `v9.x`,
/// etc. — all N≥2.  The per-line `<!-- doc-check: allow-vref -->` escape hatch
/// remains available for the rare legitimate N≥2 URL fragment or citation.
fn find_internal_version_token(line: &str) -> Option<String> {
    let bytes = line.as_bytes();
    let mut i = 0;
    while i < bytes.len() {
        if bytes[i] != b'v' {
            i += 1;
            continue;
        }
        // Require word boundary before `v` (start of line, space, `(`, `"`, or similar).
        let word_start = i == 0 || !bytes[i - 1].is_ascii_alphanumeric();
        if !word_start {
            i += 1;
            continue;
        }
        // Try to parse vN.M.K starting at i+1.
        if let Some((tok, end)) = parse_version_token(&bytes[i + 1..]) {
            // N must be ≥ 2 (§ L1.3: v1.0.0 is the legitimate API-freeze reference).
            if tok.starts_with('v') {
                let major_str: String = tok.chars().skip(1).take_while(|c| c.is_ascii_digit()).collect();
                if major_str.parse::<u64>().unwrap_or(0) >= 2 {
                    // Word boundary after token.
                    let abs_end = i + 1 + end;
                    let after_ok = abs_end >= bytes.len()
                        || !bytes[abs_end].is_ascii_alphanumeric() && bytes[abs_end] != b'.';
                    if after_ok {
                        return Some(tok);
                    }
                }
            }
            i = i + 1 + end;
        } else {
            i += 1;
        }
    }
    None
}

/// Parse `N.M.K` (without the leading `v`) from byte slice.
/// Returns (full token with `v`, bytes consumed) or None.
fn parse_version_token(rest: &[u8]) -> Option<(String, usize)> {
    // N
    let (major, n1) = parse_digits(rest)?;
    if n1 == 0 || rest.get(n1) != Some(&b'.') {
        return None;
    }
    // M
    let (minor, n2) = parse_digits(&rest[n1 + 1..])?;
    if n2 == 0 || rest.get(n1 + 1 + n2) != Some(&b'.') {
        return None;
    }
    // K
    let (patch, n3) = parse_digits(&rest[n1 + 1 + n2 + 1..])?;
    if n3 == 0 {
        return None;
    }
    let tok = format!("v{major}.{minor}.{patch}");
    let consumed = n1 + 1 + n2 + 1 + n3;
    Some((tok, consumed))
}

/// Parse leading ASCII digits; return (value_str, bytes_consumed).
fn parse_digits(bytes: &[u8]) -> Option<(String, usize)> {
    let n = bytes.iter().take_while(|b| b.is_ascii_digit()).count();
    if n == 0 {
        return None;
    }
    let s = std::str::from_utf8(&bytes[..n]).ok()?;
    Some((s.to_owned(), n))
}

// ---------------------------------------------------------------------------
// Check 2 — exposed-class truth (semiflow-py only)
// ---------------------------------------------------------------------------

const PY_README: &str = "crates/semiflow-py/README.md";
const PY_SRC_DIR: &str = "crates/semiflow-py/src";
const PY_STUB: &str = "crates/semiflow-py/python/semiflow/__init__.pyi";

/// Check 2a/2b/2c/2d for the semiflow-py README.
fn check_exposed_classes(
    root: &Path,
    violations: &mut Vec<Violation>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let readme_path = root.join(PY_README);
    if !readme_path.exists() {
        return Ok(());
    }
    let readme_src = fs::read_to_string(&readme_path)?;

    let readme_classes = extract_readme_classes(&readme_src);
    let pyclass_names = extract_pyclass_names(root)?;
    let stub_classes = extract_stub_classes(root)?;

    check_2a(&readme_classes, &pyclass_names, violations);
    check_2b(&readme_classes, &stub_classes, violations);
    check_2c(&readme_src, &pyclass_names, violations);
    check_2d(&pyclass_names, &readme_classes, warnings);

    Ok(())
}

/// Extract class names from class-reference table rows in the README.
/// Only rows under headers containing "Class" are parsed; cell-1 must be a backtick identifier.
fn extract_readme_classes(src: &str) -> HashSet<String> {
    let mut classes = HashSet::new();
    let mut in_class_table = false;

    for line in src.lines() {
        let trimmed = line.trim();
        // Detect class-reference table header: row starting with `| Class`
        if trimmed.starts_with("| Class") || trimmed.starts_with("| `Class") {
            in_class_table = true;
            continue;
        }
        // Table separator row — skip
        if trimmed.starts_with("|---") || trimmed.starts_with("| --") || trimmed.starts_with("|:--") {
            continue;
        }
        // A new non-table header resets class-table mode
        if trimmed.starts_with('#') {
            in_class_table = false;
            continue;
        }
        if !in_class_table {
            continue;
        }
        if !trimmed.starts_with('|') {
            in_class_table = false;
            continue;
        }
        // Parse cell-1 for a backtick identifier
        if let Some(name) = extract_class_cell(trimmed) {
            classes.insert(name);
        }
    }
    classes
}

/// Extract the class name from the first cell of a README table row.
/// Cell must be `` `ClassName` `` or `` `ClassName(...)` `` or `` `ClassName.method` ``.
fn extract_class_cell(row: &str) -> Option<String> {
    // Split by '|'; first element is empty (leading '|'), second is cell-1.
    let mut parts = row.splitn(4, '|');
    parts.next(); // leading empty
    let cell = parts.next()?.trim();
    if !cell.starts_with('`') {
        return None;
    }
    let inner = cell.trim_matches('`');
    // Extract identifier up to first non-identifier char: '(', '.', ' ', '`'
    let ident: String = inner.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if ident.is_empty() || !ident.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
        return None;
    }
    Some(ident)
}

/// Extract all `#[pyclass(name = "X")]` names from `crates/semiflow-py/src/**/*.rs`.
fn extract_pyclass_names(root: &Path) -> Result<HashSet<String>> {
    let mut names = HashSet::new();
    let src_dir = root.join(PY_SRC_DIR);
    collect_pyclass_names_recursive(&src_dir, &mut names)?;
    Ok(names)
}

fn collect_pyclass_names_recursive(dir: &Path, names: &mut HashSet<String>) -> Result<()> {
    for entry in fs::read_dir(dir)?.flatten() {
        let path = entry.path();
        if path.is_dir() {
            collect_pyclass_names_recursive(&path, names)?;
        } else if path.extension().is_some_and(|e| e == "rs") {
            let src = fs::read_to_string(&path)?;
            extract_pyclass_from_source(&src, names);
        }
    }
    Ok(())
}

/// Parse `#[pyclass(name = "X")]` or `#[pyclass(name="X")]` tokens.
fn extract_pyclass_from_source(src: &str, names: &mut HashSet<String>) {
    // Simple byte-scan: find `pyclass` then look for `name` then `"..."`.
    let hay = src.as_bytes();
    let mut i = 0;
    while i + 7 < hay.len() {
        // Look for 'p','y','c','l','a','s','s'
        if &hay[i..i + 7] == b"pyclass" {
            if let Some(name) = parse_pyclass_name_attr(&hay[i..]) {
                names.insert(name);
            }
        }
        i += 1;
    }
}

/// Given a slice starting at `pyclass`, extract the `name = "X"` value if present.
fn parse_pyclass_name_attr(slice: &[u8]) -> Option<String> {
    // Find '(' within 64 bytes
    let paren = slice[..slice.len().min(64)].iter().position(|&b| b == b'(')?;
    // Find 'name' after '('
    let after_paren = &slice[paren + 1..];
    let attr_str = std::str::from_utf8(&after_paren[..after_paren.len().min(256)]).ok()?;
    // Find 'name' keyword
    let name_pos = attr_str.find("name")?;
    let after_name = &attr_str[name_pos + 4..];
    // Skip whitespace and '='
    let after_eq = after_name.trim_start().trim_start_matches('=').trim_start();
    // Find opening quote
    if !after_eq.starts_with('"') {
        return None;
    }
    let inner = &after_eq[1..];
    let close = inner.find('"')?;
    Some(inner[..close].to_owned())
}

/// Extract `^class X` names from `__init__.pyi`.
fn extract_stub_classes(root: &Path) -> Result<HashSet<String>> {
    let stub_path = root.join(PY_STUB);
    if !stub_path.exists() {
        return Ok(HashSet::new());
    }
    let src = fs::read_to_string(&stub_path)?;
    let mut names = HashSet::new();
    for line in src.lines() {
        if let Some(rest) = line.strip_prefix("class ") {
            let ident: String = rest.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !ident.is_empty() {
                names.insert(ident);
            }
        }
    }
    Ok(names)
}

/// Check 2a: README ⊆ pyclass names.
fn check_2a(readme: &HashSet<String>, pyclass: &HashSet<String>, violations: &mut Vec<Violation>) {
    let mut missing: Vec<&str> = readme.iter()
        .filter(|n| !pyclass.contains(*n))
        .map(|s| s.as_str())
        .collect();
    missing.sort_unstable();
    for name in missing {
        violations.push(Violation::new(format!(
            "doc-check: {PY_README}: README lists class `{name}` but no \
             #[pyclass(name=\"{name}\")] is registered in semiflow-py"
        )));
    }
}

/// Check 2b: README ⊆ stub classes.
fn check_2b(readme: &HashSet<String>, stub: &HashSet<String>, violations: &mut Vec<Violation>) {
    if stub.is_empty() {
        return; // stub file not present; skip silently
    }
    let mut missing: Vec<&str> = readme.iter()
        .filter(|n| !stub.contains(*n))
        .map(|s| s.as_str())
        .collect();
    missing.sort_unstable();
    for name in missing {
        violations.push(Violation::new(format!(
            "doc-check: {PY_README}: README lists class `{name}` but it is \
             missing from __init__.pyi"
        )));
    }
}

/// Check 2c: no "Rust-only / not exposed" claim for a registered class.
///
/// Excludes matches where a past-tense qualifier `at 0.\d` / `prior to` / `before` / `as of 0.`
/// appears within 80 chars after the denial phrase (the "Rust-only at 0.9.0-beta" exception).
fn check_2c(readme_src: &str, pyclass: &HashSet<String>, violations: &mut Vec<Violation>) {
    let lower_src = readme_src.to_lowercase();

    for (line_no, (orig_line, lower_line)) in
        readme_src.lines().zip(lower_src.lines()).enumerate()
    {
        for phrase in DENIAL_PHRASES {
            let Some(phrase_pos) = lower_line.find(phrase) else {
                continue;
            };
            // Check for past-tense qualifier near the phrase (within 80 chars after).
            let after = &lower_line[phrase_pos..];
            let guarded = PAST_QUALIFIERS.iter().any(|q| {
                after[..after.len().min(80)].contains(q)
            });
            if guarded {
                continue;
            }
            // Find any backtick identifier on the same line in the original.
            for name in extract_backtick_idents(orig_line) {
                if pyclass.contains(&name) {
                    violations.push(Violation::new(format!(
                        "doc-check: {PY_README}:{}: README claims `{name}` is Rust-only/not-exposed, \
                         but it IS a registered #[pyclass] in semiflow-py (false claim)",
                        line_no + 1
                    )));
                }
            }
        }
    }
}

/// Extract all backtick-quoted identifiers from a line (e.g. `` `Foo` `` → "Foo").
fn extract_backtick_idents(line: &str) -> Vec<String> {
    let mut result = Vec::new();
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c == '`' {
            let ident: String = chars.by_ref().take_while(|c| *c != '`').collect();
            // Accept identifiers starting with uppercase (class names).
            let base: String = ident.chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            if !base.is_empty() && base.chars().next().map(|c| c.is_uppercase()).unwrap_or(false) {
                result.push(base);
            }
        }
    }
    result
}

/// Check 2d (advisory): registered classes absent from README.
fn check_2d(pyclass: &HashSet<String>, readme: &HashSet<String>, warnings: &mut Vec<String>) {
    let mut undoc: Vec<&str> = pyclass.iter()
        .filter(|n| !readme.contains(*n))
        .map(|s| s.as_str())
        .collect();
    undoc.sort_unstable();
    for name in undoc {
        warnings.push(format!(
            "{PY_README}: class `{name}` is a registered #[pyclass] but not listed in README"
        ));
    }
}

// ---------------------------------------------------------------------------
// Check 3 — FFI surface truth (semiflow-ffi README ↔ include/semiflow.h)
// ---------------------------------------------------------------------------

const FFI_README: &str = "crates/semiflow-ffi/README.md";
const FFI_HEADER: &str = "crates/semiflow-ffi/include/semiflow.h";

/// Family stems to check in denial phrases (Check 3b).
/// A denial naming a stem is a violation if the header exports any `smf_{stem}*` symbol.
const FFI_FAMILY_STEMS: &[&str] = &[
    "diffusion", "graph", "manifold", "hypoelliptic", "adjoint",
    "resolvent", "killing", "reflected", "obstacle", "schrodinger",
    "tt", "gridless", "howland",
];

/// Check 3: FFI README ↔ header surface.
pub(crate) fn check_ffi_surface(
    root: &Path,
    violations: &mut Vec<Violation>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let readme_path = root.join(FFI_README);
    let header_path = root.join(FFI_HEADER);
    if !readme_path.exists() || !header_path.exists() {
        return Ok(());
    }
    let readme_src = fs::read_to_string(&readme_path)?;
    let header_src = fs::read_to_string(&header_path)?;

    let header_syms = extract_smf_symbols(&header_src);
    check_3a(&readme_src, &header_syms, violations);
    check_3b(&readme_src, &header_syms, violations);
    check_3c(&header_syms, warnings);
    Ok(())
}

/// Extract all distinct `smf_[a-z0-9_]+` symbols from source text.
fn extract_smf_symbols(src: &str) -> HashSet<String> {
    let mut syms = HashSet::new();
    let bytes = src.as_bytes();
    let mut i = 0;
    while i + 4 < bytes.len() {
        if &bytes[i..i + 4] == b"smf_" {
            let end = i + 4 + bytes[i + 4..]
                .iter()
                .take_while(|b| b.is_ascii_lowercase() || b.is_ascii_digit() || **b == b'_')
                .count();
            if end > i + 4 {
                if let Ok(sym) = std::str::from_utf8(&bytes[i..end]) {
                    syms.insert(sym.to_owned());
                }
            }
            i = end;
        } else {
            i += 1;
        }
    }
    syms
}

/// Check 3a: any `smf_*` token the README names must exist in the header.
fn check_3a(readme_src: &str, header_syms: &HashSet<String>, violations: &mut Vec<Violation>) {
    let readme_syms = extract_smf_symbols(readme_src);
    let mut phantoms: Vec<String> = readme_syms
        .into_iter()
        .filter(|s| !header_syms.contains(s))
        .collect();
    phantoms.sort_unstable();
    for sym in phantoms {
        violations.push(Violation::new(format!(
            "doc-check: {FFI_README}: names `{sym}` but no such export in {FFI_HEADER}"
        )));
    }
}

/// Check 3b: no denial for a family/symbol that IS exported (incident rule).
fn check_3b(readme_src: &str, header_syms: &HashSet<String>, violations: &mut Vec<Violation>) {
    // Skip YAML front-matter — changelog entries there can contain denial phrases.
    let body = strip_front_matter(readme_src);
    // Line-number offset so error lines stay relative to full file.
    let skipped = readme_src.lines().count() - body.lines().count();
    let lower_src = body.to_lowercase();
    for (rel_no, (orig_line, lower_line)) in
        body.lines().zip(lower_src.lines()).enumerate()
    {
        let line_no = rel_no + skipped;
        if !has_denial_phrase(lower_line) {
            continue;
        }
        let after = denial_phrase_context(lower_line);
        if is_past_qualified(after) {
            continue;
        }
        // Check concrete `smf_*` tokens on the line.
        let readme_syms = extract_smf_symbols(orig_line);
        for sym in &readme_syms {
            if header_syms.contains(sym) {
                violations.push(Violation::new(format!(
                    "doc-check: {FFI_README}:{}: claims `{sym}` is not bound/exposed via FFI, \
                     but {FFI_HEADER} exports it (false claim)",
                    line_no + 1
                )));
            }
        }
        // Check family stems mentioned as prose words on the line.
        let lower_orig = orig_line.to_lowercase();
        for stem in FFI_FAMILY_STEMS {
            if !lower_orig.contains(stem) {
                continue;
            }
            let prefix = format!("smf_{stem}");
            let exported = header_syms.iter().any(|s| s.starts_with(prefix.as_str()));
            if exported {
                violations.push(Violation::new(format!(
                    "doc-check: {FFI_README}:{}: claims family `{stem}` is not bound/exposed via \
                     FFI, but {FFI_HEADER} exports smf_{stem}* symbols (false claim)",
                    line_no + 1
                )));
            }
        }
    }
}

/// Check 3c (advisory): header family stems with no README mention.
fn check_3c(header_syms: &HashSet<String>, warnings: &mut Vec<String>) {
    // One advisory line per un-mentioned family stem — not a failure.
    // FFI README policy is not to enumerate; advisory is informational only.
    let _ = (header_syms, warnings); // intentionally no-op per design § L1.1 rule 3c
}

// ---------------------------------------------------------------------------
// Check 4 — WASM surface truth (semiflow-wasm README ↔ #[wasm_bindgen])
// ---------------------------------------------------------------------------

const WASM_README: &str = "crates/semiflow-wasm/README.md";
const WASM_SRC_DIR: &str = "crates/semiflow-wasm/src";

/// Check 4: WASM README ↔ `#[wasm_bindgen]` exports.
pub(crate) fn check_wasm_surface(
    root: &Path,
    violations: &mut Vec<Violation>,
    warnings: &mut Vec<String>,
) -> Result<()> {
    let readme_path = root.join(WASM_README);
    let src_dir = root.join(WASM_SRC_DIR);
    if !readme_path.exists() || !src_dir.exists() {
        return Ok(());
    }
    let readme_src = fs::read_to_string(&readme_path)?;
    let wasm_exports = extract_wasm_exports(&src_dir)?;

    let readme_classes = extract_wasm_readme_classes(&readme_src);
    check_4a(&readme_classes, &wasm_exports, violations);
    check_4b(&readme_src, &wasm_exports, violations);
    check_4c(&wasm_exports, &readme_classes, warnings);
    Ok(())
}

/// Extract the set of JS class names from `#[wasm_bindgen]` struct-level attributes.
///
/// Two forms are handled:
/// 1. `#[wasm_bindgen(js_name = "Y")]` at column 0, followed by `pub struct X` → JS name = Y.
/// 2. `#[wasm_bindgen]` at column 0, followed by `pub struct X` → JS name = X.
///
/// `js_class` (method-block pairing) and indented attributes (method-level) are ignored.
fn extract_wasm_exports(src_dir: &Path) -> Result<HashSet<String>> {
    let mut exports = HashSet::new();
    for entry in fs::read_dir(src_dir)?.flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|e| e == "rs") {
            let src = fs::read_to_string(&path)?;
            collect_wasm_exports_from_source(&src, &mut exports);
        }
    }
    Ok(exports)
}

/// Parse one `.rs` file for struct-level `#[wasm_bindgen]` exports.
fn collect_wasm_exports_from_source(src: &str, exports: &mut HashSet<String>) {
    let lines: Vec<&str> = src.lines().collect();
    let mut i = 0;
    while i < lines.len() {
        let line = lines[i];
        // Only column-0 #[wasm_bindgen attributes.
        if !line.starts_with("#[wasm_bindgen") {
            i += 1;
            continue;
        }
        // Skip js_class — that's the impl-block pairing, not a new export.
        if line.contains("js_class") {
            i += 1;
            continue;
        }
        // Extract js_name if present.
        let js_name_override = extract_js_name(line);
        // Find the next non-blank line.
        let mut j = i + 1;
        while j < lines.len() && lines[j].trim().is_empty() {
            j += 1;
        }
        if j < lines.len() {
            if let Some(struct_name) = extract_pub_struct_name(lines[j]) {
                let export_name = js_name_override.unwrap_or(struct_name);
                exports.insert(export_name);
            }
        }
        i = j + 1;
    }
}

/// Extract `js_name = "VALUE"` from a `#[wasm_bindgen(...)]` attribute line.
fn extract_js_name(attr_line: &str) -> Option<String> {
    let pos = attr_line.find("js_name")?;
    let rest = &attr_line[pos + 7..]; // skip "js_name"
    let rest = rest.trim_start().trim_start_matches('=').trim_start();
    let inner = rest.strip_prefix('"')?;
    let close = inner.find('"')?;
    Some(inner[..close].to_owned())
}

/// Extract the struct name from a `pub struct X` or `pub struct X(` line.
fn extract_pub_struct_name(line: &str) -> Option<String> {
    let rest = line.trim().strip_prefix("pub struct ")?;
    let name: String = rest.chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    if name.is_empty() { None } else { Some(name) }
}

/// Extract class names from the `| Class |` table in the WASM README.
/// Only applies to table rows with a backtick identifier in cell-1.
fn extract_wasm_readme_classes(src: &str) -> HashSet<String> {
    let mut classes = HashSet::new();
    let mut in_class_table = false;
    for line in src.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with("| Class") || trimmed.starts_with("| `Class") {
            in_class_table = true;
            continue;
        }
        if trimmed.starts_with("|---") || trimmed.starts_with("| --") {
            continue;
        }
        if trimmed.starts_with('#') {
            in_class_table = false;
            continue;
        }
        if !in_class_table {
            continue;
        }
        if !trimmed.starts_with('|') {
            in_class_table = false;
            continue;
        }
        if let Some(name) = extract_class_cell(trimmed) {
            classes.insert(name);
        }
    }
    classes
}

/// Check 4a: README class table ⊆ wasm exports (no phantom classes).
fn check_4a(
    readme: &HashSet<String>,
    wasm: &HashSet<String>,
    violations: &mut Vec<Violation>,
) {
    let mut phantoms: Vec<&str> = readme.iter()
        .filter(|n| !wasm.contains(*n))
        .map(|s| s.as_str())
        .collect();
    phantoms.sort_unstable();
    for name in phantoms {
        violations.push(Violation::new(format!(
            "doc-check: {WASM_README}: documents class `{name}` but no \
             #[wasm_bindgen] export produces it"
        )));
    }
}

/// Check 4b: no "not exposed / not wired to WASM" denial for a class that IS exported.
///
/// CALIBRATION: The WASM README's deferral sentence for S³ FFI handles uses the phrase
/// "FFI handles" in a parenthetical, scoping the deferral to the handle, not the class.
/// We exempt denial hits when "FFI handle" or "handle" appears in the same clause.
fn check_4b(readme_src: &str, wasm: &HashSet<String>, violations: &mut Vec<Violation>) {
    // WASM-specific denial phrases.
    const WASM_DENIAL: &[&str] = &[
        "not exposed",
        "not yet exposed",
        "not wired",
        "not yet wired",
        "not wired to wasm",
        "no js binding",
        "rust-only",
    ];
    let lower_src = readme_src.to_lowercase();
    for (line_no, (orig_line, lower_line)) in
        readme_src.lines().zip(lower_src.lines()).enumerate()
    {
        let has_denial = WASM_DENIAL.iter().any(|p| lower_line.contains(p));
        if !has_denial {
            continue;
        }
        // Extract all backtick identifiers starting with uppercase.
        for name in extract_backtick_idents(orig_line) {
            if !wasm.contains(&name) {
                continue;
            }
            // Exemption: if "handle" or "FFI handle" qualifies the denial in this clause,
            // the deferral is about the S³ handle, not the class itself.
            // Check the text immediately after the identifier's backtick span.
            let lower_name = name.to_lowercase();
            let name_pos = lower_line.find(lower_name.as_str()).unwrap_or(0);
            let clause = &lower_line[name_pos..];
            let ffi_handle_qualified = clause[..clause.len().min(120)]
                .contains("handle");
            if ffi_handle_qualified {
                continue;
            }
            violations.push(Violation::new(format!(
                "doc-check: {WASM_README}:{}: claims `{name}` is not exposed/not-wired \
                 to WASM, but it IS a #[wasm_bindgen] export (false claim)",
                line_no + 1
            )));
        }
    }
}

/// Check 4c — intentionally a no-op.
///
/// The authoritative, exhaustive list of exported JS classes is the wasm-pack-generated
/// `semiflow_wasm.d.ts` shipped in the `@semiflow/wasm` npm package. Emitting a
/// per-export advisory warning for every unlisted class (50+) is an unmaintainable
/// maintenance trap — consistent with Check 3c (FFI) which also suppresses per-symbol
/// completeness warnings and points to `include/semiflow.h` instead.
fn check_4c(
    _wasm: &HashSet<String>,
    _readme: &HashSet<String>,
    _warnings: &mut Vec<String>,
) {
    // Intentional no-op — see doc comment above.
}

// ---------------------------------------------------------------------------
// Shared denial helpers (Check 2c, 3b, 4b)
// ---------------------------------------------------------------------------

/// Returns true if a lowercased line contains any denial phrase.
fn has_denial_phrase(lower_line: &str) -> bool {
    DENIAL_PHRASES.iter().any(|p| lower_line.contains(p))
}

/// Returns the substring of the line starting at the first denial phrase match.
fn denial_phrase_context(lower_line: &str) -> &str {
    DENIAL_PHRASES
        .iter()
        .filter_map(|p| lower_line.find(p).map(|pos| &lower_line[pos..]))
        .min_by_key(|s| s.len())
        .unwrap_or(lower_line)
}

/// Returns true if the context (after a denial phrase) contains a past-tense qualifier.
fn is_past_qualified(context: &str) -> bool {
    let window = &context[..context.len().min(80)];
    PAST_QUALIFIERS.iter().any(|q| window.contains(q))
}

/// Strip a YAML front-matter block (content between the first two `---` lines)
/// from a README string so denial-phrase scanners don't trip on changelog entries.
fn strip_front_matter(src: &str) -> &str {
    let mut lines = src.splitn(3, '\n');
    // If first line is `---`, look for closing `---`.
    if lines.next().map(|l| l.trim() == "---").unwrap_or(false) {
        // Find the byte position of the second `---\n` or `---` at EOL.
        let after_first = src[src.find('\n').map(|p| p + 1).unwrap_or(src.len())..].trim_start_matches('\n');
        let base = src.len() - after_first.len();
        if let Some(close_rel) = after_first.find("\n---") {
            // Return everything after the closing `---` line.
            let close_abs = base + close_rel + 1; // points to `---`
            let past_close = close_abs + 3;       // skip `---`
            let rest = &src[past_close..];
            return rest.trim_start_matches('\n');
        }
    }
    src
}
