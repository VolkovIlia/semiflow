//! `latency-gate` subcommand — L-gate latency floor harness (ADR-0068 Track 2).
//!
//! ## Usage
//!
//! ```text
//! cargo run -p xtask -- latency-gate <gate_id>
//! cargo run -p xtask -- latency-gate --all
//!
//! Options:
//!   <gate_id>                    Run a single named gate (e.g. L_CEV_PTICK)
//!   --all                        Run every gate in contracts/semiflow-core.properties.yaml
//!   --hardware-profile <profile> Override hardware profile (default: i7-12700K)
//!   --bench-args "..."           Extra args passed to the underlying bench invocation
//!   --mock-input <path>          Substitute bench output with a fixture JSONL file (for tests)
//!   --help                       Print this help
//! ```
//!
//! ## v2.6 behaviour (ADVISORY)
//!
//! Always exits 0. Prints `L-GATE PASS` or `L-GATE WARN` per percentile.
//! TODO(v2.7): promote to `L-GATE ERR` + exit 1 when severity == RELEASE_BLOCKING.

use std::{path::PathBuf, process};

use anyhow::{bail, Result};

// ---------------------------------------------------------------------------
// Public entry point
// ---------------------------------------------------------------------------

/// Parse args and dispatch latency-gate subcommand.
pub fn run(args: &[String]) -> Result<()> {
    let cfg = CliArgs::parse(args)?;
    let root = crate::workspace_root()?;
    let props_path = root.join("contracts").join("semiflow-core.properties.yaml");

    let props_src = std::fs::read_to_string(&props_path)
        .map_err(|e| anyhow::anyhow!("cannot read {}: {e}", props_path.display()))?;

    let gates = parse_latency_gates(&props_src)?;

    if gates.is_empty() {
        bail!("no latency_gates: entries found in properties.yaml");
    }

    let selected = select_gates(&gates, &cfg)?;
    let mut any_warn = false;
    let mut any_blocking_breach = false;

    for gate in selected {
        let (warned, blocking) = run_one_gate(gate, &cfg, &root)?;
        any_warn |= warned;
        any_blocking_breach |= blocking;
    }

    if any_blocking_breach {
        eprintln!("L-GATE: one or more RELEASE_BLOCKING gates breached (advisory=false) — exit 1");
        process::exit(1);
    }
    if any_warn {
        eprintln!("L-GATE: all gates pass (blocking or advisory per profile, v2.7)");
    } else {
        println!("L-GATE: all selected gates PASS");
    }
    // v2.7 active enforcement per ADR-0069 + math.md §3.6.bis.7
    Ok(())
}

// ---------------------------------------------------------------------------
// CLI args
// ---------------------------------------------------------------------------

struct CliArgs {
    gate_id: Option<String>,
    all: bool,
    hardware_profile: String,
    bench_args: Vec<String>,
    mock_input: Option<PathBuf>,
}

impl CliArgs {
    fn parse(args: &[String]) -> Result<Self> {
        let mut gate_id: Option<String> = None;
        let mut all = false;
        let mut hardware_profile = "i7-12700K".to_owned();
        let mut bench_args: Vec<String> = Vec::new();
        let mut mock_input: Option<PathBuf> = None;

        let mut i = 0;
        while i < args.len() {
            match args[i].as_str() {
                "--all" => {
                    all = true;
                }
                "--hardware-profile" => {
                    i += 1;
                    hardware_profile = args
                        .get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--hardware-profile needs a value"))?;
                }
                "--bench-args" => {
                    i += 1;
                    let raw = args
                        .get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--bench-args needs a value"))?;
                    bench_args = raw.split_whitespace().map(str::to_owned).collect();
                }
                "--mock-input" => {
                    i += 1;
                    let p = args
                        .get(i)
                        .cloned()
                        .ok_or_else(|| anyhow::anyhow!("--mock-input needs a path"))?;
                    mock_input = Some(PathBuf::from(p));
                }
                "--help" | "-h" => {
                    print_help();
                    process::exit(0);
                }
                other if other.starts_with('-') => {
                    bail!("unknown flag '{other}'; run with --help for usage");
                }
                id => {
                    if gate_id.is_some() {
                        bail!("unexpected positional arg '{id}' (gate_id already set)");
                    }
                    gate_id = Some(id.to_owned());
                }
            }
            i += 1;
        }

        if !all && gate_id.is_none() {
            bail!("latency-gate: specify a <gate_id> or --all");
        }
        Ok(Self {
            gate_id,
            all,
            hardware_profile,
            bench_args,
            mock_input,
        })
    }
}

fn print_help() {
    println!("cargo run -p xtask -- latency-gate <gate_id> [OPTIONS]");
    println!();
    println!("ARGS:");
    println!("  <gate_id>                    Named gate (e.g. L_CEV_PTICK)");
    println!();
    println!("OPTIONS:");
    println!("  --all                        Run every gate in properties.yaml");
    println!("  --hardware-profile <profile> Hardware profile key (default: i7-12700K)");
    println!("  --bench-args \"...\"           Extra args for the bench invocation");
    println!("  --mock-input <path>          Substitute bench with fixture JSONL (for tests)");
    println!("  --help                       Print this help");
    println!();
    println!("v2.7 behaviour: RELEASE_BLOCKING + advisory=false → exit 1 on breach.");
    println!("advisory=true (default) profiles always exit 0 with warnings.");
}

// ---------------------------------------------------------------------------
// Gate selection
// ---------------------------------------------------------------------------

fn select_gates<'a>(gates: &'a [LGate], cfg: &CliArgs) -> Result<Vec<&'a LGate>> {
    if cfg.all {
        return Ok(gates.iter().collect());
    }
    let id = cfg.gate_id.as_deref().unwrap();
    let found = gates.iter().find(|g| g.id == id);
    match found {
        Some(g) => Ok(vec![g]),
        None => {
            let ids: Vec<&str> = gates.iter().map(|g| g.id.as_str()).collect();
            bail!(
                "gate '{id}' not found in properties.yaml; known: {}",
                ids.join(", ")
            )
        }
    }
}

// ---------------------------------------------------------------------------
// Run one gate
// ---------------------------------------------------------------------------

/// Returns `(any_warn, any_blocking_breach)`.
fn run_one_gate(gate: &LGate, cfg: &CliArgs, root: &std::path::Path) -> Result<(bool, bool)> {
    eprintln!(
        "L-GATE: running {} on profile '{}'",
        gate.id, cfg.hardware_profile
    );

    let profile = gate
        .profiles
        .iter()
        .find(|p| p.name == cfg.hardware_profile);
    if profile.is_none() {
        eprintln!(
            "L-GATE WARN: no profile '{}' for gate {} — skipping (advisory)",
            cfg.hardware_profile, gate.id
        );
        return Ok((true, false));
    }
    let profile = profile.unwrap();

    if !profile.has_floors() {
        eprintln!(
            "L-GATE SKIP: gate {} profile {} has null floors — advisory placeholder",
            gate.id, profile.name
        );
        return Ok((false, false));
    }

    let jsonl = if let Some(mock) = &cfg.mock_input {
        std::fs::read_to_string(mock)
            .map_err(|e| anyhow::anyhow!("mock-input {}: {e}", mock.display()))?
    } else {
        run_bench(gate, &cfg.bench_args, root)?
    };

    let metrics = parse_jsonl_metrics(&jsonl, &gate.id)?;
    check_floors(gate, profile, &metrics)
}

// ---------------------------------------------------------------------------
// Bench invocation
// ---------------------------------------------------------------------------

/// Run the bench and capture its stdout (JSONL lines).
fn run_bench(gate: &LGate, extra_args: &[String], root: &std::path::Path) -> Result<String> {
    if gate.bench_invocation.trim() == "TBD-v2.7" {
        eprintln!(
            "L-GATE SKIP: gate {} bench_invocation is TBD — placeholder gate",
            gate.id
        );
        return Ok(String::new());
    }

    // Ensure output directory exists for --out-json paths like target/lgate/*.jsonl.
    std::fs::create_dir_all(root.join("target").join("lgate"))
        .map_err(|e| anyhow::anyhow!("cannot create target/lgate/: {e}"))?;

    // Build the template command (tokens still contain {rep} placeholders).
    let (bin, bin_args_tpl) = build_bench_cmd(&gate.bench_invocation, extra_args, gate)?;

    let n_reps = gate.n_reps.max(1);
    let mut all_stdout = String::new();

    for rep in 0..n_reps {
        // Substitute {rep} placeholder in every token.
        let rep_str = rep.to_string();
        let args: Vec<String> = bin_args_tpl
            .iter()
            .map(|t| t.replace("{rep}", &rep_str))
            .collect();

        eprintln!("L-GATE [rep {rep}/{n_reps}]: $ {bin} {}", args.join(" "));

        let out = process::Command::new(&bin)
            .args(&args)
            .current_dir(root)
            .output()
            .map_err(|e| anyhow::anyhow!("failed to spawn '{bin}': {e}"))?;

        if !out.status.success() {
            let stderr = String::from_utf8_lossy(&out.stderr);
            bail!(
                "bench invocation failed (exit {})\nstderr: {stderr}",
                out.status.code().unwrap_or(-1)
            );
        }

        all_stdout.push_str(&String::from_utf8_lossy(&out.stdout));
    }

    Ok(all_stdout)
}

/// Build the command and args from the bench_invocation YAML template.
///
/// Injects `--format=jsonl --gate-id <id>` after `--`, appends extra_args.
/// Returns `(binary, args_vec)`.
fn build_bench_cmd(
    invocation: &str,
    extra_args: &[String],
    gate: &LGate,
) -> Result<(String, Vec<String>)> {
    // Normalise the multi-line YAML block: join lines, strip trailing backslashes.
    let joined: String = invocation
        .lines()
        .map(|l| l.trim().trim_end_matches('\\').trim())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join(" ");

    let mut tokens: Vec<String> = joined.split_whitespace().map(str::to_owned).collect();
    if tokens.is_empty() {
        bail!("bench_invocation is empty for gate '{}'", gate.id);
    }

    // Inject --format=jsonl and --gate-id <id> after the `--` separator.
    // Use two-token form so parse_args in latency_tail.rs handles it without
    // needing `strip_prefix("--gate-id=")` (simpler, avoids format mismatch).
    let sep_pos = tokens.iter().position(|t| t == "--");
    let inject_at = sep_pos.map_or(tokens.len(), |p| p + 1);
    tokens.insert(inject_at, gate.id.clone());
    tokens.insert(inject_at, "--gate-id".to_owned());
    tokens.insert(inject_at, "--format=jsonl".to_owned());

    // Append caller-supplied extra args.
    tokens.extend_from_slice(extra_args);

    let bin = tokens.remove(0);
    Ok((bin, tokens))
}

// ---------------------------------------------------------------------------
// JSONL parser
// ---------------------------------------------------------------------------

/// Parsed L-gate percentile record.
#[derive(Debug)]
struct Metric {
    metric: String,
    value_ns: i64,
}

/// Parse JSONL output from the bench (one record per line).
///
/// Each line: `{"gate":"...","metric":"p50","value_ns":28}`.
/// Lines that don't look like L-gate metrics are silently skipped (e.g. the
/// existing JSON summary line from --out-json /dev/stdout).
fn parse_jsonl_metrics(jsonl: &str, expected_gate: &str) -> Result<Vec<Metric>> {
    let mut metrics = Vec::new();
    for line in jsonl.lines() {
        let line = line.trim();
        if !line.starts_with('{') {
            continue;
        }
        if !line.contains("\"value_ns\"") {
            continue;
        }

        if let (Some(metric), Some(value_ns)) = (
            extract_str_field(line, "metric"),
            extract_i64_field(line, "value_ns"),
        ) {
            metrics.push(Metric { metric, value_ns });
        }
    }
    if metrics.is_empty() {
        eprintln!(
            "L-GATE WARN: no JSONL metric lines found for gate '{}'; \
             bench may not have emitted --format=jsonl output",
            expected_gate
        );
    }
    Ok(metrics)
}

/// Extract a string field value from a flat JSON object (hand-rolled, no serde).
fn extract_str_field(json: &str, key: &str) -> Option<String> {
    let needle = format!("\"{key}\":\"");
    let start = json.find(&needle)? + needle.len();
    let end = json[start..].find('"')? + start;
    Some(json[start..end].to_owned())
}

/// Extract an integer field value from a flat JSON object.
fn extract_i64_field(json: &str, key: &str) -> Option<i64> {
    let needle = format!("\"{key}\":");
    let start = json.find(&needle)? + needle.len();
    // Skip optional whitespace.
    let rest = json[start..].trim_start();
    let end = rest
        .find(|c: char| !c.is_ascii_digit() && c != '-')
        .unwrap_or(rest.len());
    rest[..end].parse().ok()
}

// ---------------------------------------------------------------------------
// Floor check
// ---------------------------------------------------------------------------

/// Check metric values against the profile's percentile floors.
///
/// Returns `(any_warn, any_blocking_breach)`.
/// A blocking breach occurs when severity == RELEASE_BLOCKING AND advisory == false
/// AND the observed value exceeds the floor (v2.7 active enforcement, ADR-0069 +
/// math.md §3.6.bis.7).
fn check_floors(gate: &LGate, profile: &Profile, metrics: &[Metric]) -> Result<(bool, bool)> {
    let mut warned = false;
    let mut any_blocking = false;

    let floor_map: &[(&str, Option<i64>)] = &[
        ("p50", profile.p50),
        ("p99", profile.p99),
        ("p99.9", profile.p999),
        ("p99.99", profile.p9999),
    ];

    let is_blocking_gate = gate.severity == "RELEASE_BLOCKING" && !profile.is_advisory();

    for (metric_name, floor_opt) in floor_map {
        let floor = match floor_opt {
            Some(f) => *f,
            None => continue, // advisory placeholder with null floor
        };
        let observed = metrics.iter().find(|m| m.metric == *metric_name);
        match observed {
            None => {
                eprintln!(
                    "L-GATE WARN: gate {} metric {} not found in bench output (advisory)",
                    gate.id, metric_name
                );
                warned = true;
            }
            Some(m) if m.value_ns > floor => {
                if is_blocking_gate {
                    eprintln!(
                        "L-GATE ERR: {} {} {}={}ns exceeds floor {}ns on {} (BLOCKING)",
                        gate.id, gate.severity, metric_name, m.value_ns, floor, profile.name
                    );
                    any_blocking = true;
                } else {
                    eprintln!(
                        "L-GATE WARN: {} {} {}={}ns exceeds floor {}ns on {} (advisory)",
                        gate.id, gate.severity, metric_name, m.value_ns, floor, profile.name
                    );
                }
                warned = true;
            }
            Some(m) => {
                println!(
                    "L-GATE PASS: {} {} {}={}ns ≤ {}ns on {}",
                    gate.id, gate.severity, metric_name, m.value_ns, floor, profile.name
                );
            }
        }
    }
    Ok((warned, any_blocking))
}

// ---------------------------------------------------------------------------
// Minimal YAML parser for latency_gates: section
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct LGate {
    id: String,
    severity: String,
    bench_invocation: String,
    /// Number of repetitions from `replication.n_reps` (default 1).
    n_reps: u32,
    profiles: Vec<Profile>,
}

#[derive(Debug)]
struct Profile {
    name: String,
    p50: Option<i64>,
    p99: Option<i64>,
    p999: Option<i64>,
    p9999: Option<i64>,
    /// v2.7: per-profile advisory flag (math.md §3.6.bis.7).
    /// `None` → default `true` (backward-compat with v2.6 entries).
    advisory: Option<bool>,
}

impl Profile {
    fn has_floors(&self) -> bool {
        self.p50.is_some() || self.p99.is_some() || self.p999.is_some() || self.p9999.is_some()
    }

    /// Returns true when this profile enforces warn-only semantics.
    /// Default: true (advisory) when the field is absent.
    fn is_advisory(&self) -> bool {
        self.advisory.unwrap_or(true)
    }
}

/// Parse the `latency_gates:` section from the properties YAML.
///
/// Hand-rolled minimal parser; avoids serde_yaml dep in xtask (build tool can
/// add deps freely, but unnecessary deps are suckless-bad). The schema is stable
/// and both ends (writer + reader) are under the project's control.
fn parse_latency_gates(yaml: &str) -> Result<Vec<LGate>> {
    let mut gates: Vec<LGate> = Vec::new();
    let mut in_section = false;
    let mut current: Option<GateBuilder> = None;
    let mut in_invocation = false;
    let mut current_profile: Option<ProfileBuilder> = None;
    let mut in_percentile_budgets = false;
    let mut in_replication = false;

    for raw_line in yaml.lines() {
        let indent = raw_line.len() - raw_line.trim_start_matches(' ').len();
        let line = raw_line.trim();

        // Section start.
        if line == "latency_gates:" {
            in_section = true;
            continue;
        }
        if !in_section {
            continue;
        }

        // Section end: any top-level key (indent=0, non-empty, not a list item).
        if indent == 0 && !line.is_empty() && !line.starts_with('-') {
            break;
        }

        // New gate entry: `  - id: ...`
        if let Some(id_raw) = line.strip_prefix("- id:") {
            flush_gate(&mut current, &mut current_profile, &mut gates);
            // Strip inline YAML comment (`# ...`) from the id value.
            let id = id_raw.split('#').next().unwrap_or("").trim().to_owned();
            current = Some(GateBuilder::new(id));
            in_invocation = false;
            in_percentile_budgets = false;
            in_replication = false;
            continue;
        }

        let Some(ref mut g) = current else { continue };

        // Multi-line bench_invocation block scalar (|).
        if in_invocation {
            if indent >= 6 {
                g.bench_invocation.push_str(line);
                g.bench_invocation.push('\n');
                continue;
            } else {
                in_invocation = false;
            }
        }

        match line {
            l if l.starts_with("severity:") => {
                // Strip inline YAML comment before storing severity value.
                let raw = l["severity:".len()..].trim();
                g.severity = raw.split('#').next().unwrap_or("").trim().to_owned();
            }
            l if l.starts_with("bench_invocation:") => {
                let after = l["bench_invocation:".len()..].trim();
                if after == "|" {
                    in_invocation = true;
                } else {
                    g.bench_invocation = after.to_owned();
                }
            }
            "replication:" => {
                in_replication = true;
                in_percentile_budgets = false;
            }
            l if in_replication && l.starts_with("n_reps:") => {
                let raw = l["n_reps:".len()..].trim();
                let val = raw.split('#').next().unwrap_or("").trim();
                if let Ok(n) = val.parse::<u32>() {
                    g.n_reps = n;
                }
            }
            "percentile_budgets_ns:" => {
                in_percentile_budgets = true;
                in_replication = false;
                flush_profile(&mut current_profile, g);
            }
            // Hardware profile key: `      "i7-12700K":` or `      "m2-pro":   # comment`
            // (indent 6, key portion ends with colon before any inline comment).
            l if in_percentile_budgets && indent == 6 && is_yaml_map_key(l) => {
                flush_profile(&mut current_profile, g);
                let key_part = l.split('#').next().unwrap_or(l).trim();
                let profile_name = key_part
                    .trim_end_matches(':')
                    .trim()
                    .trim_matches('"')
                    .to_owned();
                current_profile = Some(ProfileBuilder::new(profile_name));
            }
            l if in_percentile_budgets => {
                if let Some(ref mut pb) = current_profile {
                    parse_profile_field(l, pb);
                }
            }
            _ => {}
        }
    }

    // Flush last gate.
    flush_gate(&mut current, &mut current_profile, &mut gates);
    Ok(gates)
}

// ---------------------------------------------------------------------------
// Builder types for the minimal YAML parser
// ---------------------------------------------------------------------------

#[derive(Default)]
struct GateBuilder {
    id: String,
    severity: String,
    bench_invocation: String,
    n_reps: u32,
    profiles: Vec<Profile>,
}

impl GateBuilder {
    fn new(id: String) -> Self {
        Self {
            id,
            ..Default::default()
        }
    }

    fn build(self) -> LGate {
        LGate {
            id: self.id,
            severity: self.severity,
            bench_invocation: self.bench_invocation,
            // Default to 1 rep if YAML replication section was absent.
            n_reps: if self.n_reps == 0 { 1 } else { self.n_reps },
            profiles: self.profiles,
        }
    }
}

struct ProfileBuilder {
    name: String,
    p50: Option<i64>,
    p99: Option<i64>,
    p999: Option<i64>,
    p9999: Option<i64>,
    advisory: Option<bool>,
}

impl ProfileBuilder {
    fn new(name: String) -> Self {
        Self {
            name,
            p50: None,
            p99: None,
            p999: None,
            p9999: None,
            advisory: None,
        }
    }

    fn build(self) -> Profile {
        Profile {
            name: self.name,
            p50: self.p50,
            p99: self.p99,
            p999: self.p999,
            p9999: self.p9999,
            advisory: self.advisory,
        }
    }
}

/// Returns true if the trimmed line is a YAML map key (ends with `:` before any `#` comment).
///
/// Examples: `"i7-12700K":` → true; `"m2-pro":   # comment` → true; `p50: 28` → false
/// (because `p50` is a value key we handle separately, not a profile name).
fn is_yaml_map_key(line: &str) -> bool {
    // A profile name is quoted: starts with `"`.
    if !line.starts_with('"') {
        return false;
    }
    // The key portion (before any `#`) must end with `:`.
    let key_part = line.split('#').next().unwrap_or("").trim();
    key_part.ends_with(':')
}

fn parse_profile_field(line: &str, pb: &mut ProfileBuilder) {
    let (key, val_str) = match line.split_once(':') {
        Some(pair) => pair,
        None => return,
    };
    // Strip inline YAML comment (`# ...`) before parsing values.
    let val_clean = val_str.split('#').next().unwrap_or("").trim();
    // `null` → None; any parseable integer → Some(i64); anything else → None.
    let val: Option<i64> = if val_clean == "null" {
        None
    } else {
        val_clean.parse().ok()
    };
    match key.trim() {
        "p50" => pb.p50 = val,
        "p99" => pb.p99 = val,
        "p999" => pb.p999 = val,
        "p9999" => pb.p9999 = val,
        // v2.7: per-profile advisory flag (math.md §3.6.bis.7).
        "advisory" => match val_clean {
            "true" => pb.advisory = Some(true),
            "false" => pb.advisory = Some(false),
            _ => {} // unknown value → leave as None (default true)
        },
        _ => {}
    }
}

fn flush_profile(pb: &mut Option<ProfileBuilder>, g: &mut GateBuilder) {
    if let Some(b) = pb.take() {
        g.profiles.push(b.build());
    }
}

fn flush_gate(
    current: &mut Option<GateBuilder>,
    current_profile: &mut Option<ProfileBuilder>,
    gates: &mut Vec<LGate>,
) {
    if let Some(mut g) = current.take() {
        flush_profile(current_profile, &mut g);
        gates.push(g.build());
    }
}

// ---------------------------------------------------------------------------
// Unit tests (latency_gate module)
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_str_field_basic() {
        let json = r#"{"gate":"L_CEV_PTICK","metric":"p50","value_ns":28}"#;
        assert_eq!(
            extract_str_field(json, "gate").as_deref(),
            Some("L_CEV_PTICK")
        );
        assert_eq!(extract_str_field(json, "metric").as_deref(), Some("p50"));
    }

    #[test]
    fn extract_i64_field_basic() {
        let json = r#"{"gate":"L_CEV_PTICK","metric":"p50","value_ns":28}"#;
        assert_eq!(extract_i64_field(json, "value_ns"), Some(28));
    }

    #[test]
    fn parse_jsonl_basic() {
        let jsonl = r#"
{"gate":"L_CEV_PTICK","metric":"p50","value_ns":23}
{"gate":"L_CEV_PTICK","metric":"p99","value_ns":31}
{"gate":"L_CEV_PTICK","metric":"p99.9","value_ns":45}
{"gate":"L_CEV_PTICK","metric":"p99.99","value_ns":187}
"#;
        let metrics = parse_jsonl_metrics(jsonl, "L_CEV_PTICK").unwrap();
        assert_eq!(metrics.len(), 4);
        assert_eq!(metrics[0].metric, "p50");
        assert_eq!(metrics[0].value_ns, 23);
        assert_eq!(metrics[2].metric, "p99.9");
        assert_eq!(metrics[2].value_ns, 45);
    }

    #[test]
    fn parse_gates_minimal() {
        let yaml = r#"
latency_gates:

  - id: L_TEST_GATE
    severity: ADVISORY
    bench_invocation: cargo run --release -- --n 16
    percentile_budgets_ns:
      "test-cpu":
        p50: 10
        p99: 20
        p999: 50
        p9999: 200
        status: blocking
"#;
        let gates = parse_latency_gates(yaml).unwrap();
        assert_eq!(gates.len(), 1);
        assert_eq!(gates[0].id, "L_TEST_GATE");
        assert_eq!(gates[0].severity, "ADVISORY");
        let profile = &gates[0].profiles[0];
        assert_eq!(profile.name, "test-cpu");
        assert_eq!(profile.p50, Some(10));
        assert_eq!(profile.p999, Some(50));
    }

    #[test]
    fn profile_has_floors_null() {
        let p = Profile {
            name: "x".into(),
            p50: None,
            p99: None,
            p999: None,
            p9999: None,
            advisory: None,
        };
        assert!(!p.has_floors());
    }

    #[test]
    fn cli_args_gate_id() {
        let args: Vec<String> = vec!["L_CEV_PTICK".into()];
        let cfg = CliArgs::parse(&args).unwrap();
        assert_eq!(cfg.gate_id.as_deref(), Some("L_CEV_PTICK"));
        assert!(!cfg.all);
        assert_eq!(cfg.hardware_profile, "i7-12700K");
    }

    #[test]
    fn cli_args_all_flag() {
        let args: Vec<String> = vec!["--all".into()];
        let cfg = CliArgs::parse(&args).unwrap();
        assert!(cfg.all);
    }

    #[test]
    fn cli_args_hardware_profile_override() {
        let args: Vec<String> = vec![
            "L_CEV_PTICK".into(),
            "--hardware-profile".into(),
            "m2-pro".into(),
        ];
        let cfg = CliArgs::parse(&args).unwrap();
        assert_eq!(cfg.hardware_profile, "m2-pro");
    }

    // -----------------------------------------------------------------------
    // v2.7 blocking-semantics unit tests (ADR-0069 + math.md §3.6.bis.7)
    // -----------------------------------------------------------------------

    fn make_gate(severity: &str) -> LGate {
        LGate {
            id: "L_TEST".into(),
            severity: severity.into(),
            bench_invocation: String::new(),
            n_reps: 1,
            profiles: vec![],
        }
    }

    fn make_profile(advisory: Option<bool>, p999: Option<i64>) -> Profile {
        Profile {
            name: "i7-12700K".into(),
            p50: None,
            p99: None,
            p999,
            p9999: None,
            advisory,
        }
    }

    fn metrics_with_p999(value_ns: i64) -> Vec<Metric> {
        vec![Metric {
            metric: "p99.9".into(),
            value_ns,
        }]
    }

    /// v2.7: RELEASE_BLOCKING + advisory=false + breach → any_blocking=true.
    #[test]
    fn blocking_gate_advisory_false_breach_returns_blocking() {
        let gate = make_gate("RELEASE_BLOCKING");
        let profile = make_profile(Some(false), Some(50_000));
        // Observed p99.9 = 99_999 ns > floor 50_000 ns → blocking breach.
        let metrics = metrics_with_p999(99_999);
        let (warned, blocking) = check_floors(&gate, &profile, &metrics).unwrap();
        assert!(warned, "breach must set warned");
        assert!(
            blocking,
            "advisory=false + RELEASE_BLOCKING + breach must set blocking"
        );
    }

    /// v2.7: RELEASE_BLOCKING + advisory=false + within budget → exit 0 (no blocking).
    #[test]
    fn blocking_gate_advisory_false_within_budget_no_blocking() {
        let gate = make_gate("RELEASE_BLOCKING");
        let profile = make_profile(Some(false), Some(50_000));
        // Observed p99.9 = 30_000 ns ≤ floor 50_000 ns → PASS.
        let metrics = metrics_with_p999(30_000);
        let (warned, blocking) = check_floors(&gate, &profile, &metrics).unwrap();
        assert!(!warned, "no breach → no warning");
        assert!(!blocking, "no breach → not blocking");
    }

    /// v2.7 backward-compat: RELEASE_BLOCKING + advisory=true + breach → warn only, no blocking.
    #[test]
    fn release_blocking_advisory_true_breach_is_warn_only() {
        let gate = make_gate("RELEASE_BLOCKING");
        // advisory=true (default) → warn-only even with RELEASE_BLOCKING.
        let profile = make_profile(Some(true), Some(50_000));
        let metrics = metrics_with_p999(99_999);
        let (warned, blocking) = check_floors(&gate, &profile, &metrics).unwrap();
        assert!(warned, "breach must set warned");
        assert!(
            !blocking,
            "advisory=true → never blocking even on RELEASE_BLOCKING"
        );
    }
}
