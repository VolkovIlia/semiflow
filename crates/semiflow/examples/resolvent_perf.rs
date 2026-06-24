//! `L_RESOLVENT_N64_P99` bench harness for the xtask latency-gate.
//!
//! Measures per-call latency of [`LaplaceChernoffResolvent::eval`] at n=64,
//! λ=1.0, `GaussLaguerre32`, on a Gaussian initial datum on `Grid1D` N=256.
//!
//! Contract gate: `L_RESOLVENT_N64_P99` (ADR-0069, math.md §22).
//! Properties entry: `contracts/semiflow-core.properties.yaml`.
//!
//! Build: `cargo build --release --example resolvent_perf -p semiflow-core`
//! Smoke: `cargo run --release --example resolvent_perf -- --n-ticks 1000 --warmup-ticks 100`

// Integration test: allows for numerical / binding wrapper patterns.
#![allow(clippy::cast_possible_truncation, clippy::too_many_lines)]

use std::{
    env,
    fs::{create_dir_all, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use semiflow::{
    DiffusionChernoff, Grid1D, GridFn1D, HdrSnapshot, LaplaceChernoffResolvent, LaplaceQuadrature,
};

// ── Constants ────────────────────────────────────────────────────────────────

const DEFAULT_N_TICKS: usize = 100_000;
const DEFAULT_WARMUP: usize = 1_000;
const DEFAULT_N_RESOLVENT: usize = 64;
const DEFAULT_LAMBDA: f64 = 1.0;
const DEFAULT_GRID_N: usize = 256;
const DEFAULT_HARDWARE_PROFILE: &str = "i7-12700K";
const DEFAULT_GATE_ID: &str = "L_RESOLVENT_N64_P99";
const DEFAULT_OUT_JSON: &str = "target/lgate/resolvent.jsonl";

// ── CLI args ─────────────────────────────────────────────────────────────────

struct Args {
    n_ticks: usize,
    warmup_ticks: usize,
    n_resolvent: usize,
    lambda: f64,
    grid_n: usize,
    out_json: PathBuf,
    rep: u32,
    hardware_profile: String,
    gate_id: String,
}

fn parse_usize(raw: &str, flag: &str, min: usize) -> usize {
    match raw.parse::<usize>() {
        Ok(v) if v >= min => v,
        Ok(_) => die(&format!("{flag} must be >= {min}")),
        Err(_) => die(&format!("{flag} needs a number, got '{raw}'")),
    }
}

fn parse_f64(raw: &str, flag: &str) -> f64 {
    raw.parse::<f64>()
        .unwrap_or_else(|_| die(&format!("{flag} needs a float, got '{raw}'")))
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn parse_args(argv: Vec<String>) -> Args {
    let mut a = Args {
        n_ticks: DEFAULT_N_TICKS,
        warmup_ticks: DEFAULT_WARMUP,
        n_resolvent: DEFAULT_N_RESOLVENT,
        lambda: DEFAULT_LAMBDA,
        grid_n: DEFAULT_GRID_N,
        out_json: PathBuf::from(DEFAULT_OUT_JSON),
        rep: 0,
        hardware_profile: DEFAULT_HARDWARE_PROFILE.into(),
        gate_id: DEFAULT_GATE_ID.into(),
    };
    let mut it = argv.into_iter().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--n-ticks" => {
                let v = it.next().unwrap_or_default();
                a.n_ticks = parse_usize(&v, "--n-ticks", 1);
            }
            "--warmup-ticks" => {
                let v = it.next().unwrap_or_default();
                a.warmup_ticks = parse_usize(&v, "--warmup-ticks", 0);
            }
            "--n" => {
                let v = it.next().unwrap_or_default();
                a.n_resolvent = parse_usize(&v, "--n", 1);
            }
            "--lambda" => {
                let v = it.next().unwrap_or_default();
                a.lambda = parse_f64(&v, "--lambda");
            }
            "--grid-n" => {
                let v = it.next().unwrap_or_default();
                a.grid_n = parse_usize(&v, "--grid-n", 2);
            }
            "--out-json" => {
                a.out_json = PathBuf::from(it.next().unwrap_or_default());
            }
            "--rep" => {
                let v = it.next().unwrap_or_default();
                a.rep = parse_usize(&v, "--rep", 0) as u32;
            }
            "--hardware-profile" => {
                a.hardware_profile = it.next().unwrap_or_else(|| DEFAULT_HARDWARE_PROFILE.into());
            }
            "--gate-id" => {
                a.gate_id = it.next().unwrap_or_else(|| DEFAULT_GATE_ID.into());
            }
            "--quadrature" | "--format" | "--format=jsonl" => {
                // Accept but ignore: only one mode in v2.7.
                let _ = it.next();
            }
            "--help" | "-h" => {
                print_help();
                std::process::exit(0);
            }
            f => die(&format!("unknown flag '{f}'")),
        }
    }
    a
}

fn print_help() {
    println!("Usage: resolvent_perf [OPTIONS]");
    println!("  --n-ticks N          Measurement iterations (default {DEFAULT_N_TICKS})");
    println!("  --warmup-ticks N     Warmup iterations (default {DEFAULT_WARMUP})");
    println!("  --n N                Resolvent truncation n (default {DEFAULT_N_RESOLVENT})");
    println!("  --lambda F           Resolvent parameter λ (default {DEFAULT_LAMBDA})");
    println!("  --grid-n N           Grid1D node count (default {DEFAULT_GRID_N})");
    println!("  --out-json PATH      JSONL output path (default {DEFAULT_OUT_JSON})");
    println!("  --rep N              Replication index (default 0)");
    println!("  --hardware-profile S Hardware label in output");
    println!("  --gate-id S          Gate ID label in output");
}

// ── Harness setup ─────────────────────────────────────────────────────────────

fn build_resolvent(
    grid_n: usize,
    n: usize,
) -> (
    Grid1D,
    GridFn1D,
    LaplaceChernoffResolvent<DiffusionChernoff>,
) {
    let grid = Grid1D::new(-5.0_f64, 5.0, grid_n).expect("Grid1D construction must succeed");
    let inner = DiffusionChernoff::new(
        |_| 1.0_f64, // a(x) = 1 (unit diffusion)
        |_| 0.0_f64, // a'(x) = 0
        |_| 0.0_f64, // a''(x) = 0
        1.0_f64,     // a_norm bound
        grid,
    );
    let resolvent = LaplaceChernoffResolvent::new(inner, n, LaplaceQuadrature::GaussLaguerre32)
        .expect("resolvent construction must succeed");
    // Initial datum: Gaussian centred at origin.
    let g = GridFn1D::from_fn(grid, |x| (-x * x).exp());
    (grid, g, resolvent)
}

// ── JSONL emit (v2.6 schema, parseable by xtask latency-gate) ─────────────────

fn emit_jsonl(args: &Args, hdr: &mut HdrSnapshot) {
    if let Some(parent) = args.out_json.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).expect("create output directory");
        }
    }
    let file = File::create(&args.out_json)
        .unwrap_or_else(|e| die(&format!("create {}: {e}", args.out_json.display())));
    let mut w = BufWriter::new(file);

    let percentiles: &[(&str, f64)] = &[
        ("p50", 50.0),
        ("p99", 99.0),
        ("p99.9", 99.9),
        ("p99.99", 99.99),
    ];
    for (label, pct) in percentiles {
        let v = hdr.percentile(*pct);
        writeln!(
            w,
            r#"{{"gate":"{}","metric":"{}","value_ns":{},"rep":{}}}"#,
            args.gate_id, label, v, args.rep
        )
        .expect("write JSONL line");
    }
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args(env::args().collect());
    let (_grid, g, resolvent) = build_resolvent(args.grid_n, args.n_resolvent);

    // Warmup: not measured.
    for _ in 0..args.warmup_ticks {
        let _ = resolvent.eval(args.lambda, &g).expect("warmup eval");
    }

    // Measurement loop.
    let mut hdr = HdrSnapshot::new(args.n_ticks);
    for _ in 0..args.n_ticks {
        let t0 = Instant::now();
        let _ = resolvent.eval(args.lambda, &g).expect("timed eval");
        hdr.record(t0.elapsed().as_nanos() as i64);
    }

    let p50 = hdr.percentile(50.0);
    let p99 = hdr.percentile(99.0);
    let p999 = hdr.percentile(99.9);
    let p9999 = hdr.percentile(99.99);

    emit_jsonl(&args, &mut hdr);

    eprintln!(
        "{gate}: {n} ticks  p50={p50}ns  p99={p99}ns  p99.9={p999}ns  p99.99={p9999}ns  hw={hw}",
        gate = args.gate_id,
        n = args.n_ticks,
        hw = args.hardware_profile,
    );
}
