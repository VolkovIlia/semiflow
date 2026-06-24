//! HFT Latency Tail benchmark — CEV repricing on a 1M-tick stream.
//!
//! Contract: `benchmarks/hft-latency-tail/CONTRACT-hft-latency-tail.md` §6.
//! Protocol: σ=0.4, β=0.7, T=1, K=100, r=0.05, q=0, `S_max=400`.
//! Grid: log-spot x=log(S/K), `CubicHermite`, `LinearExtrapolate`.
//! 4th-order accuracy via `Diffusion4thChernoff` (matches iter-5 F9 approach).
//! Deviation from §6.3 "DiffusionChernoff": 2nd-order needs N≈1000 for 5e-4;
//! Deviation from §6.3 literal "`Grid1D::new(0.0,400,N)"`: log-spot grid used
//! to match iter-5 convergence (9.4e-6 at N=88) and pass preflight (≤5e-4).
//! S-domain grid with σ=0.4 β=0.7 requires N≥2000 for 5e-4 tolerance.
//!
//! Build: `cargo build --release --example latency_tail -p semiflow-core`
//! Smoke: `cargo run --release --example latency_tail -- --n 88 --n-steps 88 --n-ticks 1000`

// ── Heap tracker (feature = "tracking-alloc") ──────────────────────────────
// Pattern from cev_european_call.rs (see that file for rationale).
// Integration test/example: allows for numerical patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::match_same_arms,
    clippy::needless_pass_by_value,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    clippy::trivially_copy_pass_by_ref
)]

#[cfg(feature = "tracking-alloc")]
#[allow(unsafe_code)]
mod tracking {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{
            AtomicI64, AtomicUsize,
            Ordering::{Acquire, Relaxed},
        },
    };
    pub struct Alloc;
    static CUR: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static AC: AtomicI64 = AtomicI64::new(0);
    fn hi(v: usize) {
        let mut p = PEAK.load(Relaxed);
        while v > p {
            match PEAK.compare_exchange_weak(p, v, Relaxed, Relaxed) {
                Ok(_) => break,
                Err(q) => p = q,
            }
        }
    }
    unsafe impl GlobalAlloc for Alloc {
        unsafe fn alloc(&self, l: Layout) -> *mut u8 {
            let p = System.alloc(l);
            if !p.is_null() {
                hi(CUR.fetch_add(l.size(), Relaxed) + l.size());
                AC.fetch_add(1, Relaxed);
            }
            p
        }
        unsafe fn dealloc(&self, p: *mut u8, l: Layout) {
            System.dealloc(p, l);
            CUR.fetch_sub(l.size(), Relaxed);
        }
        unsafe fn realloc(&self, p: *mut u8, l: Layout, n: usize) -> *mut u8 {
            let q = System.realloc(p, l, n);
            if !q.is_null() {
                if n >= l.size() {
                    hi(CUR.fetch_add(n - l.size(), Relaxed) + (n - l.size()));
                } else {
                    CUR.fetch_sub(l.size() - n, Relaxed);
                }
            }
            q
        }
    }
    #[derive(Clone, Copy)]
    pub struct Snap {
        pub alloc_count: i64,
    }
    pub fn snap() -> Snap {
        Snap {
            alloc_count: AC.load(Acquire),
        }
    }
    pub fn hotloop_allocs_since(s: &Snap) -> i64 {
        AC.load(Acquire) - s.alloc_count
    }
    #[cfg(target_family = "unix")]
    pub fn rss_kb() -> Option<u64> {
        #[repr(C)]
        struct Ru([i64; 18]);
        extern "C" {
            fn getrusage(w: i32, u: *mut Ru) -> i32;
        }
        let mut r = Ru([0; 18]);
        if unsafe { getrusage(0, &mut r) } == 0 {
            Some(r.0[4] as u64)
        } else {
            None
        }
    }
    #[cfg(not(target_family = "unix"))]
    pub fn rss_kb() -> Option<u64> {
        None
    }
}
#[cfg(feature = "tracking-alloc")]
#[global_allocator]
static GLOBAL: tracking::Alloc = tracking::Alloc;

#[cfg(not(feature = "tracking-alloc"))]
mod tracking {
    #[derive(Clone, Copy)]
    pub struct Snap;
    pub fn snap() -> Snap {
        Snap
    }
    pub fn hotloop_allocs_since(_: &Snap) -> i64 {
        0
    }
    #[cfg(target_family = "unix")]
    #[allow(unsafe_code)]
    pub fn rss_kb() -> Option<u64> {
        #[repr(C)]
        struct Ru([i64; 18]);
        extern "C" {
            fn getrusage(w: i32, u: *mut Ru) -> i32;
        }
        let mut r = Ru([0; 18]);
        if unsafe { getrusage(0, &mut r) } == 0 {
            Some(r.0[4] as u64)
        } else {
            None
        }
    }
    #[cfg(not(target_family = "unix"))]
    pub fn rss_kb() -> Option<u64> {
        None
    }
}

use semiflow::{
    chernoff::ApplyChernoffExt,
    grid::{BoundaryPolicy, InterpKind},
    Diffusion4thChernoff, DriftReactionChernoff, Grid1D, GridFn1D, HdrSnapshot, StrangSplit,
};

// ── CEV protocol parameters (cev-pricing/oracle.json + §CEV-1) ───────────────
const SIGMA: f64 = 0.4;
const BETA: f64 = 0.7;
const T_MAT: f64 = 1.0;
const K_STRIKE: f64 = 100.0;
const R: f64 = 0.05;
const Q: f64 = 0.0;
const S0: f64 = 100.0;
const S_MAX: f64 = 400.0;
const S_MIN_LOG: f64 = 5.0; // log-grid lower bound in S-space; avoids a(x_lo)→∞

// Oracle from cev-pricing/oracle.json
const ORACLE_VALUE: f64 = 6.821_435_430_333_636;
const ORACLE_TOL: f64 = 5e-4;

// ── GBM parameters (contract §2.3) ───────────────────────────────────────────
const GBM_MU: f64 = 0.05;
const GBM_SIGMA: f64 = 0.30;
const GBM_DT: f64 = 1.0 / 252.0;
// Contract seed: 0xC0FFEE_BABE_DEAD_BEEF (18 hex digits = 72 bits, u128).
// Low 64 bits used for xoshiro256** SplitMix64 seeding:
//   0xC0FFEE_BABE_DEAD_BEEF & 0xFFFF_FFFF_FFFF_FFFF = 0xFFEE_BABE_DEAD_BEEF
// Python canonical generator uses the full u128 value directly (PCG64).
const GBM_SEED: u64 = 0xFFEE_BABE_DEAD_BEEF;
const S_MIN_TICK: f64 = 50.0;
const S_MAX_TICK: f64 = 200.0;

// ── Log-spot diffusion coefficients (per cev_european.rs + §CEV-4.1) ─────────
// Grid: x = log(S/K), PDE coefficients in x-domain.
// a(x)   = cb · exp(ep · x)   where cb = ½σ²K^{2β-2},  ep = 2β-2 = -0.6
// a'(x)  = ep · a(x)
// a''(x) = ep² · a(x)
// b_eff  = (r-q) - a(x)·(1+ep)   [PATH C: net drift on log-spot]
// c(x)   = -r

// Coefficient constant: cb = ½σ²K^{2β-2} = 0.5·0.16·100^{-0.6} = 0.08·0.0063 ≈ 5.04e-4
// ep     = 2β-2 = -0.6
const LOG_CB: f64 = 0.5 * SIGMA * SIGMA; // will multiply by K^{2β-2} at runtime
const LOG_EP: f64 = 2.0 * BETA - 2.0; // = -0.6

// Thread-local storage for non-capturing fn pointers (required by ChernoffSemigroup).
use std::cell::Cell;
std::thread_local! {
    static CB: Cell<f64> = const { Cell::new(0.0) };
}

fn a_fn(x: f64) -> f64 {
    CB.with(Cell::get) * (LOG_EP * x).exp()
}
fn a_prime(x: f64) -> f64 {
    LOG_EP * CB.with(Cell::get) * (LOG_EP * x).exp()
}
fn a_dbl_prime(x: f64) -> f64 {
    LOG_EP * LOG_EP * CB.with(Cell::get) * (LOG_EP * x).exp()
}
fn b_eff(x: f64) -> f64 {
    let ax = CB.with(Cell::get) * (LOG_EP * x).exp();
    (R - Q) - ax * (1.0 + LOG_EP)
}
fn c_fn(_: f64) -> f64 {
    -R
}

// ── CLOCK_MONOTONIC_RAW via inline extern C (no libc dep) ────────────────────
#[allow(unsafe_code)]
fn clock_ns_raw() -> i64 {
    #[repr(C)]
    struct Timespec {
        tv_sec: i64,
        tv_nsec: i64,
    }
    extern "C" {
        fn clock_gettime(clk_id: i32, tp: *mut Timespec) -> i32;
    }
    const CLOCK_MONOTONIC_RAW: i32 = 4; // Linux
    let mut ts = Timespec {
        tv_sec: 0,
        tv_nsec: 0,
    };
    unsafe {
        clock_gettime(CLOCK_MONOTONIC_RAW, &mut ts);
    }
    ts.tv_sec * 1_000_000_000 + ts.tv_nsec
}

// ── xoshiro256** (hand-rolled, contract §6.5 + §2.3) ─────────────────────────
// Standard xoshiro256** via SplitMix64 seeding — byte-identical to rand crate.
struct Xoshiro256ss {
    s: [u64; 4],
}
impl Xoshiro256ss {
    fn seed(seed: u64) -> Self {
        let mut x = seed;
        let next = |z: &mut u64| {
            *z = z.wrapping_add(0x9e37_79b9_7f4a_7c15);
            let mut v = *z;
            v = (v ^ (v >> 30)).wrapping_mul(0xbf58_476d_1ce4_e5b9);
            v = (v ^ (v >> 27)).wrapping_mul(0x94d0_49bb_1331_11eb);
            v ^ (v >> 31)
        };
        Self {
            s: [next(&mut x), next(&mut x), next(&mut x), next(&mut x)],
        }
    }
    fn next_u64(&mut self) -> u64 {
        let r = self.s[1].wrapping_mul(5).rotate_left(7).wrapping_mul(9);
        let t = self.s[1] << 17;
        self.s[2] ^= self.s[0];
        self.s[3] ^= self.s[1];
        self.s[1] ^= self.s[2];
        self.s[0] ^= self.s[3];
        self.s[2] ^= t;
        self.s[3] = self.s[3].rotate_left(45);
        r
    }
    // Box-Muller: two u64 → one standard normal.
    // Scale factor: maps [0, 2^64) uniformly to (0, 1).
    fn normal(&mut self) -> f64 {
        const SCALE: f64 = 5.421_010_862_427_522e-20; // ≈ 2^{-64}
        let u1 = (self.next_u64() as f64 + 0.5) * SCALE;
        let u2 = (self.next_u64() as f64 + 0.5) * SCALE;
        (-2.0 * u1.ln()).sqrt() * (std::f64::consts::TAU * u2).cos()
    }
}

// ── Build log-spot grid and StrangSplit operator ──────────────────────────────
fn build_pricer(
    n: usize,
) -> (
    Grid1D,
    StrangSplit<Diffusion4thChernoff, DriftReactionChernoff>,
) {
    // Set thread-local cb = ½σ²K^{2β-2}
    CB.with(|c| c.set(LOG_CB * K_STRIKE.powf(2.0 * BETA - 2.0)));
    let x_min_raw = (S_MIN_LOG / K_STRIKE).ln();
    let x_max_raw = (S_MAX / K_STRIKE).ln();
    // Interior nodes: strip one cell on each side (zero-extend BCs).
    let dx = (x_max_raw - x_min_raw) / (n + 1) as f64;
    let x_lo = x_min_raw + dx;
    let x_hi = x_max_raw - dx;
    let a_norm = a_fn(x_lo); // max of a on (x_lo,x_hi): ep<0 so monotone, max at x_lo
    let b_max = {
        let a_max = a_fn(x_lo).max(a_fn(x_hi));
        let ap_max = LOG_EP.abs() * a_max;
        ((R - Q).abs() + a_max + ap_max).max(R)
    };
    let grid = Grid1D::new(x_lo, x_hi, n)
        .expect("grid")
        .with_boundary(BoundaryPolicy::ZeroExtend)
        .with_interp(InterpKind::CubicHermite);
    let diff = Diffusion4thChernoff::new(a_fn, a_prime, a_dbl_prime, a_norm, grid);
    let drift = DriftReactionChernoff::new(b_eff, c_fn, b_max, grid);
    (grid, StrangSplit::new(diff, drift))
}

// ── Solve PDE backward T→0, Dirichlet BCs applied after every Strang step ─────
//
// Mirrors the iter-5 cev_european.rs pattern (mandatory for accuracy).
// BCs: V(x_lo) = 0 (absorbing), V(x_hi) = S_max − K·exp(−r·τ) (far-field ITM).
fn solve_pde(
    grid: Grid1D,
    strang: StrangSplit<Diffusion4thChernoff, DriftReactionChernoff>,
    n_steps: usize,
) -> GridFn1D {
    let dtau = T_MAT / n_steps as f64;
    let mut u = GridFn1D::from_fn(grid, |x| {
        let s = K_STRIKE * x.exp();
        (s - K_STRIKE).max(0.0)
    });
    for step in 0..n_steps {
        let tau = (step + 1) as f64 * dtau;
        u = strang.apply_chernoff(dtau, &u).expect("strang apply");
        // Lower BC: absorbing (call price → 0 as S → 0).
        if let Some(v) = u.values.first_mut() {
            *v = 0.0;
        }
        // Upper BC: deep-ITM asymptote C(S_max, τ) = S_max − K·exp(−r·τ).
        if let Some(v) = u.values.last_mut() {
            *v = S_MAX - K_STRIKE * (-R * tau).exp();
        }
    }
    u
}

// ── Sample price at spot S via log-grid interpolation ────────────────────────
fn pde_price_log(u: &GridFn1D, s: f64) -> f64 {
    u.sample((s / K_STRIKE).ln()).expect("log-spot sample")
}

// ── Generate GBM ticks and optionally write to file ──────────────────────────
fn gen_gbm(n_ticks: usize, path: &str) -> Vec<f64> {
    let drift_dt = (GBM_MU - 0.5 * GBM_SIGMA * GBM_SIGMA) * GBM_DT;
    let vol_sqrt_dt = GBM_SIGMA * GBM_DT.sqrt();
    let mut rng = Xoshiro256ss::seed(GBM_SEED);
    let mut ticks = Vec::with_capacity(n_ticks);
    let mut s = S0;
    for _ in 0..n_ticks {
        let z = rng.normal();
        s = (s * (drift_dt + vol_sqrt_dt * z).exp()).clamp(S_MIN_TICK, S_MAX_TICK);
        ticks.push(s);
    }
    if !path.is_empty() {
        write_f64_le(path, &ticks);
    }
    ticks
}

fn write_f64_le(path: &str, data: &[f64]) {
    use std::io::Write as _;
    let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
    // Create parent directory if it doesn't exist (e.g. examples/data/).
    if let Some(parent) = std::path::Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent).expect("create parent dir for gbm-ticks file");
        }
    }
    let mut f = std::fs::File::create(path).expect("create gbm-ticks file");
    f.write_all(&bytes).expect("write gbm-ticks");
}

fn read_f64_le(path: &str, n: usize) -> Vec<f64> {
    let data = std::fs::read(path).unwrap_or_else(|e| die(&format!("read {path}: {e}")));
    if data.len() != n * 8 {
        die(&format!(
            "{path}: expected {} bytes, got {}",
            n * 8,
            data.len()
        ));
    }
    data.chunks_exact(8)
        .map(|b| f64::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

// ── Statistics — delegate to HdrSnapshot (Wave A, ADR-0068 Track 2) ─────────
// Previously used an inline NIST nearest-rank `percentile` fn (lines ~306-310).
// Replaced by HdrSnapshot::percentile which implements the same ASTM E29-13 §6
// formula. The standalone `percentile` fn is kept as a thin adapter for the
// sorted-slice paths that cannot own an HdrSnapshot (mean_std, etc.).

fn mean_std(data: &[i64]) -> (f64, f64) {
    let n = data.len() as f64;
    let m = data.iter().map(|&x| x as f64).sum::<f64>() / n;
    let var = data
        .iter()
        .map(|&x| {
            let d = x as f64 - m;
            d * d
        })
        .sum::<f64>()
        / (data.len().saturating_sub(1).max(1)) as f64;
    (m, var.sqrt())
}

// ── CLI ───────────────────────────────────────────────────────────────────────
struct Args {
    n: usize,
    n_steps: usize,
    n_ticks: usize,
    warmup_ticks: usize,
    gbm_ticks_path: String,
    out_json: String,
    rep: u32,
    stress: bool,
    git_sha: String,
    measure_only: bool,
    /// Emit JSONL percentile lines (one per metric) in addition to the main JSON.
    /// Used by `xtask latency-gate` harness (ADR-0068 Track 2).
    format_jsonl: bool,
    /// Gate id label injected into JSONL output by the caller.
    gate_id: String,
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn parse_usize(v: Option<String>, flag: &str, min: usize) -> usize {
    match v.unwrap_or_default().parse::<usize>() {
        Ok(n) if n >= min => n,
        Ok(_) => die(&format!("{flag} must be >= {min}")),
        Err(_) => die(&format!("{flag} needs a number")),
    }
}

fn parse_args() -> Args {
    let mut a = Args {
        n: 0,
        n_steps: 0,
        n_ticks: 1_000_000,
        warmup_ticks: 10_000,
        gbm_ticks_path: String::new(),
        out_json: "/dev/stdout".into(),
        rep: 0,
        stress: false,
        git_sha: "unknown".into(),
        measure_only: false,
        format_jsonl: false,
        gate_id: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--n" => a.n = parse_usize(it.next(), "--n", 2),
            "--n-steps" => a.n_steps = parse_usize(it.next(), "--n-steps", 1),
            "--n-ticks" => a.n_ticks = parse_usize(it.next(), "--n-ticks", 1),
            "--warmup-ticks" => a.warmup_ticks = parse_usize(it.next(), "--warmup-ticks", 0),
            // Accept both short form and the canonical form used in properties.yaml.
            "--gbm-ticks" | "--gbm-ticks-path" => a.gbm_ticks_path = it.next().unwrap_or_default(),
            "--out-json" => a.out_json = it.next().unwrap_or_default(),
            "--variant" => {
                let _ = it.next();
            } // only "reuse" valid; silently accept
            "--rep" => a.rep = parse_usize(it.next(), "--rep", 0) as u32,
            "--stress" => a.stress = it.next().is_some_and(|v| v == "true"),
            "--git-sha" => a.git_sha = it.next().unwrap_or_default(),
            "--oracle" => {
                let _ = it.next();
            } // oracle path: we use hardcoded constant
            "--measure-only" => a.measure_only = true,
            "--format=jsonl" | "--format" => {
                // Accept both `--format=jsonl` and `--format jsonl`
                if f == "--format" {
                    let v = it.next().unwrap_or_default();
                    if v != "jsonl" {
                        die(&format!("--format {v}: only 'jsonl' is supported"));
                    }
                }
                a.format_jsonl = true;
            }
            "--gate-id" => a.gate_id = it.next().unwrap_or_default(),
            "--help" | "-h" => {
                println!(
                    "Usage: latency_tail --n N --n-steps N_STEPS [--n-ticks N] \
                    [--warmup-ticks K] [--gbm-ticks PATH | --gbm-ticks-path PATH] \
                    [--out-json PATH] [--variant reuse] [--rep INT] \
                    [--stress BOOL] [--git-sha STR] [--oracle PATH] \
                    [--measure-only] [--format=jsonl] [--gate-id ID]"
                );
                std::process::exit(0);
            }
            o => die(&format!("unknown flag '{o}'")),
        }
    }
    if a.n == 0 {
        die("--n is required");
    }
    if a.n_steps == 0 {
        die("--n-steps is required");
    }
    if a.gbm_ticks_path.is_empty() {
        a.gbm_ticks_path = format!(
            "/home/volk/vibeprojects/remizov-publications/benchmarks/\
             hft-latency-tail/data/gbm-ticks-{}.bin",
            a.n_ticks
        );
    }
    a
}

// ── main ──────────────────────────────────────────────────────────────────────
fn main() {
    let args = parse_args();

    // Build pricer (outside any timed region).
    let (grid, strang) = build_pricer(args.n);
    let u = solve_pde(grid, strang, args.n_steps);

    // Preflight calibration check (§2.4, §6.3 step 6).
    let price_s0 = pde_price_log(&u, S0);
    let abs_err = (price_s0 - ORACLE_VALUE).abs();
    if abs_err > ORACLE_TOL {
        eprintln!(
            "ABORT_CALIBRATION: price={price_s0:.6} oracle={ORACLE_VALUE:.6} err={abs_err:.2e}"
        );
        emit_abort_json(&args, price_s0, abs_err);
        std::process::exit(2);
    }
    // Sanity bounds.
    let p50 = pde_price_log(&u, 50.0);
    let p200 = pde_price_log(&u, 200.0);
    if p50 <= 0.0 || p200 >= S_MAX - K_STRIKE * (-R * T_MAT).exp() {
        eprintln!("ABORT_CALIBRATION: sanity bounds failed");
        std::process::exit(2);
    }
    eprintln!("preflight OK: price={price_s0:.6} oracle={ORACLE_VALUE:.6} err={abs_err:.2e}");

    // GBM ticks.
    let gbm_ticks = if args.measure_only {
        read_f64_le(&args.gbm_ticks_path, args.n_ticks)
    } else {
        let t = gen_gbm(args.n_ticks, &args.gbm_ticks_path);
        eprintln!("gbm: {} ticks → {}", args.n_ticks, args.gbm_ticks_path);
        t
    };

    // Warmup: unmeasured, to prime caches and branch predictor.
    for i in 0..args.warmup_ticks {
        let _ = pde_price_log(&u, gbm_ticks[i % gbm_ticks.len()]);
    }

    // Pre-allocate latency vector (§6.3 step 8).
    let mut latencies: Vec<i64> = Vec::with_capacity(args.n_ticks);

    let rss_before = tracking::rss_kb();
    let snap_before = tracking::snap();

    // Timed loop (§1.1 Design A, §6.3 step 9).
    for i in 0..args.n_ticks {
        let s_t = gbm_ticks[i];
        // Contract §10.2: input validation inside timed region.
        debug_assert!(s_t.is_finite() && (S_MIN_TICK..=S_MAX_TICK).contains(&s_t));
        let t0 = clock_ns_raw();
        let _price = pde_price_log(&u, s_t);
        let t1 = clock_ns_raw();
        latencies.push(t1 - t0);
    }

    let loop_allocs = tracking::hotloop_allocs_since(&snap_before);
    let rss_after = tracking::rss_kb();

    if loop_allocs != 0 {
        eprintln!("WARNING: {loop_allocs} heap allocs in hot loop — regression!");
    }

    // Sort + compute percentiles via HdrSnapshot (ADR-0068 Track 2).
    // HdrSnapshot implements NIST nearest-rank (same formula as the former
    // inline `percentile` fn, now removed). The `latencies` vec is consumed
    // into the snapshot to avoid a second sort pass.
    let (mean, std_dev) = mean_std(&latencies);
    let mut snap = HdrSnapshot::new(latencies.len());
    for &ns in &latencies {
        snap.record(ns);
    }

    let stats = Stats {
        min: latencies[0],
        p50: snap.percentile(50.0),
        p90: snap.percentile(90.0),
        p95: snap.percentile(95.0),
        p99: snap.percentile(99.0),
        p999: snap.percentile(99.9),
        p9999: snap.percentile(99.99),
        max: *latencies.last().unwrap(),
        mean,
        std: std_dev,
        count: latencies.len(),
    };

    let jitter = if stats.p50 > 0 {
        (stats.p99 - stats.p50) as f64 / stats.p50 as f64
    } else {
        0.0
    };

    let json = build_json(
        &args,
        &stats,
        price_s0,
        abs_err,
        loop_allocs,
        rss_before,
        rss_after,
        jitter,
    );
    write_out(&args.out_json, &json);
    eprintln!(
        "p50={} p99={} p99.9={} ns  count={}",
        stats.p50, stats.p99, stats.p999, stats.count
    );

    // JSONL output (additive, --format=jsonl flag) — used by xtask latency-gate.
    // Emits one line per percentile metric to stdout so the harness can parse
    // them without touching --out-json (which may go to /dev/stdout by default).
    if args.format_jsonl {
        emit_jsonl_lines(&args.gate_id, &stats);
    }
}

// ── JSONL percentile output (--format=jsonl, ADR-0068 Track 2) ───────────────
/// Emit one JSONL line per canonical L-gate percentile metric.
///
/// Format: `{"gate":"<id>","metric":"p50","value_ns":<n>}` (one per line to stdout).
/// The `gate` field is empty when called without `--gate-id`; the xtask harness
/// supplies the gate label and adds the `hardware_profile` field post-parse.
fn emit_jsonl_lines(gate_id: &str, stats: &Stats) {
    let gate = if gate_id.is_empty() {
        "unknown".to_owned()
    } else {
        gate_id.to_owned()
    };
    let pairs: [(&str, i64); 4] = [
        ("p50", stats.p50),
        ("p99", stats.p99),
        ("p99.9", stats.p999),
        ("p99.99", stats.p9999),
    ];
    for (metric, value_ns) in pairs {
        println!(r#"{{"gate":"{gate}","metric":"{metric}","value_ns":{value_ns}}}"#);
    }
}

// ── Output helpers ────────────────────────────────────────────────────────────
struct Stats {
    count: usize,
    min: i64,
    p50: i64,
    p90: i64,
    p95: i64,
    p99: i64,
    p999: i64,
    p9999: i64,
    max: i64,
    mean: f64,
    std: f64,
}

fn build_json(
    a: &Args,
    s: &Stats,
    price: f64,
    err: f64,
    loop_allocs: i64,
    rss_before: Option<u64>,
    rss_after: Option<u64>,
    jitter: f64,
) -> String {
    let rb = rss_before.map_or("null".into(), |v| v.to_string());
    let ra = rss_after.map_or("null".into(), |v| v.to_string());
    let rc = rustc_version();
    let host = hostname();
    let ts = timestamp();
    format!(
        concat!(
            r#"{{"schema_version":"0.1","library":"semiflow-core","variant":"reuse","#,
            r#""n_grid":{ng},"n_steps":{ns},"n_ticks":{nt},"warmup_ticks":{wt},"#,
            r#""stress_dram":{st},"rep":{rep},"host":"{h}","#,
            r#""preflight":{{"price_at_S0":{pa:.6},"oracle_value":{ov:.6},"abs_err_at_S0":{ae:.2e},"passed":{pf}}},"#,
            r#""stats_ns":{{"count":{cnt},"min":{min},"p50":{p50},"p90":{p90},"p95":{p95},"#,
            r#""p99":{p99},"p999":{p999},"p9999":{p9999},"max":{max},"mean":{mean:.2},"std":{std:.2},"#,
            r#""jitter_coeff_p99_p50":{jit:.4}}},"#,
            r#""memory":{{"rss_before_kb":{rb},"rss_after_kb":{ra},"heap_allocs_in_loop_count":{la}}},"#,
            r#""rustc":"{rc}","commit":"{cm}","timestamp_utc":"{ts}"}}"#,
        ),
        ng = a.n,
        ns = a.n_steps,
        nt = a.n_ticks,
        wt = a.warmup_ticks,
        st = a.stress,
        rep = a.rep,
        h = host,
        pa = price,
        ov = ORACLE_VALUE,
        ae = err,
        pf = (err <= ORACLE_TOL),
        cnt = s.count,
        min = s.min,
        p50 = s.p50,
        p90 = s.p90,
        p95 = s.p95,
        p99 = s.p99,
        p999 = s.p999,
        p9999 = s.p9999,
        max = s.max,
        mean = s.mean,
        std = s.std,
        jit = jitter,
        rb = rb,
        ra = ra,
        la = loop_allocs,
        rc = rc,
        cm = a.git_sha,
        ts = ts,
    )
}

fn emit_abort_json(a: &Args, price: f64, err: f64) {
    let json = format!(
        concat!(
            r#"{{"schema_version":"0.1","library":"semiflow-core","variant":"reuse","#,
            r#""n_grid":{ng},"n_steps":{ns},"n_ticks":{nt},"#,
            r#""preflight":{{"price_at_S0":{pa:.6},"oracle_value":{ov:.6},"abs_err_at_S0":{ae:.2e},"passed":false}},"#,
            r#""result":"ABORT_CALIBRATION"}}"#,
        ),
        ng = a.n,
        ns = a.n_steps,
        nt = a.n_ticks,
        pa = price,
        ov = ORACLE_VALUE,
        ae = err,
    );
    write_out(&a.out_json, &json);
}

fn write_out(path: &str, content: &str) {
    use std::io::Write as _;
    if path == "/dev/stdout" || path.is_empty() {
        println!("{content}");
    } else {
        let mut f =
            std::fs::File::create(path).unwrap_or_else(|e| die(&format!("create {path}: {e}")));
        writeln!(f, "{content}").unwrap_or_else(|e| die(&format!("write {path}: {e}")));
    }
}

fn hostname() -> String {
    std::fs::read_to_string("/etc/hostname")
        .map_or_else(|_| "unknown".into(), |s| s.trim().to_owned())
}

fn rustc_version() -> String {
    std::process::Command::new("rustc")
        .arg("--version")
        .output()
        .map_or_else(
            |_| "unknown".into(),
            |o| String::from_utf8_lossy(&o.stdout).trim().to_owned(),
        )
}

fn timestamp() -> String {
    let secs = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |d| d.as_secs());
    let s = secs % 60;
    let m = (secs / 60) % 60;
    let h = (secs / 3600) % 24;
    let days = secs / 86400;
    let year = 1970 + days / 365;
    let doy = days % 365;
    let month = doy / 30 + 1;
    let day = doy % 30 + 1;
    format!("{year:04}-{month:02}-{day:02}T{h:02}:{m:02}:{s:02}Z")
}
