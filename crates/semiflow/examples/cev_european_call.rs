//! End-to-end demo: prices a European call under the CEV model via Chernoff
//! approximation. Mirrors the gates in `tests/cev_european_call.rs`.
//!
//! Canonical parameters (Schroder 1989 benchmark, v0.3.0):
//!   S₀=100, σ₀=0.30, β=0.5, T=1, K=100, r=0.05
//!   δ² = σ₀²·S₀^(2-2β) = 0.09·100 = 9.0
//!   a(S) = 4.5·S,  a'(S) = 4.5,  a''(S) = 0
//!
//! Run with: `cargo run --release --example cev_european_call -p semiflow-core`

// Integration test/bench/example: allows for numerical patterns.
#![allow(clippy::too_many_lines, clippy::unused_self)]

use semiflow::{
    grid::{BoundaryPolicy, InterpKind},
    ChernoffSemigroup, DiffusionChernoff, DriftReactionChernoff, Grid1D, GridFn1D, StrangSplit,
};

// Reference prices from Schroder closed form at canonical parameters (computed
// by running the v0.3.0 single-point benchmark):
//   C_oracle(S=80)  ≈ 2.3097
//   C_oracle(S=100) ≈ 14.24
//   C_oracle(S=120) ≈ 28.58
// (These are illustrative; exact values are computed live below.)

const S0: f64 = 100.0;
const K: f64 = 100.0;
const R: f64 = 0.05;
const T: f64 = 1.0;
const X_MIN: f64 = 1.0;
const X_MAX: f64 = 200.0;
const N_GRID: usize = 512;
const N_STEPS: usize = 256;

// Heap tracker (feature = "tracking-alloc"). GlobalAlloc requires unsafe impl;
// scoped here, same as src/simd/mod.rs which the workspace lint policy permits.
#[cfg(feature = "tracking-alloc")]
#[allow(unsafe_code)]
mod tracking {
    use std::{
        alloc::{GlobalAlloc, Layout, System},
        sync::atomic::{AtomicU64, AtomicUsize, Ordering::*},
    };
    pub struct Alloc;
    static CUR: AtomicUsize = AtomicUsize::new(0);
    static PEAK: AtomicUsize = AtomicUsize::new(0);
    static AC: AtomicU64 = AtomicU64::new(0); // alloc count  (sanity; not in JSON)
    static DC: AtomicU64 = AtomicU64::new(0); // dealloc count (sanity; not in JSON)
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
            DC.fetch_add(1, Relaxed);
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
        pub c: usize,
        pub p: usize,
    }
    pub fn snap() -> Snap {
        Snap {
            c: CUR.load(Acquire),
            p: PEAK.load(Acquire),
        }
    }
    pub fn reset() {
        PEAK.store(CUR.load(Acquire), Release);
    }
    // getrusage: 64-bit Linux layout: timeval×2 (4×i64) then ru_maxrss (idx 4); 18 i64 total.
    // Inline extern to avoid libc dep (contract §2).
    #[cfg(target_family = "unix")]
    pub fn rss() -> Option<u64> {
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
    pub fn rss() -> Option<u64> {
        None
    }
    pub struct Rec {
        pub ph: &'static str,
        pub c: u64,
        pub p: u64,
        pub d: i64,
        pub ns: u64,
        pub rss: Option<u64>,
    }
    impl Rec {
        pub fn jsonl(&self, n: usize, ns: usize) -> String {
            let r = self.rss.map_or_else(|| "null".into(), |v| v.to_string());
            format!(
                concat!(
                    r#"{{"phase":"{ph}","n":{n},"n_steps":{ns2},"current_bytes":{c},"#,
                    r#""peak_bytes":{p},"delta_bytes":{d},"elapsed_ns":{ns3},"getrusage_kb":{r}}}"#
                ),
                ph = self.ph,
                n = n,
                ns2 = ns,
                c = self.c,
                p = self.p,
                d = self.d,
                ns3 = self.ns,
                r = r
            )
        }
    }
    pub fn start(_: &'static str) -> (Snap, std::time::Instant) {
        reset();
        (snap(), std::time::Instant::now())
    }
    pub fn end(name: &'static str, s: (Snap, std::time::Instant)) -> Rec {
        let e = snap();
        Rec {
            ph: name,
            c: e.c as u64,
            p: e.p as u64,
            d: e.c as i64 - s.0.c as i64,
            ns: s.1.elapsed().as_nanos().min(u128::from(u64::MAX)) as u64,
            rss: rss(),
        }
    }
}
#[cfg(feature = "tracking-alloc")]
#[global_allocator]
static GLOBAL: tracking::Alloc = tracking::Alloc;

// ─── No-op shim when feature is off ──────────────────────────────────────────
#[cfg(not(feature = "tracking-alloc"))]
mod tracking {
    #[derive(Clone, Copy)]
    pub struct Snap;
    pub struct Rec;
    impl Rec {
        pub fn jsonl(&self, _n: usize, _ns: usize) -> String {
            String::new()
        }
    }
    pub fn start(_: &'static str) -> (Snap, std::time::Instant) {
        (Snap, std::time::Instant::now())
    }
    pub fn end(_: &'static str, _: (Snap, std::time::Instant)) -> Rec {
        Rec
    }
}

// ─── CLI ─────────────────────────────────────────────────────────────────────
#[derive(Clone, Copy, PartialEq, Eq)]
enum Measure {
    None,
    Warmup,
    Hotloop,
    Full,
    All,
}
struct Args {
    n: Option<usize>,
    ns: Option<usize>,
    m: Measure,
    json: bool,
}
fn die(s: &str) -> ! {
    eprintln!("error: {s}");
    std::process::exit(2);
}
fn parse_usize(it: &mut impl Iterator<Item = String>, flag: &str, min: usize) -> usize {
    match it.next().unwrap_or_default().parse::<usize>() {
        Ok(v) if v >= min => v,
        Ok(_) => die(&format!("{flag} must be >= {min}")),
        Err(_) => die(&format!("{flag} requires a numeric argument")),
    }
}
fn parse_cli() -> Args {
    let mut a = Args {
        n: None,
        ns: None,
        m: Measure::None,
        json: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(f) = it.next() {
        match f.as_str() {
            "--help" => {
                println!("Usage: cev_european_call [--n N] [--n-steps N_STEPS] [--measure MODE] [--json]");
                std::process::exit(0);
            }
            "--n" => a.n = Some(parse_usize(&mut it, "--n", 4)),
            "--n-steps" => a.ns = Some(parse_usize(&mut it, "--n-steps", 1)),
            "--measure" => {
                let v = it.next().unwrap_or_default();
                a.m = match v.as_str() {
                    "none" => Measure::None,
                    "warmup" => Measure::Warmup,
                    "hotloop" => Measure::Hotloop,
                    "full" => Measure::Full,
                    "all" => Measure::All,
                    o => die(&format!("unknown --measure value '{o}'")),
                };
            }
            "--json" => a.json = true,
            o => die(&format!("unknown flag '{o}'")),
        }
    }
    a
}

fn a_fn(s: f64) -> f64 {
    4.5_f64 * s
}
fn a_prime(_: f64) -> f64 {
    4.5_f64
}
fn a_dbl_prime(_: f64) -> f64 {
    0.0_f64
}
fn b_fn(s: f64) -> f64 {
    R * s - 4.5_f64
}
fn c_fn(_: f64) -> f64 {
    -R
}

fn make_grid(n: usize) -> Grid1D {
    Grid1D::new(X_MIN, X_MAX, n)
        .unwrap()
        .with_boundary(BoundaryPolicy::LinearExtrapolate)
        .with_interp(InterpKind::CubicHermite)
}

fn make_strang(grid: Grid1D) -> StrangSplit<DiffusionChernoff, DriftReactionChernoff> {
    let a_norm = 4.5_f64 * X_MAX;
    let diff = DiffusionChernoff::new(a_fn, a_prime, a_dbl_prime, a_norm, grid);
    let drift = DriftReactionChernoff::new(b_fn, c_fn, R, grid);
    StrangSplit::new(diff, drift)
}

/// Interpolate PDE solution at an arbitrary S in [`X_MIN`, `X_MAX`] (linear).
fn pde_price(u: &GridFn1D, s: f64) -> f64 {
    let grid = u.grid;
    let dx = grid.dx();
    // s ≥ X_MIN by construction; result is non-negative integer index.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i = ((s - X_MIN) / dx) as usize;
    if i + 1 >= grid.n {
        return u.values[grid.n - 1];
    }
    let frac = (s - grid.x_at(i)) / dx;
    u.values[i] * (1.0 - frac) + u.values[i + 1] * frac
}

fn main() {
    let full_s = tracking::start("full");
    let args = parse_cli();
    let n = args.n.unwrap_or(N_GRID);
    let n_steps = args.ns.unwrap_or(N_STEPS);

    let warm_s = tracking::start("warmup");
    let grid = make_grid(n);
    let strang = make_strang(grid);
    let f0 = GridFn1D::from_fn(grid, |s| (s - K).max(0.0));
    let sg = ChernoffSemigroup::new(strang, n_steps).expect("n >= 1");
    let warm_u = sg.evolve(T, &f0).expect("evolve ok");
    let warm_r = tracking::end("warmup", warm_s);

    // Hotloop: drop warmup result (-N bytes) then evolve (+N bytes) → delta = 0.
    let hot_s = tracking::start("hotloop");
    drop(warm_u);
    let u = sg.evolve(T, &f0).expect("evolve ok");
    let hot_r = tracking::end("hotloop", hot_s);

    // Emit human-readable output before dropping `u` (needed for pde_price).
    let query_points = [
        ("S=80  (OTM)", 80.0_f64),
        ("S=100 (ATM)", S0),
        ("S=120 (ITM)", 120.0_f64),
    ];
    let emit = |w: &mut dyn std::io::Write| {
        writeln!(
            w,
            "CEV European call prices (S₀={S0}, K={K}, σ₀=0.30, β=0.5, T={T}, r={R})"
        )
        .unwrap();
        writeln!(
            w,
            "{:<18}  {:>10}  {:>10}",
            "Point", "PDE price", "Intrinsic"
        )
        .unwrap();
        writeln!(w, "{}", "-".repeat(44)).unwrap();
        for (label, s) in query_points {
            let price = pde_price(&u, s);
            let intrinsic = (s - K).max(0.0);
            writeln!(w, "{label:<18}  {price:>10.6}  {intrinsic:>10.4}").unwrap();
        }
        writeln!(w).unwrap();
        writeln!(
            w,
            "ATM price ≈ 14.24 (Schroder 1989 closed form; see tests/cev_european_call.rs)"
        )
        .unwrap();
        writeln!(
            w,
            "Run `cargo test --release --test cev_european_call` to verify all three gates."
        )
        .unwrap();
    };

    if args.json && args.m != Measure::None {
        use std::io::Write as _;
        // Human output to stderr first (§4.3), then drop heap allocs, then record full.
        emit(&mut std::io::stderr());
        drop(u);
        drop(sg);
        drop(f0);
        let _ = grid;
        let full_r = tracking::end("full", full_s);
        let stdout = std::io::stdout();
        let mut out = stdout.lock();
        if matches!(args.m, Measure::Warmup | Measure::All) {
            writeln!(out, "{}", warm_r.jsonl(n, n_steps)).unwrap();
        }
        if matches!(args.m, Measure::Hotloop | Measure::All) {
            writeln!(out, "{}", hot_r.jsonl(n, n_steps)).unwrap();
        }
        if matches!(args.m, Measure::Full | Measure::All) {
            writeln!(out, "{}", full_r.jsonl(n, n_steps)).unwrap();
        }
    } else {
        emit(&mut std::io::stdout());
    }
}
