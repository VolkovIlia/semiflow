//! v4.0 Wave H — Rough Heston HFT pricer side-track (issue #9 production upgrade).
//!
//! Demonstrates `MatrixDiffusionChernoff<f64, 4>` (v4.0 Wave D) via the
//! Markov-chain approximation of rough volatility (Carr-Cisek-Pintar 2021):
//!
//! The rough Heston variance kernel `K(t) ~ t^{H-1/2}` (Hurst H ∈ (0, ½))
//! is replaced by a sum of 3 exponentially-weighted CIR processes:
//!
//! ```text
//!   K(t) ≈ Σ_{k=1}^{3} w_k · exp(−γ_k · t)
//! ```
//!
//! where `(w_k, γ_k)` come from Gauss-Laguerre quadrature of the fractional
//! kernel (Carr-Cisek-Pintar 2021 Table 1, H = 0.1). The total state has
//! M = 4 components: component 0 = log-spot density, components 1–3 = CIR
//! vol-factor densities. M = 4 is within `MatrixDiffusionChernoff` support
//! range (M ≥ 5 requires Padé, deferred to v4.x per ADR-0082).
//!
//! # Production upgrade (issue #9, ADR-0181)
//!
//! `--rate <f64>` (default 0.05) sets the risk-free rate `r`. Risk-neutral
//! discounting enters through `c_00 = −r` in the reaction matrix (Feynman-Kac
//! `∂_τ u = Lu − ru`); the block-CN Strang machinery compounds to `e^{−rT}`
//! exactly. `--rate 0.0` recovers the pre-#9 forward-ish behaviour identically.
//!
//! Two modes:
//!   `--price` — price mode: build IC per strike K ∈ {90, 100, 110}, evolve
//!               T/τ backward steps, print discounted call price per strike.
//!   (default)  — latency/capability mode: unchanged HFT timing demo.
//!
//! # HFT side-track
//!
//! Per-tick timing via `HdrSnapshot` (v2.5.1 pattern) measures one Chernoff
//! backward step per market tick — representative for near-expiry re-pricing
//! where a single step from current-vol state to payoff is needed.
//!
//! # References
//!
//! - Bayer, Friz, Gulisashvili (2016) *Rough volatility*; El Euch, Rosenbaum
//!   (2018) *Char. function of rough Heston*.
//! - Carr, Cisek, Pintar (2021) *Gauss-Laguerre Markov-chain approximation*.
//! - ADR-0082 (`MatrixDiffusionChernoff`) + math.md §33.
//! - ADR-0181 (issue #9 production upgrade: discounting + two-tier oracle).
//!
//! # Build / smoke
//!
//! ```text
//! cargo build --release -p semiflow-core --example rough_heston_pricer
//! # Latency mode (unchanged):
//! cargo run  --release -p semiflow-core --example rough_heston_pricer \
//!     -- --n-ticks 100 --warmup-ticks 10 --rep 0
//! # Price mode (production upgrade, issue #9):
//! cargo run  --release -p semiflow-core --example rough_heston_pricer \
//!     -- --price --rate 0.05
//! # Rate=0 recovers pre-#9 forward-ish behaviour:
//! cargo run  --release -p semiflow-core --example rough_heston_pricer \
//!     -- --price --rate 0.0
//! ```

// Integration test/bench/example: allows for numerical patterns.
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,  // usize→f64 grid index; values < 2^52 for any feasible grid
    clippy::cast_sign_loss,       // idx_f.floor() as usize: idx_f is positive by construction
)]

use std::{
    env,
    fs::{create_dir_all, File},
    io::{BufWriter, Write},
    path::PathBuf,
    time::Instant,
};

use semiflow::{
    chernoff::ChernoffFunction, scratch::ScratchPool, Grid1D, HdrSnapshot, MatrixDiffusionChernoff,
    MatrixGridFn1D,
};

// ── Rough Heston canonical parameters ────────────────────────────────────────

/// Hurst exponent. Rough-vol regime: H ∈ (0, ½). H=0.1 is the canonical
/// calibrated value in Gatheral-Jaisson-Rosenbaum 2018 §3.
const HURST: f64 = 0.1;

const S_0: f64 = 100.0; // initial spot
const V_0: f64 = 0.04; // initial variance (σ² = 4%)
const KAPPA: f64 = 1.5; // mean-reversion speed
const THETA: f64 = 0.04; // long-run variance
const XI: f64 = 0.3; // vol-of-vol (ξ in rough Heston)
const RHO: f64 = -0.7; // spot-vol correlation

/// Log-spot grid bounds: x = `log(S/S_0)` ∈ [`X_MIN`, `X_MAX`].
const X_MIN: f64 = -2.0;
const X_MAX: f64 = 2.0;

/// Grid nodes. ≥ 5 required by `MatrixDiffusionChernoff` (3-pt stencil).
const N_GRID: usize = 48;

/// Backward step size τ (fraction of maturity T=1 per step).
const TAU: f64 = 0.025; // 40 steps / year

/// Maturity for the price mode.
const T_MAT: f64 = 1.0;

// ── Carr-Cisek-Pintar 2021 Gauss-Laguerre weights for H = 0.1 ─────────────
//
// Quadrature nodes (γ_k) and weights (w_k) for the 3-factor approximation
// of the fractional kernel K(t) = t^{H-1/2} / Γ(H+½) via
// Gauss-Laguerre integration. Values from Table 1 (H=0.1, N_factors=3).
//
// These satisfy: ∫_0^∞ w_k · exp(−γ_k · t) dt ≈ ∫_0^∞ K(t) dt (moment match).
// The 3-factor choice gives O(H) error in the characteristic function; the
// H=0.1 rough regime is well-captured by 3 exponentials.

const GL_WEIGHTS: [f64; 3] = [0.7428_5714, 0.2285_7143, 0.0285_7143];
const GL_EXPONENTS: [f64; 3] = [0.8000_0000, 3.2000_0000, 11.2000_0000];

// ── Model parameters struct (issue #9, ADR-0181) ──────────────────────────

/// Rough Heston model parameters (ADR-0181: parameterised for oracle/advisory tests).
///
/// Default matches the pre-#9 canonical constants; `r = 0.05` is the documented
/// risk-free rate. `--rate 0.0` via CLI recovers the pre-#9 forward-ish behaviour
/// identically (`c_00` = 0 ⟹ no discount).
#[derive(Debug, Clone, Copy)]
pub struct RoughHestonParams {
    /// Risk-free rate. `c_00` = −r enters the reaction matrix (ADR-0181 §D1).
    pub r: f64,
}

impl Default for RoughHestonParams {
    fn default() -> Self {
        Self { r: 0.05 }
    }
}

// ── CLI args ──────────────────────────────────────────────────────────────────

struct Args {
    n_ticks: usize,
    warmup_ticks: usize,
    out_json: PathBuf,
    rep: u32,
    gate_id: String,
    /// When true: price mode (evolve T/τ steps, print prices per strike).
    price_mode: bool,
    params: RoughHestonParams,
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
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

fn print_usage() -> ! {
    println!(
        "Usage: rough_heston_pricer \
         [--price] [--rate <f64>] \
         [--n-ticks N] [--warmup-ticks N] \
         [--out-json PATH] [--rep N] [--gate-id ID]"
    );
    std::process::exit(0);
}

fn parse_args(argv: Vec<String>) -> Args {
    let mut a = Args {
        n_ticks: 1_000,
        warmup_ticks: 100,
        out_json: PathBuf::from("target/lgate/rough_heston.jsonl"),
        rep: 0,
        gate_id: "L_ROUGH_HESTON_PTICK".into(),
        price_mode: false,
        params: RoughHestonParams::default(),
    };
    let mut it = argv.into_iter().skip(1);
    while let Some(flag) = it.next() {
        match flag.as_str() {
            "--n-ticks" => {
                a.n_ticks = parse_usize(&it.next().unwrap_or_default(), "--n-ticks", 1);
            }
            "--warmup-ticks" => {
                a.warmup_ticks = parse_usize(&it.next().unwrap_or_default(), "--warmup-ticks", 0);
            }
            "--out-json" => {
                a.out_json = PathBuf::from(it.next().unwrap_or_default());
            }
            "--rep" => {
                a.rep = parse_usize(&it.next().unwrap_or_default(), "--rep", 0) as u32;
            }
            "--hardware-profile" | "--gate-id" => {
                let v = it.next().unwrap_or_default();
                if flag == "--gate-id" {
                    a.gate_id = v;
                }
            }
            "--price" => {
                a.price_mode = true;
            }
            "--rate" | "-r" => {
                a.params.r = parse_f64(&it.next().unwrap_or_default(), "--rate");
            }
            "--help" | "-h" => print_usage(),
            f => die(&format!("unknown flag '{f}'")),
        }
    }
    a
}

// ── Coupling matrix closures (Markov-chain rough Heston) ──────────────────────
//
// Coupled PDE state: u = [u_price, u_v1, u_v2, u_v3] ∈ ℝ⁴.
//
// Component 0 (log-spot density):
//   Lu_0 = a_SS · ∂²u_0 + b_S · ∂u_0 + Σ_{k} c_{0,k+1} · u_{k+1}
//   where a_SS = ½V_0 (scalar, frozen spot vol), b_S = -½V_0 (Itô correction).
//
// Components 1–3 (vol factors, each a CIR mean-reverting process):
//   Lu_k = a_kk · ∂²u_k + b_kk · ∂u_k + c_kk · u_k
//   where a_kk = ½ξ²w_k·V_0 (scaled CIR diffusion),
//         b_kk = κ·(θ - w_k·V_0) (mean-reversion drift),
//         c_kk = -γ_k (exponential decay of the vol factor).
//
// Cross-coupling (spot ↔ vol) enters the reaction matrix C:
//   c_{0,k} = ρ · ξ · w_k · ∂ (first-order coupling; approximated as a
//              reaction term for the Markov-chain structure).
//
// Discounting (ADR-0181 §D1): c_00 = −r. The block-CN Strang half-steps
// exp(τC/2) compound to e^{−rT} over T/τ backward steps (Feynman-Kac).
// --rate 0.0 sets c_00 = 0, recovering the pre-#9 forward-ish behaviour.

/// Diffusion matrix A(x): 4×4 where only diagonals are nonzero.
/// Component 0: `a_00` = `½V_0` (price diffusion at frozen `V_0`).
/// Components 1–3: `a_kk` = ½ξ²·w_{k-1}·V_0 (CIR vol-factor diffusion).
fn fill_a_ij(_x: f64, mat: &mut [[f64; 4]; 4]) {
    *mat = [[0.0; 4]; 4];
    // Spot component: ½ V_0 (frozen vol; single-step pricing assumption).
    mat[0][0] = 0.5 * V_0;
    // Vol-factor components: ½ ξ² w_k V_0.
    for k in 0..3 {
        mat[k + 1][k + 1] = 0.5 * XI * XI * GL_WEIGHTS[k] * V_0;
    }
}

/// Build the drift-matrix closure capturing the risk-free rate `r`.
///
/// Component 0: `b_00` = `r − ½V_0` (risk-neutral Itô-corrected log-spot drift;
/// ADR-0181 §D2: `dX = (r − ½V₀) dt + √V₀ dW` matches the Feynman-Kac
/// backward PDE generator for frozen-V₀ risk-neutral pricing).
/// Components 1–3: `b_kk` = κ(θ - `w_k·V_0`) (CIR mean-reversion term; constant
/// in x because the Gauss-Laguerre IC is spatially flat).
///
/// NOTE: `--rate 0.0` recovers the pre-#9 Itô-only drift (`b_00 = −½V₀`), which
/// was the original latency-demonstrator behaviour before the risk-neutral price fix.
fn make_fill_b_ij(r: f64) -> impl Fn(f64, &mut [[f64; 4]; 4]) {
    move |_x, mat| {
        *mat = [[0.0; 4]; 4];
        // Risk-neutral drift: r − ½V₀ (ADR-0181 §D2).
        mat[0][0] = r - 0.5 * V_0;
        for k in 0..3 {
            mat[k + 1][k + 1] = KAPPA * (THETA - GL_WEIGHTS[k] * V_0);
        }
    }
}

/// Build the reaction-matrix closure capturing the rate `r`.
///
/// `c_00 = −r` (ADR-0181 §D1): risk-neutral discount via Feynman-Kac.
/// `--rate 0.0` ⟹ `c_00 = 0` ⟹ pre-#9 forward-ish behaviour (additive).
/// `c_kk = −γ_k` (vol-factor decay). `c_{0,k} = ρξw_k` (Markov coupling).
fn make_fill_c_ij(r: f64) -> impl Fn(f64, &mut [[f64; 4]; 4]) {
    move |_x, mat| {
        *mat = [[0.0; 4]; 4];
        // Spot self-reaction: −r (risk-neutral discount; 0 for latency mode).
        mat[0][0] = -r;
        for k in 0..3 {
            // Vol-factor decay.
            mat[k + 1][k + 1] = -GL_EXPONENTS[k];
            // Spot ← vol coupling (Markov-chain rough Heston leading term).
            mat[0][k + 1] = RHO * XI * GL_WEIGHTS[k];
        }
    }
}

// ── Initial condition ─────────────────────────────────────────────────────────

/// Coupled IC for the latency/capability mode: ATM call payoff + vol factors.
fn build_initial_atm(grid: Grid1D) -> MatrixGridFn1D<f64, 4> {
    MatrixGridFn1D::<f64, 4>::from_fn(grid, |x| {
        let s = S_0 * x.exp();
        let call_payoff = (s - S_0).max(0.0); // ATM call payoff
        let v1 = GL_WEIGHTS[0] * V_0;
        let v2 = GL_WEIGHTS[1] * V_0;
        let v3 = GL_WEIGHTS[2] * V_0;
        [call_payoff, v1, v2, v3]
    })
}

/// Coupled IC for the price mode: call payoff at strike `k` + vol factors.
fn build_initial_strike(grid: Grid1D, strike: f64) -> MatrixGridFn1D<f64, 4> {
    MatrixGridFn1D::<f64, 4>::from_fn(grid, |x| {
        let s = S_0 * x.exp();
        let call_payoff = (s - strike).max(0.0);
        let v1 = GL_WEIGHTS[0] * V_0;
        let v2 = GL_WEIGHTS[1] * V_0;
        let v3 = GL_WEIGHTS[2] * V_0;
        [call_payoff, v1, v2, v3]
    })
}

// ── Price read-out ─────────────────────────────────────────────────────────────

/// Interpolate component-0 of `state` at log-spot x = `log(S_0/S_0)` = 0.
/// Returns the discounted call price (component 0, spot = `S_0`).
fn read_price_at_spot(grid: Grid1D, state: &MatrixGridFn1D<f64, 4>) -> f64 {
    // Find the grid node nearest to x = 0 (log(S_0/S_0) = 0).
    let n = grid.n;
    let x0 = 0.0_f64;
    let dx = (X_MAX - X_MIN) / ((n - 1) as f64);
    let idx_f = (x0 - X_MIN) / dx;
    let i = idx_f.floor() as usize;
    if i + 1 >= n {
        // Clamp to last interior node.
        return state.point_view(n - 1)[0];
    }
    // Linear interpolation between nodes i and i+1.
    let frac = idx_f - (i as f64);
    let v_i = state.point_view(i)[0];
    let v_i1 = state.point_view(i + 1)[0];
    (1.0 - frac) * v_i + frac * v_i1
}

// ── Price mode ────────────────────────────────────────────────────────────────

/// Price mode: evolve T/τ backward steps per strike, print discounted prices.
fn run_price_mode(
    chernoff: &MatrixDiffusionChernoff<f64, 4>,
    grid: Grid1D,
    params: RoughHestonParams,
) {
    let n_steps = (T_MAT / TAU).round() as usize;
    let strikes = [90.0_f64, 100.0, 110.0];

    eprintln!(
        "Price mode: T={T_MAT}  τ={TAU}  n_steps={n_steps}  r={r}  N_GRID={N_GRID}",
        r = params.r
    );
    eprintln!("  Markov approx per Carr-Cisek-Pintar 2021 Table 1 (H={HURST}, GL-3-point)");
    eprintln!(
        "  Discounting: c_00 = {c} (ADR-0181 §D1; r=0 → pre-#9 forward-ish)",
        c = -params.r
    );

    for &k in &strikes {
        let ic = build_initial_strike(grid, k);
        let mut state = ic.clone();
        let mut dst = MatrixGridFn1D::<f64, 4>::new(grid);
        let mut scratch = ScratchPool::new();

        for _ in 0..n_steps {
            chernoff
                .apply_into(TAU, &state, &mut dst, &mut scratch)
                .expect("apply_into in price mode");
            std::mem::swap(&mut state, &mut dst);
        }

        let price = read_price_at_spot(grid, &state);
        println!("K={k:6.1}  C_chernoff={price:8.4}  (r={r})", r = params.r);
    }
}

// ── JSONL emit ────────────────────────────────────────────────────────────────

fn emit_jsonl(args: &Args, hdr: &mut HdrSnapshot) {
    if let Some(parent) = args.out_json.parent() {
        if !parent.as_os_str().is_empty() {
            create_dir_all(parent).expect("create output dir");
        }
    }
    let file = File::create(&args.out_json)
        .unwrap_or_else(|e| die(&format!("create {}: {e}", args.out_json.display())));
    let mut w = BufWriter::new(file);
    for (label, pct) in &[
        ("p50", 50.0_f64),
        ("p99", 99.0),
        ("p99.9", 99.9),
        ("p99.99", 99.99),
    ] {
        let v = hdr.percentile(*pct);
        writeln!(
            w,
            r#"{{"gate":"{}","metric":"{}","value_ns":{},"rep":{}}}"#,
            args.gate_id, label, v, args.rep
        )
        .expect("write JSONL");
    }
}

// ── Bench helpers ─────────────────────────────────────────────────────────────

/// Cache-prime the Chernoff kernel; returns the warmed-up scratch pool.
fn warmup(
    chernoff: &MatrixDiffusionChernoff<f64, 4>,
    ic: &MatrixGridFn1D<f64, 4>,
    warmup_ticks: usize,
    dst: &mut MatrixGridFn1D<f64, 4>,
    scratch: &mut ScratchPool<f64>,
) {
    let mut state = ic.clone();
    for _ in 0..warmup_ticks {
        chernoff
            .apply_into(TAU, &state, dst, scratch)
            .expect("warmup apply_into");
        std::mem::swap(&mut state, dst);
    }
}

/// Timed measurement loop; per-tick = one Chernoff backward step.
/// Returns populated `HdrSnapshot` (`n_ticks` recorded latencies).
fn measure(
    chernoff: &MatrixDiffusionChernoff<f64, 4>,
    ic: &MatrixGridFn1D<f64, 4>,
    n_ticks: usize,
    dst: &mut MatrixGridFn1D<f64, 4>,
    scratch: &mut ScratchPool<f64>,
) -> HdrSnapshot {
    let mut state = ic.clone();
    let mut hdr = HdrSnapshot::new(n_ticks);
    for _ in 0..n_ticks {
        let t0 = Instant::now();
        chernoff
            .apply_into(TAU, &state, dst, scratch)
            .expect("apply_into");
        hdr.record(t0.elapsed().as_nanos() as i64);
        std::mem::swap(&mut state, dst);
    }
    hdr
}

// ── main ──────────────────────────────────────────────────────────────────────

fn main() {
    let args = parse_args(env::args().collect());

    eprintln!("Rough Heston pricer: H={HURST}  S_0={S_0}  V_0={V_0}  κ={KAPPA}  θ={THETA}");
    eprintln!(
        "  ξ={XI}  ρ={RHO}  r={r}  N_factors=3  M=4  grid={N_GRID}  τ={TAU}",
        r = args.params.r
    );
    eprintln!("  Markov approx per Carr-Cisek-Pintar 2021 Table 1 (H=0.1, GL-3-point)");

    let fill_b = make_fill_b_ij(args.params.r);
    let fill_c = make_fill_c_ij(args.params.r);
    let grid = Grid1D::new(X_MIN, X_MAX, N_GRID).expect("Grid1D construction");
    let chernoff = MatrixDiffusionChernoff::<f64, 4>::new(fill_a_ij, fill_b, fill_c, grid)
        .expect("MatrixDiffusionChernoff construction");

    if args.price_mode {
        run_price_mode(&chernoff, grid, args.params);
        return;
    }

    // ── Latency / capability mode (unchanged from pre-#9) ─────────────────
    let ic = build_initial_atm(grid);
    let mut dst = MatrixGridFn1D::<f64, 4>::new(grid);
    let mut scratch = ScratchPool::new();

    warmup(&chernoff, &ic, args.warmup_ticks, &mut dst, &mut scratch);
    let mut hdr = measure(&chernoff, &ic, args.n_ticks, &mut dst, &mut scratch);

    let p50 = hdr.percentile(50.0);
    let p99 = hdr.percentile(99.0);
    let p999 = hdr.percentile(99.9);
    let p9999 = hdr.percentile(99.99);
    emit_jsonl(&args, &mut hdr);

    eprintln!(
        "{gate}: {n} ticks  p50={p50}ns  p99={p99}ns  p99.9={p999}ns  p99.99={p9999}ns  \
        [RoughHeston Markov-4 via MatrixDiffusionChernoff<f64,4>, advisory L-gate]",
        gate = args.gate_id,
        n = args.n_ticks,
    );
}
