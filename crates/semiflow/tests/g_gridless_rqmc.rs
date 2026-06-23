//! `G_GRIDLESS_RQMC` — path-space RQMC decisive experiment (§3, ADR-0158, v9.0.0)
//!
//! **Purpose:** Measure whether Sobol + Owen-scramble + Brownian-bridge RQMC achieves
//! ≥ 2× lower MSE than plain MC at equal budget for the anisotropic heat functional
//! at d ∈ {4, 8} (and optionally d=10).  This is the SINGLE MISSING LEG of the §50.5
//! airtight refutation — see `v9-shift-c-resolution-research.md` §2.2 leg 3.
//!
//! ## Anti-gaming design (§3.1 / research §2.1 and §5.5)
//!
//! The TWO arms (RQMC and MC) differ in EXACTLY ONE THING: how the P path-points
//! are chosen.  Everything else is IDENTICAL:
//! - Same budget P per replication.
//! - Same number of replications R (RQMC randomised via independent Owen scrambles;
//!   MC via independent LCG seeds).
//! - Same smooth functional f(x) = ∏_j `cos(ξ_j` `x_j`).
//! - Same closed-form truth E[f] = ∏_j exp(−T `ξ_j²` `a_j`).
//! - Same path-space model: `X_T,j` = Brownian motion with diffusion `a_j` (per axis).
//!   With n sub-steps the path is `X_T,j` = sum of n Gaussian increments.
//! - MSE = `mean_R`[(`estimate_r` − truth)²] is the ONLY score.
//!
//! A single fixed Sobol sequence has no measurable variance.
//! The Owen scramble per replication is what makes RQMC a genuine estimator with
//! measurable variance (this was the prior-attempt bug, now fixed).
//!
//! ## Pre-registered pass criterion (§3.2 — HARD-CODED)
//!
//! PASS (§50.5 CONFIRMED): `MSE_MC` / `MSE_RQMC(Sobol+Owen+BB)` ≥ 2.0
//!   at BOTH d=4 AND d=8.
//!
//! FAIL (§50.5 FALSIFIED-by-measurement): ratio < 2.0 at d=4 OR d=8.
//!
//! ## Parameters
//!
//! - P = 1024 paths per replication (stable MC MSE regime)
//! - R = 64 replications (stable MSE estimate)
//! - n = per-d (must satisfy n*d ≤ `MAX_SOBOL_DIM=21)`:
//!   d=4: n=4 (n*d=16)
//!   d=8: n=2 (n*d=16)
//!   d=10: n=2 (n*d=20)
//! - T = 1.0
//!
//! ## No new deps: all arithmetic is integer XOR / bitwise (Sobol + Owen) +
//! Moro inverse-normal (rational approx, all arithmetic) + LCG64.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::unreadable_literal)]
#![allow(clippy::too_many_arguments)]
#![allow(clippy::too_many_lines)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::many_single_char_names)]

extern crate alloc;

// ═══════════════════════════════════════════════════════════════════════════════
// §A — LCG pseudo-random number generator (for MC baseline)
// ═══════════════════════════════════════════════════════════════════════════════

struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    fn new(seed: u64) -> Self {
        Self {
            state: seed.wrapping_add(1),
        }
    }

    fn next_u64(&mut self) -> u64 {
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    fn next_unit(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0 + 1e-15
    }

    fn next_std_normal(&mut self) -> f64 {
        // Box-Muller
        let u1 = self.next_unit();
        let u2 = (self.next_u64() >> 11) as f64 / 9_007_199_254_740_992.0;
        let r = libm::sqrt(-2.0 * libm::log(u1));
        r * libm::cos(core::f64::consts::TAU * u2)
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §B — Moro (1995) rational approximation to Φ⁻¹(p)
//
// Verified: inv_normal(0.1) ≈ −1.2816, inv_normal(0.5) = 0, inv_normal(0.9) ≈ 1.2816.
// This is the CORRECT inverse normal used for the QMC uniform → normal transform.
// (Note: g_gridless.rs uses a different Acklam formula that has a transcription
// error — it gives wrong values. This Moro implementation has been independently
// verified in the debug scripts before integration.)
// ═══════════════════════════════════════════════════════════════════════════════

/// Inverse normal CDF: Φ⁻¹(p).
///
/// Uses the Abramowitz & Stegun 26.2.17 rational approximation (tail region)
/// combined with the Moro-like central rational for the interior.
/// Accuracy ~3 decimal places (sufficient for QMC path generation).
///
/// Verified: `inv_normal(0.1587)` ≈ -1.0, `inv_normal(0.5)` = 0, `inv_normal(0.001)` ≈ -3.09.
fn inv_normal(p: f64) -> f64 {
    // Central region: Moro rational (|y| < 0.42)
    const A: [f64; 4] = [
        2.50662823884,
        -18.61500062529,
        41.39119773534,
        -25.44106049637,
    ];
    const B: [f64; 4] = [
        -8.47351093090,
        23.08336743743,
        -21.06224101826,
        3.13082909833,
    ];
    // Tail region: Abramowitz & Stegun 26.2.17 (works for all p including extremes)
    const C: [f64; 3] = [2.515517, 0.802853, 0.010328];
    const D: [f64; 3] = [1.432788, 0.189269, 0.001308];

    let y = p - 0.5;
    if y.abs() < 0.42 {
        let r = y * y;
        let num = (((A[3] * r + A[2]) * r + A[1]) * r + A[0]) * y;
        let den = (((B[3] * r + B[2]) * r + B[1]) * r + B[0]) * r + 1.0;
        num / den
    } else {
        // Abramowitz & Stegun 26.2.17 (one-sided)
        let r = if p < 0.5 { p } else { 1.0 - p };
        let t = libm::sqrt(-2.0 * libm::log(r));
        let z = t
            - (C[0] + C[1] * t + C[2] * t * t) / (1.0 + D[0] * t + D[1] * t * t + D[2] * t * t * t);
        if p < 0.5 {
            -z
        } else {
            z
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════════
// §C — Per-axis anisotropic model (same as g_gridless.rs)
// ═══════════════════════════════════════════════════════════════════════════════

fn a_j(j: usize) -> f64 {
    0.5 * (1.0 + 0.1 * j as f64)
}
fn xi_j(j: usize) -> f64 {
    1.0 / (1.0 + 0.05 * j as f64)
}

/// Closed-form `E[∏_j cos(ξ_j X_T,j)] = ∏_j exp(−T ξ_j² a_j)`.
fn truth_d(d: usize, t: f64) -> f64 {
    (0..d)
        .map(|j| libm::exp(-t * xi_j(j) * xi_j(j) * a_j(j)))
        .product()
}

/// Product functional `f(x) = ∏_j cos(ξ_j x_j)`.
fn functional(pos: &[f64], d: usize) -> f64 {
    (0..d).map(|j| libm::cos(xi_j(j) * pos[j])).product()
}

// ═══════════════════════════════════════════════════════════════════════════════
// §D — Joe-Kuo Sobol direction numbers (dims 1..=MAX_SOBOL_DIM)
//
// Source: S. Joe and F. Y. Kuo, "Constructing Sobol sequences with better
// two-dimensional projections," SIAM J. Sci. Comput. 30 (2008) 2635-2654.
// Direction-number initialisation from the standard Joe-Kuo table.
// All integers stored at word width w=32 bits.
// ═══════════════════════════════════════════════════════════════════════════════

const SOBOL_BITS: u32 = 32;
const MAX_SOBOL_DIM: usize = 21;

const JK_INIT: &[(u32, u32, &[u32])] = &[
    (1, 0, &[1]),
    (2, 1, &[1, 1]),
    (3, 1, &[1, 1, 1]),
    (3, 2, &[1, 1, 3]),
    (4, 1, &[1, 1, 3, 3]),
    (4, 4, &[1, 3, 5, 13]),
    (5, 2, &[1, 1, 1, 1, 1]),
    (5, 4, &[1, 1, 1, 3, 7]),
    (5, 7, &[1, 1, 7, 9, 23]),
    (5, 11, &[1, 1, 5, 9, 29]),
    (5, 13, &[1, 1, 1, 3, 13]),
    (6, 9, &[1, 3, 5, 3, 7, 35]),
    (6, 20, &[1, 3, 7, 5, 5, 9]),
    (6, 22, &[1, 1, 1, 15, 11, 3]),
    (6, 31, &[1, 3, 7, 11, 23, 35]),
    (7, 4, &[1, 1, 7, 3, 23, 35, 5]),
    (7, 14, &[1, 3, 7, 9, 13, 63, 25]),
    (7, 22, &[1, 1, 7, 11, 19, 25, 17]),
    (7, 55, &[1, 3, 7, 13, 7, 5, 9]),
    (7, 62, &[1, 3, 5, 13, 13, 7, 27]),
];

fn build_direction_vectors() -> [[u32; 32]; MAX_SOBOL_DIM] {
    let mut v = [[0u32; 32]; MAX_SOBOL_DIM];

    // Dimension 1: standard van-der-Corput base 2
    for k in 0..32 {
        v[0][k] = 1u32 << (SOBOL_BITS - 1 - k as u32);
    }

    // Dimensions 2..=MAX_SOBOL_DIM from Joe-Kuo table (JK_INIT[i] → dim i+2)
    for (di, &(s, a, m)) in JK_INIT.iter().enumerate() {
        let dim = di + 1; // 0-indexed in v[]
                          // Initial direction numbers: v[dim][k] = m[k] * 2^{w-1-k}
        for k in 0..s as usize {
            v[dim][k] = m[k] << (SOBOL_BITS - 1 - k as u32);
        }
        // Recurrence for k >= s:  v_k = v_{k-s} XOR (v_{k-s} >> s)
        //   XOR ( a_{s-1} * v_{k-1} XOR ... XOR a_1 * v_{k-s+1} )
        // where a_i = bit i of the polynomial `a` (bit s-1-i of a, 1-indexed)
        for k in s as usize..32 {
            let mut vk = v[dim][k - s as usize] ^ (v[dim][k - s as usize] >> s);
            for i in 1..s as usize {
                if (a >> (s - 1 - i as u32)) & 1 == 1 {
                    vk ^= v[dim][k - i];
                }
            }
            v[dim][k] = vk;
        }
    }
    v
}

// ═══════════════════════════════════════════════════════════════════════════════
// §E — Owen (nested-uniform) scrambling
//
// Each bit at position k (from MSB=31 downward) is XOR'd with a bit derived
// from a hash of (seed, upper bits of x, k).  This preserves the stratification
// of the Sobol sequence while randomising it — giving RQMC measurable variance.
//
// Wang hash is used as the hash function (fast, good avalanche).
// Each dimension gets its own seed derived from (base, dim, replication).
// ═══════════════════════════════════════════════════════════════════════════════

fn wang_hash(mut x: u64) -> u64 {
    x = x.wrapping_add(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
    x ^ (x >> 31)
}

/// Apply one round of Owen scramble to integer x using the given seed.
fn owen_scramble(x: u32, seed: u64) -> u32 {
    let mut result = x;
    let mut upper: u32 = 0; // upper bits of the ORIGINAL x (for tree-path dependency)
    for k in (0..32_u32).rev() {
        let hash_in =
            wang_hash(seed ^ u64::from(upper) ^ u64::from(k).wrapping_mul(0x9E3779B97F4A7C15));
        let flip = ((hash_in >> 63) as u32) & 1;
        let orig_bit = (x >> k) & 1; // use original x for upper bits (not mutated)
        let new_bit = ((result >> k) & 1) ^ flip;
        result = (result & !(1u32 << k)) | (new_bit << k);
        upper = (upper << 1) | orig_bit;
    }
    result
}

/// Generate one P-th Owen-scrambled Sobol point in `n_dims` dimensions.
/// `scramble_base` + `replication` together seed the per-dimension scrambles
/// so that each replication gets an independent scramble.
fn sobol_owen_point(
    i: u32,
    dir: &[[u32; 32]; MAX_SOBOL_DIM],
    n_dims: usize,
    scramble_base: u64,
    replication: u64,
) -> alloc::vec::Vec<f64> {
    debug_assert!(n_dims <= MAX_SOBOL_DIM);
    let scale = f64::from(u32::MAX) + 1.0;
    let g = i ^ (i >> 1); // Gray code

    // Accumulate direction vectors for each dimension
    let mut raw = alloc::vec![0u32; n_dims];
    for k in 0..32 {
        if (g >> k) & 1 == 1 {
            for d in 0..n_dims {
                raw[d] ^= dir[d][k];
            }
        }
    }

    // Owen-scramble each dimension with an independent seed
    raw.iter()
        .enumerate()
        .map(|(d, &x)| {
            let seed = wang_hash(
                scramble_base.wrapping_add((d as u64).wrapping_mul(0x9E3779B97F4A7C15))
                    ^ replication.wrapping_mul(0x6C62272E07BB0142),
            );
            let scrambled = owen_scramble(x, seed);
            // Map to (0,1) — clamp away from exact 0 and 1 for inv_normal stability
            (f64::from(scrambled) / scale).clamp(1e-8, 1.0 - 1e-8)
        })
        .collect()
}

// ═══════════════════════════════════════════════════════════════════════════════
// §F — Brownian-bridge path ordering for n sub-steps
//
// Standard binary bisection ordering: the increment spanning the widest time
// interval is assigned the LEADING Sobol coordinate (lowest discrepancy).
//
// For n steps (power-of-2), the BFS bisection of the interval [0,n) gives:
//   n=4:  order = [2, 1, 3, 0]  (step 2 = midpoint gets coord 0)
//   n=8:  order = [4, 2, 6, 1, 3, 5, 7, 0]
//
// `order[sobol_coord_k]` = time_step_index.
// ═══════════════════════════════════════════════════════════════════════════════

fn bb_order(n: usize) -> alloc::vec::Vec<usize> {
    debug_assert!(n.is_power_of_two());
    let mut perm = alloc::vec::Vec::with_capacity(n);
    let mut queue: alloc::collections::VecDeque<(usize, usize)> =
        alloc::collections::VecDeque::new();
    // half-open [lo, hi): pick midpoint (lo+hi)/2 first
    queue.push_back((0, n));
    while let Some((lo, hi)) = queue.pop_front() {
        let mid = (lo + hi) / 2;
        perm.push(mid);
        if mid > lo {
            queue.push_back((lo, mid));
        }
        if mid + 1 < hi {
            queue.push_back((mid + 1, hi));
        }
    }
    perm
}

// ═══════════════════════════════════════════════════════════════════════════════
// §G — Path builders (terminal position X_T from X_0 = 0)
// ═══════════════════════════════════════════════════════════════════════════════

/// MC random walk: n Euler-Maruyama steps per axis, fresh LCG increments.
fn build_path_mc(d: usize, n_steps: usize, tau: f64, lcg: &mut Lcg64) -> alloc::vec::Vec<f64> {
    let mut pos = alloc::vec![0.0f64; d];
    for _ in 0..n_steps {
        for j in 0..d {
            let sigma = libm::sqrt(2.0 * a_j(j) * tau);
            pos[j] += sigma * lcg.next_std_normal();
        }
    }
    pos
}

/// Plain Sobol RQMC (no BB reordering): uniform → normal per (step,axis) increment.
/// `u[step * d + j]` = Sobol coordinate for (step, axis j).
fn build_path_sobol_no_bb(d: usize, n_steps: usize, tau: f64, u: &[f64]) -> alloc::vec::Vec<f64> {
    let mut pos = alloc::vec![0.0f64; d];
    for step in 0..n_steps {
        for j in 0..d {
            let sigma = libm::sqrt(2.0 * a_j(j) * tau);
            pos[j] += sigma * inv_normal(u[step * d + j]);
        }
    }
    pos
}

/// Sobol + Brownian-bridge RQMC:
/// Leading Sobol coordinate (k=0) drives the BB midpoint increment,
/// so the coarse path shape consumes the most stratified coordinates.
/// `u[sobol_k * d + j]` = Sobol coordinate k for axis j.
/// `order[sobol_k]` = which time-step that coordinate drives.
fn build_path_sobol_bb(
    d: usize,
    n_steps: usize,
    tau: f64,
    u: &[f64],
    order: &[usize],
) -> alloc::vec::Vec<f64> {
    // Accumulate increments into a temporary array indexed by time-step
    let mut inc = alloc::vec![[0.0f64; 16]; n_steps]; // n_steps ≤ 16 guaranteed by caller
    for (sobol_k, &step_idx) in order.iter().enumerate() {
        for j in 0..d {
            let sigma = libm::sqrt(2.0 * a_j(j) * tau);
            inc[step_idx][j] = sigma * inv_normal(u[sobol_k * d + j]);
        }
    }
    let mut pos = alloc::vec![0.0f64; d];
    for step in 0..n_steps {
        for j in 0..d {
            pos[j] += inc[step][j];
        }
    }
    pos
}

// ═══════════════════════════════════════════════════════════════════════════════
// §H — Single-replication estimators
// ═══════════════════════════════════════════════════════════════════════════════

fn estimate_mc(d: usize, n_steps: usize, t: f64, p: usize, lcg: &mut Lcg64) -> f64 {
    let tau = t / n_steps as f64;
    let sum: f64 = (0..p)
        .map(|_| {
            let path = build_path_mc(d, n_steps, tau, lcg);
            functional(&path, d)
        })
        .sum();
    sum / p as f64
}

fn estimate_sobol_no_bb(
    d: usize,
    n_steps: usize,
    t: f64,
    p: usize,
    dir: &[[u32; 32]; MAX_SOBOL_DIM],
    scramble_base: u64,
    rep: u64,
) -> f64 {
    let tau = t / n_steps as f64;
    let n_dims = n_steps * d;
    let sum: f64 = (0..p as u32)
        .map(|i| {
            let u = sobol_owen_point(i, dir, n_dims, scramble_base, rep);
            let path = build_path_sobol_no_bb(d, n_steps, tau, &u);
            functional(&path, d)
        })
        .sum();
    sum / p as f64
}

fn estimate_sobol_bb(
    d: usize,
    n_steps: usize,
    t: f64,
    p: usize,
    dir: &[[u32; 32]; MAX_SOBOL_DIM],
    scramble_base: u64,
    rep: u64,
    order: &[usize],
) -> f64 {
    let tau = t / n_steps as f64;
    let n_dims = n_steps * d;
    let sum: f64 = (0..p as u32)
        .map(|i| {
            let u = sobol_owen_point(i, dir, n_dims, scramble_base, rep);
            let path = build_path_sobol_bb(d, n_steps, tau, &u, order);
            functional(&path, d)
        })
        .sum();
    sum / p as f64
}

// ═══════════════════════════════════════════════════════════════════════════════
// §I — MSE over R replications and the main experiment runner
// ═══════════════════════════════════════════════════════════════════════════════

fn mse_over_reps(estimates: &[f64], truth: f64) -> f64 {
    estimates.iter().map(|&e| (e - truth).powi(2)).sum::<f64>() / estimates.len() as f64
}

/// Run the three-arm experiment for a given dimension `d` and sub-step count `n_steps`.
/// Returns `(mse_mc, mse_nobb, mse_bb)`.
fn run_experiment(
    d: usize,
    n_steps: usize,
    t: f64,
    p: usize,
    r_reps: usize,
) -> (f64, f64, f64, f64) {
    debug_assert!(n_steps * d <= MAX_SOBOL_DIM);
    debug_assert!(n_steps.is_power_of_two());

    let truth = truth_d(d, t);
    let dir = build_direction_vectors();
    let order = bb_order(n_steps);

    // Two different scramble bases: one for no-BB, one for BB.
    // They are independent — both derived from a fixed experiment-wide constant
    // combined with the dimension and step count.
    let base_nobb: u64 = 0xDEAD_BEEF_0000_0000u64
        .wrapping_add((d as u64).wrapping_mul(0x9E3779B97F4A7C15))
        .wrapping_add((n_steps as u64).wrapping_mul(0x6C62272E07BB0142));
    let base_bb: u64 = base_nobb ^ 0xAAAA_BBBB_CCCC_DDDDu64;

    let mut mc_ests = alloc::vec::Vec::with_capacity(r_reps);
    let mut nobb_ests = alloc::vec::Vec::with_capacity(r_reps);
    let mut bb_ests = alloc::vec::Vec::with_capacity(r_reps);

    for rep in 0..r_reps {
        let mc_seed = 0xC0FF_EE00_0000_0000u64.wrapping_add((rep as u64).wrapping_mul(1_000_003));
        let mut lcg = Lcg64::new(mc_seed);
        mc_ests.push(estimate_mc(d, n_steps, t, p, &mut lcg));

        nobb_ests.push(estimate_sobol_no_bb(
            d, n_steps, t, p, &dir, base_nobb, rep as u64,
        ));
        bb_ests.push(estimate_sobol_bb(
            d, n_steps, t, p, &dir, base_bb, rep as u64, &order,
        ));
    }

    let mse_mc = mse_over_reps(&mc_ests, truth);
    let mse_nobb = mse_over_reps(&nobb_ests, truth);
    let mse_bb = mse_over_reps(&bb_ests, truth);

    // Anti-gaming asserts (§3.1): both MC and RQMC must be genuine estimators
    assert!(mse_mc > 0.0, "MSE_MC = 0 — MC estimator is degenerate");
    assert!(
        mse_bb > 0.0,
        "MSE_RQMC_BB = 0 — RQMC estimator is degenerate"
    );
    assert!(
        mse_nobb > 0.0,
        "MSE_RQMC_noBB = 0 — RQMC no-BB estimator is degenerate"
    );

    (mse_mc, mse_nobb, mse_bb, truth)
}

fn run_and_report_d(
    d: usize,
    n_steps: usize,
    p: usize,
    r: usize,
    t: f64,
) -> (f64, f64, f64, f64, f64)
// Returns (mse_mc, mse_nobb, mse_bb, ratio_bb, ratio_nobb)
{
    println!(
        "\n  d={d}: n_steps={n_steps}, n*d={}, P={p}, R={r}, T={t}",
        n_steps * d
    );
    let (mse_mc, mse_nobb, mse_bb, truth) = run_experiment(d, n_steps, t, p, r);

    let ratio_bb = mse_mc / mse_bb.max(1e-300);
    let ratio_nobb = mse_mc / mse_nobb.max(1e-300);

    println!("  truth            = {truth:.8}");
    println!("  MSE_MC           = {mse_mc:.4e}");
    println!("  MSE_Sobol_noBB   = {mse_nobb:.4e}  ratio(noBB) = {ratio_nobb:.3}×");
    println!("  MSE_Sobol_BB     = {mse_bb:.4e}  ratio(BB)   = {ratio_bb:.3}×");

    let verdict = if ratio_bb >= 2.0 {
        "PASS ≥2×"
    } else {
        "FAIL <2×"
    };
    println!("  d={d} verdict(BB): {verdict}");

    (mse_mc, mse_nobb, mse_bb, ratio_bb, ratio_nobb)
}

// ═══════════════════════════════════════════════════════════════════════════════
// §J — Self-calibration sanity check (validates the inv_normal and Sobol)
// ═══════════════════════════════════════════════════════════════════════════════

/// Sanity-check: E[cos(Z)] with d=1, n=1 (single-step terminal), P=1024, R=8.
/// Truth = `exp(-a_0 * xi_0^2 * T)` = `exp(-0.5 * 1.0 * 1.0)` = `exp(-0.5)` ≈ 0.6065.
/// Both MC and Sobol+Owen should be close to truth (within a few % at P=1024).
fn self_calibration_check() {
    println!("\n  [Self-calibration] d=1, n=1, P=1024, R=8:");
    println!("  truth = {:.6}", truth_d(1, 1.0));

    let dir = build_direction_vectors();
    // d=1, n=1, n*d=1 Sobol dimension
    let p = 1024usize;

    // Verify inv_normal at a few points
    println!(
        "  inv_normal(0.1587) = {:.4} (expect ≈ -1.0)",
        inv_normal(0.1587)
    );
    println!(
        "  inv_normal(0.5000) = {:.4} (expect   0.0)",
        inv_normal(0.5)
    );
    println!(
        "  inv_normal(0.8413) = {:.4} (expect ≈  1.0)",
        inv_normal(0.8413)
    );

    let t = 1.0f64;
    let sigma = libm::sqrt(2.0 * a_j(0) * t);
    let truth1d = truth_d(1, t);
    let mut errors_sobol = alloc::vec::Vec::new();
    let mut errors_mc = alloc::vec::Vec::new();
    let base: u64 = 0x1234_5678_9ABC_DEF0;
    for rep in 0..8u64 {
        let sobol_est: f64 = (0..p as u32)
            .map(|i| {
                let u = sobol_owen_point(i, &dir, 1, base, rep);
                libm::cos(xi_j(0) * sigma * inv_normal(u[0]))
            })
            .sum::<f64>()
            / p as f64;
        errors_sobol.push((sobol_est - truth1d).abs());

        let mc_seed = 0xDEAD_0000_0000_0000u64.wrapping_add(rep.wrapping_mul(999_983));
        let mut lcg = Lcg64::new(mc_seed);
        let mc_est: f64 = (0..p)
            .map(|_| libm::cos(xi_j(0) * sigma * lcg.next_std_normal()))
            .sum::<f64>()
            / p as f64;
        errors_mc.push((mc_est - truth1d).abs());
    }
    let mean_err_s = errors_sobol.iter().sum::<f64>() / 8.0;
    let mean_err_m = errors_mc.iter().sum::<f64>() / 8.0;
    println!("  Mean |error| Sobol+Owen: {mean_err_s:.4e}");
    println!("  Mean |error| MC:         {mean_err_m:.4e}");
    assert!(
        mean_err_s < 0.05,
        "Self-calibration FAIL: Sobol+Owen 1D error {mean_err_s:.4e} >= 5e-2 — \
         inv_normal or Sobol generation is broken"
    );
    assert!(
        mean_err_m < 0.10,
        "Self-calibration FAIL: MC 1D error {mean_err_m:.4e} >= 0.1 — LCG or Box-Muller broken"
    );
    println!("  Self-calibration PASS ✓");
}

// ═══════════════════════════════════════════════════════════════════════════════
// §K — Experiment constants (pre-registered)
// ═══════════════════════════════════════════════════════════════════════════════

const T_EXP: f64 = 1.0; // integration time
const P_EXP: usize = 1024; // paths per replication
const R_EXP: usize = 64; // replications

// ═══════════════════════════════════════════════════════════════════════════════
// §L — The decisive experiment (pre-registered §3.2)
// ═══════════════════════════════════════════════════════════════════════════════

/// `G_GRIDLESS_RQMC`: path-space RQMC decisive experiment.
///
/// Pre-registered PASS criterion (§3.2, HARD-CODED):
/// - PASS (§50.5 CONFIRMED):   `MSE_MC` / `MSE_RQMC_BB` ≥ 2.0  at BOTH d=4 AND d=8.
/// - FAIL (§50.5 FALSIFIED):   `ratio_BB` < 2.0  at d=4 OR d=8.
///
/// No hard assert on the final verdict — both outcomes are valid scientific results.
/// The self-calibration check and the per-estimator `mse > 0` asserts catch
/// degenerate setups; the ratio verdict is PRINTED HONESTLY whatever it is.
#[test]
#[ignore = "slow path-space RQMC decisive experiment; run with --ignored"]
fn g_gridless_rqmc() {
    println!("\n{}", "═".repeat(72));
    println!("G_GRIDLESS_RQMC — Path-space RQMC decisive experiment (§3, v9.0.0)");
    println!("{}", "═".repeat(72));
    println!();
    println!("Design (anti-gaming §3.1 — equal budget, equal functional, equal truth):");
    println!("  f(x) = ∏_j cos(ξ_j x_j),  ξ_j = 1/(1+0.05j)");
    println!("  Truth = ∏_j exp(-T ξ_j² a_j),  a_j = 0.5(1+0.1j)");
    println!("  P={P_EXP} paths/rep,  R={R_EXP} replications,  T={T_EXP}");
    println!("  RQMC arm: Sobol (Joe-Kuo) + Owen scramble per rep + Brownian-bridge order");
    println!("  MC arm:   LCG pseudorandom, independent seed per rep");
    println!("  Baseline: Sobol + Owen, NO Brownian-bridge (isolates BB contribution)");
    println!("  n*d per dimension: d=4→n=4 (16-dim), d=8→n=2 (16-dim), d=10→n=2 (20-dim)");
    println!();
    println!("Pre-registered PASS criterion (§3.2, HARD-CODED, NOT tuned):");
    println!("  PASS (§50.5 CONFIRMED):  ratio_BB ≥ 2.0 at BOTH d=4 AND d=8");
    println!("  FAIL (§50.5 FALSIFIED):  ratio_BB < 2.0 at d=4 OR d=8");

    // Self-calibration: verify inv_normal and Sobol generation are correct
    self_calibration_check();

    println!("\nMain experiment results:");

    // d=4: n_steps=4 → n*d=16 ≤ 21 ✓
    let (mc4, nobb4, bb4, ratio_bb_d4, ratio_nobb_d4) = run_and_report_d(4, 4, P_EXP, R_EXP, T_EXP);

    // d=8: n_steps=2 → n*d=16 ≤ 21 ✓
    let (mc8, nobb8, bb8, ratio_bb_d8, ratio_nobb_d8) = run_and_report_d(8, 2, P_EXP, R_EXP, T_EXP);

    // d=10: n_steps=2 → n*d=20 ≤ 21 ✓  (informational)
    let (_mc10, _nobb10, _bb10, ratio_bb_d10, ratio_nobb_d10) =
        run_and_report_d(10, 2, P_EXP, R_EXP, T_EXP);

    // ── Summary table ────────────────────────────────────────────────────────
    println!();
    println!("{}", "─".repeat(72));
    println!("RESULTS TABLE");
    println!("{}", "─".repeat(72));
    println!(
        "  {:>3} | {:>10} | {:>10} | {:>10} | {:>10} | {:>10}",
        "d", "MSE_MC", "MSE_Sobol_BB", "MSE_Sobol_noB", "ratio(BB)", "ratio(noB)"
    );
    println!("  {}", "─".repeat(68));
    println!(
        "  {:>3} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>9.3}× | {:>9.3}×",
        4, mc4, bb4, nobb4, ratio_bb_d4, ratio_nobb_d4
    );
    println!(
        "  {:>3} | {:>10.4e} | {:>10.4e} | {:>10.4e} | {:>9.3}× | {:>9.3}×",
        8, mc8, bb8, nobb8, ratio_bb_d8, ratio_nobb_d8
    );

    // ── Pre-registered verdict ───────────────────────────────────────────────
    let pass_d4 = ratio_bb_d4 >= 2.0;
    let pass_d8 = ratio_bb_d8 >= 2.0;
    let both_pass = pass_d4 && pass_d8;

    println!();
    println!("{}", "═".repeat(72));
    println!("PRE-REGISTERED VERDICT (§3.2)");
    println!("{}", "═".repeat(72));
    println!();
    println!(
        "  d=4:  ratio_BB = {ratio_bb_d4:.3}×  {}",
        if pass_d4 {
            "≥ 2.0  → PASS"
        } else {
            "< 2.0  → FAIL"
        }
    );
    println!(
        "  d=8:  ratio_BB = {ratio_bb_d8:.3}×  {}",
        if pass_d8 {
            "≥ 2.0  → PASS"
        } else {
            "< 2.0  → FAIL"
        }
    );
    println!("  d=10: ratio_BB = {ratio_bb_d10:.3}×  (informational only)");
    println!();
    println!("  BB vs no-BB:");
    println!(
        "    d=4:  ratio_noBB={ratio_nobb_d4:.3}×  ratio_BB={ratio_bb_d4:.3}× \
              (BB delta = {:+.3}×)",
        ratio_bb_d4 - ratio_nobb_d4
    );
    println!(
        "    d=8:  ratio_noBB={ratio_nobb_d8:.3}×  ratio_BB={ratio_bb_d8:.3}× \
              (BB delta = {:+.3}×)",
        ratio_bb_d8 - ratio_nobb_d8
    );
    println!("    d=10: ratio_noBB={ratio_nobb_d10:.3}×  ratio_BB={ratio_bb_d10:.3}×");
    println!();

    if both_pass {
        println!("  *** §50.5 CONFIRMED ***");
        println!("  MSE_MC / MSE_RQMC_BB ≥ 2.0 at BOTH d=4 AND d=8.");
        println!("  The path-space RQMC (Sobol+Owen+BB) thesis IS met.");
        println!("  → Engineer builds the RQMC path-space evolver as v9.x/v10 ship item.");
        println!("  → §50.5 thesis CONFIRMED via path-space (not spatial-merge).");
    } else {
        println!("  *** §50.5 FALSIFIED-by-measurement ***");
        let which = if !pass_d4 && !pass_d8 {
            "d=4 AND d=8 (both)"
        } else if !pass_d4 {
            "d=4"
        } else {
            "d=8"
        };
        println!("  ratio_BB < 2.0 at {which} — pre-registered conjunctive FAIL.");
        println!("  Spatial-merge (§50.7), van-der-Corput, AND proper Sobol+Owen+BB");
        println!("  all tested — none meets the ≥ 2× bar.");
        println!();
        println!("  NOTE (Owen bound, Grade A theorem): RQMC is never provably WORSE");
        println!("  than MC — this is falsification-by-measurement, NOT impossibility.");
        println!();
        println!("  → Shift C ships NARROW on H-MEM/determinism axis only (§4).");
        println!("  → d=2 validated envelope (err 1.197e-3, ratio 1.417×) stands.");
        println!("  → Shift B (ReverseChernoff, §51) is the v9.0.0 headline.");
        println!("  → Publishable honest negative: first QMC path-space test of");
        println!("     Chernoff branching; ≥2× bar not met; airtight refutation.");
    }

    println!();
    println!("{}", "═".repeat(72));
    println!("G_GRIDLESS_RQMC complete.");
    println!("{}", "═".repeat(72));
}
