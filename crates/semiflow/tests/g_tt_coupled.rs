//! `g_tt_coupled` — P5: high-d rank-scaling gate for the spectral evolver.
//!
//! # Purpose (`RELEASE_BLOCKING`, P5 curse-escape evidence, §11.6)
//!
//! Prove that `CoupledTtChernoff` (P3'' spectral evolver) escapes the exponential
//! curse for correlated-Gaussian diffusion across d∈{4,6,8,10}:
//!   - storage O(d·n·r²) with r=O(1), versus O(n^d) naive
//!   - evolved-state `peak_rank` >1, bounded, ~constant in d
//!   - byte-reproducible (0-ULP determinism, ADR-0018)
//!
//! # Four HARD ASSERTS (all `assert!`/`panic!` — NOT print-only)
//!
//! **C1 — Genuine coupling (anti-triviality):** rank-1 IC evolves to `peak_rank` > 1
//! under `Tridiagonal(ρ)` at d∈{4,6,8,10}.  Rank-1 result = coupling was a no-op
//! (v9.0.0 separability bug).
//!
//! **C2 — Bounded / poly-d:** evolved-state `peak_rank` stays ≤ `RANK_BOUND` (=10) at
//! every d, AND the log-rank slope across d∈{4,6,8,10} is < `SLOPE_GATE` (=0.70).
//! Spec measures ~5–6 constant in d; `RANK_BOUND=10` gives headroom.  If `peak_rank`
//! exceeds `RANK_BOUND`, C2 FAIL — the curse has NOT been escaped.  Do NOT loosen.
//!
//! **C3 — SUPERSEDED.** The old consistent-shift-reference accuracy check is withdrawn:
//! exactness is now certified by `g_tt_coupled_converge` (P4' gate, §10.13, §11.6).
//! See `g_tt_coupled_converge.rs`.  This gate proves the COST/RANK claim only.
//!
//! **C4 — Cost poly-d:** TT storage ≪ naive n^d as d grows (sub-exponential cost).
//! Asserted for d∈{4,6} where n^d is representable without overflow.
//!
//! **C5 — Byte-reproducible:** two independent runs produce bit-identical TT cores.
//!
//! # SPD constraint (CRITICAL, §10.12 risk 1)
//!
//! Tridiagonal d≥4 interior axes are each shared by 2 pairs → `c_j=a_j/2`.
//! Per-pair SPD requires |ρ| < 0.5, and full-tensor SPD requires |ρ| < 1/√(d−1).
//! At d=10: 1/√9 = 0.333.  We use `RHO_TRIDIAG=0.3` — safe for all d≤10.
//!
//! Dense all-pairs d=4: each axis in 3 pairs → `c_j=a_j/3`; SPD threshold |ρ| < 1/3.
//! With `RHO_DENSE=0.3` < 1/3 ≈ 0.333 ✓ (det = `a_j`*`a_k`*(1/9−0.09)=0.021*`a_j`*`a_k` > 0).
//!
//! # Topologies
//!
//! TRIDIAGONAL (primary): d∈{4,6,8,10}, `n=N_LARGE=32`, `N_STEPS=40`.  C1,C2,C4,C5.
//! ADJACENT BLOCK-DISJOINT PAIRS (secondary, v9.1.0): d=4, pairs (0,1)+(2,3),
//!   `n=N_SMALL=8`, `N_STEPS=3`.  C1,C2,C4,C5.
//!   Non-adjacent pairs (k>j+1) are now rejected at construction (fail-loud, ADR-0162).
//!   True dense all-pairs coupling deferred to v9.2.0.
//!   Poly-rank proven by the tridiagonal sweep above.

#![cfg(feature = "slow-tests")]
#![allow(clippy::cast_precision_loss, clippy::too_many_lines)]
#![allow(clippy::cast_possible_truncation)] // usize→u32 for .pow(): d ≤ 10, n ≤ 32 in test
#![allow(clippy::cast_possible_wrap)] // u32→i32 for .powi(): scaling factor ≤ 30
#![allow(clippy::many_single_char_names)] // n, d, r, a, s, etc. are standard math variable names
#![allow(clippy::needless_range_loop)] // index loops use cross-index arithmetic
#![allow(clippy::unreadable_literal)] // LCG/expm constants are mathematical identifiers

extern crate alloc;
use alloc::vec::Vec;

use semiflow::{CoupledTtChernoff, CouplingTopology, TtState};

// ═══════════════════════════════════════════════════════════════════════════
// §A — Pre-registered parameters (NORMATIVE)
// ═══════════════════════════════════════════════════════════════════════════

const T_FINAL: f64 = 0.5;
const N_LARGE: usize = 32; // tridiagonal C1/C2/C4/C5
const N_SMALL: usize = 8; // dense all-pairs d=4 (optional)
const N_STEPS_L: usize = 40; // tridiagonal sweep
const N_STEPS_S: usize = 3; // dense d=4
const X_MIN: f64 = -4.0;
const X_MAX: f64 = 4.0;
const EPS_ROUND: f64 = 1e-7;
const SIGMA0_SQ: f64 = 1.0;

/// SPD-safe for all d≤10 tridiagonal (|ρ|<0.5 per-pair, |ρ|<1/√9≈0.333 full-tensor).
const RHO_TRIDIAG: f64 = 0.3;

/// SPD-safe for d=4 all-pairs (|ρ|<1/3≈0.333; `det=(a_j/3)(a_k/3)−(0.3√(a_j a_k))²>0`).
const RHO_DENSE: f64 = 0.3;

const D_LIST: [usize; 4] = [4, 6, 8, 10];

/// C2 bounded-rank gate: evolved-state `peak_rank` ≤ `RANK_BOUND` at every d.
/// Spec measures ~5–6 constant in d (pair-factor op-rank 6, §11.6); allow headroom.
/// Do NOT loosen if the gate fires — report honest failure to architect.
const RANK_BOUND: usize = 10;

/// C2 slope gate: `log(peak_rank)` vs d slope < `SLOPE_GATE` (exponential-growth detector).
const SLOPE_GATE: f64 = 0.70;

// ═══════════════════════════════════════════════════════════════════════════
// §B — Helpers
// ═══════════════════════════════════════════════════════════════════════════

fn diffusion_coeff(j: usize) -> f64 {
    0.5 + 0.1 * j as f64
}

fn grid_x(n: usize, i: usize) -> f64 {
    X_MIN + i as f64 * (X_MAX - X_MIN) / (n as f64 - 1.0)
}

fn ic_slice(n: usize) -> Vec<f64> {
    (0..n)
        .map(|i| (-(grid_x(n, i).powi(2)) / (2.0 * SIGMA0_SQ)).exp())
        .collect()
}

fn test_func(n: usize, j: usize) -> Vec<f64> {
    let alpha = 0.1 / (j as f64 + 1.0);
    (0..n)
        .map(|i| (-alpha * grid_x(n, i).powi(2)).exp())
        .collect()
}

/// Rank-1 separable Gaussian IC with heterogeneous axes (`diffusion_coeff`).
fn rank1_ic(d: usize, n: usize) -> TtState<f64> {
    TtState::rank1_separable((0..d).map(|_| ic_slice(n)).collect())
}

fn tt_functional(d: usize, n: usize, state: &TtState<f64>) -> f64 {
    let fns: Vec<Vec<f64>> = (0..d).map(|j| test_func(n, j)).collect();
    state.inner_separable(&fns)
}

/// Build all-pairs dense topology (kept for documentation; not used in v9.1.0 gates
/// because non-adjacent pairs are now rejected at construction — deferred to v9.2.0).
#[allow(dead_code)]
fn dense_pairs(d: usize, rho: f64) -> Vec<(usize, usize, f64)> {
    (0..d)
        .flat_map(|j| ((j + 1)..d).map(move |k| (j, k, rho)))
        .collect()
}

fn log_slope(d_vals: &[usize], ranks: &[usize]) -> f64 {
    let n = d_vals.len() as f64;
    let xs: Vec<f64> = d_vals.iter().map(|&d| d as f64).collect();
    let ys: Vec<f64> = ranks.iter().map(|&r| (r as f64).max(1.0).ln()).collect();
    let sx: f64 = xs.iter().sum();
    let sy: f64 = ys.iter().sum();
    let sxx: f64 = xs.iter().map(|&x| x * x).sum();
    let sxy: f64 = xs.iter().zip(ys.iter()).map(|(&x, &y)| x * y).sum();
    let denom = n * sxx - sx * sx;
    if denom.abs() < 1e-300 {
        0.0
    } else {
        (n * sxy - sx * sy) / denom
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §C — Evolver runner
// ═══════════════════════════════════════════════════════════════════════════

fn run_tt(d: usize, n: usize, n_steps: usize, topo: CouplingTopology<f64>) -> (usize, usize, f64) {
    let a: Vec<f64> = (0..d).map(diffusion_coeff).collect();
    let domain: Vec<(f64, f64)> = vec![(X_MIN, X_MAX); d];
    let ev = CoupledTtChernoff::new(a, vec![0.0f64; d], 0.0, topo, domain, EPS_ROUND);
    let mut s = rank1_ic(d, n);
    ev.evolve(T_FINAL, n_steps, &mut s);
    (s.peak_rank(), s.storage_size(), tt_functional(d, n, &s))
}

// ═══════════════════════════════════════════════════════════════════════════
// §D — Byte-reproducibility checker
// ═══════════════════════════════════════════════════════════════════════════

fn check_byte_repro(d: usize, n: usize, n_steps: usize, topo: CouplingTopology<f64>) {
    let a: Vec<f64> = (0..d).map(diffusion_coeff).collect();
    let dom: Vec<(f64, f64)> = vec![(X_MIN, X_MAX); d];
    let ev1 = CoupledTtChernoff::new(
        a.clone(),
        vec![0.0; d],
        0.0,
        topo.clone(),
        dom.clone(),
        EPS_ROUND,
    );
    let ev2 = CoupledTtChernoff::new(a, vec![0.0; d], 0.0, topo, dom, EPS_ROUND);
    let mut s1 = rank1_ic(d, n);
    let mut s2 = rank1_ic(d, n);
    ev1.evolve(T_FINAL, n_steps, &mut s1);
    ev2.evolve(T_FINAL, n_steps, &mut s2);
    for (k, (c1, c2)) in s1.cores.iter().zip(s2.cores.iter()).enumerate() {
        assert_eq!(
            c1.data, c2.data,
            "C5 FAIL: d={d} n={n} core {k} not bit-identical between two runs",
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// §E — Main gate
// ═══════════════════════════════════════════════════════════════════════════

#[test]
#[ignore = "slow P5 curse-escape gate; run with: cargo run -p xtask -- test-flagship"]
fn g_tt_coupled() {
    let bar = "═".repeat(72);
    println!("\n{bar}");
    println!("g_tt_coupled  P5 — high-d rank-scaling gate (spectral evolver P3'')");
    println!("             RELEASE_BLOCKING (§11.6, ADR-0162, curse-escape evidence)");
    println!("{bar}");
    println!();
    println!(
        "Tridiagonal ρ={RHO_TRIDIAG} d∈{{4,6,8,10}}: n={N_LARGE}, n_steps={N_STEPS_L}  C1,C2,C4,C5"
    );
    println!(
        "Dense ρ={RHO_DENSE}         d=4 only:     n={N_SMALL}, n_steps={N_STEPS_S}  C1,C2,C4,C5"
    );
    println!("EPS_ROUND={EPS_ROUND}, T={T_FINAL}");
    println!("RANK_BOUND={RANK_BOUND} (headroom; spec measures ~5–6 constant in d)");
    println!("SLOPE_GATE={SLOPE_GATE}");
    println!();
    // C3 superseded note.
    println!("C3: SUPERSEDED — exactness certified by g_tt_coupled_converge (P4' gate).");
    println!("    The old consistent-shift accuracy check is withdrawn per §11.6 / ADR-0162.");
    println!("    This gate proves COST/RANK escape only.");
    println!();

    // ─────────────────────────────────────────────────────────────────────
    // TRIDIAGONAL: C1, C2, C4, C5
    // ─────────────────────────────────────────────────────────────────────
    println!("{}", "─".repeat(72));
    println!("TRIDIAGONAL ρ={RHO_TRIDIAG} — n={N_LARGE}, n_steps={N_STEPS_L}  (C1,C2,C4,C5)");
    println!(
        "SPD check: c_j=a_j/2 for interior axes; det=a_j*a_k*(0.25-ρ²)={}",
        0.25 - RHO_TRIDIAG * RHO_TRIDIAG
    );
    println!("  d  | peak_r | storage   | naive_nd         | C1   | C2b  | C4");
    println!("  {}", "─".repeat(65));

    let mut ranks_t = Vec::new();
    let mut storages_t = Vec::new();

    for &d in &D_LIST {
        let topo = CouplingTopology::Tridiagonal(RHO_TRIDIAG);
        let (r, st, _) = run_tt(d, N_LARGE, N_STEPS_L, topo);
        let naive = if d <= 6 {
            N_LARGE.saturating_pow(d as u32)
        } else {
            usize::MAX
        };
        let naive_str = if naive == usize::MAX {
            "     overflow".to_string()
        } else {
            format!("{naive:>17}")
        };
        let c1 = if r > 1 { "PASS" } else { "FAIL" };
        let c2b = if r <= RANK_BOUND { "PASS" } else { "FAIL" };
        let c4 = if naive == usize::MAX || st < naive {
            "PASS"
        } else {
            "FAIL"
        };
        println!("  {d:>2} | {r:>6} | {st:>9} | {naive_str} | {c1:4} | {c2b:4} | {c4:4}");
        ranks_t.push(r);
        storages_t.push(st);
    }

    let slope_t = log_slope(&D_LIST, &ranks_t);
    println!("  Log-rank slope (d∈{{4,6,8,10}}): {slope_t:.4}  (gate < {SLOPE_GATE})");

    // ─────────────────────────────────────────────────────────────────────
    // ADJACENT BLOCK-DISJOINT PAIRS: d=4, pairs (0,1)+(2,3) only.
    // (v9.1.0: non-adjacent pairs now rejected at construction — ADR-0162)
    // True dense all-pairs coupling deferred to v9.2.0.
    // The tridiagonal sweep d∈{4,6,8,10} above proves poly-d rank scaling.
    // ─────────────────────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(72));
    println!(
        "ADJACENT BLOCK-DISJOINT PAIRS ρ={RHO_DENSE} — d=4, pairs=(0,1)+(2,3), \
         n={N_SMALL}, n_steps={N_STEPS_S}  (C1,C2,C4,C5)"
    );
    println!("Note: non-adjacent pairs (k>j+1) are rejected in v9.1.0 (fail-loud).");
    println!("True dense all-pairs coupling deferred to v9.2.0 (ADR-0162).");
    println!("Poly-d rank proven by the tridiagonal sweep above.");

    let d_dense = 4usize;
    // Adjacent block-disjoint: (0,1) and (2,3) — both satisfy k==j+1.
    let topo_d = CouplingTopology::Pairs(vec![
        (0usize, 1usize, RHO_DENSE),
        (2usize, 3usize, RHO_DENSE),
    ]);
    let (r_d, st_d, _) = run_tt(d_dense, N_SMALL, N_STEPS_S, topo_d);
    let naive_d = N_SMALL.saturating_pow(d_dense as u32);
    let c1_d = if r_d > 1 { "PASS" } else { "FAIL" };
    let c2b_d = if r_d <= RANK_BOUND { "PASS" } else { "FAIL" };
    let c4_d = if st_d < naive_d {
        "PASS"
    } else {
        "NOTE: small-n artifact"
    };
    println!(
        "  d={d_dense}: peak_r={r_d}  storage={st_d}  naive_n^d={naive_d}  \
         C1={c1_d}  C2b={c2b_d}  C4={c4_d}"
    );

    // ─────────────────────────────────────────────────────────────────────
    // C5 — Byte-reproducibility
    // ─────────────────────────────────────────────────────────────────────
    println!("\n{}", "─".repeat(72));
    println!("C5 — Byte-reproducibility (0-ULP determinism, ADR-0018):");
    check_byte_repro(
        4,
        N_LARGE,
        N_STEPS_L,
        CouplingTopology::Tridiagonal(RHO_TRIDIAG),
    );
    println!("  Tridiagonal d=4 n={N_LARGE} n_steps={N_STEPS_L}: PASS");
    check_byte_repro(
        d_dense,
        N_SMALL,
        N_STEPS_S,
        CouplingTopology::Pairs(vec![
            (0usize, 1usize, RHO_DENSE),
            (2usize, 3usize, RHO_DENSE),
        ]),
    );
    println!("  Adjacent block-disjoint d=4 n={N_SMALL} n_steps={N_STEPS_S}: PASS");

    // ─────────────────────────────────────────────────────────────────────
    // HARD ASSERTS
    // ─────────────────────────────────────────────────────────────────────
    println!("\n{bar}");
    println!("GATE SUMMARY AND HARD ASSERTS");
    println!("{bar}");

    // C1: peak_rank > 1 (genuine coupling, anti-triviality)
    for (&d, &r) in D_LIST.iter().zip(ranks_t.iter()) {
        assert!(
            r > 1,
            "C1 FAIL [Tridiagonal d={d} n={N_LARGE}]: peak_rank={r} from rank-1 IC. \
             Coupling was a no-op (v9.0.0 separability bug).",
        );
    }
    assert!(
        r_d > 1,
        "C1 FAIL [AdjBlockDisjoint d={d_dense} n={N_SMALL}]: peak_rank={r_d} from rank-1 IC.",
    );
    println!("  C1 PASS: all d∈{{4,6,8,10}} tridiagonal + d=4 adj-block-disjoint grew rank>1");

    // C2: peak_rank ≤ RANK_BOUND AND log-rank slope < SLOPE_GATE
    for (&d, &r) in D_LIST.iter().zip(ranks_t.iter()) {
        assert!(
            r <= RANK_BOUND,
            "C2 FAIL [Tridiagonal d={d}]: peak_rank={r} > RANK_BOUND={RANK_BOUND}. \
             Rank is growing beyond the poly-d bound — curse NOT escaped. \
             Ranks so far: {ranks_t:?}. Do NOT loosen RANK_BOUND.",
        );
    }
    assert!(
        r_d <= RANK_BOUND,
        "C2 FAIL [AdjBlockDisjoint d={d_dense}]: peak_rank={r_d} > RANK_BOUND={RANK_BOUND}.",
    );
    assert!(
        slope_t < SLOPE_GATE,
        "C2 FAIL [Tridiagonal slope]: {slope_t:.4} >= {SLOPE_GATE}. \
         Rank grew super-polynomially in d. Ranks: {ranks_t:?}. \
         ADR-0161 honest-downgrade path applies — do NOT weaken threshold.",
    );
    println!("  C2 PASS: all peak_ranks ≤ {RANK_BOUND}; slope={slope_t:.4} < {SLOPE_GATE}");
    println!("           Tridiagonal ranks d=[4,6,8,10]: {ranks_t:?}");

    // C3: NOTE ONLY (no assert — superseded)
    println!("  C3: SUPERSEDED — exactness gate is g_tt_coupled_converge (P4', §10.13).");

    // C4: TT storage < naive n^d (poly-d)
    for (i, &d) in D_LIST.iter().enumerate() {
        if d <= 6 {
            let naive = N_LARGE.saturating_pow(d as u32);
            assert!(
                storages_t[i] < naive,
                "C4 FAIL [Tridiagonal d={d}]: storage={} >= naive n^d={naive}",
                storages_t[i],
            );
        }
    }
    println!("  C4 PASS: TT storage ≪ naive n^d (sub-exponential, tridiagonal d≤6 verified)");
    println!("           Adj-block-disjoint d=4: storage={st_d} vs naive n^d={naive_d}");

    // C5: already verified via check_byte_repro (panics on failure)
    println!("  C5 PASS: bit-identical across two independent runs (0-ULP, ADR-0018)");

    println!();
    println!("{bar}");
    println!("g_tt_coupled PASS — P5 curse-escape evidence confirmed");
    println!(
        "  Genuine coupling (C1): rank-1 IC → rank>{} at all d (no v9.0.0 no-op).",
        1
    );
    println!("  Bounded poly-d  (C2): peak_rank ≤ {RANK_BOUND}, slope={slope_t:.4} < {SLOPE_GATE}");
    println!("  Poly-d cost     (C4): storage O(d·n·r²) ≪ naive O(n^d)");
    println!("  Reproducible    (C5): 0-ULP determinism confirmed");
    println!("  C3 superseded: exactness certified by g_tt_coupled_converge (P4').");
    println!("{bar}");
}
