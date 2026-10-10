//! Shared oracle for the ADR-0203 gates: cyclic-Jacobi eigensolver, the
//! cancellation-free Daleckii-Krein form of §63.1.b and the fixtures F1/F2.
//!
//! Scipy-free and dependency-free: everything is plain `f64` arithmetic.

#![allow(dead_code)] // each gate file uses a subset
#![allow(clippy::many_single_char_names, clippy::needless_range_loop)] // textbook Jacobi / index notation
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_possible_wrap
)] // test arithmetic on small indices/counts (< 2^32), never on user data

#[allow(unused_imports)] // also compiled inside the no_std crate's unit tests
use std::{vec, vec::Vec};

/// Unit roundoff `u = 2^-53`.
pub const U: f64 = 1.110_223_024_625_156_5e-16;
/// A-priori quadrature constant of §63.4.
pub const EPS_Q: f64 = 1.1e-14;

/// splitmix64 generator: deterministic fixtures without a dependency.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^ (z >> 31)
    }

    /// Uniform in `[lo, hi)`.
    pub fn uniform(&mut self, lo: f64, hi: f64) -> f64 {
        let unit = (self.next_u64() >> 11) as f64 / (1_u64 << 53) as f64;
        lo + (hi - lo) * unit
    }
}

/// Dense symmetric matrix, row-major `Vec<Vec<f64>>`.
pub type Dense = Vec<Vec<f64>>;

/// Sum of squares of the strict upper triangle.
fn off_diagonal_sq(a: &Dense) -> f64 {
    let n = a.len();
    let mut s = 0.0;
    for p in 0..n {
        for q in p + 1..n {
            s += a[p][q] * a[p][q];
        }
    }
    s
}

/// One Jacobi rotation zeroing `a[p][q]`; updates `a` and accumulates into `v`.
fn rotate(a: &mut Dense, v: &mut Dense, p: usize, q: usize) {
    let theta = (a[q][q] - a[p][p]) / (2.0 * a[p][q]);
    let t = theta.signum() / (theta.abs() + (theta * theta + 1.0).sqrt());
    let c = 1.0 / (t * t + 1.0).sqrt();
    let s = t * c;
    let n = a.len();
    for k in 0..n {
        let (akp, akq) = (a[k][p], a[k][q]);
        a[k][p] = c * akp - s * akq;
        a[k][q] = s * akp + c * akq;
    }
    for k in 0..n {
        let (apk, aqk) = (a[p][k], a[q][k]);
        a[p][k] = c * apk - s * aqk;
        a[q][k] = s * apk + c * aqk;
    }
    for row in v.iter_mut() {
        let (vkp, vkq) = (row[p], row[q]);
        row[p] = c * vkp - s * vkq;
        row[q] = s * vkp + c * vkq;
    }
}

/// Eigendecomposition `L = Σ λ_i φ_i φ_iᵀ`; `vec[i]` is the eigenvector of `lam[i]`.
pub struct Eig {
    pub lam: Vec<f64>,
    pub vec: Vec<Vec<f64>>,
}

/// Cyclic Jacobi: sweep until the off-diagonal mass is `<= 1e-300` or 100 sweeps.
pub fn jacobi_eigh(mat: &Dense) -> Eig {
    let n = mat.len();
    let mut a = mat.clone();
    let mut v: Dense = (0..n)
        .map(|i| (0..n).map(|j| f64::from(u8::from(i == j))).collect())
        .collect();
    for _ in 0..100 {
        if off_diagonal_sq(&a) <= 1e-300 {
            break;
        }
        for p in 0..n {
            for q in p + 1..n {
                if a[p][q] != 0.0 {
                    rotate(&mut a, &mut v, p, q);
                }
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&x, &y| a[x][x].total_cmp(&a[y][y]));
    Eig {
        lam: order.iter().map(|&i| a[i][i]).collect(),
        vec: order
            .iter()
            .map(|&i| (0..n).map(|k| v[k][i]).collect())
            .collect(),
    }
}

/// Cancellation-free divided difference of `-t e^{-t x}` (§63.1.b).
pub fn f1(t: f64, x: f64, y: f64) -> f64 {
    let (lo, hi) = (x.min(y), x.max(y));
    let theta = t * (hi - lo);
    let phi = if theta > 0.0 {
        -(-theta).exp_m1() / theta
    } else {
        1.0
    };
    -t * (-t * lo).exp() * phi
}

/// A symmetric direction `M_k = ∂L/∂θ_k`.
#[derive(Clone, Copy, Debug)]
pub enum Stencil {
    /// Edge conductance: `(e_i - e_j)(e_i - e_j)ᵀ`, `‖M‖₂ = 2`.
    Edge(usize, usize),
    /// Symmetric entry pair `e_i e_jᵀ + e_j e_iᵀ` (`i ≠ j`) or `e_i e_iᵀ`, `‖M‖₂ = 1`.
    Entry(usize, usize),
}

impl Stencil {
    /// `φ_aᵀ M φ_b`.
    pub fn form(&self, pa: &[f64], pb: &[f64]) -> f64 {
        match *self {
            Self::Edge(i, j) => (pa[i] - pa[j]) * (pb[i] - pb[j]),
            Self::Entry(i, j) if i == j => pa[i] * pb[i],
            Self::Entry(i, j) => pa[i] * pb[j] + pa[j] * pb[i],
        }
    }

    /// `‖M‖₂`.
    pub fn norm2(&self) -> f64 {
        match *self {
            Self::Edge(..) => 2.0,
            Self::Entry(..) => 1.0,
        }
    }

    /// `(i, j)` pair, for building the library sensitivity.
    pub fn pair(&self) -> (usize, usize) {
        match *self {
            Self::Edge(i, j) | Self::Entry(i, j) => (i, j),
        }
    }
}

fn dot(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

pub fn norm2(a: &[f64]) -> f64 {
    dot(a, a).sqrt()
}

/// Oracle gradient `g_k` and absolute magnitude `G_k` (§63.1.b–c) for every stencil.
pub fn oracle(eig: &Eig, d: &[f64], v: &[f64], t: f64, st: &[Stencil]) -> Vec<(f64, f64)> {
    let n = eig.lam.len();
    let alpha: Vec<f64> = eig.vec.iter().map(|p| dot(p, d)).collect();
    let beta: Vec<f64> = eig.vec.iter().map(|p| dot(p, v)).collect();
    st.iter()
        .map(|s| {
            let (mut g, mut g_abs) = (0.0, 0.0);
            for i in 0..n {
                for j in 0..n {
                    let term = alpha[i]
                        * beta[j]
                        * s.form(&eig.vec[i], &eig.vec[j])
                        * f1(t, eig.lam[i], eig.lam[j]);
                    g += term;
                    g_abs += term.abs();
                }
            }
            (g, g_abs)
        })
        .collect()
}

/// `e^{-τL} x` with exact eigen propagators.
pub fn propagate(eig: &Eig, tau: f64, x: &[f64]) -> Vec<f64> {
    let n = x.len();
    let mut out = vec![0.0; n];
    for (lam, phi) in eig.lam.iter().zip(&eig.vec) {
        let c = (-tau * lam).exp() * dot(phi, x);
        for (o, p) in out.iter_mut().zip(phi) {
            *o += c * p;
        }
    }
    out
}

/// Pre-ADR-0203 rule (one GL8 panel on `[0, t]`) with EXACT propagators.
pub fn legacy_gl8(eig: &Eig, d: &[f64], v: &[f64], t: f64, st: &[Stencil]) -> Vec<f64> {
    const GL8: [(f64, f64); 8] = [
        (0.019_855_071_751_231_88, 0.050_614_268_145_188_29),
        (0.101_666_761_293_186_65, 0.111_190_517_226_687_24),
        (0.237_233_795_041_835_5, 0.156_853_322_938_943_64),
        (0.408_282_678_752_175_1, 0.181_341_891_689_180_6),
        (0.591_717_321_247_825, 0.181_341_891_689_180_6),
        (0.762_766_204_958_164_5, 0.156_853_322_938_943_64),
        (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
        (0.980_144_928_248_768_1, 0.050_614_268_145_188_29),
    ];
    let mut g = vec![0.0; st.len()];
    for &(s, w) in &GL8 {
        let a = propagate(eig, (1.0 - s) * t, d);
        let b = propagate(eig, s * t, v);
        for (gk, m) in g.iter_mut().zip(st) {
            let (i, j) = m.pair();
            let bil = match *m {
                Stencil::Edge(..) => (a[i] - a[j]) * (b[i] - b[j]),
                Stencil::Entry(..) if i == j => a[i] * b[i],
                Stencil::Entry(..) => a[i] * b[j] + a[j] * b[i],
            };
            *gk -= t * w * bil;
        }
    }
    g
}

fn add_edge(mat: &mut Dense, i: usize, j: usize, w: f64) {
    mat[i][i] += w;
    mat[j][j] += w;
    mat[i][j] -= w;
    mat[j][i] -= w;
}

/// Fixture F1: random weighted graph (`U[0.2, 2]`, density ~35%) containing a path.
pub fn random_graph(n: usize, seed: u64) -> (Dense, Vec<(usize, usize, f64)>) {
    let mut rng = Rng::new(seed);
    let mut mat = vec![vec![0.0; n]; n];
    let mut edges = Vec::new();
    for i in 0..n {
        for j in i + 1..n {
            if rng.uniform(0.0, 1.0) < 0.35 || j == i + 1 {
                let w = rng.uniform(0.2, 2.0);
                add_edge(&mut mat, i, j, w);
                edges.push((i, j, w));
            }
        }
    }
    (mat, edges)
}

/// Fixture F2: two fast rings, one slow bridge, a Robin leak (SPD).
pub fn stiff_contrast(n: usize, fast: f64, seed: u64) -> (Dense, Vec<(usize, usize)>) {
    let mut rng = Rng::new(seed);
    let mut mat = vec![vec![0.0; n]; n];
    let mut edges = Vec::new();
    let h = n / 2;
    for ring in [(0..h).collect::<Vec<_>>(), (h..n).collect::<Vec<_>>()] {
        let mut pairs: Vec<(usize, usize)> = ring.windows(2).map(|w| (w[0], w[1])).collect();
        pairs.push((ring[0], ring[ring.len() - 1]));
        for (i, j) in pairs {
            add_edge(&mut mat, i, j, fast * rng.uniform(0.5, 1.5));
            edges.push((i.min(j), i.max(j)));
        }
    }
    add_edge(&mut mat, h - 1, h, 1.0);
    edges.push((h - 1, h));
    mat[0][0] += 0.3;
    (mat, edges)
}

/// CSR triple `(row_ptr, col_idx, vals)` of a dense matrix (nonzeros only).
pub fn csr_of(mat: &Dense) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let (mut row_ptr, mut cols, mut vals) = (vec![0_usize], Vec::new(), Vec::new());
    for row in mat {
        for (j, &x) in row.iter().enumerate() {
            if x != 0.0 {
                cols.push(j as u32);
                vals.push(x);
            }
        }
        row_ptr.push(cols.len());
    }
    (row_ptr, cols, vals)
}

/// Maximum stored entries in a CSR row (diagonal included).
pub fn max_row_nnz(row_ptr: &[usize]) -> usize {
    row_ptr.windows(2).map(|w| w[1] - w[0]).max().unwrap_or(0)
}

/// Deterministic test vectors `u0`, `dj` in `[-1, 1]`.
pub fn test_vectors(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>) {
    let mut rng = Rng::new(seed);
    let u0 = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    let dj = (0..n).map(|_| rng.uniform(-1.0, 1.0)).collect();
    (u0, dj)
}

/// Inputs of the §63.7.a bound.
pub struct BoundInputs {
    pub n: usize,
    pub row_nnz: usize,
    pub tol: f64,
    pub n_chain: u64,
    /// `FrechetPlan::chain_weight` (§63.7.a, Amendment 4).
    pub chain_weight: u64,
    pub rho_t: f64,
    pub t: f64,
}

impl BoundInputs {
    /// `η` of §63.7.a (Amendment 4: rounding weight `W` replaces `N_chain·m_max²`).
    pub fn eta(&self) -> f64 {
        let r = self.row_nnz as f64;
        2.0 * (self.n_chain as f64 * self.tol + (r + 3.0) * self.chain_weight as f64 * U)
            + self.tol
            + (r + self.n as f64) * U * self.rho_t
    }

    /// `τ_k` of §63.7.a for one parameter.
    pub fn tau(&self, big_g: f64, norm_k: f64, norm_d: f64, norm_v: f64) -> f64 {
        let nn = self.n as f64;
        (EPS_Q + 2.0 * nn * nn * U) * big_g + self.eta() * self.t * norm_k * norm_d * norm_v
    }

    /// Propagator-free part `τ_k^Q` used against the legacy rule (non-vacuity b).
    pub fn tau_quad(&self, big_g: f64, norm_k: f64, norm_d: f64, norm_v: f64) -> f64 {
        let nn = self.n as f64;
        let r = self.row_nnz as f64;
        (EPS_Q + 2.0 * nn * nn * U) * big_g
            + (r + nn) * U * self.rho_t * self.t * norm_k * norm_d * norm_v
    }
}

/// `a.max(b)` that propagates NaN (`f64::max` silently drops it).
pub fn nan_max(a: f64, b: f64) -> f64 {
    if a.is_nan() || b.is_nan() {
        f64::NAN
    } else {
        a.max(b)
    }
}

/// `true` unless `err <= tau`; NaN in either argument counts as a violation.
#[allow(clippy::neg_cmp_op_on_partial_ord)] // the negation IS the point: NaN must fail
pub fn violates(err: f64, tau: f64) -> bool {
    !(err <= tau)
}

// ---------------------------------------------------------------------------
// Fixture F3 (ADR-0203 Amendment 1): four triangle clusters + three bridges.
// ---------------------------------------------------------------------------

/// Number of nodes of F3.
pub const F3_N: usize = 12;
/// Index range of the three bridge edges in [`f3_matrix`]'s edge list.
pub const F3_BRIDGES: std::ops::Range<usize> = 12..15;

/// F3 combinatorial Laplacian and edge list `(i, j, weight)` for stiffness `w`.
///
/// Intra-cluster edges first (`e = 0, 1, 2` = `(a,b), (b,d), (a,d)` of cluster
/// `c`, weight `w·(0.5 + ((7·(3c+e)) mod 11)/10)`), then the bridges
/// `(3c+2, 3c+3)` of weights `{0.6, 1.0, 1.4}`. Deterministic, no RNG.
pub fn f3_matrix(w: f64) -> (Dense, Vec<(usize, usize, f64)>) {
    let mut mat = vec![vec![0.0; F3_N]; F3_N];
    let mut edges = Vec::new();
    for c in 0..4_usize {
        let (a, b, d) = (3 * c, 3 * c + 1, 3 * c + 2);
        for (e, (i, j)) in [(a, b), (b, d), (a, d)].into_iter().enumerate() {
            let weight = w * (0.5 + ((7 * (3 * c + e)) % 11) as f64 / 10.0);
            add_edge(&mut mat, i, j, weight);
            edges.push((i, j, weight));
        }
    }
    for (c, weight) in [0.6, 1.0, 1.4].into_iter().enumerate() {
        add_edge(&mut mat, 3 * c + 2, 3 * c + 3, weight);
        edges.push((3 * c + 2, 3 * c + 3, weight));
    }
    (mat, edges)
}

/// F3 signals `(u0, dj)`, concentrated on the slow inter-cluster modes.
pub fn f3_signals() -> (Vec<f64>, Vec<f64>) {
    let cluster = |i: usize| (i / 3) as f64;
    let u0 = (0..F3_N)
        .map(|i| cluster(i) - 1.5 + 0.1 * (1.7 * i as f64 + 0.3).sin())
        .collect();
    let dj = (0..F3_N)
        .map(|i| (-1.0_f64).powi((i / 3) as i32) + 0.1 * (2.3 * i as f64).cos())
        .collect();
    (u0, dj)
}

/// `λ_max/λ_2` of F3 at stiffness `w` (`λ_1 = 0` is the constant mode).
fn f3_ratio(w: f64) -> f64 {
    let eig = jacobi_eigh(&f3_matrix(w).0);
    eig.lam[F3_N - 1] / eig.lam[1]
}

/// Per-point F3 stiffness `W` and time `t` for a target `λ_max·t` (Amendment 1).
///
/// If `target >= R1 = λ_max/λ_2 (W = 1)`: bisection on `log W` until
/// `λ_max/λ_2 = target` to `1e-3` (asserted), `t = 1/λ_2`. Otherwise `W = 1`,
/// `t = target/λ_max`.
pub fn f3_point(target: f64) -> (f64, f64) {
    let eig = jacobi_eigh(&f3_matrix(1.0).0);
    let (lam2, lam_max) = (eig.lam[1], eig.lam[F3_N - 1]);
    if target < lam_max / lam2 {
        return (1.0, target / lam_max);
    }
    let (mut lo, mut hi) = (0.0_f64, 30.0_f64);
    for _ in 0..200 {
        let mid = 0.5 * (lo + hi);
        if f3_ratio(mid.exp()) < target {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    let w = (0.5 * (lo + hi)).exp();
    let ratio = f3_ratio(w);
    assert!(
        (ratio / target - 1.0).abs() <= 1e-3,
        "F3 bisection: ratio {ratio:e} vs target {target:e}"
    );
    let lam2 = jacobi_eigh(&f3_matrix(w).0).lam[1];
    (w, 1.0 / lam2)
}
