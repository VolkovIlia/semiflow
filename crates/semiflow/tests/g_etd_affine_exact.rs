//! `G_ETD_AFFINE_EXACT` (`RELEASE_BLOCKING`, ADR-0202 D3, math §62.5, closes MODEL gap G1).
//!
//! (1) Affine evolution `M u' = −A u + s`: `phi_combination(h, [u0, M⁻¹s])` equals
//!     `u* + e^{hG}(u0 − u*)`, `u*` from `SpdResolvent(λ = 0)`, `e^{hG}` from the dense
//!     test-local scaling-and-squaring exponential (`tests/phi_dense`), n = 10,
//!     `h ∈ {1e-3, 1, 1e2}`. The last is deep in the stiff regime.
//! (2) The documented ETDRK4 recipe (four `phi_combination` calls, §62.5) on
//!     Allen–Cahn (the `G_ETDRK4_ORDER` setup) agrees with the shipped `Etdrk4`
//!     driver after 4 steps (the two sum the φ-weights along different paths).

// Small index -> f64 conversions on tiny test operators.
#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use semiflow::{
    generator_action::{CsrGenerator, GeneratorAction},
    phi_action::phi_combination,
    scratch::ScratchPool,
    AllenCahn, Etdrk4, SpdSolver, SymmetricOperator,
};

mod phi_dense;
mod spdr_common;
use phi_dense::{dense_expm, sup};
use spdr_common::{dense_of, half_cell_mass, lcg, matvec, op_1d, rel_sup_err};

const N: usize = 10;

/// `n×n` dense `G = −M⁻¹A`, row-major.
fn dense_generator(op: &SymmetricOperator<f64>, mass: &[f64]) -> Vec<f64> {
    let mut g = dense_of(op);
    for i in 0..N {
        for j in 0..N {
            g[i * N + j] = -g[i * N + j] / mass[i];
        }
    }
    g
}

/// `u* + e^{hG}(u0 − u*)`.
fn affine_oracle(g: &[f64], h: f64, u0: &[f64], ustar: &[f64]) -> Vec<f64> {
    let hg: Vec<f64> = g.iter().map(|x| h * x).collect();
    let e = dense_expm(&hg, N);
    let diff: Vec<f64> = u0.iter().zip(ustar).map(|(a, b)| a - b).collect();
    let prop = matvec(&e, &diff);
    ustar.iter().zip(&prop).map(|(s, p)| s + p).collect()
}

fn affine_case(h: f64) -> f64 {
    let c: Vec<f64> = (0..N).map(|i| 0.5 + 0.25 * i as f64).collect();
    let op = op_1d(N).with_diagonal(&c).unwrap();
    let mass = half_cell_mass(N);
    let gen = CsrGenerator::from_symmetric(&op, Some(&mass)).unwrap();
    let (u0, src) = (lcg(5, N), lcg(17, N));
    let res = op.resolvent(0.0, None, SpdSolver::Auto, 1e-12).unwrap();
    let mut ustar = vec![0.0; N];
    res.solve_into(&src, &mut ustar, &mut ScratchPool::new())
        .unwrap();
    let minv_s: Vec<f64> = src.iter().zip(&mass).map(|(s, m)| s / m).collect();
    let mut got = vec![0.0; N];
    phi_combination(&gen, h, &[&u0, &minv_s], &mut got, &mut ScratchPool::new()).unwrap();
    let want = affine_oracle(&dense_generator(&op, &mass), h, &u0, &ustar);
    rel_sup_err(&got, &want)
}

// --- part (2): ETDRK4 recipe vs the shipped driver ---------------------------

const NP: usize = 8;
const EPS: f64 = 0.01;

struct PeriodLaplacian {
    eps_over_dxsq: f64,
}

impl GeneratorAction<f64> for PeriodLaplacian {
    fn dim(&self) -> usize {
        NP
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        for i in 0..NP {
            let im = if i == 0 { NP - 1 } else { i - 1 };
            let ip = if i + 1 == NP { 0 } else { i + 1 };
            dst[i] = self.eps_over_dxsq * (src[im] - 2.0 * src[i] + src[ip]);
        }
    }
    fn norm_bound(&self) -> f64 {
        4.0 * self.eps_over_dxsq
    }
}

fn make_op() -> PeriodLaplacian {
    let dx = 1.0 / NP as f64;
    PeriodLaplacian {
        eps_over_dxsq: EPS / (dx * dx),
    }
}

/// Allen–Cahn `N(u) = u − u³`.
fn nl(u: &[f64]) -> Vec<f64> {
    u.iter().map(|x| x - x * x * x).collect()
}

fn combo(op: &PeriodLaplacian, tau: f64, w: &[&[f64]]) -> Vec<f64> {
    let mut out = vec![0.0; NP];
    phi_combination(op, tau, w, &mut out, &mut ScratchPool::new()).unwrap();
    out
}

/// One ETDRK4 step in the §62.5 recipe form (four `phi_combination` calls).
fn recipe_step(op: &PeriodLaplacian, h: f64, u: &[f64]) -> Vec<f64> {
    let nu = nl(u);
    let a = combo(op, h / 2.0, &[u, &nu]);
    let na = nl(&a);
    let b = combo(op, h / 2.0, &[u, &na]);
    let nb = nl(&b);
    let w_c: Vec<f64> = nb.iter().zip(&nu).map(|(x, y)| 2.0 * x - y).collect();
    let c = combo(op, h / 2.0, &[&a, &w_c]);
    let nc = nl(&c);
    let w2: Vec<f64> = (0..NP)
        .map(|i| (-3.0 * nu[i] + 2.0 * na[i] + 2.0 * nb[i] - nc[i]) / h)
        .collect();
    let w3: Vec<f64> = (0..NP)
        .map(|i| 4.0 * (nu[i] - na[i] - nb[i] + nc[i]) / (h * h))
        .collect();
    combo(op, h, &[u, &nu, &w2, &w3])
}

fn recipe_vs_driver() -> f64 {
    let dx = 1.0 / NP as f64;
    let u0: Vec<f64> = (0..NP)
        .map(|i| 0.5 * (2.0 * std::f64::consts::PI * i as f64 * dx).sin())
        .collect();
    let h = 0.05;
    let mut u = u0.clone();
    for _ in 0..4 {
        u = recipe_step(&make_op(), h, &u);
    }
    let driver = Etdrk4::new(make_op(), AllenCahn::<f64>::new(), h).unwrap();
    let mut want = u0.clone();
    driver
        .integrate(&u0, 4, &mut want, &mut ScratchPool::new())
        .unwrap();
    assert!(sup(&want) > 0.1, "driver state must be non-trivial");
    rel_sup_err(&u, &want)
}

#[test]
fn g_etd_affine_exact() {
    for h in [1e-3, 1.0, 1e2] {
        let err = affine_case(h);
        eprintln!("G_ETD_AFFINE_EXACT (1) h={h:e}: rel_sup_err={err:.3e}");
        assert!(err <= 1e-12, "affine h={h}: rel err {err:.3e} > 1e-12");
    }
    let diff = recipe_vs_driver();
    eprintln!("G_ETD_AFFINE_EXACT (2) recipe vs Etdrk4: rel_sup_diff={diff:.3e}");
    assert!(diff <= 1e-11, "recipe vs driver: {diff:.3e} > 1e-11");
}
