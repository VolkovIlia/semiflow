//! `G_ETDRK4_ORDER` (`RELEASE_BLOCKING`): ETDRK4 achieves order 4 on the
//! Allen–Cahn equation `∂ₜu = ε u_xx + N(u)`, `N(u) = u − u³`, periodic 1D.
//!
//! Method: self-convergence with step sizes `h`, `h/2` vs reference `h/4`.
//! Gate: empirical slope `∈ [3.7, 4.3]`.
//!
//! Grid: N=8 points, ε=0.01. Step h=0.05, T=0.2.
//! `τ·‖L‖ = 0.05 · 2.56 = 0.128` (safe for augmented Horner series).

use semiflow::{generator_action::GeneratorAction, scratch::ScratchPool, AllenCahn, Etdrk4};

const N: usize = 8;
const EPS: f64 = 0.01;

struct PeriodLaplacian {
    eps_over_dxsq: f64,
}

impl GeneratorAction<f64> for PeriodLaplacian {
    fn dim(&self) -> usize { N }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        let c = self.eps_over_dxsq;
        for i in 0..N {
            let im = if i == 0 { N - 1 } else { i - 1 };
            let ip = if i + 1 == N { 0 } else { i + 1 };
            dst[i] = c * (src[im] - 2.0 * src[i] + src[ip]);
        }
    }
    fn norm_bound(&self) -> f64 { 4.0 * self.eps_over_dxsq }
}

#[allow(clippy::cast_precision_loss)]
fn make_op() -> PeriodLaplacian {
    let dx = 1.0 / N as f64;
    PeriodLaplacian { eps_over_dxsq: EPS / (dx * dx) }
}

fn integrate_ac(h: f64, n_steps: usize, u0: &[f64]) -> Vec<f64> {
    let driver = Etdrk4::new(make_op(), AllenCahn::<f64>::new(), h).unwrap();
    let mut out = u0.to_vec();
    driver.integrate(u0, n_steps, &mut out, &mut ScratchPool::new()).unwrap();
    out
}

fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0_f64, f64::max)
}

#[test]
#[ignore = "slow gate: G_ETDRK4_ORDER"]
#[allow(clippy::cast_precision_loss)]
fn g_etdrk4_order() {
    let dx = 1.0 / N as f64;
    let u0: Vec<f64> = (0..N)
        .map(|i| 0.5 * (2.0 * std::f64::consts::PI * i as f64 * dx).sin())
        .collect();

    let n_base = 4_usize;
    let h = 0.2_f64 / n_base as f64; // h = 0.05; tau*||L|| ≈ 0.128

    let u_h   = integrate_ac(h,       n_base,     &u0);
    let u_h2  = integrate_ac(h / 2.0, n_base * 2, &u0);
    let u_ref = integrate_ac(h / 4.0, n_base * 4, &u0);

    let e1 = sup_diff(&u_h,  &u_ref);
    let e2 = sup_diff(&u_h2, &u_ref);

    let slope = (e1 / e2).log2();
    assert!(
        (3.7_f64..=4.3_f64).contains(&slope),
        "G_ETDRK4_ORDER: slope={slope:.4} not in [3.7, 4.3] (e1={e1:.2e}, e2={e2:.2e})",
    );
}
