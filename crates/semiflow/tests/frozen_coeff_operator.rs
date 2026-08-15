//! `G_FROZEN_COEFF_*` — which PDE the zeroed-derivative `DiffusionChernoff`
//! solves, and at what order (ADR-0191 AMENDMENT 1, §9.2.3.B.bis).
//!
//! `Heat2DVarA`/`Heat3DVarA` build each axis kernel by handing
//! `DiffusionChernoff::with_closure` a variable `a` together with `a' = 0` and
//! `a'' = 0`. Because that kernel is documented as the ζ-A discretisation of the
//! **divergence** form `∂_x(a ∂_x)` — and genuinely consumes `a'`/`a''`, both in
//! the γ-A sampling offsets and in the whole ζ-A correction — passing zeros with
//! a varying `a` reads as a bug.
//!
//! It is not one, and these two gates are the evidence. Zeroing the derivatives
//! collapses the kernel to a frozen-coefficient stencil, which is a consistent
//! order-1 discretisation of the **non-divergence** operator `a(x)·∂_xx`, the
//! operator all three `Heat2DVarA` binding surfaces advertise.
//!
//! Nothing previously in the suite could tell the two operators apart: every
//! existing oracle for this kernel (`F(0) = I`, constant preservation, temporal
//! self-convergence) is satisfied by *both*, and the only quantitative test used
//! `a ≡ 1`, exactly where they coincide.

// Dense reference construction on small test matrices.
#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use semiflow::{
    chernoff::ChernoffFunction, BoundaryPolicy, DiffusionChernoff, Grid1D, GridFn1D, ScratchPool,
};

const N: usize = 128;
const L: f64 = 1.0;
const T: f64 = 0.02;

fn dx() -> f64 {
    L / N as f64
}

/// `a(x) = 1 + ½·sin(2πx)` — periodic, so the circulant references are exact.
fn a_at(i: usize) -> f64 {
    1.0 + 0.5 * libm::sin(2.0 * core::f64::consts::PI * (i as f64) * dx())
}

fn u0() -> Vec<f64> {
    (0..N)
        .map(|i| libm::exp(libm::cos(2.0 * core::f64::consts::PI * (i as f64) * dx())))
        .collect()
}

/// `a(x)·∂_xx` on a periodic grid: `diag(a) · D₂`.
fn dense_nondivergence() -> Vec<Vec<f64>> {
    let h2 = dx() * dx();
    let mut m = vec![vec![0.0_f64; N]; N];
    for i in 0..N {
        m[i][(i + N - 1) % N] += a_at(i) / h2;
        m[i][i] -= 2.0 * a_at(i) / h2;
        m[i][(i + 1) % N] += a_at(i) / h2;
    }
    m
}

/// `∂_x(a ∂_x)` on a periodic grid, arithmetic face averages.
fn dense_divergence() -> Vec<Vec<f64>> {
    let h2 = dx() * dx();
    let mut m = vec![vec![0.0_f64; N]; N];
    for i in 0..N {
        let face_up = 0.5 * (a_at(i) + a_at((i + 1) % N));
        let face_dn = 0.5 * (a_at(i) + a_at((i + N - 1) % N));
        m[i][(i + 1) % N] += face_up / h2;
        m[i][i] -= (face_up + face_dn) / h2;
        m[i][(i + N - 1) % N] += face_dn / h2;
    }
    m
}

/// `e^{t·M}·v` by scaling-and-squaring with a degree-12 Taylor inner series.
fn expm_action(m: &[Vec<f64>], v: &[f64], t: f64) -> Vec<f64> {
    const K: u32 = 14;
    let s = t / f64::from(1_u32 << K);
    // e^{sM} ≈ Σ_{j≤12} (sM)^j / j!
    let mut acc = vec![vec![0.0_f64; N]; N];
    let mut term = vec![vec![0.0_f64; N]; N];
    for i in 0..N {
        acc[i][i] = 1.0;
        term[i][i] = 1.0;
    }
    for j in 1..=12 {
        term = matmul_scaled(&term, m, s / f64::from(j));
        for i in 0..N {
            for k in 0..N {
                acc[i][k] += term[i][k];
            }
        }
    }
    for _ in 0..K {
        acc = matmul_scaled(&acc, &acc, 1.0);
    }
    (0..N)
        .map(|i| (0..N).map(|k| acc[i][k] * v[k]).sum())
        .collect()
}

fn matmul_scaled(a: &[Vec<f64>], b: &[Vec<f64>], scale: f64) -> Vec<Vec<f64>> {
    let mut out = vec![vec![0.0_f64; N]; N];
    for i in 0..N {
        for k in 0..N {
            let aik = a[i][k];
            if aik == 0.0 {
                continue;
            }
            for j in 0..N {
                out[i][j] += aik * b[k][j] * scale;
            }
        }
    }
    out
}

/// Run the frozen-coefficient kernel (`a' = a'' = 0`) for `n_steps`.
fn run_frozen(n_steps: usize) -> Vec<f64> {
    let grid = Grid1D::new(0.0, L, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::Periodic);
    let a_vals: Vec<f64> = (0..N).map(a_at).collect();
    let norm = a_vals.iter().copied().fold(0.0_f64, f64::max);
    let arc = alloc_arc(a_vals);
    let kernel = DiffusionChernoff::<f64>::with_closure(
        move |x: f64| periodic_lerp(&arc, x),
        |_: f64| 0.0,
        |_: f64| 0.0,
        norm,
        grid,
    );
    let mut src = GridFn1D::new(grid, u0()).unwrap();
    let mut dst = src.clone();
    let mut pool = ScratchPool::<f64>::new();
    let tau = T / n_steps as f64;
    for k in 0..n_steps {
        if k % 2 == 0 {
            kernel.apply_into(tau, &src, &mut dst, &mut pool).unwrap();
        } else {
            kernel.apply_into(tau, &dst, &mut src, &mut pool).unwrap();
        }
    }
    if n_steps % 2 == 0 {
        src.values
    } else {
        dst.values
    }
}

fn alloc_arc(v: Vec<f64>) -> std::sync::Arc<Vec<f64>> {
    std::sync::Arc::new(v)
}

/// Periodic linear lookup of the tabulated `a`.
fn periodic_lerp(vals: &[f64], x: f64) -> f64 {
    let t = x / dx();
    let i = t.floor();
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let i0 = (i.rem_euclid(N as f64)) as usize;
    let frac = t - i;
    let i1 = (i0 + 1) % N;
    (1.0 - frac) * vals[i0] + frac * vals[i1]
}

fn sup_diff(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(p, q)| (p - q).abs())
        .fold(0.0_f64, f64::max)
}

/// `G_FROZEN_COEFF_NONDIV` — the zeroed-derivative kernel solves `a(x)·u_xx`.
///
/// Both dense references are exponentiated by scaling-and-squaring and compared
/// against the kernel at a step count where the temporal error is already below
/// the spatial one. The gate asserts the kernel is at least 5× closer to the
/// non-divergence reference than to the divergence one — with a separation
/// between the two references that is itself an order of magnitude larger than
/// either residual, so the comparison is not measuring noise.
#[test]
#[ignore = "G_FROZEN_COEFF_NONDIV — two dense 128x128 matrix exponentials; Pattern B slow gate"]
fn g_frozen_coeff_nondiv() {
    let v0 = u0();
    let ref_nd = expm_action(&dense_nondivergence(), &v0, T);
    let ref_dv = expm_action(&dense_divergence(), &v0, T);
    let separation = sup_diff(&ref_nd, &ref_dv);
    assert!(
        separation > 1e-2,
        "the two candidate operators agree to {separation:.3e} — the gate is vacuous"
    );

    let got = run_frozen(3200);
    let e_nd = sup_diff(&got, &ref_nd);
    let e_dv = sup_diff(&got, &ref_dv);
    assert!(
        e_nd * 5.0 < e_dv,
        "frozen-coefficient kernel is not clearly the non-divergence operator: \
         |u - a·u_xx| = {e_nd:.3e}, |u - (a u_x)_x| = {e_dv:.3e}, \
         reference separation {separation:.3e}"
    );
}

/// `G_FROZEN_COEFF_ORDER1` — and it is order 1, not 2, for variable `a`.
///
/// `S(τ)f = f + τ·a f'' + (τ²/2)·a² f'''' + …` against
/// `e^{τ a ∂_xx} f = f + τ·a f'' + (τ²/2)·a(a f'')'' + …`: the τ² terms differ by
/// `(τ²/2)·a·(a'' f'' + 2a' f''')` whenever `a` varies. This is why
/// `Heat2DVarA::order()` reports 1.
#[test]
#[ignore = "G_FROZEN_COEFF_ORDER1 — 5 solves incl. a 40960-step reference; Pattern B slow gate"]
fn g_frozen_coeff_order1() {
    let reference = run_frozen(40960);
    let ns = [80_usize, 160, 320, 640];
    let errs: Vec<f64> = ns
        .iter()
        .map(|&n| sup_diff(&run_frozen(n), &reference))
        .collect();
    let (mut sx, mut sy, mut sxx, mut sxy) = (0.0, 0.0, 0.0, 0.0);
    for (&n, &e) in ns.iter().zip(errs.iter()) {
        let (x, y) = (libm::log(n as f64), libm::log(e));
        sx += x;
        sy += y;
        sxx += x * x;
        sxy += x * y;
    }
    let k = ns.len() as f64;
    let slope = (k * sxy - sx * sy) / (k * sxx - sx * sx);
    assert!(
        (-1.25..=-0.85).contains(&slope),
        "frozen-coefficient global order is {:.3}, expected ≈1 \
         (errors {errs:?} at n_steps {ns:?}); if this has moved to ≈2 the \
         kernel changed and `Heat2DVarA::order()` must be revisited",
        -slope
    );
}
