//! Private helpers for `graph_sensitivity` — δΩ₄ and Neumann JVP stages.
//!
//! All items are `pub(crate)` so `graph_sensitivity.rs` can import them
//! without re-exporting to the public API.

use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Laplacian,
    graph_sensitivity::{GeneratorSensitivity, SQRT3_12},
    graph_signal::GraphSignal,
    magnus_graph::{apply_omega4, MagnusGraphHeatChernoff, GL4_C1_F64, GL4_C2_F64},
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// d_omega4_lap — δΩ₄ via explicit Laplacian objects (§43.2)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn d_omega4_lap<F: SemiflowFloat>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    d1: &Laplacian<F>,
    d2: &Laplacian<F>,
    tau: F,
    v: &[F],
    out: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
) {
    let n = v.len();
    let half = from_f64::<F>(0.5);
    let cs = from_f64::<F>(SQRT3_12) * tau * tau;
    d1.apply_into_slice(v, ta);
    d2.apply_into_slice(v, tb);
    for (k, o) in out.iter_mut().enumerate() {
        *o = -half * tau * (ta[k] + tb[k]);
    }
    // [δA₂,A₁]v = δL₂(L₁v) − L₁(δL₂v).
    l1.apply_into_slice(v, ta);
    d2.apply_into_slice(ta, tc);
    d2.apply_into_slice(v, tb);
    l1.apply_into_slice(tb, ta);
    for k in 0..n {
        out[k] += cs * (tc[k] - ta[k]);
    }
    // [A₂,δA₁]v = L₂(δL₁v) − δL₁(L₂v).
    d1.apply_into_slice(v, ta);
    l2.apply_into_slice(ta, tc);
    l2.apply_into_slice(v, tb);
    d1.apply_into_slice(tb, ta);
    for k in 0..n {
        out[k] += cs * (tc[k] - ta[k]);
    }
}

// ---------------------------------------------------------------------------
// d_omega4_tr — δΩ₄ via GeneratorSensitivity trait (§43.3)
// ---------------------------------------------------------------------------

/// Second-commutator stage of `d_omega4_tr`: `[A₂,δA₁]w` accumulation.
#[allow(clippy::too_many_arguments)]
fn comm_a2_da1<F, P>(
    pd: &P,
    p: usize,
    ts: F,
    c1: F,
    l2: &Laplacian<F>,
    tau: F,
    w: &[F],
    out: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    cs: F,
    n: usize,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    // [A₂,δA₁]w = A₂(δA₁w) − δA₁(A₂w).  A₂w = −L₂w.
    pd.apply_param_deriv(p, ts + c1 * tau, w, ta)?; // ta = δA₁w
    l2.apply_into_slice(ta, tc);
    for x in tc.iter_mut() {
        *x = -*x;
    } // tc = A₂(δA₁w)
    l2.apply_into_slice(w, tb);
    for x in tb.iter_mut() {
        *x = -*x;
    } // tb = A₂w
    pd.apply_param_deriv(p, ts + c1 * tau, tb, ta)?; // ta = δA₁(A₂w)
    for k in 0..n {
        out[k] += cs * (tc[k] - ta[k]);
    }
    Ok(())
}

/// Compute `δΩ₄ · w` for parameter `p` using the trait.
#[allow(clippy::too_many_arguments)]
pub(crate) fn d_omega4_tr<F, P>(
    pd: &P,
    p: usize,
    ts: F,
    c1: F,
    c2: F,
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    w: &[F],
    out: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = w.len();
    let half = from_f64::<F>(0.5);
    let cs = from_f64::<F>(SQRT3_12) * tau * tau;
    pd.apply_param_deriv(p, ts + c1 * tau, w, ta)?; // δA₁w
    pd.apply_param_deriv(p, ts + c2 * tau, w, tb)?; // δA₂w
    for k in 0..n {
        out[k] = half * tau * (ta[k] + tb[k]);
    }
    // [δA₂,A₁]w = δA₂(A₁w) − A₁(δA₂w).  A₁w = −L₁w.
    l1.apply_into_slice(w, tc);
    for x in tc.iter_mut() {
        *x = -*x;
    } // tc = A₁w
    pd.apply_param_deriv(p, ts + c2 * tau, tc, ta)?; // ta = δA₂(A₁w)
    l1.apply_into_slice(tb, tc);
    for x in tc.iter_mut() {
        *x = -*x;
    } // tc = A₁(δA₂w)
    for k in 0..n {
        out[k] += cs * (ta[k] - tc[k]);
    }
    comm_a2_da1(pd, p, ts, c1, l2, tau, w, out, ta, tb, tc, cs, n)
}

// ---------------------------------------------------------------------------
// jvp_neumann — m=3 and m=4 helpers + top-level accumulator
// ---------------------------------------------------------------------------

/// m=3 Neumann term: `(1/6)(Σ_{p=0}^{2} Ω^p δΩ Ω^{2-p}) w`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jvp_m3<F, D>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    six: F,
    n: usize,
    w: &[F],
    pw1: &[F],
    pw2: &[F],
    d: &D,
    dv: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    out: &mut [F],
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    D: Fn(&[F], &mut [F], &mut [F], &mut [F], &mut [F]) -> Result<(), SemiflowError>,
{
    let one = F::one();
    let mut ac = alloc::vec![F::zero(); n];
    let mut od = alloc::vec![F::zero(); n];
    d(pw2, dv, ta, tb, tc)?;
    for k in 0..n {
        ac[k] += dv[k];
    }
    d(pw1, dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, dv, &mut od, ta, tb, tc);
    for k in 0..n {
        ac[k] += od[k];
    }
    d(w, dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, dv, &mut od, ta, tb, tc);
    let od2 = od.clone();
    apply_omega4(l1, l2, tau, one, &od2, &mut od, ta, tb, tc);
    for k in 0..n {
        ac[k] += od[k];
    }
    for k in 0..n {
        out[k] += ac[k] / six;
    }
    Ok(())
}

/// m=4 Neumann term: `(1/24)(Σ_{p=0}^{3} Ω^p δΩ Ω^{3-p}) w`.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jvp_m4<F, D>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    tf: F,
    n: usize,
    w: &[F],
    pw1: &[F],
    pw2: &[F],
    pw3: &[F],
    d: &D,
    dv: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    out: &mut [F],
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    D: Fn(&[F], &mut [F], &mut [F], &mut [F], &mut [F]) -> Result<(), SemiflowError>,
{
    let one = F::one();
    let mut ac = alloc::vec![F::zero(); n];
    let mut od = alloc::vec![F::zero(); n];
    d(pw3, dv, ta, tb, tc)?;
    ac.iter_mut().zip(dv.iter()).for_each(|(a, &d)| *a += d);
    d(pw2, dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, dv, &mut od, ta, tb, tc);
    ac.iter_mut().zip(od.iter()).for_each(|(a, &o)| *a += o);
    d(pw1, dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, dv, &mut od, ta, tb, tc);
    let od2 = od.clone();
    apply_omega4(l1, l2, tau, one, &od2, &mut od, ta, tb, tc);
    ac.iter_mut().zip(od.iter()).for_each(|(a, &o)| *a += o);
    d(w, dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, dv, &mut od, ta, tb, tc);
    let od2 = od.clone();
    apply_omega4(l1, l2, tau, one, &od2, &mut od, ta, tb, tc);
    let od3 = od.clone();
    apply_omega4(l1, l2, tau, one, &od3, &mut od, ta, tb, tc);
    ac.iter_mut().zip(od.iter()).for_each(|(a, &o)| *a += o);
    out.iter_mut()
        .zip(ac.iter())
        .for_each(|(o, &a)| *o += a / tf);
    Ok(())
}

/// Accumulate m=2..4 Neumann JVP terms.
#[allow(clippy::too_many_arguments)]
pub(crate) fn jvp_neumann<F, D>(
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    w: &[F],
    pw1: &[F],
    pw2: &[F],
    pw3: &[F],
    d: &D,
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    D: Fn(&[F], &mut [F], &mut [F], &mut [F], &mut [F]) -> Result<(), SemiflowError>,
{
    let n = w.len();
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    let tf = (two + two) * (two + one) * two;
    let mut dv = scratch.take_vec(n);
    let mut od = scratch.take_vec(n);
    // m=2: (1/2)(δΩ(Ω w) + Ω(δΩ w)).
    d(pw1, &mut dv, ta, tb, tc)?;
    let ac0 = dv[..n].to_vec();
    d(w, &mut dv, ta, tb, tc)?;
    apply_omega4(l1, l2, tau, one, &dv, &mut od, ta, tb, tc);
    out.iter_mut()
        .zip(ac0.iter().zip(od.iter()))
        .for_each(|(o, (&a, &b))| *o += (a + b) / two);
    scratch.return_vec(od);
    jvp_m3(
        l1, l2, tau, six, n, w, pw1, pw2, d, &mut dv, ta, tb, tc, out,
    )?;
    jvp_m4(
        l1, l2, tau, tf, n, w, pw1, pw2, pw3, d, &mut dv, ta, tb, tc, out,
    )?;
    scratch.return_vec(dv);
    Ok(())
}

// ---------------------------------------------------------------------------
// fwd_traj — forward trajectory (§43.4)
// ---------------------------------------------------------------------------

pub(crate) fn fwd_traj<F: SemiflowFloat>(
    mc: &MagnusGraphHeatChernoff<F>,
    u0: &GraphSignal<F>,
    n_steps: usize,
    tau: F,
    scratch: &mut ScratchPool<F>,
) -> Result<Vec<GraphSignal<F>>, SemiflowError> {
    let g = u0.graph_arc();
    let mut t = Vec::with_capacity(n_steps + 1);
    t.push(u0.clone());
    for k in 0..n_steps {
        #[allow(clippy::cast_precision_loss)]
        let ts = from_f64::<F>(k as f64) * tau;
        let mut un = GraphSignal::zeros(g.clone());
        mc.apply_into_at(ts, tau, t.last().unwrap(), &mut un, scratch)?;
        t.push(un);
    }
    Ok(t)
}

// ---------------------------------------------------------------------------
// grad_backward — §43.4 backward pass
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn grad_backward<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    traj: &[GraphSignal<F>],
    lam: &mut GraphSignal<F>,
    lam_next: &mut GraphSignal<F>,
    n_steps: usize,
    tau: F,
    pd: &P,
    grad: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = lam.len();
    let c1 = from_f64::<F>(GL4_C1_F64);
    let c2 = from_f64::<F>(GL4_C2_F64);
    let mut ta = scratch.take_vec(n);
    let mut tb = scratch.take_vec(n);
    let mut tc = scratch.take_vec(n);
    let mut jvp = scratch.take_vec(n);
    for sk in (0..n_steps).rev() {
        grad_step(
            mc, traj, lam, lam_next, sk, tau, c1, c2, pd, grad, &mut jvp, &mut ta, &mut tb,
            &mut tc, scratch,
        )?;
    }
    scratch.return_vec(jvp);
    scratch.return_vec(tc);
    scratch.return_vec(tb);
    scratch.return_vec(ta);
    Ok(())
}

/// Inner product `a · b` without allocating.
fn dot_vec<F: SemiflowFloat>(a: &[F], b: &[F]) -> F {
    a.iter()
        .zip(b.iter())
        .fold(F::zero(), |s, (&x, &y)| s + x * y)
}

/// One backward step: adjoint advance + gradient accumulation.
#[allow(clippy::too_many_arguments)]
fn grad_step<F, P>(
    mc: &MagnusGraphHeatChernoff<F>,
    traj: &[GraphSignal<F>],
    lam: &mut GraphSignal<F>,
    lam_next: &mut GraphSignal<F>,
    sk: usize,
    tau: F,
    c1: F,
    c2: F,
    pd: &P,
    grad: &mut [F],
    jvp: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    #[allow(clippy::cast_precision_loss)]
    let ts = from_f64::<F>(sk as f64) * tau;
    mc.apply_state_adjoint_into_at(ts, tau, lam, lam_next, scratch)?;
    let l1 = (mc.lap_at_t)(ts + c1 * tau);
    let l2 = (mc.lap_at_t)(ts + c2 * tau);
    let uk = &traj[sk];
    for (pp, g) in grad.iter_mut().enumerate() {
        step_jvp_tr(
            pd,
            pp,
            ts,
            c1,
            c2,
            &l1,
            &l2,
            tau,
            uk.values(),
            jvp,
            ta,
            tb,
            tc,
            scratch,
        )?;
        *g += dot_vec(lam.values(), jvp);
    }
    core::mem::swap(lam, lam_next);
    Ok(())
}

// ---------------------------------------------------------------------------
// step_jvp_tr — §43.3 JVP for one Magnus step (trait variant)
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
pub(crate) fn step_jvp_tr<F, P>(
    pd: &P,
    p: usize,
    ts: F,
    c1: F,
    c2: F,
    l1: &Laplacian<F>,
    l2: &Laplacian<F>,
    tau: F,
    u: &[F],
    out: &mut [F],
    ta: &mut [F],
    tb: &mut [F],
    tc: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = u.len();
    let mut pw1 = scratch.take_vec(n);
    let mut pw2 = scratch.take_vec(n);
    let mut pw3 = scratch.take_vec(n);
    apply_omega4(l1, l2, tau, F::one(), u, &mut pw1, ta, tb, tc);
    apply_omega4(l1, l2, tau, F::one(), &pw1.clone(), &mut pw2, ta, tb, tc);
    apply_omega4(l1, l2, tau, F::one(), &pw2.clone(), &mut pw3, ta, tb, tc);
    d_omega4_tr(pd, p, ts, c1, c2, l1, l2, tau, u, out, ta, tb, tc)?;
    let df = |w: &[F], o: &mut [F], a: &mut [F], b: &mut [F], c: &mut [F]| {
        d_omega4_tr(pd, p, ts, c1, c2, l1, l2, tau, w, o, a, b, c)
    };
    jvp_neumann(
        l1, l2, tau, u, &pw1, &pw2, &pw3, &df, ta, tb, tc, out, scratch,
    )?;
    scratch.return_vec(pw3);
    scratch.return_vec(pw2);
    scratch.return_vec(pw1);
    Ok(())
}
