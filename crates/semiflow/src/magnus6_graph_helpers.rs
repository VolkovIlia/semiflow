//! Internal computation helpers for [`super::MagnusGraphHeat6thChernoff`].
//!
//! Separated for the ≤500-line file budget (constitution Override #1).
//! All items are `pub(super)` — NOT part of the public API.
//! Buffer-operation order is NORMATIVE (bit-identity gates); do not reorder.

use super::{ONE_OVER_12, ONE_OVER_240, ONE_OVER_60, SQRT15_OVER_3, TEN_OVER_3};
use crate::{
    float::{from_f64, SemiflowFloat},
    graph::Laplacian,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
    state::State,
};

// ---------------------------------------------------------------------------
// Taylor-truncation kernel: exp(Ω₆)·src via degree-6 expansion
// ---------------------------------------------------------------------------

/// Compute `Σ_{k=0..6} Ω₆^k/k! · src`, write to `dst`.
///
/// Acquires **22 scratch buffers** and returns all before returning
/// (R4 zero-alloc invariant, ADR-0114).
#[allow(clippy::too_many_arguments)]
pub(super) fn apply_exp_omega6_kernel<F: SemiflowFloat>(
    lap1: &Laplacian<F>,
    lap2: &Laplacian<F>,
    lap3: &Laplacian<F>,
    tau: F,
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    scratch: &mut ScratchPool<F>,
) {
    let n = src.len();
    let gl6 = [lap1, lap2, lap3];
    let (mut ov, mut op) = (scratch.take_vec(n), scratch.take_vec(n));
    // 20 scratch buffers acquired as array (s00..s19); all returned before return.
    let mut sv: [Vec<F>; 20] = core::array::from_fn(|_| scratch.take_vec(n));
    let s: &mut [&mut [F]; 20] = &mut sv.each_mut().map(|v| -> &mut [F] { v });

    macro_rules! omega6 {
        ($pv:expr, $dv:expr) => {
            apply_omega6(&gl6, tau, $pv, $dv, s)
        };
    }

    omega6!(src.values(), &mut ov); // k=1: Ω₆·src → ov
    accumulate_taylor6(src, dst, &mut ov, &mut op, |pv, dv| omega6!(pv, dv));

    scratch.return_vec(ov);
    scratch.return_vec(op);
    for buf in sv {
        scratch.return_vec(buf);
    }
}

/// Accumulate `Σ_{k=0..6} Ω^k/k! · src` into `dst`.
///
/// On entry `omega_v` holds `Ω·src` (k=1). Operations in fixed order
/// (bit-identity invariant).
fn accumulate_taylor6<F, Step>(
    src: &GraphSignal<F>,
    dst: &mut GraphSignal<F>,
    omega_v: &mut [F],
    omega_pow: &mut [F],
    mut step: Step,
) where
    F: SemiflowFloat,
    Step: FnMut(&[F], &mut [F]),
{
    let one = F::one();
    let two = one + one;
    let six = two + two + two;
    let f24 = six * (two + two);
    let f120 = f24 * (two + two + two - one); // 24 * 5
    let f720 = f120 * (two + two + two); // 120 * 6

    dst.copy_from(src);
    dst.axpy_into_slice(one, omega_v);
    omega_pow.copy_from_slice(omega_v);

    step(omega_pow, omega_v);
    dst.axpy_into_slice(one / two, omega_v);
    omega_pow.copy_from_slice(omega_v);

    step(omega_pow, omega_v);
    dst.axpy_into_slice(one / six, omega_v);
    omega_pow.copy_from_slice(omega_v);

    step(omega_pow, omega_v);
    dst.axpy_into_slice(one / f24, omega_v);
    omega_pow.copy_from_slice(omega_v);

    step(omega_pow, omega_v);
    dst.axpy_into_slice(one / f120, omega_v);
    omega_pow.copy_from_slice(omega_v);

    step(omega_pow, omega_v);
    dst.axpy_into_slice(one / f720, omega_v);
}

// ---------------------------------------------------------------------------
// Ω₆·v helper (NORMATIVE — math.md §16.2 BCOR-6 / ADR-0114)
// ---------------------------------------------------------------------------

/// Apply `Ω₆(τ)·v` → `out`.
///
/// `laps`: `[lap1, lap2, lap3]`. `s`: 20-slot scratch array (s00..s19).
/// Buffer-operation order is NORMATIVE (bit-identity).
fn apply_omega6<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    v: &[F],
    out: &mut [F],
    s: &mut [&mut [F]; 20],
) {
    let sq = from_f64::<F>(SQRT15_OVER_3);
    let t3 = from_f64::<F>(TEN_OVER_3);
    let i12 = from_f64::<F>(ONE_OVER_12);
    let i60 = from_f64::<F>(ONE_OVER_60);
    let i240 = from_f64::<F>(ONE_OVER_240);
    let n20 = from_f64::<F>(-20.0);
    let n = v.len();
    let [s00, s01, s02, s03, s04, s05, s06, s07, s08, s09, s10, s11, s12, s13, s14, s15, s16, _s17, _s18, s19] =
        s;

    phase12_b_and_c1(laps, tau, v, out, s00, s01, s02, s03, s04, sq, t3);
    phase3_c2(
        laps, tau, out, s00, s01, s02, s03, s05, s06, s07, s08, s09, s10, s11, sq, t3, i60,
    );
    for i in 0..n {
        s06[i] = n20 * s03[i] - out[i] + s00[i]; // Lv
        s07[i] = s04[i] + s05[i]; // Rv
    }
    phase5a_lop_rv(
        laps, tau, out, s00, s01, s02, s07, s08, s09, s10, s11, s12, s15, sq, t3, n20,
    );
    phase5b_rop_lv(
        laps, tau, s00, s01, s02, s06, s08, s09, s10, s11, s12, s13, s14, s16, s19, sq, t3, i60,
    );
    for i in 0..n {
        out[i] = s03[i] + i12 * out[i] + i240 * (s15[i] - s19[i]);
    }
}

/// Phases 1–2: B-vectors then C₁v → s00.
///
/// After: s03=B₁v, s04=B₂v, out=B₃v, s00=C₁v.
#[allow(clippy::too_many_arguments)]
fn phase12_b_and_c1<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    v: &[F],
    out: &mut [F],
    s00: &mut [F],
    s01: &mut [F],
    s02: &mut [F],
    s03: &mut [F],
    s04: &mut [F],
    sq: F,
    t3: F,
) {
    let n = v.len();
    laps[0].apply_into_slice(v, s00);
    laps[1].apply_into_slice(v, s01);
    laps[2].apply_into_slice(v, s02);
    for i in 0..n {
        let (a1, a2, a3) = (-s00[i], -s01[i], -s02[i]);
        s03[i] = tau * a2;
        s04[i] = tau * sq * (a3 - a1);
        out[i] = tau * t3 * (a3 - a2 - a2 + a1);
    }
    // C₁v = B₁(B₂v) − B₂(B₁v); result → s00.
    laps[1].apply_into_slice(s04, s00);
    laps[0].apply_into_slice(s03, s01);
    laps[2].apply_into_slice(s03, s02);
    for i in 0..n {
        let b1_b2v = -tau * s00[i];
        let b2_b1v = tau * sq * (s01[i] - s02[i]);
        s00[i] = b1_b2v - b2_b1v;
    }
}

/// Phase 3: C₂v = −(1/60)[B₁, 2B₃+C₁]v → s05.
///
/// Reads: s00=C₁v, s03=B₁v, out=B₃v.
#[allow(clippy::too_many_arguments)]
fn phase3_c2<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    out: &[F],
    s00: &[F],
    s01: &mut [F],
    s02: &mut [F],
    s03: &[F],
    s05: &mut [F],
    s06: &mut [F],
    s07: &mut [F],
    s08: &mut [F],
    s09: &mut [F],
    s10: &mut [F],
    s11: &mut [F],
    sq: F,
    t3: F,
    i60: F,
) {
    let n = out.len();
    for i in 0..n {
        s01[i] = out[i] + out[i] + s00[i]; // wv = 2B₃v + C₁v
    }
    laps[1].apply_into_slice(s01, s02);
    for v in &mut s02[..n] {
        *v = -tau * *v;
    } // B₁(wv)
    laps[0].apply_into_slice(s03, s05);
    laps[1].apply_into_slice(s03, s06);
    laps[2].apply_into_slice(s03, s07);
    for i in 0..n {
        s08[i] = tau * sq * (s05[i] - s07[i]); // B₂(B₁v)
    }
    laps[1].apply_into_slice(s08, s09);
    laps[0].apply_into_slice(s06, s10);
    laps[2].apply_into_slice(s06, s11);
    for i in 0..n {
        let b1_b2_b1v = -tau * s09[i];
        let b2_b1_b1v = tau * sq * (-tau) * (s10[i] - s11[i]);
        let c1_b1v = b1_b2_b1v - b2_b1_b1v;
        let two_b3_b1v = (tau + tau) * t3 * (-s07[i] + s06[i] + s06[i] - s05[i]);
        s05[i] = -i60 * (s02[i] - (two_b3_b1v + c1_b1v));
    }
}

/// Phase 5a: `lop_rv = (−20B₁−B₃+C₁)(Rv)` → s15.
///
/// Reads: s07=Rv, out=B₃v. Workspace: s00..s02,s08..s12.
#[allow(clippy::too_many_arguments)]
fn phase5a_lop_rv<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    out: &[F],
    s00: &mut [F],
    s01: &mut [F],
    s02: &mut [F],
    s07: &[F],
    s08: &mut [F],
    s09: &mut [F],
    s10: &mut [F],
    s11: &mut [F],
    s12: &mut [F],
    s15: &mut [F],
    sq: F,
    t3: F,
    n20: F,
) {
    let n = s07.len();
    laps[0].apply_into_slice(s07, s08);
    laps[1].apply_into_slice(s07, s09);
    laps[2].apply_into_slice(s07, s10);
    for i in 0..n {
        s00[i] = -tau * s09[i]; // B₁(Rv)
        s01[i] = tau * sq * (s08[i] - s10[i]); // B₂(Rv)
    }
    laps[1].apply_into_slice(s01, s02);
    laps[0].apply_into_slice(s00, s11);
    laps[2].apply_into_slice(s00, s12);
    for i in 0..n {
        let b3rv = tau * t3 * (-s10[i] + s09[i] + s09[i] - s08[i]);
        let b1_b2rv = -tau * s02[i];
        let b2_b1rv = tau * sq * (s11[i] - s12[i]);
        s15[i] = n20 * s00[i] - b3rv + (b1_b2rv - b2_b1rv);
    }
    let _ = out; // B₃v kept alive by caller for phase 6
}

/// Phase 5b: `rop_lv = (B₂+C₂)(Lv)` → s19.
///
/// Reads: s06=Lv. Workspace: s00..s02,s08..s14,s16.
#[allow(clippy::too_many_arguments)]
fn phase5b_rop_lv<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    s00: &mut [F],
    s01: &mut [F],
    s02: &mut [F],
    s06: &[F],
    s08: &mut [F],
    s09: &mut [F],
    s10: &mut [F],
    s11: &mut [F],
    s12: &mut [F],
    s13: &mut [F],
    s14: &mut [F],
    s16: &mut [F],
    s19: &mut [F],
    sq: F,
    t3: F,
    i60: F,
) {
    let n = s06.len();
    laps[0].apply_into_slice(s06, s08);
    laps[1].apply_into_slice(s06, s09);
    laps[2].apply_into_slice(s06, s10);
    for i in 0..n {
        s00[i] = -tau * s09[i]; // B₁Lv
        s01[i] = tau * sq * (s08[i] - s10[i]); // B₂Lv
    }
    // C₁(Lv) and wLv = (2B₃+C₁)(Lv) → s02.
    laps[1].apply_into_slice(s01, s02);
    laps[0].apply_into_slice(s00, s11);
    laps[2].apply_into_slice(s00, s12);
    for i in 0..n {
        let b3lv = tau * t3 * (-s10[i] + s09[i] + s09[i] - s08[i]);
        let c1lv = -tau * s02[i] - tau * sq * (s11[i] - s12[i]);
        s02[i] = b3lv + b3lv + c1lv; // wLv
    }
    phase5b_c2lv(
        laps, tau, s00, s01, s02, s08, s09, s11, s12, s13, s14, s16, s19, sq, t3, i60,
    );
}

/// Phase 5b part 2: compute C₂(Lv) and assemble `rop_lv` → s19.
///
/// On entry: s00=B₁Lv, s01=B₂Lv, s02=wLv=(2B₃+C₁)(Lv), s11/s12 from C₁(Lv).
#[allow(clippy::too_many_arguments)]
fn phase5b_c2lv<F: SemiflowFloat>(
    laps: &[&Laplacian<F>; 3],
    tau: F,
    s00: &[F],
    s01: &[F],
    s02: &[F],
    s08: &mut [F],
    s09: &mut [F],
    s11: &[F],
    s12: &[F],
    s13: &mut [F],
    s14: &mut [F],
    s16: &mut [F],
    s19: &mut [F],
    sq: F,
    t3: F,
    i60: F,
) {
    let n = s00.len();
    laps[1].apply_into_slice(s02, s08);
    for v in &mut s08[..n] {
        *v = -tau * *v; // B₁(wLv)
    }
    laps[1].apply_into_slice(s00, s09);
    for i in 0..n {
        s13[i] = tau * sq * (s11[i] - s12[i]);
    }
    laps[1].apply_into_slice(s13, s14);
    laps[0].apply_into_slice(s09, s13);
    laps[2].apply_into_slice(s09, s16);
    for i in 0..n {
        let b1_b2_b1lv = -tau * s14[i];
        let b2_b1_b1lv = tau * sq * (-tau) * (s13[i] - s16[i]);
        let c1_b1lv = b1_b2_b1lv - b2_b1_b1lv;
        let two_b3_b1lv = (tau + tau) * t3 * (-s12[i] + s09[i] + s09[i] - s11[i]);
        let c2lv = -i60 * (s08[i] - (two_b3_b1lv + c1_b1lv));
        s19[i] = s01[i] + c2lv;
    }
}
