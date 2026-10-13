//! Unit tests of the `DiffusionChernoff` plan's identity checks (ADR-0215): every
//! header and engine word that a stale plan could get wrong is checked, and using
//! the stale plan anyway would change the bits (so the checks are not vacuous).

use alloc::{vec, vec::Vec};

use super::*;
use crate::{boundary::InterpKind, grid_fn::GridFn1D, step_plan::ptr_bits};

fn bump(x: f64) -> f64 {
    libm::exp(-2.0 * x * x) + 0.1 * x
}

fn plan(dc: &DiffusionChernoff<f64>, tau: f64, u: &GridFn1D<f64>) -> (Vec<f64>, Vec<f64>) {
    let (d, w) = len(dc, u).unwrap();
    let mut data = vec![0.0; d];
    fill(dc, tau, u, &mut data).unwrap();
    (data, vec![0.0; w.max(4 * 200)])
}

fn applied(
    dc: &DiffusionChernoff<f64>,
    tau: f64,
    plan: Option<(&[f64], &mut [f64])>,
    u: &GridFn1D<f64>,
) -> Vec<f64> {
    let (mut dst, mut pool) = (u.clone(), ScratchPool::new());
    match plan {
        Some((data, work)) => apply(dc, tau, data, work, u, &mut dst, &mut pool).unwrap(),
        None => dc.apply_into(tau, u, &mut dst, &mut pool).unwrap(),
    }
    dst.values
}

fn bits(v: &[f64]) -> Vec<u64> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A plan for grid `a` presented with a state on grid `b` (same length) is not
/// used; forging its header to `b` would give other bits.
#[test]
fn sampling_grid_fields_are_checked_and_matter() {
    let base = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let dc = DiffusionChernoff::new(|x| 0.5 + 0.1 * x * x, |x| 0.2 * x, |_| 0.2, 0.7, base);
    let u = GridFn1D::from_fn(base, bump);
    let (data, mut work) = plan(&dc, 0.01, &u);
    for other in [
        Grid1D::new(-1.0, 1.2, 37).unwrap(),
        Grid1D::new(-0.9, 1.0, 37).unwrap(),
        base.with_interp(InterpKind::CubicHermite),
    ] {
        let v = GridFn1D {
            values: u.values.clone(),
            grid: other,
        };
        assert!(!usable(&dc, 0.01, &data, &work, &v), "{other:?}");
        let want = applied(&dc, 0.01, None, &v);
        assert_eq!(
            bits(&applied(&dc, 0.01, Some((&data, &mut work)), &v)),
            bits(&want)
        );
        if other.interp == base.interp {
            let mut forged = data.clone();
            forged[..HEADER].copy_from_slice(&header(&dc, 2.0, 0.01, 37, &other));
            assert!(usable(&dc, 0.01, &forged, &work, &v));
            let stale = applied(&dc, 0.01, Some((&forged, &mut work)), &v);
            assert_ne!(bits(&stale), bits(&want), "{other:?}: field never matters");
        }
    }
}

/// A plan is tied to the engine VALUE: a changed public grid, or a reassigned
/// engine at the same address, is detected.
#[test]
fn engine_state_is_checked() {
    let g = Grid1D::new(-1.0, 1.0, 37).unwrap();
    let u = GridFn1D::from_fn(g, bump);
    let mut dc = DiffusionChernoff::new_const_a(0.35, 0.35, g);
    let (data, work) = plan(&dc, 0.01, &u);
    assert!(usable(&dc, 0.01, &data, &work, &u));
    dc.grid = Grid1D::new(-1.0, 1.1, 37).unwrap();
    assert!(!usable(&dc, 0.01, &data, &work, &u), "engine grid");
    dc = DiffusionChernoff::new_const_a(0.5, 0.5, g);
    assert!(
        !usable(&dc, 0.01, &data, &work, &u),
        "reassigned constant a"
    );
    let short = &data[..data.len() - 1];
    assert!(!usable(&dc, 0.01, short, &work, &u), "truncated plan");
    let mut wrong = vec![0.0; data.len() + 1];
    assert!(
        fill(&dc, 0.01, &u, &mut wrong).is_err(),
        "wrong fill length"
    );
}

/// Engine addresses are stored as bit patterns: lossless for tagged pointers.
#[test]
fn addresses_are_bit_exact() {
    let (p, q) = (0xb400_007a_1c2d_3000_usize, 0xb400_007a_1c2d_3010_usize);
    assert_ne!(
        ptr_bits(p as *const u8).to_bits(),
        ptr_bits(q as *const u8).to_bits()
    );
}
