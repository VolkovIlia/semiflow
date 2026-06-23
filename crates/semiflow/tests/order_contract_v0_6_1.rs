//! Canonical pin: `ChernoffFunction::order()` returns the τ-axis Chernoff
//! consistency order (NOT spatial dx accuracy) for all v0.6.0 inner types.
//!
//! Motivation — audit D1 (`docs/audit-findings-v0_6_0.md`): `Diffusion4thChernoff`
//! and `TruncatedExp4thDiffusionChernoff` returned `4` (dx spatial accuracy), conflating
//! it with τ-axis consistency p (`chernoff.rs:39`, `math.md §11.1`). True: `p = 2`.
//! Bug propagated into `AdaptivePI` Richardson divisor `2^p−1` (`adaptive.rs:152`)
//! and gains `α=0.7/p`, `β=0.4/p` (`adaptive.rs:93-94`): with `p=4` divisor=15 and
//! gains 0.175/0.1, under-estimating LTE ~5× and violating the `tol` contract.
//! See `math.md §11.1.bis` (NORMATIVE, v0.6.1). Intentionally redundant with
//! `diffusion4_unit.rs::order_is_4` and `truncated_exp4_unit.rs::order_is_2` (both
//! assert 2) — those names reflect the v0.6.1 axis-distinction clarification.

use semiflow_core::{
    AdaptivePI, ChernoffFunction, Diffusion4thChernoff, DiffusionChernoff, Grid1D, Strang2D,
    TruncatedExp4thDiffusionChernoff, TruncatedExpDiffusionChernoff,
};

const MSG: &str =
    "::order() must return 2 (τ-axis); spatial accuracy is independent (math.md §11.1.bis, audit D1)";

fn grid32() -> Grid1D {
    Grid1D::new(-1.0, 1.0, 32).unwrap()
}

#[test]
/// `DiffusionChernoff` (ζ-A baseline): τ-axis order is 2.
fn diffusion_chernoff_order_is_2_tau_axis() {
    let c = DiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid32());
    assert_eq!(c.order(), 2, "DiffusionChernoff{MSG}");
}

#[test]
/// `Diffusion4thChernoff` (ζ⁴, 4th-order dx stencil): τ-axis order is still 2.
fn diffusion4th_chernoff_order_is_2_tau_axis() {
    let c = Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid32());
    assert_eq!(c.order(), 2, "Diffusion4thChernoff{MSG}");
}

#[test]
/// `TruncatedExpDiffusionChernoff` (truncated-exp K=4, v0.4.0): τ-axis order is 2.
fn truncated_exp_diffusion_chernoff_order_is_2_tau_axis() {
    let c = TruncatedExpDiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid32());
    assert_eq!(c.order(), 2, "TruncatedExpDiffusionChernoff{MSG}");
}

#[test]
/// `TruncatedExp4thDiffusionChernoff` (4th-order dx + truncated-exp K=4, v0.6.0): τ-axis order is 2.
fn truncated_exp4th_diffusion_chernoff_order_is_2_tau_axis() {
    let c = TruncatedExp4thDiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid32());
    assert_eq!(c.order(), 2, "TruncatedExp4thDiffusionChernoff{MSG}");
}

#[test]
/// `Strang2D` composed from two `Diffusion4thChernoff` inners returns τ-order 2.
/// min(2, 2, 4) = 2 — the cap of 4 in min(x, y, 4) is a forward-compatibility
/// ceiling for hypothetical higher-τ-order inners, not a floor (math.md §11.1.bis).
fn strang2d_with_4th_order_inner_returns_tau_order_2() {
    let gx = Grid1D::new(-1.0, 1.0, 32).unwrap();
    let gy = Grid1D::new(-1.0, 1.0, 32).unwrap();
    let strang = Strang2D::new(
        Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gx),
        Diffusion4thChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, gy),
    );
    assert_eq!(
        strang.order(),
        2,
        "Strang2D<Diffusion4thChernoff,Diffusion4thChernoff>{MSG}"
    );
}

#[test]
/// `AdaptivePI` gains α=0.35, β=0.2 confirm chain: inner.order()==2 → α=0.7/2, β=0.4/2 → divisor=3.
fn adaptive_pi_inherits_inner_tau_order_2() {
    let inner = TruncatedExp4thDiffusionChernoff::new(|_| 0.5, |_| 0.0, |_| 0.0, 0.5, grid32());
    let pi = AdaptivePI::new(inner);
    let msg = format!("TruncatedExp4thDiffusionChernoff{MSG}");
    assert!(
        (pi.alpha() - 0.7 / 2.0).abs() < 1e-15,
        "alpha={} ≠ 0.35; {msg}",
        pi.alpha()
    );
    assert!(
        (pi.beta() - 0.4 / 2.0).abs() < 1e-15,
        "beta={} ≠ 0.2; {msg}",
        pi.beta()
    );
}
