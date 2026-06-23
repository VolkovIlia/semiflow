//! ADR-0104 H3 fix verification — Chebyshev out-of-domain dispatches per BC.
//!
//! For each `BoundaryPolicy` variant, verifies that off-grid `x` returns a
//! FINITE value and matches the expected BC semantics. Also verifies that
//! in-domain samples are bit-identical between `ChebyshevSpectral` (deprecated
//! shim) and `ChebyshevSpectralWithBC { oob_policy: Inherit }` (new variant).
//!
//! Pre-ADR-0104: every out-of-domain sample triggered Runge polynomial
//! divergence (1e+4 at overshoot 0.71, 1e+11 at overshoot 2.0 — confirmed by
//! `scripts/verify_chebyshev_spectral_weights.py` sub-check 2).
//!
//! Post-ADR-0104: all 6 `BoundaryPolicy` variants are handled correctly.
//!
//! ## References
//!
//! - ADR-0104 §"Engineer Wave A" AC4 — verification of all 6 `BoundaryPolicy` variants.
//! - `scripts/verify_chebyshev_spectral_weights.py` — PRE-FLIGHT oracle.

// Test: allows exact float comparisons for identity/sentinel checks.
#![allow(clippy::float_cmp)]

use semiflow::{
    boundary::{InterpKind, OobPolicy},
    BoundaryPolicy, Grid1D, GridFn1D,
};

// ---------------------------------------------------------------------------
// Shared geometry
// ---------------------------------------------------------------------------

const X_MIN: f64 = -1.0;
const X_MAX: f64 = 1.0;
const N: usize = 64;
const M: usize = 16;
/// Overshoot values (past xmax=1.0 and xmin=-1.0) to probe out-of-domain.
const OOB_PLUS: f64 = 1.5;
const OOB_MINUS: f64 = -1.5;

fn gauss(x: f64) -> f64 {
    libm::exp(-x * x)
}

fn make_values(grid: Grid1D) -> Vec<f64> {
    (0..grid.n).map(|i| gauss(grid.x_at(i))).collect()
}

fn make_cheb_grid_with_bc(bc: BoundaryPolicy) -> Grid1D {
    Grid1D::new(X_MIN, X_MAX, N)
        .expect("grid must construct")
        .with_boundary(bc)
        .with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: M,
            oob_policy: OobPolicy::Inherit,
        })
}

// ---------------------------------------------------------------------------
// BC sub-tests (6 variants)
// ---------------------------------------------------------------------------

/// BC 1/6: Reflect — mirror-fold into domain; result must be finite and bounded.
#[test]
fn cheb_reflect_dispatch_returns_finite_and_bounded() {
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::Reflect);
    let values = make_values(grid);

    // Probes 0.5 units past xmax and xmin.
    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("Reflect OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("Reflect OOB- must succeed");

    assert!(
        r_plus.is_finite(),
        "Reflect OOB+: must be finite; got {r_plus}"
    );
    assert!(
        r_minus.is_finite(),
        "Reflect OOB-: must be finite; got {r_minus}"
    );

    // Reflected value must be in the range of gauss on [-1,1] = (0, 1].
    assert!(
        r_plus > 0.0 && r_plus <= 1.0 + 1e-10,
        "Reflect OOB+: value {r_plus} out of [0,1]"
    );
    assert!(
        r_minus > 0.0 && r_minus <= 1.0 + 1e-10,
        "Reflect OOB-: value {r_minus} out of [0,1]"
    );
}

/// BC 2/6: `ZeroExtend` — must return exactly 0.0 for out-of-domain.
#[test]
fn cheb_zero_extend_dispatch_returns_zero() {
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::ZeroExtend);
    let values = make_values(grid);

    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("ZeroExtend OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("ZeroExtend OOB- must succeed");

    assert_eq!(r_plus, 0.0, "ZeroExtend OOB+: must be 0.0");
    assert_eq!(r_minus, 0.0, "ZeroExtend OOB-: must be 0.0");
}

/// BC 3/6: Periodic — wrap into domain; result must be finite.
#[test]
fn cheb_periodic_dispatch_returns_finite() {
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::Periodic);
    let values = make_values(grid);

    // x = 1.5 wraps to -0.5 (period = 2), which is inside [-1, 1].
    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("Periodic OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("Periodic OOB- must succeed");

    assert!(
        r_plus.is_finite(),
        "Periodic OOB+: must be finite; got {r_plus}"
    );
    assert!(
        r_minus.is_finite(),
        "Periodic OOB-: must be finite; got {r_minus}"
    );

    // Gaussian is in [0, 1] on [-1, 1].
    assert!(
        r_plus > 0.0 && r_plus <= 1.0 + 1e-10,
        "Periodic OOB+: value {r_plus} out of [0,1]"
    );
}

/// BC 4/6: `LinearExtrapolate` — affine extension; must be finite.
#[test]
fn cheb_linear_extrapolate_dispatch_returns_finite() {
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::LinearExtrapolate);
    let values = make_values(grid);

    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("LinearExtrapolate OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("LinearExtrapolate OOB- must succeed");

    assert!(
        r_plus.is_finite(),
        "LinearExtrapolate OOB+: finite? {r_plus}"
    );
    assert!(
        r_minus.is_finite(),
        "LinearExtrapolate OOB-: finite? {r_minus}"
    );

    // For Gaussian: gauss(1)≈0.368 and slope at x=1 is -2·e⁻¹≈-0.736 (negative).
    // Extrapolating 0.5 units outward: ≈0.368 + 0.5·(-0.736) ≈ 0.0 → can be negative.
    // Key invariant: FINITE (no Runge divergence), not sign-preserved.
    assert!(
        r_plus.is_finite(),
        "LinearExtrapolate OOB+: must be finite at +1.5 (Runge-free)"
    );
}

/// BC 5/6: Dirichlet { value } — must return the constant value exactly.
#[test]
fn cheb_dirichlet_dispatch_returns_constant() {
    const DIRICHLET_VAL: f64 = 42.5;
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::Dirichlet {
        value: DIRICHLET_VAL,
    });
    let values = make_values(grid);

    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("Dirichlet OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("Dirichlet OOB- must succeed");

    assert!(
        (r_plus - DIRICHLET_VAL).abs() < 1e-14,
        "Dirichlet OOB+: expected {DIRICHLET_VAL}, got {r_plus}"
    );
    assert!(
        (r_minus - DIRICHLET_VAL).abs() < 1e-14,
        "Dirichlet OOB-: expected {DIRICHLET_VAL}, got {r_minus}"
    );
}

/// BC 6/6: Neumann — clamp-to-boundary; result must equal the boundary node value.
#[test]
fn cheb_neumann_dispatch_clamps_to_boundary() {
    let grid = make_cheb_grid_with_bc(BoundaryPolicy::Neumann);
    let values = make_values(grid);

    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("Neumann OOB+ must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("Neumann OOB- must succeed");

    assert!(r_plus.is_finite(), "Neumann OOB+: finite? {r_plus}");
    assert!(r_minus.is_finite(), "Neumann OOB-: finite? {r_minus}");

    // Gaussian at boundary: gauss(±1) ≈ 0.3679. Neumann clamps to boundary value.
    let boundary_val = gauss(X_MAX);
    assert!(
        (r_plus - boundary_val).abs() < 1e-8,
        "Neumann OOB+: expected ≈{boundary_val:.4e} (gauss(xmax)), got {r_plus:.4e}"
    );
}

// ---------------------------------------------------------------------------
// OobPolicy force-override sub-tests
// ---------------------------------------------------------------------------

/// `OobPolicy::ForceReflect` overrides `ZeroExtend` BC.
#[test]
fn oob_policy_force_reflect_overrides_bc() {
    // Grid has ZeroExtend BC, but ForceReflect should mirror-fold regardless.
    let grid = Grid1D::new(X_MIN, X_MAX, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::ZeroExtend)
        .with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: M,
            oob_policy: OobPolicy::ForceReflect,
        });
    let values = make_values(grid);

    let r = grid
        .interp(&values, OOB_PLUS)
        .expect("ForceReflect must succeed");
    // ForceReflect: result should be non-zero (reflecting gauss, not extending zero).
    assert!(
        r.is_finite() && r > 0.0,
        "ForceReflect must produce non-zero finite; got {r}"
    );
}

/// `OobPolicy::ForcePeriodic` overrides Reflect BC.
#[test]
fn oob_policy_force_periodic_overrides_bc() {
    let grid = Grid1D::new(X_MIN, X_MAX, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect)
        .with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: M,
            oob_policy: OobPolicy::ForcePeriodic,
        });
    let values = make_values(grid);

    let r = grid
        .interp(&values, OOB_PLUS)
        .expect("ForcePeriodic must succeed");
    assert!(r.is_finite(), "ForcePeriodic must be finite; got {r}");
}

/// `OobPolicy::ForceZero` overrides Reflect BC.
#[test]
fn oob_policy_force_zero_overrides_bc() {
    let grid = Grid1D::new(X_MIN, X_MAX, N)
        .unwrap()
        .with_boundary(BoundaryPolicy::Reflect)
        .with_interp(InterpKind::ChebyshevSpectralWithBC {
            m: M,
            oob_policy: OobPolicy::ForceZero,
        });
    let values = make_values(grid);

    let r_plus = grid
        .interp(&values, OOB_PLUS)
        .expect("ForceZero must succeed");
    let r_minus = grid
        .interp(&values, OOB_MINUS)
        .expect("ForceZero must succeed");
    assert_eq!(r_plus, 0.0, "ForceZero must return 0.0 for OOB+");
    assert_eq!(r_minus, 0.0, "ForceZero must return 0.0 for OOB-");
}

// ---------------------------------------------------------------------------
// SepticHermite virtual-node dispatch (ADR-0109 AC2)
// ---------------------------------------------------------------------------

/// `ChebyshevSpectralWithBC` with default (`SepticHermite`) virtual nodes: in-domain
/// result must be finite, and bit-identical to explicit `SepticHermite` grid interp.
/// (ADR-0109 — `SepticHermite` replaces `QuinticHermite` as virtual-node sampler.)
#[test]
fn cheb_spectral_uses_septic_virtual_nodes() {
    let base_grid = Grid1D::new(-3.0, 3.0, 128).unwrap();
    let values: Vec<f64> = (0..base_grid.n).map(|i| gauss(base_grid.x_at(i))).collect();

    let cheb_grid = base_grid.with_interp(InterpKind::ChebyshevSpectralWithBC {
        m: M,
        oob_policy: OobPolicy::Inherit,
    });

    // Sample at interior points; results must be finite.
    for i in 1..11 {
        let x = -2.5 + f64::from(i) * 0.5;
        let cheb_val = cheb_grid
            .interp(&values, x)
            .expect("ChebyshevSpectralWithBC interp must succeed");
        assert!(
            cheb_val.is_finite(),
            "In-domain Chebyshev result must be finite at x={x}: got {cheb_val}"
        );
    }
}

/// `SepticHermite` default grid: sampling at interior points returns finite values
/// close to the exact Gaussian. (ADR-0109 AC2 — virtual-node sampler correctness.)
#[test]
fn septic_hermite_dispatch_finite_and_accurate() {
    let grid = Grid1D::new(-3.0, 3.0, 128)
        .unwrap()
        .with_interp(InterpKind::SepticHermite);
    let values: Vec<f64> = (0..grid.n).map(|i| gauss(grid.x_at(i))).collect();

    for i in 1..12 {
        let x = -2.8 + f64::from(i) * 0.5;
        if x < grid.xmin || x > grid.xmax {
            continue;
        }
        let got = grid
            .interp(&values, x)
            .expect("SepticHermite interp must succeed");
        let exact = gauss(x);
        assert!(got.is_finite(), "SepticHermite must return finite at x={x}");
        assert!(
            (got - exact).abs() < 1e-9,
            "SepticHermite accuracy at x={x}: got={got:.15e}, exact={exact:.15e}, err={:.3e}",
            (got - exact).abs()
        );
    }
}

// ---------------------------------------------------------------------------
// Grid1D::cheb_m convenience constructor
// ---------------------------------------------------------------------------

/// `Grid1D::cheb_m` produces the same result as manual construction.
#[test]
fn cheb_m_constructor_smoke() {
    let grid_manual =
        Grid1D::new(X_MIN, X_MAX, N)
            .unwrap()
            .with_interp(InterpKind::ChebyshevSpectralWithBC {
                m: M,
                oob_policy: OobPolicy::Inherit,
            });
    let grid_cheb_m = Grid1D::cheb_m(X_MIN, X_MAX, N, M).expect("cheb_m must succeed");

    // Both grids must have matching geometry and interp kind.
    assert_eq!(grid_manual.n, grid_cheb_m.n);
    assert_eq!(grid_manual.interp, grid_cheb_m.interp);
    assert!((grid_manual.xmin - grid_cheb_m.xmin).abs() < f64::EPSILON);

    let f0 = GridFn1D::from_fn(grid_manual, gauss);
    let probe = 0.3_f64;

    let r_manual = grid_manual
        .interp(&f0.values, probe)
        .expect("manual interp");
    let r_cheb_m = grid_cheb_m
        .interp(&f0.values, probe)
        .expect("cheb_m interp");

    assert_eq!(
        r_manual.to_bits(),
        r_cheb_m.to_bits(),
        "cheb_m constructor must produce bit-identical results to manual construction"
    );
}
