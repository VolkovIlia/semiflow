//! Coverage-gap tests for `CoupledTtChernoff` (v9.1.0 QA audit HIGH findings).
//!
//! ## Gap 1 — drift `b!=0` — **FAIL-LOUD (v9.1.0 scope decision)**
//!
//! `CoupledTtChernoff::new` now panics if any `b[j] != 0`.  The per-axis shift
//! used is symmetric (mean-preserving), so drift advection is silently broken;
//! the constructor rejects it loudly rather than silently giving wrong results.
//! Correct drift advection is deferred to v9.2.0 (ADR-0162 / §52.9).
//!
//! `drift_nonzero_is_rejected_v9_1` asserts the panic fires with the expected message.
//!
//! ## Gap 2 — non-adjacent `Pairs` — **FAIL-LOUD (v9.1.0 scope decision)**
//!
//! `CouplingTopology::Pairs([(0, 2, rho)])` for d=3 has `k=2 > j+1=1`.  The pair
//! factor would be silently skipped AND axes 0,2 would lose their diagonal diffusion
//! (worse than a no-op — they are frozen).  The constructor now rejects any pair
//! with `k > j+1` with a clear panic message.  True dense / non-adjacent coupling
//! is deferred to v9.2.0.  Adjacent block-disjoint pairs (e.g. (0,1),(2,3)) are
//! still accepted.
//!
//! `non_adjacent_pair_is_rejected_v9_1` asserts the panic fires with the expected message.

use semiflow::{CoupledTtChernoff, CouplingTopology};

// ===========================================================================
// §A — Gap 1: drift b!=0 is now rejected at construction
// ===========================================================================

/// Gap 1 (v9.1.0 fail-loud): `b[0]=0.5` with `Tridiagonal` coupling panics.
///
/// ## What changed (v9.1.0 scope decision)
///
/// Drift advection on coupled axes was silently broken: the drift-only branch in
/// `diagonal_sweep_with_mask` called `apply_per_axis_shift_pub(core, drift_shift, dx, n)`
/// which implements the SYMMETRIC formula `(1/4)S_{+h} + (1/2)I + (1/4)S_{-h}`.
/// That formula preserves the mean — it is a diffusion step, NOT advection.
/// For `b=-b` the result is identical (sign-invisible), so the defect was
/// completely undetectable at runtime.
///
/// Rather than silently produce wrong results, `CoupledTtChernoff::new` now
/// panics with a clear message pointing to ADR-0162 / §52.9 v9.2.0 deferral.
///
/// Correct drift advection (a one-sided S_{b*tau} scheme) is deferred to v9.2.0.
/// To evolve with drift in v9.1.0: use `TtChernoff` (separable, drift supported)
/// or wait for the v9.2.0 coupled-drift fix.
#[test]
#[should_panic(expected = "CoupledTtChernoff drift b")]
fn drift_nonzero_is_rejected_v9_1() {
    const XMIN: f64 = -8.0;
    const XMAX: f64 = 8.0;
    // This must panic before any evolution takes place.
    let _ev = CoupledTtChernoff::new(
        vec![0.5, 0.3],
        vec![0.5, 0.0], // b[0] = 0.5 != 0 — must trigger the guard
        0.0,
        CouplingTopology::Tridiagonal(0.4f64),
        vec![(XMIN, XMAX); 2],
        1e-8,
    );
}

// ===========================================================================
// §B — Gap 2: non-adjacent Pairs are now rejected at construction
// ===========================================================================

/// Gap 2 (v9.1.0 fail-loud): `Pairs([(0,2,rho)])` for d=3 panics at construction.
///
/// ## What changed (v9.1.0 scope decision)
///
/// `CouplingTopology::Pairs([(0,2,rho)])` for d=3 had `k=2 > j+1=1`.  The old
/// code silently pushed an empty expsym placeholder and skipped the spectral apply,
/// while simultaneously marking axes 0 and 2 as `is_coupled=true` and suppressing
/// their diagonal diffusion.  The net effect was that axes 0 and 2 were FROZEN —
/// neither diagonal diffusion nor coupling was applied.  This was worse than `None`
/// and completely invisible to the caller.
///
/// The constructor now panics with a clear message if any `Pairs` entry has `k > j+1`.
/// True dense / non-adjacent coupling is deferred to v9.2.0.
///
/// Adjacent block-disjoint pairs (e.g. `Pairs([(0,1,rho),(2,3,rho)])` for d=4)
/// are still accepted — they are valid adjacent pairs that work correctly.
#[test]
#[should_panic(expected = "CoupledTtChernoff non-adjacent pair (0,2) with k>j+1 is not supported")]
fn non_adjacent_pair_is_rejected_v9_1() {
    const XMIN: f64 = -4.0;
    const XMAX: f64 = 4.0;
    const RHO: f64 = 0.3;
    // Pair (0,2): k=2 > j+1=1 — non-adjacent — must trigger the guard.
    let _ev = CoupledTtChernoff::new(
        vec![0.5, 0.4, 0.3],
        vec![0.0; 3],
        0.0,
        CouplingTopology::Pairs(vec![(0usize, 2usize, RHO)]),
        vec![(XMIN, XMAX); 3],
        1e-10,
    );
}

// ===========================================================================
// §C — Adjacent block-disjoint Pairs are still accepted (regression guard)
// ===========================================================================

/// Adjacent block-disjoint Pairs `[(0,1),(2,3)]` for d=4 must NOT panic.
///
/// This confirms that the non-adjacent guard does not accidentally block
/// valid configurations. Both (0,1) and (2,3) satisfy k==j+1.
#[test]
fn adjacent_block_disjoint_pairs_are_accepted() {
    const XMIN: f64 = -3.0;
    const XMAX: f64 = 3.0;
    const RHO: f64 = 0.3;
    // (0,1): k=1=j+1=1 OK; (2,3): k=3=j+1=3 OK.
    let _ev = CoupledTtChernoff::new(
        vec![0.5, 0.4, 0.3, 0.2],
        vec![0.0; 4],
        0.0,
        CouplingTopology::Pairs(vec![(0usize, 1usize, RHO), (2usize, 3usize, RHO)]),
        vec![(XMIN, XMAX); 4],
        1e-8,
    );
    // Reaching here means no panic — construction succeeded.
}
