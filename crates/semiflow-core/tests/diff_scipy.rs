//! G5 — scipy diff-test (NIGHTLY-ONLY, FEATURE-GATED).
//!
//! Gated behind `#[cfg(feature = "diff-scipy")]`. Compares
//! `ShiftChernoff1D::evolve` to `scipy.linalg.expm` of the 5-point
//! finite-difference stencil for `(1/2) d^2/dx^2` on N=64, tau=0.01.
//!
//! This test is a documented stub for v0.1.0. It is marked `#[ignore]` so
//! `cargo test --features diff-scipy` compiles and reports it as ignored
//! rather than failing. The nightly CI workflow is expected to un-ignore it
//! when full Python subprocess scaffolding is in place.

#![cfg(feature = "diff-scipy")]

#[test]
#[ignore = "G5 stub: full Python subprocess scaffolding required (see TODO below)"]
fn g5_diff_scipy() {
    // TODO(nightly): Build 5-point stencil for (1/2) ∂_xx on N=64.
    //
    // Steps:
    //   1. Compute the (N x N) matrix L where L[i,i] = -1, L[i,i±1] = 0.5,
    //      L[i,i±2] = 0  (standard 5-point FD for 0.5 ∂_xx).
    //   2. Call Python: `python3 -c "
    //        import numpy as np, scipy.linalg, json, sys
    //        n=64; h=20.0/(n-1)
    //        L = np.diag(np.full(n,-1.0/h**2)) + np.diag(np.full(n-1,0.5/h**2),1)
    //            + np.diag(np.full(n-1,0.5/h**2),-1)
    //        x = np.linspace(-10,10,n)
    //        f0 = np.exp(-x**2)
    //        eL = scipy.linalg.expm(0.01 * L)
    //        u = eL @ f0
    //        print(json.dumps(u.tolist()))
    //      "` via std::process::Command.
    //   3. Parse JSON output as Vec<f64>.
    //   4. Run ShiftChernoff1D.evolve(0.01, &f0) on same N=64 grid.
    //   5. Assert max relative error < 1e-3.
    //
    // Blocked on: Python availability check in CI nightly environment.
    todo!("full G5 implementation requires Python subprocess scaffolding");
}
