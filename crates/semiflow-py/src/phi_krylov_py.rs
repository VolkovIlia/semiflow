//! Python plumbing of the adaptive Krylov φ path (ADR-0211).
//!
//! ```python
//! y = semiflow.phi_combination(op, tau, w, method="krylov", tol=1e-10, m=30)
//! Y = general_op.evolve_batched(t, V_nc, method="krylov", tol=1e-10)
//! drv = semiflow.Etdrk4.from_symmetric_op(op, "allen_cahn", h, phi="krylov")
//! ```
//!
//! `"taylor"` (the default everywhere) is the augmented Taylor sweep with exact
//! application counts; `"krylov"` is the adaptive Arnoldi path, far cheaper at
//! large `τ‖G‖`.

use pyo3::prelude::*;
use semiflow::{
    general_operator::GeneralOperator, generator_action::GeneratorAction, phi_combination,
    phi_combination_krylov, KrylovPhiOptions, PhiSolver, ScratchPool, SemiflowError,
};

use crate::error::new_pyerr;

/// `method=` of the φ entry points: `"taylor"` (default) or `"krylov"` with `tol`
/// (relative to the augmented start vector) and Krylov dimension `m`.
pub(crate) fn parse_phi_method(method: &str, tol: f64, m: usize) -> PyResult<PhiSolver> {
    match method {
        "taylor" => Ok(PhiSolver::Taylor),
        "krylov" => Ok(PhiSolver::Krylov(
            KrylovPhiOptions::default().with_tol(tol).with_m(m),
        )),
        other => Err(new_pyerr(
            "OutOfDomain",
            &format!("method must be 'taylor' or 'krylov', got '{other}'"),
        )),
    }
}

/// `out ← Σ τᵏφ_k(τG)w_k` with the chosen evaluator.
///
/// # Errors
/// As [`phi_combination`] / [`phi_combination_krylov`].
pub(crate) fn combination<Op: GeneratorAction<f64>>(
    solver: &PhiSolver,
    op: &Op,
    tau: f64,
    w: &[&[f64]],
    out: &mut [f64],
    scratch: &mut ScratchPool<f64>,
) -> Result<(), SemiflowError> {
    match solver {
        PhiSolver::Krylov(opts) => {
            phi_combination_krylov(op, tau, w, out, *opts, scratch).map(|_| ())
        }
        _ => phi_combination(op, tau, w, out, scratch),
    }
}

/// `e^{−tA}` on every column of a column-major `[C][N]` buffer by adaptive Arnoldi.
///
/// # Errors
/// As [`semiflow::expmv_krylov`].
pub(crate) fn expmv_krylov_columns(
    op: &GeneralOperator<f64>,
    t: f64,
    opts: KrylovPhiOptions,
    src_cn: &[f64],
    dst_cn: &mut [f64],
) -> Result<(), SemiflowError> {
    let n = op.n();
    let mut scratch = ScratchPool::new();
    for (src, dst) in src_cn.chunks_exact(n).zip(dst_cn.chunks_exact_mut(n)) {
        semiflow::expmv_krylov(op, t, src, dst, opts, &mut scratch)?;
    }
    Ok(())
}
