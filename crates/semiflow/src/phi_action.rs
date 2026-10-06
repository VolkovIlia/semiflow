//! φ-function actions via the augmented-matrix construction (ADR-0189 §58.2).
//!
//! ## Summary
//!
//! `φ_k(z) = Σ_{n≥0} zⁿ / (n+k)!` with `φ_0(z) = eᶻ`.
//!
//! [`phi_action`] computes `φ_k(τA)·v` for a single `k ≤ PHI_MAX`.
//! [`phi_action_batched`] computes all `φ_0…φ_p` in a single call
//! (each at the same `τ` and `v`).
//!
//! ## Algorithm
//!
//! Build the `(n+p) × (n+p)` augmented operator
//! `B̃_η = [[τG, η·W], [0, J_p]]` (ADR-0202 §62.4).  A Horner sweep on
//! `(1/s)·B̃_η` with `(s,m)` from Al-Mohy–Higham Algorithm 3.2 yields
//! `exp(B̃_η)·z_init`; the top-n rows equal the requested φ combination.
//!
//! **Cost is independent of the input vectors.**  The coupling block is
//! normalised by a power of two `η = 2^{−⌈log₂‖v‖∞⌉}` (the companion block by
//! `1/η`), so `‖B̃_η‖ ≤ τ‖G‖ + 2` regardless of `‖v‖`; `(s,m)` follows from
//! [`phi_cost_probe`].  Powers of two are exact in IEEE arithmetic, hence
//! `φ(2^j v) = 2^j φ(v)` bit for bit (gate `G_PHI_COST_V_INVARIANT`).
//!
//! ## References
//! - ADR-0189, ADR-0202; math.md §58, §62 (NORMATIVE).
//! - Al-Mohy & Higham (2011) SIAM J. Sci. Comput. 33:488–511.

extern crate alloc;

use alloc::vec;
use alloc::vec::Vec;

use crate::{
    error::SemiflowError,
    expmv::select_s_m,
    float::SemiflowFloat,
    generator_action::GeneratorAction,
    graph_krylov::MAX_DENSE_N,
    matrix_pade::mat_exp_pade13,
    phi_action_helpers::{eta_scaling, run_sweeps, Coupling},
    scratch::ScratchPool,
};

/// Maximum supported φ index.
pub const PHI_MAX: usize = 3;

/// Norm-tightening factor for `select_s_m` in the φ-action.
///
/// `THETA_M` is calibrated for the exponential BACKWARD error; the φ-extraction
/// from the augmented block requires FORWARD accuracy.  Multiplying the norm
/// estimate by this factor nudges `select_s_m` to choose a higher Taylor degree
/// (m=18 instead of m=13 at the canonical z≈2 test point), reducing the per-substep
/// truncation from ~9e-9 to ~8e-14 without adding extra squarings.
const PHI_NORM_TIGHTEN: f64 = 2.0;

/// `(s, m)` used by every φ entry point for a generator-norm bound `norm_g` and step `tau`.
///
/// The augmented norm is `τ‖G‖ + 2` (the η-scaled coupling block contributes at
/// most 1, the nilpotent block at most 1), tightened by `PHI_NORM_TIGHTEN`.
/// Independent of every input vector by construction (§62.4,
/// `G_PHI_COST_V_INVARIANT`).
#[must_use]
pub fn phi_cost_probe(norm_g: f64, tau: f64) -> (u32, u32) {
    select_s_m((tau * norm_g + 2.0) * PHI_NORM_TIGHTEN, 1.0)
}

/// `(s, m)` for `op` and `tau`.
fn sweep_params<F: SemiflowFloat, Op: GeneratorAction<F>>(op: &Op, tau: F) -> (u32, u32) {
    phi_cost_probe(
        op.norm_bound().to_f64().unwrap_or(0.0),
        tau.to_f64().unwrap_or(0.0),
    )
}

/// `‖v‖∞` as `f64`.
fn sup_norm<F: SemiflowFloat>(v: &[F]) -> f64 {
    v.iter()
        .map(|x| x.abs().to_f64().unwrap_or(0.0))
        .fold(0.0_f64, f64::max)
}

// ---------------------------------------------------------------------------
// phi_action_batched
// ---------------------------------------------------------------------------

/// Compute `φ_k(τG)·v` for all `k = 0 … p` simultaneously.
///
/// # Arguments
/// - `op`: linear generator providing `G`-matvec.
/// - `p`: max φ index (must be `≤ PHI_MAX = 3`).
/// - `tau`: time step.
/// - `v`: input vector, length `op.dim()`.
/// - `out`: output buffer, length `(p+1) * op.dim()`.
///   Slice `out[k*n .. (k+1)*n]` receives `φ_k(τG)·v`.
/// - `scratch`: reusable allocation pool.
///
/// # Errors
/// Returns `DomainViolation` if `p > PHI_MAX`.
#[allow(clippy::many_single_char_names)]
pub fn phi_action_batched<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    p: usize,
    tau: F,
    v: &[F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if p > PHI_MAX {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "phi_action_batched: p > PHI_MAX (3)",
            value: p as f64,
        });
    }
    let n = op.dim();
    let dim_aug = n + PHI_MAX; // always n+3; uniform across all k
    let (s, m) = sweep_params(op, tau);
    let (eta, inv_eta) = eta_scaling::<F>(sup_norm(v));
    let coupling = Coupling {
        cols: &[v],
        coef: &[eta],
    };

    // Scratch: y (n+3), w (n+3), av (n)
    let mut y_aug = scratch.take_vec(dim_aug);
    let mut w_aug = scratch.take_vec(dim_aug);
    let mut av_buf = scratch.take_vec(n);

    for k in 0..=p {
        init_aug_vector(v, n, PHI_MAX, k, inv_eta, &mut y_aug);
        run_sweeps(
            op,
            &coupling,
            tau,
            s,
            m,
            &mut y_aug,
            &mut w_aug,
            &mut av_buf,
        )?;
        // Extract top-n into out[k*n .. (k+1)*n]
        out[k * n..(k + 1) * n].copy_from_slice(&y_aug[..n]);
    }

    scratch.return_vec(av_buf);
    scratch.return_vec(w_aug);
    scratch.return_vec(y_aug);
    Ok(())
}

// ---------------------------------------------------------------------------
// phi_action (single k)
// ---------------------------------------------------------------------------

/// Compute `φ_k(τG)·v` for a single index `k ≤ PHI_MAX`.
///
/// Equivalent to `phi_action_batched` restricted to one k.
///
/// # Errors
/// Returns `DomainViolation` if `k > PHI_MAX`.
#[allow(clippy::many_single_char_names)]
pub fn phi_action<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    k: usize,
    tau: F,
    v: &[F],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    if k > PHI_MAX {
        #[allow(clippy::cast_precision_loss)]
        return Err(SemiflowError::DomainViolation {
            what: "phi_action: k > PHI_MAX (3)",
            value: k as f64,
        });
    }
    let n = op.dim();
    let dim_aug = n + PHI_MAX;
    let (s, m) = sweep_params(op, tau);
    let (eta, inv_eta) = eta_scaling::<F>(sup_norm(v));
    let coupling = Coupling {
        cols: &[v],
        coef: &[eta],
    };

    let mut y_aug = scratch.take_vec(dim_aug);
    let mut w_aug = scratch.take_vec(dim_aug);
    let mut av_buf = scratch.take_vec(n);

    init_aug_vector(v, n, PHI_MAX, k, inv_eta, &mut y_aug);
    run_sweeps(
        op,
        &coupling,
        tau,
        s,
        m,
        &mut y_aug,
        &mut w_aug,
        &mut av_buf,
    )?;
    out[..n].copy_from_slice(&y_aug[..n]);

    scratch.return_vec(av_buf);
    scratch.return_vec(w_aug);
    scratch.return_vec(y_aug);
    Ok(())
}

// ---------------------------------------------------------------------------
// phi_combination
// ---------------------------------------------------------------------------

/// `DomainViolation` helper.
fn domain_err(what: &'static str, value: f64) -> SemiflowError {
    SemiflowError::DomainViolation { what, value }
}

/// Validate `phi_combination` inputs (early returns, no allocation).
fn validate_combination<F: SemiflowFloat>(
    n: usize,
    tau: F,
    w: &[&[F]],
    out_len: usize,
) -> Result<(), SemiflowError> {
    #[allow(clippy::cast_precision_loss)]
    let as_f64 = |x: usize| x as f64;
    if w.is_empty() || w.len() > PHI_MAX + 1 {
        return Err(domain_err(
            "phi_combination: w.len() not in 1..=PHI_MAX+1",
            as_f64(w.len()),
        ));
    }
    if !tau.is_finite() || tau < F::zero() {
        return Err(domain_err(
            "phi_combination: tau < 0 or non-finite",
            tau.to_f64().unwrap_or(f64::NAN),
        ));
    }
    if out_len != n {
        return Err(domain_err(
            "phi_combination: out.len() != n",
            as_f64(out_len),
        ));
    }
    for wk in w {
        if wk.len() != n {
            return Err(domain_err(
                "phi_combination: w_k.len() != n",
                as_f64(wk.len()),
            ));
        }
        if let Some(bad) = wk.iter().find(|x| !x.is_finite()) {
            return Err(domain_err(
                "phi_combination: non-finite entry in w_k",
                bad.to_f64().unwrap_or(f64::NAN),
            ));
        }
    }
    Ok(())
}

/// Coupling columns `cols[j] = w_{p-j}`, weights `gains[j] = τ^{p-j}`, and the
/// weighted magnitude `Σ_k τ^k ‖w_k‖∞` (drives η, §62.4).  Being the SUM, it
/// bounds the max row sum of the coupling block, so `‖W_η‖∞ ≤ 1` rigorously
/// and `‖B̃_η‖∞ ≤ τ‖G‖ + 2` holds for any number of columns.
type Columns<'a, F> = ([&'a [F]; PHI_MAX], [F; PHI_MAX], f64);

/// Column `j` of the coupling block carries `τ^{p-j}·w_{p-j}` (§62.4.a).
fn weighted_columns<'a, F: SemiflowFloat>(
    w: &[&'a [F]],
    tau: F,
) -> Result<Columns<'a, F>, SemiflowError> {
    let degree = w.len() - 1;
    let mut cols: [&[F]; PHI_MAX] = [w[0]; PHI_MAX];
    let mut gains = [F::one(); PHI_MAX];
    let mut max_mag = 0.0_f64;
    for k in 1..=degree {
        cols[degree - k] = w[k];
        let w_sup = sup_norm(w[k]);
        if w_sup == 0.0 {
            gains[degree - k] = F::zero(); // zero column: never overflows
            continue;
        }
        let gain = (1..k).fold(tau, |acc, _| acc * tau);
        gains[degree - k] = gain;
        max_mag += gain.to_f64().unwrap_or(f64::NAN) * w_sup;
    }
    if !max_mag.is_finite() {
        return Err(domain_err("phi_combination: tau^k*|w_k| overflow", max_mag));
    }
    Ok((cols, gains, max_mag))
}

/// Compute `out ← Σ_{k=0}^{p} τ^k φ_k(τG) w_k`, `p = w.len() − 1 ≤ PHI_MAX` (§62.4).
///
/// ONE augmented Horner sweep (Al-Mohy–Higham 2011, Thm 2.1, η-scaled): the cost
/// equals a single [`phi_action`], not `p + 1` of them, and is independent of
/// the magnitudes of the `w_k` (see [`phi_cost_probe`]).  Covers the affine
/// evolution `u' = Gu + s` (`w = [u₀, s]`) and every ETD / Krogstad /
/// Cox–Matthews / exponential-Rosenbrock stage (§62.5).
///
/// # Errors
/// `DomainViolation` if `w` is empty or `w.len() > PHI_MAX + 1`; any
/// `w_k.len() != n` or non-finite entry; `τ < 0` or non-finite; `out.len() != n`;
/// or `τ^k·‖w_k‖∞` overflows.
pub fn phi_combination<F: SemiflowFloat, Op: GeneratorAction<F>>(
    op: &Op,
    tau: F,
    w: &[&[F]],
    out: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError> {
    let n = op.dim();
    validate_combination(n, tau, w, out.len())?;
    let degree = w.len() - 1;
    let (cols, mut gains, max_mag) = weighted_columns(w, tau)?;
    let (eta, inv_eta) = eta_scaling::<F>(max_mag);
    for gain in &mut gains[..degree] {
        *gain *= eta;
        if !gain.is_finite() {
            return Err(domain_err("phi_combination: tau^k*eta overflow", f64::NAN));
        }
    }
    let coupling = Coupling {
        cols: &cols[..degree],
        coef: &gains[..degree],
    };

    let (s, m) = sweep_params(op, tau);
    let mut y_aug = scratch.take_vec(n + degree);
    let mut w_aug = scratch.take_vec(n + degree);
    let mut av_buf = scratch.take_vec(n);
    y_aug.fill(F::zero());
    y_aug[..n].copy_from_slice(w[0]);
    if degree > 0 {
        y_aug[n + degree - 1] = inv_eta;
    }
    let result = run_sweeps(
        op,
        &coupling,
        tau,
        s,
        m,
        &mut y_aug,
        &mut w_aug,
        &mut av_buf,
    );
    out.copy_from_slice(&y_aug[..n]);
    scratch.return_vec(av_buf);
    scratch.return_vec(w_aug);
    scratch.return_vec(y_aug);
    result
}

// ---------------------------------------------------------------------------
// Initial-vector setup
// ---------------------------------------------------------------------------

/// Fill `y_aug` (size `n+p`) with the initial augmented vector for `φ_k`.
///
/// - `k = 0`: `[v; 0…0]` → `exp(B̃_η)·y_aug` top-n = `φ_0(τG)·v = e^{τG}·v`.
/// - `k ≥ 1`: `[0…0; e_{k-1}/η]` → top-n = `φ_k(τG)·v` (`inv_eta = 1/η`
///   cancels the `η` on the coupling block).
fn init_aug_vector<F: SemiflowFloat>(
    v: &[F],
    n: usize,
    p: usize,
    k: usize,
    inv_eta: F,
    y_aug: &mut [F],
) {
    // Zero everything first
    for yi in y_aug.iter_mut() {
        *yi = F::zero();
    }
    if k == 0 {
        // [v; 0 … 0]
        y_aug[..n].copy_from_slice(&v[..n]);
    } else {
        // [0; e_{k-1}/η in R^p] — 1/η at position n + (k-1)
        let idx = n + k - 1;
        if idx < n + p {
            y_aug[idx] = inv_eta;
        }
    }
}

// ---------------------------------------------------------------------------
// Dense Padé-13 oracle (gate tests only)
// ---------------------------------------------------------------------------

/// Fill the `MAX_DENSE_N×MAX_DENSE_N` matrix for the augmented φ-oracle.
///
/// Materialises τA column-by-column (top-left n×n block), sets V=v in column n,
/// and sets the J₃ superdiagonal in rows `n..n+PHI_MAX-1`.
fn fill_aug_dense_mat(
    gen: &dyn GeneratorAction<f64>,
    tau: f64,
    v: &[f64],
    n: usize,
) -> [[f64; MAX_DENSE_N]; MAX_DENSE_N] {
    let mut mat = [[0.0_f64; MAX_DENSE_N]; MAX_DENSE_N];
    let mut e_j = vec![0.0_f64; n];
    let mut col_j = vec![0.0_f64; n];
    for j in 0..n {
        e_j[j] = 1.0;
        gen.apply_generator(&e_j, &mut col_j);
        for i in 0..n {
            mat[i][j] = tau * col_j[i];
        }
        e_j[j] = 0.0;
    }
    for i in 0..n {
        mat[i][n] = v[i];
    }
    for i in n..n + PHI_MAX - 1 {
        mat[i][i + 1] = 1.0;
    }
    mat
}

/// Dense Padé-13 oracle for the `G_PHI_AUG_DENSE` gate test.
///
/// Builds the `(n+PHI_MAX)×(n+PHI_MAX)` augmented matrix
/// `Ã = [[τA, v·e₁ᵀ], [0, J₃]]` (zero-padded to `MAX_DENSE_N = 12`),
/// exponentiates via `mat_exp_pade13`, and extracts `φ_k(τA)·v`
/// for `k = 0 … PHI_MAX`.
///
/// # Errors
/// `DomainViolation` if `n + PHI_MAX > MAX_DENSE_N = 12`.
pub fn dense_phi_aug_ref(
    gen: &dyn GeneratorAction<f64>,
    tau: f64,
    v: &[f64],
) -> Result<Vec<Vec<f64>>, SemiflowError> {
    let n = gen.dim();
    let dim_aug = n + PHI_MAX;
    if dim_aug > MAX_DENSE_N {
        return Err(SemiflowError::DomainViolation {
            what: "dense_phi_aug_ref: n + PHI_MAX > MAX_DENSE_N (12)",
            #[allow(clippy::cast_precision_loss)]
            value: dim_aug as f64,
        });
    }
    let mat = fill_aug_dense_mat(gen, tau, v, n);
    let exp_mat = mat_exp_pade13::<f64, MAX_DENSE_N>(&mat)?;
    let mut out = Vec::with_capacity(PHI_MAX + 1);
    // φ_0: exp_mat[0:n, 0:n] · v.
    out.push(
        (0..n)
            .map(|i| (0..n).map(|j| exp_mat[i][j] * v[j]).sum())
            .collect(),
    );
    // φ_k (k = 1 … PHI_MAX): column n+(k−1) of exp_mat, top n rows.
    for k in 1..=PHI_MAX {
        out.push((0..n).map(|i| exp_mat[i][n + k - 1]).collect());
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Unit tests (compile-time only; slow gate is in tests/g_phi_aug_dense.rs)
// ---------------------------------------------------------------------------

#[cfg(test)]
include!("phi_action_tests_mod.rs");
