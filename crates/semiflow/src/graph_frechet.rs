//! `graph_expmv_frechet` — VJP gradient via §54.5 augmented Fréchet (ADR-0185).
//!
//! Computes `∂J/∂w_k` for `J = Σ_c ⟨dj_c, e^{−tL} u0_c⟩` using the exact
//! Duhamel integral via 8-point Gauss-Legendre quadrature on [0,1]:
//!
//! ```text
//! ∂J/∂w_k = −t ∫₀¹ ⟨e^{−(1−s)tL} dj, (∂L/∂w_k) e^{−stL} u0⟩ ds
//!          =  t ∫₀¹ ⟨a(s), (∂A/∂w_k) b(s)⟩ ds      (A = −L)
//! ```
//!
//! This is exact for ALL directions (including non-commuting `[L,∂L/∂w_k]≠0`).
//! The rectangle approximation `t⟨e^{−tL}dj, (∂A/∂w_k)u0⟩` used previously
//! was only exact when `[L,∂L/∂w_k]=0` (e.g. N=2 single-edge graph).
//! See §54.5 for the full mathematical derivation.

use alloc::sync::Arc;

use crate::{
    chernoff::ChernoffFunction,
    error::SemiflowError,
    float::{from_f64, SemiflowFloat},
    graph::Graph,
    graph_krylov::GraphKrylovChernoff,
    graph_sensitivity::GeneratorSensitivity,
    graph_signal::GraphSignal,
    scratch::ScratchPool,
};

// ---------------------------------------------------------------------------
// 8-point Gauss-Legendre nodes and weights on [0, 1]
// ---------------------------------------------------------------------------
// Nodes: s_k = (1 + x_k) / 2,  Weights: w_k = w_GL_k / 2.
// Degree-15 exactness; sum of weights = 1.0.
// Source: Abramowitz & Stegun §25.4.29.

const GL8: [(f64, f64); 8] = [
    (0.019_855_071_751_231_88, 0.050_614_268_145_188_29),
    (0.101_666_761_293_186_65, 0.111_190_517_226_687_24),
    (0.237_233_795_041_835_5, 0.156_853_322_938_943_64),
    (0.408_282_678_752_175_1, 0.181_341_891_689_180_6),
    (0.591_717_321_247_825, 0.181_341_891_689_180_6),
    (0.762_766_204_958_164_5, 0.156_853_322_938_943_64),
    (0.898_333_238_706_813_4, 0.111_190_517_226_687_24),
    (0.980_144_928_248_768_1, 0.050_614_268_145_188_29),
];

// ---------------------------------------------------------------------------
// Argument validation helper
// ---------------------------------------------------------------------------

#[allow(clippy::too_many_arguments)]
fn validate_args<F: SemiflowFloat>(
    t_final: F,
    u0_len: usize,
    dj_len: usize,
    grad_len: usize,
    n_cols: usize,
    n: usize,
    n_p: usize,
) -> Result<(), SemiflowError> {
    if !t_final.is_finite() || t_final <= F::zero() {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: t_final must be finite and positive",
            value: t_final.to_f64().unwrap_or(f64::NAN),
        });
    }
    if u0_len != n_cols * n {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: u0_cols length != n_cols * n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: u0_len as f64,
        });
    }
    if dj_len != n_cols * n {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: dj_cols length != n_cols * n_nodes",
            #[allow(clippy::cast_precision_loss)]
            value: dj_len as f64,
        });
    }
    if grad_len != n_p {
        return Err(SemiflowError::DomainViolation {
            what: "graph_expmv_frechet: grad_w.len() != n_params",
            #[allow(clippy::cast_precision_loss)]
            value: grad_len as f64,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Quadrature accumulation for one GL node
// ---------------------------------------------------------------------------

/// Accumulate gradient contribution from one GL node `(s, w_q)`.
///
/// `a_buf = e^{−(1−s)t L} dj_c`,  `b_buf = e^{−s t L} u0_c`.
/// Adds `t * w_q * ⟨a_buf, (∂A/∂θ_k) b_buf⟩` to `grad_w[k]` for all k.
#[allow(clippy::too_many_arguments)] // 7 scalar/slice physics args; no natural grouping
fn accumulate_gl_node<F, P>(
    t: F,
    wq: F,
    a_buf: &[F],
    b_buf: &[F],
    param_deriv: &P,
    grad_w: &mut [F],
    tmp: &mut [F],
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    for (k, gk) in grad_w.iter_mut().enumerate() {
        // tmp = (∂A/∂w_k) b_buf  (= −E_k b_buf for edge-weight sensitivity)
        param_deriv.apply_param_deriv(k, t, b_buf, tmp)?;
        let dot = a_buf.iter().zip(tmp.iter()).fold(F::zero(), |acc, (&a, &b)| acc + a * b);
        *gk += t * wq * dot;
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Single-channel Duhamel integral via GL8 quadrature
// ---------------------------------------------------------------------------

/// Exact VJP for one channel: accumulates into `grad_w` (not zeroed here).
///
/// For each GL8 node `s_q`: computes `a(s_q) = e^{-(1-s_q)t L} dj_c` and
/// `b(s_q) = e^{-s_q t L} u0_c`, then calls `accumulate_gl_node`.
#[allow(clippy::too_many_arguments)] // 8 physics args: solver, signal pair, graph, time, sens, grad, scratch
fn frechet_channel<F, P>(
    gk: &GraphKrylovChernoff<F>,
    u0_c: &[F],
    dj_c: &[F],
    g_arc: &Arc<Graph<F>>,
    t: F,
    param_deriv: &P,
    grad_w: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = u0_c.len();
    let mut a_buf = scratch.take_vec(n);
    let mut b_buf = scratch.take_vec(n);
    let mut tmp = scratch.take_vec(n);

    for &(s_f64, wq_f64) in &GL8 {
        let s = from_f64::<F>(s_f64);
        let wq = from_f64::<F>(wq_f64);

        // a(s) = e^{-(1-s)*t*L} dj_c
        let src_a = GraphSignal::from_fn(Arc::clone(g_arc), |i| dj_c[i as usize]);
        let mut dst_a = GraphSignal::zeros(Arc::clone(g_arc));
        gk.apply_into((F::one() - s) * t, &src_a, &mut dst_a, scratch)?;
        a_buf.copy_from_slice(dst_a.values());

        // b(s) = e^{-s*t*L} u0_c
        let src_b = GraphSignal::from_fn(Arc::clone(g_arc), |i| u0_c[i as usize]);
        let mut dst_b = GraphSignal::zeros(Arc::clone(g_arc));
        gk.apply_into(s * t, &src_b, &mut dst_b, scratch)?;
        b_buf.copy_from_slice(dst_b.values());

        accumulate_gl_node(t, wq, &a_buf, &b_buf, param_deriv, grad_w, &mut tmp)?;
    }
    scratch.return_vec(tmp);
    scratch.return_vec(b_buf);
    scratch.return_vec(a_buf);
    Ok(())
}

// ---------------------------------------------------------------------------
// Public API
// ---------------------------------------------------------------------------

/// VJP gradient `∂J/∂θ` for `J = Σ_c ⟨dj_c, e^{−t L} u0_c⟩` (§54.5, ADR-0185).
///
/// Uses 8-point Gauss-Legendre quadrature of the Duhamel integral:
/// `∂J/∂θ_k = t ∫₀¹ ⟨e^{−(1−s)tL}dj_c, (∂A/∂θ_k) e^{−stL}u0_c⟩ ds`.
/// Exact for ALL graph topologies, including non-commuting directions.
///
/// # Arguments
///
/// * `gk` — Krylov solver owning the fixed Laplacian `L` (A1 primitive).
/// * `u0_cols` — flat row-major initial states; slice `c·n … (c+1)·n` = `u0_c`.
/// * `dj_cols` — flat row-major loss-gradient vectors; same layout as `u0_cols`.
/// * `n_cols` — number of channels (D4 ascending sweep).
/// * `t_final` — evolution time `t > 0`.
/// * `param_deriv` — `(∂A/∂θ_k) v` provider (`A = −L`, generator sign).
/// * `grad_w` — output slice of length `n_params`; zeroed then accumulated.
/// * `scratch` — reusable buffer pool.
///
/// # Panics
///
/// Never panics in practice: the internal `.expect("empty graph never fails")`
/// guard constructs a zero-edge domain graph and `Graph::from_edges` only
/// errors on malformed edge data, which cannot occur here.
///
/// # Errors
///
/// Returns `DomainViolation` if `t_final ≤ 0`, any length mismatch, or the
/// inner Krylov solve fails.
#[allow(clippy::too_many_arguments)]
pub fn graph_expmv_frechet<F, P>(
    gk: &GraphKrylovChernoff<F>,
    u0_cols: &[F],
    dj_cols: &[F],
    n_cols: usize,
    t_final: F,
    param_deriv: &P,
    grad_w: &mut [F],
    scratch: &mut ScratchPool<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat,
    P: GeneratorSensitivity<F>,
{
    let n = gk.n_nodes();
    let n_p = param_deriv.n_params();
    validate_args(t_final, u0_cols.len(), dj_cols.len(), grad_w.len(), n_cols, n, n_p)?;

    // D4: zero accumulator before ascending-channel sweep (ADR-0185 §D4).
    for g in grad_w.iter_mut() {
        *g = F::zero();
    }

    // Dummy graph: `n` nodes, no edges.  Used only as the GraphSignal domain.
    let g_dummy = Arc::new(
        Graph::<F>::from_edges(n, core::iter::empty()).expect("empty graph never fails"),
    );

    // D4: ascending channel order.
    for c in 0..n_cols {
        let u0_c = &u0_cols[c * n..(c + 1) * n];
        let dj_c = &dj_cols[c * n..(c + 1) * n];
        frechet_channel(gk, u0_c, dj_c, &g_dummy, t_final, param_deriv, grad_w, scratch)?;
    }
    Ok(())
}
