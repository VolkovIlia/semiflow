//! Stiff kinetics, node by node (`RELEASE_BLOCKING`, ADR-0212, math §69).
//!
//! `G_STIFF_KINETICS_PER_NODE`: `u′ = C(x)u` with a NODE-DEPENDENT `2×2` matrix
//! `C_i = R_i·diag(λ_i, −1)·R_iᵀ`, `λ_i = −10^{6x_i}` (`−1, −10², −10⁴, −10⁶` on the
//! four nodes: node 0 non-stiff, node 3 stiff), `R_i` the rotation by
//! `θ_i = 0.2 + 1.2x_i` (partial pivoting swaps rows at nodes 2–3 only), from
//! non-uniform data `u₀ = (1 + 3x, 0.3 − 2x)` (the difference steps `δ_i` differ per
//! node), `T = 1`. Adaptive ROS2 with the analytic and the forward-difference
//! Jacobian at `rtol ∈ {1e−4, 1e−6, 1e−8}` (`atol = 1e−14`) must meet `≤ 10·rtol`
//! relative at EVERY node against the exact `R_i·diag(e^{λ_iT}, e^{−T})·R_iᵀu₀_i`.
//! A per-node indexing bug (node 0's `J`, `LU`, pivots or `δ` used for all nodes)
//! fails it; the uniform fields of `tests/g_stiff_kinetics.rs` cannot see one.

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

mod stiff_common;

use semiflow::{Kinetics, SemiflowError};
use stiff_common::{evolve_nodes, ros, NODES};

/// Node `i`'s `(λ_i, cos θ_i, sin θ_i)` from its coordinate.
fn node_params(x: f64) -> (f64, f64, f64) {
    let theta = 0.2 + 1.2 * x;
    (-(10f64.powf(6.0 * x)), theta.cos(), theta.sin())
}

/// `C_i` row-major.
fn node_matrix(x: f64) -> [f64; 4] {
    let (l, c, s) = node_params(x);
    let off = (l + 1.0) * c * s;
    [l * c * c - s * s, off, off, l * s * s - c * c]
}

/// `u′ = C(x)u`; analytic Jacobian unless `fd`.
struct StiffLinearX {
    fd: bool,
}

impl Kinetics<f64> for StiffLinearX {
    fn species(&self) -> usize {
        2
    }
    fn eval_batch(
        &self,
        _t: f64,
        n: usize,
        x: &[f64],
        u: &[f64],
        du: &mut [f64],
    ) -> Result<(), SemiflowError> {
        for i in 0..n {
            let m = node_matrix(x[i]);
            du[i] = m[0] * u[i] + m[1] * u[n + i];
            du[n + i] = m[2] * u[i] + m[3] * u[n + i];
        }
        Ok(())
    }
    fn jacobian_batch(
        &self,
        _t: f64,
        n: usize,
        x: &[f64],
        _u: &[f64],
        jac: &mut [f64],
    ) -> Option<Result<(), SemiflowError>> {
        if self.fd {
            return None;
        }
        for i in 0..n {
            for (e, v) in node_matrix(x[i]).into_iter().enumerate() {
                jac[e * n + i] = v;
            }
        }
        Some(Ok(()))
    }
}

/// `R·diag(e^{λT}, e^{−T})·Rᵀu₀` at one node.
fn node_exact(x: f64, u0: [f64; 2], t: f64) -> [f64; 2] {
    let (l, c, s) = node_params(x);
    let (a, b) = (c * u0[0] + s * u0[1], -s * u0[0] + c * u0[1]);
    let (a, b) = ((l * t).exp() * a, (-t).exp() * b);
    [c * a - s * b, s * a + c * b]
}

#[test]
fn g_stiff_kinetics_per_node() {
    let xs: Vec<f64> = (0..NODES).map(|i| i as f64 / (NODES - 1) as f64).collect();
    let u0 = vec![
        xs.iter().map(|x| 1.0 + 3.0 * x).collect::<Vec<f64>>(),
        xs.iter().map(|x| 0.3 - 2.0 * x).collect(),
    ];
    for fd in [false, true] {
        for rtol in [1e-4, 1e-6, 1e-8] {
            let kin = Box::new(StiffLinearX { fd });
            let got = evolve_nodes(kin, ros(rtol, 1e-14), 1, &u0, (1.0, 1)).expect("ros2");
            let worst = xs.iter().enumerate().fold(0.0_f64, |m, (i, &x)| {
                let want = node_exact(x, [u0[0][i], u0[1][i]], 1.0);
                let err = (got[0][i] - want[0]).abs().max((got[1][i] - want[1]).abs());
                m.max(err / want[0].abs().max(want[1].abs()))
            });
            eprintln!(
                "G_STIFF_KINETICS_PER_NODE fd={fd} rtol={rtol:e}: worst node {:.3}·rtol",
                worst / rtol
            );
            assert!(worst <= 10.0 * rtol, "fd={fd} rtol={rtol:e}: {worst:e}");
        }
    }
}
