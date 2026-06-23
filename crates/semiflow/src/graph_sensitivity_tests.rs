//! Unit tests for `graph_sensitivity.rs`.
//!
//! Moved here to keep `graph_sensitivity.rs` within the 500-line suckless limit.

#[cfg(test)]
mod tests {
    use alloc::sync::Arc;
    use alloc::vec;

    use crate::{
        graph::{Graph, Laplacian},
        graph_sensitivity::{
            adjoint_state_gradient, apply_edge_weight_deriv, magnus_step_jvp_into,
            EdgeWeightSensitivity, GeneratorSensitivity, NodeTimescaleSensitivity,
        },
        graph_signal::GraphSignal,
        scratch::ScratchPool,
        state::State,
    };

    #[test]
    fn rank1_stencil_edge() {
        let n = 3;
        let v = vec![1.0_f64, 2.0, 3.0];
        let mut out = vec![0.0_f64; n];
        apply_edge_weight_deriv(0, 1, &v, &mut out).unwrap();
        let diff = v[0] - v[1]; // = -1.0
        assert!((out[0] - diff).abs() < 1e-14, "out[0]={}", out[0]);
        assert!((out[1] + diff).abs() < 1e-14, "out[1]={}", out[1]);
        assert!(out[2].abs() < 1e-14, "out[2]={}", out[2]);
        assert!(apply_edge_weight_deriv(1, 1, &v, &mut out).is_err());
        assert!(apply_edge_weight_deriv(0, 5, &v, &mut out).is_err());
    }

    #[test]
    fn node_timescale_finite() {
        let n = 3;
        let g = Arc::new(Graph::<f64>::path(n));
        let bare_lap = Laplacian::assemble_combinatorial(&g);
        let sqrt_a = vec![1.0_f64, 2.0_f64.sqrt(), 0.5_f64.sqrt()];
        let sens = NodeTimescaleSensitivity { sqrt_a, bare_lap };
        let v = vec![1.0_f64, -0.5, 0.3];
        let mut out = vec![0.0_f64; n];
        for k in 0..n {
            sens.apply_param_deriv(k, 0.0, &v, &mut out).unwrap();
            assert!(out.iter().all(|x| x.is_finite()), "node {k}: {out:?}");
        }
    }

    #[test]
    fn adjoint_grad_finite() {
        use crate::magnus_graph::{LaplacianAtTime, MagnusGraphHeatChernoff};
        let n = 3;
        let g = Arc::new(Graph::<f64>::path(n));
        let g2 = Arc::clone(&g);
        let lap_fn: LaplacianAtTime<f64> =
            alloc::boxed::Box::new(move |_t| Arc::new(Laplacian::assemble_combinatorial(&g2)));
        let mc = MagnusGraphHeatChernoff::new(Arc::clone(&g), lap_fn, 3.0, true).unwrap();
        let u0 = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.3 + 0.1);
        let tgt = GraphSignal::from_fn(Arc::clone(&g), |i| f64::from(i) * 0.1 - 0.2);
        let n_steps = 2;
        let tau = 0.02_f64;
        let mut scratch = ScratchPool::new();
        let mut cur = u0.clone();
        let mut nxt = GraphSignal::zeros(Arc::clone(&g));
        for k in 0..n_steps {
            #[allow(clippy::cast_precision_loss)]
            mc.apply_into_at(k as f64 * tau, tau, &cur, &mut nxt, &mut scratch)
                .unwrap();
            core::mem::swap(&mut cur, &mut nxt);
        }
        let mut dj = cur;
        State::<f64>::axpy_into(&mut dj, -1.0, &tgt);
        let sens = EdgeWeightSensitivity {
            params: vec![(0, 1), (1, 2)],
            n_nodes: n,
        };
        let mut grad = vec![0.0_f64; 2];
        adjoint_state_gradient(&mc, &u0, n_steps, tau, &dj, &sens, &mut grad, &mut scratch)
            .unwrap();
        assert!(grad.iter().all(|x| x.is_finite()), "gradient: {grad:?}");
    }

    #[test]
    fn jvp_into_finite() {
        let n = 3;
        let g = Arc::new(Graph::<f64>::path(n));
        let lap = Laplacian::assemble_combinatorial(&g);
        let u = vec![1.0_f64, 0.5, -0.3];
        let mut out = vec![0.0_f64; n];
        let mut scratch = ScratchPool::new();
        magnus_step_jvp_into(&lap, &lap, &lap, &lap, 0.01, &u, &mut out, &mut scratch).unwrap();
        assert!(out.iter().all(|x| x.is_finite()), "jvp: {out:?}");
    }
}
