//! `G_ETDRK4_SWEEPS` (`RELEASE_BLOCKING`, ADR-0205): one ETDRK4 step costs exactly
//! four augmented φ sweeps — three at `τ = h/2` (stages a, b, c) and one at
//! `τ = h` (the update) — counted as generator applications, `3·C(h/2) + C(h)`
//! with `C(τ) = s·m` of `phi_cost_probe`. The former assembly (one φ per call)
//! made nine sweeps, `5·C(h/2) + 4·C(h)`; the gate also asserts that saving is
//! at least 2.2× at every step size.

use std::sync::{
    atomic::{AtomicUsize, Ordering},
    Arc,
};

use semiflow::{
    generator_action::GeneratorAction, phi_cost_probe, scratch::ScratchPool, AllenCahn, Etdrk4,
};

const N: usize = 32;

/// Periodic `c·(u_{i−1} − 2u_i + u_{i+1})` that counts its applications.
struct Counting {
    c: f64,
    calls: Arc<AtomicUsize>,
}

impl GeneratorAction<f64> for Counting {
    fn dim(&self) -> usize {
        N
    }
    fn apply_generator(&self, src: &[f64], dst: &mut [f64]) {
        self.calls.fetch_add(1, Ordering::Relaxed);
        for i in 0..N {
            let (im, ip) = ((i + N - 1) % N, (i + 1) % N);
            dst[i] = self.c * (src[im] - 2.0 * src[i] + src[ip]);
        }
    }
    fn norm_bound(&self) -> f64 {
        4.0 * self.c
    }
}

fn cost(norm: f64, tau: f64) -> usize {
    let (s, m) = phi_cost_probe(norm, tau);
    assert!(s >= 1 && m >= 1, "no schedule at norm={norm}, tau={tau}");
    (s * m) as usize
}

#[test]
fn g_etdrk4_sweeps() {
    #[allow(clippy::cast_precision_loss)]
    let u0: Vec<f64> = (0..N).map(|i| (0.3 * i as f64).sin()).collect();
    for (c, h) in [(1.0, 0.01), (10.0, 0.05), (100.0, 0.1), (1e4, 0.2)] {
        let calls = Arc::new(AtomicUsize::new(0));
        let op = Counting {
            c,
            calls: Arc::clone(&calls),
        };
        let norm = op.norm_bound();
        let driver = Etdrk4::new(op, AllenCahn::<f64>::new(), h).expect("driver");
        let mut out = vec![0.0; N];
        driver
            .step(&u0, &mut out, &mut ScratchPool::new())
            .expect("step");
        assert!(out.iter().all(|x| x.is_finite()), "c={c} h={h}: non-finite");
        let calls = calls.load(Ordering::Relaxed);
        let (half, full) = (cost(norm, 0.5 * h), cost(norm, h));
        let expected = 3 * half + full;
        let legacy = 5 * half + 4 * full;
        eprintln!("G_ETDRK4_SWEEPS c={c} h={h}: calls={calls} expected={expected} legacy={legacy}");
        assert_eq!(calls, expected, "c={c} h={h}: not four sweeps");
        assert!(
            10 * legacy >= 22 * expected,
            "c={c} h={h}: saving below 2.2×"
        );
    }
}
