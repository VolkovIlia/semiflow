//! Shared step of the frozen-coefficient N-D shift engines, with a separable
//! whole-array path for constant coefficients (ADR-0214).
//!
//! `SmolyakGridND` and `AnisotropicShiftChernoffND` both compute, per node `k`,
//! `exp(τc(x_k))·π^{−D/2}·Σ_q w_q f(y_q(k))`, `y_q(k) = x_k + τb(x_k) + 2√τ·L_k η_q`,
//! where `f(y)` is the tensor-product sample of [`GridFnND::sample`]: a recursion
//! that collapses axis 0 innermost, axis `D − 1` outermost, `K^D` node reads.
//!
//! When `b` and `L` are the same (bitwise) at every node, axis `d` of `y_q(k)`
//! depends only on `(q, k_d)`, so the stencil of axis `d` depends only on `k_d`.
//! The recursion then equals `D` whole-array passes — axis 0 first, each output
//! entry `Σ_j w_j·bc(hit_j, i ↦ previous pass at axis index i)` with
//! `acc = 0; acc += w_j·v_j` in `j` order, exactly the arithmetic of
//! `GridFnND::collapse` (a boundary hit that reads several nodes reads the stored
//! previous-pass values the recursion would recompute). Same operations, same
//! order: bit-identical for every non-NaN output (NaN outputs coincide as a class;
//! their payload is unspecified), at `D·K` instead of `K^D` reads per node and `q`.
//!
//! Consecutive `q` whose axis offsets agree on axes `0..=d` (bitwise) share the
//! first `d + 1` passes: Smolyak nodes are sorted axis 0 first, so the shared
//! prefixes cut the passes ≈ 2.6× further at `D = 6`. The outer sum over `q`
//! still runs in stored order per node.

use alloc::vec::Vec;
use core::any::Any;

use crate::{
    boundary::bc_value_from_hit,
    error::SemiflowError,
    float::SemiflowFloat,
    grid_nd::{GridFnND, GridND},
    interp_stencil::supports_nd,
    scratch::ScratchPool,
    shift_nd::{flat_to_x, SquareMatrix},
};

/// Largest scratch (in `F` elements) the separable path may hold: prefix
/// sharing needs `(D + 1)·N`, the plain per-`q` passes `3·N`.
const SWEEP_MAX_ELEMS: usize = 1 << 24;

/// Drift closure `b(x)`.
pub(crate) type DriftRef<'a, F, const D: usize> = &'a (dyn Fn(&[F; D], &mut [F; D]) + Send + Sync);
/// Reaction closure `c(x)`.
pub(crate) type ReactionRef<'a, F, const D: usize> = &'a (dyn Fn(&[F; D]) -> F + Send + Sync);

/// Everything one shift step needs from the engine.
pub(crate) struct ShiftStep<'a, F: SemiflowFloat, const D: usize> {
    /// Step size `τ`.
    pub(crate) tau: F,
    /// The engine's grid (node coordinates `x_k`).
    pub(crate) grid: &'a GridND<F, D>,
    /// Drift `b(x)`.
    pub(crate) b: DriftRef<'a, F, D>,
    /// Reaction `c(x)`.
    pub(crate) c: ReactionRef<'a, F, D>,
    /// Per-node Cholesky factors `L_k`.
    pub(crate) chol: &'a [SquareMatrix<F, D>],
    /// Quadrature `(η_q, w_q)` in the engine's stored order and exact conversion.
    pub(crate) quad: &'a [([F; D], F)],
}

/// `dst = S(τ) src` for a shift engine: the separable path when it applies, the
/// per-node loop otherwise. Both give the same bits (NaN outputs: same class).
pub(crate) fn apply_shift_step<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
    dst: &mut GridFnND<F, D>,
    scratch: &mut ScratchPool<F>,
) {
    if !try_separable(step, src, dst, scratch, SWEEP_MAX_ELEMS) {
        per_node(step, src, dst);
    }
}

/// `2√τ`, the node scale of ADR-0112.
fn two_sqrt_tau<F: SemiflowFloat>(tau: F) -> F {
    crate::float::from_f64::<F>(2.0_f64) * tau.sqrt()
}

/// Output of node `flat`: today's per-node formula, verbatim.
fn node_value<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
    ns: &[usize; D],
    flat: usize,
) -> F {
    let tau = step.tau;
    let two_sqrt_tau = two_sqrt_tau(tau);
    let inv_pi_dhalf = crate::float::inv_pi_pow_half::<F>(D);
    let xk = flat_to_x::<F, D>(flat, ns, step.grid);
    let mut b_val = [F::zero(); D];
    (step.b)(&xk, &mut b_val);
    let c_val = (step.c)(&xk);
    let exp_factor = (tau * c_val).libm_exp();
    let l_k = &step.chol[flat];
    let mut acc = F::zero();
    for &(eta_q, w_q) in step.quad {
        let mut l_eta = [F::zero(); D];
        l_k.mat_vec(&eta_q, &mut l_eta);
        let y_q: [F; D] =
            core::array::from_fn(|d| xk[d] + tau * b_val[d] + two_sqrt_tau * l_eta[d]);
        // CRITICAL: weight may be negative — do NOT clamp or abs.
        acc += w_q * src.sample(&y_q).unwrap_or(F::zero());
    }
    exp_factor * inv_pi_dhalf * acc
}

/// The per-node loop (variable coefficients). Nodes are independent, so for
/// `F = f64` they are evaluated in parallel under the `parallel` feature.
fn per_node<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
    dst: &mut GridFnND<F, D>,
) {
    let ns: [usize; D] = core::array::from_fn(|d| step.grid.axes[d].n);
    let total = src.values.len();
    if let Some(out) = (&mut dst.values as &mut dyn Any).downcast_mut::<Vec<f64>>() {
        let eval = |flat: usize| {
            Ok(node_value(step, src, &ns, flat)
                .to_f64()
                .unwrap_or(f64::NAN))
        };
        let res: Result<(), SemiflowError> =
            crate::parallel1d::parallel_eval_into(&mut out[..total], eval);
        debug_assert!(res.is_ok(), "node_value is infallible");
        return;
    }
    for flat in 0..total {
        dst.values[flat] = node_value(step, src, &ns, flat);
    }
}

/// Bitwise equality for the primitive floats; `false` (path disabled) otherwise.
fn same_bits<F: SemiflowFloat>(a: &F, b: &F) -> bool {
    let (a, b): (&dyn Any, &dyn Any) = (a, b);
    match (a.downcast_ref::<f64>(), b.downcast_ref::<f64>()) {
        (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
        _ => match (a.downcast_ref::<f32>(), b.downcast_ref::<f32>()) {
            (Some(x), Some(y)) => x.to_bits() == y.to_bits(),
            _ => false,
        },
    }
}

/// `b(x_k)` and `L_k` are bitwise the same at every node; returns `b`.
fn constant_coefficients<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    ns: &[usize; D],
    total: usize,
) -> Option<[F; D]> {
    let probe = |flat: usize| {
        let mut b = [F::zero(); D];
        (step.b)(&flat_to_x::<F, D>(flat, ns, step.grid), &mut b);
        b
    };
    if !same_bits(&F::zero(), &F::zero()) {
        return None; // not a primitive float (e.g. `Dual`): no bitwise test available
    }
    let (b0, l0) = (probe(0), step.chol.first()?);
    let same_l = |l: &SquareMatrix<F, D>| {
        (0..D).all(|i| (0..D).all(|j| same_bits(&l.get(i, j), &l0.get(i, j))))
    };
    let all = (1..total).all(|flat| {
        let b = probe(flat);
        (0..D).all(|d| same_bits(&b[d], &b0[d])) && same_l(&step.chol[flat])
    });
    all.then_some(b0)
}

/// Shapes agree, the interpolant has an N-D stencil, coefficients are constant.
fn separable_inputs<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
    dst: &GridFnND<F, D>,
) -> Option<[F; D]> {
    let ns: [usize; D] = core::array::from_fn(|d| step.grid.axes[d].n);
    let total = src.values.len();
    let shapes = D > 0
        && supports_nd(src.grid.interp)
        && (0..D).all(|d| src.grid.axes[d].n == ns[d])
        && ns.iter().product::<usize>() == total
        && total > 0
        && dst.values.len() >= total
        && step.chol.len() >= total;
    if !shapes {
        return None;
    }
    constant_coefficients(step, &ns, total)
}

/// The separable path within `cap` scratch elements; `false` when it does not
/// apply (nothing written).
fn try_separable<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    src: &GridFnND<F, D>,
    dst: &mut GridFnND<F, D>,
    scratch: &mut ScratchPool<F>,
    cap: usize,
) -> bool {
    let Some(b0) = separable_inputs(step, src, dst) else {
        return false;
    };
    let total = src.values.len();
    let shared = (D + 1).saturating_mul(total) <= cap;
    if !shared && 3 * total > cap {
        return false;
    }
    // Every buffer comes from the pool: steady-state calls allocate nothing.
    let mut levels: [Vec<F>; D] = core::array::from_fn(|d| {
        if shared || d < 2 {
            scratch.take_vec(total)
        } else {
            Vec::new()
        }
    });
    let mut acc = scratch.take_vec(total);
    let (base, offsets) = axis_bases(step, b0, scratch);
    let mut sweep = Sweep {
        src,
        levels: &mut levels,
        shared,
        base: &base,
        offsets,
    };
    let done = sweep.accumulate(step, &mut acc);
    if done {
        finish(step, &acc, dst);
    }
    for buf in levels.into_iter().filter(|b| b.capacity() > 0) {
        scratch.return_vec(buf);
    }
    scratch.return_vec(acc);
    scratch.return_vec(base);
    done
}

/// `x_k[d] + τ·b[d]` per axis node, axes concatenated (the first addend of `y`),
/// and the start of each axis in that buffer.
fn axis_bases<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    b0: [F; D],
    scratch: &mut ScratchPool<F>,
) -> (Vec<F>, [usize; D]) {
    let mut base = scratch.take_vec(step.grid.axes.iter().map(|ax| ax.n).sum());
    let mut offsets = [0_usize; D];
    let mut at = 0;
    for (d, ax) in step.grid.axes.iter().enumerate() {
        offsets[d] = at;
        for k in 0..ax.n {
            base[at + k] = step.grid.x_at(d, k) + step.tau * b0[d];
        }
        at += ax.n;
    }
    (base, offsets)
}

/// `dst[k] = exp(τc(x_k))·π^{−D/2}·acc[k]`, the per-node tail.
fn finish<F: SemiflowFloat, const D: usize>(
    step: &ShiftStep<'_, F, D>,
    acc: &[F],
    dst: &mut GridFnND<F, D>,
) {
    let ns: [usize; D] = core::array::from_fn(|d| step.grid.axes[d].n);
    let inv_pi_dhalf = crate::float::inv_pi_pow_half::<F>(D);
    for (flat, &a) in acc.iter().enumerate() {
        let c_val = (step.c)(&flat_to_x::<F, D>(flat, &ns, step.grid));
        let exp_factor = (step.tau * c_val).libm_exp();
        dst.values[flat] = exp_factor * inv_pi_dhalf * a;
    }
}

/// Pass buffers of the separable path.
struct Sweep<'s, F: SemiflowFloat, const D: usize> {
    src: &'s GridFnND<F, D>,
    /// `D` pass outputs (prefix sharing) or two ping-pong buffers (slots 0, 1).
    levels: &'s mut [Vec<F>; D],
    shared: bool,
    /// `x_k[d] + τ·b[d]`, axis `d` at `base[offsets[d]..offsets[d] + n_d]`.
    base: &'s [F],
    offsets: [usize; D],
}

impl<F: SemiflowFloat, const D: usize> Sweep<'_, F, D> {
    /// `acc[k] = Σ_q w_q·f(y_q(k))` in stored `q` order; `false` if a stencil
    /// could not be built (then nothing was written to `dst`).
    fn accumulate(&mut self, step: &ShiftStep<'_, F, D>, acc: &mut [F]) -> bool {
        let two_sqrt_tau = two_sqrt_tau(step.tau);
        let mut prev: Option<[F; D]> = None;
        for &(eta_q, w_q) in step.quad {
            let mut l_eta = [F::zero(); D];
            step.chol[0].mat_vec(&eta_q, &mut l_eta);
            let key: [F; D] = core::array::from_fn(|d| two_sqrt_tau * l_eta[d]);
            let first = match (self.shared, prev) {
                (true, Some(p)) => (0..D).find(|&d| !same_bits(&p[d], &key[d])).unwrap_or(D),
                _ => 0,
            };
            for (d, &off) in key.iter().enumerate().skip(first) {
                if !self.pass(d, off) {
                    return false;
                }
            }
            prev = Some(key);
            let last = &self.levels[self.slot(D - 1)];
            for (a, &g) in acc.iter_mut().zip(last.iter()) {
                *a += w_q * g;
            }
        }
        true
    }

    fn slot(&self, d: usize) -> usize {
        if self.shared {
            d
        } else {
            d % 2
        }
    }

    /// Pass `d` at axis offset `off`.
    fn pass(&mut self, d: usize, off: F) -> bool {
        let n = self.src.grid.axes[d].n;
        let xs = &self.base[self.offsets[d]..self.offsets[d] + n];
        let out_slot = self.slot(d);
        let (input, out): (&[F], &mut [F]) = if d == 0 {
            (&self.src.values, &mut self.levels[out_slot])
        } else {
            let in_slot = self.slot(d - 1);
            let (lo, hi) = self.levels.split_at_mut(in_slot.max(out_slot));
            if in_slot < out_slot {
                (&lo[in_slot], &mut hi[0])
            } else {
                (&hi[0], &mut lo[out_slot])
            }
        };
        axis_pass(self.src, d, xs, off, input, out)
    }
}

/// One axis of `GridFnND::collapse` over the whole array: for each axis node `k`
/// the stencil of `y = (x_k + τb) + off` (the engine's own association), then every
/// output entry with axis-`d` index `k`. `false` if a stencil cannot be built
/// (impossible after `supports_nd`; the caller then runs the per-node loop).
fn axis_pass<F: SemiflowFloat, const D: usize>(
    src: &GridFnND<F, D>,
    d: usize,
    xs: &[F],
    off: F,
    input: &[F],
    out: &mut [F],
) -> bool {
    let ax = &src.grid.axes[d];
    let n = ax.n;
    for (k, &x) in xs.iter().enumerate() {
        let Ok(st) = src.axis_stencil(d, x + off) else {
            return false;
        };
        let block = st.stride * n;
        for hi in (0..out.len()).step_by(block) {
            for lo in 0..st.stride {
                let base = hi + lo;
                let mut acc = F::zero();
                for j in 0..st.k {
                    let get = |i: usize| input[base + i * st.stride];
                    let v = bc_value_from_hit(st.hits[j], ax.boundary, get, n, ax.dx());
                    acc += st.weights[j] * v;
                }
                out[base + k * st.stride] = acc;
            }
        }
    }
    true
}

#[cfg(test)]
#[path = "grid_nd_sweep_tests.rs"]
mod tests;

#[cfg(test)]
#[path = "grid_nd_sweep_engine_tests.rs"]
mod engine_tests;
