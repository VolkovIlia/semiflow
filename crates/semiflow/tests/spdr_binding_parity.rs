//! Binding-parity pin (ADR-0202 Wave 3): the Python `SpdResolvent.solve` must be
//! bitwise equal to the core result. This test runs the same fixed inputs through
//! the core and pins the FNV-1a 64 digest of the output bits. The Python test
//! `crates/semiflow-py/tests/test_spd_resolvent.py::test_binding_parity` rebuilds the
//! identical inputs (all dyadic rationals, so no libm is involved) and pins the same
//! constants. Change one side and the other fails.

// Small index -> f64 conversions on tiny, exactly representable inputs.
#![allow(clippy::cast_precision_loss)]

use semiflow::{scratch::ScratchPool, Precond, SpdSolver, SymmetricOperator};

/// FNV-1a 64 over the little-endian bytes of each `f64` (mirrors the nostd-check `Digest`).
fn fnv(xs: &[f64]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    for x in xs {
        for byte in x.to_bits().to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    h
}

/// Path-graph Neumann Laplacian, conductance `1 + 0.25·(i mod 5)`, n = 32.
fn path_csr(n: usize) -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let cond = |i: usize| 1.0 + 0.25 * (i % 5) as f64;
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for i in 0..n {
        let left = if i > 0 { cond(i - 1) } else { 0.0 };
        let right = if i + 1 < n { cond(i) } else { 0.0 };
        if i > 0 {
            ci.push(u32::try_from(i - 1).unwrap());
            va.push(-left);
        }
        ci.push(u32::try_from(i).unwrap());
        va.push(left + right);
        if i + 1 < n {
            ci.push(u32::try_from(i + 1).unwrap());
            va.push(-right);
        }
        rp.push(ci.len());
    }
    (rp, ci, va)
}

/// 6×6 five-point Neumann Laplacian, row-major, n = 36.
fn grid_csr() -> (Vec<usize>, Vec<u32>, Vec<f64>) {
    let side = 6_usize;
    let (mut rp, mut ci, mut va) = (vec![0_usize], Vec::new(), Vec::new());
    for r in 0..side {
        for c in 0..side {
            let i = r * side + c;
            let mut nb = Vec::new();
            if r > 0 {
                nb.push(i - side);
            }
            if c > 0 {
                nb.push(i - 1);
            }
            if c + 1 < side {
                nb.push(i + 1);
            }
            if r + 1 < side {
                nb.push(i + side);
            }
            let mut row: Vec<(usize, f64)> = nb.iter().map(|&j| (j, -1.0)).collect();
            row.push((i, nb.len() as f64));
            row.sort_by_key(|&(j, _)| j);
            for (j, v) in row {
                ci.push(u32::try_from(j).unwrap());
                va.push(v);
            }
            rp.push(ci.len());
        }
    }
    (rp, ci, va)
}

fn rhs(n: usize) -> Vec<f64> {
    (0..n).map(|i| ((i * 7) % 11) as f64 / 4.0 - 1.25).collect()
}

fn digest_tridiag() -> u64 {
    let n = 32;
    let (rp, ci, va) = path_csr(n);
    let op = SymmetricOperator::from_csr(n, &rp, &ci, &va, 1e-10).unwrap();
    let c: Vec<f64> = (0..n).map(|i| 0.5 + 0.125 * (i % 3) as f64).collect();
    let op = op.with_diagonal(&c).unwrap();
    let mass: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i % 4) as f64).collect();
    let r = op
        .resolvent(0.5, Some(&mass), SpdSolver::Auto, 1e-12)
        .unwrap();
    let mut x = vec![0.0; n];
    r.solve_into(&rhs(n), &mut x, &mut ScratchPool::new())
        .unwrap();
    fnv(&x)
}

fn digest_pcg() -> u64 {
    let (rp, ci, va) = grid_csr();
    let op = SymmetricOperator::from_csr(36, &rp, &ci, &va, 1e-10).unwrap();
    let op = op.with_diagonal(&[0.25; 36]).unwrap();
    let solver = SpdSolver::Pcg {
        precond: Precond::Ic0,
        max_iter: None,
    };
    let r = op.resolvent(0.0, None, solver, 1e-12).unwrap();
    let mut x = vec![0.0; 36];
    r.solve_into(&rhs(36), &mut x, &mut ScratchPool::new())
        .unwrap();
    fnv(&x)
}

#[test]
fn spdr_binding_parity_digests() {
    let (tri, pcg) = (digest_tridiag(), digest_pcg());
    eprintln!("tridiag=0x{tri:016x} pcg=0x{pcg:016x}");
    // Pinned; the Python test pins the same constants.
    assert_eq!(tri, 0x0cfd_7d64_90fe_5ac2, "tridiag digest");
    assert_eq!(pcg, 0xdeb8_3d11_4b38_4d33, "pcg digest");
}
