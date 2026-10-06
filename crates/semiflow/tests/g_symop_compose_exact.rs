//! `G_SYMOP_COMPOSE_EXACT` (`RELEASE_BLOCKING`, ADR-0202 D2, math §62.3).
//!
//! `with_diagonal(c)` equals the dense `A + diag(c)` entrywise, bitwise, including on a
//! hand-built CSR whose rows lack a stored diagonal; `csr()` round-trips through
//! `from_csr` with identical arrays and identical `lambda_max_bound` bits.
#![allow(
    clippy::cast_precision_loss,
    clippy::doc_markdown,
    clippy::many_single_char_names,
    clippy::too_many_lines,
    clippy::useless_vec,
    clippy::similar_names
)]

use semiflow::SymmetricOperator;

mod spdr_common;
use spdr_common::{dense_of, is_domain};

/// n = 6 hand-built CSR; rows 1, 4, 5 lack a stored diagonal.
fn hand_built() -> SymmetricOperator<f64> {
    // rows: 0:{0,1} 1:{0,2} 2:{1,2,3} 3:{2,3,4} 4:{3,5} 5:{4}
    let rp = vec![0, 2, 4, 7, 10, 12, 13];
    let ci: Vec<u32> = vec![0, 1, 0, 2, 1, 2, 3, 2, 3, 4, 3, 5, 4];
    // Rows 1 and 4 have NO stored diagonal; (4,5)/(5,4) are explicit zeros and (5,5)
    // is not stored.
    let va = vec![
        1.0, -1.0, -1.0, -0.5, -0.5, 2.0, -0.75, -0.75, 1.5, -0.25, -0.25, 0.0, 0.0,
    ];
    SymmetricOperator::from_csr(6, &rp, &ci, &va, 1e-14).unwrap()
}

fn assert_bits(a: &[f64], b: &[f64], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: length");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert_eq!(x.to_bits(), y.to_bits(), "{what}[{i}]: {x} vs {y}");
    }
}

#[test]
fn g_symop_compose_exact() {
    let op = hand_built();
    let n = op.n();
    let c = [0.0, 0.5, 0.0, 1.25, 3.0, 0.125];
    let out = op.with_diagonal(&c).expect("with_diagonal");
    // Dense reference: A + diag(c), entrywise, bitwise.
    let mut want = dense_of(&op);
    for i in 0..n {
        want[i * n + i] += c[i];
    }
    assert_bits(&dense_of(&out), &want, "dense A+diag(c)");
    // Columns sorted and unique in every row; every row has its diagonal.
    let (rp, ci, _va) = out.csr();
    for i in 0..n {
        let row = &ci[rp[i]..rp[i + 1]];
        assert!(row.windows(2).all(|w| w[0] < w[1]), "row {i} sorted");
        assert!(row.contains(&u32::try_from(i).unwrap()), "row {i} diagonal");
    }
    // csr() round trip: identical arrays and identical lambda_max_bound bits.
    let (rp, ci, va) = out.csr();
    let back = SymmetricOperator::from_csr(n, rp, ci, va, 0.0).expect("round trip");
    let (rp2, ci2, va2) = back.csr();
    assert_eq!(rp, rp2);
    assert_eq!(ci, ci2);
    assert_bits(va, va2, "vals");
    assert_eq!(
        out.lambda_max_bound().to_bits(),
        back.lambda_max_bound().to_bits()
    );
    // Validation: wrong length / negative / non-finite are DomainViolation.
    assert!(is_domain(
        &op.with_diagonal(&c[..5]).err().expect("expected error")
    ));
    assert!(is_domain(
        &op.with_diagonal(&[0.0, -1e-9, 0.0, 0.0, 0.0, 0.0])
            .err()
            .expect("expected error")
    ));
    assert!(is_domain(&op.with_diagonal(&[f64::NAN; 6]).err().unwrap()));
    eprintln!("G_SYMOP_COMPOSE_EXACT: dense bitwise equal, round trip identical");
}
