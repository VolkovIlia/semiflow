// Unit tests for `matrix_2d3d` — included via `include!` in `matrix_2d3d.rs`.

use super::*;
use crate::{Grid1D, Grid2D, Grid3D};

const M2: usize = 2;

fn k2(n: usize) -> MatrixDiffusionChernoff<f64, M2> {
    let g = Grid1D::new(-3.0, 3.0, n).unwrap();
    MatrixDiffusionChernoff::<f64, M2>::new(
        |_, a| {
            a[0][0] = 0.3;
            a[1][1] = 0.5;
        },
        |_, _b| {},
        |_, c| {
            c[0][1] = 0.05;
            c[1][0] = 0.05;
        },
        g,
    )
    .unwrap()
}

#[test]
fn shape_preserved_2d() {
    use crate::chernoff::ApplyChernoffExt;
    let (gx, gy) = (
        Grid1D::new(-3.0, 3.0, 10).unwrap(),
        Grid1D::new(-3.0, 3.0, 8).unwrap(),
    );
    let s2d = MatrixDiffusionChernoff2D::new(k2(10), k2(8));
    let f0 = MatrixGridFn2D::<f64, M2>::from_fn(Grid2D::new(gx, gy), |x, y| {
        [(-x * x).exp(), (-y * y).exp()]
    });
    let f1 = s2d.apply_chernoff(0.01, &f0).unwrap();
    assert_eq!(f1.values.len(), f0.values.len());
}

#[test]
fn shape_preserved_3d() {
    use crate::chernoff::ApplyChernoffExt;
    let g = Grid1D::new(-2.0, 2.0, 6).unwrap();
    let s3d = MatrixDiffusionChernoff3D::new(k2(6), k2(6), k2(6));
    let f0 = MatrixGridFn3D::<f64, M2>::from_fn(Grid3D::new(g, g, g).unwrap(), |x, y, _z| {
        [(-(x * x + y * y)).exp(), 0.5_f64]
    });
    let f1 = s3d.apply_chernoff(0.01, &f0).unwrap();
    assert_eq!(f1.values.len(), f0.values.len());
}
