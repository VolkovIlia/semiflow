    use super::*;
    fn ok(n: usize, d: usize) -> AxisCoef<f64> {
        AxisCoef {
            a_axis: (0..d).map(|_| vec![1.0f64; n]).collect(),
            b_axis: (0..d).map(|_| vec![0.0f64; n]).collect(),
            v_axis: (0..d).map(|_| vec![0.0f64; n]).collect(),
        }
    }
    #[test]
    fn rejects_out_of_class() {
        assert!(S3VarCoefEvolver::<f64>::new(1, 2, 0.1, ok(1, 2)).is_err()); // n<2
        assert!(S3VarCoefEvolver::<f64>::new(4, 2, 0.0, ok(4, 2)).is_err()); // dx≤0
        let mut c = ok(4, 2);
        c.a_axis[0][1] = -0.5;
        assert!(S3VarCoefEvolver::<f64>::new(4, 2, 0.1, c).is_err()); // a≤0
        let bad = AxisCoef {
            a_axis: vec![vec![1.0f64; 4]],
            b_axis: vec![vec![0.0; 4], vec![0.0; 4]],
            v_axis: vec![vec![0.0; 4], vec![0.0; 4]],
        };
        assert!(S3VarCoefEvolver::<f64>::new(4, 2, 0.1, bad).is_err()); // shape mismatch
    }
    #[test]
    fn accepts_valid() {
        assert!(S3VarCoefEvolver::<f64>::new(4, 2, 0.1, ok(4, 2)).is_ok());
    }
