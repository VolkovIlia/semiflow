//! Gauss-Hermite quadrature constants for `AnisotropicShiftChernoffND`.
//!
//! Extracted from `shift_nd.rs` to keep that file under 500 lines.
//! All constants are `pub(crate)` — re-exported from `shift_nd.rs`.

/// 1-pt Gauss-Hermite node (physicist weight exp(-x²)).  Σ w = √π.
pub(crate) const GH1_NODES_F64: [f64; 1] = [0.0];
pub(crate) const GH1_WEIGHTS_F64: [f64; 1] = [1.772_453_850_905_516];

/// 3-pt Gauss-Hermite nodes (physicist weight exp(-x²)).  Σ w = √π.
pub(crate) const GH3_NODES_F64: [f64; 3] = [-1.224_744_871_391_589, 0.0, 1.224_744_871_391_589];
pub(crate) const GH3_WEIGHTS_F64: [f64; 3] = [
    0.295_408_975_150_919,
    1.181_635_900_603_677,
    0.295_408_975_150_919,
];

/// 5-pt Gauss-Hermite nodes for physicist weight exp(-x²).
///
/// Computed via numpy.polynomial.hermite.hermgauss(5); verified against
/// Abramowitz-Stegun Table 25.10.  Σ wᵢ = √π exactly in double precision.
pub(crate) const GH5_NODES_F64: [f64; 5] = [
    -2.020_182_870_456_086,
    -0.958_572_464_613_819,
    0.000_000_000_000_000,
    0.958_572_464_613_819,
    2.020_182_870_456_086,
];

/// 5-pt Gauss-Hermite weights for physicist weight exp(-x²).
///
/// Σ wᵢ = √π (verified: sum == 1.7724538509055159).
pub(crate) const GH5_WEIGHTS_F64: [f64; 5] = [
    0.019_953_242_059_046,
    0.393_619_323_152_241,
    0.945_308_720_482_942,
    0.393_619_323_152_241,
    0.019_953_242_059_046,
];

/// 7-pt Gauss-Hermite nodes (physicist weight exp(-x²)).  Σ w = √π.
pub(crate) const GH7_NODES_F64: [f64; 7] = [
    -2.651_961_356_835_233,
    -1.673_551_628_767_471,
    -0.816_287_882_858_965,
    0.0,
    0.816_287_882_858_965,
    1.673_551_628_767_471,
    2.651_961_356_835_233,
];
pub(crate) const GH7_WEIGHTS_F64: [f64; 7] = [
    0.000_971_781_245_099_520,
    0.054_515_582_819_127_05,
    0.425_607_252_610_127_8,
    0.810_264_617_556_807_2,
    0.425_607_252_610_127_8,
    0.054_515_582_819_127_05,
    0.000_971_781_245_099_520,
];

/// 9-pt Gauss-Hermite nodes (physicist weight exp(-x²)).  Σ w = √π.
pub(crate) const GH9_NODES_F64: [f64; 9] = [
    -3.190_993_201_781_528,
    -2.266_580_584_531_843,
    -1.468_553_289_216_668,
    -0.723_551_018_752_838,
    0.0,
    0.723_551_018_752_838,
    1.468_553_289_216_668,
    2.266_580_584_531_843,
    3.190_993_201_781_528,
];
pub(crate) const GH9_WEIGHTS_F64: [f64; 9] = [
    3.960_697_726_326_437e-5,
    0.004_943_624_275_536_941,
    0.088_474_527_394_376_64,
    0.432_651_559_002_555_64,
    0.720_235_215_606_051,
    0.432_651_559_002_555_64,
    0.088_474_527_394_376_64,
    0.004_943_624_275_536_941,
    3.960_697_726_326_437e-5,
];
