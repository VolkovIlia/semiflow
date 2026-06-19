//! `G_BINDING_RESOLVENT_JUMP_PARITY` — sub-test 2 (FFI v3, 0-ULP against core golden).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0138, ADR-0134, slow-tests):
//!   Call `smf_resolvent_jump_new_heat_1d_unit_v3` and
//!   `smf_resolvent_jump_apply_v3` at the CANONICAL smoke params
//!   (§1.1 `V8_1_TIER3_BINDING_DESIGN.md)`:
//!     XMIN=-10.0, XMAX=10.0, N=64, `M_NODES=16`, T=0.5,
//!     u0(x)=exp(-x²), unit diffusion a=1, DEFAULT grid.
//!   Assert that the returned jump values are byte-identical (0 ULP) to the
//!   CORE GOLDEN — the values produced directly by `semiflow-core`.
//!
//! ## Why this is GENUINE
//!
//! The FFI path crosses an `extern "C"` boundary + `Box<ResolventJumpInnerV3>`
//! construction + `smf_resolvent_jump_apply_v3` + buffer-copy into caller-owned
//! memory.  Any precision loss or marshalling bug would show up as a non-zero ULP.
//! The core golden is independently produced by
//! `crates/semiflow-core/tests/binding_resolvent_jump_parity.rs`, not this file.
//!
//! ## ABI-safety
//!
//! `Complex<f64>` / TWS contour arithmetic never crosses the boundary.
//! This test confirms the real-valued output only.

#![allow(unsafe_code)]
// Binding layer: allows for FFI/PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::unreadable_literal
)]

use semiflow_ffi::{
    smf_resolvent_jump_apply_v3, smf_resolvent_jump_free_v3,
    smf_resolvent_jump_new_heat_1d_unit_v3, SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 64;
const M_NODES: usize = 16;
const T: f64 = 0.5;

// ---------------------------------------------------------------------------
// Core golden (produced by crates/semiflow-core/tests/binding_resolvent_jump_parity.rs
// `canonical_resolvent_jump_core` — verified against M_ref=40 self-convergence).
// These are the bit-exact results; any binding that diverges has a marshalling bug.
// ---------------------------------------------------------------------------

const GOLDEN_JUMP: [f64; 64] = [
    -1.6604697501999292e-8,
    -1.8196979968145994e-8,
    -2.1043919050724678e-8,
    -2.451_194_770_765_213e-8,
    -2.7750673863565958e-8,
    -2.9755645262172097e-8,
    -2.917_848_183_837_303e-8,
    -2.306_449_231_500_052e-8,
    -1.8294588627155877e-9,
    6.809_894_126_809_966e-8,
    3.002_520_399_144_616e-7,
    1.0590372225268815e-6,
    3.4540117153368524e-6,
    1.0688839435941088e-5,
    3.152_082_477_107_465e-5,
    8.855_724_001_185_934e-5,
    2.3674745541133347e-4,
    6.013_939_537_988_8e-4,
    1.449_419_876_860_6e-3,
    3.309_231_098_559_233e-3,
    7.146_410_442_600_213e-3,
    1.4574931605988384e-2,
    2.8029691282033423e-2,
    5.075_419_489_013_081e-2,
    8.640_474_326_358_671e-2,
    1.3810723156721388e-1,
    2.0699074294690928e-1,
    2.9055989251532677e-1,
    3.816_174_464_356_022e-1,
    4.685_539_837_423_257e-1,
    5.374_571_700_683_741e-1,
    5.756_858_277_151_045e-1,
    5.756_858_277_151_047e-1,
    5.374_571_700_683_745e-1,
    4.685_539_837_423_263e-1,
    3.8161744643560264e-1,
    2.9055989251532705e-1,
    2.0699074294690953e-1,
    1.3810723156721413e-1,
    8.640_474_326_358_69e-2,
    5.075_419_489_013_094e-2,
    2.8029691282033503e-2,
    1.4574931605988432e-2,
    7.146_410_442_600_238e-3,
    3.3092310985592456e-3,
    1.4494198768606056e-3,
    6.013_939_537_988_825e-4,
    2.3674745541133445e-4,
    8.855_724_001_185_972e-5,
    3.152_082_477_107_474e-5,
    1.0688839435941097e-5,
    3.4540117153368545e-6,
    1.059_037_222_526_882e-6,
    3.0025203991446105e-7,
    6.809_894_126_809_99e-8,
    -1.829_458_862_714_814e-9,
    -2.3064492315000017e-8,
    -2.9178481838372693e-8,
    -2.9755645262171945e-8,
    -2.7750673863565958e-8,
    -2.4511947707652185e-8,
    -2.1043919050724764e-8,
    -1.8196979968146083e-8,
    -1.6604697501999388e-8,
];

// ---------------------------------------------------------------------------
// Helper: u0[i] = exp(-x_i²)
// ---------------------------------------------------------------------------

fn make_u0() -> Vec<f64> {
    let dx = (XMAX - XMIN) / (N - 1) as f64;
    (0..N)
        .map(|i| {
            #[allow(clippy::cast_precision_loss)]
            let x = XMIN + i as f64 * dx;
            (-x * x).exp()
        })
        .collect()
}

// ---------------------------------------------------------------------------
// G_BINDING_RESOLVENT_JUMP_PARITY sub-test 2: FFI v3 byte-identical to core golden
// ---------------------------------------------------------------------------

/// `G_BINDING_RESOLVENT_JUMP_PARITY` sub-test 2 (FFI v3, 0-ULP).
///
/// Calls `smf_resolvent_jump_new_heat_1d_unit_v3` + `smf_resolvent_jump_apply_v3`
/// from Rust (same mechanics as a C caller).  Asserts that the jump output is
/// byte-identical (0 ULP) to the CORE GOLDEN.
#[test]
fn g_binding_resolvent_jump_parity_ffi_v3() {
    let u0 = make_u0();
    let mut ev: *mut semiflow_ffi::SmfResolventJumpV3 = std::ptr::null_mut();

    // --- Construct the FFI handle ---
    let rc = unsafe { smf_resolvent_jump_new_heat_1d_unit_v3(XMIN, XMAX, N, M_NODES, &mut ev) };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI new failed: {rc:?}");
    assert!(!ev.is_null(), "FFI handle must be non-null on Ok");

    // --- Evaluate jump ---
    let mut out = vec![0.0f64; N];
    let rc = unsafe { smf_resolvent_jump_apply_v3(ev, T, u0.as_ptr(), N, out.as_mut_ptr(), N) };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI apply failed: {rc:?}");

    // --- Free the handle ---
    unsafe { smf_resolvent_jump_free_v3(ev) };

    // --- 0-ULP check against core golden ---
    let max_ulp = max_ulp_diff(&out, &GOLDEN_JUMP);

    println!(
        "G_BINDING_RESOLVENT_JUMP_PARITY sub-test 2 (FFI v3):\n\
         How called: smf_resolvent_jump_new_heat_1d_unit_v3 + smf_resolvent_jump_apply_v3\n\
         max ULP diff vs core golden = {max_ulp}  (expected 0)\n\
         out[32] = {:.16e}  golden[32] = {:.16e}",
        out[32], GOLDEN_JUMP[32],
    );

    assert_eq!(
        max_ulp, 0,
        "FFI v3 jump is NOT byte-identical to core golden (max ULP diff = {max_ulp})"
    );
}

// ---------------------------------------------------------------------------
// ULP helpers
// ---------------------------------------------------------------------------

fn max_ulp_diff(got: &[f64], want: &[f64]) -> u64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want.iter())
        .map(|(&g, &w)| ulp_dist(g, w))
        .max()
        .unwrap_or(0)
}

fn ulp_dist(a: f64, b: f64) -> u64 {
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    ai.wrapping_sub(bi).unsigned_abs()
}
