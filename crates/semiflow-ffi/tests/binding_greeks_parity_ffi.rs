//! `G_BINDING_GREEKS_PARITY` — sub-test 2 (FFI v3, 0-ULP against core golden).
//!
//! Gate (`RELEASE_BLOCKING`, ADR-0028 Amendment 2, ADR-0133 A1):
//!   Call `smf_greeks_evolver_new_heat_1d_unit_v3` and `smf_heat1d_greeks_v3`
//!   at the CANONICAL smoke params (§5 `V8_PHASE5_BINDING_GREEKS_DESIGN.md`,
//!   contracts/semiflow-core.properties.yaml §`G_BINDING_GREEKS_PARITY)`:
//!     θ₀=0.5, N=64, `n_chernoff=32`, t=0.05, u0=exp(-x²), domain [-10, 10].
//!   Assert that the returned (value, delta, gamma) triples are byte-identical
//!   (0 ULP) to the CORE GOLDEN — the same values produced directly by the
//!   `semiflow-core` hyper-dual sweep with identical parameters.
//!
//! ## Why this is GENUINE and not tautological
//!
//! The FFI path crosses an `extern "C"` boundary + pointer round-trip +
//! `Box<GreeksInnerV3>` construction + `apply_f` loop + three separate
//! `write_output_buffers` calls.  Any precision loss in the buffer-copy or
//! grid-index computation would show up as a ULP difference here.
//! The golden constants are independently produced by `semiflow-core` directly
//! (see `crates/semiflow-core/tests/binding_greeks_parity.rs`), not by this file.
//!
//! ## How the FFI is called
//!
//! This Rust integration test invokes the `extern "C"` symbols directly
//! via raw pointer manipulation — equivalent to a C caller, but without
//! a separate C binary.  This is the same pattern used in `ffi_v3_smoke.rs`.

#![allow(unsafe_code)]
// Binding layer: allows for FFI/PyO3/wasm-bindgen wrapper patterns.
#![allow(
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::unreadable_literal
)]

use semiflow_ffi::{
    smf_greeks_evolver_free_v3, smf_greeks_evolver_new_heat_1d_unit_v3, smf_heat1d_greeks_v3,
    SemiflowStatus,
};

// ---------------------------------------------------------------------------
// Canonical smoke parameters (contracts/semiflow-core.properties.yaml)
// ---------------------------------------------------------------------------

const XMIN: f64 = -10.0;
const XMAX: f64 = 10.0;
const N: usize = 64;
const N_CHERNOFF: usize = 32;
const T: f64 = 0.05;
const THETA: f64 = 0.5;

// ---------------------------------------------------------------------------
// Core golden (produced by crates/semiflow-core/tests/binding_greeks_parity.rs
// `print_golden_for_binding_tests` — verified against Richardson FD anchor).
// These are the bit-exact results of the hyper-dual core sweep; any binding
// that diverges has a marshalling bug.
// ---------------------------------------------------------------------------

const GOLDEN_VALUE: [f64; 64] = [
    1.2273665126349644e-23,
    1.5634124948354552e-22,
    1.8630596457416103e-21,
    1.3334389244702536e-20,
    6.273_179_086_584_449e-20,
    2.622_834_887_323_565e-19,
    2.699_695_137_364_158e-18,
    3.9812079022180766e-17,
    4.217_694_214_819_308e-16,
    3.149_309_632_109_956e-15,
    1.7743373610017136e-14,
    9.138_916_096_653_64e-14,
    6.662_497_282_186_772e-13,
    6.843_910_818_911_789e-12,
    6.516_457_269_242_25e-11,
    4.962_994_021_179_617e-10,
    3.0456011009952755e-9,
    1.6334237696875445e-8,
    9.273_584_444_886_448e-8,
    6.612_706_512_377_167e-7,
    5.280_969_591_421_713e-6,
    3.8923073423285504e-5,
    2.436_846_172_240_2e-4,
    1.270_919_783_254_598e-3,
    5.510_056_311_249_373e-3,
    1.9873008874759564e-2,
    5.966_231_347_555_289e-2,
    1.4912925268685812e-1,
    3.103_620_750_105_716e-1,
    5.377_865_572_717_248e-1,
    7.758_480_746_924_641e-1,
    9.318_838_944_541_147e-1,
    9.318_838_944_541_168e-1,
    7.758_480_746_924_552e-1,
    5.377_865_572_717_286e-1,
    3.1036207501055646e-1,
    1.491_292_526_868_596e-1,
    5.9662313475555125e-2,
    1.987_300_887_475_921e-2,
    5.510_056_311_249_474e-3,
    1.2709197832546308e-3,
    2.4368461722399043e-4,
    3.892_307_342_328_329e-5,
    5.2809695914224884e-6,
    6.612_706_512_375_904e-7,
    9.273_584_444_886_177e-8,
    1.633_423_769_688_029e-8,
    3.0456011009940273e-9,
    4.962_994_021_179_928e-10,
    6.516_457_269_244_97e-11,
    6.8439108189034605e-12,
    6.662_497_282_195_187e-13,
    9.138_916_096_663_29e-14,
    1.7743373609967763e-14,
    3.1493096321175424e-15,
    4.217_694_214_817_302e-16,
    3.981_207_902_197_535e-17,
    2.6996951374135936e-18,
    2.6228348872722076e-19,
    6.273_179_086_549_98e-20,
    1.3334389244918855e-20,
    1.8630596457028758e-21,
    1.5634124948587485e-22,
    1.2273665127349842e-23,
];
const GOLDEN_DELTA: [f64; 64] = [
    1.5116386225478912e-22,
    1.9747025981399162e-21,
    2.1906503466756586e-20,
    1.4072643655854709e-19,
    5.614_818_662_543_216e-19,
    1.9820087363987545e-18,
    2.3424935159904366e-17,
    3.553_833_532_471_729e-16,
    3.5412389753390115e-15,
    2.406_556_399_717_728e-14,
    1.2020239792222649e-13,
    5.435_485_161_935_972e-13,
    3.750_541_618_655_134e-12,
    3.766_372_181_468_426e-11,
    3.361_733_606_772_356e-10,
    2.325_678_771_654_368e-9,
    1.2651129008784693e-8,
    5.828_926_497_689_721e-8,
    2.7412537590325326e-7,
    1.6178048223466443e-6,
    1.089_682_897_531_993e-5,
    6.720_229_366_323_835e-5,
    3.4327755576714857e-4,
    1.4150456337306452e-3,
    4.667_385_844_686_427e-3,
    1.2186025656545101e-2,
    2.463_615_034_778_377e-2,
    3.6732162269824856e-2,
    3.5110862017445975e-2,
    7.114_672_671_154_658e-3,
    -4.1440107702895135e-2,
    -8.084_553_899_123_452e-2,
    -8.084_553_899_124_042e-2,
    -4.1440107702884074e-2,
    7.114_672_671_143_167e-3,
    3.511_086_201_745_903e-2,
    3.673_216_226_981_944e-2,
    2.4636150347783634e-2,
    1.2186025656546107e-2,
    4.667_385_844_686_287e-3,
    1.4150456337307053e-3,
    3.432_775_557_671_518e-4,
    6.720_229_366_322_965e-5,
    1.0896828975321073e-5,
    1.6178048223465736e-6,
    2.7412537590319503e-7,
    5.8289264976922364e-8,
    1.2651129008781323e-8,
    2.3256787716538776e-9,
    3.361_733_606_774_681e-10,
    3.7663721814636734e-11,
    3.750_541_618_656_485e-12,
    5.435_485_161_952_685e-13,
    1.2020239792176395e-13,
    2.4065563997225146e-14,
    3.541_238_975_343_665e-15,
    3.5538335324416834e-16,
    2.3424935160441756e-17,
    1.9820087363691325e-18,
    5.614_818_662_433_887e-19,
    1.4072643656202276e-19,
    2.1906503466277156e-20,
    1.974_702_598_141_9e-21,
    1.5116386228096023e-22,
];
const GOLDEN_GAMMA: [f64; 64] = [
    1.7668632693894237e-21,
    2.153_287_902_160_806e-20,
    2.1330711897272125e-19,
    1.1663813226807447e-18,
    3.520_393_579_470_559e-18,
    9.714_529_829_736_497e-18,
    1.703_349_661_133_509e-16,
    2.5972351777791464e-15,
    2.321_367_133_876_84e-14,
    1.355_230_362_450_13e-13,
    5.573_829_518_431_403e-13,
    2.1338674016780794e-12,
    1.5395048314131047e-11,
    1.545_005_607_629_165e-10,
    1.2246814727029343e-9,
    7.083_502_263_074_629e-9,
    3.088_217_449_235_859e-8,
    1.1337699163707118e-7,
    4.801_941_569_292_22e-7,
    2.883_162_681_139_556e-6,
    1.8036862783792596e-5,
    9.204_715_469_355_409e-5,
    3.593_819_298_109_211e-4,
    1.047_395_287_632_07e-3,
    2.1674943139937688e-3,
    2.697_883_920_208_277e-3,
    1.982_409_361_292_772e-4,
    -6.789_643_828_716_816e-3,
    -1.3891784394294955e-2,
    -1.1320284783412078e-2,
    4.472_162_698_827_893e-3,
    2.094_555_380_540_857e-2,
    2.0945553805424004e-2,
    4.472_162_698_805_449e-3,
    -1.1320284783389995e-2,
    -1.389_178_439_431_667e-2,
    -6.789_643_828_705_265e-3,
    1.9824093612621886e-4,
    2.697_883_920_207_46e-3,
    2.1674943139943026e-3,
    1.0473952876319218e-3,
    3.593_819_298_109_522e-4,
    9.204_715_469_355_4e-5,
    1.803_686_278_379_028e-5,
    2.8831626811404476e-6,
    4.801_941_569_289_303e-7,
    1.1337699163711157e-7,
    3.0882174492368686e-8,
    7.083_502_263_069_067e-9,
    1.2246814727040287e-9,
    1.5450056076284108e-10,
    1.5395048314089992e-11,
    2.1338674016943157e-12,
    5.573_829_518_405_228e-13,
    1.3552303624500194e-13,
    2.3213671338881788e-14,
    2.5972351777476566e-15,
    1.7033496611697046e-16,
    9.714_529_829_989_011e-18,
    3.520_393_579_267_902e-18,
    1.1663813227231383e-18,
    2.133_071_189_690_885e-19,
    2.1532879021021273e-20,
    1.7668632698973982e-21,
];

// ---------------------------------------------------------------------------
// Helper: build u0[i] = exp(-x_i²)
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
// G_BINDING_GREEKS_PARITY sub-test 2: FFI v3 byte-identical to core golden
// ---------------------------------------------------------------------------

/// `G_BINDING_GREEKS_PARITY` sub-test 2 (FFI v3).
///
/// Calls `smf_greeks_evolver_new_heat_1d_unit_v3` + `smf_heat1d_greeks_v3`
/// from Rust (same mechanics as a C caller).  Asserts that value, delta, and
/// gamma are byte-identical (0 ULP) to the CORE GOLDEN.
///
/// How this is run: the `extern "C"` functions are called directly via raw
/// pointer manipulation — equivalent to the C smoke (`examples/greeks.c`) but
/// exercised from a Rust integration test so it can use the `assert_eq!` macro
/// and report 0-ULP failures precisely.
#[test]
fn g_binding_greeks_parity_sub2_ffi_0ulp() {
    let u0 = make_u0();
    let mut ev = std::ptr::null_mut();

    // --- Construct the FFI Greeks evolver ---
    let rc = unsafe {
        smf_greeks_evolver_new_heat_1d_unit_v3(
            XMIN,
            XMAX,
            N,
            N_CHERNOFF,
            THETA,
            u0.as_ptr(),
            N,
            &mut ev,
        )
    };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI constructor failed: {rc:?}");
    assert!(!ev.is_null(), "FFI constructor returned null ev");

    // --- Evaluate Greeks at t=0.05 ---
    let mut value = vec![0.0f64; N];
    let mut delta = vec![0.0f64; N];
    let mut gamma = vec![0.0f64; N];

    let rc = unsafe {
        smf_heat1d_greeks_v3(
            ev,
            T,
            value.as_mut_ptr(),
            delta.as_mut_ptr(),
            gamma.as_mut_ptr(),
            N,
        )
    };
    assert_eq!(rc, SemiflowStatus::Ok, "FFI greeks eval failed: {rc:?}");

    // --- Free the handle ---
    unsafe { smf_greeks_evolver_free_v3(ev) };

    // --- Compute ULP statistics ---
    let max_ulp_value = max_ulp_diff(&value, &GOLDEN_VALUE);
    let max_ulp_delta = max_ulp_diff(&delta, &GOLDEN_DELTA);
    let max_ulp_gamma = max_ulp_diff(&gamma, &GOLDEN_GAMMA);

    println!(
        "G_BINDING_GREEKS_PARITY sub-test 2 (FFI v3):\n\
         How called: smf_greeks_evolver_new_heat_1d_unit_v3 (extern C, raw-ptr from Rust)\n\
         value: max ULP diff = {max_ulp_value}  (expected 0)\n\
         delta: max ULP diff = {max_ulp_delta}  (expected 0)\n\
         gamma: max ULP diff = {max_ulp_gamma}  (expected 0)"
    );

    assert_eq!(
        max_ulp_value, 0,
        "FFI v3 value is NOT byte-identical to core golden (max ULP diff = {max_ulp_value})"
    );
    assert_eq!(
        max_ulp_delta, 0,
        "FFI v3 delta is NOT byte-identical to core golden (max ULP diff = {max_ulp_delta})"
    );
    assert_eq!(
        max_ulp_gamma, 0,
        "FFI v3 gamma is NOT byte-identical to core golden (max ULP diff = {max_ulp_gamma})"
    );
}

// ---------------------------------------------------------------------------
// ULP helper: max bits-difference across a pair of f64 slices
// ---------------------------------------------------------------------------

/// Compute max ULP distance between corresponding elements of two f64 slices.
///
/// ULP distance is defined as the bit distance in IEEE-754 representation.
/// 0 means byte-identical; 1 means adjacent floating-point numbers.
/// NaN/Inf are treated as ±`i64::MAX` to surface them as failures.
fn max_ulp_diff(got: &[f64], want: &[f64]) -> u64 {
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(want.iter())
        .map(|(&g, &w)| ulp_dist(g, w))
        .max()
        .unwrap_or(0)
}

/// IEEE-754 ULP distance between two finite f64 values.
fn ulp_dist(a: f64, b: f64) -> u64 {
    let ai = a.to_bits() as i64;
    let bi = b.to_bits() as i64;
    ai.wrapping_sub(bi).unsigned_abs()
}
