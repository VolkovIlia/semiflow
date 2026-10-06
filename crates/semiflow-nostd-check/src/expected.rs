//! Committed output digests, one per scenario (ADR-0200).
//!
//! Recorded from the host `no_std` run; every other configuration — the
//! `std-ref` feature, AVX2 (`-C target-cpu=x86-64-v3`), aarch64 NEON and
//! QEMU Cortex-M3/M4F — must reproduce the table unchanged. A mismatch means
//! a build-dependent computation reached semiflow's numerics; fix the source
//! (libm or portable lanes), never the table per platform.

/// `(scenario name, FNV-1a 64 digest of its output bits)`.
static EXPECTED: &[(&str, u64)] = &[
    ("heat_shift1d", 0xac0e_1482_a362_f4fe),
    ("heat_diffusion", 0xa97b_9aac_14fc_25a8),
    ("strang_heat_decay", 0xc85d_454f_b09e_6b00),
    ("truncated_exp_heat", 0x2002_589c_cbd8_2972),
    ("chebyshev_bc_reflect", 0x0e2d_4db6_7c90_0c95),
    ("chebyshev_bc_periodic", 0x85c7_9f3a_c734_e39d),
    ("schrodinger_unitarity", 0x6d8c_7d76_49ed_717e),
    ("expmv_div_form", 0x869f_d2e1_999b_9b85),
    ("graph_krylov_chebyshev", 0x29d5_3d4a_fadf_cbe5),
    ("symop_implicit_euler_pcg", 0x4819_db47_e626_d11b),
    ("graph_frechet_large_t", 0x2f31_5b1c_9b43_770b),
    ("subordinated_gamma", 0x2dd5_2304_f635_4d71),
    ("gridless_moments", 0x5b45_603f_9e4b_42bc),
    ("smolyak_heat_d2", 0x94d0_efd5_d729_e3cb),
    ("reverse_ad_sqrt_n", 0x998d_f499_a6d4_d979),
    ("texp4_cached_heat", 0x533e_7a1b_4e4c_fcf5),
    ("diffusion6_heat_f64", 0x8754_5239_241e_fe09),
    ("diffusion6_catmull_f32", 0x6180_7208_60df_84af),
    ("cubic_hermite_heat", 0xc0ea_9dbe_d6e1_71cb),
    ("octonic_heat", 0x399e_f380_eb86_fb81),
];

/// Committed digest for `name`, if any.
pub(crate) fn lookup(name: &str) -> Option<u64> {
    EXPECTED
        .iter()
        .find(|(n, _)| *n == name)
        .map(|&(_, digest)| digest)
}
