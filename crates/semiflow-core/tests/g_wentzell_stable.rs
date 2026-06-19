//! `G_WENTZELL_STABLE` — von-Neumann stability sweep for the Cayley boundary block.
//!
//! `RELEASE_BLOCKING` gate (ADR-0151, math.md §49.5).
//!
//! Verifies:
//! 1. The implicit Cayley map `K_CN = (I − τC_∂/2)⁻¹(I + τC_∂/2)` has `ρ(K_CN) ≤ 1 + 1e-9`
//!    for all `(dx, γ)` in the sweep (A-stable, UNCONDITIONAL).
//! 2. The explicit map `I + τC_∂` has `ρ > 1` somewhere in the sweep — the candidate MUST
//!    fix a real instability (not a trivially stable regime).
//!
//! Sweep: `dx ∈ {1/16, 1/64, 1/256, 1/1024}`, `γ ∈ {0.5, 1, 4, 16}`,
//! `κ = π/dx` (highest wavenumber), `τ = 0.4·dx²/a` (parabolic-scale step).
//!
//! Feature gate: `slow-tests`.

#![cfg(feature = "slow-tests")]

/// 2×2 generator block (same model as the preflight Python script, math §49.3).
fn coupled_generator(a: f64, kappa: f64, gamma: f64, c: f64, dx: f64) -> [[f64; 2]; 2] {
    [
        [-a * kappa * kappa, 1.0 / dx],
        [-gamma / dx, -(gamma / dx + c)],
    ]
}

/// Spectral radius via closed-form 2×2 characteristic polynomial.
///
/// Eigenvalues of M satisfy `λ² − tr(M)·λ + det(M) = 0`.
fn spectral_radius_2x2(m: [[f64; 2]; 2]) -> f64 {
    let tr = m[0][0] + m[1][1];
    let det = m[0][0] * m[1][1] - m[0][1] * m[1][0];
    let disc = tr * tr - 4.0 * det;
    if disc >= 0.0 {
        // Two real eigenvalues: largest absolute value.
        let r1 = 0.5 * (tr + disc.sqrt());
        let r2 = 0.5 * (tr - disc.sqrt());
        r1.abs().max(r2.abs())
    } else {
        // Complex conjugate pair: modulus = √det.
        det.abs().sqrt()
    }
}

/// Closed-form Cayley map `K_CN = (I − τC/2)⁻¹(I + τC/2)` for a 2×2 block.
fn cayley_amp(c: [[f64; 2]; 2], tau: f64) -> [[f64; 2]; 2] {
    let h = 0.5 * tau;
    // L = I - h*C
    let l = [
        [1.0 - h * c[0][0], -h * c[0][1]],
        [-h * c[1][0], 1.0 - h * c[1][1]],
    ];
    // R = I + h*C
    let r = [
        [1.0 + h * c[0][0], h * c[0][1]],
        [h * c[1][0], 1.0 + h * c[1][1]],
    ];
    // R * [unit vectors] then solve L * x = R * e_i
    let det_l = l[0][0] * l[1][1] - l[0][1] * l[1][0];
    let inv_det = det_l.recip();
    // K_CN = L^{-1} R using adjugate
    // L^{-1} = (1/det) * [[l11, -l01], [-l10, l00]]
    let l_inv = [
        [l[1][1] * inv_det, -l[0][1] * inv_det],
        [-l[1][0] * inv_det, l[0][0] * inv_det],
    ];
    // K = L_inv @ R (2x2 product)
    [
        [
            l_inv[0][0] * r[0][0] + l_inv[0][1] * r[1][0],
            l_inv[0][0] * r[0][1] + l_inv[0][1] * r[1][1],
        ],
        [
            l_inv[1][0] * r[0][0] + l_inv[1][1] * r[1][0],
            l_inv[1][0] * r[0][1] + l_inv[1][1] * r[1][1],
        ],
    ]
}

/// Explicit step map `I + τ·C` (Stephan forward-Euler / freezing).
fn explicit_amp(c: [[f64; 2]; 2], tau: f64) -> [[f64; 2]; 2] {
    [
        [1.0 + tau * c[0][0], tau * c[0][1]],
        [tau * c[1][0], 1.0 + tau * c[1][1]],
    ]
}

#[test]
#[ignore] // RELEASE_BLOCKING slow-test — run with `-- --include-ignored`
fn g_wentzell_stable() {
    let a = 1.0_f64;
    let c_reaction = 0.5_f64;
    let dx_list = [1.0 / 16.0, 1.0 / 64.0, 1.0 / 256.0, 1.0 / 1024.0];
    let gamma_list = [0.5_f64, 1.0, 4.0, 16.0];

    let mut cayley_max_rho = 0.0_f64;
    let mut explicit_max_rho = 0.0_f64;
    let mut explicit_unstable_found = false;

    println!(
        "{:>10} {:>8} {:>14} {:>14}",
        "dx", "gamma", "rho_cayley", "rho_explicit"
    );

    for &dx in &dx_list {
        let kappa = core::f64::consts::PI / dx; // highest wavenumber
        let tau = 0.4 * dx * dx / a; // parabolic-scale step
        for &gamma in &gamma_list {
            let c_block = coupled_generator(a, kappa, gamma, c_reaction, dx);
            let k_cay = cayley_amp(c_block, tau);
            let k_expl = explicit_amp(c_block, tau);
            let rho_cay = spectral_radius_2x2(k_cay);
            let rho_expl = spectral_radius_2x2(k_expl);
            cayley_max_rho = cayley_max_rho.max(rho_cay);
            explicit_max_rho = explicit_max_rho.max(rho_expl);
            if rho_expl > 1.0 + 1e-9 {
                explicit_unstable_found = true;
            }
            println!("{dx:>10.6} {gamma:>8.2} {rho_cay:>14.8} {rho_expl:>14.4}");
        }
    }

    println!("\nG_WENTZELL_STABLE: Cayley max rho = {cayley_max_rho:.9}");
    println!("G_WENTZELL_STABLE: Explicit max rho = {explicit_max_rho:.4}");
    println!("G_WENTZELL_STABLE: Explicit unstable found = {explicit_unstable_found}");

    assert!(
        cayley_max_rho <= 1.0 + 1e-9,
        "G_WENTZELL_STABLE FAIL: Cayley rho = {cayley_max_rho:.9} > 1 + 1e-9 (A-stable gate, math §49.5)"
    );
    assert!(
        explicit_unstable_found,
        "G_WENTZELL_STABLE FAIL: explicit map never exceeded rho=1 — must verify a real instability is being fixed"
    );

    println!("G_WENTZELL_STABLE PASS: rho_cayley_max = {cayley_max_rho:.9} <= 1 + 1e-9");
}
