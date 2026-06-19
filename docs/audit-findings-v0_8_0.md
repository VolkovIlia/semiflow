# v0.8.0 Math Fidelity Audit (ROADMAP item 6)

**Auditor**: researcher agent
**Date**: 2026-05-06
**Scope**: `v0.7.0..HEAD` (commits 1399e7f, a1d8c78, 300054f, 3a33e98, a927aac, 7d39938, 217b168, 3091c12, 920993a)
**Theme**: PERFORMANCE (math frozen since v0.7.0)

## Executive Summary

**No math changed.** v0.8.0 ships parallelism (Block B), SIMD intrinsics (Block C),
clippy hygiene (Block A), and FLAGSHIP-deferral docs (Block D), each with explicit
bit-equal contracts to the v0.7.0 scalar/serial reference. The v0.5.0 frozen golden,
3 SIMD/parallel determinism gates, 12 sympy NORMATIVE verification scripts, and the
149-test fast suite all pass. The single ignored test is the deferred G3⁶-2D
flagship (ADR-0020 Amendment #2). **APPROVE for v0.8.0 release.**

## 1. Source-Diff Review (`v0.7.0..HEAD`)

| File | Lines Δ | Verdict | Evidence |
|------|---------|---------|----------|
| `diffusion.rs` | 36 | PASS — renames + rustdoc backticks | `fph→f_pos1`, `ap_x→a_prime_x`; coeffs W0/W1/W2, ζ-A correction polynomial untouched |
| `diffusion4.rs` | 22 | PASS — renames + `as f64`→`f64::from` | `near_n_*→near_neg_*`, `ap_x→a_prime_x`; ζ⁴ correction & C3/C2/C1 stencils untouched |
| `diffusion6.rs` | 82 | PASS — adds SIMD path with bit-equal scalar fallback | New `fd9_simd` (4+4+1 split, FMA-free); `fd9_scalar` is verbatim v0.7.0 body; dispatcher gated by `cfg!(test) && FORCE_SCALAR`; K7_W*/Fornberg-9 weights untouched |
| `strang2d.rs` | 199 | PASS — refactor (no body change) + parallel feature-gated impl | Original `apply` body moved verbatim to `pub(crate) fn apply_serial`; `serial_order`/`serial_growth` extracted helpers preserve `min(p_x,p_y,4)` and `(M_x²·M_y, ω_x+ω_y)` exactly; parallel impl falls back to serial when `ny < 2*MIN_ROWS_PER_THREAD` |
| `strang2d_parallel.rs` | +350 (new) | PASS — new file behind `parallel` feature | `parallel_x_pass` / `parallel_y_pass` via `std::thread::scope`; output bit-equality is the test contract (verified §2 below) |
| `nonseparable2d.rs` | 49 | PASS — duplicate impl block under `cfg(feature="parallel")` adding `Send + Sync` bounds | Same 5-leg formula `X(τ/2)·Y(τ/2)·M(τ)·Y(τ/2)·X(τ/2)`; CFL θ=1/4; growth `m_mixed = 1.0+0.25+0.5*0.25²`; `apply_strang2d_path` switches to `s3.apply_serial` (same body) |
| `adaptive.rs` | 78 | PASS — `evolve_adaptive` extracts `substep` helper | Same accept/reject semantics: `richardson_err`→`mixed_tolerance`→`pi_step_factor`/`reject_step_factor`→`clamp_step`; tol divisor `((1u64<<p)-1)`; α/β PI gains and safety unchanged |
| `grid.rs` | 39 | PASS — extracts `catmull_rom` to `grid_cubic.rs` (verbatim) + cast hygiene | `bc_value` linear-extrapolate formula `f0 - d·0.5·slope_combo` preserved (`as f64`→`f64::from`) |
| `grid_cubic.rs` | +62 (new) | PASS — Catmull-Rom SIMD with bit-equal scalar fallback | `catmull_rom_scalar` is verbatim relocated v0.7.0 body; `catmull_rom_simd` reformulates as `0.5·dot(coeffs(s),pts)`; dispatcher gated by `FORCE_SCALAR` |
| `grid_quintic.rs` | 60 | PASS — same SIMD-with-bit-equal-scalar pattern | `fd_scaled_prime_scalar` verbatim; new `_simd` zero-pads lane 3, divides by 60.0 the same way; quintic-Hermite Horner weights untouched |
| `truncated_exp.rs` | 4 | PASS — rustdoc backticks + `#[allow(cast_possible_truncation)]` for compile-time const | No code change |
| `truncated_exp4.rs` | 37 | PASS — pure renames | `h_p2/h_p1/h_i/h_m1/h_m2 → rp2/rp1/ctr/lm1/lm2`; `a_p3h/a_p1h/a_m1h/a_m3h → ar3h/ar1h/al1h/al3h`; ADR-0013 5-pt divergence-form coefficients (5/4 and 1/12) intact |
| `drift_reaction.rs` | 2 | PASS — single rustdoc backtick fix | No code change |
| `grid_fn2d.rs` | 9 | PASS — test-only `assert_eq!(... 1.0)` → `(.-1.0).abs() < EPSILON` | Pedantic `float_cmp` lint hygiene; library code unchanged |
| `error.rs` | 64 | PASS — extracts 2 `write!` arms into `fmt_*` helpers | User-facing strings byte-identical (verified by re-read) |
| `lib.rs` | 7 | PASS — `forbid(unsafe_code)`→`deny(unsafe_code)` + 3 module decls | Required by SIMD intrinsics; module-level `#![allow(unsafe_code)]` confined to `src/simd/{x86_64,aarch64}.rs` per ADR-0019 |
| `simd/mod.rs` | +104 (new) | PASS — bit-equal contract codified | Trait doc forbids FMA, mandates `((l0+l1)+l2)+l3` reduction order |
| `simd/scalar.rs` | +71 (new) | PASS — pure scalar baseline used for cfg(test) and non-SIMD targets |
| `simd/x86_64.rs` | +102 (new) | PASS — AVX2 intrinsics in scoped `unsafe` |
| `simd/aarch64.rs` | +127 (new) | PASS — NEON intrinsics in scoped `unsafe` |

**Conclusion §1**: zero algorithmic change. All numerical constants (stencil weights,
CFL bounds, splitting orders, BCH/oracle coefficients, PI gains, growth bounds) are
preserved verbatim. Every new code path has an explicit bit-equal contract.

## 2. Determinism Gates (release-blocking)

| Gate | Command | Expected | Observed |
|------|---------|----------|----------|
| v0.5.0 regression (scalar) | `cargo test --release --test v0_5_0_regression_bit_equal` | 4/0/0 | **4/0/0 ✓** |
| v0.5.0 regression (parallel+simd) | `RUSTFLAGS=-C target-cpu=native cargo test --release --features parallel,simd --test v0_5_0_regression_bit_equal` | 4/0/0 | **4/0/0 ✓** |
| Block B `STRANG2D_PARALLEL_BIT_EQUAL` | `RUSTFLAGS=-C target-cpu=native cargo test --release --features parallel,slow-tests --test strang2d_parallel_bit_equal` | 3/0/0 | **3/0/0 ✓** (DiffusionChernoff, Diffusion4thChernoff, speedup_gate_informational) |
| Block C `SIMD_BIT_EQUAL` | `RUSTFLAGS=-C target-cpu=native cargo test --release --features simd --test simd_bit_equal` | 2/0/0 | **2/0/0 ✓** (diffusion6_bit_equal_all, quintic_hermite_bit_equal_all) |
| Block C `SIMD_BIT_EQUAL_PARALLEL` | `RUSTFLAGS=-C target-cpu=native cargo test --release --features parallel,simd,slow-tests --test strang2d_parallel_bit_equal` | 3/0/0 | **3/0/0 ✓** |

The v0.5.0 frozen golden vector reproduces byte-for-byte both on the plain scalar
build and with `parallel,simd` enabled — the SIMD/parallel paths converge to the
same f64 bytes as the v0.5.0 reference scalar path.

## 3. Sympy NORMATIVE Verification Scripts

Local runner: `python3 .dev-docs/verification/scripts/verify_*.py` (sympy 1.14.0).

| Script | Result |
|--------|--------|
| `verify_liouville_oracle.py` | PASS |
| `verify_v0_2_3_beta.py` | PASS |
| `verify_v0_2_3_variants.py` | PASS |
| `verify_v0_3_0_gamma.py` | PASS |
| `verify_v0_3_0_zeta_engineer_impl.py` | PASS |
| `verify_v0_3_0_zeta.py` | PASS |
| `verify_v0_6_0_magnus4.py` | PASS |
| `verify_v0_6_0_zeta4.py` | PASS |
| `verify_v0_7_0_kkernel.py` | PASS |
| `verify_v0_7_0_nonseparable.py` | PASS |
| `verify_v0_7_0_quintic_hermite.py` | PASS |
| `verify_v0_7_0_zeta6.py` | PASS |

**12/12 scripts exit 0**, covering the 16 individual NORMATIVE gates per ROADMAP
v0.7.0 (each script bundles ≥1 named sympy gate; e.g., `verify_v0_7_0_kkernel.py`
covers `K7_sum-to-1`, `K7_xi6-match`, …). Math is frozen.

## 4. Block D Deferral Consistency (G3⁶-2D FLAGSHIP → v0.8.1)

| Artefact | Required | Evidence |
|----------|----------|----------|
| `docs/adr/0020-g3-6th-2d-flagship.md` | Amendment 2026-05-06 (#2) present | line 232: `## Amendment 2026-05-06 (#2): Defer FLAGSHIP gate to v0.8.1` ✓ |
| `contracts/semiflow-core.properties.yaml` | `G3_6_2D status: DEFERRED, deferred_to: v0.8.1` | lines 3315–3317: `status: DEFERRED`, `deferred_to: "v0.8.1"`, `deferred_reason: "Two-run calibration failures; ADR-0020 Amendment 2"` ✓ |
| `crates/semiflow-core/tests/convergence_rate_6th_2d.rs` | `#[ignore = "..."]` annotation | line 202: `#[ignore = "Block D FLAGSHIP deferred to v0.8.1 — see ADR-0020 Amendment 2"]` ✓ |
| `ROADMAP.md` v0.8.0 item 4 | Deferral note (NOT crossed out) | lines 92–102: `[ ] **G3⁶-2D FLAGSHIP** — … Test infrastructure shipped … `#[ignore]`. Gate calibration deferred to v0.8.1` ✓ |

All four points consistent. The flagship is the single ignored test in the workspace
(verified §6 below).

## 5. Workspace Test Suite (`cargo test --workspace`)

```
TOTAL: 149 passed, 0 failed, 1 ignored
```

The 1 ignored test = `convergence_rate_6th_2d::*` (FLAGSHIP, expected). All 149 active
tests pass on the fast `profile.test opt-level=2` profile.

## 6. Suckless Invariants

- **Runtime deps**: 2 (`num-traits`, `libm`) — unchanged from v0.7.0 ✓
- **Files ≤ 500 lines**: largest = `grid.rs` 460 LoC ✓ (none over)
- **Function ≤ 50 lines**: enforced by `cargo clippy --all-targets --features parallel,simd,slow-tests -- -D warnings` exit 0 ✓
- **Workspace version**: still `0.7.0` in `Cargo.toml` line 6 — Block E will bump to `0.8.0` (out of scope for this audit)

## 7. Final Verdict

**APPROVE FOR RELEASE** — v0.8.0 is a true performance MINOR. Math invariants are
intact end-to-end:

1. Source diff: 100% non-numerical changes (renames, rustdoc, feature-gated dispatch
   with verbatim scalar fallback).
2. Bit-equal regression: v0.5.0 frozen golden reproduces with and without
   `parallel,simd`.
3. All 5 release-blocking determinism gates (3 parallel × 2 SIMD runs) pass.
4. All 12 sympy NORMATIVE scripts (16 gates) pass.
5. Block D FLAGSHIP deferral is consistent across ADR, contract, test, and roadmap.
6. 149/149 active tests pass; the single ignored test is the documented flagship.
7. Suckless invariants (deps=2, files≤500 LoC, clippy clean) hold.

**Russian release-CHANGELOG seed**: «v0.8.0 — производительность: параллелизм
Strang2D, SIMD-интринсики (AVX2/NEON), все математические инварианты v0.7.0
сохранены побитово (gates: v0.5.0 regression, STRANG2D_PARALLEL_BIT_EQUAL,
SIMD_BIT_EQUAL, 16 sympy NORMATIVE). FLAGSHIP G3⁶-2D перенесён в v0.8.1.»
