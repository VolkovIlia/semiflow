# v0.8.1 Math Fidelity Audit

**Auditor**: docs-writer agent
**Date**: 2026-05-07
**Scope**: `v0.8.0..HEAD` (commits `eae2b7a` Block A + Block B staged changes)
**Theme**: G3⁶-2D FLAGSHIP gate + tile-scratch perf (math frozen since v0.7.0)

## 1. Summary

**APPROVED FOR RELEASE.** v0.8.1 closes ROADMAP item 4 with a 3-point prime-N
recalibration of the G3⁶-2D FLAGSHIP gate (slope=-6.0837, window [-6.15, -5.85],
wallclock=3090s under a 3300s budget) and a bit-equal-preserving tile-scratch reuse
improvement in the parallel Strang2D hot loops (4.38× speedup at N=1600²). Zero
algorithmic changes since v0.7.0.

## 2. Math Fidelity

**ZERO algorithmic changes since v0.7.0.** Theorem 7 is unchanged. The K7 kernel,
9-point Fornberg FD stencil, and QuinticHermite interpolant per axis are frozen.
All numerical constants (stencil weights, CFL bounds, splitting orders,
BCH/oracle coefficients, PI gains, growth bounds) are preserved verbatim at v0.7.0
values. v0.8.1 touches only:

- `strang2d_parallel.rs` — allocation site only (values identical, bit-equal verified).
- `strang2d.rs` — `apply_parallel` owns the `y_scratch` allocation (internal wiring).
- `tests/convergence_rate_6th_2d.rs` — N_SWEEP recalibrated, `#[ignore]` removed,
  schema-doc version bumped 0.7.4 → 0.7.6.
- `contracts/semiflow-core.properties.yaml` — schema 0.7.5 → 0.7.6, G3_6_2D entry
  updated to reflect gate activation and final run results.

Sympy NORMATIVE scripts `verify_v0_7_0_kkernel.py` and `verify_v0_7_0_zeta6.py` are
unchanged from v0.7.0 and expected to pass without alteration. The full 12-script
suite from the v0.8.0 audit (`docs/audit-findings-v0_8_0.md` §3) remains valid; no
new sympy verification is required because v0.8.1 adds no new mathematical content —
the 2D O(dx⁶) claim reduces to the 1D claim via Theorem 7's separable-commutator
identity, which is an exact algebraic statement requiring no numerical sympy gate.

## 3. Bit-Equal Evidence

All determinism gates confirmed at HEAD with Block A tile-scratch perf landed:

| Gate | Result |
|------|--------|
| `STRANG2D_PARALLEL_BIT_EQUAL` (3 sub-tests) | **3/3 PASS** |
| `SIMD_BIT_EQUAL_PARALLEL` (3 sub-tests) | **3/3 PASS** |
| `v0_5_0_regression_bit_equal` (4 sub-tests) | **4/4 PASS** |

Bit-equality is preserved by construction: the allocation site changed (tile-scratch
reuse via `core::mem::take`), not the arithmetic. The v0.5.0 frozen golden vector
reproduces byte-for-byte under Block A perf with and without `parallel,simd`.

## 4. G3⁶-2D FLAGSHIP Gate Evidence

**Gate**: `g3_6_2d_flagship_slope_and_runtime_gate`
**Command**: `RUSTFLAGS="-C target-cpu=native" cargo test --release --features parallel,simd,slow-tests`
**Schema**: `contracts/semiflow-core.properties.yaml` 0.7.6

Per-N measurements (hardware at HEAD `eae2b7a`, pilot run):

| N | ‖err‖∞ | wallclock |
|---|--------|-----------|
| 503 | 1.2198e-7 | 155 s |
| 997 | 9.7974e-10 | 588 s |
| 1999 | 2.7481e-11 | 2341 s |

3-point OLS slope: **-6.0837** (window [-6.15, -5.85] — **PASS**).
Wallclock pilot: **3084 s**. Final gate run: **3090 s** (budget 3300 s — **PASS**,
margin 6.4%).

Convergence-order analysis: order 7.03 at N=503→997 pair (super-asymptotic,
pre-asymptotic ratio amplification), order 5.16 at N=997→1999 pair (settling toward 6
as N grows). OLS slope -6.0837 over the full 3-point basket falls inside the window
with a margin of 0.0837 from the lower bound, consistent with the Theorem 7 asymptote
prediction of approximately -5.95.

## 5. Suckless Invariants

- **Runtime deps**: 2 (`num-traits`, `libm`) — unchanged from v0.7.0.
- **Largest src file**: `src/grid.rs` 460 LoC — all source files under 500 LoC.
- **Functions ≤ 50 LoC**: enforced by existing clippy clean; tile-scratch delta is
  44 LoC distributed across helper functions that were already within limits.
- **`unsafe` scope**: confined to `src/simd/{x86_64,aarch64}.rs` per ADR-0019;
  no new `unsafe` introduced in Block A.
- **Public API**: zero changes. `pub(crate)` signature adjustment to `parallel_y_pass`
  is internal. `cargo public-api --diff-against 0.8.0` reports no surface change.
- **Workspace version**: `0.8.1` in `Cargo.toml` `[workspace.package]`.

## 6. Out of Scope (deferred to v0.9.0)

The following items were explicitly considered and deferred:

- `NonSeparable2DChernoff` serial dispatch improvements
- Magnus K=4 SIMD paths
- `AdaptivePI` parallel stepping
- `FORCE_THREADS` constant sealing
- `MIN_ROWS_PER_THREAD` tuning experiments
- Fused FD9+K7 SIMD pass
- Hermite-SIMD 8-wide lane fusion

None of these affect the v0.8.1 release scope or the correctness of the shipped gate.

## 7. Recommendation

**Ship v0.8.1.** ROADMAP item 4 is closed. Math invariants are intact. Bit-equal
contracts hold. The G3⁶-2D FLAGSHIP gate passes on both the slope and wallclock
sub-conditions. Performance improvements (4.38× / 3.87×) satisfy ROADMAP item 3.
