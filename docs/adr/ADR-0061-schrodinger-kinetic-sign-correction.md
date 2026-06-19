# ADR-0061 — Schrödinger Kinetic Off-Diagonal Sign Correction

- **Date**: 2026-05-22
- **Status**: Accepted
- **Supersedes**: nothing
- **Amends**: ADR-0057 (Schrödinger Crank-Nicolson Cayley map) — Amendment 2

## Decision

v2.2.0 `SchrodingerChernoff::apply_strang_step` passed positive `a_off = half_tau · a₀ / dx²` to `cn_kinetic_step_f64`, yielding matrix `A = +τK/2` with positive eigenvalues. The Cayley update `(I − iA)(I + iA)⁻¹` then implemented `e^{+iτK}` (time-reversal). Unitarity gates G18a/b passed because norm is preserved under sign flip, but the harmonic oscillator period gate G19 surfaced a 4×10³ accuracy violation. v2.2.1 negates `a_off` so `A = −τK/2 = +τ·∂²ₓ/2`, restoring forward-time evolution `e^{-iτK}` per math.md §17.4. Identical in spirit to the well-known FD-Schrödinger sign-of-Laplacian gotcha. ADR-0057 design (CN Cayley) unchanged; only the off-diagonal sign passed to the helper changed. New non-`#[ignore]` regression test `regression_iter5_schrodinger.rs` guards against re-introduction.

## Consequences

- v2.2.0 numerical results from `SchrodingerChernoff` (any caller) are INCORRECT. No external callers exist (rlib-only, no bindings until v2.2.0 graph FFI which routes through different kernels).
- All 4 B.3 v2.2 `#[ignore]` gates (G17 K=6 slope, G18c slope f64, G18a/b unitarity, G19 period) reconfirmed PASS at v2.2.1.
- SemVer: PATCH (correctness fix, no API change).
