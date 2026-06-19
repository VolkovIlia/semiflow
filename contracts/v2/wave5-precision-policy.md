# Wave 5 Contract: Precision-Policy Bands (Sister Contract)

**Status**: NORMATIVE
**ADR**: docs/adr/0046-precision-policy-bands.md
**Parent contract**: contracts/v2/wave5-bindings.md
**Scope**: semiflow-core v2.0 Wave 5 — f32 vs f64 slope-gate bands.

This document holds the machine-readable band table for ADR-0046. Engineer pass
ingests this when wiring `tests/generic_float_strang.rs`. Reviewer pass cites
this when checking precision claims.

---

## §1 — Sympy gates (oracle-class)

| Gate | f64 | f32 |
|---|---|---|
| T9N_* (var-a 1D Taylor) | NORMATIVE (exact) | VACUOUSLY SATISFIED |
| T10N_* (3D tensor Taylor) | NORMATIVE (exact) | VACUOUSLY SATISFIED |
| T11N_* (non-sep 2D Taylor) | NORMATIVE (exact) | VACUOUSLY SATISFIED |

**Rationale**: sympy/mpmath uses arbitrary precision. f32 rounding is a
property of the LLVM runtime, not the Taylor expansion. Forcing sympy to
round to 23-bit mantissa invents a fake oracle (mpmath's IEEE-754 binary32
emulator is not bit-equal to LLVM `f32`).

VACUOUSLY SATISFIED means: gate does not run on f32, and the absence is not
a failure. Same constitutional status as guardrail #6 (MCP) under
Override #2.

---

## §2 — Slope gates (numerical self-convergence)

| Gate | Description | f64 band | f32 band | f32 disabled? |
|---|---|---|---|---|
| G3 | 1D Strang, order 2 | ≥ −1.95 | ≥ −1.80 | no |
| G3⁴ | 1D 4th-order, ζ⁴ Chernoff | ≥ −3.85 | ≥ −3.50 | no |
| G3⁶ | 1D 6th-order, ζ⁶ Chernoff | ≥ −5.80 | — | **YES** (no asymptotic window at 23-bit) |
| G3⁶-2D | 2D flagship (ADR-0020) | ≥ −1.95 in τ | — | **YES** (flagship is f64-only) |
| G4_NS2D_aniso | Non-sep 2D (ADR-0023) | ≥ −1.95 | ≥ −1.80 | no |
| G5_3D | 3D Strang (ADR-0024) | ≥ −1.95 | ≥ −1.80 | no |

### §2.1 Sweep parameters (per precision)

| Param | f64 | f32 |
|---|---|---|
| τ sequence | {1e-2, 5e-3, 2.5e-3, 1.25e-3, 6.25e-4} | {1e-1, 5e-2, 2.5e-2, 1.25e-2, 6.25e-3} |
| τ_min recommended | 1e-4 | 5e-3 |
| Reference grid N (1D self-conv) | 4096 | 1024 |
| Reference grid N (2D self-conv) | 511 | 255 |
| Reference grid N (3D self-conv) | 127 | 63 |
| n_steps for slope fit | 200 | 100 |

### §2.2 Rounding-floor derivation (informative)

Mantissa width drives the rounding floor:

- f64: 52 bits → ε ≈ 2.2e-16; per-step rounding ≈ N · ε ≈ 1e-13 on a 1024² grid;
  accumulated over 200 steps ≈ 2e-11. **Always below truncation error** in
  the τ sweep above. Slope gate ≥ −1.95 (one mantissa-decade margin).
- f32: 23 bits → ε ≈ 1.2e-7; per-step rounding ≈ N · ε ≈ 6e-5 on 1024²;
  accumulated over 100 steps ≈ 6e-3. **Comparable to truncation at τ_min = 5e-3**
  for order 2. Slope gate ≥ −1.80 widens the band to absorb the rounding
  pre-asymptotic regime.

For order 6 on f32, the rounding floor (~6e-3 accumulated) is **larger** than
the leading-order truncation (~τ⁶ at τ = 1e-1 is ~1e-6 — totally swamped).
G3⁶ has no asymptotic window on f32; the gate is disabled rather than
relaxed.

---

## §3 — Bit-equality (parallel = serial)

| Gate | f64 | f32 |
|---|---|---|
| `strang2d_parallel_bit_equal` | NORMATIVE (byte-equal vs v1.0.0 SHA-256 manifest) | NORMATIVE (f32 parallel = f32 serial byte-equal within precision class — no cross-precision) |
| `strang3d_parallel_bit_equal` | NORMATIVE | NORMATIVE |
| `diffusion4_unit_simd` (ADR-0019) | NORMATIVE | DOES NOT APPLY (SIMD path is f64-only) |

**Note**: cross-precision comparison (f32 parallel ≈ f64 parallel) is **not
gated** at any tolerance. Mixed-precision composition is out of Wave 5 scope.

---

## §4 — Implementation requirements

### §4.1 `tests/generic_float_strang.rs`

```rust
mod bands {
    pub const F64_STRANG: f64 = -1.95;
    pub const F32_STRANG: f64 = -1.80;
    pub const F64_FOURTH: f64 = -3.85;
    pub const F32_FOURTH: f64 = -3.50;
    pub const F64_SIXTH:  f64 = -5.80;
    // No F32_SIXTH — gate is disabled
}

#[test] fn strang2d_self_conv_f64() {
    let s = strang2d_slope::<f64>(&TAU_F64, 511, 200);
    assert!(s <= bands::F64_STRANG, "f64: slope={s} > {}", bands::F64_STRANG);
}

#[test] fn strang2d_self_conv_f32() {
    let s = strang2d_slope::<f32>(&TAU_F32, 255, 100);
    assert!(s <= bands::F32_STRANG, "f32: slope={s} > {}", bands::F32_STRANG);
}

// Same pattern for Strang3D.
```

Sweep constants `TAU_F64`, `TAU_F32` mirror §2.1 above.

### §4.2 `tests/strang2d_parallel_bit_equal_f32.rs`

Mirror of the existing f64 test, but with `F = f32` and a separate
SHA-256 manifest `tests/fixtures/strang2d_parallel_f32_v2.sha256` (NORMATIVE
new file; first Wave 5 commit snapshots f32 serial outputs as the truth).

### §4.3 Rustdoc on disabled gates

`Diffusion6thChernoff<F>` rustdoc cites ADR-0046 §2 row "G3⁶":

> Sixth-order spatial accuracy is **mathematically** preserved for any
> `F: SemiflowFloat`, but **numerically demonstrable** only on `F = f64`. At
> `F = f32`, the rounding floor (~6e-3 accumulated over a 100-step sweep)
> exceeds the leading τ⁶ truncation term in any sub-1.0 τ range. The slope
> gate `G3⁶` is therefore disabled on f32 (see ADR-0046, gate table). Users
> needing 6th-order asymptotic verification MUST run on f64.

### §4.4 `docs/precision-policy.md` (user-facing)

Single-screen summary table cribbed from ADR-0046 §3.4. Linked from
`crates/semiflow-core/src/lib.rs` crate-level rustdoc and from
`crates/semiflow-py/README.md`.

---

## §5 — Acceptance gate (precision-policy subset)

This contract is satisfied when:

1. `tests/generic_float_strang.rs` passes both f64 and f32 bands per §2.
2. `tests/strang2d_parallel_bit_equal_f32.rs` byte-equal within f32 class.
3. `docs/precision-policy.md` exists and is linked from `lib.rs`.
4. `Diffusion6thChernoff` rustdoc cites ADR-0046 §2 G3⁶ row.
5. ADR-0046 references this contract from its `References` section.

---

## §6 — Out of scope

- Mixed-precision composition (f32 input, f64 internal compute) — separate
  ADR if requested.
- f16 / bf16 — `SemiflowFloat` sealed at `{f32, f64}` per ADR-0026.
- Sympy on f32-rounded oracles — see §1 rationale.
- Cross-precision bit-equality — see §3 note.
