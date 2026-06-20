# ADR-0171 — S³ non-grid engines get an opaque carrier-handle C-ABI; ragged arrays flatten as data+offsets; Py/WASM mirror it

**Status:** IMPLEMENTED · **Date:** 2026-06-19 · **Wired:** 2026-06-20 · **Branch:** `experiment/triz-s3-curse-escape`
**Theme:** v9.2.0 — binding surface for the three S³ non-grid carriers (TtState/TtChernoff, CoupledTtChernoff, GridlessChernoff/MeasureState)
**Cross-refs:** ADR-0028 (FFI/PyO3/WASM split — C ABI is the canonical source, Py/WASM mirror; opaque-handle + `SemiflowStatus` + `catch_panic!` baseline), ADR-0076 (additive `_v3` binding surface), ADR-0162 (`CoupledTtChernoff` fail-loud construction), ADR-0169 (boundary-as-type S³ public scope).
**Contract:** `contracts/semiflow-ffi.s3-carrier-handle.yaml`.
**Implementation note (2026-06-20):** FFI and WASM wiring is complete.
`crates/semiflow-ffi/src/lib.rs` declares the `tt_ffi`, `tt_coupled_ffi`, and
`gridless_ffi` modules unconditionally; `crates/semiflow-wasm/src/lib.rs` mirrors
them with wasm-bindgen JS classes (`TtState`, `TtEvolver`, `TtCoupledEvolver`,
`MeasureState`, `GridlessEvolver`).  PyO3 wiring was already present.
Smoke tests: `crates/semiflow-ffi/tests/ffi_s3_smoke.rs` and
`crates/semiflow-wasm/tests/s3_smoke.rs`.

## Decision

The grid evolvers pass `&[f64]` through the value-buffer `smf_evolver_*_v3` surface; the three S³ engines carry non-grid state (TT cores; weighted-Dirac particles) and so get a NEW **opaque carrier-handle** surface modelled exactly on `SmfEvolverV3` / `SmfWentzellEvolverV3`: each engine and each state is a zero-sized `#[repr(C)] struct Smf* { _private: [u8;0] }` returned as `*mut Smf*` (backed by a heap `Box<…Inner>`), constructed via an out-param `*mut *mut Smf*`, advanced in place behind its handle, read out only as scalars/marginals, and dropped by a null-safe `smf_*_free`; every `extern "C"` returns `SemiflowStatus` (reused verbatim — no new variants), null-checks before `catch_panic!`, and the **dense `n^d` tensor / `3^d` particle tree is NEVER materialised across the ABI** (the curse-escape is structural, not incidental). C has no ragged arrays, so any `Vec<Vec<f64>>` (separable IC slices, per-axis functionals) crosses as a flat `data: *const f64` plus a prefix-sum `offsets: *const usize` (length `n_axes+1`, `offsets[0]=0`, axis `j` = `data[offsets[j]..offsets[j+1]]`) — the CSR row-pointer convention, which uniquely and allocation-free-ly handles ragged per-axis lengths; `CouplingTopology` crosses as a `u32` tag plus two parallel `pairs_jk`/`pairs_rho` arrays (usize and f64 cannot share one buffer), `ParticleReduction` as a tag + scalar `cap`, and particles as a row-major `positions[n_part*D]` + `weights[n_part]` pair; the v9.1.0 fail-loud walls (drift `b≠0`, non-adjacent pairs, non-SPD blocks) are pre-checked in the binding and surfaced as `OutOfDomain`, never panicked; and per ADR-0028 `semiflow-py` (PyO3) and `semiflow-wasm` (wasm-bindgen) MIRROR this one canonical surface 1:1 (same operations, same flattening, idiomatic host types) rather than inventing divergent ad-hoc APIs.
