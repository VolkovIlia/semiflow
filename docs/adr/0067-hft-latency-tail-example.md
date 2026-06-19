# ADR-0067 — latency_tail.rs example

- **Status**: ACCEPTED (v2.5.1)
- **Date**: 2026-05-26
- **Wave**: v2.5.1 (cache-residency + HFT benchmarks)
- **Authors**: claude (docs-writer)
- **Depends on**: ADR-0005 (cubic-Hermite spatial 4th-order — used by
  `Diffusion4thChernoff`), ADR-0066 (tracking-alloc feature),
  ADR-0028 (FFI/PyO3/WASM release).
- **Mathematical foundation**: `Diffusion4thChernoff` per ADR-0005; CEV model
  a(x) = ½σ²e^{(2β-2)x} on log-spot grid x = log(S/K) (Emanuel-MacBeth 1982).

## Context

The cache-residency experiment (ADR-0066 Phase 1–4) established that the CEV
hot loop is L1d-resident and cache-optimal. An HFT-style benchmark is needed to
demonstrate the operational consequence — p99.9 tail latency — and to provide a
reproducible companion to the publications-repo experiment data. The natural
vehicle is an `examples/` binary that:

1. Loads a canonical GBM trajectory (1M f64 spot values, PCG64 seed
   `0xC0FFEE_BABE_DEAD_BEEF`, sha256
   `beac36ebe1c541bff4641debe170cb00b35aedb0e893ac8603470d98508b2863`).
2. Prices the CEV European call at each tick via `Diffusion4thChernoff` N=1536
   on a log-spot grid (passes the 5e-4 calibration gate vs Emanuel-MacBeth 1982
   ncx2 oracle; V2/V3 QuantLib BS proxies fail the gate and are excluded from
   the primary comparison).
3. Records per-tick latency with nanosecond resolution, computes HDR percentiles
   (p50/p99/p99.9/p99.99), and emits a JSONL result file.

## Decision

Add `crates/semiflow-core/examples/latency_tail.rs` (~569 LoC). Key design
choices:

- **ScratchPool pre-warm**: `GridFn1D` scratch buffers allocated once before the
  hot loop; zero allocations per tick (verified via ADR-0066 `tracking-alloc`
  and the `--measure` flag on `cev_european_call`).
- **Matched-accuracy comparison baseline**: QuantLib V3 Schroder ncx2 closed-form
  (`NonCentralCumulativeChiSquareDistribution`) is the only QL variant that passes
  the 5e-4 gate (calibration error 1.74e-10). V1/V2 FDM Black-Scholes proxies are
  retained in the harness with an explicit `notes` field in the JSONL output but
  are NOT cited as the primary comparison.
- **Output format**: JSONL per rep with `{rep, p50_ns, p99_ns, p999_ns, p9999_ns,
  rss_kb, calibration_err}`. Aggregated summary (median + 95% CI across reps)
  written to `data/phase-d-summary.json` in the publications repo.
- **Binary location**: `examples/` rather than `src/` — this is an end-to-end
  demonstrator, not part of the library API.

## Consequences

- Provides a reproducible head-to-head comparison: RC p99.9 = 45 ns vs QL V3
  p99.9 = 6711 ns → 149× tail-latency advantage (95% CI [145×, 153×]) at
  matched accuracy; RSS 18 MB vs 76 MB (4.2× lower). Source data in
  `remizov-publications/benchmarks/hft-latency-tail/data/phase-d-summary.json`.
- Binary is built only in `--release` (hot-loop latency is not meaningful in
  debug). Run with `cargo run --release --example latency_tail`.
- No changes to `crates/semiflow-core/src/`; the example imports only public API.
- ADR-0066 `tracking-alloc` is a dev-dependency of this example; not required for
  the primary latency measurement loop.
- Phase 4 A/B testing (ADR-0066) found NULL speedup from `#[inline]` and Y-pass
  chunking — the library was already cache-optimal. The `#[inline]` annotations on
  five hot-path entry points (apply_into_x, apply_into_y, apply_strang2d_into,
  apply_strang3d_into, run_axislift_into_2d) are retained to preserve hot-path
  inlining at module boundary without shipping a performance regression.
