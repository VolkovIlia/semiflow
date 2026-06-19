# ADR-0022 — Parallel Strang2D tile-scratch reuse

**Status**: Accepted
**Date**: 2026-05-07
**Authors**: agentic-engineer
**Cross-refs**: ADR-0018 (parallel Strang2D — original allocation strategy),
ADR-0020 Amendment 3 (G3⁶-2D FLAGSHIP, lists this ADR as prerequisite),
ROADMAP.md v0.8.0 item 3 (≥5× combined speedup); commit `eae2b7a`.

v0.8.1 Block A identified three per-iteration heap allocations in the hot loops
of `strang2d_parallel.rs` that were unnecessary: the row buffer in `x_pass_chunk`,
the column buffer in `y_apply_cols`, and the scratch vector threaded through
`parallel_y_pass`. Each was allocated fresh on every call to the parallel Strang2D
sweep, creating allocator contention and cache pressure at large N.

The decision is to eliminate these hot-path allocations via `core::mem::take`:
`x_pass_chunk` takes its row_buf from a pre-allocated pool (returned after use),
`parallel_y_pass` accepts `&mut Vec<f64>` scratch threaded in from
`Strang2D::apply_parallel` (which owns the allocation across Chernoff steps), and
`y_apply_cols` takes its col_buf from the same scratch using the same take-and-return
pattern. The `pub(crate)` signature of `parallel_y_pass` changes to accept the
scratch argument; this is an internal interface with no public API impact.

Bit-equal correctness is preserved by construction: the values written to the output
buffer are computed from the same arithmetic operations in the same order as v0.8.0;
only the allocation site moves. All existing bit-equal contracts confirmed at HEAD:
`STRANG2D_PARALLEL_BIT_EQUAL` (3/3 pass), `SIMD_BIT_EQUAL_PARALLEL` (3/3 pass),
`v0_5_0_regression_bit_equal` (4/4 pass). Bench evidence at N=1600²:
`benches/heat_2d` speedup **4.38×**, `benches/advdiff_2d` speedup **3.87×** vs
v0.7.0 serial baseline (ROADMAP item 3 target was ≥1.8× incremental over v0.8.0;
both exceed it). No new unsafe introduced; `unsafe` scope remains confined to
`src/simd/{x86_64,aarch64}.rs` per ADR-0019. Dependencies unchanged at 2
(`num-traits`, `libm`). Source delta: 44 LoC in `strang2d_parallel.rs`,
9 LoC in `strang2d.rs`.

---

## Amendment 1 — v0.13.0 Wave A2 propagation (Strang3D serial path)

**Status**: Proposed 2026-05-19 for v0.13.0.

**Context**: Iter-3 bench (`BENCHMARKS-STATUS.md`) showed F7 (3D heat, N=32) ran on Strang3D **serial** path because N=32 < `2*MIN_PENCILS_PER_THREAD=64`. Serial path (`strang3d.rs:415-419`) allocates 5 fresh `GridFn3D` per Strang step; at M≈300 steps the churn ≈ 1.4 s of 7.8 s wall (~18% genuine speedup potential). Parallel path (lines 452-454) already applies the original ADR-0022 scratch pattern.

**Decision**: Mirror the parallel-path scratch pattern in `Strang3D::apply` serial branch: pre-allocate `y_scratch: Vec<F>` and `z_scratch: Vec<F>` once at constructor, store on `Strang3D` struct (RefCell-wrapped if `&self apply` retained, otherwise `&mut self`), reuse across all steps. Add `STRANG3D_SERIAL_SCRATCH_BIT_EQUAL` regression test: byte-identical output vs pre-amendment scalar reference at N=32. Wave A1 (1D `StrangSplit` scratch pool) is **explicitly DROPPED** — researcher confirmed 1D alloc overhead is 0.2-0.7% of wall, not the bottleneck.

**Consequences**: F7 target 15-25% wall-time reduction at N=32. Memory cost: 2·N³·8 bytes permanent state per Strang3D instance (~256 KB at N=32; ~2 MB at N=64). 1D StrangSplit is left unchanged — alloc-bound is not its bottleneck (real culprit is cubic-Hermite sample() chain in DiffusionChernoff, deferred to post-v0.13). Compatible with existing parallel-path scratch (no double-allocation). Citation: researcher memo `/home/volk/.claude/plans/vast-riding-bachman.md` Wave A weakening analysis.
