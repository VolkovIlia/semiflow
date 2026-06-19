# ADR-0060 — Parallel benchmark suite for thread-scaling

- **Status**: ACCEPTED
- **Date**: 2026-05-21
- **Wave**: v2.2 Wave C (refactor + bindings, optional addition)
- **Authors**: ai-solutions-architect
- **Depends on**: ADR-0018 (`Strang2D` parallel SIMD), ADR-0036 (parallel
  1D Chernoff), ADR-0051 (`MagnusGraphHeatChernoff`).
- **Mathematical foundation**: none — benchmark harness, not a library
  feature.

## Context

v0.8.0 shipped parallel `Strang2D::apply` (ADR-0018) and v0.9.0
shipped parallel `Strang3D::apply` (mirror, ADR-0024). v2.0 added
parallel 1D Chernoff (`parallel_eval`, ADR-0036) for unbounded-domain
applications. v2.1 shipped sparse-graph Magnus K=4 (sequential).

We have NO formal thread-scaling benchmark that quantifies:
- Speedup curves (1 → 2 → 4 → 8 → 16 threads) per kernel.
- Strong-scaling efficiency vs problem size.
- Cross-platform (x86_64 vs aarch64) consistency.

The iter-3 bench (MEMORY.md "head-to-head bench iter 1" + iter-3
H-MEM Pareto results) measured **memory** Pareto dominance — SemiFlow's
flagship hypothesis — but did not include thread-scaling.

## Decision

**OPTIONAL** (engineer may skip if v2.2 LoC budget tight): add a
benchmark harness measuring thread-scaling 1 → 16 for the parallel
kernels:

```text
benchmarks/v2_2_0/
├── parallel_bench.rs              (NEW, ~280 LoC, criterion benchmark)
├── baseline-v2_2_0.json           (NEW, output snapshot)
├── parallel_bench_report.md       (NEW, markdown report)
└── threads_vs_kernel_matrix.csv   (NEW, raw data)
```

The schema for `baseline-v2_2_0.json`:

```json
{
  "version": "2.2.0",
  "host": "bestfriend",
  "cpu": "i7-12700K, 12C20T",
  "rust_version": "1.78.0",
  "date": "2026-05-21",
  "kernels": [
    {
      "name": "Strang2D::apply",
      "feature_flags": ["parallel"],
      "n": 1024,
      "thread_counts": [1, 2, 4, 8, 12, 16],
      "time_per_step_ns": [12340, 6420, 3520, 1980, 1450, 1380],
      "allocs_per_step": [0, 0, 0, 0, 0, 0]
    },
    {
      "name": "MagnusGraphHeatChernoff::apply_into_at",
      "feature_flags": ["parallel"],
      "n_nodes": 65536,
      "thread_counts": [1, 2, 4, 8, 12, 16],
      "time_per_step_ns": [...],
      "allocs_per_step": [0, ...]
    }
  ]
}
```

Kernels measured:
- `Strang2D::apply` (ADR-0018; parallel f64-only)
- `Strang3D::apply` (ADR-0024; parallel f64-only)
- `parallel_eval_1d` for `DiffusionChernoff` (ADR-0036)
- `GraphHeatChernoff::apply_into` (v2.1; serial-only — single-thread baseline)
- `MagnusGraphHeatChernoff::apply_into_at` (v2.1; serial-only — baseline)
- `MagnusGraphHeat6thChernoff::apply_into_at` (ADR-0056, if Wave B
  parallel impl ships in v2.2; else single-thread baseline)

**Comparison**: speedup S(p) = T(1) / T(p); efficiency η(p) = S(p) / p.
Targets per kernel (rough; document actual in `parallel_bench_report.md`):

| Kernel | S(8) target | η(8) target |
|---|---|---|
| `Strang2D` | ≥ 6.0 | ≥ 0.75 |
| `Strang3D` | ≥ 6.0 | ≥ 0.75 |
| `parallel_eval_1d` | ≥ 5.0 | ≥ 0.62 |

Sparse-graph kernels (single-thread): baseline only; no speedup
because no parallel impl yet (defer to v2.3+).

## Rationale

- **Memory-first measurement remains primary** (per MEMORY "bench
  memory-first presentation"). This ADR adds thread-scaling as
  **secondary** evidence; memory plots still come first in
  presentations.
- **Single source of thread-scaling truth.** Without this benchmark,
  customers can only infer scaling from anecdote — bad for adoption.
- **Reproducibility**: `cargo bench -p remizov-bench parallel_bench`
  (single command, ADR-0001 build path).
- **Optional**: doesn't block v2.2 release. Can ship in v2.2.1 if
  Wave A + B + C take precedence.

## Consequences

- New benchmark module (~280 LoC); under file cap, lives in `benchmarks/`
  not `crates/semiflow-core/src/`.
- `xtask` may grow a new `bench-parallel` subcommand (~50 LoC).
- CI `bench.yml` adds optional job (manual trigger only; expensive).

## Acceptance gates

- **No CI gate** (benchmarks are informational, not regression-gated).
- **G_bench_artefact gate** (NORMATIVE — only if Wave C ships this):
  `benchmarks/v2_2_0/baseline-v2_2_0.json` is committed at v2.2.0 tag;
  JSON schema validation passes.

## Out of scope

- Parallel sparse-graph kernels. v2.3+.
- Parallel `NonSeparableMixedChernoff` (`NS2D-aniso`). The 5-leg composition
  `apply_five_leg` calls `AxisLift::apply()` (serial) for all four axis passes.
  `AxisLift` has no parallel impl; the `Strang2D` parallel infrastructure
  (`parallel_x_pass` / `parallel_y_pass`) operates directly on the inner 1D
  operator and cannot be reused without adding a parallel sweep API to `AxisLift`.
  Measured η(8) = 0.126 (INTRINSIC_LIMIT). Deferred to v2.3+.
  See `docs/perf/scaling-v2_2_0.md` §NS2D-aniso.
- Per-platform bench (M1 macOS, ARM linux). v2.2 single-host on bestfriend.
- iter-4 head-to-head benchmarks. Separate project per iter-3 deliverable.

## Risks

| # | Risk | Mitigation |
|---|------|------------|
| R1 | Bench result regressions vs v0.11.0 baseline detected | Compare to v0.11.0 in `parallel_bench_report.md`; flag if S(8) regresses by >10% on any kernel. |
| R2 | thread-pinning vs OS scheduler interference | Document `taskset` recipe + `nice` invocation in `parallel_bench_report.md`. |

## Cost (LoC estimate)

| Artefact | LoC |
|---|---|
| `benchmarks/v2_2_0/parallel_bench.rs` | ~280 |
| `benchmarks/v2_2_0/parallel_bench_report.md` | ~250 |
| `xtask` extension (`bench-parallel`) | ~50 |
| ADR-0060 (this) | ~170 |
| **Total** | **~750** |

## References

- ADR-0018 (`Strang2D` parallel).
- ADR-0036 (1D parallel `parallel_eval`).
- ADR-0051 (Magnus K=4 graph; serial).
- MEMORY.md "head-to-head bench iter 1" + iter-3 H-MEM.

## Amendment 1 (2026-05-21) — NS2D-aniso parallel explicitly deferred to v2.3

**Context**: Phase 5 bench QA empirically measured `NonSeparableMixedChernoff`
thread-scaling efficiency at η(8) = 0.126, confirming INTRINSIC_LIMIT.

**Decision**: `NonSeparableMixedChernoff` parallel impl (NS2D-aniso) is
explicitly OUT OF SCOPE for v2.2. The "Out of scope" section above documents
the root cause: `apply_five_leg` calls `AxisLift::apply()` (serial) for all
four axis passes; `AxisLift` has no parallel impl. Measured η(8) = 0.126
confirms the bottleneck. Deferred to v2.3+ pending a parallel `AxisLift` sweep
API (see `docs/perf/scaling-v2_2_0.md` §NS2D-aniso).
