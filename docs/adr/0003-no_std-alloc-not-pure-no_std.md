# ADR-0003 — `no_std + alloc`, not pure `no_std`

**Status**: Accepted
**Date**: 2026-04-28
**Authors**: ai-solutions-architect (Stage 3)
**Resolves risk**: R1 (technical-constraints.md)

## Decision

`semiflow-core` is `#![no_std]` with `extern crate alloc` and uses
`alloc::vec::Vec<f64>` for grid storage. We do NOT pursue pure no_std with
`heapless::Vec<N>` in v0.1.0 because (a) `N=1000` (frozen test grid) would
either waste a fixed const-generic slot or force compile-time monomorphisation
across grid sizes, and (b) reusing one `Vec` across the n=1000 iterations of
`ChernoffSemigroup::evolve` is the natural in-place pattern that delivers
G6 performance (<10 ms median) without SIMD or Rayon. `ndarray` is dropped
entirely (no `std`, no slow build, no transitive dep blow-up). A pure
`heapless` variant for embedded/WASM is a v0.9 concern that will be authored
when WASM bring-up forces the issue. This is consistent with framework
guardrail #1 (minimalism) — `alloc` is core's stdlib, not a third-party
crate, and adds zero dependency budget.
