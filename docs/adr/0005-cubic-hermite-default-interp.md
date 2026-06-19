# ADR-0005 — Cubic Hermite (Catmull-Rom) is the default interpolation kernel

**Status**: Accepted
**Date**: 2026-04-28
**Authors**: ai-solutions-architect (Stage 3)
**Resolves risk**: R7 (technical-constraints.md)

## Decision

`InterpKind::CubicHermite` (Catmull-Rom 4-point) is the default for
`Grid1D::sample`. Linear interpolation (`InterpKind::Linear`) is gated behind
the `linear-interp` cargo feature and exists only for debug/sanity. Catmull-Rom
is C^1, costs four multiply-adds per sample, has a closed-form expression
without per-point spline solves, and on a smooth Gaussian over `N=1000` nodes
delivers point-wise error well below `10^-6` — comfortably under G1's `10^-4`
budget at `n=100`. Linear interpolation cannot meet G1: its truncation error
scales like `(dx)^2 * ||f''||/8` ≈ `(0.02)^2 * 2 / 8 = 1e-4`, which exhausts
G1's tolerance before the Chernoff truncation contributes. Engineer MUST
implement Catmull-Rom from scratch (≤50 lines) and forbid any cubic-spline
solver dep; this keeps us inside the ≤3-direct-deps budget (G7) and the
500-line `grid.rs` budget (G10).
