# ADR-0004 — f64 monomorphic in v0.1.0

**Status**: Accepted
**Date**: 2026-04-28
**Authors**: ai-solutions-architect (Stage 3)
**Resolves risk**: R2 (technical-constraints.md)

## Decision

The `State` trait, `ChernoffFunction` trait, and all concrete types use `f64`
directly — no associated `type Scalar: Float` and no `<F: Float>` generics.
A generic-over-Float refactor pays its weight only when there is a concrete
client demanding `f128`, interval arithmetic, or `Complex<f64>`; v0.1.0 has
none. Pre-emptively generalising would (i) explode trait-bound noise across
every signature, (ii) require pinning `num-traits = "0.2"` and importing
`Float`, `Zero`, `One` everywhere, (iii) make rustdoc almost unreadable for
the heat-Mehler example. The clean refactor target is v0.5 when the adaptive
controller (`remizov-adaptive`) lands with a real motivation for non-f64
scalars (interval arithmetic for guaranteed bounds). Public API users will
see one breaking change at v0.5; we accept that cost in exchange for v0.1.0
clarity. `num-traits` MAY still appear as an indirect dep via PRD §5.3 hints
but is NOT mandatory for the v0.1.0 surface.
