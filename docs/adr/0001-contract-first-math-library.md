# ADR-0001 — Contract-first adapted for a Rust scientific math library

**Status**: Accepted
**Date**: 2026-04-28
**Authors**: ai-solutions-architect (Stage 3)

## Decision

`semiflow-core` has no HTTP/RPC surface, so the framework's default contract
artifact (OpenAPI YAML) is inapplicable. We adopt a four-file contract bundle
under `contracts/`: `semiflow-core.traits.yaml` (JSON-Schema-flavored Rust
trait/struct IDL), `semiflow-core.math.md` (KaTeX-rich semantic spec, normative
for formulas and theorems), `semiflow-core.errors.yaml` (the closed `SemiflowError`
enum), and `semiflow-core.properties.yaml` (proptest invariants). Codegen is
performed by `cargo xtask gen-stubs` (DevOps writes the xtask in Stage 5),
emitting Rust skeletons with rustdoc into `crates/semiflow-core/src/_stubs/`
that the Engineer fills in. This satisfies framework guardrail #2 (Contract-first)
for non-network libraries and is the canonical pattern for any future
`remizov-*` math crate. Nothing else changes: message-flows, adapters,
capabilities, MCP, build, ADRs all retain framework-default locations.
