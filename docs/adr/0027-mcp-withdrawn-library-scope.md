# ADR-0027 — MCP withdrawn for library scope

**Status**: Accepted (waiver, supersedes ADR-0002)
**Date**: 2026-05-08
**Authors**: ai-solutions-architect
**Guardrail waived**: #6 (MCP Everywhere) — full, scope `v0.x` and `v1.x`
**Constitution override**: #2 of 2 active (see `.dev-docs/constitution.md`)

## Decision

`semiflow-core` is a `no_std + alloc` Rust `rlib` with no daemon, no runtime,
no I/O, no log buffer, and no service surface — every "operation" is a
synchronous function call inside the caller's process. Each guardrail-#6
endpoint is therefore either tautological or already covered by the toolchain:
`health.get` is constant `Ok` for a pure function, `control.start/stop/reload`
is meaningless without a runnable, `logs.tail` has no buffer in `no_std`,
`contracts.list/describe` is already published by `cargo doc` (rustdoc), and
`test.run`/`metrics.snapshot` is already provided by `cargo test` and
`cargo bench`/Criterion. ADR-0002's v0.1.0 deferral (which promised MCP at
v0.9.0 alongside FFI) is therefore superseded: there is nothing for an MCP
server to introspect across the v0.x and v1.x release lines, the FFI/PyO3/WASM
crates landing in v0.10.0 (ADR-0028) inherit the same library-call semantics,
and standing up a ceremonial MCP shim would burn dependency budget against
guardrail #1. **We withdraw guardrail #6 entirely** through v1.x; if a
future service-shaped product (e.g. a daemon for distributed Chernoff
workers, or a long-running `remizov-server` exposing `evolve` over RPC)
ever materialises, MCP can be re-introduced under a fresh ADR scoped to
that artifact. supersedes ADR-0002.
