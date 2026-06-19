# ADR-0002 — MCP server deferred to v0.9.0 (FFI/WASM milestone)

**Status**: Superseded by ADR-0027 (2026-05-08)
**Superseded**: 2026-05-08 — see ADR-0027 (MCP withdrawn entirely; library has no runtime). Body retained as historical context.
**Date**: 2026-04-28
**Authors**: ai-solutions-architect (Stage 3)
**Guardrail waived**: #6 (MCP Everywhere) — partial, scope-limited

## Decision

v0.1.0 is a pure-CPU `rlib` with no service, no daemon, no runtime, no I/O.
There is nothing for an MCP server to introspect or control: `health.get`
would be tautologically `Ok`, `control.start/stop/reload` is meaningless for
a synchronous library function, `logs.tail` has no log buffer (no_std).
Standing up an MCP shim in v0.1.0 would be ceremonial weight against suckless
guardrail #1 (one-screen build path, <=3 deps, <500-line files). We therefore
WAIVE guardrail #6 for v0.1.0 only and re-introduce MCP at v0.9.0 alongside
the `semiflow-ffi` C ABI and `semiflow-wasm` WASM crates — that milestone ships
a runtime that genuinely benefits from health/contracts/control/test endpoints.
The deferred design sketch lives in `.dev-docs/mcp/server-design.md`. This
waiver counts as 1 of the project's permitted ≤3 framework overrides.
