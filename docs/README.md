# SemiFlow documentation

Documentation is split into **user-facing** material (how to use the library) and
**developer-facing** material (how the library is built and decided). For the API
reference, see [docs.rs/semiflow](https://docs.rs/semiflow).

## For users

| Document | Purpose |
|----------|---------|
| [Quickstart](QUICKSTART.md) | Smallest runnable heat-equation program |
| [User Guide](USER_GUIDE.md) | Use-case-driven tour ("I want to solve …") |
| [Install](INSTALL.md) | Toolchain and installation pointers |
| [Bindings Guide](BINDINGS.md) | C, Python, and WASM usage |
| [Project README](../README.md) | Install commands, [feature flags](../README.md#feature-flags), [`no_std`](../README.md#no_std), [engine catalogue](../README.md#engine-catalogue), bindings overview |
| [`examples/`](../crates/semiflow/examples/README.md) | Worked examples, beginner → advanced |
| [precision-policy.md](precision-policy.md) | Accuracy / performance trade-offs |
| [python-coverage.md](python-coverage.md) | Binding parity matrix (Rust / C / Python / WASM) |
| [api-stability.md](api-stability.md) | Versioning and API-stability policy |
| [Role and roadmap](semiflow_role_and_roadmap.md) | Positioning, honest scope limits, roadmap pointers |

## For developers & contributors

| Document | Purpose |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Workflow, conventions, ADR process |
| [release-process.md](release-process.md) | How releases are cut |
| [`readme/`](readme) | Sources of the generated READMEs — edit here, then `cargo run -p xtask -- readme` |
| [`adr/`](adr) | Architecture Decision Records (the "why" behind the design) |
| [`migration/`](migration) | API-evolution notes across versions |
| [`audit-findings-*.md`](.) | Per-release math-fidelity audit records |
| [SECURITY.md](../SECURITY.md) | Vulnerability disclosure |
| [`contracts/`](../contracts) | Contract-first IDL / property specs |

## Mathematical specification

The normative mathematical specification lives in
[`contracts/semiflow-core.math.md`](../contracts/semiflow-core.math.md). The method
implements Theorem 6 of Remizov (2025), *Vladikavkaz Math. J.* 27(4), 124–135.
