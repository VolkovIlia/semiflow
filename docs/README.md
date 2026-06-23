# SemiFlow documentation

Documentation is split into **user-facing** material (how to use the library) and
**developer-facing** material (how the library is built and decided). For the API
reference, see [docs.rs/semiflow](https://docs.rs/semiflow).

## For users

| Document | Purpose |
|----------|---------|
| [Quickstart](QUICKSTART.md) | Smallest runnable heat-equation program |
| [User Guide](USER_GUIDE.md) | Use-case-driven tour ("I want to solve …") |
| [Install](INSTALL.md) | Installation and feature flags |
| [Bindings Guide](BINDINGS.md) | C, Python, and WASM usage |
| [`semiflow/README.md`](../crates/semiflow/README.md) | Full type catalogue + cargo features |
| [`examples/`](../crates/semiflow/examples/README.md) | Worked examples, beginner → advanced |
| [precision-policy.md](precision-policy.md) | Accuracy / performance trade-offs |
| [python-coverage.md](python-coverage.md) | Python binding parity matrix |

## For developers & contributors

| Document | Purpose |
|----------|---------|
| [CONTRIBUTING.md](../CONTRIBUTING.md) | Workflow, conventions, ADR process |
| [api-stability.md](api-stability.md) | Semantic-versioning and API-stability policy |
| [release-process.md](release-process.md) | How releases are cut |
| [`adr/`](adr) | Architecture Decision Records (the "why" behind the design) |
| [`migration/`](migration) | API-evolution notes across development versions |
| [`audit-findings-*.md`](.) | Per-release math-fidelity audit records |
| [SECURITY.md](../SECURITY.md) | Vulnerability disclosure |
| [`contracts/`](../contracts) | Contract-first IDL / property specs |

## Mathematical specification

The normative mathematical specification lives in
[`contracts/semiflow-core.math.md`](../contracts/semiflow-core.math.md). The method
implements Theorem 6 of Remizov (2025), *Vladikavkaz Math. J.* 27(4), 124–135.
