# ADR-0170: Rebrand to SemiFlow

## Status
Accepted — 2026-06-19

## Context
The library was published under the working name "RemizovCore" / "remizov-core",
derived from the primary theorem author's name. This conflates the scientific
attribution (Remizov 2025, Vladikavkaz Math. J.) with the software brand, making
the package name appear as a personal eponym rather than a subject-matter name.
"Semiflow" / "SemiFlow" accurately describes what the library computes: operator
semiflows approximated via Chernoff's theorem.

## Decision
Rename all brand tokens: package names (`remizov-core` → `semiflow-core`,
`remizov-ffi` → `semiflow-ffi`, `remizov-py` → `semiflow-py`,
`remizov-wasm` → `semiflow-wasm`), Rust type prefixes (`RemizovFloat` →
`SemiflowFloat`, `RemizovComplex` → `SemiflowComplex`, `RemizovError` →
`SemiflowError`), C ABI prefix (`rmz_` → `smf_`, legacy `remizov_` → `smf_`),
and C opaque typedefs (`Rmz*` → `Smf*`). Repository slug: `remizovcore` →
`semiflow`. The rebrand coincides with the first public release, which is cut as
**`0.9.0-beta`** — a `0.x` public beta opened for community testing and bug
reports ahead of a stable `1.0`. (The library was developed privately through
extensive internal iteration; that history is not part of the public record.)
PyPI distribution name is `semiflow-pde` because `semiflow` is already taken on
PyPI; the Python import module remains `semiflow`. Scientific citations —
"Remizov 2025", "Galkin–Remizov", "Theorem 6 of Remizov", the Vladikavkaz DOI —
are preserved verbatim in all files.

## Consequences
The public surface is the `semiflow-*` crates, the `smf_` C ABI
(`#include "semiflow.h"`), the Python package `semiflow-pde` (`import semiflow`),
and the npm package `semiflow`. As a `0.x` beta, minor versions may introduce
breaking changes while the API stabilizes toward `1.0`.
