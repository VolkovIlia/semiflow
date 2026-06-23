---
name: Bug report
about: Report a defect — incorrect math, panic, build break, doc error
title: "[bug] "
labels: bug
assignees: ''
---

## Summary

<!-- One line. -->

## Affected component

- [ ] `semiflow`
- [ ] `semiflow-ffi`
- [ ] `semiflow-py`
- [ ] `semiflow-wasm`
- [ ] docs
- [ ] build

## Version

<!-- Output of one of:
       cargo pkgid -p semiflow
       pip show semiflow-pde
       npm list @semiflow/wasm
-->

## Toolchain

<!-- rustc --version, OS, and (if applicable) Python / Node version. -->

## Reproducer

<!-- Minimal code block that triggers the bug. -->

```rust
// or python / js / wasm
```

## Expected behaviour

<!-- What should happen. Cite math.md section / ADR / paper if relevant. -->

## Actual behaviour

<!-- What actually happens. Include full error message, stack trace, or
     panic message. -->

## For numerical bugs

- Oracle (closed-form / reference grid / sympy gate):
- Observed convergence rate vs expected:
- Regime where the bug appears (parameter ranges, grid sizes, σ, T, β, …):
