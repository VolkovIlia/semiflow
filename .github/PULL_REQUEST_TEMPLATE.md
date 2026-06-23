## Summary

<!-- 1-3 sentences: what changed and why -->

## Type of change

- [ ] Bug fix
- [ ] New feature (non-breaking)
- [ ] Breaking change
- [ ] Documentation
- [ ] Refactor / perf
- [ ] Build / CI

## Affected crates

- [ ] `semiflow`
- [ ] `semiflow-ffi`
- [ ] `semiflow-py`
- [ ] `semiflow-wasm`

## Math content (if applicable)

- [ ] Updates `contracts/semiflow-core.math.md` §___
- [ ] New ADR: `docs/adr/NNNN-___.md`
- [ ] Sympy gate added/updated: `crates/semiflow/sympy/___.py`

## Verification

- [ ] `cargo run -p xtask -- test-fast` passes
- [ ] `cargo run -p xtask -- check-lints` clean
- [ ] `cargo clippy --all-targets --all-features -- -D warnings` clean
- [ ] FFI: `cargo run -p xtask -- ffi-smoke` (if FFI changed)
- [ ] PyO3: `cargo run -p xtask -- py-smoke` (if Python changed)
- [ ] WASM: `cargo run -p xtask -- wasm-test` (if WASM changed)
- [ ] `cargo doc --all-features --no-deps` builds without warnings (if public API changed)

## Trailers

This PR's commits include the required trailers:

```
Agent: <agent-or-human>
Task-ID: <kebab-case>
```

Bug-fix PRs additionally include `Fixes-Agent:` and `Fixes-Commit:` trailers.
