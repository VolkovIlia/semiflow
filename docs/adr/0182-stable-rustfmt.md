# ADR-0182: Switch CI `fmt` job from nightly to stable rustfmt

**Status**: Accepted

**Context**: `rustfmt.toml` contained `imports_granularity` and `group_imports`,
which are nightly-only options. This forced the CI `fmt` job to use a nightly
toolchain, adding fragility (nightly breaks occasionally) and divergence from
all other jobs that run on stable.

**Decision**: Remove the two nightly-only options from `rustfmt.toml` (they
controlled import grouping style — not correctness). Switch the CI `fmt` job
from `dtolnay/rust-toolchain@nightly` to `dtolnay/rust-toolchain@stable`.

**Consequences**: `cargo fmt --all --check` now passes on stable. All seven CI
jobs run on stable or a pinned MSRV (1.78), eliminating the nightly dependency.
Import ordering is still stable-fmt-canonical; only the grouping heuristic is
gone, which has no effect on compilation or semantics.
