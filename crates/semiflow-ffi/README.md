---
version: 1.2.0
last_updated: 2026-05-10
freshness_score: 1.0
dependencies:
  - crates/semiflow-ffi/src/ffi.rs
  - crates/semiflow-ffi/src/status.rs
  - crates/semiflow-ffi/include/semiflow.h
  - crates/semiflow-ffi/examples/heat.c
  - docs/adr/0028-ffi-pyo3-wasm-v0_10.md
changelog:
  - 1.0.0: Initial documentation for Wave A (v0.10.0)
  - 1.1.0: v0.11.0 sync — defer variable-a to v0.12.0, heat.c link confirmed
  - 1.2.0: Promote build-profile safety warning to top-level ⚠ Safety section
---

# semiflow-ffi

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![Version](https://img.shields.io/badge/version-1.0.0-blue)](https://github.com/VolkovIlia/semiflow/releases)
[![Docs](https://img.shields.io/badge/docs-README-blue)](https://github.com/VolkovIlia/semiflow/tree/master/crates/semiflow-ffi)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

C ABI bindings for [`semiflow-core`](../../crates/semiflow-core): Chernoff
approximations of operator semigroups. Exposes an opaque-handle C API backed
by a `catch_unwind` panic boundary and a status-code enum for all error paths.

**Aligned with `semiflow-core` v2.8.0.** The FFI surface itself did not change in
v2.6.0–v2.8.0; the 1D heat + graph kernel entry points from v2.4.0 remain the
current exposed surface. New v2.6–v2.8 Rust types (`KillingChernoff`,
`LaplaceChernoffResolvent`, `HowlandLift`, `ManifoldChernoff`,
`ReflectedHeatChernoff`) are **not yet bound via FFI** — deferred to v2.9+ per
the roadmap.

**EXPERIMENTAL** — API may break before v1.0.0. Wave A (this crate) exposes
1D heat with `a(x) = 1.0` only. Variable diffusion coefficients are deferred
to v0.12.0. See [ADR-0028](../../docs/adr/0028-ffi-pyo3-wasm-v0_10.md).

---

## ⚠ Safety: Build Profile

**Always build with `--profile release-ffi`. Never use `--release`.**

The workspace `[profile.release]` sets `panic = "abort"`. Under that setting,
`catch_unwind` at each `extern "C"` boundary becomes a no-op: a Rust panic
instead unwinds through the C stack as **undefined behavior**. The
`[profile.release-ffi]` profile inherits `release` and overrides exactly one
field — `panic = "unwind"` — so that every `catch_unwind` actually catches
panics and converts them to a `SemiflowStatus::Panic` return code.

Building with `--release` looks identical at link time but produces a binary
with a broken panic boundary. There is no runtime warning.

**Verification**: `cargo run -p xtask -- ffi-build` uses the correct profile
automatically. If you invoke `cargo build` directly, you must pass
`--profile release-ffi` explicitly.

See [ADR-0028](../../docs/adr/0028-ffi-pyo3-wasm-v0_10.md) (and Amendment 1)
for the rationale behind keeping `release-ffi` as a separate profile rather
than patching `[profile.release]`.

---

## Build

```sh
# Linux  — produces target/release-ffi/libsemiflow_ffi.so
cargo build -p semiflow-ffi --profile release-ffi

# macOS  — produces target/release-ffi/libsemiflow_ffi.dylib
cargo build -p semiflow-ffi --profile release-ffi

# Windows — produces target/release-ffi/semiflow_ffi.dll
cargo build -p semiflow-ffi --profile release-ffi
```

---

## Usage from C

```c
#include "semiflow.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    int n = 1000;
    double *u0 = malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        double x = -10.0 + i * (20.0 / (n - 1));
        u0[i] = exp(-x * x);
    }

    SemiflowState *state = NULL;
    SemiflowStatus st = smf_state_new_heat_1d_unit(
        -10.0, 10.0, n, u0, n, &state);
    if (st != Ok) { fprintf(stderr, "%s\n", smf_status_str(st)); return 1; }

    st = smf_evolve(state, 1.0, 100);
    if (st != Ok) { fprintf(stderr, "%s\n", smf_status_str(st)); return 1; }

    double *out = malloc(n * sizeof(double));
    smf_state_values(state, out, n);
    printf("u[500] = %.6f\n", out[500]);   /* near x=0: ≈ 0.447 */

    smf_state_free(state);
    free(out); free(u0);
    return 0;
}
```

**Compile and link**:

```sh
# Linux / clang
clang heat.c \
  -I crates/semiflow-ffi/include \
  -L target/release-ffi -lsemiflow_ffi \
  -lm -o heat
LD_LIBRARY_PATH=target/release-ffi ./heat

# macOS
cc heat.c \
  -I crates/semiflow-ffi/include \
  -L target/release-ffi -lsemiflow_ffi \
  -lm -o heat
DYLD_LIBRARY_PATH=target/release-ffi ./heat

# Windows (MSVC cl.exe)
cl.exe heat.c /I crates\semiflow-ffi\include \
  target\release-ffi\semiflow_ffi.dll /link /out:heat.exe
heat.exe
```

Example output: `sup_error=1.460e-6  version=0.10.0`

A fully-annotated smoke program lives in
[`examples/heat.c`](examples/heat.c).

---

## Status codes

| Variant | Integer | Meaning | When |
|---------|---------|---------|------|
| `Ok` | 0 | Success | Operation completed. |
| `GridMismatch` | 1 | Grid geometry invalid | `n < 4`; `xmin >= xmax`; `u0_len != n`. |
| `NanInf` | 2 | Non-finite input | NaN or Inf in `u0`, `xmin`, `xmax`, or `t`. |
| `OutOfDomain` | 3 | Domain precondition | `t < 0`; `n_steps == 0`. |
| `BoundaryFailure` | 4 | Grid too coarse | Chernoff shift exceeds grid spacing. |
| `NullPtr` | 5 | Null pointer | Required pointer argument was null. |
| `CflViolated` | 6 | CFL exceeded | TruncatedExp K=4 CFL bound violated. |
| `ConvergenceFailed` | 7 | Solver diverged | Iterative solver hit cap (rare). |
| `Unsupported` | 8 | Not in this build | Feature disabled at compile time. |
| `Panic` | 99 | Internal panic caught | Rust panic at FFI boundary — file a bug. |

Integer values are stable ABI. Adding variants requires a major bump (ADR-0028).

---

## API reference

All functions return `SemiflowStatus` except `smf_state_free`,
`smf_state_size`, `smf_status_str`, and `smf_version`.

| Function | Signature | Description |
|----------|-----------|-------------|
| `smf_state_new_heat_1d_unit` | `(xmin, xmax, n, u0, u0_len, out_state) → Status` | Allocate 1D heat state with `a=1.0`. |
| `smf_state_free` | `(state) → void` | Free a state handle. Null-safe. |
| `smf_evolve` | `(state, t, n_steps) → Status` | Advance state in-place by time `t`. |
| `smf_state_values` | `(state, out_buf, out_buf_len) → Status` | Copy grid values into caller buffer. |
| `smf_state_size` | `(state) → usize` | Number of grid nodes (0 if null). |
| `smf_status_str` | `(status) → const char *` | Static string for status code. Do not free. |
| `smf_version` | `() → const char *` | Crate version string, e.g. `"0.10.0"`. Do not free. |

Full signatures are in [`include/semiflow.h`](include/semiflow.h).

---

## Lifecycle invariants

- The caller owns the handle returned by `smf_state_new_*`.
- `smf_state_free` is null-safe but **not** double-free safe. Set the
  pointer to `NULL` immediately after calling it.
- Arrays are `(ptr, len)` pairs. Never null-terminated.
- `smf_state_size(NULL)` returns `0`.
- `t = 0.0` in `smf_evolve` is accepted but does NOT produce an identity
  transform. Numerical underflow from `n_steps` kernel applications means
  the result will not equal `u0`. Skip the call if you need an identity.
- Static strings from `smf_status_str` and `smf_version` are valid
  for the lifetime of the process; do not free them.

---

## Generating the C header

```sh
# Regenerate include/semiflow.h from Rust source
cargo run -p xtask -- ffi-headers

# Check for drift (used in CI)
cargo run -p xtask -- ffi-headers --check
```

---

## Testing

```sh
# 32 integration tests (12 round-trip + 20 edge-case)
cargo test -p semiflow-ffi

# End-to-end C smoke (build cdylib, compile heat.c, check sup_error < 5e-4)
cargo run -p xtask -- ffi-smoke
```

---

## Roadmap

- **Wave A (v0.10.0)** — 1D heat, `a(x) = 1.0`, 7 entry points. Released.
- **v0.10.0 Wave B -- semiflow-py` (PyO3 + maturin wheels, Python 3.10–3.13). Released.
- **v0.10.0 Wave C** — `semiflow-wasm` (wasm-bindgen, `wasm32-unknown-unknown`). Released.
- **v2.4.0 Wave C** — graph kernel FFI entry points (`smf_ghc6_*`, `smf_vc_mghc_*`). Released.
- **v2.6.0–v2.8.0** — core library advanced (BoundaryPolicy widening, KillingChernoff,
  LaplaceChernoffResolvent, HowlandLift, ManifoldChernoff, ReflectedHeatChernoff). FFI
  surface UNCHANGED; new types deferred to v2.9+ per roadmap.
- **v0.12.0** — variable `a(x)` via FFI callback. Blocked on the `with_closure`
  core API design in `semiflow-core`; `DiffusionChernoff::new` currently only
  accepts non-capturing `fn(F) -> F` pointers (ADR-0028).
- **v1.0.0** — ABI freeze. No variants removed or reordered after this point.

---

## License

MIT OR Apache-2.0 (workspace inheritance).
