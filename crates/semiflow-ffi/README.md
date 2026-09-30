# semiflow-ffi

[![CI](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml/badge.svg)](https://github.com/VolkovIlia/semiflow/actions/workflows/ci.yml)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

C ABI bindings for [`semiflow`](../../crates/semiflow): Chernoff
approximations of operator semigroups. Exposes an opaque-handle C API backed
by a `catch_unwind` panic boundary and a status-code enum for all error paths.
It shares the workspace version with the other SemiFlow packages (see the
[project README](../../README.md)).

The surface spans the kernel families: 1D/2D/3D diffusion (constant and
variable coefficients), higher-order and ζ-ladder kernels, drift–reaction,
graph kernels, manifold, hypoelliptic, adjoint, resolvent, killing,
reflected/Neumann, obstacle, Schrödinger, tensor-train, gridless, and more.
The authoritative, complete list of exported `smf_*` functions is the generated
header [`include/semiflow.h`](include/semiflow.h) (regenerated via
`cargo run -p xtask -- ffi-headers`); each declaration carries its doc comment.
Consult that file rather than any per-type enumeration in prose.

**Experimental** — the C API may change before 1.0.0. See
[ADR-0028](../../docs/adr/0028-ffi-pyo3-wasm-v0_10.md).

---

## ⚠ Safety: Build Profile

**Always build with `--profile release-ffi`. Never use `--release`.**

The workspace `[profile.release]` sets `panic = "abort"`. Under that setting,
`catch_unwind` at each `extern "C"` boundary becomes a no-op: a Rust panic
instead unwinds through the C stack as **undefined behavior**. The
`[profile.release-ffi]` profile inherits `release` and overrides
`panic = "unwind"` (and strips symbols) so that every `catch_unwind` actually
catches panics and converts them to a `Panic` status code.

Building with `--release` looks identical at link time but produces a binary
with a broken panic boundary. There is no runtime warning.

**Verification**: `cargo run -p xtask -- ffi-smoke` builds with the correct
profile automatically. If you invoke `cargo build` directly, you must pass
`--profile release-ffi` explicitly.

See [ADR-0028](../../docs/adr/0028-ffi-pyo3-wasm-v0_10.md) (and Amendment 1)
for the rationale behind keeping `release-ffi` as a separate profile rather
than patching `[profile.release]`.

---

## Build

No prebuilt binaries are published; build from a checkout of the repository:

```sh
cargo build -p semiflow-ffi --profile release-ffi
# Linux:   target/release-ffi/libsemiflow_ffi.so   (+ libsemiflow_ffi.a)
# macOS:   target/release-ffi/libsemiflow_ffi.dylib (+ libsemiflow_ffi.a)
# Windows: target/release-ffi/semiflow_ffi.dll
# Header:  crates/semiflow-ffi/include/semiflow.h
```

---

## Usage from C

Solve `∂ₜu = ∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1`:

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

Output: `u[500] = 0.447206`.

Runnable programs in [`examples/`](examples):

| File | Shows |
|------|-------|
| [`heat.c`](examples/heat.c) | Unit-diffusion 1D heat vs the closed-form oracle (the CI smoke test) |
| [`heat_var_a.c`](examples/heat_var_a.c) | Variable `a(x)` through a C callback (`smf_state_new_with_closure`) |
| [`graph_heat.c`](examples/graph_heat.c) | Graph heat on a path graph (`smf_graph_path`, `smf_ghc_new`) |
| [`greeks.c`](examples/greeks.c) | Hyper-dual Greeks evolver |

---

## Status codes

`SemiflowStatus` enumerators are unprefixed in C:

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
`smf_status_str(st)` returns the variant name as a static string.

---

## API reference

Every kernel family follows the same lifecycle: a `smf_<family>_new…`
constructor that writes an opaque handle through an out-pointer and returns
`SemiflowStatus`, an evolve/apply call, a read-back call into a caller buffer,
and a null-safe `…_free` / `…_drop`. The 1D heat state used above:

| Function | Description |
|----------|-------------|
| `smf_state_new_heat_1d_unit` | Allocate a 1D heat state with `a = 1`. |
| `smf_state_new_with_closure` | Allocate a 1D heat state with variable `a(x)` given as C callbacks (`a`, `a'`, `a''` plus `user_data`). |
| `smf_evolve` | Advance the state in place by time `t` in `n_steps` steps. |
| `smf_state_values` | Copy grid values into a caller buffer. |
| `smf_state_size` | Number of grid nodes (0 if null). |
| `smf_state_free` | Free a state handle. Null-safe. |
| `smf_status_str` | Static string for a status code. Do not free. |
| `smf_version` | Crate version string. Do not free. |

For everything else — signatures, preconditions and error cases — read
[`include/semiflow.h`](include/semiflow.h). The graph entry points are also
documented in the hand-maintained companion header
[`include/semiflow_graph.h`](include/semiflow_graph.h).

---

## Lifecycle invariants

- The caller owns every handle returned by an `smf_*_new*` constructor.
- `smf_state_free` is null-safe but **not** double-free safe. Set the
  pointer to `NULL` immediately after calling it.
- Arrays are `(ptr, len)` pairs. Never null-terminated.
- `smf_state_size(NULL)` returns `0`.
- `t = 0.0` in `smf_evolve` is accepted but does NOT produce an identity
  transform. Numerical underflow from `n_steps` kernel applications means
  the result will not equal `u0`. Skip the call if you need an identity.
- Static strings from `smf_status_str` and `smf_version` are valid
  for the lifetime of the process; do not free them.
- Callback-based constructors (e.g. `smf_state_new_with_closure`) require the
  callbacks to be pure and panic-free, and `user_data` to outlive the handle.

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
# Rust integration tests (round-trip, edge cases, parity, panic boundary)
cargo test -p semiflow-ffi

# End-to-end C smoke (build cdylib, compile heat.c, check sup_error < 5e-4)
cargo run -p xtask -- ffi-smoke

# Graph C smoke (graph_heat.c)
cargo run -p xtask -- ffi-graph-smoke
```

---

## Roadmap

- **1.0.0** — ABI freeze. No variants removed or reordered after this point.

---

## License

MIT OR Apache-2.0 (workspace inheritance).
