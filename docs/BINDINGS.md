# SemiFlow Bindings Guide

SemiFlow's numerical core (`semiflow`, Rust) is exposed to three other
ecosystems. All bindings wrap the same kernels; the Rust crate is the source of
truth for behaviour. Install commands and versions are in
[README § Install](../README.md#install); the binding overview is in
[README § Bindings](../README.md#bindings).

| Language | Package | Import / header | Crate |
|----------|---------|-----------------|-------|
| Rust | `semiflow` (crates.io) | `use semiflow::…;` | `crates/semiflow` |
| C / C++ | `semiflow-ffi` (built from source) | `#include "semiflow.h"` | `crates/semiflow-ffi` |
| Python | `semiflow-pde` (PyPI) | `import semiflow` | `crates/semiflow-py` |
| JS / TS / WASM | `@semiflow/wasm` (npm) | `require("@semiflow/wasm")` / `import … from "@semiflow/wasm"` | `crates/semiflow-wasm` |

> The PyPI **distribution** is named `semiflow-pde` because `semiflow` is already
> taken on PyPI, but it still imports as `import semiflow`.

## C / C++ (FFI)

The C ABI is a surface over opaque handles. Functions are prefixed `smf_`; all
fallible calls return a `SemiflowStatus` enum whose enumerators are unprefixed
(`Ok = 0`, `GridMismatch`, `NanInf`, `OutOfDomain`, `BoundaryFailure`,
`NullPtr`, `CflViolated`, `ConvergenceFailed`, `Unsupported`, `Panic = 99`);
`smf_status_str(st)` gives a static name string. Memory is owned by the
library — create a handle, evolve, read results, then free.

```c
#include "semiflow.h"   /* graph PDEs: semiflow_graph.h */

SemiflowState *state = NULL;
SemiflowStatus st = smf_state_new_heat_1d_unit(-10.0, 10.0, n, u0, n, &state);
if (st == Ok) st = smf_evolve(state, 1.0, 100);
if (st == Ok) st = smf_state_values(state, out, n);
if (st != Ok) fprintf(stderr, "semiflow: %s\n", smf_status_str(st));
smf_state_free(state);
```

No prebuilt binaries are published; build the library from a checkout. See
[`crates/semiflow-ffi/README.md`](../crates/semiflow-ffi/README.md) for the
build command (use the `release-ffi` profile so the panic boundary works), the
link flags, and the header regeneration step. The complete, authoritative list
of functions is the header itself,
[`crates/semiflow-ffi/include/semiflow.h`](../crates/semiflow-ffi/include/semiflow.h).

## Python

```python
import numpy as np
import semiflow

x = np.linspace(-10.0, 10.0, 1000)
state = semiflow.Heat1D(-10.0, 10.0, 1000, np.exp(-x**2))   # ∂ₜu = ∂ₓₓu
state.evolve(1.0, 100)
u = state.values()          # ≈ exp(-x²/5)/√5
```

Errors raise `semiflow.SemiflowError`; the message starts with the error kind
in brackets (e.g. `[OutOfDomain] t must be finite and >= 0`), using the same
names as the C `SemiflowStatus` enum.

The wheel is `abi3` (one wheel covers CPython 3.10+). Most `evolve` calls
release the GIL while they compute. See
[`crates/semiflow-py/README.md`](../crates/semiflow-py/README.md) and
[python-coverage.md](python-coverage.md) for the class inventory and the parity
matrix against the Rust API; the type stubs
(`crates/semiflow-py/python/semiflow/__init__.pyi`) carry every signature.

## JavaScript / WebAssembly

Node.js (CommonJS):

```js
const { Heat1D } = require("@semiflow/wasm");

const n = 1000;
const u0 = Float64Array.from({ length: n }, (_, i) => Math.exp(-((-10 + (20 * i) / (n - 1)) ** 2)));
const state = new Heat1D(-10, 10, n, u0);
state.evolve(1.0, 100);
const u = state.values();   // Float64Array (copy)
```

Browsers and bundlers load the ES-module build and initialise it once:

```js
import init, { Heat1D } from "@semiflow/wasm";
await init();
```

Errors are thrown as JS `Error` objects with a `.kind` string property (same
names as the C enum). The npm package is the **lite** build; the heavy-grid
engines (higher-order, 2D/3D, boundary-condition, manifold, hypoelliptic, …)
require building `crates/semiflow-wasm` with `--features full`. See
[`crates/semiflow-wasm/README.md`](../crates/semiflow-wasm/README.md) for the
class tables of both builds and the build commands.

## Cross-language parity

Binding results are checked against the Rust core in CI (parity tests in each
binding crate). If a number differs between languages, treat the Rust result as
canonical and file an issue.
