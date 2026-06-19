# SemiFlow Bindings Guide

SemiFlow's numerical core (`semiflow-core`, Rust) is exposed to three other
ecosystems. All bindings wrap the same kernels and pass the same convergence
gates; the Rust crate is the source of truth for behaviour.

| Language | Package | Import / header | Crate |
|----------|---------|-----------------|-------|
| Rust | `semiflow-core` (crates.io) | `use semiflow_core::…;` | `crates/semiflow-core` |
| C / C++ | release artifact (cdylib) | `#include "semiflow.h"` | `crates/semiflow-ffi` |
| Python | `semiflow-pde` (PyPI) | `import semiflow` | `crates/semiflow-py` |
| JS / TS / WASM | `semiflow` (npm) | `import … from "semiflow"` | `crates/semiflow-wasm` |

> The PyPI **distribution** is named `semiflow-pde` because `semiflow` is already
> taken on PyPI, but it still imports as `import semiflow`.

## C / C++ (FFI)

The C ABI is a small, stable surface over opaque handles. Symbols are prefixed
`smf_` and status codes are `SMF_*`. Memory is owned by the library — create a
handle, evolve, read results, then free.

```c
#include "semiflow.h"   /* graph PDEs: semiflow_graph.h */

/* create → evolve → read → free; every call returns an SMF_* status code */
```

Build the cdylib and generate the header with the workspace's `xtask`; see
[`crates/semiflow-ffi/README.md`](../crates/semiflow-ffi/README.md) for the exact
commands, the full function list, and the panic-unwind requirement.

## Python

```bash
pip install semiflow-pde
```

```python
import semiflow

# pyclass evolvers mirror the Rust kernels; errors raise SemiflowError(.kind)
```

The wheel is `abi3` (one wheel covers CPython 3.10+). The GIL is released during
evolution. See [`crates/semiflow-py/README.md`](../crates/semiflow-py/README.md)
and [python-coverage.md](python-coverage.md) for the complete class inventory and
parity matrix against the Rust API.

## JavaScript / WebAssembly

```bash
npm install semiflow
```

```js
import init, { Heat1D } from "semiflow";
await init();
```

Built with `wasm-bindgen` / `wasm-pack`; `pkg-web` and `pkg-node` targets are
published. See [`crates/semiflow-wasm/README.md`](../crates/semiflow-wasm/README.md).

## Cross-language parity

All bindings are validated to agree with the Rust core to sub-ULP / sub-`1e-9`
tolerances in CI. If a number differs between languages, treat the Rust result as
canonical and file an issue.
