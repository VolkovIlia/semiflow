# semiflow-wasm — WebAssembly bindings for semiflow-core

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![npm](https://img.shields.io/badge/npm-pending%20publication-lightgrey)](https://www.npmjs.com/package/@semiflow/wasm)
[![Docs](https://img.shields.io/badge/docs-README-blue)](https://github.com/VolkovIlia/semiflow/tree/master/crates/semiflow-wasm)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

WebAssembly bindings for [`semiflow-core`](../../crates/semiflow-core) —
Chernoff approximations of operator semigroups (Remizov 2025).

**Status**: Experimental — API not stabilised until v1.0.0

**Aligned with `semiflow-core` v9.0.0** (ADR-0154, 2026-06-10). The v9.0.0
addition exposes `ReverseHeat1D` — reverse-mode AD for constant-a 1D heat
(math §51, ADR-0156). The 1D heat (`Heat1D`) entry point from prior releases
remains unchanged.

`TtChernoff` / `TtState` and `GridlessChernoff` / `ParticleReduction` are
**Rust-only at v9.0.0** — not exposed via WASM (binding design deferred).

npm publish workflow (`release-wasm.yml`) is in place as of v0.11.0; the
package requires a maintainer `NPM_TOKEN` secret before the first publish.
See [ADR-0028](../../docs/adr/0028-ffi-pyo3-wasm-v0_10.md).

## npm install

```sh
npm install @semiflow/wasm
```

> **Note**: as of v0.11.0 the package has not yet been published to npm. The
> `release-wasm.yml` workflow is in place and will publish automatically on
> version tags once the maintainer configures the `NPM_TOKEN` repository
> secret. Until then, install via `wasm-pack build` (see below) and
> `npm link` or a local `file:` specifier.

## Build from source

Requires Rust toolchain and [`wasm-pack`](https://rustwasm.github.io/wasm-pack/).

```sh
cargo install wasm-pack

# Browser target (ES module with .wasm sidecar)
wasm-pack build crates/semiflow-wasm --target web --out-dir pkg-web

# Node.js target (CommonJS)
wasm-pack build crates/semiflow-wasm --target nodejs --out-dir pkg-node
```

## Quick start (Node.js)

```js
const { Heat1D, panic_hook_init, version } = require('./pkg-node/semiflow_wasm.js');

panic_hook_init();

const n = 1000;
const u0 = new Float64Array(n);
for (let i = 0; i < n; i++) {
    const x = -10 + 20 * i / (n - 1);
    u0[i] = Math.exp(-x * x);
}

const state = new Heat1D(-10, 10, n, u0);
state.evolve(1.0, 100);

const vals = state.values();  // Float64Array
console.log('values len:', vals.length, 'version:', version());
```

## Quick start (browser)

```html
<script type="module">
import init, { Heat1D, panic_hook_init, version } from './pkg-web/semiflow_wasm.js';
await init();  // fetch + compile .wasm
panic_hook_init();

const n = 1000;
const u0 = new Float64Array(n);
for (let i = 0; i < n; i++) {
    const x = -10 + 20 * i / (n - 1);
    u0[i] = Math.exp(-x * x);
}

const state = new Heat1D(-10, 10, n, u0);
state.evolve(1.0, 100);
console.log('max:', Math.max(...state.values()), 'version:', version());
</script>
```

## panic_hook_init

Call `panic_hook_init()` once at application startup in **development** builds.
It installs `console_error_panic_hook` so Rust panics appear as readable
messages in the browser console or Node.js stderr rather than opaque
`RuntimeError: unreachable executed`.

Production WASM builds use `panic = "abort"` (ADR-0028 Amendment 1) for
minimal code size; the hook is still safe to install but has no effect in that
mode.

## release-wasm.yml

The automated publish workflow lives at
[`.github/workflows/release-wasm.yml`](../../../../.github/workflows/release-wasm.yml).
It runs `wasm-pack build`, stamps `package.json` from the template, and calls
`npm publish --provenance` on version tags. Two idempotency guards prevent
double-publishing the same version.

## API

### Heat1D (existing)

| Symbol | Description |
|--------|-------------|
| `new Heat1D(xmin, xmax, n, u0)` | Create state on `[xmin, xmax]` with `n` nodes and Float64Array initial datum `u0` |
| `.evolve(t, n_steps)` | Advance state by time `t` using `n_steps` Chernoff steps |
| `.values()` | Current grid values as `Float64Array` (copy) |
| `.len()` | Number of grid nodes |

### ReverseHeat1D (v9.0.0, ADR-0154/0156)

Reverse-mode AD over constant-a 1D heat. **NARROW scope (§51.5):** constant-a
`DiffusionChernoff` ONLY; θ is the uniform diffusivity. Variable-coefficient and
nonlinear kernels are out of scope at v9.0.0.

| Symbol | Description |
|--------|-------------|
| `new ReverseHeat1D(theta, xmin, xmax, nGrid, nSteps)` | Construct evolver. `theta > 0` (diffusivity), `xmin < xmax`, `nGrid >= 4`, `nSteps >= 1`. Throws on validation failure. |
| `.valueAndGrad(tau, u0, target)` | Compute `(J, ∂J/∂θ)`. Returns `Float64Array[2]`: `[0]` = loss, `[1]` = gradient. Throws on invalid inputs. |
| `.theta()` | Returns the diffusivity parameter θ |
| `.nSteps()` | Returns the configured Chernoff step count |
| `.nGrid()` | Returns the number of grid nodes |

**`.valueAndGrad(tau, u0, target) -> Float64Array[2]`:**

| Parameter | Type | Notes |
|-----------|------|-------|
| `tau` | `number` | Per-step time increment (> 0, finite) |
| `u0` | `Float64Array` | Initial condition, length `nGrid` |
| `target` | `Float64Array` | Target state, length `nGrid` |
| returns `[0]` | `number` | L² loss `‖(F_θ(τ))ⁿ u₀ − target‖²` |
| returns `[1]` | `number` | `∂J/∂θ` (K=1 forward-mode Dual; 0-ULP vs core, §51.4) |

```js
import init, { ReverseHeat1D, panic_hook_init } from './pkg-web/semiflow_wasm.js';
await init();
panic_hook_init();

const nGrid = 24;
const u0 = new Float64Array(nGrid);
const target = new Float64Array(nGrid);
for (let i = 0; i < nGrid; i++) {
    const x = -4.0 + 8.0 * i / (nGrid - 1);
    u0[i] = Math.exp(-x * x);
    // target stays 0
}

const rc = new ReverseHeat1D(
    0.4,    // theta: diffusivity
    -4.0,   // xmin
     4.0,   // xmax
    nGrid,  // nGrid
    8       // nSteps
);
const result = rc.valueAndGrad(0.05, u0, target);
// result[0]: loss (number)
// result[1]: ∂J/∂θ (number)
console.log(`loss=${result[0].toFixed(6)}  grad=${result[1].toFixed(6)}`);
```

Errors are thrown as JavaScript `Error` objects with a `.kind` string property:
`GridMismatch`, `NanInf`, `OutOfDomain`.

### Utilities

| Symbol | Description |
|--------|-------------|
| `version()` | Crate version string |
| `panic_hook_init()` | Install `console_error_panic_hook` for readable panic messages (dev builds) |

All errors are thrown as JavaScript `Error` objects with a `.kind` string property
matching the C-ABI discriminator names:
`GridMismatch`, `NanInf`, `OutOfDomain`, `BoundaryFailure`, `CflViolated`,
`ConvergenceFailed`, `Unsupported`, `Panic`.

## Scope

- **In (v9.0.0)**: 1D heat (`Heat1D`, unit diffusion `a=1`), reverse-mode AD
  (`ReverseHeat1D`, constant-a narrow scope), Node.js + browser targets,
  CI-built `.wasm` artifacts, Firefox cross-engine CI, `release-wasm.yml`
  npm publish workflow.
- **Out (Rust-only at v9.0.0)**: `TtChernoff` / `TtState`, `GridlessChernoff`
  / `ParticleReduction` (binding design deferred).
- **Out (deferred to future)**: variable `a(x)`, 2D/3D, bundler integration tests.

## Mathematical reference

I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135.
DOI [10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q) |
[arXiv:2301.06765](https://arxiv.org/abs/2301.06765)

## License

MIT OR Apache-2.0 — same as `semiflow-core`.
