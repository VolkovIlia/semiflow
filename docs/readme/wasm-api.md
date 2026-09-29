## API

The authoritative signatures are the TypeScript declarations shipped in the
package (`web/semiflow_wasm.d.ts`). The tables below are generated from the
Rust sources, so every name listed is exported under exactly that JS name.

### npm build ("lite")

The published package contains:

{{wasm_lite_table}}

### Full build

The heavy-grid, multi-dimensional, boundary-condition and hypoelliptic
engines are compiled only with the `full` Cargo feature, which the npm package
does not enable. Build it yourself to use them (see Install above):

{{wasm_full_table}}

### `Heat1D`

| Member | Description |
|--------|-------------|
| `new Heat1D(xmin, xmax, n, u0)` | Unit diffusion on `[xmin, xmax]` with `n ≥ 4` nodes; `u0` is a `Float64Array` of length `n` |
| `.evolve(t, nSteps)` | Advance by time `t` with `nSteps` Chernoff steps |
| `.values()` | Current values as a `Float64Array` (copy) |
| `.len()` | Number of grid nodes |

### `ReverseHeat1D`

Reverse-mode AD over constant-coefficient 1D heat (the diffusivity θ is the
parameter):

| Member | Description |
|--------|-------------|
| `new ReverseHeat1D(theta, xmin, xmax, nGrid, nSteps)` | `theta > 0`, `xmin < xmax`, `nGrid ≥ 4`, `nSteps ≥ 1` |
| `.valueAndGrad(tau, u0, target)` | Returns `Float64Array [J, ∂J/∂θ]` with `J = ‖(F_θ(τ))ⁿ u₀ − target‖²` |
| `.theta()`, `.nSteps()`, `.nGrid()` | Configuration accessors |

```js
const { ReverseHeat1D } = require('@semiflow/wasm');

const nGrid = 24;
const u0 = Float64Array.from({ length: nGrid }, (_, i) => {
  const x = -4 + (8 * i) / (nGrid - 1);
  return Math.exp(-x * x);
});
const target = new Float64Array(nGrid);   // all zeros

const rc = new ReverseHeat1D(0.4, -4.0, 4.0, nGrid, 8);
const [loss, grad] = rc.valueAndGrad(0.05, u0, target);
if (!(Number.isFinite(loss) && Number.isFinite(grad))) throw new Error('non-finite');
```

### Utilities and errors

| Function | Description |
|----------|-------------|
| `version()` | Version string of the Rust crate |
| `panic_hook_init()` | Installs `console_error_panic_hook` so Rust panics show a readable message instead of `RuntimeError: unreachable` |

Errors are thrown as JS `Error` objects with a `.kind` string:
`GridMismatch`, `NanInf`, `OutOfDomain`, `BoundaryFailure`, `CflViolated`,
`ConvergenceFailed`, `Unsupported` or `Panic`.
