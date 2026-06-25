# semiflow-wasm — WebAssembly bindings for semiflow

[![CI](https://img.shields.io/badge/CI-passing-brightgreen)](https://github.com/VolkovIlia/semiflow/actions)
[![npm](https://img.shields.io/npm/v/@semiflow/wasm)](https://www.npmjs.com/package/@semiflow/wasm)
[![Docs](https://img.shields.io/badge/docs-README-blue)](https://github.com/VolkovIlia/semiflow/tree/master/crates/semiflow-wasm)
[![License](https://img.shields.io/badge/license-MIT%20OR%20Apache--2.0-blue)](../../LICENSE-MIT)

WebAssembly bindings for [`semiflow`](../../crates/semiflow) —
Chernoff approximations of operator semigroups (Remizov 2025).

**Status**: Experimental — API not stabilised until v1.0.0

## npm install

```sh
npm install @semiflow/wasm
```

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

Production WASM builds use `panic = "abort"` (see ADR-0028 Amendment 1) for
minimal code size; the hook is still safe to install but has no effect in that
mode.

## release-wasm.yml

The automated publish workflow lives at
[`.github/workflows/release-wasm.yml`](../../../../.github/workflows/release-wasm.yml).
It runs `wasm-pack build`, stamps `package.json` from the template, and calls
`npm publish --provenance` on version tags. Two idempotency guards prevent
double-publishing the same version.

## API

The package ships two build sizes controlled by a Cargo feature flag.

### Default build ("lite") — ≈ 768 KB raw Wasm

The default npm package exposes the lightweight baseline engine set:

| Class | Description |
|-------|-------------|
| `Heat1D` | 1D heat equation (`a = 1`), Chernoff stepping |
| `ReverseHeat1D` | Reverse-mode AD over constant-a 1D heat: returns `(loss, ∂J/∂θ)` |
| `EvolverHeat1DUnitV3` | v3 unit-diffusivity 1D evolver (Chernoff-approximation API) |
| `EvolverHeat1DGreeksV3` | v3 Greeks evolver: evolves state and computes finite-difference sensitivities |
| `GrowthV3` | Growth bound helper used by the v3 evolver API |
| `GraphPath` | Heat semigroup on a path graph |
| `GraphHeat` | Heat semigroup on a general graph (variable edge weights) |
| `GraphHeat6` | 6th-order Magnus graph heat evolver |
| `ResolventJumpV8` | 1D resolvent-jump operator (v8 API) |
| `ResolventJump2DV8` | 2D resolvent-jump operator (v8 API) |
| `ResolventJump3DV8` | 3D resolvent-jump operator (v8 API) |
| `TtEvolver` | Tensor-train 1D evolver |
| `TtState` | Mutable state carrier for `TtEvolver` |
| `TtCoupledEvolver` | Tensor-train coupled-dimension evolver |
| `VarCoefTtEvolver` | Tensor-train variable-coefficient evolver |
| `GridlessEvolver` | Particle-based gridless evolver |
| `MeasureState` | Discrete measure state carrier for `GridlessEvolver` |
| `WentzellV8` | Wentzell boundary condition evolver (v8 API) |
| `GammaFamily` | Gamma-family semigroup approximation |
| `AdjointFokkerPlanckV8` | Adjoint Fokker–Planck evolver (v8 API) |

### `--features full` build — ≈ 1.4 MB raw Wasm

Building with `--features full` adds all heavy-grid, multi-dimensional, boundary-condition,
and hypoelliptic engines. Additions include:

- **Higher-order 1D** — `Heat1D4th`, `Heat1D6th`, `Heat1DZeta4/6/8`, `TruncatedExp1D`,
  `TruncatedExp4th1D`, `DriftReaction1D`, `DriftReaction4th1D`, `Shift1D`, `Strang1D`,
  `DiffusionExpmv1D`
- **Matrix / Schrödinger** — `MatrixDiffusion1D`, `MatrixDiffusion2D`, `MatrixDiffusion3D`,
  `Schrodinger1D`, `SchrodingerComplex1D`
- **Boundary conditions** — `Killing1D`, `Killing2nd1D`, `Reflected1D`, `Robin1D`,
  `Resolvent1D`, `KilledDirichlet1D`, `DirichletHeat2nd1D`
- **2D/3D tensor** — `Heat2D`, `Heat3D`, `Heat2DVarA`, `Heat3DVarA`
- **Non-separable / anisotropic** — `NonSeparable2D`, `NonSeparable2DAniso`,
  `AnisotropicShiftND2`, `AnisotropicShiftND3`
- **High-dimensional** — `SmolyakD6`
- **Nonautonomous** — `Howland1D`, `Subordinated1D`
- **Manifold** — `Manifold2D` (Torus, Sphere2, Hyperbolic2)
- **Hypoelliptic** — `HypoellipticChernoffHeisenberg`, `HypoellipticChernoffKolmogorov`,
  `HypoellipticChernoffEngel` (step-2 Carnot groups, Strang–Hörmander splitting)
- **Graph extensions** — `GraphHeat4thWasm`, `VarCoefGraphHeatWasm`, `MagnusGraphHeatWasm`,
  `MagnusGraphHeat6Wasm`, `VarCoefMagnusGraphWasm`, `QuantumGraphWasm`,
  `QuantumGraphHeatWasm`, `StrangGraphWasm`
- **Other** — `Obstacle1D` (JS name), `ObstacleND2Wasm`, `ObstacleGammaV8Wasm`,
  `Adjoint1D`, `AdaptivePI1D`, `ComplexTripleJumpWasm`, `PointEvalWasm`

**Documented deferrals (not yet wired to WASM):** `ObstacleND`, `GraphTraj`,
Laplacian introspection, `GraphAdjoint` dense read-back, S³ carrier handles
(`TtEvolver`/`GridlessEvolver` FFI handles — deferred to a follow-up release).

### Heat1D

| Symbol | Description |
|--------|-------------|
| `new Heat1D(xmin, xmax, n, u0)` | Create state on `[xmin, xmax]` with `n` nodes and Float64Array initial datum `u0` |
| `.evolve(t, n_steps)` | Advance state by time `t` using `n_steps` Chernoff steps |
| `.values()` | Current grid values as `Float64Array` (copy) |
| `.len()` | Number of grid nodes |

### ReverseHeat1D

Reverse-mode AD over constant-a 1D heat. **Narrow scope:** constant-a
`DiffusionChernoff` only; θ is the uniform diffusivity.

| Symbol | Description |
|--------|-------------|
| `new ReverseHeat1D(theta, xmin, xmax, nGrid, nSteps)` | Construct evolver. `theta > 0`, `xmin < xmax`, `nGrid >= 4`, `nSteps >= 1`. Throws on validation failure. |
| `.valueAndGrad(tau, u0, target)` | Compute `(J, ∂J/∂θ)`. Returns `Float64Array[2]`: `[0]` = loss, `[1]` = gradient. |
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
| returns `[1]` | `number` | `∂J/∂θ` |

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

### Utilities

| Symbol | Description |
|--------|-------------|
| `version()` | Crate version string |
| `panic_hook_init()` | Install `console_error_panic_hook` for readable panic messages (dev builds) |

All errors are thrown as JavaScript `Error` objects with a `.kind` string property:
`GridMismatch`, `NanInf`, `OutOfDomain`, `BoundaryFailure`, `CflViolated`,
`ConvergenceFailed`, `Unsupported`, `Panic`.

## Mathematical reference

I. D. Remizov, *Vladikavkaz Math. J.* **27**(4) (2025) 124–135.
DOI [10.46698/a3908-1212-5385-q](https://doi.org/10.46698/a3908-1212-5385-q) |
[arXiv:2301.06765](https://arxiv.org/abs/2301.06765)

## License

MIT OR Apache-2.0 — same as `semiflow`.
