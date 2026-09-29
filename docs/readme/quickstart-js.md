Solve `∂ₜu = ∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1` and compare with the
closed-form solution `u(1,x) = 5^{-1/2}·e^{-x²/5}` (Node.js):

```js
const { Heat1D, panic_hook_init } = require('@semiflow/wasm');
panic_hook_init();   // readable Rust panic messages (optional)

const n = 1000;
const x = (i) => -10 + (20 * i) / (n - 1);
const u0 = Float64Array.from({ length: n }, (_, i) => Math.exp(-(x(i) ** 2)));

const state = new Heat1D(-10, 10, n, u0);
state.evolve(1.0, 100);
const u = state.values();   // Float64Array (copy)

let err = 0;
for (let i = 0; i < n; i++) {
  err = Math.max(err, Math.abs(u[i] - Math.exp(-(x(i) ** 2) / 5) / Math.sqrt(5)));
}
if (!(err < 5e-4)) throw new Error(`sup-norm error ${err}`);
```

In a browser or with a bundler, import the ES-module build and initialise it
once:

<!-- readme-test: skip (needs a browser to fetch the .wasm file) -->
```js
import init, { Heat1D } from '@semiflow/wasm';
await init();   // fetches and compiles the .wasm module
const state = new Heat1D(-10, 10, 1000, new Float64Array(1000));
```

Node.js examples on this page are executed by the project's CI against the
current build.
