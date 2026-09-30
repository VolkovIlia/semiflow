Solve `∂ₜu = ∂ₓₓu` from `u₀(x) = e^{-x²}` to `t = 1` and compare with the
closed-form solution `u(1,x) = 5^{-1/2}·e^{-x²/5}`:

```python
import numpy as np
import semiflow

n = 1000
xs = np.linspace(-10.0, 10.0, n)
state = semiflow.Heat1D(-10.0, 10.0, n, np.exp(-xs**2))
state.evolve(t=1.0, n_steps=100)   # releases the GIL while Rust runs

exact = np.exp(-xs**2 / 5.0) / np.sqrt(5.0)
err = np.max(np.abs(state.values() - exact))
assert err < 5e-4, err
```

Every example on this page is executed by the project's CI against the
current build.
