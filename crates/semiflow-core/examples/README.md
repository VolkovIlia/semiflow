# SemiFlow examples

Run any example with:

```bash
cargo run --release --example <name>            # add --features parallel,simd for the HFT ones
```

| Example | What it shows | Level |
|---------|---------------|-------|
| [`heat_2d_demo`](heat_2d_demo.rs) | 2D heat equation via `Strang2D` tensor-product splitting | beginner |
| [`boundary_demo`](boundary_demo.rs) | `BoundaryPolicy` options (reflect / Dirichlet / Robin) | beginner |
| [`strang_advdiff_demo`](strang_advdiff_demo.rs) | Advection–diffusion via operator splitting | beginner |
| [`cev_european_call`](cev_european_call.rs) | CEV European-call pricing vs the Schröder ncx2 oracle | intermediate |
| [`resolvent_perf`](resolvent_perf.rs) | Laplace-resolvent evaluation `(λI−A)⁻¹g` | intermediate |
| [`heston_pricer`](heston_pricer.rs) | Heston stochastic-volatility pricing | intermediate |
| [`sabr_pricer`](sabr_pricer.rs) | SABR model priced on the hyperbolic manifold `H²` | advanced |
| [`rough_heston_pricer`](rough_heston_pricer.rs) | Rough Heston via matrix-diffusion kernels | advanced |
| [`latency_tail`](latency_tail.rs) | HFT p99.9 per-tick latency benchmark (writes deterministic ticks to `examples/data/`) | advanced |

New to the library? Start with `heat_2d_demo` and `boundary_demo`, then read the
[User Guide](../../../docs/USER_GUIDE.md).
