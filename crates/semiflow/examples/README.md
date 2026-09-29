# SemiFlow examples

Run any example from the repository root with:

```bash
cargo run -p semiflow --release --example <name>
```

| Example | What it shows | Level |
|---------|---------------|-------|
| [`heat_2d_demo`](heat_2d_demo.rs) | 2D heat equation via `Strang2D` tensor-product splitting | beginner |
| [`boundary_demo`](boundary_demo.rs) | Sampling under the `BoundaryPolicy` options (reflect / zero-extend / periodic / linear extrapolation) | beginner |
| [`strang_advdiff_demo`](strang_advdiff_demo.rs) | Advection–diffusion via `StrangSplit` (order 2) vs `ShiftChernoff1D` (order 1) | beginner |
| [`cev_european_call`](cev_european_call.rs) | CEV European-call pricing vs the Schröder ncx2 oracle | intermediate |
| [`resolvent_perf`](resolvent_perf.rs) | Laplace-resolvent evaluation `(λI−A)⁻¹g` | intermediate |
| [`heston_pricer`](heston_pricer.rs) | Heston stochastic-volatility pricing | intermediate |
| [`graph_par_speedup`](graph_par_speedup.rs) | Multicore speedup of batched graph evolution (compare runs with and without `--features parallel`) | intermediate |
| [`sabr_pricer`](sabr_pricer.rs) | SABR model priced on the hyperbolic manifold `H²` | advanced |
| [`rough_heston_pricer`](rough_heston_pricer.rs) | Rough Heston via a 4-factor Markov approximation on `MatrixDiffusionChernoff` (`--rate` / `--price`) | advanced |
| [`latency_tail`](latency_tail.rs) | HFT p99.9 per-tick latency benchmark (writes deterministic ticks to `examples/data/`) | advanced |

Add `--features parallel` for multi-threaded kernels (needed for a meaningful
`graph_par_speedup` comparison); latency-oriented examples are best built with
`RUSTFLAGS="-C target-cpu=native"`.

New to the library? Start with `heat_2d_demo` and `boundary_demo`, then read the
[User Guide](../../../docs/USER_GUIDE.md).
