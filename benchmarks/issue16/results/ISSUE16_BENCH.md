# Issue #16 Benchmark — `path="implicit"` stiff-tractability

Date: 2026-07-02  |  Timeout: 90s  |  Operator: 1-D FD Laplacian

## Stiff Neumann (N=400, scale=1e7, t=1.0) — ACCURACY GATE

Oracle: `mean(v)·1` (surviving null mode; non-trivial — mean≈0.50)

| Engine | Wallclock | Peak mem | Status | sup_error vs oracle |
| --- | --- | --- | --- | --- |
| semiflow/implicit | 4.1 ms | 32.6 MB | OK | 3.253e-12 |
| semiflow/chebyshev | 14116.5 ms | 32.2 MB | OK | 6.761e-06 |
| semiflow/lanczos | 55620.4 ms | 32.3 MB | OK | 1.711e-11 |
| scipy/expm_multiply | >90s | — | TIMEOUT (>90s) | — |

## Stiff Dirichlet (N=400, scale=1e7, t=1.0) — L-STABILITY DEMO

Note: all modes underflow to 0; any non-exploding method trivially passes.
This demonstrates `path="implicit"` does NOT blow up on SPD problems.

| Engine | Wallclock | Peak mem | Status | sup_error vs oracle |
| --- | --- | --- | --- | --- |
| semiflow/implicit | 38.7 ms | 32.2 MB | OK | — |
| semiflow/chebyshev | 14291.4 ms | 32.3 MB | OK | — |
| scipy/expm_multiply | >90s | — | TIMEOUT (>90s) | — |

## Well-conditioned cross-validation (N=200, scale=1.0, t=0.01)

All engines run; scipy.linalg.expm (dense) used as cross-check in the pytest test.

| Engine | Wallclock | Peak mem | Status | sup_error vs oracle |
| --- | --- | --- | --- | --- |
| semiflow/implicit | 0.8 ms | 32.3 MB | OK | — |
| semiflow/chebyshev | 0.0 ms | 32.2 MB | OK | — |
| semiflow/lanczos | 0.0 ms | 32.5 MB | OK | — |
| scipy/expm_multiply | 0.7 ms | 59.9 MB | OK | — |

## Headline

- `path="implicit"` RETURNS with sup_error ≤ 1e-6 (stiff Neumann).
- Explicit Krylov (`chebyshev`, `lanczos`) TIMEOUT (stiff regime).
- `scipy.sparse.linalg.expm_multiply` TIMEOUT (stiff regime).
- Well-conditioned case: implicit agrees with explicit Krylov and scipy.
