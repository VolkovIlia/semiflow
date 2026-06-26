# ADR-0184 — Batched multi-channel evolve for graph-heat kernels + adjoints

- **Status**: Proposed (design only — Issue #10, branch `issue-10-batched-evolve`)
- **Date**: 2026-06-26
- **Supersedes**: none — purely additive over ADR-0047 (graph-heat), ADR-0051
  (Magnus graph), ADR-0115 (graph adjoint sensitivity), ADR-0180 (presampled
  adjoint), ADR-0031 (PyO3 GIL-release pattern).
- **Contract**: `contracts/graph-batched-evolve.contract.md`

## Context

The PyO3 bindings evolve **one channel (1-D state) per call**. Using the
graph-heat kernels + adjoints as a torch `autograd.Function` for a GNN diffusion
layer over an `[N, C]` feature matrix requires `C` separate PyO3 calls with
numpy↔Python round-trips on every forward AND backward. Profiled (90-node graph,
`C=4`, `n_steps=8`): raw 1-channel Rust evolve ≈ 580 µs; full 4-channel forward
≈ 7700 µs (only ≈ 30 % Rust, rest = per-channel Python/PyO3/numpy overhead);
torch dense-unroll fwd+bwd ≈ 620 µs. The binding is ≈ 4× behind a batched dense
baseline because of the `O(C)` Python loop + array conversions, even though the
GIL is released per call (ADR-0031). The single-channel Rust path is already
fast — the overhead is **entirely at the binding boundary**.

## Decision

Add a **batched multi-channel evolve** that evolves all `C` channels in ONE Rust
call under a SINGLE `py.detach`, for the six forward kernels (`GraphHeat`,
`GraphHeat4th`/`GraphHeat6`, `MagnusGraphHeat`, `MagnusGraphHeat6`,
`VarCoefGraphHeat`, `VarCoefMagnusGraph`) and the adjoint stack
(`GraphAdjoint(.Presampled).evolve_state_adjoint`, `edge_weight_grad`). New
methods are ADDITIVE (`*_batched`); existing 1-D `evolve` is byte-unchanged.

### D1 — Memory layout: Python canonical `[N, C]`; core canonical `[C, N]`; the transpose is dissolved into the mandatory GIL-boundary copy

There is a genuine contradiction: the array must be `[N, C]` (torch-native →
zero Python-side copy) AND `[C, N]` (each channel contiguous → the existing 1-D
kernel runs unchanged with best cache locality). Resolved (not compromised) via
TRIZ ИКР in the **super-system**: a kernel running under a released GIL cannot
borrow the numpy buffer, so the input **must** be copied into an owned
`Vec<f64>` regardless. That mandatory marshalling copy is performed with a
*transposing* access pattern — `ndarray.column(c)` strided-gather of `[N, C]`
into the contiguous core buffer `[C, N]` — so the transpose **is** the boundary
copy, not an extra pass. Cost of the reorganisation = 0 beyond the copy already
required; the `n_steps` inner applies all run on contiguous data; output is
scattered back to `[N, C]` inside the equally-mandatory result copy.
**Python contract = `[N, C]`** (torch GNN feature matrix, passed as-is — no
`.t()`). **Core contract = `[C, N]` flat, channel-major** (channel `c` occupies
`cols[c*N .. c*N + N]`), zero-dep, each channel a trivially-contiguous subslice.
The strided gather/scatter is `O(N)` per channel, amortised over `n_steps ≥ 1`.

### D2 — Single GIL release, C-loop entirely in Rust

The PyO3 wrapper does: (1) pre-flight (GIL held) — validate, gather `[N, C]`
PyReadonlyArray2 → owned `[C, N]` `Vec<f64>`; (2) compute (`py.detach`, GIL
released) — the whole C-loop in pure Rust; (3) post-flight (GIL held) — scatter
the `[C, N]` result into one `[N, C]` `PyArray2`. ONE GIL release, ONE numpy
input read, ONE numpy output write per batched call. No per-channel round-trip.
Mirrors the v0.11.0 I6 pattern (ADR-0031, commit 07a4689, `Heat1D.evolve` via
`py.detach`) extended from 1-D to 2-D.

### D3 — Batching strategy: thin loop over the existing single-channel step, with channel-independent work hoisted; optional channel-parallelism feature-gated

Channels are mathematically independent (same graph operator, different RHS), so
the batched method is a **thin loop over the existing single-channel evolve** —
NO new per-channel math (suckless: reuse, do not reimplement). Two reuse wins
the binding loop cannot get today are realised in core: (a) the ping-pong
buffers + `ScratchPool` are allocated ONCE and reused across channels;
(b) channel-independent operator work — Laplacian assembly (heat) and the
GL₄ Laplacian/a-sequence sampling (Magnus/VarCoef, per ADR-0180) — is hoisted
**out** of the channel loop and shared by all channels. Per-channel parallelism
(`std::thread::scope`, ADR-0018) is **feature-gated behind `parallel` and a
channel-count threshold** (`C ≥ 2`): channels are embarrassingly parallel with
no cross-channel reduction in the **forward** path, so a parallel batched
forward is **bit-identical** to the serial loop (0 ULP). Each worker owns its
own `ScratchPool`. Intra-channel SpMV SIMD (the existing `simd` feature) is
orthogonal and composes. v1 ships **serial** core loops + the hoist; the
`parallel` channel-split is specified but may land as a follow-up — the
bit-equality gate (D5) is independent of which is used.

### D4 — Adjoint batching: stacked state-VJP `[C, N]`, summed edge-weight grad

For a multi-channel loss `J = Σ_c J_c(u_n^c)`, by linearity of the inner product
in the discrete adjoint `∂J/∂θ = Σ_k ⟨λ_{k+1}, (∂S_k/∂θ) u_k⟩`
(`contracts/semiflow-core.math.md` §43.4, NORMATIVE) the contributions sum over
channels. Therefore: `evolve_state_adjoint_batched` takes `C` RHS columns
`[C, N]` and returns the **stacked** state-VJP `[C, N]` (each `λ^c` evolved
independently — embarrassingly parallel, like the forward path);
`adjoint_state_gradient_batched` (the core of `edge_weight_grad`) takes `C` RHS
columns + `C` `u0` columns and returns ONE **summed** `∂J/∂w` of length
`n_params` (`Σ_c ∂J_c/∂w`). To preserve bit-equality with the per-channel Python
loop (which sums `g_0 + g_1 + … + g_{C-1}` in index order), the summed gradient
is accumulated **in ascending channel index** — `grad_theta` is zeroed once,
then each channel's contribution is added without re-zeroing. If channel-parallel
is enabled for the grad path, per-channel partials are reduced in ascending
index order (NOT in thread-completion order) to keep the 0-ULP identity.

### D5 — Correctness gate: structural bit-equality, no new math oracle

`batched == per-channel-loop` is a structural identity (the batched path calls
the SAME single-channel kernel `C` times), so a **0-ULP bit-equality gate**
suffices — no new numerical claim, hence **no new sympy oracle**. A Rust
`#[test]` (and Python parity test) asserts, for every kernel + the two adjoint
methods, that the batched `[C, N]` output equals the column-wise per-channel
output exactly (`assert_eq!` on the bit pattern). The existing §43.6
finite-difference oracle (`T_ADJOINT_STATE_SENSITIVITY`) still covers the
underlying gradient correctness, unchanged.

### D6 — Wheel features (sub-issue): published wheel is built WITHOUT `simd`/`parallel`

`crates/semiflow-py/Cargo.toml` declares
`semiflow = { path = "../semiflow", default-features = false, features = ["std"] }`
— this **disables the core `default = ["simd"]`** (ADR-0019) and never enables
`parallel`. `pyproject.toml` (`[tool.maturin] features = ["pyo3/extension-module"]`)
and `.github/workflows/release-wheels.yml` (cibuildwheel) add nothing. So the
PyPI wheel today has **no SpMV SIMD and no channel parallelism compiled in** —
the batched API's hoist still helps, but the per-channel SIMD/parallel speedups
are absent. Recommended quick win (DO NOT apply in this design ADR — separate
change): set `features = ["std", "simd"]` (and, once D3's channel-parallel
lands, `"parallel"`) on the `semiflow` dependency in `semiflow-py/Cargo.toml`.
This is the minimal, single-line enabler; verify wheel build time + manylinux
target-feature baseline (SIMD uses runtime feature detection per the core SIMD
path, so manylinux portability is preserved).

## Consequences

- **Positive**: removes the `O(C)` Python-loop overhead (target: batched-C
  forward ≈ raw-1ch × small constant, not × C); zero new core math; no breaking
  change; reuses buffers + hoists operator assembly; bit-identical to the 1-D
  path; unblocks a torch `autograd.Function` GNN diffusion layer in `revssm`
  without per-channel marshalling.
- **Negative / risk**: strided gather/scatter at the `[N, C]`↔`[C, N]` boundary
  has a worse access pattern than a contiguous copy (amortised, acceptable);
  channel-parallel grad reduction must be order-pinned (D4) or the 0-ULP gate
  breaks; six forward kernels + two adjoint paths = a non-trivial but mechanical
  surface — kept suckless via one generic helper for the three plain-`GraphSignal`
  ChernoffFunction kernels + typed helpers for the Magnus/VarCoef trajectory
  kernels (see contract).
- **Boundary (ADR-0115 reaffirmed)**: core ships the batched MATH primitives
  only; the `torch.autograd.Function`/neural-ODE tape stays in `revssm`.
