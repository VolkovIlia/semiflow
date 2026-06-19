# ADR-0157 — Shift A: Hardware-native Chernoff spike (`remizov-gpu`, feature-gated)

**Status:** DEFERRED (not built at v9.0.0; advisory-only; withdraw-on-dep-budget-breach) · **Date:** 2026-06-08 · **Branch:** `feat/v9.0.0-planning`
**Theme:** v9.0.0 — GPU spike (Shift A, SPIKE-ONLY, advisory, R2 = locality)
**Gate:** `G_GPU_PARITY` (ADVISORY, NON-blocking, feature-gated `--features gpu`) · **No math.md section** (zero new mathematics)
**Parent:** ADR-0154 · **Source:** research §4

## Context

Resource **R2** (solver-free locality: only local shifts + scalar multiplies, no global solve) maps naturally onto data-parallel / non-von-Neumann hardware where sparse-LU and Krylov stall (they need global communication / irregular memory). A shift-and-scale + interpolation kernel is *embarrassingly stencil-like* → a GPU compute shader is trivial to emit. The CPU path is already excellent (45 ns HFT tail, SIMD + threads), so this is **pure perf engineering, not new mathematics** (research §4 confirms zero new math — hence NO `math.md` section for Shift A, ADR only). The only portable Rust path is **`wgpu`** (Vulkan/Metal/D3D12 + WASM/WebGPU); the genuinely novel hardware (analog crossbar, optical) is **Fourier-domain**, the *opposite* of the shift kernel's solver-free locality, and is rejected outright.

## Decision

Pursue Shift A **only as a strictly-optional, out-of-core, feature-gated spike**. **TRIZ resolution (research §4.3).** **ТП**: we want *maximum throughput* (⇒ specialized hardware / many deps) AND *minimal dependencies & portability* (suckless, `no_std`, Override #1 ≤3-direct-dep budget). **ФП**: the codebase must *use the GPU* AND *not depend on a GPU stack*. **ИКР**: a feature-gated optional backend that **compiles away to nothing when absent** — the `no_std` core stays pure `alloc`; a `--features gpu` path mounts a `wgpu` translator (the Hurd-translator pattern from the adapter registry). **ВПР**: the kernel's locality is the resource that makes a stencil shader trivial. Realize this as a **separate crate `remizov-gpu`** (mirroring how FFI/PyO3/WASM are separate crates), `wgpu`-only; the `no_std` `semiflow-core` gains NO GPU dependency. Analog/optical Fourier-domain ports are a permanent NON-GOAL (carried from ADR-0154 anti-directions).

## Consequences

`remizov-gpu` drags in a large graphics dependency tree — a direct collision with suckless "few deps" — which is **why it is isolated in a separate crate**: the `semiflow-core` dependency-count and `size_check` gates are **unchanged**. **Strongest failure mode (adversarial, research §4.5):** zero new math + the CPU path is already fast + **GPU `f64` is slow and NOT bit-identical to the AVX2/NEON path**, which breaks the **0-ULP parity culture** (ADR-0018) that *is* part of the moat. Therefore parity is **explicitly WAIVED** for this spike. Advisory gate **`G_GPU_PARITY`** (NON-blocking, `--features gpu`): 2D/3D heat, `wgpu` compute-shader vs SIMD `f64` CPU path — end-state agreement <1e-10 relative (NOT 0-ULP; GPU `f64` parity is waived and documented) AND ≥5× throughput over the CPU SIMD path at `N≥512` per axis to justify the dependency. **Hard constraint:** the gate lives in `remizov-gpu`; if the dep / binary-size budget is breached → the spike is **WITHDRAWN**, mirroring the v8.0.0 `EigenrotatedAnisotropicChernoff` "no crutches, strong result" withdrawal precedent (ADR-0137). No `math.md` section is authored (zero new mathematics). Gate PLANNED.

## v9.0.0 Deferral Record (2026-06-10)

`remizov-gpu` was **not built at v9.0.0**. The v9.0.0 ship prioritized Shift B (`ReverseChernoff`, HEADLINE) and Shift C (`TtChernoff`, co-HEADLINE). The dep/binary-size budget was not the deciding factor — this was an execution-bandwidth deferral. Advisory gate `G_GPU_PARITY` remains PLANNED. The `semiflow-core` dep budget (Override #1 ≤3 direct deps) is unchanged and unaffected by this deferral. The withdraw-on-dep-budget-breach posture is in force for any future release that picks up Shift A.
