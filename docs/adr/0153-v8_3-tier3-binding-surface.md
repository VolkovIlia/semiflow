# ADR-0153 — v8.3.0 TIER-3 binding surface (Wentzell, ResolventJumpND, Obstacle Γ)

**Status:** ACCEPTED (2026-06-09) · **Branch:** `feat/v8.3.0-bindings`
**Supersedes:** the TIER-3 *deferral* rows for the three v8.2.0 math kernels (now
scheduled, not deferred). **Cross-refs:** ADR-0138 (v8.1.0 TIER-3 binding surface,
structure mirrored), ADR-0028 Amendment 2 (per-crate-dup mandate, binding-scope
tiering), ADR-0076 (additive `_v3`/`V8` surface), ADR-0031 (PyO3 three-phase GIL
release), ADR-0151 (Wentzell Cayley kernel NARROW scope), ADR-0148 (ResolventJump
2D/3D NARROW-parabolic scope), ADR-0152 (obstacle Γ ill-posedness),
`.dev-docs/reports/V8_3_TIER3_BINDING_DESIGN.md`.

## Context

v8.2.0 shipped three new core kernels with **no binding surface**:
**`DynamicWentzellChernoff`** (C-9, time-dependent Wentzell/Robin BC via implicit
Cayley boundary step, `gamma: fn(F)->F` + `c`), **`ResolventJumpChernoff2D/3D`**
(B-5, TWS parabolic-contour `e^{tA}g` over `Grid2D`/`Grid3D`), and the inherent
primitive **`ObstacleChernoff::apply_inactive_gamma_into`** (B-7, inactive-set Γ on
the open continuation set, plus `ObstacleChernoffND` D≥2 forward evolution). The Rust
core is fully usable; the FFI/PyO3/WASM siblings lag. v8.3.0 closes this debt.

Two non-trivial ABI seams forced a design pass, not a copy: (1) the Wentzell kernel
carries a host-supplied `γ(t)` function-pointer that cannot soundly cross the
`py.detach` GIL-release / WASM boundary; (2) the obstacle Γ primitive returns a count
AND a companion `defined: &mut [bool]` refusal mask, so a single real-valued buffer
copy does not capture the surface. The ND resolvent and ObstacleND additionally
re-raise the row-major layout question (v8.1.0 bugs C1/F4 were C-vs-Fortran order).

## Decision

**(1) γ(t) ABI — PRE-SAMPLED γ-SCHEDULE (Altshuller preliminary-action #10 +
copying #26).** γ is frozen at the left endpoint of each Chernoff step (Howland
freeze, `t_k = t_offset + k·τ`, `τ = t/n_steps`), so the sample grid is deterministic
up front. The **primary ABI on all three bindings (uniform)** passes a flat
`gamma_schedule: &[f64]` of length `n_steps`; the GIL-off kernel reads
`gamma_step[k]` instead of `(gamma)(t_k)`. The host evaluates ITS OWN arbitrary γ at
the sample times BEFORE detaching. Length-1 = constant-γ for free; cross-binding 0-ULP
parity is trivial (identical f64 arrays → identical core path). PyO3+WASM add an
**ergonomic-sugar** `GammaFamily { Constant, Linear, Exponential }` constructor that
expands internally to a schedule ("covers 90% ergonomically; use the schedule overload
for arbitrary γ"). A NORMATIVE rustdoc note requires host sampling to match the kernel
freeze point exactly (else silent order-1 error) and validates `γ ≥ 0`, finite
(NanInf/OutOfDomain), and schedule length (GridMismatch).

**(2) Obstacle Γ two-output return.** `apply_inactive_gamma_into` returns a `usize`
count AND writes a companion `defined: &mut [bool]` mask. PyO3 returns a tuple
`(gamma: np.ndarray[f64], defined: np.ndarray[bool], count: int)`; FFI uses two
out-params (`double* gamma_out`, `uint8_t* defined_out`) plus the returned count;
WASM returns `{gamma: Float64Array, defined: Uint8Array, count: number}`. A `false`
mask entry means "Γ undefined here", NEVER "Γ = 0" — callers MUST consult it.

**(3) ND numpy layout — Fortran order (NORMATIVE).** Rust ND state is axis-0-fastest
(column-major). The Python side MUST `.ravel(order="F")` on input and reshape with
`order="F"` on output for ResolventJump 2D/3D and ObstacleND. This is recorded as a
NORMATIVE contract; it is the direct fix for the v8.1.0 C1/F4 C-vs-Fortran bug class.

**(4) Tiering by kernel shape, not importance.** Wentzell + ResolventJumpND are
evolver-style (real-valued scalar/buffer `apply`/`jump`) → **FULL FFI+PyO3+WASM**.
Obstacle Γ + ObstacleND are a research/analysis surface (Γ second-derivative + bool
mask, primary consumer Python/numpy) → **PyO3-first TIER-2**; FFI/WASM are
opportunistic and may slip within v8.3.x with zero headline impact (same posture as F3
KilledDirichlet1D in v8.0.0). **(5) NARROW-scope echo:** each binding repeats its
kernel's NARROW limitation in rustdoc (Wentzell 1D half-line collapse + order-1;
ResolventJumpND self-adjoint/sectorial parabolic only; obstacle Γ refused at the kink).

## Consequences

Per-crate duplication is REQUIRED (ADR-0028 Amdt 2): each of `remizov-{ffi,py,wasm}`
owns its boundary code with NO shared util. No new dependency in any crate; FFI bodies
stay `catch_panic!`-wrapped under `[profile.release-ffi]` (panic=unwind), PyO3 compute
releases the GIL via `py.detach`, WASM uses `[profile.release]` (panic=abort) + `Result
<_, JsValue>`. No complex type leaks any boundary (TWS contour math stays sealed; only
real-valued `jump`/Γ outputs cross). Each new binding file stays ≤500 LoC (default cap;
child-splits only if a kernel would exceed — see design report §6). Three 0-ULP
cross-binding parity gates are added — `G_BINDING_WENTZELL_PARITY`,
`G_BINDING_RESOLVENT_JUMP_ND_PARITY`, `G_BINDING_OBSTACLE_GAMMA_PARITY` — modelled on
`G_BINDING_RESOLVENT_JUMP_PARITY`. Contracts: `traits.yaml` 4.12.0 → **4.13.0** (binding
entry-point changelog, no core type changed — additive crate-local wrappers per the
ADR-0076 precedent); `properties.yaml` 4.13.0 → **4.14.0** (three new gates, no existing
gate changed, no removal). (NB: the working-tree schema versions are 4.12.0/4.13.0 from
the v8.2.0 math layer — grep-confirmed — so the additive bumps land at 4.13.0/4.14.0.)

## Rejected Alternatives

**FFI-only true `double(*)(double)` γ fn-pointer.** Sound across the C ABI (no GIL,
no JS boundary), but rejected for **signature uniformity**: a fn-pointer ABI on FFI
diverges from the schedule ABI PyO3/WASM are forced into, fracturing the one-golden
parity plumbing and forcing a second core seam. Binding the same `gamma_schedule`
array on FFI keeps all three bindings on a single path. **Per-step host callback
across the boundary** (call back into Python/JS for `γ(t_k)` from inside the GIL-off /
WASM sweep). Rejected: re-acquiring the GIL (or crossing into JS) per step destroys the
`py.detach` performance win, is unsound while the GIL is released, and has no analogue
in WASM. The pre-sampled schedule resolves the apparent "arbitrary-γ vs sound-boundary"
contradiction by exploiting a resource already in the topology — the freeze grid is
deterministic, so the host can sample ahead of time and the boundary only ever carries
inert f64 data. No genuine irreducible contradiction remains; no TRIZ compromise forced.
