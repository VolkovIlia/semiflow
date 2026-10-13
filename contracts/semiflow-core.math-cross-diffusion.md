# §70 — Cross-diffusion with a constant matrix (ADR-0213, NORMATIVE library)

## §70.1 — Modal factorisation

For `∂ₜu = (D ⊗ L)u` with `u = (u_1, …, u_K)`, `L` a linear operator applied
identically to every species and `D = PΛP⁻¹` (real `Λ = diag(λ_k)`, `λ_k ≥ 0`):
`(D ⊗ L) = (P ⊗ I)(Λ ⊗ L)(P⁻¹ ⊗ I)`, hence

`e^{t(D⊗L)} = (P ⊗ I)·diag_k(e^{tλ_k L})·(P⁻¹ ⊗ I)`.

A Chernoff step replaces `e^{τλ_k L}` by a mode engine `C_k(τ)`; the step error is
`(P ⊗ I)·diag_k(C_k(τ)ⁿ − e^{nτλ_k L})·(P⁻¹ ⊗ I)`, so for initial data `c ⊗ v`

`|error_i| ≤ Σ_k |P_ik|·|(P⁻¹c)_k|·‖C_k(τ)ⁿv − e^{nτλ_k L}v‖`.

## §70.2 — Decomposition (NORMATIVE)

`D` (`K×K`, `1 ≤ K ≤ 16`, finite), `s = ‖D‖_max = m·2^e`: the iteration runs on
`2^{−e}D` (exact). Eigenvalues by Hessenberg reduction with stabilised elimination
and Francis double-shift QR (≤ 60 iterations per eigenvalue; a non-finite value or
no convergence → `DomainViolation`). `|Im λ| > 1e−6·s` → complex; `λ < −1e−6·s` →
negative; else `λ ← max(Re λ, 0)`, sorted ascending (a later reconstruction failure
caused by a dropped `Im λ` or clamped `λ` is reported as complex / negative).
Eigenvectors: for widths `w ∈ (1e−6, 1e−9, 1e−12, 1e−14, 0)·s` in turn, consecutive
`λ` within `w` of the cluster's first form a cluster of size `m`; its basis is the
null space of `D − λ̄I` after exactly `K − m` complete-pivoting elimination steps
(an exactly zero pivot earlier → defective); if `m > 1` and
`max_j ‖Dv_j − λ_j v_j‖_∞ > 1e−10·s/K`, the cluster is refined at the next width.
Accept only if, in `f64`, `cond₁(P) ≤ min(1e8, 1e−2/ε_F)` and
`‖PΛP⁻¹ − D‖_max ≤ 1e−10·s`; `F` stores the decomposition rounded. `D = 0`:
`P = I`, `Λ = 0`.

## §70.3 — Step

`ReactionDiffusion` with cross-diffusion: diffusion step = gather species
(species-major `u[k·N + i]`), `m = P⁻¹u` per node, `m_k ← C_k(τ)m_k`, `u = Pm` per
node, scatter. Reaction steps act on `u` (unchanged, ADR-0208). Admissibility: every
species field `mixes_with` the first (same nodes, interpolant, homogeneous linear
boundary; not `Dirichlet { value ≠ 0 }`), else `DomainViolation`.

## §70.4 — Gates

`G_CROSSDIFF_GRAPH_EXACT`, `G_CROSSDIFF_STRANG_ORDER`, `G_CROSSDIFF_FOURIER_GRID`,
`G_CROSSDIFF_REJECT` (ADR-0213).
