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

`D` (`K×K`, `1 ≤ K ≤ 16`, finite): eigenvalues by Hessenberg reduction with
stabilised elimination and Francis double-shift QR (≤ 60 iterations per eigenvalue,
else `DomainViolation`). With `s = ‖D‖_max`: `|Im λ| > 1e−6·s` → complex (rejected);
`λ < −1e−6·s` → negative (rejected); `λ ← max(λ, 0)`, sorted ascending. Clusters:
consecutive `λ` within `1e−6·s`; per cluster the null space of `D − λ̄I` (complete
pivoting, pivots `≤ 1e−6·s` count as zero) must have the cluster's size, else
defective. Accept only if `cond₁(P) ≤ 1e8` and `‖PΛP⁻¹ − D‖_max ≤ 1e−10·s`.
`D = 0`: `P = I`, `Λ = 0`.

## §70.3 — Step

`ReactionDiffusion` with cross-diffusion: diffusion step = gather species
(species-major `u[k·N + i]`), `m = P⁻¹u` per node, `m_k ← C_k(τ)m_k`, `u = Pm` per
node, scatter. Reaction steps act on `u` (unchanged, ADR-0208). Admissibility: every
species field `mixes_with` the first (same nodes, interpolant, homogeneous linear
boundary; not `Dirichlet { value ≠ 0 }`), else `DomainViolation`.

## §70.4 — Gates

`G_CROSSDIFF_GRAPH_EXACT`, `G_CROSSDIFF_STRANG_ORDER`, `G_CROSSDIFF_FOURIER_GRID`,
`G_CROSSDIFF_REJECT` (ADR-0213).
