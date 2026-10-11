# §65 — Separable sampling for the constant-coefficient N-D shift engines (ADR-0214, NORMATIVE library)

## §65.1 — Statement

For `SmolyakGridND` / `AnisotropicShiftChernoffND` (§32.4, eq. 32.3),
`(S(τ)f)_k = e^{τc(x_k)} π^{−D/2} Σ_q w_q f(y_q(k))`,
`y_q(k)_d = (x_{k,d} + τ b_d(x_k)) + 2√τ (L_k η_q)_d`, and `f(y)` the tensor-product
sample of `GridFnND::sample` (ADR-0191):
`f(y) = Σ_{j_{D−1}} w^{(D−1)}_{j_{D−1}} bc_{D−1}( … Σ_{j_0} w^{(0)}_{j_0} bc_0(f[…]) … )`,
axis 0 innermost.

**Claim (NORMATIVE).** If `b(x_k) ≡ b` and `L_k ≡ L` bitwise, then for every `q`
the per-axis weights and resolved boundary hits of axis `d` depend only on `k_d`,
and `f(y_q(k))` equals the result of the passes
`G_{−1} = f`, `G_d[k] = Σ_j w^{(d)}_j(k_d) · bc_d(hit_j(k_d), i ↦ G_{d−1}[k|_{k_d = i}])`
(`d = 0 … D−1`, accumulation `0 + w_0v_0 + w_1v_1 + …`), bit for bit for every
non-NaN result (a NaN result stays NaN; its payload is unspecified by IEEE 754).
*Proof.* By induction on `d`: the recursion's value of the lower-axis collapse at
higher-axis indices `(i_{d+1}, …)` uses the output node's stencils on axes `≤ d`
(unchanged by higher-axis hits) — that is `G_d` at the position with those
higher-axis indices. The boundary resolver `bc_value_from_hit` only calls its
accessor, which returns the same number either way. ∎

## §65.2 — Prefix sharing

`G_d` depends on `q` only through the offsets `(2√τ(Lη_q)_e)_{e ≤ d}`. Consecutive
`q` (stored order) with bitwise-equal offset prefixes share `G_0 … G_d`. The final
sum `Σ_q w_q G_{D−1}` is accumulated per node in stored `q` order, as the per-node
loop does.

## §65.3 — Gate and limits

Gate `G_ND_SWEEP_BIT_EQUAL` (0 ULP against the pre-ADR-0214 loop for non-NaN outputs;
NaN outputs coincide as a class). Limits:
variable `b` or `L` (e.g. the nightly tanh-coupled `A` gates), non-primitive
scalars (`Dual`), and scratch above `2²⁴` elements use the per-node loop.
