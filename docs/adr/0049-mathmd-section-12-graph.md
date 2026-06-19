# ADR-0049 — `contracts/semiflow-core.math.md` §12 — graph PDE: CITATION + NORMATIVE only, NO new theorems

- **Status**: PROPOSED
- **Date**: 2026-05-20
- **Wave**: v2.1 Wave A (Graph PDE Foundations)
- **Companion**: ADR-0047, ADR-0048, ADR-0050.

## Decision

Open a new top-level section §12 in `contracts/semiflow-core.math.md` titled
**"Graph PDE — discrete heat semigroup on finite weighted graphs (NORMATIVE library
choices; CITATION-only mathematics)"**.

§12 contains **NO new theorems**. Every mathematical statement is one of:

- a **citation** of an established result (Pazy 1983, Engel-Nagel 2000, Chung 1997)
  with verbatim hypothesis + conclusion + page reference, restated for `ℝ^N` with
  the finite-graph Laplacian playing the role of bounded `−A`;
- a **NORMATIVE library choice** — CSR layout invariants, stability-window scaling
  factor, normalization convention — that exists to pin the implementation to a
  single semantic and to be cited verbatim by rustdoc / ADRs.

Audience: anyone (reviewer, downstream user, future Wave 2.1B author) needing to
verify that `GraphHeatChernoff<F>` correctly implements the Chernoff product
formula on a finite-dimensional Hilbert space.

## Sub-section layout

| §     | Title                                                       | Classification |
|-------|-------------------------------------------------------------|----------------|
| §12   | Graph PDE — discrete heat semigroup on finite weighted graphs (NORMATIVE library choices; CITATION-only mathematics) | header |
| §12.1 | Setting: finite weighted graph `G = (V, E, w)` and its Laplacian `L_G` | NORMATIVE |
| §12.2 | Chernoff product formula on `ℝ^N` (CITATION: Pazy 1983 §1.3 Thm 1.3; Engel-Nagel 2000 §III.5 Thm 5.2) | CITATION |
| §12.3 | Order-1 leading Chernoff `S(τ) f = f − τ L_G f` — hypothesis check | CITATION + NORMATIVE |
| §12.4 | Stability envelope: `τ ≤ ½ ρ̄^{−1}`; Gershgorin upper bound `ρ̄` (CITATION: Chung 1997 §1.2–1.3) | NORMATIVE |
| §12.5 | CSR storage layout and invariants (NORMATIVE — see ADR-0048) | NORMATIVE |

## Hard rules

1. **No new theorems.** Statements that look like theorems must be restatements
   of cited results with the citation in-line, never freshly proved.
2. **No new lemmas.** If algebraic manipulation is needed (e.g. showing
   `S'(0) = −L_G` on `ℝ^N`), it is a single line of arithmetic, NOT framed as a
   lemma. Compare to §10.8 Lemma 10.1 (which IS a new lemma, and IS allowed
   because it ships under ADR-0024 and Wave 2.1A explicitly excludes its scope).
3. **Page-level citations.** Every CITATION block must cite (a) author + year,
   (b) section number, (c) theorem/equation number. Not just "Pazy 1983".
4. **NORMATIVE blocks must be implementable.** A NORMATIVE statement that no
   piece of code in `crates/semiflow-core/src/graph*.rs` enforces is a bug.
5. **Symbol consistency with rest of math.md.** Reuse `f` for state, `τ` for
   step, `t` for total time, `n` for number of Chernoff iterations, `N` for
   number of nodes. Do NOT redefine.

## §12.1 NORMATIVE content (outline)

**Setting.** A finite weighted undirected graph `G = (V, E, w)` with:

- `V = {0, 1, …, N − 1}` (node set).
- `E ⊆ {(u, v) : u, v ∈ V, u ≠ v}` (no self-loops).
- `w : E → (0, ∞)` (positive finite edge weights).

**Combinatorial Laplacian.** `L_G ∈ ℝ^{N×N}` with

```
L_G[i, j] = { −w(i, j)      if (i, j) ∈ E
            { deg_w(i)      if i == j
            { 0             otherwise
```

where `deg_w(i) = Σ_{j : (i,j) ∈ E} w(i, j)`.

**Properties.** `L_G` is symmetric positive semidefinite. Smallest eigenvalue
`λ_0 = 0` (eigenvector `1`); largest eigenvalue `λ_{N-1} ≤ 2 · max_i deg_w(i)`
(Gershgorin). Discrete heat semigroup `e^{−t L_G}` is well-defined and is a
strongly continuous contraction semigroup on `(ℝ^N, ℓ²)`.

**Symmetric normalized Laplacian** (opt-in, ADR-0048):
`L_sym = I − D^{−½} W D^{−½}` with `D = diag(deg_w)`. Eigenvalues lie in `[0, 2]`
per Chung 1997 §1.3.

## §12.2 CITATION content (outline)

> **Theorem (Chernoff product formula on Banach spaces, Pazy 1983, §1.3, Theorem
> 1.3, p. 19; Engel-Nagel 2000, §III.5, Theorem 5.2, p. 220).** Let `A` be the
> generator of a `C_0` semigroup `T(t)` on a Banach space `X` with `‖T(t)‖ ≤
> M e^{ωt}`. Let `S : [0, ∞) → ℒ(X)` be strongly continuous with `S(0) = I` and
> `‖S(τ)‖ ≤ M e^{ωτ}` for all `τ ≥ 0`. If `S'(0) f = A f` for all `f` in a
> core `D ⊂ D(A)`, then for every `t > 0`,
> ```
>   (S(t/n))^n f  →  T(t) f      strongly as n → ∞.
> ```
> Quantitative error: `‖(S(t/n))^n f − T(t) f‖ = O(1/n)` for `f ∈ D(A^2)`.

**Specialisation to `X = ℝ^N`, `A = −L_G`.** Since `−L_G` is bounded, `D(A) =
D(A^2) = ℝ^N`. The semigroup `e^{−t L_G}` is the matrix exponential, a
contraction (because `L_G ⪰ 0`). The above theorem applies verbatim.

## §12.3 CITATION + NORMATIVE content (outline)

**Hypothesis check for `S(τ) = I − τ L_G`** (with citations into §12.2):

1. `S(0) = I` — trivial.
2. `τ ↦ S(τ) f` is strongly continuous — `S(τ) f = f − τ L_G f` is linear in `τ`.
3. `S'(0) f = (d/dτ)|_{τ=0} (f − τ L_G f) = −L_G f` ✓.
4. `‖S(τ)‖_{op,ℓ²} ≤ 1` for `τ ∈ [0, 2 / λ_{N-1}]` — eigenvalues of `S(τ)` are
   `1 − τ λ_k`, all in `[−1, 1]` for this τ range. (CITATION: Engel-Nagel 2000
   Thm 5.2 hypothesis.) For `τ` beyond this window the operator-norm bound
   `‖S(τ)‖ ≤ 1 + τ · λ_{N-1}` is still finite and exponentially-bounded
   `‖S(τ)‖ ≤ e^{τ λ_{N-1}}` via `1 + x ≤ e^x`, so the Chernoff product still
   converges — see ADR-0047 §"Why one scratch borrow per step".

Therefore `(S(t/n))^n f → e^{−t L_G} f` per §12.2.

**Order claim**: `S(τ) = e^{−τ L_G} + O(τ²)` on `D(A^2) = ℝ^N` — by Taylor
expansion of `e^{−τ L_G} = I − τ L_G + ½ τ² L_G^2 + O(τ³)`. So global error
on a fixed time interval `[0, t]` is `O(1/n)` per the Pazy / Engel-Nagel theorem.

## §12.4 NORMATIVE content (outline)

**Stability scaling factor.** `semiflow-core` accepts the user's `τ` without
clamping. Quasi-contractivity (operator-norm `≤ 1`) is guaranteed iff

```
   τ ≤ ½ · ρ̄^{−1}                           (NORMATIVE, ADR-0047 growth())
```

where `ρ̄ = max_i Σ_j |L_G[i, j]|` is the Gershgorin spectral-radius bound
(CITATION: Varga 2000, *Matrix Iterative Analysis*, §1.5, Theorem 1.11). The
prefactor ½ is the literature standard (Engel-Nagel 2000 §III.5 Cor 5.5).
`Laplacian::spectral_radius_bound()` returns `ρ̄` for use by callers.

**`growth()` return value.** `GraphHeatChernoff::growth() → (M, ω) = (1.0, ρ̄)`.
The pair `(M, ω)` is the literature `‖S(τ)‖ ≤ M e^{ωτ}` quasi-contractivity
parameters (`chernoff.rs:135-142`).

## §12.5 NORMATIVE content (outline)

(Mirrors ADR-0048 verbatim — restated here in math.md so the math file is
self-contained for downstream users who read only the math contract.)

CSR layout invariants:
1. `row_ptr.len() == n_nodes + 1`, `row_ptr[0] == 0`, `row_ptr[n_nodes] ==
   col_idx.len() == vals.len()`.
2. `row_ptr` non-strictly monotonic.
3. `col_idx[row_ptr[i]..row_ptr[i+1]]` strictly sorted ascending, no duplicates.
4. No self-loops in `Graph<F>`. `Laplacian<F>` diagonal entry is the LAST entry
   of each row.
5. All `vals` finite. All `col_idx[k] < n_nodes`.
6. Immutable post-assembly.

## Acceptance criteria

1. §12 is appended to `contracts/semiflow-core.math.md` between current §11.4 and
   the existing §10.9 (which becomes §13 — renumbered to "Storage refactors for
   v0.13.0"; cross-references updated). Net diff: ~250 LoC added, ~5 LoC
   renamed.
2. No new theorem environments (no `**Theorem n.**` blocks) introduced by §12
   itself — only restatements of cited theorems.
3. Every CITATION block names author, year, section, page, theorem number.
4. Every NORMATIVE block is enforced by code in `crates/semiflow-core/src/graph*.rs`
   (cross-checked by reviewer-suckless).
5. Cross-reference index updated: `Theorem 6` and `Theorem 7` references untouched;
   new entries `§12.{1,2,3,4,5}` indexed under "Graph PDE" in the glossary.

## Renumber consequence

The existing §10.9 (storage refactors, line 4605) is renumbered to §13 to keep
§12 contiguous for the graph content. All ADR / rustdoc references to "§10.9"
become "§13" — affects `docs/adr/0022-parallel-tile-scratch.md` Amendment 1,
ADR-0034 Amendment 1, ADR-0019 Amendment 2 (per `git grep -nE 'math.md.*§10\.9'`,
3 hits expected). Engineer must perform this renumber atomically with the §12
introduction.

Alternative considered: keep §10.9 numbered, add §12 starting at §12.1 — fine,
but breaks the convention that storage-refactor sections sit after the
theory sections. The renumber is cleaner.
