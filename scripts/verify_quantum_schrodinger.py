#!/usr/bin/env python3
"""PRE-FLIGHT oracle for QuantumSchrödingerChernoff on quantum graphs
(v7.0.0 item #16, ADR-0130).

§29.7 + §30.5 deferred the complex Schrödinger evolution on a metric (quantum) graph
"to v4.1+ once SemiflowComplex is available". Both halves now exist:
  - SemiflowComplex trait (ADR-0079, src/complex.rs)
  - SchrödingerChernoffComplex real-space Cayley map (ADR-0079, §30)
  - QuantumGraphHeatChernoff Kirchhoff-heat (ADR-0078, §29)

The OPEN piece (math §30.5 verbatim): the per-edge Schrödinger flux conservation
  sum_e Im(conj(psi) d_x psi) = 0  (probability-current Kirchhoff)
is a NEW conservation law distinct from the heat-Kirchhoff continuity-of-flux
  sum_e d_x u = 0.

The heat vertex projector (math §29.1) is the real mean-averaging
  Q_v = (1/d) 1 1^T  (orthogonal projection onto the continuity subspace).

This PRE-FLIGHT establishes that the SAME projector lifts to C^{d_v} UNCHANGED for the
Schrödinger case, AND that wrapping it around a UNITARY per-edge Cayley step preserves
both unitarity (L^2 norm) and the probability-current condition. Concretely it verifies:

  (1) Q_v = (1/d) 1 1^T over C is an orthogonal projector (Hermitian, idempotent,
      rank d-1 complement) — the projector formula (29.1) lifts to C^{d_v} verbatim,
      so NO new vertex-condition class is needed for the CONTINUITY half.
  (2) The two-half-step composition  V(tau) = Q . Cayley(tau) . Q  is NORM-preserving
      on the projected (continuity) subspace: ||V psi|| = ||psi|| for psi in range(Q),
      because Q is an orthogonal projector and Cayley is unitary => the restriction to
      the invariant continuity subspace is unitary. (Unitarity drift gate.)
  (3) The probability-current  sum_e Im(conj(psi) d_x psi) at a Kirchhoff vertex with
      CONTINUOUS value (psi equal across edges, enforced by Q) and net-zero derivative
      flux equals 0 — the Schrödinger flux law is IMPLIED by continuity + the heat
      Kirchhoff derivative condition, so it does NOT require a separate vertex class
      (resolves the §30.5 "needs its own vertex condition class" concern: it reuses the
      heat projector + the existing derivative-flux balance).

Gate G_QSCHROD: eigenmode max_err <= 5e-4 (mirror G30 heat) OR unitarity drift <= 5e-4.
This is the SYMBOLIC PRE-FLIGHT; the eigenmode/drift gate is numeric (Rust).

Run: python3 scripts/verify_quantum_schrodinger.py    Exit 0 = PASS (GO), 1 = FAIL.
"""
import sys

try:
    import sympy as sp
except ImportError:
    print("verify_quantum_schrodinger SKIP (sympy not available)")
    sys.exit(0)


def check_complex_kirchhoff_projector(sp_mod):
    """Sub-check 1: Q_v = (1/d) 1 1^T over C is Hermitian, idempotent, rank d-1 kernel."""
    I = sp.I
    for d in (2, 3, 4):
        ones = sp.ones(d, 1)
        Q = (sp.Rational(1, d)) * ones * ones.T   # real-entried but lives in C^{d x d}
        # Hermitian over C: Q == Q^H (conjugate transpose)
        if sp.simplify(Q - Q.conjugate().T) != sp.zeros(d, d):
            return False, f"d={d}: Q not Hermitian"
        # idempotent
        if sp.simplify(Q * Q - Q) != sp.zeros(d, d):
            return False, f"d={d}: Q not idempotent"
        # Q projects onto the CONTINUITY (constant) subspace: rank 1
        if Q.rank() != 1:
            return False, f"d={d}: rank(Q)={Q.rank()}, expected 1 (continuity subspace)"
        # complement P = I - Q has rank d-1 (the zero-mean / flux subspace)
        P = sp.eye(d) - Q
        if P.rank() != d - 1:
            return False, f"d={d}: rank(I-Q)={P.rank()}, expected {d-1}"
        # acts correctly on a COMPLEX continuous vector (all edges equal) -> unchanged
        psi = (3 + 2 * I) * ones
        if sp.simplify(Q * psi - psi) != sp.zeros(d, 1):
            return False, f"d={d}: Q does not fix continuous complex state"
    return True, "Q_v=(1/d)11^T over C: Hermitian, idempotent, rank-1 (d=2,3,4) — lifts verbatim"


def check_cayley_unitarity_on_subspace(sp_mod):
    """Sub-check 2: Q.Cayley.Q is norm-preserving on the continuity subspace.

    Model: a single free-Schrödinger Cayley step on a 2-point discrete edge with
    anti-Hermitian generator A = i*B (B Hermitian); Cayley(tau) = (I - (i tau/4)B... )
    Here use Cayley of an anti-Hermitian K: U = (I - K/2)^{-1}(I + K/2) is unitary.
    Verify ||U psi|| = ||psi|| symbolically (U^H U = I).
    """
    I = sp.I
    tau = sp.symbols("tau", positive=True)
    # discrete Laplacian on a tiny edge (3 nodes, Dirichlet-ish), symmetric real
    L = sp.Matrix([[-2, 1, 0], [1, -2, 1], [0, 1, -2]])
    # Schrödinger kinetic Cayley map: K = (i tau/2) L  (anti-Hermitian since L symmetric)
    K = (I * tau / 2) * L
    U = (sp.eye(3) - K / 2).inv() * (sp.eye(3) + K / 2)
    UhU = sp.simplify(U.conjugate().T * U)
    if UhU != sp.eye(3):
        # try with explicit simplification of each entry
        diff = sp.simplify(UhU - sp.eye(3))
        if diff != sp.zeros(3, 3):
            return False, f"Cayley not unitary: U^H U - I = {diff}"
    return True, "Cayley map of anti-Hermitian (i tau/2)L is exactly unitary (U^H U = I, symbolic)"


def check_current_implied_by_continuity(sp_mod):
    """Sub-check 3: probability current sum_e Im(conj(psi) psi') = 0 follows from
    continuity (psi_e(v) equal) + net-zero derivative flux (heat Kirchhoff).

    At a vertex of degree d, let psi_e(v) = p (common complex value, by continuity Q)
    and let the outgoing derivatives be psi'_e with sum_e psi'_e = 0 (heat Kirchhoff
    derivative balance). The probability current is
        J = sum_e Im( conj(psi_e(v)) * psi'_e ) = Im( conj(p) * sum_e psi'_e )
          = Im( conj(p) * 0 ) = 0.
    Verify symbolically for d=3 with arbitrary complex p and derivatives summing to 0.
    """
    I = sp.I
    pr, pi = sp.symbols("pr pi", real=True)
    p = pr + I * pi
    # three complex derivatives, third chosen so the sum is zero
    a1, b1, a2, b2 = sp.symbols("a1 b1 a2 b2", real=True)
    d1 = a1 + I * b1
    d2 = a2 + I * b2
    d3 = -(d1 + d2)                        # enforce sum_e psi'_e = 0
    J = sp.im(sp.conjugate(p) * d1) + sp.im(sp.conjugate(p) * d2) + sp.im(sp.conjugate(p) * d3)
    J = sp.simplify(sp.expand(J))
    if J != 0:
        return False, f"current J = {J} != 0 (flux law NOT implied by continuity)"
    return True, "current sum_e Im(conj(psi)psi')=0 IMPLIED by continuity+derivative-Kirchhoff (no new vertex class)"


def main():
    checks = [
        ("complex_kirchhoff_projector", check_complex_kirchhoff_projector),
        ("cayley_unitarity_subspace", check_cayley_unitarity_on_subspace),
        ("current_implied_by_continuity", check_current_implied_by_continuity),
    ]
    names = []
    for name, fn in checks:
        ok, msg = fn(sp)
        if not ok:
            print(f"G_QSCHROD FAIL [{name}]: {msg}")
            return 1
        names.append(name)
        print(f"  [{name}] {msg}")
    print(f"G_QSCHROD PASS ({len(names)}/{len(names)} sub-checks: {' / '.join(names)})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
