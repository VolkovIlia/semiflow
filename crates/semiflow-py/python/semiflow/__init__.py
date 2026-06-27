# Re-export everything from the native extension module.
# This __init__.py enables the mixed maturin layout (python-source = "python")
# while preserving the flat import API: `from semiflow import Heat1D`.
from .semiflow import (  # pyright: ignore[reportMissingImports]
    Heat1D,
    Heat1D4th,
    Heat1D6th,
    Heat2D,
    Heat3D,
    DriftReaction1D,
    Graph,
    GraphPath,
    GraphHeat,
    GraphHeat4th,
    Laplacian,
    MagnusGraphHeat,
    MagnusGraphHeat6,
    SemiflowError,
    Schrodinger1D,
    Shift1D,
    NonSeparable2D,
    Adjoint,
    AdaptivePI,
    VarCoefGraphHeat,
    GraphHeat6,
    VarCoefMagnusGraph,
    version,
    # v3.0 surface (ADR-0076 Wave E — additive)
    GrowthV3,
    EvolverHeat1DUnitV3,
    # v4.1 Phase D — PyO3 parity for Heisenberg / ζ⁴ / ζ⁶ APIs
    HypoellipticChernoffHeisenberg,
    heisenberg_heat_kernel,
    Heat1DZeta4,
    Heat1DZeta6,
    # ADR-0111 Wave P1 — 1-D diffusion completeness
    Heat1DZeta8,
    TruncatedExp1D,
    TruncatedExp4th1D,
    Strang1D,
    # ADR-0111 Wave P2 — complex Schrödinger
    SchrodingerComplex1D,
    # ADR-0111 Wave P3 — boundary-condition kernels
    Resolvent1D,
    Killing1D,
    Reflected1D,
    Robin1D,
    DirichletHeat2nd1D,
    # ADR-0111 Wave P4 — nonautonomous + subordinated
    Howland1D,
    Subordinated1D,
    # ADR-0111 Wave P5 — geometry: manifold + hypoelliptic backends
    Manifold2D,
    HypoellipticChernoffKolmogorov,
    HypoellipticChernoffEngel,
    # ADR-0111 Wave P6 — quantum graphs, matrix diffusion, point-eval, graph traj
    QuantumGraph,
    QuantumGraphHeat,
    MatrixDiffusion1D,
    PointEval,
    sample_gridfn2d,
    GraphTraj,
    StrangGraph,
    # ADR-0111 Wave P7 — multi-D anisotropic + 2D/3D variable-coefficient
    AnisotropicShiftND2,
    AnisotropicShiftND3,
    NonSeparable2DAniso,
    Heat2DVarA,
    Heat3DVarA,
    # Issue #1 — adjoint-state parameter-sensitivity (ADR-0115)
    # Issue #10 — batched evolve (ADR-0184)
    GraphAdjoint,
    GraphAdjointPresampled,
    edge_weight_grad,
    edge_weight_grad_batched,
    # v6.3.0 — obstacle / variational-inequality Chernoff (math §44)
    ObstacleChernoff,
    # v8.0.0 F1 — Dual-AD Greeks (ADR-0133 A3)
    EvolverHeat1DGreeksV3,
    KilledDirichlet1D,
    # v8.1.0 F2 — ResolventJumpV8 (TWS contour, TIER 1, ADR-0138)
    ResolventJumpV8,
    # v8.1.0 C2 — AdjointFokkerPlanckV8 (Lemma A.1 push, TIER 1, ADR-0138)
    AdjointFokkerPlanckV8,
    # v8.1.0 C1 — SmolyakD6V8 (sparse-grid D=6, unit a=I, TIER 2, ADR-0138)
    SmolyakD6V8,
    # v8.1.0 F4 — ComplexTripleJumpV8 (filiform-N5, apply_real, TIER 2, ADR-0138)
    ComplexTripleJumpV8,
    # v8.3.0 C-9 — WentzellV8 + GammaFamily (dynamic Wentzell/Robin BC, TIER 1, ADR-0153)
    WentzellV8,
    GammaFamily,
    # v8.3.0 F2-ND — ResolventJump2DV8 + ResolventJump3DV8 (2D/3D parabolic, TIER 1, ADR-0153)
    ResolventJump2DV8,
    ResolventJump3DV8,
    # v8.3.0 B-7 — ObstacleGammaV8 (inactive-set Γ, two-output, TIER 2, ADR-0153)
    # and ObstacleNDV8 (D=2 forward evolution, Fortran-order, TIER 2, ADR-0153)
    ObstacleGammaV8,
    ObstacleNDV8,
    # v9 S³ flagship carriers (ADR-0171) — tensor-train + gridless particle
    TtState,
    TtEvolver,
    TtCoupledEvolver,
    MeasureState,
    GridlessEvolver,
    VarCoefTtEvolver,
    # bind-remaining-operators wave
    DiffusionExpmv1D,
    DriftReaction4th1D,
    Killing2nd1D,
    MatrixDiffusion2D,
    MatrixDiffusion3D,
    # feat/graph-krylov-frechet-a1a2 — A1 Krylov action + A2 Fréchet VJP (ADR-0185)
    GraphKrylov,
    graph_expmv_frechet,
    # issue #11 — conservative diffusion (FV divergence-form)
    ConservativeDiffusionChernoff,
    assemble_conservative_csr_1d,
    # issue #13 — externally-assembled symmetric PSD operator + entry-sensitivity VJP
    SymmetricOperator,
    symmetric_op_expmv_frechet,
    # issue #14 — mass-operator Krylov evolution
    MassKOperator,
    mass_lumped_evolve,
    # issue #12 — φ-function actions + ETDRK4 driver
    phi_action,
    phi_action_batched,
    Etdrk4,
)

__all__ = [
    "Heat1D",
    "Heat1D4th",
    "Heat1D6th",
    "Heat2D",
    "Heat3D",
    "DriftReaction1D",
    "Graph",
    "GraphPath",
    "GraphHeat",
    "GraphHeat4th",
    "Laplacian",
    "MagnusGraphHeat",
    "MagnusGraphHeat6",
    "SemiflowError",
    "Schrodinger1D",
    "Shift1D",
    "NonSeparable2D",
    "Adjoint",
    "AdaptivePI",
    "VarCoefGraphHeat",
    "GraphHeat6",
    "VarCoefMagnusGraph",
    "version",
    # v3.0 surface (ADR-0076 Wave E)
    "GrowthV3",
    "EvolverHeat1DUnitV3",
    # v4.1 Phase D
    "HypoellipticChernoffHeisenberg",
    "heisenberg_heat_kernel",
    "Heat1DZeta4",
    "Heat1DZeta6",
    # ADR-0111 Wave P1
    "Heat1DZeta8",
    "TruncatedExp1D",
    "TruncatedExp4th1D",
    "Strang1D",
    # ADR-0111 Wave P2
    "SchrodingerComplex1D",
    # ADR-0111 Wave P3
    "Resolvent1D",
    "Killing1D",
    "Reflected1D",
    "Robin1D",
    "DirichletHeat2nd1D",
    # ADR-0111 Wave P4
    "Howland1D",
    "Subordinated1D",
    # ADR-0111 Wave P5
    "Manifold2D",
    "HypoellipticChernoffKolmogorov",
    "HypoellipticChernoffEngel",
    # ADR-0111 Wave P6
    "QuantumGraph",
    "QuantumGraphHeat",
    "MatrixDiffusion1D",
    "PointEval",
    "sample_gridfn2d",
    "GraphTraj",
    "StrangGraph",
    # ADR-0111 Wave P7
    "AnisotropicShiftND2",
    "AnisotropicShiftND3",
    "NonSeparable2DAniso",
    "Heat2DVarA",
    "Heat3DVarA",
    # Issue #1 — adjoint-state parameter-sensitivity (ADR-0115)
    # Issue #10 — batched evolve (ADR-0184)
    "GraphAdjoint",
    "GraphAdjointPresampled",
    "edge_weight_grad",
    "edge_weight_grad_batched",
    # v6.3.0 — obstacle / variational-inequality Chernoff (math §44)
    "ObstacleChernoff",
    # v8.0.0 F1 — Dual-AD Greeks (ADR-0133 A3)
    "EvolverHeat1DGreeksV3",
    "KilledDirichlet1D",
    # v8.1.0 F2 — ResolventJumpV8 (TWS contour, TIER 1, ADR-0138)
    "ResolventJumpV8",
    # v8.1.0 C2 — AdjointFokkerPlanckV8 (Lemma A.1 push, TIER 1, ADR-0138)
    "AdjointFokkerPlanckV8",
    # v8.1.0 C1 — SmolyakD6V8 (sparse-grid D=6, unit a=I, TIER 2, ADR-0138)
    "SmolyakD6V8",
    # v8.1.0 F4 — ComplexTripleJumpV8 (filiform-N5, apply_real, TIER 2, ADR-0138)
    "ComplexTripleJumpV8",
    # v8.3.0 C-9 — WentzellV8 + GammaFamily (dynamic Wentzell/Robin BC, TIER 1, ADR-0153)
    "WentzellV8",
    "GammaFamily",
    # v8.3.0 F2-ND — ResolventJump2DV8 + ResolventJump3DV8 (2D/3D, TIER 1, ADR-0153)
    "ResolventJump2DV8",
    "ResolventJump3DV8",
    # v8.3.0 B-7 — ObstacleGammaV8 + ObstacleNDV8 (TIER 2, ADR-0153)
    "ObstacleGammaV8",
    "ObstacleNDV8",
    # v9 S³ flagship carriers (ADR-0171) — tensor-train + gridless particle
    "TtState",
    "TtEvolver",
    "TtCoupledEvolver",
    "MeasureState",
    "GridlessEvolver",
    # VarCoefTt (issue #2, ADR-0178): additive-separable variable-coefficient TT evolver
    "VarCoefTtEvolver",
    # bind-remaining-operators wave
    "DiffusionExpmv1D",
    "DriftReaction4th1D",
    "Killing2nd1D",
    "MatrixDiffusion2D",
    "MatrixDiffusion3D",
    # feat/graph-krylov-frechet-a1a2 — A1 Krylov action + A2 Fréchet VJP (ADR-0185)
    "GraphKrylov",
    "graph_expmv_frechet",
    # issue #11
    "ConservativeDiffusionChernoff",
    "assemble_conservative_csr_1d",
    # issue #13
    "SymmetricOperator",
    "symmetric_op_expmv_frechet",
    # issue #14
    "MassKOperator",
    "mass_lumped_evolve",
    # issue #12
    "phi_action",
    "phi_action_batched",
    "Etdrk4",
]
