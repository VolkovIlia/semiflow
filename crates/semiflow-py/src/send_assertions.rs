//! Compile-time `Send + Sync` assertions for types that cross `py.allow_threads`.
//!
//! `Heat1D::evolve` releases the GIL and passes `DiffusionChernoff<f64>`,
//! `Grid1D<f64>`, and `Vec<f64>` into a `py.detach` closure.  `PyO3`
//! enforces that the closure is `Send`; Rust's borrow checker would reject
//! the code if any captured value is not `Send`.  These `assert_impl_all!`
//! assertions make the requirement explicit and produce a clear compile error
//! if the invariant is ever violated by upstream changes (ADR-0031, ADR-0026).
//!
//! Phase 4 additions (v2.3): `NonSeparableMixedChernoff`, `AdjointChernoff`,
//! `AdaptivePI` (ADR-0058, ADR-0055, ADR-0044).
//!
//! **No runtime cost**: `static_assertions` is a zero-overhead macro library
//! that generates compile-time type-level proofs only.

#[cfg(test)]
mod tests {
    use semiflow_core::{
        AdaptivePI, AdjointChernoff, ChernoffSemigroup, Diffusion4thChernoff, Diffusion6thChernoff,
        DiffusionChernoff, DriftReactionChernoff, Graph, GraphHeatChernoff, GraphSignal, Grid1D,
        Grid2D, Grid3D, GridFn1D, GridFn3D, NonSeparableMixedChernoff, SchrodingerChernoff,
        SchrodingerState, ShiftChernoff1D, Strang3D,
    };
    use static_assertions::assert_impl_all;

    // `CoeffClosure` = `Box<dyn Fn(f64) -> f64 + Send + Sync + 'static>`.
    // Send + Sync are explicit bounds in the definition (handle.rs / ADR-0034).
    assert_impl_all!(crate::handle::CoeffClosure: Send, Sync);

    // `DiffusionChernoff<f64>` holds only `fn(f64) -> f64` pointers,
    // `Grid1D<f64>` scalars, and `f64` scalars — all `Send + Sync`.
    assert_impl_all!(DiffusionChernoff<f64>: Send, Sync);

    // `Grid1D<f64>` holds only `f64` scalars — trivially `Send + Sync`.
    assert_impl_all!(Grid1D<f64>: Send, Sync);

    // `GridFn1D<f64>` holds `Vec<f64>` + `Grid1D<f64>` — auto `Send + Sync`.
    assert_impl_all!(GridFn1D<f64>: Send, Sync);

    // `ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>` holds
    // `DiffusionChernoff<f64>` + `usize` + `PhantomData<GridFn1D<f64>>`.
    // All fields are `Send + Sync` (ADR-0026 generic-over-Float invariant).
    assert_impl_all!(
        ChernoffSemigroup<DiffusionChernoff<f64>, GridFn1D<f64>>: Send, Sync
    );

    // Higher-order diffusion kernels (v2.3 Phase 2): `Arc<dyn Fn+Send+Sync>` is Send+Sync.
    assert_impl_all!(Diffusion4thChernoff<f64>: Send, Sync);
    assert_impl_all!(Diffusion6thChernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<Diffusion4thChernoff<f64>, GridFn1D<f64>>: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<Diffusion6thChernoff<f64>, GridFn1D<f64>>: Send, Sync
    );

    // Drift-reaction kernel (v2.3 Phase 2).
    assert_impl_all!(DriftReactionChernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<DriftReactionChernoff<f64>, GridFn1D<f64>>: Send, Sync
    );

    // Shift1D kernel (v2.3 Phase 2).
    assert_impl_all!(ShiftChernoff1D<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<ShiftChernoff1D<f64>, GridFn1D<f64>>: Send, Sync
    );

    // 3D types: `Grid3D<f64>` and `GridFn3D<f64>` hold only `Send + Sync` fields.
    assert_impl_all!(Grid3D<f64>: Send, Sync);
    assert_impl_all!(GridFn3D<f64>: Send, Sync);

    // `Strang3D` wraps three `AxisLift3D<DiffusionChernoff<f64>, f64>` — all `Send + Sync`
    // because `DiffusionChernoff<f64>` and `Axis` are `Send + Sync`.
    assert_impl_all!(
        Strang3D<DiffusionChernoff<f64>, DiffusionChernoff<f64>, DiffusionChernoff<f64>>:
            Send, Sync
    );

    // Schrödinger types (v2.3 Phase 3 — ADR-0057 / ADR-0031).
    // `SchrodingerChernoff<f64>` holds `RefCell<[Vec<f64>; 12]>` — Send but !Sync.
    // The py.detach closure moves it by value, so Send is sufficient.
    assert_impl_all!(SchrodingerChernoff<f64>: Send);
    // `SchrodingerState<f64>` holds two `GridFn1D<f64>` — auto Send + Sync.
    assert_impl_all!(SchrodingerState<f64>: Send, Sync);
    // `Schrodinger1DInner` = SchrodingerChernoff<f64> + SchrodingerState<f64>.
    // Send: both fields are Send.  !Sync: RefCell in SchrodingerChernoff.
    assert_impl_all!(crate::schrodinger::Schrodinger1DInner: Send);
    // `Mutex<Schrodinger1DInner>` is Send + Sync (Mutex wraps !Sync in Sync).
    assert_impl_all!(std::sync::Mutex<crate::schrodinger::Schrodinger1DInner>: Send, Sync);

    // Graph types (ADR-0059 / ADR-0031 — must cross py.detach boundary).
    // `Graph<f64>` holds only `Vec<usize>`, `Vec<u32>`, `Vec<f64>` — auto `Send + Sync`.
    assert_impl_all!(Graph<f64>: Send, Sync);
    // `GraphHeatChernoff<f64>` holds `Arc<Laplacian<f64>>` — auto `Send + Sync`.
    assert_impl_all!(GraphHeatChernoff<f64>: Send, Sync);
    // `GraphSignal<f64>` holds `Vec<f64>` + `Arc<Graph<f64>>` — auto `Send + Sync`.
    assert_impl_all!(GraphSignal<f64>: Send, Sync);
    // `ChernoffSemigroup<GraphHeatChernoff<f64>, GraphSignal<f64>>` inherits.
    assert_impl_all!(
        ChernoffSemigroup<GraphHeatChernoff<f64>, GraphSignal<f64>>: Send, Sync
    );

    // Phase 5 graph types (v2.3 Phase 5).
    // `Laplacian<f64>` holds `Vec<usize>`, `Vec<u32>`, `Vec<f64>`, scalar — auto Send+Sync.
    assert_impl_all!(semiflow_core::Laplacian<f64>: Send, Sync);
    // `GraphHeat4thChernoff<f64>` holds `Arc<Laplacian<f64>>` — auto Send+Sync.
    assert_impl_all!(semiflow_core::GraphHeat4thChernoff<f64>: Send, Sync);
    // `ChernoffSemigroup<GraphHeat4thChernoff<f64>, GraphSignal<f64>>` inherits.
    assert_impl_all!(
        ChernoffSemigroup<semiflow_core::GraphHeat4thChernoff<f64>, GraphSignal<f64>>: Send, Sync
    );
    // `VarCoefGraphHeatChernoff<f64>` holds `Arc<Graph>`, `Arc<Laplacian>`, `Vec<f64>` — Send+Sync.
    assert_impl_all!(semiflow_core::VarCoefGraphHeatChernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<semiflow_core::VarCoefGraphHeatChernoff<f64>, GraphSignal<f64>>: Send, Sync
    );

    // ADR-0111 Wave P1 types.
    // All four kernel types are f64 fn-pointer / scalar structs — auto Send+Sync.
    assert_impl_all!(semiflow_core::Diffusion8thZeta8Chernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<semiflow_core::Diffusion8thZeta8Chernoff<f64>, GridFn1D<f64>>:
            Send, Sync
    );
    assert_impl_all!(semiflow_core::TruncatedExpDiffusionChernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<semiflow_core::TruncatedExpDiffusionChernoff<f64>, GridFn1D<f64>>:
            Send, Sync
    );
    assert_impl_all!(semiflow_core::TruncatedExp4thDiffusionChernoff<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<semiflow_core::TruncatedExp4thDiffusionChernoff<f64>, GridFn1D<f64>>:
            Send, Sync
    );
    assert_impl_all!(
        semiflow_core::StrangSplit<
            semiflow_core::DiffusionChernoff<f64>,
            semiflow_core::DriftReactionChernoff<f64>,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::StrangSplit<
                semiflow_core::DiffusionChernoff<f64>,
                semiflow_core::DriftReactionChernoff<f64>,
            >,
            GridFn1D<f64>,
        >: Send, Sync
    );

    // ADR-0111 Wave P2 — SchrodingerComplex1D (SchrödingerChernoffComplex).
    // `SchrödingerChernoffComplex<C>` holds Vec<C::Real>, Grid1D<C::Real>, PhantomData<C>.
    // All fields are Send+Sync when C: Send+Sync (Complex<f64> is Send+Sync).
    assert_impl_all!(semiflow_core::SchrödingerChernoffComplex<numpy::Complex64>: Send, Sync);
    // `GridFnComplex1D<C>` holds Vec<C> + Grid1D<C::Real> — auto Send+Sync.
    assert_impl_all!(semiflow_core::GridFnComplex1D<numpy::Complex64>: Send, Sync);

    // ADR-0111 Wave P3 — BC kernels (Resolvent1D, Killing1D, Reflected1D, Robin1D).
    // All four kernel types wrap fn-pointer DiffusionChernoff<f64> + scalar region — auto Send+Sync.
    assert_impl_all!(
        semiflow_core::resolvent::LaplaceChernoffResolvent<
            semiflow_core::DiffusionChernoff<f64>, f64
        >: Send, Sync
    );
    assert_impl_all!(
        semiflow_core::killing::KillingChernoff<
            semiflow_core::DiffusionChernoff<f64>,
            semiflow_core::killing::BoxRegion<f64, 1>,
            f64,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::killing::KillingChernoff<
                semiflow_core::DiffusionChernoff<f64>,
                semiflow_core::killing::BoxRegion<f64, 1>,
                f64,
            >,
            GridFn1D<f64>,
        >: Send, Sync
    );
    assert_impl_all!(
        semiflow_core::reflection::ReflectedHeatChernoff<
            semiflow_core::DiffusionChernoff<f64>,
            semiflow_core::reflection::HalfSpaceRegion<f64, 1>,
            f64,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::reflection::ReflectedHeatChernoff<
                semiflow_core::DiffusionChernoff<f64>,
                semiflow_core::reflection::HalfSpaceRegion<f64, 1>,
                f64,
            >,
            GridFn1D<f64>,
        >: Send, Sync
    );
    assert_impl_all!(
        semiflow_core::robin::RobinHeatChernoff<
            semiflow_core::DiffusionChernoff<f64>,
            semiflow_core::robin::HalfSpaceRobin<f64, 1>,
            f64,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::robin::RobinHeatChernoff<
                semiflow_core::DiffusionChernoff<f64>,
                semiflow_core::robin::HalfSpaceRobin<f64, 1>,
                f64,
            >,
            GridFn1D<f64>,
        >: Send, Sync
    );

    // ADR-0111 Wave P6 — structured types (QuantumGraph, MatrixDiffusion1D,
    //                    PointEval, GraphTraj, StrangGraph).
    //
    // QuantumGraphHeatChernoff<f64>: Clone + Send + Sync (all fields are
    //   QuantumGraph<f64> + Vec<KirchhoffVertex<f64>> + Vec<ShiftChernoff1D<f64>>
    //   + Option<(ShiftChernoff1D<f64>, usize)>; all Send+Sync per fn-ptr invariant).
    assert_impl_all!(semiflow_core::quantum_graph::QuantumGraphHeatChernoff<f64>: Send, Sync);
    assert_impl_all!(semiflow_core::quantum_graph::QuantumGraphSignal<f64>: Send, Sync);
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::quantum_graph::QuantumGraphHeatChernoff<f64>,
            semiflow_core::quantum_graph::QuantumGraphSignal<f64>,
        >: Send, Sync
    );
    // StrangSplitGraph<GraphHeatChernoff<f64>, GraphHeatChernoff<f64>, f64>:
    //   both inner kernels are Send+Sync.
    assert_impl_all!(
        semiflow_core::strang_graph::StrangSplitGraph<
            semiflow_core::graph_heat::GraphHeatChernoff<f64>,
            semiflow_core::graph_heat::GraphHeatChernoff<f64>,
            f64,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::strang_graph::StrangSplitGraph<
                semiflow_core::graph_heat::GraphHeatChernoff<f64>,
                semiflow_core::graph_heat::GraphHeatChernoff<f64>,
                f64,
            >,
            semiflow_core::graph_signal::GraphSignal<f64>,
        >: Send, Sync
    );

    // ADR-0111 Wave P7 — AnisotropicShiftChernoffND<f64, 2/3> + NonSeparableMixed +
    //                    Strang2D/3D with closure DiffusionChernoff.
    // AnisotropicShiftChernoffND<f64,D> holds Box<dyn Fn+Send+Sync> closures +
    // Vec<SquareMatrix> + GaussHermiteTensor + GridND — all Send+Sync.
    assert_impl_all!(semiflow_core::shift_nd::AnisotropicShiftChernoffND<f64, 2>: Send, Sync);
    assert_impl_all!(semiflow_core::shift_nd::AnisotropicShiftChernoffND<f64, 3>: Send, Sync);

    // ADR-0111 Wave P5 — geometry types (Manifold2D, Kolmogorov, Engel).
    // ManifoldEnum is Send+Sync (unsafe impl in geometry.rs; all three backends
    // contain only BoundedGeometryManifold<f64> + PhantomData<f64>).
    // KolmogorovHypoelliptic contains Box<dyn VectorField<f64, 2>> — Send+Sync
    // because VectorField has Send+Sync+'static bounds (hormander.rs:43).
    // HypoellipticChernoff<f64, 4, 2> same argument.
    assert_impl_all!(crate::geometry::PyManifold2D: Send);
    assert_impl_all!(crate::geometry::PyHypoellipticChernoffKolmogorov: Send);
    assert_impl_all!(crate::geometry::PyHypoellipticChernoffEngel: Send);

    // ADR-0111 Wave P4 types.
    // HowlandLift<DiffusionChernoff<f64>, f64> holds DiffusionChernoff<f64> + f64 scalars.
    // All fields are Send + Sync.
    assert_impl_all!(
        semiflow_core::howland::HowlandLift<
            semiflow_core::DiffusionChernoff<f64>,
            f64,
        >: Send, Sync
    );
    // HowlandState<GridFn1D<f64>, f64> holds Vec<GridFn1D<f64>> — auto Send + Sync.
    assert_impl_all!(
        semiflow_core::howland::HowlandState<GridFn1D<f64>, f64>: Send, Sync
    );
    // SubordinatedChernoff<DiffUnit, SubordinatorEnum, f64>:
    // DiffUnit is Send+Sync; SubordinatorEnum wraps three f64-scalar types — Send+Sync.
    assert_impl_all!(
        semiflow_core::subordinated::SubordinatedChernoff<
            semiflow_core::DiffusionChernoff<f64>,
            crate::subordinated_py::SubordinatorEnum,
            f64,
        >: Send, Sync
    );
    assert_impl_all!(
        ChernoffSemigroup<
            semiflow_core::subordinated::SubordinatedChernoff<
                semiflow_core::DiffusionChernoff<f64>,
                crate::subordinated_py::SubordinatorEnum,
                f64,
            >,
            GridFn1D<f64>,
        >: Send, Sync
    );

    // Phase 4 types (v2.3 ADR-0058, ADR-0055, ADR-0044).

    // `NonSeparableMixedChernoff` holds `AxisLift` wrappers of `DiffusionChernoff<f64>`
    // + `Grid2D<f64>` + `Box<dyn MixedDerivOp<f64>>` (Send+Sync per trait bound).
    assert_impl_all!(
        NonSeparableMixedChernoff<DiffusionChernoff<f64>, DiffusionChernoff<f64>>: Send, Sync
    );

    // `AdjointChernoff<C, f64>` holds `C + PhantomData<f64>` — inherits Send+Sync from C.
    assert_impl_all!(AdjointChernoff<DiffusionChernoff<f64>>: Send, Sync);
    assert_impl_all!(AdjointChernoff<DriftReactionChernoff<f64>>: Send, Sync);

    // `AdaptivePI<C, f64>` holds `C` + scratch fields (Vec, ScratchPool) — all Send+Sync.
    // `Grid2D<f64>` also Send+Sync (two `Grid1D<f64>` scalars).
    assert_impl_all!(Grid2D<f64>: Send, Sync);
    assert_impl_all!(AdaptivePI<DiffusionChernoff<f64>>: Send, Sync);
    assert_impl_all!(AdaptivePI<DriftReactionChernoff<f64>>: Send, Sync);
}
