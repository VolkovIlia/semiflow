// Parallel `Strang3D` impl block — compiled only when `parallel` feature is active.
// Included into `strang3d.rs` via `include!`; private items from `strang3d`
// (e.g. `apply_strang3d_into`, `run_lift_into`) are accessible because this
// snippet is lexically part of that module.
// See `strang3d.rs`, ADR-0042, ADR-0045 Wave 5.

// Generic over F: SemiflowFloat + Send + Sync + ParallelPool3D (ADR-0045 Wave 5).
#[cfg(feature = "parallel")]
#[allow(private_bounds)]
impl<X, Y, Z, F> Strang3D<X, Y, Z, F>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    /// Parallel palindromic Strang — allocation-free ping-pong path.
    ///
    /// Used by `ChernoffSemigroup::evolve` so that multi-step integration
    /// benefits from the parallel X/Y/Z passes.
    ///
    /// For grids smaller than `2 * MIN_PENCILS_PER_THREAD` pencils, falls back
    /// to the serial scratch-pool path (zero allocation in steady state).
    fn apply_parallel_into(
        &self,
        tau: F,
        src: &GridFn3D<F>,
        dst: &mut GridFn3D<F>,
        scratch: &mut ScratchPool<F>,
    ) -> Result<(), SemiflowError> {
        let nx = src.grid.nx();
        let ny = src.grid.ny();
        let nz = src.grid.nz();
        let min_pencils = nx.min(ny * nz).min(nx * nz).min(nx * ny);
        if min_pencils < 2 * MIN_PENCILS_PER_THREAD {
            // Serial fallback — zero-alloc steady state via scratch pool.
            return apply_strang3d_into(tau, src, dst, &self.x, &self.y, &self.z, scratch);
        }
        let n_threads = resolve_threads_3d(nx * ny);
        let half_tau = half::<F>() * tau;
        let mut state = src.values.clone();
        let mut y_scratch: Vec<F> = Vec::with_capacity(nx * nz * ny);
        let mut z_scratch: Vec<F> = Vec::with_capacity(nx * ny * nz);
        run_parallel_passes_3d(
            &mut state,
            src.grid,
            nx,
            ny,
            nz,
            n_threads,
            &self.x.inner,
            &self.y.inner,
            &self.z.inner,
            tau,
            half_tau,
            &mut y_scratch,
            &mut z_scratch,
        )?;
        dst.values.resize(state.len(), F::zero());
        dst.values.copy_from_slice(&state);
        dst.grid = src.grid;
        Ok(())
    }
}

/// Execute the palindromic 5-pass X-Y-Z-Y-X sequence in parallel.
///
/// Extracted from `apply_parallel` and `apply_parallel_into` to keep each
/// caller under the 50-line budget. Float-operation order is verbatim
/// identical to the inlined version (ADR-0018 bit-identity).
#[cfg(feature = "parallel")]
#[allow(clippy::too_many_arguments, private_bounds)]
fn run_parallel_passes_3d<X, Y, Z, F>(
    state: &mut [F],
    grid: crate::grid3d::Grid3D<F>,
    nx: usize,
    ny: usize,
    nz: usize,
    n_threads: usize,
    x_inner: &X,
    y_inner: &Y,
    z_inner: &Z,
    tau: F,
    half_tau: F,
    y_scratch: &mut Vec<F>,
    z_scratch: &mut Vec<F>,
) -> Result<(), SemiflowError>
where
    F: SemiflowFloat + Send + Sync + ParallelPool3D,
    X: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Y: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
    Z: ChernoffFunction<F, S = GridFn1D<F>> + Clone + Send + Sync,
{
    parallel_x_pass_3d(state, grid.x, n_threads, x_inner, half_tau)?;
    parallel_y_pass_3d(state, nx, grid.y, nz, n_threads, y_inner, half_tau, y_scratch)?;
    parallel_z_pass_3d(state, nx, ny, grid.z, n_threads, z_inner, tau, z_scratch)?;
    parallel_y_pass_3d(state, nx, grid.y, nz, n_threads, y_inner, half_tau, y_scratch)?;
    parallel_x_pass_3d(state, grid.x, n_threads, x_inner, half_tau)?;
    Ok(())
}
