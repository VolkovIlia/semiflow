fn unit_dc(grid: Grid1D<f64>) -> DiffusionChernoff<f64> {
    extern "Rust" fn one(_: f64) -> f64 {
        1.0
    }
    extern "Rust" fn zer(_: f64) -> f64 {
        0.0
    }
    DiffusionChernoff::new(one, zer, zer, 1.0, grid)
}

fn build_grid_2d(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
) -> Result<(Grid1D<f64>, Grid1D<f64>, Grid2D<f64>), semiflow::SemiflowError> {
    let gx = Grid1D::new(xmin, xmax, nx)?.with_boundary(BoundaryPolicy::Reflect);
    let gy = Grid1D::new(ymin, ymax, ny)?.with_boundary(BoundaryPolicy::Reflect);
    Ok((gx, gy, Grid2D::new(gx, gy)))
}

fn build_nonsep2d_const(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    c: f64,
    u0: &[f64],
) -> Result<InnerNonSep2D, semiflow::SemiflowError> {
    let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny)?;
    let c_norm = c.abs();
    let c_val = c;
    let arc_c: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static> = Arc::new(move |_x, _y| c_val);
    let kernel =
        nonseparable_mixed_closure::with_closure_c(unit_dc(gx), unit_dc(gy), arc_c, c_norm, grid)?;
    let size = nx * ny;
    Ok(InnerNonSep2D {
        kernel,
        grid,
        current: u0.to_vec(),
        size,
    })
}

fn build_nonsep2d_aniso(
    xmin: f64,
    xmax: f64,
    nx: usize,
    ymin: f64,
    ymax: f64,
    ny: usize,
    beta: &[f64],
    norm_bound: f64,
    u0: &[f64],
) -> Result<InnerNonSep2D, semiflow::SemiflowError> {
    let (gx, gy, grid) = build_grid_2d(xmin, xmax, nx, ymin, ymax, ny)?;
    let arc = Arc::new(beta.to_vec());
    let (nx2, ny2) = (nx, ny);
    let (xmn, xmx, ymn, ymx) = (xmin, xmax, ymin, ymax);
    let arc2 = Arc::clone(&arc);
    let beta_cls: Arc<dyn Fn(f64, f64) -> f64 + Send + Sync + 'static> = Arc::new(move |x, y| {
        let fi = ((x - xmn) / (xmx - xmn)) * (nx2 as f64 - 1.0);
        let fj = ((y - ymn) / (ymx - ymn)) * (ny2 as f64 - 1.0);
        let fi = fi.clamp(0.0, (nx2 - 1) as f64);
        let fj = fj.clamp(0.0, (ny2 - 1) as f64);
        let i0 = (fi as usize).min(nx2 - 2);
        let j0 = (fj as usize).min(ny2 - 2);
        let ti = fi - i0 as f64;
        let tj = fj - j0 as f64;
        let idx = |i: usize, j: usize| j * nx2 + i;
        arc2[idx(i0, j0)] * (1.0 - ti) * (1.0 - tj)
            + arc2[idx(i0 + 1, j0)] * ti * (1.0 - tj)
            + arc2[idx(i0, j0 + 1)] * (1.0 - ti) * tj
            + arc2[idx(i0 + 1, j0 + 1)] * ti * tj
    });
    let kernel = nonseparable_mixed_closure::with_closure_beta(
        unit_dc(gx),
        unit_dc(gy),
        beta_cls,
        norm_bound,
        grid,
    )?;
    let size = nx * ny;
    Ok(InnerNonSep2D {
        kernel,
        grid,
        current: u0.to_vec(),
        size,
    })
}

// ---------------------------------------------------------------------------
// Compute helper
// ---------------------------------------------------------------------------

fn evolve_nonsep(
    kernel: &Nsm,
    grid: Grid2D<f64>,
    input: Vec<f64>,
    tau: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow::SemiflowError> {
    let mut src = GridFn2D::new(grid, input)?;
    let mut dst = GridFn2D::new(grid, vec![0.0; src.values.len()])?;
    let mut scratch = ScratchPool::<f64>::new();
    for _ in 0..n_steps {
        kernel.apply_into(tau, &src, &mut dst, &mut scratch)?;
        core::mem::swap(&mut src, &mut dst);
    }
    Ok(src.values)
}

// ---------------------------------------------------------------------------
// Shared validators
// ---------------------------------------------------------------------------

fn validate_tau_steps(tau: f64, n_steps: usize) -> Result<(), SemiflowStatus> {
    if n_steps == 0 || !tau.is_finite() || tau <= 0.0 {
        return Err(SemiflowStatus::OutOfDomain);
    }
    Ok(())
}

fn validate_finite_slice(vals: &[f64]) -> Result<(), SemiflowStatus> {
    for &v in vals {
        if !v.is_finite() {
            return Err(SemiflowStatus::NanInf);
        }
    }
    Ok(())
}

/// Resolve `beta_norm_bound`: negative hint → auto-compute as `1.1 * max|β|`.
fn compute_beta_norm_bound(beta_slice: &[f64], hint: f64) -> f64 {
    if hint < 0.0 {
        let m = beta_slice
            .iter()
            .copied()
            .fold(0.0_f64, |a, v| a.max(v.abs()));
        if m == 0.0 { 0.0 } else { m * 1.1 }
    } else {
        hint
    }
}
