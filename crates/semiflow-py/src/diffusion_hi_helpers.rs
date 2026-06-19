// Helper functions for diffusion_hi.rs — included via include!() (batch H8).
// Items are defined in the same module scope as diffusion_hi.rs.

// ---------------------------------------------------------------------------
// Array-based builders (extracted to keep methods ≤50 lines)
// ---------------------------------------------------------------------------

/// Build `Heat1D4th` from pre-sampled coefficient arrays.
#[allow(clippy::too_many_arguments)]
fn build_heat4th_from_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: &Bound<'_, PyAny>,
    u0: &Bound<'_, PyAny>,
    a_prime: Option<&Bound<'_, PyAny>>,
    a_double_prime: Option<&Bound<'_, PyAny>>,
    a_norm_bound: Option<f64>,
    boundary: &str,
) -> PyResult<Heat1D4th> {
    let (policy, slice, a_fn, ap_fn, app_fn, norm) =
        extract_abc_arrays(xmin, xmax, n, a, u0, a_prime, a_double_prime, a_norm_bound, boundary)?;
    validate_u0_finite(&slice).map_err(|e| from_core(&e))?;
    let grid = semiflow_core::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let chernoff = Diffusion4thChernoff::with_closure(a_fn, ap_fn, app_fn, norm, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, 100).map_err(|e| from_core(&e))?;
    let current = GridFn1D::new(grid, slice).map_err(|e| from_core(&e))?;
    Ok(Heat1D4th {
        inner: Diff4StateInner { semigroup, current },
    })
}

/// Build `Heat1D6th` from pre-sampled coefficient arrays.
#[allow(clippy::too_many_arguments)]
fn build_heat6th_from_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: &Bound<'_, PyAny>,
    u0: &Bound<'_, PyAny>,
    a_prime: Option<&Bound<'_, PyAny>>,
    a_double_prime: Option<&Bound<'_, PyAny>>,
    a_norm_bound: Option<f64>,
    boundary: &str,
) -> PyResult<Heat1D6th> {
    let (policy, slice, a_fn, ap_fn, app_fn, norm) =
        extract_abc_arrays(xmin, xmax, n, a, u0, a_prime, a_double_prime, a_norm_bound, boundary)?;
    validate_u0_finite(&slice).map_err(|e| from_core(&e))?;
    let grid = semiflow_core::Grid1D::new(xmin, xmax, n)
        .map_err(|e| from_core(&e))?
        .with_boundary(policy);
    let chernoff = Diffusion6thChernoff::with_closure(a_fn, ap_fn, app_fn, norm, grid);
    let semigroup = ChernoffSemigroup::new(chernoff, 100).map_err(|e| from_core(&e))?;
    let current = GridFn1D::new(grid, slice).map_err(|e| from_core(&e))?;
    Ok(Heat1D6th {
        inner: Diff6StateInner { semigroup, current },
    })
}

// ---------------------------------------------------------------------------
// Shared array extraction (used by both 4th and 6th builders)
// ---------------------------------------------------------------------------

type CoeffTriple = (
    semiflow_core::BoundaryPolicy,
    Vec<f64>,
    crate::handle::CoeffClosure,
    crate::handle::CoeffClosure,
    crate::handle::CoeffClosure,
    f64,
);

/// Extract and validate all coefficient arrays from Python arguments.
///
/// Returns `(policy, u0_slice, a_fn, ap_fn, app_fn, norm_bound)`.
#[allow(clippy::too_many_arguments)]
fn extract_abc_arrays(
    xmin: f64,
    xmax: f64,
    n: usize,
    a: &Bound<'_, PyAny>,
    u0: &Bound<'_, PyAny>,
    a_prime: Option<&Bound<'_, PyAny>>,
    a_double_prime: Option<&Bound<'_, PyAny>>,
    a_norm_bound: Option<f64>,
    boundary: &str,
) -> PyResult<CoeffTriple> {
    let policy = parse_boundary(boundary)?;
    let slice = extract_f64_slice(u0)?;
    let a_vals: Vec<f64> = a
        .extract::<Vec<f64>>()
        .map_err(|_| new_pyerr("GridMismatch", "a must be numpy.ndarray[float64]"))?;
    let dx = if n > 1 { (xmax - xmin) / (n as f64 - 1.0) } else { 1.0 };
    let ap_vals = extract_or_fd1(a_prime, &a_vals, dx, "a_prime")?;
    let app_vals = extract_or_fd2(a_double_prime, &a_vals, dx, "a_double_prime")?;
    let norm = a_norm_bound
        .unwrap_or_else(|| 1.1 * a_vals.iter().copied().fold(f64::NEG_INFINITY, f64::max));
    let a_fn = crate::coeff::closure_from_array(a, xmin, xmax, n)?;
    let ap_fn = build_vec_closure(&ap_vals, xmin, xmax, n)?;
    let app_fn = build_vec_closure(&app_vals, xmin, xmax, n)?;
    Ok((policy, slice, a_fn, ap_fn, app_fn, norm))
}

/// Extract `a'` from Python or compute via 4th-order FD.
fn extract_or_fd1(
    arr: Option<&Bound<'_, PyAny>>,
    base_vals: &[f64],
    dx: f64,
    name: &str,
) -> PyResult<Vec<f64>> {
    match arr {
        Some(py_arr) => py_arr.extract::<Vec<f64>>().map_err(|_| {
            new_pyerr("GridMismatch", &format!("{name} must be numpy.ndarray[float64]"))
        }),
        None => Ok(crate::coeff::derivative_4th(base_vals, dx)),
    }
}

/// Extract `a''` from Python or compute via 4th-order FD.
fn extract_or_fd2(
    arr: Option<&Bound<'_, PyAny>>,
    base_vals: &[f64],
    dx: f64,
    name: &str,
) -> PyResult<Vec<f64>> {
    match arr {
        Some(py_arr) => py_arr.extract::<Vec<f64>>().map_err(|_| {
            new_pyerr("GridMismatch", &format!("{name} must be numpy.ndarray[float64]"))
        }),
        None => Ok(crate::coeff::second_derivative_4th(base_vals, dx)),
    }
}

/// Build a `CoeffClosure` from a `Vec<f64>` (already-extracted).
fn build_vec_closure(
    vals: &[f64],
    xmin: f64,
    xmax: f64,
    n: usize,
) -> PyResult<crate::handle::CoeffClosure> {
    use std::sync::Arc;
    if vals.len() != n {
        return Err(new_pyerr(
            "GridMismatch",
            &format!("derivative array length {} != n={}", vals.len(), n),
        ));
    }
    let dx = if n > 1 { (xmax - xmin) / (n as f64 - 1.0) } else { 1.0 };
    let shared = Arc::new(vals.to_vec());
    Ok(Box::new(move |x: f64| {
        crate::coeff::interp_catmull_rom_pub(&shared, xmin, dx, x)
    }))
}

// ---------------------------------------------------------------------------
// Phase 2 compute helpers (called inside py.detach)
// ---------------------------------------------------------------------------

/// Evolve `Diffusion4thChernoff` for `n_steps` steps of `t/n_steps`.
///
/// No Python types. All params are `Send + Sync`.
///
/// # Errors
/// Propagates `SemiflowError` from `ChernoffSemigroup`.
fn compute_evolve_4th(
    func: Diffusion4thChernoff<f64>,
    grid: semiflow_core::Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, input)?;
    Ok(sg.evolve(t, &f)?.values)
}

/// Evolve `Diffusion6thChernoff` for `n_steps` steps of `t/n_steps`.
///
/// # Errors
/// Propagates `SemiflowError` from `ChernoffSemigroup`.
fn compute_evolve_6th(
    func: Diffusion6thChernoff<f64>,
    grid: semiflow_core::Grid1D<f64>,
    input: Vec<f64>,
    t: f64,
    n_steps: usize,
) -> Result<Vec<f64>, semiflow_core::SemiflowError> {
    let sg = ChernoffSemigroup::new(func, n_steps)?;
    let f = GridFn1D::new(grid, input)?;
    Ok(sg.evolve(t, &f)?.values)
}
