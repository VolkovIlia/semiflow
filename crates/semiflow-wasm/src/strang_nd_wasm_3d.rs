// ---------------------------------------------------------------------------
// Heat3D — unit coefficient 3D heat
// ---------------------------------------------------------------------------

/// 3-D heat equation (`a = 1`, palindromic Strang splitting).
///
/// Solves `∂_t u = ∂_xx u + ∂_yy u + ∂_zz u`.
/// Buffer layout: x-fastest row-major, `values[k*nx*ny + j*nx + i] ≈ u(x_i,y_j,z_k)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Heat3D {
    strang: Strang3Dc,
    grid: Grid3D<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[wasm_bindgen]
impl Heat3D {
    /// Construct `Heat3D` on a `Grid3D` (unit `a = 1`).
    ///
    /// ## Parameters
    /// - `xmin`/`xmax`/`nx` — X-axis.
    /// - `ymin`/`ymax`/`ny` — Y-axis.
    /// - `zmin`/`zmax`/`nz` — Z-axis. Each axis: ≥ 4 nodes, finite bounds.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
    ) -> Result<Heat3D, JsValue> {
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let gz = Grid1D::new(zmin, zmax, nz).map_err(|e| err_to_js(&e))?;
        let grid = Grid3D::new(gx, gy, gz).map_err(|e| err_to_js(&e))?;
        let strang = Strang3D::new(unit_diff(gx), unit_diff(gy), unit_diff(gz));
        Ok(Heat3D {
            strang,
            grid,
            nx,
            ny,
            nz,
        })
    }

    /// Evolve flat x-fastest `u0` (length `nx * ny * nz`) by `n_steps` of size `tau`.
    ///
    /// Returns a new `Float64Array` with the evolved state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(
        &self,
        u0: &Float64Array,
        tau: f64,
        n_steps: usize,
    ) -> Result<Float64Array, JsValue> {
        validate_tau_nsteps(tau, n_steps)?;
        let input = extract_flat(u0, self.nx * self.ny * self.nz)?;
        let result = evolve_3d(&self.strang, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Z-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Total number of grid nodes (`nx * ny * nz`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}

// ---------------------------------------------------------------------------
// Heat2DVarA — variable-coefficient 2D heat
// ---------------------------------------------------------------------------

/// 2-D heat with per-axis variable diffusion coefficient.
///
/// Solves `∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u` (palindromic Strang splitting).
/// Buffer layout: flat row-major, `values[j*nx + i] ≈ u(x_i, y_j)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Heat2DVarA {
    strang: Strang2Dc,
    grid: Grid2D<f64>,
    nx: usize,
    ny: usize,
}

#[wasm_bindgen]
impl Heat2DVarA {
    /// Construct `Heat2DVarA`.
    ///
    /// ## Parameters
    /// - `xmin`, `xmax`, `nx` — X-axis (finite, `xmin < xmax`, `nx >= 4`).
    /// - `ymin`, `ymax`, `ny` — Y-axis (finite, `ymin < ymax`, `ny >= 4`).
    /// - `a_x` — diffusion at X-grid nodes, `Float64Array` length `nx`, all > 0 and finite.
    /// - `a_y` — diffusion at Y-grid nodes, `Float64Array` length `ny`, all > 0 and finite.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        a_x: &Float64Array,
        a_y: &Float64Array,
    ) -> Result<Heat2DVarA, JsValue> {
        let coeff_x = extract_pos_coeff(a_x, nx, "a_x")?;
        let coeff_y = extract_pos_coeff(a_y, ny, "a_y")?;
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let grid = Grid2D::new(gx, gy);
        let dx = var_diff(coeff_x, xmin, xmax, nx, gx);
        let dy = var_diff(coeff_y, ymin, ymax, ny, gy);
        let strang = Strang2D::new(dx, dy);
        Ok(Heat2DVarA {
            strang,
            grid,
            nx,
            ny,
        })
    }

    /// Evolve flat row-major `u0` (length `nx * ny`) by `n_steps` of size `tau`.
    ///
    /// Returns a new `Float64Array` with the evolved state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(
        &self,
        u0: &Float64Array,
        tau: f64,
        n_steps: usize,
    ) -> Result<Float64Array, JsValue> {
        validate_tau_nsteps(tau, n_steps)?;
        let input = extract_flat(u0, self.nx * self.ny)?;
        let result = evolve_2d(&self.strang, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Total number of grid nodes (`nx * ny`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.ny
    }
}

// ---------------------------------------------------------------------------
// Heat3DVarA — variable-coefficient 3D heat
// ---------------------------------------------------------------------------

/// 3-D heat with per-axis variable diffusion coefficient.
///
/// Solves `∂_t u = a_x(x)·∂_xx u + a_y(y)·∂_yy u + a_z(z)·∂_zz u`.
/// Buffer layout: x-fastest row-major, `values[k*nx*ny + j*nx + i] ≈ u(x_i,y_j,z_k)`.
///
/// # Errors
/// Throws JS `Error` with `.kind` — see crate-level error table.
#[wasm_bindgen]
pub struct Heat3DVarA {
    strang: Strang3Dc,
    grid: Grid3D<f64>,
    nx: usize,
    ny: usize,
    nz: usize,
}

#[wasm_bindgen]
impl Heat3DVarA {
    /// Construct `Heat3DVarA`.
    ///
    /// ## Parameters
    /// - `xmin`/`xmax`/`nx`, `ymin`/`ymax`/`ny`, `zmin`/`zmax`/`nz` — axes.
    /// - `a_x`, `a_y`, `a_z` — per-axis `Float64Array` diffusion coefficients; must be > 0.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    #[wasm_bindgen(constructor)]
    pub fn new(
        xmin: f64,
        xmax: f64,
        nx: usize,
        ymin: f64,
        ymax: f64,
        ny: usize,
        zmin: f64,
        zmax: f64,
        nz: usize,
        a_x: &Float64Array,
        a_y: &Float64Array,
        a_z: &Float64Array,
    ) -> Result<Heat3DVarA, JsValue> {
        let coeff_x = extract_pos_coeff(a_x, nx, "a_x")?;
        let coeff_y = extract_pos_coeff(a_y, ny, "a_y")?;
        let coeff_z = extract_pos_coeff(a_z, nz, "a_z")?;
        let gx = Grid1D::new(xmin, xmax, nx).map_err(|e| err_to_js(&e))?;
        let gy = Grid1D::new(ymin, ymax, ny).map_err(|e| err_to_js(&e))?;
        let gz = Grid1D::new(zmin, zmax, nz).map_err(|e| err_to_js(&e))?;
        let grid = Grid3D::new(gx, gy, gz).map_err(|e| err_to_js(&e))?;
        let dx = var_diff(coeff_x, xmin, xmax, nx, gx);
        let dy = var_diff(coeff_y, ymin, ymax, ny, gy);
        let dz = var_diff(coeff_z, zmin, zmax, nz, gz);
        let strang = Strang3D::new(dx, dy, dz);
        Ok(Heat3DVarA {
            strang,
            grid,
            nx,
            ny,
            nz,
        })
    }

    /// Evolve flat x-fastest `u0` (length `nx * ny * nz`) by `n_steps` of size `tau`.
    ///
    /// Returns a new `Float64Array` with the evolved state.
    ///
    /// # Errors
    /// Throws JS `Error` with `.kind`.
    pub fn evolve(
        &self,
        u0: &Float64Array,
        tau: f64,
        n_steps: usize,
    ) -> Result<Float64Array, JsValue> {
        validate_tau_nsteps(tau, n_steps)?;
        let input = extract_flat(u0, self.nx * self.ny * self.nz)?;
        let result = evolve_3d(&self.strang, self.grid, input, tau, n_steps)?;
        Ok(vec_to_js(&result))
    }

    /// X-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nx(&self) -> usize {
        self.nx
    }

    /// Y-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn ny(&self) -> usize {
        self.ny
    }

    /// Z-axis node count.
    #[must_use]
    #[wasm_bindgen(getter)]
    pub fn nz(&self) -> usize {
        self.nz
    }

    /// Total number of grid nodes (`nx * ny * nz`).
    #[must_use]
    #[wasm_bindgen(js_name = "len")]
    pub fn len_method(&self) -> usize {
        self.nx * self.ny * self.nz
    }
}
