use crate::utils::{CondMulAdd as _, DimMapping, PackedColor};
use hypervox_expr::{Node, VarMap};
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Per-phase timing for grid generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GridTimingsMs {
    pub sign_grid: f64,
    pub composite: f64,
}

/// Configuration for N-dimensional to 3D spatial mapping.
/// Maps N dimensions (0..ndim) to 3D spatial axes (X, Y, Z).
/// Dimensions not mapped to any axis are held at fixed values.
/// Expression variables: x,y,z (spatial axes) and x0..x{N-1} (dimension coords).
#[derive(Debug)]
pub struct DimConfig {
    pub ndim: usize,
    /// Which dimension index varies along the X axis
    pub x_dim: usize,
    /// Which dimension index varies along the Y axis
    pub y_dim: usize,
    /// Which dimension index varies along the Z axis
    pub z_dim: usize,
    /// Fixed coordinate value for each dimension (used for non-spatial dims)
    pub fixed: Vec<f64>,
    /// Offset of the evaluation window in world units
    pub world_offset: (f64, f64, f64),
}

impl Default for DimConfig {
    fn default() -> Self {
        DimMapping::default().into()
    }
}

impl From<&DimMapping> for DimConfig {
    fn from(value: &DimMapping) -> Self {
        Self {
            ndim: value.ndim,
            x_dim: value.x_dim,
            y_dim: value.y_dim,
            z_dim: value.z_dim,
            fixed: value.fixed.clone(),
            world_offset: value.world_offset,
        }
    }
}

impl From<DimMapping> for DimConfig {
    fn from(value: DimMapping) -> Self {
        Self {
            ndim: value.ndim,
            x_dim: value.x_dim,
            y_dim: value.y_dim,
            z_dim: value.z_dim,
            fixed: value.fixed,
            world_offset: value.world_offset,
        }
    }
}

impl VarMap for DimConfig {
    fn ndim(&self) -> usize {
        self.ndim
    }
    fn resolve_alias(&self, name: &str) -> Option<usize> {
        match name {
            "x" => Some(self.x_dim),
            "y" => Some(self.y_dim),
            "z" => Some(self.z_dim),
            _ => None,
        }
    }
    fn primary_prefix(&self) -> &'static str {
        "x"
    }
}

/// Generates a voxel grid of size `size^3` from N-dimensional expressions.
/// - `exprs`: parsed expressions with their base colors (0xRRGGBB stored as `color + 1`)
/// - `world_half_extent`: half the size of the region in world units.
/// - `dim`: N-dimensional mapping config
/// - `use_parallel`: whether to use parallel computation (rayon)
pub fn generate_voxel_grid_multi_with_composing(
    size: usize,
    exprs: &[(Node, PackedColor)],
    world_half_extent: f64,
    dim: &DimConfig,
    use_parallel: bool,
) -> Result<(Vec<PackedColor>, GridTimingsMs, usize), String> {
    if size == 0 {
        return Ok((Vec::new(), GridTimingsMs::default(), 0));
    }

    if dim.x_dim >= dim.ndim || dim.y_dim >= dim.ndim || dim.z_dim >= dim.ndim {
        return Err(format!(
            "Axis mapping out of range: x_dim={}, y_dim={}, z_dim={} with ndim={}",
            dim.x_dim, dim.y_dim, dim.z_dim, dim.ndim
        ));
    }

    let node_dim = size.saturating_add(1);

    let step = (world_half_extent * 2.0_f64) / size as f64;

    let sign_start = Instant::now();
    let sign_grids: Vec<(Vec<bool>, PackedColor)> = if use_parallel {
        exprs
            .par_iter()
            .map(|(expr, base_color)| {
                (
                    compute_sign_grid_par(expr, node_dim, step, world_half_extent, dim),
                    *base_color,
                )
            })
            .collect()
    } else {
        exprs
            .iter()
            .map(|(expr, base_color)| {
                (
                    compute_sign_grid(expr, node_dim, step, world_half_extent, dim),
                    *base_color,
                )
            })
            .collect()
    };
    let sign_grid = sign_start.elapsed().as_secs_f64() * 1000.0_f64;

    let composite_start = Instant::now();

    let (voxel_grid, count) = if use_parallel {
        fill_voxels_composing_par(&sign_grids, node_dim)
    } else {
        fill_voxels_composing(&sign_grids, node_dim)
    };
    let composite = composite_start.elapsed().as_secs_f64() * 1000.0_f64;

    Ok((
        voxel_grid,
        GridTimingsMs {
            sign_grid,
            composite,
        },
        count,
    ))
}

#[inline]
fn eval_sign(val: f64) -> bool {
    !val.is_finite() || val >= 0.0
}

#[inline]
fn init_fixed_vars(dim_conf: &DimConfig) -> Vec<f64> {
    let mut vars = vec![0.0_f64; dim_conf.ndim];
    for (dim, var) in vars.iter_mut().enumerate() {
        if dim != dim_conf.x_dim && dim != dim_conf.y_dim && dim != dim_conf.z_dim {
            *var = if dim < dim_conf.fixed.len() {
                // SAFETY: the enclosing `if` guarantees `dim < fixed.len()`.
                unsafe { *dim_conf.fixed.get_unchecked(dim) }
            } else {
                0.0_f64
            };
        }
    }

    vars
}

fn compute_sign_grid(
    expr: &hypervox_expr::Node,
    node_dim: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Vec<bool> {
    let node_dim_sq = node_dim.saturating_mul(node_dim);

    let mut sign_grid: Vec<bool> = vec![false; node_dim.saturating_mul(node_dim_sq)];

    let mut vars = init_fixed_vars(dim);

    let mut vars_options: Vec<Option<f64>> = vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        if let Some(var) = vars_options.get_mut(idx) {
            *var = None;
        }
    }

    let mut expr_mut_clone = expr.clone();
    let multi = expr_mut_clone.prepare_multi(&vars_options, &[dim.x_dim, dim.y_dim, dim.z_dim]);

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let mut cache = vec![0.0_f64; multi.cse_slots];
    for nz in 0..node_dim {
        let fz = (nz as f64).cond_mul_add(step, z0);
        // SAFETY: `vars` has length `ndim`; `z_dim < ndim` is validated in
        // `generate_voxel_grid_multi_with_composing`.
        *unsafe { vars.get_unchecked_mut(dim.z_dim) } = fz;
        for group in &multi.groups {
            if group.level == 2 {
                (group.combined)(&vars, &mut cache);
            }
        }
        for ny in 0..node_dim {
            let fy = (ny as f64).cond_mul_add(step, y0);
            // SAFETY: `vars` has length `ndim`; `y_dim < ndim` is validated in
            // `generate_voxel_grid_multi_with_composing`.
            *unsafe { vars.get_unchecked_mut(dim.y_dim) } = fy;
            for group in &multi.groups {
                if group.level == 1 {
                    (group.combined)(&vars, &mut cache);
                }
            }
            for nx in 0..node_dim {
                let fx = (nx as f64).cond_mul_add(step, x0);
                // SAFETY: `vars` has length `ndim`; `x_dim < ndim` is validated in
                // `generate_voxel_grid_multi_with_composing`.
                *unsafe { vars.get_unchecked_mut(dim.x_dim) } = fx;
                let idx = nx
                    .wrapping_add(ny.wrapping_mul(node_dim))
                    .wrapping_add(nz.wrapping_mul(node_dim_sq));
                // SAFETY: `idx = nx + ny*node_dim + nz*node_dim_sq` with each of
                // nx, ny, nz < node_dim, so `idx < node_dim^3 == sign_grid.len()`.
                *unsafe { sign_grid.get_unchecked_mut(idx) } =
                    eval_sign((multi.main)(&vars, &mut cache));
            }
        }
    }

    sign_grid
}

#[expect(clippy::similar_names)]
fn compute_sign_grid_par(
    expr: &hypervox_expr::Node,
    node_dim: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Vec<bool> {
    let node_dim_sq = node_dim.saturating_mul(node_dim);

    let mut sign_grid: Vec<bool> = vec![false; node_dim.saturating_mul(node_dim_sq)];

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let base_vars = init_fixed_vars(dim);

    let mut base_vars_options: Vec<Option<f64>> = base_vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        if let Some(var) = base_vars_options.get_mut(idx) {
            *var = None;
        }
    }

    let mut expr_clone = expr.clone();
    let multi = expr_clone.prepare_multi(&base_vars_options, &[dim.x_dim, dim.y_dim, dim.z_dim]);

    let total = node_dim.saturating_mul(node_dim_sq);
    let num_threads = rayon::current_num_threads();
    let chunk_size = total.div_ceil(num_threads);

    sign_grid
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx.wrapping_mul(chunk_size);
            let start_nz = start.checked_div(node_dim_sq).unwrap_or_default();
            let start_ny = start
                .checked_rem(node_dim_sq)
                .unwrap_or_default()
                .checked_div(node_dim)
                .unwrap_or_default();
            let start_nx = start.checked_rem(node_dim).unwrap_or_default();

            let mut vars = base_vars.clone();
            let mut cache = vec![0.0_f64; multi.cse_slots];
            let mut nz = start_nz;
            let mut ny = start_ny;
            let mut nx = start_nx;

            let fz = (nz as f64).cond_mul_add(step, z0);
            // SAFETY: `vars` has length `ndim`; `z_dim < ndim` is validated in
            // `generate_voxel_grid_multi_with_composing`.
            *unsafe { vars.get_unchecked_mut(dim.z_dim) } = fz;
            for group in &multi.groups {
                if group.level == 2 {
                    (group.combined)(&vars, &mut cache);
                }
            }
            let fy = (ny as f64).cond_mul_add(step, y0);
            // SAFETY: `vars` has length `ndim`; `y_dim < ndim` is validated in
            // `generate_voxel_grid_multi_with_composing`.
            *unsafe { vars.get_unchecked_mut(dim.y_dim) } = fy;
            for group in &multi.groups {
                if group.level == 1 {
                    (group.combined)(&vars, &mut cache);
                }
            }

            for cell in chunk.iter_mut() {
                let fx = (nx as f64).cond_mul_add(step, x0);
                // SAFETY: `vars` has length `ndim`; `x_dim < ndim` is validated in
                // `generate_voxel_grid_multi_with_composing`.
                *unsafe { vars.get_unchecked_mut(dim.x_dim) } = fx;
                *cell = eval_sign((multi.main)(&vars, &mut cache));

                nx = nx.wrapping_add(1);
                if nx == node_dim {
                    nx = 0;
                    ny = ny.wrapping_add(1);
                    if ny == node_dim {
                        ny = 0;
                        nz = nz.wrapping_add(1);
                        let fz = (nz as f64).cond_mul_add(step, z0);
                        // SAFETY: `vars` has length `ndim`; `z_dim < ndim` is validated in
                        // `generate_voxel_grid_multi_with_composing`.
                        *unsafe { vars.get_unchecked_mut(dim.z_dim) } = fz;
                        for group in &multi.groups {
                            if group.level == 2 {
                                (group.combined)(&vars, &mut cache);
                            }
                        }
                    }
                    let fy = (ny as f64).cond_mul_add(step, y0);
                    // SAFETY: `vars` has length `ndim`; `y_dim < ndim` is validated in
                    // `generate_voxel_grid_multi_with_composing`.
                    *unsafe { vars.get_unchecked_mut(dim.y_dim) } = fy;
                    for group in &multi.groups {
                        if group.level == 1 {
                            (group.combined)(&vars, &mut cache);
                        }
                    }
                }
            }
        });

    sign_grid
}

#[expect(clippy::inline_always)]
#[inline(always)]
#[expect(clippy::too_many_arguments, clippy::fn_params_excessive_bools)]
fn should_fill_voxel(
    s000: bool,
    s100: bool,
    s010: bool,
    s110: bool,
    s001: bool,
    s101: bool,
    s011: bool,
    s111: bool,
) -> bool {
    let sum = u8::from(s000)
        .wrapping_add(u8::from(s100))
        .wrapping_add(u8::from(s010))
        .wrapping_add(u8::from(s110))
        .wrapping_add(u8::from(s001))
        .wrapping_add(u8::from(s101))
        .wrapping_add(u8::from(s011))
        .wrapping_add(u8::from(s111));
    sum != 8 && sum != 0
}

/// # Safety
///
/// `base`, `node_dim` and `node_dim_sq` must satisfy
/// `base + node_dim_sq + node_dim + 1 < grid.len()` for every grid in
/// `sign_grids`, and `cell` must be a valid (in-bounds, uniquely owned)
/// mutable reference into the voxel grid.
#[expect(clippy::inline_always)]
#[inline(always)]
unsafe fn fill_voxel(
    sign_grids: &[(Vec<bool>, PackedColor)],
    cell: &mut PackedColor,
    base: usize,
    node_dim: usize,
    node_dim_sq: usize,
    count: &mut usize,
) {
    for (sign_grid, packed_color) in sign_grids {
        // SAFETY: caller guarantees base + node_dim^2 + node_dim + 1 < grid.len().
        let (s000, s100, s010, s110, s001, s101, s011, s111) = unsafe {
            (
                *sign_grid.get_unchecked(base),
                *sign_grid.get_unchecked(base.wrapping_add(1)),
                *sign_grid.get_unchecked(base.wrapping_add(node_dim)),
                *sign_grid.get_unchecked(base.wrapping_add(node_dim).wrapping_add(1)),
                *sign_grid.get_unchecked(base.wrapping_add(node_dim_sq)),
                *sign_grid.get_unchecked(base.wrapping_add(node_dim_sq).wrapping_add(1)),
                *sign_grid.get_unchecked(base.wrapping_add(node_dim_sq).wrapping_add(node_dim)),
                *sign_grid.get_unchecked(
                    base.wrapping_add(node_dim_sq)
                        .wrapping_add(node_dim)
                        .wrapping_add(1),
                ),
            )
        };

        #[expect(clippy::arithmetic_side_effects)]
        if should_fill_voxel(s000, s100, s010, s110, s001, s101, s011, s111) {
            *cell = *packed_color;
            *count += 1;
            break;
        }
    }
}

fn fill_voxels_composing(
    sign_grids: &[(Vec<bool>, PackedColor)],
    node_dim: usize,
) -> (Vec<PackedColor>, usize) {
    let node_dim_sq = node_dim.saturating_mul(node_dim);
    let size = node_dim.saturating_sub(1);
    let size_sq = size.saturating_mul(size);

    debug_assert!(
        sign_grids
            .iter()
            .all(|(grid, _)| grid.len() == node_dim.saturating_mul(node_dim_sq)),
        "sign grid must be node_dim^3"
    );

    let mut voxel_grid = vec![None; size.saturating_mul(size_sq)];

    let mut count = 0;
    for vz in 0..size {
        let base_z = vz.wrapping_mul(node_dim_sq);
        let voxel_base_z = vz.wrapping_mul(size_sq);

        for vy in 0..size {
            let base_y = base_z.wrapping_add(vy.wrapping_mul(node_dim));
            let voxel_base_y = voxel_base_z.wrapping_add(vy.wrapping_mul(size));

            for vx in 0..size {
                let base = base_y.wrapping_add(vx);

                // SAFETY: the largest read offset is `node_dim^2 + node_dim + 1`.
                // Since the loops run vx, vy, vz < size, the largest index is:
                //   base + node_dim^2 + node_dim + 1
                //   = (size-1)*(1 + node_dim + node_dim^2) + node_dim^2 + node_dim + 1
                //   = size*(1 + node_dim + node_dim^2)
                //   = (node_dim-1)*(1 + node_dim + node_dim^2) [node_dim = size+1]
                //   = node_dim^3 - 1,
                // and the debug_asserts above guarantee every grid holds
                // exactly node_dim^3 entries, so all reads are in bounds.
                unsafe {
                    fill_voxel(
                        sign_grids,
                        // SAFETY: voxel_base_y + vx = vz*size^2 + vy*size + vx,
                        // with vx, vy, vz < size, so the largest index is
                        // size^3 - 1, within voxel_grid's size^3 entries.
                        voxel_grid.get_unchecked_mut(voxel_base_y.wrapping_add(vx)),
                        base,
                        node_dim,
                        node_dim_sq,
                        &mut count,
                    );
                }
            }
        }
    }

    (voxel_grid, count)
}

fn fill_voxels_composing_par(
    sign_grids: &[(Vec<bool>, PackedColor)],
    node_dim: usize,
) -> (Vec<PackedColor>, usize) {
    let node_dim_sq = node_dim.saturating_mul(node_dim);
    let size = node_dim.saturating_sub(1);
    let size_sq = size.saturating_mul(size);

    debug_assert!(
        sign_grids
            .iter()
            .all(|(grid, _)| grid.len() == node_dim.saturating_mul(node_dim_sq)),
        "sign grid must be node_dim^3"
    );

    let mut voxel_grid = vec![None; size.saturating_mul(size_sq)];

    let total_voxels = size.saturating_mul(size_sq);
    let num_threads = rayon::current_num_threads();
    let chunk_size = total_voxels.div_ceil(num_threads);

    let count = voxel_grid
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let start_linear = chunk_idx.wrapping_mul(chunk_size);
            let mut vz = start_linear.checked_div(size_sq).unwrap_or_default();
            let mut vy = start_linear
                .checked_rem(size_sq)
                .unwrap_or_default()
                .checked_div(size)
                .unwrap_or_default();
            let mut vx = start_linear.checked_rem(size).unwrap_or_default();
            let mut local = 0_usize;

            for cell in chunk.iter_mut() {
                let base = vx
                    .wrapping_add(vy.wrapping_mul(node_dim))
                    .wrapping_add(vz.wrapping_mul(node_dim_sq));

                // SAFETY: the largest read offset is `node_dim^2 + node_dim + 1`.
                // Since the loops run vx, vy, vz < size, the largest index is:
                //   base + node_dim^2 + node_dim + 1
                //   = (size-1)*(1 + node_dim + node_dim^2) + node_dim^2 + node_dim + 1
                //   = size*(1 + node_dim + node_dim^2)
                //   = (node_dim-1)*(1 + node_dim + node_dim^2) [node_dim = size+1]
                //   = node_dim^3 - 1,
                // and the debug_asserts above guarantee every grid holds
                // exactly node_dim^3 entries, so all reads are in bounds.
                unsafe { fill_voxel(sign_grids, cell, base, node_dim, node_dim_sq, &mut local) }

                vx = vx.wrapping_add(1);
                if vx == size {
                    vx = 0;
                    vy = vy.wrapping_add(1);
                    if vy == size {
                        vy = 0;
                        vz = vz.wrapping_add(1);
                    }
                }
            }

            local
        })
        .sum();

    (voxel_grid, count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::pack_color;

    fn fill_one(size: usize, expr: &str, color: (u8, u8, u8), half: f64, dim: &DimConfig) -> usize {
        let node = hypervox_expr::parse(expr, dim).expect("expression should be valid");
        let (_, _, count) = generate_voxel_grid_multi_with_composing(
            size,
            &[(node, pack_color(color))],
            half,
            dim,
            true,
        )
        .expect("generation should not fail");
        count
    }

    #[test]
    fn sphere_generation() {
        let dim = DimConfig::default();
        let filled = fill_one(32, "x^2 + y^2 + z^2 - 4", (0xFF, 0x00, 0x00), 5.0, &dim);
        assert!(filled > 0 && filled < 32_usize.pow(3));
    }

    #[test]
    fn sinusoidal_surface() {
        let dim = DimConfig::default();
        let filled = fill_one(16, "sin(x) + cos(y) + z", (0x00, 0xFF, 0x00), 8.0, &dim);
        assert!(filled > 0 && filled < 16_usize.pow(3));
    }

    #[test]
    fn four_d_nd_variables() {
        let dim = DimConfig {
            ndim: 4,
            x_dim: 1,
            y_dim: 2,
            z_dim: 3,
            fixed: vec![0.0_f64, 0.0_f64, 0.0_f64, 0.0_f64],
            ..DimConfig::default()
        };
        let filled = fill_one(
            16,
            "x1^2 + x2^2 + x3^2 - x0^2",
            (0x00, 0xFF, 0x00),
            8.0,
            &dim,
        );
        assert_eq!(filled, 0);
    }
}
