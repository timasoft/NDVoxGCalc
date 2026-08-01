use crate::utils::{DimMapping, PackedColor};
use hypervox_expr::{Node, VarMap};
use rayon::prelude::*;

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

/// Per-phase timing for grid generation.
#[derive(Debug, Clone, Copy, Default)]
pub struct GridTimings {
    pub sign_grid_ms: f64,
    pub composite_ms: f64,
}

/// Maximum supported number of dimensions (stack-allocated vars buffer size).
const MAX_NDIM: usize = 128;

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
    fn primary_prefix(&self) -> &str {
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
) -> Result<(Vec<PackedColor>, GridTimings, usize), String> {
    if size == 0 {
        return Ok((Vec::new(), GridTimings::default(), 0));
    }

    if dim.x_dim >= dim.ndim || dim.y_dim >= dim.ndim || dim.z_dim >= dim.ndim {
        return Err(format!(
            "Axis mapping out of range: x_dim={}, y_dim={}, z_dim={} with ndim={}",
            dim.x_dim, dim.y_dim, dim.z_dim, dim.ndim
        ));
    }

    let node_dim = size + 1;

    let step = (world_half_extent * 2.0) / size as f64;

    let sign_start = Instant::now();
    let sign_grids: Vec<(Vec<bool>, PackedColor)> = exprs
        .iter()
        .map(|(expr, base_color)| {
            let sign_grid = if use_parallel {
                compute_sign_grid_par(expr, node_dim, step, world_half_extent, dim)
            } else {
                compute_sign_grid(expr, node_dim, step, world_half_extent, dim)
            };
            (sign_grid, *base_color)
        })
        .collect();
    let sign_grid_ms = sign_start.elapsed().as_secs_f64() * 1000.0;

    let composite_start = Instant::now();

    let (voxel_grid, count) = if use_parallel {
        fill_voxels_composing_par(&sign_grids, size, node_dim)
    } else {
        fill_voxels_composing(&sign_grids, size, node_dim)
    };
    let composite_ms = composite_start.elapsed().as_secs_f64() * 1000.0;

    Ok((
        voxel_grid,
        GridTimings {
            sign_grid_ms,
            composite_ms,
        },
        count,
    ))
}

#[inline]
fn eval_sign(val: f64) -> bool {
    !val.is_finite() || val >= 0.0
}

#[inline]
fn init_fixed_vars(dim: &DimConfig) -> [f64; MAX_NDIM] {
    let mut vars = [0.0; MAX_NDIM];
    for (d, v) in vars.iter_mut().enumerate() {
        if d != dim.x_dim && d != dim.y_dim && d != dim.z_dim {
            *v = if d < dim.fixed.len() {
                dim.fixed[d]
            } else {
                0.0
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
    let node_dim_sq = node_dim * node_dim;

    let mut sign_grid: Vec<bool> = vec![false; node_dim * node_dim_sq];

    let mut vars = init_fixed_vars(dim);

    let mut vars_options: Vec<Option<f64>> = vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        vars_options[idx] = None;
    }

    let mut expr = expr.clone();
    let multi = expr.prepare_multi(&vars_options, &[dim.x_dim, dim.y_dim, dim.z_dim]);

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let mut cache = vec![0.0; multi.cse_slots];
    for nz in 0..node_dim {
        let fz = z0 + nz as f64 * step;
        vars[dim.z_dim] = fz;
        for group in &multi.groups {
            if group.level == 2 {
                (group.combined)(&vars, &mut cache);
            }
        }
        for ny in 0..node_dim {
            let fy = y0 + ny as f64 * step;
            vars[dim.y_dim] = fy;
            for group in &multi.groups {
                if group.level == 1 {
                    (group.combined)(&vars, &mut cache);
                }
            }
            for nx in 0..node_dim {
                let fx = x0 + nx as f64 * step;
                vars[dim.x_dim] = fx;
                let idx = nx + ny * node_dim + nz * node_dim_sq;
                sign_grid[idx] = eval_sign((multi.main)(&vars, &mut cache));
            }
        }
    }

    sign_grid
}

fn compute_sign_grid_par(
    expr: &hypervox_expr::Node,
    node_dim: usize,
    step: f64,
    world_half_extent: f64,
    dim: &DimConfig,
) -> Vec<bool> {
    let node_dim_sq = node_dim * node_dim;

    let mut sign_grid: Vec<bool> = vec![false; node_dim * node_dim_sq];

    let x0 = -world_half_extent + dim.world_offset.0;
    let y0 = -world_half_extent + dim.world_offset.1;
    let z0 = -world_half_extent + dim.world_offset.2;

    let base_vars = init_fixed_vars(dim);

    let mut base_vars_options: Vec<Option<f64>> = base_vars.iter().copied().map(Some).collect();
    for idx in [dim.x_dim, dim.y_dim, dim.z_dim] {
        base_vars_options[idx] = None;
    }

    let mut expr = expr.clone();
    let multi = expr.prepare_multi(&base_vars_options, &[dim.x_dim, dim.y_dim, dim.z_dim]);

    let total = node_dim * node_dim_sq;
    let num_threads = rayon::current_num_threads();
    let chunk_size = total.div_ceil(num_threads);

    sign_grid
        .par_chunks_mut(chunk_size)
        .enumerate()
        .for_each(|(chunk_idx, chunk)| {
            let start = chunk_idx * chunk_size;
            let start_nz = start / node_dim_sq;
            let start_ny = (start % node_dim_sq) / node_dim;
            let start_nx = start % node_dim;

            let mut vars = base_vars;
            let mut cache = vec![0.0; multi.cse_slots];
            let mut nz = start_nz;
            let mut ny = start_ny;
            let mut nx = start_nx;

            vars[dim.z_dim] = z0 + nz as f64 * step;
            for group in &multi.groups {
                if group.level == 2 {
                    (group.combined)(&vars, &mut cache);
                }
            }
            vars[dim.y_dim] = y0 + ny as f64 * step;
            for group in &multi.groups {
                if group.level == 1 {
                    (group.combined)(&vars, &mut cache);
                }
            }

            for cell in chunk.iter_mut() {
                let fx = x0 + nx as f64 * step;
                vars[dim.x_dim] = fx;
                *cell = eval_sign((multi.main)(&vars, &mut cache));

                nx += 1;
                if nx == node_dim {
                    nx = 0;
                    ny += 1;
                    if ny == node_dim {
                        ny = 0;
                        nz += 1;
                        vars[dim.z_dim] = z0 + nz as f64 * step;
                        for group in &multi.groups {
                            if group.level == 2 {
                                (group.combined)(&vars, &mut cache);
                            }
                        }
                    }
                    vars[dim.y_dim] = y0 + ny as f64 * step;
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

#[inline]
#[expect(clippy::too_many_arguments)]
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
    let sum = s000 as u8
        + s100 as u8
        + s010 as u8
        + s110 as u8
        + s001 as u8
        + s101 as u8
        + s011 as u8
        + s111 as u8;
    sum != 8 && sum != 0
}

fn fill_voxels_composing(
    sign_grids: &[(Vec<bool>, PackedColor)],
    size: usize,
    node_dim: usize,
) -> (Vec<PackedColor>, usize) {
    let node_dim_sq = node_dim * node_dim;
    let size_sq = size * size;

    let mut voxel_grid = vec![None; size * size_sq];

    let mut count = 0;
    for vz in 0..size {
        let base_z = vz * node_dim_sq;
        let voxel_base_z = vz * size_sq;

        for vy in 0..size {
            let base_y = base_z + vy * node_dim;
            let voxel_base_y = voxel_base_z + vy * size;

            for vx in 0..size {
                let base = base_y + vx;

                for (sign_grid, packed_color) in sign_grids {
                    let s000 = sign_grid[base];
                    let s100 = sign_grid[base + 1];
                    let s010 = sign_grid[base + node_dim];
                    let s110 = sign_grid[base + node_dim + 1];
                    let s001 = sign_grid[base + node_dim_sq];
                    let s101 = sign_grid[base + node_dim_sq + 1];
                    let s011 = sign_grid[base + node_dim_sq + node_dim];
                    let s111 = sign_grid[base + node_dim_sq + node_dim + 1];

                    if should_fill_voxel(s000, s100, s010, s110, s001, s101, s011, s111) {
                        voxel_grid[voxel_base_y + vx] = *packed_color;
                        count += 1;
                        break;
                    }
                }
            }
        }
    }

    (voxel_grid, count)
}

fn fill_voxels_composing_par(
    sign_grids: &[(Vec<bool>, PackedColor)],
    size: usize,
    node_dim: usize,
) -> (Vec<PackedColor>, usize) {
    let node_dim_sq = node_dim * node_dim;
    let size_sq = size * size;

    let mut voxel_grid = vec![None; size * size_sq];

    let total_voxels = size * size_sq;
    let num_threads = rayon::current_num_threads();
    let chunk_size = total_voxels.div_ceil(num_threads);

    let count = voxel_grid
        .par_chunks_mut(chunk_size)
        .enumerate()
        .map(|(chunk_idx, chunk)| {
            let start_linear = chunk_idx * chunk_size;
            let mut vz = start_linear / size_sq;
            let mut vy = (start_linear % size_sq) / size;
            let mut vx = start_linear % size;
            let mut local = 0usize;

            for cell in chunk.iter_mut() {
                let base = vx + vy * node_dim + vz * node_dim_sq;

                for (sign_grid, packed_color) in sign_grids {
                    let s000 = sign_grid[base];
                    let s100 = sign_grid[base + 1];
                    let s010 = sign_grid[base + node_dim];
                    let s110 = sign_grid[base + node_dim + 1];
                    let s001 = sign_grid[base + node_dim_sq];
                    let s101 = sign_grid[base + node_dim_sq + 1];
                    let s011 = sign_grid[base + node_dim_sq + node_dim];
                    let s111 = sign_grid[base + node_dim_sq + node_dim + 1];

                    if should_fill_voxel(s000, s100, s010, s110, s001, s101, s011, s111) {
                        *cell = *packed_color;
                        local += 1;
                        break;
                    }
                }

                vx += 1;
                if vx == size {
                    vx = 0;
                    vy += 1;
                    if vy == size {
                        vy = 0;
                        vz += 1;
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
        let node = hypervox_expr::parse(expr, dim).unwrap();
        let (_, _, count) = generate_voxel_grid_multi_with_composing(
            size,
            &[(node, pack_color(color))],
            half,
            dim,
            true,
        )
        .unwrap();
        count
    }

    #[test]
    fn test_sphere_generation() {
        let dim = DimConfig::default();
        let filled = fill_one(32, "x^2 + y^2 + z^2 - 4", (0xFF, 0x00, 0x00), 5.0, &dim);
        assert!(filled > 0 && filled < 32usize.pow(3));
    }

    #[test]
    fn test_sinusoidal_surface() {
        let dim = DimConfig::default();
        let filled = fill_one(16, "sin(x) + cos(y) + z", (0x00, 0xFF, 0x00), 8.0, &dim);
        assert!(filled > 0 && filled < 16usize.pow(3));
    }

    #[test]
    fn test_4d_nd_variables() {
        let dim = DimConfig {
            ndim: 4,
            x_dim: 1,
            y_dim: 2,
            z_dim: 3,
            fixed: vec![0.0, 0.0, 0.0, 0.0],
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
