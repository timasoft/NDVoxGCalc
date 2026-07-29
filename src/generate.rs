#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use bevy::prelude::*;

use crate::math;
use crate::math::DimConfig;
use crate::utils::{
    DimMapping, ExpressionConfig, ExpressionStatus, GridConfig, parallel_available,
};

pub struct GenerationTimings {
    pub parse_ms: f64,
    pub sign_grid_ms: f64,
    pub voxel_fill_ms: f64,
    pub composite_ms: f64,
}

pub fn generate_voxels(
    grid_config: &GridConfig,
    expr_config: &ExpressionConfig,
    expr_status: &mut ExpressionStatus,
    dim_mapping: &DimMapping,
) -> (Vec<u32>, usize, GenerationTimings) {
    let size_usize = grid_config.size as usize;
    let half_extent = (grid_config.size as f64) / 2.0 * grid_config.voxel_size;

    let mut timings = GenerationTimings {
        parse_ms: 0.0,
        sign_grid_ms: 0.0,
        voxel_fill_ms: 0.0,
        composite_ms: 0.0,
    };

    expr_status.errors.clear();

    let mut grids = Vec::with_capacity(expr_config.entries.len());
    let mut last_voxel_count = 0;

    for (idx, entry) in expr_config.entries.iter().enumerate() {
        if !entry.enabled {
            continue;
        }

        let parse_start = Instant::now();
        if let Err(e) = hypervox_expr::validate(&entry.expr, &DimConfig::from(dim_mapping)) {
            timings.parse_ms += parse_start.elapsed().as_secs_f64() * 1000.0;
            expr_status.is_valid = false;
            expr_status
                .errors
                .push(format!("Expression #{} '{}': {}", idx + 1, entry.expr, e));
            continue;
        }
        timings.parse_ms += parse_start.elapsed().as_secs_f64() * 1000.0;

        let base_color =
            ((entry.color.0 as u32) << 16) | ((entry.color.1 as u32) << 8) | (entry.color.2 as u32);

        let gen_start = Instant::now();
        match math::generate_voxel_grid(
            size_usize,
            &entry.expr,
            base_color,
            half_extent,
            &DimConfig::from(dim_mapping),
            parallel_available(),
        ) {
            Ok((grid, grid_timings, voxel_count)) => {
                timings.sign_grid_ms += grid_timings.sign_grid_ms;
                timings.voxel_fill_ms += grid_timings.voxel_fill_ms;
                last_voxel_count = voxel_count;
                grids.push(grid);
            }
            Err(e) => {
                timings.sign_grid_ms += gen_start.elapsed().as_secs_f64() * 1000.0;
                error!(
                    "Error evaluating expression #{} '{}': {}",
                    idx + 1,
                    entry.expr,
                    e
                );
                expr_status.is_valid = false;
                expr_status.errors.push(format!(
                    "Eval error for #{} '{}': {}",
                    idx + 1,
                    entry.expr,
                    e
                ));
            }
        }
    }

    let compose_start = Instant::now();
    let total_positions = size_usize.pow(3);
    let (composite, rendered_voxel_count) = if grids.is_empty() {
        (vec![0u32; total_positions], 0)
    } else if grids.len() == 1
        && let Some(grid) = grids.pop()
    {
        (grid, last_voxel_count)
    } else {
        let mut composite = vec![0u32; total_positions];
        let mut rendered_voxel_count = 0;
        for idx in 0..total_positions {
            for grid in &grids {
                let val = grid[idx];
                if val != 0 {
                    composite[idx] = val;
                    rendered_voxel_count += 1;
                    break;
                }
            }
        }
        (composite, rendered_voxel_count)
    };
    timings.composite_ms = compose_start.elapsed().as_secs_f64() * 1000.0;

    // Only mark as valid if no errors occurred AND at least one enabled expression exists
    if expr_status.errors.is_empty()
        && expr_config
            .entries
            .iter()
            .any(|e| e.enabled && !e.expr.trim().is_empty())
    {
        expr_status.is_valid = true;
    }

    (composite, rendered_voxel_count, timings)
}
