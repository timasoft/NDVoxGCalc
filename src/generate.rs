#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use bevy::prelude::*;

use crate::math::{DimConfig, generate_voxel_grid_multi_with_composing};
use crate::utils::{
    DimMapping, ExpressionConfig, ExpressionStatus, GridConfig, PackedColor, pack_color,
    parallel_available,
};

pub struct GenerationTimingsMs {
    pub parse: f64,
    pub sign_grid: f64,
    pub composite: f64,
}

pub fn generate_voxels(
    grid_config: &GridConfig,
    expr_config: &ExpressionConfig,
    expr_status: &mut ExpressionStatus,
    dim_mapping: &DimMapping,
) -> (Vec<PackedColor>, usize, GenerationTimingsMs) {
    let size_usize = grid_config.size as usize;
    let half_extent = f64::from(grid_config.size) / 2.0_f64 * grid_config.voxel_size;

    let mut timings = GenerationTimingsMs {
        parse: 0.0,
        sign_grid: 0.0,
        composite: 0.0,
    };

    expr_status.errors.clear();

    let parse_start = Instant::now();
    let mut exprs = Vec::with_capacity(expr_config.entries.len());
    for (idx, entry) in expr_config.entries.iter().enumerate() {
        if !entry.enabled {
            continue;
        }
        match hypervox_expr::parse(&entry.expr, &DimConfig::from(dim_mapping)) {
            Ok(expr) => exprs.push((expr, pack_color(entry.color))),
            Err(err) => {
                expr_status.is_valid = false;
                expr_status.errors.push(format!(
                    "Expression #{} '{}': {}",
                    idx.wrapping_add(1),
                    entry.expr,
                    err
                ));
            }
        }
    }
    timings.parse = parse_start.elapsed().as_secs_f64() * 1000.0_f64;

    let (composite, grid_timings, rendered_voxel_count) =
        match generate_voxel_grid_multi_with_composing(
            size_usize,
            &exprs,
            half_extent,
            &DimConfig::from(dim_mapping),
            parallel_available(),
        ) {
            Ok(result) => result,
            Err(err) => {
                expr_status.is_valid = false;
                expr_status.errors.push(err);
                return (Vec::new(), 0, timings);
            }
        };

    timings.sign_grid = grid_timings.sign_grid;
    timings.composite = grid_timings.composite;

    // Only mark as valid if no errors occurred AND at least one enabled expression exists
    if expr_status.errors.is_empty()
        && expr_config
            .entries
            .iter()
            .any(|entry| entry.enabled && !entry.expr.trim().is_empty())
    {
        expr_status.is_valid = true;
    }

    (composite, rendered_voxel_count, timings)
}
