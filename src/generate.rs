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

pub struct GenerationTimings {
    pub parse_ms: f64,
    pub sign_grid_ms: f64,
    pub composite_ms: f64,
}

pub fn generate_voxels(
    grid_config: &GridConfig,
    expr_config: &ExpressionConfig,
    expr_status: &mut ExpressionStatus,
    dim_mapping: &DimMapping,
) -> (Vec<PackedColor>, usize, GenerationTimings) {
    let size_usize = grid_config.size as usize;
    let half_extent = (grid_config.size as f64) / 2.0 * grid_config.voxel_size;

    let mut timings = GenerationTimings {
        parse_ms: 0.0,
        sign_grid_ms: 0.0,
        composite_ms: 0.0,
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
            Err(e) => {
                expr_status.is_valid = false;
                expr_status
                    .errors
                    .push(format!("Expression #{} '{}': {}", idx + 1, entry.expr, e));
            }
        }
    }
    timings.parse_ms = parse_start.elapsed().as_secs_f64() * 1000.0;

    let (composite, grid_timings, rendered_voxel_count) =
        match generate_voxel_grid_multi_with_composing(
            size_usize,
            &exprs,
            half_extent,
            &DimConfig::from(dim_mapping),
            parallel_available(),
        ) {
            Ok(result) => result,
            Err(e) => {
                expr_status.is_valid = false;
                expr_status.errors.push(e);
                return (Vec::new(), 0, timings);
            }
        };

    timings.sign_grid_ms = grid_timings.sign_grid_ms;
    timings.composite_ms = grid_timings.composite_ms;

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
