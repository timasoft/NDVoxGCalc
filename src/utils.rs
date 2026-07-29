use bevy::prelude::*;

/// First value `v` where `v + step == v` due to f64 rounding.
/// Always a power of two: `pow2_ceil(step * 2^53)`.
#[inline]
pub fn first_bad_offset(step: f64) -> f64 {
    let v = step * (1u64 << f64::MANTISSA_DIGITS) as f64;
    let bits = v.to_bits();
    let mantissa_bits = f64::MANTISSA_DIGITS as u64 - 1;
    let mantissa_mask = (1u64 << mantissa_bits) - 1;
    if bits & mantissa_mask == 0 {
        v
    } else {
        f64::from_bits((bits & !mantissa_mask) + (1u64 << mantissa_bits))
    }
}

/// Largest `voxel_size` where `first_bad_offset(v)` stays finite.
/// Bound: `v * 2^53 < f64::MAX` → `v < f64::MAX / 2^53`.
/// One ULP below that threshold to avoid `first_bad_offset` returning ∞.
pub const MAX_VOXEL_SIZE: f64 = (f64::MAX / (1u64 << f64::MANTISSA_DIGITS) as f64).next_down();

pub const CAMERA_RADIUS: f32 = 1.5;
pub const CAMERA_HEIGHT: f32 = 0.8;

#[derive(Resource)]
pub struct GridConfig {
    pub size: u32,
    pub voxel_size: f64,
    pub voxel_count: usize,
}

#[derive(Resource)]
pub struct DimMapping {
    pub ndim: usize,
    pub x_dim: usize,
    pub y_dim: usize,
    pub z_dim: usize,
    pub fixed: Vec<f64>,
    pub world_offset: (f64, f64, f64),
}

impl Default for DimMapping {
    fn default() -> Self {
        Self {
            ndim: 3,
            x_dim: 0,
            y_dim: 1,
            z_dim: 2,
            fixed: vec![0.0; 3],
            world_offset: (0.0, 0.0, 0.0),
        }
    }
}

#[derive(Resource)]
pub struct SceneEntities {
    pub camera: Entity,
    pub voxel_mesh: Entity,
}

#[derive(Resource)]
pub struct CameraState {
    pub angle: f32,
    pub speed: f32,
    pub mode: CameraMode,
}

#[derive(PartialEq, Eq)]
pub enum CameraMode {
    AutoOrbit,
    Manual,
}

#[derive(Resource, Clone, PartialEq, Eq)]
pub struct ShowAxesPlanes {
    pub show_axes: bool,
    pub show_ground_grid: bool,
    pub show_planes: bool,
}

impl Default for ShowAxesPlanes {
    fn default() -> Self {
        Self {
            show_axes: true,
            show_ground_grid: false,
            show_planes: false,
        }
    }
}

#[derive(Resource)]
pub struct RegenerateEveryFrame {
    pub enabled: bool,
}

#[derive(Clone)]
pub struct ExpressionEntry {
    pub expr: String,
    pub color: (u8, u8, u8),
    pub enabled: bool,
}

#[derive(Resource)]
pub struct ExpressionConfig {
    pub entries: Vec<ExpressionEntry>,
}

impl Default for ExpressionConfig {
    fn default() -> Self {
        Self {
            entries: vec![ExpressionEntry {
                expr: "x^2 + z * y - 64.0".into(),
                color: rand::random(),
                enabled: true,
            }],
        }
    }
}

#[derive(Resource, Default)]
pub struct ExpressionStatus {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

#[derive(Resource, Default)]
pub struct ProfilingData {
    pub parse_ms: f64,
    pub sign_grid_ms: f64,
    pub voxel_fill_ms: f64,
    pub composite_ms: f64,
    pub mesh_build_ms: f64,
    pub total_ms: f64,
}

#[cfg(target_arch = "wasm32")]
use std::sync::OnceLock;

#[cfg(target_arch = "wasm32")]
static PARALLEL_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
pub fn set_parallel_available(val: bool) {
    let _ = PARALLEL_AVAILABLE.set(val);
}

pub fn parallel_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        *PARALLEL_AVAILABLE.get().unwrap_or(&false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}
