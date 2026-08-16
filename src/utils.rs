use std::num::NonZeroU32;

use bevy::prelude::*;

/// First value `v` where `v + step == v` due to f64 rounding.
/// Always a power of two: `pow2_ceil(step * 2^53)`.
#[inline]
pub fn first_bad_offset(step: f64) -> f64 {
    let val = step * (1_u64 << f64::MANTISSA_DIGITS) as f64;
    let bits = val.to_bits();
    let mantissa_bits = f64::MANTISSA_DIGITS - 1;
    #[expect(clippy::arithmetic_side_effects)]
    let mantissa_mask = (1_u64 << mantissa_bits) - 1;
    if bits & mantissa_mask == 0 {
        val
    } else {
        f64::from_bits((bits & !mantissa_mask).wrapping_add(1_u64 << mantissa_bits))
    }
}

/// Largest `voxel_size` where `first_bad_offset(v)` stays finite.
/// Bound: `v * 2^53 < f64::MAX` → `v < f64::MAX / 2^53`.
/// One ULP below that threshold to avoid `first_bad_offset` returning ∞.
pub const MAX_VOXEL_SIZE: f64 = (f64::MAX / (1_u64 << f64::MANTISSA_DIGITS) as f64).next_down();

pub const CAMERA_RADIUS: f32 = 1.5;
pub const CAMERA_HEIGHT: f32 = 0.8;

#[derive(Resource)]
pub struct GridConfig {
    pub size: u16,
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

impl Default for ExpressionEntry {
    fn default() -> Self {
        Self {
            expr: "x^2 + z * y - 64.0".into(),
            color: rand::random(),
            enabled: true,
        }
    }
}

#[derive(Resource)]
pub struct ExpressionConfig {
    pub entries: Vec<ExpressionEntry>,
}

impl Default for ExpressionConfig {
    fn default() -> Self {
        Self {
            entries: vec![ExpressionEntry::default()],
        }
    }
}

#[derive(Resource, Default)]
pub struct ExpressionStatus {
    pub is_valid: bool,
    pub errors: Vec<String>,
}

#[derive(Resource, Default)]
pub struct ProfilingDataMs {
    pub parse: f64,
    pub sign_grid: f64,
    pub composite: f64,
    pub mesh_build: f64,
    pub total: f64,
}

#[cfg(target_arch = "wasm32")]
use std::sync::OnceLock;

#[cfg(target_arch = "wasm32")]
static PARALLEL_AVAILABLE: OnceLock<bool> = OnceLock::new();

#[cfg(target_arch = "wasm32")]
pub fn set_parallel_available(val: bool) {
    let _ = PARALLEL_AVAILABLE.set(val);
}

pub const fn parallel_available() -> bool {
    #[cfg(target_arch = "wasm32")]
    {
        *PARALLEL_AVAILABLE.get().unwrap_or(&false)
    }
    #[cfg(not(target_arch = "wasm32"))]
    {
        true
    }
}

/// A 24-bit RGB color plus a "filled" marker, packed into `Option<NonZeroU32>`
pub type PackedColor = Option<NonZeroU32>;

#[expect(clippy::min_ident_chars)]
#[inline]
pub const fn pack_color((r, g, b): (u8, u8, u8)) -> PackedColor {
    NonZeroU32::new((((r as u32) << 16) | ((g as u32) << 8) | b as u32).wrapping_add(1))
}

#[expect(clippy::inline_always, clippy::min_ident_chars)]
#[inline(always)]
pub fn unpack_color(val: NonZeroU32) -> LinearRgba {
    let raw = val.get() - 1;
    let r = f32::from(((raw >> 16_usize) & 0xFF) as u8) / 255.0;
    let g = f32::from(((raw >> 8_usize) & 0xFF) as u8) / 255.0;
    let b = f32::from((raw & 0xFF) as u8) / 255.0;
    LinearRgba::rgb(r, g, b)
}

pub trait CondMulAdd {
    #[expect(clippy::min_ident_chars)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self;
}

#[cfg(feature = "fma")]
impl CondMulAdd for f64 {
    #[inline]
    #[expect(clippy::disallowed_methods)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(not(feature = "fma"))]
impl CondMulAdd for f64 {
    #[inline]
    #[expect(clippy::suboptimal_flops)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}

#[cfg(feature = "fma")]
impl CondMulAdd for f32 {
    #[inline]
    #[expect(clippy::disallowed_methods)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self.mul_add(b, c)
    }
}

#[cfg(not(feature = "fma"))]
impl CondMulAdd for f32 {
    #[inline]
    #[expect(clippy::suboptimal_flops)]
    fn cond_mul_add(self, b: Self, c: Self) -> Self {
        self * b + c
    }
}
