use bevy::{
    asset::RenderAssetUsages,
    diagnostic::{DiagnosticsStore, FrameTimeDiagnosticsPlugin},
    prelude::*,
    render::render_resource::PrimitiveTopology,
};
use bevy_egui::{EguiContexts, egui};
use bevy_panorbit_camera::PanOrbitCamera;
use hypervox_expr::{f0_list, f1_list, f2_list};

#[cfg(not(target_arch = "wasm32"))]
use std::time::{Duration, Instant};
#[cfg(target_arch = "wasm32")]
use web_time::{Duration, Instant};

use crate::generate::generate_voxels;
use crate::utils::{
    CAMERA_HEIGHT, CAMERA_RADIUS, CameraMode, CameraState, CondMulAdd as _, DimMapping,
    ExpressionConfig, ExpressionEntry, ExpressionStatus, GridConfig, MAX_VOXEL_SIZE,
    ProfilingDataMs, RegenerateEveryFrame, SceneEntities, ShowAxesPlanes, first_bad_offset,
    parallel_available,
};

const REGEN_DEBOUNCE: Duration = Duration::from_millis(300);

#[derive(Default)]
pub enum RegenRequest {
    #[default]
    None,
    Debounce(Instant),
    Force,
}

#[expect(clippy::arithmetic_side_effects)]
#[inline]
const fn snap_size_up(size: u16) -> u16 {
    let step = if size < 4 {
        2
    } else if size < 8 {
        4
    } else {
        8
    };
    (size / step + 1) * step
}

#[expect(clippy::arithmetic_side_effects)]
#[inline]
const fn snap_size_down(size: u16) -> u16 {
    if size > 8 {
        (size - 1) / 8 * 8
    } else if size > 4 {
        4
    } else {
        2
    }
}

fn format_drag_value(val: f64) -> String {
    let abs_val = val.abs();
    if abs_val != 0.0 && (abs_val < 1e-6 || abs_val >= 1e6) {
        format!("{val:e}")
    } else {
        let str = format!("{val:.6}");
        str.trim_end_matches('0').trim_end_matches('.').to_owned()
    }
}

pub fn egui_overlays_system(
    mut egui_contexts: EguiContexts,
    mut show_axes_planes: ResMut<ShowAxesPlanes>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    egui::Window::new("Overlays")
        .anchor(egui::Align2::RIGHT_BOTTOM, [-12.0, -12.0])
        .show(ctx, |ui| {
            ui.checkbox(&mut show_axes_planes.show_axes, "Show Axes");
            ui.checkbox(&mut show_axes_planes.show_ground_grid, "Show Ground Grid");
            ui.checkbox(&mut show_axes_planes.show_planes, "Show Reference Planes");
            ui.separator();
            ui.colored_label(egui::Color32::from_rgb(255, 51, 51), "X — red");
            ui.colored_label(egui::Color32::from_rgb(51, 255, 51), "Y — green");
            ui.colored_label(egui::Color32::from_rgb(51, 51, 255), "Z — blue");
        });
}

#[cfg(target_arch = "wasm32")]
pub fn update_ui_scale_from_browser(mut egui_contexts: EguiContexts) {
    if let Some(window) = web_sys::window() {
        let scale = window.device_pixel_ratio() as f32;
        if let Ok(ctx) = egui_contexts.ctx_mut() {
            ctx.set_pixels_per_point(scale);
        }
    }
}

#[cfg(not(target_arch = "wasm32"))]
pub const fn update_ui_scale_from_browser() {}

#[expect(
    clippy::too_many_arguments,
    clippy::needless_pass_by_value,
    clippy::shadow_reuse
)]
pub fn egui_ui_system(
    mut commands: Commands,
    mut egui_contexts: EguiContexts,
    diagnostics: Res<DiagnosticsStore>,
    mut grid_config: ResMut<GridConfig>,
    mut dim_mapping: ResMut<DimMapping>,
    mut camera: ResMut<CameraState>,
    mut query_cam: Query<&mut PanOrbitCamera>,
    mut auto_regen: ResMut<RegenerateEveryFrame>,
    mut regenerate_request: Local<RegenRequest>,
    mut expr_config: ResMut<ExpressionConfig>,
    mut expr_status: ResMut<ExpressionStatus>,
    entities: Res<SceneEntities>,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut query_cam_transform: Query<&mut Transform>,
    mut profiling: ResMut<ProfilingDataMs>,
) {
    let Ok(ctx) = egui_contexts.ctx_mut() else {
        return;
    };

    let mut viewport_ui = egui::Ui::new(
        ctx.clone(),
        egui::Id::new("viewport_ui"),
        egui::UiBuilder::new()
            .layer_id(egui::LayerId::background())
            .max_rect(ctx.viewport_rect()),
    );
    egui::Panel::left("left_panel")
        .resizable(true)
        .show(&mut viewport_ui, |ui| {
            ui.heading("Generation");

            ui.label("Expressions:");

            // Scrollable area for expression list with fixed max height
            egui::ScrollArea::vertical()
                .id_salt("expr_list_scroll")
                .max_height(300.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    fill_expr_list_scroll(
                        ui,
                        &mut expr_config,
                        &mut regenerate_request,
                        &mut expr_status,
                    );
                });

            ui.separator();

            if ui.button("➕ Add Expression").clicked() {
                expr_config.entries.push(ExpressionEntry {
                    expr: String::new(),
                    color: rand::random(),
                    enabled: true,
                });
                *regenerate_request = RegenRequest::Debounce(Instant::now());
            }

            if !expr_status.errors.is_empty() {
                for error in &expr_status.errors {
                    ui.label(
                        egui::RichText::new(format!("❌ {error}"))
                            .color(egui::Color32::RED)
                            .small(),
                    );
                }
            } else if expr_status.is_valid
                && expr_config
                    .entries
                    .iter()
                    .any(|entry| entry.enabled && !entry.expr.trim().is_empty())
            {
                ui.label(
                    egui::RichText::new("✅ Valid expressions")
                        .color(egui::Color32::GREEN)
                        .small(),
                );
            }

            if dim_mapping.ndim > 3 {
                ui.label(
                    egui::RichText::new(format!(
                        "Tip: ^=power, vars: x,y,z (axes), x0..x{} (dims)",
                        dim_mapping.ndim.saturating_sub(1)
                    ))
                    .italics()
                    .small()
                    .color(egui::Color32::from_gray(180)),
                );
            } else {
                ui.label(
                    egui::RichText::new(
                        "Tip: ^ = power (x^2), * = multiply, + - / work as expected",
                    )
                    .italics()
                    .small()
                    .color(egui::Color32::from_gray(180)),
                );
            }

            egui::CollapsingHeader::new("Available functions")
                .default_open(false)
                .id_salt("functions_help")
                .show(ui, |ui| {
                    ui.strong("Constants:");
                    ui.label(f0_list());
                    ui.separator();
                    ui.strong("Functions (1 arg):");
                    ui.label(f1_list());
                    ui.separator();
                    ui.strong("Functions (2 arg):");
                    ui.label(f2_list());
                });

            ui.separator();

            ui.label("Grid Size:");
            ui.horizontal(|ui| {
                if ui.button("-").clicked() && grid_config.size > 2 {
                    grid_config.size = snap_size_down(grid_config.size);
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }

                if ui
                    .add(
                        egui::Slider::new(&mut grid_config.size, 2_u16..=256_u16)
                            .logarithmic(false)
                            .custom_formatter(|n, _| format!("{n:.0}")),
                    )
                    .changed()
                {
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }

                if ui.button("+").clicked() && grid_config.size < 256 {
                    grid_config.size = snap_size_up(grid_config.size);
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            ui.label("Voxel Size:");
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut grid_config.voxel_size)
                            .speed(0.0_f64)
                            .custom_formatter(|val, _| format_drag_value(val))
                            .range(f64::MIN_POSITIVE..=MAX_VOXEL_SIZE),
                    )
                    .changed()
                {
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            ui.separator();

            // --- Dimension Configuration ---
            ui.label("Dimensions (N):");
            ui.horizontal(|ui| {
                if ui
                    .add(
                        egui::DragValue::new(&mut dim_mapping.ndim)
                            .speed(0.0_f64)
                            .custom_formatter(|val, _| format_drag_value(val))
                            .range(3..=usize::MAX),
                    )
                    .changed()
                {
                    let n = dim_mapping.ndim;
                    dim_mapping.fixed.resize(n, 0.0_f64);
                    dim_mapping.x_dim = dim_mapping.x_dim.min(dim_mapping.ndim.saturating_sub(1));
                    dim_mapping.y_dim = dim_mapping.y_dim.min(dim_mapping.ndim.saturating_sub(1));
                    dim_mapping.z_dim = dim_mapping.z_dim.min(dim_mapping.ndim.saturating_sub(1));
                    if dim_mapping.y_dim == dim_mapping.x_dim {
                        dim_mapping.y_dim = (dim_mapping.y_dim.wrapping_add(1))
                            .checked_rem(dim_mapping.ndim)
                            .unwrap_or_default();
                    }
                    if dim_mapping.z_dim == dim_mapping.x_dim
                        || dim_mapping.z_dim == dim_mapping.y_dim
                    {
                        for dim in 0..dim_mapping.ndim {
                            if dim != dim_mapping.x_dim && dim != dim_mapping.y_dim {
                                dim_mapping.z_dim = dim;
                                break;
                            }
                        }
                    }
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }
            });

            append_axis_mapping(ui, &mut dim_mapping, &mut regenerate_request);

            // Fixed values for non-spatial dims
            let max_offset = grid_config.voxel_size.cond_mul_add(
                -f64::from(grid_config.size) / 2.0_f64,
                first_bad_offset(grid_config.voxel_size),
            );
            egui::ScrollArea::vertical()
                .id_salt("dims_fixed_scroll")
                .max_height(300.0)
                .auto_shrink([false, true])
                .show(ui, |ui| {
                    for dim in 0..dim_mapping.ndim {
                        if dim != dim_mapping.x_dim
                            && dim != dim_mapping.y_dim
                            && dim != dim_mapping.z_dim
                        {
                            ui.horizontal(|ui| {
                                ui.label(format!("x{dim}:"));
                                if ui
                                    .add(
                                        egui::DragValue::new(
                                            dim_mapping.fixed.get_mut(dim).unwrap_or(&mut 0.0_f64),
                                        )
                                        .speed(0.0_f64)
                                        .custom_formatter(|val, _| format_drag_value(val))
                                        .range(-max_offset..=max_offset),
                                    )
                                    .changed()
                                {
                                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                                }
                            });
                        }
                    }
                });

            ui.separator();

            if ui.button("Regenerate").clicked() {
                *regenerate_request = RegenRequest::Force;
            }

            ui.checkbox(&mut auto_regen.enabled, "Auto Regenerate");

            ui.separator();

            ui.collapsing("View Offset", |ui| {
                fill_view_offset(ui, &mut dim_mapping, max_offset, &mut regenerate_request);
            });

            let cam_label = match camera.mode {
                CameraMode::AutoOrbit => "Camera: Auto",
                CameraMode::Manual => "Camera: Manual",
            };
            if ui.button(cam_label).clicked() {
                camera.mode = match camera.mode {
                    CameraMode::AutoOrbit => {
                        for mut poc in &mut query_cam {
                            poc.enabled = true;
                            poc.focus = Vec3::ZERO;
                            poc.target_focus = Vec3::ZERO;
                        }
                        CameraMode::Manual
                    }
                    CameraMode::Manual => {
                        for mut poc in &mut query_cam {
                            poc.enabled = false;
                        }
                        CameraMode::AutoOrbit
                    }
                };
            }
        });

    egui::Window::new("Statistics")
        .anchor(egui::Align2::RIGHT_TOP, [-12.0, 12.0])
        .show(ctx, |ui| {
            let fps = diagnostics
                .get(&FrameTimeDiagnosticsPlugin::FPS)
                .and_then(bevy::diagnostic::Diagnostic::smoothed)
                .map_or(0.0_f64, f64::round);
            ui.label(format!("FPS: {fps:.0}"));
            if dim_mapping.ndim > 3 {
                ui.label(format!(
                    "Grid: {}³ (N={}, dims 0..{})",
                    grid_config.size,
                    dim_mapping.ndim,
                    dim_mapping.ndim.saturating_sub(1)
                ));
            } else {
                ui.label(format!("Grid: {}³", grid_config.size));
            }
            ui.label(format!("Rendered Voxels: {}", grid_config.voxel_count));
            ui.label(format!(
                "Fill: {:.1}%",
                (grid_config.voxel_count as f64 / f64::from(grid_config.size).powi(3)) * 100.0_f64
            ));

            ui.separator();
            ui.label("Regeneration Timing:");
            let total = profiling.total;
            if total > 0.0_f64 {
                ui.label(format!(
                    "  Parse:     {:.1} ms  ({:.0}%)",
                    profiling.parse,
                    profiling.parse / total * 100.0_f64
                ));
                ui.label(format!(
                    "  Sign grid: {:.1} ms  ({:.0}%)",
                    profiling.sign_grid,
                    profiling.sign_grid / total * 100.0_f64
                ));
                ui.label(format!(
                    "  Composite:  {:.1} ms  ({:.0}%)",
                    profiling.composite,
                    profiling.composite / total * 100.0_f64
                ));
                ui.label(format!(
                    "  Mesh build: {:.1} ms  ({:.0}%)",
                    profiling.mesh_build,
                    profiling.mesh_build / total * 100.0_f64
                ));
                ui.label("  -----------------");
                ui.label(format!("  Total:     {total:.1} ms"));
            } else {
                ui.label("  (waiting for generation...)");
            }
        });

    let should_regenerate = auto_regen.enabled
        || match *regenerate_request {
            RegenRequest::None => false,
            RegenRequest::Debounce(time) => time.elapsed() >= REGEN_DEBOUNCE,
            RegenRequest::Force => true,
        };
    if should_regenerate {
        *regenerate_request = RegenRequest::None;
        let total_start = Instant::now();
        let (composite, count, gen_timings) =
            generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
        grid_config.voxel_count = count;

        let mut mesh = Mesh::new(
            PrimitiveTopology::TriangleList,
            RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
        );

        let mesh_start = Instant::now();
        if parallel_available() {
            crate::mesh::build_batched_mesh_with_global_corner_ambient_occlusion_par(
                &mut mesh,
                &composite,
                &grid_config,
                count,
            );
        } else {
            crate::mesh::build_batched_mesh_with_global_corner_ambient_occlusion(
                &mut mesh,
                &composite,
                &grid_config,
                count,
            );
        }
        let mesh_build = mesh_start.elapsed().as_secs_f64() * 1000.0_f64;
        let total = total_start.elapsed().as_secs_f64() * 1000.0_f64;

        profiling.parse = gen_timings.parse;
        profiling.sign_grid = gen_timings.sign_grid;
        profiling.composite = gen_timings.composite;
        profiling.mesh_build = mesh_build;
        profiling.total = total;

        commands
            .entity(entities.voxel_mesh)
            .insert(Mesh3d(meshes.add(mesh)))
            .insert(MeshMaterial3d(materials.add(StandardMaterial {
                base_color: Color::WHITE,
                unlit: true,
                perceptual_roughness: 0.0,
                metallic: 0.0,
                reflectance: 0.0,
                ..default()
            })));

        match camera.mode {
            CameraMode::AutoOrbit => {
                let angle = camera.angle;
                commands.entity(entities.camera).insert(
                    Transform::from_xyz(
                        CAMERA_RADIUS * angle.cos(),
                        CAMERA_HEIGHT,
                        CAMERA_RADIUS * angle.sin(),
                    )
                    .looking_at(Vec3::ZERO, Vec3::Y),
                );
            }
            CameraMode::Manual => {
                commands.entity(entities.camera).insert(PanOrbitCamera {
                    focus: Vec3::ZERO,
                    target_focus: Vec3::ZERO,
                    ..default()
                });
                if let Ok(mut transform) = query_cam_transform.get_mut(entities.camera) {
                    transform.look_at(Vec3::ZERO, Vec3::Y);
                }
            }
        }
    }
}

#[expect(clippy::shadow_reuse)]
fn fill_expr_list_scroll(
    ui: &mut egui::Ui,
    expr_config: &mut ExpressionConfig,
    regenerate_request: &mut RegenRequest,
    expr_status: &mut ExpressionStatus,
) {
    let mut remove_idx = None;
    let mut duplicate_idx = None;
    let mut move_up_idx = None;
    let mut move_down_idx = None;
    let entries_len = expr_config.entries.len();
    for (idx, entry) in expr_config.entries.iter_mut().enumerate() {
        egui::CollapsingHeader::new(format!("Function #{}", idx.wrapping_add(1)))
            .id_salt(("func_header", idx))
            .default_open(true)
            .show(ui, |ui| {
                // Trigger regeneration when enabled state changes
                if ui.checkbox(&mut entry.enabled, "Enabled").changed() {
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                }

                ui.label("Expression:");
                if ui.text_edit_singleline(&mut entry.expr).changed() {
                    *regenerate_request = RegenRequest::Debounce(Instant::now());
                    expr_status.errors.clear();
                }

                ui.label("Color:");
                ui.horizontal(|ui| {
                    let mut color_edit =
                        egui::Color32::from_rgb(entry.color.0, entry.color.1, entry.color.2);
                    if ui.color_edit_button_srgba(&mut color_edit).changed() {
                        entry.color = (color_edit.r(), color_edit.g(), color_edit.b());
                        *regenerate_request = RegenRequest::Debounce(Instant::now());
                    }
                });

                ui.horizontal(|ui| {
                    if idx > 0 && ui.small_button("⬆").clicked() {
                        move_up_idx = Some(idx);
                    }
                    if idx < entries_len.saturating_sub(1) && ui.small_button("⬇").clicked() {
                        move_down_idx = Some(idx);
                    }
                    if ui.small_button("📋 Duplicate").clicked() {
                        duplicate_idx = Some(idx);
                    }
                    if ui.small_button("❌ Remove").clicked() {
                        remove_idx = Some(idx);
                    }
                });
            });
        ui.separator();
    }

    // Apply removals after iteration to avoid borrow issues
    if let Some(idx) = remove_idx {
        expr_config.entries.remove(idx);
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }

    if let Some(idx) = duplicate_idx {
        let entry = expr_config.entries.get(idx).cloned().unwrap_or_default();
        expr_config.entries.insert(idx.wrapping_add(1), entry);
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }

    if let Some(idx) = move_up_idx {
        expr_config.entries.swap(idx, idx.saturating_sub(1));
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }

    if let Some(idx) = move_down_idx {
        expr_config.entries.swap(idx, idx.saturating_add(1));
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }
}

#[expect(clippy::shadow_reuse)]
fn append_axis_mapping(
    ui: &mut egui::Ui,
    dim_mapping: &mut DimMapping,
    regenerate_request: &mut RegenRequest,
) {
    ui.label("Axis Mapping:");
    let ndim = dim_mapping.ndim;
    let mut x_dim = dim_mapping.x_dim;
    let mut y_dim = dim_mapping.y_dim;
    let mut z_dim = dim_mapping.z_dim;

    ui.horizontal(|ui| {
        ui.label("X ⬅");
        egui::ComboBox::from_id_salt("x_dim_map")
            .selected_text(format!("x{x_dim}"))
            .show_ui(ui, |ui| {
                for dim in 0..ndim {
                    let is_taken = dim == y_dim || dim == z_dim;
                    if !is_taken || dim == x_dim {
                        ui.selectable_value(&mut x_dim, dim, format!("x{dim}"));
                    }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Y ⬅");
        egui::ComboBox::from_id_salt("y_dim_map")
            .selected_text(format!("x{y_dim}"))
            .show_ui(ui, |ui| {
                for dim in 0..ndim {
                    let is_taken = dim == x_dim || dim == z_dim;
                    if !is_taken || dim == y_dim {
                        ui.selectable_value(&mut y_dim, dim, format!("x{dim}"));
                    }
                }
            });
    });
    ui.horizontal(|ui| {
        ui.label("Z ⬅");
        egui::ComboBox::from_id_salt("z_dim_map")
            .selected_text(format!("x{z_dim}"))
            .show_ui(ui, |ui| {
                for dim in 0..ndim {
                    let is_taken = dim == x_dim || dim == y_dim;
                    if !is_taken || dim == z_dim {
                        ui.selectable_value(&mut z_dim, dim, format!("x{dim}"));
                    }
                }
            });
    });

    if x_dim != dim_mapping.x_dim || y_dim != dim_mapping.y_dim || z_dim != dim_mapping.z_dim {
        // Ensure distinct mapping
        if x_dim == y_dim {
            y_dim = y_dim.wrapping_add(1).checked_rem(ndim).unwrap_or_default();
        }
        if x_dim == z_dim || y_dim == z_dim {
            for dim in 0..ndim {
                if dim != x_dim && dim != y_dim {
                    z_dim = dim;
                    break;
                }
            }
        }
        dim_mapping.x_dim = x_dim;
        dim_mapping.y_dim = y_dim;
        dim_mapping.z_dim = z_dim;
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }
}

#[expect(clippy::shadow_reuse)]
fn fill_view_offset(
    ui: &mut egui::Ui,
    dim_mapping: &mut DimMapping,
    max_abs_offset: f64,
    regenerate_request: &mut RegenRequest,
) {
    let mut changed = false;
    let mut ox = dim_mapping.world_offset.0;
    let mut oy = dim_mapping.world_offset.1;
    let mut oz = dim_mapping.world_offset.2;

    ui.horizontal(|ui| {
        ui.label("X:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut ox)
                    .speed(0.0_f64)
                    .custom_formatter(|val, _| format_drag_value(val))
                    .range(-max_abs_offset..=max_abs_offset),
            )
            .changed();
        if ui.small_button("↺").clicked() {
            ox = 0.0_f64;
            changed = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Y:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut oy)
                    .speed(0.0_f64)
                    .custom_formatter(|val, _| format_drag_value(val))
                    .range(-max_abs_offset..=max_abs_offset),
            )
            .changed();
        if ui.small_button("↺").clicked() {
            oy = 0.0_f64;
            changed = true;
        }
    });
    ui.horizontal(|ui| {
        ui.label("Z:");
        changed |= ui
            .add(
                egui::DragValue::new(&mut oz)
                    .speed(0.0_f64)
                    .custom_formatter(|val, _| format_drag_value(val))
                    .range(-max_abs_offset..=max_abs_offset),
            )
            .changed();
        if ui.small_button("↺").clicked() {
            oz = 0.0_f64;
            changed = true;
        }
    });
    if changed {
        dim_mapping.world_offset = (ox, oy, oz);
        *regenerate_request = RegenRequest::Debounce(Instant::now());
    }
}
