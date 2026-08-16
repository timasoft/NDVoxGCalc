use bevy::{
    asset::RenderAssetUsages,
    camera::Exposure,
    core_pipeline::tonemapping::Tonemapping,
    diagnostic::FrameTimeDiagnosticsPlugin,
    prelude::*,
    render::render_resource::PrimitiveTopology,
    window::{PresentMode, WindowMode, WindowResolution},
};
use bevy_egui::{EguiPlugin, EguiPrimaryContextPass};
use bevy_panorbit_camera::{PanOrbitCamera, PanOrbitCameraPlugin};

#[cfg(not(target_arch = "wasm32"))]
use std::time::Instant;
#[cfg(target_arch = "wasm32")]
use web_time::Instant;

use crate::generate::generate_voxels;
use crate::utils::{
    CAMERA_HEIGHT, CAMERA_RADIUS, CameraMode, CameraState, CondMulAdd as _, DimMapping,
    ExpressionConfig, ExpressionStatus, GridConfig, ProfilingDataMs, RegenerateEveryFrame,
    SceneEntities, ShowAxesPlanes, parallel_available,
};

#[cfg(target_arch = "wasm32")]
use wasm_bindgen::prelude::*;
#[cfg(target_arch = "wasm32")]
use wasm_bindgen_futures::JsFuture;

mod generate;
mod math;
mod mesh;
mod ui;
mod utils;

#[cfg(target_arch = "wasm32")]
#[wasm_bindgen(start)]
async fn wasm_start() {
    let num_threads = web_sys::window()
        .map(|w| w.navigator().hardware_concurrency())
        .map(|c| c as usize)
        .unwrap_or(1)
        .max(1);
    let result = JsFuture::from(wasm_bindgen_rayon::init_thread_pool(num_threads)).await;
    if result.is_ok() {
        crate::utils::set_parallel_available(true);
    } else {
        web_sys::console::log_1(&"init_thread_pool failed, falling back to sequential".into());
    }
    main();
}

fn main() {
    #[cfg(not(target_arch = "wasm32"))]
    let _: Result<(), rayon::ThreadPoolBuildError> = rayon::ThreadPoolBuilder::new().build_global();
    App::new()
        .add_plugins(DefaultPlugins.set(WindowPlugin {
            primary_window: Some(Window {
                title: "HyperVox".into(),
                mode: WindowMode::BorderlessFullscreen(MonitorSelection::Current),
                resolution: WindowResolution::default(),
                present_mode: PresentMode::Fifo,
                ..default()
            }),
            ..default()
        }))
        .insert_resource(ClearColor(Color::WHITE))
        .insert_resource(GridConfig {
            size: 64,
            voxel_size: 1.0,
            voxel_count: 0,
        })
        .insert_resource(DimMapping::default())
        .insert_resource(ExpressionConfig::default())
        .insert_resource(ExpressionStatus::default())
        .insert_resource(RegenerateEveryFrame { enabled: false })
        .insert_resource(ShowAxesPlanes::default())
        .add_plugins(FrameTimeDiagnosticsPlugin::default())
        .add_plugins(PanOrbitCameraPlugin)
        .add_plugins(EguiPlugin::default())
        .add_systems(Startup, setup)
        .add_systems(
            Update,
            (
                rotate_camera,
                ui::update_ui_scale_from_browser,
                draw_axes_and_planes,
            ),
        )
        .add_systems(EguiPrimaryContextPass, ui::egui_overlays_system)
        .add_systems(EguiPrimaryContextPass, ui::egui_ui_system)
        .run();
}

#[expect(clippy::needless_pass_by_value)]
fn setup(
    mut commands: Commands,
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<StandardMaterial>>,
    mut grid_config: ResMut<GridConfig>,
    mut expr_status: ResMut<ExpressionStatus>,
    expr_config: Res<ExpressionConfig>,
    dim_mapping: Res<DimMapping>,
) {
    let cam_entity = commands
        .spawn((
            Camera3d::default(),
            Transform::from_xyz(CAMERA_RADIUS * 0.7, CAMERA_HEIGHT, CAMERA_RADIUS * 0.7)
                .looking_at(Vec3::ZERO, Vec3::Y),
            Exposure { ev100: 0.0 },
            Tonemapping::None,
            PanOrbitCamera {
                enabled: false,
                focus: Vec3::ZERO,
                target_focus: Vec3::ZERO,
                ..default()
            },
        ))
        .id();

    let total_start = Instant::now();
    let (composite, voxel_count, gen_timings) =
        generate_voxels(&grid_config, &expr_config, &mut expr_status, &dim_mapping);
    grid_config.voxel_count = voxel_count;
    let mut mesh = Mesh::new(
        PrimitiveTopology::TriangleList,
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    let mesh_start = Instant::now();

    if parallel_available() {
        mesh::build_batched_mesh_with_global_corner_ambient_occlusion_par(
            &mut mesh,
            &composite,
            &grid_config,
            voxel_count,
        );
    } else {
        mesh::build_batched_mesh_with_global_corner_ambient_occlusion(
            &mut mesh,
            &composite,
            &grid_config,
            voxel_count,
        );
    }

    let mesh_build = mesh_start.elapsed().as_secs_f64() * 1000.0_f64;
    let total = total_start.elapsed().as_secs_f64() * 1000.0_f64;

    commands.insert_resource(ProfilingDataMs {
        parse: gen_timings.parse,
        sign_grid: gen_timings.sign_grid,
        composite: gen_timings.composite,
        mesh_build,
        total,
    });

    let mesh_handle = meshes.add(mesh);

    let material_handle = materials.add(StandardMaterial {
        base_color: Color::WHITE,
        unlit: true,
        perceptual_roughness: 0.0,
        metallic: 0.0,
        reflectance: 0.0,
        ..default()
    });

    let mesh_entity = commands
        .spawn((Mesh3d(mesh_handle), MeshMaterial3d(material_handle)))
        .id();

    commands.insert_resource(SceneEntities {
        camera: cam_entity,
        voxel_mesh: mesh_entity,
    });
    commands.insert_resource(CameraState {
        angle: 0.0,
        speed: 0.5,
        mode: CameraMode::AutoOrbit,
    });
}

#[expect(clippy::needless_pass_by_value)]
fn draw_axes_and_planes(
    mut gizmos: Gizmos,
    grid_config: Res<GridConfig>,
    show: Res<ShowAxesPlanes>,
    camera_query: Query<&Transform, With<Camera3d>>,
) {
    let inv_size = 1.0 / f32::from(grid_config.size);
    let n = grid_config.size;
    #[expect(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    let step = ((f32::from(n) / 32.0).round() as u32).next_power_of_two();

    if show.show_axes {
        gizmos.line(
            Vec3::new(-0.5, 0.0, 0.0),
            Vec3::new(0.5, 0.0, 0.0),
            Color::srgb(1.0, 0.2, 0.2),
        );
        gizmos.line(
            Vec3::new(0.0, -0.5, 0.0),
            Vec3::new(0.0, 0.5, 0.0),
            Color::srgb(0.2, 1.0, 0.2),
        );
        gizmos.line(
            Vec3::new(0.0, 0.0, -0.5),
            Vec3::new(0.0, 0.0, 0.5),
            Color::srgb(0.2, 0.2, 1.0),
        );

        let cam_t = camera_query.iter().next().copied().unwrap_or_default();

        for (pos, label, color) in [
            (Vec3::new(0.55, 0.0, 0.0), "X", Color::srgb(1.0, 0.2, 0.2)),
            (Vec3::new(0.0, 0.55, 0.0), "Y", Color::srgb(0.2, 1.0, 0.2)),
            (Vec3::new(0.0, 0.0, 0.55), "Z", Color::srgb(0.2, 0.2, 1.0)),
        ] {
            #[expect(clippy::arithmetic_side_effects)]
            let view_dir = (cam_t.translation - pos).normalize();
            #[expect(clippy::arithmetic_side_effects)]
            let cam_up = cam_t.rotation * Vec3::Y;
            let right = cam_up.cross(view_dir).normalize();
            let up = view_dir.cross(right);
            let rot = Quat::from_mat3(&Mat3::from_cols(right, up, view_dir));
            let isometry = Isometry3d::new(pos, rot);
            gizmos.text(isometry, label, 0.04, Vec2::ZERO, color);
        }
    }

    if show.show_ground_grid {
        let gc = Color::srgba(0.5, 0.5, 0.5, 0.35);

        for i in (0..=n).step_by(step as usize) {
            let pos = f32::from(i).cond_mul_add(inv_size, -0.5);
            gizmos.line(Vec3::new(-0.5, -0.5, pos), Vec3::new(0.5, -0.5, pos), gc);
            gizmos.line(Vec3::new(pos, -0.5, -0.5), Vec3::new(pos, -0.5, 0.5), gc);
        }
    }

    if show.show_planes {
        // XY plane at z = center.z — red grid
        let cr = Color::srgba(1.0, 0.2, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let pos = f32::from(i).cond_mul_add(inv_size, -0.5);
            gizmos.line(Vec3::new(-0.5, pos, 0.0), Vec3::new(0.5, pos, 0.0), cr);
            gizmos.line(Vec3::new(pos, -0.5, 0.0), Vec3::new(pos, 0.5, 0.0), cr);
        }

        // XZ plane at y = center.y — green grid
        let cg = Color::srgba(0.2, 1.0, 0.2, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let pos = f32::from(i).cond_mul_add(inv_size, -0.5);
            gizmos.line(Vec3::new(-0.5, 0.0, pos), Vec3::new(0.5, 0.0, pos), cg);
            gizmos.line(Vec3::new(pos, 0.0, -0.5), Vec3::new(pos, 0.0, 0.5), cg);
        }

        // YZ plane at x = center.x — blue grid
        let cb = Color::srgba(0.2, 0.2, 1.0, 0.3);
        for i in (0..=n).step_by(step as usize) {
            let pos = f32::from(i).cond_mul_add(inv_size, -0.5);
            gizmos.line(Vec3::new(0.0, -0.5, pos), Vec3::new(0.0, 0.5, pos), cb);
            gizmos.line(Vec3::new(0.0, pos, -0.5), Vec3::new(0.0, pos, 0.5), cb);
        }
    }
}

#[expect(clippy::needless_pass_by_value)]
fn rotate_camera(
    time: Res<Time>,
    mut camera: ResMut<CameraState>,
    mut query_camera: Query<&mut Transform, With<Camera3d>>,
) {
    if camera.mode != CameraMode::AutoOrbit {
        return;
    }
    camera.angle = time.delta_secs().cond_mul_add(camera.speed, camera.angle);
    for mut transform in query_camera.iter_mut() {
        transform.translation = Vec3::new(
            CAMERA_RADIUS * camera.angle.cos(),
            CAMERA_HEIGHT,
            CAMERA_RADIUS * camera.angle.sin(),
        );
        transform.look_at(Vec3::ZERO, Vec3::Y);
    }
}
