use bevy::{mesh::Indices, prelude::*};
use rayon::prelude::*;

use crate::utils::{CondMulAdd as _, PackedColor, unpack_color};

type Vec3Arr = [f32; 3];
type IVec3Arr = [i32; 3];
type CornerArr = [Vec3Arr; 4];
type MeshData = (Vec<[f32; 3]>, Vec<[f32; 3]>, Vec<[f32; 4]>, Vec<u32>);

const AMBIENT_OCCLUSION_FACTORS: [f32; 4] = [1.0, 0.75, 0.5, 0.3];

const FACE_DEFS: [(Vec3Arr, CornerArr, IVec3Arr); 6] = [
    // +X face
    (
        [1.0, 0.0, 0.0],
        [
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 1.0, 1.0],
            [1.0, 0.0, 1.0],
        ],
        [1, 0, 0],
    ),
    // -X face
    (
        [-1.0, 0.0, 0.0],
        [
            [0.0, 0.0, 1.0],
            [0.0, 1.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        [-1, 0, 0],
    ),
    // +Y face
    (
        [0.0, 1.0, 0.0],
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 0.0],
        ],
        [0, 1, 0],
    ),
    // -Y face
    (
        [0.0, -1.0, 0.0],
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 0.0, 1.0],
        ],
        [0, -1, 0],
    ),
    // +Z face
    (
        [0.0, 0.0, 1.0],
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ],
        [0, 0, 1],
    ),
    // -Z face
    (
        [0.0, 0.0, -1.0],
        [
            [0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        [0, 0, -1],
    ),
];

#[inline]
const fn grid_index(x: usize, y: usize, z: usize, grid_size: usize) -> usize {
    x.wrapping_add(y.wrapping_mul(grid_size))
        .wrapping_add(z.wrapping_mul(grid_size).wrapping_mul(grid_size))
}

#[inline]
fn is_occupied(
    composite: &[PackedColor],
    x: i32,
    y: i32,
    z: i32,
    grid_size: i32,
    grid_size_usize: usize,
) -> bool {
    if !(0_i32..grid_size).contains(&x)
        || !(0_i32..grid_size).contains(&y)
        || !(0_i32..grid_size).contains(&z)
    {
        return false;
    }
    composite
        .get(grid_index(
            x.cast_unsigned() as usize,
            y.cast_unsigned() as usize,
            z.cast_unsigned() as usize,
            grid_size_usize,
        ))
        .is_some_and(Option::is_some)
}

#[expect(clippy::similar_names)]
fn process_z_range_multi(
    z_start: u16,
    z_end: u16,
    composite: &[PackedColor],
    size: u16,
    voxel_count: usize,
) -> MeshData {
    let size_i32 = i32::from(size);
    let size_usize = size as usize;
    let inv_size = 1.0 / f32::from(size);

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(voxel_count.saturating_mul(30));
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(voxel_count.saturating_mul(30));
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(voxel_count.saturating_mul(30));
    let mut indices: Vec<u32> = Vec::with_capacity(voxel_count.saturating_mul(72));

    for z in z_start..z_end {
        for y in 0..size {
            for x in 0..size {
                let idx = grid_index(x as usize, y as usize, z as usize, size_usize);

                let Some(Some(voxel_val)) = composite.get(idx).copied() else {
                    continue;
                };

                // Skip interior voxels with all 6 faces occluded
                let all_occluded = FACE_DEFS.iter().all(|(_, _, off)| {
                    #[expect(clippy::arithmetic_side_effects)]
                    is_occupied(
                        composite,
                        i32::from(x) + off[0],
                        i32::from(y) + off[1],
                        i32::from(z) + off[2],
                        size_i32,
                        size_usize,
                    )
                });
                if all_occluded {
                    continue;
                }

                let base_linear = unpack_color(voxel_val);
                let offset = Vec3::new(
                    f32::from(x).cond_mul_add(inv_size, -0.5),
                    f32::from(y).cond_mul_add(inv_size, -0.5),
                    f32::from(z).cond_mul_add(inv_size, -0.5),
                );

                let mut ao_cache = [0.0_f32; 8];
                let mut ao_mask: u8 = 0;

                for (normal, corners, neighbor_offset) in FACE_DEFS {
                    #[expect(clippy::arithmetic_side_effects)]
                    let nx = i32::from(x) + neighbor_offset[0];
                    #[expect(clippy::arithmetic_side_effects)]
                    let ny = i32::from(y) + neighbor_offset[1];
                    #[expect(clippy::arithmetic_side_effects)]
                    let nz = i32::from(z) + neighbor_offset[2];

                    if is_occupied(composite, nx, ny, nz, size_i32, size_usize) {
                        continue;
                    }

                    let mut ao_vals = [0.0_f32; 4];
                    let mut corner_colors = [[0.0; 4]; 4];

                    for (i, &corner) in corners.iter().enumerate() {
                        let [fcx, fcy, fcz] = corner;

                        let cx_off = i32::from(fcx > 0.0);
                        let cy_off = i32::from(fcy > 0.0);
                        let cz_off = i32::from(fcz > 0.0);
                        let corner_idx = (cx_off | (cy_off << 1_usize) | (cz_off << 2_usize))
                            .cast_unsigned() as usize;

                        if (ao_mask & (1 << corner_idx)) == 0 {
                            ao_mask |= 1 << corner_idx;

                            #[expect(clippy::arithmetic_side_effects)]
                            let cx = i32::from(x) + cx_off;
                            #[expect(clippy::arithmetic_side_effects)]
                            let cy = i32::from(y) + cy_off;
                            #[expect(clippy::arithmetic_side_effects)]
                            let cz = i32::from(z) + cz_off;

                            let mut occlusion = 0;
                            #[expect(clippy::arithmetic_side_effects)]
                            if is_occupied(composite, cx - 1, cy, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            #[expect(clippy::arithmetic_side_effects)]
                            if is_occupied(composite, cx, cy - 1, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            #[expect(clippy::arithmetic_side_effects)]
                            if is_occupied(composite, cx, cy, cz - 1, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            // SAFETY: `corner_idx` packs 3 bits (cx/cy/cz offsets),
                            // so it is in 0..8, within `ao_cache`'s 8 entries.
                            *unsafe { ao_cache.get_unchecked_mut(corner_idx) } =
                                // SAFETY: `occlusion` counts 3 neighbor checks (0..=3),
                                // within `AMBIENT_OCCLUSION_FACTORS`' 4 entries.
                                unsafe { *AMBIENT_OCCLUSION_FACTORS.get_unchecked(occlusion) };
                        }
                        // SAFETY: `corner_idx` packs 3 bits (cx/cy/cz offsets),
                        // so it is in 0..8, within `ao_cache`'s 8 entries.
                        let ambient_occlusion = unsafe { *ao_cache.get_unchecked(corner_idx) };
                        // SAFETY: `i` iterates the 4 face corners, within `ao_vals`' 4 entries.
                        *unsafe { ao_vals.get_unchecked_mut(i) } = ambient_occlusion;

                        // SAFETY: `i` iterates the 4 face corners, within `corner_colors`' 4 entries.
                        *unsafe { corner_colors.get_unchecked_mut(i) } = [
                            base_linear.red * ambient_occlusion,
                            base_linear.green * ambient_occlusion,
                            base_linear.blue * ambient_occlusion,
                            base_linear.alpha,
                        ];
                    }

                    #[expect(clippy::cast_possible_truncation)]
                    let start_idx = positions.len() as u32;

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        positions.push([
                            cx.cond_mul_add(inv_size, offset.x),
                            cy.cond_mul_add(inv_size, offset.y),
                            cz.cond_mul_add(inv_size, offset.z),
                        ]);
                        normals.push(normal);
                        // SAFETY: `i` iterates the 4 face corners, within `corner_colors`' 4 entries.
                        colors.push(unsafe { *corner_colors.get_unchecked(i) });
                    }

                    if ao_vals[0] + ao_vals[2] > ao_vals[1] + ao_vals[3] {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx.wrapping_add(1),
                            start_idx.wrapping_add(2),
                            start_idx,
                            start_idx.wrapping_add(2),
                            start_idx.wrapping_add(3),
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx.wrapping_add(1),
                            start_idx.wrapping_add(3),
                            start_idx.wrapping_add(1),
                            start_idx.wrapping_add(2),
                            start_idx.wrapping_add(3),
                        ]);
                    }
                }
            }
        }
    }

    (positions, normals, colors, indices)
}

pub fn build_batched_mesh_with_global_corner_ambient_occlusion(
    mesh: &mut Mesh,
    composite: &[PackedColor],
    grid_config: &crate::utils::GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let (positions, normals, colors, indices) =
        process_z_range_multi(0, size, composite, size, voxel_count);
    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}

#[inline]
fn split_disjoint<'a, T>(slice: &'a mut [T], counts: &[usize]) -> Vec<&'a mut [T]> {
    debug_assert_eq!(
        counts.iter().copied().sum::<usize>(),
        slice.len(),
        "split_disjoint: counts sum ({}) != slice len ({})",
        counts.iter().copied().sum::<usize>(),
        slice.len()
    );
    let mut parts = Vec::with_capacity(counts.len());
    let mut rest = slice;
    for &count in counts {
        let (part, tail) = rest.split_at_mut(count);
        parts.push(part);
        rest = tail;
    }
    parts
}

pub fn build_batched_mesh_with_global_corner_ambient_occlusion_par(
    mesh: &mut Mesh,
    composite: &[PackedColor],
    grid_config: &crate::utils::GridConfig,
    voxel_count: usize,
) {
    info!("voxel_count: {}", voxel_count);
    let size = grid_config.size;
    let chunk_count = rayon::current_num_threads();
    let chunk_size = (size as usize).div_ceil(chunk_count);
    let voxel_count_per_chunk = voxel_count.div_ceil(chunk_count);
    let results: Vec<_> = (0..size)
        .collect::<Vec<_>>()
        .par_chunks(chunk_size)
        .map(|z_range| {
            let z_start = z_range.first().copied().unwrap_or_default();
            #[expect(clippy::arithmetic_side_effects)]
            let z_end = z_range.last().copied().unwrap_or_default() + 1;
            process_z_range_multi(z_start, z_end, composite, size, voxel_count_per_chunk)
        })
        .collect();

    let vertex_counts: Vec<usize> = results.iter().map(|(pos, _, _, _)| pos.len()).collect();
    let index_counts: Vec<usize> = results.iter().map(|(_, _, _, ind)| ind.len()).collect();
    let total_vertices: usize = vertex_counts.iter().sum();
    let total_indices: usize = index_counts.iter().sum();

    let mut vertex_offsets = Vec::with_capacity(results.len());
    {
        let mut v_off = 0_u32;
        #[expect(clippy::cast_possible_truncation)]
        for &vc in &vertex_counts {
            vertex_offsets.push(v_off);
            v_off = v_off.wrapping_add(vc as u32);
        }
    }

    let mut positions = Vec::with_capacity(total_vertices);
    let mut normals = Vec::with_capacity(total_vertices);
    let mut colors = Vec::with_capacity(total_vertices);
    let mut indices = Vec::with_capacity(total_indices);

    // SAFETY: every element is overwritten below before any read
    unsafe {
        positions.set_len(total_vertices);
        normals.set_len(total_vertices);
        colors.set_len(total_vertices);
        indices.set_len(total_indices);
    }

    let pos_parts = split_disjoint(&mut positions, &vertex_counts);
    let norm_parts = split_disjoint(&mut normals, &vertex_counts);
    let col_parts = split_disjoint(&mut colors, &vertex_counts);
    let idx_parts = split_disjoint(&mut indices, &index_counts);

    rayon::scope(|scope| {
        for ((result, &v_offset), (((pos_part, norm_part), col_part), idx_part)) in
            results.iter().zip(&vertex_offsets).zip(
                pos_parts
                    .into_iter()
                    .zip(norm_parts)
                    .zip(col_parts)
                    .zip(idx_parts),
            )
        {
            scope.spawn(move |_| {
                pos_part.copy_from_slice(&result.0);
                norm_part.copy_from_slice(&result.1);
                col_part.copy_from_slice(&result.2);
                for (dst, &src) in idx_part.iter_mut().zip(result.3.iter()) {
                    *dst = src.wrapping_add(v_offset);
                }
            });
        }
    });

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}
