use bevy::{mesh::Indices, prelude::*};
use rayon::prelude::*;

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
fn grid_index(x: usize, y: usize, z: usize, grid_size: usize) -> usize {
    x + y * grid_size + z * grid_size.pow(2)
}

/// Decodes color from u32 with +1 offset.
/// 0 = empty, 1 = #000000 (black), 0xFFFFFF+1 = #FFFFFF (white)
#[inline]
fn decode_color(packed: u32) -> LinearRgba {
    let val = packed - 1;
    let r = ((val >> 16) & 0xFF) as f32 / 255.0;
    let g = ((val >> 8) & 0xFF) as f32 / 255.0;
    let b = (val & 0xFF) as f32 / 255.0;
    LinearRgba::rgb(r, g, b)
}

#[inline]
fn is_occupied(
    composite: &[u32],
    x: i32,
    y: i32,
    z: i32,
    grid_size: i32,
    grid_size_usize: usize,
) -> bool {
    if !(0..grid_size).contains(&x) || !(0..grid_size).contains(&y) || !(0..grid_size).contains(&z)
    {
        return false;
    }
    composite[grid_index(x as usize, y as usize, z as usize, grid_size_usize)] != 0
}

fn process_z_range_multi(
    z_start: u32,
    z_end: u32,
    composite: &[u32],
    size: u32,
    voxel_count: usize,
) -> MeshData {
    let size_i32 = size as i32;
    let size_usize = size as usize;
    let inv_size = 1.0 / size as f32;

    let mut positions: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut normals: Vec<[f32; 3]> = Vec::with_capacity(voxel_count * 30);
    let mut colors: Vec<[f32; 4]> = Vec::with_capacity(voxel_count * 30);
    let mut indices: Vec<u32> = Vec::with_capacity(voxel_count * 72);

    for z in z_start..z_end {
        for y in 0..size {
            for x in 0..size {
                let idx = grid_index(x as usize, y as usize, z as usize, size_usize);

                let voxel_val = composite[idx];
                if voxel_val == 0 {
                    continue;
                }

                // Skip interior voxels with all 6 faces occluded
                let all_occluded = FACE_DEFS.iter().all(|(_, _, off)| {
                    is_occupied(
                        composite,
                        x as i32 + off[0],
                        y as i32 + off[1],
                        z as i32 + off[2],
                        size_i32,
                        size_usize,
                    )
                });
                if all_occluded {
                    continue;
                }

                let base_linear = decode_color(voxel_val);
                let offset = Vec3::new(
                    x as f32 * inv_size - 0.5,
                    y as f32 * inv_size - 0.5,
                    z as f32 * inv_size - 0.5,
                );

                let mut ao_cache = [0.0f32; 8];
                let mut ao_mask: u8 = 0;

                for (normal, corners, neighbor_offset) in FACE_DEFS {
                    let nx = x as i32 + neighbor_offset[0];
                    let ny = y as i32 + neighbor_offset[1];
                    let nz = z as i32 + neighbor_offset[2];

                    if is_occupied(composite, nx, ny, nz, size_i32, size_usize) {
                        continue;
                    }

                    let mut ao_vals = [0.0f32; 4];
                    let mut corner_colors = [[0.0; 4]; 4];

                    for (i, &corner) in corners.iter().enumerate() {
                        let [fcx, fcy, fcz] = corner;

                        let cx_off = (fcx > 0.0) as i32;
                        let cy_off = (fcy > 0.0) as i32;
                        let cz_off = (fcz > 0.0) as i32;
                        let corner_idx = (cx_off | (cy_off << 1) | (cz_off << 2)) as usize;

                        let ambient_occlusion = if (ao_mask & (1 << corner_idx)) != 0 {
                            ao_cache[corner_idx]
                        } else {
                            ao_mask |= 1 << corner_idx;

                            let cx = x as i32 + cx_off;
                            let cy = y as i32 + cy_off;
                            let cz = z as i32 + cz_off;

                            let mut occlusion = 0;
                            if is_occupied(composite, cx - 1, cy, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(composite, cx, cy - 1, cz, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            if is_occupied(composite, cx, cy, cz - 1, size_i32, size_usize) {
                                occlusion += 1;
                            }
                            ao_cache[corner_idx] = AMBIENT_OCCLUSION_FACTORS[occlusion];
                            ao_cache[corner_idx]
                        };
                        ao_vals[i] = ambient_occlusion;

                        corner_colors[i] = [
                            base_linear.red * ambient_occlusion,
                            base_linear.green * ambient_occlusion,
                            base_linear.blue * ambient_occlusion,
                            base_linear.alpha,
                        ];
                    }

                    let start_idx = positions.len() as u32;

                    for (i, &corner) in corners.iter().enumerate() {
                        let [cx, cy, cz] = corner;
                        positions.push([
                            cx * inv_size + offset.x,
                            cy * inv_size + offset.y,
                            cz * inv_size + offset.z,
                        ]);
                        normals.push(normal);
                        colors.push(corner_colors[i]);
                    }

                    if ao_vals[0] + ao_vals[2] > ao_vals[1] + ao_vals[3] {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx + 1,
                            start_idx + 2,
                            start_idx,
                            start_idx + 2,
                            start_idx + 3,
                        ]);
                    } else {
                        indices.extend_from_slice(&[
                            start_idx,
                            start_idx + 1,
                            start_idx + 3,
                            start_idx + 1,
                            start_idx + 2,
                            start_idx + 3,
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
    composite: &[u32],
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
    composite: &[u32],
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
            let z_start = z_range[0];
            let z_end = z_range[z_range.len() - 1] + 1;
            process_z_range_multi(z_start, z_end, composite, size, voxel_count_per_chunk)
        })
        .collect();

    let vertex_counts: Vec<usize> = results.iter().map(|(pos, _, _, _)| pos.len()).collect();
    let index_counts: Vec<usize> = results.iter().map(|(_, _, _, ind)| ind.len()).collect();
    let total_vertices: usize = vertex_counts.iter().sum();
    let total_indices: usize = index_counts.iter().sum();

    let mut vertex_offsets = Vec::with_capacity(results.len());
    {
        let mut v_off = 0u32;
        for &vc in &vertex_counts {
            vertex_offsets.push(v_off);
            v_off += vc as u32;
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

    rayon::scope(|s| {
        for (i, (((pos_part, norm_part), col_part), idx_part)) in pos_parts
            .into_iter()
            .zip(norm_parts)
            .zip(col_parts)
            .zip(idx_parts)
            .enumerate()
        {
            let result = &results[i];
            let v_offset = vertex_offsets[i];

            s.spawn(move |_| {
                pos_part.copy_from_slice(&result.0);
                norm_part.copy_from_slice(&result.1);
                col_part.copy_from_slice(&result.2);
                for (dst, &src) in idx_part.iter_mut().zip(result.3.iter()) {
                    *dst = src + v_offset;
                }
            });
        }
    });

    mesh.insert_attribute(Mesh::ATTRIBUTE_POSITION, positions);
    mesh.insert_attribute(Mesh::ATTRIBUTE_NORMAL, normals);
    mesh.insert_attribute(Mesh::ATTRIBUTE_COLOR, colors);
    mesh.insert_indices(Indices::U32(indices));
}
