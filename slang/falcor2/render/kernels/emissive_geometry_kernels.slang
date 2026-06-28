// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;
import falcor2.utils;

/// Returns the index of the first element in the sorted buffer that is not less than `value`.
/// The range [lo, hi) must be sorted in non-descending order.
uint lower_bound<T : IComparable>(StructuredBuffer<T> buffer, uint lo, uint hi, T value)
{
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (buffer[mid] < value) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    return lo;
}

/// Returns the index of the first element in the sorted buffer that is greater than `value`.
/// The range [lo, hi) must be sorted in non-descending order.
uint upper_bound<T : IComparable>(StructuredBuffer<T> buffer, uint lo, uint hi, T value)
{
    while (lo < hi) {
        uint mid = lo + (hi - lo) / 2;
        if (value < buffer[mid]) {
            hi = mid;
        } else {
            lo = mid + 1;
        }
    }
    return lo;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void gather_triangle_count(
    uint geometry_instance_index: SV_DispatchThreadID,
    RWStructuredBuffer<uint> is_emissive,
    RWStructuredBuffer<uint> triangle_count,
    uniform uint geometry_instance_count
)
{
    if (geometry_instance_index >= geometry_instance_count)
        return;

    GeometryInstanceDesc geometry_instance_desc
        = g_scene.get_geometry_instance_desc(GeometryInstanceID(geometry_instance_index));
    IMaterial material = g_scene.get_material(geometry_instance_desc.header.material_id);
    bool potentially_emissive = material.is_emissive();
    if (geometry_instance_desc.header.type == GeometryType::triangle && potentially_emissive) {
        TriangleGeometryInstanceDesc triangle_geometry_instance_desc
            = reinterpret<TriangleGeometryInstanceDesc>(geometry_instance_desc);
        is_emissive[geometry_instance_index] = 1;
        triangle_count[geometry_instance_index] = triangle_geometry_instance_desc.index_count / 3;
    } else {
        is_emissive[geometry_instance_index] = 0;
        triangle_count[geometry_instance_index] = 0;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void setup_geometry_instance_to_triangle_id(
    uint geometry_instance_index: SV_DispatchThreadID,
    StructuredBuffer<uint> is_emissive,
    StructuredBuffer<uint> triangle_offset,
    RWStructuredBuffer<EmissiveTriangleID> geometry_instance_to_triangle_id,
    uniform uint geometry_instance_count
)
{
    if (geometry_instance_index >= geometry_instance_count)
        return;

    if (is_emissive[geometry_instance_index] == 0) {
        geometry_instance_to_triangle_id[geometry_instance_index] = EmissiveTriangleID::invalid;
    } else {
        geometry_instance_to_triangle_id[geometry_instance_index]
            = EmissiveTriangleID(triangle_offset[geometry_instance_index]);
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void setup_triangle(
    uint triangle_index: SV_DispatchThreadID,
    RWStructuredBuffer<PackedEmissiveTriangle> triangles,
    StructuredBuffer<uint> triangle_offset,
    uniform uint triangle_count,
    uniform uint geometry_instance_count
)
{
    if (triangle_index >= triangle_count)
        return;

    // Binary search to find the geometry instance this triangle belongs to.
    uint geometry_instance_index = upper_bound(triangle_offset, 0, geometry_instance_count, triangle_index) - 1;
    GeometryInstanceID geometry_instance_id = GeometryInstanceID(geometry_instance_index);

    uint primitive_index = triangle_index - triangle_offset[geometry_instance_index];

    TriangleVertex vertices[3] = g_scene.get_triangle_vertices(geometry_instance_id, primitive_index);

    EmissiveTriangle triangle = {};
    triangle.uv[0] = vertices[0].uv[0];
    triangle.uv[1] = vertices[1].uv[0];
    triangle.uv[2] = vertices[2].uv[0];
    triangle.geometry_instance_id = geometry_instance_id;
    triangle.material_id = g_scene.get_geometry_instance_desc(triangle.geometry_instance_id).header.material_id;

    triangles[triangle_index] = detail::pack_emissive_triangle(triangle);
}

static constexpr uint64_t SAMPLE_BUCKET_SIZE = 32;

[shader("compute")]
[numthreads(128, 1, 1)]
void setup_triangle_sampling(
    uint triangle_index: SV_DispatchThreadID,
    StructuredBuffer<PackedEmissiveTriangle> triangles,
    RWStructuredBuffer<uint64_t> sample_count,
    RWStructuredBuffer<uint2> sample_grid,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    EmissiveTriangle triangle = detail::unpack_emissive_triangle(triangles[triangle_index]);
    IMaterial material = g_scene.get_material(triangle.material_id);
    uint2 grid = material.compute_emission_sampling_grid(triangle.uv);
    uint64_t count = grid.x * grid.y;
    count = (count + SAMPLE_BUCKET_SIZE - 1) / SAMPLE_BUCKET_SIZE * SAMPLE_BUCKET_SIZE;
    sample_count[triangle_index] = count;
    sample_grid[triangle_index] = grid;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void compute_triangle_max_emission(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<PackedEmissiveTriangle> triangles,
    StructuredBuffer<uint64_t> sample_offset,
    StructuredBuffer<uint64_t> sample_count,
    StructuredBuffer<uint2> sample_grid,
    uniform uint triangle_count,
    uniform uint thread_count,
    uniform uint64_t total_sample_count,
    RWStructuredBuffer<uint> max_emission
)
{
    uint64_t sample_index = tid;
    while (sample_index < total_sample_count) {
        // Binary search to find the triangle this sample belongs to.
        uint triangle_index = upper_bound(sample_offset, 0, triangle_count, sample_index) - 1;
        EmissiveTriangle triangle = detail::unpack_emissive_triangle(triangles[triangle_index]);

        // Read sampling parameters.
        uint2 grid = sample_grid[triangle_index];
        uint64_t count = sample_count[triangle_index];

        // Compute the local sample index for this triangle.
        uint local_sample_index = uint(sample_index - sample_offset[triangle_index]);

        // Compute the local number of samples for this triangle.
        float max_emission_component = 0.0;
        uint local_sample_count = grid.x * grid.y;
        if (local_sample_index < local_sample_count) {
            uint2 sample_coord = uint2(local_sample_index % grid.x, local_sample_index / grid.x);
            float2 sample_uv = (float2(sample_coord) + 0.5) / float2(grid);
            float3 sample_bary = sample_triangle(sample_uv);
            IMaterial material = g_scene.get_material(triangle.material_id);
            SurfaceInteraction si = {};
            si.uv = triangle.get_uv(sample_bary);
            float3 emission = material.eval_emission(si, ExplicitLodSampler());
            max_emission_component = max(emission.x, max(emission.y, emission.z));
        }
        float wave_max_emission_component = WaveActiveMax(max_emission_component);
        if (WaveIsFirstLane()) {
            // Compute maximum emission component.
            // This works because the bit-level representation of float is monotonic for non-negative numbers.
            InterlockedMax(max_emission[triangle_index], asuint(wave_max_emission_component));
        }

        sample_index += thread_count;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void compute_triangle_max_emission_factor(
    uint triangle_index: SV_DispatchThreadID,
    StructuredBuffer<uint64_t> sample_count,
    StructuredBuffer<float> max_emission,
    RWStructuredBuffer<float> emission_factor,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    // Scale emission into uint64 fixed-point for atomic accumulation.
    // Worst-case sum is max_emission * factor * sample_count, which must fit in uint64.
    // Use factor = C / (max_emission * sample_count) so the max sum equals C,
    // maximizing precision without overflow. 1e18 stays safely below UINT64_MAX (~1.8e19).
    float factor = 1e18 / (max(max_emission[triangle_index], 1e-30) * float(sample_count[triangle_index]));
    emission_factor[triangle_index] = factor;
}

[shader("compute")]
[numthreads(128, 1, 1)]
void accumulate_triangle_emission(
    uint tid: SV_DispatchThreadID,
    StructuredBuffer<PackedEmissiveTriangle> triangles,
    StructuredBuffer<uint64_t> sample_offset,
    StructuredBuffer<uint64_t> sample_count,
    StructuredBuffer<uint2> sample_grid,
    StructuredBuffer<float> emission_factor,
    uniform uint triangle_count,
    uniform uint thread_count,
    uniform uint64_t total_sample_count,
    RWStructuredBuffer<uint64_t> accumulated_emission
)
{
    uint64_t sample_index = tid;
    while (sample_index < total_sample_count) {
        // Binary search to find the triangle this sample belongs to.
        uint triangle_index = upper_bound(sample_offset, 0, triangle_count, sample_index) - 1;
        EmissiveTriangle triangle = detail::unpack_emissive_triangle(triangles[triangle_index]);

        // Read sampling parameters.
        uint2 grid = sample_grid[triangle_index];
        uint64_t count = sample_count[triangle_index];

        // Compute the local sample index for this triangle.
        uint local_sample_index = uint(sample_index - sample_offset[triangle_index]);

        // Compute the local number of samples for this triangle.
        uint64_t3 emission_fixed_point = uint64_t3(0);
        uint local_sample_count = grid.x * grid.y;
        if (local_sample_index < local_sample_count) {
            uint2 sample_coord = uint2(local_sample_index % grid.x, local_sample_index / grid.x);
            float2 sample_uv = (float2(sample_coord) + 0.5) / float2(grid);
            float3 sample_bary = sample_triangle(sample_uv);
            IMaterial material = g_scene.get_material(triangle.material_id);
            SurfaceInteraction si = {};
            si.uv = triangle.get_uv(sample_bary);
            float3 emission = material.eval_emission(si, ExplicitLodSampler());
            emission_fixed_point = uint64_t3(emission * emission_factor[triangle_index]);
        }
        uint64_t3 wave_emission_fixed_point = WaveActiveSum(emission_fixed_point);
        if (WaveIsFirstLane()) {
            InterlockedAdd(accumulated_emission[triangle_index * 3], wave_emission_fixed_point.x);
            InterlockedAdd(accumulated_emission[triangle_index * 3 + 1], wave_emission_fixed_point.y);
            InterlockedAdd(accumulated_emission[triangle_index * 3 + 2], wave_emission_fixed_point.z);
        }

        sample_index += thread_count;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
void finalize_triangle_emission(
    uint triangle_index: SV_DispatchThreadID,
    StructuredBuffer<uint2> sample_grid,
    StructuredBuffer<uint64_t> accumulated_emission,
    StructuredBuffer<float> emission_factor,
    RWStructuredBuffer<float4> triangle_emission,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    uint2 grid = sample_grid[triangle_index];
    uint64_t count = grid.x * grid.y;

    uint64_t3 total = uint64_t3(
        accumulated_emission[triangle_index * 3],
        accumulated_emission[triangle_index * 3 + 1],
        accumulated_emission[triangle_index * 3 + 2]
    );

    float3 emission = float3(total / count) / emission_factor[triangle_index];
    triangle_emission[triangle_index] = float4(emission, 0.0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void gather_active_triangles(
    uint triangle_index: SV_DispatchThreadID,
    StructuredBuffer<float4> triangle_emission,
    RWStructuredBuffer<uint> is_active,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    is_active[triangle_index] = select(any(triangle_emission[triangle_index] > 0.0), 1, 0);
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void setup_active_triangle_mapping(
    uint triangle_index: SV_DispatchThreadID,
    StructuredBuffer<uint> is_active,
    StructuredBuffer<uint> active_offset,
    RWStructuredBuffer<ActiveEmissiveTriangleID> triangle_id_to_active_triangle_id,
    RWStructuredBuffer<EmissiveTriangleID> active_triangle_id_to_triangle_id,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    if (is_active[triangle_index] != 0) {
        uint active_index = active_offset[triangle_index];
        triangle_id_to_active_triangle_id[triangle_index] = ActiveEmissiveTriangleID(active_index);
        active_triangle_id_to_triangle_id[active_index] = EmissiveTriangleID(triangle_index);
    } else {
        triangle_id_to_active_triangle_id[triangle_index] = ActiveEmissiveTriangleID::invalid;
    }
}

[shader("compute")]
[numthreads(128, 1, 1)]
public void update_triangle(
    uint triangle_index: SV_DispatchThreadID,
    RWStructuredBuffer<PackedEmissiveTriangle> triangles,
    RWStructuredBuffer<float> triangle_flux,
    RWStructuredBuffer<float> active_triangle_flux,
    StructuredBuffer<ActiveEmissiveTriangleID> triangle_id_to_active_triangle_id,
    StructuredBuffer<EmissiveTriangleID> geometry_instance_to_triangle_id,
    StructuredBuffer<float4> triangle_emission,
    uniform uint triangle_count
)
{
    if (triangle_index >= triangle_count)
        return;

    EmissiveTriangle triangle = detail::unpack_emissive_triangle(triangles[triangle_index]);

    TriangleGeometryInstanceDesc geometry_instance_desc
        = reinterpret<TriangleGeometryInstanceDesc>(g_scene.get_geometry_instance_desc(triangle.geometry_instance_id));

    uint primitive_index = triangle_index - uint(geometry_instance_to_triangle_id[uint(triangle.geometry_instance_id)]);

    TriangleVertex vertices[3] = g_scene.get_triangle_vertices(triangle.geometry_instance_id, primitive_index);
    float4x4 transform = g_scene.get_world_from_object(triangle.geometry_instance_id);
    triangle.pos_ws[0] = transform_point(transform, vertices[0].position);
    triangle.pos_ws[1] = transform_point(transform, vertices[1].position);
    triangle.pos_ws[2] = transform_point(transform, vertices[2].position);

    float3 edge0 = triangle.pos_ws[1] - triangle.pos_ws[0];
    float3 edge1 = triangle.pos_ws[2] - triangle.pos_ws[0];
    triangle.normal_ws = normalize(cross(edge0, edge1));

    float flux = 0.0;
    if (any(isnan(triangle.normal_ws))) {
        triangle.area_ws = 0.0;
        triangle.normal_ws = float3(0.0, 0.0, 0.0);
    } else {
        triangle.area_ws = length(cross(edge0, edge1)) / 2;
        float3 emission = triangle_emission[triangle_index].xyz;
        flux = triangle.area_ws * luminance(emission);
    }

    triangles[triangle_index] = detail::pack_emissive_triangle(triangle);
    triangle_flux[triangle_index] = flux;

    ActiveEmissiveTriangleID active_triangle_id = triangle_id_to_active_triangle_id[triangle_index];
    if (active_triangle_id != ActiveEmissiveTriangleID::invalid)
        active_triangle_flux[uint(active_triangle_id)] = flux;
}
