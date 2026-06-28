// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module example_animated;
__exported import falcor2.minitracer.scene;
import falcor2.utils;
import slangpy;
import sgl.device.print;

uint to_index1d(uint2 index2, uint2 dimensions)
{
    return index2.x + index2.y * dimensions.x;
}

public void animate_sine(
    uint2 index2d,
    RWStructuredBuffer<Vertex> vertices,
    uint vertex_start,
    uint2 dimensions,
    float amplitude,
    float time
)
{
    uint vertex_index = vertex_start + to_index1d(index2d, dimensions);

    // Normalize coordinates to [0, 1]
    float u = float(index2d.x) / float(dimensions.x - 1);
    float v = float(index2d.y) / float(dimensions.y - 1);

    // Diagonal sine wave: 10 waves along the diagonal (u + v)
    float wave_count = 10.0;
    float height = amplitude * sin((u + v) * wave_count * 3.14159265 * 2.0 + time);

    vertices[vertex_index].position.z = height;
}

public void
recalc_normals(uint2 index2d, RWStructuredBuffer<Vertex> vertices, uniform uint vertex_start, uniform uint2 dimensions)
{
    int2 this_index2d = int2(index2d);
    int2 right_index2d = this_index2d + int2(1, 0);
    int2 down_index2d = this_index2d + int2(0, 1);
    int2 left_index2d = this_index2d - int2(1, 0);
    int2 up_index2d = this_index2d - int2(0, 1);

    // Clamp to edge
    right_index2d = min(right_index2d, int2(dimensions) - 1);
    down_index2d = min(down_index2d, int2(dimensions) - 1);
    left_index2d = max(left_index2d, int2(0));
    up_index2d = max(up_index2d, int2(0));

    float3 center_pos = vertices[vertex_start + to_index1d(this_index2d, dimensions)].position;
    float3 right_pos = vertices[vertex_start + to_index1d(right_index2d, dimensions)].position;
    float3 down_pos = vertices[vertex_start + to_index1d(down_index2d, dimensions)].position;
    float3 left_pos = vertices[vertex_start + to_index1d(left_index2d, dimensions)].position;
    float3 up_pos = vertices[vertex_start + to_index1d(up_index2d, dimensions)].position;

    float3 right_vec = right_pos - center_pos;
    float3 down_vec = down_pos - center_pos;
    float3 left_vec = left_pos - center_pos;
    float3 up_vec = up_pos - center_pos;

    // Average the cross products of the surrounding edges to get a smoother normal
    float3 normal = normalize(
        cross(right_vec, down_vec) + cross(down_vec, left_vec) + cross(left_vec, up_vec) + cross(up_vec, right_vec)
    );

    Frame frame = { normal };

    vertices[vertex_start + to_index1d(this_index2d, dimensions)].normal = frame.normal;
    vertices[vertex_start + to_index1d(this_index2d, dimensions)].tangent = frame.tangent;
    vertices[vertex_start + to_index1d(this_index2d, dimensions)].bitangent = frame.bitangent;
}

public void reset_heightfield(uint2 index2d, RWStructuredBuffer<Vertex> vertices, uint vertex_start, uint2 dimensions)
{
    uint vertex_index = vertex_start + to_index1d(index2d, dimensions);

    vertices[vertex_index].position.y = 0.0;
}

public void apply_heightfield(
    uint2 index2d,
    RWStructuredBuffer<Vertex> vertices,
    uint vertex_start,
    uint2 dimensions,
    Tensor<float, 2> heightfield,
    float height_offset
)
{
    uint vertex_index = vertex_start + to_index1d(index2d, dimensions);

    float height = heightfield[index2d] + height_offset;
    vertices[vertex_index].position.z = height;
}

public void update_material_index(
    RWStructuredBuffer<GeometryInstance> geometry_instances,
    uint instance_index,
    uint new_material_index
)
{
    geometry_instances[instance_index].material = MaterialID(new_material_index);
}
