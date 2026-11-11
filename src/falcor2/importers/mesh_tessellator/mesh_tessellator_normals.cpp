// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator_internal.h"

#include <algorithm>
#include <cmath>

namespace falcor {
namespace mesh_tessellator {

void calculate_smooth_normals(const TessellatorInputMesh& input, size_t sample, std::vector<float>& normal_values)
{
    FALCOR_CHECK(normal_values.empty(), "Smooth-normal output storage must be empty and separate from input storage");
    FALCOR_ASSERT(sample < input.positions.size());
    const MeshAttributeStream& positions = input.positions[sample];

    const size_t vertex_count = positions.value_count();
    detail::check_stream_layout(
        positions,
        input.face_vertex_counts.size(),
        vertex_count,
        input.face_vertex_indices.size()
    );
    FALCOR_CHECK(positions.component_count() == 3, "Smooth-normal input positions must have three components");
    FALCOR_CHECK(positions.indices.empty(), "Smooth-normal input positions cannot be indexed separately");

    std::vector<bool> holes(input.face_vertex_counts.size(), false);
    for (int face_index : input.hole_indices) {
        FALCOR_CHECK(
            face_index >= 0 && size_t(face_index) < holes.size(),
            "Smooth-normal hole references face {} but the mesh has {} faces",
            face_index,
            holes.size()
        );
        holes[face_index] = true;
    }

    normal_values.assign(vertex_count * 3, 0.f);
    const bool left_handed = detail::is_left_handed(input.orientation);

    size_t face_vertex_offset = 0;
    for (size_t face_index = 0; face_index < input.face_vertex_counts.size(); ++face_index) {
        const int face_vertex_count = input.face_vertex_counts[face_index];
        FALCOR_CHECK(
            face_vertex_count >= 0,
            "Mesh face {} has negative vertex count {}",
            face_index,
            face_vertex_count
        );
        FALCOR_CHECK(
            face_vertex_offset + size_t(face_vertex_count) <= input.face_vertex_indices.size(),
            "Mesh face-vertex counts exceed the face-vertex index array"
        );

        const size_t current_face_vertex_offset = face_vertex_offset;
        face_vertex_offset += size_t(face_vertex_count);
        if (holes[face_index] || face_vertex_count < 3)
            continue;

        for (int local_vertex = 0; local_vertex < face_vertex_count; ++local_vertex) {
            const int previous_offset = (local_vertex + face_vertex_count - 1) % face_vertex_count;
            const int next_offset = (local_vertex + 1) % face_vertex_count;
            const int vertex_index = input.face_vertex_indices[current_face_vertex_offset + local_vertex];
            int previous_index = input.face_vertex_indices[current_face_vertex_offset + previous_offset];
            int next_index = input.face_vertex_indices[current_face_vertex_offset + next_offset];
            if (left_handed)
                std::swap(previous_index, next_index);
            FALCOR_CHECK(
                vertex_index >= 0 && size_t(vertex_index) < vertex_count && previous_index >= 0
                    && size_t(previous_index) < vertex_count && next_index >= 0 && size_t(next_index) < vertex_count,
                "Mesh face {} references a vertex outside [0, {})",
                face_index,
                vertex_count
            );
            const float3 position = detail::stream_element<float3>(positions, vertex_index);
            const float3 previous = detail::stream_element<float3>(positions, previous_index);
            const float3 next = detail::stream_element<float3>(positions, next_index);
            const float3 corner_normal = sgl::math::cross(next - position, previous - position);
            normal_values[vertex_index * 3 + 0] += corner_normal.x;
            normal_values[vertex_index * 3 + 1] += corner_normal.y;
            normal_values[vertex_index * 3 + 2] += corner_normal.z;
        }
    }
    FALCOR_CHECK(
        face_vertex_offset == input.face_vertex_indices.size(),
        "Mesh has {} face vertices but {} face-vertex indices",
        face_vertex_offset,
        input.face_vertex_indices.size()
    );

    for (size_t vertex_index = 0; vertex_index < vertex_count; ++vertex_index) {
        float3 normal{
            normal_values[vertex_index * 3 + 0],
            normal_values[vertex_index * 3 + 1],
            normal_values[vertex_index * 3 + 2],
        };
        const float length_squared = sgl::math::dot(normal, normal);
        if (length_squared > 0.f)
            normal *= 1.f / std::sqrt(length_squared);
        normal_values[vertex_index * 3 + 0] = normal.x;
        normal_values[vertex_index * 3 + 1] = normal.y;
        normal_values[vertex_index * 3 + 2] = normal.z;
    }
}

} // namespace mesh_tessellator
} // namespace falcor
