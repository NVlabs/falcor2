// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator_internal.h"

#include <algorithm>

namespace falcor {
namespace mesh_tessellator {
namespace detail {

namespace {

struct TriangulationLayout {
    size_t vertex_count = 0;
    std::vector<size_t> subgeometry_triangle_counts;
};

TriangulationLayout prepare_triangulation_layout(
    const TessellatorInputMesh& input,
    std::span<int> subgeometry_mapping,
    size_t subgeometry_count
)
{
    TriangulationLayout layout;
    layout.subgeometry_triangle_counts.resize(subgeometry_count, 0);
    bool found_degenerate_face = false;
    const int face_count = int(input.face_vertex_counts.size());
    for (int face_index = 0; face_index < face_count; ++face_index) {
        if (subgeometry_mapping[face_index] < 0)
            continue;

        const int face_vertex_count = input.face_vertex_counts[face_index];
        if (face_vertex_count < 3) {
            subgeometry_mapping[face_index] = -1;
            found_degenerate_face = true;
            continue;
        }
        layout.vertex_count += size_t(face_vertex_count);
        layout.subgeometry_triangle_counts[subgeometry_mapping[face_index]] += size_t(face_vertex_count - 2);
    }
    if (found_degenerate_face)
        sgl::log_warn("Degenerate face found on mesh '{}'.", input.name);
    return layout;
}

float3 calculate_flat_normal(
    const TessellatorInputMesh& input,
    size_t face_vertex_offset,
    int face_vertex_count,
    const int (&fan_vertex_offsets)[2]
)
{
    const float3 v0 = stream_element<float3>(input.positions, input.face_vertex_indices[face_vertex_offset]);
    float3 normal(0.f);
    for (int triangle = 0; triangle < face_vertex_count - 2; ++triangle) {
        const float3 v1 = stream_element<float3>(
            input.positions,
            input.face_vertex_indices[face_vertex_offset + triangle + fan_vertex_offsets[0]]
        );
        const float3 v2 = stream_element<float3>(
            input.positions,
            input.face_vertex_indices[face_vertex_offset + triangle + fan_vertex_offsets[1]]
        );
        normal += sgl::math::cross(v1 - v0, v2 - v0);
    }
    return sgl::math::normalize(normal);
}

} // namespace

ImporterMesh triangulate_mesh(const TessellatorInputMesh& input)
{
    const size_t vertex_count = input.positions.value_count();
    check_stream_layout(
        input.positions,
        input.face_vertex_counts.size(),
        vertex_count,
        input.face_vertex_indices.size()
    );
    FALCOR_CHECK(input.positions.component_count() == 3, "Mesh tessellator positions must have three components");
    FALCOR_CHECK(input.positions.indices.empty(), "Mesh tessellator positions cannot be indexed separately");

    MeshAttributeStream default_uv;
    const std::vector<const MeshAttributeStream*> source_streams = collect_streams(input, default_uv);
    for (const MeshAttributeStream* stream : source_streams)
        check_stream_layout(*stream, input.face_vertex_counts.size(), vertex_count, input.face_vertex_indices.size());

    const MeshAttributeStream* authored_normals = find_stream(input.streams, ImporterSemantic::normal);
    if (authored_normals)
        FALCOR_CHECK(authored_normals->component_count() == 3, "Mesh tessellator normals must have three components");

    const ImporterMeshDefaultAttribute position_attribute = input.positions.attribute;
    const ImporterMeshDefaultAttribute normal_attribute
        = authored_normals ? authored_normals->attribute : ImporterMeshDefaultAttribute{ImporterSemantic::normal, 0, 3};
    std::vector<ImporterMeshDefaultAttribute> output_attributes;
    output_attributes.reserve(source_streams.size() + 2);
    add_output_attribute(output_attributes, position_attribute);
    add_output_attribute(output_attributes, normal_attribute);

    struct StreamBinding {
        const MeshAttributeStream* input_stream;
        ImporterMeshDefaultAttribute output_attribute;
        OutputBinding output;
    };
    std::vector<StreamBinding> stream_bindings;
    stream_bindings.reserve(source_streams.size());
    for (const MeshAttributeStream* stream : source_streams) {
        if (stream == authored_normals)
            continue;
        add_output_attribute(output_attributes, stream->attribute);
        stream_bindings.push_back({.input_stream = stream, .output_attribute = stream->attribute});
    }

    ImporterMesh result;
    std::vector<int> subgeometry_mapping;
    initialize_result(input, result, subgeometry_mapping);
    exclude_holes_from_output(input, subgeometry_mapping);

    int fan_vertex_offsets[2] = {1, 2};
    if (is_left_handed(input.orientation))
        std::swap(fan_vertex_offsets[0], fan_vertex_offsets[1]);

    const TriangulationLayout layout
        = prepare_triangulation_layout(input, subgeometry_mapping, result.subgeometries.size());
    for (size_t subgeometry_index = 0; subgeometry_index < result.subgeometries.size(); ++subgeometry_index)
        result.subgeometries[subgeometry_index].indices.resize(layout.subgeometry_triangle_counts[subgeometry_index]);

    // Faces may interleave subgeometries, so each pre-sized index array needs its own write offset.
    std::vector<size_t> subgeometry_triangle_offsets(result.subgeometries.size(), 0);

    result.ensure_attributes(output_attributes);
    result.allocate_vertices(layout.vertex_count);

    // All mesh attributes and vertices are now final. These bindings remain stable throughout the fill phase.
    OutputBinding position_output = bind_output(result, position_attribute);
    OutputBinding normal_output = bind_output(result, normal_attribute);
    for (StreamBinding& binding : stream_bindings)
        binding.output = bind_output(result, binding.output_attribute);

    size_t next_output_vertex = 0;
    size_t face_vertex_offset = 0;
    const int face_count = int(input.face_vertex_counts.size());
    for (int face_index = 0; face_index < face_count;
         face_vertex_offset += input.face_vertex_counts[face_index], ++face_index) {
        if (subgeometry_mapping[face_index] < 0)
            continue;

        const int face_vertex_count = input.face_vertex_counts[face_index];
        const size_t subgeometry_index = size_t(subgeometry_mapping[face_index]);
        float3 flat_normal(0.f);
        if (!authored_normals)
            flat_normal = calculate_flat_normal(input, face_vertex_offset, face_vertex_count, fan_vertex_offsets);

        const int first_output_vertex = int(next_output_vertex);
        next_output_vertex += size_t(face_vertex_count);

        for (int local_face_vertex = 0; local_face_vertex < face_vertex_count; ++local_face_vertex) {
            const int output_vertex = first_output_vertex + local_face_vertex;
            const PrimvarElementIndices element_indices{
                .uniform = face_index,
                .vertex = input.face_vertex_indices[face_vertex_offset + local_face_vertex],
                .face_varying = int(face_vertex_offset) + local_face_vertex,
            };

            copy_stream_element(input.positions, element_indices.vertex, position_output.value(output_vertex));
            if (authored_normals) {
                copy_stream_element(
                    *authored_normals,
                    interpolation_element_index(authored_normals->interpolation, element_indices),
                    normal_output.value(output_vertex)
                );
            } else {
                normal_output.value<float3>(output_vertex) = flat_normal;
            }

            for (const StreamBinding& binding : stream_bindings) {
                copy_stream_element(
                    *binding.input_stream,
                    interpolation_element_index(binding.input_stream->interpolation, element_indices),
                    binding.output.value(output_vertex)
                );
            }
        }

        ImporterMesh::Subgeometry& subgeometry = result.subgeometries[subgeometry_index];
        size_t& triangle_offset = subgeometry_triangle_offsets[subgeometry_index];
        for (int triangle = 0; triangle < face_vertex_count - 2; ++triangle) {
            const int3 triangle_indices{
                first_output_vertex,
                first_output_vertex + triangle + fan_vertex_offsets[0],
                first_output_vertex + triangle + fan_vertex_offsets[1],
            };
            subgeometry.indices[triangle_offset++] = uint3(triangle_indices);
        }
    }

    FALCOR_ASSERT_EQ(face_vertex_offset, input.face_vertex_indices.size());
    FALCOR_ASSERT_EQ(next_output_vertex, layout.vertex_count);
    for (size_t subgeometry_index = 0; subgeometry_index < result.subgeometries.size(); ++subgeometry_index) {
        FALCOR_ASSERT_EQ(
            subgeometry_triangle_offsets[subgeometry_index],
            layout.subgeometry_triangle_counts[subgeometry_index]
        );
    }

    return result;
}

} // namespace detail
} // namespace mesh_tessellator
} // namespace falcor
