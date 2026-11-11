// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mesh_tessellator_internal.h"

#include "falcor2/importers/importer_utils.h"

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
    const MeshAttributeStream& positions,
    size_t face_vertex_offset,
    int face_vertex_count,
    const int (&fan_vertex_offsets)[2]
)
{
    const float3 v0 = stream_element<float3>(positions, input.face_vertex_indices[face_vertex_offset]);
    float3 normal(0.f);
    for (int triangle = 0; triangle < face_vertex_count - 2; ++triangle) {
        const float3 v1 = stream_element<float3>(
            positions,
            input.face_vertex_indices[face_vertex_offset + triangle + fan_vertex_offsets[0]]
        );
        const float3 v2 = stream_element<float3>(
            positions,
            input.face_vertex_indices[face_vertex_offset + triangle + fan_vertex_offsets[1]]
        );
        normal += sgl::math::cross(v1 - v0, v2 - v0);
    }
    return normalize_or(normal, float3(0, 0, 1));
}

} // namespace

ImporterMesh triangulate_mesh(const TessellatorInputMesh& input)
{
    const size_t vertex_count = input.positions[0].value_count();
    for (const MeshAttributeStream& positions : input.positions) {
        check_stream_layout(positions, input.face_vertex_counts.size(), vertex_count, input.face_vertex_indices.size());
        FALCOR_CHECK(positions.component_count() == 3, "Mesh tessellator positions must have three components");
        FALCOR_CHECK(positions.indices.empty(), "Mesh tessellator positions cannot be indexed separately");
    }
    for (const MeshAttributeStream& normals : input.normals) {
        check_stream_layout(normals, input.face_vertex_counts.size(), vertex_count, input.face_vertex_indices.size());
        FALCOR_CHECK(normals.component_count() == 3, "Mesh tessellator normals must have three components");
    }

    MeshAttributeStream default_uv;
    const std::vector<const MeshAttributeStream*> source_streams = collect_streams(input, default_uv);
    for (const MeshAttributeStream* stream : source_streams)
        check_stream_layout(*stream, input.face_vertex_counts.size(), vertex_count, input.face_vertex_indices.size());

    std::vector<ImporterMeshDefaultAttribute> output_attributes;
    output_attributes.reserve(source_streams.size() + 2 * input.positions.size());
    for (size_t sample = 0; sample < input.positions.size(); ++sample) {
        add_output_attribute(output_attributes, {ImporterSemantic::position, uint32_t(sample), 3});
        add_output_attribute(output_attributes, {ImporterSemantic::normal, uint32_t(sample), 3});
    }

    struct StreamBinding {
        const MeshAttributeStream* input_stream;
        ImporterMeshDefaultAttribute output_attribute;
        OutputBinding output;
    };
    std::vector<StreamBinding> stream_bindings;
    stream_bindings.reserve(source_streams.size());
    for (const MeshAttributeStream* stream : source_streams) {
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
    struct SampleOutput {
        ImporterVertexStream<float3> positions;
        ImporterVertexStream<float3> normals;
    };
    std::vector<SampleOutput> sample_outputs(input.positions.size());
    for (size_t sample = 0; sample < sample_outputs.size(); ++sample)
        sample_outputs[sample] = {result.position_stream(uint32_t(sample)), result.normal_stream(uint32_t(sample))};
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
        const int first_output_vertex = int(next_output_vertex);
        next_output_vertex += size_t(face_vertex_count);

        for (size_t sample = 0; sample < input.positions.size(); ++sample) {
            const MeshAttributeStream& positions = input.positions[sample];
            const MeshAttributeStream* normals = input.normals.empty() ? nullptr : &input.normals[sample];
            const SampleOutput& output = sample_outputs[sample];
            float3 flat_normal(0.f);
            if (!normals)
                flat_normal = calculate_flat_normal(
                    input,
                    positions,
                    face_vertex_offset,
                    face_vertex_count,
                    fan_vertex_offsets
                );
            for (int corner = 0; corner < face_vertex_count; ++corner) {
                const int output_vertex = first_output_vertex + corner;
                const PrimvarElementIndices element_indices{
                    .uniform = face_index,
                    .vertex = input.face_vertex_indices[face_vertex_offset + corner],
                    .face_varying = int(face_vertex_offset) + corner,
                };
                output.positions[output_vertex] = stream_element<float3>(positions, element_indices.vertex);
                output.normals[output_vertex] = normals
                    ? stream_element<float3>(
                          *normals,
                          interpolation_element_index(normals->interpolation, element_indices)
                      )
                    : flat_normal;
            }
        }

        for (int corner = 0; corner < face_vertex_count; ++corner) {
            const PrimvarElementIndices element_indices{
                .uniform = face_index,
                .vertex = input.face_vertex_indices[face_vertex_offset + corner],
                .face_varying = int(face_vertex_offset) + corner,
            };
            for (const StreamBinding& binding : stream_bindings) {
                copy_stream_element(
                    *binding.input_stream,
                    interpolation_element_index(binding.input_stream->interpolation, element_indices),
                    binding.output.value(first_output_vertex + corner)
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
