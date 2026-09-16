// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/vertex_remap.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("VertexDeduplication");

namespace {

ImporterMesh make_mesh_with_duplicate_vertices()
{
    ImporterMesh mesh;
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}});
    mesh.allocate_vertices(7, true);
    mesh.ensure_attributes({{ImporterSemantic::tex_coord}, {ImporterSemantic::color, 0, 3}});

    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    auto texcoords = mesh.texcoord_stream();
    auto colors = mesh.color_stream();

    const std::array<float3, 4> unique_positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 0.f),
    };
    const std::array<float2, 4> unique_texcoords = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(0.f, 1.f),
        float2(0.5f, 0.5f),
    };
    const std::array<float3, 4> unique_colors = {
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 1.f),
        float3(1.f, 1.f, 1.f),
    };
    constexpr std::array<uint32_t, 7> source_values = {0, 1, 2, 0, 2, 3, 3};

    for (size_t vertex_index = 0; vertex_index < source_values.size(); ++vertex_index) {
        const uint32_t source_value = source_values[vertex_index];
        positions[vertex_index] = unique_positions[source_value];
        normals[vertex_index] = float3(0.f, 0.f, 1.f);
        texcoords[vertex_index] = unique_texcoords[source_value];
        colors[vertex_index] = unique_colors[source_value];
    }

    ImporterMesh::Subgeometry subgeometry;
    subgeometry.indices = {uint3(0, 1, 2), uint3(3, 4, 5), uint3(6, 1, 2)};
    mesh.subgeometries.push_back(std::move(subgeometry));
    return mesh;
}

} // namespace

TEST_CASE("vertex deduplication remaps multiple strided streams")
{
    struct PaddedValue {
        uint32_t value;
        uint32_t ignored_padding;
    };

    const std::array<PaddedValue, 4> values = {{{7, 10}, {7, 20}, {7, 30}, {8, 40}}};
    const std::array<uint32_t, 4> categories = {1, 2, 1, 1};
    const std::array<VertexRemapStream, 2> streams = {
        VertexRemapStream{reinterpret_cast<const std::byte*>(values.data()), sizeof(uint32_t), sizeof(PaddedValue)},
        VertexRemapStream{reinterpret_cast<const std::byte*>(categories.data()), sizeof(uint32_t), sizeof(uint32_t)},
    };

    const VertexRemap remap = generate_vertex_remap(streams, values.size());
    CHECK_EQ(remap.old_to_new, std::vector<uint32_t>({0, 1, 0, 2}));
    CHECK_EQ(remap.new_to_old, std::vector<uint32_t>({0, 1, 3}));
    CHECK_EQ(remap.unique_vertex_count(), 3);
}

TEST_CASE("vertex deduplication hashes partial words")
{
    struct ThreeByteValue {
        uint8_t bytes[3];
    };
    static_assert(sizeof(ThreeByteValue) == 3);

    const std::array<ThreeByteValue, 5> values = {{{{1, 2, 3}}, {{1, 2, 3}}, {{1, 2, 4}}, {{0, 2, 3}}, {{1, 2, 4}}}};
    const VertexRemapStream stream{
        reinterpret_cast<const std::byte*>(values.data()),
        sizeof(ThreeByteValue),
        sizeof(ThreeByteValue),
    };

    const VertexRemap remap = generate_vertex_remap(std::span(&stream, 1), values.size());
    CHECK_EQ(remap.old_to_new, std::vector<uint32_t>({0, 0, 1, 2, 1}));
    CHECK_EQ(remap.new_to_old, std::vector<uint32_t>({0, 2, 3}));
}

TEST_CASE("vertex deduplication resolves hash table collisions")
{
    std::vector<uint64_t> values(100);
    for (size_t index = 0; index < values.size(); ++index)
        values[index] = index;

    const VertexRemapStream stream{
        reinterpret_cast<const std::byte*>(values.data()),
        sizeof(uint64_t),
        sizeof(uint64_t),
    };
    const VertexRemap remap = generate_vertex_remap(std::span(&stream, 1), values.size());
    CHECK(remap.is_identity());
    CHECK(remap.new_to_old.empty());
    REQUIRE_EQ(remap.old_to_new.size(), values.size());
    for (size_t index = 0; index < values.size(); ++index)
        CHECK_EQ(remap.old_to_new[index], index);
}

TEST_CASE("vertex deduplication compacts mesh streams and indices")
{
    ImporterMesh mesh = make_mesh_with_duplicate_vertices();
    mesh.deduplicate_vertices();
    mesh.deduplicate_vertices(); // The already-compact identity path must preserve the mesh.

    REQUIRE_EQ(mesh.vertex_count(), 4);
    const std::array<uint3, 3> expected_triangles = {uint3(0, 1, 2), uint3(0, 2, 3), uint3(3, 1, 2)};
    REQUIRE_EQ(mesh.subgeometries[0].indices.size(), expected_triangles.size());
    for (size_t triangle_index = 0; triangle_index < expected_triangles.size(); ++triangle_index) {
        for (int corner = 0; corner < 3; ++corner)
            CHECK_EQ(mesh.subgeometries[0].indices[triangle_index][corner], expected_triangles[triangle_index][corner]);
    }

    const std::array<float3, 4> expected_positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 0.f),
    };
    const std::array<float2, 4> expected_texcoords = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(0.f, 1.f),
        float2(0.5f, 0.5f),
    };
    const std::array<float3, 4> expected_colors = {
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 1.f),
        float3(1.f, 1.f, 1.f),
    };

    const auto positions = mesh.position_stream();
    const auto normals = mesh.normal_stream();
    const auto texcoords = mesh.texcoord_stream();
    const auto colors = mesh.color_stream();
    for (size_t vertex_index = 0; vertex_index < mesh.vertex_count(); ++vertex_index) {
        CHECK_EQ(positions[vertex_index], expected_positions[vertex_index]);
        CHECK_EQ(normals[vertex_index], float3(0.f, 0.f, 1.f));
        CHECK_EQ(texcoords[vertex_index], expected_texcoords[vertex_index]);
        CHECK_EQ(colors[vertex_index], expected_colors[vertex_index]);
    }
}

TEST_SUITE_END();
