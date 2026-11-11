// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/vertex_remap.h"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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


TEST_CASE_NIGHTLY("mesh motion streams share vertex operations")
{
    ImporterMesh sparse;
    CHECK_FALSE(sparse.position_attribute().valid());
    CHECK_FALSE(sparse.normal_attribute().valid());
    CHECK_FALSE(sparse.tangent_attribute().valid());
    sparse.ensure_attributes(
        {{ImporterSemantic::position, 2}, {ImporterSemantic::normal, 2}, {ImporterSemantic::tangent, 2}}
    );
    sparse.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tangent}});
    for (uint32_t sample : {0u, 2u}) {
        CHECK_EQ(sparse.position_attribute(sample).index, sample);
        CHECK_EQ(sparse.normal_attribute(sample).index, sample);
        CHECK_EQ(sparse.tangent_attribute(sample).index, sample);
    }
    CHECK_FALSE(sparse.position_attribute(1).valid());
    CHECK_FALSE(sparse.normal_attribute(1).valid());
    CHECK_FALSE(sparse.tangent_attribute(1).valid());
    CHECK_THROWS(sparse.position_attribute(3));
    CHECK_THROWS(sparse.normal_attribute(3));
    CHECK_THROWS(sparse.tangent_attribute(3));

    ImporterMesh mesh;
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::position, 1}, {ImporterSemantic::normal}});
    mesh.allocate_vertices(4, true);
    mesh.ensure_attributes({{ImporterSemantic::tex_coord, 1}});
    mesh.subgeometries.push_back({.indices = {uint3(0, 1, 2), uint3(3, 1, 2)}});
    const float3 values[] = {float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0), float3(0, 0, 0)};
    for (size_t i = 0; i < 4; ++i) {
        mesh.position_stream()[i] = values[i];
        mesh.position_stream(1)[i] = values[i];
        mesh.normal_stream()[i] = float3(0, 0, -1);
        mesh.find_stream<float2>(ImporterSemantic::tex_coord, 1)[i] = float2(0);
    }
    mesh.position_stream(1)[3].z = 1.f;
    auto opposite = mesh;
    opposite.subgeometries[0].indices = {uint3(0, 1, 2), uint3(3, 2, 1)};
    CHECK_EQ(opposite.collapse_exact_opposite_triangles(), 0);
    CHECK_EQ(opposite.subgeometries[0].indices.size(), 2);
    CHECK_THROWS(opposite.bake_pose({}));
    const ImporterMesh& view = mesh;
    CHECK_EQ(view.position_stream().data, view.position_stream(0).data);
    CHECK_EQ(view.position_stream(1)[3], float3(0, 0, 1));
    CHECK_THROWS(view.normal_stream(1));
    CHECK_FALSE(mesh.find_stream<float3>(ImporterSemantic::normal, 1).valid());
    auto duplicate = mesh.position_attribute(1);
    duplicate.name = "position_01";
    duplicate.buffer = uint32_t(mesh.buffers().size());
    CHECK_THROWS(mesh.add_attribute(duplicate));
    mesh.deduplicate_vertices();
    CHECK_EQ(mesh.vertex_count(), 4);
    mesh.position_stream(1)[3] = mesh.position_stream(1)[0];
    mesh.deduplicate_vertices();
    CHECK_EQ(mesh.vertex_count(), 3);
    CHECK_EQ(mesh.subgeometries[0].indices[0], mesh.subgeometries[0].indices[1]);
    CHECK(mesh.motion_times.empty());
    mesh.add_normals_from_faces();
    mesh.add_tangents_from_normals();
    for (uint32_t sample = 0; sample < 2; ++sample) {
        for (size_t vertex = 0; vertex < 3; ++vertex) {
            CAPTURE(sample);
            CAPTURE(vertex);
            CHECK_EQ(mesh.normal_stream(sample)[vertex], float3(0, 0, 1));
            const float3 tangent = mesh.tangent_stream(sample)[vertex];
            const float3 reference = mesh.tangent_stream()[vertex];
            // Frame reconstruction and normalization need not reproduce identical bits across architectures.
            for (int component = 0; component < 3; ++component) {
                CAPTURE(component);
                CHECK(std::isfinite(tangent[component]));
                CHECK_LE(std::abs(tangent[component] - reference[component]), 1e-5f);
            }
            CHECK_LE(std::abs(length(tangent) - 1.f), 1e-5f);
            CHECK_LE(std::abs(dot(mesh.normal_stream(sample)[vertex], tangent)), 1e-5f);
        }
    }
    CHECK_EQ(mesh.find_stream<float2>(ImporterSemantic::tex_coord, 1).size, 3);
}

TEST_CASE_NIGHTLY("opposite triangle collapse compares complete vertex motion")
{
    ImporterMesh mesh;
    mesh.ensure_attributes(
        {{ImporterSemantic::position}, {ImporterSemantic::position, 1}, {ImporterSemantic::position, 2}}
    );
    mesh.allocate_vertices(6);
    mesh.subgeometries.push_back({.indices = {uint3(0, 1, 2), uint3(3, 5, 4)}});
    const float3 positions[] = {float3(0, 0, 0), float3(1, 0, 0), float3(0, 1, 0)};
    for (uint32_t sample = 0; sample < 3; ++sample)
        for (size_t vertex = 0; vertex < 6; ++vertex)
            mesh.position_stream(sample)[vertex] = positions[vertex % 3] + float3(0, 0, float(sample));

    auto separated = mesh;
    separated.position_stream(1)[3].z += 1.f;
    CHECK_EQ(separated.collapse_exact_opposite_triangles(), 0);

    auto last_sample_separated = mesh;
    last_sample_separated.position_stream(2)[3].z += 1.f;
    CHECK_EQ(last_sample_separated.collapse_exact_opposite_triangles(), 0);

    // Reject one candidate while retaining another in a later subgeometry.
    auto filtered = separated;
    filtered.subgeometries.push_back({.indices = {uint3(0, 2, 1)}});
    CHECK_EQ(filtered.collapse_exact_opposite_triangles(), 1);
    REQUIRE_EQ(filtered.subgeometries.size(), 1);
    CHECK_EQ(filtered.subgeometries[0].indices.size(), 2);

    // The same triangle at both endpoints can still follow different paths between them.
    auto permuted = mesh;
    for (size_t corner = 0; corner < 3; ++corner)
        permuted.position_stream(1)[corner + 3] = mesh.position_stream(1)[(corner + 1) % 3];
    CHECK_EQ(permuted.collapse_exact_opposite_triangles(), 0);

    auto non_finite = mesh;
    non_finite.position_stream(1)[3].x = std::numeric_limits<float>::quiet_NaN();
    CHECK_EQ(non_finite.collapse_exact_opposite_triangles(), 0);
    non_finite.position_stream(1)[0].x = non_finite.position_stream(1)[3].x = std::numeric_limits<float>::infinity();
    CHECK_EQ(non_finite.collapse_exact_opposite_triangles(), 0);

    // Sample zero supplies candidates, so coincident corners there conservatively prevent collapse.
    auto coincident = mesh;
    coincident.position_stream()[1] = coincident.position_stream()[0];
    coincident.position_stream()[4] = coincident.position_stream()[3];
    CHECK_EQ(coincident.collapse_exact_opposite_triangles(), 0);

    mesh.position_stream(1)[3].x = -0.f;
    CHECK_EQ(mesh.collapse_exact_opposite_triangles(), 1);
    CHECK_EQ(mesh.subgeometries[0].indices.size(), 1);
    CHECK_EQ(mesh.subgeometries[0].indices[0], uint3(0, 1, 2));
    CHECK_EQ(mesh.vertex_count(), 6);
}

TEST_SUITE_END();
