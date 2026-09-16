// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/blob_cache.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/mikkt.h"
#include "falcor2/importers/mikkt_internal.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace falcor;

TEST_SUITE_BEGIN("MikkTCache");

namespace {

BlobCache::Options make_cache_options()
{
    BlobCache::Options opts;
    opts.root_dir = falcor::testing::get_case_temp_directory() / "mikkt_cache";
    opts.max_size = 1024 * 1024;
    return opts;
}

ImporterMesh make_textured_quad()
{
    ImporterMesh mesh;
    mesh.name = "quad";
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tex_coord}});
    mesh.allocate_vertices(4, true);

    auto positions = mesh.position_stream();
    positions[0] = float3(0.f, 0.f, 0.f);
    positions[1] = float3(1.f, 0.f, 0.f);
    positions[2] = float3(1.f, 1.f, 0.f);
    positions[3] = float3(0.f, 1.f, 0.f);

    auto normals = mesh.normal_stream();
    for (size_t i = 0; i < mesh.vertex_count(); ++i)
        normals[i] = float3(0.f, 0.f, 1.f);

    auto uvs = mesh.texcoord_stream();
    uvs[0] = float2(0.f, 0.f);
    uvs[1] = float2(1.f, 0.f);
    uvs[2] = float2(1.f, 1.f);
    uvs[3] = float2(0.f, 1.f);

    ImporterMesh::Subgeometry subgeometry;
    subgeometry.name = "quad";
    subgeometry.indices = {uint3(0, 1, 2), uint3(0, 2, 3)};
    mesh.subgeometries.push_back(std::move(subgeometry));

    return mesh;
}

ImporterMesh make_indexed_tangent_discontinuity()
{
    ImporterMesh mesh;
    mesh.name = "indexed_tangent_discontinuity";
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tex_coord},
        {ImporterSemantic::color, 0, 3},
    });
    mesh.allocate_vertices(5);

    auto positions = mesh.position_stream();
    positions[0] = float3(0.f, 0.f, 0.f);
    positions[1] = float3(1.f, 0.f, 0.f);
    positions[2] = float3(0.f, 1.f, 0.f);
    positions[3] = float3(-1.f, 0.f, 0.f);
    positions[4] = float3(0.f, -1.f, 0.f);

    auto normals = mesh.normal_stream();
    for (size_t i = 0; i < mesh.vertex_count(); ++i)
        normals[i] = float3(0.f, 0.f, 1.f);

    auto uvs = mesh.texcoord_stream();
    uvs[0] = float2(0.f, 0.f);
    uvs[1] = float2(1.f, 0.f);
    uvs[2] = float2(0.f, 1.f);
    uvs[3] = float2(1.f, 0.f);
    uvs[4] = float2(0.f, 1.f);

    auto colors = mesh.color_stream();
    for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex)
        colors[vertex] = float3(float(vertex), float(vertex + 10), float(vertex + 20));

    ImporterMesh::Subgeometry subgeometry;
    subgeometry.name = mesh.name;
    subgeometry.indices = {uint3(0, 1, 2), uint3(0, 3, 4)};
    mesh.subgeometries.push_back(std::move(subgeometry));

    return mesh;
}

ImporterMesh make_multi_subgeometry_tangent_discontinuity()
{
    ImporterMesh mesh = make_indexed_tangent_discontinuity();
    ImporterMesh::Subgeometry second;
    second.name = "second";
    second.indices.push_back(mesh.subgeometries[0].indices.back());
    mesh.subgeometries[0].indices.pop_back();
    mesh.subgeometries.push_back(std::move(second));
    return mesh;
}

ImporterMesh make_tangent_test_mesh(
    std::span<const float3> positions,
    std::span<const float3> normals,
    std::span<const float2> uvs,
    std::span<const uint3> faces
);

ImporterMesh make_handedness_only_discontinuity()
{
    const std::array positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(-1.f, 0.f, 0.f),
        float3(0.f, -1.f, 0.f),
    };
    const std::array normals = {
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
    };
    // Both triangles produce the same +X tangent at vertex 0, but the second UV parameterization is mirrored.
    const std::array uvs = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(0.f, 1.f),
        float2(-1.f, 0.f),
        float2(0.f, 1.f),
    };
    const std::array faces = {uint3(0, 1, 2), uint3(0, 3, 4)};
    return make_tangent_test_mesh(positions, normals, uvs, faces);
}

ImporterMesh make_non_manifold_edge_fan()
{
    const std::array positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(0.2f, 1.f, 0.f),
        float3(0.4f, -1.f, 0.f),
        float3(0.7f, 2.f, 0.f),
    };
    const std::array normals = {
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
    };
    const std::array uvs = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(0.1f, 1.f),
        float2(0.9f, 1.f),
        float2(0.2f, -1.f),
    };
    // Three faces share edge 0-1. The reversed faces have opposite UV orientations so Mikk's deterministic
    // non-manifold pairing affects which corner groups are connected.
    const std::array faces = {uint3(0, 1, 2), uint3(1, 0, 3), uint3(1, 0, 4)};
    return make_tangent_test_mesh(positions, normals, uvs, faces);
}

ImporterMesh make_textured_grid(uint32_t width, uint32_t height)
{
    ImporterMesh mesh;
    mesh.name = "textured_grid";
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tex_coord}});
    mesh.allocate_vertices(size_t(width + 1) * (height + 1));

    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    auto uvs = mesh.texcoord_stream();
    for (uint32_t y = 0; y <= height; ++y) {
        for (uint32_t x = 0; x <= width; ++x) {
            const size_t vertex = size_t(y) * (width + 1) + x;
            positions[vertex] = float3(float(x), float(y), 0.f);
            normals[vertex] = float3(0.f, 0.f, 1.f);
            uvs[vertex] = float2(float(x) / width, float(y) / height);
        }
    }

    ImporterMesh::Subgeometry subgeometry;
    subgeometry.name = mesh.name;
    subgeometry.indices.reserve(size_t(width) * height * 2);
    for (uint32_t y = 0; y < height; ++y) {
        for (uint32_t x = 0; x < width; ++x) {
            const uint32_t v00 = y * (width + 1) + x;
            const uint32_t v10 = v00 + 1;
            const uint32_t v01 = v00 + width + 1;
            const uint32_t v11 = v01 + 1;
            subgeometry.indices.push_back(uint3(v00, v10, v11));
            subgeometry.indices.push_back(uint3(v00, v11, v01));
        }
    }
    mesh.subgeometries.push_back(std::move(subgeometry));
    return mesh;
}

ImporterMesh make_expanded_textured_quad()
{
    ImporterMesh mesh;
    mesh.name = "expanded_quad";
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tex_coord}});
    mesh.allocate_vertices(6);
    mesh.ensure_attributes({{ImporterSemantic::color, 0, 3}});

    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    auto uvs = mesh.texcoord_stream();
    auto colors = mesh.color_stream();
    const std::array<float3, 4> unique_positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(1.f, 1.f, 0.f),
        float3(0.f, 1.f, 0.f),
    };
    const std::array<float2, 4> unique_uvs = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(1.f, 1.f),
        float2(0.f, 1.f),
    };
    const std::array<float3, 4> unique_colors = {
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 1.f),
        float3(1.f, 1.f, 1.f),
    };
    constexpr std::array<uint32_t, 6> source_values = {0, 1, 2, 0, 2, 3};
    for (size_t vertex_index = 0; vertex_index < source_values.size(); ++vertex_index) {
        const uint32_t source_value = source_values[vertex_index];
        positions[vertex_index] = unique_positions[source_value];
        normals[vertex_index] = float3(0.f, 0.f, 1.f);
        uvs[vertex_index] = unique_uvs[source_value];
        colors[vertex_index] = unique_colors[source_value];
    }

    ImporterMesh::Subgeometry subgeometry;
    subgeometry.name = mesh.name;
    subgeometry.indices = {uint3(0, 1, 2), uint3(3, 4, 5)};
    mesh.subgeometries.push_back(std::move(subgeometry));
    return mesh;
}

ImporterMesh make_tangent_test_mesh(
    std::span<const float3> positions,
    std::span<const float3> normals,
    std::span<const float2> uvs,
    std::span<const uint3> faces
)
{
    REQUIRE_EQ(normals.size(), positions.size());
    REQUIRE_EQ(uvs.size(), positions.size());

    ImporterMesh mesh;
    mesh.ensure_attributes({{ImporterSemantic::position}, {ImporterSemantic::normal}, {ImporterSemantic::tex_coord}});
    mesh.allocate_vertices(positions.size());
    for (size_t vertex = 0; vertex < positions.size(); ++vertex) {
        mesh.position_stream()[vertex] = positions[vertex];
        mesh.normal_stream()[vertex] = normals[vertex];
        mesh.texcoord_stream()[vertex] = uvs[vertex];
    }
    ImporterMesh::Subgeometry subgeometry;
    subgeometry.indices.assign(faces.begin(), faces.end());
    mesh.subgeometries.push_back(std::move(subgeometry));
    return mesh;
}

std::vector<float4> generate_reference_tangent_space(const ImporterMesh& mesh)
{
    return mikkt_detail::generate_corner_tangent_space(mesh, mikkt_detail::Generator::official_mikktspace);
}

void check_corner_tangent_space_matches(const ImporterMesh& mesh, std::span<const float4> expected)
{
    auto tangents = mesh.tangent_stream();
    auto handedness = mesh.handedness_stream();
    size_t face_index = 0;
    for (const auto& subgeometry : mesh.subgeometries) {
        for (const uint3& triangle : subgeometry.indices) {
            for (int corner = 0; corner < 3; ++corner) {
                const float4 expected_value = expected[face_index * 3 + corner];
                const size_t vertex_index = triangle[corner];
                CHECK_EQ(tangents[vertex_index].x, doctest::Approx(expected_value.x));
                CHECK_EQ(tangents[vertex_index].y, doctest::Approx(expected_value.y));
                CHECK_EQ(tangents[vertex_index].z, doctest::Approx(expected_value.z));
                CHECK_EQ(handedness[vertex_index], doctest::Approx(expected_value.w));
            }
            ++face_index;
        }
    }
}

std::filesystem::path find_cache_payload(const std::filesystem::path& root_dir)
{
    for (const auto& entry : std::filesystem::recursive_directory_iterator(root_dir)) {
        if (entry.is_regular_file()
            && entry.path().filename().string() == std::string(BlobCache::DEFAULT_SUB_DATA_NAME))
            return entry.path();
    }
    return {};
}

void write_bytes(const std::filesystem::path& path, std::span<const uint8_t> bytes)
{
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    REQUIRE(file.is_open());
    file.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(bytes.size()));
    file.close();
    REQUIRE(file.good());
}

size_t corner_count(const ImporterMesh& mesh)
{
    size_t result = 0;
    for (const auto& subgeometry : mesh.subgeometries)
        result += subgeometry.indices.size() * 3;
    return result;
}

std::vector<uint8_t> make_tangent_blob(size_t count, const float3& tangent, float handedness)
{
    std::vector<uint8_t> blob(count * sizeof(float4));
    const float4 tangent_space(tangent, handedness);

    for (size_t i = 0; i < count; ++i)
        std::memcpy(blob.data() + i * sizeof(float4), &tangent_space, sizeof(float4));

    return blob;
}

void check_tangent_space_matches(const ImporterMesh& expected, const ImporterMesh& actual)
{
    REQUIRE_EQ(expected.vertex_count(), actual.vertex_count());

    auto expected_tangents = expected.tangent_stream();
    auto actual_tangents = actual.tangent_stream();
    auto expected_handedness = expected.handedness_stream();
    auto actual_handedness = actual.handedness_stream();

    REQUIRE(expected_tangents.valid());
    REQUIRE(actual_tangents.valid());
    REQUIRE(expected_handedness.valid());
    REQUIRE(actual_handedness.valid());

    for (size_t i = 0; i < expected.vertex_count(); ++i) {
        CHECK_EQ(actual_tangents[i].x, doctest::Approx(expected_tangents[i].x));
        CHECK_EQ(actual_tangents[i].y, doctest::Approx(expected_tangents[i].y));
        CHECK_EQ(actual_tangents[i].z, doctest::Approx(expected_tangents[i].z));
        CHECK_EQ(actual_handedness[i], doctest::Approx(expected_handedness[i]));
    }
}

void check_parallel_matches_official(const ImporterMesh& mesh)
{
    const std::vector<float4> official
        = mikkt_detail::generate_corner_tangent_space(mesh, mikkt_detail::Generator::official_mikktspace);
    const std::vector<float4> parallel
        = mikkt_detail::generate_corner_tangent_space(mesh, mikkt_detail::Generator::parallel_mikktspace);
    REQUIRE_EQ(parallel.size(), official.size());
    float max_component_error = 0.f;
    size_t handedness_mismatches = 0;
    size_t nonfinite_values = 0;
    for (size_t corner = 0; corner < official.size(); ++corner) {
        const bool official_is_finite = std::isfinite(official[corner].x) && std::isfinite(official[corner].y)
            && std::isfinite(official[corner].z) && std::isfinite(official[corner].w);
        const bool parallel_is_finite = std::isfinite(parallel[corner].x) && std::isfinite(parallel[corner].y)
            && std::isfinite(parallel[corner].z) && std::isfinite(parallel[corner].w);
        if (!official_is_finite || !parallel_is_finite) {
            ++nonfinite_values;
            continue;
        }
        max_component_error = std::max(
            max_component_error,
            std::max(
                {std::abs(parallel[corner].x - official[corner].x),
                 std::abs(parallel[corner].y - official[corner].y),
                 std::abs(parallel[corner].z - official[corner].z)}
            )
        );
        handedness_mismatches += parallel[corner].w != official[corner].w;
    }
    CHECK_EQ(nonfinite_values, 0);
    CHECK_LE(max_component_error, 1e-5f);
    CHECK_EQ(handedness_mismatches, 0);
}

void check_parallel_assembly_matches_official(const ImporterMesh& source)
{
    ImporterMesh official = source;
    ImporterMesh parallel = source;
    mikkt_detail::generate_tangent_space(official, mikkt_detail::Generator::official_mikktspace);
    mikkt_detail::generate_tangent_space(parallel, mikkt_detail::Generator::parallel_mikktspace);

    REQUIRE_EQ(parallel.vertex_count(), official.vertex_count());
    REQUIRE_EQ(parallel.subgeometries.size(), official.subgeometries.size());
    for (size_t subgeometry = 0; subgeometry < official.subgeometries.size(); ++subgeometry)
        CHECK_EQ(parallel.subgeometries[subgeometry].indices, official.subgeometries[subgeometry].indices);

    const auto official_positions = official.position_stream();
    const auto parallel_positions = parallel.position_stream();
    const auto official_normals = official.normal_stream();
    const auto parallel_normals = parallel.normal_stream();
    const auto official_uvs = official.texcoord_stream();
    const auto parallel_uvs = parallel.texcoord_stream();
    const auto official_colors = official.color_stream();
    const auto parallel_colors = parallel.color_stream();
    const auto official_tangents = official.tangent_stream();
    const auto parallel_tangents = parallel.tangent_stream();
    const auto official_handedness = official.handedness_stream();
    const auto parallel_handedness = parallel.handedness_stream();
    for (size_t vertex = 0; vertex < official.vertex_count(); ++vertex) {
        CHECK_EQ(parallel_positions[vertex], official_positions[vertex]);
        CHECK_EQ(parallel_normals[vertex], official_normals[vertex]);
        CHECK_EQ(parallel_uvs[vertex], official_uvs[vertex]);
        if (official_colors.valid())
            CHECK_EQ(parallel_colors[vertex], official_colors[vertex]);
        CHECK_LE(std::abs(parallel_tangents[vertex].x - official_tangents[vertex].x), 1e-5f);
        CHECK_LE(std::abs(parallel_tangents[vertex].y - official_tangents[vertex].y), 1e-5f);
        CHECK_LE(std::abs(parallel_tangents[vertex].z - official_tangents[vertex].z), 1e-5f);
        CHECK_EQ(parallel_handedness[vertex], official_handedness[vertex]);
    }
}

template<typename Mutate>
void check_cache_input_invalidation(std::string_view name, Mutate&& mutate)
{
    BlobCache::Options options = make_cache_options();
    options.root_dir = *options.root_dir / name;
    BlobCache cache(options);

    ImporterMesh source = make_textured_quad();
    mikkt_generate_tangent_space(source, cache);
    const std::filesystem::path payload_path = find_cache_payload(*options.root_dir);
    REQUIRE_FALSE(payload_path.empty());
    write_bytes(payload_path, make_tangent_blob(corner_count(source), float3(2.f, 3.f, 4.f), -1.f));

    ImporterMesh modified = make_textured_quad();
    mutate(modified);
    const std::vector<float4> reference = generate_reference_tangent_space(modified);
    mikkt_generate_tangent_space(modified, cache);
    check_corner_tangent_space_matches(modified, reference);
}

} // namespace

TEST_CASE("Parallel MikkTSpace is numerically equivalent on regular meshes")
{
    for (const ImporterMesh& mesh :
         {make_textured_quad(), make_indexed_tangent_discontinuity(), make_expanded_textured_quad()})
        check_parallel_matches_official(mesh);
}

TEST_CASE("Parallel MikkTSpace assembled output matches official MikkTSpace")
{
    check_parallel_assembly_matches_official(make_indexed_tangent_discontinuity());
    check_parallel_assembly_matches_official(make_multi_subgeometry_tangent_discontinuity());
}

TEST_CASE("MikkTSpace assembly splits handedness-only discontinuities")
{
    ImporterMesh mesh = make_handedness_only_discontinuity();
    const std::vector<float4> reference = generate_reference_tangent_space(mesh);
    REQUIRE_EQ(reference[0].xyz(), reference[3].xyz());
    REQUIRE_NE(reference[0].w, reference[3].w);

    check_parallel_assembly_matches_official(mesh);
    mikkt_generate_tangent_space(mesh);

    REQUIRE_EQ(mesh.vertex_count(), 6);
    REQUIRE_NE(mesh.subgeometries[0].indices[0].x, mesh.subgeometries[0].indices[1].x);
    check_corner_tangent_space_matches(mesh, reference);
}

TEST_CASE("Parallel MikkTSpace preserves non-manifold edge pairing")
{
    check_parallel_assembly_matches_official(make_non_manifold_edge_fan());
}

TEST_CASE("Parallel MikkTSpace executes inner parallel paths")
{
    // 2,880 triangles produce 8,640 corners, enough to schedule multiple jobs without making normal tests scene-sized.
    check_parallel_assembly_matches_official(make_textured_grid(40, 36));
}

TEST_CASE("Parallel MikkTSpace executes partitioned edge sorting")
{
    // 22,000 triangles produce 66,000 directed edges, just above the production sort partition threshold.
    ImporterMesh mesh = make_textured_grid(110, 100);
    auto positions = mesh.position_stream();
    for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex)
        positions[vertex].y += 0.001f * positions[vertex].x * positions[vertex].y;
    check_parallel_assembly_matches_official(mesh);
}

TEST_CASE("Parallel MikkTSpace preserves degenerate and zero tangent behavior")
{
    SUBCASE("valid projected tangent is zero")
    {
        const std::array positions = {float3(0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f)};
        const std::array normals = {float3(1.f, 0.f, 0.f), float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f)};
        const std::array uvs = {float2(0.f), float2(1.f, 0.f), float2(0.f, 1.f)};
        const std::array faces = {uint3(0, 1, 2)};
        check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, faces));
    }
    SUBCASE("collapsed position extent")
    {
        const std::array positions = {float3(0.f), float3(0.f), float3(0.f)};
        const std::array normals = {float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f)};
        const std::array uvs = {float2(0.f), float2(1.f, 0.f), float2(0.f, 1.f)};
        const std::array faces = {uint3(0, 1, 2)};
        check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, faces));
    }
    SUBCASE("position-degenerate face copies from a welded healthy corner")
    {
        const std::array positions = {
            float3(0.f, 0.f, 0.f),
            float3(1.f, 0.f, 0.f),
            float3(0.f, 1.f, 0.f),
            float3(0.f, 0.f, 0.f),
        };
        const std::array normals = {
            float3(0.f, 0.f, 1.f),
            float3(0.f, 0.f, 1.f),
            float3(0.f, 0.f, 1.f),
            float3(0.f, 0.f, 1.f),
        };
        const std::array uvs = {
            float2(0.f, 0.f),
            float2(1.f, 0.f),
            float2(0.f, 1.f),
            float2(0.f, 0.f),
        };
        const std::array faces = {uint3(0, 1, 2), uint3(3, 0, 1)};
        check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, faces));
    }
    SUBCASE("extreme finite UVs preserve GROUP_WITH_ANY behavior")
    {
        const std::array positions = {float3(0.f), float3(1.f, 0.f, 0.f), float3(0.f, 1.f, 0.f)};
        const std::array normals = {float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f)};
        const std::array uvs = {float2(0.f), float2(1e20f, 0.f), float2(0.f, 1e20f)};
        const std::array faces = {uint3(0, 1, 2)};
        check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, faces));
    }
}

TEST_CASE("Parallel MikkTSpace rejects unsupported topology domains")
{
    SUBCASE("empty mesh")
    {
        ImporterMesh mesh = make_textured_quad();
        mesh.subgeometries.front().indices.clear();
        CHECK_THROWS_WITH(
            mikkt_detail::generate_corner_tangent_space(mesh, mikkt_detail::Generator::parallel_mikktspace),
            doctest::Contains("empty mesh")
        );
    }
    SUBCASE("position extent overflows float")
    {
        constexpr float MAX_FLOAT = std::numeric_limits<float>::max();
        const std::array positions = {
            float3(-MAX_FLOAT, 0.f, 0.f),
            float3(MAX_FLOAT, 0.f, 0.f),
            float3(0.f, 1.f, 0.f),
        };
        const std::array normals = {float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f), float3(0.f, 0.f, 1.f)};
        const std::array uvs = {float2(0.f), float2(1.f, 0.f), float2(0.f, 1.f)};
        const std::array faces = {uint3(0, 1, 2)};
        const ImporterMesh mesh = make_tangent_test_mesh(positions, normals, uvs, faces);
        CHECK_THROWS_WITH(
            mikkt_detail::generate_corner_tangent_space(mesh, mikkt_detail::Generator::parallel_mikktspace),
            doctest::Contains("position extent")
        );
    }
}

TEST_CASE("Parallel MikkTSpace preserves UV-degenerate traversal order")
{
    const std::array positions = {
        float3(0.f, 0.f, 0.f),
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, -1.f, 0.f),
        float3(1.f, -1.f, 0.f),
    };
    const std::array normals = {
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
        float3(0.f, 0.f, 1.f),
    };
    const std::array uvs = {
        float2(0.f, 0.f),
        float2(1.f, 0.f),
        float2(0.f, 1.f),
        float2(2.f, 0.f),
        float2(2.f, 1.f),
    };
    // The first face is UV-degenerate and connects contributing faces with opposite UV orientations.
    const std::array faces = {uint3(1, 0, 3), uint3(0, 1, 2), uint3(3, 0, 4)};
    check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, faces));
    const std::array reversed_faces = {faces[2], faces[1], faces[0]};
    check_parallel_matches_official(make_tangent_test_mesh(positions, normals, uvs, reversed_faces));
}

TEST_CASE("cache hit loads cached tangent data")
{
    BlobCache::Options opts = make_cache_options();
    BlobCache cache(opts);

    ImporterMesh mesh = make_textured_quad();
    mikkt_generate_tangent_space(mesh, cache);

    std::filesystem::path payload_path = find_cache_payload(*opts.root_dir);
    REQUIRE_FALSE(payload_path.empty());

    const float3 cached_tangent(2.f, 3.f, 4.f);
    const float cached_handedness = -1.f;
    std::vector<uint8_t> cached_blob = make_tangent_blob(corner_count(mesh), cached_tangent, cached_handedness);
    write_bytes(payload_path, cached_blob);

    ImporterMesh cached_mesh = make_textured_quad();
    mikkt_generate_tangent_space(cached_mesh, cache);

    auto tangents = cached_mesh.tangent_stream();
    auto handedness = cached_mesh.handedness_stream();
    REQUIRE(tangents.valid());
    REQUIRE(handedness.valid());

    for (size_t i = 0; i < cached_mesh.vertex_count(); ++i) {
        CHECK_EQ(tangents[i].x, doctest::Approx(cached_tangent.x));
        CHECK_EQ(tangents[i].y, doctest::Approx(cached_tangent.y));
        CHECK_EQ(tangents[i].z, doctest::Approx(cached_tangent.z));
        CHECK_EQ(handedness[i], doctest::Approx(cached_handedness));
    }
}

TEST_CASE("cache size mismatch recomputes tangent data")
{
    BlobCache::Options opts = make_cache_options();
    BlobCache cache(opts);

    ImporterMesh expected_mesh = make_textured_quad();
    mikkt_generate_tangent_space(expected_mesh, cache);

    std::filesystem::path payload_path = find_cache_payload(*opts.root_dir);
    REQUIRE_FALSE(payload_path.empty());

    std::vector<uint8_t> invalid_blob(1, 0xff);
    write_bytes(payload_path, invalid_blob);

    ImporterMesh actual_mesh = make_textured_quad();
    mikkt_generate_tangent_space(actual_mesh, cache);

    check_tangent_space_matches(expected_mesh, actual_mesh);
}

TEST_CASE("cache key covers every MikkTSpace input")
{
    check_cache_input_invalidation(
        "position",
        [](ImporterMesh& mesh)
        {
            mesh.position_stream()[1] = float3(2.f, 0.5f, 0.f);
        }
    );
    check_cache_input_invalidation(
        "normal",
        [](ImporterMesh& mesh)
        {
            mesh.normal_stream()[0] = float3(0.70710678f, 0.f, 0.70710678f);
        }
    );
    check_cache_input_invalidation(
        "uv",
        [](ImporterMesh& mesh)
        {
            mesh.texcoord_stream()[1] = float2(2.f, 0.5f);
        }
    );
    check_cache_input_invalidation(
        "indices",
        [](ImporterMesh& mesh)
        {
            std::swap(mesh.subgeometries[0].indices[0], mesh.subgeometries[0].indices[1]);
        }
    );
}

TEST_CASE("indexed tangent variants are preserved")
{
    ImporterMesh mesh = make_indexed_tangent_discontinuity();
    const std::vector<float4> reference = generate_reference_tangent_space(mesh);

    REQUIRE_NE(reference[0].xyz(), reference[3].xyz());
    mikkt_generate_tangent_space(mesh);

    REQUIRE_EQ(mesh.vertex_count(), 6);
    check_corner_tangent_space_matches(mesh, reference);
    CHECK_EQ(mesh.color_stream()[5], mesh.color_stream()[0]);
}

TEST_CASE("multiple subgeometries preserve corner ordering")
{
    ImporterMesh mesh = make_multi_subgeometry_tangent_discontinuity();

    check_parallel_matches_official(mesh);
    const std::vector<float4> reference = generate_reference_tangent_space(mesh);
    mikkt_generate_tangent_space(mesh);

    REQUIRE_EQ(mesh.vertex_count(), 6);
    REQUIRE_EQ(mesh.subgeometries[0].indices.size(), 1);
    REQUIRE_EQ(mesh.subgeometries[1].indices.size(), 1);
    check_corner_tangent_space_matches(mesh, reference);
}

TEST_CASE("cache preserves indexed tangent variants")
{
    BlobCache::Options opts = make_cache_options();
    BlobCache cache(opts);

    ImporterMesh cold_mesh = make_indexed_tangent_discontinuity();
    const std::vector<float4> reference = generate_reference_tangent_space(cold_mesh);
    mikkt_generate_tangent_space(cold_mesh, cache);
    REQUIRE_EQ(cold_mesh.vertex_count(), 6);
    check_corner_tangent_space_matches(cold_mesh, reference);

    ImporterMesh cached_mesh = make_indexed_tangent_discontinuity();
    mikkt_generate_tangent_space(cached_mesh, cache);
    REQUIRE_EQ(cached_mesh.vertex_count(), 6);
    check_corner_tangent_space_matches(cached_mesh, reference);
}

TEST_CASE("tangent generation only splits authored vertices")
{
    ImporterMesh mesh = make_expanded_textured_quad();
    const std::vector<float4> reference = generate_reference_tangent_space(mesh);
    const std::vector<uint3> source_indices = mesh.subgeometries[0].indices;
    mikkt_generate_tangent_space(mesh);

    REQUIRE_EQ(mesh.vertex_count(), 6);
    CHECK_EQ(mesh.subgeometries[0].indices, source_indices);
    check_corner_tangent_space_matches(mesh, reference);
}

TEST_CASE("explicit deduplication compacts before tangent generation")
{
    ImporterMesh mesh = make_expanded_textured_quad();
    const std::vector<float4> reference = generate_reference_tangent_space(mesh);

    mesh.deduplicate_vertices();
    mikkt_generate_tangent_space(mesh);

    REQUIRE_EQ(mesh.vertex_count(), 4);
    const std::array<uint3, 2> expected_indices = {uint3(0, 1, 2), uint3(0, 2, 3)};
    CHECK_EQ(mesh.subgeometries[0].indices, std::vector<uint3>(expected_indices.begin(), expected_indices.end()));
    check_corner_tangent_space_matches(mesh, reference);
    const std::array<float3, 4> expected_colors = {
        float3(1.f, 0.f, 0.f),
        float3(0.f, 1.f, 0.f),
        float3(0.f, 0.f, 1.f),
        float3(1.f, 1.f, 1.f),
    };
    const auto colors = mesh.color_stream();
    for (size_t vertex_index = 0; vertex_index < mesh.vertex_count(); ++vertex_index)
        CHECK_EQ(colors[vertex_index], expected_colors[vertex_index]);
}

TEST_CASE("importer cache accessor retains a safe reference")
{
    ref<BlobCache> previous_cache = importer_cache();
    BlobCache::Options opts = make_cache_options();
    ref<BlobCache> cache = make_ref<BlobCache>(opts);
    set_importer_cache(cache);

    cache = nullptr;
    ref<BlobCache> retained_cache = importer_cache();
    CHECK(retained_cache);

    set_importer_cache(nullptr);
    CHECK_FALSE(importer_cache());
    set_importer_cache(previous_cache);
}

TEST_SUITE_END();
