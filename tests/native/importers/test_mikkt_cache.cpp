// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/core/blob_cache.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/mikkt.h"

#include <cstring>
#include <filesystem>
#include <fstream>
#include <span>
#include <string>
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

std::vector<uint8_t> make_tangent_blob(size_t vertex_count, const float3& tangent, float handedness)
{
    size_t tangent_bytes = vertex_count * sizeof(float3);
    std::vector<uint8_t> blob(tangent_bytes + vertex_count * sizeof(float));

    for (size_t i = 0; i < vertex_count; ++i)
        std::memcpy(blob.data() + i * sizeof(float3), &tangent, sizeof(float3));
    for (size_t i = 0; i < vertex_count; ++i)
        std::memcpy(blob.data() + tangent_bytes + i * sizeof(float), &handedness, sizeof(float));

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

} // namespace

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
    std::vector<uint8_t> cached_blob = make_tangent_blob(mesh.vertex_count(), cached_tangent, cached_handedness);
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
