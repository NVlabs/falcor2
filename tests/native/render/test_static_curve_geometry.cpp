// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/geometry/static_curve_geometry.h"
#include "falcor2/render/scene.h"
#include <cstring>

using namespace falcor;

static shared::PackedLSSVertex make_vertex(float x, float y, float z, float r)
{
    shared::LSSVertex v = {};
    v.position = {x, y, z};
    v.radius = r;
    return shared::detail::pack_lss_vertex(v);
}

static shared::LSSVertex unpack(const shared::PackedLSSVertex& pv)
{
    return shared::detail::unpack_lss_vertex(pv);
}

TEST_SUITE_BEGIN("StaticCurveGeometry");

TEST_CASE_GPU("missing_primary_uvs_are_zero")
{
    auto scene = Scene::create(ref(ctx.device));
    auto& geometry = *scene->create_geometry<StaticCurveGeometry>();
    const std::vector<float3> positions = {{0, 0, 0}, {1, 0, 0}, {4, 0, 0}};
    const std::vector<float> radii(positions.size(), 0.1f);
    std::vector<uint32_t> indices = {0, 1, 1, 2};
    CurveIndexingMode mode = CurveIndexingMode::list;
    SUBCASE("list") { }
    SUBCASE("successive")
    {
        indices = {0, 1};
        mode = CurveIndexingMode::successive;
    }
    StaticCurveGeometryDataDesc desc = {};
    desc.vertex_count = uint32_t(positions.size());
    desc.position_stream = {positions.data(), sizeof(float3)};
    desc.radius_stream = {radii.data(), sizeof(float)};
    desc.index_count = uint32_t(indices.size());
    desc.index_stream = {indices.data(), sizeof(uint32_t)};
    desc.indexing_mode = mode;
    geometry.set_curve_data(desc);
    CHECK(geometry.vertices().size() == positions.size());
    REQUIRE(geometry.indices().size() == 2);
    CHECK(geometry.indices()[0] == 0);
    CHECK(geometry.indices()[1] == 1);
    for (const auto& vertex : geometry.vertices())
        CHECK(sgl::math::all(unpack(vertex).texcoord == float2(0.f)));
}

TEST_CASE_GPU("primary_uv_packing_and_wider_promotion")
{
    static_assert(sizeof(shared::PackedLSSVertex) == 24);
    auto scene = Scene::create(ref(ctx.device));
    auto& geometry = *scene->create_geometry<StaticCurveGeometry>();
    const float3 positions[] = {{0, 0, 0}, {1, 0, 0}};
    const float radii[] = {0.1f, 0.2f};
    const uint32_t indices[] = {1, 0};
    const float4 uvs[] = {{0.2f, 0.3f, 7, 8}, {0.7f, 0.9f, 9, 10}};
    GeometryVertexAttributeDesc uv = {
        .name = "st",
        .semantic = MeshAttributeSemantic::tex_coord,
        .type = sgl::DataType::float32,
        .component_count = 4,
        .data = uvs,
        .stride = sizeof(float4),
        .data_size = sizeof(uvs),
    };
    StaticCurveGeometryDataDesc desc = {};
    desc.vertex_count = 2;
    desc.position_stream = {positions, sizeof(float3)};
    desc.radius_stream = {radii, sizeof(float)};
    desc.index_count = 2;
    desc.index_stream = {indices, sizeof(uint32_t)};
    desc.vertex_attributes = std::span(&uv, 1);
    geometry.set_curve_data(desc);
    CHECK(unpack(geometry.vertices()[0]).texcoord.x == 0.7f);
    CHECK(unpack(geometry.vertices()[1]).texcoord.y == 0.3f);
    float4 copied;
    std::memcpy(&copied, geometry.vertex_attributes()[0].data.data(), sizeof(copied));
    CHECK(copied.z == 9.f);
    CHECK(copied.w == 10.f);
}

TEST_CASE("convert_list_to_successive_consecutive_chain")
{
    // A simple chain: segments (0,1), (1,2), (2,3)
    // All pairs are consecutive, so fast path should be taken.
    std::vector<shared::PackedLSSVertex> vertices = {
        make_vertex(0, 0, 0, 0.1f),
        make_vertex(1, 0, 0, 0.1f),
        make_vertex(2, 0, 0, 0.1f),
        make_vertex(3, 0, 0, 0.1f),
    };
    std::vector<uint32_t> indices = {0, 1, 1, 2, 2, 3};

    auto original_vertices = vertices;
    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    // Successive indices: [0, 1, 2]
    REQUIRE(indices.size() == 3);
    CHECK(indices[0] == 0);
    CHECK(indices[1] == 1);
    CHECK(indices[2] == 2);

    // Vertices should be unchanged (fast path).
    REQUIRE(vertices.size() == 4);
    CHECK(vertex_remap == (std::vector<uint32_t>{0, 1, 2, 3}));
}

TEST_CASE("convert_list_to_successive_disconnected_chains")
{
    // Two disconnected chains: (0,1), (1,2) and (5,6), (6,7)
    std::vector<shared::PackedLSSVertex> vertices;
    for (int i = 0; i < 8; ++i)
        vertices.push_back(make_vertex(static_cast<float>(i), 0, 0, 0.1f));

    std::vector<uint32_t> indices = {0, 1, 1, 2, 5, 6, 6, 7};

    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    // All pairs are consecutive, so fast path: [0, 1, 5, 6]
    REQUIRE(indices.size() == 4);
    CHECK(indices[0] == 0);
    CHECK(indices[1] == 1);
    CHECK(indices[2] == 5);
    CHECK(indices[3] == 6);

    // Vertices unchanged.
    REQUIRE(vertices.size() == 8);
    CHECK(vertex_remap == (std::vector<uint32_t>{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST_CASE("convert_list_to_successive_single_segment")
{
    std::vector<shared::PackedLSSVertex> vertices = {
        make_vertex(0, 0, 0, 0.1f),
        make_vertex(1, 0, 0, 0.1f),
    };
    std::vector<uint32_t> indices = {0, 1};

    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    REQUIRE(indices.size() == 1);
    CHECK(indices[0] == 0);
    REQUIRE(vertices.size() == 2);
    CHECK(vertex_remap == (std::vector<uint32_t>{0, 1}));
}

TEST_CASE("convert_list_to_successive_empty")
{
    std::vector<shared::PackedLSSVertex> vertices;
    std::vector<uint32_t> indices;

    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    CHECK(indices.empty());
    CHECK(vertices.empty());
    CHECK(vertex_remap.empty());
}

TEST_CASE("convert_list_to_successive_non_consecutive_pairs")
{
    // Pairs are not consecutive: (0,3), (1,4) - this triggers the slow path.
    std::vector<shared::PackedLSSVertex> vertices = {
        make_vertex(0, 0, 0, 0.1f), // 0
        make_vertex(1, 0, 0, 0.1f), // 1
        make_vertex(2, 0, 0, 0.1f), // 2
        make_vertex(3, 0, 0, 0.1f), // 3
        make_vertex(4, 0, 0, 0.1f), // 4
    };
    std::vector<uint32_t> indices = {0, 3, 1, 4};

    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    // After rearrangement, each segment's endpoints should be adjacent.
    REQUIRE(indices.size() == 2);

    // Segment 0: vertices[indices[0]] and vertices[indices[0]+1] should be original v0 and v3
    auto v0 = unpack(vertices[indices[0]]);
    auto v0_next = unpack(vertices[indices[0] + 1]);
    CHECK(v0.position.x == doctest::Approx(0.0f));
    CHECK(v0_next.position.x == doctest::Approx(3.0f));

    // Segment 1: vertices[indices[1]] and vertices[indices[1]+1] should be original v1 and v4
    auto v1 = unpack(vertices[indices[1]]);
    auto v1_next = unpack(vertices[indices[1] + 1]);
    CHECK(v1.position.x == doctest::Approx(1.0f));
    CHECK(v1_next.position.x == doctest::Approx(4.0f));
    CHECK(vertex_remap == (std::vector<uint32_t>{0, 3, 1, 4}));
}

TEST_CASE("convert_list_to_successive_shared_vertex_non_consecutive")
{
    // Reversed pair followed by forward pair: (1,0), (0,1)
    // First pair is non-consecutive (1 != 0+1=2), so slow path.
    std::vector<shared::PackedLSSVertex> vertices = {
        make_vertex(0, 0, 0, 0.1f), // 0
        make_vertex(1, 0, 0, 0.1f), // 1
    };
    std::vector<uint32_t> indices = {1, 0, 0, 1};

    const std::vector<uint32_t> vertex_remap
        = StaticCurveGeometry::convert_list_to_successive_indices(indices, vertices);

    REQUIRE(indices.size() == 2);

    // Segment 0: v1 -> v0
    auto s0_start = unpack(vertices[indices[0]]);
    auto s0_end = unpack(vertices[indices[0] + 1]);
    CHECK(s0_start.position.x == doctest::Approx(1.0f));
    CHECK(s0_end.position.x == doctest::Approx(0.0f));

    // Segment 1: v0 -> v1
    auto s1_start = unpack(vertices[indices[1]]);
    auto s1_end = unpack(vertices[indices[1] + 1]);
    CHECK(s1_start.position.x == doctest::Approx(0.0f));
    CHECK(s1_end.position.x == doctest::Approx(1.0f));
    CHECK(vertex_remap == (std::vector<uint32_t>{1, 0, 1}));
}

TEST_SUITE_END();
