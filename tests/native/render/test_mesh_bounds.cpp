// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/render/geometry.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/scene.h"

#include <cmath>
#include <limits>

using namespace falcor;

TEST_SUITE_BEGIN("MeshBounds");

TEST_CASE("MeshBounds merges vertex bounds in a shared coordinate system")
{
    MeshBounds bounds;
    CHECK_FALSE(bounds.aabb.is_valid());
    CHECK(bounds.radius == 0.f);
    bounds.expand(MeshBounds{});
    CHECK_FALSE(bounds.aabb.is_valid());

    bounds.expand(float3(3.f, 4.f, 0.f));
    CHECK(sgl::math::all(bounds.aabb.min == float3(3.f, 4.f, 0.f)));
    CHECK(sgl::math::all(bounds.aabb.max == float3(3.f, 4.f, 0.f)));
    CHECK(bounds.radius == 5.f);

    MeshBounds other;
    other.expand(float3(-6.f, 0.f, 8.f));
    other.expand(float3(1.f, -2.f, 3.f));
    bounds.expand(other);
    bounds.expand(MeshBounds{});
    CHECK(sgl::math::all(bounds.aabb.min == float3(-6.f, -2.f, 0.f)));
    CHECK(sgl::math::all(bounds.aabb.max == float3(3.f, 4.f, 8.f)));
    CHECK(bounds.radius == 10.f);
}

TEST_CASE("MeshBounds preserves large finite radii and rejects invalid points")
{
    MeshBounds bounds;
    bounds.expand(float3(1e20f, 1e20f, 0.f));
    CHECK(std::isfinite(bounds.radius));
    CHECK(bounds.radius == doctest::Approx(std::sqrt(2.f) * 1e20f));
    CHECK_THROWS_WITH(
        bounds.expand(float3(std::numeric_limits<float>::quiet_NaN(), 0.f, 0.f)),
        doctest::Contains("finite")
    );
    CHECK_THROWS_WITH(
        bounds.expand(float3(std::numeric_limits<float>::infinity(), 0.f, 0.f)),
        doctest::Contains("finite")
    );
    CHECK_THROWS_WITH(bounds.expand(float3(std::numeric_limits<float>::max())), doctest::Contains("finite"));
    CHECK(sgl::math::all(bounds.aabb.min == float3(1e20f, 1e20f, 0.f)));
    CHECK(sgl::math::all(bounds.aabb.max == float3(1e20f, 1e20f, 0.f)));
}

TEST_CASE_GPU("MeshBounds uses exact transformed vertices across all submeshes")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* geometry = scene->create_geometry<StaticMeshGeometry>();
    const float3 positions[] = {{0, 0, 0}, {2, 0, 0}, {0, 1, 0}, {-3, -2, 0}, {1, -2, 0}, {-3, 1, 0}};
    const uint32_t first_indices[] = {0, 1, 2};
    const uint32_t second_indices[] = {3, 4, 5};
    const MeshVertexAttributeDesc position_attribute = {
        .name = "POSITION",
        .semantic = MeshAttributeSemantic::position,
        .type = sgl::DataType::float32,
        .component_count = 3,
        .data = positions,
        .stride = sizeof(float3),
        .data_size = sizeof(positions),
    };
    StaticMeshGeometryDataDesc desc = {};
    desc.vertex_count = 6;
    desc.vertex_attributes = std::span(&position_attribute, 1);
    desc.sub_meshes = {
        {.name = "first", .index_count = 3, .index_stream = {first_indices, sizeof(uint32_t)}},
        {.name = "second", .index_count = 3, .index_stream = {second_indices, sizeof(uint32_t)}},
    };
    geometry->set_mesh_data(desc);

    auto transform = float4x4::identity();
    transform[0][1] = -1.f;
    transform[0][3] = 2.f;
    transform[1][0] = 1.f;
    transform[1][3] = -1.f;
    const MeshBounds bounds = geometry->calculate_bounds(transform);
    CHECK(sgl::math::all(bounds.aabb.min == float3(-2.f, -6.f, 0.f)));
    CHECK(sgl::math::all(bounds.aabb.max == float3(5.f, 1.f, 0.f)));
    CHECK(bounds.radius == doctest::Approx(std::sqrt(37.f)));
    CHECK(geometry->local_aabb().transform(transform).max.y > bounds.aabb.max.y);

    geometry->mutable_vertices(1)[0].position[0] = std::numeric_limits<float>::quiet_NaN();
    CHECK_THROWS_WITH(geometry->calculate_bounds(), doctest::Contains("finite"));
}

TEST_CASE_GPU("MeshBounds validates affine transforms even for empty geometry")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* geometry = scene->create_geometry<StaticMeshGeometry>();
    const MeshBounds bounds = geometry->calculate_bounds();
    CHECK_FALSE(bounds.aabb.is_valid());
    CHECK(bounds.radius == 0.f);

    auto transform = float4x4::identity();
    transform[0][0] = std::numeric_limits<float>::infinity();
    CHECK_THROWS_WITH(geometry->calculate_bounds(transform), doctest::Contains("finite"));
    transform = float4x4::identity();
    transform[3][0] = 0.1f;
    CHECK_THROWS_WITH(geometry->calculate_bounds(transform), doctest::Contains("affine"));
    transform = float4x4::identity();
    transform[3][3] = 2.f;
    CHECK_THROWS_WITH(geometry->calculate_bounds(transform), doctest::Contains("affine"));
}

TEST_SUITE_END();
