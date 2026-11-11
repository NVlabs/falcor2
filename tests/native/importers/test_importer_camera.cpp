// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/render/transform.h"

#include <array>
#include <cmath>
#include <limits>

using namespace falcor;

namespace {

float4x4 make_transform(float3 translation, float3 scale = float3(1.f))
{
    Transform transform;
    transform.set_translation(translation);
    transform.set_scale(scale);
    return transform.matrix();
}

void check_camera_fit(const ImporterScene& scene, int camera_index, std::span<const AABB> bounds, float aspect)
{
    const auto& camera = scene.cameras[camera_index];
    const auto view = sgl::math::inverse(scene.nodes[scene.root_nodes.back()].transform);
    const float tan_y = std::tan(sgl::math::radians(camera.vertical_fov_degrees()) * 0.5f);
    for (const auto& aabb : bounds) {
        for (uint32_t corner = 0; corner < 8; ++corner) {
            const float3 point(
                corner & 1 ? aabb.max.x : aabb.min.x,
                corner & 2 ? aabb.max.y : aabb.min.y,
                corner & 4 ? aabb.max.z : aabb.min.z
            );
            const auto p = sgl::math::mul(view, float4(point, 1.f));
            const float depth = -p.z;
            CHECK_GT(depth, camera.depth_range.x);
            CHECK_LT(depth, camera.depth_range.y);
            CHECK_LE(std::abs(p.x) / (depth * tan_y * aspect), 0.901f);
            CHECK_LE(std::abs(p.y) / (depth * tan_y), 0.901f);
        }
    }
}

} // namespace

TEST_CASE("Importer camera - manual addition preserves existing cameras and handles empty bounds")
{
    ImporterScene scene;
    CHECK_FALSE(scene.has_camera());
    CHECK_EQ(scene.add_framed_camera(40.f, 2.f, "First"), 0);
    CHECK(scene.has_camera());
    CHECK_EQ(scene.add_framed_camera(), 1);
    REQUIRE_EQ(scene.cameras.size(), 2);
    REQUIRE_EQ(scene.nodes.size(), 2);
    CHECK_EQ(scene.root_nodes, (std::vector<int>{0, 1}));
    CHECK_EQ(scene.cameras[0].name, "First");
    CHECK_EQ(scene.cameras[0].vertical_fov_degrees(), doctest::Approx(40.f));
    CHECK_EQ(scene.cameras[0].projection, ImporterCamera::Projection::perspective);
    CHECK_FALSE(scene.cameras[0].enable_depth_of_field);
    CHECK_EQ(scene.cameras[0].focus_distance, 5.f);
    CHECK_EQ(scene.cameras[0].depth_range, float2(0.01f, 1000.f));
    CHECK_EQ(scene.nodes[0].name, "First");
    CHECK_EQ(scene.nodes[0].parent, -1);
    CHECK_EQ(scene.nodes[0].camera_index, 0);
    CHECK_EQ(scene.nodes[0].transform.get_col(3).xyz(), float3(0.f, 0.f, 5.f));
    CHECK_EQ(scene.cameras[1].name, "FramedCamera");
    CHECK_EQ(scene.cameras[1].vertical_fov_degrees(), doctest::Approx(50.f));
}

TEST_CASE("Importer camera - invalid framing arguments do not append objects")
{
    ImporterScene scene;
    for (float fov : {0.f, 180.f, -1.f, std::numeric_limits<float>::quiet_NaN()})
        CHECK_THROWS(scene.add_framed_camera(fov));
    for (float aspect : {0.f, -1.f, std::numeric_limits<float>::infinity(), std::numeric_limits<float>::quiet_NaN()})
        CHECK_THROWS(scene.add_framed_camera(50.f, aspect));
    CHECK(scene.cameras.empty());
    CHECK(scene.nodes.empty());
    CHECK(scene.root_nodes.empty());
}

TEST_CASE("Importer camera - camera detection follows reachable nodes and nested prototypes")
{
    ImporterScene scene;
    scene.cameras.resize(1);
    scene.nodes.resize(5);
    scene.nodes[4].camera_index = 0;
    CHECK_FALSE(scene.has_camera());
    scene.root_nodes = {0};
    scene.nodes[0].children = {1};
    scene.nodes[1].prototype_index = 0;
    scene.prototypes.resize(2);
    scene.prototypes[0].root_node = 2;
    scene.nodes[2].prototype_index = 1;
    scene.prototypes[1].root_node = 3;
    CHECK_FALSE(scene.has_camera());
    scene.nodes[3].camera_index = 10;
    CHECK_FALSE(scene.has_camera());
    scene.nodes[3].camera_index = 0;
    CHECK(scene.has_camera());
    scene.nodes[3].camera_index = -1;
    scene.nodes[3].children = {0};
    CHECK_THROWS_WITH(scene.has_camera(), doctest::Contains("Cycle in importer scene node hierarchy."));
    CHECK_THROWS_WITH(scene.add_framed_camera(), doctest::Contains("Cycle in importer scene node hierarchy."));
    CHECK_EQ(scene.cameras.size(), 1);
    CHECK_EQ(scene.nodes.size(), 5);
}

TEST_CASE("Importer camera - framing includes transformed mesh and curve prototype instances")
{
    ImporterScene scene;
    scene.meshes.resize(1);
    auto& mesh = scene.meshes[0];
    mesh.ensure_attributes({{ImporterSemantic::position, 0}});
    mesh.allocate_vertices(2);
    mesh.position_stream()[0] = float3(-1.f, -2.f, -3.f);
    mesh.position_stream()[1] = float3(1.f, 2.f, 3.f);
    mesh.local_aabb = AABB(float3(10000.f), float3(10001.f));

    scene.curves.resize(1);
    auto& curve = scene.curves[0];
    curve.resize_vertices(2);
    curve.position_stream()[0] = float3(0.f);
    curve.position_stream()[1] = float3(2.f, 1.f, 0.f);
    curve.radius_stream()[0] = 0.5f;
    curve.radius_stream()[1] = 1.f;

    scene.nodes.resize(8);
    scene.root_nodes = {0};
    scene.nodes[0].transform = make_transform(float3(20.f, 5.f, 0.f));
    scene.nodes[0].children = {1, 2};
    scene.nodes[1].prototype_index = 0;
    scene.nodes[1].transform = make_transform(float3(-10.f, 0.f, 0.f));
    scene.nodes[2].prototype_index = 0;
    scene.nodes[2].transform = make_transform(float3(30.f, 0.f, 0.f));
    scene.prototypes.resize(2);
    scene.prototypes[0].root_node = 3;
    scene.nodes[3].transform = make_transform(float3(0.f, 3.f, 0.f));
    scene.nodes[3].children = {4, 5};
    scene.nodes[4].mesh_index = 0;
    scene.nodes[4].transform = make_transform(float3(0.f), float3(-2.f, 1.f, 3.f));
    scene.nodes[5].prototype_index = 1;
    scene.nodes[5].transform = make_transform(float3(0.f, 0.f, 12.f));
    scene.prototypes[1].root_node = 6;
    scene.nodes[6].curve_index = 0;
    scene.nodes[6].transform = make_transform(float3(0.f), float3(3.f, 2.f, 1.f));
    // A disconnected instance must not pull the view away from the actual scene.
    scene.nodes[7].mesh_index = 0;
    scene.nodes[7].transform = make_transform(float3(10000.f));

    const std::array<AABB, 4> bounds{
        AABB(float3(8.f, 6.f, -9.f), float3(12.f, 10.f, 9.f)),
        AABB(float3(8.5f, 7.f, 11.f), float3(19.f, 12.f, 13.f)),
        AABB(float3(48.f, 6.f, -9.f), float3(52.f, 10.f, 9.f)),
        AABB(float3(48.5f, 7.f, 11.f), float3(59.f, 12.f, 13.f)),
    };
    for (float aspect : {0.25f, 2.f, 4.f}) {
        const int index = scene.add_framed_camera(45.f, aspect);
        check_camera_fit(scene, index, bounds, aspect);
        CHECK_LT(scene.cameras[index].focus_distance, 500.f);
    }
}

TEST_CASE("Importer camera - explicit fallback only changes the completed scene")
{
    auto importer = Importer::create();
    importer->cameras().create_fov("Unused");
    auto unchanged = importer->build_importer_scene();
    CHECK_EQ(unchanged->cameras.size(), 1);
    CHECK_FALSE(unchanged->has_camera());
    auto generated = importer->build_importer_scene();
    if (!generated->has_camera())
        generated->add_framed_camera(35.f, 2.f);
    REQUIRE_EQ(generated->cameras.size(), 2);
    CHECK(generated->has_camera());
    CHECK_EQ(generated->cameras[1].vertical_fov_degrees(), doctest::Approx(35.f));
    CHECK_EQ(importer->build_importer_scene()->cameras.size(), 1);
    CHECK_FALSE(importer->build_importer_scene()->has_camera());

    importer->nodes().create_camera_fov("Authored");
    auto authored = importer->build_importer_scene();
    if (!authored->has_camera())
        authored->add_framed_camera();
    REQUIRE_EQ(authored->cameras.size(), 2);
    CHECK_EQ(authored->cameras[1].name, "Authored");
    CHECK_EQ(authored->nodes.size(), 1);
}
