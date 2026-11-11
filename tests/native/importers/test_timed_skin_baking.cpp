// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/render/transform.h"

#include <array>

using namespace falcor;

namespace {

float4x4 translated(float x)
{
    Transform transform;
    transform.set_translation(float3(x, 0.f, 0.f));
    return transform.matrix();
}

ImporterScene make_scene()
{
    ImporterScene scene;
    ImporterMesh mesh;
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::joints},
        {ImporterSemantic::weights},
    });
    mesh.allocate_vertices(1, true);
    mesh.position_stream()[0] = float3(1.f, 0.f, 0.f);
    mesh.normal_stream()[0] = sgl::math::normalize(float3(1.f, 1.f, 0.f));
    mesh.find_stream<int4>(ImporterSemantic::joints)[0] = int4(0, 0, 0, 0);
    mesh.find_stream<float4>(ImporterSemantic::weights)[0] = float4(1.f, 0.f, 0.f, 0.f);
    scene.meshes.push_back(std::move(mesh));
    scene.nodes.resize(3);
    scene.nodes[0].mesh_index = 0;
    scene.nodes[0].skin_index = 0;
    scene.nodes[0].parent = 2;
    scene.nodes[0].transform = translated(2.f);
    scene.nodes[1].parent = 2;
    scene.nodes[1].transform = translated(4.f);
    scene.nodes[2].transform = translated(10.f);
    scene.nodes[2].children = {0, 1};
    scene.root_nodes = {2};
    ImporterSkin skin;
    skin.joints = {1};
    skin.joint_from_bind = {float4x4::identity()};
    scene.skins.push_back(std::move(skin));
    return scene;
}

void check_vector(float3 actual, float3 expected)
{
    for (int i = 0; i < 3; ++i)
        CHECK(actual[i] == doctest::Approx(expected[i]).epsilon(1e-5));
}

} // namespace

TEST_CASE("ImporterSkinning - timed baking preserves stored placement and animation")
{
    for (auto time : std::array<std::optional<double>, 6>{std::nullopt, -1., 0., 1., 2., 4.}) {
        auto scene = make_scene();
        scene.animation.channels = {
            {.value_type = AnimationValueType::float3, .times = {0.f, 2.f}, .values = {20, 0, 0, 24, 0, 0}},
            {.value_type = AnimationValueType::float3, .times = {1.f, 3.f}, .values = {6, 0, 0, 10, 0, 0}},
            {.value_type = AnimationValueType::float3, .times = {0.f, 2.f}, .values = {100, 0, 0, 200, 0, 0}},
        };
        scene.nodes[2].animation_translation = 0;
        scene.nodes[1].animation_translation = 1;
        scene.nodes[0].animation_translation = 2;
        if (time)
            scene.bake_pose(*time);
        else
            scene.bake_pose();

        // The mesh retains world x=12. Sampled root and joint translations contribute to deformation;
        // the mesh's own animated translation never replaces its stored output placement.
        const float root_x = time ? (*time <= 0 ? 20.f : *time >= 2 ? 24.f : 22.f) : 10.f;
        const float joint_x = time ? (*time <= 1 ? 6.f : *time >= 3 ? 10.f : 8.f) : 4.f;
        check_vector(scene.meshes[0].position_stream()[0], float3(root_x + joint_x + 1.f - 12.f, 0.f, 0.f));
        CHECK(scene.skins.empty());
        CHECK_EQ(scene.nodes[0].skin_index, -1);
        CHECK_EQ(scene.nodes[0].transform, translated(2.f));
        CHECK_EQ(scene.nodes[1].transform, translated(4.f));
        CHECK_EQ(scene.nodes[2].transform, translated(10.f));
        CHECK_EQ(scene.nodes[0].animation_translation, 2);
        CHECK_EQ(scene.nodes[1].animation_translation, 1);
        CHECK_EQ(scene.nodes[2].animation_translation, 0);
        CHECK_EQ(scene.animation.channels.size(), 3);
    }
}

TEST_CASE("ImporterSkinning - timed rotation and held scale preserve unanimated translation")
{
    auto scene = make_scene();
    scene.animation.channels = {
        {.value_type = AnimationValueType::quatf, .times = {0.f, 2.f}, .values = {0, 0, 0, 1, 0, 0, 1, 0}},
        {.value_type = AnimationValueType::float3,
         .interpolation_mode = AnimationInterpolationMode::step,
         .times = {0.f, 2.f},
         .values = {2, 1, 1, 4, 1, 1}},
    };
    scene.nodes[1].animation_rotation = 0;
    scene.nodes[1].animation_scale = 1;
    scene.bake_pose(1.);
    check_vector(scene.meshes[0].position_stream()[0], float3(2.f, 2.f, 0.f));
    check_vector(scene.meshes[0].normal_stream()[0], sgl::math::normalize(float3(-1.f, 0.5f, 0.f)));
}

TEST_CASE("ImporterSkinning - timed cubic translation and shared mesh instances")
{
    auto scene = make_scene();
    scene.animation.channels = {{
        .value_type = AnimationValueType::float3,
        .interpolation_mode = AnimationInterpolationMode::cubic_spline,
        .times = {0.f, 2.f},
        .values = {4, 0, 0, 8, 0, 0},
        .in_tangents = {0, 0, 0, 0, 0, 0},
        .out_tangents = {4, 0, 0, 0, 0, 0},
    }};
    scene.nodes[1].animation_translation = 0;
    scene.nodes.push_back(scene.nodes[0]);
    scene.nodes[3].transform = translated(5.f);
    scene.nodes[2].children.push_back(3);
    scene.bake_pose(1.);
    REQUIRE_EQ(scene.meshes.size(), 3);
    // Cubic translation is 7 at the midpoint, so the posed world point is at x=18.
    check_vector(scene.meshes[scene.nodes[0].mesh_index].position_stream()[0], float3(6.f, 0.f, 0.f));
    check_vector(scene.meshes[scene.nodes[3].mesh_index].position_stream()[0], float3(3.f, 0.f, 0.f));
    check_vector(scene.meshes[0].position_stream()[0], float3(1.f, 0.f, 0.f));
}

TEST_CASE("ImporterSkinning - timed bake without skin leaves animation untouched")
{
    auto scene = make_scene();
    scene.skins.clear();
    scene.nodes[0].skin_index = -1;
    scene.animation.channels = {
        {.value_type = AnimationValueType::float3, .times = {0.f, 1.f}, .values = {20, 0, 0, 24, 0, 0}},
    };
    scene.nodes[2].animation_translation = 0;
    scene.bake_pose(1.);
    check_vector(scene.meshes[0].position_stream()[0], float3(1.f, 0.f, 0.f));
    CHECK_EQ(scene.nodes[2].transform, translated(10.f));
    CHECK_EQ(scene.nodes[2].animation_translation, 0);
    CHECK_EQ(scene.animation.channels.size(), 1);
}
