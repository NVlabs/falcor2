// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/render/transform.h"

#include <array>
#include <cmath>
#include <limits>
#include <utility>

using namespace falcor;

namespace {

float4x4 translated(float3 translation)
{
    Transform transform;
    transform.set_translation(translation);
    return transform.matrix();
}

ImporterMesh make_mesh(float3 position, float4 weights = float4(1.f, 0.f, 0.f, 0.f), bool with_normals = true)
{
    ImporterMesh mesh;
    mesh.name = "Analytic skin";
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::joints},
        {ImporterSemantic::weights},
    });
    if (with_normals)
        mesh.ensure_attributes({{ImporterSemantic::normal}});
    mesh.allocate_vertices(1, true);
    mesh.position_stream()[0] = position;
    if (with_normals)
        mesh.normal_stream()[0] = sgl::math::normalize(float3(1.f, 1.f, 0.f));
    mesh.find_stream<int4>(ImporterSemantic::joints)[0] = int4(0, 1, 0, 0);
    mesh.find_stream<float4>(ImporterSemantic::weights)[0] = weights;
    return mesh;
}

void check_vector(float3 actual, float3 expected)
{
    CHECK(actual.x == doctest::Approx(expected.x).epsilon(1e-5));
    CHECK(actual.y == doctest::Approx(expected.y).epsilon(1e-5));
    CHECK(actual.z == doctest::Approx(expected.z).epsilon(1e-5));
}

ImporterScene make_scene(float4x4 joint_pose, float4x4 mesh_pose = float4x4::identity())
{
    ImporterScene scene;
    scene.meshes.push_back(make_mesh(float3(1.f, 0.f, 0.f)));
    scene.nodes.resize(2);
    scene.nodes[0].mesh_index = 0;
    scene.nodes[0].skin_index = 0;
    scene.nodes[0].transform = mesh_pose;
    scene.nodes[1].transform = joint_pose;
    scene.root_nodes = {0, 1};
    ImporterSkin skin;
    skin.joints = {1};
    skin.bind_from_mesh = translated(float3(10.f, 0.f, 0.f));
    skin.joint_from_bind = {translated(float3(-10.f, 0.f, 0.f))};
    scene.skins.push_back(std::move(skin));
    return scene;
}

} // namespace

TEST_CASE("ImporterSkinning - documented binding spaces and rest pose")
{
    auto bind = make_scene(translated(float3(10.f, 0.f, 0.f)));
    bind.bake_pose();
    check_vector(bind.meshes[0].position_stream()[0], float3(11.f, 0.f, 0.f));

    // An unanimated rest transform is independent of the bind transform.
    auto rest = make_scene(translated(float3(10.f, 1.f, 0.f)));
    rest.bake_pose();
    check_vector(rest.meshes[0].position_stream()[0], float3(11.f, 1.f, 0.f));

    float4x4 rotation = float4x4::identity();
    rotation[0][0] = 0.f;
    rotation[0][1] = -1.f;
    rotation[1][0] = 1.f;
    rotation[1][1] = 0.f;
    const auto joint_pose = sgl::math::mul(translated(float3(10.f, 2.f, 0.f)), rotation);
    auto posed = make_scene(joint_pose, translated(float3(100.f, 0.f, 0.f)));
    posed.bake_pose();
    const auto p_world = sgl::math::mul(posed.nodes[0].transform, float4(posed.meshes[0].position_stream()[0], 1.f));
    check_vector(p_world.xyz(), float3(10.f, 3.f, 0.f));

    auto gltf_equivalent = make_scene(joint_pose);
    gltf_equivalent.meshes[0].position_stream()[0] = float3(11.f, 0.f, 0.f);
    gltf_equivalent.skins[0].bind_from_mesh = float4x4::identity();
    gltf_equivalent.bake_pose();
    check_vector(gltf_equivalent.meshes[0].position_stream()[0], float3(10.f, 3.f, 0.f));

    auto parented = make_scene(joint_pose);
    parented.nodes.resize(3);
    parented.nodes[2].transform = translated(float3(0.f, 0.f, 5.f));
    parented.nodes[2].children = {1};
    parented.nodes[1].parent = 2;
    parented.root_nodes = {0, 2};
    parented.bake_pose();
    check_vector(parented.meshes[0].position_stream()[0], float3(10.f, 3.f, 5.f));
}

TEST_CASE("ImporterSkinning - baking normalizes weights")
{
    const std::array palette = {translated(float3(2.f, 0.f, 0.f))};
    for (const float weight : {-1.f, 0.f, 1e-30f, 0.5f, 2.f}) {
        auto authored = make_mesh(float3(1.f, 0.f, 0.f), float4(weight, 0.f, 0.f, 0.f));
        authored.bake_pose(palette);
        check_vector(authored.position_stream()[0], float3(weight == 0.f ? 1.f : 3.f, 0.f, 0.f));
    }
}

TEST_CASE("ImporterSkinning - explicit sanitization preserves surviving weights and shared source")
{
    auto source = make_mesh(float3(1.f), float4(-1.f, 3.f, 2.f, std::numeric_limits<float>::infinity()));
    source.find_stream<int4>(ImporterSemantic::joints)[0] = int4(0, 1, -1, 1);
    source.ensure_attributes({{ImporterSemantic::joints, 1}, {ImporterSemantic::weights, 1}});
    source.find_stream<int4>(ImporterSemantic::joints, 1)[0] = int4(1, 9, 9, 9);
    source.find_stream<float4>(ImporterSemantic::weights, 1)[0] = float4(2.f, 0.f, 0.f, 0.f);

    auto large_palette = source;
    large_palette.sanitize_skinning(2);
    CHECK_EQ(large_palette.find_stream<float4>(ImporterSemantic::weights)[0], float4(-1.f, 3.f, 0.f, 0.f));
    CHECK_EQ(large_palette.find_stream<float4>(ImporterSemantic::weights, 1)[0], float4(2.f, 0.f, 0.f, 0.f));
    const auto joints = large_palette.find_stream<int4>(ImporterSemantic::joints)[0];
    CHECK_EQ(joints[2], 0);
    CHECK_EQ(joints[3], 0);
    CHECK_EQ(large_palette.find_stream<int4>(ImporterSemantic::joints, 1)[0][1], 0);

    auto small_palette = source;
    small_palette.sanitize_skinning(1);
    CHECK_EQ(small_palette.find_stream<float4>(ImporterSemantic::weights)[0], float4(-1.f, 0.f, 0.f, 0.f));
    CHECK_EQ(small_palette.find_stream<float4>(ImporterSemantic::weights, 1)[0], float4(0.f));
    CHECK_EQ(source.find_stream<float4>(ImporterSemantic::weights)[0].y, 3.f);
    CHECK_EQ(source.find_stream<int4>(ImporterSemantic::joints)[0][2], -1);
    CHECK_THROWS(source.sanitize_skinning(0));

    large_palette.normalize_skinning_weights();
    CHECK_EQ(large_palette.find_stream<float4>(ImporterSemantic::weights)[0], float4(-0.25f, 0.75f, 0.f, 0.f));
    CHECK_EQ(large_palette.find_stream<float4>(ImporterSemantic::weights, 1)[0], float4(0.5f, 0.f, 0.f, 0.f));
}

TEST_CASE("ImporterSkinning - explicit normalization handles zero and tiny sums")
{
    for (const float value : {0.f, 1e-30f, 2.f}) {
        auto mesh = make_mesh(float3(1.f), float4(value));
        mesh.ensure_attributes({{ImporterSemantic::joints, 1}, {ImporterSemantic::weights, 1}});
        mesh.find_stream<float4>(ImporterSemantic::weights, 1)[0] = float4(value);
        mesh.normalize_skinning_weights();
        for (uint32_t set = 0; set < 2; ++set)
            CHECK_EQ(mesh.find_stream<float4>(ImporterSemantic::weights, set)[0], float4(value == 0.f ? 0.f : 0.125f));
    }
    auto cancelling = make_mesh(float3(1.f, 0.f, 0.f), float4(1.f, -1.f, 0.f, 0.f));
    cancelling.normalize_skinning_weights();
    CHECK_EQ(cancelling.find_stream<float4>(ImporterSemantic::weights)[0], float4(1.f, -1.f, 0.f, 0.f));
    const std::array palette = {translated(float3(2.f, 0.f, 0.f)), translated(float3(5.f, 0.f, 0.f))};
    cancelling.bake_pose(palette);
    check_vector(cancelling.position_stream()[0], float3(1.f, 0.f, 0.f));
}

TEST_CASE("ImporterSkinning - documented normal example")
{
    auto mesh = make_mesh(float3(1.f), float4(0.5f, 0.5f, 0.f, 0.f));
    std::array palette = {float4x4::identity(), float4x4::identity()};
    palette[0][0][0] = 2.f;
    palette[1][1][1] = 3.f;
    mesh.bake_pose(palette);
    check_vector(mesh.normal_stream()[0], sgl::math::normalize(float3(0.75f, 2.f / 3.f, 0.f)));
    check_vector(mesh.position_stream()[0], float3(1.5f, 2.f, 1.f));
}

TEST_CASE("ImporterSkinning - normalized weights preserve placement and zero weights preserve the source")
{
    for (float weight : {0.f, 0.5f, 1.f, 2.f}) {
        auto scene = make_scene(translated(float3(10.f, 0.f, 0.f)), translated(float3(50.f, 0.f, 0.f)));
        scene.skins[0].bind_from_mesh = float4x4::identity();
        scene.meshes[0].find_stream<float4>(ImporterSemantic::weights)[0] = float4(weight, 0.f, 0.f, 0.f);
        scene.nodes.resize(3);
        scene.nodes[2].transform = translated(float3(100.f, 0.f, 0.f));
        scene.nodes[2].children = {1};
        scene.nodes[1].parent = 2;
        scene.root_nodes = {0, 2};
        scene.bake_pose();
        check_vector(scene.meshes[0].position_stream()[0], float3(weight == 0.f ? 1.f : 51.f, 0.f, 0.f));
        const auto world = sgl::math::mul(scene.nodes[0].transform, float4(scene.meshes[0].position_stream()[0], 1.f));
        check_vector(world.xyz(), float3(weight == 0.f ? 51.f : 101.f, 0.f, 0.f));
        if (weight == 0.f)
            check_vector(scene.meshes[0].normal_stream()[0], sgl::math::normalize(float3(1.f, 1.f, 0.f)));
    }
}

TEST_CASE("ImporterSkinning - singular normal transforms preserve the source normal")
{
    std::array palette = {float4x4::identity()};
    palette[0][0][0] = 0.f;
    auto mesh = make_mesh(float3(1.f));
    const auto source_normal = mesh.normal_stream()[0];
    mesh.bake_pose(palette);
    check_vector(mesh.position_stream()[0], float3(0.f, 1.f, 1.f));
    check_vector(mesh.normal_stream()[0], source_normal);
}

TEST_CASE("ImporterSkinning - singular skinning placement does not require an inverse for positions")
{
    auto scene = make_scene(translated(float3(10.f, 0.f, 0.f)));
    scene.meshes[0] = make_mesh(float3(1.f, 2.f, 3.f), float4(1.f, 0.f, 0.f, 0.f), false);
    scene.skins[0].bind_from_mesh = float4x4::identity();
    scene.nodes.resize(3);
    scene.nodes[2].transform = translated(float3(100.f, 0.f, 0.f));
    scene.nodes[2].transform[0][0] = 0.f;
    scene.nodes[1].parent = 2;
    scene.nodes[2].children = {1};
    scene.root_nodes = {0, 2};
    scene.bake_pose();
    check_vector(scene.meshes[0].position_stream()[0], float3(100.f, 2.f, 3.f));
}

TEST_CASE("ImporterSkinning - baking supports 32-bit joint indices")
{
    auto mesh = make_mesh(float3(1.f, 0.f, 0.f));
    mesh.find_stream<int4>(ImporterSemantic::joints)[0] = int4(65536, -1, 0, 0);
    std::vector<float4x4> palette(65537, float4x4::identity());
    palette.back() = translated(float3(2.f, 0.f, 0.f));
    mesh.bake_pose(palette);
    check_vector(mesh.position_stream()[0], float3(3.f, 0.f, 0.f));
}

TEST_CASE("ImporterSkinning - baking normalizes across all influence sets")
{
    auto mesh = make_mesh(float3(1.f, 0.f, 0.f), float4(2.f));
    mesh.ensure_attributes({{ImporterSemantic::joints, 1}, {ImporterSemantic::weights, 1}});
    mesh.find_stream<int4>(ImporterSemantic::joints)[0] = int4(0, 1, 2, 3);
    mesh.find_stream<int4>(ImporterSemantic::joints, 1)[0] = int4(4, 5, 6, 7);
    mesh.find_stream<float4>(ImporterSemantic::weights, 1)[0] = float4(2.f);
    std::array<float4x4, 8> palette;
    for (size_t i = 0; i < palette.size(); ++i)
        palette[i] = translated(float3(static_cast<float>(i), 0.f, 0.f));
    mesh.bake_pose(palette);
    check_vector(mesh.position_stream()[0], float3(4.5f, 0.f, 0.f));
    CHECK(mesh.find_attribute(ImporterSemantic::joints, 1) == nullptr);
}

TEST_CASE("ImporterSkinning - invalid influences")
{
    const std::array palette = {float4x4::identity()};
    for (auto semantic : {ImporterSemantic::joints, ImporterSemantic::weights}) {
        auto mesh = make_mesh(float3(1.f));
        mesh.ensure_attributes({{semantic, 1}});
        CHECK_THROWS_WITH(mesh.bake_pose(palette), doctest::Contains("matching joint and weight attributes for set 1"));
    }
    for (float weight : {std::numeric_limits<float>::quiet_NaN(), std::numeric_limits<float>::infinity()}) {
        auto mesh = make_mesh(float3(1.f), float4(weight, 0.f, 0.f, 0.f));
        CHECK_THROWS(mesh.bake_pose(palette));
        CHECK_THROWS(mesh.normalize_skinning_weights());
        mesh.sanitize_skinning(palette.size());
        CHECK_EQ(mesh.find_stream<float4>(ImporterSemantic::weights)[0], float4(0.f));
    }
    auto mesh = make_mesh(float3(1.f));
    mesh.find_stream<int4>(ImporterSemantic::joints)[0][0] = 4;
    CHECK_THROWS(mesh.bake_pose(palette));
}
