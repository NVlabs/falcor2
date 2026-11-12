// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"
#include "falcor2/importers/usd_importer/usd_importer.h"
#include "falcor2/importers/usd_importer/usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/stage.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/metrics.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/skeleton.h>
#include <pxr/usd/usdSkel/tokens.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <cmath>
#include <limits>

using namespace falcor;

namespace {

pxr::UsdStageRefPtr make_retained_skin_stage()
{
    using namespace pxr;
    auto stage = UsdStage::CreateInMemory();
    UsdGeomSetStageMetersPerUnit(stage, 1.0);
    UsdGeomSetStageUpAxis(stage, UsdGeomTokens->y);
    stage->SetTimeCodesPerSecond(24.0);
    UsdSkelRoot::Define(stage, SdfPath("/Rig"));
    auto skeleton = UsdSkelSkeleton::Define(stage, SdfPath("/Rig/Skeleton"));
    skeleton.CreateJointsAttr().Set(VtTokenArray{TfToken("root"), TfToken("root/child")});
    GfMatrix4d root(1), child(1), child_bind(1);
    root.SetTranslate(GfVec3d(10, 0, 0));
    child.SetTranslate(GfVec3d(2, 0, 0));
    child_bind.SetTranslate(GfVec3d(12, 0, 0));
    skeleton.CreateRestTransformsAttr().Set(VtMatrix4dArray{root, child});
    skeleton.CreateBindTransformsAttr().Set(VtMatrix4dArray{root, child_bind});
    auto mesh = UsdGeomMesh::Define(stage, SdfPath("/Rig/Mesh"));
    mesh.CreateSubdivisionSchemeAttr().Set(UsdGeomTokens->none);
    mesh.CreatePointsAttr().Set(VtVec3fArray{{1, 0, 0}, {0, 1, 0}, {0, 0, 0}});
    mesh.CreateFaceVertexCountsAttr().Set(VtIntArray{3});
    mesh.CreateFaceVertexIndicesAttr().Set(VtIntArray{0, 1, 2});
    mesh.CreateNormalsAttr().Set(VtVec3fArray{{0, 0, 1}, {0, 0, 1}, {0, 0, 1}});
    mesh.SetNormalsInterpolation(UsdGeomTokens->faceVarying);
    auto binding = UsdSkelBindingAPI::Apply(mesh.GetPrim());
    binding.CreateSkeletonRel().SetTargets({skeleton.GetPath()});
    binding.CreateJointsAttr().Set(VtTokenArray{TfToken("root/child"), TfToken("root")});
    binding.CreateGeomBindTransformAttr().Set(root);
    binding.CreateJointIndicesPrimvar(true, 8).Set(VtIntArray{0, 1, 0, 1, 0, 1, 0, 1});
    binding.CreateJointWeightsPrimvar(true, 8).Set(VtFloatArray(8, 0.125f));
    return stage;
}

} // namespace

TEST_CASE("ImporterSkinning - USD preserves influences for explicit preparation")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    const auto binding = UsdSkelBindingAPI(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    binding.GetJointIndicesPrimvar().Set(VtIntArray{0, 1, 65536, -1, 0, 0, 0, 0});
    binding.GetJointWeightsPrimvar().Set(
        VtFloatArray{1.25f, -0.25f, 2.f, std::numeric_limits<float>::quiet_NaN(), 0, 0, 0, 0}
    );
    auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    REQUIRE_EQ(scene->meshes.size(), 1);
    auto& mesh = scene->meshes[0];
    const auto authored = mesh.find_stream<float4>(ImporterSemantic::weights)[0];
    CHECK_EQ(authored.x, 1.25f);
    CHECK_EQ(authored.y, -0.25f);
    CHECK_EQ(authored.z, 2.f);
    CHECK(std::isnan(authored.w));
    CHECK_EQ(mesh.find_stream<int4>(ImporterSemantic::joints)[0][2], 65536);
    CHECK_EQ(mesh.find_stream<int4>(ImporterSemantic::joints)[0][3], -1);
    mesh.sanitize_skinning(scene->skins[0].joints.size());
    CHECK_EQ(mesh.find_stream<float4>(ImporterSemantic::weights)[0], float4(1.25f, -0.25f, 0.f, 0.f));
}

TEST_CASE("ImporterSkinning - USD skips skinning validation on non-renderable meshes")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    const auto mesh = UsdGeomMesh(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    UsdSkelBindingAPI(mesh.GetPrim()).CreateSkinningMethodAttr().Set(UsdSkelTokens->dualQuaternion);
    SUBCASE("invisible mesh")
    {
        mesh.CreateVisibilityAttr().Set(UsdGeomTokens->invisible);
    }
    SUBCASE("invisible ancestor")
    {
        UsdGeomImageable(stage->GetPrimAtPath(SdfPath("/Rig"))).CreateVisibilityAttr().Set(UsdGeomTokens->invisible);
    }
    SUBCASE("guide")
    {
        mesh.CreatePurposeAttr().Set(UsdGeomTokens->guide);
    }
    SUBCASE("proxy")
    {
        mesh.CreatePurposeAttr().Set(UsdGeomTokens->proxy);
    }
    const auto scene = UsdImporter().import_stage(*stage, {.bake_poses = true});
    CHECK(scene->meshes.empty());
    CHECK(scene->skins.empty());
    CHECK(scene->animation.channels.empty());
}

TEST_CASE("ImporterSkinning - USD retains binding factors and eight influences")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    REQUIRE_EQ(scene->skins.size(), 1);
    REQUIRE_EQ(scene->meshes.size(), 1);
    const auto& skin = scene->skins[0];
    REQUIRE_EQ(skin.joints.size(), 2);
    CHECK(scene->nodes[skin.joints[0]].name == "/Rig/Skeleton/root/child");
    CHECK(skin.bind_from_mesh[0][3] == doctest::Approx(10.f));
    CHECK(skin.joint_from_bind[0][0][3] == doctest::Approx(-12.f));
    auto& mesh = scene->meshes[0];
    for (uint32_t set = 0; set < 2; ++set) {
        REQUIRE(mesh.find_attribute(ImporterSemantic::joints, set));
        REQUIRE(mesh.find_attribute(ImporterSemantic::weights, set));
        const auto joints = mesh.get_stream<int4>(*mesh.find_attribute(ImporterSemantic::joints, set));
        const auto weights = mesh.get_stream<float4>(*mesh.find_attribute(ImporterSemantic::weights, set));
        for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex) {
            for (int component = 0; component < 4; ++component) {
                CHECK(joints[vertex][component] == component % 2);
                CHECK(weights[vertex][component] == doctest::Approx(0.125f));
            }
        }
    }
    CHECK(UsdImporter().import_stage(*stage)->skins.empty());
    const auto baked = UsdImporter().import_stage(*stage, {.usd_import_skinning = true, .bake_poses = true});
    CHECK(baked->skins.empty());
    REQUIRE_EQ(baked->meshes.size(), 1);
    CHECK(baked->meshes[0].find_attribute(ImporterSemantic::joints) == nullptr);
}

TEST_CASE("ImporterSkinning - USD typed streams preserve point influences and pad the final set")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto source_mesh = UsdGeomMesh(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    const VtVec3fArray source_points{{0, 0, 0}, {1, 0, 0}, {1, 1, 0}, {0, 1, 0}};
    source_mesh.CreatePointsAttr().Set(source_points);
    source_mesh.CreateFaceVertexCountsAttr().Set(VtIntArray{3, 3});
    const VtIntArray face_points{2, 0, 1, 0, 2, 3};
    source_mesh.CreateFaceVertexIndicesAttr().Set(face_points);
    source_mesh.CreateNormalsAttr().Set(VtVec3fArray(6, GfVec3f(0, 0, 1)));
    source_mesh.CreateOrientationAttr().Set(UsdGeomTokens->leftHanded);
    const auto binding = UsdSkelBindingAPI(source_mesh.GetPrim());
    VtIntArray joint_values(20);
    VtFloatArray weight_values(20);
    for (size_t point = 0; point < 4; ++point) {
        for (size_t component = 0; component < 5; ++component) {
            joint_values[5 * point + component] = int((point + component) % 2);
            weight_values[5 * point + component] = float(5 * point + component + 1);
        }
    }
    binding.CreateJointIndicesPrimvar(false, 5).Set(joint_values);
    binding.CreateJointWeightsPrimvar(false, 5).Set(weight_values);

    auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    REQUIRE_EQ(scene->meshes.size(), 1);
    auto& mesh = scene->meshes[0];
    // Shared corners must merge when all real attributes agree, including joint indices and weights.
    REQUIRE_EQ(mesh.vertex_count(), 4);
    for (size_t vertex = 0; vertex < mesh.vertex_count(); ++vertex) {
        const auto position = mesh.position_stream()[vertex];
        const auto point
            = std::find(source_points.begin(), source_points.end(), GfVec3f(position.x, position.y, position.z));
        REQUIRE(point != source_points.end());
        const size_t point_index = size_t(point - source_points.begin());
        for (uint32_t set = 0; set < 2; ++set) {
            const auto joints = mesh.find_stream<int4>(ImporterSemantic::joints, set);
            const auto weights = mesh.find_stream<float4>(ImporterSemantic::weights, set);
            REQUIRE(joints.valid());
            REQUIRE(weights.valid());
            for (uint32_t component = 0; component < 4; ++component) {
                const size_t influence = 4 * set + component;
                const size_t source = 5 * point_index + influence;
                CHECK_EQ(joints[vertex][component], influence < 5 ? joint_values[source] : 0);
                CHECK_EQ(weights[vertex][component], influence < 5 ? weight_values[source] : 0.f);
            }
        }
    }
}

TEST_CASE("ImporterSkinning - USD skinning overrides global and mesh refinement")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    const UsdGeomMesh source(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    source.GetSubdivisionSchemeAttr().Set(UsdGeomTokens->catmullClark);
    source.GetPrim().CreateAttribute(TfToken("refinementLevel"), SdfValueTypeNames->Int).Set(3);
    ImportOptions options;
    options.usd_import_skinning = true;
    options.mesh_refinement_level = 2;
    const auto scene = UsdImporter().import_stage(*stage, options);
    REQUIRE_EQ(scene->skins.size(), 1);
    REQUIRE_EQ(scene->meshes.size(), 1);
    auto& mesh = scene->meshes[0];
    CHECK_EQ(mesh.properties.get<int>("refinement_level"), 0);
    CHECK_EQ(mesh.vertex_count(), 3);
    REQUIRE_EQ(mesh.subgeometries.size(), 1);
    CHECK_EQ(mesh.subgeometries[0].indices.size(), 1);
    CHECK(mesh.find_attribute(ImporterSemantic::joints, 1));
    CHECK(mesh.find_attribute(ImporterSemantic::weights, 1));

    options.usd_import_skinning = false;
    const auto unskinned = UsdImporter().import_stage(*stage, options);
    CHECK_EQ(unskinned->meshes[0].properties.get<int>("refinement_level"), 3);
    CHECK_GT(unskinned->meshes[0].subgeometries[0].indices.size(), 1);
}

TEST_CASE("ImporterSkinning - USD sparse animation preserves rest and component times")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto animation = UsdSkelAnimation::Define(stage, SdfPath("/Rig/Animation"));
    animation.CreateJointsAttr().Set(VtTokenArray{TfToken("root/child")});
    animation.CreateTranslationsAttr().Set(VtVec3fArray{{2, 0, 0}}, 0);
    animation.GetTranslationsAttr().Set(VtVec3fArray{{2, 2, 0}}, 24);
    animation.CreateRotationsAttr().Set(VtQuatfArray{GfQuatf(1)});
    animation.CreateScalesAttr().Set(VtVec3hArray{GfVec3h(1)});
    UsdSkelBindingAPI::Apply(stage->GetPrimAtPath(SdfPath("/Rig/Skeleton")))
        .CreateAnimationSourceRel()
        .SetTargets({animation.GetPath()});
    auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    const auto& skin = scene->skins.at(0);
    const auto& animated = scene->nodes[skin.joints[0]];
    const auto& unanimated = scene->nodes[skin.joints[1]];
    REQUIRE(animated.animation_translation >= 0);
    CHECK(unanimated.animation_translation == -1);
    CHECK(unanimated.transform[0][3] == doctest::Approx(10.f));
    const auto& channel = scene->animation.channels[animated.animation_translation];
    REQUIRE_EQ(channel.times.size(), 2);
    CHECK(channel.times[0] == 0.f);
    CHECK(channel.times[1] == 1.f);
    CHECK(channel.values[4] == 2.f);
    REQUIRE_EQ(scene->animation.channels[animated.animation_rotation].times.size(), 1);

    // A mesh-only override does not change the resolved skeleton animation source.
    auto mesh_animation = UsdSkelAnimation::Define(stage, SdfPath("/Rig/MeshAnimation"));
    UsdSkelBindingAPI(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")))
        .CreateAnimationSourceRel()
        .SetTargets({mesh_animation.GetPath()});
    CHECK_FALSE(UsdImporter().import_stage(*stage, {.usd_import_skinning = true})->animation.channels.empty());

    // Missing one component makes USD fall back to the whole rest pose, including the translation.
    animation.GetScalesAttr().Block();
    scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    CHECK(scene->nodes[scene->skins[0].joints[0]].animation_translation == -1);
    CHECK(scene->nodes[scene->skins[0].joints[0]].transform[0][3] == doctest::Approx(2.f));
}

TEST_CASE("ImporterSkinning - USD imports a fixed binding from time-sampled attributes")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto binding = UsdSkelBindingAPI(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    SUBCASE("weights")
    {
        auto weights = binding.GetJointWeightsAttr();
        weights.Set(VtFloatArray(8, 0.125f), 1);
        weights.Set(VtFloatArray(8, 0.125f), 2);
        CHECK_NOTHROW(UsdImporter().import_stage(*stage, {.usd_import_skinning = true}));
        VtFloatArray changed(8, 0.125f);
        changed[0] = 0.25f;
        changed[1] = 0.f;
        weights.Set(changed, 2);
    }
    SUBCASE("indices")
    {
        binding.GetJointIndicesAttr().Set(VtIntArray{0, 1, 0, 1, 0, 1, 0, 1}, 1);
        binding.GetJointIndicesAttr().Set(VtIntArray{1, 0, 0, 1, 0, 1, 0, 1}, 2);
    }
    SUBCASE("bind transform and weights")
    {
        GfMatrix4d transform(1);
        transform.SetTranslate(GfVec3d(10, 0, 0));
        binding.GetGeomBindTransformAttr().Set(transform, 1);
        transform.SetTranslate(GfVec3d(20, 0, 0));
        binding.GetGeomBindTransformAttr().Set(transform, 2);
        binding.GetJointWeightsAttr().Set(VtFloatArray(8, 0.125f), 1);
        binding.GetJointWeightsAttr().Set(VtFloatArray(8, 0.25f), 2);
    }
    const auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    REQUIRE_EQ(scene->skins.size(), 1);
    CHECK_EQ(scene->skins[0].bind_from_mesh[0][3], 10.f);
    REQUIRE_EQ(scene->meshes.size(), 1);
    auto& mesh = scene->meshes[0];
    CHECK_EQ(mesh.find_stream<float4>(ImporterSemantic::weights)[0], float4(0.125f));
    const auto joints = mesh.find_stream<int4>(ImporterSemantic::joints)[0];
    CHECK_EQ(joints[0], 0);
    CHECK_EQ(joints[1], 1);
}

TEST_CASE("ImporterSkinning - USD diagnoses unsupported deformation")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto mesh = UsdGeomMesh(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    auto binding = UsdSkelBindingAPI(mesh.GetPrim());
    binding.CreateSkinningMethodAttr().Set(UsdSkelTokens->dualQuaternion);
    CHECK_THROWS_WITH(
        UsdImporter().import_stage(*stage, {.usd_import_skinning = true}),
        doctest::Contains("dualQuaternion")
    );
    binding.GetSkinningMethodAttr().Set(UsdSkelTokens->classicLinear);
    binding.CreateBlendShapesAttr().Set(VtTokenArray{TfToken("smile")});
    for (bool has_skeleton : {true, false}) {
        if (!has_skeleton)
            binding.GetSkeletonRel().ClearTargets(true);
        CHECK_THROWS_WITH(
            UsdImporter().import_stage(*stage, {.usd_import_skinning = true}),
            doctest::Contains("unsupported blend shapes")
        );
    }
}

TEST_CASE("ImporterSkinning - USD accepts animated half scales and normalizes rotations")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto animation = UsdSkelAnimation::Define(stage, SdfPath("/Rig/Animation"));
    animation.CreateJointsAttr().Set(VtTokenArray{TfToken("root")});
    animation.CreateTranslationsAttr().Set(VtVec3fArray{{10, 0, 0}});
    animation.CreateRotationsAttr().Set(VtQuatfArray{GfQuatf(1.001f)});
    animation.CreateScalesAttr().Set(VtVec3hArray{GfVec3h(1)}, 0);
    animation.GetScalesAttr().Set(VtVec3hArray{GfVec3h(2)}, 24);
    UsdSkelBindingAPI::Apply(stage->GetPrimAtPath(SdfPath("/Rig/Skeleton")))
        .CreateAnimationSourceRel()
        .SetTargets({animation.GetPath()});
    const auto binding = UsdSkelBindingAPI(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    binding.CreateJointIndicesPrimvar(true, 1).Set(VtIntArray{1});
    binding.CreateJointWeightsPrimvar(true, 1).Set(VtFloatArray{1.f});
    for (const auto interpolation : {UsdInterpolationTypeLinear, UsdInterpolationTypeHeld}) {
        stage->SetInterpolationType(interpolation);
        auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
        const auto source = scene->meshes[0];
        scene->bake_pose(0.3);
        const float scale = interpolation == UsdInterpolationTypeLinear ? 1.3f : 1.f;
        for (size_t vertex = 0; vertex < source.vertex_count(); ++vertex) {
            const auto expected = source.position_stream()[vertex] * scale + float3(10.f, 0.f, 0.f);
            const auto actual = scene->meshes[0].position_stream()[vertex];
            for (int component = 0; component < 3; ++component)
                CHECK(actual[component] == doctest::Approx(expected[component]));
        }
    }
}

TEST_CASE("ImporterSkinning - USD skeleton placements use ordinary transform import")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    const UsdGeomXformable placement(stage->GetPrimAtPath(SdfPath("/Rig/Skeleton")));
    auto rotation = placement.AddRotateXYZOp();
    rotation.Set(GfVec3f(0), 0);
    rotation.Set(GfVec3f(90, 90, 0), 24);
    const auto ordinary = UsdImporter().import_stage(*stage);
    const auto skinned = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    auto find_placement = [](const auto& node)
    {
        return node.name == "/Rig/Skeleton";
    };
    const auto expected = std::find_if(ordinary->nodes.begin(), ordinary->nodes.end(), find_placement);
    const auto actual = std::find_if(skinned->nodes.begin(), skinned->nodes.end(), find_placement);
    REQUIRE(expected != ordinary->nodes.end());
    REQUIRE(actual != skinned->nodes.end());
    CHECK_EQ(actual->transform, expected->transform);
    REQUIRE(expected->animation_rotation >= 0);
    REQUIRE(actual->animation_rotation >= 0);
    const auto& expected_channel = ordinary->animation.channels[expected->animation_rotation];
    const auto& actual_channel = skinned->animation.channels[actual->animation_rotation];
    CHECK_EQ(actual_channel.times, expected_channel.times);
    CHECK_EQ(actual_channel.values, expected_channel.values);
    CHECK_EQ(actual_channel.interpolation_mode, expected_channel.interpolation_mode);
}

TEST_CASE("ImporterSkinning - USD retains static joint matrices")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    auto skeleton = UsdSkelSkeleton(stage->GetPrimAtPath(SdfPath("/Rig/Skeleton")));
    VtMatrix4dArray rest;
    REQUIRE(skeleton.GetRestTransformsAttr().Get(&rest));
    rest[0][0][1] = 0.25;
    skeleton.GetRestTransformsAttr().Set(rest);
    auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = true});
    const auto& node = scene->nodes[scene->skins[0].joints[1]];
    CHECK_EQ(node.transform[1][0], 0.25f);
    CHECK_NOTHROW(scene->bake_pose(0.3));
}

TEST_CASE("ImporterSkinning - USD static meshes use the initial viewing time")
{
    using namespace pxr;
    auto stage = make_retained_skin_stage();
    const UsdGeomMesh mesh(stage->GetPrimAtPath(SdfPath("/Rig/Mesh")));
    VtVec3fArray points, normals;
    REQUIRE(mesh.GetPointsAttr().Get(&points));
    REQUIRE(mesh.GetNormalsAttr().Get(&normals));
    bool author_samples = true;
    SUBCASE("default only")
    {
        author_samples = false;
    }
    SUBCASE("samples without default")
    {
        mesh.GetPointsAttr().Clear();
        mesh.GetNormalsAttr().Clear();
    }
    SUBCASE("samples override default")
    {
        mesh.GetPointsAttr().Set(VtVec3fArray(3, GfVec3f(100)));
        mesh.GetNormalsAttr().Set(VtVec3fArray(3, GfVec3f(0, 1, 0)));
    }
    if (author_samples) {
        auto before = points, after = points;
        for (size_t i = 0; i < points.size(); ++i) {
            before[i][2] -= 2;
            after[i][2] += 2;
        }
        mesh.GetPointsAttr().Set(before, -1);
        mesh.GetPointsAttr().Set(after, 1);
        mesh.GetNormalsAttr().Set(normals, -1);
        mesh.GetNormalsAttr().Set(normals, 1);
    }
    for (double time : {0.0, 0.5}) {
        CAPTURE(time);
        if (time != 0.0) {
            stage->SetStartTimeCode(time);
            stage->SetEndTimeCode(600);
        }
        for (bool skinning : {false, true}) {
            CAPTURE(skinning);
            const auto scene = UsdImporter().import_stage(*stage, {.usd_import_skinning = skinning});
            CHECK_EQ(scene->skins.size(), skinning ? 1 : 0);
            REQUIRE_EQ(scene->meshes.size(), 1);
            auto& imported = scene->meshes[0];
            REQUIRE_EQ(imported.vertex_count(), 3);
            CHECK(imported.motion_times.empty());
            const auto positions = imported.position_stream();
            const auto imported_normals = imported.find_stream<float3>(ImporterSemantic::normal);
            for (size_t i = 0; i < points.size(); ++i) {
                CHECK_EQ(
                    positions[i],
                    float3(points[i][0], points[i][1], points[i][2] + (author_samples ? float(2 * time) : 0.f))
                );
                CHECK_EQ(imported_normals[i], float3(0, 0, 1));
            }
        }
    }
}
