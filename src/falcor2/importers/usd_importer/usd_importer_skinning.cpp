// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_context.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/usd/usd/primRange.h>
#include <pxr/usd/usdGeom/mesh.h>
#include <pxr/usd/usdGeom/pointBased.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdSkel/animation.h>
#include <pxr/usd/usdSkel/bindingAPI.h>
#include <pxr/usd/usdSkel/cache.h>
#include <pxr/usd/usdSkel/root.h>
#include <pxr/usd/usdSkel/tokens.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>

namespace falcor::usd_importer {
namespace {

// Keep each component's authored times: adding translation/scale keys to the rotation channel can change slerp.
template<typename Array, bool IS_QUATERNION = false>
std::vector<ImporterAnimationChannel> read_joint_channels(
    const pxr::UsdAttribute& attribute,
    size_t joint_count,
    double rate,
    AnimationInterpolationMode interpolation
)
{
    std::vector<double> times;
    attribute.GetTimeSamples(&times);
    if (times.empty())
        times.push_back(0.0);
    std::vector<ImporterAnimationChannel> channels(joint_count);
    for (auto& channel : channels) {
        channel.value_type = IS_QUATERNION ? AnimationValueType::quatf : AnimationValueType::float3;
        channel.interpolation_mode = interpolation;
    }
    for (double time : times) {
        Array values;
        FALCOR_CHECK(
            attribute.Get(&values, time) && values.size() == joint_count,
            "Invalid joint channel '{}' at {}",
            attribute.GetPath().GetString(),
            time
        );
        for (size_t joint = 0; joint < joint_count; ++joint) {
            auto& channel = channels[joint];
            channel.times.push_back(float(time / rate));
            if constexpr (IS_QUATERNION) {
                const auto value = values[joint].GetNormalized();
                for (int component = 0; component < 3; ++component)
                    channel.values.push_back(value.GetImaginary()[component]);
                channel.values.push_back(value.GetReal());
            } else {
                for (int component = 0; component < 3; ++component)
                    channel.values.push_back(float(values[joint][component]));
            }
        }
    }
    return channels;
}

} // namespace

void UsdImporterContext::prepare_skinning(const pxr::UsdPrim& root)
{
    using namespace pxr;
    if (!m_import_options.usd_import_skinning && !m_import_options.bake_poses)
        return;
    UsdSkelCache cache;
    const auto predicate = UsdTraverseInstanceProxies();
    for (const UsdPrim& prim : UsdPrimRange(root, predicate)) {
        if (prim.IsA<UsdSkelRoot>())
            FALCOR_CHECK(
                cache.Populate(UsdSkelRoot(prim), predicate),
                "Cannot read skeletons at '{}'",
                prim.GetPath().GetString()
            );
    }
    for (const UsdPrim& prim : UsdPrimRange(root, predicate)) {
        if (!prim.IsA<UsdGeomPointBased>() || !is_renderable(UsdGeomImageable(prim), m_initial_time))
            continue;
        const UsdSkelBindingAPI binding(prim);
        const std::string path = prim.GetPath().GetString();
        VtTokenArray blend_shapes;
        SdfPathVector blend_targets;
        binding.GetBlendShapesAttr().Get(&blend_shapes);
        if (const auto targets = binding.GetBlendShapeTargetsRel())
            targets.GetTargets(&blend_targets);
        FALCOR_CHECK(blend_shapes.empty() && blend_targets.empty(), "Skin '{}' uses unsupported blend shapes", path);
        const auto skeleton = binding.GetInheritedSkeleton();
        if (!skeleton)
            continue;
        const auto query = cache.GetSkinningQuery(prim);
        if (!query) {
            const UsdGeomPrimvarsAPI primvars(prim);
            const auto indices = primvars.FindPrimvarWithInheritance(TfToken("skel:jointIndices"));
            const auto weights = primvars.FindPrimvarWithInheritance(TfToken("skel:jointWeights"));
            FALCOR_CHECK(
                (!indices || !indices.HasValue()) && (!weights || !weights.HasValue()),
                "Invalid skin binding at '{}'",
                prim.GetPath().GetString()
            );
            continue;
        }
        if (!query.HasJointInfluences())
            continue; // A skeleton relationship alone does not deform a rigid mesh.
        FALCOR_CHECK(prim.IsA<UsdGeomMesh>(), "Skin '{}' uses unsupported non-mesh geometry", path);
        FALCOR_CHECK(
            !prim.IsInstanceProxy() && !prim.IsInPrototype() && !skeleton.GetPrim().IsInstanceProxy()
                && !skeleton.GetPrim().IsInstance(),
            "Skin '{}' uses unsupported skeletal instancing",
            path
        );
        FALCOR_CHECK(
            query.GetSkinningMethod() == UsdSkelTokens->classicLinear,
            "Skin '{}' uses unsupported method '{}'",
            path,
            query.GetSkinningMethod().GetString()
        );
        auto skeleton_query = cache.GetSkelQuery(skeleton);
        FALCOR_CHECK(skeleton_query, "Invalid skeleton for skin '{}'", path);
        m_skin_bindings.emplace(prim.GetPath(), SkinBinding{query, skeleton_query});
    }
}

ImporterSkin UsdImporterContext::import_skeleton(const pxr::UsdSkelSkeletonQuery& query)
{
    using namespace pxr;
    const auto path = query.GetPrim().GetPath();
    const auto placement = m_prim_path_to_nodes.find(path);
    FALCOR_CHECK(
        placement != m_prim_path_to_nodes.end() && placement->second.size() == 1,
        "Skeleton '{}' has no unique imported placement",
        path.GetString()
    );
    const auto& chain = placement->second.front();
    ImporterSkin skin;
    skin.skeleton_root = chain.pivot_inverse >= 0 ? chain.pivot_inverse : chain.authored;
    const auto order = query.GetJointOrder();
    const auto& topology = query.GetTopology();
    std::string error;
    FALCOR_CHECK(topology.Validate(&error), "Invalid skeleton '{}': {}", path.GetString(), error);
    VtMatrix4dArray locals, binds;
    FALCOR_CHECK(
        query.ComputeJointLocalTransforms(&locals, m_initial_time) && query.GetJointWorldBindTransforms(&binds)
            && locals.size() == order.size() && binds.size() == order.size(),
        "Invalid bind/rest transforms for skeleton '{}'",
        path.GetString()
    );
    for (size_t joint = 0; joint < order.size(); ++joint) {
        const int parent = topology.GetParent(joint);
        skin.joints.push_back(add_transform(
            to_falcor(locals[joint]),
            path.GetString() + "/" + order[joint].GetString(),
            parent < 0 ? skin.skeleton_root : skin.joints.at(parent)
        ));
        // USD uses row vectors; to_falcor transposes. Units and skeleton placement are already on ancestor nodes.
        skin.joint_from_bind.push_back(to_falcor(binds[joint].GetInverse()));
    }
    const auto animation_query = query.GetAnimQuery();
    if (!animation_query)
        return skin;
    const UsdSkelAnimation animation(animation_query.GetPrim());
    if (!animation.GetTranslationsAttr().HasValue() || !animation.GetRotationsAttr().HasValue()
        || !animation.GetScalesAttr().HasValue())
        return skin; // Missing TRS components select the rest pose, as in UsdSkel.
    const auto animation_order = animation_query.GetJointOrder();
    const double rate = get_time_codes_per_second(*m_usd_stage);
    const auto interpolation = m_usd_stage->GetInterpolationType() == UsdInterpolationTypeHeld
        ? AnimationInterpolationMode::step
        : AnimationInterpolationMode::linear;
    auto translations = read_joint_channels<VtVec3fArray>(
        animation.GetTranslationsAttr(),
        animation_order.size(),
        rate,
        interpolation
    );
    auto rotations = read_joint_channels<VtQuatfArray, true>(
        animation.GetRotationsAttr(),
        animation_order.size(),
        rate,
        interpolation
    );
    auto scales
        = read_joint_channels<VtVec3hArray>(animation.GetScalesAttr(), animation_order.size(), rate, interpolation);
    auto attach = [&](ImporterAnimationChannel channel, int& index)
    {
        index = int(m_scene->animation.channels.size());
        m_scene->animation.channels.push_back(std::move(channel));
    };
    for (size_t joint = 0; joint < animation_order.size(); ++joint) {
        const auto found = std::find(order.begin(), order.end(), animation_order[joint]);
        if (found == order.end())
            continue;
        auto& node = m_scene->nodes[skin.joints[size_t(found - order.begin())]];
        attach(std::move(translations[joint]), node.animation_translation);
        attach(std::move(rotations[joint]), node.animation_rotation);
        attach(std::move(scales[joint]), node.animation_scale);
    }
    m_scene->animation.name = "UsdAnimation";
    return skin;
}

void UsdImporterContext::extract_skinning()
{
    using namespace pxr;
    std::map<SdfPath, ImporterSkin> skeletons;
    for (const auto& [path, binding] : m_skin_bindings) {
        const auto nodes = m_prim_path_to_nodes.find(path);
        if (nodes == m_prim_path_to_nodes.end())
            continue;
        for (const auto& chain : nodes->second) {
            const int node = chain.pivot_inverse >= 0 ? chain.pivot_inverse : chain.authored;
            if (m_scene->nodes[node].mesh_index < 0)
                continue;
            const auto skeleton_path = binding.skeleton.GetPrim().GetPath();
            auto skeleton = skeletons.find(skeleton_path);
            if (skeleton == skeletons.end())
                skeleton = skeletons.emplace(skeleton_path, import_skeleton(binding.skeleton)).first;
            ImporterSkin skin;
            skin.name = path.GetString();
            skin.skeleton_root = skeleton->second.skeleton_root;
            skin.bind_from_mesh = to_falcor(binding.mesh.GetGeomBindTransform(m_initial_time));
            const auto order = binding.skeleton.GetJointOrder();
            VtTokenArray palette;
            if (!binding.mesh.GetJointOrder(&palette))
                palette = order;
            for (const TfToken& joint : palette) {
                const auto found = std::find(order.begin(), order.end(), joint);
                FALCOR_CHECK(
                    found != order.end(),
                    "Skin '{}' references unknown joint '{}'",
                    path.GetString(),
                    joint.GetString()
                );
                const size_t index = size_t(found - order.begin());
                skin.joints.push_back(skeleton->second.joints[index]);
                skin.joint_from_bind.push_back(skeleton->second.joint_from_bind[index]);
            }
            m_scene->nodes[node].skin_index = int(m_scene->skins.size());
            m_scene->skins.push_back(std::move(skin));
        }
    }
}

} // namespace falcor::usd_importer
