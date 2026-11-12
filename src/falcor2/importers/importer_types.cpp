// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_utils.h"
#include "falcor2/importers/mikkt.h"
#include "falcor2/importers/vertex_remap.h"
#include "falcor2/geometry/tangent_transfer.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/transform.h"
#include "falcor2/utils/camera_framing.h"

#include "falcor2/core/types.h"
#include "falcor2/utils/indexed_vector.h"

#include <sgl/core/type_utils.h>

#include <algorithm>
#include <array>
#include <bit>
#include <cmath>
#include <limits>
#include <map>

namespace falcor {

void detail::validate_refinement_level(int level)
{
    FALCOR_CHECK(level >= 0 && level <= 256, "Refinement level {} must be between 0 and 256", level);
}

Properties ImporterMesh::make_properties(const ImportOptions& options)
{
    detail::validate_refinement_level(options.mesh_refinement_level);
    Properties properties;
    // Zero triangulates the control cage; positive N samples N+1 edge segments on the limit surface.
    // Polygon meshes remain polygonal. The mesh option defaults to zero; valid levels are 0 through 256.
    properties.set("refinement_level", options.mesh_refinement_level);
    // Request the additional shared face-normal/tangent recomputation, not missing-normal generation.
    properties.set("recompute_normals", options.recompute_normals);
    return properties;
}

Properties ImporterCurve::make_properties(const ImportOptions& options)
{
    detail::validate_refinement_level(options.curve_refinement_level);
    Properties properties;
    // Zero uses the authored CV polygon; positive N evaluates N+1 segments per cubic span.
    // Linear curves are unchanged. The curve option defaults to zero; valid levels are 0 through 256.
    properties.set("refinement_level", options.curve_refinement_level);
    return properties;
}

Properties ImporterLight::make_properties(const ImportOptions& options, Type type)
{
    bool convert_to_mesh = false;
    switch (type) {
    case Type::rectangular:
        convert_to_mesh = options.force_rectangle_lights_to_geometry;
        break;
    case Type::disk:
        convert_to_mesh = options.force_disk_lights_to_geometry;
        break;
    case Type::sphere:
        convert_to_mesh = options.force_sphere_lights_to_geometry;
        break;
    default:
        break;
    }
    Properties properties;
    // Convert to emissive triangles, using the type-specific option as the default. Only rectangles, disks
    // and non-point spheres are eligible; shaped lights retain their analytic representation.
    properties.set("convert_to_mesh", convert_to_mesh);
    return properties;
}

namespace {

// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
float3 branchlessONB(float3 normal)
{
    float sign = std::copysign(1.f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    return float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
}

// Visit each instance with its composed transform. A visitor returning true stops the traversal.
template<typename Visitor>
bool visit_importer_nodes(const ImporterScene& scene, Visitor&& visitor)
{
    std::vector<bool> visiting(scene.nodes.size(), false);
    std::function<bool(int, const float4x4&)> visit_node = [&](int index, const float4x4& world_from_parent)
    {
        if (index < 0 || static_cast<size_t>(index) >= scene.nodes.size())
            return false;
        FALCOR_CHECK(!visiting[index], "Cycle in importer scene node hierarchy.");
        visiting[index] = true;
        const auto& node = scene.nodes[index];
        const auto world_from_node = sgl::math::mul(world_from_parent, node.transform);
        if (visitor(node, world_from_node))
            return true;
        for (int child : node.children) {
            if (visit_node(child, world_from_node))
                return true;
        }
        if (node.prototype_index >= 0 && static_cast<size_t>(node.prototype_index) < scene.prototypes.size()) {
            if (visit_node(scene.prototypes[node.prototype_index].root_node, world_from_node))
                return true;
        }
        visiting[index] = false;
        return false;
    };
    for (int root : scene.root_nodes) {
        if (visit_node(root, float4x4::identity()))
            return true;
    }
    return false;
}

} // namespace

int ImporterAnimationChannel::components_per_value() const
{
    switch (value_type) {
    case AnimationValueType::float_:
        return 1;
    case AnimationValueType::float3:
        return 3;
    case AnimationValueType::quatf:
        return 4;
    default:
        SGL_THROW("Unknown animation channel value type");
    }
}

size_t ImporterAnimationChannel::keyframe_count() const
{
    return times.size();
}

AnimationTimeRange ImporterAnimation::key_range() const
{
    AnimationTimeRange range;
    for (const ImporterAnimationChannel& channel : channels) {
        if (!channel.times.empty())
            range.expand({channel.times.front(), channel.times.back()});
    }
    return range;
}

float ImporterCamera::focal_length_from_fov_degrees(float fov_degrees, float sensor_size_mm)
{
    SGL_CHECK(fov_degrees > 0.f && fov_degrees < 180.f, "Camera field of view must be between 0 and 180 degrees");
    SGL_CHECK(sensor_size_mm > 0.f, "Camera sensor size must be positive");

    const float fov_rad = sgl::math::radians(fov_degrees);
    return sensor_size_mm / (2.f * sgl::math::tan(fov_rad * 0.5f));
}

float ImporterCamera::fov_degrees_from_focal_length(float sensor_size_mm, float focal_length_mm)
{
    SGL_CHECK(sensor_size_mm > 0.f, "Camera sensor size must be positive");
    SGL_CHECK(focal_length_mm > 0.f, "Camera focal length must be positive");

    return sgl::math::degrees(2.f * std::atan(sensor_size_mm / (2.f * focal_length_mm)));
}

float ImporterCamera::fov_degrees() const
{
    return fov_degrees_from_focal_length(sensor_size_mm, focal_length);
}

float ImporterCamera::vertical_sensor_size_mm(float horizontal_to_vertical_sensor_ratio) const
{
    SGL_CHECK(horizontal_to_vertical_sensor_ratio > 0.f, "Camera sensor ratio must be positive");

    return fov_direction == FOVDirection::vertical ? sensor_size_mm
                                                   : sensor_size_mm / horizontal_to_vertical_sensor_ratio;
}

float ImporterCamera::vertical_fov_degrees(float horizontal_to_vertical_sensor_ratio) const
{
    return fov_degrees_from_focal_length(vertical_sensor_size_mm(horizontal_to_vertical_sensor_ratio), focal_length);
}

void ImporterCurve::resize_vertices(size_t vertex_count, size_t sample_count)
{
    FALCOR_CHECK(sample_count > 0, "Curve must have at least one sample");
    m_positions.resize(sample_count);
    m_radii.resize(sample_count);
    for (size_t sample = 0; sample < sample_count; ++sample) {
        m_positions[sample].stride = sizeof(float3);
        m_positions[sample].resize_vertices(vertex_count);
        m_radii[sample].stride = sizeof(float);
        m_radii[sample].resize_vertices(vertex_count);
    }
    m_vertex_count = vertex_count;
}

void ImporterCurve::calculate_local_aabb()
{
    local_aabb = AABB();
    const auto positions = position_stream();
    const auto radii = radius_stream();
    for (size_t i = 0; i < positions.size; ++i) {
        float r = i < radii.size ? radii[i] : 0.f;
        local_aabb.expand(positions[i], r);
    }
}

namespace {

struct SkinningStreams {
    ImporterVertexStream<int4> joints;
    ImporterVertexStream<float4> weights;
};

std::vector<SkinningStreams> get_skinning_streams(ImporterMesh& mesh)
{
    // A mesh may have more than four influences per vertex, represented by matching JOINTS_n/WEIGHTS_n pairs.
    // Collect views of every pair once so vertex processing is independent of the importer attribute layout.
    std::vector<SkinningStreams> streams;
    for (const ImporterMeshAttribute& attribute : mesh.attributes()) {
        if (attribute.semantic != ImporterSemantic::joints && attribute.semantic != ImporterSemantic::weights)
            continue;

        ImporterMeshAttribute* joint_attribute = mesh.find_attribute(ImporterSemantic::joints, attribute.index);
        ImporterMeshAttribute* weight_attribute = mesh.find_attribute(ImporterSemantic::weights, attribute.index);
        SGL_CHECK(
            joint_attribute && weight_attribute,
            "Mesh '{}' must provide matching joint and weight attributes for set {}",
            mesh.name,
            attribute.index
        );
        if (attribute.semantic == ImporterSemantic::joints)
            streams.push_back({mesh.get_stream<int4>(*joint_attribute), mesh.get_stream<float4>(*weight_attribute)});
    }
    return streams;
}

} // namespace

void ImporterMesh::sanitize_skinning(size_t joint_count)
{
    FALCOR_CHECK(joint_count > 0, "Cannot sanitize skinning without a joint palette");
    for (const auto& stream : get_skinning_streams(*this)) {
        for (size_t vertex = 0; vertex < vertex_count(); ++vertex) {
            for (int component = 0; component < 4; ++component) {
                auto& joint = stream.joints[vertex][component];
                auto& weight = stream.weights[vertex][component];
                if (joint < 0 || static_cast<size_t>(joint) >= joint_count || !std::isfinite(weight))
                    weight = 0.f;
                if (weight == 0.f)
                    joint = 0;
            }
        }
    }
}

void ImporterMesh::normalize_skinning_weights()
{
    const auto streams = get_skinning_streams(*this);
    for (size_t vertex = 0; vertex < vertex_count(); ++vertex) {
        double sum = 0.0;
        for (const auto& stream : streams)
            for (int component = 0; component < 4; ++component)
                sum += stream.weights[vertex][component];
        FALCOR_CHECK(std::isfinite(sum), "Mesh '{}' has nonfinite skin weights", name);
        if (sum == 0.0 || sum == 1.0)
            continue;
        for (const auto& stream : streams)
            for (int component = 0; component < 4; ++component)
                stream.weights[vertex][component] = static_cast<float>(stream.weights[vertex][component] / sum);
    }
}

void ImporterMesh::bake_pose(std::span<const float4x4> mesh_from_bind_mesh_by_joint)
{
    for (const auto& attribute : m_attributes)
        FALCOR_CHECK(
            attribute.semantic != ImporterSemantic::position || attribute.index == 0,
            "Cannot bake a skin pose into a mesh with motion samples"
        );
    normalize_skinning_weights();
    const auto streams = get_skinning_streams(*this);

    SGL_CHECK(!streams.empty(), "Skinned mesh '{}' has no joint influence streams", name);

    auto positions = position_stream();
    auto normals = normal_stream();
    auto tangents = tangent_stream();
    auto handedness = handedness_stream();

    // Match the Khronos glTF Sample Renderer and OpenUSD: inverse-transpose each joint's linear transform before
    // blending normals. Inverse-transposing the blended matrix gives a different result.
    std::vector<float3x3> normal_matrices;
    if (normals.valid()) {
        normal_matrices.reserve(mesh_from_bind_mesh_by_joint.size());
        for (const float4x4& joint_matrix : mesh_from_bind_mesh_by_joint)
            normal_matrices.push_back(sgl::math::transpose(sgl::math::inverse(float3x3(joint_matrix))));
    }

    for (size_t vertex_index = 0; vertex_index < vertex_count(); ++vertex_index) {
        // Linear-blend skinning forms one matrix per vertex by taking the weighted sum of its joint matrices.
        float4x4 mesh_from_bind_mesh = float4x4::zeros();
        float3 transformed_normal = {};
        float weight_sum = 0.f;
        for (const SkinningStreams& skinning_stream : streams) {
            const int4& joints = skinning_stream.joints[vertex_index];
            const float4& weights = skinning_stream.weights[vertex_index];
            for (int component = 0; component < 4; ++component) {
                const float weight = weights[component];
                if (weight == 0.f)
                    continue;
                SGL_CHECK(
                    static_cast<size_t>(joints[component]) < mesh_from_bind_mesh_by_joint.size(),
                    "Vertex {} of mesh '{}' references joint {} but its skin has only {} joints",
                    vertex_index,
                    name,
                    joints[component],
                    mesh_from_bind_mesh_by_joint.size()
                );
                const float4x4& mesh_from_bind_mesh_for_joint = mesh_from_bind_mesh_by_joint[joints[component]];
                for (int row = 0; row < 4; ++row)
                    mesh_from_bind_mesh[row] += mesh_from_bind_mesh_for_joint[row] * weight;
                if (normals.valid()) {
                    transformed_normal
                        += sgl::math::transform_vector(normal_matrices[joints[component]], normals[vertex_index])
                        * weight;
                }
                weight_sum += weight;
            }
        }

        // Preserve vertices without a normalized pose, including cancellation of signed weights.
        if (weight_sum == 0.f)
            continue;

        positions[vertex_index] = sgl::math::transform_point(mesh_from_bind_mesh, positions[vertex_index]);

        const float3x3 mesh_from_bind_mesh_linear(mesh_from_bind_mesh);
        const float determinant = sgl::math::determinant(mesh_from_bind_mesh_linear);
        if (normals.valid()) {
            transformed_normal = normalize_or(transformed_normal, normals[vertex_index]);
            normals[vertex_index] = transformed_normal;
        }

        if (tangents.valid()) {
            // Tangents are ordinary direction vectors. Transform them with the linear part, then remove any component
            // along the transformed normal to recover an orthonormal tangent frame.
            float3 transformed_tangent
                = sgl::math::transform_vector(mesh_from_bind_mesh_linear, tangents[vertex_index]);
            if (normals.valid())
                transformed_tangent -= transformed_normal * sgl::math::dot(transformed_normal, transformed_tangent);

            float3 fallback_tangent = tangents[vertex_index];
            if (normals.valid()) {
                const float3 axis
                    = std::abs(transformed_normal.z) < 0.999f ? float3(0.f, 0.f, 1.f) : float3(0.f, 1.f, 0.f);
                fallback_tangent = sgl::math::cross(axis, transformed_normal);
            }
            tangents[vertex_index] = normalize_or(transformed_tangent, fallback_tangent);
        }

        // A reflection reverses the orientation of the tangent frame.
        if (handedness.valid() && determinant < 0.f)
            handedness[vertex_index] = -handedness[vertex_index];
    }

    // The result is now an ordinary static mesh. Removing the influence streams makes baking idempotent and prevents
    // later mesh processing (such as subdivision) from treating stale skinning data as live attributes.
    std::erase_if(
        m_attributes,
        [](const ImporterMeshAttribute& attribute)
        {
            return attribute.semantic == ImporterSemantic::joints || attribute.semantic == ImporterSemantic::weights;
        }
    );
    calculate_local_aabb();
}

namespace {

std::vector<float4x4>
evaluate_local_transforms(std::span<const ImporterNode> nodes, const ImporterAnimation& animation, float time_seconds)
{
    // Detached channels use the same interpolation as Scene without requiring a Scene or a GPU device.
    std::vector<ref<AnimationChannel>> channels;
    channels.reserve(animation.channels.size());
    for (const ImporterAnimationChannel& source : animation.channels) {
        auto channel = make_ref<AnimationChannel>();
        channel->set_value_type(source.value_type);
        channel->set_interpolation_mode(source.interpolation_mode);
        channel->set_times(source.times);
        channel->set_values(source.values);
        channel->set_in_tangents(source.in_tangents);
        channel->set_out_tangents(source.out_tangents);
        channels.push_back(std::move(channel));
    }
    auto find_channel = [&](int index) -> const AnimationChannel*
    {
        return index >= 0 && static_cast<size_t>(index) < channels.size() && !channels[index]->times().empty()
            ? channels[index].get()
            : nullptr;
    };

    std::vector<float4x4> local_transforms;
    local_transforms.reserve(nodes.size());
    for (const ImporterNode& node : nodes) {
        const auto* translation = find_channel(node.animation_translation);
        const auto* rotation = find_channel(node.animation_rotation);
        const auto* scale = find_channel(node.animation_scale);
        if (!translation && !rotation && !scale) {
            local_transforms.push_back(node.transform);
            continue;
        }
        Transform transform(node.transform);
        if (translation)
            transform.set_translation(translation->evaluate_float3(time_seconds));
        if (rotation)
            transform.set_rotation(rotation->evaluate_quatf(time_seconds));
        if (scale)
            transform.set_scale(scale->evaluate_float3(time_seconds));
        local_transforms.push_back(transform.matrix());
    }
    return local_transforms;
}

} // namespace

AnimationTimeRange ImporterScene::animation_range() const
{
    if (playback_range.is_valid())
        return playback_range;
    AnimationTimeRange range = animation.key_range();
    for (const ImporterMesh& mesh : meshes) {
        if (!mesh.motion_times.empty())
            range.expand({mesh.motion_times.front(), mesh.motion_times.back()});
    }
    for (const ImporterCurve& curve : curves) {
        if (!curve.motion_times.empty())
            range.expand({curve.motion_times.front(), curve.motion_times.back()});
    }
    return range;
}

void ImporterScene::bake_pose(std::optional<double> time_seconds)
{
    if (skins.empty())
        return;

    std::vector<float4x4> sampled_local_transforms;
    if (time_seconds) {
        FALCOR_CHECK(std::isfinite(*time_seconds), "Skin bake time must be finite");
        sampled_local_transforms = evaluate_local_transforms(nodes, animation, static_cast<float>(*time_seconds));
    }

    // Skin joints and mesh instances are both importer nodes. Resolve every node's world transform first; nodes are
    // not required to be stored in parent-before-child order, hence the recursive memoized walk.
    std::vector<float4x4> world_from_node(nodes.size());
    std::vector<float4x4> sampled_world_from_node(nodes.size());
    std::vector<uint8_t> visit_state(nodes.size(), 0);
    std::function<const float4x4&(size_t)> resolve_world_from_node = [&](size_t node_index) -> const float4x4&
    {
        SGL_CHECK_LT(node_index, nodes.size());
        SGL_CHECK(visit_state[node_index] != 1, "Cycle detected in importer node hierarchy at node {}", node_index);
        if (visit_state[node_index] == 2)
            return world_from_node[node_index];

        visit_state[node_index] = 1;
        const ImporterNode& node = nodes[node_index];
        world_from_node[node_index] = node.parent >= 0
            ? sgl::math::mul(resolve_world_from_node(static_cast<size_t>(node.parent)), node.transform)
            : node.transform;
        const float4x4& sampled_local = time_seconds ? sampled_local_transforms[node_index] : node.transform;
        sampled_world_from_node[node_index]
            = node.parent >= 0 ? sgl::math::mul(sampled_world_from_node[node.parent], sampled_local) : sampled_local;
        visit_state[node_index] = 2;
        return world_from_node[node_index];
    };
    for (size_t node_index = 0; node_index < nodes.size(); ++node_index)
        resolve_world_from_node(node_index);

    // Baking is instance-dependent because the mesh node's world transform is part of the joint palette. A mesh used
    // by multiple nodes must therefore be copied before one instance is deformed.
    std::vector<size_t> mesh_reference_counts(meshes.size(), 0);
    for (const ImporterNode& node : nodes) {
        if (node.mesh_index >= 0) {
            SGL_CHECK_LT(static_cast<size_t>(node.mesh_index), mesh_reference_counts.size());
            ++mesh_reference_counts[node.mesh_index];
        }
    }

    for (size_t node_index = 0; node_index < nodes.size(); ++node_index) {
        ImporterNode& node = nodes[node_index];
        if (node.mesh_index < 0 || node.skin_index < 0)
            continue;
        SGL_CHECK_LT(static_cast<size_t>(node.skin_index), skins.size());

        const size_t source_mesh_index = static_cast<size_t>(node.mesh_index);
        SGL_CHECK_LT(source_mesh_index, meshes.size());
        const ImporterSkin& skin = skins[node.skin_index];
        SGL_CHECK(
            skin.joints.size() == skin.joint_from_bind.size(),
            "Skin '{}' must provide one bind transform per joint",
            skin.name
        );

        // A joint matrix produces world-space positions (bind_from_mesh is identity for glTF):
        //
        //     world_from_joint * joint_from_bind * bind_from_mesh
        //
        // Falcor retains and later applies the mesh node transform, so bake each palette entry back into that node's
        // local mesh space. This preserves the authored hierarchy and avoids applying the mesh transform twice:
        //
        //     mesh_from_bind_mesh = mesh_from_world * world_from_joint(time) * joint_from_bind * bind_from_mesh
        //
        // mesh_from_world uses the stored output placement, which this operation leaves unchanged. For example,
        // a mesh stored at x=10 and a joint sampled at x=12 produce local x=2 for a bind point at the origin.
        const float4x4 mesh_from_world = sgl::math::inverse(world_from_node[node_index]);
        std::vector<float4x4> mesh_from_bind_mesh_by_joint;
        mesh_from_bind_mesh_by_joint.reserve(skin.joints.size());
        for (size_t joint_index = 0; joint_index < skin.joints.size(); ++joint_index) {
            const int joint_node_index = skin.joints[joint_index];
            SGL_CHECK(
                joint_node_index >= 0 && joint_node_index < static_cast<int>(world_from_node.size()),
                "Skin '{}' references invalid joint node {}",
                skin.name,
                joint_node_index
            );
            mesh_from_bind_mesh_by_joint.push_back(
                sgl::math::mul(
                    sgl::math::mul(
                        sgl::math::mul(mesh_from_world, sampled_world_from_node[joint_node_index]),
                        skin.joint_from_bind[joint_index]
                    ),
                    skin.bind_from_mesh
                )
            );
        }

        // ImporterMesh owns all vertex stream mutation; the scene only chooses the correct instance and supplies its
        // already-composed mesh-local joint palette.
        ImporterMesh* target_mesh = &meshes[source_mesh_index];
        if (mesh_reference_counts[source_mesh_index] > 1) {
            ImporterMesh posed_mesh = *target_mesh;
            posed_mesh.name += fmt::format("_skin_{}_node_{}", node.skin_index, node_index);
            meshes.push_back(std::move(posed_mesh));
            node.mesh_index = sgl::narrow_cast<int>(meshes.size() - 1);
            target_mesh = &meshes.back();
        }
        target_mesh->bake_pose(mesh_from_bind_mesh_by_joint);
        node.skin_index = -1;
    }

    // All bound meshes are static now, so the scene no longer needs skeleton bindings. Joint nodes themselves remain
    // untouched because they may still be useful as ordinary authored hierarchy nodes.
    skins.clear();
    calculate_aabbs();
}

size_t ImporterScene::collapse_exact_opposite_triangles()
{
    // A mesh shared by static and skinned nodes must stay intact for every instance.
    std::vector<uint8_t> skinned_meshes(meshes.size());
    for (const ImporterNode& node : nodes) {
        if (node.mesh_index >= 0 && node.skin_index >= 0) {
            SGL_CHECK_LT(static_cast<size_t>(node.mesh_index), meshes.size());
            skinned_meshes[node.mesh_index] = 1;
        }
    }

    size_t removed_triangle_count = 0;
    for (size_t mesh_index = 0; mesh_index < meshes.size(); ++mesh_index) {
        if (!skinned_meshes[mesh_index])
            removed_triangle_count += meshes[mesh_index].collapse_exact_opposite_triangles();
    }
    return removed_triangle_count;
}

void ImporterScene::calculate_aabbs()
{
    // First, calculate local AABBs for all meshes
    for (auto& mesh : meshes) {
        mesh.calculate_local_aabb();
    }

    // Calculate local AABBs for all curves
    for (auto& curve : curves) {
        curve.calculate_local_aabb();
    }

    // Then calculate world AABBs for all nodes using a recursive approach
    // We need to process nodes in dependency order (parents before children)

    // Helper function to recursively calculate world AABB for a node
    std::function<AABB(int, const float4x4&)> calculate_node_aabb_recursive
        = [&](int node_idx, const float4x4& world_from_parent) -> AABB
    {
        if (node_idx < 0 || node_idx >= static_cast<int>(nodes.size())) {
            return AABB();
        }

        auto& node = nodes[node_idx];

        // Calculate world transform for this node
        float4x4 world_from_node = sgl::math::mul(world_from_parent, node.transform);

        AABB node_aabb;

        // If this node has a mesh, include its transformed AABB
        if (node.mesh_index >= 0 && node.mesh_index < static_cast<int>(meshes.size())) {
            const auto& mesh = meshes[node.mesh_index];
            if (mesh.local_aabb.is_valid()) {
                AABB transformed_mesh_aabb = mesh.local_aabb.transform(world_from_node);
                node_aabb.expand(transformed_mesh_aabb);
            }
        }

        // If this node has a curve, include its transformed AABB
        if (node.curve_index >= 0 && node.curve_index < static_cast<int>(curves.size())) {
            const auto& curve = curves[node.curve_index];
            if (curve.local_aabb.is_valid()) {
                AABB transformed_curve_aabb = curve.local_aabb.transform(world_from_node);
                node_aabb.expand(transformed_curve_aabb);
            }
        }

        // Include AABBs from all children
        for (int child_idx : node.children) {
            AABB child_aabb = calculate_node_aabb_recursive(child_idx, world_from_node);
            if (child_aabb.is_valid())
                node_aabb.expand(child_aabb);
        }

        // Store the calculated AABB in the node
        node.world_aabb = node_aabb;

        return node_aabb;
    };

    // Calculate AABBs starting from root nodes with identity transform
    float4x4 identity = float4x4::identity();
    for (int root_idx : root_nodes) {
        calculate_node_aabb_recursive(root_idx, identity);
    }
}

AABB ImporterScene::get_scene_aabb() const
{
    AABB scene_aabb;

    // Union all root node AABBs to get the overall scene AABB
    for (int root_idx : root_nodes) {
        if (root_idx >= 0 && root_idx < static_cast<int>(nodes.size())) {
            const auto& node = nodes[root_idx];
            if (node.world_aabb.is_valid())
                scene_aabb.expand(node.world_aabb);
        }
    }

    return scene_aabb;
}

bool ImporterScene::has_camera() const
{
    return visit_importer_nodes(
        *this,
        [&](const ImporterNode& node, const float4x4&)
        {
            return node.camera_index >= 0 && static_cast<size_t>(node.camera_index) < cameras.size();
        }
    );
}

int ImporterScene::add_framed_camera(float fov_degrees, float aspect, std::string_view name)
{
    for (auto& mesh : meshes)
        mesh.calculate_local_aabb();
    for (auto& curve : curves)
        curve.calculate_local_aabb();

    std::vector<AABB> bounds;
    visit_importer_nodes(
        *this,
        [&](const ImporterNode& node, const float4x4& world_from_node)
        {
            if (node.mesh_index >= 0 && static_cast<size_t>(node.mesh_index) < meshes.size()) {
                const auto& aabb = meshes[node.mesh_index].local_aabb;
                if (aabb.is_valid())
                    bounds.push_back(aabb.transform(world_from_node));
            }
            if (node.curve_index >= 0 && static_cast<size_t>(node.curve_index) < curves.size()) {
                const auto& aabb = curves[node.curve_index].local_aabb;
                if (aabb.is_valid())
                    bounds.push_back(aabb.transform(world_from_node));
            }
            return false;
        }
    );
    const auto framing = detail::frame_camera(bounds, fov_degrees, aspect);

    ImporterCamera camera;
    camera.name = name;
    camera.sensor_size_mm = 24.f;
    camera.focal_length = ImporterCamera::focal_length_from_fov_degrees(fov_degrees, camera.sensor_size_mm);
    camera.focus_distance = framing.focus_distance;
    camera.depth_range = framing.depth_range;
    camera.projection = ImporterCamera::Projection::perspective;
    camera.fov_direction = ImporterCamera::FOVDirection::vertical;
    camera.enable_depth_of_field = false;

    const int camera_index = static_cast<int>(cameras.size());
    ImporterNode node;
    node.name = name;
    node.transform = framing.world_from_camera;
    node.camera_index = camera_index;
    cameras.push_back(std::move(camera));
    nodes.push_back(std::move(node));
    root_nodes.push_back(static_cast<int>(nodes.size()) - 1);
    return camera_index;
}

void ImporterScene::rescale_to_fit(float box_size)
{
    // First ensure AABBs are calculated
    calculate_aabbs();

    // Get the overall scene AABB
    AABB scene_aabb = get_scene_aabb();

    if (!scene_aabb.is_valid()) {
        // Scene has no valid geometry, nothing to rescale
        return;
    }

    // Calculate the current scene size and center
    float3 scene_size = scene_aabb.size();
    float3 scene_center = scene_aabb.center();

    // Find the maximum dimension to maintain aspect ratio
    float max_dimension = sgl::math::max(scene_size.x, sgl::math::max(scene_size.y, scene_size.z));

    if (max_dimension <= 0.0f) {
        // Degenerate case - scene has no size
        return;
    }

    // Calculate uniform scale factor
    float scale = box_size / max_dimension;

    // Create the transformation matrix:
    // 1. Translate to origin (center the scene at origin)
    // 2. Apply uniform scale
    float4x4 newtransform_from_transform
        = mul(matrix_from_scaling(float3(scale)), matrix_from_translation(-scene_center));

    // Apply the transformation to all root nodes
    for (int root_idx : root_nodes) {
        if (root_idx >= 0 && root_idx < static_cast<int>(nodes.size())) {
            auto& node = nodes[root_idx];
            node.transform = sgl::math::mul(newtransform_from_transform, node.transform);
        }
    }

    // Recalculate AABBs after transformation
    calculate_aabbs();
}

void ImporterScene::flatten_node_hierarchy()
{
    if (nodes.empty()) {
        return;
    }

    // Flattening discards joint-only nodes, so existing skin bindings would reference nodes that no longer exist.
    // Treat skins like the rest of the discarded hierarchy data and leave the flattened scene explicitly unskinned.
    skins.clear();

    // Clear any existing flattened nodes
    std::vector<ImporterNode> original_nodes = std::move(nodes);
    std::vector<int> original_root_nodes = std::move(root_nodes);

    // Helper function to recursively accumulate transforms and create flattened nodes
    std::function<void(int, const float4x4&)> flatten_node_recursive
        = [&](int node_idx, const float4x4& accumulated_transform)
    {
        if (node_idx < 0 || node_idx >= static_cast<int>(original_nodes.size())) {
            SGL_THROW("Invalid node index during flattening");
        }

        const auto& node = original_nodes[node_idx];

        // Accumulate the transform (parent * current)
        float4x4 world_transform = sgl::math::mul(accumulated_transform, node.transform);

        // If this node references something (like a mesh), create a flattened node
        if (node.mesh_index >= 0 || node.curve_index >= 0 || node.camera_index >= 0 || node.light_index >= 0) {
            ImporterNode flattened_node;
            flattened_node.name = node.name;
            flattened_node.transform = world_transform;
            flattened_node.mesh_index = node.mesh_index;
            flattened_node.curve_index = node.curve_index;
            flattened_node.camera_index = node.camera_index;
            flattened_node.light_index = node.light_index;
            flattened_node.parent = -1; // Flattened nodes have no hierarchy
            flattened_node.animation_translation = node.animation_translation;
            flattened_node.animation_rotation = node.animation_rotation;
            flattened_node.animation_scale = node.animation_scale;
            // Note: children are not copied since this is a flattened representation

            nodes.push_back(std::move(flattened_node));
            root_nodes.push_back(static_cast<int>(nodes.size()) - 1);
        }

        // Recursively process children
        for (int child_idx : node.children) {
            flatten_node_recursive(child_idx, world_transform);
        }

        // Process instanced prototype
        if (node.prototype_index >= 0) {
            flatten_node_recursive(prototypes[node.prototype_index].root_node, world_transform);
        }
    };

    // Start flattening from root nodes with identity transform
    float4x4 identity = float4x4::identity();
    for (int root_idx : original_root_nodes) {
        flatten_node_recursive(root_idx, identity);
    }
}

void ImporterScene::make_clay_scene()
{
    // Remove all materials.
    materials.clear();
    textures.clear();

    // Add clay material.
    Properties props;
    props.set("type", "gltf_pbrMetallicRoughness");
    props.set("baseColorFactor", float4(0.8f, 0.8f, 0.8f, 1.f));
    materials.push_back(ImporterMaterial{.name = "Clay", .params = props});

    // Set all geometry to use the clay material.
    for (auto& mesh : meshes)
        for (auto& subgeo : mesh.subgeometries)
            subgeo.material_index = 0;
    for (auto& curve : curves)
        curve.material_index = 0;

    // Remove all lights.
    lights.clear();
    for (auto& node : nodes)
        node.light_index = -1;

    // Add constant light.
    lights.push_back(
        ImporterLight{
            .properties = ImporterLight::make_properties({}, ImporterLight::Type::constant),
            .name = "ConstantLight",
            .type = ImporterLight::Type::constant,
        }
    );
    nodes.push_back(
        ImporterNode{
            .name = "ConstantLight",
            .transform = float4x4::identity(),
            .light_index = 0,
        }
    );
    root_nodes.push_back(static_cast<int>(nodes.size()) - 1);
}

void ImporterMesh::add_normals_from_faces()
{
    std::vector<ImporterMeshDefaultAttribute> normals_to_create;
    for (const auto& attribute : m_attributes) {
        if (attribute.semantic == ImporterSemantic::position)
            normals_to_create.push_back({ImporterSemantic::normal, attribute.index});
    }
    ensure_attributes(normals_to_create);
    for (const auto& attribute : normals_to_create) {
        auto normals = normal_stream(attribute.index);
        auto positions = position_stream(attribute.index);

        // Reset all normals to zero
        std::fill(normals.begin(), normals.end(), float3(0.f));

        // Accumulate face normals into vertex normals
        for (const auto& subgeo : subgeometries) {
            for (const auto& tri : subgeo.indices) {
                const float3 v0 = positions[tri.x];
                const float3 v1 = positions[tri.y];
                const float3 v2 = positions[tri.z];

                float3 edge1 = v1 - v0;
                float3 edge2 = v2 - v0;
                float3 face_normal = sgl::math::cross(edge1, edge2);

                // Add the face normal to each vertex normal
                normals[tri.x] += face_normal;
                normals[tri.y] += face_normal;
                normals[tri.z] += face_normal;
            }
        }

        // Normalize all vertex normals
        for (float3& normal : normals)
            normal = normalize_or(normal, float3(0, 0, 1));
    }
}


void ImporterMesh::add_tangents_from_uvs()
{
    if (auto cache = importer_cache())
        mikkt_generate_tangent_space(*this, *cache);
    else
        mikkt_generate_tangent_space(*this);
}

void ImporterMesh::add_tangents_from_normals()
{
    SGL_CHECK(normal_attribute().valid(), "Cannot generate tangents from normals when normals do not exist");
    ensure_attributes({{ImporterSemantic::tangent}, {ImporterSemantic::handedness}});

    auto normals = normal_stream();
    auto tangents = tangent_stream();
    auto handedness = handedness_stream();

    for (size_t i = 0; i < m_vertex_count; i++) {
        tangents[i] = branchlessONB(normals[i]);
        handedness[i] = 1.f; // Default handedness
    }
    update_motion_tangents();
}

void ImporterMesh::update_motion_tangents()
{
    std::vector<ImporterMeshDefaultAttribute> tangents_to_create;
    for (const auto& attribute : m_attributes) {
        if (attribute.semantic == ImporterSemantic::position && attribute.index != 0) {
            FALCOR_CHECK(find_attribute(ImporterSemantic::normal, attribute.index), "Motion sample is missing normals");
            tangents_to_create.push_back({ImporterSemantic::tangent, attribute.index});
        }
    }
    if (tangents_to_create.empty())
        return;
    FALCOR_ASSERT(position_attribute().valid());
    FALCOR_ASSERT(normal_attribute().valid());
    FALCOR_ASSERT(tangent_attribute().valid());
    ensure_attributes(tangents_to_create);

    // One reference frame and topology serve every sample; only the per-vertex scratch is overwritten.
    std::vector<uint3> triangles;
    for (const auto& subgeometry : subgeometries)
        triangles.insert(triangles.end(), subgeometry.indices.begin(), subgeometry.indices.end());
    std::vector<float3> positions(m_vertex_count), normals(m_vertex_count), tangents(m_vertex_count);
    const auto reference_positions = position_stream();
    const auto reference_normals = normal_stream();
    const auto reference_tangents = tangent_stream();
    for (size_t i = 0; i < m_vertex_count; ++i) {
        positions[i] = reference_positions[i];
        normals[i] = normalize_or(reference_normals[i], float3(0, 0, 1));
        tangents[i] = reference_tangents[i];
    }
    TangentTransfer transfer(triangles, positions, normals, tangents);
    std::vector<float3> sampled_positions(m_vertex_count), sampled_normals(m_vertex_count),
        sampled_tangents(m_vertex_count);
    for (const auto& attribute : tangents_to_create) {
        const auto p = position_stream(attribute.index);
        const auto n = normal_stream(attribute.index);
        auto t = tangent_stream(attribute.index);
        for (size_t i = 0; i < m_vertex_count; ++i) {
            sampled_positions[i] = p[i];
            sampled_normals[i] = normalize_or(n[i], float3(0, 0, 1));
        }
        transfer.evaluate(sampled_positions, sampled_normals, sampled_tangents);
        std::copy(sampled_tangents.begin(), sampled_tangents.end(), t.begin());
    }
}

namespace {

using ExactVertexKey = std::array<uint32_t, 3>;
using ExactTriangleKey = std::array<ExactVertexKey, 3>;

// Canonical position order identifies coincident triangles; permutation parity distinguishes their windings.
bool make_triangle_key(
    ImporterVertexStream<const float3> positions,
    const uint3& indices,
    ExactTriangleKey& key,
    bool& odd_winding
)
{
    for (int corner = 0; corner < 3; ++corner) {
        if (indices[corner] >= positions.size)
            return false;
        const float3 position = positions[indices[corner]];
        for (int axis = 0; axis < 3; ++axis) {
            if (!std::isfinite(position[axis]))
                return false;
            // Positive and negative zero describe the same authored position.
            key[corner][axis] = std::bit_cast<uint32_t>(position[axis] == 0.f ? 0.f : position[axis]);
        }
    }

    const ExactTriangleKey authored_order = key;
    std::sort(key.begin(), key.end());
    if (key[0] == key[1] || key[1] == key[2])
        return false;

    uint32_t inversion_count = 0;
    for (size_t i = 0; i < authored_order.size(); ++i) {
        for (size_t j = i + 1; j < authored_order.size(); ++j)
            inversion_count += authored_order[j] < authored_order[i];
    }
    odd_winding = (inversion_count & 1) != 0;
    return true;
}

// Compact triangle indices in place, preserving their order and removing any emptied subgeometries.
template<typename Predicate>
size_t remove_triangles_if(std::vector<ImporterMesh::Subgeometry>& subgeometries, Predicate remove)
{
    size_t removed_triangle_count = 0;
    for (size_t subgeometry_index = 0; subgeometry_index < subgeometries.size(); ++subgeometry_index) {
        auto& subgeometry = subgeometries[subgeometry_index];
        size_t output_index = 0;
        for (size_t triangle_index = 0; triangle_index < subgeometry.indices.size(); ++triangle_index) {
            if (remove(subgeometry_index, triangle_index)) {
                ++removed_triangle_count;
                continue;
            }
            subgeometry.indices[output_index++] = subgeometry.indices[triangle_index];
        }
        subgeometry.indices.resize(output_index);
    }
    if (removed_triangle_count > 0) {
        std::erase_if(
            subgeometries,
            [](const ImporterMesh::Subgeometry& subgeometry)
            {
                return subgeometry.indices.empty();
            }
        );
    }
    return removed_triangle_count;
}

} // namespace

size_t ImporterMesh::collapse_exact_opposite_triangles()
{
    // Coincident positions can separate under skinning when their influences differ.
    for (const ImporterMeshAttribute& attribute : attributes()) {
        if (attribute.semantic == ImporterSemantic::joints || attribute.semantic == ImporterSemantic::weights)
            return 0;
    }

    const auto positions = position_stream();
    if (!positions.valid())
        return 0;

    struct Candidate {
        size_t subgeometry_index;
        size_t triangle_index;
        uint3 retained_vertices;
    };
    std::vector<Candidate> candidates;
    {
        struct RetainedTriangle {
            uint3 vertices;
            bool odd_winding;
        };
        // Only this mesh's local positions and material indices participate.
        std::map<std::pair<ExactTriangleKey, int>, RetainedTriangle> retained_triangles;
        for (size_t subgeometry_index = 0; subgeometry_index < subgeometries.size(); ++subgeometry_index) {
            const auto& subgeometry = subgeometries[subgeometry_index];
            for (size_t triangle_index = 0; triangle_index < subgeometry.indices.size(); ++triangle_index) {
                const uint3 indices = subgeometry.indices[triangle_index];
                ExactTriangleKey key;
                bool odd_winding = false;
                if (!make_triangle_key(positions, indices, key, odd_winding))
                    continue;
                const auto [it, inserted] = retained_triangles.try_emplace(
                    std::make_pair(key, subgeometry.material_index),
                    RetainedTriangle{indices, odd_winding}
                );
                // Keep the first authored winding and all duplicates of that winding.
                if (inserted || it->second.odd_winding == odd_winding)
                    continue;

                // Fix corner correspondence at sample zero; later samples must follow the same vertices.
                Candidate candidate{subgeometry_index, triangle_index, {}};
                for (int corner = 0; corner < 3; ++corner) {
                    for (int retained_corner = 0; retained_corner < 3; ++retained_corner) {
                        const uint32_t vertex = it->second.vertices[retained_corner];
                        if (sgl::math::all(positions[indices[corner]] == positions[vertex])) {
                            candidate.retained_vertices[corner] = vertex;
                            break;
                        }
                    }
                }
                candidates.push_back(candidate);
            }
        }
    }

    if (candidates.empty())
        return 0;
    for (const auto& attribute : attributes()) {
        if (attribute.semantic != ImporterSemantic::position || attribute.index == 0)
            continue;
        const auto sample = get_stream<float3>(attribute);
        std::erase_if(
            candidates,
            [&](const Candidate& candidate)
            {
                const uint3 indices = subgeometries[candidate.subgeometry_index].indices[candidate.triangle_index];
                for (int corner = 0; corner < 3; ++corner) {
                    const float3 position = sample[indices[corner]];
                    const float3 retained_position = sample[candidate.retained_vertices[corner]];
                    for (int axis = 0; axis < 3; ++axis) {
                        if (!std::isfinite(position[axis]) || position[axis] != retained_position[axis])
                            return true;
                    }
                }
                return false;
            }
        );
        if (candidates.empty())
            return 0;
    }

    // Filtering preserves candidate order, so a single cursor suffices during compaction.
    auto candidate = candidates.begin();
    return remove_triangles_if(
        subgeometries,
        [&](size_t subgeometry_index, size_t triangle_index)
        {
            if (candidate != candidates.end() && candidate->subgeometry_index == subgeometry_index
                && candidate->triangle_index == triangle_index) {
                ++candidate;
                return true;
            }
            return false;
        }
    );
}

void ImporterMesh::deduplicate_vertices()
{
    // TODO: Replacing vertex buffers below can invalidate non-owning Python stream views. Move the buffers to
    // allocation-owned copy-on-write storage before treating topology-changing operations as generally view-safe.
    std::vector<VertexRemapStream> streams;
    streams.reserve(m_attributes.size());
    uint32_t previous_buffer = 0;
    size_t previous_end = 0;
    for (const ImporterMeshAttribute& attribute : m_attributes) {
        const void* data = nullptr;
        size_t stride = 0;
        get_stream_info(attribute, data, stride);
        const size_t element_size = data_type_size(attribute.type) * attribute.num_components;
        // Adjacent occupied ranges have the same bytewise equality as their individual attributes. Keep gaps
        // and separate buffers as separate streams so padding never participates in vertex identity.
        if (!streams.empty() && attribute.buffer == previous_buffer && attribute.offset == previous_end) {
            streams.back().element_size += element_size;
            previous_end += element_size;
            continue;
        }
        streams.push_back(
            VertexRemapStream{
                .data = static_cast<const std::byte*>(data),
                .element_size = element_size,
                .stride = stride,
            }
        );
        previous_buffer = attribute.buffer;
        previous_end = attribute.offset + element_size;
    }

    const VertexRemap remap = generate_vertex_remap(streams, m_vertex_count);
    apply_vertex_remap(remap);
}

void ImporterMesh::apply_vertex_remap(const VertexRemap& remap)
{
    FALCOR_ASSERT(remap.old_to_new.size() == m_vertex_count);
    if (remap.is_identity())
        return;

    for (Subgeometry& subgeometry : subgeometries) {
        for (uint3& triangle : subgeometry.indices) {
            for (int corner = 0; corner < 3; ++corner)
                triangle[corner] = remap.old_to_new[triangle[corner]];
        }
    }

    std::vector<ImporterVertexBuffer> new_buffers(m_buffers.size());
    for (size_t buffer_index = 0; buffer_index < m_buffers.size(); ++buffer_index) {
        const ImporterVertexBuffer& old_buffer = m_buffers[buffer_index];
        ImporterVertexBuffer& new_buffer = new_buffers[buffer_index];
        new_buffer.stride = old_buffer.stride;
        new_buffer.data.resize(remap.new_to_old.size() * new_buffer.stride);
        for (size_t new_index = 0; new_index < remap.new_to_old.size(); ++new_index) {
            const size_t old_index = remap.new_to_old[new_index];
            std::memcpy(
                new_buffer.data.data() + new_index * new_buffer.stride,
                old_buffer.data.data() + old_index * old_buffer.stride,
                old_buffer.stride
            );
        }
    }

    m_vertex_count = remap.new_to_old.size();
    m_buffers = std::move(new_buffers);
}

void ImporterMesh::add_attribute(const ImporterMeshAttribute& attrib)
{
    SGL_CHECK(attrib.valid(), "Cannot add invalid attribute");
    if (attrib.buffer < m_buffers.size()) {
        SGL_CHECK(
            m_buffers[attrib.buffer].stride == 0,
            "Attribute buffer has already been created, new attributes cannot be added to it"
        );
    }

    size_t req_alignment = data_type_size(attrib.type);
    SGL_CHECK(attrib.offset % req_alignment == 0, "Attribute offset must be aligned to its type size");
    if (attrib.semantic == ImporterSemantic::position) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Position attribute must be float3"
        );
        while (m_positions.size() <= attrib.index)
            m_positions.emplace_back();
        FALCOR_CHECK(!position_attribute(attrib.index).valid(), "Position attribute is already set");
        m_positions[attrib.index] = attrib;
    } else if (attrib.semantic == ImporterSemantic::normal) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Normal attribute must be float3"
        );
        while (m_normals.size() <= attrib.index)
            m_normals.emplace_back();
        FALCOR_CHECK(!normal_attribute(attrib.index).valid(), "Normal attribute is already set");
        m_normals[attrib.index] = attrib;
    } else if (attrib.semantic == ImporterSemantic::tangent) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Tangent attribute must be float3"
        );
        while (m_tangents.size() <= attrib.index)
            m_tangents.emplace_back();
        FALCOR_CHECK(!tangent_attribute(attrib.index).valid(), "Tangent attribute is already set");
        m_tangents[attrib.index] = attrib;
    } else if (attrib.semantic == ImporterSemantic::handedness && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 1,
            "Handedness attribute must be float"
        );
        SGL_CHECK(!m_handedness.valid(), "Handedness attribute is already set");
        m_handedness = attrib;
    } else if (attrib.semantic == ImporterSemantic::tex_coord && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components >= 2,
            "Texcoord attribute must be float2/3/4"
        );
        SGL_CHECK(!m_texcoords.valid(), "Texcoord attribute is already set");
        m_texcoords = attrib;
    } else if (attrib.semantic == ImporterSemantic::color && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && (attrib.num_components == 3 || attrib.num_components == 4),
            "Color attribute must be float3/4"
        );
        SGL_CHECK(!m_colors.valid(), "Color attribute is already set");
        m_colors = attrib;
    }

    m_attributes.push_back(attrib);
}

void ImporterMesh::create_buffer(size_t buffer_index, size_t stride)
{
    if (buffer_index >= m_buffers.size()) {
        m_buffers.resize(buffer_index + 1);
    }
    SGL_CHECK(m_buffers[buffer_index].stride == 0, "Buffer has already been created");

    // Calculate stride from attributes in this buffer
    size_t max_offset = 0;
    size_t max_alignment = 1;
    for (const auto& attrib : m_attributes) {
        if (attrib.buffer == buffer_index) {
            size_t dtype_size = data_type_size(attrib.type);
            size_t attrib_end = attrib.offset + attrib.num_components * dtype_size;
            if (attrib_end > max_offset) {
                max_offset = attrib_end;
            }
            max_alignment = std::max(max_alignment, dtype_size);
        }
    }
    SGL_CHECK(max_offset > 0, "Cannot create buffer with zero stride");

    // Either calculate or validate specified stride
    if (stride == 0) {
        stride = (max_offset + max_alignment - 1) / max_alignment * max_alignment;
    } else {
        SGL_CHECK(stride >= max_offset, "Specified stride is too small for attributes in this buffer");
        SGL_CHECK(stride % max_alignment == 0, "Specified stride must be aligned to largest attribute alignment");
    }

    // Set buffer stride
    m_buffers[buffer_index].stride = stride;

    // Allocate buffer data if adding a new buffer to existing mesh
    if (m_vertex_count > 0) {
        m_buffers[buffer_index].resize_vertices(m_vertex_count);
    }
}

void ImporterMesh::create_buffers_from_attributes()
{
    // Create buffers for all attributes that don't have a buffer yet
    for (const auto& attrib : m_attributes) {
        if (attrib.buffer >= m_buffers.size() || m_buffers[attrib.buffer].stride == 0) {
            create_buffer(attrib.buffer);
        }
    }
}

void ImporterMesh::ensure_attributes(std::span<ImporterMeshDefaultAttribute> attribs)
{
    size_t offset = 0;
    size_t buffer = m_buffers.size(); // New buffer if needed
    for (const auto& descriptor : attribs) {
        auto* attrib = find_attribute(descriptor);
        if (!attrib) {

            // Select defaults for semantic
            ImporterMeshAttribute new_attrib;
            switch (descriptor.semantic) {
            case ImporterSemantic::unknown:
                SGL_CHECK(!descriptor.name.empty(), "Cannot ensure an unknown attribute without a name");
                SGL_CHECK(descriptor.components > 0, "Cannot ensure a named attribute without a component count");
                new_attrib.name = descriptor.name;
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = descriptor.components;
                break;
            case ImporterSemantic::position:
                new_attrib.name = "POSITION";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 3;
                break;
            case ImporterSemantic::normal:
                new_attrib.name = "NORMAL";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 3;
                break;
            case ImporterSemantic::tangent:
                new_attrib.name = "TANGENT";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 3;
                break;
            case ImporterSemantic::handedness:
                new_attrib.name = "HANDNESS";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 1;
                break;
            case ImporterSemantic::tex_coord:
                new_attrib.name = "TEXCOORD";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 2;
                break;
            case ImporterSemantic::color:
                new_attrib.name = "COLOR";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 4;
                break;
            case ImporterSemantic::joints:
                new_attrib.name = "JOINTS";
                new_attrib.type = sgl::DataType::int32;
                new_attrib.num_components = 4;
                break;
            case ImporterSemantic::weights:
                new_attrib.name = "WEIGHTS";
                new_attrib.type = sgl::DataType::float32;
                new_attrib.num_components = 4;
                break;
            default:
                SGL_THROW("Cannot ensure attribute for unknown semantic");
                break;
            }

            // Override requested components if specified.
            if (descriptor.components != 0) {
                new_attrib.num_components = descriptor.components;
            }

            if (descriptor.type != sgl::DataType::void_)
                new_attrib.type = descriptor.type;

            // Align offset to attribute type size
            size_t dtype_size = data_type_size(new_attrib.type);
            if (offset % dtype_size != 0) {
                offset = (offset + dtype_size - 1) / dtype_size * dtype_size;
            }

            // Add attribute
            if (descriptor.semantic != ImporterSemantic::unknown)
                new_attrib.name += "_" + std::to_string(descriptor.index);
            new_attrib.buffer = (uint32_t)buffer;
            new_attrib.offset = (uint32_t)offset;
            new_attrib.semantic = descriptor.semantic;
            new_attrib.index = descriptor.index;
            add_attribute(new_attrib);

            // Advance offset
            offset += new_attrib.num_components * dtype_size;
        }
    }

    // If any attributes created, create the new buffer
    if (offset > 0) {
        create_buffer(buffer);
    }
}

size_t ImporterMesh::allocate_vertices(size_t count, bool init_to_zero)
{
    size_t start_index = m_vertex_count;
    m_vertex_count += count;
    for (auto& buffer : m_buffers) {
        buffer.resize_vertices(m_vertex_count);
        if (init_to_zero) {
            std::fill(buffer.data.begin() + start_index * buffer.stride, buffer.data.end(), (uint8_t)0);
        }
    }
    return start_index;
}

} // namespace falcor
