// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/importer.h"
#include "falcor2/importers/mikkt.h"

#include "falcor2/core/types.h"
#include "falcor2/utils/fnv_hash.h"
#include "falcor2/utils/indexed_vector.h"

#include <algorithm>
#include <array>
#include <limits>

namespace falcor {

namespace {

// https://graphics.pixar.com/library/OrthonormalB/paper.pdf
float3 branchlessONB(float3 normal)
{
    float sign = std::copysign(1.f, normal.z);
    const float a = -1.0f / (sign + normal.z);
    const float b = normal.x * normal.y * a;
    return float3(1.0f + sign * normal.x * normal.x * a, sign * b, -sign * normal.x);
}

constexpr float DEFAULT_SENSOR_HEIGHT_MM = 24.f;

float compute_focal_length_from_fov_y(float fov_degrees, float sensor_height_mm)
{
    const float fov_rad = sgl::math::radians(fov_degrees);
    return sensor_height_mm / (2.f * sgl::math::tan(fov_rad * 0.5f));
}

float compute_percentile(std::vector<float>& values, float percentile)
{
    if (values.empty()) {
        return 0.f;
    }

    percentile = std::clamp(percentile, 0.f, 1.f);
    std::sort(values.begin(), values.end());

    if (values.size() == 1) {
        return values.front();
    }

    const float f_index = percentile * static_cast<float>(values.size() - 1);
    const size_t index0 = static_cast<size_t>(std::floor(f_index));
    const size_t index1 = std::min(index0 + 1, values.size() - 1);
    const float t = f_index - static_cast<float>(index0);
    return values[index0] * (1.f - t) + values[index1] * t;
}

std::array<float3, 8> aabb_corners(const AABB& aabb)
{
    return {
        float3(aabb.min.x, aabb.min.y, aabb.min.z),
        float3(aabb.max.x, aabb.min.y, aabb.min.z),
        float3(aabb.min.x, aabb.max.y, aabb.min.z),
        float3(aabb.max.x, aabb.max.y, aabb.min.z),
        float3(aabb.min.x, aabb.min.y, aabb.max.z),
        float3(aabb.max.x, aabb.min.y, aabb.max.z),
        float3(aabb.min.x, aabb.max.y, aabb.max.z),
        float3(aabb.max.x, aabb.max.y, aabb.max.z),
    };
}

std::vector<AABB> collect_mesh_instance_world_aabbs(const ImporterScene& scene)
{
    std::vector<AABB> result;

    std::function<void(int, const float4x4&)> collect_recursive = [&](int node_idx, const float4x4& world_from_parent)
    {
        if (node_idx < 0 || node_idx >= static_cast<int>(scene.nodes.size())) {
            return;
        }

        const auto& node = scene.nodes[node_idx];
        const float4x4 world_from_node = sgl::math::mul(world_from_parent, node.transform);

        if (node.mesh_index >= 0 && node.mesh_index < static_cast<int>(scene.meshes.size())) {
            const auto& mesh = scene.meshes[node.mesh_index];
            if (mesh.local_aabb.is_valid()) {
                result.push_back(mesh.local_aabb.transform(world_from_node));
            }
        }

        for (int child_idx : node.children) {
            collect_recursive(child_idx, world_from_node);
        }

        if (node.prototype_index >= 0 && node.prototype_index < static_cast<int>(scene.prototypes.size())) {
            collect_recursive(scene.prototypes[node.prototype_index].root_node, world_from_node);
        }
    };

    const float4x4 identity = float4x4::identity();
    for (int root_idx : scene.root_nodes) {
        collect_recursive(root_idx, identity);
    }

    return result;
}

std::pair<AABB, float3> compute_robust_bounds_and_center(
    const std::vector<AABB>& mesh_world_aabbs,
    const AABB& fallback_scene_aabb,
    float outlier_percentile
)
{
    if (mesh_world_aabbs.empty()) {
        return {fallback_scene_aabb, fallback_scene_aabb.center()};
    }

    outlier_percentile = std::clamp(outlier_percentile, 0.f, 0.49f);

    std::vector<float> mins_x, mins_y, mins_z;
    std::vector<float> maxs_x, maxs_y, maxs_z;
    std::vector<float> centers_x, centers_y, centers_z;

    mins_x.reserve(mesh_world_aabbs.size());
    mins_y.reserve(mesh_world_aabbs.size());
    mins_z.reserve(mesh_world_aabbs.size());
    maxs_x.reserve(mesh_world_aabbs.size());
    maxs_y.reserve(mesh_world_aabbs.size());
    maxs_z.reserve(mesh_world_aabbs.size());
    centers_x.reserve(mesh_world_aabbs.size());
    centers_y.reserve(mesh_world_aabbs.size());
    centers_z.reserve(mesh_world_aabbs.size());

    for (const auto& aabb : mesh_world_aabbs) {
        mins_x.push_back(aabb.min.x);
        mins_y.push_back(aabb.min.y);
        mins_z.push_back(aabb.min.z);
        maxs_x.push_back(aabb.max.x);
        maxs_y.push_back(aabb.max.y);
        maxs_z.push_back(aabb.max.z);

        const float3 c = aabb.center();
        centers_x.push_back(c.x);
        centers_y.push_back(c.y);
        centers_z.push_back(c.z);
    }

    AABB robust_aabb;
    robust_aabb.min = float3(
        compute_percentile(mins_x, outlier_percentile),
        compute_percentile(mins_y, outlier_percentile),
        compute_percentile(mins_z, outlier_percentile)
    );
    robust_aabb.max = float3(
        compute_percentile(maxs_x, 1.f - outlier_percentile),
        compute_percentile(maxs_y, 1.f - outlier_percentile),
        compute_percentile(maxs_z, 1.f - outlier_percentile)
    );

    if (!robust_aabb.is_valid()) {
        robust_aabb = fallback_scene_aabb;
    }

    float3 robust_center = float3(
        compute_percentile(centers_x, 0.5f),
        compute_percentile(centers_y, 0.5f),
        compute_percentile(centers_z, 0.5f)
    );

    return {robust_aabb, robust_center};
}

float3 choose_safe_up(float3 view_dir)
{
    float3 up = float3(0.f, 1.f, 0.f);
    if (std::abs(sgl::math::dot(view_dir, up)) > 0.98f) {
        up = float3(0.f, 0.f, 1.f);
    }
    return up;
}

int append_camera_and_node(
    ImporterScene& scene,
    std::string_view camera_name,
    std::string_view node_name,
    const float3& cam_pos,
    const float3& target,
    const float3& up,
    float fov_degrees,
    float focus_distance,
    float near_plane,
    float far_plane
)
{
    ImporterCamera camera;
    camera.name = std::string(camera_name);
    camera.focus_distance = std::max(focus_distance, 0.01f);
    camera.focal_length = compute_focal_length_from_fov_y(fov_degrees, DEFAULT_SENSOR_HEIGHT_MM);
    camera.fstop = 8.f;
    camera.depth_range = float2(std::max(near_plane, 0.001f), std::max(far_plane, near_plane + 0.1f));
    camera.projection = ImporterCamera::Projection::perspective;
    camera.fov_direction = ImporterCamera::FOVDirection::vertical;
    camera.aperture = DEFAULT_SENSOR_HEIGHT_MM;

    const int camera_index = static_cast<int>(scene.cameras.size());
    scene.cameras.push_back(std::move(camera));

    ImporterNode camera_node;
    camera_node.name = std::string(node_name);
    camera_node.transform = sgl::math::inverse(sgl::math::matrix_from_look_at(cam_pos, target, up));
    camera_node.camera_index = camera_index;
    camera_node.parent = -1;

    scene.nodes.push_back(std::move(camera_node));
    scene.root_nodes.push_back(static_cast<int>(scene.nodes.size()) - 1);
    return camera_index;
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

float ImporterAnimation::duration() const
{
    float max_time = 0.f;
    for (const auto& channel : channels) {
        if (!channel.times.empty()) {
            max_time = std::max(max_time, channel.times.back());
        }
    }
    return max_time;
}

void ImporterCurve::calculate_local_aabb()
{
    local_aabb = AABB();
    for (size_t i = 0; i < positions.size(); ++i) {
        float r = i < radii.size() ? radii[i] : 0.f;
        local_aabb.expand(positions[i], r);
    }
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

int ImporterScene::add_default_camera_robust_fit(
    float fov_degrees,
    float aspect,
    float coverage,
    float outlier_percentile
)
{
    aspect = std::max(aspect, 1e-3f);
    coverage = std::clamp(coverage, 0.1f, 0.99f);

    calculate_aabbs();

    AABB scene_aabb = get_scene_aabb();
    if (!scene_aabb.is_valid()) {
        const float3 target = float3(0.f);
        const float3 view_dir = float3(0.f, 0.f, -1.f);
        const float distance = 5.f;
        const float3 cam_pos = target - view_dir * distance;
        return append_camera_and_node(
            *this,
            "auto_camera_robust_fit",
            "AutoCamera_RobustFit",
            cam_pos,
            target,
            choose_safe_up(view_dir),
            fov_degrees,
            distance,
            0.01f,
            1000.f
        );
    }

    const std::vector<AABB> mesh_world_aabbs = collect_mesh_instance_world_aabbs(*this);
    auto [robust_aabb, robust_center]
        = compute_robust_bounds_and_center(mesh_world_aabbs, scene_aabb, outlier_percentile);

    const float3 extent = robust_aabb.max - robust_aabb.min;
    const float half_width = std::max(extent.x * 0.5f, 1e-3f);
    const float half_height = std::max(extent.y * 0.5f, 1e-3f);

    const float vfov_rad = sgl::math::radians(fov_degrees);
    const float tan_half_vfov = sgl::math::tan(vfov_rad * 0.5f);
    const float tan_half_hfov = tan_half_vfov * aspect;

    const float dist_horiz = half_width / (tan_half_hfov * coverage);
    const float dist_vert = half_height / (tan_half_vfov * coverage);
    const float distance = std::max(dist_horiz, dist_vert);

    const float3 view_dir = float3(0.f, 0.f, -1.f);
    const float3 cam_pos = robust_center - view_dir * distance;

    const float3 scene_extent = scene_aabb.size();
    const float scene_radius = std::max(0.5f * sgl::math::length(scene_extent), 1.f);
    const float near_plane = std::max(0.01f, distance - scene_radius);
    const float far_plane = distance + scene_radius;

    return append_camera_and_node(
        *this,
        "auto_camera_robust_fit",
        "AutoCamera_RobustFit",
        cam_pos,
        robust_center,
        choose_safe_up(view_dir),
        fov_degrees,
        distance,
        near_plane,
        far_plane
    );
}

int ImporterScene::add_default_camera_best_view(
    float fov_degrees,
    float aspect,
    float coverage,
    float outlier_percentile
)
{
    aspect = std::max(aspect, 1e-3f);
    coverage = std::clamp(coverage, 0.1f, 0.99f);

    calculate_aabbs();

    AABB scene_aabb = get_scene_aabb();
    if (!scene_aabb.is_valid()) {
        const float3 target = float3(0.f);
        const float3 view_dir = float3(0.f, 0.f, -1.f);
        const float distance = 5.f;
        const float3 cam_pos = target - view_dir * distance;
        return append_camera_and_node(
            *this,
            "auto_camera_best_view",
            "AutoCamera_BestView",
            cam_pos,
            target,
            choose_safe_up(view_dir),
            fov_degrees,
            distance,
            0.01f,
            1000.f
        );
    }

    const std::vector<AABB> mesh_world_aabbs = collect_mesh_instance_world_aabbs(*this);
    auto [robust_aabb, robust_center]
        = compute_robust_bounds_and_center(mesh_world_aabbs, scene_aabb, outlier_percentile);

    std::vector<float3> sample_points;
    sample_points.reserve(std::max<size_t>(8, mesh_world_aabbs.size() * 8));
    if (!mesh_world_aabbs.empty()) {
        for (const auto& aabb : mesh_world_aabbs) {
            for (const auto& c : aabb_corners(aabb)) {
                sample_points.push_back(c);
            }
        }
    } else {
        for (const auto& c : aabb_corners(robust_aabb)) {
            sample_points.push_back(c);
        }
    }

    const float vfov_rad = sgl::math::radians(fov_degrees);
    const float tan_half_vfov = sgl::math::tan(vfov_rad * 0.5f);
    const float tan_half_hfov = tan_half_vfov * aspect;

    const std::array<float3, 14> candidate_dirs = {
        sgl::math::normalize(float3(0.f, 0.f, -1.f)),
        sgl::math::normalize(float3(0.f, 0.f, 1.f)),
        sgl::math::normalize(float3(1.f, 0.f, 0.f)),
        sgl::math::normalize(float3(-1.f, 0.f, 0.f)),
        sgl::math::normalize(float3(0.707f, 0.f, -0.707f)),
        sgl::math::normalize(float3(-0.707f, 0.f, -0.707f)),
        sgl::math::normalize(float3(0.707f, 0.f, 0.707f)),
        sgl::math::normalize(float3(-0.707f, 0.f, 0.707f)),
        sgl::math::normalize(float3(0.5f, 0.5f, -0.707f)),
        sgl::math::normalize(float3(-0.5f, 0.5f, -0.707f)),
        sgl::math::normalize(float3(0.5f, -0.5f, -0.707f)),
        sgl::math::normalize(float3(-0.5f, -0.5f, -0.707f)),
        sgl::math::normalize(float3(0.f, 0.5f, -0.866f)),
        sgl::math::normalize(float3(0.f, -0.5f, -0.866f)),
    };

    float best_score = -std::numeric_limits<float>::infinity();
    float best_distance = 5.f;
    float best_depth_min = 0.f;
    float best_depth_max = 0.f;
    float3 best_view_dir = float3(0.f, 0.f, -1.f);

    for (const float3& view_dir : candidate_dirs) {
        const float3 up = choose_safe_up(view_dir);
        const float3 right = sgl::math::normalize(sgl::math::cross(up, view_dir));
        const float3 camera_up = sgl::math::normalize(sgl::math::cross(view_dir, right));

        float max_abs_x = 0.f;
        float max_abs_y = 0.f;
        float min_z = std::numeric_limits<float>::infinity();
        float max_z = -std::numeric_limits<float>::infinity();

        for (const float3& p : sample_points) {
            const float3 q = p - robust_center;
            const float x = sgl::math::dot(q, right);
            const float y = sgl::math::dot(q, camera_up);
            const float z = sgl::math::dot(q, view_dir);
            max_abs_x = std::max(max_abs_x, std::abs(x));
            max_abs_y = std::max(max_abs_y, std::abs(y));
            min_z = std::min(min_z, z);
            max_z = std::max(max_z, z);
        }

        const float dist_fit_x = max_abs_x / (tan_half_hfov * coverage);
        const float dist_fit_y = max_abs_y / (tan_half_vfov * coverage);
        const float dist_visibility = -min_z + 1e-3f;
        const float distance = std::max(std::max(dist_fit_x, dist_fit_y), dist_visibility);

        if (!(distance > 0.f) || !std::isfinite(distance)) {
            continue;
        }

        const float fill_x = max_abs_x / std::max(tan_half_hfov * distance, 1e-6f);
        const float fill_y = max_abs_y / std::max(tan_half_vfov * distance, 1e-6f);
        const float fill = std::max(fill_x, fill_y);
        const float depth_span = (max_z - min_z) / std::max(distance, 1e-4f);
        const float score = fill - 0.15f * depth_span;

        if (score > best_score) {
            best_score = score;
            best_distance = distance;
            best_depth_min = min_z;
            best_depth_max = max_z;
            best_view_dir = view_dir;
        }
    }

    const float3 cam_pos = robust_center - best_view_dir * best_distance;
    const float near_plane = std::max(0.01f, best_distance + best_depth_min);
    const float far_plane = std::max(near_plane + 0.1f, best_distance + best_depth_max);

    return append_camera_and_node(
        *this,
        "auto_camera_best_view",
        "AutoCamera_BestView",
        cam_pos,
        robust_center,
        choose_safe_up(best_view_dir),
        fov_degrees,
        best_distance,
        near_plane,
        far_plane
    );
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

    // Set all subgeometries to use the clay material.
    for (auto& mesh : meshes)
        for (auto& subgeo : mesh.subgeometries)
            subgeo.material_name = "Clay";

    // Remove all lights.
    lights.clear();
    for (auto& node : nodes)
        node.light_index = -1;

    // Add constant light.
    lights.push_back(
        ImporterLight{
            .name = "ConstantLight",
            .type = ImporterLight::Type::constant,
            .intensity = float3(1.f),
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
    // Make sure we have normals
    ensure_attributes({{ImporterSemantic::normal}});

    auto normals = normal_stream();
    auto positions = position_stream();

    // Reset all normals to zero
    for (size_t i = 0; i < m_vertex_count; i++) {
        normals[i] = float3(0.f);
    }

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
    for (size_t i = 0; i < m_vertex_count; i++) {
        normals[i] = sgl::math::normalize(normals[i]);
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
    SGL_CHECK(m_normals.valid(), "Cannot generate tangents from normals when normals do not exist");
    ensure_attributes({{ImporterSemantic::tangent}, {ImporterSemantic::handedness}});

    auto normals = normal_stream();
    auto tangents = tangent_stream();
    auto handedness = handedness_stream();

    for (size_t i = 0; i < m_vertex_count; i++) {
        tangents[i] = branchlessONB(normals[i]);
        handedness[i] = 1.f; // Default handedness
    }
}

void ImporterMesh::deduplicate_vertices()
{
    // Build a mapping of vertex hash to new, deduplicated index and
    // Mapping of old vertex index to new vertex index
    std::unordered_map<size_t, size_t> hash_to_index; // vertex hash -> new index
    hash_to_index.reserve(m_vertex_count);
    std::vector<size_t> old_to_new_index(m_vertex_count, size_t(-1)); // old index -> new index
    std::vector<uint8_t> is_first_instance(m_vertex_count, false);
    size_t next_index = 0;
    for (size_t i = 0; i < m_vertex_count; i++) {
        size_t hash = hash_vertex(i);
        if (hash_to_index.find(hash) == hash_to_index.end()) {
            hash_to_index[hash] = next_index++;
            is_first_instance[i] = true;
        }
        old_to_new_index[i] = hash_to_index[hash];
    }

    // Early out if there are no duplicated vertices.
    if (next_index == m_vertex_count)
        return;

    // Remap the vertex indices in each subgeometry
    for (auto& subgeo : subgeometries) {
        for (auto& indices : subgeo.indices) {
            for (int i = 0; i < 3; ++i) {
                indices[i] = (uint32_t)old_to_new_index[indices[i]];
            }
        }
    }

    // Generate new buffers with deduplicated vertices
    std::vector<ImporterMeshBuffer> new_buffers(m_buffers.size());
    for (size_t buffer_idx = 0; buffer_idx < m_buffers.size(); buffer_idx++) {
        const auto& old_buffer = m_buffers[buffer_idx];
        auto& new_buffer = new_buffers[buffer_idx];
        new_buffer.stride = old_buffer.stride;
        new_buffer.data.resize(next_index * new_buffer.stride);
        for (size_t old_idx = 0; old_idx < m_vertex_count; old_idx++) {
            if (is_first_instance[old_idx]) {
                // Copy the old vertex data to the new buffer
                size_t new_idx = old_to_new_index[old_idx];
                std::memcpy(
                    &new_buffer.data[new_idx * new_buffer.stride],
                    &old_buffer.data[old_idx * old_buffer.stride],
                    old_buffer.stride
                );
            }
        }
    }

    m_vertex_count = next_index;
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
    if (attrib.semantic == ImporterSemantic::position && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Position attribute must be float3"
        );
        SGL_CHECK(!m_positions.valid(), "Position attribute is already set");
        m_positions = attrib;
    } else if (attrib.semantic == ImporterSemantic::normal && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Normal attribute must be float3"
        );
        SGL_CHECK(!m_normals.valid(), "Normal attribute is already set");
        m_normals = attrib;
    } else if (attrib.semantic == ImporterSemantic::tangent && attrib.index == 0) {
        SGL_CHECK(
            attrib.type == sgl::DataType::float32 && attrib.num_components == 3,
            "Tangent attribute must be float3"
        );
        SGL_CHECK(!m_tangents.valid(), "Tangent attribute is already set");
        m_tangents = attrib;
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
        m_buffers[buffer_index].data.resize(m_vertex_count * stride);
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
    for (const auto& sem_and_idx : attribs) {
        auto* attrib = find_attribute(sem_and_idx.semantic, sem_and_idx.index);
        if (!attrib) {

            // Select defaults for semantic
            ImporterMeshAttribute new_attrib;
            switch (sem_and_idx.semantic) {
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
                new_attrib.type = sgl::DataType::uint16;
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

            // Override requested componnents if specified
            if (sem_and_idx.components != 0) {
                new_attrib.num_components = sem_and_idx.components;
            }

            // Align offset to attribute type size
            size_t dtype_size = data_type_size(new_attrib.type);
            if (offset % dtype_size != 0) {
                offset = (offset + dtype_size - 1) / dtype_size * dtype_size;
            }

            // Add attribute
            new_attrib.name += "_" + std::to_string(sem_and_idx.index);
            new_attrib.buffer = (uint32_t)buffer;
            new_attrib.offset = (uint32_t)offset;
            new_attrib.semantic = sem_and_idx.semantic;
            new_attrib.index = sem_and_idx.index;
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
        buffer.data.resize(m_vertex_count * buffer.stride);
        if (init_to_zero) {
            std::fill(buffer.data.begin() + start_index * buffer.stride, buffer.data.end(), (uint8_t)0);
        }
    }
    return start_index;
}

size_t ImporterMesh::hash_vertex(size_t vertex_index)
{
    FNVHash64 hash;
    for (auto& attrib : m_attributes) {
        void* base_ptr;
        size_t stride;
        get_stream_info(attrib, base_ptr, stride);
        size_t size = data_type_size(attrib.type) * attrib.num_components;
        size_t offset = vertex_index * stride;
        const uint8_t* data_ptr = (const uint8_t*)base_ptr + offset;
        // Simple hash combine
        hash.insert(data_ptr, size);
    }
    return hash.get();
}

} // namespace falcor
