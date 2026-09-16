// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer_area_light_geometry.h"

#include "falcor2/importers/importer_types.h"

#include <sgl/core/logger.h>

#include <cmath>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

namespace falcor::detail {

namespace {

constexpr float PI = 3.14159265358979323846f;
constexpr uint32_t DISK_SEGMENT_COUNT = 64;
constexpr uint32_t SPHERE_LONGITUDE_SEGMENT_COUNT = 64;
constexpr uint32_t SPHERE_LATITUDE_SEGMENT_COUNT = 32;

struct MeshVertex {
    float3 position;
    float3 normal;
    float3 tangent;
    float handedness;
    float2 texcoord;
};

ImporterMesh
make_mesh(std::string name, std::string material_name, std::vector<MeshVertex> vertices, std::vector<uint3> indices)
{
    ImporterMesh mesh;
    mesh.name = std::move(name);
    mesh.ensure_attributes({
        {ImporterSemantic::position},
        {ImporterSemantic::normal},
        {ImporterSemantic::tangent},
        {ImporterSemantic::handedness},
        {ImporterSemantic::tex_coord},
    });
    mesh.allocate_vertices(vertices.size());

    auto positions = mesh.position_stream();
    auto normals = mesh.normal_stream();
    auto tangents = mesh.tangent_stream();
    auto handedness = mesh.handedness_stream();
    auto texcoords = mesh.texcoord_stream();
    for (size_t i = 0; i < vertices.size(); ++i) {
        positions[i] = vertices[i].position;
        normals[i] = vertices[i].normal;
        tangents[i] = vertices[i].tangent;
        handedness[i] = vertices[i].handedness;
        texcoords[i] = vertices[i].texcoord;
    }

    mesh.subgeometries.push_back(
        ImporterMesh::Subgeometry{
            .name = mesh.name,
            .indices = std::move(indices),
            .material_name = std::move(material_name),
        }
    );
    mesh.calculate_local_aabb();
    return mesh;
}

ImporterMesh make_rect_mesh(const ImporterLight& light, std::string mesh_name, std::string material_name)
{
    const float half_width = 0.5f * light.width;
    const float half_height = 0.5f * light.height;
    std::vector<MeshVertex> vertices = {
        {{-half_width, -half_height, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, -1.f, {0.f, 0.f}},
        {{half_width, -half_height, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, -1.f, {1.f, 0.f}},
        {{half_width, half_height, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, -1.f, {1.f, 1.f}},
        {{-half_width, half_height, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, -1.f, {0.f, 1.f}},
    };
    return make_mesh(std::move(mesh_name), std::move(material_name), std::move(vertices), {{0, 2, 1}, {0, 3, 2}});
}

ImporterMesh make_disk_mesh(const ImporterLight& light, std::string mesh_name, std::string material_name)
{
    std::vector<MeshVertex> vertices;
    vertices.reserve(DISK_SEGMENT_COUNT + 1);
    vertices.push_back({{0.f, 0.f, 0.f}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, -1.f, {0.5f, 0.5f}});
    for (uint32_t i = 0; i < DISK_SEGMENT_COUNT; ++i) {
        const float angle = 2.f * PI * static_cast<float>(i) / static_cast<float>(DISK_SEGMENT_COUNT);
        const float x = std::cos(angle);
        const float y = std::sin(angle);
        vertices.push_back({
            {light.radius * x, light.radius * y, 0.f},
            {0.f, 0.f, -1.f},
            {1.f, 0.f, 0.f},
            -1.f,
            {0.5f + 0.5f * x, 0.5f + 0.5f * y},
        });
    }

    std::vector<uint3> indices;
    indices.reserve(DISK_SEGMENT_COUNT);
    for (uint32_t i = 0; i < DISK_SEGMENT_COUNT; ++i) {
        const uint32_t next = (i + 1) % DISK_SEGMENT_COUNT;
        indices.push_back({0, next + 1, i + 1});
    }
    return make_mesh(std::move(mesh_name), std::move(material_name), std::move(vertices), std::move(indices));
}

ImporterMesh make_sphere_mesh(const ImporterLight& light, std::string mesh_name, std::string material_name)
{
    const uint32_t ring_count = SPHERE_LATITUDE_SEGMENT_COUNT - 1;
    std::vector<MeshVertex> vertices;
    vertices.reserve(2 + ring_count * SPHERE_LONGITUDE_SEGMENT_COUNT);
    vertices.push_back({{0.f, 0.f, light.radius}, {0.f, 0.f, 1.f}, {1.f, 0.f, 0.f}, 1.f, {0.5f, 0.f}});

    for (uint32_t latitude = 1; latitude < SPHERE_LATITUDE_SEGMENT_COUNT; ++latitude) {
        const float theta = PI * static_cast<float>(latitude) / static_cast<float>(SPHERE_LATITUDE_SEGMENT_COUNT);
        const float sin_theta = std::sin(theta);
        const float cos_theta = std::cos(theta);
        for (uint32_t longitude = 0; longitude < SPHERE_LONGITUDE_SEGMENT_COUNT; ++longitude) {
            const float phi
                = 2.f * PI * static_cast<float>(longitude) / static_cast<float>(SPHERE_LONGITUDE_SEGMENT_COUNT);
            const float sin_phi = std::sin(phi);
            const float cos_phi = std::cos(phi);
            const float3 normal(sin_theta * cos_phi, sin_theta * sin_phi, cos_theta);
            vertices.push_back({
                light.radius * normal,
                normal,
                {-sin_phi, cos_phi, 0.f},
                1.f,
                {static_cast<float>(longitude) / static_cast<float>(SPHERE_LONGITUDE_SEGMENT_COUNT),
                 static_cast<float>(latitude) / static_cast<float>(SPHERE_LATITUDE_SEGMENT_COUNT)},
            });
        }
    }

    const uint32_t bottom_vertex = static_cast<uint32_t>(vertices.size());
    vertices.push_back({{0.f, 0.f, -light.radius}, {0.f, 0.f, -1.f}, {1.f, 0.f, 0.f}, 1.f, {0.5f, 1.f}});

    auto ring_vertex = [](uint32_t ring, uint32_t longitude)
    {
        return 1 + ring * SPHERE_LONGITUDE_SEGMENT_COUNT + longitude % SPHERE_LONGITUDE_SEGMENT_COUNT;
    };

    std::vector<uint3> indices;
    indices.reserve(2 * SPHERE_LONGITUDE_SEGMENT_COUNT * (SPHERE_LATITUDE_SEGMENT_COUNT - 1));
    for (uint32_t longitude = 0; longitude < SPHERE_LONGITUDE_SEGMENT_COUNT; ++longitude) {
        indices.push_back({0, ring_vertex(0, longitude), ring_vertex(0, longitude + 1)});
    }
    for (uint32_t ring = 0; ring + 1 < ring_count; ++ring) {
        for (uint32_t longitude = 0; longitude < SPHERE_LONGITUDE_SEGMENT_COUNT; ++longitude) {
            const uint32_t upper = ring_vertex(ring, longitude);
            const uint32_t upper_next = ring_vertex(ring, longitude + 1);
            const uint32_t lower = ring_vertex(ring + 1, longitude);
            const uint32_t lower_next = ring_vertex(ring + 1, longitude + 1);
            indices.push_back({upper, lower, lower_next});
            indices.push_back({upper, lower_next, upper_next});
        }
    }
    for (uint32_t longitude = 0; longitude < SPHERE_LONGITUDE_SEGMENT_COUNT; ++longitude) {
        indices.push_back({
            ring_vertex(ring_count - 1, longitude),
            bottom_vertex,
            ring_vertex(ring_count - 1, longitude + 1),
        });
    }

    return make_mesh(std::move(mesh_name), std::move(material_name), std::move(vertices), std::move(indices));
}

bool geometry_requested(const ImporterLight& light, const ImportOptions& options)
{
    bool force_type = false;
    switch (light.type) {
    case ImporterLight::Type::rectangular:
        force_type = options.force_rectangle_lights_to_geometry;
        break;
    case ImporterLight::Type::disk:
        force_type = options.force_disk_lights_to_geometry;
        break;
    case ImporterLight::Type::sphere:
        force_type = options.force_sphere_lights_to_geometry;
        break;
    default:
        return false;
    }
    return light.geometry_visible || force_type;
}

ImporterMaterial make_emissive_material(const ImporterLight& light, std::string name)
{
    const float3 emissive_radiance = light.intensity * std::exp2(light.exposure);
    const float emission_luminance
        = 0.2126f * emissive_radiance.x + 0.7152f * emissive_radiance.y + 0.0722f * emissive_radiance.z;
    const float3 emission_color = emission_luminance > 0.f ? emissive_radiance / emission_luminance : float3(1.f);

    ImporterMaterial material;
    material.name = std::move(name);
    material.params.set("_scene_material_type", "EmissionOnlyMaterial");
    material.params.set("emission_color", emission_color);
    material.params.set("enable_color_temperature", light.enable_color_temperature);
    material.params.set("color_temperature", light.color_temperature);
    material.params.set("emission_luminance", emission_luminance);
    return material;
}

} // namespace

void convert_area_lights_to_geometry(ImporterScene& scene, const ImportOptions& options)
{
    if (scene.lights.empty())
        return;

    std::vector<int> replacement_mesh_indices(scene.lights.size(), -1);
    std::vector<int> retained_light_indices(scene.lights.size(), -1);
    std::vector<ImporterLight> retained_lights;
    retained_lights.reserve(scene.lights.size());

    for (size_t light_index = 0; light_index < scene.lights.size(); ++light_index) {
        const ImporterLight& light = scene.lights[light_index];
        const bool selected = geometry_requested(light, options);
        if (!selected || light.enable_shaping) {
            if (selected && light.enable_shaping) {
                sgl::log_warn(
                    "Keeping shaped area light '{}' analytic because emissive geometry cannot reproduce its angular "
                    "shaping.",
                    light.name
                );
            }
            retained_light_indices[light_index] = static_cast<int>(retained_lights.size());
            retained_lights.push_back(light);
            continue;
        }

        const std::string generated_base = light.name + ".__emissive_geometry_" + std::to_string(light_index);
        const std::string material_name = generated_base + "_material";
        scene.materials.push_back(make_emissive_material(light, material_name));

        ImporterMesh mesh;
        switch (light.type) {
        case ImporterLight::Type::rectangular:
            mesh = make_rect_mesh(light, generated_base, material_name);
            break;
        case ImporterLight::Type::disk:
            mesh = make_disk_mesh(light, generated_base, material_name);
            break;
        case ImporterLight::Type::sphere:
            mesh = make_sphere_mesh(light, generated_base, material_name);
            break;
        default:
            break;
        }
        replacement_mesh_indices[light_index] = static_cast<int>(scene.meshes.size());
        scene.meshes.push_back(std::move(mesh));
    }

    struct PendingChild {
        size_t parent_index;
        int mesh_index;
        std::string name;
    };
    std::vector<PendingChild> pending_children;
    const size_t original_node_count = scene.nodes.size();
    for (size_t node_index = 0; node_index < original_node_count; ++node_index) {
        ImporterNode& node = scene.nodes[node_index];
        if (node.light_index < 0)
            continue;

        const size_t old_light_index = static_cast<size_t>(node.light_index);
        if (old_light_index >= scene.lights.size()) {
            sgl::log_warn("Ignoring invalid light index {} on importer node '{}'.", node.light_index, node.name);
            node.light_index = -1;
            continue;
        }

        const int mesh_index = replacement_mesh_indices[old_light_index];
        if (mesh_index >= 0) {
            node.light_index = -1;
            if (node.mesh_index < 0) {
                node.mesh_index = mesh_index;
            } else {
                pending_children.push_back({node_index, mesh_index, node.name + ".__emissive_geometry"});
            }
        } else {
            node.light_index = retained_light_indices[old_light_index];
        }
    }

    for (PendingChild& child : pending_children) {
        const int child_index = static_cast<int>(scene.nodes.size());
        scene.nodes[child.parent_index].children.push_back(child_index);
        scene.nodes.push_back(
            ImporterNode{
                .name = std::move(child.name),
                .mesh_index = child.mesh_index,
                .parent = static_cast<int>(child.parent_index),
            }
        );
    }

    scene.lights = std::move(retained_lights);
    scene.calculate_aabbs();
}

} // namespace falcor::detail
