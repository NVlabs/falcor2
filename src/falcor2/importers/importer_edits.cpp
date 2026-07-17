// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/importers/importer_edits.h"

#include "falcor2/core/error.h"

#include <algorithm>
#include <iterator>
#include <limits>
#include <string_view>
#include <utility>

namespace falcor {

namespace {

int resolve_selector(const std::unordered_map<uint32_t, int>& selectors, uint32_t selector_id, std::string_view kind)
{
    auto it = selectors.find(selector_id);
    if (it == selectors.end()) {
        FALCOR_THROW("Unresolved importer {} selector {}", kind, selector_id);
    }
    return it->second;
}

int remap_index(int index, size_t offset, std::string_view kind)
{
    if (index < 0) {
        return index;
    }

    const size_t remapped = static_cast<size_t>(index) + offset;
    if (remapped > static_cast<size_t>(std::numeric_limits<int>::max())) {
        FALCOR_THROW("Importer {} index exceeds int range during asset merge", kind);
    }

    return static_cast<int>(remapped);
}

bool has_scene_content(const ImporterScene& scene)
{
    return !scene.materials.empty() || !scene.textures.empty() || !scene.meshes.empty() || !scene.curves.empty()
        || !scene.nodes.empty() || !scene.cameras.empty() || !scene.lights.empty() || !scene.prototypes.empty()
        || !scene.root_nodes.empty() || !scene.animation.channels.empty();
}

template<typename T>
void append_move(std::vector<T>& destination, std::vector<T>& source)
{
    destination
        .insert(destination.end(), std::make_move_iterator(source.begin()), std::make_move_iterator(source.end()));
}

void append_imported_scene(ImporterScene& destination, ImporterScene& source)
{
    if (!has_scene_content(destination)) {
        destination.uv_origin = source.uv_origin;
    }

    const size_t mesh_offset = destination.meshes.size();
    const size_t curve_offset = destination.curves.size();
    const size_t node_offset = destination.nodes.size();
    const size_t camera_offset = destination.cameras.size();
    const size_t light_offset = destination.lights.size();
    const size_t prototype_offset = destination.prototypes.size();
    const size_t animation_channel_offset = destination.animation.channels.size();

    destination.materials.reserve(destination.materials.size() + source.materials.size());
    destination.textures.reserve(destination.textures.size() + source.textures.size());
    destination.meshes.reserve(destination.meshes.size() + source.meshes.size());
    destination.curves.reserve(destination.curves.size() + source.curves.size());
    destination.cameras.reserve(destination.cameras.size() + source.cameras.size());
    destination.lights.reserve(destination.lights.size() + source.lights.size());
    destination.prototypes.reserve(destination.prototypes.size() + source.prototypes.size());
    destination.animation.channels.reserve(destination.animation.channels.size() + source.animation.channels.size());
    destination.nodes.reserve(destination.nodes.size() + source.nodes.size());
    destination.root_nodes.reserve(destination.root_nodes.size() + source.root_nodes.size());

    append_move(destination.materials, source.materials);
    append_move(destination.textures, source.textures);
    append_move(destination.meshes, source.meshes);
    append_move(destination.curves, source.curves);
    append_move(destination.cameras, source.cameras);
    append_move(destination.lights, source.lights);

    // Apply source animation if we don't aleady have one.
    if (animation_channel_offset == 0 && !source.animation.channels.empty()) {
        destination.animation.name = source.animation.name;
    }
    append_move(destination.animation.channels, source.animation.channels);

    for (auto& prototype : source.prototypes) {
        prototype.root_node = remap_index(prototype.root_node, node_offset, "prototype root node");
        destination.prototypes.push_back(std::move(prototype));
    }

    for (auto& node : source.nodes) {
        node.mesh_index = remap_index(node.mesh_index, mesh_offset, "node mesh");
        node.curve_index = remap_index(node.curve_index, curve_offset, "node curve");
        node.light_index = remap_index(node.light_index, light_offset, "node light");
        node.camera_index = remap_index(node.camera_index, camera_offset, "node camera");
        node.prototype_index = remap_index(node.prototype_index, prototype_offset, "node prototype");
        node.parent = remap_index(node.parent, node_offset, "node parent");
        node.animation_translation
            = remap_index(node.animation_translation, animation_channel_offset, "node translation animation");
        node.animation_rotation
            = remap_index(node.animation_rotation, animation_channel_offset, "node rotation animation");
        node.animation_scale = remap_index(node.animation_scale, animation_channel_offset, "node scale animation");

        for (int& child : node.children) {
            child = remap_index(child, node_offset, "node child");
        }

        destination.nodes.push_back(std::move(node));
    }

    for (const int root_node : source.root_nodes) {
        destination.root_nodes.push_back(remap_index(root_node, node_offset, "root node"));
    }
}

} // namespace

std::filesystem::path
ImporterBuildContext::resolve_required_asset(const std::filesystem::path& path, std::string_view kind) const
{
    FALCOR_CHECK(asset_resolver, "Importer build context has no asset resolver");
    const auto resolved = asset_resolver->resolve_path(path);
    if (!resolved.empty())
        return resolved;

    FALCOR_THROW(
        "Failed to resolve importer {} path \"{}\". Source directory: \"{}\". {}.",
        kind,
        path,
        source_directory,
        asset_resolver->to_string()
    );
}

ImportAssetEdit::ImportAssetEdit(std::filesystem::path path, ImportOptions import_options)
    : m_path(std::move(path))
    , m_import_options(import_options)
{
}

void ImportAssetEdit::apply(ImporterBuildContext& context) const
{
    const auto resolved_path = context.resolve_required_asset(m_path, "scene asset");
    ref<ImporterScene> imported_scene = import_scene(resolved_path, m_import_options);
    append_imported_scene(*context.scene, *imported_scene);
}

ReplaceMaterialEdit::ReplaceMaterialEdit(std::string name, std::string material_type, Properties props)
    : m_name(std::move(name))
    , m_material_type(std::move(material_type))
    , m_props(std::move(props))
{
}

ReplaceMaterialEdit::ReplaceMaterialEdit(std::string name, ImporterMaterial::Constructor constructor, Properties props)
    : m_name(std::move(name))
    , m_constructor(std::move(constructor))
    , m_props(std::move(props))
{
}

void ReplaceMaterialEdit::apply(ImporterBuildContext& context) const
{
    Properties replacement_params = m_props;
    if (m_constructor) {
        replacement_params.remove_property("_scene_material_type");
    } else {
        replacement_params.set("_scene_material_type", m_material_type);
    }

    for (ImporterMaterial& material : context.scene->materials) {
        if (material.name != m_name) {
            continue;
        }
        material.params = replacement_params;
        material.constructor = m_constructor;
        material.output_to_material_network.clear();
    }
}

SetEnvironmentEdit::SetEnvironmentEdit(std::filesystem::path path, float exposure, std::string name)
    : m_path(std::move(path))
    , m_exposure(exposure)
    , m_name(std::move(name))
{
}

void SetEnvironmentEdit::apply(ImporterBuildContext& context) const
{
    const auto resolved_path = context.resolve_required_asset(m_path, "environment map");
    const auto dome_it = std::find_if(
        context.scene->lights.begin(),
        context.scene->lights.end(),
        [](const ImporterLight& light)
        {
            return light.type == ImporterLight::Type::dome;
        }
    );

    int light_index = -1;
    if (dome_it != context.scene->lights.end()) {
        dome_it->name = m_name;
        dome_it->env_map_path = resolved_path.string();
        dome_it->exposure = m_exposure;
        light_index = static_cast<int>(std::distance(context.scene->lights.begin(), dome_it));
    } else {
        ImporterLight light;
        light.name = m_name;
        light.type = ImporterLight::Type::dome;
        light.env_map_path = resolved_path.string();
        light.exposure = m_exposure;
        light_index = static_cast<int>(context.scene->lights.size());
        context.scene->lights.push_back(std::move(light));
    }

    for (ImporterNode& node : context.scene->nodes) {
        if (node.light_index == light_index) {
            node.name = m_name;
            return;
        }
    }

    ImporterNode node;
    node.name = m_name;
    node.light_index = light_index;
    const int node_index = static_cast<int>(context.scene->nodes.size());
    context.scene->nodes.push_back(std::move(node));
    context.scene->root_nodes.push_back(node_index);
}

CreateCameraEdit::CreateCameraEdit(uint32_t selector_id, ImporterCamera camera)
    : m_selector_id(selector_id)
    , m_camera(std::move(camera))
{
}

void CreateCameraEdit::apply(ImporterBuildContext& context) const
{
    const int camera_index = static_cast<int>(context.scene->cameras.size());
    context.scene->cameras.push_back(m_camera);
    context.cameras[m_selector_id] = camera_index;
}

CreateNodeEdit::CreateNodeEdit(
    uint32_t selector_id,
    std::string name,
    float4x4 transform,
    std::optional<ImporterCameraSelector> camera,
    std::optional<ImporterNodeSelector> parent
)
    : m_selector_id(selector_id)
    , m_name(std::move(name))
    , m_transform(transform)
    , m_camera(camera)
    , m_parent(parent)
{
}

void CreateNodeEdit::apply(ImporterBuildContext& context) const
{
    ImporterNode node;
    node.name = m_name;
    node.transform = m_transform;

    if (m_camera) {
        node.camera_index = resolve_selector(context.cameras, m_camera->id, "camera");
    }

    if (m_parent) {
        node.parent = resolve_selector(context.nodes, m_parent->id, "node");
    }

    const int node_index = static_cast<int>(context.scene->nodes.size());
    context.scene->nodes.push_back(std::move(node));
    context.nodes[m_selector_id] = node_index;

    if (m_parent) {
        context.scene->nodes[context.scene->nodes[node_index].parent].children.push_back(node_index);
    } else {
        context.scene->root_nodes.push_back(node_index);
    }
}

} // namespace falcor
