// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "falcor2/core/python_interpreter.h"
#include "falcor2/core/platform.h"
#include "falcor2/render/scene.h"
#include "falcor2/importers/mesh_import.h"
#include "falcor2/importers/light_import.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/importers/gltf_importer/gltf_conversions.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/static_curve_geometry.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/component/transform_animation.h"

#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/utils/math/transform.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"
#include "falcor2/importers/material_conversions.h"
#include "falcor2/importers/usd_importer/usd_importer.h"

#include <sgl/core/type_utils.h>
#include <sgl/core/string.h>
#include <sgl/core/thread.h>
#include <sgl/core/memory_stream.h>
#include <sgl/core/crypto.h>
#include <sgl/device/device.h>
#include <sgl/device/sampler.h>
#include <sgl/utils/texture_loader.h>

#include <filesystem>
#include <fstream>
#include <cmath>
#include <map>
#include <span>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

namespace falcor {

/// Helper function to create a texture from importer embedded texture data.
/// Returns a ref<sgl::Texture> on success, or an empty ref if no embedded data.
static ref<sgl::Texture> create_texture_from_embedded_data(
    Scene* scene,
    std::span<const uint8_t> texture_data,
    TextureLoadFlags flags,
    std::string_view source_name
)
{
    if (texture_data.empty()) {
        return {};
    }

    try {
        sgl::MemoryStream stream(texture_data.data(), texture_data.size());
        sgl::TextureLoader loader{ref<sgl::Device>(scene->device())};
        const auto options
            = texture_loader_options(flags, sgl::TextureUsage::shader_resource, scene->config().max_mip_count);
        return loader.load_texture(&stream, options);
    } catch (const std::exception& e) {
        sgl::log_warn("Failed to load embedded texture data for '{}': {}", source_name, e.what());
    }

    return {};
}

/// Helper function to create a texture from importer embedded texture data.
/// Returns a ref<sgl::Texture> on success, or an empty ref if no embedded data.
static ref<sgl::Texture>
create_texture_from_embedded_data(Scene* scene, const ImporterTexture& importer_texture, TextureLoadFlags flags)
{
    return create_texture_from_embedded_data(scene, importer_texture.texture_data, flags, importer_texture.source_name);
}

struct ImportedTextureSource {
    ref<sgl::Texture> texture;
    std::filesystem::path path;
    ref<sgl::Sampler> sampler;
};

class ImportedTextureResolver {
public:
    ImportedTextureResolver(
        Scene* scene,
        std::span<const ImporterAsset> importer_assets,
        std::span<const ImporterTexture> importer_textures
    )
        : m_scene(scene)
    {
        for (const ImporterTexture& texture : importer_textures) {
            if (!texture.texture_path.empty() && !texture.texture_data.empty())
                m_assets.try_emplace(texture.texture_path, std::span<const uint8_t>(texture.texture_data));
        }
        for (const ImporterAsset& asset : importer_assets)
            m_assets.try_emplace(asset.path, std::span<const uint8_t>(asset.data));
    }

    ImportedTextureSource resolve(const ImporterTexture& importer_texture, TextureLoadFlags flags)
    {
        auto sampler = resolve_sampler(importer_texture);
        if (auto texture = resolve(importer_texture.texture_path, flags))
            return {.texture = std::move(texture), .sampler = sampler};
        if (auto texture = create_texture_from_embedded_data(m_scene, importer_texture, flags))
            return {.texture = std::move(texture), .sampler = sampler};
        return {.path = importer_texture.texture_path, .sampler = sampler};
    }

    std::filesystem::path materialx_path(const ImporterTexture& texture)
    {
        std::span<const uint8_t> bytes = texture.texture_data;
        if (bytes.empty()) {
            if (auto it = m_assets.find(texture.texture_path); it != m_assets.end())
                bytes = it->second;
        }
        if (bytes.empty())
            return std::filesystem::absolute(texture.texture_path);

        // MaterialX's existing resource loader consumes filenames. Keep embedded
        // images content-addressed so deferred loads outlive the importer data.
        const auto directory = platform::app_data_directory() / "gltf-images";
        std::filesystem::create_directories(directory);
        const auto hash = sgl::SHA1(bytes.data(), bytes.size()).hex_digest();
        const auto extension = texture.mime_type == "image/jpeg" ? ".jpg" : ".png";
        const auto path = directory / (hash + extension);
        platform::FileLock lock(directory / (hash + ".lock"));
        FALCOR_CHECK(lock.try_lock(), "Timed out caching embedded glTF texture '{}'", texture.source_name);
        if (!std::filesystem::exists(path) || std::filesystem::file_size(path) != bytes.size()) {
            std::ofstream stream(path, std::ios::binary | std::ios::trunc);
            stream.write(reinterpret_cast<const char*>(bytes.data()), std::streamsize(bytes.size()));
            stream.close();
            FALCOR_CHECK(stream.good(), "Failed to cache embedded glTF texture '{}'", path);
        }
        return path;
    }

    ref<sgl::Texture> resolve(const std::filesystem::path& path, TextureLoadFlags flags)
    {
        const auto key = std::make_tuple(path, flags);
        if (auto it = m_textures.find(key); it != m_textures.end())
            return it->second;

        auto asset_it = m_assets.find(path);
        if (asset_it == m_assets.end())
            return {};

        ref<sgl::Texture> texture = create_texture_from_embedded_data(m_scene, asset_it->second, flags, path.string());
        if (!texture)
            FALCOR_THROW("Failed to decode embedded texture '{}'", path.string());

        m_textures.emplace(key, texture);
        return texture;
    }

private:
    ref<sgl::Sampler> resolve_sampler(const ImporterTexture& texture)
    {
        const auto key = std::make_tuple(
            texture.min_filter,
            texture.mag_filter,
            texture.mip_filter,
            texture.wrap_s,
            texture.wrap_t,
            texture.use_mipmaps
        );
        if (auto it = m_samplers.find(key); it != m_samplers.end())
            return it->second;

        const auto filter = [](TextureFilterMode mode)
        {
            return mode == TextureFilterMode::nearest ? sgl::TextureFilteringMode::point
                                                      : sgl::TextureFilteringMode::linear;
        };
        const auto address = [](TextureWrapMode mode)
        {
            switch (mode) {
            case TextureWrapMode::repeat:
                return sgl::TextureAddressingMode::wrap;
            case TextureWrapMode::mirror_repeat:
                return sgl::TextureAddressingMode::mirror_repeat;
            case TextureWrapMode::clamp_to_edge:
                return sgl::TextureAddressingMode::clamp_to_edge;
            }
            FALCOR_UNREACHABLE();
        };
        sgl::SamplerDesc desc;
        desc.min_filter = filter(texture.min_filter);
        desc.mag_filter = filter(texture.mag_filter);
        desc.mip_filter = filter(texture.mip_filter);
        desc.address_u = address(texture.wrap_s);
        desc.address_v = address(texture.wrap_t);
        if (!texture.use_mipmaps)
            desc.max_lod = 0.f;

        // Leave unspecified/default bindings on the existing scene sampler.
        const sgl::SamplerDesc defaults;
        ref<sgl::Sampler> sampler;
        if (desc.min_filter != defaults.min_filter || desc.mag_filter != defaults.mag_filter
            || desc.mip_filter != defaults.mip_filter || desc.address_u != defaults.address_u
            || desc.address_v != defaults.address_v || desc.max_lod != defaults.max_lod) {
            if (m_scene->device()->type() == sgl::DeviceType::cuda && desc.min_filter != desc.mag_filter)
                sgl::log_warn(
                    "Texture '{}': CUDA uses the minification filter for magnification too; "
                    "the authored min/mag filters differ.",
                    texture.source_name
                );
            sampler = m_scene->device()->create_sampler(desc);
        }
        m_samplers.emplace(key, sampler);
        return sampler;
    }

    Scene* m_scene;
    std::unordered_map<std::filesystem::path, std::span<const uint8_t>> m_assets;
    std::map<std::tuple<std::filesystem::path, TextureLoadFlags>, ref<sgl::Texture>> m_textures;
    std::map<
        std::tuple<TextureFilterMode, TextureFilterMode, TextureFilterMode, TextureWrapMode, TextureWrapMode, bool>,
        ref<sgl::Sampler>>
        m_samplers;
};

static void set_texture_property(
    Properties& properties,
    const std::string& property_name,
    const std::string& property_name_path,
    const ImportedTextureSource& source
)
{
    if (source.texture)
        properties.set(property_name, source.texture);
    else if (!source.path.empty())
        properties.set(property_name_path, source.path);
    if (source.sampler && (source.texture || !source.path.empty()))
        properties.set(property_name + "_sampler", source.sampler);
}

static void resolve_embedded_texture_properties(Properties& properties, ImportedTextureResolver& texture_resolver)
{
    std::vector<std::string> path_property_names;
    for (std::string_view key : properties.keys()) {
        if (key.ends_with("_texture_path") && properties.type(key) == PropertyType::string)
            path_property_names.emplace_back(key);
    }

    for (const std::string& path_property_name : path_property_names) {
        const std::filesystem::path path = properties.get<std::filesystem::path>(path_property_name);
        const bool srgb = importer_material::is_texture_srgb(properties, path_property_name);

        auto flags = TextureLoadFlags::generate_mips;
        if (srgb)
            flags |= TextureLoadFlags::srgb;
        ref<sgl::Texture> texture = texture_resolver.resolve(path, flags);
        if (!texture)
            continue;

        const std::string texture_property_name
            = path_property_name.substr(0, path_property_name.size() - std::string_view("_path").size());
        properties.remove_property(path_property_name);
        properties.set(texture_property_name, std::move(texture));
    }
}

static Material* load_importer_material(
    Scene* scene,
    const ImporterMaterial& importer_material,
    std::span<const ImporterTexture> importer_textures,
    ImportedTextureResolver& texture_resolver,
    bool use_vertex_color
)
{
    const Properties& params = importer_material.params;
    if (importer_material.constructor) {
        Material* material = importer_material.constructor(*scene, importer_material);
        if (!material) {
            FALCOR_THROW("Constructor for importer material '{}' returned null", importer_material.name);
        }
        return material;
    }

    if (auto scene_material_type = params.get_optional<std::string_view>("_scene_material_type")) {
        Properties resolved_params = params;
        resolve_embedded_texture_properties(resolved_params, texture_resolver);
        auto material = scene->create_material(*scene_material_type, resolved_params);
        return material;
    }

    auto type = params.get_optional<std::string>("_type");

    if (type && (*type == "usd_UsdPreviewSurface" || *type == "usd_NodeMaterial")) {
        if (auto converted = convert_material(importer_material)) {
            resolve_embedded_texture_properties(*converted, texture_resolver);
            auto material
                = scene->create_material(converted->get<std::string_view>("_scene_material_type"), *converted);
            return material;
        }
    }

    std::optional<Properties> converted;
    const GLTFTextureSetter set_texture
        = [&](Properties& properties, std::string_view name, const ImporterTexture& texture, TextureLoadFlags flags)
    {
        set_texture_property(
            properties,
            std::string(name),
            fmt::format("{}_path", name),
            texture_resolver.resolve(texture, flags)
        );
    };
    const auto target = params.get<std::string>("_gltf_target", "");
    if (target == "standard_material")
        converted = gltf_to_standard(importer_material, importer_textures, use_vertex_color, set_texture);
    else if (target == "openpbr")
        converted = gltf_to_openpbr(importer_material, importer_textures, use_vertex_color, set_texture);
    else if (target == "materialx")
        converted = gltf_to_mtlx(
            importer_material,
            importer_textures,
            use_vertex_color,
            scene->config().uv_origin,
            [&](const ImporterTexture& texture)
            {
                return texture_resolver.materialx_path(texture);
            }
        );
    if (!target.empty())
        FALCOR_CHECK(
            converted,
            "glTF material '{}' cannot be converted to requested target '{}'",
            importer_material.name,
            target
        );
    if (!converted && params.get("_gltf_use_mdl", false)) {
        converted = gltf_to_mdl(
            importer_material,
            importer_textures,
            scene->config().uv_origin,
            [&](Properties& properties, std::string_view name, const ImporterTexture& texture, bool srgb)
            {
                const auto flags = srgb ? TextureLoadFlags::color : TextureLoadFlags::data_rgb;
                auto resolved = texture_resolver.resolve(texture, flags);
                if (!resolved.texture) {
                    sgl::TextureLoader loader{ref<sgl::Device>(scene->device())};
                    resolved.texture = loader.load_texture(
                        resolved.path,
                        texture_loader_options(flags, sgl::TextureUsage::shader_resource, scene->config().max_mip_count)
                    );
                }
                properties.set(fmt::format("{}:texture", name), resolved.texture);
                properties.set(fmt::format("{}:sampler", name), resolved.sampler);
                properties.set(fmt::format("{}:srgb", name), srgb);
            }
        );
    }
    if (!converted) {
        converted = gltf_to_gltf(
            importer_material,
            importer_textures,
            use_vertex_color,
            [&](Properties& properties, std::string_view name, const ImporterTexture& texture, TextureLoadFlags flags)
            {
                set_texture_property(
                    properties,
                    std::string(name),
                    fmt::format("{}_path", name),
                    texture_resolver.resolve(texture, flags)
                );
            }
        );
    }
    if (converted) {
        auto material = scene->create_material(converted->get<std::string_view>("_scene_material_type"), *converted);
        material->set_name(importer_material.name);
        return material;
    }

    Properties properties;
    properties.set("alpha_mode", params.get("alphaMode", AlphaMode::opaque));
    properties.set("alpha_cutoff", params.get("alphaCutoff", 0.5f));
    properties.set("use_vertex_color", use_vertex_color);
    auto material = scene->create_material<StandardMaterial>();
    material->set_properties(properties);
    return material;
}

static StaticMeshGeometry* load_importer_mesh(Scene* scene, const ImporterMesh& importer_mesh)
{
    auto* geometry = scene->create_geometry<StaticMeshGeometry>();
    update_imported_mesh(geometry, importer_mesh, scene->config().uv_origin);
    return geometry;
}

static StaticCurveGeometry* load_importer_curve(Scene* scene, const ImporterCurve& importer_curve)
{
    StaticCurveGeometryDataDesc desc = {};
    desc.name = importer_curve.name;
    desc.vertex_count = sgl::narrow_cast<uint32_t>(importer_curve.vertex_count());
    const auto positions = importer_curve.position_stream();
    const auto radii = importer_curve.radius_stream();
    desc.position_stream.data = reinterpret_cast<const float3*>(positions.data);
    desc.position_stream.stride = positions.stride;
    desc.radius_stream.data = reinterpret_cast<const float*>(radii.data);
    desc.radius_stream.stride = radii.stride;
    desc.index_count = sgl::narrow_cast<uint32_t>(importer_curve.indices.size());
    desc.index_stream.data = importer_curve.indices.data();
    desc.index_stream.stride = sizeof(uint32_t);
    desc.indexing_mode = importer_curve.indexing_mode;

    std::span<const float2> texcoords = importer_curve.texcoords;
    std::vector<float2> converted_texcoords;
    GeometryVertexAttributeDesc texcoord_attribute = {};
    if (!texcoords.empty()) {
        FALCOR_CHECK(texcoords.size() == desc.vertex_count, "Curve '{}' UV count does not match positions", desc.name);
        if (importer_curve.uv_origin != scene->config().uv_origin) {
            converted_texcoords.assign(texcoords.begin(), texcoords.end());
            for (float2& uv : converted_texcoords)
                uv.y = 1.f - uv.y;
            texcoords = converted_texcoords;
        }
        texcoord_attribute = {
            .semantic = MeshAttributeSemantic::tex_coord,
            .index = 0,
            .type = sgl::DataType::float32,
            .component_count = 2,
            .data = texcoords.data(),
            .stride = sizeof(float2),
            .data_size = texcoords.size() * sizeof(float2),
        };
        desc.vertex_attributes = std::span(&texcoord_attribute, 1);
    }

    StaticCurveGeometry* geometry = scene->create_geometry<StaticCurveGeometry>();
    geometry->set_curve_data(desc);
    geometry->_set_source_asset_ref(importer_curve.source_asset_ref);
    return geometry;
}

static SceneConfig resolve_scene_config(std::optional<SceneConfig> config, const ImporterScene& importer_scene)
{
    SceneConfig result = config.value_or(SceneConfig{.uv_origin = importer_scene.uv_origin});
    if (importer_scene.max_mip_count)
        result.max_mip_count = result.max_mip_count ? std::min(result.max_mip_count, importer_scene.max_mip_count)
                                                    : importer_scene.max_mip_count;
    return result;
}

static void append_importer_scene(Scene* scene, const ImporterScene& importer_scene)
{
    ImportedTextureResolver texture_resolver(scene, importer_scene.assets, importer_scene.textures);

    std::vector<bool> indexed_vertex_color_materials(importer_scene.materials.size(), false);
    bool default_material_uses_vertex_color = false;
    for (const ImporterMesh& mesh : importer_scene.meshes) {
        if (!mesh.color_attribute().valid())
            continue;
        for (const ImporterMesh::Subgeometry& subgeometry : mesh.subgeometries) {
            if (subgeometry.material_index >= 0
                && subgeometry.material_index < static_cast<int>(importer_scene.materials.size()))
                indexed_vertex_color_materials[subgeometry.material_index] = true;
            else
                default_material_uses_vertex_color = true;
        }
    }

    // Import materials.
    std::vector<Material*> imported_materials;
    imported_materials.reserve(importer_scene.materials.size());
    for (size_t material_index = 0; material_index < importer_scene.materials.size(); ++material_index) {
        const ImporterMaterial& importer_material = importer_scene.materials[material_index];
        Material* material = load_importer_material(
            scene,
            importer_material,
            importer_scene.textures,
            texture_resolver,
            indexed_vertex_color_materials[material_index]
        );
        if (importer_material.params.get<bool>("double_sided", false)) {
            Properties properties;
            properties.set("double_sided", true);
            properties.set("enable_backface_culling", false);
            material->set_properties(properties);
        }
        material->set_name(importer_material.name);
        if (importer_material.source_asset_ref)
            material->_set_source_asset_ref(importer_material.source_asset_ref);
        imported_materials.push_back(material);
    }

    Material* default_material = nullptr;
    auto resolve_material = [&](int material_index) -> Material*
    {
        if (material_index >= 0 && material_index < static_cast<int>(imported_materials.size()))
            return imported_materials[material_index];
        if (material_index != -1)
            sgl::log_warn(
                "Importer geometry references invalid material index {}; using default material",
                material_index
            );

        if (!default_material) {
            default_material = scene->create_material<StandardMaterial>();
            default_material->set_name("DefaultMaterial");
            Properties properties;
            properties.set("use_vertex_color", default_material_uses_vertex_color);
            default_material->set_properties(properties);
        }
        return default_material;
    };

    // Create a StaticMeshGeometry for each importer mesh.
    std::vector<StaticMeshGeometry*> mesh_geometries;
    std::map<StaticMeshGeometry*, std::vector<Material*>> mesh_geometry_to_materials;
    for (const ImporterMesh& importer_mesh : importer_scene.meshes) {
        StaticMeshGeometry* mesh_geometry = load_importer_mesh(scene, importer_mesh);
        mesh_geometries.push_back(mesh_geometry);
        std::vector<Material*> materials;
        for (const ImporterMesh::Subgeometry& subgeo : importer_mesh.subgeometries) {
            materials.push_back(resolve_material(subgeo.material_index));
        }
        mesh_geometry_to_materials[mesh_geometry] = std::move(materials);
    }

    // Create a StaticCurveGeometry for each importer curve.
    std::vector<StaticCurveGeometry*> curve_geometries;
    std::map<StaticCurveGeometry*, Material*> curve_geometry_to_material;
    for (const ImporterCurve& importer_curve : importer_scene.curves) {
        StaticCurveGeometry* curve_geometry = load_importer_curve(scene, importer_curve);
        curve_geometries.push_back(curve_geometry);
        curve_geometry_to_material[curve_geometry] = resolve_material(importer_curve.material_index);
    }

    // Create Animation object from importer scene (before entities, so TransformAnimation can reference it).
    Animation* animation = nullptr;
    if (!importer_scene.animation.channels.empty()) {
        animation = scene->create_animation();
        animation->set_name(importer_scene.animation.name);
        animation->_set_source_asset_ref(importer_scene.animation.source_asset_ref);
        std::vector<ref<AnimationChannel>> channels;
        channels.reserve(importer_scene.animation.channels.size());
        for (const ImporterAnimationChannel& src : importer_scene.animation.channels) {
            auto dst = make_ref<AnimationChannel>();
            dst->set_value_type(src.value_type);
            dst->set_interpolation_mode(src.interpolation_mode);
            dst->set_times(src.times);
            dst->set_values(src.values);
            dst->set_in_tangents(src.in_tangents);
            dst->set_out_tangents(src.out_tangents);
            channels.push_back(std::move(dst));
        }
        animation->set_channels(std::move(channels));
    }

    auto create_entities_recursive = [&](int node_idx, Entity* parent, std::string_view prefix, auto recurse) -> void
    {
        const ImporterNode& importer_node = importer_scene.nodes[node_idx];

        Entity* entity = scene->create_entity();
        entity->set_transform(Transform(importer_node.transform));
        entity->set_parent(parent);
        entity->set_name(fmt::format("{}{}", prefix, importer_node.name));
        entity->_set_source_asset_ref(importer_node.source_asset_ref);

        // Create TransformAnimation component if this node has animation channels.
        if (animation
            && (importer_node.animation_translation >= 0 || importer_node.animation_rotation >= 0
                || importer_node.animation_scale >= 0)) {
            const int channel_count = static_cast<int>(animation->channels().size());
            TransformAnimation* ta = entity->create_component<TransformAnimation>();
            ta->_set_source_asset_ref(importer_node.source_asset_ref);
            ta->set_animation(animation);
            if (importer_node.animation_translation >= 0 && importer_node.animation_translation < channel_count)
                ta->set_translation_channel(importer_node.animation_translation);
            if (importer_node.animation_rotation >= 0 && importer_node.animation_rotation < channel_count)
                ta->set_rotation_channel(importer_node.animation_rotation);
            if (importer_node.animation_scale >= 0 && importer_node.animation_scale < channel_count)
                ta->set_scale_channel(importer_node.animation_scale);
        }

        if (importer_node.mesh_index >= 0) {
            FALCOR_ASSERT_LT(static_cast<size_t>(importer_node.mesh_index), mesh_geometries.size());
            GeometryInstance* instance = entity->create_component<GeometryInstance>();
            // Generated transform helpers have no source identity of their own.
            instance->_set_source_asset_ref(
                importer_node.source_asset_ref ? importer_node.source_asset_ref
                                               : importer_scene.meshes[importer_node.mesh_index].source_asset_ref
            );
            StaticMeshGeometry* mesh_geometry = mesh_geometries[importer_node.mesh_index];
            instance->set_geometry(mesh_geometry);
            auto it = mesh_geometry_to_materials.find(mesh_geometry);
            FALCOR_ASSERT(it != mesh_geometry_to_materials.end());
            instance->set_materials(it->second);
        }
        if (importer_node.curve_index >= 0) {
            FALCOR_ASSERT_LT(static_cast<size_t>(importer_node.curve_index), curve_geometries.size());
            GeometryInstance* instance = entity->create_component<GeometryInstance>();
            instance->_set_source_asset_ref(
                importer_node.source_asset_ref ? importer_node.source_asset_ref
                                               : importer_scene.curves[importer_node.curve_index].source_asset_ref
            );
            StaticCurveGeometry* curve_geometry = curve_geometries[importer_node.curve_index];
            instance->set_geometry(curve_geometry);
            auto it = curve_geometry_to_material.find(curve_geometry);
            if (it != curve_geometry_to_material.end()) {
                instance->set_materials({it->second});
            }
        }
        if (importer_node.light_index >= 0) {
            FALCOR_ASSERT_LT(static_cast<size_t>(importer_node.light_index), importer_scene.lights.size());
            const ImporterLight& importer_light = importer_scene.lights[importer_node.light_index];
            Light* light = update_imported_light(entity, nullptr, importer_light);
            if (auto* env_map_light = dynamic_cast<EnvMapLight*>(light)) {
                if (ref<sgl::Texture> texture
                    = texture_resolver.resolve(importer_light.env_map_path, TextureLoadFlags::environment))
                    env_map_light->set_env_map_texture(std::move(texture));
            }
        }
        if (importer_node.camera_index >= 0) {
            FALCOR_ASSERT_LT(static_cast<size_t>(importer_node.camera_index), importer_scene.cameras.size());
            const ImporterCamera& importer_camera = importer_scene.cameras[importer_node.camera_index];
            Camera* camera = entity->create_component<Camera>();
            camera->set_name(importer_camera.name);
            camera->_set_source_asset_ref(importer_camera.source_asset_ref);
            // Normalize directional importer data to Camera's vertical film-fit model.
            camera->set_focal_length(importer_camera.focal_length);
            camera->set_fstop(importer_camera.fstop);
            camera->set_exposure(importer_camera.exposure);
            camera->set_sensor_height(importer_camera.vertical_sensor_size_mm());
            camera->set_enable_depth_of_field(importer_camera.enable_depth_of_field);
            camera->set_focus_distance(importer_camera.focus_distance);
            camera->set_depth_range(importer_camera.depth_range);
            if (scene->active_camera() == nullptr) {
                scene->set_active_camera(camera);
            }
        }

        // Recursively process children
        for (int child_idx : importer_node.children) {
            recurse(child_idx, entity, prefix, recurse);
        }

        // Process instanced prototype
        if (importer_node.prototype_index >= 0) {
            FALCOR_ASSERT_LT(static_cast<size_t>(importer_node.prototype_index), importer_scene.prototypes.size());
            recurse(
                importer_scene.prototypes[importer_node.prototype_index].root_node,
                entity,
                entity->name() + "/",
                recurse
            );
        }
    };

    for (auto root_idx : importer_scene.root_nodes)
        create_entities_recursive(root_idx, nullptr, "", create_entities_recursive);
}

struct LoadedImporter {
    ref<Importer> importer;
    std::vector<std::filesystem::path> python_search_paths;
};

static LoadedImporter load_pyscene_importer(const std::filesystem::path& path, const ImportOptions& import_options)
{
    const std::filesystem::path absolute_path = std::filesystem::absolute(path).lexically_normal();
    if (sgl::string::to_lower(absolute_path.extension().string()) != ".py")
        FALCOR_THROW("Expected a Python scene file with extension '.py', got '{}'", path);

    ref<Importer> importer = Importer::create(import_options);
    importer->set_source_path(absolute_path);
    ScopedCurrentImporter current_importer(importer);

    std::vector<std::filesystem::path> python_search_paths{absolute_path.parent_path()};
    const std::filesystem::path project_directory = platform::project_directory();
    if (std::filesystem::exists(project_directory / "falcor2" / "__init__.py"))
        python_search_paths.push_back(project_directory);

    PythonInterpreter::get().create_context().execute_file(absolute_path, "__falcor2_scene__", python_search_paths);
    return {.importer = std::move(importer), .python_search_paths = std::move(python_search_paths)};
}

static LoadedImporter load_scene_importer(const std::filesystem::path& path, const ImportOptions& import_options)
{
    const std::filesystem::path absolute_path = std::filesystem::absolute(path).lexically_normal();
    if (sgl::string::to_lower(absolute_path.extension().string()) == ".py")
        return load_pyscene_importer(absolute_path, import_options);

    ref<Importer> importer = Importer::create(import_options);
    importer->set_source_path(absolute_path);
    importer->import_asset(absolute_path);
    return {.importer = std::move(importer)};
}

ref<Scene> Scene::load(
    ref<sgl::Device> device,
    const std::filesystem::path& path,
    std::optional<ImportOptions> import_options,
    std::optional<SceneConfig> config
)
{
    LoadedImporter loaded = load_scene_importer(path, import_options.value_or(ImportOptions{}));
    if (loaded.python_search_paths.empty())
        return from_importer(std::move(device), *loaded.importer, config);

    ref<Scene> scene;
    PythonInterpreter::get().with_search_paths(
        loaded.python_search_paths,
        [&]
        {
            scene = from_importer(std::move(device), *loaded.importer, config);
        }
    );
    return scene;
}

ref<Scene> Scene::from_stage(
    ref<sgl::Device> device,
    const pxr::UsdStage& stage,
    std::optional<ImportOptions> import_options,
    std::optional<SceneConfig> config
)
{
    ref<ImporterScene> importer_scene = UsdImporter().import_stage(stage, import_options.value_or(ImportOptions{}));
    return from_importer_scene(std::move(device), *importer_scene, config);
}

ref<Scene> Scene::from_importer_scene(
    ref<sgl::Device> device,
    const ImporterScene& importer_scene,
    std::optional<SceneConfig> config
)
{
    auto scene = ref<Scene>{new Scene(std::move(device), resolve_scene_config(config, importer_scene))};
    append_importer_scene(scene.get(), importer_scene);
    return scene;
}

ref<Scene> Scene::from_importer(ref<sgl::Device> device, const Importer& importer, std::optional<SceneConfig> config)
{
    ref<ImporterScene> importer_scene = importer.build_importer_scene();
    FALCOR_ASSERT(importer_scene);

    ref<Scene> scene = from_importer_scene(std::move(device), *importer_scene, config);
    importer.run_scene_loaded_callbacks(scene);
    return scene;
}

void Scene::append(const std::filesystem::path& path, std::optional<ImportOptions> import_options)
{
    LoadedImporter loaded = load_scene_importer(path, import_options.value_or(ImportOptions{}));
    if (loaded.python_search_paths.empty()) {
        append(*loaded.importer);
        return;
    }

    PythonInterpreter::get().with_search_paths(
        loaded.python_search_paths,
        [&]
        {
            append(*loaded.importer);
        }
    );
}

void Scene::append(const ImporterScene& importer_scene)
{
    append_importer_scene(this, importer_scene);
}

void Scene::append(const Importer& importer)
{
    ref<ImporterScene> importer_scene = importer.build_importer_scene();
    FALCOR_ASSERT(importer_scene);
    append_importer_scene(this, *importer_scene);
    importer.run_scene_loaded_callbacks(ref(this));
}

} // namespace falcor
