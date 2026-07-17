// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0


#include "scene_import.h"

#include "falcor2/core/python_interpreter.h"
#include "falcor2/render/scene.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/standard_specgloss_material.h"
#include "falcor2/render/material/openpbr_material.h"
#include "falcor2/render/geometry/static_mesh_geometry.h"
#include "falcor2/render/geometry/static_curve_geometry.h"
#include "falcor2/render/component/geometry_instance.h"
#include "falcor2/render/component/camera.h"
#include "falcor2/render/component/light.h"
#include "falcor2/render/animation.h"
#include "falcor2/render/component/transform_animation.h"

#include "falcor2/importers/importer.h"
#include "falcor2/importers/importer_types.h"
#include "falcor2/importers/gltf_importer/gltf_importer.h"
#include "falcor2/importers/material_conversions.h"
#include "falcor2/importers/usd_importer/usd_importer.h"

#include <sgl/core/type_utils.h>
#include <sgl/core/string.h>
#include <sgl/core/thread.h>
#include <sgl/core/bitmap.h>
#include <sgl/core/memory_stream.h>
#include <sgl/device/device.h>
#include <sgl/utils/texture_loader.h>

#include <map>
#include <utility>
#include <vector>

namespace falcor {

/// Temporary switch for using the OpenPBRMaterial vs StandardMaterial.
static constexpr bool USE_OPENPBR_MATERIAL = false;

/// Helper function to create a texture from importer embedded texture data.
/// Returns a ref<sgl::Texture> on success, or an empty ref if no embedded data.
static ref<sgl::Texture>
create_texture_from_embedded_data(sgl::Device* device, const ImporterTexture& importer_texture, bool load_as_srgb)
{
    if (importer_texture.texture_data.empty()) {
        return {};
    }

    try {
        // Load bitmap from embedded texture data
        sgl::MemoryStream stream(importer_texture.texture_data.data(), importer_texture.texture_data.size());
        ref<sgl::Bitmap> bitmap = ref(new sgl::Bitmap(&stream));

        // Create texture from bitmap via TextureLoader
        if (!bitmap->empty()) {
            sgl::ref<sgl::Device> device_ref(device);
            sgl::TextureLoader loader(device_ref);
            return loader.load_texture(bitmap.get(), sgl::TextureLoader::Options{.load_as_srgb = load_as_srgb});
        }
    } catch (const std::exception& e) {
        sgl::log_warn("Failed to load embedded texture data: {}", e.what());
    }

    return {};
}

/// Helper function to set texture properties from importer texture.
/// Sets the texture reference if embedded data is available, and sets the path property
/// only if embedded data is not available and path is available.
static void set_texture_property(
    Properties& properties,
    sgl::Device* device,
    const std::string& property_name,
    const std::string& property_name_path,
    const ImporterTexture& importer_texture,
    bool load_as_srgb = true
)
{
    // Set texture reference if embedded data is available
    if (auto texture = create_texture_from_embedded_data(device, importer_texture, load_as_srgb)) {
        properties.set(property_name, texture);
    } else if (!importer_texture.texture_path.empty()) {
        // Only set path if no embedded data is available
        properties.set(property_name_path, importer_texture.texture_path);
    }
}

static Material* load_importer_material(
    Scene* scene,
    const ImporterMaterial& importer_material,
    std::span<const ImporterTexture> importer_textures
)
{
    const Properties& params = importer_material.params;
    if (importer_material.constructor) {
        Material* material = importer_material.constructor(*scene, importer_material);
        if (!material) {
            FALCOR_THROW("Constructor for importer material '{}' returned null", importer_material.name);
        }
        material->set_name(importer_material.name);
        return material;
    }

    if (auto scene_material_type = params.get_optional<std::string_view>("_scene_material_type")) {
        auto material = scene->create_material(*scene_material_type, params);
        material->set_name(importer_material.name);
        return material;
    }

    auto type = params.get_optional<std::string>("_type");

    if (type && (*type == "usd_UsdPreviewSurface" || *type == "usd_NodeMaterial")) {
        if (auto converted = convert_material(importer_material)) {
            auto material
                = scene->create_material(converted->get<std::string_view>("_scene_material_type"), *converted);
            material->set_name(importer_material.name);
            return material;
        }
    }

    if (USE_OPENPBR_MATERIAL) {
        OpenPBRMaterial* material = scene->create_material<OpenPBRMaterial>();
        material->set_name(importer_material.name);
        Properties properties;

        if (type && type.value() == "gltf_pbrMetallicRoughness") {
            if (auto property = params.get_optional<float4>("baseColorFactor")) {
                properties.set("base_color", property.value().xyz());
            }
            if (auto property = params.get_optional<int>("baseColorTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "base_color_texture",
                    "base_color_texture_path",
                    importer_textures[property.value()]
                );
            }
            if (auto property = params.get_optional<float>("metallicFactor")) {
                properties.set("base_metalness", property.value());
            }
            if (auto property = params.get_optional<float>("roughnessFactor")) {
                properties.set("specular_roughness", property.value());
            }
            if (auto property = params.get_optional<int>("metallicRoughnessTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "base_metalness_texture",
                    "base_metalness_texture_path",
                    importer_textures[property.value()],
                    false
                );
                // glTF metallic-roughness texture uses B channel for metallic
                properties.set("base_metalness_texture_channel", 2u);
                set_texture_property(
                    properties,
                    scene->device(),
                    "specular_roughness_texture",
                    "specular_roughness_texture_path",
                    importer_textures[property.value()],
                    false
                );
                // glTF metallic-roughness texture uses G channel for roughness
                properties.set("specular_roughness_texture_channel", 1);
            }
            if (auto property = params.get_optional<int>("normalTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "normal_texture",
                    "normal_texture_path",
                    importer_textures[property.value()],
                    false
                );
            }
            if (auto property = params.get_optional<float>("normalTexture_strength")) {
                properties.set("normal_texture_scale", property.value());
            }
            if (auto property = params.get_optional<float3>("emissiveFactor")) {
                properties.set("emission_luminance", 1.f);
                properties.set("emission_color", property.value());
            }
            if (auto property = params.get_optional<int>("emissiveTexture")) {
                properties.set("emission_luminance", 1.f);
                set_texture_property(
                    properties,
                    scene->device(),
                    "emission_color_texture",
                    "emission_color_texture_path",
                    importer_textures[property.value()]
                );
            }
        } else if (type && type.value() == "gltf_pbrSpecularGlossiness") {

        } else {
        }

        material->set_properties(properties);

        return material;

    } else {
        Material* material = nullptr;
        Properties properties;

        if (type && type.value() == "gltf_pbrMetallicRoughness") {
            material = scene->create_material<StandardMaterial>();

            if (auto property = params.get_optional<float4>("baseColorFactor")) {
                properties.set("base_color_factor", property.value().xyz());
            }
            if (auto property = params.get_optional<int>("baseColorTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "base_color_texture",
                    "base_color_texture_path",
                    importer_textures[property.value()]
                );
            }
            if (auto property = params.get_optional<float>("metallicFactor")) {
                properties.set("metallic_factor", property.value());
            }
            if (auto property = params.get_optional<float>("roughnessFactor")) {
                properties.set("roughness_factor", property.value());
            }
            if (auto property = params.get_optional<int>("metallicRoughnessTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "metallic_roughness_texture",
                    "metallic_roughness_texture_path",
                    importer_textures[property.value()],
                    false
                );
                // glTF metallic-roughness texture uses B channel for metallic
                properties.set("metallic_texture_channel", 2u);
            }
            if (auto property = params.get_optional<int>("normalTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "normal_texture",
                    "normal_texture_path",
                    importer_textures[property.value()],
                    false
                );
            }
            if (auto property = params.get_optional<float>("normalTexture_strength")) {
                properties.set("normal_texture_scale", property.value());
            }
            if (auto property = params.get_optional<float3>("emissiveFactor")) {
                properties.set("emissive_factor", property.value());
            }
            if (auto property = params.get_optional<int>("emissiveTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "emissive_texture",
                    "emissive_texture_path",
                    importer_textures[property.value()]
                );
            }
        } else if (type && type.value() == "gltf_pbrSpecularGlossiness") {
            material = scene->create_material<StandardSpecGlossMaterial>();

            if (auto property = params.get_optional<float4>("diffuseFactor")) {
                properties.set("diffuse_factor", property.value().xyz());
            }
            if (auto property = params.get_optional<int>("diffuseTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "diffuse_texture",
                    "diffuse_texture_path",
                    importer_textures[property.value()]
                );
            }
            if (auto property = params.get_optional<float3>("specularFactor")) {
                properties.set("specular_factor", property.value());
            }
            if (auto property = params.get_optional<float>("glossinessFactor")) {
                properties.set("glossiness_factor", property.value());
            }
            if (auto property = params.get_optional<int>("specularGlossinessTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "specular_glossiness_texture",
                    "specular_glossiness_texture_path",
                    importer_textures[property.value()]
                );
            }
            if (auto property = params.get_optional<int>("normalTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "normal_texture",
                    "normal_texture_path",
                    importer_textures[property.value()],
                    false
                );
            }
            if (auto property = params.get_optional<float>("normalTexture_strength")) {
                properties.set("normal_texture_scale", property.value());
            }
            if (auto property = params.get_optional<float3>("emissiveFactor")) {
                properties.set("emissive_factor", property.value());
            }
            if (auto property = params.get_optional<int>("emissiveTexture")) {
                set_texture_property(
                    properties,
                    scene->device(),
                    "emissive_texture",
                    "emissive_texture_path",
                    importer_textures[property.value()]
                );
            }
        } else {
            material = scene->create_material<StandardMaterial>();
        }

        if (auto property = params.get_optional<bool>("doubleSided")) {
            properties.set("double_sided", property.value());
        }

        material->set_name(importer_material.name);
        material->set_properties(properties);

        return material;
    }
}

static StaticMeshGeometry* load_importer_mesh(Scene* scene, const ImporterMesh& importer_mesh)
{
    StaticMeshGeometryDataDesc desc = {};
    std::vector<float2> converted_texcoords;
    auto texcoord_stream = importer_mesh.texcoord_stream();
    // SceneOptions define the target convention for loaded scene UVs.
    // ImporterMesh::uv_origin describes the mesh's authored texcoords, so convert only when they differ.
    if (importer_mesh.uv_origin != scene->options().uv_origin && texcoord_stream.valid()) {
        converted_texcoords.resize(importer_mesh.vertex_count());
        for (size_t i = 0; i < importer_mesh.vertex_count(); ++i) {
            converted_texcoords[i] = texcoord_stream[i];
            converted_texcoords[i].y = 1.f - converted_texcoords[i].y;
        }
    }

    desc.name = importer_mesh.name;
    desc.vertex_count = sgl::narrow_cast<uint32_t>(importer_mesh.vertex_count());
    desc.position_stream.data = reinterpret_cast<const float3*>(importer_mesh.position_stream().data);
    desc.position_stream.stride = importer_mesh.position_stream().stride;
    desc.normal_stream.data = reinterpret_cast<const float3*>(importer_mesh.normal_stream().data);
    desc.normal_stream.stride = importer_mesh.normal_stream().stride;
    desc.tangent_stream.data = reinterpret_cast<const float3*>(importer_mesh.tangent_stream().data);
    desc.tangent_stream.stride = importer_mesh.tangent_stream().stride;
    desc.handedness_stream.data = reinterpret_cast<const float*>(importer_mesh.handedness_stream().data);
    desc.handedness_stream.stride = importer_mesh.handedness_stream().stride;
    if (converted_texcoords.empty()) {
        desc.texcoord_stream.data = reinterpret_cast<const float2*>(texcoord_stream.data);
        desc.texcoord_stream.stride = texcoord_stream.stride;
    } else {
        desc.texcoord_stream.data = converted_texcoords.data();
        desc.texcoord_stream.stride = sizeof(float2);
    }
    for (const ImporterMesh::Subgeometry& subgeo : importer_mesh.subgeometries) {
        StaticMeshGeometryDataDesc::SubMesh sub_mesh = {};
        sub_mesh.name = subgeo.name;
        sub_mesh.index_count = sgl::narrow_cast<uint32_t>(subgeo.indices.size()) * 3;
        sub_mesh.index_stream.data = reinterpret_cast<const uint32_t*>(subgeo.indices.data());
        sub_mesh.index_stream.stride = sizeof(float);
        desc.sub_meshes.push_back(sub_mesh);
    }

    StaticMeshGeometry* geometry = scene->create_geometry<StaticMeshGeometry>();
    geometry->set_mesh_data(desc);
    return geometry;
}

static StaticCurveGeometry* load_importer_curve(Scene* scene, const ImporterCurve& importer_curve)
{
    StaticCurveGeometryDataDesc desc = {};
    desc.name = importer_curve.name;
    desc.vertex_count = sgl::narrow_cast<uint32_t>(importer_curve.positions.size());
    desc.position_stream.data = importer_curve.positions.data();
    desc.position_stream.stride = sizeof(float3);
    desc.radius_stream.data = importer_curve.radii.data();
    desc.radius_stream.stride = sizeof(float);
    desc.index_count = sgl::narrow_cast<uint32_t>(importer_curve.indices.size());
    desc.index_stream.data = importer_curve.indices.data();
    desc.index_stream.stride = sizeof(uint32_t);
    desc.indexing_mode = importer_curve.indexing_mode;

    StaticCurveGeometry* geometry = scene->create_geometry<StaticCurveGeometry>();
    geometry->set_curve_data(desc);
    return geometry;
}

static ref<ImporterScene> import_scene_for_create(const std::filesystem::path& path, bool recompute_normals)
{
    ImportOptions import_options{.recompute_normals = recompute_normals};

    ref<ImporterScene> importer_scene = import_scene(path, import_options);
    FALCOR_ASSERT(importer_scene);
    return importer_scene;
}

static SceneOptions resolve_scene_options(std::optional<UVOrigin> uv_origin, const ImporterScene& importer_scene)
{
    return SceneOptions(uv_origin.value_or(importer_scene.uv_origin));
}

void load_importer_scene(Scene* scene, const ImporterScene& importer_scene)
{
    // Import materials.
    std::map<std::string, Material*> name_to_material;
    for (const ImporterMaterial& importer_material : importer_scene.materials) {
        name_to_material[importer_material.name]
            = load_importer_material(scene, importer_material, importer_scene.textures);
    }

    Material* default_material = nullptr;
    auto resolve_material = [&](const std::string& material_name) -> Material*
    {
        if (auto it = name_to_material.find(material_name); it != name_to_material.end())
            return it->second;

        if (!material_name.empty())
            sgl::log_warn("Importer geometry references missing material '{}'; using default material", material_name);

        if (!default_material) {
            default_material = scene->create_material<StandardMaterial>();
            default_material->set_name("DefaultMaterial");
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
            materials.push_back(resolve_material(subgeo.material_name));
        }
        mesh_geometry_to_materials[mesh_geometry] = std::move(materials);
    }

    // Create a StaticCurveGeometry for each importer curve.
    std::vector<StaticCurveGeometry*> curve_geometries;
    std::map<StaticCurveGeometry*, Material*> curve_geometry_to_material;
    for (const ImporterCurve& importer_curve : importer_scene.curves) {
        StaticCurveGeometry* curve_geometry = load_importer_curve(scene, importer_curve);
        curve_geometries.push_back(curve_geometry);
        curve_geometry_to_material[curve_geometry] = resolve_material(importer_curve.material_name);
    }

    // Create Animation object from importer scene (before entities, so TransformAnimation can reference it).
    Animation* animation = nullptr;
    if (!importer_scene.animation.channels.empty()) {
        animation = scene->create_animation();
        animation->set_name(importer_scene.animation.name);
        animation->channels.reserve(importer_scene.animation.channels.size());
        for (const ImporterAnimationChannel& src : importer_scene.animation.channels) {
            AnimationChannel dst;
            dst.value_type = src.value_type;
            dst.interpolation_mode = src.interpolation_mode;
            dst.times = src.times;
            dst.values = src.values;
            dst.in_tangents = src.in_tangents;
            dst.out_tangents = src.out_tangents;
            animation->channels.push_back(std::move(dst));
        }
    }

    auto create_entities_recursive = [&](int node_idx, Entity* parent, std::string_view prefix, auto recurse) -> void
    {
        const ImporterNode& importer_node = importer_scene.nodes[node_idx];

        Entity* entity = scene->create_entity();
        entity->set_name(fmt::format("{}{}", prefix, importer_node.name));
        entity->set_transform(Transform(importer_node.transform));
        entity->set_parent(parent);

        // Create TransformAnimation component if this node has animation channels.
        if (animation
            && (importer_node.animation_translation >= 0 || importer_node.animation_rotation >= 0
                || importer_node.animation_scale >= 0)) {
            const int channel_count = static_cast<int>(animation->channels.size());
            TransformAnimation* ta = entity->create_component<TransformAnimation>();
            ta->set_animation(animation);
            if (importer_node.animation_translation >= 0 && importer_node.animation_translation < channel_count)
                ta->set_translation_channel(importer_node.animation_translation);
            if (importer_node.animation_rotation >= 0 && importer_node.animation_rotation < channel_count)
                ta->set_rotation_channel(importer_node.animation_rotation);
            if (importer_node.animation_scale >= 0 && importer_node.animation_scale < channel_count)
                ta->set_scale_channel(importer_node.animation_scale);
        }

        if (importer_node.mesh_index >= 0) {
            FALCOR_ASSERT_LT(importer_node.mesh_index, mesh_geometries.size());
            GeometryInstance* instance = entity->create_component<GeometryInstance>();
            StaticMeshGeometry* mesh_geometry = mesh_geometries[importer_node.mesh_index];
            instance->set_geometry(mesh_geometry);
            auto it = mesh_geometry_to_materials.find(mesh_geometry);
            FALCOR_ASSERT(it != mesh_geometry_to_materials.end());
            instance->set_materials(it->second);
        }
        if (importer_node.curve_index >= 0) {
            FALCOR_ASSERT_LT(importer_node.curve_index, curve_geometries.size());
            GeometryInstance* instance = entity->create_component<GeometryInstance>();
            StaticCurveGeometry* curve_geometry = curve_geometries[importer_node.curve_index];
            instance->set_geometry(curve_geometry);
            auto it = curve_geometry_to_material.find(curve_geometry);
            if (it != curve_geometry_to_material.end()) {
                instance->set_materials({it->second});
            }
        }
        if (importer_node.light_index >= 0) {
            FALCOR_ASSERT_LT(importer_node.light_index, importer_scene.lights.size());
            const ImporterLight& importer_light = importer_scene.lights[importer_node.light_index];
            switch (importer_light.type) {
            case ImporterLight::Type::distant: {
                DistantLight* distant_light = entity->create_component<DistantLight>();
                distant_light->set_radiance(importer_light.intensity);
                // Scene distant light expects a half angle
                distant_light->set_cutoff_angle(0.5f * importer_light.degree_angular_diameter);
                break;
            }
            case ImporterLight::Type::rectangular:
                // TODO(scene): handle (create geometry and emissive material)
                break;
            case ImporterLight::Type::sphere:
                // TODO(scene): handle (create geometry and emissive material)
                break;
            case ImporterLight::Type::point: {
                PointLight* point_light = entity->create_component<PointLight>();
                point_light->set_intensity(importer_light.intensity);
                break;
            }
            case ImporterLight::Type::disk:
                // TODO(scene): handle (create geometry and emissive material)
                break;
            case ImporterLight::Type::dome: {
                if (importer_light.env_map_path.empty()) {
                    ConstantLight* constant_light = entity->create_component<ConstantLight>();
                    constant_light->set_radiance(importer_light.intensity);
                    constant_light->set_exposure(importer_light.exposure);
                } else {
                    EnvMapLight* env_map_light = entity->create_component<EnvMapLight>();
                    env_map_light->set_env_map_path(importer_light.env_map_path);
                    env_map_light->set_exposure(importer_light.exposure);
                }
                break;
            }
            case ImporterLight::Type::constant: {
                ConstantLight* constant_light = entity->create_component<ConstantLight>();
                constant_light->set_radiance(importer_light.intensity);
                break;
            }
            }
        }
        if (importer_node.camera_index >= 0) {
            FALCOR_ASSERT_LT(importer_node.camera_index, importer_scene.cameras.size());
            const ImporterCamera& importer_camera = importer_scene.cameras[importer_node.camera_index];
            Camera* camera = entity->create_component<Camera>();
            camera->set_name(importer_camera.name);
            camera->set_focal_length(importer_camera.focal_length);
            camera->set_focus_distance(importer_camera.focus_distance);
            camera->set_fstop(importer_camera.fstop);
            camera->set_fov_y(importer_camera.vertical_fov_degrees());
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
            FALCOR_ASSERT_LT(importer_node.prototype_index, importer_scene.prototypes.size());
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

void load_scene(Scene* scene, const std::filesystem::path& path, bool recompute_normals)
{
    ImportOptions import_options{.recompute_normals = recompute_normals};

    ref<ImporterScene> importer_scene = import_scene(path, import_options);
    FALCOR_ASSERT(importer_scene);

    // TODO: Allow this to be controlled somehow.
    // importer_scene->make_clay_scene();
    return load_importer_scene(scene, *importer_scene);
}

namespace detail {

ref<Scene> create_scene(
    ref<sgl::Device> device,
    const std::filesystem::path& path,
    bool recompute_normals,
    std::optional<UVOrigin> uv_origin
)
{
    const std::string extension = sgl::string::to_lower(path.extension().string());

    if (extension == ".py") {
        ImportOptions import_options{.recompute_normals = recompute_normals};
        ref<Importer> importer = Importer::create(import_options);
        importer->set_source_path(path);
        ScopedCurrentImporter current_importer(importer);

        PythonInterpreter::get().create_context().execute_file(path, "__falcor2_scene__");

        return create_scene(std::move(device), *importer, uv_origin);
    }

    ref<ImporterScene> importer_scene = import_scene_for_create(path, recompute_normals);
    return create_scene(std::move(device), *importer_scene, uv_origin);
}

ref<Scene> create_scene(ref<sgl::Device> device, const ImporterScene& importer_scene, std::optional<UVOrigin> uv_origin)
{
    auto scene = ref<Scene>{new Scene(std::move(device), resolve_scene_options(uv_origin, importer_scene))};
    load_importer_scene(scene.get(), importer_scene);
    return scene;
}

ref<Scene> create_scene(
    ref<sgl::Device> device,
    const Importer& importer,
    std::optional<UVOrigin> uv_origin,
    bool add_default_camera_best_view,
    float camera_aspect
)
{
    ref<ImporterScene> importer_scene = importer.build_importer_scene();
    FALCOR_ASSERT(importer_scene);
    if (add_default_camera_best_view && importer_scene->cameras.empty())
        importer_scene->add_default_camera_best_view(50.f, camera_aspect);

    ref<Scene> scene = create_scene(std::move(device), *importer_scene, uv_origin);
    importer.run_scene_created_callbacks(scene);
    return scene;
}

} // namespace detail

} // namespace falcor
