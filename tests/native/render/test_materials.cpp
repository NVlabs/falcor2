// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "testing.h"

#include "falcor2/core/platform.h"
#include "falcor2/core/properties.h"
#include "falcor2/core/asset_resolver.h"
#include "falcor2/core/reflection/class_descriptor.h"
#include "falcor2/render/material/mdl/mdl_material.h"
#include "falcor2/render/geometry/geometry_vertex_attribute.h"
#include "falcor2/render/material/emission_only_material.h"
#include "falcor2/render/material/mtlx/codegen/codegen.h"
#include "falcor2/render/material/mtlx/codegen/codegen_cache.h"
#include "falcor2/render/material/mtlx/mtlx_material.h"
#include "falcor2/render/material/standard_material.h"
#include "falcor2/render/material/standard_specgloss_material.h"
#include "falcor2/render/material/gltf_material.h"
#include "falcor2/render/material/gltf_specgloss_material.h"
#include "falcor2/render/material/shared_gltf_texture_selectors.h"
#include "falcor2/render/material/unlit_material.h"
#include "falcor2/render/material_aov_registry.h"
#include "falcor2/render/scene.h"
#include "falcor2/utils/buffer_offset_pair.h"

#include <fmt/format.h>
#include <sgl/device/device.h>
#include <sgl/device/kernel.h>

#include <algorithm>
#include <array>
#include <barrier>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <optional>
#include <regex>
#include <sstream>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

using namespace falcor;

namespace {
namespace F = shared::gltfmat::FeatureBits;

class DataTestMaterial final : public Material {
    FALCOR_SCENE_OBJECT(DataTestMaterial, Material)
public:
    DataTestMaterial() { set_slang_type_name("InvalidMaterial"); }
    using Material::data;
    using Material::data_ref;
    using Material::ensure_data_size;

    void set_probe_module(ref<sgl::SlangModule> module)
    {
        m_probe_module = std::move(module);
        set_slang_type_name("DataPoolProbeMaterial");
        mark_dirty_properties();
    }

    ref<sgl::SlangModule> required_module() const override { return m_probe_module; }

    void write_to_cursor(sgl::BufferElementCursor cursor) const override
    {
        if (m_probe_module) {
            cursor["extra_data"] = data_ref();
            cursor["has_data"] = uint32_t(!data().empty());
        }
    }

    void set_value(size_t size, uint32_t value)
    {
        ensure_data_size(size);
        std::memcpy(mutable_data().data(), &value, sizeof(value));
    }

private:
    ref<sgl::SlangModule> m_probe_module;
};

class UpdateProbeMaterial final : public Material {
    FALCOR_SCENE_OBJECT(UpdateProbeMaterial, Material)
public:
    UpdateProbeMaterial() { set_slang_type_name("InvalidMaterial"); }

    void update(SceneUpdateContext&) override { ++m_update_count; }
    uint32_t update_count() const { return m_update_count; }

private:
    uint32_t m_update_count{0};
};

class GLTFDataTestMaterial final : public GLTFMaterial {
public:
    using Material::data;
    using Material::data_ref;
};

class GLTFSpecGlossDataTestMaterial final : public GLTFSpecGlossMaterial {
public:
    using Material::data;
    using Material::data_ref;
};
} // namespace

TEST_CASE_GPU("material_system_updates_clean_non_gltf_materials")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* probe = scene->create_material<UpdateProbeMaterial>();
    auto* gltf = scene->create_material<GLTFMaterial>();
    scene->update();
    const uint32_t before = probe->update_count();
    REQUIRE_GT(before, 0u);

    // Editing a different material dirties the collection, while this probe
    // remains clean. Non-glTF update hooks must still run in that pass.
    gltf->set_property<float16_t>("roughness_factor", float16_t(0.5f));
    scene->update();
    CHECK_GT(probe->update_count(), before);
}

TEST_CASE("glTF material reflected property inventory")
{
    GLTFMaterial metallic_roughness;
    GLTFSpecGlossMaterial specular_glossiness;
    const std::array<const Material*, 2> materials{&metallic_roughness, &specular_glossiness};

    std::string output_env;
    bool output_requested = false;
#ifdef _WIN32
    char* output_env_buffer = nullptr;
    size_t output_env_length = 0;
    if (_dupenv_s(&output_env_buffer, &output_env_length, "FALCOR2_GLTF_PROPERTY_INVENTORY_PATH") == 0
        && output_env_buffer) {
        output_env = output_env_buffer;
        output_requested = true;
        std::free(output_env_buffer);
    }
#else
    if (const char* value = std::getenv("FALCOR2_GLTF_PROPERTY_INVENTORY_PATH")) {
        output_env = value;
        output_requested = true;
    }
#endif
    std::ofstream output;
    if (output_requested) {
        const std::filesystem::path default_path
            = testing::project_directory() / ".tmp/gltf-material-specialization/property-inventory-before.tsv";
        const std::filesystem::path output_path
            = output_env.empty() || output_env == "default" ? default_path : std::filesystem::path(output_env);
        if (!output_path.parent_path().empty())
            std::filesystem::create_directories(output_path.parent_path());
        output.open(output_path, std::ios::binary | std::ios::trunc);
        REQUIRE(output.is_open());
        output << "class\towner\tindex\tname\tcpp_type\tstorage_type\tread_only\tserializable\thas_default\t"
                  "is_default\tinitial_value\tui_flags\trange\tenum_is_flags\tenum_items\tdoc\n";
    }

    const auto escape_tsv = [](std::string_view value)
    {
        std::string result;
        for (char character : value) {
            switch (character) {
            case '\\':
                result += "\\\\";
                break;
            case '\t':
                result += "\\t";
                break;
            case '\n':
                result += "\\n";
                break;
            case '\r':
                result += "\\r";
                break;
            default:
                result += character;
                break;
            }
        }
        return result;
    };

    for (const Material* material : materials) {
        const auto& class_desc = material->class_descriptor();
        const Properties values = material->properties();
        size_t property_count = 0;
        size_t serializable_count = 0;
        for (const auto* property : class_desc.all_property_range()) {
            CHECK(class_desc.find_property(property->name()) == property);
            if (property->is_serializable_to_properties()) {
                ++serializable_count;
                CHECK(values.has_property(property->name()));
            }

            if (output.is_open()) {
                std::string owner;
                for (const auto* level = &class_desc; level; level = level->base()) {
                    const auto own_properties = level->properties();
                    if (std::find(own_properties.begin(), own_properties.end(), property) != own_properties.end()) {
                        owner = std::string(level->name());
                        break;
                    }
                }

                std::string storage_type;
                std::string initial_value;
                if (property->is_serializable_to_properties()) {
                    storage_type = sgl::enum_to_string(values.type(property->name()));
                    initial_value = values.as_string(property->name());
                }

                std::string range;
                if (const auto* value_range = property->value_range())
                    range = fmt::format("{},{}", value_range->min, value_range->max);

                std::string enum_items;
                bool enum_is_flags = false;
                if (const auto* enum_desc = property->enum_descriptor()) {
                    enum_is_flags = enum_desc->is_flags;
                    for (const auto& item : enum_desc->items)
                        enum_items += fmt::format("{}:{};", item.value, item.name);
                }

                output << escape_tsv(class_desc.name()) << '\t' << escape_tsv(owner) << '\t' << property_count << '\t'
                       << escape_tsv(property->name()) << '\t'
                       << escape_tsv(platform::demangle_type_name(property->type())) << '\t' << escape_tsv(storage_type)
                       << '\t' << property->is_read_only() << '\t' << property->is_serializable_to_properties() << '\t'
                       << property->has_default_value() << '\t'
                       << (property->has_default_value() && property->is_default(material)) << '\t'
                       << escape_tsv(initial_value) << '\t' << static_cast<uint32_t>(property->ui_flags()) << '\t'
                       << escape_tsv(range) << '\t' << enum_is_flags << '\t' << escape_tsv(enum_items) << '\t'
                       << escape_tsv(property->doc()) << '\n';
            }
            ++property_count;
        }
        CHECK(property_count > 0);
        CHECK(values.size() == serializable_count);
    }

    if (output.is_open()) {
        output.close();
        CHECK(output.good());
    }
}

TEST_CASE_GPU("glTF material grouped properties round-trip")
{
    auto scene = Scene::create(ref(ctx.device));
    auto* metallic_roughness = scene->create_material<GLTFMaterial>();
    Properties metallic_edits;
    metallic_edits.set("base_color_factor", float16_t3(0.2f, 0.4f, 0.6f));
    metallic_edits.set("base_color_texture_path", std::filesystem::path("base_color.png"));
    metallic_edits.set("base_color_texture_texcoord", 2u);
    metallic_edits.set("base_color_texture_offset", float2(0.25f, 0.5f));
    metallic_edits.set("metallic_factor", float16_t(0.75f));
    metallic_edits.set("normal_texture_scale", float16_t(0.4f));
    metallic_edits.set("emissive_factor", float3(0.1f, 0.2f, 0.3f));
    metallic_edits.set("emissive_texture_rotation", 0.3f);
    metallic_edits.set("clearcoat_factor", float16_t(0.5f));
    metallic_edits.set("clearcoat_normal_texture_path", std::filesystem::path("coat_normal.png"));
    metallic_edits.set("sheen_color_factor", float3(0.3f, 0.2f, 0.1f));
    metallic_edits.set("iridescence_thickness_texture_uv_scale", float2(0.8f, 0.7f));
    metallic_edits.set("attenuation_distance", 3.f);
    metallic_edits.set("diffuse_transmission_color_texture_texcoord", 3u);
    metallic_roughness->set_properties(metallic_edits);

    auto* metallic_copy = scene->create_material<GLTFMaterial>();
    metallic_copy->set_properties(metallic_roughness->properties());
    CHECK(metallic_copy->properties() == metallic_roughness->properties());

    auto* specular_glossiness = scene->create_material<GLTFSpecGlossMaterial>();
    Properties specular_edits;
    specular_edits.set("diffuse_factor", float16_t3(0.6f, 0.4f, 0.2f));
    specular_edits.set("diffuse_texture_path", std::filesystem::path("diffuse.png"));
    specular_edits.set("diffuse_texture_offset", float2(0.1f, 0.2f));
    specular_edits.set("specular_factor", float16_t3(0.9f, 0.8f, 0.7f));
    specular_edits.set("specular_glossiness_texture_texcoord", 4u);
    specular_edits.set("glossiness_factor", float16_t(0.35f));
    specular_edits.set("normal_texture_scale", float16_t(0.6f));
    specular_edits.set("emissive_factor", float3(0.1f, 0.05f, 0.025f));
    specular_edits.set("emissive_texture_uv_scale", float2(0.5f, 0.75f));
    specular_glossiness->set_properties(specular_edits);

    auto* specular_copy = scene->create_material<GLTFSpecGlossMaterial>();
    specular_copy->set_properties(specular_glossiness->properties());
    CHECK(specular_copy->properties() == specular_glossiness->properties());
}

TEST_CASE_GPU("material property edits mark emission dirty independently")
{
    auto scene = Scene::create(ref(ctx.device));
    auto& collection = scene->material_collection();
    using Flags = Material::DirtyFlags;
    const auto emission_properties = Flags::properties | Flags::emission;
    const auto emission_resources = Flags::resources | Flags::emission;
    Material* material = nullptr;
    const auto edit = [&](std::string_view name, const auto& value, Flags expected)
    {
        CAPTURE(name);
        Properties props;
        props.set(name, value);
        collection.reset_dirty_flags();
        material->set_properties(props);
        CHECK(collection.combined_dirty_flags() == expected);
        CHECK(collection.dirty_flags().back() == expected);
    };

    // Resource paths need not exist: dirty marking does not load them or update the scene.
    // Each material checks ordinary edits while dark and lit, then disables emission again.
    SUBCASE("StandardMaterial")
    {
        material = scene->create_material("StandardMaterial");
        edit("emissive_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("base_color_texture_path", std::filesystem::path("shading.png"), Flags::resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("base_color_factor", float3(0.25f), Flags::properties);
        edit("emissive_factor", float3(1.f), emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("base_color_factor", float3(0.5f), Flags::properties);
        edit("emissive_factor", float3(0.f), emission_properties);
        edit("emissive_texture_texcoord", 1u, emission_properties);
    }
    SUBCASE("StandardSpecGlossMaterial")
    {
        material = scene->create_material("StandardSpecGlossMaterial");
        edit("emissive_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("diffuse_texture_path", std::filesystem::path("shading.png"), Flags::resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("diffuse_factor", float3(0.25f), Flags::properties);
        edit("emissive_factor", float3(1.f), emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("diffuse_factor", float3(0.5f), Flags::properties);
        edit("emissive_factor", float3(0.f), emission_properties);
        edit("emissive_texture_texcoord", 1u, emission_properties);
    }
    SUBCASE("GLTFMaterial")
    {
        material = scene->create_material("GLTFMaterial");
        edit("emissive_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("base_color_texture_path", std::filesystem::path("shading.png"), Flags::resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("base_color_factor", float3(0.25f), Flags::properties);
        edit("emissive_factor", float3(1.f), emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("base_color_factor", float3(0.5f), Flags::properties);
        edit("emissive_factor", float3(0.f), emission_properties);
        edit("emissive_texture_texcoord", 1u, emission_properties);
        edit("emissive_texture_offset", float2(0.25f), emission_properties);
        edit("emissive_texture_rotation", 0.5f, emission_properties);
        edit("emissive_texture_uv_scale", float2(2.f), emission_properties);
    }
    SUBCASE("GLTFSpecGlossMaterial")
    {
        material = scene->create_material("GLTFSpecGlossMaterial");
        edit("emissive_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("diffuse_texture_path", std::filesystem::path("shading.png"), Flags::resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("diffuse_factor", float3(0.25f), Flags::properties);
        edit("emissive_factor", float3(1.f), emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("diffuse_factor", float3(0.5f), Flags::properties);
        edit("emissive_factor", float3(0.f), emission_properties);
        edit("emissive_texture_texcoord", 1u, emission_properties);
        edit("emissive_texture_offset", float2(0.25f), emission_properties);
        edit("emissive_texture_rotation", 0.5f, emission_properties);
        edit("emissive_texture_uv_scale", float2(2.f), emission_properties);
    }
    SUBCASE("OpenPBRMaterial")
    {
        material = scene->create_material("OpenPBRMaterial");
        edit("emission_color_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("base_color_texture_path", std::filesystem::path("shading.png"), Flags::resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("base_color", float3(0.25f), Flags::properties);
        edit("emission_luminance", 1.f, emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("base_color", float3(0.5f), Flags::properties);
        edit("emission_luminance", 0.f, emission_properties);
        edit("emission_color", float3(0.5f), emission_properties);
        edit("emission_color_texture_texcoord", 1u, emission_properties);
        edit("emission_luminance_texture_path", std::filesystem::path("luminance.png"), emission_resources);
        edit("emission_luminance_texture_channel", 1u, emission_resources);
    }
    SUBCASE("EmissionOnlyMaterial")
    {
        material = scene->create_material("EmissionOnlyMaterial");
        edit("emission_color_texture_path", std::filesystem::path("emission.png"), emission_resources);
        edit("enable_backface_culling", true, Flags::properties | Flags::backface_culling);
        edit("nested_priority", 1u, Flags::properties);
        edit("emission_luminance", 1.f, emission_properties);
        edit("nested_priority", 2u, Flags::properties);
        edit("emission_luminance", 0.f, emission_properties);
        edit("emission_color", float3(0.5f), emission_properties);
        edit("enable_color_temperature", true, emission_properties);
        edit("color_temperature", 4000.f, emission_properties);
    }
}

TEST_CASE_GPU("gltf_transform_data_lifecycle")
{
    auto scene = Scene::create(ref(ctx.device));
    auto first = ref(scene->create_material<GLTFDataTestMaterial>());
    auto second = ref(scene->create_material<GLTFSpecGlossDataTestMaterial>());
    scene->update();
    auto layout = scene->render_module()->layout();
    auto check_material_size = [&](const std::string& type_name)
    {
        const std::string buffer_type_name = fmt::format("StructuredBuffer<{}>", type_name);
        auto buffer_type = layout->find_type_by_name(buffer_type_name.c_str());
        REQUIRE(buffer_type);
        CHECK_LT(layout->get_type_layout(buffer_type)->element_type_layout()->stride(), sizeof(shared::MaterialData));
    };
    check_material_size(first->slang_type_name());
    check_material_size(second->slang_type_name());
    CHECK(first->data().empty());
    CHECK(second->data().empty());
    CHECK_EQ(first->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());

    // Authored UV transforms are retained even when there is no usable texture.
    first->set_property<float2>("base_color_texture_offset", float2(0.25f, 0.f));
    second->set_property<float2>("diffuse_texture_offset", float2(0.5f, 0.f));
    scene->update();
    CHECK(first->data().empty());
    CHECK_EQ(first->effective_masks().textures, 0u);
    CHECK_EQ(first->effective_masks().transforms, 0u);
    CHECK(second->data().empty());
    CHECK_EQ(second->effective_masks().textures, 0u);
    CHECK_EQ(second->effective_masks().transforms, 0u);

    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm_srgb,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    first->set_property("base_color_texture", texture);
    first->set_property("base_color_texture_path", std::filesystem::path("ignored_base_color.png"));
    second->set_property("diffuse_texture", texture);
    second->set_property("diffuse_texture_path", std::filesystem::path("ignored_diffuse.png"));
    scene->update();
    CHECK(first->opacity_desc().texture_handle.texture() == texture);
    CHECK(second->opacity_desc().texture_handle.texture() == texture);
    CHECK(sgl::math::all(first->get_property<float2>("base_color_texture_offset") == float2(0.25f, 0.f)));
    CHECK(sgl::math::all(second->get_property<float2>("diffuse_texture_offset") == float2(0.5f, 0.f)));
    REQUIRE_EQ(first->data().size(), 32u);
    const auto first_offset = first->data_ref().offset;
    CHECK_NE(first_offset, second->data_ref().offset);
    CHECK_EQ(first->data_ref().buffer.data(), second->data_ref().buffer.data());
    CHECK_EQ(first->effective_masks().textures, shared::gltfmat::TextureBits::BASE_COLOR);
    CHECK_EQ(first->effective_masks().transforms, shared::gltfmat::TextureBits::BASE_COLOR);
    REQUIRE_EQ(second->data().size(), 32u); // 24-byte transform, padded to the pool's 16-byte alignment.
    CHECK_EQ(second->effective_masks().textures, shared::gltfmat::SpecGlossTextureBits::DIFFUSE);
    CHECK_EQ(second->effective_masks().transforms, shared::gltfmat::SpecGlossTextureBits::DIFFUSE);
    float authored_offset;
    std::memcpy(&authored_offset, first->data().data() + 2 * sizeof(float), sizeof(authored_offset));
    CHECK_EQ(authored_offset, 0.25f);
    const std::vector<uint8_t> first_bytes(first->data().begin(), first->data().end());

    first->set_property<uint32_t>("base_color_texture_texcoord", 2u);
    scene->update();
    CHECK_EQ(first->effective_masks().nonzero_texcoords, shared::gltfmat::TextureBits::BASE_COLOR);
    first->set_property<uint32_t>("base_color_texture_texcoord", 0u);
    scene->update();
    CHECK_EQ(first->effective_masks().nonzero_texcoords, 0u);

    std::vector<GLTFSpecGlossDataTestMaterial*> neighbors;
    for (uint32_t i = 0; i < 100; ++i) {
        auto* neighbor = scene->create_material<GLTFSpecGlossDataTestMaterial>();
        neighbor->set_property("diffuse_texture", texture);
        neighbor->set_property<float2>("diffuse_texture_offset", float2(float(i + 1), 0.f));
        neighbors.push_back(neighbor);
    }
    scene->update();
    CHECK_EQ(first->data_ref().offset, first_offset);
    CHECK(std::equal(first_bytes.begin(), first_bytes.end(), first->data().begin()));
    CHECK_EQ(first->data_ref().buffer.data(), neighbors.back()->data_ref().buffer.data());

    // A factor-only edit must rewrite both workflows' transform payloads.
    auto module = ctx.device->load_module_from_source("gltf_transform_upload_probe", R"(
import falcor2.utils;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(BufferOffsetPair first, BufferOffsetPair second, RWStructuredBuffer<float> result)
{
    result[0] = first.load<float>(0);
    result[1] = second.load<float>(0);
}
)");
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program({module}, {module->entry_point("main")})}
    );
    auto result = ctx.device->create_buffer({.size = 2 * sizeof(float), .usage = sgl::BufferUsage::unordered_access});
    const float canary = 7.f;
    float original_first, original_second;
    std::memcpy(&original_first, first->data().data(), sizeof(float));
    std::memcpy(&original_second, second->data().data(), sizeof(float));
    std::memcpy(const_cast<uint8_t*>(first->data().data()), &canary, sizeof(float));
    std::memcpy(const_cast<uint8_t*>(second->data().data()), &canary, sizeof(float));
    first->set_property<float16_t>("roughness_factor", float16_t(0.25f));
    second->set_property<float16_t>("glossiness_factor", float16_t(0.75f));
    scene->update();
    kernel->dispatch(
        uint3(1),
        [&](sgl::ShaderCursor cursor)
        {
            cursor = cursor.find_entry_point(0);
            cursor["first"] = first->data_ref();
            cursor["second"] = second->data_ref();
            cursor["result"] = result;
        }
    );
    const std::vector<float> expected{original_first, original_second};
    CHECK(result->get_elements<float>() == expected);
    float shadow_first, shadow_second;
    std::memcpy(&shadow_first, first->data().data(), sizeof(float));
    std::memcpy(&shadow_second, second->data().data(), sizeof(float));
    CHECK_EQ(shadow_first, original_first);
    CHECK_EQ(shadow_second, original_second);

    second->set_property<float3>("emissive_factor", float3(0.25f));
    second->set_property("emissive_texture", texture);
    second->set_property<float>("emissive_texture_rotation", 0.3f);
    scene->update();
    CHECK_EQ(second->data().size(), 48);
    CHECK(std::equal(first_bytes.begin(), first_bytes.end(), first->data().begin()));
    second->set_property<float2>("diffuse_texture_offset", float2(0.f));
    scene->update();
    CHECK_EQ(second->data().size(), 48); // Nonzero shrink retains capacity by design.
    second->set_property<float>("emissive_texture_rotation", 0.f);
    first->set_property<float2>("base_color_texture_offset", float2(0.f));
    scene->update();
    CHECK(first->data().empty());
    CHECK(second->data().empty());
    CHECK_EQ(first->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
    first->set_property<float2>("base_color_texture_offset", float2(0.25f, 0.f));
    scene->update();
    CHECK_EQ(first->data_ref().offset % 16u, 0u);
    CHECK(std::equal(first_bytes.begin(), first_bytes.end(), first->data().begin()));

    // Transmission's extension record precedes all transform records, and its
    // later texture slot is packed after base color in the transform region.
    first->set_property<float16_t>("transmission_factor", float16_t(0.5f));
    first->set_property("transmission_texture", texture);
    first->set_property<float2>("transmission_texture_offset", float2(0.75f, 0.f));
    scene->update();
    REQUIRE_GE(first->data().size(), 32u + 2u * 24u);
    CHECK_EQ(
        first->effective_masks().transforms,
        shared::gltfmat::TextureBits::BASE_COLOR | shared::gltfmat::TextureBits::TRANSMISSION
    );
    float transmission_offset;
    std::memcpy(&transmission_offset, first->data().data() + 32 + 24 + 2 * sizeof(float), sizeof(float));
    CHECK_EQ(transmission_offset, 0.75f);
    std::vector<uint8_t> with_multiscatter(first->data().begin(), first->data().begin() + 80);
    first->set_property<float16_t>("metallic_factor", float16_t(0.f));
    scene->update();
    CHECK_EQ(first->effective_masks().features & shared::gltfmat::FeatureBits::MULTISCATTER, 0u);
    CHECK(std::equal(with_multiscatter.begin(), with_multiscatter.end(), first->data().begin()));
    first->set_property<float16_t>("metallic_factor", float16_t(1.f));
    scene->update();
    CHECK_NE(first->effective_masks().features & shared::gltfmat::FeatureBits::MULTISCATTER, 0u);
    CHECK(std::equal(with_multiscatter.begin(), with_multiscatter.end(), first->data().begin()));
    first->set_property<float2>("base_color_texture_offset", float2(0.f));
    scene->update();
    std::memcpy(&transmission_offset, first->data().data() + 32 + 2 * sizeof(float), sizeof(float));
    CHECK_EQ(transmission_offset, 0.75f); // Sparse packing moves binding 6 to slot zero.
    first->set_property<float2>("transmission_texture_offset", float2(0.f));
    first->set_property<float16_t>("transmission_factor", float16_t(0.f));
    scene->update();
    CHECK(first->data().empty());
    CHECK_EQ(first->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
    first->remove();
    scene->update();
    CHECK_EQ(first->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
    CHECK_EQ(first->material_id(), shared::MaterialID::invalid);
}

TEST_CASE_GPU("gltf_volume_data_lifecycle")
{
    auto scene = Scene::create(ref(ctx.device));
    auto material = ref(scene->create_material<GLTFDataTestMaterial>());
    scene->update();
    CHECK(material->data().empty());
    material->set_property<float>("thickness_factor", 1.f);
    material->set_property<float3>("attenuation_color", float3(0.5f, 1.f, 0.f));
    material->set_property<float>("attenuation_distance", 2.f);
    scene->update();
    auto layout = scene->render_module()->layout();
    auto volume_type = layout->find_type_by_name("StructuredBuffer<GLTFVolumeRecord>");
    REQUIRE(volume_type);
    const auto stride = layout->get_type_layout(volume_type)->element_type_layout()->stride();
    REQUIRE_EQ(stride, 32u);
    REQUIRE_EQ(material->data().size(), 32u);
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::VOLUME,
        shared::gltfmat::FeatureBits::VOLUME
    );
    float3 sigma;
    std::memcpy(&sigma, material->data().data(), sizeof(sigma));
    CHECK_EQ(sigma.x, doctest::Approx(std::log(2.f) / 2.f));
    CHECK_EQ(sigma.y, 0.f);
    CHECK(std::isinf(sigma.z));

    // A transform on a missing texture consumes neither an extension nor UV slot.
    material->set_property<float2>("transmission_texture_offset", float2(0.75f, 0.f));
    scene->update();
    CHECK_EQ(material->data().size(), 32u);
    CHECK_EQ(material->effective_masks().transforms, 0u);
    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("transmission_texture", texture);
    material->set_property<float16_t>("transmission_factor", float16_t(0.5f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::TRANSMISSION,
        shared::gltfmat::FeatureBits::TRANSMISSION
    );
    CHECK_EQ(material->effective_masks().transforms, shared::gltfmat::TextureBits::TRANSMISSION);
    REQUIRE_GE(material->data().size(), 2u * stride + 24u);
    float offset;
    std::memcpy(&offset, material->data().data() + 2 * stride + 2 * sizeof(float), sizeof(float));
    CHECK_EQ(offset, 0.75f);
    const std::vector<uint8_t> packed(material->data().begin(), material->data().begin() + 2 * stride + 24);
    for (int i = 0; i < 100; ++i) {
        auto neighbor = scene->create_material<GLTFDataTestMaterial>();
        neighbor->set_property<float>("thickness_factor", 1.f);
    }
    scene->update();
    CHECK(std::equal(packed.begin(), packed.end(), material->data().begin()));
    material->set_property<float>("attenuation_distance", 4.f);
    scene->update();
    std::memcpy(&sigma, material->data().data() + stride, sizeof(sigma));
    CHECK_EQ(sigma.x, doctest::Approx(std::log(2.f) / 4.f));
    material->set_property<float>("thickness_factor", 0.f);
    scene->update();
    CHECK_EQ(material->effective_masks().features & shared::gltfmat::FeatureBits::VOLUME, 0u);
    std::memcpy(&offset, material->data().data() + stride + 2 * sizeof(float), sizeof(float));
    CHECK_EQ(offset, 0.75f);
    material->set_property<float2>("transmission_texture_offset", float2(0.f));
    material->set_property<float16_t>("transmission_factor", float16_t(0.f));
    scene->update();
    CHECK(material->data().empty());
    material->set_property<float>("thickness_factor", 1.f);
    scene->update();
    CHECK_FALSE(material->data().empty());
    material->remove();
    scene->update();
    CHECK_EQ(material->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
}

TEST_CASE_GPU("gltf_clearcoat_data_lifecycle")
{
    auto scene = Scene::create(ref(ctx.device));
    auto material = ref(scene->create_material<GLTFDataTestMaterial>());
    scene->update();
    CHECK(material->data().empty());
    material->set_property<float16_t>("clearcoat_factor", float16_t(1.f));
    material->set_property<float16_t>("clearcoat_normal_texture_scale", float16_t(0.25f));
    scene->update();
    auto layout = scene->render_module()->layout();
    auto coat_type = layout->find_type_by_name("StructuredBuffer<GLTFClearcoatRecord>");
    REQUIRE(coat_type);
    const auto stride = layout->get_type_layout(coat_type)->element_type_layout()->stride();
    REQUIRE_EQ(stride, 32u);
    REQUIRE_EQ(material->data().size(), stride);
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::CLEARCOAT,
        shared::gltfmat::FeatureBits::CLEARCOAT
    );
    float16_t scale;
    std::memcpy(&scale, material->data().data() + 28, sizeof(scale));
    CHECK_EQ(float(scale), doctest::Approx(0.25f));

    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("clearcoat_texture", texture);
    material->set_property("clearcoat_roughness_texture", texture);
    material->set_property("clearcoat_normal_texture", texture);
    scene->update();
    CHECK_EQ(
        material->effective_masks().textures,
        shared::gltfmat::TextureBits::CLEARCOAT | shared::gltfmat::TextureBits::CLEARCOAT_ROUGHNESS
            | shared::gltfmat::TextureBits::CLEARCOAT_NORMAL
    );

    // Each logical texture-info stores its own effective UV selector.
    uint32_t selectors;
    for (uint32_t texcoord : {7u, 0u}) {
        material->set_property<uint32_t>("clearcoat_texture_texcoord", texcoord);
        scene->update();
        std::memcpy(&selectors, material->data().data() + 4, sizeof(selectors));
        CHECK_EQ(selectors, texcoord);
    }
    for (uint32_t texcoord : {6u, 0u}) {
        material->set_property<uint32_t>("clearcoat_roughness_texture_texcoord", texcoord);
        scene->update();
        std::memcpy(&selectors, material->data().data() + 12, sizeof(selectors));
        CHECK_EQ(selectors, texcoord);
    }
    for (uint32_t texcoord : {5u, 0u}) {
        material->set_property<uint32_t>("clearcoat_normal_texture_texcoord", texcoord);
        scene->update();
        std::memcpy(&selectors, material->data().data() + 20, sizeof(selectors));
        CHECK_EQ(selectors & 7u, texcoord);
    }
    material->set_property<float>("thickness_factor", 1.f);
    material->set_property<float2>("clearcoat_normal_texture_offset", float2(0.75f, 0.f));
    scene->update();
    REQUIRE_GE(material->data().size(), 2u * stride + 24u);
    CHECK_EQ(material->effective_masks().transforms, shared::gltfmat::TextureBits::CLEARCOAT_NORMAL);
    float offset;
    std::memcpy(&offset, material->data().data() + 2 * stride + 2 * sizeof(float), sizeof(float));
    CHECK_EQ(offset, 0.75f);
    material->set_property<float16_t>("clearcoat_normal_texture_scale", float16_t(0.5f));
    scene->update();
    std::memcpy(&scale, material->data().data() + stride + 28, sizeof(scale));
    CHECK_EQ(float(scale), doctest::Approx(0.5f));
    material->set_property<float16_t>("clearcoat_factor", float16_t(0.f));
    scene->update();
    CHECK_EQ(material->effective_masks().features & shared::gltfmat::FeatureBits::CLEARCOAT, 0u);
    CHECK_EQ(material->effective_masks().transforms, 0u);
    REQUIRE_GE(material->data().size(), stride); // The volume slot remains active.
    material->set_property<float>("thickness_factor", 0.f);
    material->set_property<float2>("clearcoat_normal_texture_offset", float2(0.f));
    scene->update();
    CHECK(material->data().empty());
}

TEST_CASE_GPU("gltf_sheen_iridescence_data_lifecycle")
{
    auto scene = Scene::create(ref(ctx.device));
    auto material = ref(scene->create_material<GLTFDataTestMaterial>());
    scene->update();
    CHECK(material->data().empty());
    auto layout = scene->render_module()->layout();
    const std::string material_type_name = fmt::format("StructuredBuffer<{}>", material->slang_type_name());
    auto material_type = layout->find_type_by_name(material_type_name.c_str());
    REQUIRE(material_type);
    CHECK_LT(layout->get_type_layout(material_type)->element_type_layout()->stride(), sizeof(shared::MaterialData));
    for (const char* name : {"StructuredBuffer<GLTFSheenRecord>", "StructuredBuffer<GLTFIridescenceRecord>"}) {
        auto type = layout->find_type_by_name(name);
        REQUIRE(type);
        CHECK_EQ(layout->get_type_layout(type)->element_type_layout()->stride(), 32u);
    }
    material->set_property<float3>("sheen_color_factor", float3(0.5f));
    material->set_property<float>("iridescence_factor", 0.75f);
    scene->update();
    REQUIRE_EQ(material->data().size(), 64u);
    CHECK_EQ(
        material->effective_masks().features
            & (shared::gltfmat::FeatureBits::SHEEN | shared::gltfmat::FeatureBits::IRIDESCENCE),
        shared::gltfmat::FeatureBits::SHEEN | shared::gltfmat::FeatureBits::IRIDESCENCE
    );
    float value;
    std::memcpy(&value, material->data().data() + 32 + 16, sizeof(float));
    CHECK_EQ(value, 0.75f);
    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("iridescence_texture", texture);
    material->set_property<uint32_t>("iridescence_texture_texcoord", 3u);
    scene->update();
    CHECK_EQ(material->effective_masks().textures, shared::gltfmat::TextureBits::IRIDESCENCE);
    CHECK_EQ(material->effective_masks().nonzero_texcoords, shared::gltfmat::TextureBits::IRIDESCENCE);
    uint32_t selectors;
    std::memcpy(&selectors, material->data().data() + 32 + 4, sizeof(selectors));
    CHECK_EQ(selectors, 3u);
    const std::vector<uint8_t> packed(material->data().begin(), material->data().end());
    for (int i = 0; i < 100; ++i) {
        auto neighbor = scene->create_material<GLTFDataTestMaterial>();
        neighbor->set_property<float3>("sheen_color_factor", float3(1.f));
    }
    scene->update();
    CHECK(std::equal(packed.begin(), packed.end(), material->data().begin()));
    material->set_property<float3>("sheen_color_factor", float3(0.f));
    scene->update();
    CHECK_EQ(material->effective_masks().features & shared::gltfmat::FeatureBits::SHEEN, 0u);
    std::memcpy(&value, material->data().data() + 16, sizeof(float));
    CHECK_EQ(value, 0.75f);
    material->set_property<float>("iridescence_factor", 0.f);
    scene->update();
    CHECK(material->data().empty());
}

TEST_CASE_GPU("gltf_normal_and_emission_feature_gating")
{
    auto scene = Scene::create(ref(ctx.device));
    auto material = scene->create_material<GLTFDataTestMaterial>();
    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("normal_texture", texture);
    material->set_property<float2>("normal_texture_offset", float2(0.25f, 0.f));
    material->set_property("emissive_texture", texture);
    material->set_property<float2>("emissive_texture_offset", float2(0.5f, 0.f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::NORMAL,
        shared::gltfmat::FeatureBits::NORMAL
    );
    CHECK_EQ(material->effective_masks().features & shared::gltfmat::FeatureBits::EMISSION, 0u);
    CHECK_EQ(material->effective_masks().textures, shared::gltfmat::TextureBits::NORMAL);
    CHECK_EQ(material->effective_masks().transforms, shared::gltfmat::TextureBits::NORMAL);
    REQUIRE_EQ(material->data().size(), 32u);

    // A bound normal texture remains active at zero scale to preserve the
    // normal-map decode behavior, while zero emission removes emission work.
    material->set_property<float16_t>("normal_texture_scale", float16_t(0.f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::NORMAL,
        shared::gltfmat::FeatureBits::NORMAL
    );
    CHECK_EQ(material->effective_masks().textures, shared::gltfmat::TextureBits::NORMAL);
    CHECK_EQ(material->effective_masks().transforms, shared::gltfmat::TextureBits::NORMAL);
    CHECK_EQ(material->data().size(), 32u);
    CHECK(sgl::math::all(material->get_property<float2>("normal_texture_offset") == float2(0.25f, 0.f)));

    material->set_property<float3>("emissive_factor", float3(1.f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::EMISSION,
        shared::gltfmat::FeatureBits::EMISSION
    );
    CHECK_EQ(
        material->effective_masks().textures,
        shared::gltfmat::TextureBits::NORMAL | shared::gltfmat::TextureBits::EMISSIVE
    );
    CHECK_EQ(
        material->effective_masks().transforms,
        shared::gltfmat::TextureBits::NORMAL | shared::gltfmat::TextureBits::EMISSIVE
    );
    material->set_property<float16_t>("normal_texture_scale", float16_t(1.f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().transforms,
        shared::gltfmat::TextureBits::NORMAL | shared::gltfmat::TextureBits::EMISSIVE
    );
}

TEST_CASE_GPU("gltf_scene_feature_cap_omits_disabled_extra_data")
{
    SceneConfig config;
    config.gltf_allowed_features = GLTFFeatureFlags::all & ~GLTFFeatureFlags::normal & ~GLTFFeatureFlags::volume;
    auto scene = Scene::create(ref(ctx.device), config);
    auto* material = scene->create_material<GLTFDataTestMaterial>();
    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("normal_texture", texture);
    material->set_property<float2>("normal_texture_offset", float2(0.25f, 0.f));
    material->set_property<float3>("emissive_factor", float3(0.5f));
    material->set_property("emissive_texture", texture);
    material->set_property<float2>("emissive_texture_offset", float2(0.5f, 0.f));
    material->set_property<uint32_t>("feature_mask_filter", F::ALL_MR & ~F::EMISSION);
    scene->update();

    CHECK_EQ(material->effective_masks().features & (F::NORMAL | F::EMISSION), 0u);
    CHECK_EQ(material->effective_masks().textures, 0u);
    CHECK_EQ(material->effective_masks().transforms, 0u);
    CHECK(material->data().empty());

    material->set_property<float16_t>("transmission_factor", float16_t(1.f));
    material->set_property<float>("thickness_factor", 1.f);
    scene->update();
    CHECK_EQ(material->effective_masks().features & (F::TRANSMISSION | F::VOLUME), F::TRANSMISSION);
    CHECK_EQ(material->data().size(), 32u); // Transmission record only; volume is disabled by the scene cap.
    CHECK(!scene->required_capabilities().has(SceneCapabilityFlags::solid_transmission));
}

TEST_CASE_GPU("gltf_sparse_extension_rank_gpu_readback")
{
    auto module = ctx.device->load_module_from_source("gltf_sparse_extension_rank_probe", R"(
import falcor2.render;
import falcor2.utils;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint feature_mask, uint transform_mask, BufferOffsetPair extra_data,
          RWStructuredBuffer<float4> result, uint index)
{
    result[index] = float4(
        extra_data.load<float>(gltf_material::feature_record_offset(
            feature_mask, gltfmat::FeatureBits::SHEEN) + 16u),
        extra_data.load<float>(gltf_material::feature_record_offset(
            feature_mask, gltfmat::FeatureBits::IRIDESCENCE) + 16u),
        extra_data.load<float>(gltf_material::transform_record_offset(
            feature_mask, transform_mask, gltfmat::TextureBits::BASE_COLOR) + 8u),
        extra_data.load<float>(gltf_material::transform_record_offset(
            feature_mask, transform_mask, gltfmat::TextureBits::IRIDESCENCE) + 8u)
    );
}
)");
    REQUIRE(module);

    auto scene = Scene::create(ref(ctx.device));
    auto material = scene->create_material<GLTFDataTestMaterial>();
    material->set_property<float16_t>("specular_factor", float16_t(0.5f));
    material->set_property<float16_t>("transmission_factor", float16_t(0.5f));
    material->set_property<float16_t>("ior", float16_t(2.f));
    material->set_property<float>("thickness_factor", 1.f);
    material->set_property<float16_t>("clearcoat_factor", float16_t(0.5f));
    material->set_property<float3>("sheen_color_factor", float3(0.25f));
    material->set_property<float>("iridescence_factor", 0.5f);
    material->set_property<float>("anisotropy_strength", 0.5f);
    material->set_property<float>("diffuse_transmission_factor", 0.5f);
    scene->update();
    REQUIRE_EQ(
        material->effective_masks().features & shared::gltfmat::FeatureBits::EXTENSION_BITS,
        shared::gltfmat::FeatureBits::EXTENSION_BITS
    );
    REQUIRE_EQ(material->data().size(), 9u * 32u);
    CHECK_EQ(material->effective_masks().transforms, 0u);

    const ref<sgl::Texture> texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });
    material->set_property("base_color_texture", texture);
    material->set_property("iridescence_texture", texture);
    material->set_property<float2>("base_color_texture_offset", float2(0.125f, 0.f));
    material->set_property<float2>("iridescence_texture_offset", float2(0.75f, 0.f));
    scene->update();
    REQUIRE_EQ(
        material->effective_masks().transforms,
        shared::gltfmat::TextureBits::BASE_COLOR | shared::gltfmat::TextureBits::IRIDESCENCE
    );
    REQUIRE_GE(material->data().size(), 9u * 32u + 2u * 24u);

    const auto requirements = scene->requirements();
    std::vector<ref<sgl::SlangModule>> modules(requirements->modules.begin(), requirements->modules.end());
    modules.push_back(module);
    auto composed = ctx.device->compose_modules(
        "gltf_sparse_extension_rank_probe_composed",
        modules,
        requirements->type_conformances
    );
    REQUIRE(composed);
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program({composed}, {composed->entry_point("main")})}
    );
    REQUIRE(kernel);
    auto result = ctx.device->create_buffer({.size = 2 * sizeof(float4), .usage = sgl::BufferUsage::unordered_access});
    auto read = [&](uint32_t index)
    {
        const auto masks = material->effective_masks();
        kernel->dispatch(
            uint3(1),
            [&](sgl::ShaderCursor cursor)
            {
                scene->bind(cursor, SceneBindFlags::materials);
                cursor = cursor.find_entry_point(0);
                cursor["feature_mask"] = masks.features;
                cursor["transform_mask"] = masks.transforms;
                cursor["extra_data"] = material->data_ref();
                cursor["result"] = result;
                cursor["index"] = index;
            }
        );
    };
    read(0);

    // A SpecGloss neighbor shares the pool and forces the mixed-workflow path.
    auto neighbor = scene->create_material<GLTFSpecGlossDataTestMaterial>();
    neighbor->set_property("diffuse_texture", texture);
    neighbor->set_property<float2>("diffuse_texture_offset", float2(0.5f, 0.f));
    scene->update();
    CHECK_EQ(material->data_ref().buffer.data(), neighbor->data_ref().buffer.data());

    // Removing lower feature bits moves later records, while both transform
    // records move with the shorter feature prefix. The same GPU kernel reads
    // the new addresses using the current runtime masks and allocation.
    material->set_property<float16_t>("specular_factor", float16_t(1.f));
    material->set_property<float16_t>("ior", float16_t(1.5f));
    material->set_property<float16_t>("clearcoat_factor", float16_t(0.f));
    scene->update();
    CHECK_EQ(
        material->effective_masks().features
            & (shared::gltfmat::FeatureBits::SPECULAR | shared::gltfmat::FeatureBits::IOR
               | shared::gltfmat::FeatureBits::CLEARCOAT),
        0u
    );
    REQUIRE_GE(material->data().size(), 6u * 32u + 2u * 24u);
    CHECK(
        std::all_of(
            material->data().begin() + 6u * 32u + 2u * 24u,
            material->data().end(),
            [](uint8_t value)
            {
                return value == 0;
            }
        )
    );
    read(1);
    const auto values = result->get_elements<float4>();
    REQUIRE_EQ(values.size(), 2u);
    CHECK_EQ(values[0], float4(0.25f, 0.5f, 0.125f, 0.75f));
    CHECK_EQ(values[1], values[0]);
}

TEST_CASE_GPU("material_data_lifecycle")
{
    auto scene = Scene::create(ref(ctx.device));
    auto first = ref(scene->create_material<DataTestMaterial>());
    auto* second = scene->create_material<DataTestMaterial>();
    first->set_value(13, 11);
    second->set_value(29, 22);
    scene->update();
    CHECK_EQ(first->data_ref().offset % 16, 0);
    CHECK_EQ(second->data_ref().offset % 16, 0);
    CHECK_NE(first->data_ref().offset, second->data_ref().offset);
    CHECK_EQ(first->data_ref().buffer.data(), second->data_ref().buffer.data());

    auto module = ctx.device->load_module_from_source("material_data_lifecycle", R"(
import falcor2.utils;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(BufferOffsetPair first, BufferOffsetPair second, RWStructuredBuffer<uint> result, uint index)
{
    result[index] = first.load<uint>(0);
    result[index + 1] = second.load_aligned<uint, 4>(0);
}
)");
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program({module}, {module->entry_point("main")})}
    );
    auto result
        = ctx.device->create_buffer({.size = 8 * sizeof(uint32_t), .usage = sgl::BufferUsage::unordered_access});
    auto read = [&](DataTestMaterial* a, DataTestMaterial* b, uint32_t index)
    {
        kernel->dispatch(
            uint3(1),
            [&](sgl::ShaderCursor cursor)
            {
                cursor = cursor.find_entry_point(0);
                cursor["first"] = a->data_ref();
                cursor["second"] = b->data_ref();
                cursor["result"] = result;
                cursor["index"] = index;
            }
        );
    };
    read(first, second, 0);
    first->ensure_data_size(4096);
    uint32_t preserved_value;
    std::memcpy(&preserved_value, first->data().data(), sizeof(preserved_value));
    CHECK_EQ(preserved_value, 11);
    auto grown_bytes = first->data().subspan(sizeof(uint32_t));
    CHECK(
        std::all_of(
            grown_bytes.begin(),
            grown_bytes.end(),
            [](uint8_t value)
            {
                return value == 0;
            }
        )
    );
    first->set_value(4096, 33);
    scene->update();
    CHECK_EQ(first->data_ref().buffer.data(), second->data_ref().buffer.data());
    read(first, second, 2);

    const auto freed_offset = first->data_ref().offset;
    first->remove();
    scene->update();
    CHECK_EQ(first->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
    auto* third = scene->create_material<DataTestMaterial>();
    third->ensure_data_size(4096);
    auto reused_bytes = third->data();
    CHECK(
        std::all_of(
            reused_bytes.begin(),
            reused_bytes.end(),
            [](uint8_t value)
            {
                return value == 0;
            }
        )
    );
    third->set_value(4096, 44);
    scene->update();
    CHECK_EQ(third->data_ref().offset, freed_offset);
    read(third, second, 4);

    third->ensure_data_size(0);
    auto transient = ref(scene->create_material<DataTestMaterial>());
    transient->set_value(16, 55);
    transient->remove();
    scene->update();
    CHECK_EQ(third->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());
    CHECK_EQ(transient->material_id(), shared::MaterialID::invalid);
    read(second, second, 6);
    // Read back only after submitted work has crossed growth, removal and reuse.
    const std::vector<uint32_t> expected{11, 22, 33, 22, 44, 22, 22, 22};
    CHECK(result->get_elements<uint32_t>() == expected);
}

TEST_CASE_GPU("material_data_reference_and_dynamic_dispatch")
{
    auto module = ctx.device->load_module_from_source("material_data_scene_binding_probe", R"(
import falcor2.render;
import falcor2.utils;

public struct DataPoolProbeMaterial : IMaterial
{
    typedef InvalidMaterialInstance MaterialInstance;
    public MaterialHeader header;
    public BufferOffsetPair extra_data;
    public uint has_data;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        InvalidMaterial fallback = {};
        fallback.header = header;
        return fallback.setup_material_instance(si, lod_sampler, hints);
    }

    public override float3 eval_emission<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (has_data == 0u || !extra_data.buffer.is_valid())
            return float3(0.0);
        return float3(
            float(extra_data.load<uint>(0u)),
            1.0,
            float(extra_data.offset)
        );
    }
};

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint material_id, RWStructuredBuffer<float4> result, uint index)
{
    SurfaceInteraction si = {};
    let material = g_scene.get_material(MaterialID(material_id));
    const float3 value = material.eval_emission(si, ExplicitLodSampler(0.0));
    result[index] = float4(value, value.y);
}
)");
    REQUIRE(module);

    auto scene = Scene::create(ref(ctx.device));
    auto* empty = scene->create_material<DataTestMaterial>();
    empty->set_probe_module(module);
    scene->update();
    CHECK_EQ(empty->data_ref().buffer.data(), std::numeric_limits<uint64_t>::max());

    const auto requirements = scene->requirements();
    auto composed = ctx.device->compose_modules(
        "material_data_scene_binding_probe_composed",
        requirements->modules,
        requirements->type_conformances
    );
    REQUIRE(composed);
    auto kernel = ctx.device->create_compute_kernel(
        {.program = ctx.device->link_program({composed}, {composed->entry_point("main")})}
    );
    REQUIRE(kernel);
    auto result = ctx.device->create_buffer({.size = 5 * sizeof(float4), .usage = sgl::BufferUsage::unordered_access});

    auto read = [&](DataTestMaterial* material, uint32_t index)
    {
        kernel->dispatch(
            uint3(1),
            [&](sgl::ShaderCursor cursor)
            {
                scene->bind(cursor, SceneBindFlags::materials);
                cursor = cursor.find_entry_point(0);
                cursor["material_id"] = uint32_t(material->material_id());
                cursor["result"] = result;
                cursor["index"] = index;
            }
        );
    };
    read(empty, 0);

    auto* first = scene->create_material<DataTestMaterial>();
    auto* second = scene->create_material<DataTestMaterial>();
    first->set_probe_module(module);
    second->set_probe_module(module);
    first->set_value(13, 11);
    second->set_value(29, 22);
    scene->update();
    const uint32_t first_offset = first->data_ref().offset;
    const uint32_t second_offset = second->data_ref().offset;
    CHECK_NE(first_offset, second_offset);
    CHECK_EQ(first->data_ref().buffer.data(), second->data_ref().buffer.data());
    read(first, 1);
    read(second, 2);

    // This exceeds the initial GPU buffer size, so each material record must receive the replacement handle.
    const uint64_t initial_handle = first->data_ref().buffer.data();
    first->set_value(4096, 33);
    scene->update();
    const uint32_t grown_offset = first->data_ref().offset;
    CHECK_NE(first->data_ref().buffer.data(), initial_handle);
    CHECK_NE(grown_offset, second_offset);
    CHECK_EQ(first->data_ref().buffer.data(), second->data_ref().buffer.data());
    read(first, 3);
    read(second, 4);

    const auto values = result->get_elements<float4>();
    REQUIRE_EQ(values.size(), 5);
    CHECK_EQ(values[0], float4(0.f, 0.f, 0.f, 0.f));
    CHECK_EQ(values[1], float4(11.f, 1.f, float(first_offset), 1.f));
    CHECK_EQ(values[2], float4(22.f, 1.f, float(second_offset), 1.f));
    CHECK_EQ(values[3], float4(33.f, 1.f, float(grown_offset), 1.f));
    CHECK_EQ(values[4], float4(22.f, 1.f, float(second_offset), 1.f));
}

TEST_CASE("material header metadata codec")
{
    CHECK_EQ(sizeof(shared::MaterialHeader), 16);
    CHECK_EQ(sizeof(shared::MaterialData), 128);

    shared::MaterialHeader header = {};
    header.material_data = shared::detail::pack_material_data(
        shared::MaterialFlags::two_sided | shared::MaterialFlags::thin_walled,
        255
    );
    CHECK_EQ(
        shared::detail::get_material_flags(header),
        shared::MaterialFlags::two_sided | shared::MaterialFlags::thin_walled
    );
    CHECK_EQ(shared::detail::get_nested_priority(header), 255);
    CHECK_EQ(header.material_data & 0xffff0000u, 0u);
}

TEST_CASE_GPU_NIGHTLY("material nested priority properties")
{
    // This checks API-independent host behavior, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    auto scene = Scene::create(ref(ctx.device));

    StandardMaterial* standard = scene->create_material<StandardMaterial>();
    CHECK_EQ(standard->nested_priority(), 0);
    standard->set_property<uint32_t>("nested_priority", 17);
    CHECK_EQ(standard->nested_priority(), 17);

    StandardSpecGlossMaterial* spec_gloss = scene->create_material<StandardSpecGlossMaterial>();
    spec_gloss->set_property<uint32_t>("nested_priority", 23);
    CHECK_EQ(spec_gloss->nested_priority(), 23);

    MaterialXMaterial* materialx = scene->create_material<MaterialXMaterial>();
    materialx->set_property<uint32_t>("nested_priority", 31);
    CHECK_EQ(materialx->nested_priority(), 31);
    CHECK_EQ(materialx->flags(), shared::MaterialFlags::two_sided);
}

namespace {

const char* LAYERING_TRAVERSAL_ORDER_PROPERTY = "materialx_layering_traversal_order";

class RequiredModuleTestMaterial final : public Material {
    FALCOR_SCENE_OBJECT(RequiredModuleTestMaterial, Material)
public:
    RequiredModuleTestMaterial() { set_slang_type_name("RequiredModuleTestMaterialData"); }

    void set_required_module(ref<sgl::SlangModule> module) { m_required_module = std::move(module); }
    ref<sgl::SlangModule> required_module() const override { return m_required_module; }

private:
    ref<sgl::SlangModule> m_required_module;
};

std::string upgrade_test_document_version(std::string source)
{
    const std::string old_version = "version=\"1.38\"";
    const std::string new_version = "version=\"1.39\"";
    size_t pos = 0;
    while ((pos = source.find(old_version, pos)) != std::string::npos) {
        source.replace(pos, old_version.size(), new_version);
        pos += new_version.size();
    }
    return source;
}

mtlx::CodeGenDesc make_materialx_codegen_desc(const std::string& shader)
{
    mtlx::CodeGenDesc desc;
    desc.document = upgrade_test_document_version(shader);
    desc.node_name = "Tiled_Brass";
    desc.positionfree_layering = false;
    desc.transmissive_bsdfs.clear();
    desc.make_editable = false;
    return desc;
}

std::vector<std::string> layering_traversal_order(const mtlx::CodeGenResult& result)
{
    return result.codegen_metadata.get_list<std::string>(LAYERING_TRAVERSAL_ORDER_PROPERTY);
}

std::string join_lines(const std::vector<std::string>& lines)
{
    std::string result;
    for (const std::string& line : lines) {
        result += line;
        result += "\n";
    }
    return result;
}

bool starts_with(std::string_view value, std::string_view prefix)
{
    return value.size() >= prefix.size() && value.substr(0, prefix.size()) == prefix;
}

bool contains(const std::vector<std::string>& values, const std::string& value)
{
    return std::find(values.begin(), values.end(), value) != values.end();
}

bool contains(std::string_view source, std::string_view needle)
{
    return source.find(needle) != std::string_view::npos;
}

std::string read_text_file(const std::filesystem::path& path)
{
    std::ifstream file(path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

struct PreviewOutputCase {
    std::string type;
    std::string value;
};

std::vector<PreviewOutputCase> preview_output_cases()
{
    return {
        {"float", "0.5"},
        {"vector2", "0.2, 0.4"},
        {"vector3", "0.2, 0.4, 0.8"},
        {"vector4", "0.2, 0.4, 0.8, 0.5"},
        {"color3", "0.2, 0.4, 0.8"},
        {"color4", "0.2, 0.4, 0.8, 0.5"},
    };
}

std::string preview_output_node_graph_name(const PreviewOutputCase& preview_case)
{
    return "NG_" + preview_case.type;
}

std::string preview_output_name(const PreviewOutputCase& preview_case)
{
    return preview_output_node_graph_name(preview_case) + "/out";
}

std::string preview_output_node_graph(const PreviewOutputCase& preview_case)
{
    const std::string graph_name = preview_output_node_graph_name(preview_case);
    return "  <nodegraph name=\"" + graph_name + "\">\n" + "    <constant name=\"value\" type=\"" + preview_case.type
        + "\">\n" + "      <input name=\"value\" type=\"" + preview_case.type + "\" value=\"" + preview_case.value
        + "\" />\n" + "    </constant>\n" + "    <output name=\"out\" type=\"" + preview_case.type
        + "\" nodename=\"value\" />\n" + "  </nodegraph>\n";
}

std::string preview_output_shader(const PreviewOutputCase& preview_case)
{
    return "<?xml version=\"1.0\"?>\n"
           "<materialx version=\"1.38\" colorspace=\"lin_rec709\">\n"
        + preview_output_node_graph(preview_case) + "</materialx>\n";
}

std::string transformmatrix_vector4_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_transformmatrix_vector4">
    <transformmatrix name="transform1" type="vector4">
      <input name="in" type="vector4" value="0.2, 0.4, 0.8, 1.0" />
      <input name="mat" type="matrix44" value="1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0" />
    </transformmatrix>
    <output name="out" type="vector4" nodename="transform1" />
  </nodegraph>
</materialx>
)";
}

std::string image_vector2_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_image_vector2">
    <image name="image1" type="vector2">
      <input name="file" type="filename" value="resources/Images/grid.png" />
      <input name="default" type="vector2" value="0.25, 0.75" />
    </image>
    <output name="out" type="vector2" nodename="image1" />
  </nodegraph>
</materialx>
)";
}

std::string color4_to_float_channel_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_color4_to_float_g">
    <constant name="value" type="color4">
      <input name="value" type="color4" value="0.0, 0.5, 0.75, 1.0" />
    </constant>
    <output name="out" type="float" nodename="value" channels="g" />
  </nodegraph>
</materialx>
)";
}

std::string vector2_to_vector3_channel_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_vector2_to_vector3_xy0">
    <constant name="value" type="vector2">
      <input name="value" type="vector2" value="0.25, 0.75" />
    </constant>
    <output name="out" type="vector3" nodename="value" channels="xy0" />
  </nodegraph>
</materialx>
)";
}

std::string difference_color3_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_difference_color3">
    <difference name="difference1" type="color3">
      <input name="fg" type="color3" value="0.9, 0.2, 0.1" />
      <input name="bg" type="color3" value="0.1, 0.4, 0.8" />
      <input name="mix" type="float" value="0.5" />
    </difference>
    <output name="out" type="color3" nodename="difference1" />
  </nodegraph>
</materialx>
)";
}

std::string blur_vector3_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_blur_vector3">
    <blur name="blur1" type="vector3">
      <input name="in" type="vector3" value="0.2, 0.4, 0.8" />
      <input name="size" type="float" value="0.1" />
    </blur>
    <output name="out" type="vector3" nodename="blur1" />
  </nodegraph>
</materialx>
)";
}

std::string worleynoise3d_vector2_preview_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_worleynoise3d_vector2">
    <worleynoise3d name="worley1" type="vector2">
      <input name="position" type="vector3" value="0.2, 0.4, 0.8" />
      <input name="jitter" type="float" value="0.5" />
    </worleynoise3d>
    <output name="out" type="vector2" nodename="worley1" />
  </nodegraph>
</materialx>
)";
}

std::string mix_edf_surface_shader()
{
    return R"(<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodegraph name="NG_mix_edf_surface">
    <uniform_edf name="edf1" type="EDF">
      <input name="color" type="color3" value="0.1, 0.8, 0.2" />
    </uniform_edf>
    <uniform_edf name="edf2" type="EDF">
      <input name="color" type="color3" value="0.8, 0.1, 0.4" />
    </uniform_edf>
    <mix name="mix1" type="EDF">
      <input name="fg" type="EDF" nodename="edf1" />
      <input name="bg" type="EDF" nodename="edf2" />
      <input name="mix" type="float" value="0.5" />
    </mix>
    <surface name="surface1" type="surfaceshader">
      <input name="edf" type="EDF" nodename="mix1" />
    </surface>
    <output name="test_surface_result" type="surfaceshader" nodename="surface1" />
  </nodegraph>
</materialx>
)";
}

std::string asymmetric_bsdf_mix_shader(std::string_view weight_type, bool keep_foreground)
{
    const bool color_weight = weight_type == "color3";
    const std::string mix_nodedef = color_weight ? R"( nodedef="ND_mix_bsdfC")" : "";
    const std::string foreground = keep_foreground ? "live" : "empty";
    const std::string background = keep_foreground ? "empty" : "live";
    return fmt::format(
        R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="live" type="BSDF" />
  <oren_nayar_diffuse_bsdf name="empty" type="BSDF">
    <input name="weight" type="float" value="0.0" />
  </oren_nayar_diffuse_bsdf>
  <mix name="asymmetric_mix" type="BSDF"{1}>
    <input name="fg" type="BSDF" nodename="{2}" />
    <input name="bg" type="BSDF" nodename="{3}" />
    <input name="mix" type="{0}" value="{4}" />
  </mix>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="asymmetric_mix" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
        weight_type,
        mix_nodedef,
        foreground,
        background,
        color_weight ? "0.25, 0.5, 0.75" : "0.25"
    );
}

std::string empty_surface_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <uniform_edf name="emission" type="EDF">
    <input name="color" type="color3" value="0.1, 0.8, 0.2" />
  </uniform_edf>
  <surface name="surface" type="surfaceshader">
    <input name="edf" type="EDF" nodename="emission" />
    <input name="opacity" type="float" value="0.25" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string empty_combiner_child_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <multiply name="empty" type="BSDF">
    <input name="in2" type="float" value="0.5" />
  </multiply>
  <oren_nayar_diffuse_bsdf name="live" type="BSDF" />
  <add name="add" type="BSDF">
    <input name="in1" type="BSDF" nodename="empty" />
    <input name="in2" type="BSDF" nodename="live" />
  </add>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="add" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string mtlx_layered_property_update_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="base" type="BSDF">
    <input name="color" type="color3" value="0.1, 0.2, 0.3" />
  </oren_nayar_diffuse_bsdf>
  <oren_nayar_diffuse_bsdf name="top" type="BSDF">
    <input name="color" type="color3" value="0.7, 0.6, 0.5" />
  </oren_nayar_diffuse_bsdf>
  <layer name="layer1" type="BSDF">
    <input name="top" type="BSDF" nodename="top" />
    <input name="base" type="BSDF" nodename="base" />
  </layer>
  <surface name="surface1" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="layer1" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>)";
}

std::string rootless_srgb_material_shader()
{
    return R"(<materialx version="1.39">
  <image name="albedo" type="color3">
    <input name="file" type="filename" value="albedo.png" colorspace="srgb_texture" />
  </image>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
    <input name="color" type="color3" nodename="albedo" />
  </oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string unconsumed_bsdf_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="root_bsdf" type="BSDF">
    <input name="color" type="color3" value="0.2, 0.4, 0.8" />
  </oren_nayar_diffuse_bsdf>
</materialx>)";
}

std::string contextual_root_weight_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_weighted_root" node="weighted_root" nodegroup="pbr">
    <input name="in" type="BSDF" />
    <input name="weight" type="color3" value="1, 1, 1" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="NG_weighted_root" nodedef="ND_weighted_root">
    <multiply name="multiply" type="BSDF" nodedef="ND_multiply_bsdfC">
      <input name="in1" type="BSDF" interfacename="in" />
      <input name="in2" type="color3" interfacename="weight" />
    </multiply>
    <output name="out" type="BSDF" nodename="multiply" />
  </nodegraph>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <weighted_root name="weighted" type="BSDF">
    <input name="in" type="BSDF" nodename="diffuse" />
    <input name="weight" type="color3" value="0.2, 0.4, 0.8" />
  </weighted_root>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="weighted" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string weighted_layer_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="shared_top" type="BSDF" />
  <oren_nayar_diffuse_bsdf name="base" type="BSDF" />
  <weighted_layer name="covered" type="BSDF">
    <input name="top" type="BSDF" nodename="shared_top" />
    <input name="top_weight" type="color3" value="0.25, 0.5, 0.75" />
    <input name="base" type="BSDF" nodename="base" />
  </weighted_layer>
  <mix name="root" type="BSDF">
    <input name="fg" type="BSDF" nodename="covered" />
    <input name="bg" type="BSDF" nodename="base" />
    <input name="mix" type="float" value="0.5" />
  </mix>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="root" /></surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="surface" /></surfacematerial>
</materialx>)";
}

std::string custom_material_implementation_shader(const std::string& gain = {})
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_test_material" node="test_material">
    <input name="gain" type="float" value="0.2" />
    <output name="out" type="material" />
  </nodedef>
  <nodegraph name="NG_test_material" nodedef="ND_test_material">
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
      <input name="weight" type="float" interfacename="gain" />
      <input name="color" type="color3" value="0.2, 0.4, 0.8" />
    </oren_nayar_diffuse_bsdf>
    <surface name="surface" type="surfaceshader">
      <input name="bsdf" type="BSDF" nodename="diffuse" />
    </surface>
    <surfacematerial name="material" type="material">
      <input name="surfaceshader" type="surfaceshader" nodename="surface" />
    </surfacematerial>
    <output name="out" type="material" nodename="material" />
  </nodegraph>
  <constant name="override" type="float"><input name="value" type="float" value="0.8" /></constant>
  <test_material name="root_material" type="material">)"
        + gain + R"(</test_material>
</materialx>)";
}

std::string provider_geomprop_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <geompropvalue name="custom_stream" type="vector3">
    <input name="geomprop" type="string" value="Pref" />
    <input name="default" type="vector3" value="0.9, 0.8, 0.7" />
  </geompropvalue>
  <texcoord name="uv1" type="vector2"><input name="index" type="integer" value="1" /></texcoord>
  <convert name="uv3" type="vector3"><input name="in" type="vector2" nodename="uv1" /></convert>
  <add name="sum" type="vector3">
    <input name="in1" type="vector3" nodename="custom_stream" />
    <input name="in2" type="vector3" nodename="uv3" />
  </add>
  <convert name="tint" type="color3"><input name="in" type="vector3" nodename="sum" /></convert>
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" nodename="tint" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::string geometry_route_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <geompropdef name="uv_alias" type="vector2" geomprop="texcoord" index="1" />
  <geompropdef name="color_alias" type="color3" geomprop="geomcolor" index="0" />
  <geompropdef name="view_alias" type="vector3" geomprop="viewdirection" />
  <texcoord name="uv_direct" type="vector2"><input name="index" type="integer" value="1" /></texcoord>
  <geompropvalue name="uv_defined" type="vector2"><input name="geomprop" type="string" value="uv_alias" /></geompropvalue>
  <add name="uv_sum" type="vector2">
    <input name="in1" type="vector2" nodename="uv_direct" />
    <input name="in2" type="vector2" nodename="uv_defined" />
  </add>
  <convert name="uv3" type="vector3"><input name="in" type="vector2" nodename="uv_sum" /></convert>
  <geomcolor name="color_direct" type="color3"><input name="index" type="integer" value="0" /></geomcolor>
  <geompropvalue name="color_defined" type="color3"><input name="geomprop" type="string" value="color_alias" /></geompropvalue>
  <convert name="color_direct3" type="vector3"><input name="in" type="color3" nodename="color_direct" /></convert>
  <convert name="color_defined3" type="vector3"><input name="in" type="color3" nodename="color_defined" /></convert>
  <add name="color_sum" type="vector3">
    <input name="in1" type="vector3" nodename="color_direct3" />
    <input name="in2" type="vector3" nodename="color_defined3" />
  </add>
  <viewdirection name="view_direct" type="vector3"><input name="space" type="string" value="world" /></viewdirection>
  <geompropvalue name="view_defined" type="vector3"><input name="geomprop" type="string" value="view_alias" /></geompropvalue>
  <add name="view_sum" type="vector3">
    <input name="in1" type="vector3" nodename="view_direct" />
    <input name="in2" type="vector3" nodename="view_defined" />
  </add>
  <add name="uv_color" type="vector3">
    <input name="in1" type="vector3" nodename="uv3" />
    <input name="in2" type="vector3" nodename="color_sum" />
  </add>
  <add name="sum" type="vector3">
    <input name="in1" type="vector3" nodename="uv_color" />
    <input name="in2" type="vector3" nodename="view_sum" />
  </add>
  <convert name="tint" type="color3"><input name="in" type="vector3" nodename="sum" /></convert>
  <standard_surface name="surface" type="surfaceshader"><input name="base_color" type="color3" nodename="tint" /></standard_surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="surface" /></surfacematerial>
</materialx>)";
}

std::string missing_geomcolor_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <geomcolor name="missing_color" type="color3">
    <input name="index" type="integer" value="0" />
  </geomcolor>
  <output name="out" type="color3" nodename="missing_color" />
</materialx>)";
}

std::string retroreflective_microfacet_shader()
{
    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <conductor_bsdf name="conductor" type="BSDF">
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="thinfilm_thickness" type="float" value="100.0" />
  </conductor_bsdf>
  <dielectric_bsdf name="dielectric" type="BSDF">
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="scatter_mode" type="string" value="RT" />
  </dielectric_bsdf>
  <generalized_schlick_bsdf name="generalized_schlick" type="BSDF">
    <input name="color82" type="color3" value="0.8, 0.7, 0.6" />
    <input name="roughness" type="vector2" value="0.2, 0.4" />
    <input name="retroreflective" type="boolean" value="true" />
    <input name="scatter_mode" type="string" value="RT" />
  </generalized_schlick_bsdf>
  <add name="microfacet_add" type="BSDF">
    <input name="in1" type="BSDF" nodename="conductor" />
    <input name="in2" type="BSDF" nodename="dielectric" />
  </add>
  <add name="all_bsdfs" type="BSDF">
    <input name="in1" type="BSDF" nodename="microfacet_add" />
    <input name="in2" type="BSDF" nodename="generalized_schlick" />
  </add>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="all_bsdfs" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

enum class MaterialXOpacityCase {
    opaque,
    constant,
    connected,
    mixed_out_of_range,
    luminance_custom_coefficients,
    luminance_nonuniform_color,
    luminance_overflow,
    standard_surface_white,
    standard_surface_red,
    open_pbr_constant,
    open_pbr_connected,
};

std::string materialx_opacity_shader(MaterialXOpacityCase opacity_case)
{
    if (opacity_case == MaterialXOpacityCase::standard_surface_white
        || opacity_case == MaterialXOpacityCase::standard_surface_red) {
        const char* opacity = opacity_case == MaterialXOpacityCase::standard_surface_white ? "1, 1, 1" : "1, 0, 0";
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader">
    <input name="opacity" type="color3" value="{}" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
            opacity
        );
    }

    if (opacity_case == MaterialXOpacityCase::open_pbr_constant
        || opacity_case == MaterialXOpacityCase::open_pbr_connected) {
        const bool connected = opacity_case == MaterialXOpacityCase::open_pbr_connected;
        const std::string opacity_node = connected ? R"(  <constant name="opacity_value" type="float">
    <input name="value" type="float" value="0.25" />
  </constant>
)"
                                                   : "";
        const std::string opacity_input = connected
            ? R"(    <input name="geometry_opacity" type="float" nodename="opacity_value" />
)"
            : R"(    <input name="geometry_opacity" type="float" value="0.25" />
)";
        return R"(<materialx version="1.39" colorspace="lin_rec709">
)" + opacity_node
            + R"(  <open_pbr_surface name="surface" type="surfaceshader">
)" + opacity_input
            + R"(  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    }

    if (opacity_case == MaterialXOpacityCase::mixed_out_of_range) {
        return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="background" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" value="0.2" />
  </surface>
  <surface name="foreground" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" value="2.0" />
  </surface>
  <mix name="surface_mix" type="surfaceshader">
    <input name="bg" type="surfaceshader" nodename="background" />
    <input name="fg" type="surfaceshader" nodename="foreground" />
    <input name="mix" type="float" value="0.25" />
  </mix>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface_mix" />
  </surfacematerial>
</materialx>)";
    }

    if (opacity_case == MaterialXOpacityCase::luminance_custom_coefficients
        || opacity_case == MaterialXOpacityCase::luminance_nonuniform_color
        || opacity_case == MaterialXOpacityCase::luminance_overflow) {
        const char* color = "1.0, 1.0, 1.0";
        const char* coefficients = "0.0, 0.0, 0.0";
        if (opacity_case == MaterialXOpacityCase::luminance_nonuniform_color) {
            color = "1.0, 0.5, 0.25";
            coefficients = "0.2, 0.3, 0.5";
        } else if (opacity_case == MaterialXOpacityCase::luminance_overflow) {
            color = "3.0e38, 1.0, 1.0";
            coefficients = "2.0, 0.0, 0.0";
        }
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <constant name="opacity_color" type="color3">
    <input name="value" type="color3" value="{}" />
  </constant>
  <luminance name="opacity_luminance" type="color3">
    <input name="in" type="color3" nodename="opacity_color" />
    <input name="lumacoeffs" type="color3" value="{}" />
  </luminance>
  <extract name="opacity" type="float">
    <input name="in" type="color3" nodename="opacity_luminance" />
    <input name="index" type="integer" value="0" />
  </extract>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="opacity" type="float" nodename="opacity" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
            color,
            coefficients
        );
    }

    std::string opacity_node;
    std::string opacity_input;
    if (opacity_case == MaterialXOpacityCase::constant)
        opacity_input = R"(    <input name="opacity" type="float" value="0.25" />
)";
    else if (opacity_case == MaterialXOpacityCase::connected) {
        opacity_node = R"(  <constant name="opacity_value" type="float">
    <input name="value" type="float" value="0.25" />
  </constant>
)";
        opacity_input = R"(    <input name="opacity" type="float" nodename="opacity_value" />
)";
    }

    return R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
)" + opacity_node
        + R"(  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
)" + opacity_input
        + R"(  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
}

std::vector<std::string>
normalize_layering_traversal_order(const std::vector<std::string>& order, bool include_node_events)
{
    std::vector<std::string> result;
    for (const std::string& event : order) {
        if (event == "add:Tiled_Brass")
            continue;

        if (!include_node_events && starts_with(event, "add:"))
            continue;

        result.push_back(event);
    }
    return result;
}

std::string recursive_closure_dag(size_t depth)
{
    std::string document = R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="base_0" type="BSDF">
    <input name="color" type="color3" value="0.1, 0.2, 0.3" />
  </oren_nayar_diffuse_bsdf>
)";
    std::string base = "base_0";
    for (size_t index = 1; index <= depth; ++index) {
        document += fmt::format(
            R"(  <oren_nayar_diffuse_bsdf name="top_{0}" type="BSDF">
    <input name="color" type="color3" value="0.{1}, 0.2, 0.3" />
  </oren_nayar_diffuse_bsdf>
  <layer name="layer_{0}" type="BSDF">
    <input name="top" type="BSDF" nodename="top_{0}" />
    <input name="base" type="BSDF" nodename="{2}" />
  </layer>
  <mix name="coverage_{0}" type="BSDF">
    <input name="fg" type="BSDF" nodename="layer_{0}" />
    <input name="bg" type="BSDF" nodename="{2}" />
    <input name="mix" type="float" value="0.7" />
  </mix>
)",
            index,
            index + 1,
            base
        );
        base = "coverage_" + std::to_string(index);
    }
    document += fmt::format(
        R"(  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="{}" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>
)",
        base
    );
    return document;
}

} // namespace

TEST_SUITE_BEGIN("scene_materials");

TEST_CASE("material opacity data round-trips through packed header")
{
    const auto flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
    const float threshold = 0.5f;
    const float factor = 0.002f;
    shared::MaterialHeader header = {};
    header.opacity_data = shared::detail::pack_opacity_data(flags, 3, threshold, factor);

    CHECK_EQ(shared::detail::get_opacity_flags(header), flags);
    CHECK_EQ(shared::detail::get_opacity_texture_channel(header), 3);
    CHECK_EQ(shared::detail::get_opacity_threshold(header), 128.f / 255.f);
    CHECK(std::abs(shared::detail::get_opacity_factor(header) - factor) <= 0.5f / 65535.f);
}

TEST_CASE_GPU("scene materialx representative layered material loads")
{
    // Only the Slang frontend is tested here, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    // Python covers numerical evaluation. Keep one regular native generation
    // and module-load witness, using the default backend and normal optimization.
    mtlx::CodeGenDesc desc;
    desc.document = weighted_layer_shader();
    desc.node_name = "M";
    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK_FALSE(result->module_source.empty());
    CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 2);
    CHECK_EQ(result->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::opaque);
    REQUIRE(ctx.device->load_module_from_source(result->module_name, result->module_source));
}

TEST_CASE_NIGHTLY("scene materialx classifies root opacity and emits traversal evaluation")
{
    using OpacityKind = mtlx::CodeGenResult::OpacityResult::Kind;

    struct OpacityVariant {
        MaterialXOpacityCase opacity_case;
        bool editable;
        OpacityKind expected_kind;
        float expected_constant;
    };

    const std::array opacity_variants = {
        OpacityVariant{MaterialXOpacityCase::opaque, false, OpacityKind::opaque, 1.f},
        OpacityVariant{MaterialXOpacityCase::constant, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::connected, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::mixed_out_of_range, false, OpacityKind::constant, 0.4f},
        OpacityVariant{MaterialXOpacityCase::luminance_custom_coefficients, false, OpacityKind::constant, 0.f},
        OpacityVariant{MaterialXOpacityCase::luminance_nonuniform_color, false, OpacityKind::constant, 0.475f},
        OpacityVariant{MaterialXOpacityCase::luminance_overflow, false, OpacityKind::dynamic, 1.f},
        OpacityVariant{MaterialXOpacityCase::opaque, true, OpacityKind::dynamic, 1.f},
        OpacityVariant{MaterialXOpacityCase::standard_surface_white, false, OpacityKind::opaque, 1.f},
        OpacityVariant{MaterialXOpacityCase::standard_surface_red, false, OpacityKind::constant, 0.272229f},
        OpacityVariant{MaterialXOpacityCase::open_pbr_constant, false, OpacityKind::constant, 0.25f},
        OpacityVariant{MaterialXOpacityCase::open_pbr_connected, false, OpacityKind::constant, 0.25f},
    };
    const std::array layering_modes = {
        mtlx::LayeringMode::closure_tree,
        mtlx::LayeringMode::bsdf_mix,
    };

    for (mtlx::LayeringMode layering_mode : layering_modes) {
        for (const OpacityVariant& variant : opacity_variants) {
            // Classification is shared by the backends. Keep its full semantic
            // matrix on the default, plus opaque/constant/dynamic emission on
            // every alternate backend. The specialization test below checks
            // backend-specific storage and emission details separately.
            if (layering_mode != mtlx::default_layering_mode() && variant.opacity_case != MaterialXOpacityCase::opaque
                && variant.opacity_case != MaterialXOpacityCase::constant)
                continue;
            CAPTURE(static_cast<int>(layering_mode));
            CAPTURE(static_cast<int>(variant.opacity_case));
            mtlx::CodeGenDesc desc;
            desc.document = materialx_opacity_shader(variant.opacity_case);
            desc.node_name = "M";
            desc.make_editable = variant.editable;
            if (layering_mode != mtlx::default_layering_mode())
                desc.layering_mode = layering_mode;

            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->opacity.kind, variant.expected_kind);
            CHECK_EQ(result->opacity.constant, doctest::Approx(variant.expected_constant));
            const bool evaluates_opacity = variant.expected_kind == OpacityKind::dynamic;
            CHECK(contains(result->module_source, "eval_opacity<TLodSampler"));
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_EVALUATE_OPACITY ") + (evaluates_opacity ? "1" : "0")
            ));
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_OPACITY_IS_OPAQUE ")
                    + (variant.expected_kind == OpacityKind::opaque ? "1" : "0")
            ));
            const bool conservatively_emissive = variant.opacity_case == MaterialXOpacityCase::standard_surface_white
                || variant.opacity_case == MaterialXOpacityCase::standard_surface_red
                || variant.opacity_case == MaterialXOpacityCase::open_pbr_constant
                || variant.opacity_case == MaterialXOpacityCase::open_pbr_connected;
            CHECK(contains(
                result->module_source,
                std::string("#define MATERIALX_HAS_EMISSION ") + (conservatively_emissive ? "1" : "0")
            ));
            CHECK(contains(result->module_source, "out float3 opacity_weighted_emission"));
            CHECK(contains(result->module_source, "eval_emission<TLodSampler"));
            CHECK(contains(result->module_source, "data_buffer.load<"));
            CHECK_FALSE(contains(result->module_source, "data_buffer.load_aligned<"));
            CHECK_FALSE(contains(result->module_source, "MxSimpleBTDF()"));
            CHECK_FALSE(contains(result->module_source, "synthetic_opacity_mix"));
        }
    }
}

TEST_CASE_NIGHTLY("scene materialx keeps geometry selectors out of editable promotion")
{
    struct Case {
        const char* category;
        const char* type;
        const char* inputs;
        const char* editable_input;
    };
    const std::array cases = {
        Case{"texcoord", "vector2", R"(<input name="index" type="integer" value="1" />)", ""},
        Case{"geomcolor", "color3", R"(<input name="index" type="integer" value="0" />)", ""},
        Case{"position", "vector3", R"(<input name="space" type="string" value="world" />)", ""},
        Case{"normal", "vector3", R"(<input name="space" type="string" value="world" />)", ""},
        Case{"tangent", "vector3", R"(<input name="index" type="integer" value="0" />)", ""},
        Case{"bitangent", "vector3", R"(<input name="index" type="integer" value="0" />)", ""},
        Case{
            "geompropvalue",
            "vector3",
            R"(<input name="geomprop" type="string" value="Pref" /><input name="default" type="vector3" value="0.1, 0.2, 0.3" />)",
            "default"
        },
        Case{
            "transformpoint",
            "vector3",
            R"(<input name="fromspace" type="string" value="model" /><input name="tospace" type="string" value="world" />)",
            "in"
        },
        Case{
            "transformvector",
            "vector3",
            R"(<input name="fromspace" type="string" value="object" /><input name="tospace" type="string" value="world" />)",
            "in"
        },
        Case{
            "transformnormal",
            "vector3",
            R"(<input name="fromspace" type="string" value="object" /><input name="tospace" type="string" value="world" />)",
            "in"
        },
        // An unrelated arithmetic index must still be promotable.
        Case{
            "extract",
            "float",
            R"(<input name="in" type="vector3" value="0.1, 0.2, 0.3" /><input name="index" type="integer" value="1" />)",
            "index"
        },
    };
    // Retain every node as an additional output so one generation can check all
    // selectors without pruning unused nodes or repeating document preparation.
    std::string document = R"(<materialx version="1.39">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
)";
    for (const Case& test : cases) {
        document += fmt::format(
            R"(<{0} name="read_{0}" type="{1}">{2}</{0}><output name="out_{0}" type="{1}" nodename="read_{0}" />)",
            test.category,
            test.type,
            test.inputs
        );
    }
    document += "</materialx>";
    mtlx::CodeGenDesc desc;
    desc.document = std::move(document);
    desc.node_name = "surface";
    desc.make_editable = true;
    desc.reserve_geometry_attribute_id = [](std::string_view)
    {
        return 77u;
    };
    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    for (const Case& test : cases) {
        CAPTURE(test.category);
        CHECK(contains(result->module_source, " out_" + std::string(test.category) + " = "));
        const std::string prefix = "read_" + std::string(test.category) + "_";
        bool found_editable = std::string_view(test.editable_input).empty();
        for (const mtlx::MxParamInfo& param : result->all_material_params.m_params) {
            if (!param.param_name.starts_with(prefix))
                continue;
            if (param.param_name == prefix + test.editable_input) {
                CHECK(param.is_editable);
                found_editable = true;
            } else {
                CHECK(param.param_name != prefix + "index");
                CHECK(param.param_name != prefix + "space");
                CHECK(param.param_name != prefix + "fromspace");
                CHECK(param.param_name != prefix + "tospace");
                CHECK(param.param_name != prefix + "geomprop");
            }
        }
        CHECK(found_editable);
        if (std::string_view(test.category) == "transformpoint") {
            // Check the emitted selector, not just its exclusion from editable parameters.
            CHECK(contains(result->module_source, "int(GeomPropSpace::model)"));
        }
    }
}

TEST_CASE_NIGHTLY("scene materialx selects editable parameters by generated name")
{
    auto make_desc = []
    {
        mtlx::CodeGenDesc desc;
        desc.document = materialx_opacity_shader(MaterialXOpacityCase::constant);
        desc.node_name = "M";
        return desc;
    };

    auto restricted = mtlx::CodeGen::generate(make_desc());
    REQUIRE(restricted);
    REQUIRE(restricted->all_material_params.m_params.size() >= 2);
    CHECK_EQ(restricted->all_material_params.m_editable_count, 0);

    CHECK_EQ(restricted->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);

    mtlx::CodeGenDesc unrelated_desc = make_desc();
    unrelated_desc.editable_param_names = {"diffuse_color"};
    auto unrelated = mtlx::CodeGen::generate(unrelated_desc);
    REQUIRE(unrelated);
    CHECK_EQ(unrelated->all_material_params.m_editable_count, 1);
    CHECK_EQ(unrelated->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);
    CHECK(unrelated->codegen_metadata.get<bool>("closure_pruning_effective_enabled"));

    mtlx::CodeGenDesc whitelist_desc = make_desc();
    whitelist_desc.editable_param_names = {"surface_opacity"};
    auto whitelist = mtlx::CodeGen::generate(whitelist_desc);
    REQUIRE(whitelist);
    CHECK_EQ(whitelist->all_material_params.m_editable_count, 1);
    for (const mtlx::MxParamInfo& param : whitelist->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.param_name == "surface_opacity");
    CHECK_EQ(whitelist->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::dynamic);
    CHECK(whitelist->codegen_metadata.get<bool>("closure_pruning_effective_enabled"));

    mtlx::CodeGenDesc complete_desc = make_desc();
    complete_desc.make_editable = true;
    auto complete = mtlx::CodeGen::generate(complete_desc);
    REQUIRE(complete);
    for (const mtlx::MxParamInfo& param : complete->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());
    CHECK_EQ(complete->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::dynamic);
}

TEST_CASE_NIGHTLY("scene materialx preserves authored editable inputs through rewrites")
{
    mtlx::CodeGenDesc desc;
    desc.node_name = "surface";
    std::string name;
    SUBCASE("color conversion")
    {
        name = "tint_value";
        desc.document = std::string(R"(<materialx version="1.39">
  <constant name="tint" type="color3"><input name="value" type="color3" value="0.2, 0.4, 0.6" colorspace="srgb_texture" /></constant>
  <surface_unlit name="surface" type="surfaceshader"><input name="emission_color" type="color3" nodename="tint" /></surface_unlit>
</materialx>)");
    }
    SUBCASE("unit conversion")
    {
        name = "measure_value";
        desc.document = std::string(R"(<materialx version="1.39">
  <constant name="measure" type="float"><input name="value" type="float" value="25" unit="centimeter" unittype="distance" /></constant>
  <surface_unlit name="surface" type="surfaceshader"><input name="emission" type="float" nodename="measure" /></surface_unlit>
</materialx>)");
    }
    SUBCASE("premultiplied mix preserves an initially zero leaf weight")
    {
        name = "fg_weight";
        desc.document = std::string(R"(<materialx version="1.39">
  <constant name="amount" type="float"><input name="value" type="float" value="0.5" /></constant>
  <oren_nayar_diffuse_bsdf name="fg" type="BSDF"><input name="weight" type="float" value="0" /></oren_nayar_diffuse_bsdf>
  <oren_nayar_diffuse_bsdf name="bg" type="BSDF" />
  <mix name="blend" type="BSDF">
    <input name="fg" type="BSDF" nodename="fg" />
    <input name="bg" type="BSDF" nodename="bg" />
    <input name="mix" type="float" nodename="amount" />
  </mix>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="blend" /></surface>
</materialx>)");
    }
    desc.editable_param_names = {name};
    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    REQUIRE_EQ(result->all_material_params.m_editable_count, 1);
    for (const auto& param : result->all_material_params.m_params) {
        if (!param.is_editable)
            continue;
        CHECK_EQ(param.param_name, name);
        CHECK(contains(result->module_source, param.binding_name));
        if (name == "tint_value") {
            REQUIRE(param.extract_value<float3>());
            CHECK_EQ(*param.extract_value<float3>(), float3(0.2f, 0.4f, 0.6f));
        } else if (name == "measure_value") {
            REQUIRE(param.extract_value<float>());
            CHECK_EQ(*param.extract_value<float>(), 25.f);
        }
    }
    if (name == "fg_weight" || name == "covered_top_weight")
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 2);
}

TEST_CASE_GPU_NIGHTLY("scene material system deduplicates logical modules")
{
    const std::string module_name = "required_module_dedup_test";
    const std::string module_source = R"(
import falcor2.render;

public struct RequiredModuleTestMaterialData : IMaterial
{
    typedef InvalidMaterialInstance MaterialInstance;
    public MaterialHeader header;

    public MaterialInstance setup_material_instance<LodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        InvalidMaterial material = {};
        material.header = header;
        return material.setup_material_instance(si, lod_sampler, hints);
    }
};
)";
    ref<sgl::SlangModule> first_module = ctx.device->load_module_from_source(module_name, module_source);
    ref<sgl::SlangModule> second_module = ctx.device->load_module_from_source(module_name, module_source);
    REQUIRE(first_module);
    REQUIRE(second_module);
    REQUIRE_NE(first_module.get(), second_module.get());
    REQUIRE_EQ(first_module->slang_module(), second_module->slang_module());
    REQUIRE_EQ(first_module->slang_component_type(), second_module->slang_component_type());
    REQUIRE_EQ(first_module->name(), second_module->name());

    SceneRequirements first_shader;
    first_shader.modules = {first_module};
    SceneRequirements second_shader;
    second_shader.modules = {second_module};
    CHECK(first_shader == second_shader);
    CHECK(first_shader.matches_shader_inputs(&second_shader));

    auto changed_shader = first_shader;
    changed_shader.modules[0] = nullptr;
    CHECK_FALSE(changed_shader.matches_shader_inputs(&first_shader));
    CHECK(changed_shader != first_shader);
    changed_shader = first_shader;
    changed_shader.modules.push_back(nullptr);
    CHECK_FALSE(changed_shader.matches_shader_inputs(&first_shader));
    CHECK(changed_shader != first_shader);

    auto scene = Scene::create(ref(ctx.device));
    auto* first = scene->create_material<RequiredModuleTestMaterial>();
    auto* second = scene->create_material<RequiredModuleTestMaterial>();
    first->set_required_module(first_module);
    second->set_required_module(second_module);
    scene->update();

    const auto requirements = scene->requirements();
    const size_t shared_module_count = std::count_if(
        requirements->modules.begin(),
        requirements->modules.end(),
        [&](const ref<sgl::SlangModule>& module)
        {
            return module->name() == module_name;
        }
    );
    CHECK_EQ(shared_module_count, 1);
}

TEST_CASE_GPU_NIGHTLY("scene materialx preserves independent parameter and texture state")
{
    // Texture registration requires a scene with a device. This test
    // never updates the scene, compiles a generated material, or dispatches work.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    auto scene = Scene::create(ref(ctx.device));
    const std::string shader = R"(<materialx version="1.39" colorspace="lin_rec709">
  <image name="albedo" type="color3">
    <input name="file" type="filename" value="regular_blue.png" />
  </image>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
    <input name="color" type="color3" nodename="albedo" />
    <input name="roughness" type="float" value="0.25" />
  </oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    Properties props;
    props.set("mtlx_buffer", shader);
    props.set("mtlx_node_name", "M");
    props.set("mtlx_basepath", testing::project_directory() / "data/assets/textures/test_texture_manager");
    props.set("mtlx_editable_params", "*");
    auto create = [&](const Properties& material_props)
    {
        auto* material = scene->create_material<MaterialXMaterial>();
        material->set_properties(material_props);
        return material;
    };

    Properties first_props = props;
    first_props.set("inputs:diffuse_roughness", 0.1f);
    auto* first = create(first_props);
    Properties second_props = props;
    second_props.set("inputs:diffuse_roughness", 0.8f);
    auto* second = create(second_props);
    CHECK_EQ(first->get_property<float>("inputs:diffuse_roughness"), doctest::Approx(0.1f));
    CHECK_EQ(second->get_property<float>("inputs:diffuse_roughness"), doctest::Approx(0.8f));
    first->set_property<float>("inputs:diffuse_roughness", 0.35f);
    CHECK_EQ(second->get_property<float>("inputs:diffuse_roughness"), doctest::Approx(0.8f));

    REQUIRE_EQ(first->build_texture_list().size(), 1);
    REQUIRE_EQ(second->build_texture_list().size(), 1);
    CHECK_FALSE(first->build_texture_list().front().is_valid());
    CHECK_FALSE(second->build_texture_list().front().is_valid());
    first->on_load_resources();
    CHECK(first->build_texture_list().front().is_valid());
    CHECK_FALSE(second->build_texture_list().front().is_valid());

    // A later material starts with authored values and its own unloaded texture state.
    auto* third = create(props);
    CHECK_EQ(third->get_property<float>("inputs:diffuse_roughness"), doctest::Approx(0.25f));
    CHECK_EQ(first->get_property<float>("inputs:diffuse_roughness"), doctest::Approx(0.35f));
    REQUIRE_EQ(third->build_texture_list().size(), 1);
    CHECK_FALSE(third->build_texture_list().front().is_valid());
    third->on_load_resources();
    CHECK(third->build_texture_list().front().is_valid());
    CHECK_FALSE(second->build_texture_list().front().is_valid());

    // A nodegraph interface may be published through several downstream uniforms.
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39">
  <nodegraph name="NG">
    <input name="gain" type="float" value="0.25" />
    <constant name="a" type="float"><input name="value" type="float" interfacename="gain" /></constant>
    <constant name="b" type="float"><input name="value" type="float" interfacename="gain" /></constant>
    <add name="sum" type="float">
      <input name="in1" type="float" nodename="a" /><input name="in2" type="float" nodename="b" />
    </add>
    <output name="out" type="float" nodename="sum" />
  </nodegraph>
  <surface_unlit name="M" type="surfaceshader"><input name="emission" type="float" nodegraph="NG" output="out" /></surface_unlit>
</materialx>)"));
    auto* fanout = create(props);
    CHECK_EQ(fanout->get_property<float>("inputs:NG_gain"), 0.25f);
    fanout->set_property<float>("inputs:NG_gain", 0.5f);
    CHECK_EQ(fanout->get_property<float>("inputs:NG_gain"), 0.5f);
}

TEST_CASE_NIGHTLY("scene materialx canonical naming normalizes graph identity")
{
    struct UserGraphNames {
        std::string_view node_def;
        std::string_view category;
        std::string_view graph;
        std::string_view tint_input;
        std::string_view file_input;
        std::string_view output;
        std::string_view image;
        std::string_view multiply;
        std::string_view instance;
        std::string_view surface;
        std::string_view material;
    };
    const UserGraphNames names_a{
        "ND_nested_color_a",
        "nested_color_a",
        "nested_graph_a",
        "tint_a",
        "file_a",
        "out_a",
        "base_image_a",
        "multiply_a",
        "instance_a",
        "surface_a",
        "material_a",
    };
    const UserGraphNames names_b{
        "ND_nested_color_b",
        "nested_color_b",
        "nested_graph_b",
        "tint_b",
        "file_b",
        "out_b",
        "renamed_texture_b",
        "multiply_b",
        "instance_b",
        "surface_b",
        "material_b",
    };

    auto make_document = [](const UserGraphNames& names,
                            std::string_view roughness,
                            std::string_view filename,
                            std::string_view tint,
                            bool reverse_node_declarations)
    {
        const std::string image = fmt::format(
            R"(    <image name="{}" type="color3">
      <input name="file" type="filename" interfacename="{}" />
    </image>
)",
            names.image,
            names.file_input
        );
        const std::string multiply = fmt::format(
            R"(    <multiply name="{}" type="color3">
      <input name="in1" type="color3" nodename="{}" />
      <input name="in2" type="color3" interfacename="{}" />
    </multiply>
)",
            names.multiply,
            names.image,
            names.tint_input
        );
        return fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="{}" node="{}">
    <input name="{}" type="color3" value="0.8, 0.8, 0.8" />
    <input name="{}" type="filename" value="" />
    <output name="{}" type="color3" />
  </nodedef>
  <nodegraph name="{}" nodedef="{}">
{}{}    <output name="{}" type="color3" nodename="{}" />
  </nodegraph>
  <{} name="{}" type="color3">
    <input name="{}" type="color3" value="{}" />
    <input name="{}" type="filename" value="{}" />
  </{}>
  <standard_surface name="{}" type="surfaceshader">
    <input name="base_color" type="color3" nodename="{}" />
    <input name="specular_roughness" type="float" value="{}" />
  </standard_surface>
  <surfacematerial name="{}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="{}" />
  </surfacematerial>
</materialx>)",
            names.node_def,
            names.category,
            names.tint_input,
            names.file_input,
            names.output,
            names.graph,
            names.node_def,
            reverse_node_declarations ? multiply : image,
            reverse_node_declarations ? image : multiply,
            names.output,
            names.multiply,
            names.category,
            names.instance,
            names.tint_input,
            tint,
            names.file_input,
            filename,
            names.category,
            names.surface,
            names.instance,
            roughness,
            names.material,
            names.surface
        );
    };

    auto generate_canonical = [&](const UserGraphNames& names,
                                  std::string_view roughness,
                                  std::string_view filename,
                                  std::string_view tint,
                                  bool reverse_node_declarations,
                                  bool make_editable)
    {
        mtlx::CodeGenDesc desc;
        desc.document = make_document(names, roughness, filename, tint, reverse_node_declarations);
        desc.node_name = names.material;
        desc.canonicalize_names = true;
        desc.make_editable = make_editable;
        return mtlx::CodeGen::generate(desc);
    };

    auto first = generate_canonical(names_a, "0.2", "first.png", "0.1, 0.2, 0.3", false, true);
    auto second = generate_canonical(names_b, "0.8", "second.png", "0.7, 0.8, 0.9", true, true);
    REQUIRE(first);
    REQUIRE(second);
    CHECK_EQ(first->module_source, second->module_source);
    CHECK_EQ(first->module_name, second->module_name);
    CHECK_EQ(first->material_name, second->material_name);
    CHECK_EQ(first->material_data_name, second->material_data_name);
    CHECK_EQ(first->data_struct_size, second->data_struct_size);
    CHECK(contains(first->module_source, "mx_canonical_graph_0"));
    CHECK_FALSE(contains(first->module_source, names_a.graph));
    CHECK_FALSE(contains(second->module_source, names_b.graph));
    CHECK(contains(first->module_source, "NG_standard_surface_surfaceshader_100"));
    CHECK(
        std::regex_search(
            first->module_source,
            std::regex(R"(\[mutating\]\s+void NG_standard_surface_surfaceshader_100\()")
        )
    );
    for (const mtlx::MxParamInfo& param : first->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());
    for (const mtlx::MxParamInfo& param : second->all_material_params.m_params)
        CHECK_EQ(param.is_editable, param.holds_value());

    mtlx::CodeGenDesc different_topology_desc;
    different_topology_desc.document = std::string(R"(<materialx version="1.39" colorspace="lin_rec709">
  <standard_surface name="surface" type="surfaceshader">
    <input name="base_color" type="color3" value="0.5, 0.5, 0.5" />
  </standard_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)");
    different_topology_desc.node_name = "M";
    different_topology_desc.canonicalize_names = true;
    different_topology_desc.make_editable = true;
    auto different_topology = mtlx::CodeGen::generate(different_topology_desc);
    REQUIRE(different_topology);
    CHECK_NE(first->module_name, different_topology->module_name);

    std::vector<const mtlx::MxParamInfo*> first_params;
    std::vector<const mtlx::MxParamInfo*> second_params;
    for (const mtlx::MxParamInfo& param : first->all_material_params.m_params)
        first_params.push_back(&param);
    for (const mtlx::MxParamInfo& param : second->all_material_params.m_params)
        second_params.push_back(&param);
    const auto binding_less = [](const mtlx::MxParamInfo* lhs, const mtlx::MxParamInfo* rhs)
    {
        return lhs->binding_name < rhs->binding_name;
    };
    std::sort(first_params.begin(), first_params.end(), binding_less);
    std::sort(second_params.begin(), second_params.end(), binding_less);
    REQUIRE_EQ(first_params.size(), second_params.size());

    bool found_distinct_property_names = false;
    bool found_distinct_float_values = false;
    bool found_distinct_texture_paths = false;
    for (size_t i = 0; i < first_params.size(); ++i) {
        CHECK_EQ(first_params[i]->binding_name, second_params[i]->binding_name);
        found_distinct_property_names |= first_params[i]->param_name != second_params[i]->param_name;
        if (const float* first_value = first_params[i]->extract_value<float>()) {
            if (const float* second_value = second_params[i]->extract_value<float>())
                found_distinct_float_values |= *first_value != *second_value;
        }
        if (const auto* first_texture = first_params[i]->extract_value<mtlx::params::TextureInfo>()) {
            if (const auto* second_texture = second_params[i]->extract_value<mtlx::params::TextureInfo>())
                found_distinct_texture_paths |= first_texture->file_name != second_texture->file_name;
        }
    }
    CHECK(found_distinct_property_names);
    CHECK(found_distinct_float_values);
    CHECK(found_distinct_texture_paths);

    auto generate_authored = [&](const UserGraphNames& names)
    {
        mtlx::CodeGenDesc desc;
        desc.document = make_document(names, "0.2", "texture.png", "0.5, 0.5, 0.5", false);
        desc.node_name = names.material;
        desc.make_editable = true;
        return mtlx::CodeGen::generate(desc);
    };
    auto authored_a = generate_authored(names_a);
    auto authored_b = generate_authored(names_b);
    REQUIRE(authored_a);
    REQUIRE(authored_b);
    CHECK_NE(authored_a->module_source, authored_b->module_source);
    CHECK(contains(authored_a->module_source, names_a.graph));
    CHECK(contains(authored_b->module_source, names_b.graph));

    auto baked_a = generate_canonical(names_a, "0.2", "first.png", "0.1, 0.2, 0.3", false, false);
    auto baked_b = generate_canonical(names_b, "0.2", "second.png", "0.1, 0.2, 0.3", true, false);
    auto baked_different_roughness = generate_canonical(names_b, "0.8", "second.png", "0.1, 0.2, 0.3", true, false);
    auto baked_different_tint = generate_canonical(names_b, "0.2", "second.png", "0.7, 0.8, 0.9", true, false);
    REQUIRE(baked_a);
    REQUIRE(baked_b);
    REQUIRE(baked_different_roughness);
    REQUIRE(baked_different_tint);
    CHECK_EQ(baked_a->module_source, baked_b->module_source);
    CHECK_NE(baked_a->module_source, baked_different_roughness->module_source);
    CHECK_NE(baked_a->module_source, baked_different_tint->module_source);
    CHECK_EQ(baked_a->all_material_params.m_editable_count, 1);
    CHECK_EQ(baked_b->all_material_params.m_editable_count, 1);

    auto generate_subset = [&](const UserGraphNames& names, std::string_view roughness)
    {
        mtlx::CodeGenDesc desc;
        desc.document = make_document(names, roughness, "texture.png", "0.5, 0.5, 0.5", false);
        desc.node_name = names.material;
        desc.canonicalize_names = true;
        desc.editable_param_names = {fmt::format("{}_specular_roughness", names.surface)};
        return mtlx::CodeGen::generate(desc);
    };
    auto first_editable = generate_subset(names_a, "0.2");
    auto second_editable = generate_subset(names_b, "0.8");
    REQUIRE(first_editable);
    REQUIRE(second_editable);
    CHECK_EQ(first_editable->module_source, second_editable->module_source);
    CHECK_EQ(first_editable->module_name, second_editable->module_name);
    CHECK_EQ(first_editable->all_material_params.m_editable_count, 2);
    CHECK_EQ(second_editable->all_material_params.m_editable_count, 2);
    for (const mtlx::MxParamInfo& param : first_editable->all_material_params.m_params)
        CHECK_EQ(param.interface.name, param.param_name);
}

TEST_CASE_NIGHTLY("scene materialx canonical naming treats user includes as authored")
{
    const std::filesystem::path root = testing::get_case_temp_directory();
    auto write_file = [](const std::filesystem::path& path, std::string_view contents)
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        REQUIRE(stream);
        stream << contents;
        REQUIRE(stream.good());
    };
    auto write_document = [&](std::string_view suffix)
    {
        const std::string include_name = fmt::format("included_{}.mtlx", suffix);
        write_file(
            root / include_name,
            fmt::format(
                R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_user_graph_{0}" node="user_graph_{0}">
    <input name="tint_{0}" type="color3" value="0.2, 0.3, 0.4" />
    <output name="out_{0}" type="color3" />
  </nodedef>
  <nodegraph name="user_graph_impl_{0}" nodedef="ND_user_graph_{0}">
    <constant name="constant_{0}" type="color3">
      <input name="value" type="color3" interfacename="tint_{0}" />
    </constant>
    <output name="out_{0}" type="color3" nodename="constant_{0}" />
  </nodegraph>
</materialx>)",
                suffix
            )
        );

        const std::filesystem::path document_path = root / fmt::format("material_{}.mtlx", suffix);
        write_file(
            document_path,
            fmt::format(
                R"(<materialx version="1.39" colorspace="lin_rec709" xmlns:xi="http://www.w3.org/2001/XInclude">
  <xi:include href="{0}" />
  <user_graph_{1} name="instance_{1}" type="color3">
    <input name="tint_{1}" type="color3" value="0.6, 0.7, 0.8" />
  </user_graph_{1}>
  <standard_surface name="surface_{1}" type="surfaceshader">
    <input name="base_color" type="color3" nodename="instance_{1}" />
  </standard_surface>
  <surfacematerial name="material_{1}" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface_{1}" />
  </surfacematerial>
</materialx>)",
                include_name,
                suffix
            )
        );
        return document_path;
    };
    auto generate = [](const std::filesystem::path& path, std::string_view suffix)
    {
        mtlx::CodeGenDesc desc;
        desc.document = path;
        desc.node_name = fmt::format("material_{}", suffix);
        desc.canonicalize_names = true;
        desc.make_editable = true;
        return mtlx::CodeGen::generate(desc);
    };

    auto first = generate(write_document("a"), "a");
    auto second = generate(write_document("b"), "b");
    REQUIRE(first);
    REQUIRE(second);
    CHECK_EQ(first->module_source, second->module_source);
    CHECK_FALSE(contains(first->module_source, "user_graph_impl_a"));
    CHECK_FALSE(contains(second->module_source, "user_graph_impl_b"));
    CHECK(contains(first->module_source, "NG_standard_surface_surfaceshader_100"));
}

TEST_CASE_NIGHTLY("scene materialx emits opacity and emission specialization defines")
{
    const std::string shader = R"(<materialx version="1.39" colorspace="lin_rec709">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <uniform_edf name="emission" type="EDF">
    <input name="color" type="color3" value="1, 0, 0" />
  </uniform_edf>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
    <input name="edf" type="EDF" nodename="emission" />
    <input name="opacity" type="float" value="0.25" />
  </surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)";
    const std::array layering_modes = {
        mtlx::LayeringMode::closure_tree,
        mtlx::LayeringMode::bsdf_mix,
    };

    for (mtlx::LayeringMode layering_mode : layering_modes) {
        mtlx::CodeGenDesc desc;
        desc.document = shader;
        desc.node_name = "M";
        desc.layering_mode = layering_mode;

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);
        CHECK(contains(result->module_source, "#define MATERIALX_EVALUATE_OPACITY 0"));
        CHECK(contains(result->module_source, "#define MATERIALX_OPACITY_IS_OPAQUE 0"));
        CHECK(contains(result->module_source, "#define MATERIALX_STATIC_OPACITY 0.250000"));
        CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
        CHECK(contains(result->module_source, "eval_opacity<TLodSampler"));
        CHECK(contains(result->module_source, "out float3 opacity_weighted_emission"));
        CHECK(contains(result->module_source, "let graph_opacity = MATERIALX_STATIC_OPACITY"));
        CHECK(contains(result->module_source, "eval_emission<TLodSampler"));
        CHECK(contains(result->module_source, "public override bool is_emissive()"));
        bool carries_bsdf_weight = false;
        const size_t storage_define = result->module_source.find("#define MTLX_SCALARIZED_GRAPH_STORAGE");
        const size_t guarded_weight
            = result->module_source.find("#ifdef MTLX_SCALARIZED_GRAPH_STORAGE\n    float3 weight = float3(1.0);");
        CHECK_NE(guarded_weight, std::string::npos);
        CHECK_EQ(storage_define != std::string::npos, carries_bsdf_weight);
        if (carries_bsdf_weight)
            CHECK_LT(storage_define, guarded_weight);
    }
}

namespace {

std::string
emission_test_document(std::string_view category, std::string_view inputs, std::string_view surface_name = "surface")
{
    return fmt::format(
        R"(<materialx version="1.39" colorspace="lin_rec709">
  <{} name="{}" type="surfaceshader">{}</{}>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="{}" />
  </surfacematerial>
</materialx>)",
        category,
        surface_name,
        inputs,
        category,
        surface_name
    );
}

std::string generated_emission_predicate(const mtlx::CodeGenResult& result)
{
    // The material instance has another is_emissive(). Inspect the material's
    // current-uniform predicate, before the capability-disabled fallback.
    const std::string local_name = result.material_name.substr(result.material_name.rfind("::") + 2);
    const size_t material = result.module_source.find("struct " + local_name + " ");
    REQUIRE_NE(material, std::string::npos);
    const size_t method = result.module_source.find("public override bool is_emissive()", material);
    REQUIRE_NE(method, std::string::npos);
    const size_t start = result.module_source.find("return ", method);
    REQUIRE_NE(start, std::string::npos);
    const size_t end = result.module_source.find(';', start);
    REQUIRE_NE(end, std::string::npos);
    return std::regex_replace(result.module_source.substr(start + 7, end - start - 7), std::regex(R"(\s+)"), "");
}

void check_editable_emission_predicate(const mtlx::CodeGenResult& result, std::string_view public_name)
{
    const auto& params = result.all_material_params.m_params;
    const auto param = std::find_if(
        params.begin(),
        params.end(),
        [&](const mtlx::MxParamInfo& candidate)
        {
            return candidate.param_name == public_name;
        }
    );
    REQUIRE(param != params.end());
    CHECK(param->is_editable);
    CHECK_EQ(generated_emission_predicate(result), "mx_emission_is_nonzero(data)");
    const size_t query = result.module_source.find("bool mx_emission_is_nonzero(");
    REQUIRE_NE(query, std::string::npos);
    const size_t end = result.module_source.find('}', query);
    const auto body = result.module_source.substr(query, end - query);
    CHECK(contains(body, "data." + param->binding_name));
    CHECK_FALSE(contains(body, "SurfaceInteraction"));
    CHECK_FALSE(contains(body, "lod_sampler"));
}

} // namespace

TEST_CASE("scene materialx current uniform emission eligibility")
{
    // Keep the primary behavior matrix on the build's default backend.
    const std::array surfaces = {
        std::pair{"standard_surface", "emission"},
        std::pair{"open_pbr_surface", "emission_luminance"},
    };
    for (const auto& [category, factor] : surfaces) {
        for (std::string_view value : {"", "0", "2"}) {
            CAPTURE(category);
            CAPTURE(value);
            mtlx::CodeGenDesc desc;
            desc.document = emission_test_document(
                category,
                value.empty() ? "" : fmt::format(R"(<input name="{}" type="float" value="{}" />)", factor, value)
            );
            desc.node_name = "M";
            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(generated_emission_predicate(*result), value == "2" ? "mx_emission_is_nonzero(data)" : "false");
            // Eligibility must not remove the graph's emission capability.
            CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
        }

        mtlx::CodeGenDesc desc;
        desc.document
            = emission_test_document(category, fmt::format(R"(<input name="{}" type="float" value="0" />)", factor));
        desc.node_name = "M";
        const std::string public_name = fmt::format("surface_{}", factor);
        desc.editable_param_names = {public_name};
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        check_editable_emission_predicate(*result, public_name);
        CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
    }
}

TEST_CASE("scene materialx emission eligibility uses canonical uniform bindings")
{
    auto generate = [](std::string_view surface_name, std::string_view value)
    {
        mtlx::CodeGenDesc desc;
        desc.document = emission_test_document(
            "standard_surface",
            fmt::format(R"(<input name="emission" type="float" value="{}" />)", value),
            surface_name
        );
        desc.node_name = "M";
        desc.editable_param_names = {fmt::format("{}_emission", surface_name)};
        desc.canonicalize_names = true;
        return mtlx::CodeGen::generate(desc);
    };
    auto dark = generate("dark_surface", "0");
    auto bright = generate("renamed_surface", "2");
    REQUIRE(dark);
    REQUIRE(bright);
    CHECK_EQ(dark->module_source, bright->module_source);
    check_editable_emission_predicate(*dark, "dark_surface_emission");
    check_editable_emission_predicate(*bright, "renamed_surface_emission");
    CHECK_FALSE(contains(generated_emission_predicate(*dark), "dark_surface"));
}

TEST_CASE("scene materialx emission eligibility follows compound and connected expressions")
{
    const std::array graphs = {
        // Follow the custom graph's actual output to its statically dark surface.
        R"(<nodedef name="ND_custom_surface" node="custom_surface">
  <output name="out" type="surfaceshader" />
</nodedef>
<nodegraph name="NG_custom_surface" nodedef="ND_custom_surface">
  <standard_surface name="inner" type="surfaceshader" />
  <output name="out" type="surfaceshader" nodename="inner" />
</nodegraph>
<custom_surface name="surface" type="surfaceshader" />)",
        // Editable connected values are evaluated from current data.
        R"(<constant name="factor" type="float"><input name="value" type="float" value="0" /></constant>
<standard_surface name="surface" type="surfaceshader">
  <input name="emission" type="float" nodename="factor" />
</standard_surface>)",
        R"(<image name="factor" type="float"><input name="file" type="filename" value="unused.exr" /></image>
<standard_surface name="surface" type="surfaceshader">
  <input name="emission" type="float" nodename="factor" />
</standard_surface>)",
        R"(<standard_surface name="dark" type="surfaceshader" />
<standard_surface name="bright" type="surfaceshader"><input name="emission" type="float" value="1" /></standard_surface>
<mix name="surface" type="surfaceshader">
  <input name="bg" type="surfaceshader" nodename="dark" />
  <input name="fg" type="surfaceshader" nodename="bright" />
  <input name="mix" type="float" value="0.5" />
</mix>)",
    };
    for (size_t i = 0; i < graphs.size(); ++i) {
        const std::string_view graph = graphs[i];
        CAPTURE(graph);
        mtlx::CodeGenDesc desc;
        desc.document = fmt::format(
            R"(<materialx version="1.39" colorspace="lin_rec709">
{}
<surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="surface" /></surfacematerial>
</materialx>)",
            graph
        );
        desc.node_name = "M";
        desc.make_editable = true;
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(generated_emission_predicate(*result), i == 0 ? "false" : "mx_emission_is_nonzero(data)");
    }
}

TEST_CASE("scene materialx emission eligibility does not trust authored builtin names")
{
    const std::array definitions = {
        std::pair{"NG_standard_surface_surfaceshader_100", "emission"},
        std::pair{"NG_open_pbr_surface_surfaceshader", "emission_luminance"},
    };
    for (const auto& [graph_name, factor_name] : definitions) {
        for (bool canonicalize : {false, true}) {
            CAPTURE(graph_name);
            CAPTURE(factor_name);
            CAPTURE(canonicalize);
            mtlx::CodeGenDesc desc;
            // The authored graph ignores its zero multiplier and emits through
            // an independent EDF. Query its actual emission, never its graph name.
            desc.document = fmt::format(
                R"(<materialx version="1.39" colorspace="lin_rec709">
  <nodedef name="ND_custom_surface" node="custom_surface">
    <input name="{1}" type="float" value="0" />
    <output name="out" type="surfaceshader" />
  </nodedef>
  <nodegraph name="{0}" nodedef="ND_custom_surface">
    <uniform_edf name="light" type="EDF">
      <input name="color" type="color3" value="1, 1, 1" />
    </uniform_edf>
    <surface name="inner" type="surfaceshader">
      <input name="edf" type="EDF" nodename="light" />
    </surface>
    <output name="out" type="surfaceshader" nodename="inner" />
  </nodegraph>
  <custom_surface name="surface" type="surfaceshader">
    <input name="{1}" type="float" value="0" />
  </custom_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)",
                graph_name,
                factor_name
            );
            desc.node_name = "M";
            desc.canonicalize_names = canonicalize;
            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
            CHECK_EQ(generated_emission_predicate(*result), "mx_emission_is_nonzero(data)");
        }
    }
}

TEST_CASE("scene materialx editable emission eligibility across layering modes")
{
    const std::array layering_modes = {
        mtlx::LayeringMode::closure_tree,
        mtlx::LayeringMode::bsdf_mix,
    };
    for (mtlx::LayeringMode layering_mode : layering_modes) {
        mtlx::CodeGenDesc desc;
        desc.document
            = emission_test_document("standard_surface", R"(<input name="emission" type="float" value="0" />)");
        desc.node_name = "M";
        desc.layering_mode = layering_mode;
        desc.editable_param_names = {"surface_emission"};
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        check_editable_emission_predicate(*result, "surface_emission");
        const std::string local_name = result->material_name.substr(result->material_name.rfind("::") + 2);
        const size_t material = result->module_source.find("struct " + local_name + " ");
        const size_t method = result->module_source.find("public override bool is_emissive()", material);
        const size_t end = result->module_source.find('}', method);
        CHECK(contains(result->module_source.substr(method, end - method), "data_buffer.load<"));
    }
}

TEST_CASE_GPU("scene materialx emission eligibility inline and spilled payloads")
{
    const std::array factors = {0.f, 2.f, 0.f};
    auto storage = ctx.device->create_buffer(
        {.size = sizeof(factors),
         .usage = sgl::BufferUsage::shader_resource,
         .data = factors.data(),
         .data_size = sizeof(factors)}
    );
    for (size_t payload_size : {size_t(0), size_t(64)}) {
        CAPTURE(payload_size);
        mtlx::CodeGenDesc desc;
        desc.document
            = emission_test_document("standard_surface", R"(<input name="emission" type="float" value="0" />)");
        desc.node_name = "M";
        desc.editable_param_names = {"surface_emission"};
        desc.canonicalize_names = true;
        desc.payload_size = payload_size;
        auto generated = mtlx::CodeGen::generate(desc);
        REQUIRE(generated);
        // Exactly one editable float lets each buffer offset hold one payload.
        REQUIRE_EQ(generated->data_struct_size, sizeof(float));
        const auto& params = generated->all_material_params.m_params;
        const auto param = std::find_if(
            params.begin(),
            params.end(),
            [](const mtlx::MxParamInfo& candidate)
            {
                return candidate.param_name == "surface_emission";
            }
        );
        REQUIRE(param != params.end());
        REQUIRE(param->is_editable);
        auto material_module = ctx.device->load_module_from_source(generated->module_name, generated->module_source);
        REQUIRE(material_module);
        const bool spilled = payload_size == 0;
        const std::string source = fmt::format(
            R"(import {};
import falcor2.utils;
[shader("compute")]
[numthreads(1, 1, 1)]
void main(BufferOffsetPair payload, float factor, RWStructuredBuffer<uint> result, uint index)
{{
    {} material = ({})0;
    {}
    result[index] = material.is_emissive() ? 1 : 0;
}}
)",
            generated->module_name,
            generated->material_name,
            generated->material_name,
            spilled ? "material.data_buffer = payload;" : fmt::format("material.data.{} = factor;", param->binding_name)
        );
        auto probe_module = ctx.device->load_module_from_source(generated->module_name + "_emission_probe", source);
        REQUIRE(probe_module);
        auto kernel = ctx.device->create_compute_kernel(
            {.program = ctx.device->link_program({probe_module}, {probe_module->entry_point("main")})}
        );
        auto result = ctx.device->create_buffer(
            {.size = factors.size() * sizeof(uint32_t), .usage = sgl::BufferUsage::unordered_access}
        );
        for (uint32_t index = 0; index < factors.size(); ++index) {
            kernel->dispatch(
                uint3(1),
                [&](sgl::ShaderCursor cursor)
                {
                    cursor = cursor.find_entry_point(0);
                    if (spilled)
                        cursor["payload"]
                            = BufferOffsetPair{.buffer = to_handle(storage), .offset = index * uint32_t(sizeof(float))};
                    else
                        cursor["factor"] = factors[index];
                    cursor["result"] = result;
                    cursor["index"] = index;
                }
            );
        }
        const std::vector<uint32_t> expected = {0, 1, 0};
        CHECK(result->get_elements<uint32_t>() == expected);
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx publishes root opacity classification")
{
    // This checks API-independent host behavior, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    // This checks the mapping from codegen's three opacity kinds to Material.
    // The CPU matrix above covers folding, luma conversion and authored forms;
    // Python test_materialx_material_opacity_evaluation covers GPU evaluation.
    auto scene = Scene::create(ref(ctx.device));
    auto classify = [&](MaterialXOpacityCase opacity_case, bool editable = false)
    {
        Properties props;
        props.set("mtlx_buffer", materialx_opacity_shader(opacity_case));
        props.set("mtlx_node_name", std::string("M"));
        if (editable)
            props.set("mtlx_editable_params", std::string("*"));
        MaterialXMaterial* material = scene->create_material<MaterialXMaterial>(props);
        return material->opacity_desc();
    };

    CHECK_EQ(classify(MaterialXOpacityCase::opaque).flags, shared::OpacityFlags::none);

    const Material::OpacityDesc constant = classify(MaterialXOpacityCase::constant);
    CHECK_EQ(constant.flags, shared::OpacityFlags::enabled);
    CHECK_EQ(constant.factor, doctest::Approx(0.25f));

    const auto evaluated_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
    CHECK_EQ(classify(MaterialXOpacityCase::opaque, true).flags, evaluated_flags);
}

TEST_CASE_NIGHTLY("scene materialx 1.39 successor variants compile without 1.38")
{
    struct Variant {
        std::string name;
        std::string shader;
        bool expects_layering = false;
    };

    const std::vector<Variant> variants = {
        {
            "standard_surface",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <standard_surface name="SR_brass1" type="surfaceshader">
    <input name="base" type="float" value="1" />
    <input name="base_color" type="color3" value="0.8, 0.6, 0.25" />
    <input name="specular" type="float" value="0" />
    <input name="metalness" type="float" value="1" />
    <input name="coat" type="float" value="1" />
  </standard_surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="SR_brass1" />
  </surfacematerial>
</materialx>
)",
            false,
        },
        {
            "multioutput_layer",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodedef name="ND_double_diffuse" node="double_diffuse">
    <input name="color1" type="color3" value="0.8, 0.8, 0.8" />
    <output name="out_bsdf1" type="BSDF"/>
    <output name="out_bsdf2" type="BSDF"/>
  </nodedef>
  <nodegraph name="IM_double_diffuse" nodedef="ND_double_diffuse">
    <oren_nayar_diffuse_bsdf name="diffuse1_bsdf" type="BSDF">
        <input name="color" type="color3" interfacename="color1"/>
    </oren_nayar_diffuse_bsdf>
    <oren_nayar_diffuse_bsdf name="diffuse2_bsdf" type="BSDF" />
    <output name="out_bsdf1" type="BSDF" nodename="diffuse1_bsdf" />
    <output name="out_bsdf2" type="BSDF" nodename="diffuse2_bsdf" />
  </nodegraph>
  <double_diffuse name="NG_diffuse1" type="multioutput" />
  <double_diffuse name="NG_diffuse2" type="multioutput" />
  <layer name="layer1" type="BSDF">
    <input name="top" type="BSDF" nodename="NG_diffuse1" output="out_bsdf1"/>
    <input name="base" type="BSDF" nodename="NG_diffuse2" output="out_bsdf2"/>
  </layer>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="layer1" />
  </surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>
)",
            true,
        },
        {
            "interface_layer",
            R"(
<?xml version="1.0"?>
<materialx version="1.38" colorspace="lin_rec709">
  <nodedef name="ND_interface_layer" node="interface_layer" nodegroup="pbr">
    <input name="top" type="BSDF" />
    <input name="base" type="BSDF" />
    <output name="out" type="BSDF" />
  </nodedef>
  <nodegraph name="IM_interface_layer" nodedef="ND_interface_layer">
    <layer name="layer1" type="BSDF">
      <input name="top" type="BSDF" interfacename="top" />
      <input name="base" type="BSDF" interfacename="base" />
    </layer>
    <output name="out" type="BSDF" nodename="layer1" />
  </nodegraph>
  <oren_nayar_diffuse_bsdf name="diffuse1" type="BSDF" />
  <oren_nayar_diffuse_bsdf name="diffuse2" type="BSDF" />
  <interface_layer name="wrapped_layer" type="BSDF">
    <input name="top" type="BSDF" nodename="diffuse1" />
    <input name="base" type="BSDF" nodename="diffuse2" />
  </interface_layer>
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="wrapped_layer" />
  </surface>
  <surfacematerial name="Tiled_Brass" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>
)",
            true,
        },
    };

    for (const Variant& variant : variants) {
        INFO("MaterialX 1.39 successor variant " << variant.name);
        auto result = mtlx::CodeGen::generate(make_materialx_codegen_desc(variant.shader));
        REQUIRE(result);
        CHECK(starts_with(result->module_name, "mtlx_"));
        CHECK(contains(result->module_source, "namespace mtlx"));
        CHECK_FALSE(contains(result->module_source, "MaterialX138"));
        if (variant.expects_layering)
            CHECK_FALSE(normalize_layering_traversal_order(layering_traversal_order(*result), true).empty());
    }
}

TEST_CASE_NIGHTLY("scene materialx targets Falcor linear Rec.709 for rootless documents")
{
    mtlx::CodeGenDesc desc;
    desc.document = rootless_srgb_material_shader();
    desc.node_name = "M";

    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(contains(result->module_source, "NG_srgb_texture_to_lin_rec709_color3(albedo_out, albedo_out_cm_out)"));
}

TEST_CASE("materialx rejects shader source in user documents")
{
    const std::filesystem::path root = testing::get_case_temp_directory();
    const auto write_file = [](const std::filesystem::path& path, std::string_view contents)
    {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        REQUIRE(stream);
        stream << contents;
        REQUIRE(stream.good());
    };
    const auto check_rejected = [](const mtlx::CodeGenDesc& desc)
    {
        const auto diagnostic
            = doctest::Contains("MaterialX source implementation 'IM_user' is unsupported in material documents");
        CHECK_THROWS_WITH_AS(mtlx::CodeGen::generate(desc), diagnostic, std::exception);
        CHECK_THROWS_WITH_AS(mtlx::CodeGen::discover_renderable_elements(desc), diagnostic, std::exception);
    };
    for (const char* source : {R"(sourcecode="0.5")", R"(file="missing.slang" function="user_function")"}) {
        for (const char* target : {"genslangpt", "genglsl"}) {
            CAPTURE(source);
            CAPTURE(target);
            // The source implementation is deliberately unused by the material.
            const std::string document = fmt::format(
                R"(<materialx version="1.39">
  <nodedef name="ND_user" node="user"><output name="out" type="float" /></nodedef>
  <implementation name="IM_user" nodedef="ND_user" target="{}" {} />
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
</materialx>)",
                target,
                source
            );
            mtlx::CodeGenDesc desc;
            desc.node_name = "surface";
            desc.document = document;
            check_rejected(desc);

            write_file(root / "source.mtlx", document);
            write_file(
                root / "included.mtlx",
                R"(<materialx version="1.39" xmlns:xi="http://www.w3.org/2001/XInclude">
  <xi:include href="source.mtlx" />
</materialx>)"
            );
            write_file(
                root / "material.mtlx",
                R"(<materialx version="1.39" xmlns:xi="http://www.w3.org/2001/XInclude">
  <xi:include href="included.mtlx" />
</materialx>)"
            );
            desc.document = root / "material.mtlx";
            check_rejected(desc);
        }
    }
}

TEST_CASE("materialx permits library shader source and user graph implementations")
{
    mtlx::CodeGenDesc desc;
    desc.node_name = "surface";
    desc.document = std::string(R"(<materialx version="1.39">
  <nodedef name="ND_user" node="user"><output name="out" type="color3" /></nodedef>
  <nodegraph name="NG_user">
    <constant name="color" type="color3"><input name="value" type="color3" value="0.2, 0.3, 0.4" /></constant>
    <output name="out" type="color3" nodename="color" />
  </nodegraph>
  <implementation name="IM_user" nodedef="ND_user" nodegraph="NG_user" target="genslangpt" />
  <user name="color" type="color3" />
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF"><input name="color" type="color3" nodename="color" /></oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
</materialx>)");
    CHECK_FALSE(mtlx::CodeGen::generate(desc)->module_source.empty());
    CHECK_FALSE(mtlx::CodeGen::discover_renderable_elements(desc).empty());

    // weighted_layer is defined by our source-backed custom_nodes.mtlx library.
    desc.document = weighted_layer_shader();
    desc.node_name = "M";
    CHECK_FALSE(mtlx::CodeGen::generate(desc)->module_source.empty());
}

TEST_CASE_NIGHTLY("scene materialx keeps shared library preparation isolated")
{
    mtlx::CodeGenDesc desc;
    const std::string nodes = R"(
  <noise3d name="height" type="float" />
  <heighttonormal name="normal" type="vector3"><input name="in" type="float" nodename="height" /></heighttonormal>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF"><input name="normal" type="vector3" nodename="normal" /></oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
)";
    desc.document = "<materialx version=\"1.39\">" + nodes + "</materialx>";
    desc.node_name = "surface";
    const auto original = mtlx::CodeGen::generate(desc);
    desc.use_slang_derivatives = true;
    CHECK_NE(mtlx::CodeGen::generate(desc)->module_source, original->module_source);
    desc.use_slang_derivatives = false;
    CHECK_EQ(mtlx::CodeGen::generate(desc)->module_source, original->module_source);

    // An unused authored graph selects private imports. Both ownership paths
    // must generate the same material on each supported backend.
    for (const auto mode :
         {mtlx::default_layering_mode(), mtlx::LayeringMode::bsdf_mix, mtlx::LayeringMode::closure_tree}) {
        desc.layering_mode = mode;
        const auto shared = mtlx::CodeGen::generate(desc);
        auto private_desc = desc;
        private_desc.document = "<materialx version=\"1.39\">" + nodes + R"(<nodegraph name="unused" /></materialx>)";
        const auto private_copy = mtlx::CodeGen::generate(private_desc);
        CHECK_EQ(shared->module_source, private_copy->module_source);
        CHECK(shared->codegen_metadata == private_copy->codegen_metadata);
    }
    desc.layering_mode = mtlx::default_layering_mode();

    // An unused malformed authored node must still be validated.
    desc.document = "<materialx version=\"1.39\">" + nodes
        + R"(<add name="unused" type="float"><input name="in1" type="float" nodename="missing" /></add></materialx>)";
    CHECK_THROWS(mtlx::CodeGen::generate(desc));
    desc.document = "<materialx version=\"1.39\">" + nodes + "</materialx>";
    CHECK_EQ(mtlx::CodeGen::generate(desc)->module_source, original->module_source);
}

TEST_CASE_NIGHTLY("materialx codegen cache preserves instance data and isolates programs")
{
    auto make_desc = [](int instance)
    {
        mtlx::CodeGenDesc desc;
        desc.canonicalize_names = true;
        desc.editable_param_names.insert(fmt::format("diffuse{}_roughness", instance));
        desc.document = fmt::format(
            R"(<materialx version="1.39">
  <image name="image{0}" type="color3"><input name="file" type="filename" value="texture{0}.png" /></image>
  <oren_nayar_diffuse_bsdf name="diffuse{0}" type="BSDF">
    <input name="color" type="color3" nodename="image{0}" />
    <input name="roughness" type="float" value="{1}" />
  </oren_nayar_diffuse_bsdf>
  <surface name="surface{0}" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse{0}" /></surface>
</materialx>)",
            instance,
            0.2f + 0.1f * instance
        );
        desc.node_name = fmt::format("surface{}", instance);
        return desc;
    };
    mtlx::CodeGenCache cache;
    std::barrier start(4);
    std::vector<std::future<std::unique_ptr<mtlx::CodeGenResult>>> tasks;
    for (int i = 0; i < 4; ++i) {
        tasks.push_back(
            std::async(
                std::launch::async,
                [&, i]
                {
                    auto desc = make_desc(i);
                    start.arrive_and_wait();
                    return mtlx::CodeGen::generate(desc, &cache);
                }
            )
        );
    }
    for (int i = 0; i < 4; ++i) {
        CAPTURE(i);
        const auto result = tasks[i].get();
        const auto reference = mtlx::CodeGen::generate(make_desc(i));
        CHECK_EQ(result->module_source, reference->module_source);
        CHECK(result->codegen_metadata == reference->codegen_metadata);
        bool found_roughness = false, found_texture = false;
        for (const auto& param : result->all_material_params.m_params) {
            if (param.param_name == fmt::format("diffuse{}_roughness", i)) {
                const auto* value = param.extract_value<float>();
                REQUIRE(value);
                CHECK_EQ(*value, doctest::Approx(0.2f + 0.1f * i));
                found_roughness = true;
            }
            if (const auto* texture = param.extract_value<mtlx::params::TextureInfo>()) {
                CHECK(contains(texture->file_name, fmt::format("texture{}.png", i)));
                found_texture = true;
            }
        }
        CHECK(found_roughness);
        CHECK(found_texture);
    }
    CHECK_EQ(cache.size(), 1);

    auto desc = make_desc(7);
    const auto check = [&]
    {
        const auto reference = mtlx::CodeGen::generate(desc);
        const auto result = mtlx::CodeGen::generate(desc, &cache);
        CHECK_EQ(result->module_source, reference->module_source);
        CHECK_EQ(result->data_struct_size, reference->data_struct_size);
        CHECK(result->codegen_metadata == reference->codegen_metadata);
        REQUIRE_EQ(result->all_material_params.m_params.size(), reference->all_material_params.m_params.size());
        for (size_t i = 0; i < result->all_material_params.m_params.size(); ++i) {
            const auto& actual = result->all_material_params.m_params[i];
            const auto& expected = reference->all_material_params.m_params[i];
            CHECK_EQ(actual.param_name, expected.param_name);
            CHECK_EQ(actual.binding_name, expected.binding_name);
            if (const auto* value = expected.extract_value<float>()) {
                REQUIRE(actual.extract_value<float>());
                CHECK_EQ(*actual.extract_value<float>(), *value);
            }
            if (const auto* texture = expected.extract_value<mtlx::params::TextureInfo>()) {
                REQUIRE(actual.extract_value<mtlx::params::TextureInfo>());
                CHECK_EQ(actual.extract_value<mtlx::params::TextureInfo>()->file_name, texture->file_name);
            }
        }
    };
    check(); // A warm renamed hit must also retain its own metadata.
    CHECK_EQ(cache.size(), 1);
    desc.editable_param_names.clear();
    check(); // Baking a formerly live value changes the program.
    CHECK_EQ(cache.size(), 2);
    desc = make_desc(8);
    desc.editable_param_names.clear();
    check(); // Same runtime mask, different baked value.
    CHECK_EQ(cache.size(), 3);
    for (auto mode : {mtlx::LayeringMode::bsdf_mix, mtlx::LayeringMode::closure_tree}) {
        desc.layering_mode = mode;
        check();
    }
    desc = make_desc(7);
    size_t size = cache.size();
    auto xml = std::get<std::string>(desc.document);
    desc.document = xml.substr(0, xml.find("</materialx>")) + "<nodegraph name=\"private\" /></materialx>";
    check(); // Private library ownership does not change the generated program.
    CHECK_EQ(cache.size(), size);
    desc.document = std::string("<materialx version=\"1.39\"><missing name=\"bad\" type=\"BSDF\" /></materialx>");
    CHECK_THROWS(mtlx::CodeGen::generate(desc, &cache));
    desc = make_desc(7);
    check();
    CHECK_EQ(cache.size(), size);

    // Custom graphs with the same implementation name must remain distinct
    // when their supported operations differ, including on a warm cache hit.
    desc.node_name = "surface";
    for (const char* operation : {"add", "multiply", "add"}) {
        desc.document = fmt::format(
            R"xml(<materialx version="1.39">
  <nodedef name="ND_custom" node="custom"><output name="out" type="color3" /></nodedef>
  <nodegraph name="NG_custom" nodedef="ND_custom">
    <{0} name="value" type="color3">
      <input name="in1" type="color3" value="0.1, 0.2, 0.3" />
      <input name="in2" type="color3" value="0.4, 0.5, 0.6" />
    </{0}>
    <output name="out" type="color3" nodename="value" />
  </nodegraph>
  <custom name="color" type="color3" />
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF"><input name="color" type="color3" nodename="color" /></oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
</materialx>)xml",
            operation
        );
        check();
    }
    size += 2;
    CHECK_EQ(cache.size(), size);

    desc.document = std::string(R"(<materialx version="1.39">
  <geompropvalue name="attr" type="color3"><input name="geomprop" type="string" value="custom_color" /></geompropvalue>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF"><input name="color" type="color3" nodename="attr" /></oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
</materialx>)");
    desc.node_name = "surface";
    int registrations = 0;
    for (uint32_t id : {4200u, 4201u, 4200u}) {
        desc.reserve_geometry_attribute_id = [&](std::string_view)
        {
            ++registrations;
            return id;
        };
        const auto result = mtlx::CodeGen::generate(desc, &cache);
        CHECK(contains(result->module_source, std::to_string(id)));
        CHECK(result->uses_geometry_properties);
    }
    CHECK_EQ(registrations, 3); // Registration still runs on the warm hit.
    size += 2;
    CHECK_EQ(cache.size(), size);

    // Editable defaults can still affect baked lobe flags.
    desc = {};
    desc.canonicalize_names = true;
    desc.node_name = "surface";
    desc.editable_param_names.insert("dielectric_scatter_mode");
    for (const char* scatter : {"R", "T"}) {
        desc.document = fmt::format(
            R"(<materialx version="1.39">
  <dielectric_bsdf name="dielectric" type="BSDF"><input name="scatter_mode" type="string" value="{}" /></dielectric_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="dielectric" /></surface>
</materialx>)",
            scatter
        );
        check();
    }
    CHECK_EQ(cache.size(), size + 2);
}

TEST_CASE_NIGHTLY("scene materialx normalizes supported authored roots")
{
    auto check_root = [&](const std::string& document, const std::string& node_name, int expected_bsdf_count)
    {
        mtlx::CodeGenDesc desc;
        desc.document = document;
        desc.node_name = node_name;
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), expected_bsdf_count);
    };

    SUBCASE("qualified value output is wrapped as a material preview")
    {
        check_root(color4_to_float_channel_preview_shader(), "NG_color4_to_float_g/out", 1);
    }
    SUBCASE("unique local output name resolves inside a nodegraph")
    {
        check_root(mix_edf_surface_shader(), "test_surface_result", 0);
    }
    SUBCASE("unconsumed top-level BSDF is an automatic root")
    {
        check_root(unconsumed_bsdf_shader(), "", 1);
    }
    SUBCASE("custom material keeps instance values and connections")
    {
        const std::array gains = {
            std::string{},
            std::string{R"(<input name="gain" type="float" value="0.8" />)"},
            std::string{R"(<input name="gain" type="float" nodename="override" />)"},
        };
        for (size_t i = 0; i < gains.size(); ++i) {
            CAPTURE(i);
            mtlx::CodeGenDesc desc;
            desc.document = custom_material_implementation_shader(gains[i]);
            desc.node_name = "root_material";
            desc.editable_param_names = {i == 2 ? "override_value" : i == 0 ? "root_material/gain" : "gain"};
            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 1);
            REQUIRE_EQ(result->all_material_params.m_editable_count, 1);
            for (const auto& param : result->all_material_params.m_params) {
                if (!param.is_editable)
                    continue;
                REQUIRE(param.extract_value<float>());
                CHECK_EQ(*param.extract_value<float>(), i == 0 ? 0.2f : 0.8f);
            }
        }
    }
    SUBCASE("preview root on alternate layering backends")
    {
        // Root resolution is shared. One preview witnesses its integration with
        // each other backend without multiplying every authored root form.
        const std::array modes = {
            mtlx::LayeringMode::closure_tree,
            mtlx::LayeringMode::bsdf_mix,
        };
        for (mtlx::LayeringMode mode : modes) {
            if (mode == mtlx::default_layering_mode())
                continue;
            CAPTURE(mode);
            mtlx::CodeGenDesc desc;
            desc.document = color4_to_float_channel_preview_shader();
            desc.node_name = "NG_color4_to_float_g/out";
            desc.layering_mode = mode;
            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 1);
        }
    }
}


TEST_CASE_NIGHTLY("scene materialx discovery ignores generation-only settings")
{
    mtlx::CodeGenDesc desc;
    desc.document = empty_surface_shader();
    // Discovery only parses the document. Options rejected by shader
    // generation must not prevent callers from enumerating its materials.
    desc.positionfree_layering = true;
    desc.sigma_a = float3(1.f);

    const std::vector<mtlx::RenderableElement> elements = mtlx::CodeGen::discover_renderable_elements(desc);
    REQUIRE_EQ(elements.size(), 1);
    CHECK_EQ(elements.front().name, "M");
    CHECK_EQ(elements.front().type, "material");
}

TEST_CASE_NIGHTLY("scene materialx retains every dynamic OpenPBR lobe")
{
    const std::vector modes = {
        mtlx::LayeringMode::bsdf_mix,
    };
    for (mtlx::LayeringMode mode : modes) {
        CAPTURE(mode);
        mtlx::CodeGenDesc desc;
        desc.document = testing::project_directory()
            / "external/MaterialX/resources/Materials/Examples/OpenPbr/open_pbr_default.mtlx";
        desc.node_name = "Default";
        desc.layering_mode = mode;
        desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::all;
        desc.make_editable = true;

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 11);
    }
}

TEST_CASE_NIGHTLY("scene materialx preserves saturated weights when pruning asymmetric mixes")
{
    for (std::string_view weight_type : {"float", "color3"}) {
        for (bool keep_foreground : {false, true}) {
            CAPTURE(weight_type);
            CAPTURE(keep_foreground);
            mtlx::CodeGenDesc desc;
            desc.document = asymmetric_bsdf_mix_shader(weight_type, keep_foreground);
            desc.node_name = "M";
            desc.layering_mode = mtlx::LayeringMode::bsdf_mix;
            desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::all;
            // Keep only the mix weight dynamic. The zero-weight BSDF must
            // remain statically empty to exercise the asymmetric rewrite.
            desc.editable_param_names = {"asymmetric_mix_mix"};

            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->codegen_metadata.get<int>("closure_pruning_asymmetric_mix_rewrites"), 1);
            CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 1);
            CHECK(contains(result->module_source, "_closure_prune_clamp_"));
            CHECK_EQ(contains(result->module_source, "_closure_prune_inv_"), !keep_foreground);
            if (weight_type == "color3")
                CHECK(contains(result->module_source, "float3 _closure_prune_clamp_"));
            else
                CHECK(contains(result->module_source, "float _closure_prune_clamp_"));
        }
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx compiles color-weighted asymmetric mix pruning")
{
    // Only the Slang frontend is tested here, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    mtlx::CodeGenDesc desc;
    desc.document = asymmetric_bsdf_mix_shader("color3", false);
    desc.node_name = "M";
    desc.layering_mode = mtlx::LayeringMode::bsdf_mix;
    desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::all;
    desc.editable_param_names = {"asymmetric_mix_mix"};

    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(ctx.device->load_module_from_source(result->module_name, result->module_source));
}

TEST_CASE_NIGHTLY("scene materialx distinguishes frames in nested reused nodegraphs")
{
    const std::vector modes = {
        mtlx::LayeringMode::bsdf_mix,
    };
    for (mtlx::LayeringMode mode : modes) {
        for (bool canonicalize_names : {false, true}) {
            CAPTURE(mode);
            CAPTURE(canonicalize_names);
            mtlx::CodeGenDesc desc;
            desc.document = testing::project_directory() / "tests/data/materialx/nested_reused_distinct_normals.mtlx";
            desc.node_name = "NestedDistinctNormals";
            desc.layering_mode = mode;
            desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::all;
            desc.canonicalize_names = canonicalize_names;

            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 2);
        }
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx imports shared unlit transmission BSDF")
{
    // Compile the public shared type independently of generated material source.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    REQUIRE(ctx.device->load_module_from_source(
        "test_shared_unlit_transmission",
        R"(import falcor2.render.materials.mtlx.mtlx;
public mtlx::MxSurfaceUnlitTransparentBSDF make_transparent(float3 tint)
{
    return mtlx::create_mx_transparent(tint);
})"
    ));

    std::vector modes = {mtlx::default_layering_mode()};
    for (const auto mode : {mtlx::LayeringMode::closure_tree, mtlx::LayeringMode::bsdf_mix})
        if (mode != mtlx::default_layering_mode())
            modes.push_back(mode);
    for (const auto mode : modes) {
        CAPTURE(mode);
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <surface_unlit name="surface" type="surfaceshader">
    <input name="transmission" type="float" value="0" />
  </surface_unlit>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
</materialx>)");
        desc.node_name = "M";
        desc.editable_param_names = {"surface_transmission"};
        if (mode != mtlx::default_layering_mode())
            desc.layering_mode = mode;
        const auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        REQUIRE(ctx.device->load_module_from_source(result->module_name, result->module_source));
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx preserves an emissive surface without a BSDF")
{
    // Only the Slang frontend is tested here, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    const std::vector modes = {
        mtlx::LayeringMode::bsdf_mix,
    };
    for (mtlx::LayeringMode mode : modes) {
        CAPTURE(mode);
        mtlx::CodeGenDesc desc;
        desc.document = empty_surface_shader();
        desc.node_name = "M";
        desc.layering_mode = mode;

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 0);
        CHECK_EQ(result->opacity.kind, mtlx::CodeGenResult::OpacityResult::Kind::constant);
        CHECK_EQ(result->opacity.constant, doctest::Approx(0.25f));
        CHECK(contains(result->module_source, "#define MATERIALX_HAS_EMISSION 1"));
        CHECK(contains(result->module_source, "float3(0.100000001, 0.800000012, 0.200000003)"));
        CHECK(ctx.device->load_module_from_source(result->module_name, result->module_source));
    }
}

// These cases inspect host-generated source. Runtime compilation and writes are
// covered by the material AOV integration and layering-mode image tests.
TEST_CASE_NIGHTLY("scene materialx preserves declared multioutput root outputs")
{
    auto registry = make_ref<MaterialAOVRegistry>();
    mtlx::CodeGenDesc desc;
    desc.material_aov_registry = registry.get();
    desc.document = testing::project_directory() / "tests/data/materialx/additional_graph_outputs.mtlx";
    desc.node_name = "stormtrooper_dust_surface";
    desc.layering_mode = mtlx::LayeringMode::bsdf_mix;

    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(contains(result->module_source, "float dust_mask_output = "));
    CHECK(contains(result->module_source, "float metallicity_output = "));

    desc.additional_output_names = {"dust_mask_output"};
    result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    REQUIRE_EQ(result->material_aov_outputs.size(), 1u);
    CHECK_EQ(result->material_aov_outputs[0].name, "dust_mask_output");

    desc.additional_output_names.clear();
    result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(result->material_aov_outputs.empty());
    CHECK_FALSE(contains(result->module_source, "write_material_aovs"));
    desc.additional_output_names = {"*"};

    // A downstream standard material exposes only its own output, not unused
    // sibling sockets on the upstream multi-output node.
    desc.node_name = "stormtrooper_dust_material";
    result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(result->material_aov_outputs.empty());
    CHECK_FALSE(contains(result->module_source, "write_material_aovs"));
}

TEST_CASE_NIGHTLY("scene materialx combines explicitly selected additional graph outputs")
{
    std::string document
        = read_text_file(testing::project_directory() / "tests/data/materialx/additional_independent_outputs.mtlx");
    const size_t document_end = document.rfind("</materialx>");
    REQUIRE(document_end != std::string::npos);
    document.insert(document_end, R"(
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="dust_surface" />
  </surfacematerial>
)");

    mtlx::CodeGenDesc desc;
    desc.document = document;
    desc.layering_mode = mtlx::LayeringMode::bsdf_mix;

    // AOVs do not change primary-root selection: explicit output, surface node,
    // material node, and automatic single-material discovery all work.
    for (std::string_view root : {"surface_output", "dust_surface", "M", ""}) {
        CAPTURE(root);
        desc.node_name = root;
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(contains(result->module_source, "float dust_mask_output = "));
        // Shared authored image work is emitted once in the Graph definition.
        const std::string_view sample = "sample_texture(dust_image_file";
        REQUIRE(result->module_source.find(sample) != std::string::npos);
        CHECK_EQ(result->module_source.find(sample), result->module_source.rfind(sample));
    }

    desc.additional_output_names = {"dust_*", "*mask*", "dust_mask_output"};
    auto patterned = mtlx::CodeGen::generate(desc);
    REQUIRE(patterned);
    CHECK(contains(patterned->module_source, "float dust_mask_output = "));
    desc.additional_output_names = {"absent_*"};
    CHECK_FALSE(contains(mtlx::CodeGen::generate(desc)->module_source, "float dust_mask_output = "));
    desc.additional_output_names = {"missing"};
    CHECK_THROWS_WITH(mtlx::CodeGen::generate(desc), doctest::Contains("could not find additional output"));

    desc.additional_output_names.clear();
    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK_FALSE(contains(result->module_source, "float dust_mask_output = "));

    // A document AOV may use the same name as the material's primary socket.
    document.insert(document.rfind("</materialx>"), R"(
  <constant name="mask" type="float"><input name="value" type="float" value="0.5" /></constant>
  <output name="out" type="float" nodename="mask" />
)");
    desc.document = document;
    desc.additional_output_names = {"*"};
    result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    // 'out' is a Slang keyword, so the generated field uses a sanitized identifier.
    CHECK(contains(result->module_source, "float out1"));

    document.insert(document.rfind("</materialx>"), R"(
  <surfacematerial name="M2" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="dust_surface" />
  </surfacematerial>
)");
    desc.document = document;
    desc.additional_output_names = {"dust_mask_output"};
    CHECK_THROWS_WITH(mtlx::CodeGen::generate(desc), doctest::Contains("provide node_name"));
    desc.node_name = "M2";
    result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    CHECK(contains(result->module_source, "float dust_mask_output = "));
}

TEST_CASE_NIGHTLY("scene materialx rejects a closure as an additional graph output")
{
    mtlx::CodeGenDesc desc;
    desc.document = std::string(R"(
<materialx version="1.39">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader">
    <input name="bsdf" type="BSDF" nodename="diffuse" />
  </surface>
  <surfacematerial name="material" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface" />
  </surfacematerial>
  <output name="surface_output" type="surfaceshader" nodename="surface" />
  <output name="material_output" type="material" nodename="material" />
</materialx>
)");
    desc.node_name = "surface_output";
    desc.additional_output_names = {"material_output"};
    desc.layering_mode = mtlx::LayeringMode::bsdf_mix;

    CHECK_THROWS_WITH(
        mtlx::CodeGen::generate(desc),
        doctest::Contains("only the main output may be material or closure-shaped")
    );
}

TEST_CASE_NIGHTLY("scene materialx emits registered material AOV producers")
{
    const auto count_occurrences = [](std::string_view source, std::string_view needle)
    {
        size_t count = 0;
        for (size_t offset = 0; (offset = source.find(needle, offset)) != std::string_view::npos;
             offset += needle.size()) {
            ++count;
        }
        return count;
    };

    SUBCASE("selected scalar and vector outputs write Graph values with stable IDs")
    {
        std::string document
            = read_text_file(testing::project_directory() / "tests/data/materialx/additional_graph_outputs.mtlx");
        const size_t document_end = document.rfind("</materialx>");
        REQUIRE(document_end != std::string::npos);
        // Combine vector transport with scalar selection in the same graph,
        // retaining the compound root and its metallicity output.
        document.insert(document_end, R"(
  <constant name="mask2_value" type="vector2">
    <input name="value" type="vector2" value="0.1, 0.2" />
  </constant>
  <constant name="mask3_value" type="vector3">
    <input name="value" type="vector3" value="0.1, 0.2, 0.3" />
  </constant>
  <constant name="mask4_value" type="vector4">
    <input name="value" type="vector4" value="0.1, 0.2, 0.3, 0.4" />
  </constant>
  <output name="mask2" type="vector2" nodename="mask2_value" />
  <output name="mask3" type="vector3" nodename="mask3_value" />
  <output name="mask4" type="vector4" nodename="mask4_value" />
)");
        std::vector layering_modes = {mtlx::default_layering_mode()};
        for (auto mode : {mtlx::LayeringMode::closure_tree, mtlx::LayeringMode::bsdf_mix}) {
            if (mode != mtlx::default_layering_mode())
                layering_modes.push_back(mode);
        }
        for (mtlx::LayeringMode mode : layering_modes) {
            CAPTURE(mode);
            auto registry = make_ref<MaterialAOVRegistry>();
            registry->register_output("metallicity_output", sgl::DataType::float32, 1);
            mtlx::CodeGenDesc desc;
            desc.document = document;
            desc.node_name = "stormtrooper_dust_surface";
            if (mode != mtlx::default_layering_mode())
                desc.layering_mode = mode;
            desc.material_aov_registry = registry.get();

            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            const auto candidates = registry->outputs();
            REQUIRE_EQ(candidates.size(), 5u);
            CHECK_EQ(candidates[1].name, "dust_mask_output");
            CHECK_EQ(candidates[1].type, sgl::DataType::float32);
            CHECK_EQ(candidates[1].component_count, 1u);
            CHECK_EQ(candidates[0].name, "metallicity_output");
            REQUIRE_EQ(result->material_aov_outputs.size(), 5u);
            CHECK_EQ(result->material_aov_outputs[0].id, 1u);
            CHECK_EQ(result->material_aov_outputs[1].id, 0u);
            CHECK_FALSE(contains(result->module_source, "mtlx_output_"));
            CHECK_EQ(count_occurrences(result->module_source, "write_material_aovs"), 1u);
            const auto writer = result->module_source.find("public override void write_material_aovs_impl<");
            REQUIRE_NE(writer, std::string::npos);
            CHECK_LT(result->module_source.find(" : IMaterial\n"), writer);
            const auto guard = result->module_source.find("if (!(Writer::generic_data_enabled", writer);
            CHECK_LT(guard, result->module_source.find("data_buffer.load<", writer));
            CHECK_LT(guard, result->module_source.find("let mtlx_graph = Graph(", writer));
            CHECK(contains(result->module_source, "Writer::generic_data_enabled(0u)"));
            CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 1);
            CHECK(contains(result->module_source, "Writer::generic_data_enabled(1u)"));
            CHECK(contains(
                result->module_source,
                "Writer::generic_data(coord, 1u, float4(mtlx_graph.dust_mask_output, 0.0, 0.0, 0.0));"
            ));
            for (uint32_t component_count : {2u, 3u, 4u}) {
                CAPTURE(component_count);
                const auto& candidate = candidates[component_count];
                CHECK_EQ(candidate.name, fmt::format("mask{}", component_count));
                CHECK_EQ(candidate.type, sgl::DataType::float32);
                CHECK_EQ(candidate.component_count, component_count);
                const auto& output = result->material_aov_outputs[component_count];
                CHECK_EQ(output.id, component_count);
            }
            CHECK(
                contains(result->module_source, "Writer::generic_data(coord, 2u, float4(mtlx_graph.mask2, 0.0, 0.0));")
            );
            CHECK(contains(result->module_source, "Writer::generic_data(coord, 3u, float4(mtlx_graph.mask3, 0.0));"));
            CHECK(contains(result->module_source, "Writer::generic_data(coord, 4u, float4(mtlx_graph.mask4));"));
        }
    }

    SUBCASE("independent AOV-only work follows selection")
    {
        const std::filesystem::path document
            = testing::project_directory() / "tests/data/materialx/material_aov_independent_output.mtlx";
        auto generate = [&](bool select_dust)
        {
            auto registry = make_ref<MaterialAOVRegistry>();
            mtlx::CodeGenDesc desc;
            desc.document = document;
            desc.node_name = "surface_output";
            if (!select_dust)
                desc.additional_output_names.clear();
            desc.material_aov_registry = registry.get();
            auto result = mtlx::CodeGen::generate(desc);
            CHECK_EQ(registry->outputs().size(), select_dust ? 1u : 0u);
            return result;
        };

        auto unselected = generate(false);
        REQUIRE(unselected);
        CHECK(unselected->material_aov_outputs.empty());
        CHECK_FALSE(contains(unselected->module_source, "sample_texture(dust_image_file"));
        CHECK_FALSE(contains(unselected->module_source, "float dust_mask_output = "));

        auto selected = generate(true);
        REQUIRE(selected);
        REQUIRE_EQ(selected->material_aov_outputs.size(), 1u);
        CHECK_EQ(selected->material_aov_outputs[0].id, 0u);
        CHECK_EQ(count_occurrences(selected->module_source, "sample_texture(dust_image_file"), 1u);
        CHECK_EQ(count_occurrences(selected->module_source, "Writer::generic_data(coord, 0u"), 1u);
    }

    SUBCASE("same local name in independent graphs keeps distinct inputs")
    {
        auto registry = make_ref<MaterialAOVRegistry>();
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <nodegraph name="NGmain">
    <constant name="v" type="float"><input name="value" type="float" value="0.2" /></constant>
    <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF"><input name="weight" type="float" nodename="v" /></oren_nayar_diffuse_bsdf>
    <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
    <output name="out" type="surfaceshader" nodename="surface" />
  </nodegraph>
  <nodegraph name="NGaov">
    <constant name="v" type="float"><input name="value" type="float" value="0.8" /></constant>
    <output name="out" type="float" nodename="v" />
  </nodegraph>
  <dot name="aov" type="float"><input name="in" type="float" nodegraph="NGaov" output="out" /></dot>
  <output name="mask" type="float" nodename="aov" />
</materialx>)");
        desc.node_name = "NGmain/out";
        desc.material_aov_registry = registry.get();
        desc.editable_param_names = {"NGmain/v/value", "NGaov/v/value"};
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        REQUIRE_EQ(result->all_material_params.m_editable_count, 2);
        std::unordered_set<std::string> bindings;
        for (const auto& param : result->all_material_params.m_params) {
            if (!param.is_editable)
                continue;
            REQUIRE(param.extract_value<float>());
            CHECK_EQ(*param.extract_value<float>(), param.param_name == "NGmain/v/value" ? 0.2f : 0.8f);
            bindings.insert(param.binding_name);
        }
        CHECK_EQ(bindings.size(), 2);
        desc.editable_param_names = {"v_value"};
        CHECK_THROWS_WITH_AS(mtlx::CodeGen::generate(desc), doctest::Contains("ambiguous"), std::runtime_error);
    }

    SUBCASE("independent outputs support a node-shaped primary root")
    {
        auto registry = make_ref<MaterialAOVRegistry>();
        mtlx::CodeGenDesc desc;
        desc.document = testing::project_directory() / "tests/data/materialx/material_aov_independent_output.mtlx";
        desc.node_name = "surface";
        desc.layering_mode = mtlx::LayeringMode::bsdf_mix;
        desc.material_aov_registry = registry.get();
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(count_occurrences(result->module_source, "sample_texture(dust_image_file"), 1u);
    }

    SUBCASE("patterns select candidates before ID resolution")
    {
        auto registry = make_ref<MaterialAOVRegistry>();
        mtlx::CodeGenDesc desc;
        desc.document = testing::project_directory() / "tests/data/materialx/material_aov_independent_output.mtlx";
        desc.node_name = "surface_output";
        desc.additional_output_names = {"dust_*"};
        desc.material_aov_registry = registry.get();
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(contains(result->module_source, "Writer::generic_data(coord, 0u"));
        desc.additional_output_names.clear();
        result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(result->material_aov_outputs.empty());
        CHECK_FALSE(contains(result->module_source, "write_material_aovs"));
    }

    SUBCASE("duplicate authored candidate names are rejected")
    {
        auto registry = make_ref<MaterialAOVRegistry>();
        std::string document
            = read_text_file(testing::project_directory() / "tests/data/materialx/additional_graph_outputs.mtlx");
        const size_t document_end = document.rfind("</materialx>");
        REQUIRE(document_end != std::string::npos);
        document.insert(
            document_end,
            R"(<output name="dust_mask_output" type="float" nodename="stormtrooper_dust_surface" output="dust_mask_output" />
)"
        );

        mtlx::CodeGenDesc desc;
        desc.document = std::move(document);
        desc.node_name = "stormtrooper_dust_surface";
        desc.layering_mode = mtlx::LayeringMode::bsdf_mix;
        desc.material_aov_registry = registry.get();
        CHECK_THROWS_WITH(
            mtlx::CodeGen::generate(desc),
            doctest::Contains("material AOV candidate 'dust_mask_output' was declared more than once")
        );
    }
}


TEST_CASE_GPU_NIGHTLY("scene materialx reconciles shared microfacet specializations")
{
    // Only the Slang frontend is tested here, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    const std::vector modes = {
        mtlx::LayeringMode::bsdf_mix,
    };
    for (mtlx::LayeringMode mode : modes) {
        CAPTURE(mode);
        mtlx::CodeGenDesc desc;
        desc.document = testing::project_directory() / "tests/data/materialx/shared_microfacet_specializations.mtlx";
        desc.node_name = "SharedMicrofacetSpecializations";
        desc.layering_mode = mode;
        desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::all;

        // One MaterialX compound function serves both call sites. Their R and
        // T scatter modes must reconcile to a correct common specialization.
        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 2);
        CHECK(ctx.device->load_module_from_source(result->module_name, result->module_source));
    }
}

TEST_CASE_NIGHTLY("scene materialx preserves producer identity through recursive closure reuse")
{
    for (mtlx::LayeringMode mode : {mtlx::LayeringMode::closure_tree}) {
        CAPTURE(mode);
        mtlx::CodeGenDesc desc;
        desc.document = recursive_closure_dag(4);
        desc.node_name = "M";
        desc.layering_mode = mode;
        desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::none;

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(result->codegen_metadata.get<int>("layering_bsdf_count"), 5);

        if (mode == mtlx::LayeringMode::closure_tree) {
            CHECK_FALSE(contains(result->module_source, "leaf_visit_to_bsdf_index["));
            for (size_t index = 0; index < 5; ++index) {
                const std::string prefix = fmt::format("typealias Bsdf{} = MxFramedWeightedBSDF<", index);
                const size_t begin = result->module_source.find(prefix);
                REQUIRE(begin != std::string::npos);
                const size_t end = result->module_source.find('\n', begin);
                REQUIRE(end != std::string::npos);
                const std::string_view declaration(result->module_source.data() + begin, end - begin);
                CHECK(declaration.ends_with(fmt::format(", {}>;", index)));
            }
        }
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx retroreflective microfacet nodes compile")
{
    if (ctx.device->type() == sgl::DeviceType::cuda) {
        MESSAGE("Skipping MaterialX CUDA test because MaterialXMaterial is disabled on CUDA.");
        return;
    }

    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("mtlx_buffer", retroreflective_microfacet_shader());
    props.set("mtlx_node_name", std::string("M"));

    scene->create_material("MaterialXMaterial", props);
    scene->update();
}

TEST_CASE_NIGHTLY("scene materialx emits production geometry accessor reads")
{
    SUBCASE("independent additional output materializes its implicit UVs")
    {
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
  <noise2d name="mask" type="float" />
  <output name="mask_output" type="float" nodename="mask" />
</materialx>)");
        desc.node_name = "surface";
        desc.additional_output_names = {"mask_output"};

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(contains(result->module_source, "float mask_output = "));
        CHECK(contains(result->module_source, "__mtlx_static_pruning_defaultgeom_mask_texcoord"));
    }

    SUBCASE("blind names and fixed UVs use the production accessor")
    {
        mtlx::CodeGenDesc desc;
        desc.document = provider_geomprop_shader();
        desc.node_name = "M";
        std::vector<std::string> reserved_names;
        desc.reserve_geometry_attribute_id = [&](std::string_view name)
        {
            reserved_names.emplace_back(name);
            return 77u;
        };

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(reserved_names, std::vector<std::string>{"Pref"});
        CHECK(contains(result->module_source, "si.get_geomprop_accessor().get_float3(77u, "));
        CHECK(contains(result->module_source, "si.get_geomprop_accessor().get_additional_texcoord(1u"));
        CHECK_FALSE(contains(result->module_source, "GeomPropProvider"));
        CHECK_FALSE(contains(result->module_source, "geomprop_available"));
    }

    SUBCASE("missing geometry color is black rather than UV data")
    {
        mtlx::CodeGenDesc desc;
        desc.document = missing_geomcolor_shader();
        desc.node_name = "out";

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(contains(result->module_source, "si.get_geomprop_accessor().get_color(0u"));
        CHECK_FALSE(contains(result->module_source, " = float3(si.uv, 0.0);"));
    }

    SUBCASE("standard tokens and direct nodes use the same routes")
    {
        mtlx::CodeGenDesc desc;
        desc.document = geometry_route_shader();
        desc.node_name = "M";
        desc.reserve_geometry_attribute_id = [](std::string_view)
        {
            FAIL("standard geometry tokens must not reserve blind IDs");
            return 0u;
        };

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        const auto has_two = [&](std::string_view text)
        {
            const size_t first = result->module_source.find(text);
            return first != std::string::npos
                && result->module_source.find(text, first + text.size()) != std::string::npos;
        };
        CHECK(has_two("si.get_geomprop_accessor().get_additional_texcoord(1u"));
        CHECK(has_two("si.get_geomprop_accessor().get_color(0u"));
        CHECK(has_two("mx_viewdirection_vector3(si)"));
    }

    SUBCASE("view direction uses the same space conversion for direct and alias reads")
    {
        // Keep world-only reads separate so transformed reads cannot mask an
        // unnecessary accessor. Within each group, retain and check every node.
        for (const bool world : {true, false}) {
            CAPTURE(world);
            std::string document = R"(<materialx version="1.39">
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
)";
            std::vector<std::pair<std::string, std::string>> expected_assignments;
            for (const std::string space : (world ? std::array{"", "world"} : std::array{"object", "model"})) {
                const std::string space_input
                    = space.empty() ? "" : "<input name=\"space\" type=\"string\" value=\"" + space + "\" />";
                const std::string space_attribute = space.empty() ? "" : " space=\"" + space + "\"";
                const std::string expression = world
                    ? "mx_viewdirection_vector3(si)"
                    : "normalize(mul(si.get_geomprop_accessor().get_space_transform(GeomPropSpace::" + space
                        + ", GeomPropSpace::world), float4(mx_viewdirection_vector3(si), 0.0)).xyz)";
                for (const bool alias : {false, true}) {
                    const std::string name
                        = "view_" + (space.empty() ? "default" : space) + (alias ? "_alias" : "_direct");
                    if (alias) {
                        document += fmt::format(
                            R"(<geompropdef name="{0}_def" type="vector3" geomprop="viewdirection"{1} />
<geompropvalue name="{0}" type="vector3"><input name="geomprop" type="string" value="{0}_def" /></geompropvalue>)",
                            name,
                            space_attribute
                        );
                    } else {
                        document += fmt::format(
                            R"(<viewdirection name="{}" type="vector3">{}</viewdirection>)",
                            name,
                            space_input
                        );
                    }
                    document += fmt::format(R"(<output name="out_{0}" type="vector3" nodename="{0}" />)", name);
                    expected_assignments.emplace_back(name, expression);
                }
            }
            document += "</materialx>";
            mtlx::CodeGenDesc desc;
            desc.document = std::move(document);
            desc.node_name = "surface";
            auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(contains(result->module_source, "si.get_geomprop_accessor()"), !world);
            for (const auto& [name, expression] : expected_assignments) {
                CAPTURE(name);
                CHECK(contains(result->module_source, " out_" + name + " = "));
                CHECK(contains(result->module_source, name + "_out = " + expression + ";"));
            }
        }
    }

    SUBCASE("unsupported fixed indices use defaults without accessor work")
    {
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <texcoord name="uv8" type="vector2"><input name="index" type="integer" value="8" /></texcoord>
  <output name="out" type="vector2" nodename="uv8" />
</materialx>)");
        desc.node_name = "out";

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_FALSE(contains(result->module_source, "get_additional_texcoord"));
        CHECK_FALSE(contains(result->module_source, "si.get_geomprop_accessor()"));
        CHECK_FALSE(result->uses_geometry_properties);
    }

    SUBCASE("world-only geometry reads do not acquire an accessor")
    {
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <position name="position" type="vector3"><input name="space" type="string" value="world" /></position>
  <output name="out" type="vector3" nodename="position" />
</materialx>)");
        desc.node_name = "out";

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK(contains(result->module_source, "si.position_ws"));
        CHECK_FALSE(contains(result->module_source, "si.get_geomprop_accessor()"));
    }

    SUBCASE("statically pruned blind reads leave no reservation or accessor")
    {
        mtlx::CodeGenDesc desc;
        desc.document = std::string(R"(<materialx version="1.39">
  <geompropvalue name="pref" type="color3"><input name="geomprop" type="string" value="Pref" /></geompropvalue>
  <oren_nayar_diffuse_bsdf name="diffuse" type="BSDF">
    <input name="weight" type="float" value="0" />
    <input name="color" type="color3" nodename="pref" />
  </oren_nayar_diffuse_bsdf>
  <surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
  <surfacematerial name="M" type="material"><input name="surfaceshader" type="surfaceshader" nodename="surface" /></surfacematerial>
</materialx>)");
        desc.node_name = "M";
        size_t reservation_count = 0;
        desc.reserve_geometry_attribute_id = [&](std::string_view)
        {
            ++reservation_count;
            return 77u;
        };

        auto result = mtlx::CodeGen::generate(desc);
        REQUIRE(result);
        CHECK_EQ(reservation_count, 0);
        CHECK_FALSE(contains(result->module_source, "si.get_geomprop_accessor()"));
        CHECK_FALSE(result->uses_geometry_properties);
    }
}

TEST_CASE_GPU_NIGHTLY("scene materialx binds blind names to its Scene registry")
{
    // This checks API-independent host behavior, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    auto scene = Scene::create(ref(ctx.device));
    Properties props;
    props.set("mtlx_buffer", provider_geomprop_shader());
    props.set("mtlx_node_name", std::string("M"));
    scene->create_material("MaterialXMaterial", props);
    // Material construction reserves the name; shader loading and LUT uploads are not part of this check.
    CHECK(scene->_get_geometry_attribute_id(MeshAttributeSemantic::unknown, 0, "Pref").has_value());
}

TEST_CASE_NIGHTLY("scene materialx rejects unsupported geometry property contracts")
{
    struct Case {
        const char* document;
        const char* diagnostic;
    };
    const std::array cases = {
        Case{
            R"(<materialx version="1.39">
  <geompropdef name="P" type="vector3" uniform="true" />
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="P" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "is uniform"
        },
        Case{
            R"(<materialx version="1.39">
  <geompropdef name="P" type="vector2" />
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="P" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "has type 'vector2'"
        },
        Case{
            R"(<materialx version="1.39">
  <geompropdef name="P" type="vector3" unit="meter" />
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="P" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "uses unit metadata"
        },
        Case{
            R"(<materialx version="1.39">
  <geompropdef name="P" type="vector3" geomprop="unsupported" />
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="P" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "unsupported geomprop token"
        },
        Case{
            R"(<materialx version="1.39">
  <geompropdef name="V" type="vector3" geomprop="viewdirection" space="camera" />
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="V" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "unsupported space 'camera'"
        },
        Case{
            R"(<materialx version="1.39">
  <viewdirection name="read" type="vector3"><input name="space" type="string" value="camera" /></viewdirection>
  <output name="out" type="vector3" nodename="read" />
</materialx>)",
            "unsupported space 'camera'"
        },
        Case{
            R"(<materialx version="1.39">
  <add name="index" type="integer">
    <input name="in1" type="integer" value="0" />
    <input name="in2" type="integer" value="1" />
  </add>
  <texcoord name="read" type="vector2"><input name="index" type="integer" nodename="index" /></texcoord>
  <output name="out" type="vector2" nodename="read" />
</materialx>)",
            "must be a compile-time value"
        },
        Case{
            R"(<materialx version="1.39">
  <geompropvalueuniform name="read" type="string">
    <input name="geomprop" type="string" value="asset_name" />
  </geompropvalueuniform>
  <output name="out" type="string" nodename="read" />
</materialx>)",
            "geompropvalueuniform node 'read' is unsupported"
        },
    };

    for (const Case& test : cases) {
        CAPTURE(test.diagnostic);
        mtlx::CodeGenDesc desc;
        desc.document = std::string(test.document);
        desc.node_name = "out";
        desc.optimize_graph_flags = mtlx::OptimizeGraphFlags::none;
        desc.reserve_geometry_attribute_id = [](std::string_view)
        {
            return 77u;
        };
        CHECK_THROWS_WITH_AS(mtlx::CodeGen::generate(desc), doctest::Contains(test.diagnostic), std::exception);
    }

    mtlx::CodeGenDesc missing_resolver;
    missing_resolver.document = std::string(R"(<materialx version="1.39">
  <geompropvalue name="read" type="vector3"><input name="geomprop" type="string" value="Pref" /></geompropvalue>
  <output name="out" type="vector3" nodename="read" />
</materialx>)");
    missing_resolver.node_name = "out";
    CHECK_THROWS_WITH(
        mtlx::CodeGen::generate(missing_resolver),
        doctest::Contains("requires a Scene geometry-attribute ID resolver")
    );

    SUBCASE("unused uniform geometry declarations do not affect codegen")
    {
        mtlx::CodeGenDesc desc;
        const std::string graph = R"(
  <constant name="value" type="float"><input name="value" type="float" value="0.25" /></constant>
  <output name="out" type="float" nodename="value" />
</materialx>)";
        desc.document = std::string("<materialx version=\"1.39\">") + graph;
        desc.node_name = "out";
        size_t reservations = 0;
        desc.reserve_geometry_attribute_id = [&](std::string_view)
        {
            ++reservations;
            return 77u;
        };
        auto baseline = mtlx::CodeGen::generate(desc);
        REQUIRE(baseline);
        desc.document = std::string(R"(<materialx version="1.39">
  <geompropdef name="unused" type="float" uniform="true" />
  <geompropvalue name="unused_read" type="float"><input name="geomprop" type="string" value="unused" /></geompropvalue>
)") + graph;
        auto with_unused = mtlx::CodeGen::generate(desc);
        REQUIRE(with_unused);
        CHECK_EQ(reservations, 0);
        CHECK_EQ(with_unused->module_source, baseline->module_source);
        CHECK_FALSE(contains(with_unused->module_source, "si.get_geomprop_accessor()"));
    }
}

TEST_CASE_NIGHTLY("scene materialx OpenPBR uses generated source without the legacy shortcut")
{
    mtlx::CodeGenDesc desc;
    desc.document = std::string(R"(<materialx version="1.39" colorspace="lin_rec709">
  <open_pbr_surface name="openpbr1" type="surfaceshader">
    <input name="base_color" type="color3" value="0.1, 0.2, 0.3" />
    <input name="thin_film_weight" type="float" value="1.0" />
    <input name="geometry_thin_walled" type="boolean" value="true" />
  </open_pbr_surface>
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="openpbr1" />
  </surfacematerial>
</materialx>)");
    desc.node_name = "M";
    auto result = mtlx::CodeGen::generate(desc);
    REQUIRE(result);
    REQUIRE(!result->module_source.empty());
    CHECK_FALSE(contains(result->module_source, "OpenPBRMaterialInstance openpbr_instance"));
}

TEST_CASE_GPU_NIGHTLY("scene materialx 1.39 static closure pruning property is removed")
{
    // This checks API-independent host behavior, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("mtlx_buffer", std::string(R"(<materialx version="1.39">
  <standard_surface name="surface1" type="surfaceshader" />
  <surfacematerial name="M" type="material">
    <input name="surfaceshader" type="surfaceshader" nodename="surface1" />
  </surfacematerial>
</materialx>)"));
    props.set("mtlx_node_name", std::string("M"));
    props.set("mtlx_static_closure_pruning", std::string("on"));

    bool rejected = false;
    try {
        scene->create_material("MaterialXMaterial", props);
    } catch (const std::exception& e) {
        rejected = true;
        CHECK(contains(e.what(), "mtlx_static_closure_pruning"));
        CHECK(contains(e.what(), "mtlx_optimize_graph::closure_pruning"));
    }
    CHECK(rejected);
}

TEST_CASE_GPU_NIGHTLY("scene materialx 1.39 reflected property changes regenerate shader")
{
    // This checks API-independent host behavior, so one device is sufficient.
    if (ctx.device->type() != testing::DEFAULT_DEVICE_TYPES.front())
        return;
    // Exercise the C++ reflected setter and its invalidation once. GPU backend
    // compilation is covered independently by the material integration tests.
    auto scene = Scene::create(ref(ctx.device));
    const std::filesystem::path initial_shader_path
        = testing::get_case_temp_directory() / "materialx139_property_initial.slang";
    const std::filesystem::path updated_shader_path
        = testing::get_case_temp_directory() / "materialx139_property_updated.slang";

    Properties props;
    props.set("mtlx_buffer", mtlx_layered_property_update_shader());
    props.set("mtlx_node_name", std::string("M"));
    props.set("mtlx_layering_mode", mtlx::LayeringMode::closure_tree);
    props.set("debug_write_shader_path", initial_shader_path.string());

    auto* material = checked_cast<MaterialXMaterial*>(scene->create_material("MaterialXMaterial", props));
    scene->update();

    const std::string initial_shader = read_text_file(initial_shader_path);
    REQUIRE(!initial_shader.empty());
    CHECK_FALSE(contains(initial_shader, "FlatRootBSDF"));

    const auto& desc = material->class_descriptor();
    const auto* debug_shader_path_property = desc.find_property("debug_write_shader_path");
    const auto* layering_property = desc.find_property("mtlx_layering_mode");
    REQUIRE(debug_shader_path_property);
    REQUIRE(layering_property);
    debug_shader_path_property->set<std::string>(material, updated_shader_path.string());
    layering_property->set<mtlx::LayeringMode>(material, mtlx::LayeringMode::bsdf_mix);
    scene->update();

    const std::string updated_shader = read_text_file(updated_shader_path);
    REQUIRE(!updated_shader.empty());
    CHECK(contains(updated_shader, "FlatRootBSDF"));
    CHECK_NE(initial_shader, updated_shader);
}

TEST_CASE_GPU_NIGHTLY("scene unlit material ABI and opacity")
{
    auto scene = Scene::create(ref(ctx.device));

    Properties props;
    props.set("alpha_mode", AlphaMode::mask);
    props.set("alpha_factor", 0.6f);
    props.set("alpha_cutoff", 0.4f);
    props.set("base_color_factor", float3(1.5f, 2.f, 4.f));
    props.set("double_sided", true);

    auto* material = checked_cast<UnlitMaterial*>(scene->create_material("UnlitMaterial", props));
    scene->update();

    CHECK_EQ(material->flags(), shared::MaterialFlags::unlit | shared::MaterialFlags::two_sided);
    CHECK_EQ(material->_base_color_factor(), float3(1.5f, 2.f, 4.f));
    CHECK(material->_double_sided());

    const Material::OpacityDesc opacity = material->opacity_desc();
    CHECK_EQ(opacity.flags, shared::OpacityFlags::enabled | shared::OpacityFlags::use_threshold);
    CHECK_EQ(opacity.texture_channel, 3u);
    CHECK_EQ(opacity.factor, doctest::Approx(0.6f));
    CHECK_EQ(opacity.threshold, doctest::Approx(0.4f));

    auto requirements = scene->requirements();
    CHECK(!requirements->modules.empty());
}

TEST_CASE_GPU_NIGHTLY("scene emission-only material ABI")
{
    auto scene = Scene::create(ref(ctx.device));

    const ref<sgl::Texture> emission_texture = ctx.device->create_texture({
        .format = sgl::Format::rgba8_unorm_srgb,
        .width = 1,
        .height = 1,
        .usage = sgl::TextureUsage::shader_resource,
    });

    Properties props;
    props.set("emission_color", float3(0.25f, 0.5f, 1.f));
    props.set("emission_color_texture", emission_texture);
    props.set("enable_color_temperature", true);
    props.set("color_temperature", 3000.f);
    props.set("emission_luminance", 4.f);

    auto* material = checked_cast<EmissionOnlyMaterial*>(scene->create_material("EmissionOnlyMaterial", props));
    scene->update();

    CHECK_EQ(material->flags(), shared::MaterialFlags::none);
    CHECK(material->enable_color_temperature());
    CHECK_EQ(material->color_temperature(), 3000.f);
    CHECK(!scene->requirements()->modules.empty());
}

TEST_CASE_NIGHTLY("MDL material AOV compilation")
{
    auto& context = MDLContext::get();
    const std::filesystem::path library = testing::project_directory() / "data/assets/mdl_sdk_examples";
    MDLSearchPathScope scoped_search_path(context, {&library, 1});

    MDLContext::CompileRequest request;
    request.qualified_material_name = "::tutorials_aov::green";
    request.target_material_type = "::tutorials_aov::aov_material";
    const auto standalone = context.compile(request);
    INFO(standalone.error_msg);
    REQUIRE(standalone.success);
    CHECK(standalone.material_aov_outputs.empty());
    CHECK(standalone.code.find("material_aov_") == std::string::npos);

    auto registry = make_ref<MaterialAOVRegistry>();
    request.material_aov_registry = registry.get();
    const auto registered = context.compile(request);
    INFO(registered.error_msg);
    REQUIRE(registered.success);
    REQUIRE_EQ(registered.material_aov_outputs.size(), 1u);
    const auto& output = registered.material_aov_outputs.front();
    CHECK_EQ(output.id, 0u);
    const auto declaration = registry->find_output("example");
    REQUIRE(declaration);
    CHECK_EQ(output.id, declaration->id);
    CHECK_EQ(output.component_count, 3u);
    CHECK(registered.code.find(output.callable_name + "(") != std::string::npos);

    // Same SDK graph, but a different scene assigns another callable name to this output.
    auto other_registry = make_ref<MaterialAOVRegistry>();
    other_registry->register_output("unrelated", sgl::DataType::float32, 1);
    request.material_aov_registry = other_registry.get();
    const auto other = context.compile(request);
    REQUIRE(other.success);
    REQUIRE_EQ(other.material_aov_outputs.size(), 1u);
    CHECK_EQ(other.material_aov_outputs.front().id, 1u);
    CHECK(other.code.find("material_aov_1(") != std::string::npos);
    request.material_aov_registry = registry.get();
    CHECK_EQ(context.compile(request).code, registered.code);
}

TEST_CASE("MDL target code cache")
{
    auto& context = MDLContext::get();
    const std::filesystem::path library = testing::project_directory() / "data/assets/mdl_sdk_examples";
    MDLSearchPathScope scoped_search_path(context, {&library, 1});
    MDLContext::CompileRequest request;
    request.use_class_compilation = true;
    request.module_source = R"(mdl 1.9;
import ::df::*;
import ::tex::*;
export material First(uniform color tint = color(0.2),
    uniform texture_2d image = texture_2d("/resources/example.png", gamma: tex::gamma_srgb))
    = material(surface: material_surface(scattering:
        df::diffuse_reflection_bsdf(tint: tint * tex::lookup_color(image, float2(0.5)))));
export material Second(uniform color tint = color(0.7),
    uniform texture_2d image = texture_2d("/resources/example_roughness.png", gamma: tex::gamma_linear))
    = material(surface: material_surface(scattering:
        df::diffuse_reflection_bsdf(tint: tint * tex::lookup_color(image, float2(0.5)))));
export material SameSides(uniform color tint = color(0.2))
    = let {
        material_surface side = material_surface(scattering: df::diffuse_reflection_bsdf(tint: tint));
    } in material(thin_walled: true, surface: side, backface: side);
)";
    auto compile = [&](const char* name)
    {
        request.qualified_material_name = std::string("::target_cache_test::") + name;
        auto result = context.compile(request);
        INFO(result.error_msg);
        REQUIRE(result.success);
        return result;
    };
    auto check_arguments = [](const MDLContext::CompileResult& result, float tint, const char* image, bool srgb)
    {
        CHECK_EQ(result.textures.size(), 1u); // Do not retain the first instance's unused default texture.
        REQUIRE_EQ(result.arg_block_segments.size(), 1u);
        const auto& data = result.arg_block_segments.front().data;
        float3 color;
        std::memcpy(&color, data.data() + result.arg_block_layout.at("tint").offset, sizeof(color));
        CHECK_EQ(color, float3(tint));
        uint32_t index;
        std::memcpy(&index, data.data() + result.arg_block_layout.at("image").offset, sizeof(index));
        REQUIRE(index > 0);
        REQUIRE(index <= result.textures.size());
        const auto& texture = result.textures[index - 1];
        CHECK_EQ(texture.source.path.filename().string(), image);
        CHECK_EQ(texture.source.srgb, srgb);
    };
    const auto first = compile("First");
    const auto second = compile("Second");
    CHECK_EQ(first.code, second.code);
    check_arguments(first, 0.2f, "example.png", true);
    check_arguments(second, 0.7f, "example_roughness.png", false);
    check_arguments(compile("First"), 0.2f, "example.png", true);

    // Backend options are not part of the SDK graph hash.
    request.options.enable_auxiliary = false;
    const auto no_auxiliary = compile("First");
    CHECK_NE(no_auxiliary.code, first.code);
    CHECK(no_auxiliary.code.find("surface_scattering_auxiliary(") == std::string::npos);
    request.options.enable_auxiliary = true;
    CHECK_EQ(compile("First").code, first.code);

    // Keep distinct backfaces, but do not generate duplicate functions for identical sides.
    CHECK(first.code.find("backface_scattering_evaluate(") != std::string::npos);
    const auto same_sides = compile("SameSides");
    CHECK(same_sides.code.find("surface_scattering_evaluate(") != std::string::npos);
    CHECK(same_sides.code.find("backface_scattering_evaluate(") == std::string::npos);
    CHECK(same_sides.code.find("backface_emission_intensity(") == std::string::npos);
}

TEST_CASE_NIGHTLY("MDL shared emission condition")
{
    auto& context = MDLContext::get();
    const std::filesystem::path library
        = testing::project_directory() / "tests/python/render/data/mdl_generated_material_pipeline";
    MDLSearchPathScope scoped_search_path(context, {&library, 1});

    MDLContext::CompileRequest request;
    request.qualified_material_name = "::emission::SharedConditionalSpatial";
    request.use_class_compilation = true;
    const auto result = context.compile(request);
    INFO(result.error_msg);
    REQUIRE(result.success);
    CHECK(result.potentially_emissive);
    REQUIRE_EQ(result.emission_zero_checks.size(), 1u);
    CHECK_FALSE(result.emission_zero_checks.front().condition_function.empty());
    CHECK_FALSE(result.emission_zero_checks.front().condition_value);
}

TEST_CASE_NIGHTLY("scene mdl geometry metadata")
{
    auto& context = MDLContext::get();
    const std::filesystem::path library
        = testing::project_directory() / "tests/python/render/data/mdl_generated_material_pipeline";
    MDLSearchPathScope scoped_search_path(context, {&library, 1});
    auto compile = [&](std::string_view name, bool class_compilation)
    {
        MDLContext::CompileRequest request;
        request.qualified_material_name = fmt::format("scene_data_metadata::{}", name);
        request.use_class_compilation = class_compilation;
        return context.compile(request);
    };
    for (bool class_compilation : {false, true}) {
        for (std::string_view name : {"Static", "Helper", "Uniform", "UnrelatedString"}) {
            const auto result = compile(name, class_compilation);
            INFO(result.error_msg);
            REQUIRE(result.success);
            CHECK(result.referenced_scene_data == std::vector<std::string>{"Pref"});
            CHECK(
                std::find(result.string_constants.begin(), result.string_constants.end(), "Pref")
                != result.string_constants.end()
            );
        }
        const auto no_read = compile("NoRead", class_compilation);
        REQUIRE(no_read.success);
        CHECK(no_read.referenced_scene_data.empty());
        const auto state = compile("State", class_compilation);
        REQUIRE(state.success);
        CHECK(state.referenced_scene_data.empty());
        CHECK(state.render_state_usage != 0);
        const auto presence = compile("Presence", class_compilation);
        REQUIRE(presence.success);
        CHECK(presence.referenced_scene_data == std::vector<std::string>{"other"});
    }
    const auto captured = compile("DynamicName", true);
    REQUIRE(captured.success);
    REQUIRE(captured.arg_block_layout.contains("name"));
    REQUIRE(captured.arg_block_segments.size() == 1);
    const auto& name_layout = captured.arg_block_layout.at("name");
    REQUIRE(name_layout.size == sizeof(uint32_t));
    REQUIRE(name_layout.offset + sizeof(uint32_t) <= captured.arg_block_segments[0].data.size());
    uint32_t name_id = 0;
    std::memcpy(&name_id, captured.arg_block_segments[0].data.data() + name_layout.offset, sizeof(name_id));
    REQUIRE(name_id > 0);
    REQUIRE(name_id < captured.string_constants.size());
    CHECK_EQ(captured.string_constants[name_id], "Pref");
    const auto selector = compile("Selector", true);
    REQUIRE(selector.success);
    CHECK(selector.referenced_scene_data == std::vector<std::string>{"Pref", "other"});
    CHECK(
        std::find(selector.string_constants.begin(), selector.string_constants.end(), "other")
        != selector.string_constants.end()
    );
    for (bool class_compilation : {false, true}) {
        const auto conservative = compile("Conservative", class_compilation);
        REQUIRE(conservative.success);
        CHECK(conservative.referenced_scene_data == std::vector<std::string>{"Pref", "unrelated_in_helper"});
    }
    for (std::string_view name : {"DynamicName", "Selector"}) {
        const auto folded = compile(name, false);
        INFO(folded.error_msg);
        REQUIRE(folded.success);
        CHECK(folded.referenced_scene_data == std::vector<std::string>{"Pref"});
    }
    MDLContext::CompileRequest learnable;
    learnable.qualified_material_name = "scene_data_metadata::UnrelatedString";
    learnable.use_class_compilation = true;
    learnable.options.enable_param_mollification = true;
    learnable.options.speculars_as_glossy = true;
    learnable.options.df_handle_slot_mode = 4;
    learnable.options.enable_tagged_df_properties = true;
    const auto modified = context.compile(learnable);
    INFO(modified.error_msg);
    REQUIRE(modified.success);
    CHECK(modified.referenced_scene_data == std::vector<std::string>{"Pref"});
    // Valid static-name control flow must keep compiling, even though the public SDK cannot expose its body.
    const auto control_flow = compile("ControlFlow", true);
    REQUIRE(control_flow.success);
    CHECK(control_flow.referenced_scene_data == std::vector<std::string>{"Pref"});
}

TEST_CASE_GPU_NIGHTLY("scene mdl")
{
    try {
        auto scene = Scene::create(ref(ctx.device));

        std::filesystem::path mdl_path = testing::project_directory() / "data/assets/mdl_sdk_examples";

        Properties props;
        props.set("mdl_library_path", mdl_path.string());
        props.set("mdl_material_name", "carbon_composite::carbon_composite");

        scene->create_material("MDLMaterial", props);
        scene->update();

        auto reqs = scene->requirements();
        CHECK(!reqs->modules.empty());
        CHECK(reqs->modules.front()->slang_module());
    } catch (const std::runtime_error& e) {
        std::string what = e.what();
        if (what.find("Failed to load the MDL SDK") != std::string::npos) {
            MESSAGE("Skipping scene mdl test due to MDL SDK initialization mismatch: " << what);
            return;
        }
        throw;
    }
}

TEST_CASE_GPU_NIGHTLY("scene mdl opacity classification and shader linkage")
{
    try {
        auto scene = Scene::create(ref(ctx.device));
        const std::filesystem::path mdl_path = testing::project_directory() / "data/assets/test_mdl_opacity";
        const std::filesystem::path shader_path = testing::get_case_temp_directory() / "mdl_opacity.slang";

        auto create_material = [&](std::string_view name, bool write_shader = false)
        {
            Properties props;
            props.set("mdl_library_path", mdl_path.string());
            props.set("mdl_material_name", fmt::format("opacity_test::{}", name));
            props.set("mdl_class_compilation", true);
            if (write_shader)
                props.set("debug_write_shader_path", shader_path.string());
            return scene->create_material<MDLMaterial>(props);
        };

        MDLMaterial* opaque = create_material("opaque");
        MDLMaterial* transmissive_opaque = create_material("transmissive_opaque");
        scene->update();
        CHECK(!scene->required_capabilities().has(SceneCapabilityFlags::opacity_evaluation));

        MDLMaterial* transparent = create_material("transparent");
        MDLMaterial* constant_half = create_material("constant_half");
        MDLMaterial* parameterized = create_material("parameterized", true);
        MDLMaterial* textured = create_material("textured");
        scene->update();

        CHECK_EQ(opaque->opacity_desc().flags, shared::OpacityFlags::none);
        CHECK_EQ(transmissive_opaque->opacity_desc().flags, shared::OpacityFlags::none);

        const Material::OpacityDesc transparent_desc = transparent->opacity_desc();
        CHECK_EQ(transparent_desc.flags, shared::OpacityFlags::enabled);
        CHECK_EQ(transparent_desc.factor, 0.f);

        const Material::OpacityDesc constant_half_desc = constant_half->opacity_desc();
        CHECK_EQ(constant_half_desc.flags, shared::OpacityFlags::enabled);
        CHECK_EQ(constant_half_desc.factor, 0.5f);

        const auto evaluated_flags = shared::OpacityFlags::enabled | shared::OpacityFlags::evaluate_material;
        CHECK_EQ(parameterized->opacity_desc().flags, evaluated_flags);
        CHECK_EQ(textured->opacity_desc().flags, evaluated_flags);
        CHECK_EQ(textured->build_texture_list().size(), 1);
        CHECK(scene->required_capabilities().has(SceneCapabilityFlags::opacity_evaluation));

        const std::string shader = read_text_file(shader_path);
        CHECK(contains(shader, "override float eval_opacity"));
        CHECK(contains(shader, "geometry_cutout_opacity(state)"));
        CHECK(parameterized->required_module());
        CHECK(parameterized->required_module()->slang_module());
    } catch (const std::runtime_error& e) {
        std::string what = e.what();
        if (what.find("Failed to load the MDL SDK") != std::string::npos) {
            MESSAGE("Skipping scene mdl opacity test due to MDL SDK initialization mismatch: " << what);
            return;
        }
        throw;
    }
}

TEST_CASE("materialx reports geometry capabilities from emitted operations")
{
    struct Case {
        const char* category;
        const char* type;
        const char* inputs;
        bool uses_geometry;
    };
    const std::array cases = {
        Case{"constant", "float", R"(<input name="value" type="float" value="0.5" />)", false},
        Case{"texcoord", "vector2", R"(<input name="index" type="integer" value="0" />)", false},
        Case{"texcoord", "vector2", R"(<input name="index" type="integer" value="1" />)", true},
        Case{"geomcolor", "color3", R"(<input name="index" type="integer" value="0" />)", true},
        Case{"position", "vector3", R"(<input name="space" type="string" value="world" />)", false},
        Case{"normal", "vector3", R"(<input name="space" type="string" value="world" />)", false},
        Case{"position", "vector3", R"(<input name="space" type="string" value="object" />)", true},
        Case{"normal", "vector3", R"(<input name="space" type="string" value="model" />)", true},
        Case{"geompropvalue", "vector3", R"(<input name="geomprop" type="string" value="Pref" />)", true},
        Case{"geompropvalue", "vector2", R"(<input name="geomprop" type="string" value="primary_uv" />)", false},
        Case{"geompropvalue", "vector2", R"(<input name="geomprop" type="string" value="secondary_uv" />)", true},
        Case{
            "transformpoint",
            "vector3",
            R"(<input name="fromspace" type="string" value="object" /><input name="tospace" type="string" value="world" />)",
            true
        },
    };
    // Exercise the default first, then each other supported representation without a GPU.
    std::vector<mtlx::LayeringMode> modes{mtlx::default_layering_mode()};
    for (const auto mode : {mtlx::LayeringMode::closure_tree, mtlx::LayeringMode::bsdf_mix}) {
        if (mode != modes.front())
            modes.push_back(mode);
    }
    for (const auto mode : modes) {
        for (const Case& test : cases) {
            CAPTURE(mode);
            CAPTURE(test.category);
            CAPTURE(test.inputs);
            mtlx::CodeGenDesc desc;
            desc.layering_mode = mode;
            desc.node_name = "surface";
            desc.document = fmt::format(
                R"(<materialx version="1.39">
<geompropdef name="primary_uv" type="vector2" geomprop="texcoord" index="0" />
<geompropdef name="secondary_uv" type="vector2" geomprop="texcoord" index="1" />
<oren_nayar_diffuse_bsdf name="diffuse" type="BSDF" />
<surface name="surface" type="surfaceshader"><input name="bsdf" type="BSDF" nodename="diffuse" /></surface>
<{0} name="probe" type="{1}">{2}</{0}>
<output name="out_probe" type="{1}" nodename="probe" />
</materialx>)",
                test.category,
                test.type,
                test.inputs
            );
            desc.reserve_geometry_attribute_id = [](std::string_view)
            {
                return 77u;
            };
            const auto result = mtlx::CodeGen::generate(desc);
            REQUIRE(result);
            CHECK_EQ(result->uses_geometry_properties, test.uses_geometry);
            mtlx::CodeGenCache cache;
            for (int pass = 0; pass < 2; ++pass) {
                const auto cached = mtlx::CodeGen::generate(desc, &cache);
                REQUIRE(cached);
                CHECK_EQ(cached->uses_geometry_properties, test.uses_geometry);
                CHECK_EQ(cached->module_source, result->module_source);
            }
            CHECK_EQ(cache.size(), 1);
        }
    }
}

TEST_SUITE_END();
