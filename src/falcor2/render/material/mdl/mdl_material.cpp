// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mdl_material.h"
#include "mdl_sdk_wrapper.h"

#include "falcor2/render/scene.h"

#include "falcor2/core/properties.h"
#include "falcor2/utils/string_utils.h"

#include <sgl/core/crypto.h>
#include <sgl/core/hash.h>
#include <sgl/device/sampler.h>

#include <algorithm>
#include <fstream>
#include <sstream>

namespace falcor {

namespace {
/// Enable MDL's use_renderer_adapt_normal option which allows individual normal mapped normals
/// to be adjusted instead of just the global shading normal.
const bool USE_RENDERER_ADAPT_NORMAL = true;

/// Material shader source code template.
extern std::string SHADER_SOURCE_TEMPLATE;

std::string material_aov_value(const MDLContext::CompileResult::MaterialAOVOutput& output)
{
    const std::string value = fmt::format("${{NAME}}_ns::{}(state)", output.callable_name);
    switch (output.component_count) {
    case 1:
        return fmt::format("float4({}, 0.0f, 0.0f, 0.0f)", value);
    case 2:
        return fmt::format("float4({}, 0.0f, 0.0f)", value);
    case 3:
        return fmt::format("float4({}, 0.0f)", value);
    case 4:
        return fmt::format("float4({})", value);
    default:
        FALCOR_UNREACHABLE();
    }
}

std::string emit_material_aov_writer(const MDLContext::CompileResult& result)
{
    if (result.material_aov_outputs.empty())
        return {};
    std::string source
        = R"(    public override void write_material_aovs_impl<Writer : IMaterialAOVWriter, LodSampler : ILodSampler>(
        uint2 coord, const SurfaceInteraction si, const LodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        if (!()";
    for (size_t i = 0; i < result.material_aov_outputs.size(); ++i) {
        if (i)
            source += " || ";
        source += fmt::format("Writer::generic_data_enabled({}u)", result.material_aov_outputs[i].id);
    }
    source += "))\n            return;\n        var state = init_shading_state(si, hints);\n";
    for (const MDLContext::CompileResult::MaterialAOVOutput& output : result.material_aov_outputs) {
        source += fmt::format("        if (Writer::generic_data_enabled({}u))\n", output.id);
        source += fmt::format(
            "            Writer::generic_data(coord, {0}u, {1});\n",
            output.id,
            material_aov_value(output)
        );
    }
    source += "    }\n";
    return source;
}

std::string emit_emission_predicate(const MDLContext::CompileResult& result)
{
    if (!result.potentially_emissive)
        return "        return false;\n";
    if (result.emission_zero_checks.empty())
        return "        return true;\n";
    std::string source = "        ${NAME}_ns::gMDLMaterialData = data;\n"
                         "        ${NAME}_ns::Shading_state_material state = {};\n";
    for (const auto& check : result.emission_zero_checks) {
        source += "        if (";
        if (!check.condition_function.empty()) {
            source += fmt::format(
                "${{NAME}}_ns::{}(state) == {} && ",
                check.condition_function,
                check.condition_value ? "true" : "false"
            );
        }
        source += fmt::format("all(${{NAME}}_ns::{}(state) == 0.0)) return false;\n", check.value_function);
    }
    return source + "        return true;\n";
}

} // namespace

MDLMaterial::MDLMaterial()
{
    m_material_properties.set_instance(this);
}

MDLMaterial::~MDLMaterial() { }

void MDLMaterial::set_properties(const Properties& props)
{
    // Deserialize static properties.
    class_descriptor().read_from_properties(this, props);

    // Recompile MDL if needed.
    if (m_require_codegen)
        run_codegen();

    // Deserialize material properties.
    m_material_properties.read_from_properties(props);
}

void MDLMaterial::on_load_resources()
{
    // Generate before enumerating the complete texture table.
    if (m_require_codegen) {
        // Regeneration must not reset edited parameters or texture paths.
        Properties previous_material_properties;
        m_material_properties.write_to_properties(previous_material_properties);
        run_codegen();
        m_material_properties.read_from_properties(previous_material_properties);
    }

    m_texture_table_needs_update = true;

    // Each entry owns its load source and any editable argument binding.
    for (size_t i = 0; i < m_texture_resources.size(); ++i) {
        auto& texture_resource = m_texture_resources[i];
        auto& source = texture_resource.source;
        if (texture_resource.parameter) {
            const auto& parameter = *texture_resource.parameter;
            const auto& path = m_material_properties.get_ref<std::string>(parameter.name);
            const bool srgb = m_material_properties.get<bool>(parameter.name + ":srgb");
            const auto& texture = m_material_properties.get_ref<ref<sgl::Texture>>(parameter.name + ":texture");
            const auto& sampler = m_material_properties.get_ref<ref<sgl::Sampler>>(parameter.name + ":sampler");

            // Match native materials: an explicit texture overrides the file path.
            // TextureManager reuses existing resources when only other parameters changed.
            if (texture) {
                texture_resource.texture_handle = m_scene->texture_manager()->register_texture({
                    .texture = texture,
                    .sampler = sampler,
                });
            } else if (!path.empty()) {
                texture_resource.texture_handle = m_scene->texture_manager()->load_texture({
                    .path = try_resolve(path),
                    .sampler = sampler,
                    .flags = srgb ? TextureLoadFlags::color : TextureLoadFlags::data,
                    .usage = sgl::TextureUsage::shader_resource,
                    .load_deferred = true,
                });
            } else {
                texture_resource.texture_handle = {};
            }

            // Empty defaults remain editable; no source image is needed during compilation.
            // Clearing both inputs restores texture_isvalid()'s zero resource ID.
            const uint32_t resource_id = texture || !path.empty() ? sgl::narrow_cast<uint32_t>(i + 1) : 0;
            set_material_data(mutable_data(), m_arg_block_offset + parameter.argument_offset, resource_id);
            continue;
        }
        if (texture_resource.texture_handle.is_valid())
            continue;

        // Match the resource shapes supported by MDLContext's extraction path.
        using TextureType = MDLContext::TextureSource::Type;
        switch (source.type) {
        case TextureType::Texture2D:
            if (source.path.empty())
                continue;
            texture_resource.texture_handle = m_scene->texture_manager()->load_texture({
                .path = source.path,
                .flags = source.srgb ? TextureLoadFlags::color : TextureLoadFlags::data,
                .usage = sgl::TextureUsage::shader_resource,
                .load_deferred = true,
            });
            break;
        case TextureType::TextureBSDFData: {
            sgl::SubresourceData subresource = {.data = source.data.data(), .size = source.data.size()};

            auto texture_3d = m_scene->device()->create_texture({
                .type = sgl::TextureType::texture_3d,
                .format = source.format,
                .width = source.width,
                .height = source.height,
                .depth = source.depth,
                .usage = sgl::TextureUsage::shader_resource,
                .data = {&subresource, 1},
            });

            texture_resource.texture_handle = m_scene->texture_manager()->register_texture({
                .texture = std::move(texture_3d),
            });
            break;
        }
        default:
            FALCOR_THROW("Unsupported MDL texture resource type");
        }
    }
}

void MDLMaterial::update(SceneUpdateContext&)
{
    // Write texture table only after on_load_resources() has run.
    if (m_texture_table_needs_update) {
        for (size_t i = 0; i < m_texture_resources.size(); ++i) {
            set_material_data(
                mutable_data(),
                m_texture_table_offset + i * sizeof(uint32_t),
                m_texture_resources[i].texture_handle.data()
            );
        }
        m_texture_table_needs_update = false;
    }

    /// Build the slang module, if it is missing, needs to be written or read from file.
    if (!m_slang_module || !m_debug_write_shader_path.empty() || !m_debug_load_shader_path.empty()) {
        /// We prepare the actual code, minus the type name, to allow efficient class compilation.
        std::string defines;
        defines += fmt::format("#define MDL_NUM_TEXTURE_RESULTS {}\n", m_request.options.num_texture_results);
        defines += fmt::format("#define MDL_NUM_TEXTURE_SPACES {}\n", m_request.options.num_texture_spaces);
        defines += fmt::format("#define MDL_DF_HANDLE_SLOT_MODE {}\n", m_request.options.df_handle_slot_mode);
        defines += fmt::format("#define MDL_HAS_SURFACE_EMISSION {}\n", m_result.has_surface_emission ? 1 : 0);
        defines += fmt::format("#define MDL_EMISSION_HANDLE_COUNT {}\n", m_result.surface_emission_handle_count);
        defines += fmt::format("#define MDL_HAS_SCENE_DATA {}\n", m_result.referenced_scene_data.empty() ? 0 : 1);
        // SDK state usage is coarse: texture-coordinate use requires all four supported slots.
        defines += fmt::format(
            "#define MDL_USES_TEXTURE_COORDINATES {}\n",
            (m_result.render_state_usage & mi::neuraylib::ITarget_code::SU_TEXTURE_COORDINATE) != 0 ? 1 : 0
        );
        defines += fmt::format(
            "#define MDL_USES_TRANSFORMS {}\n",
            (m_result.render_state_usage & mi::neuraylib::ITarget_code::SU_TRANSFORMS) != 0 ? 1 : 0
        );
        defines += fmt::format(
            "#define MDL_USE_RENDERER_ADAPT_NORMAL {}\n",
            m_request.options.use_renderer_adapt_normal ? "1" : "0"
        );

        std::string scene_data_cases;
        for (uint32_t string_id = 1; string_id < m_result.string_constants.size(); ++string_id) {
            const std::string binding = "mdl_scene_data:" + m_result.string_constants[string_id];
            if (m_material_properties.find_property(binding)) {
                const uint32_t attribute_id = m_material_properties.get<uint32_t>(binding);
                scene_data_cases += fmt::format("        case {}u: return {}u;\n", string_id, attribute_id);
            }
        }

        std::string source = replace_substrings(
            SHADER_SOURCE_TEMPLATE,
            std::map<std::string, std::string>({
                {"${MDL_DEFINES}", defines},
                {"${MDL_SCENE_DATA_CASES}", scene_data_cases},
                {"${MDL_CODE}", m_result.code},
                {"${MDL_WRITE_MATERIAL_AOVS}", emit_material_aov_writer(m_result)},
                {"${MDL_EMISSION_PREDICATE}", emit_emission_predicate(m_result)},
            })
        );

        // We hash the source code, with just name placeholders.
        // This way we can detect if MDL generated the same code for two different shaders,
        // even if we cannot detect actual class name.
        std::string source_hash = sgl::SHA1(source.data(), source.size()).hex_digest();
        /// Name of the Material and also the Module, it is "MDLMaterial_[hash]" with hash of the source,
        /// to avoid issues with determining the class name
        m_slang_type_name = "MDLMaterial_" + source_hash;
        source = replace_substrings(source, std::map<std::string, std::string>({{"${NAME}", m_slang_type_name}}));

        // If explicit debug_write_shader_path is provided, write shader to the given path
        if (!m_debug_write_shader_path.empty()) {
            auto filename = replace_substrings(
                m_debug_write_shader_path,
                std::map<std::string, std::string>({{"${NAME}", m_slang_type_name}})
            );
            std::ofstream ofs(filename);
            if (!ofs) {
                sgl::log_warn("Failed to write shader to: {}", filename);
            } else {
                ofs << source;
            }
        }

        // If explicit debug_load_shader_path is provided, replace the generate code with its content.
        if (!m_debug_load_shader_path.empty()) {
            std::ifstream ifs(m_debug_load_shader_path);
            if (!ifs) {
                FALCOR_THROW("Failed to read shader file: {}", m_debug_load_shader_path);
            }
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            source = buffer.str();
        }

        m_slang_module = m_scene->device()->load_module_from_source(m_slang_type_name, source);
    }

    if (m_result.arg_block_segments.size() > 0) {
        for (auto& it : m_result.arg_block_layout) {
            const std::string& arg_name = it.first;
            size_t arg_offset = m_arg_block_offset + it.second.offset;
            using ParamType = MDLContext::CompileResult::ArgumentLayout::Type;

            switch (it.second.type) {
            case ParamType::int_:
                write_mdl_material_property<int>(arg_name, arg_offset);
                break;
            case ParamType::bool_:
                write_mdl_material_property<bool>(arg_name, arg_offset);
                break;
            case ParamType::float_:
                write_mdl_material_property<float>(arg_name, arg_offset);
                break;
            case ParamType::float2_:
                write_mdl_material_property<float2>(arg_name, arg_offset);
                break;
            case ParamType::float3_:
                write_mdl_material_property<float3>(arg_name, arg_offset);
                break;
            case ParamType::float4_:
                write_mdl_material_property<float4>(arg_name, arg_offset);
                break;
            case ParamType::float4x4_:
            case ParamType::texture:
            case ParamType::unsupported:
                break;
            };
        }
    }
}

template<typename CursorT>
void MDLMaterial::write_to_cursor_impl(CursorT cursor) const
{
    auto data = cursor["data"];
    const auto data_buffer = data_ref();
    data["data"] = data_buffer.buffer;
    data["ro_data_offset"] = data_buffer.offset;
    data["arg_block_offset"] = data_buffer.offset + m_arg_block_offset;
    data["texture_table_offset"] = data_buffer.offset + m_texture_table_offset;
    data["surface_scatter_bsdf_count"] = m_result.surface_scatter_handle_count;
}

template void MDLMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void MDLMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

Material::OpacityDesc MDLMaterial::opacity_desc() const
{
    OpacityDesc result;

    if (m_result.cutout_opacity) {
        const float opacity = *m_result.cutout_opacity;
        if (opacity == 1.f)
            return result;

        result.flags = shared::OpacityFlags::enabled;
        result.factor = opacity;
        return result;
    } else {
        result.flags = shared::OpacityFlags::enabled;
    }

    result.flags |= shared::OpacityFlags::evaluate_material;
    return result;
}

SceneCapabilities MDLMaterial::required_capabilities() const
{
    // Texture-coordinate usage is SDK-wide, not per slot. The runtime populates secondary slots.
    const uint32_t geometry_state
        = mi::neuraylib::ITarget_code::SU_TRANSFORMS | mi::neuraylib::ITarget_code::SU_TEXTURE_COORDINATE;
    bool uses_geometry_properties
        = !m_result.referenced_scene_data.empty() || (m_result.render_state_usage & geometry_state) != 0;
    return material_capabilities(uses_geometry_properties);
}

ref<sgl::SlangModule> MDLMaterial::required_module() const
{
    return m_slang_module;
}

std::vector<TextureHandle> MDLMaterial::build_texture_list() const
{
    std::vector<TextureHandle> result;
    result.reserve(m_texture_resources.size());
    for (const auto& tex : m_texture_resources) {
        // Captured defaults and editable parameters can reference the same handle in different MDL slots.
        if (tex.texture_handle.is_valid()
            && std::find(result.begin(), result.end(), tex.texture_handle) == result.end())
            result.push_back(tex.texture_handle);
    }
    return result;
}

void MDLMaterial::require_codegen()
{
    mark_dirty(DirtyFlags::properties | DirtyFlags::resources | DirtyFlags::emission);
    m_require_codegen = true;
}

void MDLMaterial::run_codegen()
{
    if (!m_require_codegen)
        return;

    // Hash the properties influencing codegen to determine if we actually need to re-run.
    size_t codegen_properties_hash = sgl::hash(
        m_mdl_library_path,
        m_mdl_material_name,
        m_mdl_source,
        m_mdl_class_compilation,
        m_mdl_target_material_type,
        m_learnable
    );
    if (codegen_properties_hash == m_codegen_properties_hash) {
        m_require_codegen = false;
        return;
    }

    MDLContext& mdl_context = MDLContext::get();

    /// We are rebuilding the material.
    mark_dirty(DirtyFlags::resources | DirtyFlags::properties | DirtyFlags::emission);
    m_slang_module.reset();
    m_library_path_resolver = make_ref<AssetResolver>(m_mdl_library_path);

    FALCOR_CHECK(!m_mdl_material_name.empty(), "MDL material name must not be empty");

    // Compile material.
    m_request = {};
    m_request.qualified_material_name = m_mdl_material_name;
    m_request.module_source = m_mdl_source;
    m_request.instance_name = "";
    m_request.target_material_type = m_mdl_target_material_type;
    // m_request.options.opt_level = 1;
    m_request.options.use_renderer_adapt_normal = USE_RENDERER_ADAPT_NORMAL;
    m_request.use_class_compilation = m_mdl_class_compilation;

    m_request.material_aov_registry = m_scene->material_aov_registry();

    if (m_learnable) {
        // Learning relies on evaluating all BSDF components of the material separately (normal, roughness, tint, etc)
        // and feeding these as separate signals to the encoder network. MDL, however, aggregates the components of all
        // BSDFs with the same tag. For materials without tags (like the vMaterials) all BSDFs are aggregated into one.
        // By enabling 'enable_tagged_df_properties', Falcor scans the material and adds unique tags to each BSDF
        // component. Consequently the BSDFs in the MDL material are evaluated separately.
        //
        // Further, we enable mollifcation and replacing pure specular lobes with glossy to make learning of the lobes
        // easier.
        m_request.options.enable_param_mollification = true;
        m_request.options.speculars_as_glossy = true;
        m_request.options.df_handle_slot_mode = 4;
        m_request.options.enable_tagged_df_properties = true;
    } else {
        m_request.options.enable_param_mollification = false;
        m_request.options.speculars_as_glossy = false;
        m_request.options.df_handle_slot_mode = 0;
        m_request.options.enable_tagged_df_properties = false;
    }

    try {
        {
            MDLSearchPathScope search_paths(mdl_context, m_library_path_resolver->get_search_paths());
            m_result = mdl_context.compile(m_request);
        }
        if (!m_result.success)
            FALCOR_THROW("Failed to compile MDL material {}: {}", m_mdl_material_name, m_result.error_msg);
        if (m_result.surface_scatter_handle_count > 16)
            FALCOR_THROW(
                "MDL material exceeds supported df count {} : {}",
                m_mdl_material_name,
                m_result.surface_scatter_handle_count
            );

        m_arg_block_offset = (uint32_t)m_result.ro_data_segment_size();
        m_texture_table_offset = m_arg_block_offset + (uint32_t)m_result.arg_block_segment_size();

        // Convert SDK resources once, preserving their IDs. Runtime loading owns all source data.
        m_texture_resources.clear();
        m_texture_resources.reserve(m_result.textures.size());
        for (auto& texture : m_result.textures) {
            FALCOR_CHECK(
                texture.index == m_texture_resources.size() + 1,
                "MDL texture resource IDs must be consecutive"
            );
            m_texture_resources.push_back({.source = std::move(texture.source)});
        }

        /// Create default material properties.
        m_material_properties.clear();
        for (const auto& name : m_result.referenced_scene_data) {
            if (name.empty())
                continue;
            m_material_properties.add_property(
                "mdl_scene_data:" + name,
                m_scene->_reserve_geometry_attribute_id(name),
                false,
                "Geometry attribute ID for this MDL scene-data name. Defaults to the ordinary named attribute. "
                "An importer may override it with a fixed semantic ID. Changing it rebuilds the Slang wrapper.",
                reflection::on_change<MDLMaterial>(
                    [](MDLMaterial& self)
                    {
                        self.m_slang_module = nullptr;
                        self.mark_dirty(DirtyFlags::properties | DirtyFlags::emission);
                    }
                ),
                reflection::ui_group("Scene Data")
            );
        }
        if (m_result.arg_block_segments.size() > 0) {
            std::vector<uint8_t>& arg_block_data = m_result.arg_block_segments[0].data;

            auto on_change = reflection::on_change<MDLMaterial>(
                [](MDLMaterial& self)
                {
                    // We only need updating resources when textures change.
                    // We rely on on_load_resources() to be quick in case no work needs to be done.
                    self.mark_dirty(DirtyFlags::properties | DirtyFlags::resources | DirtyFlags::emission);
                }
            );
            auto ui_group = reflection::ui_group("Material Parameters");

            for (auto& it : m_result.arg_block_layout) {
                void* data = arg_block_data.data() + it.second.offset;
                using ParamType = MDLContext::CompileResult::ArgumentLayout::Type;

                switch (it.second.type) {
                case ParamType::int_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const int*>(data), false, on_change, ui_group);
                    break;
                case ParamType::bool_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const bool*>(data), false, on_change, ui_group);
                    break;
                case ParamType::float_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const float*>(data), false, on_change, ui_group);
                    break;
                case ParamType::float2_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const float2*>(data), false, on_change, ui_group);
                    break;
                case ParamType::float3_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const float3*>(data), false, on_change, ui_group);
                    break;
                case ParamType::float4_:
                    m_material_properties
                        .add_property(it.first, *reinterpret_cast<const float4*>(data), false, on_change, ui_group);
                    break;
                case ParamType::float4x4_:
                    break;
                case ParamType::texture: {
                    if (it.second.texture_type != MDLContext::TextureSource::Type::Texture2D)
                        break;
                    const uint32_t default_index = *reinterpret_cast<const uint32_t*>(data);
                    MDLContext::TextureSource source{.type = it.second.texture_type};
                    std::string default_path;
                    if (default_index > 0) {
                        source = m_texture_resources.at(default_index - 1).source;
                        default_path = source.path.string();
                    }
                    m_material_properties.add_property(it.first, default_path, false, on_change, ui_group);
                    m_material_properties.add_property(it.first + ":srgb", source.srgb, false, on_change, ui_group);
                    m_material_properties.add_property(
                        it.first + ":texture",
                        ref<sgl::Texture>{},
                        false,
                        "Optional GPU texture overriding the filename. Clear it to use the filename again. "
                        "Its format determines color-space decoding; :srgb only applies to file loading.",
                        on_change,
                        ui_group
                    );
                    m_material_properties.add_property(
                        it.first + ":sampler",
                        ref<sgl::Sampler>{},
                        false,
                        "Optional sampler for either a GPU texture or a filename. Null uses the scene default.",
                        on_change,
                        ui_group
                    );
                    m_texture_resources.push_back({
                        .source = std::move(source),
                        .parameter = TextureParameter{it.first, it.second.offset},
                    });
                    break;
                }
                case ParamType::unsupported:
                    break;
                };
            }
        }

        // Preserve the original default paths in properties before resolving the runtime sources.
        for (auto& texture : m_texture_resources) {
            if (!texture.source.path.empty())
                texture.source.path = try_resolve(texture.source.path);
        }

        // Include runtime parameter slots without changing the SDK's constant resource IDs.
        ensure_data_size(
            m_result.ro_data_segment_size() + m_result.arg_block_segment_size()
            + m_texture_resources.size() * sizeof(uint32_t)
        );
        if (!m_result.ro_data_segments.empty())
            set_material_data(
                mutable_data(),
                0,
                m_result.ro_data_segments[0].data.data(),
                m_result.ro_data_segment_size()
            );
        if (!m_result.arg_block_segments.empty())
            set_material_data(
                mutable_data(),
                m_arg_block_offset,
                m_result.arg_block_segments[0].data.data(),
                m_result.arg_block_segment_size()
            );

        m_codegen_properties_hash = codegen_properties_hash;
        m_require_codegen = false;
    } catch (...) {
        m_codegen_properties_hash = 0;
        throw;
    }
}

FALCOR_SCENE_REGISTER_MATERIAL(MDLMaterial);

namespace {
std::string SHADER_SOURCE_TEMPLATE = R"(
module ${NAME};

__exported import falcor2.render;
import falcor2.utils;

namespace ${NAME}_ns {

${MDL_DEFINES}

// SDK string IDs and Scene attribute IDs are distinct. UINT_MAX cannot address a Scene attribute.
uint mdl_scene_data_attribute_id(uint string_id)
{
    switch (string_id) {
${MDL_SCENE_DATA_CASES}
    default: return 0xffffffffu;
    }
}

#include "falcor2/render/materials/mdl/mdl_runtime.slangh"

${MDL_CODE}

#include "falcor2/render/materials/mdl/mdl_emission.slangh"

} // ${NAME}_ns

struct ${NAME}_Instance : IMaterialInstance
{
    MDLMaterialData data;
    ${NAME}_ns::Shading_state_material state;
    float3 ior1, ior2;
    float interior_ior;
    float3 emitted_radiance;

    override float get_interior_ior() { return interior_ior; }

    override bool is_emissive() { return any(emitted_radiance > 0.f); }
    override float3 eval_emissive_radiance() { return emitted_radiance; }

    override HomogeneousMediumProperties get_interior_medium()
    {
        // SDK expressions may take inout state; Falcor's material instance remains immutable.
        var state = this.state;
        ${NAME}_ns::gMDLMaterialData = data;
        HomogeneousMediumProperties result = {};
        result.sigma_a = ${NAME}_ns::volume_absorption_coefficient(state);
        result.sigma_s = ${NAME}_ns::volume_scattering_coefficient(state);
        return result;
    }

    BSDFEval eval_with_pdf<
        let Hints : BSDFEvalHints = BSDFEvalHints::both,
        S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo, inout S sg)
    {
        if (Hints == BSDFEvalHints::none)
            return BSDFEval();

        var state = this.state;
        ${NAME}_ns::gMDLMaterialData = data;

        BSDFEval result = {};
        if (is_set(Hints, BSDFEvalHints::eval)) {
            ${NAME}_ns::Bsdf_evaluate_data eval_data = {};
            eval_data.ior1 = ior1;     // IOR current medium
            eval_data.ior2 = ior2;     // IOR other side
            eval_data.k1 = si.wi_ws; // outgoing direction
            eval_data.k2 = wo;         // incoming direction
#if MDL_DF_HANDLE_SLOT_MODE == 0
            ${NAME}_ns::surface_scattering_evaluate(eval_data, state);
            result.value = eval_data.bsdf_diffuse + eval_data.bsdf_glossy;
#else
            uint surfaceScatterBsdfCount = ${NAME}_ns::gMDLMaterialData.surface_scatter_bsdf_count;
            uint offset = 0;
            for (; offset < surfaceScatterBsdfCount; offset += MDL_DF_HANDLE_SLOT_MODE) {
                eval_data.handle_offset = offset;
                ${NAME}_ns::surface_scattering_evaluate(eval_data, state);
                for (uint lobe = 0; (lobe < MDL_DF_HANDLE_SLOT_MODE) && ((offset + lobe) < surfaceScatterBsdfCount);
                     ++lobe) {
                    result.value += (eval_data.bsdf_diffuse[lobe] + eval_data.bsdf_glossy[lobe]);
                }
            }
#endif
            if (is_set(Hints, BSDFEvalHints::pdf))
                result.pdf = eval_data.pdf;
        } else {
            ${NAME}_ns::Bsdf_pdf_data pdf_data = {};
            pdf_data.ior1 = ior1;     // IOR current medium
            pdf_data.ior2 = ior2;     // IOR other side
            pdf_data.k1 = si.wi_ws; // outgoing direction
            pdf_data.k2 = wo;         // incoming direction
            ${NAME}_ns::surface_scattering_pdf(pdf_data, state);
            result.pdf = pdf_data.pdf;
        }
        return result;
    }

    bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out MaterialSample sample)
    {
        var state = this.state;
        ${NAME}_ns::Bsdf_sample_data sample_data = {};
        sample_data.ior1 = ior1;          // IOR current medium
        sample_data.ior2 = ior2;          // IOR other side
        sample_data.k1 = si.wi_ws;      // outgoing direction
        sample_data.xi = next_float4(sg); // pseudo-random sample numbers

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::surface_scattering_sample(sample_data, state);

        sample = {};
        sample.wo_ws = sample_data.k2;
        sample.weight = sample_data.bsdf_over_pdf;
        sample.pdf = sample_data.pdf;

        if (sample_data.event_type == BSDF_EVENT_DIFFUSE_REFLECTION)
            sample.flags = BSDFFlags::diffuse_reflection;
        else if (sample_data.event_type == BSDF_EVENT_DIFFUSE_TRANSMISSION)
            sample.flags = BSDFFlags::diffuse_transmission;
        else if (sample_data.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
            sample.flags = BSDFFlags::glossy_reflection;
        else if (sample_data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)
            sample.flags = BSDFFlags::glossy_transmission;
        else if (sample_data.event_type == BSDF_EVENT_SPECULAR_REFLECTION)
            sample.flags = BSDFFlags::delta_reflection;
        else if (sample_data.event_type == BSDF_EVENT_SPECULAR_TRANSMISSION)
            sample.flags = BSDFFlags::delta_transmission;

        return sample_data.event_type != 0;
    }

    MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        var state = this.state;
        MaterialProperties result = {};

        result.emission = emitted_radiance;

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::Bsdf_auxiliary_data aux = {};

        aux.ior1 = ior1; // IOR current medium
        aux.ior2 = ior2; // IOR other side
        aux.k1 = si.wi_ws; // outgoing direction

#if MDL_DF_HANDLE_SLOT_MODE == 0
        ${NAME}_ns::surface_scattering_auxiliary(aux, state);
        result.guide_normal = aux.normal;
        result.diffuse_reflection_albedo = aux.albedo_diffuse;
        result.specular_reflection_albedo = aux.albedo_glossy;
        result.roughness = aux.roughness.x * 0.5f + aux.roughness.y * 0.5f;
#else
        float2 roughness = float2(0.f);
        float roughness_weight = 0.f;
        uint surface_scatter_bsdf_count = ${NAME}_ns::gMDLMaterialData.surface_scatter_bsdf_count;
        for (uint offset = 0; offset < surface_scatter_bsdf_count; offset += MDL_DF_HANDLE_SLOT_MODE) {
            aux.handle_offset = offset;
            ${NAME}_ns::surface_scattering_auxiliary(aux, state);
            for (uint lobe = 0;
                 (lobe < MDL_DF_HANDLE_SLOT_MODE) && ((offset + lobe) < surface_scatter_bsdf_count);
                 ++lobe) {
                result.guide_normal += aux.normal[lobe];
                result.diffuse_reflection_albedo += aux.albedo_diffuse[lobe];
                result.specular_reflection_albedo += aux.albedo_glossy[lobe];
                roughness += aux.roughness[lobe].xy * aux.roughness[lobe].z;
                roughness_weight += aux.roughness[lobe].z;
            }
        }
        if (roughness_weight > 0.f)
            result.roughness = dot(roughness / roughness_weight, float2(0.5f));
#endif

        if (any(isnan(result.guide_normal)) || any(isinf(result.guide_normal))
            || dot(result.guide_normal, result.guide_normal) == 0.f)
            result.guide_normal = si.shading_frame_ws.normal;
        else
            result.guide_normal = normalize(result.guide_normal);

        // TODO: use albedo at normal incidence
        result.material_color = saturate(result.diffuse_reflection_albedo);
        return result;
    }

    BSDFFlags get_lobe_types(const SurfaceInteraction si) { return BSDFFlags::all; }

    override Optional<ExtraBsdfProperties> collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
        var state = this.state;
        ExtraBsdfProperties result = {};

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::Bsdf_auxiliary_data aux = {};

        aux.ior1 = ior1; // IOR current medium
        aux.ior2 = ior2; // IOR other side
        // NOTE: Keep current behavior and pass wi_ws. Bsdf_auxiliary_data comments suggest this
        // may conceptually want wo; preserve wi_ws for compatibility until semantics are reviewed.
        aux.k1 = si.wi_ws; // outgoing direction (preserved behavior)

#if MDL_DF_HANDLE_SLOT_MODE == 0
        ${NAME}_ns::surface_scattering_auxiliary(aux, state);
        result.bsdf_count = 1;
        result.bsdf_N[0] = aux.normal;
        result.bsdf_albedo[0] = aux.albedo_diffuse + aux.albedo_glossy;
        result.bsdf_roughness[0].x = aux.roughness.x;
        result.bsdf_roughness[0].y = aux.roughness.y;
        result.bsdf_weight[0] = float3(1, 1, 1);
#else
        uint surfaceScatterBsdfCount = ${NAME}_ns::gMDLMaterialData.surface_scatter_bsdf_count;
        result.bsdf_count = surfaceScatterBsdfCount;
        uint offset = 0;
        for (; offset < result.bsdf_count; offset += MDL_DF_HANDLE_SLOT_MODE) {
            aux.handle_offset = offset;
            ${NAME}_ns::surface_scattering_auxiliary(aux, state);
            for (uint lobe = 0; (lobe < MDL_DF_HANDLE_SLOT_MODE) && ((offset + lobe) < result.bsdf_count); ++lobe) {
                float3 N = aux.normal[lobe];
                if (any(isnan(N)) || any(isinf(N)))
                    N = si.front_normal_ws();
                float3 T = ${NAME}_ns::mdl_lambda_tangent(lobe + offset, state);
                float3 B = cross(N, T);

                N = si.shading_frame_ws.to_local(N);
                T = si.shading_frame_ws.to_local(T);
                B = si.shading_frame_ws.to_local(B);

                result.bsdf_N[lobe + offset] = N;
                result.bsdf_T[lobe + offset] = T;
                result.bsdf_B[lobe + offset] = B;
                // MDL Albedo is weight*tint
                result.bsdf_albedo[lobe + offset] = aux.albedo_diffuse[lobe] + aux.albedo_glossy[lobe];
                // MDL doesn't provide pure weight
                result.bsdf_weight[lobe + offset] = ${NAME}_ns::mdl_lambda_tint(lobe + offset, state);
                result.bsdf_roughness[lobe + offset].x = ${NAME}_ns::mdl_lambda_roughness_u(lobe + offset, state);
                result.bsdf_roughness[lobe + offset].y = ${NAME}_ns::mdl_lambda_roughness_v(lobe + offset, state);
            }
        }
#endif

        return result;
    }
};

public struct ${NAME} : IMaterial
{
    typealias MaterialInstance = ${NAME}_Instance;

    public MaterialHeader header;
    public MDLMaterialData data;

    ${NAME}_ns::Shading_state_material init_shading_state(
        const SurfaceInteraction si, const MaterialInstanceHints hints
    )
    {
        ${NAME}_ns::Shading_state_material state = {};

#if !MDL_USE_RENDERER_ADAPT_NORMAL
        if (hints & MaterialInstanceHints::AdjustShadingNormal)
            si.adjust_shading_normal();
#endif
        Frame sf = si.shading_frame_ws;

        state.normal = sf.normal;
        state.geom_normal = si.front_normal_ws();
        if (dot(state.normal, state.geom_normal) < 0.f)
            state.normal = -state.normal;
        state.position = si.position_ws;
        state.animation_time = 0.f;
#if MDL_USES_TEXTURE_COORDINATES || MDL_USES_TRANSFORMS
        let geomprops = si.get_geomprop_accessor();
#endif
        [ForceUnroll]
        for (uint i = 0; i < MDL_NUM_TEXTURE_SPACES; ++i) {
            // UV0 is built in; absent secondary streams use the Scene's zero-UV convention.
            float2 mdl_uv = i == 0 ? si.uv : float2(0);
#if MDL_USES_TEXTURE_COORDINATES
            if (i > 0)
                mdl_uv = geomprops.get_additional_texcoord(i, float2(0));
#endif
            // MDL always receives lower-left coordinates. If the scene was upper-left, v_mdl = 1-v_scene
            // makes positive MDL V point opposite the scene bitangent.
            if (!UV_ORIGIN_LOWER_LEFT)
                mdl_uv.y = 1.0 - mdl_uv.y;
            state.text_coords[i] = float3(mdl_uv, 0);
            // Falcor currently stores one per-vertex tangent frame, conventionally derived from UV0.
            state.tangent_u[i] = sf.tangent;
            state.tangent_v[i] = sf.bitangent;
            if (!UV_ORIGIN_LOWER_LEFT)
                state.tangent_v[i] = -state.tangent_v[i];
        }
        state.ro_data_segment_offset = 0;
        state.world_to_object = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        state.object_to_world = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
#if MDL_USES_TRANSFORMS
        state.world_to_object = geomprops.get_space_transform(GeomPropSpace::object, GeomPropSpace::world);
        state.object_to_world = geomprops.get_space_transform(GeomPropSpace::world, GeomPropSpace::object);
#endif
        // state.object_id
        state.meters_per_scene_unit = 1.f;
        state.arg_block_offset = 0;
        state.renderer_state.V = si.wi_ws;
#if MDL_HAS_SCENE_DATA
        state.renderer_state.hit_info = si.hit_info;
#endif

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::init(state);

        return state;
    }

    public override float eval_opacity<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
        var state = init_shading_state(si, hints);
        return saturate(${NAME}_ns::geometry_cutout_opacity(state));
    }

    public override bool is_emissive()
    {
${MDL_EMISSION_PREDICATE}
    }

    public override uint2 compute_emission_sampling_grid(float2 uv[3])
    {
        return uint2(100);
    }

    public override float3 eval_emission<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
#if MDL_HAS_SURFACE_EMISSION
        var state = init_shading_state(si, hints);
        return ${NAME}_ns::mdl_emission_exitance(state);
#else
        return float3(0);
#endif
    }

    public override float3 eval_emissive_radiance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints = MaterialInstanceHints::none
    )
    {
#if MDL_HAS_SURFACE_EMISSION
        var state = init_shading_state(si, hints);
        return ${NAME}_ns::mdl_emission_radiance(state, si.wi_ws);
#else
        return float3(0);
#endif
    }

    MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints
    )
    {
        var state = init_shading_state(si, hints);

        // We use ior(state) instead of the IOR from the material system because MDL compares to the IoR and we have
        // quantization errors since we store it as float16_t
        const float3 exteriorIoR = si.exterior_ior;
        const float3 interiorIoR = ${NAME}_ns::ior(state);

        const float3 ior1 = select(si.front_facing, exteriorIoR, interiorIoR);
        const float3 ior2 = select(si.front_facing, interiorIoR, exteriorIoR);
        const float interior_ior = (interiorIoR.x + interiorIoR.y + interiorIoR.z) / 3.0;

        // Reuse the already initialized state and textures. EDFs describe emitted
        // light and do not alter the SDK BSDF sampling/evaluation contract.
        float3 emission = ${NAME}_ns::mdl_emission_radiance(state, si.wi_ws);
        ${NAME}_Instance mi = { data, state, ior1, ior2, interior_ior, emission };

        return mi;
    }

${MDL_WRITE_MATERIAL_AOVS}
};
)";
}

} // namespace falcor
