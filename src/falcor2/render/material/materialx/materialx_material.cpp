// SPDX-License-Identifier: Apache-2.0

#include "materialx_material.h"

#include "codegen/codegen.h"

#include "falcor2/render/scene.h"
#include "falcor2/render/material.h"
#include "falcor2/render/material/materialx/mx139/lut_globals.h"
#include "falcor2/render/texture_manager.h"

#include "falcor2/core/types.h"

#include "falcor2/core/asset_resolver.h"
#include "falcor2/utils/managed_buffer.h"

#include <sgl/device/device.h>
#include <sgl/core/string.h>
#include <sgl/core/crypto.h>
#include <sgl/core/hash.h>

#include <algorithm>
#include <cctype>
#include <filesystem>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <variant>
#include <vector>

namespace {

template<typename T1, typename... Rest>
inline size_t hash_add(size_t hash1, const T1& t1, const Rest&... rest)
{
    size_t result = sgl::hash_combine(hash1, sgl::hash(t1));
    ((result = sgl::hash_combine(result, sgl::hash(rest))), ...);
    return result;
}

template<typename T>
inline size_t hash_add_vector(size_t hash, const std::vector<T>& values)
{
    hash = hash_add(hash, values.size());
    for (const T& value : values)
        hash = hash_add(hash, value);
    return hash;
}

bool is_ascii_space(char c)
{
    return c == ' ' || c == '\t' || c == '\n' || c == '\r';
}

std::string_view trim_ascii_space(std::string_view value)
{
    while (!value.empty() && is_ascii_space(value.front()))
        value.remove_prefix(1);
    while (!value.empty() && is_ascii_space(value.back()))
        value.remove_suffix(1);
    return value;
}

void add_materialx_search_paths(falcor::AssetResolver& resolver, const std::string& search_paths)
{
    for (std::string_view path : sgl::string::split(search_paths, ";")) {
        path = trim_ascii_space(path);
        if (!path.empty())
            resolver.add_search_path(std::filesystem::path(std::string(path)));
    }
}

sgl::TextureAddressingMode materialx_address_mode_to_sgl(std::string_view address_mode)
{
    address_mode = trim_ascii_space(address_mode);
    if (address_mode == "1" || address_mode == "clamp")
        return sgl::TextureAddressingMode::clamp_to_edge;
    if (address_mode == "3" || address_mode == "mirror")
        return sgl::TextureAddressingMode::mirror_repeat;
    if (address_mode == "0" || address_mode == "constant")
        return sgl::TextureAddressingMode::clamp_to_border;
    return sgl::TextureAddressingMode::wrap;
}

sgl::TextureFilteringMode materialx_filter_type_to_sgl(std::string_view filter_type)
{
    filter_type = trim_ascii_space(filter_type);
    if (filter_type == "0" || filter_type == "closest")
        return sgl::TextureFilteringMode::point;
    return sgl::TextureFilteringMode::linear;
}

std::vector<uint32_t> property_list_to_uint32_vector(const falcor::detail::PropertyList& value)
{
    std::vector<uint32_t> result;
    const std::vector<int64_t>& ids = value.as_vector<int64_t>();
    result.reserve(ids.size());
    for (int64_t id : ids) {
        FALCOR_CHECK(
            id >= 0 && id <= static_cast<int64_t>(std::numeric_limits<uint32_t>::max()),
            "MaterialX geomprop ID {} is outside the uint32 range.",
            id
        );
        result.push_back(static_cast<uint32_t>(id));
    }
    return result;
}

falcor::TextureManager::SamplerConfig
make_materialx_sampler_config(const falcor::materialx::params::TextureInfo& texture_info)
{
    const sgl::TextureFilteringMode filter = materialx_filter_type_to_sgl(texture_info.filtertype);
    return {
        .min_filter = filter,
        .mag_filter = filter,
        .mip_filter = filter,
        .address_u = materialx_address_mode_to_sgl(texture_info.uaddressmode),
        .address_v = materialx_address_mode_to_sgl(texture_info.vaddressmode),
        .address_w = sgl::TextureAddressingMode::wrap,
        .border_color = texture_info.border_color,
    };
}

} // namespace

namespace falcor {

MaterialXMaterial::MaterialXMaterial()
{
    m_data_buffer = ManagedBuffer::create({
        .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
        .label = fmt::format("MaterialXMaterial::m_data_buffer ({})", (void*)this),
    });

    m_material_properties.set_instance(this);
}

MaterialXMaterial::~MaterialXMaterial()
{
    // Scene globals are detached in on_remove_from_scene(); Scene internals may
    // already be partially destroyed when this destructor runs.
    m_lut_globals.reset();
}

void MaterialXMaterial::set_properties(const Properties& props_)
{
    FALCOR_CHECK(
        !props_.has_property("mtlx_static_closure_pruning"),
        "Material property 'mtlx_static_closure_pruning' was removed. Use "
        "'mtlx_optimize_graph::closure_pruning'."
    );
    FALCOR_CHECK(
        !props_.has_property("mtlx_closure_pruning"),
        "Material property 'mtlx_closure_pruning' was removed. Use 'mtlx_optimize_graph::closure_pruning'."
    );
    FALCOR_CHECK(
        !props_.has_property("mtlx_layering"),
        "Material property 'mtlx_layering' was renamed. Use 'mtlx_layering_mode'."
    );

    Properties props = props_;
    // Converts two important properties from the legacy naming.
    if (!props.has_property("mtlx_transmissive_bsdfs")) {
        if (auto it = props_.get_optional<std::string_view>("transmissiveBsdfs")) {
            props.set("mtlx_transmissive_bsdfs", *it);
        }
    }
    if (!props.has_property("mtlx_flip_v_texcoord")) {
        if (auto it = props.get_optional<std::string_view>("mtlx_source")) {
            props.set<bool>("mtlx_flip_v_texcoord", *it == "houdini");
        } else if (auto it2 = props.get_optional<std::string_view>("source")) {
            props.set<bool>("mtlx_flip_v_texcoord", *it2 == "houdini");
        }
    }
    class_descriptor().read_from_properties(this, props);

    if (m_require_codegen)
        run_codegen();

    // Deserialize material properties.
    m_material_properties.read_from_properties(props);

    // TODO(@tdavidovic) - make this on_change in m_material_properties.
    if (m_codegen_result) {
        for (auto& param : m_codegen_result->all_material_params.m_params) {
            // update Param from properties and load if needed
            FALCOR_ASSERT(!param.is_editable || param.holds_value());
            param.set_value(m_material_properties);
        }
    }
}

void MaterialXMaterial::require_codegen()
{
    mark_dirty(DirtyFlags::properties);
    m_require_codegen = true;
}

void MaterialXMaterial::validate_device_support() const
{
    if (!m_scene || !m_scene->device() || m_scene->device()->type() != sgl::DeviceType::cuda)
        return;

    FALCOR_THROW(
        "MaterialXMaterial is disabled on CUDA because generated MaterialX shaders can trigger pathological compile "
        "times with NVRTC 12.6. If you know Slang is using a newer NVRTC version, this temporary guard can be "
        "disabled in the code."
    );
}

detail::PropertyList MaterialXMaterial::mtlx_geomprop_names_property() const
{
    return detail::PropertyList::create(m_mtlx_geomprop_names);
}

void MaterialXMaterial::set_mtlx_geomprop_names_property(const detail::PropertyList& value)
{
    m_mtlx_geomprop_names = value.as_vector<std::string>();
    require_codegen();
}

detail::PropertyList MaterialXMaterial::mtlx_geomprop_ids_property() const
{
    std::vector<int64_t> ids;
    ids.reserve(m_mtlx_geomprop_ids.size());
    for (uint32_t id : m_mtlx_geomprop_ids)
        ids.push_back(static_cast<int64_t>(id));
    return detail::PropertyList::create(std::move(ids));
}

void MaterialXMaterial::set_mtlx_geomprop_ids_property(const detail::PropertyList& value)
{
    m_mtlx_geomprop_ids = property_list_to_uint32_vector(value);
    require_codegen();
}

void MaterialXMaterial::on_load_resources()
{
    if (!m_codegen_result)
        return;

    for (auto& param : m_codegen_result->all_material_params.m_params) {
        if (auto* texture_info = param.extract_value<materialx::params::TextureInfo>()) {
            // Empty filename inputs are valid MaterialX and should sample the
            // image node default value rather than trying to load the base path.
            if (texture_info->file_name.empty())
                continue;

            // TODO(tdavidovic): We need to add our path to the asset resolver used by texture_manager,
            // otherwise we can't get the UDIMs resolved.
            std::variant<std::monostate, TextureManager::SamplerConfig, ref<sgl::Sampler>> sampler;
            sampler = make_materialx_sampler_config(*texture_info);
            TextureHandle texture_handle = m_scene->texture_manager()->load_texture({
                .path = texture_info->file_name,
                .resolver = m_mtlx_path_resolver,
                .sampler = sampler,
                .generate_mips = true,
                .srgb = texture_info->is_color_output_type,
                .usage = sgl::TextureUsage::shader_resource,
                .load_deferred = true,
            });
            texture_info->texture_handle = texture_handle;
        }
    }
}

std::vector<TextureHandle> MaterialXMaterial::build_texture_list() const
{
    std::vector<TextureHandle> result;
    if (!m_codegen_result)
        return result;
    for (const auto& param : m_codegen_result->all_material_params.m_params) {
        if (auto* texture_info = param.extract_value<materialx::params::TextureInfo>()) {
            result.push_back(texture_info->texture_handle);
        }
    }
    return result;
}

void MaterialXMaterial::on_remove_from_scene()
{
    remove_lut_scene_globals();
}

void MaterialXMaterial::update(SceneUpdateContext& ctx)
{
    if (m_require_codegen)
        run_codegen();

    if (!m_codegen_result) {
        m_slang_type_name = "InvalidMaterial";
        update_lut_scene_globals(ctx);
        return;
    }

    m_slang_type_name = m_codegen_result->material_name;
    update_lut_scene_globals(ctx);

    // If explicit write_shader_path is provided, write shader to the given path
    if (!m_debug_write_shader_path.empty()) {
        std::ofstream ofs(m_debug_write_shader_path);
        ofs << m_codegen_result->module_source;
        ofs.close();
    }

    if (!m_slang_module || !m_debug_load_shader_path.empty()) {
        // If explicit shader_file is provided, replace the generate code with its content.
        if (!m_debug_load_shader_path.empty()) {
            std::ifstream ifs(m_debug_load_shader_path);
            std::stringstream buffer;
            buffer << ifs.rdbuf();
            m_codegen_result->module_source = buffer.str();
        }

        m_slang_module = m_scene->device()->load_module_from_source(
            m_codegen_result->module_name,
            m_codegen_result->module_source
        );
    }

    m_data_buffer->resize(required_data_buffer_size());
    if (m_codegen_result->all_material_params.m_editable_count > 0 && m_data_buffer->size() > 0) {
        void* dst_data = m_data_buffer->data();

        auto layout = m_slang_module->layout();
        std::string buffer_type_name = fmt::format("StructuredBuffer<{}>", m_codegen_result->material_data_name);
        auto type = layout->find_type_by_name(buffer_type_name.c_str());
        auto type_layout = layout->get_type_layout(type);
        auto element_type_layout = type_layout->element_type_layout();

        auto buffer_cursor = make_ref<sgl::BufferCursor>(
            m_scene->device()->type(),
            element_type_layout,
            dst_data,
            m_codegen_result->data_struct_size
        );
        auto element_cursor = (*buffer_cursor)[0];
        for (auto& it : m_codegen_result->all_material_params.m_params)
            it.set_cursor(element_cursor);
        buffer_cursor->apply();
    }
    m_data_buffer->update_buffer(ctx.command_encoder());

    m_has_entry_point_volume_properties = m_codegen_result->has_entry_point_volume_properties;
}

void MaterialXMaterial::remove_lut_scene_globals()
{
    if (!m_lut_globals)
        return;

    if (m_scene)
        m_scene->remove_scene_globals(m_lut_globals.get());
    m_lut_globals.reset();
}

void MaterialXMaterial::update_lut_scene_globals(SceneUpdateContext& ctx)
{
    const bool needs_mx139_globals = m_codegen_result && m_codegen_result->needs_mx139_lut_scene_globals;
    if (needs_mx139_globals) {
        if (!m_lut_globals) {
            // TODO: Not every MaterialX material instance should create a new MX139 LUT scene globals object.
            // We need to share an instance among all MX139 materials (probably cached on the scene itself).
            // Also, this ideally should happen during initialization and not during update.
            m_lut_globals = materialx::mx139::create_lut_scene_globals(ctx.device());
            m_scene->add_scene_globals(m_lut_globals);
        }
    } else {
        remove_lut_scene_globals();
    }
}

template<typename CursorT>
void MaterialXMaterial::write_to_cursor_impl(CursorT cursor) const
{
    if (m_slang_module) {
        if (requires_data_buffer()) {
            cursor["data_buffer"] = to_handle(m_data_buffer);
        } else {
            if (m_codegen_result->all_material_params.m_params.empty())
                return;

            auto element_cursor = cursor["data"];
            for (auto& it : m_codegen_result->all_material_params.m_params)
                it.set_cursor(element_cursor);
        }
    }
}

template void MaterialXMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void MaterialXMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

ref<sgl::SlangModule> MaterialXMaterial::required_module() const
{
    return m_slang_module;
}

void MaterialXMaterial::run_codegen()
{
    // If we are not forcing update and nothing has changed, bail out.
    if (!m_require_codegen)
        return;
    validate_device_support();
    m_require_codegen = false;

    // Check hash of things that control code emission.
    size_t codegen_properties_hash = sgl::hash(
        m_mtlx_basepath,
        m_mtlx_search_paths,
        m_mtlx_editable_params,
        m_mtlx_node_name,
        m_mtlx_layeringmethod,
        m_mtlx_flip_v_texcoord,
        m_mtlx_transmissive_bsdfs,
        m_mtlx_auto_transmission,
        m_mtlx_optimize_graph,
        m_mtlx_layering_mode,
        m_mtlx_compensation,
        m_mtlx_autogamma,
        m_mtlx_target_color_space_override
    );
    codegen_properties_hash = hash_add_vector(codegen_properties_hash, m_mtlx_geomprop_names);
    codegen_properties_hash = hash_add_vector(codegen_properties_hash, m_mtlx_geomprop_ids);
    if (!m_mtlx_buffer.empty()) {
        codegen_properties_hash = hash_add(codegen_properties_hash, m_mtlx_buffer);
    } else {
        codegen_properties_hash = hash_add(codegen_properties_hash, m_mtlx_path);
    }

    if (m_codegen_properties_hash == codegen_properties_hash)
        return;
    m_codegen_properties_hash = codegen_properties_hash;

    /// We are rebuilding the material.
    mark_dirty(DirtyFlags::resources | DirtyFlags::properties);
    m_slang_module.reset();
    m_codegen_result = {};
    m_mtlx_path_resolver.reset(new AssetResolver());
    m_mtlx_path_resolver->add_search_path(m_mtlx_basepath);
    add_materialx_search_paths(*m_mtlx_path_resolver, m_mtlx_search_paths);

    std::variant<std::string, std::filesystem::path> document_desc;
    if (m_mtlx_buffer.size() > 0) {
        document_desc = m_mtlx_buffer;
    } else {
        // If we have a material path, automatically add its parent directory to the search paths,
        // so that textures stored relative to it are detected.
        std::filesystem::path resolved_mtlx_path = m_mtlx_path_resolver->resolve_path(m_mtlx_path);
        FALCOR_CHECK(
            !resolved_mtlx_path.empty(),
            "MaterialX file `{}` does not exist in `mtlx_basepath` search paths `{}`.",
            m_mtlx_path,
            m_mtlx_basepath
        );
        m_mtlx_path_resolver->add_search_path(resolved_mtlx_path.parent_path());
        document_desc = std::move(resolved_mtlx_path);
    }

    m_codegen_desc = {};
    m_codegen_desc.document = document_desc;
    m_codegen_desc.node_name = m_mtlx_node_name;
    m_codegen_desc.asset_resolver = m_mtlx_path_resolver;
    m_codegen_desc.flip_v_texcoord = m_mtlx_flip_v_texcoord;
    m_codegen_desc.positionfree_layering = (m_mtlx_layeringmethod == "positionfree");
    // Currently unused
    // m_codegen_desc.sigma_a;
    // m_codegen_desc.sigma_s;
    // m_codegen_desc.phase_function_params;
    m_codegen_desc.transmissive_bsdfs = sgl::string::split(m_mtlx_transmissive_bsdfs, ";,");
    m_codegen_desc.auto_transmission = m_mtlx_auto_transmission;
    m_codegen_desc.optimize_graph_flags = m_mtlx_optimize_graph;
    m_codegen_desc.layering_mode = m_mtlx_layering_mode;
    m_codegen_desc.compensation_mode = m_mtlx_compensation;
    m_codegen_desc.auto_gamma_image = m_mtlx_autogamma;
    m_codegen_desc.target_color_space_override = m_mtlx_target_color_space_override;
    m_codegen_desc.make_editable = !m_mtlx_editable_params.empty();
    FALCOR_CHECK(
        m_mtlx_geomprop_names.size() == m_mtlx_geomprop_ids.size(),
        "MaterialX geomprop name/ID lists must have the same length ({} names, {} IDs).",
        m_mtlx_geomprop_names.size(),
        m_mtlx_geomprop_ids.size()
    );
    std::unordered_set<std::string> seen_geomprop_names;
    std::unordered_map<std::string, uint32_t> geomprop_ids;
    seen_geomprop_names.reserve(m_mtlx_geomprop_names.size());
    geomprop_ids.reserve(m_mtlx_geomprop_names.size());
    for (size_t i = 0; i < m_mtlx_geomprop_names.size(); ++i) {
        const std::string& name = m_mtlx_geomprop_names[i];
        FALCOR_CHECK(!name.empty(), "MaterialX geomprop names cannot be empty.");
        FALCOR_CHECK(seen_geomprop_names.insert(name).second, "Duplicate MaterialX geomprop name `{}`.", name);
        geomprop_ids.emplace(name, m_mtlx_geomprop_ids[i]);
    }
    if (!geomprop_ids.empty()) {
        m_codegen_desc.geomprop_id_callback = [geomprop_ids = std::move(geomprop_ids)](
                                                  std::string_view geomprop,
                                                  std::string_view /*output_type*/
                                              ) -> std::optional<uint32_t>
        {
            const auto it = geomprop_ids.find(std::string(geomprop));
            if (it == geomprop_ids.end())
                return {};
            return it->second;
        };
    }

    // We are generating new code.
    m_codegen_result = materialx::CodeGen::generate(m_codegen_desc);
}

FALCOR_SCENE_REGISTER_MATERIAL(MaterialXMaterial);

} // namespace falcor
