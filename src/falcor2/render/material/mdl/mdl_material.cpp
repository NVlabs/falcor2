// SPDX-License-Identifier: Apache-2.0

#include "mdl_material.h"

#include "falcor2/render/scene.h"

#include "falcor2/core/properties.h"
#include "falcor2/utils/string_utils.h"

#include <sgl/core/crypto.h>
#include <sgl/core/hash.h>

#include <fstream>
#include <sstream>

namespace falcor {

namespace {
/// Enable MDL's use_renderer_adapt_normal option which allows individual normal mapped normals
/// to be adjusted instead of just the global shading normal.
const bool USE_RENDERER_ADAPT_NORMAL = true;

/// Material shader source code template.
extern std::string SHADER_SOURCE_TEMPLATE;

class ScopedSearchPaths {
public:
    ScopedSearchPaths(MDLContext& mdl_context, std::filesystem::path path)
        : m_mdl_context(mdl_context)
        , m_path(std::move(path))
    {
        m_mdl_context.add_search_paths({&m_path, 1});
    }

    ~ScopedSearchPaths() { m_mdl_context.remove_search_paths({&m_path, 1}); }

private:
    MDLContext& m_mdl_context;
    std::filesystem::path m_path;
};

} // namespace

MDLMaterial::MDLMaterial()
{
    m_mdl_data = ManagedBuffer::create({
        .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
        .label = fmt::format("MDLMaterial::m_mdl_data ({})", (void*)this),
    });

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
    m_texture_table_needs_update = true;

    using ParamType = MDLContext::CompileResult::ArgumentLayout::Type;
    const uint8_t* arg_block_data = reinterpret_cast<const uint8_t*>(m_mdl_data->cdata()) + m_arg_block_offset;

    // Update textures from properties
    for (auto& it : m_result.arg_block_layout) {
        if (it.second.type != ParamType::texture)
            continue;

        const void* data = arg_block_data + it.second.offset;
        // MDL uses index 0 to mean "no texture", and then offsets by 1
        uint32_t texture_index = *reinterpret_cast<const uint32_t*>(data);
        if (texture_index == 0)
            continue;
        texture_index--;

        const auto& texture = m_result.textures[texture_index];
        auto& texture_resource = m_texture_resources[texture_index];

        // We only allow Texture2D to be controlled by properties
        if (texture.type != MDLContext::CompileResult::Texture::Type::Texture2D)
            continue;

        // At this stage we know the argument is a texture that does exist in the properties.
        const std::string& texture_path = m_material_properties.get_ref<std::string>(it.first);
        bool texture_srgb = m_material_properties.get<bool>(texture_resource.srgb_property_name);

        // If there is no change from the currently loaded texture, we just skip
        if (texture_resource.srgb == texture_srgb && texture_resource.path == texture_path)
            continue;

        // If the texture has changed, we store the new paths and reset the handle, indicating
        // that the texture needs to be reloaded.
        texture_resource.path = try_resolve(texture_path);
        texture_resource.srgb = texture_srgb;
        texture_resource.texture_handle = {};
    }

    // Actually load the textures that aren't loaded and should be (arg handling above just set the properties)
    for (size_t i = 0; i < m_result.textures.size(); ++i) {
        const auto& texture = m_result.textures[i];
        auto& texture_resource = m_texture_resources[i];

        // Already loaded, no need to re-load it
        if (texture_resource.texture_handle.is_valid())
            continue;

        if (texture.type == MDLContext::CompileResult::Texture::Type::Texture2D) {
            texture_resource.texture_handle = m_scene->texture_manager()->load_texture({
                .path = texture_resource.path,
                .generate_mips = true,
                .srgb = texture_resource.srgb,
                .usage = sgl::TextureUsage::shader_resource,
                .load_deferred = true,
            });

        } else {
            sgl::SubresourceData subresource = {.data = texture.data.data(), .size = texture.data.size()};

            auto texture_3d = m_scene->device()->create_texture({
                .type = sgl::TextureType::texture_3d,
                .format = texture.format,
                .width = texture.width,
                .height = texture.height,
                .depth = texture.depth,
                .usage = sgl::TextureUsage::shader_resource,
                .data = {&subresource, 1},
            });

            texture_resource.texture_handle = m_scene->texture_manager()->register_texture({
                .texture = std::move(texture_3d),
            });
        }
    }
}

void MDLMaterial::update(SceneUpdateContext& ctx)
{
    // Write texture table only after on_load_resources() has run.
    if (m_texture_table_needs_update) {
        for (size_t i = 0; i < m_texture_resources.size(); ++i) {
            m_mdl_data->set_data(
                m_texture_resources[i].texture_handle.data(),
                m_texture_table_offset + i * sizeof(uint32_t)
            );
        }
        m_texture_table_needs_update = false;
    }

    /// Build the slang module, if it is missing, needs to be written or read from file.
    if (!m_slang_module || !m_debug_write_shader_path.empty() || !m_debug_load_shader_path.empty()) {
        /// We prepare the actual code, minus the type name, to allow efficient class compilation.
        std::string defines;
        defines += fmt::format("#define MDL_NUM_TEXTURE_RESULTS {}\n", m_request.options.num_texture_results);
        defines += fmt::format("#define MDL_DF_HANDLE_SLOT_MODE {}\n", m_request.options.df_handle_slot_mode);
        defines += fmt::format(
            "#define MDL_USE_RENDERER_ADAPT_NORMAL {}\n",
            m_request.options.use_renderer_adapt_normal ? "1" : "0"
        );

        std::string source = replace_substrings(
            SHADER_SOURCE_TEMPLATE,
            std::map<std::string, std::string>({
                {"${MDL_DEFINES}", defines},
                {"${MDL_CODE}", m_result.code},
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

        m_ior = (float16_t)((m_result.ior.x + m_result.ior.y + m_result.ior.z) / 3.f);

        m_slang_module = m_scene->device()->load_module_from_source(m_slang_type_name, source);
    }

    if (m_result.arg_block_segments.size() > 0) {
        uint8_t* arg_block_data = reinterpret_cast<uint8_t*>(m_mdl_data->data_unsafe()) + m_arg_block_offset;
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

    m_mdl_data->update_buffer(ctx.command_encoder());
}

template<typename CursorT>
void MDLMaterial::write_to_cursor_impl(CursorT cursor) const
{
    auto data = cursor["data"];
    data["data"] = to_handle(m_mdl_data);
    data["arg_block_offset"] = m_arg_block_offset;
    data["texture_table_offset"] = m_texture_table_offset;
    data["surface_scatter_bsdf_count"] = m_result.surface_scatter_handle_count;
}

template void MDLMaterial::write_to_cursor_impl(sgl::BufferElementCursor cursor) const;
template void MDLMaterial::write_to_cursor_impl(sgl::ShaderCursor cursor) const;

ref<sgl::SlangModule> MDLMaterial::required_module() const
{
    return m_slang_module;
}

std::vector<TextureHandle> MDLMaterial::build_texture_list() const
{
    std::vector<TextureHandle> result;
    result.reserve(m_texture_resources.size());
    for (const auto& tex : m_texture_resources) {
        if (tex.texture_handle.is_valid())
            result.push_back(tex.texture_handle);
    }
    return result;
}

void MDLMaterial::require_codegen()
{
    mark_dirty(DirtyFlags::properties);
    m_require_codegen = true;
}

void MDLMaterial::run_codegen()
{
    if (!m_require_codegen)
        return;

    // Hash the properties influencing codegen to determine if we actually need to re-run.
    size_t codegen_properties_hash
        = sgl::hash(m_mdl_library_path, m_mdl_material_name, m_mdl_class_compilation, m_learnable);
    if (codegen_properties_hash == m_codegen_properties_hash)
        return;
    m_codegen_properties_hash = codegen_properties_hash;

    MDLContext& mdl_context = MDLContext::get();

    /// We are rebuilding the material.
    mark_dirty(DirtyFlags::resources | DirtyFlags::properties);
    m_slang_module.reset();
    m_library_path_resolver.reset(new AssetResolver());
    m_library_path_resolver->add_search_path(m_mdl_library_path);

    FALCOR_CHECK(!m_mdl_library_path.empty(), "MDL library path must not be empty");
    FALCOR_CHECK(!m_mdl_material_name.empty(), "MDL material name must not be empty");

    ScopedSearchPaths scoped_search_paths(mdl_context, m_mdl_library_path);

    // Compile material.
    m_request = {};
    m_request.qualified_material_name = m_mdl_material_name;
    m_request.instance_name = "";
    // m_request.options.opt_level = 1;
    m_request.options.use_renderer_adapt_normal = USE_RENDERER_ADAPT_NORMAL;
    m_request.use_class_compilation = m_mdl_class_compilation;

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

    m_result = mdl_context.compile(m_request);
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

    /// Create data buffer with the default values.
    /// TODO(@tdavidovic): Could be moved to update, but on_load_resources would need to flag
    /// if a texture_handle is dirty, instead of using mdl_data's dirty tracking
    m_mdl_data->resize(
        m_result.ro_data_segment_size() + m_result.arg_block_segment_size()
        + m_result.textures.size() * sizeof(uint32_t)
    );

    /// TODO(@tdavidovic): Can be moved to update.
    if (m_result.ro_data_segments.size() > 0)
        m_mdl_data->set_data(m_result.ro_data_segments[0].data.data(), m_result.ro_data_segment_size(), 0);

    if (m_result.arg_block_segments.size() > 0)
        m_mdl_data->set_data(
            m_result.arg_block_segments[0].data.data(),
            m_result.arg_block_segment_size(),
            m_arg_block_offset
        );

    // Set the texture_resource information for all textures.
    // TODO(@tdavidovic): This *could* be moved to on_load_resources.
    // The problem is that not all textures are exposed as properties,
    // we need to set this for both cases (constant and mutable)
    m_texture_resources.clear();
    m_texture_resources.resize(m_result.textures.size());
    for (size_t i = 0; i < m_result.textures.size(); ++i) {
        const auto& texture = m_result.textures[i];
        auto& texture_resource = m_texture_resources[i];

        if (texture.type == MDLContext::CompileResult::Texture::Type::Texture2D) {
            texture_resource.path = try_resolve(texture.path);
            texture_resource.srgb = texture.srgb;
        }
    }

    /// Create default material properties.
    m_material_properties.clear();
    if (m_result.arg_block_segments.size() > 0) {
        std::vector<uint8_t>& arg_block_data = m_result.arg_block_segments[0].data;

        auto on_change = reflection::on_change<MDLMaterial>(
            [](MDLMaterial& self)
            {
                // We only need updating resources when textures change.
                // We rely on on_load_resources() to be quick in case no work needs to be done.
                self.mark_dirty(DirtyFlags::properties | DirtyFlags::resources);
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
                uint32_t texture_index = *reinterpret_cast<const uint32_t*>(data);
                // MDL uses index 0 to mean "no texture", and then offsets by 1
                if (texture_index > 0) {
                    const auto& texture = m_result.textures[texture_index - 1];
                    auto& texture_resource = m_texture_resources[texture_index - 1];

                    if (texture.type == MDLContext::CompileResult::Texture::Type::Texture2D) {
                        texture_resource.srgb_property_name = it.first + ":srgb";
                        m_material_properties.add_property(it.first, texture.path, false, on_change, ui_group);
                        m_material_properties.add_property(
                            texture_resource.srgb_property_name,
                            texture.srgb,
                            false,
                            on_change,
                            ui_group
                        );
                    }
                }
                break;
            }
            case ParamType::unsupported:
                break;
            };
        }
    }

    m_require_codegen = false;
}

FALCOR_SCENE_REGISTER_MATERIAL(MDLMaterial);

namespace {
std::string SHADER_SOURCE_TEMPLATE = R"(
module ${NAME};

__exported import falcor2.render;
import falcor2.utils;

namespace ${NAME}_ns {

${MDL_DEFINES}

#include "falcor2/render/materials/mdl/mdl_runtime.slangh"

${MDL_CODE}

} // ${NAME}_ns

struct ${NAME}_Instance : IMaterialInstance
{
    MDLMaterialData data;
    ${NAME}_ns::Shading_state_material state;
    float3 ior1, ior2;

    float3 eval<S : ISampleGenerator>(const SurfaceInteraction si, const float3 wo, inout S sg)
    {
        ${NAME}_ns::Bsdf_evaluate_data eval_data = {};
        eval_data.ior1 = ior1;     // IOR current medium
        eval_data.ior2 = ior2;     // IOR other side
        eval_data.k1 = si.wi_ws; // outgoing direction
        eval_data.k2 = wo;         // incoming direction

        ${NAME}_ns::gMDLMaterialData = data;

        float3 result = float3(0.f);
#if MDL_DF_HANDLE_SLOT_MODE == 0
        ${NAME}_ns::surface_scattering_evaluate(eval_data, state);
        result = eval_data.bsdf_diffuse + eval_data.bsdf_glossy;
#else
        uint surfaceScatterBsdfCount = ${NAME}_ns::gMDLMaterialData.surface_scatter_bsdf_count;
        uint offset = 0;
        for (; offset < surfaceScatterBsdfCount; offset += MDL_DF_HANDLE_SLOT_MODE) {
            eval_data.handle_offset = offset;
            ${NAME}_ns::surface_scattering_evaluate(eval_data, state);
            for (uint lobe = 0; (lobe < MDL_DF_HANDLE_SLOT_MODE) && ((offset + lobe) < surfaceScatterBsdfCount);
                 ++lobe) {
                result += (eval_data.bsdf_diffuse[lobe] + eval_data.bsdf_glossy[lobe]);
            }
        }
#endif
        return result;
    }

    bool sample<S : ISampleGenerator>(const SurfaceInteraction si, inout S sg, out BSDFSample result)
    {
        ${NAME}_ns::Bsdf_sample_data sample_data = {};
        sample_data.ior1 = ior1;          // IOR current medium
        sample_data.ior2 = ior2;          // IOR other side
        sample_data.k1 = si.wi_ws;      // outgoing direction
        sample_data.xi = next_float4(sg); // pseudo-random sample numbers

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::surface_scattering_sample(sample_data, state);

        result = {};
        result.wo_ws = sample_data.k2;
        result.weight = sample_data.bsdf_over_pdf;
        result.pdf = sample_data.pdf;

        if (sample_data.event_type == BSDF_EVENT_DIFFUSE_REFLECTION)
            result.lobe_types = LobeTypes::diffuse_reflection;
        else if (sample_data.event_type == BSDF_EVENT_DIFFUSE_TRANSMISSION)
            result.lobe_types = LobeTypes::diffuse_transmission;
        else if (sample_data.event_type == BSDF_EVENT_GLOSSY_REFLECTION)
            result.lobe_types = LobeTypes::glossy_reflection;
        else if (sample_data.event_type == BSDF_EVENT_GLOSSY_TRANSMISSION)
            result.lobe_types = LobeTypes::glossy_transmission;
        else if (sample_data.event_type == BSDF_EVENT_SPECULAR_REFLECTION)
            result.lobe_types = LobeTypes::delta_reflection;
        else if (sample_data.event_type == BSDF_EVENT_SPECULAR_TRANSMISSION)
            result.lobe_types = LobeTypes::delta_transmission;

        return sample_data.event_type != 0;
    }

    float eval_pdf(const SurfaceInteraction si, const float3 wo_ws)
    {
        ${NAME}_ns::Bsdf_pdf_data pdf_data = {};
        pdf_data.ior1 = ior1;     // IOR current medium
        pdf_data.ior2 = ior2;     // IOR other side
        pdf_data.k1 = si.wi_ws; // outgoing direction
        pdf_data.k2 = wo_ws;      // incoming direction

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::surface_scattering_pdf(pdf_data, state);

        return pdf_data.pdf;
    }

    MaterialProperties collect_properties(const SurfaceInteraction si)
    {
        MaterialProperties result = {};

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::Bsdf_auxiliary_data aux = {};
        ${NAME}_ns::surface_scattering_auxiliary(aux, state);

#if MDL_DF_HANDLE_SLOT_MODE == 0
        result.guide_normal = aux.normal;
        result.diffuse_reflection_albedo = aux.albedo_diffuse;
        result.specular_reflection_albedo = aux.albedo_glossy;
        result.roughness = aux.roughness.x * 0.5f + aux.roughness.y * 0.5f;
#else
        // TODO: Weighted average for the multi-lobe case.
        result.guide_normal = aux.normal[0];
        result.diffuse_reflection_albedo = aux.albedo_diffuse[0];
        result.specular_reflection_albedo = aux.albedo_glossy[0];
        result.roughness = aux.roughness[0].x * 0.5f + aux.roughness[0].y * 0.5f;
#endif

    return result;
    }

    LobeTypes get_lobe_types(const SurfaceInteraction si) { return LobeTypes::all; }

    override Optional<ExtraBsdfProperties> collect_extra_bsdf_properties(const SurfaceInteraction si, const float3 wo)
    {
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

    MaterialInstance setup_material_instance<TLodSampler : ILodSampler>(
        const SurfaceInteraction si,
        const TLodSampler lod_sampler,
        const MaterialInstanceHints hints
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
        // Flipping v coordinate (within the same given udim range)
        state.text_coords[0] = float3(si.uv.x, floor(si.uv.y) + 1 - frac(si.uv.y), 0);
        state.tangent_u[0] = sf.tangent;
        state.tangent_v[0] = -sf.bitangent;
        state.ro_data_segment_offset = 0;
        state.world_to_object = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        state.object_to_world = float4x4(1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1);
        // state.object_id
        state.meters_per_scene_unit = 1.f;
        state.arg_block_offset = 0;
        state.renderer_state.V = si.wi_ws;

        ${NAME}_ns::gMDLMaterialData = data;
        ${NAME}_ns::init(state);

        // We use ior(state) instead of the IOR from the material system because MDL compares to the IoR and we have
        // quantization errors since we store it as float16_t
        const float3 exteriorIoR = si.ior;
        const float3 interiorIoR = ${NAME}_ns::ior(state);

        const float3 ior1 = select(si.front_facing, exteriorIoR, interiorIoR);
        const float3 ior2 = select(si.front_facing, interiorIoR, exteriorIoR);

        ${NAME}_Instance mi = { data, state, ior1, ior2 };

        return mi;
    }
};
)";
}

} // namespace falcor
