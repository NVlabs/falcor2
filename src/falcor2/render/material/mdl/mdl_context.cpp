// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "mdl_context.h"

#include "falcor2/core/error.h"
#include "falcor2/core/logger.h"
#include "falcor2/core/platform.h"
#include "falcor2/render/material_aov_registry.h"
#include "falcor2/utils/fnv_hash.h"
#include "falcor2/falcor2.h"

#include "mdl_sdk_wrapper.h"
#include "mdl_utils.h"

#include <sgl/core/platform.h>
#include <sgl/core/crypto.h>

#include <algorithm>
#include <array>
#include <compare>
#include <functional>
#include <set>
#include <string_view>
#include <utility>

namespace falcor {

namespace {

// Only inspect compiled constants. Class parameters must remain potentially emissive,
// even when their current argument values are zero, because they can be edited at runtime.
mi::base::Handle<const mi::neuraylib::IValue>
constant_expression_value(const mi::neuraylib::ICompiled_material* material, const char* path)
{
    mi::base::Handle<const mi::neuraylib::IExpression> expression(material->lookup_sub_expression(path));
    while (expression && expression->get_kind() == mi::neuraylib::IExpression::EK_TEMPORARY) {
        mi::base::Handle<const mi::neuraylib::IExpression_temporary> temporary(
            expression->get_interface<mi::neuraylib::IExpression_temporary>()
        );
        expression
            = mi::base::Handle<const mi::neuraylib::IExpression>(material->get_temporary(temporary->get_index()));
    }
    mi::base::Handle<const mi::neuraylib::IExpression_constant> constant(
        expression ? expression->get_interface<mi::neuraylib::IExpression_constant>() : nullptr
    );
    return mi::base::Handle<const mi::neuraylib::IValue>(constant ? constant->get_value() : nullptr);
}

bool constant_expression_is_zero(const mi::neuraylib::ICompiled_material* material, const char* path)
{
    const auto intensity = constant_expression_value(material, path);
    mi::base::Handle<const mi::neuraylib::IValue_float> scalar(
        intensity ? intensity->get_interface<mi::neuraylib::IValue_float>() : nullptr
    );
    if (scalar)
        return scalar->get_value() == 0.f;
    mi::base::Handle<const mi::neuraylib::IValue_color> color(
        intensity ? intensity->get_interface<mi::neuraylib::IValue_color>() : nullptr
    );
    if (!color)
        return false;
    for (mi::Size i = 0; i < 3; ++i) {
        mi::base::Handle<const mi::neuraylib::IValue_float> channel(color->get_value(i));
        if (channel->get_value() != 0.f)
            return false;
    }
    return true;
}

bool surface_emission_is_structurally_zero(const mi::neuraylib::ICompiled_material* material)
{
    const auto edf = constant_expression_value(material, "surface.emission.emission");
    return (edf && edf->get_kind() == mi::neuraylib::IValue::VK_INVALID_DF)
        || constant_expression_is_zero(material, "surface.emission.intensity");
}

bool argument_layouts_match(
    const mi::neuraylib::ITarget_value_layout* a,
    const mi::neuraylib::ITarget_value_layout* b,
    mi::neuraylib::Target_value_layout_state a_state = {},
    mi::neuraylib::Target_value_layout_state b_state = {}
)
{
    if (!a || !b || a->get_size() != b->get_size())
        return false;
    mi::neuraylib::IValue::Kind a_kind, b_kind;
    mi::Size a_size, b_size;
    const auto a_offset = a->get_layout(a_kind, a_size, a_state);
    const auto b_offset = b->get_layout(b_kind, b_size, b_state);
    if (a_offset == mi::Size(~0) || b_offset == mi::Size(~0) || a_offset != b_offset || a_size != b_size
        || a_kind != b_kind)
        return false;
    const auto count = a->get_num_elements(a_state);
    if (count != b->get_num_elements(b_state))
        return false;
    for (mi::Size i = 0; i < count; ++i) {
        if (!argument_layouts_match(a, b, a->get_nested_state(i, a_state), b->get_nested_state(i, b_state)))
            return false;
    }
    return true;
}

// The SDK state mask omits scene-data callbacks. Reject scene/state intrinsics and
// opaque calls before evaluating an expression without a hit.
bool expression_is_state_independent(
    const mi::neuraylib::ICompiled_material* material,
    mi::neuraylib::ITransaction* transaction,
    const char* path
)
{
    using namespace mi::neuraylib;
    using Expression = mi::base::Handle<const IExpression>;
    std::vector<Expression> pending;
    pending.emplace_back(material->lookup_sub_expression(path));
    std::set<mi::Size> visited_temporaries;
    while (!pending.empty()) {
        Expression expression = std::move(pending.back());
        pending.pop_back();
        if (!expression)
            return false;
        switch (expression->get_kind()) {
        case IExpression::EK_CONSTANT:
        case IExpression::EK_PARAMETER:
            break;
        case IExpression::EK_TEMPORARY: {
            mi::base::Handle<const IExpression_temporary> temporary(expression->get_interface<IExpression_temporary>());
            if (visited_temporaries.insert(temporary->get_index()).second)
                pending.emplace_back(material->get_temporary(temporary->get_index()));
            break;
        }
        case IExpression::EK_DIRECT_CALL: {
            mi::base::Handle<const IExpression_direct_call> call(expression->get_interface<IExpression_direct_call>());
            mi::base::Handle<const IFunction_definition> definition(
                transaction->access<IFunction_definition>(call->get_definition())
            );
            if (!definition)
                return false;
            const auto semantic = definition->get_semantic();
            using FD = IFunction_definition;
            if (semantic == FD::DS_UNKNOWN
                || (semantic >= FD::DS_INTRINSIC_STATE_FIRST && semantic <= FD::DS_INTRINSIC_STATE_LAST)
                || (semantic >= FD::DS_INTRINSIC_SCENE_FIRST && semantic <= FD::DS_INTRINSIC_SCENE_LAST))
                return false;
            mi::base::Handle<const IExpression_list> arguments(call->get_arguments());
            for (mi::Size i = 0; i < arguments->get_size(); ++i)
                pending.emplace_back(arguments->get_expression(i));
            break;
        }
        default:
            return false;
        }
    }
    return true;
}

struct EmissionZeroCheck {
    std::string value_path;
    std::string condition_path;
    bool condition_value;
};

std::vector<EmissionZeroCheck>
emission_zero_checks(const mi::neuraylib::ICompiled_material* material, mi::neuraylib::ITransaction* transaction)
{
    using namespace mi::neuraylib;
    using Expression = mi::base::Handle<const IExpression>;
    std::vector<EmissionZeroCheck> checks;
    auto add_check = [&](const std::string& path, const std::string& condition = "", bool condition_value = false)
    {
        Expression expression(material->lookup_sub_expression(path.c_str()));
        if (!expression)
            return false;
        mi::base::Handle<const IType> type(expression->get_type());
        mi::base::Handle<const IType> unaliased(type->skip_all_type_aliases());
        if ((unaliased->get_kind() != IType::TK_FLOAT && unaliased->get_kind() != IType::TK_COLOR)
            || !expression_is_state_independent(material, transaction, path.c_str()))
            return false;
        if (!constant_expression_value(material, path.c_str()) || constant_expression_is_zero(material, path.c_str()))
            checks.push_back({path, condition, condition_value});
        return true;
    };

    std::vector<std::pair<Expression, std::string>> pending;
    pending.emplace_back(material->lookup_sub_expression("surface.emission.intensity"), "surface.emission.intensity");
    std::set<mi::Size> visited_temporaries;
    while (!pending.empty()) {
        auto [expression, path] = std::move(pending.back());
        pending.pop_back();
        while (expression && expression->get_kind() == IExpression::EK_TEMPORARY) {
            mi::base::Handle<const IExpression_temporary> temporary(expression->get_interface<IExpression_temporary>());
            if (!visited_temporaries.insert(temporary->get_index()).second) {
                expression.reset();
                break;
            }
            expression = Expression(material->get_temporary(temporary->get_index()));
        }
        if (!expression || add_check(path))
            continue;
        mi::base::Handle<const IExpression_direct_call> call(expression->get_interface<IExpression_direct_call>());
        if (!call)
            continue;
        mi::base::Handle<const IFunction_definition> definition(
            transaction->access<IFunction_definition>(call->get_definition())
        );
        if (!definition)
            continue;
        mi::base::Handle<const IExpression_list> arguments(call->get_arguments());
        if (definition->get_semantic() == IFunction_definition::DS_MULTIPLY) {
            for (mi::Size i = 0; i < arguments->get_size(); ++i)
                pending.emplace_back(arguments->get_expression(i), path + "." + arguments->get_name(i));
        } else if (definition->get_semantic() == IFunction_definition::DS_TERNARY) {
            const std::string condition_path = path + "." + arguments->get_name(0);
            if (expression_is_state_independent(material, transaction, condition_path.c_str())) {
                for (mi::Size i = 1; i < 3; ++i)
                    add_check(path + "." + arguments->get_name(i), condition_path, i == 1);
            }
            // Do not descend into branches: their factors would need the enclosing guard.
        }
    }
    return checks;
}

MDLContext::TextureSource::Type texture_type_from_mdl_shape(mi::neuraylib::IType_texture::Shape shape)
{
    using Shape = mi::neuraylib::IType_texture::Shape;
    using Type = MDLContext::TextureSource::Type;
    switch (shape) {
    case Shape::TS_2D:
        return Type::Texture2D;
    case Shape::TS_3D:
        return Type::Texture3D;
    case Shape::TS_CUBE:
        return Type::TextureCube;
    case Shape::TS_PTEX:
        return Type::TexturePtex;
    case Shape::TS_BSDF_DATA:
        return Type::TextureBSDFData;
    default:
        return Type::Unknown;
    }
}

MDLContext::TextureSource::Type texture_type_from_mdl_shape(mi::neuraylib::ITarget_code::Texture_shape shape)
{
    using Shape = mi::neuraylib::ITarget_code::Texture_shape;
    using Type = MDLContext::TextureSource::Type;
    switch (shape) {
    case Shape::Texture_shape_2d:
        return Type::Texture2D;
    case Shape::Texture_shape_3d:
        return Type::Texture3D;
    case Shape::Texture_shape_cube:
        return Type::TextureCube;
    case Shape::Texture_shape_ptex:
        return Type::TexturePtex;
    case Shape::Texture_shape_bsdf_data:
        return Type::TextureBSDFData;
    default:
        return Type::Unknown;
    }
}

// The SDK encodes captured strings as IDs, just like target-code string constants. Defaults may introduce
// strings absent from that table; preserve its indices and append these before handing the table to the host.
class ArgumentResourceCallback : public mi::base::Interface_implement<mi::neuraylib::ITarget_resource_callback> {
public:
    ArgumentResourceCallback(
        mi::neuraylib::ITransaction* transaction,
        const mi::neuraylib::ITarget_code* target_code,
        std::vector<std::string>& strings,
        std::vector<MDLContext::CompileResult::Texture>& textures
    )
        : m_transaction(transaction)
        , m_target_code(target_code)
        , m_strings(strings)
        , m_textures(textures)
    {
        if (m_strings.empty())
            m_strings.emplace_back(); // ID zero always denotes an unknown string.
    }

    // The SDK calls this while building each instance's argument block. A cached target only knows
    // the defaults of the instance that created it: if A defaults to brick.png and B to wood.png,
    // get_known_resource_index() returns 0 (invalid) for B's wood.png despite their identical shader graph.
    // Reuse or append argument textures in the current instance's table instead. Even a known default
    // cannot blindly reuse its cached ID, since unused defaults may have been trimmed from that table.
    // Body textures are different: e.g. a fixed noise.png at ID 1 is referenced directly by generated
    // shader code, so its ID must stay unchanged. The table is seeded to preserve these IDs before this
    // callback runs. New entries record texture metadata here; image extraction/loading happens later.
    mi::Uint32 get_resource_index(const mi::neuraylib::IValue_resource* resource) override
    {
        if (const auto index = m_target_code->get_known_resource_index(m_transaction, resource)) {
            if (resource->get_kind() == mi::neuraylib::IValue::VK_TEXTURE
                && m_target_code->get_texture_is_body_resource(index))
                return index;
        }
        const char* name = resource->get_value();
        if (!name || !*name)
            return 0;

        mi::base::Handle<const mi::neuraylib::IValue_texture> value(
            resource->get_interface<mi::neuraylib::IValue_texture>()
        );
        FALCOR_CHECK(value, "Light profiles and BSDF measurements are not supported");
        mi::base::Handle<const mi::neuraylib::IType_texture> type(value->get_type());
        const auto shape = texture_type_from_mdl_shape(type->get_shape());
        for (const auto& texture : m_textures) {
            if (texture.name == name && texture.source.type == shape)
                return texture.index;
        }
        const auto index = sgl::narrow_cast<mi::Uint32>(m_textures.size() + 1);
        m_textures.push_back({
            .index = index,
            .name = name,
            .source = {.type = shape, .srgb = value->get_gamma() == 2.2f},
        });
        return index;
    }

    mi::Uint32 get_string_index(const mi::neuraylib::IValue_string* value) override
    {
        const std::string_view name(value->get_value());
        if (name.empty())
            return 0;
        const auto it = std::find(m_strings.begin() + 1, m_strings.end(), name);
        if (it != m_strings.end())
            return sgl::narrow_cast<mi::Uint32>(it - m_strings.begin());
        const auto index = sgl::narrow_cast<mi::Uint32>(m_strings.size());
        m_strings.emplace_back(name);
        return index;
    }

private:
    mi::neuraylib::ITransaction* m_transaction;
    const mi::neuraylib::ITarget_code* m_target_code;
    std::vector<std::string>& m_strings;
    std::vector<MDLContext::CompileResult::Texture>& m_textures;
};

} // namespace

struct Tagged_bsdf_info {
    struct Input {
        Input() { };
        Input(std::string _path)
            : path(_path)
            , name(_path)
        {
            std::replace(name.begin(), name.end(), '.', '_');
        }

        std::string path;
        std::string name;
    };

    Input roughness_u;
    Input roughness_v;
    Input tangent_u;
    Input tint;
};

// The SDK owns graph equivalence. Exact option/function records cover the remaining
// translation inputs; do not duplicate CompileRequest fields in a second hash recipe.
struct TargetCodeKey {
    std::array<uint32_t, 4> material_hash;
    std::vector<std::pair<std::string, std::string>> backend_options;
    std::vector<std::vector<std::pair<std::string, std::string>>> function_groups;
    auto operator<=>(const TargetCodeKey&) const = default;
};
struct TargetCodeEntry {
    struct FunctionIndices {
        mi::Size function_index;
        mi::Size argument_block_index;
    };
    mi::base::Handle<const mi::neuraylib::ITarget_code> code;
    std::vector<FunctionIndices> function_indices;
};
using TargetCodeCache = std::map<TargetCodeKey, TargetCodeEntry>;

struct MDLContextImpl : public MDLContext::IImpl {
    ~MDLContextImpl() override
    {
        target_code_cache.clear();
        if (neuray && neuray->shutdown(true) != 0)
            sgl::log_error("Failed to shutdown the MDL SDK.");

        neuray = nullptr; // free the handles that holds the INeuray instance
        mdl_utils::unload_mdl(mdl);
    }

    MDLContextImpl()
    {
        const auto mdl_library = platform::runtime_directory() / ("libmdl_sdk" MI_BASE_DLL_FILE_EXT);
        if (std::string error; !mdl_utils::load_mdl_and_get_ineuray(mdl, neuray, error, mdl_library.string().c_str())
            || !neuray.is_valid_interface())
            FALCOR_THROW("Failed to load the MDL SDK:\n{}", error);

        // Currently textures will be loaded twice in system memory, once by the MDL plugin and again by the
        // Falcor TextureManager.Unfortunately we can't easily omit this redundant MDL loading and storage
        // as the MDL compiler uses image metadata when performing constant folding for standard texture functions
        // such as width, height, etc.So far, materials relying on these intrinsics were broken( all textures had
        // resolution 1x1 for the MDL compiler ).
        //
        // TODO: To eliminate the redundant texture load in memory, we should either
        // 1) encapsulate the Falcor TextureManager in a custom MDL plugin, or
        // 2) write a stub plugin which loads the image metadata but doesn't allocate and load the image data in memory.
        if (!mdl_utils::load_mdl_plugin(
                neuray.get(),
                (platform::runtime_directory() / ("nv_openimageio" MI_BASE_DLL_FILE_EXT)).string().c_str()
            ))
            FALCOR_THROW("Failed to load the 'nv_openimageio' plugin.\n");
        if (!mdl_utils::load_mdl_plugin(
                neuray.get(),
                (platform::runtime_directory() / ("dds" MI_BASE_DLL_FILE_EXT)).string().c_str()
            ))
            FALCOR_THROW("Failed to load the 'dds' plugin.\n");

        mdl_utils::ConfigureOptions configure_options;
        configure_options.add_admin_space_search_paths = true;
        configure_options.add_user_space_search_paths = false;

        // // Add user search paths.
        // for (const auto& path : Settings::getGlobalSettings().getSearchDirectories("mdl"))
        //     configure_options.additional_mdl_paths.push_back(path.string());

        // Add search path to MDL core definitions.
        configure_options.additional_mdl_paths.push_back((platform::runtime_directory() / "mdl").string());
        configure_options.additional_mdl_paths.push_back(
            (platform::runtime_directory() / "mdl_libraries/omni_pbr").string()
        );

        if (std::string error; !mdl_utils::configure_mdl(neuray.get(), configure_options, error))
            FALCOR_THROW("Failed to configure the MDL SDK:\n{}", error);

        mi::Sint32 result = neuray->start(true);
        if (result != 0)
            FALCOR_THROW("Failed to initialize the MDL SDK. Result code: {}", result);
    }

    sgl::SharedLibraryHandle mdl;
    mi::base::Handle<mi::neuraylib::INeuray> neuray;
    // Modified materials outlive individual compile requests in the MDL database, so their tagged-BSDF metadata must
    // have the same lifetime. The outer key is the generated material name; the inner key is the unique DF handle.
    std::map<std::string, std::map<std::string, Tagged_bsdf_info>> tagged_bsdf_info_cache;
    TargetCodeCache target_code_cache;

    void* get_neuray() override { return neuray.get(); };
};

struct MDLSearchPathScope::Impl {
    std::unique_lock<std::recursive_mutex> lock;
    mi::base::Handle<mi::neuraylib::IMdl_configuration> config;
    std::vector<std::string> module_paths;
    std::vector<std::string> resource_paths;
    bool changed = false;

    explicit Impl(std::recursive_mutex& mutex)
        : lock(mutex)
    {
    }

    ~Impl() noexcept
    {
        if (!changed)
            return;
        // SDK mutation does not throw. Try every restore even when one path disappeared.
        config->clear_mdl_paths();
        config->clear_resource_paths();
        bool restored = true;
        for (const auto& path : module_paths)
            restored = (config->add_mdl_path(path.c_str()) == 0) && restored;
        for (const auto& path : resource_paths)
            restored = (config->add_resource_path(path.c_str()) == 0) && restored;
        if (!restored) {
            try {
                sgl::log_error("Failed to restore an MDL search path; a directory may have disappeared.");
            } catch (...) {
            }
        }
    }
};

MDLSearchPathScope::MDLSearchPathScope(MDLContext& context, std::span<const std::filesystem::path> paths)
    : m_impl(std::make_unique<Impl>(context.m_mutex))
{
    if (paths.empty())
        return;
    auto* neuray = static_cast<mi::neuraylib::INeuray*>(context.get_neuray());
    FALCOR_CHECK(neuray, "MDL context has no SDK implementation.");
    auto& state = *m_impl;
    state.config = neuray->get_api_component<mi::neuraylib::IMdl_configuration>();
    auto& config = *state.config;
    for (mi::Size i = 0; i < config.get_mdl_paths_length(); ++i) {
        mi::base::Handle<const mi::IString> path(config.get_mdl_path(i));
        state.module_paths.emplace_back(path->get_c_str());
    }
    for (mi::Size i = 0; i < config.get_resource_paths_length(); ++i) {
        mi::base::Handle<const mi::IString> path(config.get_resource_path(i));
        state.resource_paths.emplace_back(path->get_c_str());
    }
    auto prepend = [&](const std::vector<std::string>& previous)
    {
        std::vector<std::filesystem::path> result;
        auto append = [&](const std::filesystem::path& path)
        {
            FALCOR_CHECK(!path.empty(), "MDL search root must not be empty.");
            const auto normalized = std::filesystem::weakly_canonical(std::filesystem::absolute(path));
            FALCOR_CHECK(std::filesystem::is_directory(normalized), "MDL search root '{}' is not a directory.", path);
            for (const auto& existing : result) {
                if (std::filesystem::equivalent(existing, normalized))
                    return;
            }
            result.push_back(normalized);
        };
        for (const auto& path : paths)
            append(path);
        for (const auto& path : previous)
            append(path);
        return result;
    };
    const auto modules = prepend(state.module_paths);
    const auto resources = prepend(state.resource_paths);
    state.changed = true;
    config.clear_mdl_paths();
    config.clear_resource_paths();
    for (const auto& path : modules)
        FALCOR_CHECK(config.add_mdl_path(path.string().c_str()) == 0, "Failed to add MDL search root '{}'.", path);
    for (const auto& path : resources)
        FALCOR_CHECK(
            config.add_resource_path(path.string().c_str()) == 0,
            "Failed to add MDL resource root '{}'.",
            path
        );
}

MDLSearchPathScope::~MDLSearchPathScope() noexcept = default;

/// Return the float component count, or zero for unsupported AOV types.
uint32_t get_material_aov_component_count(const mi::neuraylib::IType* source_type)
{
    mi::base::Handle<const mi::neuraylib::IType> type(source_type->skip_all_type_aliases());
    switch (type->get_kind()) {
    case mi::neuraylib::IType::TK_FLOAT:
        return 1;
    case mi::neuraylib::IType::TK_COLOR:
        return 3;
    case mi::neuraylib::IType::TK_VECTOR: {
        mi::base::Handle<const mi::neuraylib::IType_vector> vector_type(
            type->get_interface<mi::neuraylib::IType_vector>()
        );
        mi::base::Handle<const mi::neuraylib::IType> element_type(vector_type->get_element_type());
        element_type = mi::base::Handle<const mi::neuraylib::IType>(element_type->skip_all_type_aliases());
        if (element_type->get_kind() != mi::neuraylib::IType::TK_FLOAT)
            return 0;
        return uint32_t(vector_type->get_size());
    }
    default:
        return 0;
    }
}

inline sgl::LogLevel severity_to_log_level(mi::base::details::Message_severity severity)
{
    switch (severity) {
    case mi::base::MESSAGE_SEVERITY_FATAL:
        return sgl::LogLevel::fatal;
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return sgl::LogLevel::error;
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return sgl::LogLevel::warn;
    case mi::base::MESSAGE_SEVERITY_INFO:
        return sgl::LogLevel::info;
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return sgl::LogLevel::debug;
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return sgl::LogLevel::debug;
    default:
        return sgl::LogLevel::debug;
    }
}

inline void
log_message(mi::base::details::Message_severity severity, mi::neuraylib::IMessage::Kind kind, const char* msg)
{
    sgl::Logger::get().log(severity_to_log_level(severity), fmt::format("{}: {}", mdl_utils::to_string(kind), msg));
}

/// Log all messages from the given context and return true if there were any errors.
inline bool log_messages_and_check_errors(mi::neuraylib::IMdl_execution_context* context)
{
    for (mi::Size i = 0, n = context->get_messages_count(); i < n; ++i) {
        mi::base::Handle<const mi::neuraylib::IMessage> message(context->get_message(i));
        log_message(message->get_severity(), message->get_kind(), message->get_string());
    }
    return context->get_error_messages_count() > 0;
}

MDLContext& MDLContext::get()
{
    static MDLContext* context = []()
    {
        MDLContext* ctx = new MDLContext;
        register_shutdown_callback(
            [ctx]()
            {
                delete ctx;
            }
        );
        return ctx;
    }();
    return *context;
}

MDLContext::MDLContext()
{
    m_impl = std::make_unique<MDLContextImpl>();
}

MDLContext::~MDLContext() { }

struct ModifiedParameterNames {
    // Names reported by the original compiled material. These remain the public names in Falcor, but may be paths
    // such as "tint.r" that cannot be used as parameter identifiers in the generated MDL material.
    std::vector<std::string> source;
    // Unique MDL identifiers used for the corresponding parameters in the generated material.
    std::vector<std::string> generated;
    // Name allocated for Falcor's additional mollification parameter, when enabled.
    std::optional<std::string> mollify;
};

// Build the one-to-one parameter-name mapping used when cloning a compiled material into a generated MDL module.
// The mapping must be deterministic because a previously generated module may be reused from the MDL database while
// this mapping is rebuilt to interpret its argument block.
inline bool build_modified_parameter_names(
    const MDLContext::CompileRequest& request,
    const mi::neuraylib::ICompiled_material* src_material,
    mi::neuraylib::IMdl_factory* factory,
    ModifiedParameterNames& names,
    std::string& out_error_msg
)
{
    const mi::Size parameter_count = src_material->get_parameter_count();
    names.source.resize(parameter_count);
    names.generated.resize(parameter_count);

    // Reserve every usable source name before allocating aliases. For example, if parameter 0 is the path "tint.r"
    // and a later source parameter is named "compiled_parameter_0", a one-pass allocator would give both parameters
    // the same generated name.
    std::set<std::string, std::less<>> used_parameter_names;
    for (mi::Size i = 0; i < parameter_count; ++i) {
        const char* name = src_material->get_parameter_name(i);
        if (name == nullptr) {
            out_error_msg = "Unexpected nameless parameter";
            return false;
        }
        names.source[i] = name;
        if (factory->is_valid_mdl_identifier(name)) {
            if (!used_parameter_names.insert(name).second) {
                out_error_msg = fmt::format("Duplicate compiled parameter name '{}'", name);
                return false;
            }
            names.generated[i] = name;
        }
    }

    auto allocate_parameter_name = [&used_parameter_names](const std::string& base_name)
    {
        // The preferred name is normally available, so this path performs no suffix formatting.
        if (used_parameter_names.insert(base_name).second)
            return base_name;

        // Every suffixed candidate can itself be a valid source name (for example, "mollify_1"), so probe until an
        // unused identifier is found. Each generated positional alias has a distinct base, so caching the next suffix
        // per base would not avoid work in this call site.
        for (size_t suffix = 1;; ++suffix) {
            std::string name = fmt::format("{}_{}", base_name, suffix);
            if (used_parameter_names.insert(name).second)
                return name;
        }
    };

    // Compiled-material paths are not legal declaration identifiers. Give each one a stable positional base name;
    // allocate_parameter_name() adds a suffix only if that base is already a source parameter.
    for (mi::Size i = 0; i < parameter_count; ++i) {
        if (names.generated[i].empty())
            names.generated[i] = allocate_parameter_name(fmt::format("compiled_parameter_{}", i));
    }

    // Falcor's injected parameter shares the generated material's namespace with source parameters and aliases.
    if (request.options.enable_param_mollification)
        names.mollify = allocate_parameter_name("mollify");

    return true;
}

inline bool build_modified_material(
    const MDLContext::CompileRequest& request,
    const std::string dst_simple_material_name,
    const mi::neuraylib::ICompiled_material* src_material,
    const mi::neuraylib::IFunction_definition* src_definition,
    const mi::neuraylib::IExpression_list* instance_args,
    const ModifiedParameterNames& parameter_names,
    mi::neuraylib::IMdl_factory* factory,
    mi::neuraylib::IMdl_execution_context* context,
    mi::neuraylib::ITransaction* trans,
    std::map<std::string, Tagged_bsdf_info>& tagged_bsdf_infos,
    std::string& out_error_msg
)
{
    mi::base::Handle<mi::neuraylib::IExpression_factory> ef(factory->create_expression_factory(trans));
    mi::base::Handle<mi::neuraylib::IValue_factory> vf(factory->create_value_factory(trans));
    mi::base::Handle<mi::neuraylib::IType_factory> tf(vf->get_type_factory());

    mi::base::Handle<mi::neuraylib::IValue> epsilon_value(vf->create_float(0.01f));
    mi::base::Handle<mi::neuraylib::IValue> zero_value(vf->create_float(0.0f));
    mi::base::Handle<mi::neuraylib::IValue> one_value(vf->create_float(1.0f));

    mi::base::Handle<const mi::neuraylib::IExpression> epsilon_expression(ef->create_constant(epsilon_value.get()));
    mi::base::Handle<const mi::neuraylib::IExpression> zero_expression(ef->create_constant(zero_value.get()));
    mi::base::Handle<const mi::neuraylib::IExpression> one_expression(ef->create_constant(one_value.get()));

    mi::base::Handle<const mi::neuraylib::IExpression_direct_call> src_body(src_material->get_body());

    mi::base::Handle<const mi::neuraylib::IType_list> src_parameters(src_definition->get_parameter_types());
    mi::base::Handle<const mi::neuraylib::IAnnotation_list> src_parameter_annotations(
        src_definition->get_parameter_annotations()
    );
    mi::base::Handle<const mi::neuraylib::IAnnotation_block> src_annotations(src_definition->get_annotations());
    mi::base::Handle<const mi::neuraylib::IAnnotation_block> src_return_annotations(
        src_definition->get_return_annotations()
    );

    mi::base::Handle<mi::neuraylib::IType_list> dst_parameters(tf->create_type_list());
    mi::base::Handle<mi::neuraylib::IExpression_list> dst_defaults(ef->create_expression_list());
    mi::base::Handle<mi::neuraylib::IAnnotation_list> dst_parameter_annotations(ef->create_annotation_list());

    // Build Falcor's diagnostic clone in a separate module. Editing the source module forces the MDL SDK to
    // reanalyze unrelated source declarations and can fail on otherwise valid imports that were already resolved
    // when the original module was loaded. The modified material name contains the compiled-material and option
    // hashes, so it also gives the generated module stable, collision-resistant identity.
    const std::string dst_module_db_name = fmt::format("mdl::falcor_modified_{}", dst_simple_material_name);

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(factory->create_module_builder(
        trans,
        dst_module_db_name.c_str(),
        /*ignored*/ mi::neuraylib::MDL_VERSION_1_0,
        mi::neuraylib::MDL_VERSION_LATEST,
        context
    ));
    if (!module_builder) {
        out_error_msg = "Failed to create module for modified material";
        return false;
    }

    const mi::Size parameter_count = src_material->get_parameter_count();
    FALCOR_ASSERT(parameter_names.source.size() == parameter_count);
    FALCOR_ASSERT(parameter_names.generated.size() == parameter_count);

    // clone parameter defaults
    for (mi::Size i = 0; i < parameter_count; ++i) {
        const std::string& name = parameter_names.source[i];
        const std::string& dst_name = parameter_names.generated[i];

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> src_param_annotation(
            src_parameter_annotations->get_annotation_block(name.c_str())
        );
        if (src_param_annotation) {
            if (mi::Sint32 result
                = dst_parameter_annotations->add_annotation_block(dst_name.c_str(), src_param_annotation.get())) {
                out_error_msg
                    = fmt::format("Failed to clone annotation block for parameter {} ('{}'): {}", i, name, result);
                return false;
            }
        }

        mi::base::Handle<const mi::neuraylib::IValue> compiled_default(src_material->get_argument(i));
        if (!compiled_default) {
            out_error_msg = fmt::format("Failed to get default for parameter {} ('{}')", i, name);
            return false;
        }

        mi::base::Handle<const mi::neuraylib::IType> src_type(src_parameters->get_type(name.c_str()));
        if (!src_type)
            src_type = compiled_default->get_type();
        if (!src_type) {
            out_error_msg = fmt::format("Failed to find type for parameter {} ('{}')", i, name);
            return false;
        }
        if (mi::Sint32 result = dst_parameters->add_type(dst_name.c_str(), src_type.get())) {
            out_error_msg = fmt::format("Failed to clone type for parameter {} ('{}'): {}", i, name, result);
            return false;
        }

        // If there is a provided instance argument with the same name, use it to set the default value.
        // Otherwise, use the value specified by the original source argument.
        mi::base::Handle<const mi::neuraylib::IExpression> arg(
            instance_args ? instance_args->get_expression(name.c_str()) : nullptr
        );
        if (arg) {
            if (mi::Sint32 result = dst_defaults->add_expression(dst_name.c_str(), arg.get())) {
                out_error_msg
                    = fmt::format("Failed to clone instance default for parameter {} ('{}'): {}", i, name, result);
                return false;
            }
        } else {
            mi::base::Handle<const mi::neuraylib::IExpression> dst_default(ef->create_constant(compiled_default.get()));
            if (!dst_default) {
                out_error_msg = fmt::format("Failed to create default for parameter {} ('{}')", i, name);
                return false;
            }
            if (mi::Sint32 result = dst_defaults->add_expression(dst_name.c_str(), dst_default.get())) {
                out_error_msg
                    = fmt::format("Failed to clone compiled default for parameter {} ('{}'): {}", i, name, result);
                return false;
            }
        }
    }

    mi::base::Handle<const mi::neuraylib::IExpression> specular_roughness;
    if (request.options.enable_param_mollification) {
        // add a mollification factor to approximate specular BSDFs

        // create range annotation for the mollification parameter
        mi::base::Handle<mi::neuraylib::IAnnotation_block> mollify_annotation_block(ef->create_annotation_block());
        mi::base::Handle<mi::neuraylib::IExpression_list> mollify_annotation_arguments(ef->create_expression_list());
        mollify_annotation_arguments->add_expression("min", zero_expression.get());
        mollify_annotation_arguments->add_expression("max", one_expression.get());
        mi::base::Handle<mi::neuraylib::IAnnotation> mollify_annotation(
            ef->create_annotation("::anno::hard_range(float,float)", mollify_annotation_arguments.get())
        );
        mollify_annotation_block->add_annotation(mollify_annotation.get());

        // add the mollification parameter
        FALCOR_ASSERT(parameter_names.mollify.has_value());
        const std::string& mollify_name = *parameter_names.mollify;
        mi::base::Handle<const mi::neuraylib::IType> mollify_type(tf->create_float());
        dst_parameters->add_type(mollify_name.c_str(), mollify_type.get());
        dst_defaults->add_expression(mollify_name.c_str(), epsilon_expression.get());
        dst_parameter_annotations->set_annotation_block(mollify_name.c_str(), mollify_annotation_block.get());

        // create the parameterized roughness expression used to approximate specular BSDFs
        mi::Size mollify_parameter_index = dst_parameters->get_index(mollify_name.c_str());
        mi::base::Handle<mi::neuraylib::IExpression_parameter> mollify_parameter(
            ef->create_parameter(mollify_type.get(), mollify_parameter_index)
        );

        specular_roughness = mollify_parameter;
    } else {
        // default
        specular_roughness = epsilon_expression;
    }

    // cloned temporary subexpression cache
    std::map<mi::Size, mi::base::Handle<const mi::neuraylib::IExpression>> cloned_temporaries;

    // counter used to generate unique handles for all material components
    size_t handleCount = 0;

    std::function<mi::base::Handle<
        const mi::neuraylib::IExpression>(mi::base::Handle<const mi::neuraylib::IExpression>, const std::string& path)>
        clone;
    clone = [&](mi::base::Handle<const mi::neuraylib::IExpression> src_expression,
                const std::string& path) -> mi::base::Handle<const mi::neuraylib::IExpression>
    {
        mi::neuraylib::IExpression::Kind kind = src_expression->get_kind();

        // reconstruct original dag
        if (kind == mi::neuraylib::IExpression::EK_TEMPORARY) {
            mi::base::Handle<const mi::neuraylib::IExpression_temporary> temporary(
                src_expression->get_interface<const mi::neuraylib::IExpression_temporary>()
            );
            mi::Size index = temporary->get_index();

            // cache the cloned temporary expressions to reconstruct the original dag
            mi::base::Handle<const mi::neuraylib::IExpression> dst_expression;
            auto result = cloned_temporaries.insert({index, dst_expression});

            if (result.second) {
                mi::base::Handle<const mi::neuraylib::IExpression> src_temporary(src_material->get_temporary(index));
                dst_expression = clone(src_temporary, path);
                result.first->second = dst_expression;
            } else {
                dst_expression = result.first->second;
            }

            return dst_expression;
        } else if (kind == mi::neuraylib::IExpression::EK_CONSTANT) {
            // no need to clone literals.
            return src_expression;
        } else if (kind == mi::neuraylib::IExpression::EK_PARAMETER) {
            // no need to clone parameters.
            return src_expression;
        } else if (kind == mi::neuraylib::IExpression::EK_DIRECT_CALL) {
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> src_direct_call(
                src_expression->get_interface<const mi::neuraylib::IExpression_direct_call>()
            );
            mi::base::Handle<const mi::neuraylib::IFunction_definition> func_def(
                trans->access<mi::neuraylib::IFunction_definition>(src_direct_call->get_definition())
            );
            mi::neuraylib::IFunction_definition::Semantics src_semantic = func_def->get_semantic();
            mi::neuraylib::IFunction_definition::Semantics dst_semantic = src_semantic;

            std::string dst_name(src_direct_call->get_definition());

            mi::base::Handle<const mi::neuraylib::IExpression_list> src_arguments(src_direct_call->get_arguments());
            mi::base::Handle<mi::neuraylib::IExpression_list> dst_arguments(ef->create_expression_list());

            // Approximate specular bsdf by a highly glossy bsdf
            if (request.options.speculars_as_glossy
                && src_semantic == mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_SPECULAR_BSDF) {
                dst_name = "mdl::df::simple_glossy_bsdf(float,float,color,color,float3,::df::scatter_mode,string)";
                func_def = mi::base::Handle<const mi::neuraylib::IFunction_definition>(
                    trans->access<mi::neuraylib::IFunction_definition>(dst_name.c_str())
                );

                dst_semantic = mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF;

                // Set near-specular roughness
                dst_arguments->add_expression("roughness_u", specular_roughness.get());
            }

            // recursively clone all arguments
            mi::Size num_arguments = src_arguments->get_size();
            for (mi::Size i = 0; i < num_arguments; ++i) {
                const char* name = src_arguments->get_name(i);

                // skip all handles. we will add new unique handles.
                if (request.options.enable_tagged_df_properties && strcmp(name, "handle") == 0)
                    continue;

                // don't add a dot if this is the root direct call of a material
                std::string argument_path = (path.empty() ? "" : path + ".") + name;

                mi::base::Handle<const mi::neuraylib::IExpression> src_argument(
                    src_arguments->get_expression<const mi::neuraylib::IExpression>(i)
                );
                mi::base::Handle<const mi::neuraylib::IExpression> dst_argument = clone(src_argument, argument_path);

                // Cap the roughness of glossy lobes
                if (request.options.speculars_as_glossy) {
                    switch (src_semantic) {
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
                    case mi::neuraylib::IFunction_definition::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF:
                        if (strcmp(name, "roughness_u") == 0 || strcmp(name, "roughness_v") == 0) {
                            mi::Sint32 result;

                            mi::base::Handle<mi::neuraylib::IExpression_list> max_arguments(
                                ef->create_expression_list()
                            );
                            if ((result = max_arguments->add_expression("a", dst_argument.get())) != 0)
                                return {};
                            if ((result = max_arguments->add_expression("b", specular_roughness.get())) != 0)
                                return {};

                            dst_argument = mi::base::Handle<const mi::neuraylib::IExpression_direct_call>(
                                ef->create_direct_call("mdl::math::max(float,float)", max_arguments.get(), &result)
                            );

                            if (result) {
                                return {};
                            }
                        }
                        break;
                    default:
                        // Roughness arguments for functions other than microfacet and ward ds may have different
                        // semantics, so we ignore them here.
                        break;
                    }
                }

                mi::Sint32 result = dst_arguments->add_expression(name, dst_argument.get());
                if (result) {
                    out_error_msg = "Failed to clone expression";
                    return {};
                }
            }

            // if this call accepts handle as parameter, set a unique handle
            // we could use the function semantics to limit this to dfs
            if (request.options.enable_tagged_df_properties && func_def->get_parameter_index("handle") != -1) {
                std::string tag = "tag_" + std::to_string(handleCount++);

                Tagged_bsdf_info info;

                bool isBsdf = true;
                switch (dst_semantic) {
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_DIFFUSE_REFLECTION_BSDF:
                    info.roughness_u = Tagged_bsdf_info::Input(path + ".roughness");
                    info.tint = Tagged_bsdf_info::Input(path + ".tint");
                    break;
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_DIFFUSE_TRANSMISSION_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_SPECULAR_BSDF:
                    info.tint = Tagged_bsdf_info::Input(path + ".tint");
                    break;
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_SIMPLE_GLOSSY_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::
                    DS_INTRINSIC_DF_BACKSCATTERING_GLOSSY_REFLECTION_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_MICROFACET_BECKMANN_SMITH_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_MICROFACET_GGX_SMITH_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_MICROFACET_BECKMANN_VCAVITIES_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_MICROFACET_GGX_VCAVITIES_BSDF:
                case mi::neuraylib::IFunction_definition::Semantics::DS_INTRINSIC_DF_WARD_GEISLER_MORODER_BSDF: {
                    info.roughness_u = Tagged_bsdf_info::Input(path + ".roughness_u");
                    info.roughness_v = Tagged_bsdf_info::Input(path + ".roughness_v");
                    info.tangent_u = Tagged_bsdf_info::Input(path + ".tangent_u");
                    info.tint = Tagged_bsdf_info::Input(path + ".tint");
                } break;
                default:
                    isBsdf = false;
                    break;
                };

                // Insert DF info tagged map
                if (isBsdf)
                    tagged_bsdf_infos.emplace(tag, std::move(info));

                mi::base::Handle<mi::neuraylib::IValue_string> value(vf->create_string(tag.c_str()));
                mi::base::Handle<mi::neuraylib::IExpression> handle(ef->create_constant(value.get()));
                mi::Sint32 result = dst_arguments->add_expression("handle", handle.get());
                if (result) {
                    out_error_msg = "Failed to add handle expression";
                    return {};
                }
            }

            mi::Sint32 result;
            mi::base::Handle<const mi::neuraylib::IExpression_direct_call> dst_direct_call(
                ef->create_direct_call(dst_name.c_str(), dst_arguments.get(), &result)
            );

            if (result != 0) {
                out_error_msg = "Cloning direct call expression failed";
                return {};
            }

            return dst_direct_call;
        } else if (kind == mi::neuraylib::IExpression::EK_CALL) {
            out_error_msg = "Cloning call expression not supported";
            return {};
        }

        return {};
    };

    mi::base::Handle<const mi::neuraylib::IExpression> dst_body(clone(src_body, ""));
    if (!dst_body)
        return false;

    // Add the cloned material to the module.
    // For complex materials, this tends to be the slowest operation.
    mi::Sint32 result = module_builder->add_function(
        dst_simple_material_name.c_str(),
        dst_body.get(),
        /*temporaries*/ nullptr,
        dst_parameters.get(),
        dst_defaults.get(),
        dst_parameter_annotations.get(),
        src_annotations.get(),
        src_return_annotations.get(),
        /*is_exported*/ true,
        /*is_declarative*/ false,
        /*frequency_qualifier*/ mi::neuraylib::IType::MK_NONE,
        context
    );
    if (result) {
        log_messages_and_check_errors(context);
        out_error_msg = "Failed to add modified material";
        return false;
    }

    // stores on close
    module_builder = nullptr;

    // instantiate cloned material
    mi::base::Handle<const mi::neuraylib::IModule> dst_module(
        trans->access<mi::neuraylib::IModule>(dst_module_db_name.c_str())
    );
    if (!dst_module) {
        out_error_msg = "Failed to access module for modified material";
        return false;
    }
    mi::base::Handle<const mi::IArray> dst_func_overloads(
        dst_module->get_function_overloads(dst_simple_material_name.c_str())
    );
    if (dst_func_overloads->get_length() != 1) {
        out_error_msg = "Failed to uniquely identify modified material definition";
        return false;
    }
    mi::base::Handle<const mi::IString> dst_func_db_name(dst_func_overloads->get_element<mi::IString>(0));

    mi::base::Handle<const mi::neuraylib::IFunction_definition> dst_definition(
        trans->access<mi::neuraylib::IFunction_definition>(dst_func_db_name->get_c_str())
    );
    if (!dst_definition) {
        out_error_msg = "Failed to access modified material definition";
        return false;
    }
    mi::base::Handle<mi::neuraylib::IFunction_call> dst_func_call(dst_definition->create_function_call(nullptr));
    if (!dst_func_call) {
        out_error_msg = "Failed to create modified material call";
        return false;
    }
    mi::base::Handle<mi::neuraylib::IMaterial_instance> dst_mat_instance(
        dst_func_call->get_interface<mi::neuraylib::IMaterial_instance>()
    );
    mi::Sint32 ret = trans->store(dst_mat_instance.get(), dst_simple_material_name.c_str());
    if (ret != 0) {
        out_error_msg = "Failed to compile modified material";
        return false;
    }

    return true;
}

inline MDLContext::CompileResult compileInternal(
    const MDLContext::CompileRequest& request,
    mi::neuraylib::INeuray* neuray,
    mi::neuraylib::ITransaction* trans,
    std::map<std::string, std::map<std::string, Tagged_bsdf_info>>& tagged_bsdf_info_cache,
    TargetCodeCache& target_code_cache
)
{
    auto createError = [](const std::string& msg)
    {
        MDLContext::CompileResult result;
        result.success = false;
        result.error_msg = msg;
        return result;
    };

    bool useExistingInstance = !request.instance_name.empty();

    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> mdl_impexp_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>()
    );
    mi::base::Handle<mi::neuraylib::IMdl_factory> mdl_factory(neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    // The context is used to pass options to different components and operations.
    // It also carries errors, warnings, and warnings produced by the operations.
    mi::base::Handle<mi::neuraylib::IMdl_execution_context> context(mdl_factory->create_execution_context());

    mi::base::Handle<const mi::neuraylib::IFunction_call> function_call;
    mi::base::Handle<const mi::neuraylib::IMaterial_instance> material_instance;

    // Split the material name passed on the command line into a module and (unqualified) material name.
    // The expected input is fully-qualified absolute MDL material name of the form:
    // [::<package>]::<module>::<material>
    std::string module_name, material_name;
    if (std::string error; !mdl_utils::parse_cmd_argument_material_name(
            request.qualified_material_name,
            module_name,
            material_name,
            error
        ))
        return createError(error);

    // Load the selected module.
    if (request.module_source.empty())
        mdl_impexp_api->load_module(trans, module_name.c_str(), context.get());
    else
        mdl_impexp_api
            ->load_module_from_string(trans, module_name.c_str(), request.module_source.c_str(), context.get());
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to load the selected module.");

    // Access the module by name, which has to be queried using the factory.
    mi::base::Handle<const mi::IString> module_db_name(mdl_factory->get_db_module_name(module_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        trans->access<mi::neuraylib::IModule>(module_db_name->get_c_str())
    );
    if (!module)
        return createError("Failed to access the loaded module.");

    // Include dependencies so diagnostics show which implementation won path lookup.
    std::set<std::string> logged_modules;
    auto log_origin = [&](auto&& self, const mi::neuraylib::IModule* loaded) -> void
    {
        if (!logged_modules.insert(loaded->get_mdl_name()).second)
            return;
        if (const char* filename = loaded->get_filename())
            sgl::log_debug("MDL module origin: {} = {}", loaded->get_mdl_name(), filename);
        for (mi::Size i = 0; i < loaded->get_import_count(); ++i) {
            mi::base::Handle<const mi::neuraylib::IModule> imported(
                trans->access<mi::neuraylib::IModule>(loaded->get_import(i))
            );
            if (imported)
                self(self, imported.get());
        }
    };
    log_origin(log_origin, module.get());

    // Access the material definition of the selected material.
    // A module can export multiple materials or none at all.
    std::string material_db_name = std::string(module_db_name->get_c_str()) + "::" + material_name;
    material_db_name = mdl_utils::add_missing_material_signature(module.get(), material_db_name);
    if (material_db_name.empty())
        return createError(
            fmt::format("Failed to find the material '{}' in the module '{}'.", material_name, module_name)
        );

    // Check if there is such a definition.
    // This is not really required here because the definition wrapper in the next step
    // will also check if the definition is valid.
    mi::base::Handle<const mi::neuraylib::IFunction_definition> material_definition(
        trans->access<mi::neuraylib::IFunction_definition>(material_db_name.c_str())
    );
    if (!material_definition)
        return createError("Failed to access the material definition.");

    mi::base::Handle<mi::neuraylib::IType_factory> type_factory(mdl_factory->create_type_factory(trans));
    mi::base::Handle<const mi::neuraylib::IType_struct> effective_material_type;
    if (!request.target_material_type.empty()) {
        effective_material_type = mi::base::Handle<const mi::neuraylib::IType_struct>(
            type_factory->create_struct(request.target_material_type.c_str())
        );
        if (!effective_material_type)
            return createError(
                fmt::format("Failed to resolve MDL target material type '{}'.", request.target_material_type)
            );
    } else {
        mi::base::Handle<const mi::neuraylib::IType> return_type(material_definition->get_return_type());
        mi::base::Handle<const mi::neuraylib::IType> unaliased_return_type(return_type->skip_all_type_aliases());
        effective_material_type = mi::base::Handle<const mi::neuraylib::IType_struct>(
            unaliased_return_type->get_interface<mi::neuraylib::IType_struct>()
        );
        if (!effective_material_type)
            return createError("MDL material return type is not a struct.");
    }

    mi::base::Handle<const mi::neuraylib::IStruct_category> material_category(
        effective_material_type->get_struct_category()
    );
    if (!material_category
        || material_category->get_predefined_id() != mi::neuraylib::IStruct_category::CID_MATERIAL_CATEGORY)
        return createError(
            fmt::format(
                "MDL target type '{}' is not a struct in material_category.",
                effective_material_type->get_symbol()
            )
        );

    if (!request.target_material_type.empty() && context->set_option("target_type", effective_material_type.get()) != 0)
        return createError(fmt::format("Failed to set MDL target material type '{}'.", request.target_material_type));

    if (useExistingInstance) {
        // Use the existing instance, rather than creating a new one.
        // FIXME: Perhaps simply query for its existence in the db?
        material_instance = trans->access<const mi::neuraylib::IMaterial_instance>(request.instance_name.c_str());
        function_call = material_instance->get_interface<mi::neuraylib::IFunction_call>();
    } else {
        // Create an instance of the material.
        // An easy way to do that is by using the definition wrapper which makes sure that there
        // are valid parameters set even when there are no default values defined in MDL source.
        mi::neuraylib::Definition_wrapper dw(trans, material_db_name.c_str(), mdl_factory.get());
        if (!dw.is_valid())
            return createError("Failed to access the material definition.");

        mi::Sint32 error = -1;
        function_call = dw.create_instance<mi::neuraylib::IFunction_call>(nullptr, &error);
        if (error != 0)
            return createError(
                fmt::format("Failed to create a material instance of: {}::{} ({})", module_name, material_name, error)
            );
        material_instance = function_call->get_interface<mi::neuraylib::IMaterial_instance>();
    }

    // The next step is to create a compiled material. This already applies some optimizations
    // depending on the following flags and context options.
    //
    // Compile each instance to obtain its optimized MDL graph and SDK hash.
    // Equivalent graphs can reuse the cached HLSL target code below, while
    // create_argument_block() initializes each instance's arguments separately.
    // Graph compilation is still required; linking and HLSL generation can be skipped.
    const mi::Uint32 flags = request.use_class_compilation ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION
                                                           : mi::neuraylib::IMaterial_instance::DEFAULT_OPTIONS;

    // some options require modification of the material graph
    bool modify_material = request.options.enable_tagged_df_properties || request.options.speculars_as_glossy
        || request.options.enable_param_mollification;

    if (modify_material) {
        // the options are untested with material cloning, so leave them to their defaults
        context->set_option("fold_ternary_on_df", false);
        context->set_option("fold_all_bool_parameters", false);
        context->set_option("fold_all_enum_parameters", false);

        // inlining of non-exported functions across module boundaries cause problems during material cloning.
        context->set_option("ignore_noinline", false);
    } else {
        // Set more optimization options (by default, they are all disabled).
        context->set_option("fold_ternary_on_df", request.options.fold_ternary_on_df);
        context->set_option("fold_all_bool_parameters", request.options.fold_all_bool_parameters);
        context->set_option("fold_all_enum_parameters", request.options.fold_all_enum_parameters);
        context->set_option("ignore_noinline", request.options.ignore_noinline);
    }

    // Load any pending texture elements
    if (useExistingInstance) {
        // Create an array containing the element type name ("Texture") we wish to query.
        mi::base::Handle<mi::neuraylib::IFactory> ifactory(neuray->get_api_component<mi::neuraylib::IFactory>());
        mi::IString* textStr(ifactory->create<mi::IString>("String"));
        textStr->set_c_str("Texture");
        mi::IArray* typeArray(ifactory->create<mi::IArray>("String[1]"));
        typeArray->set_element(0, textStr);

        // Get all of the texture elements associated with the material.
        mi::IArray* elements(trans->list_elements(request.instance_name.c_str(), 0, typeArray));
        const mi::Size numElements = elements ? elements->get_length() : 0;

        for (int i = 0; i < numElements; ++i) {
            const mi::IString* elementName(elements->get_element<mi::IString>(i));
            if (elementName->get_c_str()[0] == 0) {
                sgl::log_debug("Ignoring empty texture name.");
                continue;
            }
            std::string elementNameStr(elementName->get_c_str());

            mi::base::Handle<const mi::neuraylib::ITexture> tex(
                trans->access<mi::neuraylib::ITexture>(elementName->get_c_str())
            );
            if (tex == nullptr) {
                sgl::log_warn("Failed to access MDL texture {}.", elementNameStr);
                continue;
            }

            std::string imageName;
            const char* imageNameStr = tex->get_image();
            if (imageNameStr) {
                imageName = std::string(imageNameStr);
            }
            if (imageName.empty()) {
                // The texture does not yet have an associated image. Create one.
                imageName = std::string(elementName->get_c_str()) + "-img";
                mi::base::Handle<const mi::neuraylib::IImage> curImage(
                    trans->access<const mi::neuraylib::IImage>(imageName.c_str())
                );
                if (!curImage.is_valid_interface()) {
                    mi::base::Handle<mi::neuraylib::IImage> image;
                    image = trans->create<mi::neuraylib::IImage>("Image");

                    int res = image->reset_file(elementNameStr.c_str());
                    if (res < 0) {
                        if (res == -2) {
                            sgl::log_error("Could not open texture file '{}'.", elementNameStr);
                        } else {
                            sgl::log_error("Failed to set image file to '{}': {}.", elementNameStr, res);
                        }
                        continue;
                    }
                    sgl::log_debug("Created new texture image for '{}'.", elementNameStr);
                    trans->store(image.get(), imageName.c_str());
                }

                // Replace the texture with a version that points to the image
                mi::base::Handle<mi::neuraylib::ITexture> newTexture(trans->create<mi::neuraylib::ITexture>("Texture"));

                if (newTexture == nullptr) {
                    sgl::log_error("Failed to create texture for '{}'.", elementNameStr);
                    continue;
                }
                int res = newTexture->set_image(imageName.c_str());
                if (res < 0) {
                    sgl::log_error("Failed to set texture image '{}': {}.", elementNameStr, res);
                    continue;
                }
                // Copy gamma override. Note that we treat a gamma of 0 (unknown/default) as linear.
                newTexture->set_gamma(tex->get_gamma() > 0.f ? tex->get_gamma() : 1.f);
                // Overwrite the texture with the new version. Subsequent texture accesses will then retrieve the
                // version with the associated image, and not create it again.
                res = trans->store(newTexture.get(), elementName->get_c_str());
                if (res < 0) {
                    sgl::log_error("Failed to store new texture for '{}'", elementNameStr);
                }
                sgl::log_debug("Loaded texture element '{}'.", elementNameStr);
            }
        }
    }

    mi::base::Handle<const mi::neuraylib::ICompiled_material> compiled_material(
        material_instance->create_compiled_material(flags, context.get())
    );
    if (!compiled_material) {
        log_messages_and_check_errors(context.get());
        return createError("MDL SDK returned no compiled material.");
    }

    material_instance = {};

    std::map<std::string, Tagged_bsdf_info> tagged_bsdf_infos;
    // The second compilation sees the generated identifiers. Keep their original names so the argument-block API
    // continues to expose the same parameters as the source material. Injected parameters are intentionally unmapped.
    std::map<std::string, std::string, std::less<>> source_name_by_generated_name;
    if (modify_material) {
        ModifiedParameterNames parameter_names;
        std::string parameter_name_error;
        if (!build_modified_parameter_names(
                request,
                compiled_material.get(),
                mdl_factory.get(),
                parameter_names,
                parameter_name_error
            ))
            return createError(parameter_name_error);
        for (mi::Size i = 0; i < parameter_names.source.size(); ++i)
            source_name_by_generated_name.emplace(parameter_names.generated[i], parameter_names.source[i]);

        // include the options in the hash so when the same compiled material is recompiled with different modification
        // options, the modified material gets a unique name.
        const size_t optionsHash = fnv_hash_array64(&request.options, sizeof(MDLContext::CompileRequest::options));

        std::string simple_name(material_definition->get_mdl_simple_name());

        if (useExistingInstance) {
            // If using an existing instance, hash the instance name and append to the simple name.
            // This ensures that the modified material will have a unique name when compiling different instances of
            // the same material.
            simple_name += "_"
                + sgl::SHA1(request.instance_name.data(), request.instance_name.size()).hex_digest().substr(0, 8);
        }

        mi::base::Uuid src_hash = compiled_material->get_hash();
        std::string modified_mdl_simple_name = fmt::format(
            "{}_modified_{}_{}_{}_{}_{}",
            simple_name,
            src_hash.m_id1,
            src_hash.m_id2,
            src_hash.m_id3,
            src_hash.m_id4,
            optionsHash
        );

        // compile the modified material
        mi::base::Handle<const mi::neuraylib::IFunction_call> modified_material_instance(
            trans->access<const mi::neuraylib::IFunction_call>(modified_mdl_simple_name.c_str())
        );

        // only add the modified material if it doesn't already exist
        if (!modified_material_instance) {
            // Post-process the scene data.
            mi::base::Handle<const mi::neuraylib::IExpression_list> srcArgs(function_call->get_arguments());

            std::string error_msg;
            if (!build_modified_material(
                    request,
                    modified_mdl_simple_name,
                    compiled_material.get(),
                    material_definition.get(),
                    useExistingInstance ? srcArgs.get() : nullptr,
                    parameter_names,
                    mdl_factory.get(),
                    context.get(),
                    trans,
                    tagged_bsdf_infos,
                    error_msg
                )) {
                log_messages_and_check_errors(context.get());
                return createError(error_msg);
            }

            // get back after storing
            modified_material_instance = mi::base::Handle<const mi::neuraylib::IFunction_call>(
                trans->access<const mi::neuraylib::IFunction_call>(modified_mdl_simple_name.c_str())
            );

            if (!modified_material_instance)
                return createError("Failed to access the modified material instance.");

            tagged_bsdf_info_cache[modified_mdl_simple_name] = tagged_bsdf_infos;
        } else if (request.options.enable_tagged_df_properties) {
            const auto tagged_bsdf_info = tagged_bsdf_info_cache.find(modified_mdl_simple_name);
            if (tagged_bsdf_info == tagged_bsdf_info_cache.end())
                return createError("Cached modified material is missing tagged BSDF metadata.");
            tagged_bsdf_infos = tagged_bsdf_info->second;
        }

        // set the target compilation options for the final modified material
        context->set_option("fold_ternary_on_df", request.options.fold_ternary_on_df);
        context->set_option("fold_all_bool_parameters", request.options.fold_all_bool_parameters);
        context->set_option("fold_all_enum_parameters", request.options.fold_all_enum_parameters);
        context->set_option("ignore_noinline", request.options.ignore_noinline);

        // when parameterized mollification is enabled, we always compile the modified material using class compilation
        // to enable the parameter
        const mi::Uint32 modified_flags
            = request.options.enable_param_mollification ? mi::neuraylib::IMaterial_instance::CLASS_COMPILATION : flags;

        // compile the modified material
        mi::base::Handle<const mi::neuraylib::IMaterial_instance> modified_material_instance2(
            modified_material_instance->get_interface<mi::neuraylib::IMaterial_instance>()
        );

        mi::base::Handle<mi::neuraylib::ICompiled_material> compiled_modified_material(
            modified_material_instance2->create_compiled_material(modified_flags, context.get())
        );
        if (!compiled_modified_material)
            return createError("Failed to compile the modified material.");

        // use the compiled modified material in place of the original compiled material
        compiled_material = compiled_modified_material;
    }

    function_call = {};

    auto expression_has_kind = [&](const char* path, mi::neuraylib::IType::Kind kind)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> expression(compiled_material->lookup_sub_expression(path));
        if (!expression)
            return false;
        mi::base::Handle<const mi::neuraylib::IType> expression_type(expression->get_type());
        mi::base::Handle<const mi::neuraylib::IType> unaliased_type(expression_type->skip_all_type_aliases());
        return unaliased_type->get_kind() == kind;
    };

    std::string surface_scattering_path;
    if (expression_has_kind("surface.scattering", mi::neuraylib::IType::TK_BSDF))
        surface_scattering_path = "surface.scattering";
    else if (expression_has_kind("scattering", mi::neuraylib::IType::TK_BSDF))
        surface_scattering_path = "scattering";
    else
        return createError("MDL material does not provide a BSDF at 'surface.scattering' or top-level 'scattering'.");

    std::vector<MDLContext::CompileResult::MaterialAOVOutput> material_aov_outputs;
    if (request.material_aov_registry
        && effective_material_type->get_predefined_id() == mi::neuraylib::IType_struct::SID_USER) {
        material_aov_outputs.reserve(effective_material_type->get_size());
        for (mi::Size i = 0; i < effective_material_type->get_size(); ++i) {
            const std::string field_name = effective_material_type->get_field_name(i);
            mi::base::Handle<const mi::neuraylib::IType> field_type(effective_material_type->get_field_type(i));

            const bool is_render_contract_field = field_name == "thin_walled" || field_name == "surface"
                || field_name == "backface" || field_name == "ior" || field_name == "volume" || field_name == "geometry"
                || field_name == "hair" || (field_name == "scattering" && surface_scattering_path == "scattering");
            if (is_render_contract_field)
                continue;

            const uint32_t component_count = get_material_aov_component_count(field_type.get());
            if (!component_count) {
                mi::base::Handle<const mi::IString> type_name(type_factory->dump(field_type.get()));
                sgl::log_debug(
                    "Ignoring unsupported MDL material AOV field '{}' of type '{}'.",
                    field_name,
                    type_name ? type_name->get_c_str() : "<unknown>"
                );
                continue;
            }

            const uint32_t id
                = request.material_aov_registry->register_output(field_name, sgl::DataType::float32, component_count);
            mi::base::Handle<const mi::neuraylib::IExpression> expression(
                compiled_material->lookup_sub_expression(field_name.c_str())
            );
            if (!expression)
                return createError(fmt::format("Failed to resolve MDL material AOV field '{}'.", field_name));
            material_aov_outputs.push_back({field_name, component_count, id, fmt::format("material_aov_{}", id)});
        }
    }

    // Create a HLSL back-end for code generation.
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>()
    );
    mi::base::Handle<mi::neuraylib::IMdl_backend> hlsl_backend(
        backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL)
    );
    if (!hlsl_backend)
        return createError("Failed to create the MDL HLSL backend.");

    const auto material_hash = compiled_material->get_hash();
    TargetCodeKey target_key{
        .material_hash = {material_hash.m_id1, material_hash.m_id2, material_hash.m_id3, material_hash.m_id4},
    };
    auto set_backend_option = [&](const char* name, const char* value)
    {
        hlsl_backend->set_option(name, value);
        target_key.backend_options.emplace_back(name, value);
    };

    // Set code generation options.
    set_backend_option("internal_space", "coordinate_world");
    set_backend_option("compile_constants", request.options.compile_constants ? "on" : "off");
    set_backend_option("fast_math", request.options.fast_math ? "on" : "off");
    set_backend_option("opt_level", std::to_string(request.options.opt_level).c_str());
    set_backend_option("enable_auxiliary", request.options.enable_auxiliary ? "on" : "off");
    set_backend_option("use_renderer_adapt_normal", request.options.use_renderer_adapt_normal ? "on" : "off");
    switch (request.options.df_handle_slot_mode) {
    case 0:
        set_backend_option("df_handle_slot_mode", "none");
        break;
    case 1:
    case 2:
    case 4:
    case 8:
        set_backend_option("df_handle_slot_mode", fmt::format("fixed_{}", request.options.df_handle_slot_mode).c_str());
        break;
    default:
        return createError("Invalid df_handle_slot_mode.");
    }
    set_backend_option("texture_runtime_with_derivs", request.options.texture_runtime_with_derivs ? "on" : "off");
    set_backend_option("num_texture_results", std::to_string(request.options.num_texture_results).c_str());
    set_backend_option("num_texture_spaces", std::to_string(request.options.num_texture_spaces).c_str());
    set_backend_option("jit_warn_spectrum_conversion", request.options.jit_warn_spectrum_conversion ? "on" : "off");

    // Select the expressions to generate code for.
    using TD = mi::neuraylib::Target_function_description;
    std::vector<TD> descs;
    descs.reserve(16 + material_aov_outputs.size() + tagged_bsdf_infos.size() * 4);
    descs.emplace_back("init", "init");

    auto has_expression = [&](const char* path)
    {
        mi::base::Handle<const mi::neuraylib::IExpression> expression(compiled_material->lookup_sub_expression(path));
        return bool(expression);
    };
    auto add_expression = [&](const char* path, const char* callable_name, const char* same_as = nullptr)
    {
        if (!has_expression(path))
            return false;
        if (same_as
            && compiled_material->get_sub_expression_hash(path) == compiled_material->get_sub_expression_hash(same_as))
            return false;
        descs.emplace_back(path, callable_name);
        return true;
    };

    const bool has_ior = add_expression("ior", "ior");
    add_expression("thin_walled", "thin_walled");
    const mi::Size surface_scatter_descriptor_index = descs.size();
    descs.emplace_back(surface_scattering_path.c_str(), "surface_scattering");
    const mi::Size surface_emission_descriptor_index = descs.size();
    const bool has_surface_emission = add_expression("surface.emission.emission", "surface_emission_emission");
    add_expression("surface.emission.intensity", "surface_emission_intensity");
    add_expression("surface.emission.mode", "surface_emission_mode");
    // Like the SDK DXR example, omit duplicate backface functions. The Falcor wrapper only
    // calls surface functions, so no backface aliases are needed; distinct expressions remain.
    add_expression("backface.scattering", "backface_scattering", surface_scattering_path.c_str());
    add_expression("backface.emission.emission", "backface_emission_emission", "surface.emission.emission");
    add_expression("backface.emission.intensity", "backface_emission_intensity", "surface.emission.intensity");
    add_expression("backface.emission.mode", "backface_emission_mode", "surface.emission.mode");
    const bool has_volume_absorption = add_expression("volume.absorption_coefficient", "volume_absorption_coefficient");
    const bool has_volume_scattering = add_expression("volume.scattering_coefficient", "volume_scattering_coefficient");
    add_expression("geometry.normal", "geometry_normal");
    const bool has_cutout_opacity = add_expression("geometry.cutout_opacity", "geometry_cutout_opacity");
    add_expression("geometry.displacement", "geometry_displacement");

    for (const MDLContext::CompileResult::MaterialAOVOutput& output : material_aov_outputs) {
        descs.emplace_back(output.name.c_str(), output.callable_name.c_str());
    }

    // Select custom expressions for valid inputs to all tagged bsdfs
    for (const auto& [key, value] : tagged_bsdf_infos) {
        // utility function adding a valid input to the selected expressions
        auto add_input_function = [&](const Tagged_bsdf_info::Input& input) -> void
        {
            if (!input.path.empty()) {
                descs.push_back(mi::neuraylib::Target_function_description(input.path.c_str(), input.name.c_str()));
            }
        };

        add_input_function(value.roughness_u);
        add_input_function(value.roughness_v);
        add_input_function(value.tangent_u);
        add_input_function(value.tint);
    }

    std::vector<std::span<TD>> function_groups{descs};

    // Standalone expressions cannot depend on cached values from the shared shading init.
    const bool potentially_emissive
        = has_surface_emission && !surface_emission_is_structurally_zero(compiled_material.get());
    const auto zero_checks = potentially_emissive ? emission_zero_checks(compiled_material.get(), trans)
                                                  : std::vector<EmissionZeroCheck>{};
    std::vector<MDLContext::CompileResult::EmissionZeroCheck> check_functions;
    check_functions.reserve(zero_checks.size()); // Keep callable names stable until translation.
    std::vector<std::array<TD, 2>> check_probes(zero_checks.size());
    for (size_t i = 0; i < zero_checks.size(); ++i) {
        const auto& check = zero_checks[i];
        const auto& functions = check_functions.emplace_back(
            fmt::format("emission_value_{}", i),
            check.condition_path.empty() ? "" : fmt::format("emission_condition_{}", i),
            check.condition_value
        );
        auto& probes = check_probes[i];
        probes[0] = TD(check.value_path.c_str(), functions.value_function.c_str());
        probes[1] = TD(check.condition_path.c_str(), functions.condition_function.c_str());
        for (size_t j = 0; j < (check.condition_path.empty() ? 1u : 2u); ++j)
            function_groups.emplace_back(&probes[j], 1);
    }

    // Key and submit the same groups: shared-init functions and standalone emission probes
    // produce different code even when the flattened list of expressions is identical.
    for (const auto functions : function_groups) {
        auto& group = target_key.function_groups.emplace_back();
        for (const auto& function : functions)
            group.emplace_back(function.path, function.base_fname);
    }

    // Like the SDK DXR example, share target code but build each instance's arguments below.
    // Retain the SDK's indices with the target so hits also skip expensive link-unit construction.
    // Link/translation context options are fixed at their defaults. If made configurable,
    // record them in the key too; the earlier context settings only affect graph compilation.
    mi::base::Handle<const mi::neuraylib::ITarget_code> hlsl_target_code;
    std::vector<TargetCodeEntry::FunctionIndices> function_indices;
    const auto cached = target_code_cache.find(target_key);
    if (cached != target_code_cache.end()) {
        hlsl_target_code = cached->second.code;
        size_t i = 0;
        for (const auto functions : function_groups) {
            for (auto& function : functions) {
                const auto& indices = cached->second.function_indices.at(i++);
                function.function_index = indices.function_index;
                function.argument_block_index = indices.argument_block_index;
            }
        }
        sgl::log_debug("Reusing MDL target code for {}", request.qualified_material_name);
    } else {
        sgl::log_info("Compiling MDL to HLSL for material {} ...", request.qualified_material_name);
        mi::base::Handle<mi::neuraylib::ILink_unit> hlsl_link_unit(
            hlsl_backend->create_link_unit(trans, context.get())
        );
        if (log_messages_and_check_errors(context.get()) || !hlsl_link_unit)
            return createError("Failed to create a link unit");
        for (const auto functions : function_groups) {
            const mi::Sint32 status
                = hlsl_link_unit
                      ->add_material(compiled_material.get(), functions.data(), functions.size(), context.get());
            if (log_messages_and_check_errors(context.get()) || status != 0)
                return createError(
                    fmt::format(
                        "Failed to select functions for code generation (ILink_unit::add_material returned {}).",
                        status
                    )
                );
            for (const auto& function : functions)
                function_indices.push_back({function.function_index, function.argument_block_index});
        }
        hlsl_target_code = hlsl_backend->translate_link_unit(hlsl_link_unit.get(), context.get());
        if (log_messages_and_check_errors(context.get()) || !hlsl_target_code)
            return createError("Failed to translate the link unit to code.");
    }

    // Next, we query the generated target code and all the resources it needs.
    // We store all of that in a CompileResult object.
    MDLContext::CompileResult result = {};

    mi::Float32 cutout_opacity;
    if (compiled_material->get_cutout_opacity(&cutout_opacity))
        result.cutout_opacity = std::clamp(cutout_opacity, 0.f, 1.f);

    mi::base::Handle<const mi::neuraylib::IExpression_constant> thin_walled(
        compiled_material->lookup_sub_expression<mi::neuraylib::IExpression_constant>("thin_walled")
    );
    if (thin_walled) {
        mi::base::Handle<const mi::neuraylib::IValue_bool> value(thin_walled->get_value<mi::neuraylib::IValue_bool>());
        if (value)
            result.thin_walled = value->get_value();
    }

    result.code = hlsl_target_code->get_code();
    result.material_aov_outputs = std::move(material_aov_outputs);

    // Callable availability does not imply emission: even an empty EDF may have
    // a zero-returning callable. Reject structurally dark materials before triangle allocation.
    result.has_surface_emission
        = has_surface_emission && descs[surface_emission_descriptor_index].function_index != mi::Size(~0);
    result.potentially_emissive = result.has_surface_emission && potentially_emissive;
    auto probe_is_compatible = [&](const TD& probe)
    {
        if (probe.function_index == mi::Size(~0)
            || hlsl_target_code->get_callable_function_render_state_usage(probe.function_index) != 0)
            return false;
        if (probe.argument_block_index == mi::Size(~0)
            || probe.argument_block_index == descs[surface_scatter_descriptor_index].argument_block_index)
            return true;
        // The SDK allocates another layout index for the same compiled material. Reuse the
        // live argument buffer only when the complete nested layout agrees; otherwise fall back.
        mi::base::Handle<const mi::neuraylib::ITarget_value_layout> material_layout(
            hlsl_target_code->get_argument_block_layout(descs[surface_scatter_descriptor_index].argument_block_index)
        );
        mi::base::Handle<const mi::neuraylib::ITarget_value_layout> probe_layout(
            hlsl_target_code->get_argument_block_layout(probe.argument_block_index)
        );
        return argument_layouts_match(material_layout.get(), probe_layout.get());
    };
    for (size_t i = 0; i < check_probes.size(); ++i) {
        if (probe_is_compatible(check_probes[i][0])
            && (check_functions[i].condition_function.empty() || probe_is_compatible(check_probes[i][1])))
            result.emission_zero_checks.push_back(std::move(check_functions[i]));
    }
    if (result.has_surface_emission) {
        const mi::Size edf_index = descs[surface_emission_descriptor_index].function_index;
        mi::base::Handle<const mi::neuraylib::IExpression_constant> mode(
            compiled_material->lookup_sub_expression<mi::neuraylib::IExpression_constant>("surface.emission.mode")
        );
        mi::base::Handle<const mi::neuraylib::IValue_enum> value(
            mode ? mode->get_value<mi::neuraylib::IValue_enum>() : nullptr
        );
        // This adapter supports MDL radiant exitance (W/m^2). Total power (W)
        // requires the emitting object's area, which this material interface does not provide.
        if (!value || value->get_value() != 0)
            return createError(
                "MDL emission requires constant intensity_radiant_exitance; intensity_power and dynamic intensity "
                "modes are not supported"
            );
        result.surface_emission_handle_count
            = uint32_t(hlsl_target_code->get_callable_function_df_handle_count(edf_index));
    }

    if (!has_ior)
        result.code += "\nfloat3 ior(in Shading_state_material state) { return float3(1.5f); }\n";
    if (!has_volume_absorption)
        result.code
            += "\nfloat3 volume_absorption_coefficient(in Shading_state_material state) { return float3(0.0f); }\n";
    if (!has_volume_scattering)
        result.code
            += "\nfloat3 volume_scattering_coefficient(in Shading_state_material state) { return float3(0.0f); }\n";
    if (!has_cutout_opacity)
        result.code += "\nfloat geometry_cutout_opacity(in Shading_state_material state) { return 1.0f; }\n";

    for (mi::Size i = 0; i < compiled_material->get_referenced_scene_data_count(); ++i)
        result.referenced_scene_data.emplace_back(compiled_material->get_referenced_scene_data_name(i));
    result.render_state_usage = hlsl_target_code->get_render_state_usage();
    for (mi::Size i = 0; i < hlsl_target_code->get_string_constant_count(); ++i) {
        const char* value = hlsl_target_code->get_string_constant(i);
        result.string_constants.emplace_back(value ? value : "");
    }

    if (request.options.enable_tagged_df_properties) {
        // Generate lambda code for the tagged bsdf input expressions
        std::string tangent_switch_function = "float3 mdl_lambda_tangent(in uint bsdf_index, "
                                              "in Shading_state_material state) {\n"
                                              "   switch(bsdf_index) {\n";

        std::string roughness_u_switch_function = "float mdl_lambda_roughness_u(in uint bsdf_index, "
                                                  "in Shading_state_material state) {\n"
                                                  "   switch(bsdf_index) {\n";

        std::string roughness_v_switch_function = "float mdl_lambda_roughness_v(in uint bsdf_index, "
                                                  "in Shading_state_material state) {\n"
                                                  "   switch(bsdf_index) {\n";

        std::string tint_switch_function = "float3 mdl_lambda_tint(in uint bsdf_index, "
                                           "in Shading_state_material state) {\n"
                                           "   switch(bsdf_index) {\n";

        // Collect the function indices per tagged bsdf, using the material slot order for the bsdfs.
        mi::Size surface_scatter_function_index = descs[surface_scatter_descriptor_index].function_index;
        result.surface_scatter_handle_count
            = (uint32_t)hlsl_target_code->get_callable_function_df_handle_count(surface_scatter_function_index);
        for (mi::Size i = 0; i < result.surface_scatter_handle_count; i++) {
            // utility function adding a valid input to the selected expressions
            auto resolve_input_function = [&](const Tagged_bsdf_info::Input& input, std::string& switch_code) -> void
            {
                if (!input.path.empty()) {
                    switch_code += fmt::format("       case {}: return {}(state);\n", i, input.name);
                }
            };

            const char* tag = hlsl_target_code->get_callable_function_df_handle(surface_scatter_function_index, i);

            const auto itr = tagged_bsdf_infos.find(tag);
            if (itr != tagged_bsdf_infos.end()) {
                resolve_input_function(itr->second.roughness_u, roughness_u_switch_function);
                resolve_input_function(itr->second.roughness_v, roughness_v_switch_function);
                resolve_input_function(itr->second.tangent_u, tangent_switch_function);
                resolve_input_function(itr->second.tint, tint_switch_function);
            }
        }

        roughness_u_switch_function += "       default: break;\n"
                                       "   }\n"
                                       "   return 0.0f;\n"
                                       "}\n\n";

        roughness_v_switch_function += "       default: break;\n"
                                       "   }\n"
                                       "   return 0.0f;\n"
                                       "}\n\n";

        tangent_switch_function += "       default: break;\n"
                                   "   }\n"
                                   "   return float3(0.0f,0.0f,0.0f);\n"
                                   "}\n\n";

        tint_switch_function += "       default: break;\n"
                                "   }\n"
                                "   return float3(0.0f,0.0f,0.0f);\n"
                                "}\n\n";

        result.code += roughness_u_switch_function;
        result.code += roughness_v_switch_function;
        result.code += tangent_switch_function;
        result.code += tint_switch_function;
    }

    // Get read-only data segments.
    {
        size_t count = hlsl_target_code->get_ro_data_segment_count();
        for (size_t i = 0; i < count; ++i) {
            const char* name = hlsl_target_code->get_ro_data_segment_name(i);
            const uint8_t* data = reinterpret_cast<const uint8_t*>(hlsl_target_code->get_ro_data_segment_data(i));
            size_t size = hlsl_target_code->get_ro_data_segment_size(i);

            MDLContext::CompileResult::DataSegment segment{name, std::vector<uint8_t>(data, data + size)};
            result.ro_data_segments.push_back(segment);
        }
    }

    // Get argument block segments.
    {
        // Keep body IDs unchanged and the table consecutive. Discard trailing instance defaults;
        // any defaults before the last body resource must remain to preserve those IDs.
        mi::Size texture_count = hlsl_target_code->get_texture_count();
        while (texture_count > 1 && !hlsl_target_code->get_texture_is_body_resource(texture_count - 1))
            --texture_count;
        for (mi::Size i = 1; i < texture_count; ++i) {
            result.textures.push_back({
                .index = sgl::narrow_cast<uint32_t>(i),
                .name = hlsl_target_code->get_texture(i),
                .source = {
                    .type = texture_type_from_mdl_shape(hlsl_target_code->get_texture_shape(i)),
                    .srgb = hlsl_target_code->get_texture_gamma(i) == mi::neuraylib::ITarget_code::GM_GAMMA_SRGB,
                },
            });
        }
        ArgumentResourceCallback
            resource_callback(trans, hlsl_target_code.get(), result.string_constants, result.textures);
        size_t count = hlsl_target_code->get_argument_block_count();
        for (size_t i = 0; i < count; ++i) {
            mi::base::Handle<const mi::neuraylib::ITarget_argument_block> argBlock(
                hlsl_target_code->create_argument_block(i, compiled_material.get(), &resource_callback)
            );
            FALCOR_CHECK(argBlock, "Failed to initialize MDL argument block {}.", i);
            const uint8_t* data = reinterpret_cast<const uint8_t*>(argBlock->get_data());
            size_t size = argBlock->get_size();

            MDLContext::CompileResult::DataSegment segment{"", std::vector<uint8_t>(data, data + size)};
            result.arg_block_segments.push_back(segment);
        }
    }

    // Get parameters
    if (hlsl_target_code->get_argument_layout_count()) {
        result.arg_block_layout.clear();
        mi::Size argument_block_index = descs[surface_scatter_descriptor_index].argument_block_index;
        if (argument_block_index != ~0) {
            mi::base::Handle<const mi::neuraylib::ITarget_value_layout> layout(
                hlsl_target_code->get_argument_block_layout(argument_block_index)
            );

            if (layout) {
                for (mi::Size j = 0, num_params = compiled_material->get_parameter_count(); j < num_params; ++j) {
                    const char* name = compiled_material->get_parameter_name(j);
                    if (name == nullptr)
                        continue;

                    // Get the offset of the argument within the target argument block
                    mi::neuraylib::Target_value_layout_state state(layout->get_nested_state(j));
                    mi::neuraylib::IValue::Kind kind2;
                    mi::Size param_size;
                    mi::Size offset = layout->get_layout(kind2, param_size, state);

                    MDLContext::CompileResult::ArgumentLayout entry = {};
                    entry.size = (uint32_t)param_size;
                    entry.offset = (uint32_t)offset;

                    entry.type = MDLContext::CompileResult::ArgumentLayout::Type::unsupported;
                    switch (kind2) {
                    case mi::neuraylib::IValue::Kind::VK_BOOL:
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::bool_;
                        FALCOR_ASSERT(entry.size == 1);
                        break;
                    case mi::neuraylib::IValue::Kind::VK_INT:
                    case mi::neuraylib::IValue::Kind::VK_ENUM:
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::int_;
                        FALCOR_ASSERT(entry.size == 4);
                        break;
                    case mi::neuraylib::IValue::Kind::VK_FLOAT:
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float_;
                        FALCOR_ASSERT(entry.size == 4);
                        break;
                    case mi::neuraylib::IValue::Kind::VK_VECTOR:
                    case mi::neuraylib::IValue::Kind::VK_COLOR: {
                        auto element_count = layout->get_num_elements(state);
                        auto element_state = layout->get_nested_state(0, state);
                        mi::Size element_size;
                        mi::neuraylib::IValue::Kind element_kind;
                        layout->get_layout(element_kind, element_size, element_state);
                        if (element_kind == mi::neuraylib::IValue::Kind::VK_FLOAT) {
                            if (element_count == 1)
                                entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float_;
                            else if (element_count == 2)
                                entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float2_;
                            else if (element_count == 3)
                                entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float3_;
                            else if (element_count == 4)
                                entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float4_;
                        }
                    } break;
                    case mi::neuraylib::IValue::Kind::VK_MATRIX:
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::float4x4_;
                        FALCOR_ASSERT(entry.size == 64);
                        break;
                    case mi::neuraylib::IValue::Kind::VK_TEXTURE: {
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::texture;
                        FALCOR_ASSERT(entry.size == 4);
                        mi::base::Handle<const mi::neuraylib::IValue_texture> value(
                            compiled_material->get_argument<mi::neuraylib::IValue_texture>(j)
                        );
                        mi::base::Handle<const mi::neuraylib::IType_texture> type(value->get_type());
                        entry.texture_type = texture_type_from_mdl_shape(type->get_shape());
                        if (entry.texture_type == MDLContext::TextureSource::Type::Unknown)
                            return createError("Unsupported texture argument type");
                        break;
                    }
                    case mi::neuraylib::IValue::Kind::VK_DOUBLE:
                    case mi::neuraylib::IValue::Kind::VK_STRING:
                    case mi::neuraylib::IValue::Kind::VK_ARRAY:
                    case mi::neuraylib::IValue::Kind::VK_STRUCT:
                    case mi::neuraylib::IValue::Kind::VK_INVALID_DF:
                    case mi::neuraylib::IValue::Kind::VK_LIGHT_PROFILE:
                    case mi::neuraylib::IValue::Kind::VK_BSDF_MEASUREMENT:
                        break;
                    };

                    // Translate aliases such as "compiled_parameter_0_1" back to source paths such as "tint.r".
                    // An unmapped name belongs to an injected parameter and is already its intended public name.
                    const auto source_name = source_name_by_generated_name.find(name);
                    const std::string public_name
                        = source_name == source_name_by_generated_name.end() ? std::string(name) : source_name->second;
                    if (!result.arg_block_layout.insert({public_name, entry}).second)
                        return createError(fmt::format("Duplicate material parameter name '{}'.", public_name));
                }
            }
        }
    }

    // Get textures.
    {
        for (auto& result_texture : result.textures) {
            const uint32_t i = result_texture.index;
            const char* name = result_texture.name.c_str();
            mi::base::Handle<const mi::neuraylib::ITexture> texture(trans->access<mi::neuraylib::ITexture>(name));
            mi::base::Handle<const mi::neuraylib::IImage> image(
                trans->access<mi::neuraylib::IImage>(texture->get_image())
            );

            mi::Size frame_count = image->get_length();
            if (frame_count > 1)
                sgl::log_warn("Texture[{}] '{}' contains {} frames, only the first one is used", i, name, frame_count);

            mi::base::Handle<const mi::neuraylib::ICanvas> canvas(image->get_canvas(0, 0, 0));
            std::string filename;
            if (const char* filename_cstr = image->get_filename(0, 0))
                filename = filename_cstr;
            // If the filename is uvtile texture, we find the UDIM ID and replace it back to <UDIM>.
            // Ideally we should be using get_original_filename to avoid this reverse process,
            // but it returns nullptr for our common use cases.
            if (image->is_uvtile()) {
                static std::regex udimRegex("\\.1[0-9]{3}(?!000)\\."); // Valid range 1001-1999
                filename = std::regex_replace(filename, udimRegex, ".<UDIM>.", std::regex_constants::format_first_only);
            }
            bool srgb = result_texture.source.srgb;
            uint32_t width = canvas->get_resolution_x();
            uint32_t height = canvas->get_resolution_y();
            uint32_t depth = canvas->get_layers_size();
            sgl::log_debug(
                "Texture[{}] '{}': filename={} srgb={} resolution={}x{}x{}",
                i,
                name,
                !filename.empty() ? filename.c_str() : "(empty)",
                srgb,
                width,
                height,
                depth
            );

            auto& source = result_texture.source;
            source.path = filename;
            source.width = width;
            source.height = height;
            source.depth = depth;

            // Shape conversion preserves all MDL shapes; extraction currently supports these two.
            switch (source.type) {
            case MDLContext::TextureSource::Type::Texture2D:
                break;
            case MDLContext::TextureSource::Type::TextureBSDFData:
                source.format = sgl::Format::r32_float;
                {
                    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0));
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(tile->get_data());
                    size_t size = width * height * depth * sizeof(float);
                    source.data = std::vector<uint8_t>(data, data + size);
                }
                break;
            default:
                return createError("Unsupported texture type");
            }
        }
    }

    // Get BSDF measurements.
    {
        size_t count = hlsl_target_code->get_bsdf_measurement_count();
        if (count > 1)
            return createError("BSDF measurements are not supported");
        for (size_t i = 1; i < count; ++i) {
            auto name = hlsl_target_code->get_bsdf_measurement(i);
            mi::base::Handle<const mi::neuraylib::IBsdf_measurement> bsdf_measurement(
                trans->access<mi::neuraylib::IBsdf_measurement>(name)
            );
        }
    }

    // Get light profiles.
    {
        size_t count = hlsl_target_code->get_light_profile_count();
        if (count > 1)
            return createError("Light profiles are not supported");
        for (size_t i = 1; i < count; ++i) {
            auto name = hlsl_target_code->get_light_profile(i);
            mi::base::Handle<const mi::neuraylib::ILightprofile> light_profile(
                trans->access<mi::neuraylib::ILightprofile>(name)
            );
        }
    }

    if (cached == target_code_cache.end())
        target_code_cache.emplace(
            std::move(target_key),
            TargetCodeEntry{hlsl_target_code, std::move(function_indices)}
        );
    result.success = true;

    return result;
}

MDLContext::CompileResult MDLContext::compile(const CompileRequest& request)
{
    std::lock_guard lock(m_mutex);

    FALCOR_CHECK(m_impl, "MDLContext does not have an active implementation.");
    auto neuray = static_cast<mi::neuraylib::INeuray*>(m_impl->get_neuray());
    FALCOR_CHECK(neuray, "MDLContext backend did not expose a valid neuray interface.");

    // Access the database and create a transaction.
    // This is required for loading MDL modules and their dependencies.
    // All loaded elements are stored in the DB as soon as the transaction is committed.
    // Until then, changes are only visible to the current open transaction.
    mi::base::Handle<mi::neuraylib::IDatabase> database(neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> trans(scope->create_transaction());

    // Compile the MDL material using the transaction we just created.
    auto& impl = static_cast<MDLContextImpl&>(*m_impl);
    CompileResult result;
    try {
        result = compileInternal(request, neuray, trans.get(), impl.tagged_bsdf_info_cache, impl.target_code_cache);
    } catch (...) {
        trans->abort();
        throw;
    }

    // Close the transaction.
    if (result.success)
        trans->commit();
    else
        trans->abort();

    return result;
}

} // namespace falcor
