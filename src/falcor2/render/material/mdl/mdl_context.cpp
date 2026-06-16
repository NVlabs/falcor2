// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/error.h"
#include "falcor2/core/logger.h"
#include "falcor2/utils/fnv_hash.h"
#include "falcor2/falcor2.h"

#include "mdl_sdk_wrapper.h"
#include "mdl_context.h"
#include "mdl_utils.h"

#include <sgl/core/platform.h>
#include <sgl/core/crypto.h>

namespace falcor {

struct MDLContextImpl : public MDLContext::IImpl {
    ~MDLContextImpl() override
    {
        if (neuray && neuray->shutdown(true) != 0)
            sgl::log_error("Failed to shutdown the MDL SDK.");

        neuray = nullptr; // free the handles that holds the INeuray instance
        mdl_utils::unload_mdl(mdl);
    }

    MDLContextImpl()
    {
        if (std::string error; !mdl_utils::load_mdl_and_get_ineuray(mdl, neuray, error) || !neuray.is_valid_interface())
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
                (sgl::platform::runtime_directory() / ("nv_openimageio" MI_BASE_DLL_FILE_EXT)).string().c_str()
            ))
            FALCOR_THROW("Failed to load the 'nv_openimageio' plugin.\n");
        if (!mdl_utils::load_mdl_plugin(
                neuray.get(),
                (sgl::platform::runtime_directory() / ("dds" MI_BASE_DLL_FILE_EXT)).string().c_str()
            ))
            FALCOR_THROW("Failed to load the 'dds' plugin.\n");

        mdl_utils::ConfigureOptions configure_options;
        configure_options.add_admin_space_search_paths = true;
        configure_options.add_user_space_search_paths = false;

        // // Add user search paths.
        // for (const auto& path : Settings::getGlobalSettings().getSearchDirectories("mdl"))
        //     configure_options.additional_mdl_paths.push_back(path.string());

        // Add search path to MDL core definitions.
        configure_options.additional_mdl_paths.push_back((sgl::platform::runtime_directory() / "mdl").string());

        if (std::string error; !mdl_utils::configure_mdl(neuray.get(), configure_options, error))
            FALCOR_THROW("Failed to configure the MDL SDK:\n{}", error);

        mi::Sint32 result = neuray->start(true);
        if (result != 0)
            FALCOR_THROW("Failed to initialize the MDL SDK. Result code: {}", result);
    }

    sgl::SharedLibraryHandle mdl;
    mi::base::Handle<mi::neuraylib::INeuray> neuray;
    std::map<std::string, int> path_frequency;

    void* get_neuray() override { return neuray.get(); };
    bool add_search_paths(const std::span<const std::filesystem::path>& paths) override
    {
        if (!neuray)
            return false;
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>()
        );
        std::vector<std::string> new_string_paths;
        for (const auto& path : paths) {
            const std::string& str_path = path.string();
            if (path_frequency[str_path]++ == 0)
                new_string_paths.push_back(str_path);
        }
        std::string outError;
        return mdl_utils::add_search_paths(mdl_config.get(), new_string_paths, outError);
    }

    bool remove_search_paths(const std::span<const std::filesystem::path>& paths) override
    {
        if (!neuray)
            return false;
        mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
            neuray->get_api_component<mi::neuraylib::IMdl_configuration>()
        );
        std::vector<std::string> removed_paths;
        for (const auto& path : paths) {
            const std::string& str_path = path.string();
            if (--path_frequency[str_path] == 0)
                removed_paths.push_back(str_path);
        }
        std::string outError;
        return mdl_utils::remove_search_paths(mdl_config.get(), removed_paths, outError);
    }
};

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

inline bool build_modified_material(
    const MDLContext::CompileRequest& request,
    const std::string dst_simple_material_name,
    const char* src_module_db_name,
    const mi::neuraylib::ICompiled_material* src_material,
    const mi::neuraylib::IFunction_definition* src_definition,
    const mi::neuraylib::IExpression_list* instance_args,
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

    mi::base::Handle<mi::neuraylib::IMdl_module_builder> module_builder(factory->create_module_builder(
        trans,
        src_module_db_name,
        /*ignored*/ mi::neuraylib::MDL_VERSION_1_0,
        mi::neuraylib::MDL_VERSION_LATEST,
        context
    ));

    // clone parameter defaults
    mi::Size parameter_count = src_material->get_parameter_count();
    for (mi::Size i = 0; i < parameter_count; ++i) {
        const char* name = src_material->get_parameter_name(i);
        if (name == nullptr) {
            out_error_msg = "Unexpected nameless parameter";
            return false;
        }

        mi::base::Handle<const mi::neuraylib::IAnnotation_block> src_param_annotation(
            src_parameter_annotations->get_annotation_block(name)
        );
        if (src_param_annotation) {
            dst_parameter_annotations->add_annotation_block(name, src_param_annotation.get());
        }

        mi::base::Handle<const mi::neuraylib::IType> src_type(src_parameters->get_type(name));
        dst_parameters->add_type(name, src_type.get());

        // If there is a provided instance argument with the same name, use it to set the default value.
        // Otherwise, use the value specified by the original source argument.
        mi::base::Handle<const mi::neuraylib::IExpression> arg(
            instance_args ? instance_args->get_expression(name) : nullptr
        );
        if (arg) {
            dst_defaults->add_expression(name, arg.get());
        } else {
            mi::base::Handle<const mi::neuraylib::IValue> src_default(src_material->get_argument(i));
            mi::base::Handle<const mi::neuraylib::IExpression> dst_default(ef->create_constant(src_default.get()));
            dst_defaults->add_expression(name, dst_default.get());
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
        const char* mollify_name = "mollify";
        mi::base::Handle<const mi::neuraylib::IType> mollify_type(tf->create_float());
        dst_parameters->add_type(mollify_name, mollify_type.get());
        dst_defaults->add_expression(mollify_name, epsilon_expression.get());
        dst_parameter_annotations->set_annotation_block(mollify_name, mollify_annotation_block.get());

        // create the parameterized roughness expression used to approximate specular BSDFs
        mi::Size mollify_parameter_index = dst_parameters->get_index(mollify_name);
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
        trans->access<mi::neuraylib::IModule>(src_module_db_name)
    );
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
    mi::neuraylib::ITransaction* trans
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
    mdl_impexp_api->load_module(trans, module_name.c_str(), context.get());
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to load the selected module.");

    // Access the module by name, which has to be queried using the factory.
    mi::base::Handle<const mi::IString> module_db_name(mdl_factory->get_db_module_name(module_name.c_str()));
    mi::base::Handle<const mi::neuraylib::IModule> module(
        trans->access<mi::neuraylib::IModule>(module_db_name->get_c_str())
    );
    if (!module)
        return createError("Failed to access the loaded module.");

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
    // When performing class compilation of material instances (i.e., ones that were USD-embedded), we always perform
    // compilation, even if the underlying material definition has been previously compiled for another material
    // instance. We rely on the fact that difference instances of the same material will have the same generated
    // code, and create a single module that is shared between the different instances. Rather than compiling a
    // material more than once, it would be better to use the pre-existing module directly and initialize the
    // per-instance arguments and resources separately. But currently, the only easy way to generate the argument
    // blocks and the like is through HLSL code generation.
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

    material_instance = {};

    std::map<std::string, Tagged_bsdf_info> tagged_bsdf_infos;
    if (modify_material) {
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
                    module_db_name->get_c_str(),
                    compiled_material.get(),
                    material_definition.get(),
                    useExistingInstance ? srcArgs.get() : nullptr,
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

    // Create a HLSL back-end for code generation.
    mi::base::Handle<mi::neuraylib::IMdl_backend_api> backend_api(
        neuray->get_api_component<mi::neuraylib::IMdl_backend_api>()
    );
    mi::base::Handle<mi::neuraylib::IMdl_backend> hlsl_backend(
        backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_HLSL)
    );

    // Set code generation options.
    hlsl_backend->set_option("internal_space", "coordinate_world");
    hlsl_backend->set_option("compile_constants", request.options.compile_constants ? "on" : "off");
    hlsl_backend->set_option("fast_math", request.options.fast_math ? "on" : "off");
    hlsl_backend->set_option("opt_level", std::to_string(request.options.opt_level).c_str());
    hlsl_backend->set_option("enable_auxiliary", request.options.enable_auxiliary ? "on" : "off");
    hlsl_backend->set_option("use_renderer_adapt_normal", request.options.use_renderer_adapt_normal ? "on" : "off");
    switch (request.options.df_handle_slot_mode) {
    case 0:
        hlsl_backend->set_option("df_handle_slot_mode", "none");
        break;
    case 1:
    case 2:
    case 4:
    case 8:
        hlsl_backend->set_option(
            "df_handle_slot_mode",
            fmt::format("fixed_{}", request.options.df_handle_slot_mode).c_str()
        );
        break;
    default:
        return createError("Invalid df_handle_slot_mode.");
    }
    hlsl_backend->set_option("texture_runtime_with_derivs", request.options.texture_runtime_with_derivs ? "on" : "off");
    hlsl_backend->set_option("num_texture_results", std::to_string(request.options.num_texture_results).c_str());
    hlsl_backend->set_option("num_texture_spaces", std::to_string(request.options.num_texture_spaces).c_str());
    hlsl_backend->set_option(
        "jit_warn_spectrum_conversion",
        request.options.jit_warn_spectrum_conversion ? "on" : "off"
    );

    // Create a link unit that will contain all the functions we want to generate code for.
    mi::base::Handle<mi::neuraylib::ILink_unit> hlsl_link_unit(hlsl_backend->create_link_unit(trans, context.get()));
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to create a link unit");

    enum FUNCTION_DESCRIPTOR_SLOTS {
        FUNC_DESC_SLOT_INIT = 0,
        FUNC_DESC_SLOT_IOR,
        FUNC_DESC_SLOT_THIN_WALLED,
        FUNC_DESC_SLOT_SURFACE_SCATTER,
        FUNC_DESC_SLOT_EMISSION_EMISSION,
        FUNC_DESC_SLOT_EMISSION_INTENSITY,
        FUNC_DESC_SLOT_EMISSION_MODE,
        FUNC_DESC_SLOT_BACKFACE_SCATERING,
        FUNC_DESC_SLOT_BACKFACE_EMISSION_EMISSION,
        FUNC_DESC_SLOT_BACKFACE_EMISSION_INTENSITY,
        FUNC_DESC_SLOT_BACKFACE_EMISSION_MODE,
        FUNC_DESC_SLOT_VOLUME_ABSORPTION_COEFFICIENT,
        FUNC_DESC_SLOT_VOLUME_SCATTERING_COEFFICIENT,
        FUNC_DESC_SLOT_GEOMETRY_NORMAL,
        FUNC_DESC_SLOT_GEOMETRY_CUTOUT_OPACITY,
        FUNC_DESC_SLOT_GEOMETRY_DISPLACEMENT,
        FUNC_DESC_SLOT_LAMBDA_FIRST
    };

    // Select the expressions to generate code for.
    using TD = mi::neuraylib::Target_function_description;
    std::vector<TD> descs;
    descs.resize(FUNC_DESC_SLOT_LAMBDA_FIRST);
    descs[FUNC_DESC_SLOT_INIT] = (TD("init", "init"));
    descs[FUNC_DESC_SLOT_IOR] = (TD("ior", "ior"));
    descs[FUNC_DESC_SLOT_THIN_WALLED] = (TD("thin_walled", "thin_walled"));
    descs[FUNC_DESC_SLOT_SURFACE_SCATTER] = (TD("surface.scattering", "surface_scattering"));
    descs[FUNC_DESC_SLOT_EMISSION_EMISSION] = (TD("surface.emission.emission", "surface_emission_emission"));
    descs[FUNC_DESC_SLOT_EMISSION_INTENSITY] = (TD("surface.emission.intensity", "surface_emission_intensity"));
    descs[FUNC_DESC_SLOT_EMISSION_MODE] = (TD("surface.emission.mode", "surface_emission_mode"));
    descs[FUNC_DESC_SLOT_BACKFACE_SCATERING] = (TD("backface.scattering", "backface_scattering"));
    descs[FUNC_DESC_SLOT_BACKFACE_EMISSION_EMISSION] = (TD("backface.emission.emission", "backface_emission_emission"));
    descs[FUNC_DESC_SLOT_BACKFACE_EMISSION_INTENSITY]
        = (TD("backface.emission.intensity", "backface_emission_intensity"));
    descs[FUNC_DESC_SLOT_BACKFACE_EMISSION_MODE] = (TD("backface.emission.mode", "backface_emission_mode"));
    descs[FUNC_DESC_SLOT_VOLUME_ABSORPTION_COEFFICIENT]
        = (TD("volume.absorption_coefficient", "volume_absorption_coefficient"));
    descs[FUNC_DESC_SLOT_VOLUME_SCATTERING_COEFFICIENT]
        = (TD("volume.scattering_coefficient", "volume_scattering_coefficient"));
    descs[FUNC_DESC_SLOT_GEOMETRY_NORMAL] = (TD("geometry.normal", "geometry_normal"));
    descs[FUNC_DESC_SLOT_GEOMETRY_CUTOUT_OPACITY] = (TD("geometry.cutout_opacity", "geometry_cutout_opacity"));
    descs[FUNC_DESC_SLOT_GEOMETRY_DISPLACEMENT] = (TD("geometry.displacement", "geometry_displacement"));

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

    hlsl_link_unit->add_material(compiled_material.get(), descs.data(), descs.size(), context.get());
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to select functions for code generation.");

    // Translating the link unit into the target language is the last step and the only one
    // that can be time consuming. All the steps before are designed to be lightweight for
    // interactively changing materials and parameters.
    mi::base::Handle<const mi::neuraylib::ITarget_code> hlsl_target_code(
        hlsl_backend->translate_link_unit(hlsl_link_unit.get(), context.get())
    );
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to translate the link unit to code.");

    // Next, we query the generated target code and all the resources it needs.
    // We store all of that in a CompileResult object.
    MDLContext::CompileResult result = {};

    result.code = hlsl_target_code->get_code();

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
        mi::Size surface_scatter_function_index = descs[FUNC_DESC_SLOT_SURFACE_SCATTER].function_index;
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
        size_t count = hlsl_target_code->get_argument_block_count();
        for (size_t i = 0; i < count; ++i) {
            mi::base::Handle<const mi::neuraylib::ITarget_argument_block> argBlock(
                hlsl_target_code->get_argument_block(i)
            );
            const uint8_t* data = reinterpret_cast<const uint8_t*>(argBlock->get_data());
            size_t size = argBlock->get_size();

            MDLContext::CompileResult::DataSegment segment{"", std::vector<uint8_t>(data, data + size)};
            result.arg_block_segments.push_back(segment);
        }
    }

    // Get parameters
    if (hlsl_target_code->get_argument_layout_count()) {
        result.arg_block_layout.clear();

        mi::Size argument_block_index = descs[FUNC_DESC_SLOT_SURFACE_SCATTER].argument_block_index;
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
                    case mi::neuraylib::IValue::Kind::VK_TEXTURE:
                        entry.type = MDLContext::CompileResult::ArgumentLayout::Type::texture;
                        FALCOR_ASSERT(entry.size == 4);
                        break;
                    case mi::neuraylib::IValue::Kind::VK_ENUM:
                    case mi::neuraylib::IValue::Kind::VK_DOUBLE:
                    case mi::neuraylib::IValue::Kind::VK_STRING:
                    case mi::neuraylib::IValue::Kind::VK_ARRAY:
                    case mi::neuraylib::IValue::Kind::VK_STRUCT:
                    case mi::neuraylib::IValue::Kind::VK_INVALID_DF:
                    case mi::neuraylib::IValue::Kind::VK_LIGHT_PROFILE:
                    case mi::neuraylib::IValue::Kind::VK_BSDF_MEASUREMENT:
                        break;
                    };

                    result.arg_block_layout.insert({std::string(name), entry});
                }
            }
        }
    }

    // Get textures.
    {
        size_t count = hlsl_target_code->get_texture_count();
        for (size_t i = 1; i < count; ++i) {
            const char* name = hlsl_target_code->get_texture(i);
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
            bool srgb = hlsl_target_code->get_texture_gamma(i) == mi::neuraylib::ITarget_code::GM_GAMMA_SRGB;
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

            MDLContext::CompileResult::Texture resultTexture;
            resultTexture.index = (uint32_t)i;
            resultTexture.name = name;
            resultTexture.path = filename;
            resultTexture.srgb = srgb;
            resultTexture.format = sgl::Format::undefined;
            resultTexture.width = width;
            resultTexture.height = height;
            resultTexture.depth = depth;

            mi::neuraylib::ITarget_code::Texture_shape shape = hlsl_target_code->get_texture_shape(i);
            switch (shape) {
            case mi::neuraylib::ITarget_code::Texture_shape_2d:
                resultTexture.type = MDLContext::CompileResult::Texture::Type::Texture2D;
                break;
            case mi::neuraylib::ITarget_code::Texture_shape_bsdf_data:
                resultTexture.type = MDLContext::CompileResult::Texture::Type::Texture3D;
                resultTexture.format = sgl::Format::r32_float;
                {
                    mi::base::Handle<const mi::neuraylib::ITile> tile(canvas->get_tile(0));
                    const uint8_t* data = reinterpret_cast<const uint8_t*>(tile->get_data());
                    size_t size = width * height * depth * sizeof(float);
                    resultTexture.data = std::vector<uint8_t>(data, data + size);
                }
                break;
            default:
                return createError("Unsupported texture type");
            }

            result.textures.push_back(resultTexture);
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

    // We need to query the index of refraction of the material.
    // To do this, we create a native backend where we can generate code for the "ior"
    // function and then run it to query the value.

    mi::base::Handle<mi::neuraylib::IMdl_backend> native_backend(
        backend_api->get_backend(mi::neuraylib::IMdl_backend_api::MB_NATIVE)
    );

    mi::base::Handle<mi::neuraylib::ILink_unit> native_link_unit(
        native_backend->create_link_unit(trans, context.get())
    );
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to create a native link unit");

    descs.clear();
    descs.push_back(TD("ior", "ior"));

    native_link_unit->add_material(compiled_material.get(), descs.data(), descs.size(), context.get());
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to select functions for code generation.");

    mi::base::Handle<const mi::neuraylib::ITarget_code> native_target_code(
        native_backend->translate_link_unit(native_link_unit.get(), context.get())
    );
    if (log_messages_and_check_errors(context.get()))
        return createError("Failed to translate the native link unit to code.");

    // Query index of refraction.
    {
        mi::neuraylib::Shading_state_material state = {};
        mi::Sint32 execute_result = native_target_code->execute(0, state, nullptr, nullptr, &result.ior);
        if (execute_result != 0) {
            sgl::log_warn("Failed to execute target code to get index of refraction: {}", execute_result);
            result.ior = float3(1.f);
        }
    }

    result.success = true;

    return result;
}

MDLContext::CompileResult MDLContext::compile(const CompileRequest& request)
{
    sgl::log_info("Compiling MDL to HLSL for material {} ...", request.qualified_material_name);

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
    CompileResult result = compileInternal(request, neuray, trans.get());

    // Close the transaction.
    if (result.success)
        trans->commit();
    else
        trans->abort();

    return result;
}

} // namespace falcor
