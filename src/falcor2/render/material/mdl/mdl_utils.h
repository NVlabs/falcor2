// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/error.h"
#include "falcor2/core/logger.h"
#include "falcor2/core/platform.h"

#include "mdl_sdk_wrapper.h"
#include "mdl_context.h"

#include <sgl/core/string.h>

#include <fmt/format.h>
#include <regex>


namespace falcor {
namespace mdl_utils {

class DefaultLogger : public mi::base::Interface_implement<mi::base::ILogger> {
public:
    void message(
        mi::base::Message_severity level,
        const char* /*module_category*/,
        const mi::base::Message_details& /*details*/,
        const char* message
    ) override
    {
        switch (level) {
        case mi::base::MESSAGE_SEVERITY_FATAL:
            sgl::log_fatal("MDL: {}", message);
            break;
        case mi::base::MESSAGE_SEVERITY_ERROR:
            sgl::log_error("MDL: {}", message);
            break;
        case mi::base::MESSAGE_SEVERITY_WARNING:
            sgl::log_warn("MDL: {}", message);
            break;
        case mi::base::MESSAGE_SEVERITY_INFO:
            // sgl::log_info("MDL: {}", message);
            break;
        case mi::base::MESSAGE_SEVERITY_VERBOSE:
            return;
        case mi::base::MESSAGE_SEVERITY_DEBUG:
            return;
        // NB: MESSAGE_SEVERITY_FORCE_32_BIT is only defined when `SWIG` is defined, and that depends on the MDL
        // version.
        // case mi::base::MESSAGE_SEVERITY_FORCE_32_BIT:
        default:
            return;
        }
    }

    void message(mi::base::Message_severity level, const char* module_category, const char* message) override
    {
        this->message(level, module_category, mi::base::Message_details(), message);
    }
};

extern DefaultLogger default_logger;

inline bool load_mdl_plugin(mi::neuraylib::INeuray* neuray, const char* path)
{
    mi::base::Handle<mi::neuraylib::IPlugin_configuration> plugin_conf(
        neuray->get_api_component<mi::neuraylib::IPlugin_configuration>()
    );

    // try to load the requested plugin before adding any special handling
    mi::Sint32 res = plugin_conf->load_plugin_library(path);
    return (res == 0);
}

inline bool load_mdl_and_get_ineuray(
    sgl::SharedLibraryHandle& out_handle,
    mi::base::Handle<mi::neuraylib::INeuray>& out_neuray,
    std::string& out_error,
    const char* filename = nullptr
)
{
    if (!filename)
        filename = "libmdl_sdk" MI_BASE_DLL_FILE_EXT;

    sgl::SharedLibraryHandle handle = falcor::platform::load_shared_library(filename);
    if (!handle) {
        out_error = "Failed to load MDL library";
        return false;
    }

    void* symbol = falcor::platform::get_proc_address(handle, "mi_factory");
    if (!symbol) {
        out_error = "Failed to get MDL factory function";
        return false;
    }

    mi::neuraylib::INeuray* neuray = mi::neuraylib::mi_factory<mi::neuraylib::INeuray>(symbol);
    if (!neuray) {
        mi::base::Handle<mi::neuraylib::IVersion> version(mi::neuraylib::mi_factory<mi::neuraylib::IVersion>(symbol));
        if (!version)
            out_error = "Failed to get MDL version interface";
        else
            out_error = fmt::format(
                "Library version {} does not match header version {}",
                version->get_product_version(),
                MI_NEURAYLIB_PRODUCT_VERSION_STRING
            );
        return false;
    }

    out_handle = handle;
    out_neuray = neuray;
    return true;
}

inline void unload_mdl(sgl::SharedLibraryHandle handle)
{
    sgl::platform::release_shared_library(handle);
}

struct ConfigureOptions {
    /// additional search paths that are added after admin/user and the example search paths
    std::vector<std::string> additional_mdl_paths;

    /// set to false to not add the admin space search paths. It's recommend to leave this true.
    bool add_admin_space_search_paths = true;

    /// set to false to not add the user space search paths. It's recommend to leave this true.
    bool add_user_space_search_paths = true;

    /// set to true to disable (optional) plugin loading.
    bool skip_loading_plugins = false;

    /// set a custom logger if we want to use a different one than DefaultLogger.
    mi::base::ILogger* logger = nullptr;
};

inline bool add_search_paths(
    mi::neuraylib::IMdl_configuration* mdl_config,
    const std::vector<std::string>& mdl_paths,
    std::string& out_error
)
{
    bool success = true;
    // add mdl and resource paths to allow the neuray API to resolve relative textures
    for (size_t i = 0, n = mdl_paths.size(); i < n; ++i) {
        if (mdl_config->add_mdl_path(mdl_paths[i].c_str()) != 0
            || mdl_config->add_resource_path(mdl_paths[i].c_str()) != 0) {
            success = false;
            out_error += fmt::format("Failed to set MDL path \"{}\"\n", mdl_paths[i]);
        }
    }

    return success;
}

inline bool remove_search_paths(
    mi::neuraylib::IMdl_configuration* mdl_config,
    const std::vector<std::string>& mdl_paths,
    std::string& out_error
)
{
    bool success = true;
    // add mdl and resource paths to allow the neuray API to resolve relative textures
    for (size_t i = 0, n = mdl_paths.size(); i < n; ++i) {
        if (mdl_config->remove_mdl_path(mdl_paths[i].c_str()) != 0
            || mdl_config->remove_resource_path(mdl_paths[i].c_str()) != 0) {
            success = false;
            out_error += fmt::format("Failed to remove MDL path \"{}\"\n", mdl_paths[i]);
        }
    }

    return success;
}

inline bool configure_mdl(mi::neuraylib::INeuray* neuray, const ConfigureOptions& options, std::string& out_error)
{
    FALCOR_ASSERT(neuray);

    bool success = true;

    mi::base::Handle<mi::neuraylib::ILogging_configuration> mdl_logging(
        neuray->get_api_component<mi::neuraylib::ILogging_configuration>()
    );

    // set user defined or default logger
    // mdl_logging->set_receiving_logger(options.logger ? options.logger : &default_logger);


    mi::base::Handle<mi::neuraylib::IMdl_configuration> mdl_config(
        neuray->get_api_component<mi::neuraylib::IMdl_configuration>()
    );

    // set the module and texture search path.
    // mind the order
    std::vector<std::string> mdl_paths;

    if (options.add_admin_space_search_paths) {
        mdl_config->add_mdl_system_paths();
    }

    if (options.add_user_space_search_paths) {
        mdl_config->add_mdl_user_paths();
    }

    // ignore if any of these do not exist
    mdl_paths.insert(mdl_paths.end(), options.additional_mdl_paths.begin(), options.additional_mdl_paths.end());

    return add_search_paths(mdl_config.get(), mdl_paths, out_error);
}

inline bool parse_cmd_argument_material_name(
    const std::string& in_argument,
    std::string& out_module_name,
    std::string& out_material_name,
    std::string& out_error,
    bool prepend_colons_if_missing = true
)
{
    out_module_name = "";
    out_material_name = "";

    // Convert path to MDL module path syntax by replacing '/' or '\' in the input name to '::',
    // unless the character is preceded by a ':'.
    // For example, "C:\path\to\module" becomes "C:\path::to::module".
    static std::regex separatorRegex("([^:])[/\\\\]");
    std::string argument = std::regex_replace(in_argument, separatorRegex, "$1::");

    std::size_t p_left_paren = argument.rfind('(');
    if (p_left_paren == std::string::npos)
        p_left_paren = argument.size();
    std::size_t p_last = argument.rfind("::", p_left_paren - 1);

    bool starts_with_colons = argument.length() > 2 && argument[0] == ':' && argument[1] == ':';

    // check for mdle
    if (!starts_with_colons) {
        std::string potential_path = argument;
        std::string potential_material_name = "main";

        // input already has ::main attached (optional)
        if (p_last != std::string::npos) {
            potential_path = argument.substr(0, p_last);
            potential_material_name = argument.substr(p_last + 2, argument.size() - p_last);
        }

        // is it an mdle?
        if (sgl::string::has_suffix(potential_path, ".mdle", true)) {
            if (potential_material_name != "main") {
                out_error = fmt::format(
                    "Material and module name cannot be extracted from '{}'.\n"
                    "The module was detected as MDLE but the selected material is different from 'main'.\n",
                    argument
                );
                return false;
            }
            out_module_name = potential_path;
            out_material_name = potential_material_name;
            return true;
        }
    }

    if (p_last == std::string::npos || p_last == 0 || p_last == argument.length() - 2
        || (!starts_with_colons && !prepend_colons_if_missing)) {
        out_error = fmt::format(
            "Material and module name cannot be extracted from '{}'.\n"
            "An absolute fully-qualified material name of form '[::<package>]::<module>::<material>' is expected.\n",
            argument
        );
        return false;
    }

    if (!starts_with_colons && prepend_colons_if_missing) {
        out_module_name = "::";
    }

    out_module_name.append(argument.substr(0, p_last));
    out_material_name = argument.substr(p_last + 2, argument.size() - p_last);
    return true;
}

inline std::string
add_missing_material_signature(const mi::neuraylib::IModule* module, const std::string& material_name)
{
    // Return input if it already contains a signature.
    if (material_name.back() == ')')
        return material_name;

    mi::base::Handle<const mi::IArray> result(module->get_function_overloads(material_name.c_str()));
    if (!result || result->get_length() != 1)
        return std::string();

    mi::base::Handle<const mi::IString> overloads(result->get_element<mi::IString>(static_cast<mi::Size>(0)));
    return overloads->get_c_str();
}

// Returns a string-representation of the given message severity
inline std::string to_string(mi::base::Message_severity severity)
{
    switch (severity) {
    case mi::base::MESSAGE_SEVERITY_FATAL:
        return "fatal";
    case mi::base::MESSAGE_SEVERITY_ERROR:
        return "error";
    case mi::base::MESSAGE_SEVERITY_WARNING:
        return "warning";
    case mi::base::MESSAGE_SEVERITY_INFO:
        return "info";
    case mi::base::MESSAGE_SEVERITY_VERBOSE:
        return "verbose";
    case mi::base::MESSAGE_SEVERITY_DEBUG:
        return "debug";
    default:
        break;
    }
    return "";
}

// --------------------------------------------------------------------------------------------

// Returns a string-representation of the given message category
inline std::string to_string(mi::neuraylib::IMessage::Kind message_kind)
{
    switch (message_kind) {
    case mi::neuraylib::IMessage::MSG_INTEGRATION:
        return "MDL SDK";
    case mi::neuraylib::IMessage::MSG_IMP_EXP:
        return "Importer/Exporter";
    case mi::neuraylib::IMessage::MSG_COMPILER_BACKEND:
        return "Compiler Backend";
    case mi::neuraylib::IMessage::MSG_COMPILER_CORE:
        return "Compiler Core";
    case mi::neuraylib::IMessage::MSG_COMPILER_ARCHIVE_TOOL:
        return "Compiler Archive Tool";
    case mi::neuraylib::IMessage::MSG_COMPILER_DAG:
        return "Compiler DAG generator";
    default:
        break;
    }
    return "";
}
} // namespace mdl_utils

} // namespace falcor
