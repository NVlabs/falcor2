// SPDX-License-Identifier: Apache-2.0

#include "mdl_discovery.h"

#include "falcor2/core/error.h"
#include "falcor2/render/material/mdl/mdl_context.h"
#include "falcor2/render/material/mdl/mdl_sdk_wrapper.h"

#include <algorithm>
#include <cctype>
#include <mutex>

// This helper follows the MDL SDK discovery example/documentation pattern:
// add an MDL search path, run IMdl_discovery_api::discover(), traverse the
// returned package/module graph, load each selected module, and enumerate its
// material definitions. The Falcor-specific pieces are the scoped use of
// MDLContext and the conversion into MDLDiscoveredMaterial records.

namespace falcor {
namespace {

std::mutex g_mdl_discovery_mutex;

struct DiscoveredModule {
    std::string qualified_name;
    std::string simple_name;
    std::string resolved_path;
};

std::string normalized_path_string(std::filesystem::path path)
{
    std::string text = std::filesystem::weakly_canonical(path).generic_string();
#ifdef _WIN32
    std::transform(
        text.begin(),
        text.end(),
        text.begin(),
        [](unsigned char ch)
        {
            return static_cast<char>(std::tolower(ch));
        }
    );
#endif
    while (text.size() > 1 && text.back() == '/')
        text.pop_back();
    return text;
}

bool path_is_within(const std::filesystem::path& path, const std::filesystem::path& root)
{
    const std::string path_text = normalized_path_string(path);
    const std::string root_text = normalized_path_string(root);
    return path_text == root_text || path_text.starts_with(root_text + "/");
}

std::string strip_leading_scope(std::string value)
{
    while (value.starts_with("::"))
        value.erase(0, 2);
    return value;
}

bool module_included(const mi::neuraylib::IMdl_module_info* module, const std::vector<std::string>& include_modules)
{
    if (include_modules.empty())
        return true;

    const std::string qualified = strip_leading_scope(module->get_qualified_name() ? module->get_qualified_name() : "");
    const std::string simple = module->get_simple_name() ? module->get_simple_name() : "";
    for (const std::string& include_module : include_modules) {
        if (include_module == simple || include_module == qualified)
            return true;
        if (qualified.find(include_module) != std::string::npos)
            return true;
    }
    return false;
}

class ScopedSearchPath {
public:
    ScopedSearchPath(MDLContext& context, std::filesystem::path path)
        : m_context(context)
        , m_path(std::move(path))
    {
        FALCOR_CHECK(m_context.add_search_paths({&m_path, 1}), "Failed to add MDL discovery search path.");
        m_armed = true;
    }

    ~ScopedSearchPath()
    {
        if (m_armed)
            m_context.remove_search_paths({&m_path, 1});
    }

private:
    MDLContext& m_context;
    std::filesystem::path m_path;
    bool m_armed = false;
};

void collect_modules(
    const mi::neuraylib::IMdl_info* info,
    const std::filesystem::path& root,
    const std::vector<std::string>& include_modules,
    std::vector<DiscoveredModule>& modules
)
{
    if (!info)
        return;

    if (info->get_kind() == mi::neuraylib::IMdl_info::DK_MODULE) {
        mi::base::Handle<const mi::neuraylib::IMdl_module_info> module(
            info->get_interface<const mi::neuraylib::IMdl_module_info>()
        );
        if (!module || module->in_archive() || !module_included(module.get(), include_modules))
            return;
        mi::base::Handle<const mi::IString> resolved(module->get_resolved_path());
        if (!resolved || !resolved->get_c_str())
            return;
        if (!path_is_within(std::filesystem::path(resolved->get_c_str()), root))
            return;
        modules.push_back({
            module->get_qualified_name() ? module->get_qualified_name() : "",
            module->get_simple_name() ? module->get_simple_name() : "",
            resolved->get_c_str(),
        });
        return;
    }

    if (info->get_kind() != mi::neuraylib::IMdl_info::DK_PACKAGE)
        return;

    mi::base::Handle<const mi::neuraylib::IMdl_package_info> package(
        info->get_interface<const mi::neuraylib::IMdl_package_info>()
    );
    if (!package)
        return;
    for (mi::Size index = 0, count = package->get_child_count(); index < count; ++index)
        collect_modules(package->get_child(index), root, include_modules, modules);
}

} // namespace

std::vector<MDLDiscoveredMaterial>
discover_mdl_materials(const std::filesystem::path& library_path, const std::vector<std::string>& include_modules)
{
    std::lock_guard<std::mutex> lock(g_mdl_discovery_mutex);

    MDLContext& context = MDLContext::get();
    auto* neuray = static_cast<mi::neuraylib::INeuray*>(context.get_neuray());
    FALCOR_CHECK(neuray, "MDLContext backend did not expose a valid Neuray interface.");

    const std::filesystem::path root = std::filesystem::weakly_canonical(library_path);
    ScopedSearchPath scoped_search_path(context, root);

    mi::base::Handle<mi::neuraylib::IMdl_discovery_api> discovery_api(
        neuray->get_api_component<mi::neuraylib::IMdl_discovery_api>()
    );
    FALCOR_CHECK(discovery_api, "Failed to access MDL discovery API.");
    mi::base::Handle<const mi::neuraylib::IMdl_discovery_result> discovery(
        discovery_api->discover(static_cast<mi::Uint32>(mi::neuraylib::IMdl_info::DK_MODULE))
    );
    FALCOR_CHECK(discovery, "MDL discovery returned no result.");

    std::vector<DiscoveredModule> modules;
    collect_modules(discovery->get_graph(), root, include_modules, modules);
    std::sort(
        modules.begin(),
        modules.end(),
        [](const DiscoveredModule& lhs, const DiscoveredModule& rhs)
        {
            return lhs.qualified_name < rhs.qualified_name;
        }
    );

    mi::base::Handle<mi::neuraylib::IDatabase> database(neuray->get_api_component<mi::neuraylib::IDatabase>());
    mi::base::Handle<mi::neuraylib::IScope> scope(database->get_global_scope());
    mi::base::Handle<mi::neuraylib::ITransaction> transaction(scope->create_transaction());
    mi::base::Handle<mi::neuraylib::IMdl_impexp_api> import_api(
        neuray->get_api_component<mi::neuraylib::IMdl_impexp_api>()
    );
    mi::base::Handle<mi::neuraylib::IMdl_factory> factory(neuray->get_api_component<mi::neuraylib::IMdl_factory>());

    std::vector<MDLDiscoveredMaterial> records;
    for (const DiscoveredModule& module_info : modules) {
        const std::string& module_name = module_info.qualified_name;
        mi::base::Handle<mi::neuraylib::IMdl_execution_context> execution_context(factory->create_execution_context());
        import_api->load_module(transaction.get(), module_name.c_str(), execution_context.get());
        if (execution_context->get_error_messages_count() > 0)
            continue;

        mi::base::Handle<const mi::IString> module_db_name(factory->get_db_module_name(module_name.c_str()));
        mi::base::Handle<const mi::neuraylib::IModule> module(
            transaction->access<mi::neuraylib::IModule>(module_db_name->get_c_str())
        );
        if (!module)
            continue;

        for (mi::Size index = 0, count = module->get_material_count(); index < count; ++index) {
            const char* material_db_name = module->get_material(index);
            if (!material_db_name)
                continue;
            mi::base::Handle<const mi::neuraylib::IFunction_definition> definition(
                transaction->access<mi::neuraylib::IFunction_definition>(material_db_name)
            );
            if (!definition)
                continue;

            const std::string mdl_material_name = strip_leading_scope(definition->get_mdl_name());
            records.push_back({
                strip_leading_scope(module_name),
                module_info.simple_name,
                module_info.resolved_path,
                definition->get_mdl_simple_name() ? definition->get_mdl_simple_name() : "",
                mdl_material_name,
                mdl_material_name,
            });
        }
    }

    transaction->commit();
    return records;
}

} // namespace falcor
