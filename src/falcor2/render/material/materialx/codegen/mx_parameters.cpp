// SPDX-License-Identifier: Apache-2.0

#include "mx_parameters.h"
#include <falcor2/core/properties.h>

namespace falcor {
namespace materialx {
namespace params {

bool UserInterfaceInfo::operator<(const UserInterfaceInfo& rhs) const
{
    const std::vector<std::string>& lhs_folders = folders;
    const std::vector<std::string>& rhs_folders = rhs.folders;
    const size_t minCount = std::min(lhs_folders.size(), rhs_folders.size());
    // First sort alphabetically on the shared folders
    for (size_t i = 0; i < minCount; ++i)
        if (lhs_folders[i] != rhs_folders[i])
            return lhs_folders[i] < rhs_folders[i];
    // If all folders are the same, then prioritize the parameter that is,
    // in folder's root, over those in the same folder's subfolders.
    if (lhs_folders.size() != rhs_folders.size())
        return lhs_folders.size() < rhs_folders.size();
    // If parameters are in the same (sub)folder, sort by name.
    return name < rhs.name;
}

} // namespace params

void MxParamInfo::set_value(const Properties& props)
{
    if (!is_editable)
        return;

    const std::string& property_name = "inputs:" + param_name;
    if (!props.has_property(property_name))
        return;

    std::visit(
        [&](auto&& value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (!std::is_same_v<T, float4x4> && !std::is_same_v<T, std::unique_ptr<params::TextureInfo>>) {
                value = props.get<T>(property_name);
            }
        },
        *param_value
    );
}

void MxParamInfo::set_value(const reflection::DynamicPropertySet& props)
{
    if (!is_editable)
        return;

    const std::string& property_name = "inputs:" + param_name;
    if (!props.has_property(property_name))
        return;

    std::visit(
        [&](auto&& value)
        {
            using T = std::decay_t<decltype(value)>;
            if constexpr (!std::is_same_v<T, float4x4> && !std::is_same_v<T, std::unique_ptr<params::TextureInfo>>) {
                value = props.get<T>(property_name);
            }
        },
        *param_value
    );
}

void MxParamInfo::to_dictionary(IDictionary& dict) const
{
    if (!is_editable)
        return;

    std::visit(
        [&](auto&& value)
        {
            using T = std::decay_t<decltype(value)>;
            using Trait = typename params::TypeTraits<T>;
            FALCOR_ASSERT(Trait::type == param_type);
            dict[param_name] = Trait::extract_value(value);
        },
        *param_value
    );
}


template<typename CursorT>
void MxParamInfo::set_cursor(CursorT& cursor) const
{
    if (!is_editable)
        return;

    std::visit(
        [&](auto&& value)
        {
            using T = std::decay_t<decltype(value)>;
            using Trait = typename params::TypeTraits<T>;
            FALCOR_ASSERT(Trait::type == param_type);
            cursor[param_name.c_str()] = Trait::extract_value(value);
        },
        *param_value
    );
}

// Explicit instantiations for BufferElementCursor and ShaderCursor
template void MxParamInfo::set_cursor(sgl::BufferElementCursor& cursor) const;
template void MxParamInfo::set_cursor(sgl::ShaderCursor& cursor) const;


void MxParamList::add_param_info(MxParamInfo param)
{
    if (param.is_editable)
        ++m_editable_count;
    m_params.push_back(std::move(param));
}

} // namespace materialx
} // namespace falcor
