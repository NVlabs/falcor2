// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "material_aov_registry.h"
#include "falcor2/core/error.h"
#include <limits>

namespace falcor {

uint32_t MaterialAOVRegistry::register_output(std::string_view name, sgl::DataType type, uint32_t component_count)
{
    FALCOR_CHECK(!name.empty(), "Material AOV names must not be empty.");
    FALCOR_CHECK(
        type == sgl::DataType::float32 && component_count >= 1 && component_count <= 4,
        "Material AOV '{}' must use float32 with one through four components.",
        name
    );
    if (const auto iter = m_name_to_id.find(name); iter != m_name_to_id.end()) {
        const AOVDesc& desc = m_entries[iter->second];
        FALCOR_CHECK(
            desc.type == type && desc.component_count == component_count,
            "Material AOV '{}' has an incompatible value type.",
            name
        );
        return iter->second;
    }
    FALCOR_CHECK(
        m_entries.size() <= std::numeric_limits<uint32_t>::max(),
        "Material AOV registry exhausted its 32-bit ID space."
    );
    const uint32_t id = static_cast<uint32_t>(m_entries.size());
    m_entries.push_back({id, std::string(name), type, component_count});
    try {
        m_name_to_id.emplace(m_entries.back().name, id);
    } catch (...) {
        m_entries.pop_back();
        throw;
    }
    return id;
}

std::optional<AOVDesc> MaterialAOVRegistry::find_output(std::string_view name) const
{
    if (const auto iter = m_name_to_id.find(name); iter != m_name_to_id.end())
        return m_entries[iter->second];
    return {};
}
} // namespace falcor
