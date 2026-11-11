// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "codegen_cache.h"

namespace falcor::mtlx {

bool CodeGenCache::find(const Key& key, CodeGenResult& result) const
{
    std::shared_ptr<const Entry> entry;
    {
        std::lock_guard lock(m_mutex);
        auto it = m_entries.find(key);
        if (it == m_entries.end())
            return false;
        entry = it->second;
    }
    result.module_name = entry->module_name;
    result.module_source = entry->module_source;
    result.material_name = entry->material_name;
    result.material_instance_name = entry->material_instance_name;
    result.material_data_name = entry->material_data_name;
    result.data_struct_size = entry->data_struct_size;
    result.uses_geometry_properties = entry->uses_geometry_properties;
    return true;
}

void CodeGenCache::insert(const Key& key, const CodeGenResult& result)
{
    auto entry = std::make_shared<const Entry>(Entry{
        result.module_name,
        result.module_source,
        result.material_name,
        result.material_instance_name,
        result.material_data_name,
        result.data_struct_size,
        result.uses_geometry_properties,
    });
    // Concurrent misses may generate twice; only complete entries are published.
    std::lock_guard lock(m_mutex);
    m_entries.try_emplace(key, std::move(entry));
}

size_t CodeGenCache::size() const
{
    std::lock_guard lock(m_mutex);
    return m_entries.size();
}

} // namespace falcor::mtlx
