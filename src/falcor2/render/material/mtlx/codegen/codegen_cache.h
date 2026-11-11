// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "codegen_types.h"
#include <sgl/core/crypto.h>
#include <sgl/core/hash.h>

#include <mutex>
#include <unordered_map>

namespace falcor::mtlx {

/// In-memory generated code shared by materials. Instance values and MaterialX graphs stay request-local.
/// Concurrent lookup/publication is safe; callers must separately synchronize their own registries and callbacks.
class FALCOR_API CodeGenCache {
public:
    using Key = sgl::SHA1::Digest;

    /// On a hit, copy generated-code fields only; request-specific fields remain unchanged.
    /// On a miss, leave result unchanged.
    bool find(const Key& key, CodeGenResult& result) const;
    void insert(const Key& key, const CodeGenResult& result);
    size_t size() const;

private:
    struct KeyHash {
        size_t operator()(const Key& key) const
        {
            return sgl::hash(std::string_view(reinterpret_cast<const char*>(key.data()), key.size()));
        }
    };

    struct Entry {
        std::string module_name;
        std::string module_source;
        std::string material_name;
        std::string material_instance_name;
        std::string material_data_name;
        size_t data_struct_size;
        bool uses_geometry_properties;
    };

    mutable std::mutex m_mutex;
    std::unordered_map<Key, std::shared_ptr<const Entry>, KeyHash> m_entries;
};

} // namespace falcor::mtlx
