// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/reflection/class_descriptor.h"

#include "falcor2/core/macros.h"

#include <span>
#include <string_view>
#include <typeindex>
#include <typeinfo>
#include <unordered_map>
#include <vector>

namespace falcor::reflection {

// ----------------------------------------------------------------------------
// TypeRegistry
// ----------------------------------------------------------------------------

/// Singleton registry of all reflected class descriptors.
///
/// Thread safety: not thread-safe for concurrent registration. All
/// register_class() calls must happen during static initialization on a
/// single thread before any concurrent lookups. After initialization,
/// find() and all() are safe to call concurrently from multiple threads
/// since the registry is then read-only.
class FALCOR_API TypeRegistry {
public:
    /// Returns the singleton instance.
    static TypeRegistry& get();

    ~TypeRegistry();

    /// Register a class descriptor. Takes ownership.
    /// Throws if a class with the same std::type_info is already registered.
    void register_class(const ClassDescriptor* desc);

    /// Find a class descriptor by type_info. Returns null if not registered.
    const ClassDescriptor* find(const std::type_info& type) const;

    /// Find a class descriptor by name. Returns null if not registered.
    const ClassDescriptor* find(std::string_view name) const;

    /// All registered class descriptors.
    std::span<const ClassDescriptor* const> all() const { return {m_descs.data(), m_descs.size()}; }

private:
    TypeRegistry() = default;

    /// Descriptors (owned).
    std::vector<const ClassDescriptor*> m_descs;
    /// type_index -> descriptor.
    std::unordered_map<std::type_index, const ClassDescriptor*> m_by_type;
    /// name -> descriptor.
    std::unordered_map<std::string_view, const ClassDescriptor*> m_by_name;

    FALCOR_NON_COPYABLE_AND_MOVABLE(TypeRegistry);
};

/// Look up a class descriptor by type_info. Throws if the type is not registered.
FALCOR_API const ClassDescriptor& get_class_descriptor(const std::type_info& type);

} // namespace falcor::reflection
