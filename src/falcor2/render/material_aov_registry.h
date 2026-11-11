// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include <sgl/core/data_type.h>
#include <cstdint>
#include <map>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

namespace falcor {

/// Owned AOV declaration used for catalog entries and lookup results.
/// Copies do not retain the registry; IDs are meaningful only in their originating registry.
struct AOVDesc {
    uint32_t id;
    std::string name;
    sgl::DataType type;
    uint32_t component_count;
    bool operator==(const AOVDesc&) const = default;
};

/// Append-only scene-local catalog of typed material AOV declarations.
/// Only producers register names. Consumers look them up without reserving IDs.
/// IDs and types persist for the registry lifetime, including after material removal
/// or subsequent compilation failure. Registration and lookup are single-threaded.
class FALCOR_API MaterialAOVRegistry : public Object {
    FALCOR_OBJECT(MaterialAOVRegistry)
public:
    /// Declare a float1-4 producer. Repeated declarations must agree on type.
    uint32_t register_output(std::string_view name, sgl::DataType type, uint32_t component_count);
    /// Look up a registered declaration. Missing names do not allocate IDs.
    std::optional<AOVDesc> find_output(std::string_view name) const;
    /// Copied declarations in ID order, for discovery.
    std::vector<AOVDesc> outputs() const { return m_entries; }
    /// Whether the name has ever been registered in this catalog.
    bool has_output(std::string_view name) const { return m_name_to_id.contains(name); }
    /// Catalog revision; changes only when a new ID is added.
    uint64_t output_generation() const { return m_entries.size(); }

private:
    std::map<std::string, uint32_t, std::less<>> m_name_to_id;
    std::vector<AOVDesc> m_entries;
};
} // namespace falcor
