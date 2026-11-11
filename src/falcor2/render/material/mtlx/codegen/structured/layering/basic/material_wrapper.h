// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

namespace falcor::mtlx::structured {

struct EmissionContext;

struct BasicTypeAlias {
    std::string name;
    std::string type;
};

/// Backend-owned pieces inserted into the common closure-tree/BSDF-mix
/// material wrapper. This describes material concepts rather than snippet
/// tokens; common ancillary behavior and source assembly stay in one place.
struct BasicMaterialWrapperDesc {
    std::string root_type_source;
    std::string aliases_name;
    std::string aliases_comment;
    std::vector<BasicTypeAlias> aliases;
    std::string lobe_flags;
    std::string root_value_lines;
    std::string root_value;
    std::string collect_properties_method;
    std::string collect_extra_bsdf_properties_method;
};

void emit_basic_material_wrapper(EmissionContext& context, const BasicMaterialWrapperDesc& desc);

} // namespace falcor::mtlx::structured
