// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

#include <string>
#include <string_view>

namespace falcor {
namespace materialx {
namespace mx139 {

using materialx::CodeGenDesc;

// Describes the generated naming and library conventions for the MaterialX 1.39
// BSDF profile. This is front-end policy only: it derives names from CodeGenDesc
// and never reads or mutates the MaterialX graph.
struct ProfileDesc {
    std::string display_name;
    std::string generated_module_prefix;
    std::string generated_prefix;
    std::string library_import;
    std::string bsdf_prefix;
    std::string stack_data_type;

    std::string bsdf_type(std::string_view stem) const { return bsdf_prefix + std::string(stem) + "BSDF"; }

    std::string helper_type(std::string_view stem) const { return bsdf_prefix + std::string(stem); }
};

// Resolves the current Mx139 profile descriptor.
const ProfileDesc& profile_desc();

// Returns the generated BSDF type name for a Fresnel lobe after applying Mx139
// Fresnel and compensation naming policy.
std::string fresnel_bsdf_type(const ProfileDesc& profile, const CodeGenDesc& desc, std::string_view stem);

} // namespace mx139
} // namespace materialx
} // namespace falcor
