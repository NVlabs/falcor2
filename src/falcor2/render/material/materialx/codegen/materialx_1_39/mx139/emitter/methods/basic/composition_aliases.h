// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../../layering_analysis.h"
#include "../../../profile_policy.h"

#include <map>
#include <string>
#include <vector>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Generated type-alias namespace used by closure_tree and bsdf_mix composition
// emitters. It is a C++ model of Slang type aliases only; building it does not
// mutate the MaterialX graph or emit any source.
struct CompositionAliases {
    struct AliasDesc {
        std::string name;
        std::string type;
    };

    std::string namespace_name;
    std::vector<AliasDesc> aliases;
    std::map<ClosureRef, std::string> core_aliases;
    std::map<ClosureRef, std::string> weighted_aliases;
    bool uses_weighted_helpers = true;
};

// Returns the local value-name prefix for a generated combiner value.
std::string combiner_value_prefix(CombinerMode mode);

// Builds recursive composition aliases for closure_tree composition emission.
CompositionAliases
build_composition_aliases(const LayeringDesc& desc, const ProfileDesc& profile, std::string namespace_name);

// Builds the minimal alias set for bsdf_mix composition emission, where the
// bsdf_mix root type already encodes the recursive composition.
CompositionAliases build_bsdf_mix_composition_aliases(
    const LayeringDesc& desc,
    std::string namespace_name,
    const std::string& root_type_name
);

// Returns either a local `mtlx_ns::Name` reference or a fully qualified alias
// reference depending on where the emitter is writing source.
std::string alias_reference(const CompositionAliases& aliases, const std::string& name, bool use_local_aliases);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
