// SPDX-License-Identifier: Apache-2.0

#include "composition_aliases.h"

#include "falcor2/core/error.h"

#include <fmt/format.h>

#include <utility>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

namespace {

std::string combiner_alias_prefix(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return "Layer";
    case CombinerMode::mix:
        return "Mix";
    case CombinerMode::add:
        return "Add";
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

void add_type_alias(CompositionAliases& aliases, std::string name, std::string type)
{
    aliases.aliases.push_back({std::move(name), std::move(type)});
}

void build_composition_aliases_recursive(
    const LayeringDesc& desc,
    ClosureRef ref,
    const ProfileDesc& profile,
    CompositionAliases& aliases
)
{
    if (aliases.weighted_aliases.contains(ref))
        return;

    if (ref.is_bsdf()) {
        const std::size_t bsdf_index = ref.bsdf_index();
        aliases.core_aliases[ref] = desc.bsdfs[bsdf_index].bsdf_type;
        const std::string weighted_alias = fmt::format("Bsdf{}", bsdf_index);
        aliases.weighted_aliases[ref] = weighted_alias;
        add_type_alias(aliases, weighted_alias, "FramedWeighted<" + desc.bsdfs[bsdf_index].bsdf_type + ">");
        return;
    }

    const std::size_t combiner_index = ref.combiner_index();
    const LayeringDesc::CombinerDesc& combiner = desc.combiners[combiner_index];
    const ClosureRef a_ref = combiner.children[0];
    const ClosureRef b_ref = combiner.children[1];
    build_composition_aliases_recursive(desc, a_ref, profile, aliases);
    build_composition_aliases_recursive(desc, b_ref, profile, aliases);

    const std::string core_alias = fmt::format("{}{}", combiner_alias_prefix(combiner.mode), combiner_index);
    const std::string weighted_alias = "Weighted" + core_alias;
    const std::string a = aliases.weighted_aliases.at(a_ref);
    const std::string b = aliases.weighted_aliases.at(b_ref);

    std::string core_type_name;
    switch (combiner.mode) {
    case CombinerMode::layer:
        core_type_name = fmt::format("{}<{}, {}>", profile.helper_type("LayerBSDF"), a, b);
        break;
    case CombinerMode::mix:
        core_type_name = fmt::format("{}<{}, {}>", profile.helper_type("MixBSDF"), a, b);
        break;
    case CombinerMode::add:
        core_type_name = fmt::format("{}<{}, {}>", profile.helper_type("AddBSDF"), a, b);
        break;
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }

    aliases.core_aliases[ref] = core_alias;
    aliases.weighted_aliases[ref] = weighted_alias;
    add_type_alias(aliases, core_alias, core_type_name);
    add_type_alias(aliases, weighted_alias, "Weighted<" + core_alias + ">");
}

std::string qualify_alias(const CompositionAliases& aliases, const std::string& name)
{
    return aliases.namespace_name + "::" + name;
}

} // namespace

std::string combiner_value_prefix(CombinerMode mode)
{
    switch (mode) {
    case CombinerMode::layer:
        return "layer";
    case CombinerMode::mix:
        return "mix";
    case CombinerMode::add:
        return "add";
    case CombinerMode::weighted_layer:
        FALCOR_UNREACHABLE();
    }
    FALCOR_UNREACHABLE();
}

CompositionAliases
build_composition_aliases(const LayeringDesc& desc, const ProfileDesc& profile, std::string namespace_name)
{
    CompositionAliases aliases;
    aliases.namespace_name = std::move(namespace_name);
    const ClosureRef root = desc.main_layer;
    if (root.is_none())
        add_type_alias(aliases, "RootBSDF", profile.bsdf_type("Null"));
    else {
        build_composition_aliases_recursive(desc, root, profile, aliases);
        add_type_alias(aliases, "RootBSDF", aliases.weighted_aliases.at(root));
    }
    return aliases;
}

CompositionAliases build_bsdf_mix_composition_aliases(
    const LayeringDesc& desc,
    std::string namespace_name,
    const std::string& root_type_name
)
{
    FALCOR_UNUSED(desc);
    CompositionAliases aliases;
    aliases.namespace_name = std::move(namespace_name);
    aliases.uses_weighted_helpers = false;
    add_type_alias(aliases, "RootBSDF", root_type_name);
    return aliases;
}

std::string alias_reference(const CompositionAliases& aliases, const std::string& name, bool use_local_aliases)
{
    return use_local_aliases ? "mtlx_ns::" + name : qualify_alias(aliases, name);
}

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
