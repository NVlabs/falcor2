// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "composition_aliases.h"

#include "../../../layering_analysis.h"
#include "../../../profile_policy.h"

#include <MaterialXGenShader/ShaderGenerator.h>
#include <MaterialXGenShader/ShaderStage.h>

#include <memory>
#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace emitter {

// Strategy interface for the basic family's root-BSDF construction. The basic
// material wrapper (`emit_closure_tree_material_structs`) drives all basic-family
// forks via this interface so the wrapper itself does not include any
// sibling-strategy headers. Adding a new basic sibling is a single dispatch
// entry in `make_root_strategy`.
struct RootStrategy {
    virtual ~RootStrategy() = default;

    // Type declarations emitted into the stage before the wrapper body. Called
    // unconditionally; strategies with no extra types are a no-op (and must
    // also skip the trailing line break to preserve byte-identical output).
    virtual void emit_root_types(
        const MaterialX::ShaderGenerator& shadergen,
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& root_type_name,
        MaterialX::ShaderStage& stage
    ) const
        = 0;

    // Composition aliases used by the wrapper's aliases-namespace emission and
    // the per-lobe value emission.
    virtual CompositionAliases build_aliases(
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& aliases_name,
        const std::string& root_type_name
    ) const
        = 0;

    // Parameterisation for `emit_synthetic_opacity_stack_setup`. The router
    // owns the stack name; the strategy decides the flags.
    struct StackParams {
        bool use_stack_frames = false;
    };
    virtual StackParams stack_params() const = 0;

    // Emit per-lobe setup lines into the stage and return the identifier the
    // wrapper splices as the root value. This is not a pure expression
    // factory: implementations write into `stage`.
    virtual std::string emit_root_value(
        const MaterialX::ShaderGenerator& shadergen,
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& root_type_name,
        const std::string& stack_name,
        const std::string& wi_expr,
        const CompositionAliases& aliases,
        MaterialX::ShaderStage& stage
    ) const
        = 0;

    // Side-effect-free variant of `emit_root_value`: returns the same lines as
    // a single multi-line string (each line prefixed by `line_prefix` and
    // terminated by '\n') alongside the root-value expression. Used by the
    // wrapper template renderer in Step 7 to splice the cascade into a
    // $-token slot without writing through the MaterialX shader stage.
    struct RootValueBlock {
        std::string lines;
        std::string root_expr;
    };
    virtual RootValueBlock build_root_value_text(
        const LayeringDesc& layering,
        const ProfileDesc& profile,
        const std::string& root_type_name,
        const std::string& stack_name,
        const std::string& wi_expr,
        const CompositionAliases& aliases,
        const std::string& line_prefix
    ) const
        = 0;

    // Returns the material-instance collect_extra_bsdf_properties override
    // for the selected basic-family root representation.
    virtual std::string build_collect_extra_text(const LayeringDesc& layering, const std::string& line_prefix) const
        = 0;
};

// Factory: the single dispatch point that selects a basic-family strategy.
// This is the only place that knows the full set of basic siblings; the
// material wrapper depends on the interface only.
std::unique_ptr<RootStrategy> make_root_strategy(const LayeringDesc& layering, const CodeGenDesc& desc);

} // namespace emitter
} // namespace mx139
} // namespace materialx
} // namespace falcor
