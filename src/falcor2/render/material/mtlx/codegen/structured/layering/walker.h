// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include "../material/bsdf_support.h"
#include "../material/profile_policy.h"

#include "falcor2/core/error.h"

#include <MaterialXGenShader/Shader.h>

#include <array>
#include <compare>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace falcor::mtlx::structured {

enum class SurfaceRootKind {
    empty,
    non_empty,
};

enum class CombinerMode {
    layer,
    weighted_layer,
    mix,
    add,
};

struct RootContextualWeight {
    enum class Kind {
        identity,
        constant,
        dynamic,
    };

    Kind kind = Kind::identity;
    std::array<float, 3> constant = {1.0f, 1.0f, 1.0f};
};

/// Compact, type-safe reference to either a BSDF leaf or a combiner in
/// ClassifiedLayering. It does not own the referenced entry.
class ClosureRef {
public:
    constexpr ClosureRef() noexcept = default;

    static constexpr ClosureRef none() noexcept { return {}; }

    static ClosureRef bsdf(std::size_t index)
    {
        FALCOR_CHECK(
            index < static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Structured MaterialX BSDF closure index is out of range."
        );
        return ClosureRef(static_cast<int>(index) + 1);
    }

    static ClosureRef combiner(std::size_t index)
    {
        FALCOR_CHECK(
            index < static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Structured MaterialX combiner closure index is out of range."
        );
        return ClosureRef(-(static_cast<int>(index) + 1));
    }

    constexpr bool is_none() const noexcept { return m_value == 0; }
    constexpr bool is_bsdf() const noexcept { return m_value > 0; }
    constexpr bool is_combiner() const noexcept { return m_value < 0; }

    std::size_t bsdf_index() const
    {
        FALCOR_CHECK(is_bsdf(), "Structured MaterialX closure ref is not a BSDF.");
        return static_cast<std::size_t>(m_value - 1);
    }

    std::size_t combiner_index() const
    {
        FALCOR_CHECK(is_combiner(), "Structured MaterialX closure ref is not a combiner.");
        return static_cast<std::size_t>(-m_value - 1);
    }

    constexpr int raw_index() const noexcept { return m_value; }
    auto operator<=>(const ClosureRef&) const = default;

private:
    explicit constexpr ClosureRef(int value) noexcept
        : m_value(value)
    {
    }

    int m_value = 0;
};

/// Backend-independent closure topology produced by one deterministic walk of
/// the frozen MaterialX ShaderGraph. Policies may transform their owned copy in
/// representation building, but target node emission never uses this as
/// general graph storage.
struct ClassifiedLayering {
    /// One concrete BSDF leaf and the specialization facts needed by the
    /// selected layering backend and target node emitter.
    struct BsdfDesc {
        const MaterialX::ShaderNode* shader_node = nullptr;
        std::vector<std::string> invocation_chain;
        // Complex-mix frame equivalence spells the invocation identity as a
        // colon-terminated path. Retain that spelling alongside the typed
        // chain because generated frame slots use it as their stable key.
        std::string nodegraph_instance_path;
        std::string node_path;
        std::string node_impl;
        std::string scatter_mode;
        BsdfKind kind = BsdfKind::unknown;
        std::string bsdf_type;
        BsdfLobeFlags lobe_flags = BsdfLobeFlags::none;
        std::optional<MxMicrofacetSelection> microfacet_selection;
        std::optional<MxSheenSelection> sheen_selection;
        bool transmissive = false;
        bool through_material_transmissive = false;
        bool preserve_tangent_frame = false;
    };

    /// One normalized binary closure-composition operation over typed refs.
    struct CombinerDesc {
        CombinerMode mode = CombinerMode::layer;
        std::array<ClosureRef, 2> children = {ClosureRef::none(), ClosureRef::none()};
        std::string node_path;
        std::string node_impl;
    };

    SurfaceRootKind root_kind = SurfaceRootKind::empty;
    ClosureRef main_layer;
    std::vector<BsdfDesc> bsdfs;
    std::vector<CombinerDesc> combiners;
    RootContextualWeight root_contextual_weight;
    bool weighted_layers_lowered = false;
};

/// Builds ClassifiedLayering from the surface root while recording stable
/// traversal order and material-specific BSDF specialization decisions.
///
/// The walk replays MaterialX closure emission order without mutating the
/// graph. A complete invocation chain is part of every nested node identity,
/// so reused implementation graphs remain distinct even when they contain the
/// same node.
class LayeringWalker {
public:
    static ClassifiedLayering build_and_classify(
        MaterialX::ShaderGraph& graph,
        const CodeGenDesc& desc,
        const MaterialX::VariableBlock* constants,
        const StaticInputQuery* static_input_query,
        std::vector<std::string>& traversal_events
    );
};

/// Return the union of typed lobe properties reachable from a closure root.
BsdfLobeFlags reachable_lobe_flags(const ClassifiedLayering& layering, ClosureRef root);

/// Format a typed lobe mask as the canonical Slang BSDFFlags expression used
/// by the existing generated source. An empty mask produces an empty string so
/// the emission site can retain its context-specific `none` or `all` fallback.
std::string format_bsdf_lobe_flags(BsdfLobeFlags flags);

/// Reject weighted layers whose top reference is shared, because lowering
/// would otherwise change the semantics of another consumer.
void validate_weighted_layer_top_refs_are_not_shared(const ClassifiedLayering& layering);
/// Lower every supported weighted-layer combiner exactly once after any
/// backend-specific validation of the original authored topology.
void lower_weighted_layers_once(ClassifiedLayering& layering);

} // namespace falcor::mtlx::structured
