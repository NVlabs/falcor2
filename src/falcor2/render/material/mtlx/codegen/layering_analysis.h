// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "profile_policy.h"

#include "falcor2/core/error.h"
#include "falcor2/core/macros.h"

#include <MaterialXGenShader/Shader.h>

#include <array>
#include <compare>
#include <cstddef>
#include <limits>
#include <optional>
#include <string>
#include <vector>

namespace falcor {
namespace mtlx {

class StaticInputQuery;

// Composition operation represented by a MaterialX closure combiner node. The
// value describes analysis output only; emitters decide how to materialize each
// mode in generated Slang.
enum class CombinerMode { layer, weighted_layer, mix, add };

class ClosureRef {
public:
    constexpr ClosureRef() noexcept = default;

    static constexpr ClosureRef none() noexcept { return ClosureRef(); }

    static ClosureRef bsdf(std::size_t index)
    {
        FALCOR_CHECK(
            index < static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Mtlx BSDF closure ref index is out of range."
        );
        return ClosureRef(static_cast<int>(index) + 1);
    }

    static ClosureRef combiner(std::size_t index)
    {
        FALCOR_CHECK(
            index < static_cast<std::size_t>(std::numeric_limits<int>::max()),
            "Mtlx combiner closure ref index is out of range."
        );
        return ClosureRef(-(static_cast<int>(index) + 1));
    }

    static constexpr ClosureRef from_raw_layering_index(int value) noexcept { return ClosureRef(value); }

    constexpr bool is_none() const noexcept { return m_value == 0; }
    constexpr bool is_bsdf() const noexcept { return m_value > 0; }
    constexpr bool is_combiner() const noexcept { return m_value < 0; }

    std::size_t bsdf_index() const
    {
        FALCOR_CHECK(is_bsdf(), "Mtlx closure ref is not a BSDF ref.");
        return static_cast<std::size_t>(m_value - 1);
    }

    std::size_t combiner_index() const
    {
        FALCOR_CHECK(is_combiner(), "Mtlx closure ref is not a combiner ref.");
        return static_cast<std::size_t>(-m_value - 1);
    }

    constexpr int raw_layering_index() const noexcept { return m_value; }

    auto operator<=>(const ClosureRef&) const = default;

private:
    explicit constexpr ClosureRef(int value) noexcept
        : m_value(value)
    {
    }

    int m_value = 0;
};

static_assert(sizeof(ClosureRef) == sizeof(int));

// Intermediate representation of the BSDF closure graph discovered in the
// MaterialX shader graph. ClosureRef values point at BSDF closures, combiner
// nodes, or no closure. The structure is owned by generated-code analysis and
// does not own the MaterialX shader nodes it points at.
struct LayeringDesc {
    struct BsdfDesc {
        int index = 0;
        const MaterialX::ShaderNode* shader_node = nullptr;
        std::string node_path;
        std::string node_impl;
        std::string scatter_mode;
        std::string bsdf_type;
        std::string lobe_types;
        bool transmissive = false;
        bool through_material_transmissive = false;
        bool preserve_tangent_frame = false;
        bool has_microfacet_selection = false;
        int fresnel_selection = 0;
        int scatter_selection = 0;
        ClosureRef copy_source_ref;
    };

    struct CombinerDesc {
        int index = 0;
        CombinerMode mode = CombinerMode::layer;
        std::array<ClosureRef, 2> children = {ClosureRef::none(), ClosureRef::none()};
        std::string node_path;
        std::string node_impl;
        ClosureRef copy_source_ref;
    };

    std::vector<BsdfDesc> bsdfs;
    std::vector<CombinerDesc> combiners;
    ClosureRef main_layer;
};

// Builds LayeringDesc by replaying the MaterialX closure emit order and tracking
// aliases through implementation subgraphs. The builder reads shader graph
// structure and values but does not mutate the graph.
class LayeringDescBuilder {
public:
    static LayeringDesc
    build(MaterialX::ShaderGraph& shader_graph, std::vector<std::string>* traversal_events = nullptr);
};

// Fills emitter-facing fields on LayeringDesc BSDFs and combiners after the
// graph shape has been built. This pass reads profile/options and optional
// shader constants to choose BSDF type names, lobe masks, transmissive flags,
// and Mtlx microfacet policy annotations; it does not mutate the graph.
void classify_layering(
    LayeringDesc& desc,
    const ProfileDesc& profile,
    const CodeGenDesc& codegen_desc,
    const MaterialX::VariableBlock* constants = nullptr,
    const StaticInputQuery* static_input_query = nullptr
);

// Validates the 1.38-style weighted_layer lowering, where top_weight is folded
// into the top input edge. Only that edge requires single-use protection; other
// shared composition refs remain legal.
void validate_weighted_layer_top_refs_are_not_shared(const LayeringDesc& desc);

// Converts weighted_layer analysis nodes to ordinary layer nodes after all
// weighted-layer-specific structural checks have run. The generated stack has
// already folded top_weight into the top input edge, so emitters should not
// carry a separate weighted-layer composition concept.
void lower_weighted_layers_to_layers_for_emission(LayeringDesc& desc);

// Some analysis-created transparent BSDFs carry filter_o transmission as a
// guaranteed unit pass-through. This hides their private implementation names
// from emitters.
bool is_unity_layer_pass_through_transmission_bsdf(const LayeringDesc::BsdfDesc& bsdf);

} // namespace mtlx
} // namespace falcor
