// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include "../pipeline/generation_types.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/GenUserData.h>
#include <MaterialXCore/Document.h>

#include <string>
#include <vector>

namespace falcor::mtlx::structured {

inline constexpr const char* USER_DATA_NAME = "mtlx_structured";

/// Attributes Falcor must attach to a MaterialX-owned compound declaration.
/// MaterialX emits nested compound definitions before the current declaration,
/// so these facts are recorded during emission and applied after the graph is complete.
struct CompoundFunctionAnnotation {
    std::string function_name;
    bool mutating = false;
    bool force_inline = false;
};

// Mutable per-module state for emission bookkeeping and dependencies of emitted operations.
struct EmissionScratch {
    bool bsdf_preparation_helper_emitted = false;
    bool uses_geometry_properties = false;
    std::vector<CompoundFunctionAnnotation> compound_annotations;
};

class EmissionUserDataScope;

/// Data borrowed by structured MaterialX node implementations while the graph
/// is emitted. MaterialX callbacks expose only GenContext, so the outer graph
/// emission installs this bridge for the complete callback interval.
class EmissionUserData final : public MaterialX::GenUserData {
public:
    const CodeGenDesc& request() const { return *m_request; }
    const MaterialX::Document& document() const { return *m_document; }
    const NodeEmissionFacts& node_emission_facts() const { return *m_node_emission_facts; }
    EmissionScratch& emission_scratch() const { return *m_emission_scratch; }
    /// Whether scalarized complex-mix graph boundaries must disappear before
    /// CUDA lowering to avoid materializing graph state in local memory.
    bool force_inline_graph_boundaries() const { return m_force_inline_graph_boundaries; }

private:
    friend class EmissionUserDataScope;
    EmissionUserData(
        const CodeGenDesc& request,
        const MaterialX::Document& document,
        const NodeEmissionFacts& node_emission_facts,
        EmissionScratch& emission_scratch,
        bool force_inline_graph_boundaries
    );

    const CodeGenDesc* m_request;
    const MaterialX::Document* m_document;
    const NodeEmissionFacts* m_node_emission_facts;
    EmissionScratch* m_emission_scratch;
    bool m_force_inline_graph_boundaries;
};

/// Installs one complete callback bridge on MaterialX's user-data stack and
/// restores the previous same-key entry on normal or exceptional exit.
class EmissionUserDataScope {
public:
    EmissionUserDataScope(
        MaterialX::GenContext& context,
        const CodeGenDesc& request,
        const MaterialX::Document& document,
        const NodeEmissionFacts& node_emission_facts,
        EmissionScratch& emission_scratch,
        bool force_inline_graph_boundaries
    );
    EmissionUserDataScope(const EmissionUserDataScope&) = delete;
    EmissionUserDataScope& operator=(const EmissionUserDataScope&) = delete;
    EmissionUserDataScope(EmissionUserDataScope&&) = delete;
    EmissionUserDataScope& operator=(EmissionUserDataScope&&) = delete;
    ~EmissionUserDataScope();

private:
    MaterialX::GenContext* m_context;
};

} // namespace falcor::mtlx::structured
