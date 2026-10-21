// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include "../graph/closure_pruning/closure_pruning_counters.h"
#include "../graph/static_input_query/static_value_analysis.h"
#include "../material/profile_policy.h"

#include <MaterialXGenShader/GenContext.h>
#include <MaterialXGenShader/Shader.h>

#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace falcor::mtlx::structured {

class GenslangPtShaderGenerator;

/// Results retained from graph optimization for reporting and later emission.
struct GraphOptimizationFacts {
    ClosurePruningCounters closure_pruning;
};

/// Backend-independent facts about graph topology and required geometry input.
struct GraphFacts {
    bool has_surface_root = false;
};

/// Observed non-BSDF behavior shared by all layering representations.
struct AncillaryMaterialFacts {
    bool has_emission = false;
    std::string emission_predicate = "true";
    std::string emission_query_source;
};

/// Concrete closure-node facts exposed to target implementations during
/// MaterialX graph emission, after layering classification has fixed BSDF types.
struct NodeEmissionFacts {
    // Common graph storage is derived before ClassifiedLayering moves into its
    // representation owner. This is not a second closure IR: policies retain
    // composition, while common emission needs only scalar stack shape.
    std::vector<std::string> bsdf_types;
    std::size_t combiner_count = 0;
    std::unordered_map<const MaterialX::ShaderNode*, MxMicrofacetSelection> microfacet_selections;
    std::unordered_map<const MaterialX::ShaderNode*, MxSheenSelection> sheen_selections;
};

using PreCanonicalInputNames = std::unordered_map<const MaterialX::ShaderPort*, std::string>;

/// Facts derived from the optimized graph and retained through planning and
/// emission. The aggregate is populated visibly in causal order, validated
/// against ClassifiedLayering, then exposed read-only.
struct GraphAnalysis {
    GraphOptimizationFacts optimization;
    std::optional<StaticValueAnalysisResult> static_values;
    PreCanonicalInputNames pre_canonical_input_names;
    GraphFacts graph_facts;
    AncillaryMaterialFacts ancillary_facts;
    std::vector<std::string> layering_traversal_order;
    NodeEmissionFacts node_emission_facts;
};

/// Complete set of hash-substituted public names generated for one module.
struct GeneratedNames {
    std::string module_name;
    std::string graph_name;
    std::string material_data_name;
    std::string material_instance_name;
    std::string material_name;
};

// Representation decisions consumed by the common module-emission skeleton.
// The common emitter retains ordering; policies choose only the storage shape
// at the points where generated Slang genuinely diverges.
enum class GraphStorageKind {
    common,
    common_with_frames,
};

struct GraphStoragePlan {
    GraphStorageKind kind = GraphStorageKind::common;
    std::string stack_type = "MxStackData";
};

/// Non-owning view passed to a completed layering representation during module
/// emission. Every member remains alive for the complete policy call.
struct EmissionContext {
    GenslangPtShaderGenerator& shader_generator;
    MaterialX::ShaderGraph& graph;
    MaterialX::GenContext& gen_context;
    MaterialX::ShaderStage& stage;
    const CodeGenDesc& request;
    const GeneratedNames& names;
    const AncillaryMaterialFacts& ancillary;
    CodeGenResult& result;
};

} // namespace falcor::mtlx::structured
