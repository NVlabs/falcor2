// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenSlang/SlangShaderGenerator.h>

#include <cstddef>
#include <string>
#include <unordered_set>
#include <utility>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

struct EmissionScratch;

/// MaterialX Slang shader generator for Falcor's `genslangpt` target.
///
/// It owns the `genslangpt` syntax and node-implementation registry and exposes
/// only the protected MaterialX services needed by the structured generator. It
/// intentionally rejects MaterialX's monolithic generate() route: document,
/// graph, analysis, and emission ordering belong to the structured pipeline.
class GenslangPtShaderGenerator final : public mx::SlangShaderGenerator {
public:
    static constexpr const char* TARGET = "genslangpt";

    /// Built-in graph identities are excluded from canonicalization so their
    /// established library implementation names remain unchanged. The pointers
    /// are non-owning; their source document must outlive implementation
    /// creation and module emission.
    explicit GenslangPtShaderGenerator(
        bool use_slang_derivatives,
        bool canonicalize_names,
        std::unordered_set<const mx::NodeGraph*> builtin_node_graphs
    );

    static std::shared_ptr<GenslangPtShaderGenerator> create(
        bool use_slang_derivatives = false,
        bool canonicalize_names = false,
        std::unordered_set<const mx::NodeGraph*> builtin_node_graphs = {}
    );

    const std::string& getTarget() const override;
    mx::ShaderPtr generate(const std::string& name, mx::ElementPtr element, mx::GenContext& context) const override;

    // Narrow access to protected MaterialX services. The structured generator
    // owns their ordering; this generator only delegates to MaterialX.
    mx::ShaderStagePtr create_stage(const std::string& name, mx::Shader& shader) const;
    void create_variables(mx::ShaderGraphPtr graph, mx::GenContext& context, mx::Shader& shader) const;
    void convert_glsl_to_slang(mx::ShaderStage& stage) const;
    void apply_token_substitutions(mx::ShaderStage& stage) const;
    void apply_compound_annotations(mx::ShaderStage& stage, const EmissionScratch& scratch) const;
    mx::ConstSyntaxPtr syntax_ptr() const { return _syntax; }

    /// Return whether bindings inside a graph-backed implementation belong to
    /// a user graph and therefore participate in canonical naming. Unknown
    /// graph implementations are canonicalized conservatively.
    bool should_canonicalize_implementation_bindings(const mx::ShaderNodeImpl& implementation) const;

    std::string_view implementation_name(const mx::ShaderNodeImpl& implementation) const;

protected:
    mx::ShaderNodeImplPtr createShaderNodeImplForNodeGraph(const mx::NodeGraph& node_graph) const override;
    mx::ShaderNodeImplPtr
    createShaderNodeImplForImplementation(const mx::Implementation& implementation) const override;

private:
    bool m_canonicalize_names = false;
    // Non-owning graph identities; see the constructor contract.
    std::unordered_set<const mx::NodeGraph*> m_builtin_node_graphs;
    mutable std::size_t m_canonical_compound_count = 0;
};

} // namespace falcor::mtlx::structured
