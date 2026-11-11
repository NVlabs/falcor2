// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "shader_generator.h"

#include "nodes/geometry_nodes.h"
#include "nodes/height_to_normal_node.h"
#include "nodes/source_code_node.h"
#include "syntax.h"
#include "user_data.h"

#include "falcor2/core/error.h"
#include "nodes/microfacet_bsdf.h"
#include "nodes/sheen_bsdf.h"

#include <MaterialXGenShader/Nodes/CompoundNode.h>
#include <MaterialXGenShader/ShaderGraphRefactor.h>

#include <fmt/format.h>

#include <algorithm>
#include <functional>
#include <unordered_map>
#include <string_view>
#include <utility>

namespace falcor::mtlx::structured {

namespace {

void publish_input(mx::ShaderGraph& graph, mx::ShaderInput& input, const std::string& name)
{
    mx::ShaderOutput* socket = graph.addInputSocket(name, input.getType());
    socket->setPath(input.getPath());
    socket->setValue(input.getValue());
    socket->setUnit(input.getUnit());
    socket->setColorSpace(input.getColorSpace());
    socket->setMetadata(input.getMetadata());
    if (input.isUniform())
        socket->setUniform();
    socket->makeConnection(&input);
}

/// Preserve authored BSDF weight controls through the SDK's premultiplied-add
/// optimization. Bind literal weights before the SDK moves them: its
/// foldWeightIntoBsdf() preserves connections but copies only values for literals.
/// The adapter delegates the rewrite itself to that existing implementation.
/// https://github.com/AcademySoftwareFoundation/MaterialX/blob/7b64921ef1d42f2d57871e9d2c43dc11f041f26b/source/MaterialXGenShader/ShaderGraphRefactor.cpp#L42
class PreserveBSDFInputs final : public mx::PremultipliedBsdfAddRefactor {
public:
    size_t execute(mx::ShaderGraph& graph, mx::GenContext& context) override
    {
        // Do not add parameters to compound implementation signatures.
        if (context.getOptions().premultipliedBsdfAdd
            && context.getOptions().shaderInterfaceType == mx::SHADER_INTERFACE_COMPLETE) {
            for (mx::ShaderNode* node : graph.getNodes()) {
                if (!node->hasClassification(
                        mx::ShaderNode::Classification::BSDF | mx::ShaderNode::Classification::CLOSURE
                        | mx::ShaderNode::Classification::MIX
                    ))
                    continue;
                const mx::ShaderInput* mix = node->getInput("mix");
                if (!mix || !mix->getConnection())
                    continue;
                for (const char* side : {"fg", "bg"}) {
                    mx::ShaderInput* branch = node->getInput(side);
                    if (!branch || !branch->getConnection())
                        continue;
                    mx::ShaderNode* bsdf = branch->getConnection()->getNode();
                    mx::ShaderInput* weight = bsdf->getInput("weight");
                    if (!weight || weight->getConnection() || !weight->getValue() || !bsdf->isEditable(*weight))
                        continue;
                    std::string name = bsdf->getName() + "_" + weight->getName();
                    while (graph.getInputSocket(name))
                        name = mx::incrementName(name);
                    publish_input(graph, *weight, name);
                }
            }
        }
        return mx::PremultipliedBsdfAddRefactor::execute(graph, context);
    }
};

/// Prevent distinct authored inputs from sharing a public shader parameter when
/// their local node_input names collide (e.g. NGmain/v/value and NGaov/v/value).
/// Wrap the SDK refactors so collision handling runs after their rewrites and
/// dead-node cleanup, but before finalize() publishes inputs by name. Only
/// colliding inputs are published here, under unique names with their original
/// paths and values; the SDK publishes the remaining inputs normally.
///
/// Storm provides the naming precedent: _BuildEquivalentMaterialNetwork maps
/// full node paths to unique names before constructing the MaterialX document.
/// Falcor keeps authored node names and separates colliding sockets here instead.
/// https://github.com/PixarAnimationStudios/OpenUSD/blob/ff8c59e7e3d85439aa117e3c0aa5e24286e1267d/pxr/imaging/hdSt/materialXFilter.cpp#L1413
class DistinctPublicInputs final : public mx::ShaderGraphRefactor {
public:
    explicit DistinctPublicInputs(std::vector<mx::ShaderGraphRefactorPtr> refactors)
        : m_refactors(std::move(refactors))
    {
    }

    const std::string& getName() const override
    {
        static const std::string name = "distinctPublicInputs";
        return name;
    }

    size_t execute(mx::ShaderGraph& graph, mx::GenContext& context) override
    {
        size_t edits = 0;
        for (const auto& refactor : m_refactors)
            edits += refactor->execute(graph, context);
        if (context.getOptions().shaderInterfaceType != mx::SHADER_INTERFACE_COMPLETE)
            return edits;

        // Publish after the SDK passes and their cleanup, so discarded nodes
        // cannot introduce extra controls or name collisions. Returning zero
        // below tells finalize() that this cleanup has already been performed.
        if (edits)
            graph.removeUnusedNodes();

        // SDK publication keys sockets by local node_input names. Reserve every
        // pending name before separating collisions (e.g. two graphs with a
        // node named 'v'); leave ordinary inputs to the SDK's publication step.
        std::unordered_set<std::string> names;
        std::vector<mx::ShaderInput*> collisions;
        for (const mx::ShaderOutput* socket : graph.getInputSockets())
            names.emplace(socket->getName());
        for (mx::ShaderNode* node : graph.getNodes()) {
            for (mx::ShaderInput* input : node->getInputs()) {
                if (!input->getConnection() && !input->getType().isClosure() && node->isEditable(*input)) {
                    if (!names.emplace(node->getName() + "_" + input->getName()).second)
                        collisions.push_back(input);
                }
            }
        }
        for (mx::ShaderInput* input : collisions) {
            std::string name = input->getNode()->getName() + "_" + input->getName();
            do {
                name = mx::incrementName(name);
            } while (!names.emplace(name).second);
            publish_input(graph, *input, name);
        }
        return 0;
    }

private:
    std::vector<mx::ShaderGraphRefactorPtr> m_refactors;
};

class GenslangPtCompoundNode final : public mx::CompoundNode {
public:
    GenslangPtCompoundNode(std::string function_name, bool canonicalize_bindings)
        : m_function_name(std::move(function_name))
        , m_canonicalize_bindings(canonicalize_bindings)
    {
    }

    void initialize(const mx::InterfaceElement& element, mx::GenContext& context) override
    {
        mx::CompoundNode::initialize(element, context);
        if (!m_function_name.empty()) {
            _functionName = m_function_name;
            _hash = std::hash<std::string>{}(_functionName);
        }
    }

    void
    emitFunctionDefinition(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        mx::CompoundNode::emitFunctionDefinition(node, context, stage);

        // Authored compound interfaces need not put their closure output first.
        const bool mutating = has_closure_output(node);
        auto user_data = context.getUserData<EmissionUserData>(USER_DATA_NAME);
        FALCOR_CHECK(user_data, "Structured MaterialX compound node has no generation state.");
        const bool force_inline = user_data->force_inline_graph_boundaries();
        if (!mutating && !force_inline)
            return;

        // Closure-valued compounds reach the graph-storage mutators and must
        // say so at their MaterialX-owned declaration. Scalarized complex mix
        // additionally forces this boundary inline to avoid CUDA local
        // memory materialization of the scalar graph state. Record both facts
        // now; applying them after all nested definitions have been emitted
        // avoids copying the accumulated module once per compound.
        user_data->emission_scratch().compound_annotations.push_back(
            CompoundFunctionAnnotation{_functionName, mutating, force_inline}
        );
    }

    void addClassification(mx::ShaderNode& node) const override
    {
        // MaterialX classifies through output zero. Numeric siblings must neither
        // hide a closure nor mark its compound as a texture/constant node.
        uint32_t classification = 0;
        for (const mx::ShaderGraphOutputSocket* output : _rootGraph->getOutputSockets()) {
            if (output->getType().isClosure()) {
                if (const mx::ShaderOutput* connection = output->getConnection())
                    classification |= connection->getNode()->getClassification();
            }
        }
        if (classification)
            node.setClassification(classification);
        else
            mx::CompoundNode::addClassification(node);
    }

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override
    {
        // The base implementation only schedules closure inputs for closure-first
        // interfaces. Numeric-first compounds need the same dependency ordering.
        if (stage.getName() == mx::Stage::PIXEL && !nodeOutputIsClosure(node) && has_closure_output(node))
            context.getShaderGenerator()
                .emitDependentFunctionCalls(node, context, stage, mx::ShaderNode::Classification::CLOSURE);
        mx::CompoundNode::emitFunctionCall(node, context, stage);
    }

    bool canonicalize_bindings() const { return m_canonicalize_bindings; }
    const std::string& function_name() const { return _functionName; }

private:
    static bool has_closure_output(const mx::ShaderNode& node)
    {
        return std::any_of(
            node.getOutputs().begin(),
            node.getOutputs().end(),
            [](const mx::ShaderOutput* output)
            {
                return output->getType().isClosure();
            }
        );
    }

    std::string m_function_name;
    bool m_canonicalize_bindings = false;
};

} // namespace

GenslangPtShaderGenerator::GenslangPtShaderGenerator(
    bool use_slang_derivatives,
    bool canonicalize_names,
    std::unordered_set<const mx::NodeGraph*> builtin_node_graphs
)
    : mx::SlangShaderGenerator(mx::TypeSystem::create())
    , m_canonicalize_names(canonicalize_names)
    , m_builtin_node_graphs(std::move(builtin_node_graphs))
{
    _syntax = std::make_shared<GenslangPtSyntax>(_typeSystem);
    for (auto& refactor : _refactors) {
        if (dynamic_cast<mx::PremultipliedBsdfAddRefactor*>(refactor.get()))
            refactor = std::make_shared<PreserveBSDFInputs>();
    }
    auto refactors = std::move(_refactors);
    _refactors.clear();
    registerRefactor(std::make_shared<DistinctPublicInputs>(std::move(refactors)));
    registerImplementation("IM_texcoord_vector2_" + std::string(TARGET), TexCoordNode::create);
    registerImplementation("IM_texcoord_vector3_" + std::string(TARGET), TexCoordNode::create);
    if (!use_slang_derivatives)
        registerImplementation("IM_heighttonormal_vector3_" + std::string(TARGET), HeightToNormalNode::create);
    registerImplementation("IM_position_vector3_" + std::string(TARGET), IntrinsicGeomPropNode::create);
    registerImplementation("IM_normal_vector3_" + std::string(TARGET), IntrinsicGeomPropNode::create);
    registerImplementation("IM_tangent_vector3_" + std::string(TARGET), IntrinsicGeomPropNode::create);
    registerImplementation("IM_bitangent_vector3_" + std::string(TARGET), IntrinsicGeomPropNode::create);
    registerImplementation("IM_viewdirection_vector3_" + std::string(TARGET), IntrinsicGeomPropNode::create);
    registerImplementation("IM_geomcolor_float_" + std::string(TARGET), GeomColorNode::create);
    registerImplementation("IM_geomcolor_color3_" + std::string(TARGET), GeomColorNode::create);
    registerImplementation("IM_geomcolor_color4_" + std::string(TARGET), GeomColorNode::create);
    registerImplementation("IM_geompropvalue_integer_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_boolean_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_string_" + std::string(TARGET), GeomPropValueUniformNode::create);
    registerImplementation("IM_geompropvalue_float_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_color3_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_color4_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_vector2_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_vector3_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_geompropvalue_vector4_" + std::string(TARGET), GeomPropValueNode::create);
    registerImplementation("IM_dielectric_bsdf_" + std::string(TARGET), create_microfacet_dielectric_node);
    registerImplementation("IM_conductor_bsdf_" + std::string(TARGET), create_microfacet_conductor_node);
    registerImplementation(
        "IM_generalized_schlick_bsdf_" + std::string(TARGET),
        create_microfacet_generalized_schlick_node
    );
    registerImplementation("IM_sheen_bsdf_" + std::string(TARGET), create_sheen_bsdf_node);
}

std::shared_ptr<GenslangPtShaderGenerator> GenslangPtShaderGenerator::create(
    bool use_slang_derivatives,
    bool canonicalize_names,
    std::unordered_set<const mx::NodeGraph*> builtin_node_graphs
)
{
    return std::make_shared<GenslangPtShaderGenerator>(
        use_slang_derivatives,
        canonicalize_names,
        std::move(builtin_node_graphs)
    );
}

mx::ShaderNodeImplPtr GenslangPtShaderGenerator::createShaderNodeImplForNodeGraph(const mx::NodeGraph& node_graph) const
{
    const bool canonicalize_bindings = m_canonicalize_names && !m_builtin_node_graphs.contains(&node_graph);
    const std::string function_name
        = canonicalize_bindings ? fmt::format("mx_canonical_graph_{}", m_canonical_compound_count++) : std::string();

    return std::make_shared<GenslangPtCompoundNode>(function_name, canonicalize_bindings);
}

bool GenslangPtShaderGenerator::should_canonicalize_implementation_bindings(
    const mx::ShaderNodeImpl& implementation
) const
{
    const auto* compound = dynamic_cast<const GenslangPtCompoundNode*>(&implementation);
    return !compound || compound->canonicalize_bindings();
}

std::string_view GenslangPtShaderGenerator::implementation_name(const mx::ShaderNodeImpl& implementation) const
{
    const auto* compound = dynamic_cast<const GenslangPtCompoundNode*>(&implementation);
    return compound ? compound->function_name() : implementation.getName();
}

mx::ShaderNodeImplPtr GenslangPtShaderGenerator::createShaderNodeImplForImplementation(const mx::Implementation&) const
{
    return create_source_code_node();
}

const std::string& GenslangPtShaderGenerator::getTarget() const
{
    static const std::string TARGET_NAME = TARGET;
    return TARGET_NAME;
}

mx::ShaderPtr GenslangPtShaderGenerator::generate(const std::string&, mx::ElementPtr, mx::GenContext&) const
{
    FALCOR_THROW("Structured MaterialX generation must run through its explicit generator workflow.");
}

mx::ShaderStagePtr GenslangPtShaderGenerator::create_stage(const std::string& name, mx::Shader& shader) const
{
    return createStage(name, shader);
}

void GenslangPtShaderGenerator::create_variables(
    mx::ShaderGraphPtr graph,
    mx::GenContext& context,
    mx::Shader& shader
) const
{
    createVariables(std::move(graph), context, shader);
}

void GenslangPtShaderGenerator::convert_glsl_to_slang(mx::ShaderStage& stage) const
{
    SlangSyntaxFromGlsl(stage);
}

void GenslangPtShaderGenerator::apply_token_substitutions(mx::ShaderStage& stage) const
{
    replaceTokens(_tokenSubstitutions, stage);
}

void GenslangPtShaderGenerator::apply_compound_annotations(mx::ShaderStage& stage, const EmissionScratch& scratch) const
{
    if (scratch.compound_annotations.empty())
        return;

    std::unordered_map<std::string_view, std::size_t> annotation_indices;
    annotation_indices.reserve(scratch.compound_annotations.size());
    for (std::size_t i = 0; i < scratch.compound_annotations.size(); ++i) {
        const auto& annotation = scratch.compound_annotations[i];
        const auto [_, inserted] = annotation_indices.emplace(annotation.function_name, i);
        FALCOR_CHECK(
            inserted,
            "Structured MaterialX recorded compound function '{}' more than once.",
            annotation.function_name
        );
    }

    const std::string& source = stage.getSourceCode();
    std::string annotated_source;
    annotated_source.reserve(source.size() + scratch.compound_annotations.size() * 32);
    std::vector<std::size_t> match_counts(scratch.compound_annotations.size(), 0);

    // Scan each emitted line once. Only an exact MaterialX compound declaration
    // is eligible; similarly named calls, comments, and Falcor-owned functions
    // are copied unchanged.
    std::size_t line_start = 0;
    while (line_start < source.size()) {
        const std::size_t newline = source.find('\n', line_start);
        const std::size_t line_end = newline == std::string::npos ? source.size() : newline + 1;
        const std::string_view line(source.data() + line_start, line_end - line_start);
        const std::size_t declaration_start = line.find_first_not_of(" \t");

        const CompoundFunctionAnnotation* annotation = nullptr;
        if (declaration_start != std::string_view::npos && line.substr(declaration_start).starts_with("void ")) {
            const std::size_t name_start = declaration_start + 5;
            const std::size_t open_paren = line.find('(', name_start);
            if (open_paren != std::string_view::npos) {
                const auto found = annotation_indices.find(line.substr(name_start, open_paren - name_start));
                if (found != annotation_indices.end()) {
                    ++match_counts[found->second];
                    annotation = &scratch.compound_annotations[found->second];
                }
            }
        }

        if (annotation) {
            const std::string_view indentation = line.substr(0, declaration_start);
            if (annotation->force_inline) {
                annotated_source.append(indentation);
                annotated_source += "[ForceInline]\n";
            }
            annotated_source.append(indentation);
            if (annotation->mutating)
                annotated_source += "[mutating] ";
            annotated_source.append(line.substr(declaration_start));
        } else {
            annotated_source.append(line);
        }
        line_start = line_end;
    }

    for (std::size_t i = 0; i < scratch.compound_annotations.size(); ++i) {
        FALCOR_CHECK(
            match_counts[i] == 1,
            "Structured MaterialX expected exactly one declaration for compound function '{}', found {}.",
            scratch.compound_annotations[i].function_name,
            match_counts[i]
        );
    }
    stage.setSourceCode(annotated_source);
}

} // namespace falcor::mtlx::structured
