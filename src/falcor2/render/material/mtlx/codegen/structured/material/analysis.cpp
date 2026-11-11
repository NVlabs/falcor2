// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "analysis.h"

#include <MaterialXCore/Node.h>
#include <MaterialXCore/Types.h>
#include <MaterialXGenShader/ShaderNode.h>

#include <string>
#include <unordered_set>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

namespace {

bool node_may_emit(const mx::NodePtr& node, int depth)
{
    if (!node)
        return false;
    if (depth > 16)
        return true;
    if (node->getType() == mx::MATERIAL_TYPE_STRING)
        return node_may_emit(node->getConnectedNode(mx::ShaderNode::SURFACESHADER), depth + 1);
    if (node->getType() == mx::EDF_TYPE_STRING)
        return true;
    if (node->getType() != mx::SURFACE_SHADER_TYPE_STRING)
        return false;
    mx::InputPtr foreground = node->getInput("fg");
    mx::InputPtr background = node->getInput("bg");
    if (foreground && background)
        return node_may_emit(foreground->getConnectedNode(), depth + 1)
            || node_may_emit(background->getConnectedNode(), depth + 1);
    if (mx::InputPtr edf = node->getInput("edf"))
        return node_may_emit(edf->getConnectedNode(), depth + 1);
    return node->getCategory() != "surface";
}

} // namespace

bool graph_has_emission(const mx::ShaderGraph& graph)
{
    for (const mx::ShaderGraphOutputSocket* output : graph.getOutputSockets()) {
        if (output->getType() == mx::Type::EDF)
            return true;
    }
    for (const mx::ShaderNode* node : graph.getNodes()) {
        if (node->getImplementation().getName() == "IM_surface_unlit_genslangpt")
            return true;
        for (const mx::ShaderOutput* output : node->getOutputs()) {
            if (output->getType() == mx::Type::EDF)
                return true;
        }
        if (const mx::ShaderGraph* implementation = node->getImplementation().getGraph()) {
            if (graph_has_emission(*implementation))
                return true;
        }
    }
    return false;
}

bool graph_uses_space_helpers(const mx::ShaderGraph& graph)
{
    static const std::unordered_set<std::string> IMPLEMENTATIONS = {
        "IM_transformpoint_vector3_genslangpt",
        "IM_transformvector_vector3_genslangpt",
        "IM_transformnormal_vector3_genslangpt",
    };
    for (const mx::ShaderNode* node : graph.getNodes()) {
        if (IMPLEMENTATIONS.contains(node->getImplementation().getName()))
            return true;
        if (const mx::ShaderGraph* implementation = node->getImplementation().getGraph()) {
            if (graph_uses_space_helpers(*implementation))
                return true;
        }
    }
    return false;
}

bool output_block_has_material_output(const mx::VariableBlock& outputs)
{
    if (outputs.empty())
        return false;
    const mx::TypeDesc type = outputs[0]->getType();
    return type == mx::Type::SURFACESHADER || type == mx::Type::MATERIAL;
}

bool element_may_emit(const mx::ElementPtr& element)
{
    if (!element)
        return false;
    if (mx::NodePtr node = element->asA<mx::Node>())
        return node_may_emit(node, 0);
    if (mx::OutputPtr output = element->asA<mx::Output>())
        return node_may_emit(output->getConnectedNode(), 0);
    return true;
}

} // namespace falcor::mtlx::structured
