// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "emission_analysis.h"

#include <MaterialXCore/Types.h>
#include <MaterialXGenShader/ShaderNode.h>

namespace falcor {
namespace mtlx {
namespace codegen_support {

namespace mx = MaterialX;

bool materialx_node_may_emit(const mx::NodePtr& node, int depth)
{
    if (!node)
        return false;
    if (depth > 16)
        return true;

    if (node->getType() == mx::MATERIAL_TYPE_STRING)
        return materialx_node_may_emit(node->getConnectedNode(mx::ShaderNode::SURFACESHADER), depth + 1);
    if (node->getType() == mx::EDF_TYPE_STRING)
        return true;
    if (node->getType() != mx::SURFACE_SHADER_TYPE_STRING)
        return false;

    mx::InputPtr fg = node->getInput("fg");
    mx::InputPtr bg = node->getInput("bg");
    if (fg && bg) {
        return materialx_node_may_emit(fg->getConnectedNode(), depth + 1)
            || materialx_node_may_emit(bg->getConnectedNode(), depth + 1);
    }

    if (mx::InputPtr edf = node->getInput("edf"))
        return materialx_node_may_emit(edf->getConnectedNode(), depth + 1);

    // Known surface models such as standard_surface and surface_unlit can
    // produce emission without exposing an EDF input. Keep them conservative.
    return node->getCategory() != "surface";
}

bool materialx_element_may_emit(const mx::ElementPtr& element)
{
    if (!element)
        return false;
    if (mx::NodePtr node = element->asA<mx::Node>())
        return materialx_node_may_emit(node);
    if (mx::OutputPtr output = element->asA<mx::Output>())
        return materialx_node_may_emit(output->getConnectedNode());
    return true;
}

} // namespace codegen_support
} // namespace mtlx
} // namespace falcor
