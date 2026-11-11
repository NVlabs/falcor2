// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Element.h>
#include <MaterialXCore/Node.h>

#include <unordered_set>

namespace falcor::mtlx::structured {

// Walks upstream from a MaterialX Core root element and returns the reachable
// Core nodes, including all outputs of implementation graphs in the same document.
// Attached-library implementations are excluded. This is document-preparation
// reachability, not ShaderGraph call-site reachability.
std::unordered_set<MaterialX::NodePtr> collect_reachable_core_nodes(const MaterialX::ElementPtr& root_element);

} // namespace falcor::mtlx::structured
