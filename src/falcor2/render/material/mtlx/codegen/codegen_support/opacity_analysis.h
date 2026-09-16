// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../graph_prepare/static_input_query/static_input_query.h"

#include <optional>

namespace MaterialX_v1_39_5 {
class ShaderGraph;
} // namespace MaterialX_v1_39_5
namespace MaterialX = MaterialX_v1_39_5;

namespace falcor {
namespace mtlx {
namespace codegen_support {

// Returns the exact root surface opacity after traversing expanded MaterialX
// implementation graphs, or {} when any required value is not statically known.
std::optional<float>
materialx_graph_static_opacity(const MaterialX::ShaderGraph& graph, const StaticInputQuery& static_input_query);

} // namespace codegen_support
} // namespace mtlx
} // namespace falcor
