// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Element.h>
#include <MaterialXCore/Node.h>

namespace falcor {
namespace mtlx {
namespace codegen_support {

// Returns false only when the root material/surface structure proves that no
// emission closure is connected. Emission values themselves remain dynamic.
bool materialx_node_may_emit(const MaterialX::NodePtr& node, int depth = 0);
bool materialx_element_may_emit(const MaterialX::ElementPtr& element);

} // namespace codegen_support
} // namespace mtlx
} // namespace falcor
