// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/render/material/mtlx/codegen/codegen_types.h"

#include <MaterialXCore/Document.h>

#include <string>

namespace falcor {
namespace mtlx {
namespace graph_prepare {

void prepare_graph(
    MaterialX::DocumentPtr doc,
    MaterialX::ElementPtr root_element,
    const CodeGenDesc& desc,
    const std::string& target
);

} // namespace graph_prepare
} // namespace mtlx
} // namespace falcor
