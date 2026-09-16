// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Document.h>

#include <cstdint>

namespace falcor {
namespace mtlx {
namespace graph_prepare {

uint32_t
materialize_reachable_default_geomprop_inputs(MaterialX::DocumentPtr doc, const MaterialX::ElementPtr& root_element);

} // namespace graph_prepare
} // namespace mtlx
} // namespace falcor
