// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "codegen_types.h"

#include <MaterialXCore/Document.h>

#include <memory>

namespace falcor {
namespace mtlx {

std::unique_ptr<CodeGenResult> generate_code(MaterialX::DocumentPtr doc, const CodeGenDesc& desc);

} // namespace mtlx
} // namespace falcor
