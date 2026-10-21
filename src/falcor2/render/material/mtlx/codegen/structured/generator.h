// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../codegen_types.h"

#include <memory>

namespace falcor::mtlx {

class CodeGenCache;

namespace structured {

// Canonical structured-generator entry point. Invalid request combinations are
// rejected before document loading.
std::unique_ptr<CodeGenResult> generate_internal(const CodeGenDesc& desc, CodeGenCache* cache);

} // namespace structured
} // namespace falcor::mtlx
