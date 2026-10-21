// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

namespace falcor::mtlx::structured {

/// Reject request combinations that the structured backend cannot represent.
/// This validates cross-option invariants only; support for individual
/// production features is established by their implementation and tests rather
/// than a separately maintained capability table.
void validate_request(const CodeGenDesc& desc);

} // namespace falcor::mtlx::structured
