// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "policy.h"
#include "policy_factories.h"

#include "falcor2/core/error.h"

namespace falcor::mtlx::structured {

std::unique_ptr<ILayeringPolicy> create_layering_policy(const CodeGenDesc& desc)
{
    switch (desc.layering_mode) {
    case LayeringMode::closure_tree:
        return create_closure_tree_policy();
    case LayeringMode::bsdf_mix:
        return create_bsdf_mix_policy();
    }
    FALCOR_THROW("Structured MaterialX cannot create a policy for an unknown layering mode value.");
}

} // namespace falcor::mtlx::structured
