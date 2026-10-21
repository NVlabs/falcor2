// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include "walker.h"
#include "../pipeline/generation_types.h"

namespace falcor::mtlx::structured {

// Dedicated, removable owner for a valid material with no surface BSDF root.
// Emission, opacity, thin-walled, and volume behavior remain in the common
// ancillary plan and are not disguised as a layering policy.
class EmptySurfaceHandler {
public:
    /// Validate and consume an empty classified surface, leaving the fallback
    /// representation ready for storage selection and emission.
    void build_representation(ClassifiedLayering&& layering);

    GraphStoragePlan graph_storage_plan() const;

    /// Emit only the surface fallback owned by this handler; ancillary material
    /// behavior remains in the common emission plan.
    void emit_material_implementation(EmissionContext& context) const;

private:
    bool m_built = false;
};

} // namespace falcor::mtlx::structured
