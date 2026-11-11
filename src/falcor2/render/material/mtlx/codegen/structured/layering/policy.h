// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"
#include "walker.h"
#include "../pipeline/generation_types.h"

#include <memory>

namespace falcor::mtlx::structured {

/// Backend boundary for one concrete surface or fiber representation.
///
/// After document preparation and graph classification, the generator selects
/// one instance and calls these operations in order for a non-empty root. The policy may analyze the original layering
/// tree before common weighted-layer lowering, then consumes that lowered tree to build and own its complete
/// representation. Storage and emission queries are valid only after that build. Policy-specific representation state
/// stays private.
class ILayeringPolicy {
public:
    virtual ~ILayeringPolicy() = default;

    /// Inspect or annotate facts that would be lost when the common lowering
    /// replaces weighted layers. Backends that need no original-tree analysis
    /// implement a documented no-op.
    virtual void analyze_original_layering(ClassifiedLayering& layering) = 0;

    /// Validate and consume the lowered tree, leaving a complete backend-owned
    /// representation ready for storage selection and emission.
    virtual void build_representation(ClassifiedLayering&& layering, const CodeGenDesc& desc) = 0;

    /// Describe every common graph-storage decision for the completed
    /// representation, including whether stack frames are required.
    virtual GraphStoragePlan graph_storage_plan(const GeneratedNames& names) const = 0;

    /// Emit representation declarations that must precede the MaterialX graph.
    virtual void emit_representation_declarations(EmissionContext& context) const = 0;

    /// Emit the material implementation and its ancillary methods after the
    /// MaterialX graph.
    virtual void emit_material_implementation(EmissionContext& context) const = 0;
};

std::unique_ptr<ILayeringPolicy> create_layering_policy(const CodeGenDesc& desc);

} // namespace falcor::mtlx::structured
