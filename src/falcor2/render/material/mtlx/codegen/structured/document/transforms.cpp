// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "transforms.h"
#include "passes/default_geomprops.h"

#include "falcor2/core/error.h"

#include <MaterialXCore/Document.h>

namespace falcor::mtlx::structured {

void apply_document_transforms(
    DocumentTransformPass requested,
    MaterialX::Document& document,
    const MaterialX::TypedElementPtr& root,
    std::span<const MaterialX::OutputPtr> additional_outputs,
    [[maybe_unused]] const EditableInputPredicate& is_editable,
    [[maybe_unused]] MovedInputPaths& moved_inputs
)
{
    const DocumentTransformPass unknown_passes = requested & ~DocumentTransformPass::all;
    FALCOR_CHECK(
        unknown_passes == DocumentTransformPass::none,
        "Structured MaterialX requested unknown document-transform bits 0x{:x}.",
        static_cast<uint32_t>(unknown_passes)
    );

    if (is_set(requested, DocumentTransformPass::materialize_default_geomprops))
        document_passes::materialize_default_geomprops(document, root, additional_outputs);

    std::string validation_message;
    FALCOR_CHECK(
        document.validate(&validation_message),
        "Structured MaterialX document-transform validation failed:\n{}",
        validation_message
    );
}

} // namespace falcor::mtlx::structured
