// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXCore/Document.h>

#include <span>

namespace falcor::mtlx::structured::document_passes {

/// Materialize implicit default geomprops as explicit graph nodes so later
/// ShaderGraph transformations and analyses see the inputs MaterialX will use.
/// A null root prepares all nodes in a library document.
void materialize_default_geomprops(
    MaterialX::Document& document,
    const MaterialX::TypedElementPtr& root,
    std::span<const MaterialX::OutputPtr> additional_outputs
);

} // namespace falcor::mtlx::structured::document_passes
