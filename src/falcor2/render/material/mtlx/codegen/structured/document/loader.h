// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

#include <MaterialXCore/Document.h>

#include <string>
#include <unordered_set>
#include <vector>

namespace falcor::mtlx::structured {

/// Phase-2 document ownership and the root selected from the request.
struct LoadedDocument {
    MaterialX::DocumentPtr document;
    MaterialX::TypedElementPtr root;
    /// Independent document output roots retained after name and type filtering.
    std::vector<MaterialX::OutputPtr> additional_outputs;
    /// Supported material AOV declarations with stable scene IDs, populated when a registry is supplied.
    std::vector<AOVDesc> material_aov_outputs;
    std::string shader_base_name;
    /// Node graphs from Falcor's built-in MaterialX libraries. Their
    /// established implementation names are preserved during canonicalization;
    /// only user-authored graph names and bindings are canonicalized. The
    /// document owns these non-owning identities throughout code generation.
    std::unordered_set<const MaterialX::NodeGraph*> builtin_node_graphs;
};

/// Load the authored document, import libraries, make connected ports
/// compatible, and validate the resulting MaterialX document.
MaterialX::DocumentPtr load_and_validate_document(const CodeGenDesc& desc);

/// Select one unambiguous authored primary root and additional outputs matching
/// the name patterns and supported types. An optional registry assigns scene IDs to runtime-supported AOVs.
/// The primary root retains ordinary fallback discovery and value-output previews.
LoadedDocument select_document_root(MaterialX::DocumentPtr document, const CodeGenDesc& desc);

/// Load the request document, import standard libraries, validate it, and
/// resolve the typed roots suitable for ShaderGraph construction.
LoadedDocument load_and_select_document(const CodeGenDesc& desc);

/// Discover MaterialX renderable elements through the same document
/// preparation used by production generation.
std::vector<RenderableElement> discover_renderable_elements(const CodeGenDesc& desc);

} // namespace falcor::mtlx::structured
