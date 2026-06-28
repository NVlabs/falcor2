// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../../codegen_types.h"

#include <MaterialXCore/Document.h>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace root_prepare {

using materialx::CodeGenDesc;

// Result of selecting the MaterialX element that will be handed to the Slang
// generator. The selected root may be an original document element, an output
// inside a material implementation graph, or a preview material synthesized by
// prepare_root().
struct RootPrepareResult {
    MaterialX::ElementPtr root_element;
};

// Finds the requested or inferred renderable root, resolves material nodes with
// implementation graphs to their implementation output, and wraps previewable
// value outputs in a diffuse preview material.
//
// Graph mutation: only value-output preview wrapping mutates the document. It
// adds a small Oren-Nayar material graph so non-material outputs can be emitted
// by the same material codegen path as renderable materials.
RootPrepareResult prepare_root(MaterialX::DocumentPtr doc, const CodeGenDesc& desc);

} // namespace root_prepare
} // namespace mx139
} // namespace materialx
} // namespace falcor
