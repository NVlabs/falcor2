// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/enum.h"

#include <MaterialXCore/Document.h>

#include <cstdint>
#include <functional>
#include <span>
#include <unordered_map>

namespace falcor::mtlx::structured {

/// Receives the node and its local input name, not a full path or generated parameter name.
using EditableInputPredicate = std::function<bool(const MaterialX::Node&, const std::string&)>;
/// Maps replacement input paths to authored input paths removed by document rewrites.
using MovedInputPaths = std::unordered_map<std::string, std::string>;

/// Named transformations allowed to mutate the MaterialX document before
/// ShaderGraph construction. The explicit flags keep backend requests visible.
enum class DocumentTransformPass : uint32_t {
    none = 0,
    materialize_default_geomprops = 1u << 0,
    all = materialize_default_geomprops,
};
FALCOR_ENUM_CLASS_OPERATORS(DocumentTransformPass);
SGL_ENUM_FLAGS_INFO(
    DocumentTransformPass,
    {
        {DocumentTransformPass::none, "none"},
        {DocumentTransformPass::materialize_default_geomprops, "materialize_default_geomprops"},
        {DocumentTransformPass::all, "all"},
    }
);
SGL_ENUM_REGISTER(DocumentTransformPass);

/// Execute requested transformations in their fixed order.
void apply_document_transforms(
    DocumentTransformPass requested,
    MaterialX::Document& document,
    const MaterialX::TypedElementPtr& root,
    std::span<const MaterialX::OutputPtr> additional_outputs,
    const EditableInputPredicate& is_editable,
    MovedInputPaths& moved_inputs
);

} // namespace falcor::mtlx::structured
