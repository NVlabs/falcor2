// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../codegen.h"

#include <MaterialXCore/Node.h>
#include <MaterialXGenShader/ShaderNode.h>
#include <MaterialXGenShader/ShaderNodeImpl.h>

#include <optional>
#include <string>
#include <string_view>
#include <cstdint>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

namespace mx = MaterialX;

// Name of the `geomprop` shader input on the MaterialX `geompropvalue` node.
extern const char* const k_geomprop_input_name;

// Returns the type name of the first shader output, or "unknown" if absent.
std::string shader_output_type_name(const mx::ShaderNode& node);

// Slang identifier for the runtime geomprop ID constant for `stream_name`.
std::string geomprop_id_const_name(const std::string& stream_name);

// True if the requested `geomprop` is a MaterialX builtin (UV0, Pworld, ...)
// for the given `output_type`.
bool is_builtin_geomprop(const std::string& geomprop, const std::string& output_type);

// Resolves the runtime geomprop ID for `geomprop` of the given `output_type`,
// consulting the desc-provided callback first and then the test ID table.
std::optional<uint32_t>
resolve_geomprop_id(const CodeGenDesc& desc, std::string_view geomprop, std::string_view output_type);

// MaterialX `geompropvalue` pulls arbitrary named geometry data. Falcor supplies
// builtins directly and resolves non-builtin streams through the runtime
// GeomPropProvider IDs baked into the generated module.
class GeomPropValueNode : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

// MaterialX `texcoord` defaults to UV0 but may address additional UV sets by
// index. This node maps that convention to Falcor's primary UV and optional
// GeomPropProvider-backed texcoord streams.
class TexCoordNode : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

// MaterialX `geomcolor` reads vertex color sets. Falcor routes those through
// the same runtime geometry-property provider used for other named streams.
class GeomColorNode : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
