// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/ShaderNodeImpl.h>
#include <unordered_map>

namespace falcor::mtlx {
struct CodeGenDesc;
}

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

/// Emits reads from the requested MaterialX texture-coordinate set through the
/// Falcor surface-interaction geometry-property contract.
class TexCoordNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    bool isEditable(const mx::ShaderInput& input) const override;

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

/// Emits standard position and frame-vector reads in their requested space.
class IntrinsicGeomPropNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    bool isEditable(const mx::ShaderInput& input) const override;

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

/// Resolves named MaterialX geometry properties to intrinsic surface fields or
/// to Scene-owned geometry-attribute IDs.
class GeomPropValueNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    bool isEditable(const mx::ShaderInput& input) const override;

    struct ResolvedExpression {
        std::string expression;
        bool uses_geometry_properties = false;
    };

    // Resolve before cache lookup so attribute registration also happens on hits.
    // The returned expression may contain $MxGeomPropDefault, substituted during emission.
    const ResolvedExpression& resolve(
        const mx::Document& document,
        const CodeGenDesc& request,
        const mx::ShaderNode& node,
        const mx::ShaderStage& stage
    ) const;

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;

private:
    // Implementations are shared by nodes within one request, not across requests.
    mutable std::unordered_map<const mx::ShaderNode*, ResolvedExpression> m_expressions;
};

/// Rejects MaterialX's per-geometry uniform property storage, which Falcor does
/// not currently provide.
class GeomPropValueUniformNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

/// Emits indexed geometry-color reads through the production accessor.
class GeomColorNode final : public mx::ShaderNodeImpl {
public:
    static mx::ShaderNodeImplPtr create();

    bool isEditable(const mx::ShaderInput& input) const override;

    void emitFunctionCall(const mx::ShaderNode& node, mx::GenContext& context, mx::ShaderStage& stage) const override;
};

} // namespace falcor::mtlx::structured
