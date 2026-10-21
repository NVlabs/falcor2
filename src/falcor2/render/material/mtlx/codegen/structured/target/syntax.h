// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenSlang/SlangSyntax.h>

namespace falcor::mtlx::structured {

namespace mx = MaterialX;

/// MaterialX string literal rules for generated Slang source.
class GenslangPtStringTypeSyntax final : public mx::StringTypeSyntax {
public:
    explicit GenslangPtStringTypeSyntax(const mx::Syntax* parent);
    std::string getValue(const mx::Value& value, bool uniform) const override;
};

/// Falcor-specific Slang syntax overrides used by the structured target.
/// Type and identifier spelling stays here rather than leaking into graph or
/// layering analysis.
class GenslangPtSyntax final : public mx::SlangSyntax {
public:
    explicit GenslangPtSyntax(mx::TypeSystemPtr type_system);

    bool remapEnumeration(
        const std::string& value,
        mx::TypeDesc type,
        const std::string& enum_names,
        std::pair<mx::TypeDesc, mx::ValuePtr>& result
    ) const override;
};

} // namespace falcor::mtlx::structured
