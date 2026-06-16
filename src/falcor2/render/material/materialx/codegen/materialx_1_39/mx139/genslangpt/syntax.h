// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <MaterialXGenShader/Syntax.h>
#include <MaterialXGenShader/TypeDesc.h>
#include <MaterialXGenSlang/SlangSyntax.h>

#include <string>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

namespace mx = MaterialX;

class GenslangPtStringTypeSyntax : public mx::StringTypeSyntax {
public:
    explicit GenslangPtStringTypeSyntax(const mx::Syntax* parent);
    std::string getValue(const mx::Value& value, bool uniform) const override;
};

class GenslangPtSyntax : public mx::SlangSyntax {
public:
    explicit GenslangPtSyntax(mx::TypeSystemPtr type_system);

    bool remapEnumeration(
        const std::string& value,
        mx::TypeDesc type,
        const std::string& enum_names,
        std::pair<mx::TypeDesc, mx::ValuePtr>& result
    ) const override;
};

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
