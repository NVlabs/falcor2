// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "syntax.h"

#include <MaterialXCore/Types.h>
#include <MaterialXCore/Value.h>

#include <memory>
#include <unordered_map>

namespace falcor {
namespace materialx {
namespace mx139 {
namespace genslangpt {

GenslangPtStringTypeSyntax::GenslangPtStringTypeSyntax(const mx::Syntax* parent)
    : mx::StringTypeSyntax(parent, "int", "0", "0")
{
}

std::string GenslangPtStringTypeSyntax::getValue(const mx::Value& value, bool /*uniform*/) const
{
    static const std::unordered_map<std::string, std::string> k_space_values = {
        {"model", "mx_space_model"},
        {"object", "mx_space_object"},
        {"world", "mx_space_world"},
    };

    const std::string value_string = value.getValueString();
    auto it = k_space_values.find(value_string);
    if (it != k_space_values.end())
        return it->second;

    return "0";
}

GenslangPtSyntax::GenslangPtSyntax(mx::TypeSystemPtr type_system)
    : mx::SlangSyntax(type_system ? type_system : mx::TypeSystem::create())
{
    registerTypeSyntax(mx::Type::STRING, std::make_shared<GenslangPtStringTypeSyntax>(this));
    registerTypeSyntax(mx::Type::FILENAME, std::make_shared<mx::ScalarTypeSyntax>(this, "TextureHandle", "", ""));
    registerTypeSyntax(
        mx::Type::BSDF,
        std::make_shared<mx::AggregateTypeSyntax>(
            this,
            "BSDF",
            "{0}",
            "{0}",
            "",
            R"(struct BSDF
{
    uint index;
    void set_bsdf(int i) { index = i; }
    void set_layer(int i) { index = i | 0x80000000; }
    bool is_bsdf() { return (index & 0x80000000) == 0; }
    bool is_layer() { return (index & 0x80000000) != 0; }
    int get_index() { return index & ~0x80000000; }
})"
        )
    );
    registerTypeSyntax(
        mx::Type::EDF,
        std::make_shared<mx::AggregateTypeSyntax>(
            this,
            "EDF",
            "{}",
            "{0}",
            "",
            R"(struct EDF
{
    float3 radiance = {};
})"
        )
    );
    // TODO(@tdavidovic): This is a raster-compatible VDF emulation for MX139.
    // Replace it with full VDF support once MX139 can model volume scattering.
    registerTypeSyntax(
        mx::Type::VDF,
        std::make_shared<mx::AggregateTypeSyntax>(
            this,
            "VDF",
            "{}",
            "{0}",
            "",
            R"(struct VDF
{
    float3 throughput = float3(1.0);
})"
        )
    );
    registerTypeSyntax(
        mx::Type::SURFACESHADER,
        std::make_shared<mx::AggregateTypeSyntax>(
            this,
            "surfaceshader",
            "{}",
            "{0}",
            "",
            R"(struct surfaceshader
{
    BSDF root = {};
    EDF edf = {};
    float opacity = 1.0;
    bool thin_walled = false;
})"
        )
    );
    registerTypeSyntax(
        mx::Type::MATERIAL,
        std::make_shared<mx::AggregateTypeSyntax>(this, "material", "{}", "{0}", "", "#define material surfaceshader")
    );
}

bool GenslangPtSyntax::remapEnumeration(
    const std::string& value,
    mx::TypeDesc type,
    const std::string& enum_names,
    std::pair<mx::TypeDesc, mx::ValuePtr>& result
) const
{
    if (type == mx::Type::STRING && enum_names == "model,object,world")
        return false;

    if (mx::SlangSyntax::remapEnumeration(value, type, enum_names, result))
        return true;
    return false;
}

} // namespace genslangpt
} // namespace mx139
} // namespace materialx
} // namespace falcor
