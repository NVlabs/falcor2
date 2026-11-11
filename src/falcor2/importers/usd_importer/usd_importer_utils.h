// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "usd_importer_macros.h"

#include "falcor2/core/properties.h"
#include "falcor2/core/types.h"
#include "falcor2/importers/importer_types.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/pxr.h>
#include <pxr/base/gf/half.h>
#include <pxr/base/gf/size2.h>
#include <pxr/base/gf/size3.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2h.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/tf/type.h>
#include <pxr/base/tf/token.h>
#include <pxr/base/vt/value.h>

#include <pxr/base/gf/vec2f.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec4h.h>
#include <pxr/base/gf/vec4i.h>

#include <pxr/usd/sdf/assetPath.h>
#include <pxr/usd/sdf/schema.h>
#include <pxr/usd/usd/attribute.h>

#include <pxr/usd/usdGeom/tokens.h>
#include <pxr/usd/usdGeom/basisCurves.h>
#include <pxr/usd/usdGeom/imageable.h>
#include <pxr/usd/usdGeom/primvarsAPI.h>
#include <pxr/usd/usdGeom/gprim.h>

#include <pxr/usd/usdShade/input.h>
#include <pxr/usd/usdShade/connectableAPI.h>
END_DISABLE_USD_WARNINGS

#include <algorithm>
#include <filesystem>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace falcor {
namespace usd_importer {

/// Return the authored varying value count per strand for valid USD BasisCurves topology at time.
/// Counts include pinned end spans and are independent of tessellation or the rendered curve representation.
FALCOR_API pxr::VtArray<int> compute_curve_varying_counts(const pxr::UsdGeomBasisCurves& curves, pxr::UsdTimeCode time);

/// Read indices at time, rejecting changes across authored samples when importing motion.
/// Repeated identical samples are allowed; Default time uses the earliest sample when checking motion.
bool read_fixed_indices(
    const pxr::UsdAttribute& attribute,
    pxr::VtArray<int>& indices,
    pxr::UsdTimeCode time,
    bool import_motion
);

/// Shared authored-time selection and metadata for fixed-topology geometry motion.
/// no_samples selects the representation's existing fallback when motion has no authored times.
struct GeometrySampleTimes {
    /// Nonempty USD evaluation schedule; motion-disabled imports use a single Default time.
    std::vector<pxr::UsdTimeCode> sample_times;
    /// Seconds corresponding to sample_times when there are multiple samples; empty for static geometry.
    std::vector<float> motion_times;
    AnimationInterpolationMode interpolation = AnimationInterpolationMode::linear;

    GeometrySampleTimes(
        const pxr::UsdPrim& prim,
        const std::vector<pxr::UsdAttribute>& attributes,
        bool import_motion,
        pxr::UsdTimeCode no_samples = pxr::UsdTimeCode::Default()
    );
};

/// Read half/float/double UV arrays, optionally flattening authored indices.
bool get_uv_values(
    const pxr::UsdGeomPrimvar& primvar,
    pxr::VtArray<pxr::GfVec2f>& values,
    pxr::UsdTimeCode time = pxr::UsdTimeCode::Default(),
    bool flatten = false
);

FALCOR_API bool set_from_value(Properties& props, std::string_view name, const pxr::VtValue& value);

/// Return true if path identifies an asset contained in an OpenUSD package.
FALCOR_API bool is_package_asset(const std::filesystem::path& path);

/// Read an asset contained in an OpenUSD package.
///
/// Returns no value when path is not package-relative. A package-relative path
/// that cannot be opened or read raises an exception.
FALCOR_API std::optional<std::vector<uint8_t>> read_package_asset(const std::filesystem::path& path);

inline bool set_from_value(Properties& props, std::string_view name, const pxr::UsdAttribute& attr)
{
    pxr::VtValue value;
    attr.Get(&value);
    return set_from_value(props, name, value);
}

inline float2 to_falcor(const pxr::GfVec2f& v)
{
    return float2(v[0], v[1]);
}

inline float3 to_falcor(const pxr::GfVec3f& v)
{
    return float3(v[0], v[1], v[2]);
}

inline float4 to_falcor(const pxr::GfVec4f& v)
{
    return float4(v[0], v[1], v[2], v[3]);
}

inline float4x4 to_falcor(const pxr::GfMatrix4d& m)
{
    float4x4 result;
    for (int r = 0; r < 4; ++r) {
        for (int c = 0; c < 4; ++c) {
            result[r][c] = (float)m[c][r];
        }
    }
    return result;
}

inline pxr::TfToken get_purpose(const pxr::UsdGeomImageable& prim)
{
    using namespace pxr;
    TfToken purpose = UsdGeomTokens->default_;
    UsdAttribute purposeAttr = prim.GetPurposeAttr();
    if (purposeAttr) {
        purposeAttr.Get(&purpose);
    }
    return purpose;
}

inline bool is_renderable(const pxr::UsdGeomImageable& geomImageable)
{
    using namespace pxr;
    TfToken purpose = get_purpose(geomImageable);
    if (purpose == UsdGeomTokens->guide || purpose == UsdGeomTokens->proxy) {
        return false;
    }
    UsdAttribute visibilityAttr(geomImageable.GetVisibilityAttr());
    if (visibilityAttr) {
        TfToken visibility;
        visibilityAttr.Get(&visibility);
        if (visibility == UsdGeomTokens->invisible) {
            return false;
        }
    }

    // Determine the inherited visibility.
    // USD documentation warns that ComputeVisiblity() can be inefficient, as it walks up
    // towards the root every it is called, rather than caching inherited visibility.
    // However, there has yet been no indication that this is a meaningful performance issue in practice.
    return geomImageable.ComputeVisibility() != UsdGeomTokens->invisible;
}

// Helper function that returns the main attribute if it is authored (non-default),
// otherwise it tries to retrieve the fallback attribute and should that fail, the default value.
// This is used to provide compatibility, e.g. when the default disk light radius changed from "radius" to
// "inputs:radius"
template<class T>
inline T
get_authored_attribute(const pxr::UsdAttribute& mainAttrib, const pxr::UsdAttribute& fallbackAttrib, const T& def)
{
    T val = def;
    if (mainAttrib && mainAttrib.IsAuthored()) {
        mainAttrib.Get(&val);
    } else if (fallbackAttrib) {
        fallbackAttrib.Get(&val);
    }
    return val;
}

template<typename T>
inline T get_attribute(const pxr::UsdAttribute& attr, const T& default_value)
{
    T result = default_value;
    if (attr.IsValid())
        attr.Get(&result);
    return result;
}

template<typename T>
inline T get_attribute(const pxr::UsdPrim& prim, const std::string& attr_name, const T& default_value)
{
    return get_attribute<T>(prim.GetAttribute(pxr::TfToken(attr_name)), default_value);
}

/// Overlay one authored import control. Missing or blocked values retain the initialized property.
template<typename T>
inline void
read_import_property(const pxr::UsdPrim& prim, const char* attr_name, Properties& properties, const char* key)
{
    const auto attr = prim.GetAttribute(pxr::TfToken(attr_name));
    if (!attr || !attr.IsAuthored())
        return;
    const auto expected_type = pxr::SdfSchema::GetInstance().FindType(pxr::TfType::Find<T>());
    FALCOR_CHECK(
        attr.GetTypeName() == expected_type,
        "Prim '{}' attribute '{}' must have type '{}'",
        prim.GetPath().GetString(),
        attr_name,
        expected_type.GetAsToken().GetString()
    );
    if (!attr.HasValue())
        return;
    T value;
    FALCOR_CHECK(attr.Get(&value), "Cannot read prim '{}' attribute '{}'", prim.GetPath().GetString(), attr_name);
    if constexpr (std::is_same_v<T, int>) {
        FALCOR_CHECK(
            value >= 0 && value <= 256,
            "Prim '{}' attribute '{}' must be between 0 and 256 (got {})",
            prim.GetPath().GetString(),
            attr_name,
            value
        );
    }
    properties.set(key, value);
}

// Return the ultimate source of the given input.
inline pxr::UsdShadeInput get_source_input(
    pxr::UsdShadeInput input,
    pxr::UsdShadeConnectableAPI& source,
    pxr::TfToken& sourceName,
    pxr::UsdShadeAttributeType& sourceType
)
{
    using namespace pxr;

    if (input && input.HasConnectedSource()) {
        if (input.GetConnectedSource(&source, &sourceName, &sourceType)) {
            auto inputs = source.GetInputs();
            if (!inputs.empty()) {
                // If there's a connected source of type asset, return it.
                for (uint32_t i = 0; i < inputs.size(); ++i) {
                    SdfValueTypeName type_name(inputs[i].GetTypeName());
                    if (type_name == SdfValueTypeNames->Asset) {
                        return inputs[i];
                    }
                }
                // Otherwise, if there's no input of type asset, use the first connected source.
                return inputs[0];
            }
        }
    }
    return input;
}

/// String-view equivalent of UsdGeomPrimvar::StripPrimvarsName: remove only "primvars:", preserving sub-namespaces.
inline std::string_view strip_primvars_name(std::string_view name)
{
    constexpr std::string_view prefix = "primvars:";
    return name.starts_with(prefix) ? name.substr(prefix.size()) : name;
}

/// Require a valued primvar with a two-component type accepted by Falcor's existing UV import path.
inline bool has_uv_primvar_value(const pxr::UsdGeomPrimvar& primvar)
{
    using namespace pxr;

    if (!primvar || !primvar.HasValue())
        return false;
    const TfToken type_name = primvar.GetTypeName().GetAsToken();
    return type_name == SdfValueTypeNames->Half2Array.GetAsToken()
        || type_name == SdfValueTypeNames->Float2Array.GetAsToken()
        || type_name == SdfValueTypeNames->Double2Array.GetAsToken()
        || type_name == SdfValueTypeNames->TexCoord2fArray.GetAsToken()
        || type_name == SdfValueTypeNames->TexCoord2dArray.GetAsToken()
        || type_name == SdfValueTypeNames->TexCoord2hArray.GetAsToken();
}

inline pxr::UsdGeomPrimvar
get_uv_primvar(const pxr::UsdGeomGprim& prim, std::span<const std::string> fallback_exclusions = {})
{
    using namespace pxr;

    UsdGeomPrimvarsAPI primvars_API(prim);

    // Texture coordinate primvar naming is in general based on an implicit
    // agreement between the renderer and the DCC app. For example, a texcoord-holding primvar
    // can be of type float2[] with an arbitrary name, or can be of of the texcoord-related types.
    // Attempt to match using both commonly-used texcoord primvar names, and texcoord types.

    auto is_excluded = [&](const UsdGeomPrimvar& primvar)
    {
        const std::string name = primvar.GetPrimvarName().GetString();
        return std::find(fallback_exclusions.begin(), fallback_exclusions.end(), name) != fallback_exclusions.end();
    };

    UsdGeomPrimvar uv_primvar = primvars_API.GetPrimvar(TfToken("primvars:st"));
    if (has_uv_primvar_value(uv_primvar))
        return uv_primvar;

    uv_primvar = primvars_API.GetPrimvar(TfToken("primvars:st_0"));
    if (has_uv_primvar_value(uv_primvar))
        return uv_primvar;

    std::vector<UsdGeomPrimvar> all_primvars = primvars_API.GetPrimvars();
    for (auto& it : all_primvars) {
        if (has_uv_primvar_value(it) && !is_excluded(it))
            return it;
    }

    return UsdGeomPrimvar();
}

inline std::optional<float2> get_float2_value(const pxr::UsdShadeInput& input)
{
    using namespace pxr;

    SdfValueTypeName type_name = input.GetTypeName();

    if (type_name == SdfValueTypeNames->Float2) {
        GfVec2f val;
        input.Get(&val);
        return to_falcor(val);
    }

    if (type_name == SdfValueTypeNames->Float) {
        float val;
        input.Get(&val);
        return float2(val);
    }

    return {};
}

inline std::optional<float4> get_float4_value(const pxr::UsdShadeInput& input)
{
    using namespace pxr;

    SdfValueTypeName type_name = input.GetTypeName();
    if (type_name == SdfValueTypeNames->Float4) {
        GfVec4f val;
        input.Get(&val);
        return to_falcor(val);
    }

    if (type_name == SdfValueTypeNames->Float3) {
        GfVec3f val;
        input.Get(&val);
        return float4(to_falcor(val), 1.f);
    }

    if (type_name == SdfValueTypeNames->Float) {
        float val;
        input.Get(&val);
        return float4(val);
    }

    return {};
}

} // namespace usd_importer
} // namespace falcor
