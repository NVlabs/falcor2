// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_utils.h"
#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
#include <pxr/base/gf/matrix3d.h>
#include <pxr/base/gf/matrix4d.h>
#include <pxr/base/gf/numericCast.h>
#include <pxr/base/gf/vec2d.h>
#include <pxr/base/gf/vec2h.h>
#include <pxr/base/gf/vec2i.h>
#include <pxr/base/gf/vec3d.h>
#include <pxr/base/gf/vec3f.h>
#include <pxr/base/gf/vec3h.h>
#include <pxr/base/gf/vec3i.h>
#include <pxr/base/gf/vec4d.h>
#include <pxr/base/gf/vec4f.h>
#include <pxr/base/gf/vec4h.h>
#include <pxr/base/gf/vec4i.h>
#include <pxr/base/vt/array.h>
#include <pxr/base/tf/stringUtils.h>
#include <pxr/base/vt/value.h>
#include <pxr/usd/ar/asset.h>
#include <pxr/usd/ar/packageUtils.h>
#include <pxr/usd/ar/resolvedPath.h>
#include <pxr/usd/ar/resolver.h>
#include <pxr/usd/sdf/assetPath.h>
END_DISABLE_USD_WARNINGS

#include <array>
#include <stdexcept>
#include <vector>

namespace falcor {
namespace usd_importer {

pxr::VtArray<int> compute_curve_varying_counts(const pxr::UsdGeomBasisCurves& curves, pxr::UsdTimeCode time)
{
    // USD specifies one varying value per span boundary: segment count + 1 for open curves,
    // segment count for periodic curves. In particular, pinned B-spline/Catmull-Rom curves
    // with N points have N varying values. ComputeVaryingDataSize() and Storm's topology
    // helper instead use the nonperiodic count (N - 2), and the former also underflows on
    // short pinned curves. See OpenUSD issue #2775 and docs/design/usd-curve-attributes.md.
    // ComputeSegmentCounts() handles pinned end spans correctly, so derive our counts from it.
    auto counts = curves.ComputeSegmentCounts(time);
    pxr::TfToken wrap;
    curves.GetWrapAttr().Get(&wrap, time);
    if (wrap != pxr::UsdGeomTokens->periodic) {
        for (int& count : counts)
            ++count;
    }
    return counts;
}

bool read_fixed_indices(
    const pxr::UsdAttribute& attribute,
    pxr::VtArray<int>& indices,
    pxr::UsdTimeCode time,
    bool import_motion
)
{
    if (!attribute)
        return false;
    if (!import_motion)
        return attribute.Get(&indices, time);

    const bool present = attribute.Get(&indices, time.IsDefault() ? pxr::UsdTimeCode::EarliestTime() : time);
    // Index changes are held in USD and cannot be represented by interpolating our fixed-layout samples.
    std::vector<double> times;
    attribute.GetTimeSamples(&times);
    pxr::VtArray<int> sampled_indices;
    for (double sample_time : times) {
        FALCOR_CHECK(
            attribute.Get(&sampled_indices, sample_time) && present && sampled_indices == indices,
            "Motion import changes indices in '{}'",
            attribute.GetPath().GetString()
        );
    }
    return present;
}

GeometrySampleTimes::GeometrySampleTimes(
    const pxr::UsdPrim& prim,
    const std::vector<pxr::UsdAttribute>& attributes,
    bool import_motion,
    pxr::UsdTimeCode no_samples
)
{
    std::vector<double> authored_times;
    if (import_motion)
        pxr::UsdAttribute::GetUnionedTimeSamples(attributes, &authored_times);
    sample_times.assign(authored_times.begin(), authored_times.end());
    if (sample_times.empty())
        sample_times.push_back(import_motion ? no_samples : pxr::UsdTimeCode::Default());
    if (authored_times.size() > 1) {
        interpolation = prim.GetStage()->GetInterpolationType() == pxr::UsdInterpolationTypeHeld
            ? AnimationInterpolationMode::step
            : AnimationInterpolationMode::linear;
        double rate = prim.GetStage()->GetTimeCodesPerSecond();
        if (rate <= 0.0)
            rate = 24.0;
        for (double time : authored_times)
            motion_times.push_back(static_cast<float>(time / rate));
    }
}

/// Read USD's half/float/double texture-coordinate arrays into Falcor's float2 UV storage, retaining separate indices.
bool get_uv_values(
    const pxr::UsdGeomPrimvar& primvar,
    pxr::VtArray<pxr::GfVec2f>& values,
    pxr::UsdTimeCode time,
    bool flatten
)
{
    using namespace pxr;

    const SdfValueTypeName type_name = primvar.GetTypeName();
    if (type_name == SdfValueTypeNames->Float2Array || type_name == SdfValueTypeNames->TexCoord2fArray)
        return flatten ? primvar.ComputeFlattened(&values, time) : primvar.Get(&values, time);

    if (type_name != SdfValueTypeNames->Half2Array && type_name != SdfValueTypeNames->TexCoord2hArray
        && type_name != SdfValueTypeNames->Double2Array && type_name != SdfValueTypeNames->TexCoord2dArray)
        return false;

    VtValue source;
    if (!(flatten ? primvar.ComputeFlattened(&source, time) : primvar.Get(&source, time)))
        return false;
    source.Cast<VtArray<GfVec2f>>();
    if (!source.IsHolding<VtArray<GfVec2f>>())
        return false;
    values = source.UncheckedGet<VtArray<GfVec2f>>();
    return true;
}


bool is_package_asset(const std::filesystem::path& path)
{
    return pxr::ArIsPackageRelativePath(path.string());
}

std::optional<std::vector<uint8_t>> read_package_asset(const std::filesystem::path& path)
{
    const std::string path_string = path.string();
    if (!is_package_asset(path))
        return {};

    const std::shared_ptr<pxr::ArAsset> asset = pxr::ArGetResolver().OpenAsset(pxr::ArResolvedPath(path_string));
    if (!asset)
        throw std::runtime_error("Failed to open packaged USD asset: " + path_string);

    std::vector<uint8_t> data(asset->GetSize());
    if (!data.empty() && asset->Read(data.data(), data.size(), 0) != data.size())
        throw std::runtime_error("Failed to read packaged USD asset: " + path_string);

    return data;
}

namespace {

template<typename T>
bool try_set_exact(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<T>())
        return false;

    props.set(name, value.UncheckedGet<T>());
    return true;
}

template<typename Target, typename Source>
bool try_set_numeric(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<Source>())
        return false;

    auto converted = pxr::GfNumericCast<Target>(value.UncheckedGet<Source>());
    if (!converted)
        return false;

    props.set(name, *converted);
    return true;
}

template<typename Target, typename... Sources>
bool try_set_numeric_sources(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    return (try_set_numeric<Target, Sources>(props, name, value) || ...);
}

template<typename Target, typename Source>
bool try_set_numeric_array(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<pxr::VtArray<Source>>())
        return false;

    const auto& source = value.UncheckedGet<pxr::VtArray<Source>>().AsConst();
    std::vector<Target> converted;
    converted.reserve(source.size());
    for (const auto& element : source) {
        auto converted_value = pxr::GfNumericCast<Target>(element);
        if (!converted_value)
            return false;
        converted.push_back(*converted_value);
    }

    props.set_list(name, std::move(converted));
    return true;
}

template<typename Target, typename... Sources>
bool try_set_numeric_array_sources(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    return (try_set_numeric_array<Target, Sources>(props, name, value) || ...);
}

template<typename TargetVector, typename SourceVector>
std::optional<TargetVector> try_convert_vector(const SourceVector& value)
{
    // Enforce that TargetVector is a specialization of sgl::math::vector
    static_assert(
        std::is_same_v<TargetVector, sgl::math::vector<typename TargetVector::value_type, TargetVector::dimension>>,
        "TargetVector must be a specialization of sgl::math::vector"
    );
    using TargetComponent = typename TargetVector::value_type;
    constexpr int N = TargetVector::dimension;
    static_assert(N >= 2 && N <= 4, "Vector conversion only supports 2, 3, or 4 components");
    TargetVector components{};
    for (int index = 0; index < N; ++index) {
        auto converted = pxr::GfNumericCast<TargetComponent>(value[index]);
        if (!converted)
            return {};
        components[index] = *converted;
    }
    return components;
}

template<typename TargetVector, typename SourceVector>
bool try_set_vector(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<SourceVector>())
        return false;

    if constexpr (std::is_same_v<SourceVector, TargetVector>) {
        props.set(name, value.UncheckedGet<SourceVector>());
        return true;
    } else {
        auto converted = try_convert_vector<TargetVector, SourceVector>(value.UncheckedGet<SourceVector>());
        if (!converted)
            return false;

        props.set(name, *converted);
        return true;
    }
}

template<typename TargetVector, typename... SourceVectors>
bool try_set_vector_sources(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    return (try_set_vector<TargetVector, SourceVectors>(props, name, value) || ...);
}

template<typename TargetMatrix, typename SourceMatrix>
bool try_set_matrix(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<SourceMatrix>())
        return false;

    const auto& source = value.UncheckedGet<SourceMatrix>();
    TargetMatrix converted;
    // Preserve authored coefficients; geometric transforms use a separate transposing conversion.
    for (int r = 0; r < TargetMatrix::rows; ++r) {
        for (int c = 0; c < TargetMatrix::cols; ++c) {
            auto component = pxr::GfNumericCast<typename TargetMatrix::value_type>(source[r][c]);
            if (!component)
                return false;
            converted[r][c] = *component;
        }
    }
    props.set(name, converted);
    return true;
}

template<typename TargetVector, typename SourceVector>
bool try_set_vector_array(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<pxr::VtArray<SourceVector>>())
        return false;

    const auto& source = value.UncheckedGet<pxr::VtArray<SourceVector>>().AsConst();
    std::vector<TargetVector> converted;
    converted.reserve(source.size());

    if constexpr (std::is_same_v<SourceVector, TargetVector>) {
        for (const auto& element : source)
            converted.push_back(element);
    } else {
        for (const auto& element : source) {
            auto converted_value = try_convert_vector<TargetVector, SourceVector>(element);
            if (!converted_value)
                return false;
            converted.push_back(*converted_value);
        }
    }

    props.set_list(name, std::move(converted));
    return true;
}

template<typename TargetVector, typename... SourceVectors>
bool try_set_vector_array_sources(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    return (try_set_vector_array<TargetVector, SourceVectors>(props, name, value) || ...);
}

bool try_set_string(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (try_set_exact<std::string>(props, name, value))
        return true;

    if (value.IsHolding<pxr::SdfAssetPath>()) {
        const auto& asset_path = value.UncheckedGet<pxr::SdfAssetPath>();
        props.set(
            name,
            asset_path.GetResolvedPath().empty() ? asset_path.GetAssetPath() : asset_path.GetResolvedPath()
        );
        return true;
    }

    pxr::VtValue string_value = pxr::VtValue::Cast<std::string>(value);
    if (!string_value.IsEmpty() && string_value.IsHolding<std::string>()) {
        props.set(name, string_value.UncheckedGet<std::string>());
        return true;
    }

    return false;
}

template<typename Source>
bool try_set_string_array(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<pxr::VtArray<Source>>())
        return false;

    const auto& source = value.UncheckedGet<pxr::VtArray<Source>>().AsConst();
    std::vector<std::string> converted;
    converted.reserve(source.size());
    for (const auto& element : source) {
        if constexpr (std::is_same_v<Source, std::string>) {
            converted.push_back(element);
        } else if constexpr (std::is_same_v<Source, pxr::SdfAssetPath>) {
            converted.push_back(element.GetResolvedPath().empty() ? element.GetAssetPath() : element.GetResolvedPath());
        } else {
            pxr::VtValue string_value = pxr::VtValue::Cast<std::string>(pxr::VtValue(element));
            if (string_value.IsEmpty() || !string_value.IsHolding<std::string>())
                return false;
            converted.push_back(string_value.UncheckedGet<std::string>());
        }
    }

    props.set_list(name, std::move(converted));
    return true;
}

template<typename... Sources>
bool try_set_string_array_sources(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    return (try_set_string_array<Sources>(props, name, value) || ...);
}

} // namespace

bool set_from_value(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (value.IsEmpty())
        return false;
    if (value.IsArrayValued()) {
        if (try_set_numeric_array_sources<bool, bool>(props, name, value))
            return true;

        if (try_set_numeric_array_sources<
                int64_t,
                int8_t,
                uint8_t,
                int16_t,
                uint16_t,
                int32_t,
                uint32_t,
                int64_t,
                uint64_t>(props, name, value))
            return true;

        if (try_set_numeric_array_sources<double, float, double, pxr::GfHalf>(props, name, value))
            return true;

        if (try_set_vector_array_sources<int2, pxr::GfVec2i>(props, name, value))
            return true;

        if (try_set_vector_array_sources<int3, pxr::GfVec3i>(props, name, value))
            return true;

        if (try_set_vector_array_sources<int4, pxr::GfVec4i>(props, name, value))
            return true;

        if (try_set_vector_array_sources<uint2, pxr::GfSize2>(props, name, value))
            return true;

        if (try_set_vector_array_sources<uint3, pxr::GfSize3>(props, name, value))
            return true;

        if (try_set_vector_array_sources<float2, pxr::GfVec2f, pxr::GfVec2d, pxr::GfVec2h>(props, name, value))
            return true;

        if (try_set_vector_array_sources<float3, pxr::GfVec3f, pxr::GfVec3d, pxr::GfVec3h>(props, name, value))
            return true;

        if (try_set_vector_array_sources<float4, pxr::GfVec4f, pxr::GfVec4d, pxr::GfVec4h>(props, name, value))
            return true;

        if (try_set_string_array_sources<std::string, pxr::TfToken, pxr::SdfAssetPath>(props, name, value))
            return true;

        props.set(name, pxr::TfStringify(value));
        return true;
    }

    if (try_set_exact<bool>(props, name, value))
        return true;

    if (try_set_numeric_sources<int64_t, int8_t, uint8_t, int16_t, uint16_t, int32_t, uint32_t, int64_t, uint64_t>(
            props,
            name,
            value
        ))
        return true;

    if (try_set_numeric_sources<double, float, double, pxr::GfHalf>(props, name, value))
        return true;

    if (try_set_vector_sources<int2, pxr::GfVec2i>(props, name, value))
        return true;

    if (try_set_vector_sources<int3, pxr::GfVec3i>(props, name, value))
        return true;

    if (try_set_vector_sources<int4, pxr::GfVec4i>(props, name, value))
        return true;

    if (try_set_vector_sources<uint2, pxr::GfSize2>(props, name, value))
        return true;

    if (try_set_vector_sources<uint3, pxr::GfSize3>(props, name, value))
        return true;

    if (try_set_vector_sources<float2, pxr::GfVec2f, pxr::GfVec2d, pxr::GfVec2h>(props, name, value))
        return true;

    if (try_set_vector_sources<float3, pxr::GfVec3f, pxr::GfVec3d, pxr::GfVec3h>(props, name, value))
        return true;

    if (try_set_vector_sources<float4, pxr::GfVec4f, pxr::GfVec4d, pxr::GfVec4h>(props, name, value))
        return true;

    if (try_set_matrix<float3x3, pxr::GfMatrix3d>(props, name, value))
        return true;

    if (try_set_matrix<float4x4, pxr::GfMatrix4d>(props, name, value))
        return true;

    if (try_set_string(props, name, value))
        return true;

    props.set(name, pxr::TfStringify(value));
    return true;
}

} // namespace usd_importer
} // namespace falcor
