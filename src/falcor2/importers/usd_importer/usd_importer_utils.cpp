// SPDX-License-Identifier: Apache-2.0

#include "usd_importer_utils.h"
#include "usd_importer_macros.h"

BEGIN_DISABLE_USD_WARNINGS
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
#include <pxr/usd/sdf/assetPath.h>
END_DISABLE_USD_WARNINGS

#include <array>
#include <cmath>
#include <limits>
#include <vector>

namespace falcor {
namespace usd_importer {
namespace {

template<typename T>
bool try_set_exact(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<T>())
        return false;

    props.set(name, value.UncheckedGet<T>());
    return true;
}

template<typename To, typename From>
std::optional<To> try_numeric_convert(From value)
{
    if constexpr (std::is_integral_v<To> && std::is_integral_v<From>) {
        if constexpr (std::is_signed_v<From>) {
            long long signed_value = static_cast<long long>(value);
            if constexpr (std::is_unsigned_v<To>) {
                if (signed_value < 0)
                    return {};
                if (static_cast<unsigned long long>(signed_value)
                    > static_cast<unsigned long long>(std::numeric_limits<To>::max()))
                    return {};
            } else {
                if (signed_value < static_cast<long long>(std::numeric_limits<To>::min())
                    || signed_value > static_cast<long long>(std::numeric_limits<To>::max()))
                    return {};
            }
        } else {
            unsigned long long unsigned_value = static_cast<unsigned long long>(value);
            if (unsigned_value > static_cast<unsigned long long>(std::numeric_limits<To>::max()))
                return {};
        }

        return static_cast<To>(value);
    } else if constexpr (std::is_floating_point_v<To> && std::is_convertible_v<From, To>) {
        return static_cast<To>(value);
    } else if constexpr (std::is_integral_v<To> && std::is_floating_point_v<From>) {
        if (!std::isfinite(value))
            return {};
        if (std::floor(value) != value)
            return {};
        if (value < static_cast<From>(std::numeric_limits<To>::min())
            || value > static_cast<From>(std::numeric_limits<To>::max()))
            return {};
        return static_cast<To>(value);
    } else {
        return {};
    }
}

template<typename Target, typename Source>
bool try_set_numeric(Properties& props, std::string_view name, const pxr::VtValue& value)
{
    if (!value.IsHolding<Source>())
        return false;

    auto converted = try_numeric_convert<Target>(value.UncheckedGet<Source>());
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
        auto converted_value = try_numeric_convert<Target>(element);
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
        auto converted = try_numeric_convert<TargetComponent>(value[index]);
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

    if (try_set_string(props, name, value))
        return true;

    props.set(name, pxr::TfStringify(value));
    return true;
}

} // namespace usd_importer
} // namespace falcor
