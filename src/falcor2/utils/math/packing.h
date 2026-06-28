// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "warp.h"

#include "falcor2/core/types.h"
#include "falcor2/core/enum.h"

namespace falcor {

enum class PackOptions {
    /// No special handling of out-of-range values or NaNs.
    unsafe = 0,
    /// Clamp out-of-range values to valid range.
    clamp = 1 << 0,
    /// Convert NaNs to zero.
    nan_to_zero = 1 << 1,
    /// Safe conversion that clamps out-of-range values and converts NaNs to zero.
    safe = clamp | nan_to_zero,
};
FALCOR_ENUM_CLASS_OPERATORS(PackOptions);

// ----------------------------------------------------------------------------
// 8-bit snorm
// ----------------------------------------------------------------------------

/// Pack single float value to 8-bit snorm.
/// @param v Float value in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm value in low bits, high bits all zero.
inline uint32_t pack_snorm8(float v, const PackOptions options = PackOptions::safe)
{
    if (is_set(options, PackOptions::nan_to_zero))
        v = sgl::math::isnan(v) ? 0.f : v;
    if (is_set(options, PackOptions::clamp))
        v = sgl::math::clamp(v, -1.f, 1.f);
    return (int)sgl::math::round(v * 127.f) & 0xff;
}

/// Pack two float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
inline uint32_t pack_snorm2x8(float2 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8);
}

/// Pack three float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
inline uint32_t pack_snorm3x8(float3 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8) | (pack_snorm8(v.z, options) << 16);
}

/// Pack four float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
inline uint32_t pack_snorm4x8(float4 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8) | (pack_snorm8(v.z, options) << 16)
        | (pack_snorm8(v.w, options) << 24);
}

/// Unpack single 8-bit snorm value to float.
/// @param packed 8-bit snorm value in low bits.
/// @return Float value in [-1,1].
inline float unpack_snorm8(uint32_t packed)
{
    int bits = (int)(packed << 24) >> 24;
    return sgl::math::max((float)bits / 127.f, -1.f);
}

/// Unpack two 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
inline float2 unpack_snorm2x8(uint32_t packed)
{
    return float2(unpack_snorm8(packed), unpack_snorm8(packed >> 8));
}

/// Unpack three 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
inline float3 unpack_snorm3x8(uint32_t packed)
{
    return float3(unpack_snorm8(packed), unpack_snorm8(packed >> 8), unpack_snorm8(packed >> 16));
}

/// Unpack four 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
inline float4 unpack_snorm4x8(uint32_t packed)
{
    return float4(
        unpack_snorm8(packed),
        unpack_snorm8(packed >> 8),
        unpack_snorm8(packed >> 16),
        unpack_snorm8(packed >> 24)
    );
}

// ----------------------------------------------------------------------------
// 8-bit unorm
// ----------------------------------------------------------------------------

/// Pack single float value to 8-bit unorm.
/// @param v Float value in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm value in low bits, high bits all zero.
inline uint32_t pack_unorm8(float v, const PackOptions options = PackOptions::unsafe)
{
    if (is_set(options, PackOptions::nan_to_zero))
        v = sgl::math::isnan(v) ? 0.f : v;
    if (is_set(options, PackOptions::clamp))
        v = sgl::math::saturate(v);
    return (uint32_t)sgl::math::round(v * 255.f) & 0xff;
}

/// Pack two float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
inline uint32_t pack_unorm2x8(float2 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.y, options) << 8) | pack_unorm8(v.x, options);
}

/// Pack three float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
inline uint32_t pack_unorm3x8(float3 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.z, options) << 16) | (pack_unorm8(v.y, options) << 8) | pack_unorm8(v.x, options);
}

/// Pack four float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
inline uint32_t pack_unorm4x8(float4 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.w, options) << 24) | (pack_unorm8(v.z, options) << 16) | (pack_unorm8(v.y, options) << 8)
        | pack_unorm8(v.x, options);
}

/// Unpack single 8-bit unorm value to float.
/// @param packed 8-bit unorm value in low bits.
/// @return Float value in [0,1].
inline float unpack_unorm8(uint32_t packed)
{
    return float(packed & 0xff) * (1.f / 255);
}

/// Unpack two 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
inline float2 unpack_unorm2x8(uint32_t packed)
{
    return float2(uint2(packed, packed >> 8) & uint2(0xff)) * (1.f / 255);
}

/// Unpack three 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
inline float3 unpack_unorm3x8(uint32_t packed)
{
    return float3(uint3(packed, packed >> 8, packed >> 16) & uint3(0xff)) * (1.f / 255);
}

/// Unpack four 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
inline float4 unpack_unorm4x8(uint32_t packed)
{
    return float4(uint4(packed, packed >> 8, packed >> 16, packed >> 24) & uint4(0xff)) * (1.f / 255);
}

// ----------------------------------------------------------------------------
// 16-bit snorm
// ----------------------------------------------------------------------------

/// Pack single float value to 16-bit snorm.
/// @param v Float value in [-1,1].
/// @param options Packing options.
/// @return 16-bit snorm value in low bits, high bits all zero.
inline uint32_t pack_snorm16(float v, const PackOptions options = PackOptions::safe)
{
    if (is_set(options, PackOptions::nan_to_zero))
        v = sgl::math::isnan(v) ? 0.f : v;
    if (is_set(options, PackOptions::clamp))
        v = sgl::math::clamp(v, -1.f, 1.f);
    return (int)sgl::math::round(v * 32767.f) & 0xffff;
}

/// Pack two float values to 16-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 16-bit snorm values.
inline uint32_t pack_snorm2x16(float2 v, const PackOptions options = PackOptions::safe)
{
    return (pack_snorm16(v.y, options) << 16) | pack_snorm16(v.x, options);
}

/// Unpack single 16-bit snorm value to float.
/// @param packed 16-bit snorm value in low bits.
/// @return Float value in [-1,1].
inline float unpack_snorm16(uint32_t packed)
{
    int bits = (int)(packed << 16) >> 16;
    return sgl::math::max((float)bits / 32767.f, -1.f);
}

/// Unpack two 16-bit snorm values to float.
/// @param packed 16-bit snorm values.
/// @return Float value in [-1,1].
inline float2 unpack_snorm2x16(uint32_t packed)
{
    return float2(unpack_snorm16(packed), unpack_snorm16(packed >> 16));
}

// ----------------------------------------------------------------------------
// 16-bit unorm
// ----------------------------------------------------------------------------

/// Pack single float value to 16-bit unorm.
/// @param v Float value in [0,1].
/// @param options Packing options.
/// @return 16-bit unorm value in low bits, high bits all zero.
inline uint32_t pack_unorm16(float v, const PackOptions options = PackOptions::unsafe)
{
    if (is_set(options, PackOptions::nan_to_zero))
        v = sgl::math::isnan(v) ? 0.f : v;
    if (is_set(options, PackOptions::clamp))
        v = sgl::math::saturate(v);
    return (uint32_t)sgl::math::round(v * 65535.f) & 0xffff;
}

/// Pack two float values to 16-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 16-bit unorm values.
inline uint32_t pack_unorm2x16(float2 v, const PackOptions options = PackOptions::unsafe)
{
    return (pack_unorm16(v.y, options) << 16) | pack_unorm16(v.x, options);
}

/// Unpack single 16-bit unorm value to float.
/// @param packed 16-bit unorm value in low bits.
/// @return Float value in [0,1].
inline float unpack_unorm16(uint32_t packed)
{
    return float(packed & 0xffff) * (1.f / 65535);
}

/// Unpack two 16-bit unorm values to float.
/// @param packed 16-bit unorm values.
/// @return Float value in [0,1].
inline float2 unpack_unorm2x16(uint32_t packed)
{
    return float2(unpack_unorm16(packed), unpack_unorm16(packed >> 16));
}

// ----------------------------------------------------------------------------
// Normal packing in octahedral mapping
// ----------------------------------------------------------------------------

/// Pack a normal as 2x 8-bit snorms in the octahedral mapping.
/// @param normal Normal vector to encode.
/// @return Packed normal in 2x 8-bit snorms.
inline uint32_t pack_normal_oct_snorm2x8(float3 normal)
{
    float2 oct = ndir_to_oct_snorm(normal);
    return pack_snorm2x8(oct);
}

/// Unpack normal packed 2x 8-bit snorms in the octahedral mapping.
/// @param packed Packed normal in 2x 8-bit snorms.
/// @return Unpacked normal vector.
inline float3 unpack_normal_oct_snorm2x8(uint32_t packed)
{
    float2 oct = unpack_snorm2x8(packed);
    return oct_to_ndir_snorm(oct);
}

/// Pack a normal as 2x 16-bit snorms in the octahedral mapping.
/// @param normal Normal vector to encode.
/// @return Packed normal in 2x 16-bit snorms.
inline uint32_t pack_normal_oct_snorm2x16(float3 normal)
{
    float2 oct = ndir_to_oct_snorm(normal);
    return pack_snorm2x16(oct);
}

/// Unpack normal packed 2x 16-bit snorms in the octahedral mapping.
/// @param packed Packed normal in 2x 16-bit snorms.
/// @return Unpacked normal vector.
inline float3 unpack_normal_oct_snorm2x16(uint32_t packed)
{
    float2 oct = unpack_snorm2x16(packed);
    return oct_to_ndir_snorm(oct);
}

} // namespace falcor
