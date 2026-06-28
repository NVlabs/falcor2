// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public enum class PackOptions {
    /// No special handling of out-of-range values or NaNs.
    unsafe = 0,
    /// Clamp out-of-range values to valid range.
    clamp = 1 << 0,
    /// Convert NaNs to zero.
    nan_to_zero = 1 << 1,
    /// Safe conversion that clamps out-of-range values and converts NaNs to zero.
    safe = clamp | nan_to_zero,
}

// ----------------------------------------------------------------------------
// 8-bit snorm
// ----------------------------------------------------------------------------

/// Pack single float value to 8-bit snorm.
/// @param v Float value in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm value in low bits, high bits all zero.
public uint pack_snorm8(float v, const PackOptions options = PackOptions::safe)
{
    if (options & PackOptions::nan_to_zero)
        v = isnan(v) ? 0.0 : v;
    if (options & PackOptions::clamp)
        v = clamp(v, -1.0, 1.0);
    return (int)round(v * 127.0) & 0xff;
}

/// Pack two float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
public uint pack_snorm2x8(float2 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8);
}

/// Pack three float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
public uint pack_snorm3x8(float3 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8) | (pack_snorm8(v.z, options) << 16);
}

/// Pack four float values to 8-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 8-bit snorm values in low bits, high bits all zero.
public uint pack_snorm4x8(float4 v, const PackOptions options = PackOptions::safe)
{
    return pack_snorm8(v.x, options) | (pack_snorm8(v.y, options) << 8) | (pack_snorm8(v.z, options) << 16)
        | (pack_snorm8(v.w, options) << 24);
}

/// Unpack single 8-bit snorm value to float.
/// @param packed 8-bit snorm value in low bits.
/// @return Float value in [-1,1].
public float unpack_snorm8(uint packed)
{
    int bits = (int)(packed << 24) >> 24;
    return max((float)bits / 127.0, -1.0);
}

/// Unpack two 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
public float2 unpack_snorm2x8(uint packed)
{
    return float2(unpack_snorm8(packed), unpack_snorm8(packed >> 8));
}

/// Unpack three 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
public float3 unpack_snorm3x8(uint packed)
{
    return float3(unpack_snorm8(packed), unpack_snorm8(packed >> 8), unpack_snorm8(packed >> 16));
}

/// Unpack four 8-bit snorm values to float.
/// @param packed 8-bit snorm values in low bits.
/// @return Float values in [-1,1].
public float4 unpack_snorm4x8(uint packed)
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
public uint pack_unorm8(float v, const PackOptions options = PackOptions::unsafe)
{
    if (options & PackOptions::nan_to_zero)
        v = isnan(v) ? 0.0 : v;
    if (options & PackOptions::clamp)
        v = saturate(v);
    return (uint)round(v * 255.0) & 0xff;
}

/// Pack two float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
public uint pack_unorm2x8(float2 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.y, options) << 8) | pack_unorm8(v.x, options);
}

/// Pack three float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
public uint pack_unorm3x8(float3 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.z, options) << 16) | (pack_unorm8(v.y, options) << 8) | pack_unorm8(v.x, options);
}

/// Pack four float values to 8-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 8-bit unorm values in low bits, high bits all zero.
public uint pack_unorm4x8(float4 v, const PackOptions options = PackOptions::safe)
{
    return (pack_unorm8(v.w, options) << 24) | (pack_unorm8(v.z, options) << 16) | (pack_unorm8(v.y, options) << 8)
        | pack_unorm8(v.x, options);
}

/// Unpack single 8-bit unorm value to float.
/// @param packed 8-bit unorm value in low bits.
/// @return Float value in [0,1].
public float unpack_unorm8(uint packed)
{
    return float(packed & 0xff) * (1.0 / 255);
}

/// Unpack two 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
public float2 unpack_unorm2x8(uint packed)
{
    return float2(uint2(packed, packed >> 8) & 0xff) * (1.0 / 255);
}

/// Unpack three 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
public float3 unpack_unorm3x8(uint packed)
{
    return float3(uint3(packed, packed >> 8, packed >> 16) & 0xff) * (1.0 / 255);
}

/// Unpack four 8-bit unorm values to float.
/// @param packed 8-bit unorm values in low bits.
/// @return Float values in [0,1].
public float4 unpack_unorm4x8(uint packed)
{
    return float4(uint4(packed, packed >> 8, packed >> 16, packed >> 24) & 0xff) * (1.0 / 255);
}

// ----------------------------------------------------------------------------
// 16-bit snorm
// ----------------------------------------------------------------------------

/// Pack single float value to 16-bit snorm.
/// @param v Float value in [-1,1].
/// @param options Packing options.
/// @return 16-bit snorm value in low bits, high bits all zero.
public uint pack_snorm16(float v, const PackOptions options = PackOptions::safe)
{
    if (options & PackOptions::nan_to_zero)
        v = isnan(v) ? 0.0 : v;
    if (options & PackOptions::clamp)
        v = clamp(v, -1.0, 1.0);
    return (int)round(v * 32767.0) & 0xffff;
}

/// Pack two float values to 16-bit snorm.
/// @param v Float values in [-1,1].
/// @param options Packing options.
/// @return 16-bit snorm values.
public uint pack_snorm2x16(float2 v, const PackOptions options = PackOptions::safe)
{
    return (pack_snorm16(v.y, options) << 16) | pack_snorm16(v.x, options);
}

/// Unpack single 16-bit snorm value to float.
/// @param packed 16-bit snorm value in low bits.
/// @return Float value in [-1,1].
public float unpack_snorm16(uint packed)
{
    int bits = (int)(packed << 16) >> 16;
    return max((float)bits / 32767.0, -1.0);
}

/// Unpack two 16-bit snorm values to float.
/// @param packed 16-bit snorm values.
/// @return Float value in [-1,1].
public float2 unpack_snorm2x16(uint packed)
{
    return unpackSnorm2x16ToFloat(packed);
}

// ----------------------------------------------------------------------------
// 16-bit unorm
// ----------------------------------------------------------------------------

/// Pack single float value to 16-bit unorm.
/// @param v Float value in [0,1].
/// @param options Packing options.
/// @return 16-bit unorm value in low bits, high bits all zero.
public uint pack_unorm16(float v, const PackOptions options = PackOptions::unsafe)
{
    if (options & PackOptions::nan_to_zero)
        v = isnan(v) ? 0.0 : v;
    if (options & PackOptions::clamp)
        v = saturate(v);
    return (uint)round(v * 65535.0) & 0xffff;
}

/// Pack two float values to 16-bit unorm.
/// @param v Float values in [0,1].
/// @param options Packing options.
/// @return 16-bit unorm values.
public uint pack_unorm2x16(float2 v, const PackOptions options = PackOptions::unsafe)
{
    return (pack_unorm16(v.y, options) << 16) | pack_unorm16(v.x, options);
}

/// Unpack single 16-bit unorm value to float.
/// @param packed 16-bit unorm value in low bits.
/// @return Float value in [0,1].
public float unpack_unorm16(uint packed)
{
    return float(packed & 0xffff) * (1.0 / 65535);
}

/// Unpack two 16-bit unorm values to float.
/// @param packed 16-bit unorm values.
/// @return Float value in [0,1].
public float2 unpack_unorm2x16(uint packed)
{
    return unpackUnorm2x16ToFloat(packed);
}

// ----------------------------------------------------------------------------
// 32-bit HDR color format
// ----------------------------------------------------------------------------

/// Pack three positive floats into a dword.
///
/// https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
///
/// @param v RGB color with positive float components.
/// @return Packed dword.
public uint pack_r11g11b10(float3 v, const PackOptions options = PackOptions::safe)
{
    if (options & PackOptions::nan_to_zero) {
        v = select(isnan(v), 0.0, v);
    }
    if (options & PackOptions::clamp) {
        // Clamp upper bound so that it doesn't accidentally round up to INF
        // Exponent=15, Mantissa=1.11111
        v = clamp(v, 0.0, asfloat(0x477C0000));
    }
    uint r = ((f32tof16(v.x) + 8) >> 4) & 0x000007ff;
    uint g = ((f32tof16(v.y) + 8) << 7) & 0x003ff800;
    uint b = ((f32tof16(v.z) + 16) << 17) & 0xffc00000;
    return r | g | b;
}

/// Unpack three positive floats from a dword.
///
/// https://github.com/microsoft/DirectX-Graphics-Samples/blob/master/MiniEngine/Core/Shaders/PixelPacking_R11G11B10.hlsli
///
/// @param packed Packed dword.
/// @return RGB color with positive float components.
public float3 unpack_r11g11b10(uint packed)
{
    float r = f16tof32((packed << 4) & 0x7FF0);
    float g = f16tof32((packed >> 7) & 0x7FF0);
    float b = f16tof32((packed >> 17) & 0x7FE0);
    return float3(r, g, b);
}

// ----------------------------------------------------------------------------
// Normal packing in octahedral mapping
// ----------------------------------------------------------------------------

/// Pack a normal as 2x 8-bit snorms in the octahedral mapping.
/// @param normal Normal vector to encode.
/// @return Packed normal in 2x 8-bit snorms.
public uint pack_normal_oct_snorm2x8(float3 normal)
{
    float2 oct = ndir_to_oct_snorm(normal);
    return pack_snorm2x8(oct);
}

/// Unpack normal packed 2x 8-bit snorms in the octahedral mapping.
/// @param packed Packed normal in 2x 8-bit snorms.
/// @return Unpacked normal vector.
public float3 unpack_normal_oct_snorm2x8(uint packed)
{
    float2 oct = unpack_snorm2x8(packed);
    return oct_to_ndir_snorm(oct);
}

/// Pack a normal as 2x 16-bit snorms in the octahedral mapping.
/// @param normal Normal vector to encode.
/// @return Packed normal in 2x 16-bit snorms.
public uint pack_normal_oct_snorm2x16(float3 normal)
{
    float2 oct = ndir_to_oct_snorm(normal);
    return pack_snorm2x16(oct);
}

/// Unpack normal packed 2x 16-bit snorms in the octahedral mapping.
/// @param packed Packed normal in 2x 16-bit snorms.
/// @return Unpacked normal vector.
public float3 unpack_normal_oct_snorm2x16(uint packed)
{
    float2 oct = unpack_snorm2x16(packed);
    return oct_to_ndir_snorm(oct);
}
