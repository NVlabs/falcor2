// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;

module materialx_test_geomprop_provider;

__exported import falcor2.render;
__exported import falcor2.utils;

public enum class MaterialXTestGeomPropStream : uint {
    geompropvalue_integer = 1,
    geompropvalue_bool = 2,
    geompropvalue_float = 3,
    geompropvalue_vector2 = 4,
    geompropvalue_vector3 = 5,
    geompropvalue_vector4 = 6,
    geompropvalue_color2 = 7,
    geompropvalue_color3 = 8,
    geompropvalue_color4 = 9,
    myvalue = 10,
    texcoord_1 = 11,
    color_0 = 12,
    color_1 = 13,
};

float materialx_test_sine_data(float value, float frequency)
{
    return sin(value * 6.2831853071795864769 * frequency) * 0.5 + 1.0;
}

public struct MaterialXTestGeomPropProvider : IGeomPropProvider
{
    public static int get_int(SurfaceInteraction si, uint id)
    {
        switch (MaterialXTestGeomPropStream(id)) {
        case MaterialXTestGeomPropStream::geompropvalue_integer:
            return int(si.uv.y * 5.0);
        }
        return 0;
    }

    public static bool get_bool(SurfaceInteraction si, uint id) { return false; }

    public static float get_float(SurfaceInteraction si, uint id)
    {
        switch (MaterialXTestGeomPropStream(id)) {
        case MaterialXTestGeomPropStream::geompropvalue_float:
            return materialx_test_sine_data(si.uv.y, 12.0);
        }
        return 0.0;
    }

    public static float2 get_float2(SurfaceInteraction si, uint id)
    {
        switch (MaterialXTestGeomPropStream(id)) {
        case MaterialXTestGeomPropStream::texcoord_1:
            return float2(si.uv.y, si.uv.x);
        case MaterialXTestGeomPropStream::geompropvalue_vector2:
            return float2(materialx_test_sine_data(si.uv.y, 6.0), 0.0);
        case MaterialXTestGeomPropStream::geompropvalue_color2:
            return float2(materialx_test_sine_data(si.uv.x, 10.0), 0.0);
        }
        return si.uv;
    }

    public static float3 get_float3(SurfaceInteraction si, uint id)
    {
        switch (MaterialXTestGeomPropStream(id)) {
        case MaterialXTestGeomPropStream::geompropvalue_vector3:
            return float3(0.0, materialx_test_sine_data(si.uv.y, 8.0), 0.0);
        case MaterialXTestGeomPropStream::geompropvalue_color3:
            return float3(0.0, materialx_test_sine_data(si.uv.x, 8.0), 0.0);
        }
        return float3(si.uv, 0.0);
    }

    public static float4 get_float4(SurfaceInteraction si, uint id)
    {
        switch (MaterialXTestGeomPropStream(id)) {
        case MaterialXTestGeomPropStream::geompropvalue_vector4:
            return float4(0.0, 0.0, materialx_test_sine_data(si.uv.y, 10.0), 1.0);
        case MaterialXTestGeomPropStream::geompropvalue_color4:
            return float4(0.0, 0.0, materialx_test_sine_data(si.uv.x, 6.0), 1.0);
        case MaterialXTestGeomPropStream::color_0:
            return float4(si.uv.x, si.uv.y, 1.0, 1.0);
        case MaterialXTestGeomPropStream::color_1:
            return float4(1.0, si.uv.x, si.uv.y, 1.0);
        }
        return float4(si.uv, 0.0, 0.0);
    }

    public static float4x4 get_space_transform(SurfaceInteraction si, GeomPropSpace to_space, GeomPropSpace from_space)
    {
        return geomprop_identity4x4();
    }
};

export struct GeomPropProvider : IGeomPropProvider = MaterialXTestGeomPropProvider;
