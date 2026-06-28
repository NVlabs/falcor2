// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

//
// Isolated helpers for mx139. This file intentionally does not import
// falcor2 or sgl; formulas are ported from OSL/BSDL when noted locally.
//
// Constants below are the BSDL counterparts from
// external/dbg_OSL/src/libbsdl/include/BSDL/config.h:40-49 unless noted.

implementing mx139;

namespace mx139 {

// BSDL: config.h:40 EPSILON = 1e-4f
public static constexpr float BSDL_EPSILON = 1e-4;
// BSDL: config.h:41 PI = float(M_PI)
public static constexpr float PI = 3.14159265358979323846;
// BSDL: config.h:42 ONEOVERPI = 1/PI
public static constexpr float INV_PI = 0.31830988618379067154;
// BSDL: config.h:45-46 FLOAT_MIN = 1.17549435e-38f (minimum normal float)
public static constexpr float FLOAT_MIN = 1.17549435e-38;
// BSDL: config.h:48 BIG = 1e12f
public static constexpr float BIG = 1e12;
// BSDL: config.h:49 PDF_MIN = 1e-6f
public static constexpr float PDF_MIN = 1e-6;
// BSDL: config.h:43-44 ALMOSTONE = 0.999999940395355224609375f (max float < 1)
public static constexpr float ALMOST_ONE = 0.999999940395355224609375;

public float mx_hmax(float2 value)
{
    return max(value.x, value.y);
}

public float mx_hmax(float3 value)
{
    return max(value.x, max(value.y, value.z));
}

public float mx_hmax(float4 value)
{
    return max(max(value.x, value.y), max(value.z, value.w));
}

} // namespace mx139
