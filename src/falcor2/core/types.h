// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <sgl/math/scalar_types.h>
#include <sgl/math/scalar_math.h>
#include <sgl/math/vector.h>
#include <sgl/math/matrix.h>
#include <sgl/math/quaternion.h>

namespace falcor {

/// scalar_types

using uint = sgl::uint;
using float16_t = sgl::float16_t;

SGL_DIAGNOSTIC_PUSH
// disable warning about literal suffixes not starting with an underscore
SGL_DISABLE_MSVC_WARNING(4455)

using sgl::math::operator""h;

SGL_DIAGNOSTIC_POP

/// vector_types

using bool1 = sgl::bool1;
using bool2 = sgl::bool2;
using bool3 = sgl::bool3;
using bool4 = sgl::bool4;
using int1 = sgl::int1;
using int2 = sgl::int2;
using int3 = sgl::int3;
using int4 = sgl::int4;
using uint1 = sgl::uint1;
using uint2 = sgl::uint2;
using uint3 = sgl::uint3;
using uint4 = sgl::uint4;
using float1 = sgl::float1;
using float2 = sgl::float2;
using float3 = sgl::float3;
using float4 = sgl::float4;
using float16_t1 = sgl::float16_t1;
using float16_t2 = sgl::float16_t2;
using float16_t3 = sgl::float16_t3;
using float16_t4 = sgl::float16_t4;

/// matrix_types

using float2x2 = sgl::float2x2;

using float3x3 = sgl::float3x3;

using float2x4 = sgl::float2x4;
using float3x4 = sgl::float3x4;
using float4x4 = sgl::float4x4;

/// quaternion_types

using quatf = sgl::quatf;


} // namespace falcor
