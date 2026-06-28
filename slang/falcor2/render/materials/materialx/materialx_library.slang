// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module materialx_library;

// GLSL compatibility so the mx_noise and mx_hsv can stay the same they are in GLSL
typedef float2 vec2;
typedef float3 vec3;
typedef float4 vec4;
typedef uint2 uvec2;
typedef uint3 uvec3;


__include falcor2.render.materials.materialx.libraries.mx_bsdfs;
__include falcor2.render.materials.materialx.libraries.mx_hsv;
__include falcor2.render.materials.materialx.libraries.mx_noise;
__include falcor2.render.materials.materialx.libraries.mx_procedural;
