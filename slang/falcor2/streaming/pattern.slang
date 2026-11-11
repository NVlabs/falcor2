// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

RWTexture2D<float4> pattern;
uint width;
uint height;
uint frame_index;

float3 pattern_color(uint x)
{
    const float3 bars[8]
        = { float3(1, 1, 1),
            float3(1, 1, 0),
            float3(0, 1, 1),
            float3(0, 1, 0),
            float3(1, 0, 1),
            float3(1, 0, 0),
            float3(0, 0, 1),
            float3(0, 0, 0) };
    return bars[min(x * 8 / width, 7)];
}

[shader("compute")]
[numthreads(8, 8, 1)]
void make_animated_pattern(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= width || tid.y >= height)
        return;
    pattern[tid.xy] = float4(pattern_color(tid.x), 1);
    if (tid.y >= height / 2) {
        float grey = float(tid.x) / float(width - 1);
        float3 color = float3(grey);
        // Moving magenta stripe over a display-linear grey ramp. The stripe
        // stays aligned to 2x2 chroma blocks and wraps across the image.
        if ((tid.x + width - ((frame_index * 8) % width)) % width < 32)
            color = float3(1, 0, 1);
        pattern[tid.xy] = float4(color, 1);
    }
}

float3 camera_position;
float3 camera_right;
float3 camera_up;
float3 camera_forward;

[shader("compute")]
[numthreads(8, 8, 1)]
void render_camera(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= width || tid.y >= height)
        return;
    float2 uv = (float2(tid.xy) + 0.5) / float2(width, height) * 2.0 - 1.0;
    uv *= float2(float(width) / height, -1.0) * 0.7;
    float3 dir = normalize(camera_forward + uv.x * camera_right + uv.y * camera_up);
    float3 color = lerp(float3(0.08, 0.12, 0.2), float3(0.5, 0.7, 0.9), saturate(dir.y * 0.5 + 0.5));
    float nearest = 1.0e20;
    float3 normal = float3(0, 1, 0);
    float3 albedo = float3(0);
    if (dir.y < -0.0001) {
        float t = -camera_position.y / dir.y;
        if (t > 0) {
            nearest = t;
            float3 p = camera_position + dir * t;
            float checker = fmod(abs(floor(p.x) + floor(p.z)), 2.0);
            albedo = lerp(float3(0.12), float3(0.65), checker);
        }
    }
    float3 offset = camera_position - float3(0, 1, 0);
    float b = dot(offset, dir);
    float discriminant = b * b - dot(offset, offset) + 1.0;
    if (discriminant >= 0) {
        float t = -b - sqrt(discriminant);
        if (t > 0 && t < nearest) {
            nearest = t;
            normal = normalize(offset + dir * t);
            albedo = float3(0.8, 0.18, 0.06);
        }
    }
    if (nearest < 1.0e20)
        color = albedo * (0.2 + 0.8 * saturate(dot(normal, normalize(float3(1, 2, 1)))));
    pattern[tid.xy] = float4(color, 1);
}
