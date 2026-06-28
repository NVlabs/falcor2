// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Debug viewer for the DLSS Super Resolution and Frame Generation example.

module dlss_viewer;

import falcor2.utils;
import slangpy;

uint2 map_pixel_to_texture(uint2 pixel, uint2 output_dims, uint2 texture_dims)
{
    float2 uv = (float2(pixel) + 0.5) / float2(output_dims);
    return min(uint2(uv * float2(texture_dims)), texture_dims - uint2(1, 1));
}

float4 load_rgba(Texture2D<float4> texture, uint2 pixel, uint2 output_dims)
{
    uint2 texture_dims;
    texture.GetDimensions(texture_dims.x, texture_dims.y);
    return texture.Load(int3(map_pixel_to_texture(pixel, output_dims, texture_dims), 0));
}

float load_r(Texture2D<float> texture, uint2 pixel, uint2 output_dims)
{
    uint2 texture_dims;
    texture.GetDimensions(texture_dims.x, texture_dims.y);
    return texture.Load(int3(map_pixel_to_texture(pixel, output_dims, texture_dims), 0));
}

float2 load_rg(Texture2D<float2> texture, uint2 pixel, uint2 output_dims)
{
    uint2 texture_dims;
    texture.GetDimensions(texture_dims.x, texture_dims.y);
    return texture.Load(int3(map_pixel_to_texture(pixel, output_dims, texture_dims), 0));
}

float3 display_color(float3 color)
{
    return tonemap_aces_film(max(color, 0.0));
}

float3 display_hardware_depth(float depth)
{
    return float3(depth, depth, depth);
}

float3 display_motion(float2 motion)
{
    return float3(0.5 + motion.x * 0.5, 0.5 + motion.y * 0.5, 0.5);
}

public void visualize_dlss_view(
    uint2 pixel,
    int view_index,
    Texture2D<float4> path_color,
    Texture2D<float4> dlss_sr,
    Texture2D<float4> dlss_fg,
    Texture2D<float> hardware_depth,
    Texture2D<float2> motion_vectors,
    RWTexture2D<float4> output
)
{
    uint2 output_dims;
    output.GetDimensions(output_dims.x, output_dims.y);
    if (any(pixel >= output_dims)) {
        return;
    }

    float3 value = float3(0, 0, 0);
    if (view_index == 0) {
        value = display_color(load_rgba(path_color, pixel, output_dims).rgb);
    } else if (view_index == 1) {
        value = display_color(load_rgba(dlss_sr, pixel, output_dims).rgb);
    } else if (view_index == 2) {
        value = display_color(load_rgba(dlss_fg, pixel, output_dims).rgb);
    } else if (view_index == 3) {
        value = display_hardware_depth(load_r(hardware_depth, pixel, output_dims));
    } else if (view_index == 4) {
        value = display_motion(load_rg(motion_vectors, pixel, output_dims));
    }

    output[pixel] = float4(saturate(value), 1.0);
}
