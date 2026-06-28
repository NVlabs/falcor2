// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

// Debug viewer for the DLSS-RR example.
//
// The example drives DLSS-RR through PathTracerPipeline and keeps the last color,
// guide, and DLSS output textures around for inspection. This shader converts any
// selected texture into a displayable RGBA image for the editor window.

module dlss_rr_viewer;

import falcor2.utils;
import slangpy;

// Map a pixel in the editor output to a source texture pixel. The DLSS target is
// full resolution, while the path tracer guide textures can be at the lower NGX
// render resolution, so the viewer has to sample each source by normalized position.
uint2 map_pixel_to_texture(uint2 pixel, uint2 output_dims, uint2 texture_dims)
{
    float2 uv = (float2(pixel) + 0.5) / float2(output_dims);
    return min(uint2(uv * float2(texture_dims)), texture_dims - uint2(1, 1));
}

// Small load helpers for the texture shapes used by the guide outputs.
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

// Convert HDR color buffers to display-referred RGB.
float3 display_color(float3 color)
{
    return tonemap_aces_film(max(color, 0.0));
}

// Depth and hit distance are unbounded positive values, so compress them into a
// simple grayscale range for inspection.
float3 display_depth(float depth)
{
    float scaled_depth = depth / (depth + 20.0);
    return float3(scaled_depth, scaled_depth, scaled_depth);
}

// Motion vectors are signed pixel offsets. Center zero motion at mid gray and show
// horizontal/vertical motion in red/green.
float3 display_motion(float2 motion)
{
    return float3(0.5 + motion.x * 0.5, 0.5 + motion.y * 0.5, 0.5);
}

// Dispatch entry point used by the Python example. It selects one DLSS-RR input
// or output texture and writes a displayable RGBA image for the editor.
public void visualize_dlss_rr_view(
    uint2 pixel,
    int view_index,
    Texture2D<float4> color,
    Texture2D<float4> dlss_output,
    Texture2D<float4> diffuse_albedo,
    Texture2D<float4> specular_albedo,
    Texture2D<float4> normals,
    Texture2D<float> roughness,
    Texture2D<float> depth,
    Texture2D<float> specular_hit_distance,
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
        value = display_color(load_rgba(color, pixel, output_dims).rgb);
    } else if (view_index == 1) {
        value = display_color(load_rgba(dlss_output, pixel, output_dims).rgb);
    } else if (view_index == 2) {
        value = saturate(load_rgba(diffuse_albedo, pixel, output_dims).rgb);
    } else if (view_index == 3) {
        value = saturate(load_rgba(specular_albedo, pixel, output_dims).rgb);
    } else if (view_index == 4) {
        value = normalize(load_rgba(normals, pixel, output_dims).rgb) * 0.5 + 0.5;
    } else if (view_index == 5) {
        float r = saturate(load_r(roughness, pixel, output_dims));
        value = float3(r, r, r);
    } else if (view_index == 6) {
        value = display_depth(load_r(depth, pixel, output_dims));
    } else if (view_index == 7) {
        value = display_motion(load_rg(motion_vectors, pixel, output_dims));
    } else if (view_index == 8) {
        value = display_depth(load_r(specular_hit_distance, pixel, output_dims));
    }

    output[pixel] = float4(saturate(value), 1.0);
}
