// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import falcor2.render;

[shader("compute")]
[numthreads(8, 8, 1)]
public void selection_overlay(
    uint3 tid: SV_DispatchThreadID,
    RWTexture2D<float4> output_texture,
    RWTexture2D<uint> geometry_instance_id_texture,
    RWTexture2D<uint> selected_hit_texture,
    StructuredBuffer<uint> selection_bitmap,
    uniform uint selection_bitmap_bit_count,
    uniform float3 selection_color,
    uniform bool show_occluded,
    uniform float fill_opacity
)
{
    uint2 dim;
    output_texture.GetDimensions(dim.x, dim.y);

    uint2 pixel = tid.xy;
    if (any(pixel >= dim))
        return;

    // Check if center pixel is selected.
    uint center_id = GeometryInstanceID(geometry_instance_id_texture[pixel]);
    bool center_selected
        = center_id < selection_bitmap_bit_count && (selection_bitmap[center_id / 32] & (1u << (center_id % 32))) != 0;
    if (show_occluded)
        center_selected |= selected_hit_texture[pixel] != 0;

    uint count = 0;
    uint active = 0;
    for (int x = pixel.x - 2; x <= pixel.x + 2; ++x) {
        for (int y = pixel.y - 2; y <= pixel.y + 2; ++y) {
            if (x < 0 || y < 0 || x >= dim.x || y >= dim.y)
                continue;
            count += 1;
            uint2 p = uint2(x, y);
            uint id = GeometryInstanceID(geometry_instance_id_texture[p]);
            bool is_selected_visible
                = id < selection_bitmap_bit_count && (selection_bitmap[id / 32] & (1u << (id % 32))) != 0;
            if (show_occluded)
                is_selected_visible |= selected_hit_texture[p] != 0;
            if (is_selected_visible)
                active++;
        }
    }

    float ratio = float(active) / count;
    bool outline = ratio >= 0.25 && ratio <= 0.75;

    float4 color = output_texture[pixel];

    // Apply fill overlay for selected pixels.
    if (fill_opacity > 0.0 && center_selected)
        color.rgb = lerp(color.rgb, selection_color, fill_opacity);

    // Draw outlines on top of fill.
    if (outline)
        color.rgb += selection_color;

    output_texture[pixel] = color;
}
