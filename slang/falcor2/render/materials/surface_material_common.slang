// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

namespace surface_material {
// Keep existential accessor dispatch outside differentiated material code.
[TreatAsDifferentiable]
bool try_get_additional_texcoord_no_diff(
    no_diff const SurfaceInteraction si,
    const uint texture_coord_index,
    out float2 texture_coord
)
{
    const IGeomPropAccessor accessor = si.get_geomprop_accessor();
    return accessor.try_get_additional_texcoord(texture_coord_index, texture_coord);
}

// Keep existential accessor dispatch outside differentiated material code.
[TreatAsDifferentiable]
bool try_get_color_no_diff(no_diff const SurfaceInteraction si, out float4 color)
{
    const IGeomPropAccessor accessor = si.get_geomprop_accessor();
    return accessor.try_get_color(0, color);
}

[Differentiable]
float2 resolve_texcoord(const SurfaceInteraction si, const uint texture_coord_index)
{
    if (texture_coord_index == 0)
        return si.uv;

    float2 texture_coord;
    if (try_get_additional_texcoord_no_diff(si, texture_coord_index, texture_coord))
        return texture_coord;
    return si.uv;
}

[Differentiable]
float4 resolve_vertex_color(const SurfaceInteraction si)
{
    float4 color;
    if (try_get_color_no_diff(si, color))
        return color;
    return float4(1.0);
}

bool is_normal_mapping_enabled(const MaterialInstanceHints hints)
{
    // TODO: Use is_set() once https://github.com/shader-slang/slang/issues/12926 is fixed.
    return (hints & MaterialInstanceHints::disable_normal_mapping) == MaterialInstanceHints::none;
}

// Encoded maps store XY and reconstruct positive Z; ordinary maps store RGB.
public float3 decode_normal_map(const float3 encoded, bool normal_map_encoded)
{
    float3 normal = encoded * 2.0 - 1.0;
    if (normal_map_encoded)
        normal.z = sqrt(saturate(1.0 - dot(normal.xy, normal.xy)));
    return normalize(normal);
}

Frame orient_shading_frame(const SurfaceInteraction si, const Frame shading_frame_ws, const bool orient_backface)
{
    Frame result = shading_frame_ws;
    if (!si.front_facing && orient_backface)
        result.normal = -result.normal;
    return result;
}

Frame adjust_shading_frame(const SurfaceInteraction si, Frame shading_frame_ws, const bool orient_backface)
{
    const float3 geometric_normal = !si.front_facing && orient_backface ? -si.normal_ws : si.normal_ws;
    const float cos_theta = dot(shading_frame_ws.normal, geometric_normal);
    if (cos_theta < 0.0) {
        // A valid tangent-space normal can cross the triangle horizon when applied
        // to interpolated vertex normals. Keep its tangential direction, but put it
        // on the material's intended side before the side-preserving reflection repair.
        const float3 normal
            = normalize(shading_frame_ws.normal - cos_theta * geometric_normal + 0.001 * geometric_normal);
        shading_frame_ws = Frame(normal, shading_frame_ws.tangent, shading_frame_ws.handedness_sign());
    }
    adjust_shading_normal(shading_frame_ws, si.front_normal_ws(), si.wi_ws);
    return shading_frame_ws;
}

bool is_valid_hemisphere_reflection_or_transmission(
    const SurfaceInteraction si,
    const Frame shading_frame_ws,
    const float3 wi_ls,
    const float3 wo_ls,
    const float3 wo_ws
)
{
    const bool is_transmission = wi_ls.z * wo_ls.z < 0.0;

    if (min(wi_ls.z, abs(wo_ls.z)) < BSDF_MIN_COS_THETA)
        return false;

    const bool wi_top = si.front_facing;
    const bool wo_top = dot(wo_ws, si.normal_ws) >= 0.0;

    if (is_transmission) {
        if (wi_top == wo_top)
            return false;
    } else {
        if (wi_top != wo_top)
            return false;
    }

    const bool shading_top = dot(shading_frame_ws.normal, si.normal_ws) >= 0.0;
    if (wi_top != shading_top)
        return false;

    return true;
}
}
