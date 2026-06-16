// SPDX-License-Identifier: Apache-2.0

implementing render;

namespace stdmat {
bool is_normal_mapping_enabled(const MaterialInstanceHints hints)
{
    return !(hints & MaterialInstanceHints::disable_normal_mapping);
}

float get_metallic(const float3 diffuse, const float3 specular)
{
    float d = luminance(diffuse);
    float s = luminance(specular);
    if (s == 0.0)
        return 0.0;

    float b = s + d - 0.08;
    float c = 0.04 - s;
    float root = sqrt(max(b * b - 0.16 * c, 0.0));
    return saturate((root - b) * 12.5);
}

Frame orient_shading_frame(const SurfaceInteraction si, const Frame shading_frame_ws, const bool double_sided)
{
    Frame result = shading_frame_ws;
    if (!si.front_facing && double_sided)
        result.normal = -result.normal;
    return result;
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
