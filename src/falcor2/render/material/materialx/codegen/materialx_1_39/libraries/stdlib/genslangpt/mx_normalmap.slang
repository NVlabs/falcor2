// SPDX-License-Identifier: Apache-2.0

bool mx_normalmap_is_finite(float v)
{
    return v == v && abs(v) < 1.0e30;
}

bool mx_normalmap_is_finite(float3 v)
{
    return mx_normalmap_is_finite(v.x) && mx_normalmap_is_finite(v.y) && mx_normalmap_is_finite(v.z);
}

bool mx_normalmap_is_usable(float3 v)
{
    return mx_normalmap_is_finite(v) && dot(v, v) >= 1.0e-8;
}

void mx_normalmap_vector2(
    float3 value,
    float2 normal_scale,
    float3 normal,
    float3 tangent,
    float3 bitangent,
    out float3 result
)
{
    if (dot(value, value) == 0.0) {
        result = normal;
        return;
    }

    // Match the 1.38 genslang path for parity. It builds the bitangent from
    // the supplied normal and tangent with B = normalize(cross(N, T)), then
    // applies the tangent-space normal directly. Keep only local safety guards
    // for degenerate generated/default inputs.
    float3 normal_n = mx_normalmap_is_usable(normal) ? normalize(normal) : float3(0.0, 0.0, 1.0);
    float3 tangent_n = mx_normalmap_is_usable(tangent) ? tangent : Frame(normal_n).tangent;
    float3 bitangent_n = normalize(cross(normal_n, tangent_n));
    if (!mx_normalmap_is_usable(bitangent_n))
        bitangent_n = Frame(normal_n, normalize(tangent_n)).bitangent;

    float3 v = select(value == float3(0.0), float3(0.0, 0.0, 1.0), value * 2.0 - 1.0);
    v.xy *= normal_scale;
    result = normalize(tangent_n * v.x + bitangent_n * v.y + normal_n * v.z);
    if (!mx_normalmap_is_usable(result))
        result = normal_n;
}

void mx_normalmap_float(
    float3 value,
    float normal_scale,
    float3 normal,
    float3 tangent,
    float3 bitangent,
    out float3 result
)
{
    mx_normalmap_vector2(value, float2(normal_scale), normal, tangent, bitangent, result);
}
