// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

void mx_layer_bsdf(ClosureData closure_data, BSDF top, BSDF base, out BSDF result)
{
    result.set_layer(mx_stack_next_layer++);
    mx_stack.layer_weights[result.get_index()] = float3(1.0);
    mx_stack.mix_values[result.get_index()] = float3(0.0);
}

void mx_weighted_layer_bsdf(ClosureData closure_data, BSDF top, float3 top_weight, BSDF base, out BSDF result)
{
    result.set_layer(mx_stack_next_layer++);
    mx_stack.layer_weights[result.get_index()] = float3(1.0);
    mx_stack.mix_values[result.get_index()] = float3(0.0);

    float3 top_coverage = saturate(top_weight);
    if (top.is_bsdf())
        mx_stack.bsdf_weights[top.get_index()] *= top_coverage;
    else
        mx_stack.layer_weights[top.get_index()] *= top_coverage;
}

void mx_mix_bsdf(ClosureData closure_data, BSDF fg, BSDF bg, float mix_value, out BSDF result)
{
    mx_mix_bsdf(closure_data, fg, bg, float3(mix_value), result);
}

void mx_mix_bsdf(ClosureData closure_data, BSDF fg, BSDF bg, float3 mix_value, out BSDF result)
{
    result.set_layer(mx_stack_next_layer++);
    mx_stack.layer_weights[result.get_index()] = float3(1.0);
    mx_stack.mix_values[result.get_index()] = saturate(mix_value);
}

void mx_add_bsdf(ClosureData closure_data, BSDF in1, BSDF in2, out BSDF result)
{
    result.set_layer(mx_stack_next_layer++);
    mx_stack.layer_weights[result.get_index()] = float3(1.0);
    mx_stack.mix_values[result.get_index()] = float3(0.0);
}

void mx_multiply_bsdf(ClosureData closure_data, BSDF in1, float3 in2, out BSDF result)
{
    if (in1.is_bsdf())
        mx_stack.bsdf_weights[in1.get_index()] *= in2;
    else
        mx_stack.layer_weights[in1.get_index()] *= in2;
    result = in1;
}

void mx_multiply_bsdf(ClosureData closure_data, BSDF in1, float in2, out BSDF result)
{
    mx_multiply_bsdf(closure_data, in1, float3(in2), result);
}

void mx_uniform_edf(ClosureData closure_data, float3 color, out EDF result)
{
    result.radiance = color;
}

void mx_multiply_edf(ClosureData closure_data, EDF in1, float3 in2, out EDF result)
{
    result.radiance = in1.radiance * in2;
}

void mx_multiply_edf(ClosureData closure_data, EDF in1, float in2, out EDF result)
{
    result.radiance = in1.radiance * in2;
}

void mx_add_edf(ClosureData closure_data, EDF in1, EDF in2, out EDF result)
{
    result.radiance = in1.radiance + in2.radiance;
}

void mx_mix_edf(ClosureData closure_data, EDF fg, EDF bg, float mix_value, out EDF result)
{
    result.radiance = lerp(bg.radiance, fg.radiance, clamp(mix_value, 0.0, 1.0));
}

void mx_mix_surfaceshader(surfaceshader fg, surfaceshader bg, float mix_value, out surfaceshader result)
{
    float clamped_mix = clamp(mix_value, 0.0, 1.0);
    result.root.set_layer(mx_stack_next_layer++);
    mx_stack.layer_weights[result.root.get_index()] = float3(1.0);
    mx_stack.mix_values[result.root.get_index()] = float3(clamped_mix);
    result.opacity = lerp(bg.opacity, fg.opacity, clamped_mix);
    result.edf.radiance = lerp(bg.edf.radiance, fg.edf.radiance, clamped_mix);
    result.thin_walled = fg.thin_walled || bg.thin_walled;
}

void mx_generalized_schlick_edf(
    ClosureData closure_data,
    float3 color0,
    float3 color90,
    float exponent,
    EDF base,
    out EDF result
)
{
    // MaterialX generalized_schlick_edf is directional: MaterialX 1.39 GLSL tints the base
    // EDF by a generalized Schlick factor using the view-normal angle.
    float3 normal
        = dot(si.shading_frame_ws.normal, si.wi_ws) < 0.0 ? -si.shading_frame_ws.normal : si.shading_frame_ws.normal;
    float n_dot_v = clamp(dot(normal, si.wi_ws), 1.0e-6, 1.0);
    float x = clamp(1.0 - n_dot_v, 0.0, 1.0);
    result.radiance = base.radiance * lerp(color0, color90, pow(x, exponent));
}

void mx_blackbody(float temperature_kelvin, out float3 color_value)
{
    float xc;
    float yc;

    temperature_kelvin = clamp(temperature_kelvin, 800.0, 25000.0);
    float t = 1000.0 / temperature_kelvin;
    float t2 = t * t;
    float t3 = t2 * t;

    if (temperature_kelvin < 4000.0)
        xc = -0.2661239 * t3 - 0.2343580 * t2 + 0.8776956 * t + 0.179910;
    else
        xc = -3.0258469 * t3 + 2.1070379 * t2 + 0.2226347 * t + 0.240390;

    float xc2 = xc * xc;
    float xc3 = xc2 * xc;

    if (temperature_kelvin < 2222.0)
        yc = -1.1063814 * xc3 - 1.34811020 * xc2 + 2.18555832 * xc - 0.20219683;
    else if (temperature_kelvin < 4000.0)
        yc = -0.9549476 * xc3 - 1.37418593 * xc2 + 2.09137015 * xc - 0.16748867;
    else
        yc = 3.0817580 * xc3 - 5.87338670 * xc2 + 3.75112997 * xc - 0.37001483;

    if (yc <= 0.0) {
        color_value = float3(1.0);
        return;
    }

    float3 xyz = float3(xc / yc, 1.0, (1.0 - xc - yc) / yc);
    float3x3 xyz_to_rgb = float3x3(3.2406, -0.9689, 0.0557, -1.5372, 1.8758, -0.2040, -0.4986, 0.0415, 1.0570);
    color_value = max(mx_matrix_mul(xyz_to_rgb, xyz), float3(0.0));
}

void mx_surface(BSDF bsdf, EDF edf, float opacity, bool thin_walled, out surfaceshader result)
{
    result.root = bsdf;
    result.opacity = clamp(opacity, 0.0, 1.0);
    result.edf.radiance = edf.radiance * opacity;
    result.thin_walled = thin_walled;
}
