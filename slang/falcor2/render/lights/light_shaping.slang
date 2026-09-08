// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

/// Directional emission profile shared by shaped lights.
public struct LightShaping
{
    public bool enabled;
    public float cos_cutoff_angle;
    public float cos_falloff_angle;
    public float focus;

    float focus_factor(const float cos_theta)
    {
        const float exponent = max(focus, 0.0);
        return exponent > 0.0 ? pow(abs(cos_theta), exponent) : 1.0;
    }

    float cone_factor(const float cos_theta)
    {
        if (cos_theta < cos_cutoff_angle)
            return 0.0;
        if (cos_theta >= cos_falloff_angle)
            return 1.0;

        // USD specifies smoothstep interpolation in angular space.
        const float theta = acos(clamp(cos_theta, -1.0, 1.0));
        const float cutoff_cos = clamp(cos_cutoff_angle, -1.0, 1.0);
        const float falloff_cos = clamp(cos_falloff_angle, cutoff_cos, 1.0);
        const float cutoff_angle = acos(cutoff_cos);
        const float falloff_angle = acos(falloff_cos);
        const float angular_delta = saturate((cutoff_angle - theta) / (cutoff_angle - falloff_angle));
        return smoothstep(0.0, 1.0, angular_delta);
    }

    float eval_cos_theta(const float cos_theta)
    {
        if (!enabled)
            return 1.0;
        const float clamped_cos_theta = clamp(cos_theta, -1.0, 1.0);
        return cone_factor(clamped_cos_theta) * focus_factor(clamped_cos_theta);
    }

    /// Evaluates the angular profile without changing the emitting surface.
    public float eval(const float3 light_dir_ws, const float3 emitted_dir_ws)
    {
        return eval_cos_theta(dot(light_dir_ws, emitted_dir_ws));
    }

    float approximate_power_cos_cutoff()
    {
        // The exact power integral of USD's angular-space smoothstep combined with an arbitrary focus exponent has
        // no practical closed form. Power is used only to construct light-selection weights, so approximate the soft
        // transition by a hard cone at its angular midpoint. This is exact when softness is zero, preserves the USD
        // angle parameterization, and cannot bias lighting because directional emission still evaluates the exact
        // profile above and the sampler accounts for the resulting selection probability.
        const float cutoff_angle = acos(clamp(cos_cutoff_angle, -1.0, 1.0));
        const float falloff_angle = acos(clamp(cos_falloff_angle, -1.0, 1.0));
        return cos(0.5 * (cutoff_angle + falloff_angle));
    }

    float integrate_absolute_power(const float lower, const float upper)
    {
        const float exponent = max(focus, 0.0) + 1.0;
        if (upper <= 0.0)
            return (pow(-lower, exponent) - pow(-upper, exponent)) / exponent;
        if (lower >= 0.0)
            return (pow(upper, exponent) - pow(lower, exponent)) / exponent;
        return (pow(-lower, exponent) + pow(upper, exponent)) / exponent;
    }

    /// Estimates the profile integral over the sphere, normalized by 4 pi.
    public float full_sphere_power_factor()
    {
        if (!enabled)
            return 1.0;

        return saturate(0.5 * integrate_absolute_power(approximate_power_cos_cutoff(), 1.0));
    }

    /// Estimates the cosine-weighted profile integral over the emission hemisphere, normalized by pi.
    public float cosine_hemisphere_power_factor()
    {
        if (!enabled)
            return 1.0;

        const float effective_cutoff_cos = max(approximate_power_cos_cutoff(), 0.0);
        const float exponent = max(focus, 0.0) + 2.0;
        return saturate(2.0 * (1.0 - pow(effective_cutoff_cos, exponent)) / exponent);
    }
};
