// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "color.h"

#include <algorithm>

namespace falcor {

namespace {

float luminance(float3 rgb)
{
    return 0.2126f * rgb.x + 0.7152f * rgb.y + 0.0722f * rgb.z;
}

float3 blackbody_temperature_to_rgb(float color_temperature)
{
    // Approximate the Planckian locus in CIE xy using the polynomials from Kang et al. (2002).
    const float t = 1000.f / color_temperature;
    const float t2 = t * t;
    const float t3 = t2 * t;

    const float x = color_temperature < 4000.f ? -0.2661239f * t3 - 0.2343580f * t2 + 0.8776956f * t + 0.179910f
                                               : -3.0258469f * t3 + 2.1070379f * t2 + 0.2226347f * t + 0.240390f;
    const float x2 = x * x;
    const float x3 = x2 * x;

    float y;
    if (color_temperature < 2222.f)
        y = -1.1063814f * x3 - 1.3481102f * x2 + 2.1855583f * x - 0.20219683f;
    else if (color_temperature < 4000.f)
        y = -0.9549476f * x3 - 1.3741859f * x2 + 2.0913702f * x - 0.16748867f;
    else
        y = 3.0817580f * x3 - 5.8733867f * x2 + 3.7511300f * x - 0.37001483f;

    const float3 xyz(x / y, 1.f, (1.f - x - y) / y);
    return float3(
        std::max(3.2406f * xyz.x - 1.5372f * xyz.y - 0.4986f * xyz.z, 0.f),
        std::max(-0.9689f * xyz.x + 1.8758f * xyz.y + 0.0415f * xyz.z, 0.f),
        std::max(0.0557f * xyz.x - 0.2040f * xyz.y + 1.0570f * xyz.z, 0.f)
    );
}

} // namespace

float3 color_temperature_to_rgb(float color_temperature)
{
    static const float3 white = blackbody_temperature_to_rgb(6500.f);
    float3 rgb
        = blackbody_temperature_to_rgb(std::clamp(color_temperature, MIN_COLOR_TEMPERATURE, MAX_COLOR_TEMPERATURE));
    rgb /= white;
    return rgb / luminance(rgb);
}

} // namespace falcor
