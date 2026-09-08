// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module accumulator_common;

import falcor2.utils;

public float3 sanitize_accumulation_sample(float3 value, bool sanitize_non_finite)
{
    return sanitize_non_finite && !all(isfinite(value)) ? 0.f : value;
}

public float3 clamp_accumulation_sample(float3 value, float max_sample_luminance)
{
    if (!isfinite(max_sample_luminance))
        return value;

    // Normalize before computing luminance so finite values near the
    // floating-point limit cannot overflow the luminance dot product.
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (!(scale > 0.f) || !isfinite(scale))
        return value;

    float3 normalized_value = value / scale;
    float normalized_luminance = safe_luminance(normalized_value);
    if (normalized_luminance > max_sample_luminance / scale)
        return normalized_value * (max_sample_luminance / normalized_luminance);
    return value;
}

public float3 prepare_accumulation_sample(float3 value, bool sanitize_non_finite, float max_sample_luminance)
{
    value = sanitize_accumulation_sample(value, sanitize_non_finite);
    return clamp_accumulation_sample(value, max_sample_luminance);
}
