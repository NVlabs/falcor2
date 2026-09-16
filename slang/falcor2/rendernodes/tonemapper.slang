// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module tonemapper;

import slangpy;
import falcor2.utils;

static const uint HISTOGRAM_BIN_COUNT = 64;

public interface ITonemappingOperator
{
    public static float3 apply(float3 color);
}

public struct AcesFilmOperator : ITonemappingOperator
{
    public static override float3 apply(float3 color) { return tonemap_aces_film(color); }
};

public struct ReinhardOperator : ITonemappingOperator
{
    public static override float3 apply(float3 color) { return tonemap_reinhard(color); }
};

public struct ClampOperator : ITonemappingOperator
{
    public static override float3 apply(float3 color) { return tonemap_clamp(color); }
};

public float3 tonemapper_apply<TOperator : ITonemappingOperator>(float3 color, uniform float exposure_ev)
{
    return TOperator.apply(exp2(exposure_ev) * color);
}

public float4 tonemapper_apply<TOperator : ITonemappingOperator>(float4 color, uniform float exposure_ev)
{
    return float4(tonemapper_apply<TOperator>(color.xyz, exposure_ev), color.w);
}

public half4 tonemapper_apply<TOperator : ITonemappingOperator>(half4 color, uniform float exposure_ev)
{
    return half4(half3(tonemapper_apply<TOperator>(float3(color.xyz), exposure_ev)), color.w);
}

public float3 tonemapper_apply_auto<TOperator : ITonemappingOperator>(
    float3 color,
    uniform StructuredBuffer<float> adapted_exposure_ev
)
{
    return tonemapper_apply<TOperator>(color, adapted_exposure_ev[0]);
}

public float4 tonemapper_apply_auto<TOperator : ITonemappingOperator>(
    float4 color,
    uniform StructuredBuffer<float> adapted_exposure_ev
)
{
    return float4(tonemapper_apply_auto<TOperator>(color.xyz, adapted_exposure_ev), color.w);
}

public half4 tonemapper_apply_auto<TOperator : ITonemappingOperator>(
    half4 color,
    uniform StructuredBuffer<float> adapted_exposure_ev
)
{
    return half4(half3(tonemapper_apply_auto<TOperator>(float3(color.xyz), adapted_exposure_ev)), color.w);
}

/// Add one pixel's luminance to the shared auto-exposure histogram.
///
/// Luminance is represented in log2 space because exposure is multiplicative.
/// Samples are split between adjacent bins using integer weights so portable
/// uint atomics can be used on every backend. Bin zero is ignored so black
/// backgrounds have no influence on exposure.
public void tonemapper_accumulate_luminance_histogram(
    float3 color,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max
)
{
    float luminance = safe_luminance(color);
    if (luminance <= 0.f)
        return;

    float histogram_position
        = saturate((log2(luminance) - histogram_log_min) / (histogram_log_max - histogram_log_min));
    float bucket_position = histogram_position * float(HISTOGRAM_BIN_COUNT - 1);
    uint bucket0 = min(uint(bucket_position), HISTOGRAM_BIN_COUNT - 1);
    uint bucket1 = min(bucket0 + 1, HISTOGRAM_BIN_COUNT - 1);
    uint weight1 = uint(frac(bucket_position) * float(histogram_weight_scale) + 0.5f);
    uint weight0 = histogram_weight_scale - weight1;

    if (bucket0 != 0 && weight0 != 0)
        InterlockedAdd(luminance_histogram[bucket0], weight0);
    if (bucket1 != 0 && weight1 != 0)
        InterlockedAdd(luminance_histogram[bucket1], weight1);
}

public void tonemapper_accumulate_luminance_histogram(
    float4 color,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max
)
{
    tonemapper_accumulate_luminance_histogram(
        color.xyz,
        luminance_histogram,
        histogram_weight_scale,
        histogram_log_min,
        histogram_log_max
    );
}

public void tonemapper_accumulate_luminance_histogram(
    half4 color,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max
)
{
    tonemapper_accumulate_luminance_histogram(
        float3(color.xyz),
        luminance_histogram,
        histogram_weight_scale,
        histogram_log_min,
        histogram_log_max
    );
}

public void tonemapper_apply_auto_histogram<TOperator : ITonemappingOperator>(
    float3 color,
    uniform StructuredBuffer<float> adapted_exposure_ev,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max,
    out float3 mapped_color
)
{
    mapped_color = tonemapper_apply_auto<TOperator>(color, adapted_exposure_ev);
    tonemapper_accumulate_luminance_histogram(
        color,
        luminance_histogram,
        histogram_weight_scale,
        histogram_log_min,
        histogram_log_max
    );
}

public void tonemapper_apply_auto_histogram<TOperator : ITonemappingOperator>(
    float4 color,
    uniform StructuredBuffer<float> adapted_exposure_ev,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max,
    out float4 mapped_color
)
{
    mapped_color = tonemapper_apply_auto<TOperator>(color, adapted_exposure_ev);
    tonemapper_accumulate_luminance_histogram(
        color,
        luminance_histogram,
        histogram_weight_scale,
        histogram_log_min,
        histogram_log_max
    );
}

public void tonemapper_apply_auto_histogram<TOperator : ITonemappingOperator>(
    half4 color,
    uniform StructuredBuffer<float> adapted_exposure_ev,
    uniform RWStructuredBuffer<uint> luminance_histogram,
    uniform uint histogram_weight_scale,
    uniform float histogram_log_min,
    uniform float histogram_log_max,
    out half4 mapped_color
)
{
    mapped_color = tonemapper_apply_auto<TOperator>(color, adapted_exposure_ev);
    tonemapper_accumulate_luminance_histogram(
        color,
        luminance_histogram,
        histogram_weight_scale,
        histogram_log_min,
        histogram_log_max
    );
}

/// Update the one-element exposure history buffer from a luminance histogram.
public void tonemapper_update_auto_exposure(
    uniform StructuredBuffer<uint> luminance_histogram,
    uniform RWStructuredBuffer<float> adapted_exposure_ev,
    uniform float histogram_log_min,
    uniform float histogram_log_max,
    uniform float histogram_low_percent,
    uniform float histogram_high_percent,
    uniform float exposure_key,
    uniform float exposure_compensation,
    uniform float min_exposure_ev,
    uniform float max_exposure_ev,
    uniform float adaptation_speed_up,
    uniform float adaptation_speed_down,
    uniform float delta_time,
    uniform bool apply_immediately
)
{
    uint total_weight = 0;
    for (uint i = 0; i < HISTOGRAM_BIN_COUNT; ++i)
        total_weight += luminance_histogram[i];

    if (total_weight == 0) {
        if (apply_immediately)
            adapted_exposure_ev[0] = clamp(exposure_compensation, min_exposure_ev, max_exposure_ev);
        return;
    }

    float min_fraction_sum = float(total_weight) * histogram_low_percent * 0.01f;
    float max_fraction_sum = float(total_weight) * histogram_high_percent * 0.01f;
    float weighted_log_luminance = 0.f;
    float retained_weight = 0.f;

    for (uint i = 0; i < HISTOGRAM_BIN_COUNT; ++i) {
        float local_weight = float(luminance_histogram[i]);
        float discarded_weight = min(local_weight, min_fraction_sum);
        local_weight -= discarded_weight;
        min_fraction_sum -= discarded_weight;
        max_fraction_sum -= discarded_weight;

        local_weight = min(local_weight, max(max_fraction_sum, 0.f));
        max_fraction_sum -= local_weight;

        float histogram_position = float(i) / float(HISTOGRAM_BIN_COUNT - 1);
        float bucket_log_luminance = lerp(histogram_log_min, histogram_log_max, histogram_position);
        weighted_log_luminance += bucket_log_luminance * local_weight;
        retained_weight += local_weight;
    }

    if (retained_weight <= 0.f) {
        if (apply_immediately)
            adapted_exposure_ev[0] = clamp(exposure_compensation, min_exposure_ev, max_exposure_ev);
        return;
    }

    float mean_log_luminance = weighted_log_luminance / retained_weight;
    float target_ev
        = clamp(log2(exposure_key) - mean_log_luminance + exposure_compensation, min_exposure_ev, max_exposure_ev);

    if (apply_immediately) {
        adapted_exposure_ev[0] = target_ev;
        return;
    }

    float delta_ev = target_ev - adapted_exposure_ev[0];
    delta_ev = clamp(delta_ev, -adaptation_speed_down * delta_time, adaptation_speed_up * delta_time);
    adapted_exposure_ev[0] = clamp(adapted_exposure_ev[0] + delta_ev, min_exposure_ev, max_exposure_ev);
}
