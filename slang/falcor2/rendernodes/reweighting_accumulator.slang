// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

module reweighting_accumulator;

import slangpy;
import falcor2.utils;
import falcor2.rendernodes.accumulator_common;

// Implements the practical cascaded reweighting method from:
// "Reweighting Firefly Samples for Improved Finite-Sample Monte Carlo Estimates"
// Tobias Zirr, Johannes Hanika, and Carsten Dachsbacher, 2018.
// https://cg.ivd.kit.edu/publications/2018/rwmc/reweighting-fireflies-preprint.pdf
//
// Minor tweaks to blend in over a few samples, so we still have something to see
// at tiny sample counts.

static const float BIN_BASE = 8.f;
static const float INVERSE_BIN_BASE = 1.f / BIN_BASE;
static const float SPLIT_DENOMINATOR = 1.f - INVERSE_BIN_BASE;

float bin_anchor(uint bin_index)
{
    // bins are spaced by 3 stops, so the anchor is 2^(3*bin_index),
    // eg 1, 8, 64, 512, 4096, ...
    return exp2(3.f * float(bin_index));
}

float luminance_over_anchor(float3 value, float anchor)
{
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    if (!(scale > 0.f) || !isfinite(scale))
        return 0.f;

    float normalized_luminance = safe_luminance(value / scale);
    return (scale / anchor) * normalized_luminance;
}

uint history_index(uint2 pixel, uint2 image_shape, uint bin_index, uint bin_count)
{
    return (pixel.x * image_shape.y + pixel.y) * (bin_count + 1) + bin_index;
}

float3 load_bin(StructuredBuffer<float4> history, uint2 pixel, uint2 image_shape, uint bin_index, uint bin_count)
{
    return history[history_index(pixel, image_shape, bin_index, bin_count)].xyz;
}

// Recovers fractional sample count in a bin by dividing the bin's luminance by the anchor luminance for that bin.
// Works because luminance is linear, so sum(luminance/anchor) = sum(luminance)/anchor = count
float normalized_bin_count_at(
    StructuredBuffer<float4> history,
    uint2 pixel,
    uint2 image_shape,
    int bin_index,
    uint bin_count
)
{
    if (bin_index < 0 || bin_index >= int(bin_count))
        return 0.f;

    uint index = uint(bin_index);
    float3 value = load_bin(history, pixel, image_shape, index, bin_count);
    return luminance_over_anchor(value, bin_anchor(index));
}

float soft_count_at(StructuredBuffer<float4> history, uint2 pixel, uint2 image_shape, uint bin_index, uint bin_count)
{
    int index = int(bin_index);
    return normalized_bin_count_at(history, pixel, image_shape, index - 1, bin_count)
        + normalized_bin_count_at(history, pixel, image_shape, index, bin_count)
        + normalized_bin_count_at(history, pixel, image_shape, index + 1, bin_count);
}

float neighborhood_soft_count(
    StructuredBuffer<float4> history,
    uint2 pixel,
    uint2 image_shape,
    uint bin_index,
    uint bin_count
)
{
    float result = 0.f;
    uint neighbor_count = 0;
    int2 center = int2(pixel);
    int2 shape = int2(image_shape);

    for (int y_offset = -1; y_offset <= 1; ++y_offset) {
        for (int x_offset = -1; x_offset <= 1; ++x_offset) {
            int2 neighbor = center + int2(y_offset, x_offset);
            if (all(neighbor >= 0) && all(neighbor < shape)) {
                result += soft_count_at(history, uint2(neighbor), image_shape, bin_index, bin_count);
                neighbor_count += 1;
            }
        }
    }

    return result / float(neighbor_count);
}

void add_to_bin(RWTensor<float4, 1> history, uint bin_index, float3 value)
{
    float3 sum = history.load(bin_index).xyz + value;
    history.store(bin_index, float4(sum, 0.f));
}

public void reweighting_accumulator_accumulate<T : __BuiltinFloatingPointType>(
    vector<T, 4> input,
    RWTensor<float4, 1> history,
    AtomicTensor<uint, 1> overflow_count,
    uniform uint bin_count,
    uniform bool sanitize_non_finite
)
{
    // To avoid overflow, calculate normalized luminance and scale separately
    float3 value = sanitize_accumulation_sample(float3(input.xyz), sanitize_non_finite);
    float scale = max(abs(value.x), max(abs(value.y), abs(value.z)));
    float normalized_luminance = scale > 0.f ? safe_luminance(value / scale) : 0.f;

    if (!(normalized_luminance > 0.f)) {
        add_to_bin(history, 0, value);
        return;
    }

    //    L =  scale * normalized_luminance
    //=>  log2(L) = log2(scale) + log2(normalized_luminance)
    float log_luminance = log2(scale) + log2(normalized_luminance);
    if (log_luminance < 0.f) {
        add_to_bin(history, 0, value);
        return;
    }

    uint lower_bin = uint(floor(log_luminance / 3.f));
    if (lower_bin >= bin_count - 1) {
        add_to_bin(history, bin_count, value);
        overflow_count.add(0, 1);
        return;
    }

    float anchor = bin_anchor(lower_bin);
    float anchor_over_luminance = (anchor / scale) / normalized_luminance;
    float alpha = saturate((anchor_over_luminance - INVERSE_BIN_BASE) / SPLIT_DENOMINATOR);
    add_to_bin(history, lower_bin, alpha * value);
    add_to_bin(history, lower_bin + 1, (1.f - alpha) * value);
}

public void reweighting_accumulator_resolve<T : __BuiltinFloatingPointType>(
    uint pixel[2],
    StructuredBuffer<float4> history,
    uniform uint2 image_shape,
    uniform uint bin_count,
    uniform uint sample_count,
    uniform float kappa,
    uniform float kappa_min,
    uniform uint output_mode,
    out vector<T, 4> output
)
{
    uint2 pixel_index = uint2(pixel[0], pixel[1]);
    float3 result = 0.f;

    if (output_mode == 0) {
        for (uint bin_index = 0; bin_index <= bin_count; ++bin_index)
            result += load_bin(history, pixel_index, image_shape, bin_index, bin_count);
    } else {
        // Overflow cannot be statistically reweighted, but it remains part of
        // the ordinary mean used during the initial presentation transition.
        float3 unweighted_result = load_bin(history, pixel_index, image_shape, bin_count, bin_count);

        for (uint bin_index = 0; bin_index < bin_count; ++bin_index) {
            float count = soft_count_at(history, pixel_index, image_shape, bin_index, bin_count);
            float average_count = neighborhood_soft_count(history, pixel_index, image_shape, bin_index, bin_count);

            float weight = 0.f;
            if (average_count > kappa_min) {

                // Weight based on commonality
                // the more often this bin has been used, the less likely it is to be an outlier
                float count_weight = saturate((count - kappa_min) / kappa);

                // Weight based on bin energy relative to total energy summed up so far
                // i.e. if lower energy bins have already contributed a
                // lot, we can be more lenient with higher energy bins
                float value_weight = saturate(luminance_over_anchor(result, bin_anchor(bin_index)) / kappa);

                weight = max(count_weight, value_weight);
            }

            // track both weighted and unweighted result so we can blend at the end.
            float3 unweighted_bin = load_bin(history, pixel_index, image_shape, bin_index, bin_count);
            unweighted_result += unweighted_bin;
            result += unweighted_bin * weight;
        }

        // Fade from the unweighted result over the first 16 samples so an
        // interactive view remains visible immediately after accumulation resets.
        float transition = saturate((sample_count - 1) / 15.0f);
        result = lerp(unweighted_result, result, transition);
    }

    output = vector<T, 4>(float4(result / float(sample_count), 1.f));
}
