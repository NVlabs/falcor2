// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing render;

public struct ReconstructionFilterSample
{
    /// Displacement from the filter center in input pixels, excluding shared camera jitter.
    public float2 offset;
    /// Filter value divided by sampling density; one for Box (including width zero).
    public float weight;
}

/// A normalized, nonnegative filter centered at the origin, independent of camera jitter.
public interface IReconstructionFilter
{
    /// Map uniform values in [0, 1)^2 to an offset without pixel-boundary clamping.
    /// The caller consumes two random values even for a delta filter.
    ReconstructionFilterSample sample(float2 u);
    /// Evaluate the kernel for splatting approaches such as Mitsuba's reconstruction.
    /// Returns the normalized value per square input pixel; delta filters evaluate to zero.
    float eval(float2 offset);
}

public struct BoxReconstructionFilter : IReconstructionFilter
{
    public float width;

    float density()
    {
        if (width == 0.0)
            return 0.0;
        // Square the reciprocal to avoid underflow of width^2 for narrow boxes.
        float inverse_width = 1.0 / width;
        return inverse_width * inverse_width;
    }

    public ReconstructionFilterSample sample(float2 u)
    {
        float2 filter_offset = width * (u - 0.5);
        return { filter_offset, 1.0 };
    }

    public float eval(float2 offset) { return all(abs(offset) <= 0.5 * width) ? density() : 0.0; }
}

float one_minus_exp_neg(float x)
{
    return x < 0.001 ? x * (1.0 - x * (0.5 - x / 6.0)) : 1.0 - exp(-x);
}

// Boundary-subtracted, separable shape used by PBRT and Mitsuba, normalized here
// to unit integral. PBRT tabulates its sampler; Mitsuba evaluates for splatting.
// We sample an unshifted truncated-Gaussian proposal and compensate with value/pdf.
// Set radius=4*sigma for Mitsuba's cutoff, or sigma=.5/radius=1.5 for PBRT defaults.
public struct GaussianReconstructionFilter : IReconstructionFilter
{
    public float sigma;
    public float radius;
    // Prepared by the native filter once per binding, rather than recomputed per ray.
    public float cdf_limit;     // erf(radius / (sqrt(2) * sigma)).
    public float peak;          // 1 - exp(-radius^2 / (2 * sigma^2)).
    public float normalization; // Reciprocal integral of the peak-scaled 1D kernel.

    // Ratio of shifted to unshifted kernels, scaled so it is one at the center.
    float edge_factor(float x)
    {
        float y = abs(x) / radius;
        if (y >= 1.0)
            return 0.0;
        float normalized_radius = radius / sigma;
        float a = 0.5 * normalized_radius * normalized_radius;
        return one_minus_exp_neg(a * (1.0 - y) * (1.0 + y)) / peak;
    }

    public float eval(float2 offset)
    {
        if (any(abs(offset) > radius))
            return 0.0;
        float2 t = offset / sigma;
        float gaussian = exp(-0.5 * dot(t, t));
        return normalization * normalization * gaussian * edge_factor(offset.x) * edge_factor(offset.y);
    }

    public ReconstructionFilterSample sample(float2 u)
    {
        // Keep inverse erf finite when the cutoff CDF rounds to one in FP32.
        float2 v = clamp((2.0 * u - 1.0) * cdf_limit, -0.9999999403953552, 0.9999999403953552);
        float2 offset = 1.4142135623730951 * sigma * float2(erfinv(v.x), erfinv(v.y));
        // The inverse-erf approximation may overshoot the support by a few ULPs.
        offset = clamp(offset, -radius, radius);
        // Cancel the Gaussian factors analytically so tiny tail PDFs cannot cause 0/0.
        float weight_scale = normalization * (2.5066282746310005 * sigma * cdf_limit);
        float weight = weight_scale * weight_scale * edge_factor(offset.x) * edge_factor(offset.y);
        return { offset, weight };
    }
}
