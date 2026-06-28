// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Returns the smallest index i such that buffer[i] >= x, clamped to [0, count-1].
internal uint binary_search(StructuredBuffer<float> buffer, uint count, float x)
{
    int left = 0;
    int right = count;
    while (left < right) {
        int mid = (left + right) >> 1;
        if (x < buffer[mid])
            right = mid;
        else
            left = mid + 1;
    }
    return min(left, count - 1);
}

/// Discrete 1D probability distribution for importance sampling.
///
/// Supports sampling an index with probability proportional to the original weights.
/// Uses inverse transform sampling with binary search on the CDF.
public struct DiscreteDistribution1D
{
    /// Number of entries.
    public uint size;

    /// TODO(scene) replace buffers with bindless buffer handles.

    /// Probability Density Function.
    public StructuredBuffer<float> pdf_buffer;
    /// Cumulative Distribution Function.
    public StructuredBuffer<float> cdf_buffer;

    /// Evaluate the probability density function (PDF).
    /// @param index Index in [0, size-1].
    public float eval_pdf(uint index) { return pdf_buffer[index]; }

    /// Evaluate the cumulative distribution function (CDF).
    /// @param index Index in [0, size-1].
    public float eval_cdf(uint index) { return cdf_buffer[index]; }

    /// Sample an index from the distribution.
    /// @param u Uniform random number in [0, 1).
    /// @param[out] index Sampled index in [0, size-1].
    /// @param[out] pdf Probability of the sampled index.
    /// @return True if the distribution is non-empty.
    public bool sample(float u, out uint index, out float pdf)
    {
        index = binary_search(cdf_buffer, size, u);
        pdf = pdf_buffer[index];
        return size > 0;
    }

    /// Sample an index and reuse the random number.
    /// After sampling, u is remapped to [0, 1) within the selected bin,
    /// allowing it to be reused for further sampling decisions.
    /// @param[inout] u Uniform random number in [0, 1), remapped on output.
    /// @param[out] index Sampled index in [0, size-1].
    /// @param[out] pdf Probability of the sampled index.
    /// @return True if the distribution is non-empty.
    public bool sample_reuse(inout float u, out uint index, out float pdf)
    {
        index = binary_search(cdf_buffer, size, u);
        pdf = pdf_buffer[index];
        if (pdf > 0.0) {
            float cdf_prev = index == 0 ? 0.0 : cdf_buffer[index - 1];
            u = (u - cdf_prev) / pdf;
        }
        return size > 0;
    }
};

/// Discrete 1D probability distribution for importance sampling.
/// Uses the alias method for O(1) sampling after O(n) preprocessing.
public struct AliasTable1D
{
    /// Alias table entry.
    public struct Entry
    {
        /// Probability of the original index.
        float prob;
        /// Alias index for the remaining probability.
        uint alias;
    };

    /// Number of entries.
    public uint size;

    /// TODO(scene) replace buffers with bindless buffer handles.

    /// Probability Density Function.
    public StructuredBuffer<float> pdf_buffer;
    /// Alias table entries.
    public StructuredBuffer<Entry> alias_table_buffer;

    /// Evaluate the probability density function (PDF).
    /// @param index Index in [0, size-1].
    public float eval_pdf(uint index) { return pdf_buffer[index]; }

    /// Sample an index from the distribution.
    /// @param u Two independent uniform random numbers in [0, 1).
    /// @param[out] index Sampled index in [0, size-1].
    /// @param[out] pdf Probability of the sampled index.
    /// @return True if the distribution is non-empty.
    public bool sample(float2 u, out uint index, out float pdf)
    {
        index = min(uint(u.x * size), size - 1);
        Entry entry = alias_table_buffer[index];
        if (u.y >= entry.prob)
            index = entry.alias;
        pdf = pdf_buffer[index];
        return size > 0;
    }

    public bool sample(float u, out uint index, out float pdf)
    {
        u *= size;
        index = min(uint(u), size - 1);
        u = frac(u);
        Entry entry = alias_table_buffer[index];
        if (u >= entry.prob)
            index = entry.alias;
        pdf = pdf_buffer[index];
        return size > 0;
    }

    /// Sample an index and reuse the random number.
    /// After sampling, u is remapped to [0, 1) within the selected bin,
    /// allowing it to be reused for further sampling decisions.
    /// @param[inout] u Uniform random number in [0, 1), remapped on output.
    /// @param[out] index Sampled index in [0, size-1].
    /// @param[out] pdf Probability of the sampled index.
    /// @return True if the distribution is non-empty.
    public bool sample_reuse(inout float u, out uint index, out float pdf)
    {
        u *= size;
        index = min(uint(u), size - 1);
        u = frac(u);
        Entry entry = alias_table_buffer[index];
        if (u >= entry.prob) {
            index = entry.alias;
            u = (u - entry.prob) / (1.0 - entry.prob);
        } else {
            u /= entry.prob;
        }
        pdf = pdf_buffer[index];
        return size > 0;
    }
};
