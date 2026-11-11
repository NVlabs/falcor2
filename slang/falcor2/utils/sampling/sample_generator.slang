// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public interface ISampleGenerator
{
    /// Selects a point, starting at dimension zero. Keep scrambling_seed fixed across
    /// points of a QMC sequence. PRNGs combine both inputs to initialize their state.
    /// The 32-bit index wraps after 2^32 points, independently at each pixel.
    public __init(uint sample_index, uint scrambling_seed = 0);

    /// Initializes a pixel-local sequence. QMC keeps scrambling fixed across samples
    /// of the same pixel; PRNGs retain their existing pixel initialization streams.
    public __init(uint2 pixel, uint sample_index);

    /// Returns the next sample value.
    /// @return Sample value in [0,2^32).
    [mutating]
    public uint next_uint();
}

/// Stable pixel identity for QMC randomization; independent of the sample ordinal.
public uint get_pixel_scrambling_seed(uint2 pixel)
{
    return block_cipher_tea(pixel.x, pixel.y, 8).x;
}

/// Return the next sample value as a float in [0,1).
/// @param[inout] sg Sample generator.
/// @return A float in [0,1).
public float next_float<SG : ISampleGenerator>(inout SG sg)
{
    // Use upper 24 bits and divide by 2^24 to get a number u in [0,1).
    // In floating-point precision this also ensures that 1.0-u != 0.0.
    return (sg.next_uint() >> 8) * 0x1p-24;
}

/// Return the next 2 sample values as a float2 in [0,1).
/// @param[inout] sg Sample generator.
/// @return A float2 in [0,1).
public float2 next_float2<SG : ISampleGenerator>(inout SG sg)
{
    float2 sample;
    // Don't use the float2 initializer to ensure consistent order of evaluation.
    sample.x = next_float(sg);
    sample.y = next_float(sg);
    return sample;
}

/// Return the next 3 sample values as a float3 in [0,1).
/// @param[inout] sg Sample generator.
/// @return A float3 in [0,1).
public float3 next_float3<SG : ISampleGenerator>(inout SG sg)
{
    float3 sample;
    // Don't use the float3 initializer to ensure consistent order of evaluation.
    sample.x = next_float(sg);
    sample.y = next_float(sg);
    sample.z = next_float(sg);
    return sample;
}

/// Return the next 4 sample values as a float4 in [0,1).
/// @param[inout] sg Sample generator.
/// @return A float4 in [0,1).
public float4 next_float4<SG : ISampleGenerator>(inout SG sg)
{
    float4 sample;
    // Don't use the float4 initializer to ensure consistent order of evaluation.
    sample.x = next_float(sg);
    sample.y = next_float(sg);
    sample.z = next_float(sg);
    sample.w = next_float(sg);
    return sample;
}
