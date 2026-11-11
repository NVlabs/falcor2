// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public StructuredBuffer<uint> qmc_sobol_table;

// Adapted from PBRT v4 SobolSample and BinaryPermuteScrambler (Apache-2.0).
// Copyright (c) 1998-2020 Matt Pharr, Wenzel Jakob, and Greg Humphreys.
// Changes: 32-bit indices, buffer storage, integer output and PRNG overflow.
// Full direction-table notices and pinned source hashes ship in qmc_tables.json.
uint sobol_uint(uint index, uint dimension)
{
    uint result = 0;
    for (uint column = dimension * 32; index != 0; index >>= 1, ++column) {
        if (index & 1u)
            result ^= qmc_sobol_table[column];
    }
    return result;
}

/// Sobol coordinates 0-1023 with random-digit scrambling, then Xorshift32.
/// Uses 12 bytes of state. The uint sample index wraps after 2^32 points per seed.
public struct PBRTSobolSampleGenerator : ISampleGenerator
{
    // The sample index is dead after the Sobol prefix; reuse its word for PRNG state.
    uint index_or_rng;
    uint dimension;
    uint seed;

    public __init(uint sample_index, uint scrambling_seed = 0)
    {
        index_or_rng = sample_index;
        dimension = 0;
        seed = scrambling_seed;
    }
    public __init(uint2 pixel, uint sample_index)
    {
        this = PBRTSobolSampleGenerator(sample_index, get_pixel_scrambling_seed(pixel));
    }

    [mutating]
    public uint next_uint()
    {
        if (dimension < 1024) {
            uint d = dimension++;
            // Kollig and Keller, "Efficient Multidimensional Sampling" (2002),
            // section 3.2.3: base-two random digit scrambling is a digital XOR shift.
            // Keep each dimension's mask fixed across sample indices to retain the net.
            return sobol_uint(index_or_rng, d) ^ jenkins_hash(seed + d);
        }
        if (dimension == 1024) {
            index_or_rng = jenkins_hash(index_or_rng ^ jenkins_hash(seed)) | 1u;
            dimension = 1025; // Saturate: overflow must never wrap back into Sobol.
        }
        Xorshift32 rng = Xorshift32(index_or_rng);
        uint result = rng.next_uint();
        index_or_rng = rng.state;
        return result;
    }
}
