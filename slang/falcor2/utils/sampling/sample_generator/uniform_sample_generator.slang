// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Default uniform pseudorandom number generator.
///
/// This generator has 128 bit state and should have acceptable statistical
/// properties for most rendering applications.
public struct UniformSampleGenerator : ISampleGenerator
{
    Xoshiro128StarStar rng;

    public __init(uint seed)
    {
        LCG init_rng = LCG(seed);
        uint4 seed;
        seed.x = init_rng.next_uint();
        seed.y = init_rng.next_uint();
        seed.z = init_rng.next_uint();
        seed.w = init_rng.next_uint();
        rng = Xoshiro128StarStar(seed);
    }

    public __init(uint2 pixel, uint sample_index)
    {
        // Use SplitMix64 generator to generate a good pseudorandom initial state.
        // The pixel coord is expected to be max 28 bits (16K^2 is the resource limit in D3D12).
        // The sample number is expected to be practically max ~28 bits, e.g. 16spp x 16M samples.
        // As long as both stay <= 32 bits, we will always have a unique initial seed.
        // This is however no guarantee that the generated sequences will never overlap,
        // but it is very unlikely. For example, with these most extreme parameters of
        // 2^56 sequences of length L, the probability of overlap is P(overlap) = L*2^-16.
        uint2 init_seed = block_cipher_tea(interleave_32bit(pixel), sample_index);
        SplitMix64 init_rng = SplitMix64(init_seed);
        uint64_t s0 = init_rng.next_uint64();
        uint64_t s1 = init_rng.next_uint64();
        uint4 seed = { uint(s0), uint(s0 >> 32), uint(s1), uint(s1 >> 32) };

        // Create xoshiro128** pseudorandom generator.
        rng = Xoshiro128StarStar(seed);
    }

    [mutating]
    public uint next_uint() { return rng.next_uint(); }
};
