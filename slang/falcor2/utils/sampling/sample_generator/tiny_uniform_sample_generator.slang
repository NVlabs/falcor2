// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Tiny uniform random sample generator.
///
/// This generator has only 32 bit state and sub-optimal statistical properties.
/// Do not use for anything critical; correlation artifacts may be prevalent.
public struct TinyUniformSampleGenerator : ISampleGenerator
{
    LCG rng;

    public __init(uint seed) { rng = LCG(seed); }

    public __init(uint2 pixel, uint sample_index)
    {
        // Use block cipher to generate a pseudorandom initial seed.
        uint seed = block_cipher_tea(interleave_32bit(pixel), sample_index).x;
        rng = LCG(seed);
    }

    [mutating]
    public uint next_uint() { return rng.next_uint(); }
};
