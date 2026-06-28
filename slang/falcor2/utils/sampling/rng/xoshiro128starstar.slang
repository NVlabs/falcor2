// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Implementation of the xoshiro128** 32-bit all-purpose, rock-solid generator
/// written in 2018 by David Blackman and Sebastiano Vigna (vigna@acm.org).
/// The state is 128 bits and the period (2^128)-1. It has a jump function that
/// allows you to skip ahead 2^64 in the seqeuence.
///
/// Note: The state must be seeded so that it is not everywhere zero.
/// The recommendation is to initialize the state using SplitMix64.
///
/// See the original public domain code: http://xoshiro.di.unimi.it/xoshiro128starstar.c
public struct Xoshiro128StarStar : IRNG
{
    public typealias Seed = uint4;

    public uint4 state;

    public __init(Seed seed)
    {
        state[0] = seed[0];
        state[1] = seed[1];
        state[2] = seed[2];
        state[3] = seed[3];
    }

    [mutating]
    public uint next_uint()
    {
        const uint result_starstar = rotl(state[0] * 5, 7) * 9;
        const uint t = state[1] << 9;

        state[2] ^= state[0];
        state[3] ^= state[1];
        state[1] ^= state[2];
        state[0] ^= state[3];

        state[2] ^= t;
        state[3] = rotl(state[3], 11);

        return result_starstar;
    }

    /// This is equivalent to 2^64 calls to next_uint().
    /// Can be used to generate 2^64 non-overlapping subsequences for parallel computations.
    [mutating]
    public void jump()
    {
        const uint JUMP[] = { 0x8764000b, 0xf542d2d3, 0x6fa035c3, 0x77f2db5b };

        uint s0 = 0;
        uint s1 = 0;
        uint s2 = 0;
        uint s3 = 0;

        for (int i = 0; i < 4; i++) {
            for (int b = 0; b < 32; b++) {
                if ((JUMP[i] & (1u << b)) != 0) {
                    s0 ^= state[0];
                    s1 ^= state[1];
                    s2 ^= state[2];
                    s3 ^= state[3];
                }
                next_uint();
            }
        }

        state[0] = s0;
        state[1] = s1;
        state[2] = s2;
        state[3] = s3;
    }

    static uint rotl(const uint x, int k) { return (x << k) | (x >> (32 - k)); }
};
