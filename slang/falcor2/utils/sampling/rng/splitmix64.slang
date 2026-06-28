// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// SplitMix64 pseudorandom number generator.
///
/// This is a fixed-increment version of Java 8's SplittableRandom generator.
/// The period is 2^64 and its state size is 64 bits.
/// It is a very fast generator passing BigCrush. It is recommended for use with
/// other generators like xoroshiro and xorshift to initialize their state arrays.
///
/// Steele Jr, Guy L., Doug Lea, and Christine H. Flood., "Fast Splittable Pseudorandom Number
/// Generators", ACM SIGPLAN Notices 49.10 (2014): 453-472.
/// http://dx.doi.org/10.1145/2714064.2660195.
///
/// This code requires shader model 6.0 or above for 64-bit integer support.
public struct SplitMix64 : IRNG
{
    public typealias Seed = uint2;

    public uint64_t state;

    public __init(Seed seed) { state = (uint64_t(seed[1]) << 32ull) | uint64_t(seed[0]); }

    [mutating]
    public uint next_uint() { return uint(next_uint64()); }

    [mutating]
    public uint64_t next_uint64()
    {
        uint64_t z = (state += 0x9E3779B97F4A7C15ull);
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
        return z ^ (z >> 31);
    }
};
