// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Simple linear congruential generator (LCG).
///
/// The code uses the parameters from the book series "Numerical Recipes".
/// The period is 2^32 and its state size is 32 bits.
/// Note: Only for basic applications. The generator has poor statistical
/// properties and is sensitive to good seeding. If many parallel generators
/// are used (e.g. one per pixel) there will be significant correlation
/// between the generated pseudorandom sequences. In those cases, it is
/// recommended to use one of the generators with larger state.
public struct LCG : IRNG
{
    public typealias Seed = uint;

    public uint state;

    public __init(Seed seed) { state = seed; }

    [mutating]
    public uint next_uint()
    {
        const uint A = 1664525u;
        const uint C = 1013904223u;
        state = (A * state + C);
        return state;
    }
};
