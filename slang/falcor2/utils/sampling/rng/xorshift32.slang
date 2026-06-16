// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Xorshift32 is a tiny pseudorandom generator with reasonably good properties for its size.
/// It's still not good enough for many use cases due to its small state.
///
/// Prefer pseudorandom generators with a larger state.
public struct Xorshift32 : IRNG
{
    public typealias Seed = uint;

    public uint state;

    public __init(Seed seed) { state = seed; }

    [mutating]
    public uint next_uint()
    {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        return state;
    }
};
