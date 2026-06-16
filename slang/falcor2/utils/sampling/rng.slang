// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Interface for random number generators.
public interface IRNG
{
    /// Seed type.
    associatedtype Seed;

    /// Initialize the RNG with a seed.
    /// @param seed Seed value.
    __init(Seed seed);

    /// Get next pseudorandom number in the sequence (32 bits).
    /// @return Pseudorandom number.
    [mutating]
    uint next_uint();
}
