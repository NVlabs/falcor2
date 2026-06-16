// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// 32-bit (non-cryptographic) hash function by Robert Jenkins.
/// This is a perfect hash function (no collisions).
/// See https://gist.github.com/badboy/6267743.
public uint jenkins_hash(uint x)
{
    x = (x + 0x7ed55d16) + (x << 12);
    x = (x ^ 0xc761c23c) ^ (x >> 19);
    x = (x + 0x165667b1) + (x << 5);
    x = (x + 0xd3a2646c) ^ (x << 9);
    x = (x + 0xfd7046c5) + (x << 3);
    x = (x ^ 0xb55a4f09) ^ (x >> 16);
    return x;
}

/// Generates a pair of 32-bit pseudorandom numbers based on a pair of 32-bit values.
///
/// The code uses a 64-bit block cipher, the Tiny Encryption Algorithm (TEA) by Wheeler et al., 1994.
/// The 128-bit key is fixed and adapted from here: https://www.ibiblio.org/e-notes/webcl/mc.htm.
/// This function can be useful for seeding other pseudorandom number generators.
///
/// @param v0 The first value (low dword of the block).
/// @param v1 The second value (high dword of the block).
/// @param iterations Number of iterations (the authors recommend 16 at a minimum).
/// @return Two pseudorandom numbers (the block cipher of (v0,v1)).
public uint2 block_cipher_tea(uint v0, uint v1, uint iterations = 16)
{
    uint sum = 0;
    const uint delta = 0x9e3779b9;
    const uint k[4] = { 0xa341316c, 0xc8013ea4, 0xad90777d, 0x7e95761e }; // 128-bit key.
    for (uint i = 0; i < iterations; i++) {
        sum += delta;
        v0 += ((v1 << 4) + k[0]) ^ (v1 + sum) ^ ((v1 >> 5) + k[1]);
        v1 += ((v0 << 4) + k[2]) ^ (v0 + sum) ^ ((v0 >> 5) + k[3]);
    }
    return uint2(v0, v1);
}
