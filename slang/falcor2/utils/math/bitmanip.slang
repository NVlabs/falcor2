// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// 32-bit bit interleave (Morton code).
/// @param v 16-bit values in the LSBs of each component (higher bits don't matter).
/// @return Interleaved 32-bit value.
public uint interleave_32bit(uint2 v)
{
    // clang-format off
    v = v & 0x0000ffff;              // v = ---- ---- ---- ---- fedc ba98 7654 3210
    v = (v | (v << 8)) & 0x00FF00FF; // v = ---- ---- fedc ba98 ---- ---- 7654 3210
    v = (v | (v << 4)) & 0x0F0F0F0F; // v = ---- fedc ---- ba98 ---- 7654 ---- 3210
    v = (v | (v << 2)) & 0x33333333; // v = --fe --dc --ba --98 --76 --54 --32 --10
    v = (v | (v << 1)) & 0x55555555; // v = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    // clang-format on
    return v.x | (v.y << 1);
}

/// 16-bit bit interleave (Morton code).
/// @param v 8-bit values in the LSBs of each component (higher bits don't matter).
/// @return Interleaved 16-bit value in the lower word, 0 elsewhere.
public uint interleave_16bit(uint2 v)
{
    // clang-format off
    v &= 0xff;
    uint j = (v.y << 16) | v.x;      // j = ---- ---- (   y   ) ---- ---- (   x   )
                                     // j = ---- ---- fedc ba98 ---- ---- 7654 3210
    j = (j ^ (j << 4)) & 0x0f0f0f0f; // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
    j = (j ^ (j << 2)) & 0x33333333; // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j << 1)) & 0x55555555; // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    return (j >> 15) | (j & 0xffff); // j = ---- ---- ---- ---- f7e6 d5c4 b3a2 9180
    // clang-format on
}

/// 16-bit bit de-interleave (inverse Morton code).
/// @param i 16-bit value in lower word, must be 0 elsewhere.
/// @return 8-bit values in the LSBs of each component, 0 elsewhere.
public uint2 deinterleave_16bit(uint i)
{
    // clang-format off
    uint j = ((i >> 1) << 16) | i;   // j = -(     i >> 1     ) (         i       )
    j &= 0x55555555;                 // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333; // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f; // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
    j = (j ^ (j >> 4)) & 0x00ff00ff; // j = ---- ---- fedc ba98 ---- ---- 7654 3210
    return uint2(j & 0xff, j >> 16); // x = ---- ---- ---- ---- ---- ---- 7654 3210
                                     // y = ---- ---- ---- ---- ---- ---- fedc ba98
    // clang-format on
}

/// 8-bit bit de-interleave (inverse Morton code).
/// Note: This function has almost exactly the same cost as deinterleave_2x8bit, use the latter if
/// multiple values should be de-interleaved.
/// @param i 8-bit value in lower word, must be 0 elsewhere.
/// @return 4-bit values in the LSBs of each component, 0 elsewhere.
public uint2 deinterleave_8bit(uint i)
{
    // clang-format off
    uint j = ((i >> 1) << 8) | i;    // j = ---- ---- ---- ---- -(i >> 1) (   i   )
    j &= 0x00005555;                 // j = ---- ---- ---- ---- -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333; // j = ---- ---- ---- ---- --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f; // j = ---- ---- ---- ---- ---- 7654 ---- 3210
    return uint2(j & 0xf, j >> 8);   // x = ---- ---- ---- ---- ---- ---- ---- 3210
                                     // y = ---- ---- ---- ---- ---- ---- ---- 7654
    // clang-format on
}

/// 2x 8-bit bit de-interleave (inverse Morton code).
/// @param i 8-bit values in the LSBs of each 16-bit word, must be 0 elsewhere.
/// @return 4-bit values in each component in the LSBs of each 16-bit word, 0 elsewhere.
public uint2 deinterleave_2x8bit(uint i)
{
    // clang-format off
    uint j = ((i & ~0x00010001) << 7) | i; // j = -(i1 >> 1)(  i1   ) -(i0 >> 1)(  i0   )
    j &= 0x55555555;                       // j = -f-e -d-c -b-a -9-8 -7-6 -5-4 -3-2 -1-0
    j = (j ^ (j >> 1)) & 0x33333333;       // j = --fe --dc --ba --98 --76 --54 --32 --10
    j = (j ^ (j >> 2)) & 0x0f0f0f0f;       // j = ---- fedc ---- ba98 ---- 7654 ---- 3210
    return uint2(j, j >> 8) & 0x000f000f;  // x = ---- ---- ---- ba98 ---- ---- ---- 3210
                                           // y = ---- ---- ---- fedc ---- ---- ---- 7654
    // clang-format on
}
