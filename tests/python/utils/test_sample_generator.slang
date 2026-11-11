// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.utils;

public uint sample<SG : ISampleGenerator>(uint index, uint seed, uint dimension)
{
    SG sg = SG(index, seed);
    uint result = 0;
    for (uint d = 0; d <= dimension; ++d)
        result = sg.next_uint();
    return result;
}

public float4 sample_vector<SG : ISampleGenerator>(uint index, uint seed, uint dimension)
{
    SG sg = SG(index, seed);
    for (uint d = 0; d < dimension; ++d)
        sg.next_uint();
    return next_float4(sg);
}

public uint sample_pixel<SG : ISampleGenerator>(uint index, uint2 pixel, uint dimension)
{
    SG sg = SG(pixel, index);
    uint result = 0;
    for (uint d = 0; d <= dimension; ++d)
        result = sg.next_uint();
    return result;
}

public uint sampler_size<SG : ISampleGenerator>()
{
    return sizeof(SG);
}

public uint copy_matches<SG : ISampleGenerator>(uint index, uint seed, uint dimension)
{
    SG sg = SG(index, seed);
    for (uint d = 0; d < dimension; ++d)
        sg.next_uint();
    SG copy = sg;
    for (uint i = 0; i < 16; ++i) {
        uint a = sg.next_uint();
        uint b = copy.next_uint();
        if (a != b)
            return 0;
    }
    return 1;
}

public uint4 legacy_stream<SG : ISampleGenerator>(uint seed, uint2 pixel, bool pixel_mode)
{
    SG sg = pixel_mode ? SG(pixel, seed) : SG(seed);
    uint4 result;
    result.x = sg.next_uint();
    result.y = sg.next_uint();
    result.z = sg.next_uint();
    result.w = sg.next_uint();
    return result;
}
