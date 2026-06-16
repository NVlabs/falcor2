// SPDX-License-Identifier: Apache-2.0

import slangpy;
__exported import falcor2.utils;

public uint test_rng<RNG : IRNG>(uint index, RNG.Seed seed)
{
    RNG rng = RNG(seed);
    uint result = 0;
    for (uint i = 0; i <= index; ++i)
        result = rng.next_uint();
    return result;
}
