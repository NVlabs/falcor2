// SPDX-License-Identifier: Apache-2.0

__exported import falcor2.utils;

void run_test(BufferHandle src, RWByteAddressBuffer dst, uint byte_size)
{
    // Copy data from src to dst
    for (int i = 0; i < byte_size; i += 4) {
        uint data = src.load<uint>(i);
        dst.Store(i, data);
    }
}
