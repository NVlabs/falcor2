// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

__exported import falcor2.utils;

void run_test(BufferPtr src, RWByteAddressBuffer dst, uint byte_size)
{
    // Copy data from src to dst
    for (int i = 0; i < byte_size; i += 4) {
        uint data = src.load<uint>(i);
        dst.Store(i, data);
    }
}

uint2x3 load_divergent(uint64_t pointer_data, uint byte_offset)
{
    const BufferPtr src = BufferPtr<void>(pointer_data);
    const BufferPtr offset_src = src.byte_offset(byte_offset);
    return uint2x3(
        src.load<uint>(byte_offset),
        src.load_aligned<uint, 4>(byte_offset),
        offset_src.load<uint>(0),
        offset_src.load_aligned<uint, 4>(0),
        src.as<uint>().element_offset(byte_offset / 4)[0],
        src.as<uint>()[byte_offset / 4]
    );
}

[shader("compute")]
[numthreads(64, 1, 1)]
void copy_rw_vectors(uint3 tid: SV_DispatchThreadID, BufferPtr src, RWBufferPtr dst, uint count)
{
    if (tid.x < count)
        dst.as<float3>()[tid.x + 2] = src.as<float3>()[tid.x];
}

[shader("compute")]
[numthreads(64, 1, 1)]
void update_rw_vectors(uint3 tid: SV_DispatchThreadID, RWBufferPtr buffer, uint count)
{
    if (tid.x < count) {
        const uint index = tid.x + 2;
        const float3 value = buffer.as<float3>()[index];
        buffer.store<float3>(index * 12, value + float3(1.0, 2.0, 3.0));
    }
}
