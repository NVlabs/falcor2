// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

/// Non-owning shader reference to a byte offset within a buffer.
public struct BufferOffsetPair
{
    public BufferHandle buffer;
    public uint offset;
    // Keep dynamic-object packing consistent with CUDA's 8-byte struct alignment.
    public uint padding;

    public T load<T>(uint location) { return buffer.load<T>(offset + location); }
    public T load_aligned<T, uint alignment>(uint location)
    {
        return buffer.load_aligned<T, alignment>(offset + location);
    }
}
