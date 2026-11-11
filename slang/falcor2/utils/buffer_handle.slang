// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

implementing utils;

public struct BufferHandle
{
    public uint64_t data;
    /// Value indicating invalid data.
    static const uint64_t INVALID_VALUE = 0xFFFFFFFFFFFFFFFFull;
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__)
    public T load<T>(uint location)
    {
        ImmutablePtr<T> ptr = { data + location };

        return *ptr;
    }

    public T load_aligned<T, uint Alignment>(uint location)
    {
        return load<T>(location); // Alignment is not needed for these targets.
    }
#else
    public T load<T>(uint location)
    {
        DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
        return handle.Load<T>(location);
    }

    public T load_aligned<T, uint Alignment>(uint location)
    {
        DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
        return handle.LoadAligned<T>(location, Alignment);
    }
#endif

    /// Read an element by index, not by byte offset.
    /// The entire buffer must contain a homogeneous array of T, starting at byte
    /// offset zero with stride sizeof(T). Do not use this for mixed-type buffers
    /// or buffers with headers. The buffer base must be at least 16-byte aligned.
    public T load_element<T>(uint index)
    {
        // Keep sizeof generic: concrete sizes of structs with static members are
        // misreported by the current compiler; generic sizeof(T) matches CPU storage.
        static const uint STRIDE = sizeof(T);
        // Choose the largest alignment that divides the stride and is guaranteed by the 16-byte-aligned base.
        // These branches specialize away after T is known.
        if (STRIDE % 16 == 0)
            return load_aligned<T, 16>(index * STRIDE);
        if (STRIDE % 8 == 0)
            return load_aligned<T, 8>(index * STRIDE);
        if (STRIDE % 4 == 0)
            return load_aligned<T, 4>(index * STRIDE);
        if (STRIDE % 2 == 0)
            return load_aligned<T, 2>(index * STRIDE);
        return load_aligned<T, 1>(index * STRIDE);
    }

    public bool is_valid() { return data != INVALID_VALUE; }
}

/// Non-owning writable handle. The caller declares unordered-access state for the buffer.
public struct RWBufferHandle
{
    public uint64_t data;
    static const uint64_t INVALID_VALUE = 0xFFFFFFFFFFFFFFFFull;

    public T load<T>(uint location)
    {
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__)
        Ptr<T> ptr = { data + location };
        return *ptr;
#else
        DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(data), 0) };
        return handle.Load<T>(location);
#endif
    }

    public void store<T>(uint location, T value)
    {
#if defined(__TARGET_VULKAN__) || defined(__TARGET_CUDA__)
        Ptr<T> ptr = { data + location };
        *ptr = value;
#else
        DescriptorHandle<RWByteAddressBuffer> handle = { uint2(uint(data), 0) };
        handle.Store<T>(location, value);
#endif
    }

    /// Read a homogeneous array element, using sizeof(T) as the stride.
    public T load_element<T>(uint index) { return load<T>(index * sizeof(T)); }

    /// Write a homogeneous array element, using sizeof(T) as the stride.
    public void store_element<T>(uint index, T value) { store<T>(index * sizeof(T), value); }

    public bool is_valid() { return data != INVALID_VALUE; }
}

/// Non-owning, read-only view of a homogeneous buffer, indexed in elements.
public struct BufferView<T>
{
    public BufferHandle handle;

    public __subscript(uint index)->T
    {
        get {
            return handle.load_element<T>(index);
        }
    }
}

public extension BufferHandle
{
    /// View the entire buffer as a homogeneous array of T, with no header or mixed types.
    /// Elements start at byte zero with stride sizeof(T); the base must be 16-byte aligned.
    public BufferView<T> as_buffer<T>()
    {
        return { this };
    }
}
