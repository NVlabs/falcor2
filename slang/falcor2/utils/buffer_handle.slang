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

    public T load_aligned<T>(uint location, uint alignment)
    {
        return load<T>(location); // Alignment is not needed for these targets.
    }
#else
    public T load<T>(uint location)
    {
        DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
        return handle.Load<T>(location);
    }

    public T load_aligned<T>(uint location, uint alignment)
    {
        DescriptorHandle<ByteAddressBuffer> handle = { uint2(uint(data), 0) };
        return handle.LoadAligned<T>(location, alignment);
    }
#endif

    public bool is_valid() { return data != INVALID_VALUE; }
}
