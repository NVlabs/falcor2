// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "buffer_suballocator.h"

#include <cstring>

namespace falcor {

BufferSuballocator::BufferSuballocator(std::string_view label)
    : m_allocator(uint32_t(MAX_SIZE / ALIGNMENT))
    , m_data({.label = std::string(label)})
{
}

BufferSuballocator::Allocation BufferSuballocator::allocate(size_t byte_size)
{
    if (byte_size == 0)
        return {};
    FALCOR_CHECK_LE(byte_size, MAX_SIZE);
    const uint32_t units = uint32_t((byte_size + ALIGNMENT - 1) / ALIGNMENT);
    Allocation allocation = m_allocator.allocate(units);
    FALCOR_CHECK(allocation.is_valid(), "Failed to suballocate {} buffer bytes", byte_size);
    try {
        m_data.ensure_size(size_t(byte_offset(allocation)) + size_t(units) * ALIGNMENT);
    } catch (...) {
        m_allocator.free(allocation);
        throw;
    }
    return allocation;
}

void BufferSuballocator::free(Allocation allocation)
{
    if (allocation.is_valid())
        m_allocator.free(allocation);
}

uint32_t BufferSuballocator::byte_offset(Allocation allocation) const
{
    FALCOR_ASSERT(allocation.is_valid());
    return allocation.offset * ALIGNMENT;
}

size_t BufferSuballocator::allocation_size(Allocation allocation) const
{
    return allocation.is_valid() ? size_t(m_allocator.allocation_size(allocation)) * ALIGNMENT : 0;
}

void BufferSuballocator::write(Allocation allocation, std::span<const uint8_t> data)
{
    FALCOR_CHECK_LE(data.size(), allocation_size(allocation));
    if (data.empty())
        return;
    const size_t offset = byte_offset(allocation);
    std::memcpy(m_data.data() + offset, data.data(), data.size());
    m_data.mark_dirty(offset, offset + data.size());
}

std::span<const uint8_t> BufferSuballocator::data(Allocation allocation) const
{
    if (!allocation.is_valid())
        return {};
    return {m_data.data() + byte_offset(allocation), allocation_size(allocation)};
}

std::span<uint8_t> BufferSuballocator::mutable_data(Allocation allocation)
{
    if (!allocation.is_valid())
        return {};
    const size_t offset = byte_offset(allocation);
    const size_t size = allocation_size(allocation);
    m_data.mark_dirty(offset, offset + size);
    return {m_data.data() + offset, size};
}

void BufferSuballocator::update_buffer(sgl::CommandEncoder* command_encoder)
{
    m_data.update_buffer(command_encoder);
}

} // namespace falcor
