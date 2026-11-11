// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/utils/managed_vector.h"
#include "falcor2/utils/offset_allocator.h"

#include <span>
#include <string_view>

namespace falcor {

/// Stable, 16-byte-aligned ranges in one CPU-backed GPU buffer. Freeing ranges
/// retains capacity; GPU replacement on growth requires refreshing borrowed handles.
class FALCOR_API BufferSuballocator {
public:
    using Allocation = OffsetAllocator::Allocation;
    static constexpr uint32_t ALIGNMENT = 16;
    static constexpr size_t MAX_SIZE = 2ull * 1024 * 1024 * 1024;

    explicit BufferSuballocator(std::string_view label);

    /// Zero bytes returns an invalid allocation. Tokens belong to this allocator.
    Allocation allocate(size_t byte_size);
    void free(Allocation allocation);
    uint32_t byte_offset(Allocation allocation) const;
    size_t allocation_size(Allocation allocation) const;
    /// Copy to the start of an allocation and mark the written bytes dirty.
    void write(Allocation allocation, std::span<const uint8_t> data);
    /// Borrowed CPU storage, including padding. Pool growth invalidates all views.
    std::span<const uint8_t> data(Allocation allocation) const;
    /// Marks the allocation dirty. Finish writing before any other allocation can grow the pool.
    std::span<uint8_t> mutable_data(Allocation allocation);
    void update_buffer(sgl::CommandEncoder* command_encoder);
    const ref<sgl::Buffer>& buffer() const { return m_data.buffer(); }

private:
    OffsetAllocator m_allocator;
    ManagedVector<uint8_t> m_data;
};

} // namespace falcor
