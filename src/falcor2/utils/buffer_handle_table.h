// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/utils/managed_vector.h"
#include "falcor2/utils/buffer_ptr.h"

namespace falcor {

/// Stable indices into one GPU handle table. Owners retain the referenced buffers.
class BufferHandleTable {
public:
    BufferHandleTable()
        : m_handles({.label = "Scene::geometry"})
    {
    }

    /// Reserve consecutive entries. Entries remain reserved for the owner's lifetime.
    uint16_t reserve(uint16_t count)
    {
        FALCOR_CHECK_GT(count, 0);
        const size_t first = m_count;
        FALCOR_CHECK_LE(first + count, size_t(std::numeric_limits<uint16_t>::max()) + 1);
        m_handles.ensure_size(first + count);
        // Zero is a valid descriptor handle, so set() may not dirty these new entries.
        m_handles.mark_dirty(first, first + count);
        m_count += count;
        return static_cast<uint16_t>(first);
    }

    void set(uint16_t index, const ref<sgl::Buffer>& buffer)
    {
        const uint64_t handle = to_buffer_ptr(buffer).data();
        if (m_handles.data()[index] != handle)
            m_handles[index] = handle;
    }

    /// Call after all owners have updated their buffers and table entries.
    void update_buffer(sgl::CommandEncoder* encoder)
    {
        // An empty scene still binds a valid table. Its unused entry is never dereferenced.
        m_handles.ensure_size(1);
        m_handles.update_buffer(encoder);
    }

    const ref<sgl::Buffer>& buffer() const { return m_handles.buffer(); }

private:
    ManagedVector<uint64_t> m_handles;
    uint32_t m_count{0};
};

} // namespace falcor
