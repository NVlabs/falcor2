// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <algorithm>
#include <string>
#include <utility>

namespace falcor {

struct DeviceVectorDesc {
    /// Maximum size of the device buffer in bytes.
    size_t max_buffer_size = 2ull * 1024 * 1024 * 1024;
    sgl::BufferUsage usage = sgl::BufferUsage::shader_resource;
    std::string label;
};

/// Growable device-local storage with no host copy or automatic initialization.
/// Growth preserves existing elements by recording a copy on the caller's encoder.
/// Use an open encoder from the buffer's device, outside a pass, and submit work in
/// dependency order. Rebind the buffer and reacquire device addresses after growth.
/// Storage has exactly capacity() * sizeof(T) bytes; callers must satisfy shader-view
/// layout requirements (D3D12 raw views expose complete four-byte units).
template<typename T>
class DeviceVector {
public:
    /// Starts empty with zero capacity and a null buffer.
    explicit DeviceVector(const DeviceVectorDesc& desc = {})
        : m_desc(desc)
    {
    }

    size_t size() const { return m_size; }
    size_t capacity() const { return m_buffer ? m_buffer->size() / sizeof(T) : 0; }

    /// Device buffer, available after allocating nonzero capacity.
    const ref<sgl::Buffer>& buffer() const { return m_buffer; }

    /// Ensure capacity for this many elements without changing size or shrinking storage.
    /// Growth is geometric, capped by max_buffer_size / sizeof(T).
    void reserve(sgl::CommandEncoder* command_encoder, size_t capacity)
    {
        const size_t max_capacity = m_desc.max_buffer_size / sizeof(T);
        FALCOR_CHECK_LE(capacity, max_capacity);
        const size_t current_capacity = this->capacity();
        if (capacity <= current_capacity)
            return;

        const size_t doubled_capacity = current_capacity > max_capacity / 2 ? max_capacity : current_capacity * 2;
        const size_t new_capacity = std::max(capacity, doubled_capacity);
        auto new_buffer = command_encoder->device()->create_buffer({
            .size = new_capacity * sizeof(T),
            .memory_type = sgl::MemoryType::device_local,
            .usage = m_desc.usage,
            .label = m_desc.label,
        });
        if (m_size > 0) {
            // The recorded copy retains both native allocations through GPU completion.
            command_encoder->copy_buffer(new_buffer, 0, m_buffer, 0, m_size * sizeof(T));
        }
        m_buffer = std::move(new_buffer);
    }

    /// Set the logical element count, preserving the retained prefix. Shrinking retains capacity.
    /// Elements exposed by growth, including shrink followed by regrowth, are uninitialized.
    void resize(sgl::CommandEncoder* command_encoder, size_t size)
    {
        reserve(command_encoder, size);
        m_size = size;
    }

    /// Increase size to at least this many elements; never shrink it.
    void ensure_size(sgl::CommandEncoder* command_encoder, size_t size)
    {
        if (size > m_size)
            resize(command_encoder, size);
    }

private:
    DeviceVectorDesc m_desc;
    size_t m_size = 0;
    ref<sgl::Buffer> m_buffer;

    FALCOR_NON_COPYABLE_AND_MOVABLE(DeviceVector);
};

} // namespace falcor
