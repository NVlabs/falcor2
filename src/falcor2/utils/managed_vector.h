// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>

#include <limits>
#include <string>
#include <vector>

namespace falcor {

/// ManagedVector descriptor.
struct ManagedVectorDesc {
    /// Maximum size of device buffer (defaults to 2GB which is a D3D12 limit mostly).
    size_t max_buffer_size = 2ull * 1024 * 1024 * 1024;
    /// Device buffer memory type.
    sgl::MemoryType memory_type{sgl::MemoryType::device_local};
    /// Device buffer usage.
    sgl::BufferUsage usage{sgl::BufferUsage::shader_resource};
    /// Device buffer label.
    std::string label;
};

/// A vector that maintains synchronized host and device copies with dirty tracking.
/// ManagedVector provides a convenient way to manage device buffers that are frequently
/// updated from the CPU. It maintains a host-side vector and a device-side buffer,
/// tracking which elements have been modified to minimize data transfer.
template<typename T>
class ManagedVector {
public:
    /// Constructor.
    ManagedVector(const ManagedVectorDesc& desc)
        : m_desc{desc}
    {
        reset_dirty_range();
    }

    /// Destructor.
    ~ManagedVector() { }

    /// Current number of elements in the buffer.
    size_t size() const { return m_data.size(); }

    /// Device buffer.
    /// This is only available if not empty and after calling `update_buffer`.
    const ref<sgl::Buffer>& buffer() const { return m_buffer; }

    /// Host data.
    const T* data() const { return m_data.data(); }
    T* data() { return m_data.data(); }

    /// Resize buffer to a new size. Size must not exceed capacity.
    /// @param size New size.
    void resize(size_t size)
    {
        FALCOR_CHECK_LE(size * sizeof(T), m_desc.max_buffer_size);
        m_data.resize(size);
    }

    /// Ensure the buffer has a minimum size. Size must not exceed capacity.
    /// @param size New minimum size.
    void ensure_size(size_t size)
    {
        if (size > m_data.size())
            resize(size);
    }

    /// Element accessor (const access).
    const T& operator[](size_t index) const
    {
        FALCOR_ASSERT_LT(index, m_data.size());
        return m_data[index];
    }

    /// Element accessor (non-const access).
    /// Marks the element as dirty.
    T& operator[](size_t index)
    {
        FALCOR_ASSERT_LT(index, m_data.size());
        mark_dirty(index, index + 1);
        return m_data[index];
    }

    /// Explicitly reset the dirty range.
    void reset_dirty_range()
    {
        m_dirty_range[0] = std::numeric_limits<size_t>::max();
        m_dirty_range[1] = 0;
    }

    /// True if some host data has not been uploaded to the device buffer.
    bool is_dirty() const { return m_dirty_range[0] < m_dirty_range[1]; }

    /// Explicitly mark a range as dirty.
    void mark_dirty(size_t begin, size_t end)
    {
        m_dirty_range[0] = std::min(m_dirty_range[0], begin);
        m_dirty_range[1] = std::max(m_dirty_range[1], end);
    }

    /// Mark the entire buffer as dirty.
    void mark_all_dirty()
    {
        m_dirty_range[0] = 0;
        m_dirty_range[1] = m_data.size();
    }

    /// Update the device buffer to reflect the host data.
    /// If the host buffer is not empty or bigger than the device buffer, a new device buffer is allocated.
    /// The device buffer is updated with the data that changed on the host buffer since the last call to
    /// `update_buffer`.
    /// @param command_encoder Command encoder to use for upload.
    /// @param upload_dirty Upload dirty data (enabled by default).
    void update_buffer(sgl::CommandEncoder* command_encoder, bool upload_dirty = true)
    {
        if (m_data.size() == 0)
            return;
        if (!m_buffer || m_data.size() * sizeof(T) > m_buffer->size()) {
            mark_dirty(0, m_data.size());

            // Amortized growth for the device buffer.
            size_t new_buffer_size = m_buffer ? m_buffer->size() * 2 : 0;
            new_buffer_size = std::clamp(new_buffer_size, m_data.size() * sizeof(T), m_desc.max_buffer_size);
            FALCOR_CHECK(m_data.size() * sizeof(T) <= new_buffer_size, "Buffer not large enough");

            m_buffer = command_encoder->device()->create_buffer({
                .size = new_buffer_size,
                .memory_type = m_desc.memory_type,
                .usage = m_desc.usage,
                .label = m_desc.label,
            });
        }
        if (upload_dirty && is_dirty()) {
            size_t offset = m_dirty_range[0] * sizeof(T);
            size_t size = (m_dirty_range[1] - m_dirty_range[0]) * sizeof(T);
            command_encoder->upload_buffer_data(m_buffer, offset, size, &m_data[m_dirty_range[0]]);
        }
        reset_dirty_range();
    }

private:
    ManagedVectorDesc m_desc;
    /// Host data.
    std::vector<T> m_data;
    /// Device buffer.
    ref<sgl::Buffer> m_buffer;
    /// Current dirty range.
    size_t m_dirty_range[2];

    FALCOR_NON_COPYABLE_AND_MOVABLE(ManagedVector);
};

} // namespace falcor
