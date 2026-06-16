// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/utils/idictionary.h"
#include "falcor2/utils/buffer_handle.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/command.h>

#include <cstdint>
#include <limits>
#include <type_traits>
#include <vector>

namespace falcor {

/// A simple Wrapper over a Buffer, allowing setting CPU data and syncing only in update.
/// GPU address/handle can be obtained at any time, even before calling `update`.
class FALCOR_API ManagedBuffer : public Object {
    FALCOR_OBJECT(ManagedBuffer)
public:
    /// Create a new ManagedBuffer.
    /// @param desc Buffer descriptor.
    static ref<ManagedBuffer> create(sgl::BufferDesc desc) { return make_ref<ManagedBuffer>(std::move(desc)); }

    /// Constructor.
    /// @param desc Buffer descriptor.
    ManagedBuffer(sgl::BufferDesc desc)
        : m_buffer_desc(std::move(desc))
    {
        reset_dirty_range();
        resize(desc.size);
    }


    /// Destructor.
    ~ManagedBuffer() = default;

    /// Current size in bytes.
    size_t size() const { return m_data.size(); }

    /// Device buffer.
    /// This is only available if not empty and after calling `update_buffer`.
    const ref<sgl::Buffer>& buffer() const { return m_buffer; }

    /// Set data from a source buffer.
    /// @param src Source data pointer.
    /// @param src_size Size in bytes to copy.
    /// @param offset Byte offset into the buffer.
    void set_data(const void* src, size_t src_size, size_t offset)
    {
        FALCOR_ASSERT_LE(offset + src_size, size());
        mark_dirty(offset, offset + src_size);
        memcpy(m_data.data() + offset, src, src_size);
    }

    template<typename T, std::enable_if_t<!std::is_pointer_v<T>, int> = 0>
    void set_data(const T& src, size_t offset)
    {
        set_data(&src, sizeof(T), offset);
    }

    /// Returns data and marks entire buffer dirty, as it can be arbitrarily modified by the caller.
    void* data()
    {
        mark_all_dirty();
        return m_data.data();
    }

    /// Returns a mutable pointer to the host data without setting dirty flags.
    /// The caller is responsible for manually marking dirty regions if data is modified.
    void* data_unsafe() { return m_data.data(); }

    /// Host data (const access).
    const void* data() const { return m_data.data(); }

    /// Always returns constant data (without setting dirty flags), even on non-const ManagedBuffer.
    const void* cdata() const { return m_data.data(); }

    /// Resize buffer to a new size.
    /// @param size New size in bytes.
    void resize(size_t size)
    {
        if (m_data.size() == size)
            return;
        m_data.resize(size);
        m_buffer_desc.size = size;
    }

    /// Explicitly reset the dirty range.
    void reset_dirty_range()
    {
        m_dirty_range[0] = std::numeric_limits<size_t>::max();
        m_dirty_range[1] = 0;
    }

    /// True if some host data has not been uploaded to the device buffer.
    bool is_dirty() const { return m_dirty_range[0] < m_dirty_range[1]; }

    /// Explicitly mark a byte range as dirty.
    /// @param begin Start of the dirty range (byte offset).
    /// @param end End of the dirty range (byte offset, exclusive).
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
    /// Creates a temporary command encoder, uploads dirty data, and submits.
    /// @param device Device to use.
    void update_buffer(sgl::Device* device)
    {
        auto command_encored = device->create_command_encoder();
        update_buffer(command_encored);
        device->submit_command_buffer(command_encored->finish());
    }

    /// Update the device buffer to reflect the host data.
    /// If the host buffer is not empty or different size than the device buffer, a new device buffer is allocated.
    /// The device buffer is updated with the data that changed on the host buffer since the last call to
    /// `update_buffer`.
    /// @param command_encoder Command encoder to use for upload.
    void update_buffer(sgl::CommandEncoder* command_encoder)
    {
        if (m_data.empty()) {
            m_buffer.reset();
            return;
        }

        if (!m_buffer || m_data.size() != m_buffer->size()) {
            m_buffer = command_encoder->device()->create_buffer(m_buffer_desc);
            mark_all_dirty();
        }

        if (is_dirty() && m_buffer) {
            size_t offset = m_dirty_range[0];
            size_t dirty_size = m_dirty_range[1] - m_dirty_range[0];
            command_encoder->upload_buffer_data(m_buffer, offset, dirty_size, m_data.data() + offset);
            m_buffer->set_data(m_data.data(), m_data.size());
        }
        reset_dirty_range();
    }

private:
    sgl::BufferDesc m_buffer_desc;
    /// Device buffer.
    ref<sgl::Buffer> m_buffer;
    /// Host data.
    std::vector<uint8_t> m_data;
    /// Current dirty range.
    size_t m_dirty_range[2];

    /// Keeps m_buffer alive until next `update` is called, so we don't kill a used GPU resource with resize.
    ref<sgl::Buffer> m_keep_alive;
};

inline BufferHandle to_handle(const ref<ManagedBuffer>& managed_buffer)
{
    FALCOR_ASSERT(managed_buffer->size() == 0 || managed_buffer->buffer());
    return to_handle(managed_buffer->buffer());
}


} // namespace falcor
