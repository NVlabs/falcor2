// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/object.h"

#include <sgl/device/resource.h>
#include <sgl/device/device.h>

#include <array>
#include <limits>
#include <string_view>

namespace falcor {

namespace detail {
inline uint64_t buffer_handle_data(sgl::Buffer* buffer, bool writable)
{
    if (!buffer)
        return std::numeric_limits<uint64_t>::max();
    const auto type = buffer->device()->type();
    if (type == sgl::DeviceType::vulkan || type == sgl::DeviceType::cuda)
        return buffer->device_address();
    return writable ? buffer->descriptor_handle_rw().value : buffer->descriptor_handle_ro().value;
}
} // namespace detail

/// An object used to convert CPU Buffer to the GPU BufferHandle,
/// by converting to either pointer or descriptor handle, based on the backend.
class FALCOR_API BufferHandle {
public:
    /// Construct an invalid handle.
    BufferHandle() = default;

    FALCOR_STATIC_WRITE_TO_CURSOR(BufferHandle);

    /// Slang type used by the native cursor-writer functional fallback.
    static constexpr std::string_view slang_type_name = "BufferHandle";
    static constexpr std::array<std::string_view, 1> slangpy_imports() { return {"falcor2/utils.slang"}; }

    /// Data for direct upload into memory location with GPU BufferHandle.
    uint64_t data() const { return m_handle_data; }

    /// Used when writing using Buffer or Shader Cursors.
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = m_handle_data;
    }

private:
    explicit BufferHandle(sgl::Buffer* buffer)
        : m_handle_data(detail::buffer_handle_data(buffer, false))
    {
    }

    static constexpr uint64_t INVALID_HANDLE_DATA = std::numeric_limits<uint64_t>::max();

    uint64_t m_handle_data = INVALID_HANDLE_DATA;

    friend BufferHandle to_handle(const ref<sgl::Buffer>& buffer);
};

inline BufferHandle to_handle(const ref<sgl::Buffer>& buffer)
{
    return BufferHandle(buffer.get());
}

/// Non-owning shader reference for read-write buffer access. The caller must declare
/// unordered-access state on the command encoder and retain the referenced resource.
class FALCOR_API RWBufferHandle {
public:
    /// Construct an invalid handle.
    RWBufferHandle() = default;

    FALCOR_STATIC_WRITE_TO_CURSOR(RWBufferHandle);

    static constexpr std::string_view slang_type_name = "RWBufferHandle";
    static constexpr std::array<std::string_view, 1> slangpy_imports() { return {"falcor2/utils.slang"}; }

    /// Data for upload into a GPU RWBufferHandle.
    uint64_t data() const { return m_handle_data; }

    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = m_handle_data;
    }

private:
    explicit RWBufferHandle(sgl::Buffer* buffer)
        : m_handle_data(detail::buffer_handle_data(buffer, true))
    {
    }

    uint64_t m_handle_data = std::numeric_limits<uint64_t>::max();

    friend RWBufferHandle to_rw_handle(const ref<sgl::Buffer>& buffer);
};

/// Resolve a writable GPU address or UAV descriptor from the current buffer.
inline RWBufferHandle to_rw_handle(const ref<sgl::Buffer>& buffer)
{
    return RWBufferHandle(buffer.get());
}

} // namespace falcor
