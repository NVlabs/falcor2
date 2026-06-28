// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/object.h"
#include "falcor2/utils/idictionary.h"

#include <sgl/device/resource.h>
#include <sgl/device/device.h>

#include <limits>

namespace falcor {

/// An object used to convert CPU Buffer to the GPU BufferHandle,
/// by converting to either pointer or descriptor handle, based on the backend.
class FALCOR_API BufferHandle {
public:
    FALCOR_STATIC_WRITE_TO_CURSOR(BufferHandle);

    /// Data for direct upload into memory location with GPU BufferHandle.
    uint64_t data() const { return m_handle_data; }

    /// Used in get_uniforms() and get_this() calls to produce Python dictionary.
    void to_dictionary(IDictionary& dict) const { write_to_cursor(dict); }

    /// Used when writing using Buffer or Shader Cursors.
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["data"] = m_handle_data;
    }

private:
    BufferHandle() = default;

    explicit BufferHandle(sgl::Buffer* buffer)
    {
        if (!buffer)
            return;

        if (use_gpu_pointer(buffer))
            m_handle_data = buffer->device_address();
        else
            m_handle_data = buffer->descriptor_handle_ro().value;
    }

    static bool use_gpu_pointer(sgl::Buffer* buffer)
    {
        if (!buffer)
            return true;
        const auto& device = buffer->device();
        return (device->type() == sgl::DeviceType::vulkan || device->type() == sgl::DeviceType::cuda);
    }

    static constexpr uint64_t INVALID_HANDLE_DATA = std::numeric_limits<uint64_t>::max();

    uint64_t m_handle_data = INVALID_HANDLE_DATA;

    friend BufferHandle to_handle(const ref<sgl::Buffer>& buffer);
};

inline BufferHandle to_handle(const ref<sgl::Buffer>& buffer)
{
    return BufferHandle(buffer.get());
}

} // namespace falcor
