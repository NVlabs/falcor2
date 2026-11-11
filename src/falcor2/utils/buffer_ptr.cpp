// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "buffer_ptr.h"

#include <sgl/device/cursor_utils.h>
#include <sgl/device/device.h>

namespace falcor {

namespace detail {
uint64_t encode_buffer_ptr(sgl::DeviceType backend, uint64_t base, uint64_t byte_offset)
{
    constexpr auto MAX = std::numeric_limits<uint64_t>::max();
    if (backend == sgl::DeviceType::d3d12) {
        FALCOR_CHECK_LT(base, uint64_t(std::numeric_limits<uint32_t>::max()));
        FALCOR_CHECK_LE(byte_offset, uint64_t(std::numeric_limits<uint32_t>::max()));
        return base | (byte_offset << 32);
    }
    FALCOR_CHECK(
        backend == sgl::DeviceType::vulkan || backend == sgl::DeviceType::cuda,
        "BufferPtr does not support this backend"
    );
    FALCOR_CHECK_LT(base, MAX);
    FALCOR_CHECK_LT(byte_offset, MAX - base);
    return base + byte_offset;
}

template<bool Writable>
BufferPtr<Writable>
BufferPtr<Writable>::from_buffer(const ref<sgl::Buffer>& buffer, uint64_t byte_offset, uint32_t alignment)
{
    FALCOR_CHECK(alignment == 4 || alignment == 8 || alignment == 16, "Expected alignment 4, 8 or 16");
    if (!buffer) {
        FALCOR_CHECK_EQ(byte_offset, 0);
        return {};
    }
    const auto backend = buffer->device()->type();
    // TODO(EPTR-METAL-001): native pointers plus indirect resource residency.
    // Concept checked by MSL emission only; no Apple compilation or GPU tests.
    // Resolve the all-ones null sentinel versus RHI's zero-only residency skip.
    // See docs/design/buffer-pointers.md#deferred-metal-integration.
    FALCOR_CHECK(
        backend == sgl::DeviceType::d3d12 || backend == sgl::DeviceType::vulkan || backend == sgl::DeviceType::cuda,
        "BufferPtr supports D3D12, Vulkan and CUDA; Metal support is deferred"
    );
    FALCOR_CHECK_LE(byte_offset, buffer->size());
    const uint64_t address = buffer->device_address();
    FALCOR_CHECK_LE(byte_offset, std::numeric_limits<uint64_t>::max() - address);
    FALCOR_CHECK_EQ((address + byte_offset) % alignment, 0);
    if (backend == sgl::DeviceType::d3d12) {
        FALCOR_CHECK_EQ(byte_offset % alignment, 0);
        const auto descriptor = Writable ? buffer->descriptor_handle_rw() : buffer->descriptor_handle_ro();
        return BufferPtr(encode_buffer_ptr(backend, descriptor.value, byte_offset));
    }
    return BufferPtr(encode_buffer_ptr(backend, address, byte_offset));
}

template<bool Writable>
void BufferPtr<Writable>::check_cursor_type(slang::TypeLayoutReflection* type_layout)
{
    Slang::ComPtr<slang::IBlob> name;
    type_layout->getType()->getFullName(name.writeRef());
    FALCOR_CHECK(
        name
            && std::string_view(static_cast<const char*>(name->getBufferPointer()), name->getBufferSize())
                == slang_type_name,
        "BufferPtr cursor type must be {}",
        slang_type_name
    );
}

template class BufferPtr<false>;
template class BufferPtr<true>;

} // namespace detail

FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<BufferPtr>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<RWBufferPtr>());

} // namespace falcor
