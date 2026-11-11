// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/utils/managed_vector.h"
#include "falcor2/utils/buffer_handle_table.h"
#include "falcor2/utils/offset_allocator.h"

#include <sgl/device/shader_cursor.h>

#include <memory>
#include <span>

namespace falcor {

/// Typed CPU-backed ranges, allocated first-fit across GPU buffer slots. Slots are
/// constructed on first allocation, added as needed, and retain their capacity after free.
template<typename T>
class ManagedBufferPool {
public:
    struct Allocation {
        /// Pool-local tuple slot; buffer_index() resolves the shader-visible index.
        uint16_t slot_index{0};
        OffsetAllocator::Allocation range;

        bool is_valid() const { return range.is_valid(); }
    };

    ManagedBufferPool(BufferHandleTable& handles, const ManagedVectorDesc& desc)
        : m_desc(desc)
        , m_handles(handles)
    {
        FALCOR_CHECK_LE(desc.max_buffer_size / sizeof(T), std::numeric_limits<uint32_t>::max());
        // Indirect shader reads are invisible to RHI. Keep their state at command-buffer boundaries,
        // including after AS builds, so subsequent uploads wait for these reads before overwriting.
        m_desc.default_state = sgl::ResourceState::shader_resource;
    }

    /// Reserve the same element range in buffer_count same-type buffers.
    /// Allocations with different buffer counts use separate slots within this pool.
    /// Tuple members occupy consecutive handle-table entries. Zero count returns an invalid token.
    Allocation allocate(uint32_t count, uint16_t buffer_count = 1)
    {
        FALCOR_CHECK_GT(buffer_count, 0);
        if (count == 0)
            return {};
        FALCOR_CHECK_LE(count, m_desc.max_buffer_size / sizeof(T));

        OffsetAllocator::Allocation range;
        uint32_t index = 0;
        for (; index < m_slots.size(); ++index) {
            if (m_slots[index]->data.size() != buffer_count)
                continue;
            range = m_slots[index]->allocator.allocate(count);
            if (range.is_valid())
                break;
        }
        if (index == m_slots.size()) {
            FALCOR_CHECK_LE(index, std::numeric_limits<uint16_t>::max());
            auto desc = m_desc;
            desc.label += "[" + std::to_string(index) + "]";
            m_slots.push_back(std::make_unique<Slot>(desc, buffer_count, m_handles));
            range = m_slots.back()->allocator.allocate(count);
            FALCOR_CHECK(range.is_valid(), "Failed to allocate {} elements in a new slot of {}", count, m_desc.label);
        }

        Slot& slot = *m_slots[index];
        try {
            for (auto& data : slot.data)
                data->ensure_size(size_t(range.offset) + count);
        } catch (...) {
            slot.allocator.free(range);
            throw;
        }
        for (auto& data : slot.data)
            data->mark_dirty(range.offset, size_t(range.offset) + count);
        return {static_cast<uint16_t>(index), range};
    }

    void free(Allocation allocation)
    {
        if (allocation.is_valid())
            m_slots[allocation.slot_index]->allocator.free(allocation.range);
    }

    /// Borrowed CPU view of one tuple member; growth of this slot invalidates its views.
    std::span<const T> data(Allocation allocation, uint16_t member = 0) const
    {
        if (!allocation.is_valid())
            return {};
        const Slot& slot = *m_slots[allocation.slot_index];
        return {
            slot.data.at(member)->data() + allocation.range.offset,
            slot.allocator.allocation_size(allocation.range)
        };
    }

    /// Marks the entire allocation dirty. Finish writes before allocating or uploading.
    std::span<T> mutable_data(Allocation allocation, uint16_t member = 0)
    {
        if (!allocation.is_valid())
            return {};
        Slot& slot = *m_slots[allocation.slot_index];
        auto count = slot.allocator.allocation_size(allocation.range);
        slot.data.at(member)->mark_dirty(allocation.range.offset, size_t(allocation.range.offset) + count);
        return {slot.data.at(member)->data() + allocation.range.offset, count};
    }

    /// Update GPU buffers and handles, optionally uploading dirty CPU data for every slot and tuple member.
    /// Skipped dirty ranges are cleared. Buffer growth does not preserve existing GPU contents.
    void update_buffers(sgl::CommandEncoder* command_encoder, bool upload_dirty = true)
    {
        for (auto& slot : m_slots) {
            for (uint16_t member = 0; member < slot->data.size(); ++member) {
                auto& data = *slot->data[member];
                data.update_buffer(command_encoder, upload_dirty);
                m_handles.set(slot->buffer_index + member, data.buffer());
                // Handles hide these resources from automatic shader binding transitions.
                command_encoder->set_buffer_state(data.buffer(), sgl::ResourceState::shader_resource);
            }
        }
    }

    size_t slot_count() const { return m_slots.size(); }

    /// Resolve after update_buffers; growth can replace the GPU buffer.
    const ref<sgl::Buffer>& buffer(uint16_t slot, uint16_t member = 0) const
    {
        return m_slots.at(slot)->data.at(member)->buffer();
    }

    /// Shader-visible buffer index; tuple member n is buffer_index(allocation) + n.
    uint16_t buffer_index(Allocation allocation, uint16_t member = 0) const
    {
        const Slot& slot = *m_slots.at(allocation.slot_index);
        FALCOR_CHECK_LT(member, slot.data.size());
        return slot.buffer_index + member;
    }

private:
    struct Slot {
        std::vector<std::unique_ptr<ManagedVector<T>>> data;
        uint16_t buffer_index;
        OffsetAllocator allocator;

        Slot(const ManagedVectorDesc& desc, uint16_t buffer_count, BufferHandleTable& handles)
            : allocator(static_cast<uint32_t>(desc.max_buffer_size / sizeof(T)))
        {
            for (uint16_t member = 0; member < buffer_count; ++member) {
                auto member_desc = desc;
                member_desc.label += "::" + std::to_string(member);
                data.push_back(std::make_unique<ManagedVector<T>>(member_desc));
            }
            buffer_index = handles.reserve(buffer_count);
        }
    };

    ManagedVectorDesc m_desc;
    std::vector<std::unique_ptr<Slot>> m_slots;
    BufferHandleTable& m_handles;
};

} // namespace falcor
