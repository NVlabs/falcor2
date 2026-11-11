// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/core/type_utils.h>
#include <algorithm>
#include <array>
#include <vector>

namespace falcor {

/// Own packed CPU samples and their immutable, sample-major GPU copies for one output range.
template<typename T>
class GeometrySampleStorage {
public:
    struct Location {
        ref<sgl::Buffer> buffer;
        uint32_t element_offset;
    };

    GeometrySampleStorage() = default;

    /// Replace the samples and allocate their GPU copies. Empty samples release storage.
    /// Buffers contain whole samples, up to max_buffer_size bytes (capped at 2 GiB).
    /// A single sample exceeding that limit is unsupported.
    void reset(
        sgl::Device* device,
        std::vector<T> samples,
        uint32_t vertices_per_sample,
        size_t max_buffer_size = 2ull * 1024 * 1024 * 1024
    )
    {
        m_samples = std::move(samples);
        m_vertices_per_sample = vertices_per_sample;
        m_buffers.clear();
        m_samples_per_buffer = 0;
        if (m_samples.empty())
            return;
        FALCOR_ASSERT(vertices_per_sample > 0 && m_samples.size() % vertices_per_sample == 0);
        max_buffer_size = std::min(max_buffer_size, size_t(2ull * 1024 * 1024 * 1024));
        const size_t sample_bytes = size_t(vertices_per_sample) * sizeof(T);
        FALCOR_CHECK(
            sample_bytes <= max_buffer_size,
            "Geometry sample requires {} bytes, exceeding the per-buffer limit of {} bytes",
            sample_bytes,
            max_buffer_size
        );
        m_samples_per_buffer = max_buffer_size / sample_bytes;
        const size_t elements_per_buffer = m_samples_per_buffer * vertices_per_sample;
        for (size_t offset = 0; offset < m_samples.size();) {
            const size_t element_count = std::min(elements_per_buffer, m_samples.size() - offset);
            m_buffers.push_back(device->create_buffer({
                .size = element_count * sizeof(T),
                .usage = sgl::BufferUsage::shader_resource,
                .label = "GeometrySampleStorage::samples",
                .data = m_samples.data() + offset,
                .data_size = element_count * sizeof(T),
            }));
            offset += element_count;
        }
    }

    std::array<Location, 2> prepare(sgl::CommandEncoder* encoder, uint32_t source0, uint32_t source1)
    {
        std::array<Location, 2> sources{location(source0), location(source1)};
        encoder->set_buffer_state(sources[0].buffer, sgl::ResourceState::shader_resource);
        if (sources[1].buffer != sources[0].buffer)
            encoder->set_buffer_state(sources[1].buffer, sgl::ResourceState::shader_resource);
        return sources;
    }

private:
    Location location(uint32_t sample) const
    {
        FALCOR_ASSERT(size_t(sample) * m_vertices_per_sample + m_vertices_per_sample <= m_samples.size());
        const size_t offset = (sample % m_samples_per_buffer) * m_vertices_per_sample;
        return {m_buffers[sample / m_samples_per_buffer], sgl::narrow_cast<uint32_t>(offset)};
    }

    std::vector<T> m_samples;
    uint32_t m_vertices_per_sample = 0;
    size_t m_samples_per_buffer = 0;
    std::vector<ref<sgl::Buffer>> m_buffers;
};

} // namespace falcor
