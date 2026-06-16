// SPDX-License-Identifier: Apache-2.0

#include "distribution_1d.h"

#include "falcor2/core/error.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <sgl/device/cursor_utils.h>

namespace falcor {

DiscreteDistribution1D::DiscreteDistribution1D(sgl::Device* device, std::span<const float> func, std::string_view label)
{
    m_size = static_cast<uint32_t>(func.size());
    m_pdf.resize(m_size);
    m_cdf.resize(m_size);
    float sum = 0.f;
    for (uint32_t i = 0; i < m_size; ++i) {
        float value = func[i];
        if (value < 0.f) {
            FALCOR_THROW("DiscreteDistribution1D: function values must be non-negative");
        }
        sum += value;
        m_cdf[i] = sum;
    }

    float normalization = sum > 0.f ? 1.f / sum : 0.f;
    for (uint32_t i = 0; i < m_size; ++i) {
        m_pdf[i] = func[i] * normalization;
        m_cdf[i] *= normalization;
    }

    // Avoid creating zero-sized buffers.
    if (m_size == 0) {
        m_pdf.push_back(1.f);
        m_cdf.push_back(1.f);
    }

    std::string default_label;
    if (label.empty()) {
        default_label = fmt::format("DiscreteDistribution1D({})", fmt::ptr(this));
        label = default_label;
    }

    m_pdf_buffer = device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = fmt::format("{}::pdf_buffer", label),
        .data = m_pdf.data(),
        .data_size = m_pdf.size() * sizeof(float),
    });

    m_cdf_buffer = device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = fmt::format("{}::cdf_buffer", label),
        .data = m_cdf.data(),
        .data_size = m_cdf.size() * sizeof(float),
    });
}

AliasTable1D::AliasTable1D(sgl::Device* device, std::span<const float> func, std::string_view label)
{
    m_size = static_cast<uint32_t>(func.size());
    m_pdf.resize(m_size);
    m_alias_table.resize(m_size);
    float sum = 0.f;
    for (uint32_t i = 0; i < m_size; ++i) {
        float value = func[i];
        if (value < 0.f) {
            FALCOR_THROW("AliasTable1D: function values must be non-negative");
        }
        sum += value;
    }

    float normalization = sum > 0.f ? 1.f / sum : 0.f;
    for (uint32_t i = 0; i < m_size; ++i) {
        m_pdf[i] = func[i] * normalization;
    }

    // Build alias table using Vose's algorithm (Vose, 1991).
    // Each probability is scaled by n so the average is 1.0. Indices are
    // partitioned into "small" (scaled prob < 1) and "large" (>= 1) worklists.
    // We repeatedly pair a small entry with a large entry: the small entry
    // keeps its scaled probability and the large entry becomes its alias.
    // The large entry's probability is reduced by the deficit (1 - small_prob)
    // and may move to the small list. Any entries remaining due to
    // floating-point rounding get prob = 1.0 (they alias to themselves).
    if (m_size > 0) {
        // Scaled probabilities: p_i * n, so the average is 1.0.
        std::vector<float> prob(m_size);
        float scale = static_cast<float>(m_size);
        for (uint32_t i = 0; i < m_size; ++i)
            prob[i] = m_pdf[i] * scale;

        // Partition indices into small and large worklists.
        std::vector<uint32_t> small, large;
        small.reserve(m_size);
        large.reserve(m_size);
        for (uint32_t i = 0; i < m_size; ++i) {
            if (prob[i] < 1.f)
                small.push_back(i);
            else
                large.push_back(i);
        }

        // Pair small and large entries until one list is exhausted.
        while (!small.empty() && !large.empty()) {
            uint32_t s = small.back();
            small.pop_back();
            uint32_t l = large.back();
            large.pop_back();

            m_alias_table[s] = {prob[s], l};

            // Reduce the large entry's probability by the deficit.
            prob[l] = (prob[l] + prob[s]) - 1.f;

            if (prob[l] < 1.f)
                small.push_back(l);
            else
                large.push_back(l);
        }

        // Remaining entries get prob = 1.0 (numerical leftovers).
        while (!large.empty()) {
            uint32_t l = large.back();
            large.pop_back();
            m_alias_table[l] = {1.f, l};
        }
        while (!small.empty()) {
            uint32_t s = small.back();
            small.pop_back();
            m_alias_table[s] = {1.f, s};
        }
    }

    // Avoid creating zero-sized buffers.
    if (m_size == 0) {
        m_pdf.push_back(1.f);
        m_alias_table.push_back({1.f, 0});
    }

    std::string default_label;
    if (label.empty()) {
        default_label = fmt::format("AliasTable1D({})", fmt::ptr(this));
        label = default_label;
    }

    m_pdf_buffer = device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = fmt::format("{}::pdf_buffer", label),
        .data = m_pdf.data(),
        .data_size = m_pdf.size() * sizeof(float),
    });

    m_alias_table_buffer = device->create_buffer({
        .usage = sgl::BufferUsage::shader_resource,
        .label = fmt::format("{}::alias_table_buffer", label),
        .data = m_alias_table.data(),
        .data_size = m_alias_table.size() * sizeof(Entry),
    });
}

FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<DiscreteDistribution1D>());
FALCOR_STATIC_ONCE(sgl::cursor_utils::register_cursor_writer<AliasTable1D>());

} // namespace falcor
