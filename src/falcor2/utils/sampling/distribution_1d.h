// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/cursor_writer.h"
#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>

#include <span>
#include <string_view>
#include <vector>

namespace falcor {

/// Discrete 1D probability distribution for importance sampling.
///
/// Given a set of non-negative weights, this class builds a discrete probability
/// distribution that can be sampled on the device. It computes a normalized PDF and
/// a CDF for efficient inverse transform sampling using binary search.
///
/// The distribution stores both PDF and CDF in device buffers for shader access.
/// Sampling returns an index in [0, size-1] with probability proportional to the
/// original weight at that index.
///
/// Definitions:
/// sum = sum_{i=0}^{size-1} weight[i]
/// pdf[i] = weight[i] / sum
/// cdf[i] = sum(pdf[0..i]), so cdf[size-1] == 1.0 (if sum > 0).
class FALCOR_API DiscreteDistribution1D : public Object {
public:
    FALCOR_STATIC_WRITE_TO_SHADER_CURSOR(DiscreteDistribution1D);

    /// Construct a discrete distribution from a set of weights.
    /// @param device Device for buffer allocation.
    /// @param func Non-negative weights defining the distribution.
    /// @param label Debug label device resources.
    DiscreteDistribution1D(sgl::Device* device, std::span<const float> func, std::string_view label = {});

    /// Number of elements in the distribution.
    uint32_t size() const { return m_size; }

    /// Normalize probability density function.
    const std::vector<float>& pdf() const { return m_pdf; }

    /// Cumulative distribution function.
    const std::vector<float>& cdf() const { return m_cdf; }

    /// Device buffer containing the probability density function.
    sgl::Buffer* pdf_buffer() const { return m_pdf_buffer; }

    /// Device buffer containing the cumulative distribution function.
    sgl::Buffer* cdf_buffer() const { return m_cdf_buffer; }

    /// Write the handle data to a shader cursor for GPU binding.
    /// @tparam TCursor Shader cursor type.
    /// @param cursor Shader cursor to write to.
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["size"] = size();
        cursor["pdf_buffer"] = m_pdf_buffer;
        cursor["cdf_buffer"] = m_cdf_buffer;
    }

private:
    uint32_t m_size;
    std::vector<float> m_pdf;
    std::vector<float> m_cdf;

    ref<sgl::Buffer> m_pdf_buffer;
    ref<sgl::Buffer> m_cdf_buffer;

    FALCOR_NON_COPYABLE_AND_MOVABLE(DiscreteDistribution1D);
};

/// Discrete 1D probability distribution for importance sampling.
///
/// Given a set of non-negative weights, this class builds a discrete probability
/// distribution that can be sampled on the device. It computes a normalized PDF and
/// creates an alias table for efficient sampling using Vose's alias method
/// (Vose, 1991), which allows O(n) construction and O(1) sampling.
///
/// The distribution stores both PDF and alias table in device buffers for shader access.
/// Sampling returns an index in [0, size-1] with probability proportional to the
/// original weight at that index.
///
/// Definitions:
/// sum = sum_{i=0}^{size-1} weight[i]
/// pdf[i] = weight[i] / sum
/// alias_table[i] = probability of the main index i, with the alias index for the remaining probability.
class FALCOR_API AliasTable1D : public Object {
public:
    FALCOR_STATIC_WRITE_TO_SHADER_CURSOR(AliasTable1D);

    /// Alias table entry.
    struct Entry {
        /// Probability of the original index.
        float prob;
        /// Alias index for the remaining probability.
        uint32_t alias;
    };

    /// Construct a discrete distribution from a set of weights.
    /// @param device Device for buffer allocation.
    /// @param func Non-negative weights defining the distribution.
    /// @param label Debug label device resources.
    AliasTable1D(sgl::Device* device, std::span<const float> func, std::string_view label = {});

    /// Number of elements in the distribution.
    uint32_t size() const { return m_size; }

    /// Normalize probability density function.
    const std::vector<float>& pdf() const { return m_pdf; }

    /// Alias table entries.
    const std::vector<Entry>& alias_table() const { return m_alias_table; }

    /// Device buffer containing the probability density function.
    sgl::Buffer* pdf_buffer() const { return m_pdf_buffer; }

    /// Device buffer containing the alias table.
    sgl::Buffer* alias_table_buffer() const { return m_alias_table_buffer; }

    /// Write the handle data to a shader cursor for GPU binding.
    /// @tparam TCursor Shader cursor type.
    /// @param cursor Shader cursor to write to.
    template<typename TCursor>
    void write_to_cursor(TCursor& cursor) const
    {
        cursor["size"] = size();
        cursor["pdf_buffer"] = m_pdf_buffer;
        cursor["alias_table_buffer"] = m_alias_table_buffer;
    }

private:
    uint32_t m_size;
    std::vector<float> m_pdf;
    std::vector<Entry> m_alias_table;

    ref<sgl::Buffer> m_pdf_buffer;
    ref<sgl::Buffer> m_alias_table_buffer;

    FALCOR_NON_COPYABLE_AND_MOVABLE(AliasTable1D);
};

} // namespace falcor
