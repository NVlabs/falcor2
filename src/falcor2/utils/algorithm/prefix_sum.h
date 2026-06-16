// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>
#include <sgl/device/resource.h>

#include <map>
#include <optional>

namespace falcor {

/// Utility for computing prefix sum in parallel.
/// The prefix sum is computed in place using exclusive scan.
/// Each new element is y[i] = x[0] + ... + x[i-1], for i=1..N and y[0] = 0.
class FALCOR_API PrefixSum : public Object {
    FALCOR_OBJECT(PrefixSum)
public:
    /// Constructor.
    PrefixSum(ref<sgl::Device> device);

    /// Computes the parallel prefix sum over an array of elements.
    /// This supports int32, uint32, int64, uint64, float32 types.
    /// @param command_encoder Command encoder.
    /// @param type Data type of the elements.
    /// @param data Data buffer and offset to compute prefix sum over.
    /// @param element_count Number of elements to compute prefix sum over.
    /// @param total_sum (Optional) Buffer and offset to which the total sum is copied.
    void execute(
        sgl::CommandEncoder* command_encoder,
        sgl::DataType type,
        sgl::BufferOffsetPair data,
        uint32_t element_count,
        std::optional<sgl::BufferOffsetPair> total_sum = {}
    );

private:
    struct Pipelines {
        ref<sgl::ComputePipeline> group_scan_pipeline;
        ref<sgl::ComputePipeline> finalize_pipeline;
    };

    const Pipelines& get_pipelines(sgl::DataType type);

    /// Thread group size, must be a power-of-two <= 1024.
    /// Must be kept in sync with shader side.
    static constexpr const uint32_t GROUP_SIZE = 1024;

    ref<sgl::Device> m_device;

    std::map<sgl::DataType, Pipelines> m_pipelines;

    /// Temporary buffer for prefix sum computation.
    ref<sgl::Buffer> m_prefix_group_sums;
    /// Temporary buffer for total sum of an iteration.
    ref<sgl::Buffer> m_total_sum;
    /// Temporary buffer for prev total sum of an iteration.
    ref<sgl::Buffer> m_prev_total_sum;
};

/// Computes the parallel prefix sum over an array of elements.
/// This supports int32, uint32, int64, uint64, float32 types.
/// @param command_encoder Command encoder.
/// @param type Data type of the elements.
/// @param data Data buffer and offset to compute prefix sum over.
/// @param element_count Number of elements to compute prefix sum over.
/// @param total_sum (Optional) Buffer and offset to which the total sum is copied.
void FALCOR_API prefix_sum(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> total_sum = {}
);

} // namespace falcor
