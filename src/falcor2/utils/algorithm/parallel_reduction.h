// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>
#include <sgl/device/resource.h>

#include <map>
#include <optional>

namespace falcor {

/// Utility for computing parallel reduction (sum, min, max) on buffers and textures.
/// Results are written to user-specified BufferOffsetPair destinations (GPU-side, not host readback).
/// At least one of sum/min/max must be provided.
class FALCOR_API ParallelReduction : public Object {
    FALCOR_OBJECT(ParallelReduction)
public:
    /// Constructor.
    ParallelReduction(ref<sgl::Device> device);

    /// Computes parallel reduction over a buffer of scalar elements.
    /// Supports int32, uint32, int64, uint64, float32 types.
    /// element_count must be greater than zero.
    /// Each provided output writes one scalar of the requested type at the given offset.
    /// int32, uint32, and float32 outputs reserve 4 bytes and require 4-byte aligned offsets.
    /// int64 and uint64 outputs reserve 8 bytes and require 8-byte aligned offsets.
    /// At least one of sum/min/max must be provided.
    /// @param command_encoder Command encoder.
    /// @param type Data type of the elements.
    /// @param data Data buffer and byte offset to reduce. The offset must be aligned to the element size.
    /// @param element_count Number of elements to reduce.
    /// @param sum (Optional) Buffer and offset to which the sum is written.
    /// @param min (Optional) Buffer and offset to which the min is written.
    /// @param max (Optional) Buffer and offset to which the max is written.
    void execute(
        sgl::CommandEncoder* command_encoder,
        sgl::DataType type,
        sgl::BufferOffsetPair data,
        uint32_t element_count,
        std::optional<sgl::BufferOffsetPair> sum = {},
        std::optional<sgl::BufferOffsetPair> min = {},
        std::optional<sgl::BufferOffsetPair> max = {}
    );

    /// Computes parallel reduction over a 2D texture.
    /// Results are vec4 (float4 for float/unorm/snorm, int4 for sint, uint4 for uint).
    /// Each provided output writes one 16-byte vec4 and requires a 4-byte aligned offset.
    /// At least one of sum/min/max must be provided.
    /// @param command_encoder Command encoder.
    /// @param input Input non-empty 2D texture (non-owning).
    /// @param sum (Optional) Buffer and offset to which the sum is written (16 bytes).
    /// @param min (Optional) Buffer and offset to which the min is written (16 bytes).
    /// @param max (Optional) Buffer and offset to which the max is written (16 bytes).
    void execute(
        sgl::CommandEncoder* command_encoder,
        sgl::Texture* input,
        std::optional<sgl::BufferOffsetPair> sum = {},
        std::optional<sgl::BufferOffsetPair> min = {},
        std::optional<sgl::BufferOffsetPair> max = {}
    );

private:
    struct Pipelines {
        ref<sgl::ComputePipeline> reduce_pipeline;
        ref<sgl::ComputePipeline> final_reduce_pipeline;
    };

    /// Key for pipeline cache: (data type or format config, op mask).
    struct PipelineKey {
        /// DataType enum value for buffers, or encoded format config for textures.
        uint32_t type_key;
        uint32_t op_mask;
        bool is_texture;

        auto operator<=>(const PipelineKey&) const = default;
    };

    const Pipelines& get_pipelines(PipelineKey key);

    /// Thread group size for buffer reduction.
    static constexpr uint32_t GROUP_SIZE = 1024;
    /// Elements processed per thread in sequential accumulation.
    static constexpr uint32_t ELEMENTS_PER_THREAD = 8;
    /// Tile size for texture reduction (32x32 threads).
    static constexpr uint32_t TILE_SIZE = 32;
    /// Minimum wave size (conservative for shared memory sizing).
    static constexpr uint32_t MIN_WAVE_SIZE = 32;

    ref<sgl::Device> m_device;

    std::map<PipelineKey, Pipelines> m_pipelines;

    /// Intermediate ping-pong buffers (stores ReduceState/TexReduceState elements).
    ref<sgl::Buffer> m_intermediate[2];
};

/// Computes parallel reduction over a buffer of scalar elements.
/// Supports int32, uint32, int64, uint64, float32 types.
/// element_count must be greater than zero.
/// Each provided output writes one scalar of the requested type at the given offset.
/// int32, uint32, and float32 outputs reserve 4 bytes and require 4-byte aligned offsets.
/// int64 and uint64 outputs reserve 8 bytes and require 8-byte aligned offsets.
/// At least one of sum/min/max must be provided.
/// @param command_encoder Command encoder.
/// @param type Data type of the elements.
/// @param data Data buffer and byte offset to reduce. The offset must be aligned to the element size.
/// @param element_count Number of elements to reduce.
/// @param sum (Optional) Buffer and offset to which the sum is written.
/// @param min (Optional) Buffer and offset to which the min is written.
/// @param max (Optional) Buffer and offset to which the max is written.
/// @return Nothing.
void FALCOR_API parallel_reduce(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> sum = {},
    std::optional<sgl::BufferOffsetPair> min = {},
    std::optional<sgl::BufferOffsetPair> max = {}
);

/// Computes parallel reduction over a 2D texture.
/// Results are vec4 (float4 for float/unorm/snorm, int4 for sint, uint4 for uint).
/// Each provided output writes one 16-byte vec4 and requires a 4-byte aligned offset.
/// At least one of sum/min/max must be provided.
/// @param command_encoder Command encoder.
/// @param input Input non-empty 2D texture (non-owning).
/// @param sum (Optional) Buffer and offset to which the sum is written (16 bytes).
/// @param min (Optional) Buffer and offset to which the min is written (16 bytes).
/// @param max (Optional) Buffer and offset to which the max is written (16 bytes).
/// @return Nothing.
void FALCOR_API parallel_reduce(
    sgl::CommandEncoder* command_encoder,
    sgl::Texture* input,
    std::optional<sgl::BufferOffsetPair> sum = {},
    std::optional<sgl::BufferOffsetPair> min = {},
    std::optional<sgl::BufferOffsetPair> max = {}
);

} // namespace falcor
