// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "parallel_reduction.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/resource.h>
#include <sgl/device/pipeline.h>
#include <sgl/device/shader_cursor.h>
#include <sgl/device/formats.h>
#include <sgl/core/maths.h>

#include <functional>
#include <limits>
#include <map>

namespace falcor {

// Maximum number of groups for intermediate buffer sizing.
// For buffer path: ceil(max_elements / (GROUP_SIZE * ELEMENTS_PER_THREAD))
// We pre-allocate for up to ~16M elements (conservative).
static constexpr uint32_t MAX_INTERMEDIATE_GROUPS = 2048;
static constexpr uint32_t MAX_ELEMENT_SIZE = 16; // vec4 for texture path (4 * 4 bytes)
static constexpr uint32_t MAX_OPS = 3;
static constexpr uint32_t MAX_STATE_SIZE = MAX_OPS * MAX_ELEMENT_SIZE; // worst case struct stride

enum : uint32_t {
    OP_SUM = 1,
    OP_MIN = 2,
    OP_MAX = 4,
};

/// Compute the byte offset of a field within the ReduceState/TexReduceState struct.
/// Fields are laid out in order: sum (if enabled), min (if enabled), max (if enabled).
static uint32_t field_offset(uint32_t op_mask, uint32_t op_bit, uint32_t field_size)
{
    uint32_t offset = 0;
    if (op_bit == OP_SUM)
        return 0;
    if (op_mask & OP_SUM)
        offset += field_size;
    if (op_bit == OP_MIN)
        return offset;
    if (op_mask & OP_MIN)
        offset += field_size;
    return offset;
}

static bool has_buffer_range(const sgl::BufferOffsetPair& range, uint64_t byte_count)
{
    if (!range.buffer)
        return false;

    uint64_t buffer_size = range.buffer->size();
    return range.offset <= buffer_size && byte_count <= buffer_size - range.offset;
}

ParallelReduction::ParallelReduction(ref<sgl::Device> device)
    : m_device(device)
{
    sgl::BufferUsage usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access;
    uint64_t intermediate_size = uint64_t(MAX_INTERMEDIATE_GROUPS) * uint64_t(MAX_STATE_SIZE);

    for (int i = 0; i < 2; i++) {
        m_intermediate[i] = m_device->create_buffer({
            .size = intermediate_size,
            .usage = usage,
            .label = fmt::format("ParallelReduction::intermediate[{}]", i),
        });
    }
}

void ParallelReduction::execute(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> sum,
    std::optional<sgl::BufferOffsetPair> min,
    std::optional<sgl::BufferOffsetPair> max
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);
    FALCOR_CHECK_NOT_NULL(data.buffer);
    FALCOR_CHECK_GT(element_count, 0);
    FALCOR_CHECK(sum || min || max, "At least one of sum, min, or max must be provided.");

    uint32_t element_size = 0;
    switch (type) {
    case sgl::DataType::int32:
    case sgl::DataType::uint32:
    case sgl::DataType::float32:
        element_size = 4;
        break;
    case sgl::DataType::int64:
    case sgl::DataType::uint64:
        element_size = 8;
        break;
    default:
        FALCOR_THROW("Unsupported data type, only int32, uint32, int64, uint64 and float32 are supported.");
    }

    FALCOR_CHECK(data.offset % element_size == 0, "data offset must be aligned to element size.");
    uint64_t data_byte_size = uint64_t(element_count) * uint64_t(element_size);
    FALCOR_CHECK(has_buffer_range(data, data_byte_size), "data buffer too small.");
    uint64_t data_element_offset = data.offset / element_size;

    uint32_t op_mask = 0;
    if (sum)
        op_mask |= OP_SUM;
    if (min)
        op_mask |= OP_MIN;
    if (max)
        op_mask |= OP_MAX;

    PipelineKey key{static_cast<uint32_t>(type), op_mask, false};
    const Pipelines& pipelines = get_pipelines(key);

    // Compute group count for initial pass.
    uint32_t elements_per_group = GROUP_SIZE * ELEMENTS_PER_THREAD;
    uint32_t group_count = sgl::div_round_up(element_count, elements_per_group);
    FALCOR_CHECK(
        group_count <= MAX_INTERMEDIATE_GROUPS,
        "Parallel reduction requires {} intermediate groups, but only {} are supported.",
        group_count,
        MAX_INTERMEDIATE_GROUPS
    );

    // Initial buffer_reduce pass.
    {
        ref<sgl::ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();
        sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.reduce_pipeline);
        sgl::ShaderCursor cursor(root_object->get_entry_point(0));
        cursor["element_count"] = element_count;
        cursor["data"] = ref<sgl::Buffer>(data.buffer);
        cursor["data_offset"] = uint32_t(data_element_offset);
        cursor["output"] = m_intermediate[0];
        pass_encoder->dispatch_compute({group_count, 1, 1});
        pass_encoder->end();
    }

    // Iterative final_reduce passes until we have 1 group.
    uint32_t current_count = group_count;
    uint32_t ping = 0;
    while (current_count > 1) {
        uint32_t next_group_count = sgl::div_round_up(current_count, elements_per_group);
        uint32_t pong = 1 - ping;

        ref<sgl::ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();
        sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.final_reduce_pipeline);
        sgl::ShaderCursor cursor(root_object->get_entry_point(0));
        cursor["element_count"] = current_count;
        cursor["input"] = m_intermediate[ping];
        cursor["output"] = m_intermediate[pong];
        pass_encoder->dispatch_compute({next_group_count, 1, 1});
        pass_encoder->end();

        current_count = next_group_count;
        ping = pong;
    }

    // Copy results from the combined struct to user-specified destinations.
    if (sum) {
        uint32_t offset = field_offset(op_mask, OP_SUM, element_size);
        FALCOR_CHECK(has_buffer_range(*sum, element_size), "sum buffer is too small.");
        command_encoder->copy_buffer(sum->buffer, sum->offset, m_intermediate[ping], offset, element_size);
    }
    if (min) {
        uint32_t offset = field_offset(op_mask, OP_MIN, element_size);
        FALCOR_CHECK(has_buffer_range(*min, element_size), "min buffer is too small.");
        command_encoder->copy_buffer(min->buffer, min->offset, m_intermediate[ping], offset, element_size);
    }
    if (max) {
        uint32_t offset = field_offset(op_mask, OP_MAX, element_size);
        FALCOR_CHECK(has_buffer_range(*max, element_size), "max buffer is too small.");
        command_encoder->copy_buffer(max->buffer, max->offset, m_intermediate[ping], offset, element_size);
    }
}

void ParallelReduction::execute(
    sgl::CommandEncoder* command_encoder,
    sgl::Texture* input,
    std::optional<sgl::BufferOffsetPair> sum,
    std::optional<sgl::BufferOffsetPair> min,
    std::optional<sgl::BufferOffsetPair> max
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);
    FALCOR_CHECK_NOT_NULL(input);
    FALCOR_CHECK(sum || min || max, "At least one of sum, min, or max must be provided.");

    sgl::Format format = input->format();
    const sgl::FormatInfo& format_info = sgl::get_format_info(format);

    FALCOR_CHECK(!format_info.is_compressed, "Compressed texture formats are not supported.");
    FALCOR_CHECK(!format_info.is_depth_stencil(), "Depth/stencil texture formats are not supported.");

    uint32_t width = input->width();
    uint32_t height = input->height();
    uint32_t channel_count = format_info.channel_count;
    FALCOR_CHECK(width > 0 && height > 0, "input texture must be non-empty.");

    // Determine format type for shader.
    // float/unorm/snorm -> float4, sint -> int4, uint -> uint4
    // Result is always 16 bytes (vec4).
    uint32_t result_size = 16;

    uint32_t op_mask = 0;
    if (sum)
        op_mask |= OP_SUM;
    if (min)
        op_mask |= OP_MIN;
    if (max)
        op_mask |= OP_MAX;

    // Compute tile count.
    uint32_t tiles_x = sgl::div_round_up(width, TILE_SIZE);
    uint32_t tiles_y = sgl::div_round_up(height, TILE_SIZE);
    uint64_t tile_count_64 = uint64_t(tiles_x) * uint64_t(tiles_y);
    FALCOR_CHECK(
        tile_count_64 <= MAX_INTERMEDIATE_GROUPS,
        "Parallel texture reduction requires {} intermediate tiles, but only {} are supported.",
        tile_count_64,
        MAX_INTERMEDIATE_GROUPS
    );
    uint32_t tile_count = uint32_t(tile_count_64);

    // Encode format config: format_type (8 bits) | channel_count (8 bits) | format enum (8 bits)
    uint32_t format_type_value = static_cast<uint32_t>(format_info.type);
    uint32_t format_value = static_cast<uint32_t>(format);
    FALCOR_CHECK(format_type_value <= 0xFF, "Texture format type exceeds 8-bit PipelineKey encoding.");
    FALCOR_CHECK(format_value <= 0xFF, "Texture format enum value exceeds 8-bit PipelineKey encoding.");
    uint32_t type_key = (format_type_value << 16) | (channel_count << 8) | format_value;
    PipelineKey key{type_key, op_mask, true};
    const Pipelines& pipelines = get_pipelines(key);

    // Initial texture_reduce pass (2D dispatch).
    {
        ref<sgl::ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();
        sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.reduce_pipeline);
        sgl::ShaderCursor cursor(root_object->get_entry_point(0));
        cursor["tex_width"] = width;
        cursor["tex_height"] = height;
        cursor["input_texture"] = ref<sgl::Texture>(input);
        cursor["output"] = m_intermediate[0];
        cursor["tiles_x"] = tiles_x;
        pass_encoder->dispatch_compute({tiles_x, tiles_y, 1});
        pass_encoder->end();
    }

    // Iterative final reduction passes.
    uint32_t elements_per_group = GROUP_SIZE * ELEMENTS_PER_THREAD;
    uint32_t current_count = tile_count;
    uint32_t ping = 0;
    while (current_count > 1) {
        uint32_t next_group_count = sgl::div_round_up(current_count, elements_per_group);
        uint32_t pong = 1 - ping;

        ref<sgl::ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();
        sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.final_reduce_pipeline);
        sgl::ShaderCursor cursor(root_object->get_entry_point(0));
        cursor["element_count"] = current_count;
        cursor["input"] = m_intermediate[ping];
        cursor["output"] = m_intermediate[pong];
        pass_encoder->dispatch_compute({next_group_count, 1, 1});
        pass_encoder->end();

        current_count = next_group_count;
        ping = pong;
    }

    // Copy results from the combined struct to user-specified destinations.
    if (sum) {
        uint32_t offset = field_offset(op_mask, OP_SUM, result_size);
        FALCOR_CHECK(has_buffer_range(*sum, result_size), "sum buffer is too small.");
        command_encoder->copy_buffer(sum->buffer, sum->offset, m_intermediate[ping], offset, result_size);
    }
    if (min) {
        uint32_t offset = field_offset(op_mask, OP_MIN, result_size);
        FALCOR_CHECK(has_buffer_range(*min, result_size), "min buffer is too small.");
        command_encoder->copy_buffer(min->buffer, min->offset, m_intermediate[ping], offset, result_size);
    }
    if (max) {
        uint32_t offset = field_offset(op_mask, OP_MAX, result_size);
        FALCOR_CHECK(has_buffer_range(*max, result_size), "max buffer is too small.");
        command_encoder->copy_buffer(max->buffer, max->offset, m_intermediate[ping], offset, result_size);
    }
}

const ParallelReduction::Pipelines& ParallelReduction::get_pipelines(PipelineKey key)
{
    auto it = m_pipelines.find(key);
    if (it != m_pipelines.end())
        return it->second;

    auto get_type_name = [](sgl::DataType type) -> std::string_view
    {
        switch (type) {
        case sgl::DataType::int32:
            return "int32_t";
        case sgl::DataType::uint32:
            return "uint32_t";
        case sgl::DataType::int64:
            return "int64_t";
        case sgl::DataType::uint64:
            return "uint64_t";
        case sgl::DataType::float32:
            return "float32_t";
        default:
            FALCOR_ASSERT(false);
            return {};
        }
    };

    std::string module_source;
    std::string module_name;

    // Build defines for op selection.
    std::string op_defines;
    if (key.op_mask & OP_SUM)
        op_defines += "#define REDUCE_SUM\n";
    if (key.op_mask & OP_MIN)
        op_defines += "#define REDUCE_MIN\n";
    if (key.op_mask & OP_MAX)
        op_defines += "#define REDUCE_MAX\n";

    if (key.is_texture) {
        // Decode format config.
        uint32_t format_type_val = (key.type_key >> 16) & 0xFF;
        uint32_t channel_count = (key.type_key >> 8) & 0xFF;
        sgl::FormatType format_type = static_cast<sgl::FormatType>(format_type_val);

        std::string_view vec_type;
        std::string_view scalar_type;
        switch (format_type) {
        case sgl::FormatType::float_:
        case sgl::FormatType::unorm:
        case sgl::FormatType::unorm_srgb:
        case sgl::FormatType::snorm:
            vec_type = "float4";
            scalar_type = "float32_t";
            break;
        case sgl::FormatType::sint:
            vec_type = "int4";
            scalar_type = "int32_t";
            break;
        case sgl::FormatType::uint:
            vec_type = "uint4";
            scalar_type = "uint32_t";
            break;
        default:
            FALCOR_THROW("Unsupported texture format type.");
        }

        module_name = fmt::format("parallel-reduction-tex-{}-ch{}-ops{}", scalar_type, channel_count, key.op_mask);
        module_source = fmt::format(
            "#define TYPE {}\n"
            "#define IS_TEXTURE\n"
            "#define FORMAT_TYPE {}\n"
            "{}\n",
            scalar_type,
            vec_type,
            op_defines
        );
    } else {
        sgl::DataType type = static_cast<sgl::DataType>(key.type_key);
        std::string_view type_name = get_type_name(type);
        module_name = fmt::format("parallel-reduction-{}-ops{}", type_name, key.op_mask);
        module_source = fmt::format(
            "#define TYPE {}\n"
            "{}\n",
            type_name,
            op_defines
        );
    }

    module_source += m_device->slang_session()->load_source("falcor2/utils/algorithm/parallel_reduction.slang");

    ref<sgl::SlangModule> module = m_device->slang_session()->load_module_from_source(module_name, module_source);

    Pipelines pipelines;
    if (key.is_texture) {
        pipelines.reduce_pipeline = m_device->create_compute_pipeline({
            .program = m_device->link_program({module}, {module->entry_point("texture_reduce")}),
        });
        pipelines.final_reduce_pipeline = m_device->create_compute_pipeline({
            .program = m_device->link_program({module}, {module->entry_point("texture_final_reduce")}),
        });
    } else {
        pipelines.reduce_pipeline = m_device->create_compute_pipeline({
            .program = m_device->link_program({module}, {module->entry_point("buffer_reduce")}),
        });
        pipelines.final_reduce_pipeline = m_device->create_compute_pipeline({
            .program = m_device->link_program({module}, {module->entry_point("final_reduce")}),
        });
    }

    m_pipelines.emplace(key, pipelines);
    return m_pipelines[key];
}

// TODO: This should probably be a common utility. Might even be moved into sgl.
template<typename T>
T cache_per_device(sgl::Device* device, std::function<T(sgl::Device*)> factory)
{
    static std::map<sgl::Device*, T> s_cache;
    auto it = s_cache.find(device);
    if (it != s_cache.end())
        return it->second;

    T object = factory(device);
    s_cache.emplace(device, object);

    device->register_device_close_callback(
        [&](sgl::Device* device)
        {
            s_cache.erase(device);
        }
    );

    return s_cache[device];
}

void FALCOR_API parallel_reduce(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> sum,
    std::optional<sgl::BufferOffsetPair> min,
    std::optional<sgl::BufferOffsetPair> max
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);

    ref<ParallelReduction> reduction = cache_per_device<ref<ParallelReduction>>(
        command_encoder->device(),
        [](sgl::Device* device)
        {
            return make_ref<ParallelReduction>(ref<sgl::Device>(device));
        }
    );
    reduction->execute(command_encoder, type, data, element_count, sum, min, max);
}

void FALCOR_API parallel_reduce(
    sgl::CommandEncoder* command_encoder,
    sgl::Texture* input,
    std::optional<sgl::BufferOffsetPair> sum,
    std::optional<sgl::BufferOffsetPair> min,
    std::optional<sgl::BufferOffsetPair> max
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);

    ref<ParallelReduction> reduction = cache_per_device<ref<ParallelReduction>>(
        command_encoder->device(),
        [](sgl::Device* device)
        {
            return make_ref<ParallelReduction>(ref<sgl::Device>(device));
        }
    );
    reduction->execute(command_encoder, input, sum, min, max);
}

} // namespace falcor
