// SPDX-License-Identifier: Apache-2.0

#include "prefix_sum.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/resource.h>
#include <sgl/device/pipeline.h>
#include <sgl/device/shader_cursor.h>
#include <sgl/core/maths.h>

#include <functional>
#include <map>

namespace falcor {

PrefixSum::PrefixSum(ref<sgl::Device> device)
    : m_device(device)
{
    // Create and bind buffer for per-group sums and total sum.
    const uint32_t max_element_size = 8;
    m_prefix_group_sums = m_device->create_buffer({
        .size = GROUP_SIZE * max_element_size,
        .usage = sgl::BufferUsage::shader_resource | sgl::BufferUsage::unordered_access,
        .label = "PrefixSum::prefix_group_sums",
    });
    m_total_sum = m_device->create_buffer({
        .size = max_element_size,
        .usage = sgl::BufferUsage::unordered_access,
        .label = "PrefixSum::total_sum",
    });
    m_prev_total_sum = m_device->create_buffer({
        .size = GROUP_SIZE * max_element_size,
        .usage = sgl::BufferUsage::shader_resource,
        .label = "PrefixSum::prev_total_sum",
    });
}

void PrefixSum::execute(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> total_sum
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);
    FALCOR_CHECK_NOT_NULL(data.buffer);
    FALCOR_CHECK_GT(element_count, 0);

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

    FALCOR_CHECK(data.buffer->size() >= data.offset + element_count * element_size, "data buffer too small.");

    const Pipelines& pipelines = get_pipelines(type);

    // Clear total sum to zero.
    command_encoder->clear_buffer(m_total_sum);

    uint32_t max_element_count_per_iteration = GROUP_SIZE * GROUP_SIZE * 2;
    uint32_t total_element_count = element_count;
    uint32_t iteration_count = sgl::div_round_up(total_element_count, max_element_count_per_iteration);

    for (uint32_t iteration = 0; iteration < iteration_count; iteration++) {
        // Compute number of thread groups in the first pass. Each thread operates on two elements.
        uint32_t group_count
            = std::max(1u, sgl::div_round_up(std::min(element_count, max_element_count_per_iteration), GROUP_SIZE * 2));
        FALCOR_ASSERT(group_count > 0 && group_count <= GROUP_SIZE);

        // Copy previous iterations total sum to read buffer.
        command_encoder->copy_buffer(m_prev_total_sum, 0, m_total_sum, 0, element_size);

        // Clear group sums to zero.
        command_encoder->clear_buffer(m_prefix_group_sums);

        ref<sgl::ComputePassEncoder> pass_encoder = command_encoder->begin_compute_pass();

        // Pass 1: compute per-thread group prefix sums.
        {
            sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.group_scan_pipeline);
            sgl::ShaderCursor cursor(root_object->get_entry_point(0));
            cursor["group_count"] = group_count;
            cursor["total_element_count"] = total_element_count;
            cursor["iteration"] = iteration;
            cursor["data"] = ref<sgl::Buffer>(data.buffer);
            cursor["data_offset"] = uint32_t(data.offset / 4);
            cursor["prefix_group_sums"] = m_prefix_group_sums;
            cursor["total_sum"] = m_total_sum;
            cursor["prev_total_sum"] = m_prev_total_sum;
            pass_encoder->dispatch_compute({group_count, 1, 1});
        }

        // Pass 2: finalize prefix sum by adding the sums to the left to each group.
        // This is only necessary if we have more than one group.
        if (group_count > 1) {
            sgl::ShaderObject* root_object = pass_encoder->bind_pipeline(pipelines.finalize_pipeline);
            sgl::ShaderCursor cursor(root_object->get_entry_point(0));
            cursor["total_element_count"] = total_element_count;
            cursor["iteration"] = iteration;
            cursor["data"] = ref<sgl::Buffer>(data.buffer);
            cursor["data_offset"] = uint32_t(data.offset / 4);
            cursor["prefix_group_sums"] = m_prefix_group_sums;

            // Note that we're skipping the first group of 2N elements, as no add is needed (their group sum is zero).
            pass_encoder->dispatch_compute({(group_count - 1) * 2, 1, 1});
        }

        pass_encoder->end();

        // Subtract the number of elements handled this iteration.
        element_count -= max_element_count_per_iteration;
    }

    // Copy total sum to separate destination buffer, if specified.
    if (total_sum) {
        FALCOR_CHECK(total_sum->offset + element_size <= total_sum->buffer->size(), "total_sum buffer is too small.");
        command_encoder->copy_buffer(total_sum->buffer, total_sum->offset, m_total_sum, 0, element_size);
    }
}

const PrefixSum::Pipelines& PrefixSum::get_pipelines(sgl::DataType type)
{
    auto it = m_pipelines.find(type);
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

    // D3D12 doesn't support atomics on RWStructuredBuffer<float>.
    bool use_float_byte_address_buffer = m_device->type() == sgl::DeviceType::d3d12 && type == sgl::DataType::float32;

    std::string_view type_name = get_type_name(type);
    std::string module_name = fmt::format("prefix-sum-{}", type_name);
    std::string module_source = fmt::format(
        "#define TYPE {}\n"
        "#define USE_FLOAT_BYTE_ADDRESS_BUFFER {}\n\n",
        type_name,
        use_float_byte_address_buffer ? 1 : 0
    );
    module_source += m_device->slang_session()->load_source("falcor2/utils/algorithm/prefix_sum.slang");

    ref<sgl::SlangModule> module = m_device->slang_session()->load_module_from_source(module_name, module_source);

    // Create pipelines.
    Pipelines pipelines;
    pipelines.group_scan_pipeline = m_device->create_compute_pipeline({
        .program = m_device->link_program({module}, {module->entry_point("group_scan")}),
    });
    pipelines.finalize_pipeline = m_device->create_compute_pipeline({
        .program = m_device->link_program({module}, {module->entry_point("finalize")}),
    });
    m_pipelines.emplace(type, pipelines);
    return m_pipelines[type];
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

void FALCOR_API prefix_sum(
    sgl::CommandEncoder* command_encoder,
    sgl::DataType type,
    sgl::BufferOffsetPair data,
    uint32_t element_count,
    std::optional<sgl::BufferOffsetPair> total_sum
)
{
    FALCOR_CHECK_NOT_NULL(command_encoder);

    ref<PrefixSum> prefix_sum = cache_per_device<ref<PrefixSum>>(
        command_encoder->device(),
        [](sgl::Device* device)
        {
            return make_ref<PrefixSum>(ref<sgl::Device>(device));
        }
    );
    prefix_sum->execute(command_encoder, type, data, element_count, total_sum);
}

} // namespace falcor
