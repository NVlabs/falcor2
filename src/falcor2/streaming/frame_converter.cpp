// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "frame_converter.h"

#include <sgl/device/device.h>
#include <sgl/device/command.h>
#include <sgl/device/resource.h>
#include <sgl/device/shader_cursor.h>

namespace falcor::streaming {

FrameConverter::FrameConverter(sgl::Device* device)
{
    m_pipeline = device->create_compute_pipeline({
        .program = device->load_program("falcor2/streaming/convert.slang", {"convert_i420"}),
    });
}

void FrameConverter::convert(
    sgl::CommandEncoder* command_encoder,
    sgl::Texture* source,
    sgl::Buffer* output,
    uint32_t pitch,
    InputTransfer transfer
)
{
    auto pass = command_encoder->begin_compute_pass();
    auto root = pass->bind_pipeline(m_pipeline);
    sgl::ShaderCursor cursor(root);
    cursor["source"] = ref<sgl::Texture>(source);
    cursor["i420"] = ref<sgl::Buffer>(output);
    cursor["width"] = source->width();
    cursor["height"] = source->height();
    cursor["pitch"] = pitch;
    cursor["input_is_srgb"] = uint32_t(transfer == InputTransfer::srgb);
    pass->dispatch_compute({(pitch / 8 + 7) / 8, (source->height() / 2 + 7) / 8, 1});
    pass->end();
}

} // namespace falcor::streaming
