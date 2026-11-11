// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "video_encoder.h"

#include <sgl/device/pipeline.h>

namespace falcor::streaming {

/// Records GPU conversion from display-referred SDR RGB to limited-range BT.709 I420.
/// Linear input receives the sRGB transfer function; sRGB input is already encoded.
class FrameConverter {
public:
    /// Creates the conversion pipeline on the device used for all subsequent resources.
    explicit FrameConverter(sgl::Device* device);

    /// Appends a conversion pass without submitting or waiting for GPU execution.
    /// Source dimensions must be even. Pitch is the Y-plane row stride in bytes,
    /// at least the source width and a multiple of eight; U/V use half that stride.
    /// Output must support unordered access and hold pitch * height * 3 / 2 bytes.
    /// The caller keeps resources alive until the recorded work completes.
    void convert(
        sgl::CommandEncoder* command_encoder,
        sgl::Texture* source,
        sgl::Buffer* output,
        uint32_t pitch,
        InputTransfer transfer
    );

private:
    ref<sgl::ComputePipeline> m_pipeline;
};

} // namespace falcor::streaming
