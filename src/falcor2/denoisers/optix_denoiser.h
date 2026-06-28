// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/denoisers/optix_types.h"

#include <sgl/device/fwd.h>
#include "sgl/device/native_handle.h"

#include <memory>
#include <span>

namespace falcor {

struct OptixDenoiserDesc {
    /// OptiX denoiser model kind.
    optix::ModelKind model_kind{optix::ModelKind::aov};
    /// Alpha processing mode.
    optix::AlphaMode alpha_mode{optix::AlphaMode::copy};
    /// Whether to use albedo guide layer.
    bool albedo_guide_layer{false};
    /// Whether to use normal guide layer.
    bool normal_guide_layer{false};
    /// Maximum image width for denoising.
    uint32_t max_width{1920};
    /// Maximum image height for denoising.
    uint32_t max_height{1080};
};

/// OptiX denoiser wrapper.
class FALCOR_API OptixDenoiser : public Object {
    FALCOR_OBJECT(OptixDenoiser)

public:
    class Impl;

    /// Create an OptiX denoiser.
    /// @param device The device to use for denoising operations.
    /// @param desc Descriptor for denoiser initialization.
    OptixDenoiser(ref<sgl::Device> device, const OptixDenoiserDesc& desc);

    /// Destructor.
    ~OptixDenoiser();

    /// Denoise an image.
    /// @param params Denoiser parameters.
    /// @param guide_layer Guide layer containing auxiliary data (leave empty if not used).
    /// @param layers List of layers to denoise.
    /// @param cuda_stream Optional CUDA stream to use for denoising (default: null stream).
    void denoise(
        const optix::DenoiserParams& params,
        const optix::DenoiserGuideLayer& guide_layer,
        std::span<optix::DenoiserLayer> layers,
        sgl::NativeHandle cuda_stream = {}
    );

    /// Device used by this denoiser.
    sgl::Device* device() const { return m_device; }

private:
    ref<sgl::Device> m_device;
    std::unique_ptr<Impl> m_impl;
};

} // namespace falcor
