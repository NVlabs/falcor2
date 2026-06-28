// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/render/fwd.h"

namespace sgl {
class Buffer;
class Device;
} // namespace sgl

namespace falcor::materialx::mx139 {

struct MxDielectricLutBuffers {
    ref<sgl::Buffer> reflfront_energy;
    ref<sgl::Buffer> bothfront_energy;
    ref<sgl::Buffer> bothback_energy;
    ref<sgl::Buffer> bothback_fit_energy;
};

FALCOR_API MxDielectricLutBuffers create_mx_dielectric_lut_buffers(sgl::Device* device);

} // namespace falcor::materialx::mx139
