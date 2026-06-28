// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/render/fwd.h"
#include "falcor2/render/scene_globals.h"

#include <string>
#include <string_view>

namespace sgl {
class Buffer;
class Device;
class ShaderCursor;
} // namespace sgl

namespace falcor {
namespace materialx {
namespace mx139 {

class FALCOR_API LutSceneGlobals : public SceneGlobals {
    FALCOR_OBJECT(LutSceneGlobals)
public:
    struct Buffers {
        ref<sgl::Buffer> mini_microfacet_ggx_energy;
        ref<sgl::Buffer> dielectric_reflfront_energy;
        ref<sgl::Buffer> dielectric_bothfront_energy;
        ref<sgl::Buffer> dielectric_bothback_energy;
        ref<sgl::Buffer> zeltner_sheen_ltc_param;
    };

    LutSceneGlobals(sgl::Device* device, std::string_view global_block_name = "lut_globals");

    const Buffers& buffers() const { return m_buffers; }

    void bind(sgl::ShaderCursor globals) const override;

private:
    Buffers m_buffers;
    std::string m_global_block_name;
};

} // namespace mx139
} // namespace materialx
} // namespace falcor
