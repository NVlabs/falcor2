// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/object.h"
#include "falcor2/render/fwd.h"

namespace sgl {
class Buffer;
class Device;
class ShaderCursor;
} // namespace sgl

namespace falcor {
namespace materialx {
namespace mx139 {

struct LutBuffers {
    ref<sgl::Buffer> mini_microfacet_ggx_energy;
    ref<sgl::Buffer> dielectric_reflfront_energy;
    ref<sgl::Buffer> dielectric_bothfront_energy;
    ref<sgl::Buffer> dielectric_bothback_energy;
    ref<sgl::Buffer> zeltner_sheen_ltc_param;
};

FALCOR_API LutBuffers create_lut_buffers(sgl::Device* device);
FALCOR_API ref<SceneGlobals> create_lut_scene_globals(sgl::Device* device);
FALCOR_API void bind_lut_globals(sgl::ShaderCursor globals, const LutBuffers& buffers);

} // namespace mx139
} // namespace materialx
} // namespace falcor
