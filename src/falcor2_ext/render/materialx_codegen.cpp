// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/core/asset_resolver.h"
#include "falcor2/render/material/materialx/codegen/codegen.h"
#include "falcor2/render/material/materialx/mx139/lut_globals.h"

#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/vector.h>

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <utility>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(render_materialx_codegen)
{
    using namespace falcor;

    nb::sgl_enum_flags<materialx::OptimizeGraphFlags>(m, "MaterialXOptimizeGraphFlags", D_NA(OptimizeGraphFlags));
    nb::sgl_enum<materialx::LayeringMode>(m, "MaterialXLayeringMode", D_NA(LayeringMode));
    nb::sgl_enum<materialx::CompensationMode>(m, "MaterialXCompensationMode", D_NA(CompensationMode));

    m.def(
        "create_materialx_mx139_lut_bindings",
        [](sgl::Device* device)
        {
            materialx::mx139::LutBuffers buffers = materialx::mx139::create_lut_buffers(device);

            nb::dict lut_globals;
            lut_globals["mini_microfacet_ggx_energy"] = buffers.mini_microfacet_ggx_energy;
            lut_globals["dielectric_reflfront_energy"] = buffers.dielectric_reflfront_energy;
            lut_globals["dielectric_bothfront_energy"] = buffers.dielectric_bothfront_energy;
            lut_globals["dielectric_bothback_energy"] = buffers.dielectric_bothback_energy;
            lut_globals["zeltner_sheen_ltc_param"] = buffers.zeltner_sheen_ltc_param;

            nb::dict result;
            result["lut_globals"] = lut_globals;
            return result;
        },
        "device"_a
    );

    m.def(
        "discover_materialx_material_node_names",
        [](const std::filesystem::path& path, ref<AssetResolver> asset_resolver)
        {
            materialx::CodeGenDesc desc;
            desc.document = path;
            desc.asset_resolver = std::move(asset_resolver);
            return materialx::CodeGen::discover_material_node_names(desc);
        },
        "path"_a,
        "asset_resolver"_a.none() = nb::none()
    );
}
