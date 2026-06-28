// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

namespace {

nb::list discover_materialx_renderable_elements_impl(
    const std::filesystem::path& path,
    falcor::ref<falcor::AssetResolver> asset_resolver
)
{
    falcor::materialx::CodeGenDesc desc;
    desc.document = path;
    desc.asset_resolver = std::move(asset_resolver);

    nb::list records;
    for (const falcor::materialx::RenderableElement& element :
         falcor::materialx::CodeGen::discover_renderable_elements(desc)) {
        nb::dict record;
        record["name"] = element.name;
        record["type"] = element.type;
        records.append(record);
    }
    return records;
}

} // namespace

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
            auto scene_globals = make_ref<materialx::mx139::LutSceneGlobals>(device);
            const materialx::mx139::LutSceneGlobals::Buffers& buffers = scene_globals->buffers();

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
        "discover_materialx_renderable_elements",
        &discover_materialx_renderable_elements_impl,
        "path"_a,
        "asset_resolver"_a.none() = nb::none(),
        "Discover renderable MaterialX elements and their MaterialX types."
    );
}
