// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/denoisers/optix_types.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(denoisers_optix_types)
{
    using namespace falcor;
    using namespace falcor::optix;

    nb::sgl_enum<PixelFormat>(m, "OptixPixelFormat", D(optix, PixelFormat));
    nb::sgl_enum<DenoiserAOVType>(m, "OptixDenoiserAOVType", D(optix, DenoiserAOVType));
    nb::sgl_enum<ModelKind>(m, "OptixModelKind", D(optix, ModelKind));
    nb::sgl_enum<AlphaMode>(m, "OptixAlphaMode", D(optix, AlphaMode));

    nb::class_<Image2D>(m, "OptixImage2D", D(optix, Image2D))
        .def(nb::init<>())
        .DEF_RW_2(optix, Image2D, buffer)
        .DEF_RW_2(optix, Image2D, address)
        .DEF_RW_2(optix, Image2D, width)
        .DEF_RW_2(optix, Image2D, height)
        .DEF_RW_2(optix, Image2D, row_stride_in_bytes)
        .DEF_RW_2(optix, Image2D, pixel_stride_in_bytes)
        .DEF_RW_2(optix, Image2D, format);

    nb::class_<DenoiserLayer>(m, "OptixDenoiserLayer", D(optix, DenoiserLayer))
        .def(nb::init<>())
        .DEF_RW_2(optix, DenoiserLayer, input)
        .DEF_RW_2(optix, DenoiserLayer, previous_output)
        .DEF_RW_2(optix, DenoiserLayer, output)
        .DEF_RW_2(optix, DenoiserLayer, type);

    nb::class_<DenoiserGuideLayer>(m, "OptixDenoiserGuideLayer", D(optix, DenoiserGuideLayer))
        .def(nb::init<>())
        .DEF_RW_2(optix, DenoiserGuideLayer, albedo)
        .DEF_RW_2(optix, DenoiserGuideLayer, normal)
        .DEF_RW_2(optix, DenoiserGuideLayer, flow)
        .DEF_RW_2(optix, DenoiserGuideLayer, previous_output_internal_guide_layer)
        .DEF_RW_2(optix, DenoiserGuideLayer, output_internal_guide_layer)
        .DEF_RW_2(optix, DenoiserGuideLayer, flow_trustworthiness);

    nb::class_<DenoiserParams>(m, "OptixDenoiserParams", D(optix, DenoiserParams))
        .def(nb::init<>())
        .DEF_RW_2(optix, DenoiserParams, hdr_intensity, nb::none())
        .DEF_RW_2(optix, DenoiserParams, blend_factor)
        .DEF_RW_2(optix, DenoiserParams, hdr_average_color, nb::none())
        .DEF_RW_2(optix, DenoiserParams, temporal_mode_use_previous_layers);
}
