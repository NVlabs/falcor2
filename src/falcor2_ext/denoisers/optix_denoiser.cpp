// SPDX-License-Identifier: Apache-2.0

#include "nanobind.h"

#include "falcor2/denoisers/optix_denoiser.h"

#include <sgl/device/device.h>
#include <sgl/device/resource.h>

namespace nb = nanobind;
using namespace nb::literals;

FALCOR_PY_EXPORT(denoisers_optix_denoiser)
{
    using namespace falcor;
    using namespace falcor::optix;

    nb::class_<OptixDenoiserDesc>(m, "OptixDenoiserDesc", D(OptixDenoiserDesc))
        .def(nb::init<>())
        .DEF_RW(OptixDenoiserDesc, model_kind)
        .DEF_RW(OptixDenoiserDesc, alpha_mode)
        .DEF_RW(OptixDenoiserDesc, albedo_guide_layer)
        .DEF_RW(OptixDenoiserDesc, normal_guide_layer)
        .DEF_RW(OptixDenoiserDesc, max_width)
        .DEF_RW(OptixDenoiserDesc, max_height);

    nb::class_<OptixDenoiser>(m, "OptixDenoiser", D(OptixDenoiser))
        .def(
            nb::init<ref<sgl::Device>, const OptixDenoiserDesc&>(),
            "device"_a,
            "desc"_a,
            D(OptixDenoiser, OptixDenoiser)
        )
        .def(
            "denoise",
            &OptixDenoiser::denoise,
            "params"_a,
            "guide_layer"_a,
            "layers"_a,
            "cuda_stream"_a = sgl::NativeHandle{},
            D(OptixDenoiser, denoise)
        )
        .def(
            "denoise",
            [](OptixDenoiser& self,
               ref<sgl::Buffer> input,
               ref<sgl::Buffer> output,
               uint32_t width,
               uint32_t height,
               ref<sgl::Buffer> albedo,
               ref<sgl::Buffer> normal,
               ref<sgl::Buffer> flow,
               std::optional<float> hdr_intensity,
               float blend_factor,
               std::optional<float3> hdr_average_color,
               uint32_t temporal_mode_use_previous_layers,
               PixelFormat format,
               DenoiserAOVType aov_type,
               sgl::NativeHandle cuda_stream)
            {
                // Set up denoiser parameters
                DenoiserParams params;
                params.hdr_intensity = hdr_intensity;
                params.blend_factor = blend_factor;
                params.hdr_average_color = hdr_average_color;
                params.temporal_mode_use_previous_layers = temporal_mode_use_previous_layers;

                // Set up guide layer
                DenoiserGuideLayer guide_layer;
                if (albedo) {
                    guide_layer.albedo.buffer = albedo;
                    guide_layer.albedo.width = width;
                    guide_layer.albedo.height = height;
                    guide_layer.albedo.row_stride_in_bytes = width * 3 * sizeof(float);
                    guide_layer.albedo.pixel_stride_in_bytes = 3 * sizeof(float);
                    guide_layer.albedo.format = PixelFormat::float3;
                }
                if (normal) {
                    guide_layer.normal.buffer = normal;
                    guide_layer.normal.width = width;
                    guide_layer.normal.height = height;
                    guide_layer.normal.row_stride_in_bytes = width * 3 * sizeof(float);
                    guide_layer.normal.pixel_stride_in_bytes = 3 * sizeof(float);
                    guide_layer.normal.format = PixelFormat::float3;
                }
                if (flow) {
                    guide_layer.flow.buffer = flow;
                    guide_layer.flow.width = width;
                    guide_layer.flow.height = height;
                    guide_layer.flow.row_stride_in_bytes = width * 2 * sizeof(float);
                    guide_layer.flow.pixel_stride_in_bytes = 2 * sizeof(float);
                    guide_layer.flow.format = PixelFormat::float2;
                }

                // Set up denoising layer
                uint32_t channels = (format == PixelFormat::float3) ? 3 : 4;
                uint32_t pixel_size = channels * sizeof(float);
                std::vector<DenoiserLayer> layers(1);
                layers[0].input.buffer = input;
                layers[0].input.width = width;
                layers[0].input.height = height;
                layers[0].input.row_stride_in_bytes = width * pixel_size;
                layers[0].input.pixel_stride_in_bytes = static_cast<uint16_t>(pixel_size);
                layers[0].input.format = format;
                layers[0].output.buffer = output;
                layers[0].output.width = width;
                layers[0].output.height = height;
                layers[0].output.row_stride_in_bytes = width * pixel_size;
                layers[0].output.pixel_stride_in_bytes = static_cast<uint16_t>(pixel_size);
                layers[0].output.format = format;
                layers[0].type = aov_type;

                // Call the actual denoise method
                self.denoise(params, guide_layer, layers, cuda_stream);
            },
            "input"_a,
            "output"_a,
            "width"_a,
            "height"_a,
            "albedo"_a.none() = nullptr,
            "normal"_a.none() = nullptr,
            "flow"_a.none() = nullptr,
            "hdr_intensity"_a.none() = std::nullopt,
            "blend_factor"_a = 0.0f,
            "hdr_average_color"_a.none() = std::nullopt,
            "temporal_mode_use_previous_layers"_a = 0,
            "format"_a = PixelFormat::float3,
            "aov_type"_a = DenoiserAOVType::beauty,
            "cuda_stream"_a = sgl::NativeHandle{},
            R"docstring(
            Denoise an image.

            Args:
                input: Buffer containing the noisy input image
                output: Buffer to write the denoised output image
                width: Image width in pixels
                height: Image height in pixels
                albedo: Optional albedo guide buffer (RGB, float3)
                normal: Optional normal guide buffer (XYZ, float3)
                flow: Optional flow guide buffer (XY, float2)
                hdr_intensity: Optional average log intensity of input image
                blend_factor: Blend factor between denoised and original (0.0-1.0)
                hdr_average_color: Optional average log color of input image (RGB)
                temporal_mode_use_previous_layers: Whether to use previous layers in temporal mode
                format: Pixel format of input/output images
                aov_type: Type of AOV being denoised
            )docstring"
        )
        .DEF_PROP_RO(OptixDenoiser, device);
}
