// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "falcor2/core/types.h"
#include "falcor2/core/object.h"

#include <sgl/device/fwd.h>
#include <sgl/device/types.h>
#include <sgl/core/enum.h>

#include <memory>
#include <optional>

namespace falcor {
namespace optix {

// Types in this namespace map closely to OptiX denoiser types in optix headers (optix_types.h), but
// are designed to be cross version commpatible and use falcor2/sgl types where appropriate and
// use falcor2 naming conventions.

/// Pixel format for an image.
enum class PixelFormat : uint32_t {
    unknown,
    half1,
    half2,
    half3,
    half4,
    float1,
    float2,
    float3,
    float4,
    uchar3,
    uchar4,
    guide_layer
};
SGL_ENUM_INFO(
    PixelFormat,
    {
        {PixelFormat::unknown, "unknown"},
        {PixelFormat::half1, "half1"},
        {PixelFormat::half2, "half2"},
        {PixelFormat::half3, "half3"},
        {PixelFormat::half4, "half4"},
        {PixelFormat::float1, "float1"},
        {PixelFormat::float2, "float2"},
        {PixelFormat::float3, "float3"},
        {PixelFormat::float4, "float4"},
        {PixelFormat::uchar3, "uchar3"},
        {PixelFormat::uchar4, "uchar4"},
        {PixelFormat::guide_layer, "guide_layer"},
    }
);
SGL_ENUM_REGISTER(PixelFormat);

/// Type of AOV, used in temporal AOV modes as a hint to improve image quality.
enum class DenoiserAOVType : uint32_t { none, beauty, specular, reflection, refraction, diffuse };
SGL_ENUM_INFO(
    DenoiserAOVType,
    {
        {DenoiserAOVType::none, "none"},
        {DenoiserAOVType::beauty, "beauty"},
        {DenoiserAOVType::specular, "specular"},
        {DenoiserAOVType::reflection, "reflection"},
        {DenoiserAOVType::refraction, "refraction"},
        {DenoiserAOVType::diffuse, "diffuse"},
    }
);
SGL_ENUM_REGISTER(DenoiserAOVType);

enum class ModelKind {
    /// Built-in model for denoising single image.
    aov,

    /// Built-in model for denoising image sequence, temporally stable.
    temporal_aov,

    /// Built-in model for denoising single image upscaling (supports AOVs).
    upscale2x,

    /// Built-in model for denoising image sequence upscaling, temporally stable (supports AOVs).
    temporal_upscale2x,

    /// LDR/HDR denoisers. Deprecated on Optix9 in favour of aov.
    ldr,
    hdr,

    /// Temporal mode. Deprecated on Optix9 in favour of temporal_aov.
    temporal = 0x2325
};
SGL_ENUM_INFO(
    ModelKind,
    {
        {ModelKind::aov, "aov"},
        {ModelKind::temporal_aov, "temporal_aov"},
        {ModelKind::upscale2x, "upscale2x"},
        {ModelKind::temporal_upscale2x, "temporal_upscale2x"},
        {ModelKind::ldr, "ldr"},
        {ModelKind::hdr, "hdr"},
        {ModelKind::temporal, "temporal"},
    }
);
SGL_ENUM_REGISTER(ModelKind);


enum class AlphaMode {
    /// Copy alpha (if present) from input layer, no denoising.
    copy = 0,

    /// Denoise alpha.
    denoise = 1
};
SGL_ENUM_INFO(
    AlphaMode,
    {
        {AlphaMode::copy, "copy"},
        {AlphaMode::denoise, "denoise"},
    }
);
SGL_ENUM_REGISTER(AlphaMode);

/// 2D image contained in a buffer.
struct Image2D {
    /// Buffer containing the image data.
    ref<sgl::Buffer> buffer;
    /// Raw pointer to CUDA memory containing image data (CUDA only, instead of using buffer).
    sgl::DeviceAddress address{0};

    /// Image width in pixels.
    uint32_t width{0};
    /// Image height in pixels.
    uint32_t height{0};
    /// Row stride in bytes.
    uint32_t row_stride_in_bytes{0};
    /// Pixel stride in bytes.
    uint16_t pixel_stride_in_bytes{0};
    /// Pixel format.
    PixelFormat format{PixelFormat::unknown};
};

/// A layer to pass to the denoiser.
struct DenoiserLayer {
    /// Input image (beauty or AOV).
    Image2D input;
    /// Denoised output image from previous frame if temporal model kind selected.
    Image2D previous_output;
    /// Denoised output for given input.
    Image2D output;
    /// Type of AOV, used in temporal AOV modes as a hint to improve image quality.
    DenoiserAOVType type;
};

/// Guide layers to assist denoising.
struct DenoiserGuideLayer {
    /// Albedo image with three components: R, G, B.
    Image2D albedo;

    /// Normal image with two or three components: X, Y, Z.
    /// (X, Y) camera space for OPTIX_DENOISER_MODEL_KIND_LDR, OPTIX_DENOISER_MODEL_KIND_HDR models.
    /// (X, Y, Z) world space, all other models.
    Image2D normal;

    /// Flow image with two components: X, Y.
    /// Pixel movement from previous to current frame for each pixel in screen space.
    Image2D flow;

    /// Internal image used in temporal AOV denoising modes, pixel format OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER.
    Image2D previous_output_internal_guide_layer;
    /// Internal image used in temporal AOV denoising modes, pixel format OPTIX_PIXEL_FORMAT_INTERNAL_GUIDE_LAYER.
    Image2D output_internal_guide_layer;

    /// Image with a single component value that specifies how trustworthy the flow vector at x,y position in
    /// OptixDenoiserGuideLayer::flow is. Range 0..1 (low->high trustworthiness).
    /// Ignored if data pointer in the image is zero.
    Image2D flow_trustworthiness;
};

/// Denoiser parameters.
struct DenoiserParams {
    /// Average log intensity of input image (optional).
    /// If not set, autoexposure will be calculated automatically for the input image.
    /// Should be set to average log intensity of the entire image at least if tiling is used to
    /// get consistent autoexposure for all tiles.
    std::optional<float> hdr_intensity;

    /// Blend factor.
    /// If set to 0 the output is 100% of the denoised input. If set to 1, the output is 100% of
    /// the unmodified input. Values between 0 and 1 will linearly interpolate between the denoised
    /// and unmodified input.
    float blend_factor;

    /// Average log color of input image, separate for RGB channels (optional).
    /// This parameter is used when the OPTIX_DENOISER_MODEL_KIND_AOV model kind is set.
    /// If not set, average log color will be calculated automatically. See hdr_intensity for tiling,
    /// this also applies here.
    std::optional<float3> hdr_average_color;

    /// In temporal modes this parameter must be set to 1 if previous layers (e.g.
    /// previous_output_internal_guide_layer) contain valid data. This is the case in the
    /// second and subsequent frames of a sequence (for example after a change of camera
    /// angle). In the first frame of such a sequence this parameter must be set to 0.
    uint32_t temporal_mode_use_previous_layers;
};

} // namespace optix
} // namespace falcor
