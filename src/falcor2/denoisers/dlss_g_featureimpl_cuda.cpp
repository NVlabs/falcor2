// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_g_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/formats.h>
#include <sgl/device/resource.h>

#include <nvsdk_ngx_defs_dlssg.h>

#include <string>
#include <utility>

namespace falcor::ngx {
namespace {

uint64_t get_cuda_texture_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_ro();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::texture && handle.value,
        "CUDA DLSS-G evaluation {} must expose a read CUtexObject.",
        name
    );
    return handle.value;
}

uint64_t get_cuda_surface_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_rw();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::rw_texture && handle.value,
        "CUDA DLSS-G evaluation {} must expose a write CUsurfObject.",
        name
    );
    return handle.value;
}

void set_ui(NVSDK_NGX_Parameter* params, const char* name, unsigned int value)
{
    NVSDK_NGX_Parameter_SetUI(params, name, value);
}

void set_subrect(
    NVSDK_NGX_Parameter* params,
    const char* base_x_name,
    const char* base_y_name,
    const char* width_name,
    const char* height_name,
    const NVSDK_NGX_Coordinates& base,
    const NVSDK_NGX_Dimensions& size
)
{
    set_ui(params, base_x_name, base.X);
    set_ui(params, base_y_name, base.Y);
    set_ui(params, width_name, size.Width);
    set_ui(params, height_name, size.Height);
}

void set_dlssg_opt_eval_params(NVSDK_NGX_Parameter* params, NVSDK_NGX_DLSSG_Opt_Eval_Params& opt)
{
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_MultiFrameCount, opt.multiFrameCount);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_MultiFrameIndex, opt.multiFrameIndex);

    NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_DLSSG_Parameter_CameraViewToClip, opt.cameraViewToClip);
    NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_DLSSG_Parameter_ClipToCameraView, opt.clipToCameraView);
    NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_DLSSG_Parameter_ClipToLensClip, opt.clipToLensClip);
    NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_DLSSG_Parameter_ClipToPrevClip, opt.clipToPrevClip);
    NVSDK_NGX_Parameter_SetVoidPointer(params, NVSDK_NGX_DLSSG_Parameter_PrevClipToClip, opt.prevClipToClip);

    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_JitterOffsetX, opt.jitterOffset[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_JitterOffsetY, opt.jitterOffset[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_MvecScaleX, opt.mvecScale[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_MvecScaleY, opt.mvecScale[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetX, opt.cameraPinholeOffset[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraPinholeOffsetY, opt.cameraPinholeOffset[1]);

    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraPosX, opt.cameraPos[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraPosY, opt.cameraPos[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraPosZ, opt.cameraPos[2]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraUpX, opt.cameraUp[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraUpY, opt.cameraUp[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraUpZ, opt.cameraUp[2]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraRightX, opt.cameraRight[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraRightY, opt.cameraRight[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraRightZ, opt.cameraRight[2]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdX, opt.cameraFwd[0]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdY, opt.cameraFwd[1]);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraFwdZ, opt.cameraFwd[2]);

    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraNear, opt.cameraNear);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraFar, opt.cameraFar);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraFOV, opt.cameraFOV);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_CameraAspectRatio, opt.cameraAspectRatio);

    set_ui(params, NVSDK_NGX_DLSSG_Parameter_ColorBuffersHDR, opt.colorBuffersHDR ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_DepthInverted, opt.depthInverted ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_CameraMotionIncluded, opt.cameraMotionIncluded ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_Reset, opt.reset ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_AutomodeOverrideReset, opt.automodeOverrideReset ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_NotRenderingGameFrames, opt.notRenderingGameFrames ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_OrthoProjection, opt.orthoProjection ? 1u : 0u);
    NVSDK_NGX_Parameter_SetF(params, NVSDK_NGX_DLSSG_Parameter_MvecInvalidValue, opt.motionVectorsInvalidValue);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_MvecDilated, opt.motionVectorsDilated ? 1u : 0u);
    set_ui(params, NVSDK_NGX_DLSSG_Parameter_MenuDetectionEnabled, opt.menuDetectionEnabled ? 1u : 0u);

    set_subrect(
        params,
        NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseX,
        NVSDK_NGX_DLSSG_Parameter_MVecsSubrectBaseY,
        NVSDK_NGX_DLSSG_Parameter_MVecsSubrectWidth,
        NVSDK_NGX_DLSSG_Parameter_MVecsSubrectHeight,
        opt.mvecsSubrectBase,
        opt.mvecsSubrectSize
    );
    set_subrect(
        params,
        NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseX,
        NVSDK_NGX_DLSSG_Parameter_DepthSubrectBaseY,
        NVSDK_NGX_DLSSG_Parameter_DepthSubrectWidth,
        NVSDK_NGX_DLSSG_Parameter_DepthSubrectHeight,
        opt.depthSubrectBase,
        opt.depthSubrectSize
    );
    NVSDK_NGX_Parameter_SetF(
        params,
        NVSDK_NGX_DLSSG_Parameter_MinRelativeLinearDepthObjectSeparation,
        opt.minRelativeLinearDepthObjectSeparation
    );
    set_subrect(
        params,
        NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseX,
        NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectBaseY,
        NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectWidth,
        NVSDK_NGX_DLSSG_Parameter_InputBackbufferSubrectHeight,
        opt.backbufferSubrectBase,
        opt.backbufferSubrectSize
    );
    set_subrect(
        params,
        NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseX,
        NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectBaseY,
        NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectWidth,
        NVSDK_NGX_DLSSG_Parameter_OutputInterpolatedSubrectHeight,
        opt.outputInterpSubrectBase,
        opt.outputInterpSubrectSize
    );
}

class CudaDLSSGFeatureImpl final : public DLSSGFeature::Impl {
public:
    /// Create the CUDA NGX DLSS-G feature on the active CUDA command stream.
    CudaDLSSGFeatureImpl(ref<NGX> ngx, DLSSGFeatureDesc desc)
        : DLSSGFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the CUDA NGX feature and parameter map.
    ~CudaDLSSGFeatureImpl() override { release_resources(); }

    /// Record a CUDA NGX DLSS-G evaluation callback.
    void record_evaluate(
        ref<DLSSGFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSGEvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSGEvaluateResources resources = retain_evaluate_resources(desc);
        prepare_evaluate_resources(command_encoder, resources);

        reset_last_evaluate_result();
        command_encoder->execute_callback(
            [this, resources = std::move(resources), owner = std::move(owner)](sgl::NativeHandle native_handle)
            {
                FALCOR_UNUSED(owner);
                if (native_handle.type() != sgl::NativeHandleType::CUstream) {
                    set_last_evaluate_result(NVSDK_NGX_Result_FAIL_InvalidParameter);
                    return;
                }

                m_cuda_device->cudaStream = reinterpret_cast<void*>(native_handle.value());

                uint64_t backbuffer = get_cuda_texture_object(resources.backbuffer, "backbuffer");
                uint64_t depth = get_cuda_texture_object(resources.depth, "depth");
                uint64_t motion_vectors = get_cuda_texture_object(resources.motion_vectors, "motion_vectors");
                uint64_t output_interpolated
                    = get_cuda_surface_object(resources.output_interpolated_frame, "output_interpolated_frame");

                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_DLSSG_Parameter_Backbuffer, &backbuffer);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_DLSSG_Parameter_MVecs, &motion_vectors);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_DLSSG_Parameter_Depth, &depth);
                NVSDK_NGX_Parameter_SetVoidPointer(
                    m_params,
                    NVSDK_NGX_DLSSG_Parameter_OutputInterpolated,
                    &output_interpolated
                );

                NVSDK_NGX_DLSSG_Opt_Eval_Params opt_eval_params = make_opt_eval_params(resources);
                set_dlssg_opt_eval_params(m_params, opt_eval_params);

                set_last_evaluate_result(NVSDK_NGX_CUDA_EvaluateFeature_C(m_feature_handle, m_params, nullptr));
            }
        );
    }

private:
    /// Allocate the NGX parameter map and create the feature through a native command callback.
    void initialize()
    {
        validate_feature_desc();

        NGX::Impl* ngx_impl = get_ngx_impl(m_ngx.get());
        m_cuda_device = ngx_impl ? static_cast<NVSDK_NGX_CUDADevice*>(ngx_impl->ngx_device()) : nullptr;
        FALCOR_CHECK(m_cuda_device && m_cuda_device->cudaContext, "CUDA DLSS-G feature creation requires CUDA NGX.");

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_CUDA_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "CUDA DLSS-G parameter allocation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(allocate_result))
        );

        sgl::Device* device = m_ngx->device();
        ref<sgl::CommandEncoder> command_encoder = device->create_command_encoder();

        bool callback_called = false;
        std::string callback_error;
        NVSDK_NGX_Result create_result = NVSDK_NGX_Result_Fail;

        command_encoder->execute_callback(
            [&](sgl::NativeHandle native_handle)
            {
                callback_called = true;
                if (native_handle.type() != sgl::NativeHandleType::CUstream) {
                    callback_error = "CUDA DLSS-G feature creation requires an active CUstream.";
                    return;
                }

                m_cuda_device->cudaStream = reinterpret_cast<void*>(native_handle.value());

                const uint32_t dxgi_format = sgl::get_format_info(m_desc.backbuffer_format).dxgi_format;
                NVSDK_NGX_DLSSG_Create_Params create_params = make_create_params(dxgi_format);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Input1, m_cuda_device->cudaContext);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Input2, m_cuda_device->cudaStream);
                set_ui(m_params, NVSDK_NGX_Parameter_CreationNodeMask, 1);
                set_ui(m_params, NVSDK_NGX_Parameter_VisibilityNodeMask, 1);
                set_ui(m_params, NVSDK_NGX_Parameter_Width, create_params.Width);
                set_ui(m_params, NVSDK_NGX_Parameter_Height, create_params.Height);
                set_ui(m_params, NVSDK_NGX_DLSSG_Parameter_BackbufferFormat, create_params.NativeBackbufferFormat);
                set_ui(m_params, NVSDK_NGX_DLSSG_Parameter_InternalWidth, create_params.RenderWidth);
                set_ui(m_params, NVSDK_NGX_DLSSG_Parameter_InternalHeight, create_params.RenderHeight);

                create_result = NVSDK_NGX_CUDA_CreateFeature1(
                    m_cuda_device,
                    NVSDK_NGX_Feature_FrameGeneration,
                    m_params,
                    &m_feature_handle
                );
            }
        );

        ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
        device->submit_command_buffer(command_buffer);
        device->wait();

        FALCOR_CHECK(callback_called, "CUDA DLSS-G feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "CUDA DLSS-G feature creation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(create_result))
        );
    }

    /// Release backend resources in the order expected by NGX.
    void release_resources() noexcept
    {
        if (m_feature_handle) {
            FALCOR_UNUSED(NVSDK_NGX_CUDA_ReleaseFeature(m_feature_handle));
            m_feature_handle = nullptr;
        }
        if (m_params) {
            FALCOR_UNUSED(NVSDK_NGX_CUDA_DestroyParameters(m_params));
            m_params = nullptr;
        }
    }

    NVSDK_NGX_CUDADevice* m_cuda_device{nullptr};
    NVSDK_NGX_Parameter* m_params{nullptr};
    NVSDK_NGX_Handle* m_feature_handle{nullptr};
};

} // namespace

std::unique_ptr<DLSSGFeature::Impl> make_cuda_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc)
{
    return std::make_unique<CudaDLSSGFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
