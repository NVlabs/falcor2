// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_sr_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>

#include <nvsdk_ngx_helpers.h>

#include <string>
#include <utility>

namespace falcor::ngx {
namespace {

uint64_t get_cuda_texture_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_ro();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::texture && handle.value,
        "CUDA DLSS-SR evaluation {} must expose a read CUtexObject.",
        name
    );
    return handle.value;
}

uint64_t get_cuda_surface_object(sgl::Texture* texture, const char* name)
{
    sgl::DescriptorHandle handle = texture->descriptor_handle_rw();
    FALCOR_CHECK(
        handle.type == sgl::DescriptorHandleType::rw_texture && handle.value,
        "CUDA DLSS-SR evaluation {} must expose a write CUsurfObject.",
        name
    );
    return handle.value;
}

class CudaDLSSSRFeatureImpl final : public DLSSSRFeature::Impl {
public:
    /// Create the CUDA NGX DLSS-SR feature on the active CUDA command stream.
    CudaDLSSSRFeatureImpl(ref<NGX> ngx, DLSSSRFeatureDesc desc)
        : DLSSSRFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the CUDA NGX feature and parameter map.
    ~CudaDLSSSRFeatureImpl() override { release_resources(); }

    /// Record a CUDA NGX DLSS-SR evaluation callback.
    void record_evaluate(
        ref<DLSSSRFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSSREvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSSREvaluateResources resources = retain_evaluate_resources(desc);
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

                uint64_t color = get_cuda_texture_object(resources.color, "color");
                uint64_t depth = get_cuda_texture_object(resources.depth, "depth");
                uint64_t motion_vectors = get_cuda_texture_object(resources.motion_vectors, "motion_vectors");
                uint64_t output = get_cuda_surface_object(resources.output, "output");
                uint64_t exposure = resources.exposure ? get_cuda_texture_object(resources.exposure, "exposure") : 0;

                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Color, &color);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Output, &output);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Depth, &depth);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_MotionVectors, &motion_vectors);
                NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Jitter_Offset_X, resources.jitter_offset_x);
                NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Jitter_Offset_Y, resources.jitter_offset_y);
                NVSDK_NGX_Parameter_SetF(m_params, NVSDK_NGX_Parameter_Sharpness, m_desc.sharpness);
                NVSDK_NGX_Parameter_SetI(m_params, NVSDK_NGX_Parameter_Reset, resources.reset ? 1 : 0);
                NVSDK_NGX_Parameter_SetF(
                    m_params,
                    NVSDK_NGX_Parameter_MV_Scale_X,
                    resources.motion_vector_scale_x == 0.f ? 1.f : resources.motion_vector_scale_x
                );
                NVSDK_NGX_Parameter_SetF(
                    m_params,
                    NVSDK_NGX_Parameter_MV_Scale_Y,
                    resources.motion_vector_scale_y == 0.f ? 1.f : resources.motion_vector_scale_y
                );
                NVSDK_NGX_Parameter_SetVoidPointer(
                    m_params,
                    NVSDK_NGX_Parameter_ExposureTexture,
                    resources.exposure ? &exposure : nullptr
                );
                NVSDK_NGX_Parameter_SetUI(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Width,
                    evaluate_render_subrect_width(resources)
                );
                NVSDK_NGX_Parameter_SetUI(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Render_Subrect_Dimensions_Height,
                    evaluate_render_subrect_height(resources)
                );
                NVSDK_NGX_Parameter_SetF(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Pre_Exposure,
                    resources.pre_exposure == 0.f ? 1.f : resources.pre_exposure
                );
                NVSDK_NGX_Parameter_SetF(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Exposure_Scale,
                    resources.exposure_scale == 0.f ? 1.f : resources.exposure_scale
                );

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
        FALCOR_CHECK(m_cuda_device && m_cuda_device->cudaContext, "CUDA DLSS-SR feature creation requires CUDA NGX.");

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_CUDA_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "CUDA DLSS-SR parameter allocation failed: {}",
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
                    callback_error = "CUDA DLSS-SR feature creation requires an active CUstream.";
                    return;
                }

                m_cuda_device->cudaStream = reinterpret_cast<void*>(native_handle.value());

                NVSDK_NGX_DLSS_Create_Params create_params = make_create_params();
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Input1, m_cuda_device->cudaContext);
                NVSDK_NGX_Parameter_SetVoidPointer(m_params, NVSDK_NGX_Parameter_Input2, m_cuda_device->cudaStream);
                NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Width, create_params.Feature.InWidth);
                NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_Height, create_params.Feature.InHeight);
                NVSDK_NGX_Parameter_SetUI(m_params, NVSDK_NGX_Parameter_OutWidth, create_params.Feature.InTargetWidth);
                NVSDK_NGX_Parameter_SetUI(
                    m_params,
                    NVSDK_NGX_Parameter_OutHeight,
                    create_params.Feature.InTargetHeight
                );
                NVSDK_NGX_Parameter_SetI(
                    m_params,
                    NVSDK_NGX_Parameter_PerfQualityValue,
                    create_params.Feature.InPerfQualityValue
                );
                NVSDK_NGX_Parameter_SetI(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Feature_Create_Flags,
                    create_params.InFeatureCreateFlags
                );
                NVSDK_NGX_Parameter_SetI(
                    m_params,
                    NVSDK_NGX_Parameter_DLSS_Enable_Output_Subrects,
                    create_params.InEnableOutputSubrects ? 1 : 0
                );

                create_result = NVSDK_NGX_CUDA_CreateFeature1(
                    m_cuda_device,
                    NVSDK_NGX_Feature_SuperSampling,
                    m_params,
                    &m_feature_handle
                );
            }
        );

        ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
        device->submit_command_buffer(command_buffer);
        device->wait();

        FALCOR_CHECK(callback_called, "CUDA DLSS-SR feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "CUDA DLSS-SR feature creation failed: {}",
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

std::unique_ptr<DLSSSRFeature::Impl> make_cuda_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc)
{
    return std::make_unique<CudaDLSSSRFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // FALCOR_ENABLE_NGX
