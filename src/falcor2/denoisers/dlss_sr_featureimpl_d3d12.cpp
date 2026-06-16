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

#if SGL_WINDOWS

namespace falcor::ngx {
namespace {

ID3D12Resource* get_d3d12_resource(sgl::Texture* texture, const char* name)
{
    sgl::NativeHandle handle = texture->native_handle();
    FALCOR_CHECK(
        handle.type() == sgl::NativeHandleType::D3D12Resource && handle.value(),
        "D3D12 DLSS-SR evaluation {} must expose an ID3D12Resource.",
        name
    );
    return reinterpret_cast<ID3D12Resource*>(handle.value());
}

class D3D12DLSSSRFeatureImpl final : public DLSSSRFeature::Impl {
public:
    /// Create the D3D12 NGX DLSS-SR feature on an initialized command list.
    D3D12DLSSSRFeatureImpl(ref<NGX> ngx, DLSSSRFeatureDesc desc)
        : DLSSSRFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the D3D12 NGX feature and parameter map.
    ~D3D12DLSSSRFeatureImpl() override { release_resources(); }

    /// Record a D3D12 NGX DLSS-SR evaluation callback.
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
                if (native_handle.type() != sgl::NativeHandleType::D3D12GraphicsCommandList || !native_handle.value()) {
                    set_last_evaluate_result(NVSDK_NGX_Result_FAIL_InvalidParameter);
                    return;
                }

                NVSDK_NGX_D3D12_DLSS_Eval_Params eval_params{};
                eval_params.Feature.pInColor = get_d3d12_resource(resources.color, "color");
                eval_params.Feature.pInOutput = get_d3d12_resource(resources.output, "output");
                eval_params.Feature.InSharpness = m_desc.sharpness;
                eval_params.pInDepth = get_d3d12_resource(resources.depth, "depth");
                eval_params.pInMotionVectors = get_d3d12_resource(resources.motion_vectors, "motion_vectors");
                if (resources.exposure)
                    eval_params.pInExposureTexture = get_d3d12_resource(resources.exposure, "exposure");
                eval_params.InJitterOffsetX = resources.jitter_offset_x;
                eval_params.InJitterOffsetY = resources.jitter_offset_y;
                eval_params.InRenderSubrectDimensions.Width = evaluate_render_subrect_width(resources);
                eval_params.InRenderSubrectDimensions.Height = evaluate_render_subrect_height(resources);
                eval_params.InReset = resources.reset ? 1 : 0;
                eval_params.InMVScaleX = resources.motion_vector_scale_x;
                eval_params.InMVScaleY = resources.motion_vector_scale_y;
                eval_params.InPreExposure = resources.pre_exposure;
                eval_params.InExposureScale = resources.exposure_scale;

                set_last_evaluate_result(NGX_D3D12_EVALUATE_DLSS_EXT(
                    reinterpret_cast<ID3D12GraphicsCommandList*>(native_handle.value()),
                    m_feature_handle,
                    m_params,
                    &eval_params
                ));
            }
        );
    }

private:
    /// Allocate the NGX parameter map and create the feature through a native command callback.
    void initialize()
    {
        validate_feature_desc();

        NVSDK_NGX_Result allocate_result = NVSDK_NGX_D3D12_AllocateParameters(&m_params);
        FALCOR_CHECK(
            allocate_result == NVSDK_NGX_Result_Success && m_params,
            "D3D12 DLSS-SR parameter allocation failed: {}",
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
                if (native_handle.type() != sgl::NativeHandleType::D3D12GraphicsCommandList || !native_handle.value()) {
                    callback_error = "D3D12 DLSS-SR feature creation requires an active ID3D12GraphicsCommandList.";
                    return;
                }

                NVSDK_NGX_DLSS_Create_Params create_params = make_create_params();
                create_result = NGX_D3D12_CREATE_DLSS_EXT(
                    reinterpret_cast<ID3D12GraphicsCommandList*>(native_handle.value()),
                    1,
                    1,
                    &m_feature_handle,
                    m_params,
                    &create_params
                );
            }
        );

        ref<sgl::CommandBuffer> command_buffer = command_encoder->finish();
        device->submit_command_buffer(command_buffer);
        device->wait();

        FALCOR_CHECK(callback_called, "D3D12 DLSS-SR feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "D3D12 DLSS-SR feature creation failed: {}",
            ngx_result_to_string(static_cast<int32_t>(create_result))
        );
    }

    /// Release backend resources in the order expected by NGX.
    void release_resources() noexcept
    {
        if (m_feature_handle) {
            FALCOR_UNUSED(NVSDK_NGX_D3D12_ReleaseFeature(m_feature_handle));
            m_feature_handle = nullptr;
        }
        if (m_params) {
            FALCOR_UNUSED(NVSDK_NGX_D3D12_DestroyParameters(m_params));
            m_params = nullptr;
        }
    }

    NVSDK_NGX_Parameter* m_params{nullptr};
    NVSDK_NGX_Handle* m_feature_handle{nullptr};
};

} // namespace

std::unique_ptr<DLSSSRFeature::Impl> make_d3d12_dlss_sr_feature_impl(ref<NGX> ngx, const DLSSSRFeatureDesc& desc)
{
    return std::make_unique<D3D12DLSSSRFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // SGL_WINDOWS

#endif // FALCOR_ENABLE_NGX
