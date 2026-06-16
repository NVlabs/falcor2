// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/dlss_rr_feature_impl.h"
#include "falcor2/denoisers/ngx_impl.h"
#include "falcor2/falcor2.h"

#include <sgl/device/command.h>
#include <sgl/device/device.h>
#include <sgl/device/resource.h>
#include <nvsdk_ngx_helpers_dlssd.h>

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
        "D3D12 DLSS-RR evaluation {} must expose an ID3D12Resource.",
        name
    );
    return reinterpret_cast<ID3D12Resource*>(handle.value());
}

class D3D12DLSSRRFeatureImpl final : public DLSSRRFeature::Impl {
public:
    /// Create the D3D12 NGX DLSS-RR feature on an initialized command list.
    D3D12DLSSRRFeatureImpl(ref<NGX> ngx, DLSSRRFeatureDesc desc)
        : DLSSRRFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the D3D12 NGX feature and parameter map.
    ~D3D12DLSSRRFeatureImpl() override { release_resources(); }

    /// Record a D3D12 NGX DLSS-RR evaluation callback.
    void record_evaluate(
        ref<DLSSRRFeature> owner,
        sgl::CommandEncoder* command_encoder,
        const DLSSRREvaluateDesc& desc
    ) override
    {
        validate_evaluate_desc(desc);
        DLSSRREvaluateResources resources = retain_evaluate_resources(desc);
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

                NVSDK_NGX_D3D12_DLSSD_Eval_Params eval_params{};
                eval_params.pInDiffuseAlbedo = get_d3d12_resource(resources.diffuse_albedo, "diffuse_albedo");
                eval_params.pInSpecularAlbedo = get_d3d12_resource(resources.specular_albedo, "specular_albedo");
                eval_params.pInNormals = get_d3d12_resource(resources.normals, "normals");
                eval_params.pInRoughness = get_d3d12_resource(resources.roughness, "roughness");
                eval_params.pInColor = get_d3d12_resource(resources.color, "color");
                eval_params.pInOutput = get_d3d12_resource(resources.output, "output");
                eval_params.pInDepth = get_d3d12_resource(resources.depth, "depth");
                eval_params.pInMotionVectors = get_d3d12_resource(resources.motion_vectors, "motion_vectors");
                eval_params.InJitterOffsetX = resources.jitter_offset_x;
                eval_params.InJitterOffsetY = resources.jitter_offset_y;
                eval_params.InRenderSubrectDimensions.Width = m_desc.render_width;
                eval_params.InRenderSubrectDimensions.Height = m_desc.render_height;
                eval_params.InReset = resources.reset ? 1 : 0;
                eval_params.InMVScaleX = resources.motion_vector_scale_x;
                eval_params.InMVScaleY = resources.motion_vector_scale_y;
                eval_params.InPreExposure = resources.pre_exposure;
                eval_params.InExposureScale = resources.exposure_scale;
                eval_params.InFrameTimeDeltaInMsec = resources.frame_time_delta_ms;
                if (resources.motion_vectors_reflections) {
                    eval_params.pInMotionVectorsReflections
                        = get_d3d12_resource(resources.motion_vectors_reflections, "motion_vectors_reflections");
                }
                if (resources.specular_hit_distance) {
                    eval_params.pInSpecularHitDistance
                        = get_d3d12_resource(resources.specular_hit_distance, "specular_hit_distance");
                    eval_params.pInWorldToViewMatrix = const_cast<float*>(resources.view_from_world->data());
                    eval_params.pInViewToClipMatrix = const_cast<float*>(resources.clip_from_view->data());
                }

                set_last_evaluate_result(NGX_D3D12_EVALUATE_DLSSD_EXT(
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
            "D3D12 DLSS-RR parameter allocation failed: {}",
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
                    callback_error = "D3D12 DLSS-RR feature creation requires an active ID3D12GraphicsCommandList.";
                    return;
                }

                NVSDK_NGX_DLSSD_Create_Params create_params = make_create_params();
                create_result = NGX_D3D12_CREATE_DLSSD_EXT(
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

        FALCOR_CHECK(callback_called, "D3D12 DLSS-RR feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "D3D12 DLSS-RR feature creation failed: {}",
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

std::unique_ptr<DLSSRRFeature::Impl> make_d3d12_dlss_rr_feature_impl(ref<NGX> ngx, const DLSSRRFeatureDesc& desc)
{
    return std::make_unique<D3D12DLSSRRFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // SGL_WINDOWS

#endif // FALCOR_ENABLE_NGX
