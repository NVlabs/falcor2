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
#include <nvsdk_ngx_helpers_dlssg.h>

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
        "D3D12 DLSS-G evaluation {} must expose an ID3D12Resource.",
        name
    );
    return reinterpret_cast<ID3D12Resource*>(handle.value());
}

class D3D12DLSSGFeatureImpl final : public DLSSGFeature::Impl {
public:
    /// Create the D3D12 NGX DLSS-G feature on an initialized command list.
    D3D12DLSSGFeatureImpl(ref<NGX> ngx, DLSSGFeatureDesc desc)
        : DLSSGFeature::Impl(std::move(ngx), desc)
    {
        try {
            initialize();
        } catch (...) {
            release_resources();
            throw;
        }
    }

    /// Release the D3D12 NGX feature and parameter map.
    ~D3D12DLSSGFeatureImpl() override { release_resources(); }

    /// Record a D3D12 NGX DLSS-G evaluation callback.
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
                if (native_handle.type() != sgl::NativeHandleType::D3D12GraphicsCommandList || !native_handle.value()) {
                    set_last_evaluate_result(NVSDK_NGX_Result_FAIL_InvalidParameter);
                    return;
                }

                NVSDK_NGX_D3D12_DLSSG_Eval_Params eval_params{};
                eval_params.pBackbuffer = get_d3d12_resource(resources.backbuffer, "backbuffer");
                eval_params.pDepth = get_d3d12_resource(resources.depth, "depth");
                eval_params.pMVecs = get_d3d12_resource(resources.motion_vectors, "motion_vectors");
                eval_params.pOutputInterpFrame
                    = get_d3d12_resource(resources.output_interpolated_frame, "output_interpolated_frame");

                NVSDK_NGX_DLSSG_Opt_Eval_Params opt_eval_params = make_opt_eval_params(resources);
                set_last_evaluate_result(NGX_D3D12_EVALUATE_DLSSG(
                    reinterpret_cast<ID3D12GraphicsCommandList*>(native_handle.value()),
                    m_feature_handle,
                    m_params,
                    &eval_params,
                    &opt_eval_params
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
            "D3D12 DLSS-G parameter allocation failed: {}",
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
                    callback_error = "D3D12 DLSS-G feature creation requires an active ID3D12GraphicsCommandList.";
                    return;
                }

                const uint32_t dxgi_format = sgl::get_format_info(m_desc.backbuffer_format).dxgi_format;
                NVSDK_NGX_DLSSG_Create_Params create_params = make_create_params(dxgi_format);
                create_result = NGX_D3D12_CREATE_DLSSG(
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

        FALCOR_CHECK(callback_called, "D3D12 DLSS-G feature creation callback was not executed.");
        FALCOR_CHECK(callback_error.empty(), "{}", callback_error);
        FALCOR_CHECK(
            create_result == NVSDK_NGX_Result_Success && m_feature_handle,
            "D3D12 DLSS-G feature creation failed: {}",
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

std::unique_ptr<DLSSGFeature::Impl> make_d3d12_dlss_g_feature_impl(ref<NGX> ngx, const DLSSGFeatureDesc& desc)
{
    return std::make_unique<D3D12DLSSGFeatureImpl>(std::move(ngx), desc);
}

} // namespace falcor::ngx

#endif // SGL_WINDOWS

#endif // FALCOR_ENABLE_NGX
