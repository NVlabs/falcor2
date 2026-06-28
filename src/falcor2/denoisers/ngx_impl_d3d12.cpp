// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0

#include "falcor2/core/config.h"

#if FALCOR_ENABLE_NGX

#include "falcor2/denoisers/ngx_impl.h"

#include "falcor2/falcor2.h"

#include <sgl/device/device.h>

#include <nvsdk_ngx.h>

#include <memory>
#include <utility>

#if SGL_WINDOWS

namespace falcor::ngx {

namespace {

ID3D12Device* get_d3d12_device(const sgl::Device* device)
{
    if (!device)
        return nullptr;

    for (const sgl::NativeHandle& handle : device->native_handles()) {
        if (handle.type() == sgl::NativeHandleType::D3D12Device)
            return reinterpret_cast<ID3D12Device*>(handle.value());
    }

    return nullptr;
}

class D3D12NGXImpl final : public NGX::Impl {
public:
    /// Initialize NGX against the native D3D12 device.
    explicit D3D12NGXImpl(ref<sgl::Device> device)
        : NGX::Impl(std::move(device))
    {
        initialize();
    }

    /// Shut down the D3D12 NGX context if initialization reached the SDK.
    ~D3D12NGXImpl()
    {
        if (!m_d3d12_device)
            return;

        NVSDK_NGX_D3D12_Shutdown1(m_d3d12_device);
    }

private:
    /// Initialize the D3D12 NGX SDK entry point and cache DLSS-D capability data.
    void initialize()
    {
        // NGX D3D12 initialization is tied to the exact ID3D12Device used by SGL.
        m_d3d12_device = get_d3d12_device(m_device.get());
        FALCOR_CHECK(m_d3d12_device, "NGX D3D12 initialization could not get an ID3D12Device native handle.");
        m_ngx_device = m_d3d12_device;

        // NGX requires a writable app-data path and optionally accepts an extra
        // search path where it can find the runtime binary beside the executable.
        const std::filesystem::path app_data_path = NGX::Impl::app_data_path();
        const std::wstring app_data_path_wide = NGX::Impl::to_wstring_path(app_data_path);
        const wchar_t* path_list[1] = {};
        NVSDK_NGX_FeatureCommonInfo feature_common_info = NGX::Impl::make_feature_common_info(path_list);

        // This call creates the NGX D3D12 context for the native SGL device.
        NVSDK_NGX_Result init_result = NVSDK_NGX_D3D12_Init(
            k_application_id,
            app_data_path_wide.c_str(),
            m_d3d12_device,
            &feature_common_info,
            NVSDK_NGX_Version_API
        );

        FALCOR_CHECK(
            init_result == NVSDK_NGX_Result_Success,
            "NGX D3D12 initialization failed: {}",
            ngx_result_to_string(static_cast<int32_t>(init_result))
        );

        // Capability parameters are SDK-owned and must be destroyed using the
        // matching D3D12 destroy function on every exit path.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_D3D12_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_D3D12_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_D3D12_DestroyParameters
            );

        if (capability_result != NVSDK_NGX_Result_Success || !capability_params) {
            // Initialization succeeded but capability query failed, so unwind the
            // SDK context before surfacing the error.
            NVSDK_NGX_D3D12_Shutdown1(m_d3d12_device);
            FALCOR_THROW(
                "NGX D3D12 capability query failed: {}",
                ngx_result_to_string(static_cast<int32_t>(capability_result))
            );
        }

        // Cache the DLSS-D support fields needed by public NGX queries.
        read_dlssd_capability_params(capability_params);
        read_dlss_capability_params(capability_params);
        read_frame_generation_capability_params(capability_params);
    }

    /// Query D3D12 capability parameters and derive DLSS-D optimal settings.
    DLSSDOptimalSettings
    query_dlssd_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        // Request a fresh parameter block because NGX computes optimal settings
        // through the same capability parameter interface.
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_D3D12_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_D3D12_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_D3D12_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSSD NGX D3D12 capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlssd_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    /// Query D3D12 capability parameters and derive DLSS Super Resolution optimal settings.
    DLSSOptimalSettings
    query_dlss_optimal_settings(uint32_t target_width, uint32_t target_height, QualityMode quality) const override
    {
        NVSDK_NGX_Parameter* capability_params = nullptr;
        NVSDK_NGX_Result capability_result = NVSDK_NGX_D3D12_GetCapabilityParameters(&capability_params);
        auto capability_params_guard
            = std::unique_ptr<NVSDK_NGX_Parameter, decltype(&NVSDK_NGX_D3D12_DestroyParameters)>(
                capability_params,
                NVSDK_NGX_D3D12_DestroyParameters
            );

        FALCOR_CHECK(
            capability_result == NVSDK_NGX_Result_Success && capability_params,
            "DLSS NGX D3D12 capability query for optimal settings failed: {}",
            ngx_result_to_string(static_cast<int32_t>(capability_result))
        );

        return query_dlss_optimal_settings_from_params(capability_params, target_width, target_height, quality);
    }

    ID3D12Device* m_d3d12_device{nullptr};
};

} // namespace

std::unique_ptr<NGX::Impl> make_d3d12_ngx_impl(ref<sgl::Device> device)
{
    return std::make_unique<D3D12NGXImpl>(std::move(device));
}

} // namespace falcor::ngx

#endif // SGL_WINDOWS

#endif // FALCOR_ENABLE_NGX
